"""Explicitly experimental full-model correctness probe; no timing or production opt-in.

The streamed CUDA candidate is deliberately enabled only inside this process.
Run reference and candidate in different processes; compare retained tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from benchmarks.harness.b13_moe_fixture import trainables
from benchmarks.harness.b14_group_diag import find_cublas

TARGETS = ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]
REVISION = "2a205727f616995ad4aba57806f8e2d0d2f28743"


def prepare(model_dir: Path, shard_dir: Path) -> None:
    from soup_cli.utils.layer_shard import shard_checkpoint

    if shard_dir.exists() and any(shard_dir.iterdir()):
        raise ValueError("shard directory must be empty; retain previous attempts separately")
    index = shard_checkpoint(str(model_dir), str(shard_dir), dtype="bfloat16",
                             arch="deepseek_v3", expert_layout="per_expert")
    print(json.dumps({"expert_layers": index.expert_layers, "experts": index.expert_count,
                      "expert_shapes": index.expert_shapes}, default=list), flush=True)


def _load_model(model_dir: Path, shard_dir: Path, arm: str):
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, DeepseekV3Config

    from soup_cli.utils.peft_wiring import attach_adapter

    lora = LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, bias="none",
                      target_modules=TARGETS, task_type=TaskType.CAUSAL_LM)
    from soup_cli.utils.layer_shard import read_shard_index

    index = read_shard_index(str(shard_dir))
    if index.quant != "nf4" or index.quant_device != "cuda":
        raise ValueError("this probe requires a CUDA-quantized NF4 cache")
    if arm == "resident":
        from benchmarks.harness.d3_academic_nf4_reference import (
            _apply_nf4,
            _match_nf4_projection_kernel,
            _match_quantized_lora_accumulation,
        )

        config = DeepseekV3Config.from_pretrained(model_dir)
        config._attn_implementation = "eager"
        config.use_cache = False
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), config=config, dtype=torch.bfloat16, trust_remote_code=False,
            experts_implementation="eager")
        print("applying stored NF4 weights to resident reference", flush=True)
        _apply_nf4(model, shard_dir, index)
        _match_nf4_projection_kernel(model, shard_dir, index)
        model = attach_adapter(model.cuda(), lora)
        _match_quantized_lora_accumulation(model)
        return model, None

    from soup_cli.config.schema import ExpertStreamConfig
    from soup_cli.utils import kimi_k2
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    load_config = kimi_k2.load_stream_config

    def eager_config(*args, **kwargs):
        config = load_config(*args, **kwargs)
        config._attn_implementation = "eager"
        return config

    kimi_k2.load_stream_config = eager_config
    library = find_cublas()
    if library is None:
        raise ValueError("cuBLAS library missing")
    options = ExpertStreamConfig(backend=arm, weight_budget_bytes=128 * 2**20,
                                 activation_budget_bytes=512 * 2**20,
                                 call_budget_bytes=256 * 2**20,
                                 cublas_library=str(library), native_rows_max=128,
                                 native_rows_from=None)
    model, runtime = build_streamed_model(
        model_id=str(model_dir), shard_dir=str(shard_dir), index=index, lora_config=lora,
        device="cuda", dtype="bfloat16", buffers=2, pin=False, seed=71, quant="nf4",
        double_quant=index.double_quant,
        expert_options=options, trust_remote_code=False,
    )
    return model, runtime


def _initialize_rotary(model):
    from soup_cli.utils.expert_training import initialize_deepseek_rotary

    initialize_deepseek_rotary(model, device="cuda")


def run(model_dir: Path, shard_dir: Path, arm: str, output: Path, tokens: int,
        negative_control: bool = False) -> None:
    if not 2 <= tokens <= 2048:
        raise ValueError("token count must be in 2..2048")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("native BF16 CUDA required")
    free, _ = torch.cuda.mem_get_info()
    minimum = 20 * 2**30 if arm == "resident" else 4 * 2**30
    if free < minimum:
        raise ValueError(f"insufficient free GPU memory for preflight: {free} < {minimum}")
    if output.exists():
        raise ValueError("output already exists; preserve each attempt")
    output.mkdir(parents=True)
    torch.manual_seed(102)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.preferred_blas_library("cublaslt")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
    model, runtime = _load_model(model_dir, shard_dir, arm)
    try:
        _initialize_rotary(model)
        parameters = trainables(model)
        if not parameters or any("lora_" not in name for name in parameters):
            raise ValueError("unexpected trainable parameters")
        generator = torch.Generator().manual_seed(931)
        with torch.no_grad():
            for name, parameter in sorted(parameters.items()):
                values = torch.randn(parameter.shape, generator=generator).to(parameter)
                parameter.copy_(values * 0.01)
            if negative_control:
                modified = next((p for n, p in sorted(parameters.items()) if "lora_B" in n), None)
                if modified is None:
                    raise ValueError("negative control requires a LoRA B tensor")
                modified.add_(0.1)
        model.train()
        ids = ((torch.arange(tokens, device="cuda") * 17 + 7)
               % int(model.config.vocab_size)).unsqueeze(0)
        optimizer = torch.optim.SGD(parameters.values(), lr=0.01)
        tensors = {}
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            result = model(input_ids=ids, labels=ids)
            if result.loss is None or not bool(torch.isfinite(result.loss)):
                raise ValueError(f"step {step}: loss is missing or nonfinite")
            result.loss.backward()
            prefix = "" if step == 0 else "step1."
            tensors[prefix + "logits"] = result.logits.detach().cpu().contiguous()
            tensors[prefix + "loss"] = result.loss.detach().cpu().contiguous()
            for name, parameter in sorted(parameters.items()):
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                    raise ValueError(f"step {step}: missing or nonfinite gradient: {name}")
                tensors[prefix + "grad." + name] = parameter.grad.detach().cpu().contiguous()
            optimizer.step()
            for name, parameter in sorted(parameters.items()):
                tensors[prefix + "updated." + name] = parameter.detach().cpu().contiguous()
            print(f"optimizer step {step + 1} completed", flush=True)
        tensor_path = output / "tensors.safetensors"
        save_file(tensors, str(tensor_path))
        statistics = runtime.stats() if runtime is not None else None
        from soup_cli.utils.layer_shard import read_shard_index

        meta = {"arm": arm, "revision": REVISION, "token_count": ids.numel(),
                "codec": "nf4-nested" if read_shard_index(str(shard_dir)).double_quant else "nf4",
                "seed": 931,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(),
                "negative_control": negative_control,
                "scope": "exploratory real weights, synthetic tokens, two manual SGD steps",
                "attention": "native eager with runtime CUDA RoPE initialization",
                "parameters": len(parameters), "tensor_sha256": hashlib.sha256(
                    tensor_path.read_bytes()).hexdigest(), "runtime": statistics}
        (output / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        print(json.dumps(meta, default=str), flush=True)
    finally:
        if runtime is not None:
            runtime.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--arm", choices=("resident", "sequential", "grouped"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tokens", type=int, default=6)
    parser.add_argument("--negative-control", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.model, args.shards)
    elif args.arm is None or args.output is None:
        parser.error("run requires --arm and --output")
    else:
        run(args.model, args.shards, args.arm, args.output, args.tokens, args.negative_control)


if __name__ == "__main__":
    main()
