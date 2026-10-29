"""Bounded Qwen2 + LoRA pipeline experiment; not Soup CLI pipeline support.

Use prepare, torchrun ... train, then a separate reload process. No heavy
dependencies are imported until an experiment function is called.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

from rich.console import Console

from soup_cli.config.schema import SoupConfig

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
console = Console()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def plan(config: SoupConfig) -> dict[str, int]:
    """Interpret existing Pydantic batch/stage fields for this PP-only recipe."""
    training = config.training
    if training.parallelism != "pipeline" or training.pipeline_stages != 2:
        raise ValueError("this experiment requires parallelism=pipeline, pipeline_stages=2")
    batch = training.batch_size
    if not isinstance(batch, int) or isinstance(batch, bool) or batch % 2:
        raise ValueError("this experiment requires a positive even integer batch_size")
    if batch < 2:
        raise ValueError("batch_size must be >= 2")
    if (config.backend != "transformers" or training.quantization != "none"
            or training.lora.dropout != 0 or training.lora.r < 1):
        raise ValueError("this recipe requires transformers, unquantized LoRA and dropout=0")
    return {"stages": 2, "microbatches": 2, "micro_batch": batch // 2,
            "schedule_batch": batch, "accumulation": training.gradient_accumulation_steps,
            "global_batch": batch * training.gradient_accumulation_steps}


def prepare(root: Path, cpu_fixture: bool) -> None:
    import torch
    from transformers import AutoConfig, Qwen2Config, Qwen2ForCausalLM

    root.mkdir(parents=True, exist_ok=False)
    if cpu_fixture:
        base = root / "base"
        torch.manual_seed(42)
        model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True,
            max_position_embeddings=512, attention_dropout=0.0,
        ))
        model.save_pretrained(base)
        revision = "offline-random-fixture"
    else:
        from huggingface_hub import snapshot_download

        base = Path(snapshot_download(MODEL, revision=REVISION,
                                     allow_patterns=["*.json", "*.safetensors"]))
        revision = REVISION
    model_config = AutoConfig.from_pretrained(base, local_files_only=True)
    sequence = 64 if cpu_fixture else 128
    generator = torch.Generator().manual_seed(314159)
    data = torch.randint(4, model_config.vocab_size, (64, sequence), generator=generator)
    torch.save(data, root / "tokens.pt")
    config = SoupConfig.model_validate({
        "base": str(base), "backend": "transformers", "output": str(root / "adapter"),
        "data": {"train": str(root / "tokens.pt"), "max_length": sequence},
        "training": {"batch_size": 4, "gradient_accumulation_steps": 2,
                     "parallelism": "pipeline", "pipeline_stages": 2,
                     "quantization": "none", "gradient_checkpointing": False,
                     "lr": 0.0002, "weight_decay": 0.0, "seed": 42,
                     "lora": {"r": 8, "alpha": 16, "dropout": 0.0,
                              "target_modules": ["q_proj", "v_proj"]}},
    })
    write_json(root / "config.json", config.model_dump(mode="json"))
    write_json(root / "prepared.json", {
        "model": MODEL, "revision": revision, "cpu_fixture": cpu_fixture,
        "tokens_sha256": hashlib.sha256((root / "tokens.pt").read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256((root / "config.json").read_bytes()).hexdigest(),
        "base_files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in base.iterdir() if p.is_file()},
        "description": "synthetic seeded tokens; no quality claim", "plan": plan(config),
    })
    console.print(f"Prepared pipeline experiment: {root}")


def verify_prepared(root: Path, config: SoupConfig) -> dict[str, Any]:
    manifest = json.loads((root / "prepared.json").read_text())
    for filename, key in (("tokens.pt", "tokens_sha256"), ("config.json", "config_sha256")):
        if hashlib.sha256((root / filename).read_bytes()).hexdigest() != manifest[key]:
            raise ValueError(f"prepared input changed: {filename}")
    for name, expected in manifest["base_files_sha256"].items():
        if hashlib.sha256((Path(config.base) / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"immutable base changed: {name}")
    return manifest


def load_model(config: SoupConfig) -> Any:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(42 if config.training.seed is None else config.training.seed)
    model = AutoModelForCausalLM.from_pretrained(
        config.base, dtype=torch.float32, attn_implementation="eager",
        local_files_only=True,
    )
    if model.config.model_type != "qwen2" or any(
        kind != "full_attention" for kind in model.config.layer_types
    ):
        raise ValueError("this recipe supports full-attention Qwen2 only")
    model.config.use_cache = False
    return get_peft_model(model, LoraConfig(
        r=config.training.lora.r, lora_alpha=config.training.lora.alpha, lora_dropout=0.0,
        target_modules=config.training.lora.target_modules, task_type="CAUSAL_LM",
    ))


def build_stage(model: Any, rank: int) -> tuple[Any, dict[int, str], list[int]]:
    """Keep canonical layer numbers and duplicate only frozen tied embedding/head."""
    import torch

    if rank not in (0, 1):
        raise ValueError("only stage ranks 0 and 1 are supported")
    core = model.get_base_model().model
    count = len(core.layers)
    if count < 2 or count % 2:
        raise ValueError("this recipe needs an even number of decoder layers >= 2")
    indices = list(range(0, count // 2) if rank == 0 else range(count // 2, count))
    names = {id(parameter): name.replace(".default.", ".")
             for name, parameter in model.named_parameters() if parameter.requires_grad}

    class Stage(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleDict({str(index): core.layers[index] for index in indices})
            self.rotary_emb = core.rotary_emb
            self.embed = core.embed_tokens if rank == 0 else None
            self.norm = core.norm if rank == 1 else None
            self.head = model.get_output_embeddings() if rank == 1 else None

        def forward(self, value: Any) -> Any:
            hidden = self.embed(value) if self.embed is not None else value
            length = hidden.shape[1]
            positions = torch.arange(length, device=hidden.device).unsqueeze(0)
            rotary = self.rotary_emb(hidden, positions)
            mask = torch.triu(torch.full((length, length), torch.finfo(hidden.dtype).min,
                                        device=hidden.device, dtype=hidden.dtype), diagonal=1)
            for layer in self.layers.values():
                hidden = layer(hidden, attention_mask=mask[None, None], position_ids=positions,
                               position_embeddings=rotary, use_cache=False)
            return self.head(self.norm(hidden)) if self.head is not None else hidden

    stage = Stage()
    return stage, {id(p): names[id(p)] for p in stage.parameters() if p.requires_grad}, indices


def loss(logits: Any, labels: Any) -> Any:
    import torch.nn.functional as functional

    return functional.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
                                    labels[:, 1:].contiguous().view(-1))


def snapshot(parameters: Any, names: dict[int, str], gradients: bool = False) -> dict[str, Any]:
    result = {}
    for parameter in parameters:
        if id(parameter) in names:
            value = parameter.grad if gradients else parameter
            if value is None:
                raise AssertionError(f"missing gradient: {names[id(parameter)]}")
            result[names[id(parameter)]] = value.detach().cpu().clone()
    return result


def assert_state(actual: dict[str, Any], expected: dict[str, Any]) -> float:
    import torch

    if actual.keys() != expected.keys():
        raise AssertionError("adapter key sets differ")
    maximum = 0.0
    for name in actual:
        if not torch.isfinite(actual[name]).all():
            raise AssertionError(f"non-finite tensor: {name}")
        torch.testing.assert_close(actual[name], expected[name], atol=1e-5, rtol=0.02)
        maximum = max(maximum, (actual[name] - expected[name]).abs().max().item())
    return maximum


def frozen_digest(module: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def optimizer_for(module: Any, config: SoupConfig) -> Any:
    import torch

    return torch.optim.AdamW([p for p in module.parameters() if p.requires_grad],
                             lr=config.training.lr, betas=(0.9, 0.999), eps=1e-8,
                             weight_decay=config.training.weight_decay)


def baseline(model: Any, config: SoupConfig, data: Any, steps: int) -> list[dict[str, Any]]:
    import torch

    batch = plan(config)
    names = {id(p): name.replace(".default.", ".")
             for name, p in model.named_parameters() if p.requires_grad}
    optimizer = optimizer_for(model, config)
    records = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        rows = data[step * batch["global_batch"]:(step + 1) * batch["global_batch"]]
        values = []
        for tokens in rows.split(batch["micro_batch"]):
            logits = model(tokens, use_cache=False).logits
            value = loss(logits, tokens)
            values.append(value.detach())
            (value / (batch["microbatches"] * batch["accumulation"])).backward()
        gradients = snapshot(model.parameters(), names, True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
        optimizer.step()
        records.append({"gradients": gradients, "adapter": snapshot(model.parameters(), names),
                        "loss": torch.stack(values).mean().item()})
    return records


def merge_adapter_shards(shards: list[dict[str, Any]], expected_keys: set[str]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for shard in shards:
        if merged.keys() & shard.keys():
            raise AssertionError("adapter ownership overlaps between pipeline stages")
        merged.update(shard)
    if merged.keys() != expected_keys:
        raise AssertionError("pipeline adapter shards have missing or unexpected keys")
    return merged


def launch_training(root: Path, output: Path, device: str, steps: int) -> None:
    """Keep rank workers in the caller's environment, even with a global torchrun."""
    subprocess.run([
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
        str(Path(__file__).absolute()), "train", "--prepared", str(root),
        "--output", str(output), "--device", device, "--steps", str(steps),
    ], check=True)


def train(root: Path, output: Path, device_type: str, steps: int) -> None:
    import psutil
    import torch
    import torch.distributed as dist
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

    config = SoupConfig.model_validate_json((root / "config.json").read_text())
    prepared = verify_prepared(root, config)
    batch = plan(config)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if world != 2 or steps < 1 or steps * batch["global_batch"] > 64:
        raise ValueError("need exactly two ranks and 1..8 optimizer steps for the fixed data")
    device = torch.device(device_type, int(os.environ.get("LOCAL_RANK", rank)))
    if device_type == "cpu":
        device = torch.device("cpu")
    else:
        if platform.system() != "Linux" or torch.cuda.device_count() != 2:
            raise ValueError("GPU pilot requires Linux and exactly two visible CUDA devices")
        if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
            raise ValueError("GPU pilot requires a clean committed checkout")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    dist.init_process_group("gloo" if device_type == "cpu" else "nccl",
                            timeout=timedelta(seconds=120))
    try:
        directory = output / f"rank-{rank}"
        directory.mkdir(parents=True, exist_ok=False)
        model = load_model(config).to(device).train()
        initial = {key: value.detach().cpu().clone()
                   for key, value in get_peft_model_state_dict(model).items()}
        data = torch.load(root / "tokens.pt", weights_only=True).to(device)
        reference = baseline(model, config, data, steps)
        set_peft_model_state_dict(model, initial)
        model.zero_grad(set_to_none=True)
        model.cpu()
        gc.collect()
        if device_type == "cuda":
            torch.cuda.empty_cache()
        module, names, layers = build_stage(model, rank)
        module.to(device).train()
        if any(parameter.device != device for parameter in module.parameters()):
            raise AssertionError("stage parameters are on the wrong device")
        frozen = frozen_digest(module)
        optimizer = optimizer_for(module, config)
        torch.save({
            "initial": {key: value for key, value in initial.items() if key in names.values()},
            "first_gradient": {key: value for key, value in reference[0]["gradients"].items()
                               if key in names.values()},
            "first_adapter": {key: value for key, value in reference[0]["adapter"].items()
                              if key in names.values()},
            "final_adapter": {key: value for key, value in reference[-1]["adapter"].items()
                              if key in names.values()},
        }, directory / "control.pt")
        sequence, hidden = config.data.max_length, model.config.hidden_size
        shape = (batch["micro_batch"], sequence)
        example_in = torch.empty(shape, dtype=torch.long, device=device) if rank == 0 else (
            torch.empty((*shape, hidden), device=device, requires_grad=True))
        example_out = torch.empty((*shape, hidden if rank == 0 else model.config.vocab_size),
                                  device=device, requires_grad=True)
        stage = PipelineStage(module, rank, 2, device, input_args=(example_in,),
                              output_args=example_out)
        divisor = batch["microbatches"] * batch["accumulation"]
        # Native scaling divides existing .grad on every schedule.step(). For
        # accumulation across calls, normalize each micro-loss exactly once.
        schedule = ScheduleGPipe(stage, batch["microbatches"],
                                  loss_fn=lambda outputs, labels: loss(outputs, labels) / divisor,
                                  scale_grads=False)
        calls = {"forward": 0, "backward": 0}

        def backward_observed(gradient: Any) -> None:
            calls["backward"] += 1

        def forward_observed(module: Any, inputs: Any, outputs: Any) -> None:
            calls["forward"] += 1
            outputs.register_hook(backward_observed)

        hook = module.register_forward_hook(forward_observed)
        records = []
        dist.barrier()
        if device_type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        rss_before = psutil.Process().memory_info().rss
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            row_ids = list(range(step * batch["global_batch"], (step + 1) * batch["global_batch"]))
            micro_losses: list[Any] = []
            for rows in data[row_ids].split(batch["schedule_batch"]):
                if rank == 0:
                    schedule.step(rows)
                else:
                    call_losses: list[Any] = []
                    schedule.step(target=rows, losses=call_losses, return_outputs=False)
                    micro_losses.extend(call_losses)
            gradients = snapshot(module.parameters(), names, True)
            grad_error = assert_state(gradients, {key: value for key, value in
                                                 reference[step]["gradients"].items()
                                                 if key in gradients})
            if not any(value.count_nonzero() for value in gradients.values()):
                raise AssertionError("all stage gradients are zero")
            norm_squared = sum(p.grad.float().square().sum() for p in module.parameters()
                               if p.requires_grad)
            dist.all_reduce(norm_squared)
            coefficient = (config.training.max_grad_norm / (norm_squared.sqrt() + 1e-6)).clamp(
                max=1,
            )
            for parameter in module.parameters():
                if parameter.requires_grad:
                    parameter.grad.mul_(coefficient)
            optimizer.step()
            adapter = snapshot(module.parameters(), names)
            if step == 0:
                torch.save({"gradients": gradients, "adapter": adapter},
                           directory / "first-step.pt")
            update_error = assert_state(adapter, {key: value for key, value in
                                                  reference[step]["adapter"].items()
                                                  if key in adapter})
            record = {"step": step + 1, "row_ids": row_ids,
                      "gradient_error": grad_error, "adapter_error": update_error}
            if rank == 1:
                observed_loss = sum(value.item() for value in micro_losses)
                if not torch.isfinite(torch.tensor(observed_loss)):
                    raise AssertionError("non-finite pipeline loss")
                torch.testing.assert_close(torch.tensor(observed_loss),
                                           torch.tensor(reference[step]["loss"]),
                                           atol=1e-5, rtol=1e-5)
                record["loss"] = observed_loss
            records.append(record)
        if frozen_digest(module) != frozen:
            raise AssertionError("frozen stage base weights changed")
        hook.remove()
        expected_calls = steps * batch["accumulation"] * batch["microbatches"]
        if calls != {"forward": expected_calls, "backward": expected_calls}:
            raise AssertionError(f"pipeline did not execute the expected microsteps: {calls}")
        adapter = snapshot(module.parameters(), names)
        changed = sum(not torch.equal(value, initial[key]) for key, value in adapter.items())
        if changed == 0:
            raise AssertionError("optimizer did not update stage adapters")
        torch.save(adapter, directory / "adapter.pt")
        peak = None if device_type == "cpu" else {
            "allocated": torch.cuda.max_memory_allocated(device),
            "reserved": torch.cuda.max_memory_reserved(device),
        }
        # A separate small distributed eval passes activations across stages.
        fixed = data[:1, :min(32, sequence)].contiguous()
        module.eval()
        with torch.no_grad():
            if rank == 0:
                activations = module(fixed).contiguous()
                dist.send(activations, 1)
                torch.save(fixed.cpu(), output / "eval-input.pt")
            else:
                activations = torch.empty((1, fixed.shape[1], hidden), device=device)
                dist.recv(activations, 0)
                torch.save(module(activations).cpu(), output / "eval-before-save.pt")
        write_json(directory / "record.json", {
            "rank": rank, "world_size": world, "device": str(device), "layers": layers,
            "pid": os.getpid(), "observed_stage_calls": calls,
            "stage_parameter_numel": sum(p.numel() for p in module.parameters()),
            "trainable_numel": sum(p.numel() for p in module.parameters() if p.requires_grad),
            "schedule": "ScheduleGPipe", "plan": batch, "records": records,
            "changed_adapter_tensors": changed, "adapter_keys": sorted(adapter),
            "frozen_before": frozen, "frozen_after": frozen_digest(module),
            "peak_vram": peak, "wall_seconds_with_correctness_probes": time.monotonic() - started,
            "rss_before_training": rss_before,
            "rss_after_training": psutil.Process().memory_info().rss,
            "host_used_ram_after_training": psutil.virtual_memory().used,
            "host_ram_method": "point samples, not peak or unique process totals",
            "precision": "IEEE FP32; eager attention; no AMP/TF32",
            "performance_measurement": False,
        })
        dist.barrier()
        if rank == 0:
            shards = [torch.load(output / f"rank-{index}/adapter.pt", weights_only=True)
                      for index in range(2)]
            merged = merge_adapter_shards(shards, set(initial))
            set_peft_model_state_dict(model, merged)
            model.save_pretrained(output / "adapter", safe_serialization=True)
            write_json(output / "environment.json", {
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                         text=True).strip(),
                "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "platform": platform.platform(),
                "command": sys.argv, "cpu_fixture": prepared["cpu_fixture"],
                "cuda": torch.version.cuda,
                "nccl": torch.cuda.nccl.version() if device_type == "cuda" else None,
                "devices": [torch.cuda.get_device_name(i) for i in range(2)]
                if device_type == "cuda" else ["cpu", "cpu"],
                "versions": {name: importlib.metadata.version(name)
                             for name in ("torch", "transformers", "peft")},
                "prepared": prepared,
            })
        dist.barrier()
    finally:
        dist.destroy_process_group()


def reload_adapter(root: Path, output: Path, device: str) -> None:
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    config = SoupConfig.model_validate_json((root / "config.json").read_text())
    verify_prepared(root, config)
    model = AutoModelForCausalLM.from_pretrained(config.base, local_files_only=True,
                                               dtype=torch.float32, attn_implementation="eager")
    model = PeftModel.from_pretrained(model, output / "adapter").to(device).eval()
    tokens = torch.load(output / "eval-input.pt", weights_only=True).to(device)
    expected = torch.load(output / "eval-before-save.pt", weights_only=True)
    with torch.no_grad():
        logits = model(tokens, use_cache=False).logits.cpu()
    torch.testing.assert_close(logits, expected, atol=1e-5, rtol=1e-5)
    merged = merge_adapter_shards([
        torch.load(output / f"rank-{index}/adapter.pt", weights_only=True) for index in range(2)
    ], set(load_file(output / "adapter/adapter_model.safetensors")))
    serialized = load_file(output / "adapter/adapter_model.safetensors")
    reloaded = get_peft_model_state_dict(model)
    if serialized.keys() != reloaded.keys():
        raise AssertionError("reloaded adapter keys differ")
    for name, value in merged.items():
        torch.testing.assert_close(serialized[name], value, atol=0, rtol=0)
        torch.testing.assert_close(reloaded[name].cpu(), value, atol=0, rtol=0)
    train_pids = {json.loads((output / f"rank-{i}/record.json").read_text())["pid"]
                  for i in range(2)}
    if os.getpid() in train_pids:
        raise AssertionError("reload must run in a separate process")
    write_json(output / "reload.json", {"pid": os.getpid(), "device": device,
                                        "max_logit_error": (logits - expected).abs().max().item(),
                                        "adapter_keys": len(merged), "passed": True})
    console.print(f"Separate-process adapter reload passed: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "train", "reload"))
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-fixture", action="store_true")
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.prepared, args.cpu_fixture)
    elif args.output is None:
        parser.error("launch/train/reload require --output")
    elif args.action == "launch":
        launch_training(args.prepared, args.output, args.device, args.steps)
    elif args.action == "train":
        train(args.prepared, args.output, args.device, args.steps)
    else:
        reload_adapter(args.prepared, args.output, args.device)


if __name__ == "__main__":
    main()
