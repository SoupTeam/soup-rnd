"""Four-layer NF4 streaming evidence against unpatched streaming and float64.

Both low-precision arms use one immutable shard set and two buffer slots.
Synthetic evidence only. The resident float64 oracle dequantizes those same shards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path
from typing import Any

from rich.console import Console

try:
    from . import fast_lora_probe as probe
except ImportError:
    import fast_lora_probe as probe


def _canonical(name: str) -> str:
    return name.replace(".inner.", ".")


def _adapters(model: Any) -> dict[str, Any]:
    return {_canonical(name): value for name, value in model.named_parameters() if "lora_" in name}


def _snapshot(model: Any, tokens: Any, labels: Any) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    output = model(input_ids=tokens, labels=labels)
    output.loss.backward()
    result = {"Y/logits": output.logits.detach().clone(), "Y/loss": output.loss.detach().clone()}
    for name, parameter in _adapters(model).items():
        probe.require_finite(parameter.grad, name)
        result[f"gradient/{name}"] = parameter.grad.detach().clone()
    for name, parameter in model.named_parameters():
        if "lora_" not in name:
            probe._require(parameter.grad is None, f"base gradient: {name}")
    return result


def run(destination: Path, device: str, dtype: str, report: dict[str, Any]) -> None:
    import bitsandbytes as bnb
    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file, save_file
    from transformers import LlamaConfig, LlamaForCausalLM

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model, rebuild_quant_state

    base_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    store_dtype = {"bf16": "bfloat16", "fp16": "float16"}[dtype]
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=False,
        max_position_embeddings=32,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    weights = destination / "model"
    weights.mkdir()
    original = LlamaForCausalLM(config).to(dtype=base_dtype)
    original.save_pretrained(weights)
    shards = destination / "shards"
    suffixes = [
        f"{group}.{name}.weight"
        for group, names in (
            ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
            ("mlp", ("gate_proj", "up_proj", "down_proj")),
        )
        for name in names
    ]
    index = shard_checkpoint(
        str(weights),
        str(shards),
        dtype=store_dtype,
        arch="llama",
        quant="nf4",
        quant_suffixes=suffixes,
        quant_device=device,
    )
    manifest = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in shards.glob("*.safetensors")
    }
    report["shard_sha256"] = manifest
    lora = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        target_modules=[name for names in probe.PROJECTIONS.values() for name in names],
        task_type="CAUSAL_LM",
    )
    models = {}
    runtimes = {}
    handles = []
    execution: dict[str, Any] = {}
    try:
        for arm in ("PEFT", "fast"):
            models[arm], runtimes[arm] = build_streamed_model(
                model_id=str(weights),
                shard_dir=str(shards),
                index=index,
                lora_config=lora,
                device=device,
                dtype=store_dtype,
                buffers=2,
                pin=False,
                seed=792,
                quant="nf4",
            )
        with torch.no_grad():
            for name, parameter in _adapters(models["PEFT"]).items():
                if "lora_B" in name:
                    parameter.normal_(std=0.1)
                _adapters(models["fast"])[name].copy_(parameter)
        report["patch_counts"] = {
            path: probe._patch(models["fast"], path) for path in ("qkv", "mlp", "single")
        }
        # Single-projection registration includes the three MLP siblings per layer;
        # their execution is owned by the fused MLP parent, as in the strict50 harness.
        probe._require(
            report["patch_counts"] == {"qkv": 4, "mlp": 4, "single": 16},
            "wrong streaming patch counts",
        )
        execution, handles = probe._execution_hooks(models["fast"])
        # Copy every decoder base directly from the shared shard store to a dense oracle.
        oracle = original.to(device=device, dtype=torch.float64)
        codes = load_file(str(shards / "extras.safetensors"), device=device)
        for layer_index, held_cpu in enumerate(runtimes["PEFT"].source.store[: index.n_layers]):
            held = {key: value.to(device) for key, value in held_cpu.items()}
            for key, parameter in oracle.model.layers[layer_index].named_parameters():
                if key not in held:
                    raise AssertionError(f"missing oracle shard weight {key}")
                value = held[key]
                if key in index.quant_specs:
                    state = rebuild_quant_state(key, held, index.quant_specs[key], codes)
                    value = bnb.functional.dequantize_4bit(value, state)
                with torch.no_grad():
                    parameter.copy_(value.double())
        oracle = get_peft_model(oracle, lora).double()
        oracle_adapters = _adapters(oracle)
        probe._require(
            oracle_adapters.keys() == _adapters(models["PEFT"]).keys(),
            "oracle adapter topology differs",
        )
        with torch.no_grad():
            for name, parameter in _adapters(models["PEFT"]).items():
                oracle_adapters[name].copy_(parameter)
        models["float64"] = oracle
        tokens = torch.randint(0, 64, (1, 8), device=device)
        labels = torch.randint(0, 64, (1, 8), device=device)
        snapshots = {arm: _snapshot(model, tokens, labels) for arm, model in models.items()}
        report["execution"] = execution
        for path, observed in execution.items():
            for name, seen in observed.items():
                probe._require(probe.EXPECTED_GRAD_FN[path] == seen, f"silent fallback: {name}")
        probe._require(
            sum(map(len, execution.values())) == 20, "insufficient streaming execution observations"
        )
        report["runtime_stats"] = {arm: runtime.stats() for arm, runtime in runtimes.items()}
        for stats in report["runtime_stats"].values():
            probe._require(
                stats["n_layers"] == 4 and stats["buffers"] == 2,
                "streaming did not use four layers and two slots",
            )
            probe._require(stats["layer_loads"] > 4, "checkpoint buffer reuse not observed")
        keys = snapshots["PEFT"].keys()
        probe._require(
            all(snapshot.keys() == keys for snapshot in snapshots.values()), "missing quantities"
        )
        probe._require(len(keys) == 58, "expected 56 layer adapter gradients and 2 outputs")
        artifact = destination / "streaming.safetensors"
        save_file(
            {
                f"{arm}/{key}": value.cpu().contiguous()
                for arm, snapshot in snapshots.items()
                for key, value in snapshot.items()
            },
            str(artifact),
        )
        report["tensor_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        for key in keys:
            report["rows"].append(
                {
                    "quantity": key,
                    **probe._comparison(
                        snapshots["fast"][key],
                        snapshots["PEFT"][key],
                        snapshots["float64"][key],
                        dtype,
                    ),
                }
            )
        report["loss_3f_equal"] = (
            f"{snapshots['fast']['Y/loss'].item():.3f}"
            == f"{snapshots['PEFT']['Y/loss'].item():.3f}"
        )
        probe._require(report["loss_3f_equal"], "streamed loss mismatch at three decimals")
        probe._require(
            manifest
            == {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in shards.glob("*.safetensors")
            },
            "shards mutated",
        )
        report["passed"] = True
    finally:
        for handle in handles:
            handle.remove()
        for runtime in runtimes.values():
            runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    destination = Path(os.path.realpath(args.output_dir))
    destination.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "passed": False,
        "rows": [],
        "fixture": "SYNTHETIC",
        "device": args.device,
        "dtype": args.dtype,
        "seed": 792,
    }
    try:
        report["environment"] = probe._environment(args.device)
        probe._validate_environment(report["environment"])
        probe.validate_runtime(args.device, args.dtype)
        previous_cwd = os.getcwd()
        try:
            # Keep the immutable checkout untouched; sharding is contained in
            # the explicitly requested evidence workspace, not the source tree.
            os.chdir(destination)
            with probe._deterministic(792, args.device):
                run(destination, args.device, args.dtype, report)
        finally:
            os.chdir(previous_cwd)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    (destination / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    Console().print(f"Streaming {'PASS' if report['passed'] else 'FAIL'}: {destination}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
