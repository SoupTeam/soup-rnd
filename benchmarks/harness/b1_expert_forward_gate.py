"""B1.2 forward-only gate on pinned trained OLMoE layers; one cell per process.

Adapted from original Soup d500ae0d's expert_stream_gate.py. Requires committed
source/protocol and local checkpoint files. Does not download a model, train,
measure throughput or implement B1.3 backward. Every attempt is retained.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "benchmarks/protocols/b1.2-expert-forward/protocol.json"
SOURCES = (
    "benchmarks/harness/b1_expert_forward_gate.py",
    "benchmarks/protocols/b1.2-expert-forward/protocol.json",
    "src/soup_cli/utils/expert_stream.py",
    "src/soup_cli/utils/expert_shard.py",
    "src/soup_cli/utils/layer_shard.py",
    "src/soup_cli/utils/layer_stream_runtime.py",
    "src/soup_cli/utils/safetensors_reader.py",
)


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            value.update(chunk)
    return value.hexdigest()


def _frozen_source(root: Path) -> dict:
    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)

    if git("status", "--porcelain", "--untracked-files=normal").strip():
        raise ValueError("commit the source/protocol in a clean checkout before a formal gate")
    hashes = {
        name: hashlib.sha256(git("show", f"HEAD:{name}")).hexdigest() for name in SOURCES
    }
    return {"commit": git("rev-parse", "HEAD").decode().strip(), "git_blob_sha256": hashes}


def _checkpoint_identity(folder: Path, protocol: dict) -> dict:
    """Verify full bytes against prospectively pinned Hub metadata, not a folder name."""
    root = os.path.realpath(folder)
    observed = {}
    expected_shards = {
        name for name in protocol["checkpoint_files"] if name.endswith(".safetensors")
    }
    if {path.name for path in folder.glob("*.safetensors")} != expected_shards:
        raise ValueError("checkpoint shard list differs from the committed protocol")
    for name, expected in protocol["checkpoint_files"].items():
        if name != os.path.basename(name):
            raise ValueError("checkpoint names must be flat file names")
        # HF snapshots can link to ../blobs. These read-only, named files are
        # authorized by the explicit model-dir AND pinned full-content hashes.
        path = os.path.realpath(os.path.join(root, name))
        observed[name] = _sha256(Path(path))
        if observed[name] != expected:
            raise ValueError(f"checkpoint SHA-256 mismatch: {name}")
    return observed


def _layer_tensors(model_dir: Path, layer: int) -> dict:
    from safetensors import safe_open

    prefix = f"model.layers.{layer}."
    tensors = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(prefix) and (".mlp." in key or "post_attention" in key):
                    short = key[len(prefix):]
                    if short in tensors:
                        raise ValueError(f"duplicate source tensor: {key}")
                    tensors[short] = handle.get_tensor(key).clone()
    return tensors


def _experts_module(config: Any, tensors: dict, device: str, dtype: Any) -> Any:
    import torch
    from transformers.models.olmoe.modeling_olmoe import OlmoeExperts

    config._experts_implementation = "eager"
    module = OlmoeExperts(config)
    with torch.no_grad():
        for expert in range(config.num_experts):
            gate = tensors[f"mlp.experts.{expert}.gate_proj.weight"]
            up = tensors[f"mlp.experts.{expert}.up_proj.weight"]
            down = tensors[f"mlp.experts.{expert}.down_proj.weight"]
            module.gate_up_proj[expert].copy_(torch.cat((gate, up)))
            module.down_proj[expert].copy_(down)
    return module.requires_grad_(False).eval().to(device=device, dtype=dtype)


def _nf4_reference(module: Any, reader: Any, codes: dict, device: str) -> None:
    import torch
    from bitsandbytes.functional import dequantize_4bit
    from safetensors import safe_open

    from soup_cli.utils.expert_shard import EXPERT_PARTS, expert_key
    from soup_cli.utils.layer_stream_runtime import quant_sidecar_keys, rebuild_quant_state

    tables = {key: value.to(device) for key, value in codes.items() if value is not None}
    with torch.no_grad(), safe_open(reader.path, framework="pt", device="cpu") as handle:
        for expert in range(reader.count):
            for part, target in zip(EXPERT_PARTS, (module.gate_up_proj, module.down_proj)):
                key = expert_key(expert, part)
                spec = reader.quant_specs[part]
                stored = {
                    name: handle.get_tensor(name).to(device)
                    for name in quant_sidecar_keys(key, spec)
                }
                state = rebuild_quant_state(key, stored, spec, tables)
                target[expert].copy_(dequantize_4bit(stored[key], quant_state=state))


def _memory(device: str) -> dict:
    import torch

    if device == "cpu":
        return {"cuda": None}
    torch.cuda.synchronize()
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
    }


def _reset_peak(device: str) -> None:
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _tensor_hash(tensor: Any) -> str:
    import torch

    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _run_cell(cell: str, layer: int, config: Any, tensors: dict, folder: Path, rule: dict) -> dict:
    import torch
    from transformers.models.olmoe.modeling_olmoe import OlmoeTopKRouter

    from soup_cli.utils.expert_shard import ExpertShardReader, write_expert_shard
    from soup_cli.utils.expert_stream import ExpertBufferPool, streamed_experts_class
    from soup_cli.utils.layer_shard import NF4_CODE_KEY, NF4_NESTED_CODE_KEY

    device = "cpu" if cell == "cpu-fp32" else "cuda"
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    quantise = "nf4" in cell
    nested = cell.endswith("-dq")
    module = _experts_module(config, tensors, device, dtype)
    path = folder / "experts.safetensors"
    info = write_expert_shard(
        str(path), ((gate.detach(), down.detach()) for gate, down in zip(
            module.gate_up_proj, module.down_proj
        )),
        quantise=quantise, double_quant=nested, device=device,
    )
    codes = {NF4_CODE_KEY: info.codes[0], NF4_NESTED_CODE_KEY: info.codes[1]}
    reader = ExpertShardReader(
        str(path), count=info.count, shapes=info.shapes, quant_specs=info.quant_specs,
        dtype=str(dtype).removeprefix("torch."),
    )
    pool = None
    streamed = None
    try:
        if quantise:
            _nf4_reference(module, reader, codes, device)
        gen = torch.Generator().manual_seed(rule["input_seed"])
        states = torch.randn(rule["tokens"], config.hidden_size, generator=gen)
        states = (states * tensors["post_attention_layernorm.weight"].float()).to(device, dtype)
        router = OlmoeTopKRouter(config).requires_grad_(False).to(device=device, dtype=dtype)
        with torch.no_grad():
            router.weight.copy_(tensors["mlp.gate.weight"])
            _, weights, index = router(states)
        hit = sorted(set(index.cpu().flatten().tolist()))
        _reset_peak(device)
        resident_before = _memory(device)
        with torch.no_grad():
            expected = module(states, index, weights).cpu()
        resident_after = _memory(device)
        with torch.no_grad():
            module.down_proj[hit[0]].add_(rule["negative_control_down_delta"])
            negative = module(states, index, weights).cpu()
        # The streamed memory arm must not retain the resident expert stack.
        act_fn = module.act_fn
        del module, router
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        _reset_peak(device)
        streamed_before = _memory(device)
        pool = ExpertBufferPool(
            reader, device=device, dtype=reader.dtype,
            weight_budget_bytes=rule["weight_budget_bytes"], codes=codes,
        )
        streamed = streamed_experts_class()(reader, pool, act_fn, layer)
        with torch.no_grad():
            observed = streamed(states, index, weights).cpu()
        streamed_after = _memory(device)
        equal = bool(torch.equal(observed, expected))
        negative_differs = not torch.equal(negative, expected)
        order = [row["expert"] for row in streamed.last_trace]
        loads_ok = reader.fetches == len(hit) and order == hit
        finite = bool(torch.isfinite(expected).all() and torch.isfinite(observed).all())
        budget_ok = (
            pool.plan.weight_bytes <= rule["weight_budget_bytes"]
            and pool.cuda_allocation_delta <= rule["weight_budget_bytes"]
        )
        return {
            "verdict": "PASS" if all((equal, negative_differs, loads_ok, finite, budget_ok))
            else "FAIL",
            "equal": equal, "negative_differs": negative_differs, "finite": finite,
            "loads_ok": loads_ok, "budget_ok": budget_ok,
            "max_abs_diff": float((observed.float() - expected.float()).abs().max())
            if finite else None,
            "expected_sha256": _tensor_hash(expected),
            "observed_sha256": _tensor_hash(observed),
            "negative_sha256": _tensor_hash(negative),
            "routing_indices": index.cpu().tolist(), "routing_weights": weights.cpu().tolist(),
            "trace": streamed.last_trace, "pool": pool.stats(),
            "resident_memory": {"before": resident_before, "after": resident_after},
            "streamed_memory": {"before": streamed_before, "after": streamed_after},
            "experts_file_sha256": _sha256(path), "reader_direct_io": reader.direct,
            "input_kind": "synthetic normal hidden states scaled by trained layer norm",
            "comparison": "trained routed experts and real router; forward only",
        }
    finally:
        try:
            progress = {
                "reader_fetches": reader.fetches, "logical_bytes_read": reader.bytes_read,
                "trace": streamed.last_trace if streamed is not None else [],
                "pool": pool.stats() if pool is not None else None,
            }
            (folder / "progress.json").write_text(
                json.dumps(progress, indent=2) + "\n", encoding="utf-8"
            )
        finally:
            reader.close()
            if pool is not None:
                pool.close()


def _box() -> dict:
    import torch

    box = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "transformers", "bitsandbytes", "safetensors"):
        box[name] = importlib.metadata.version(name)
    box["cuda"] = torch.version.cuda
    if torch.cuda.is_available():
        box["gpu"] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        box["gpu_free_bytes"], box["gpu_total_bytes"] = free, total
        try:
            box["nvidia_smi"] = subprocess.check_output(
                ["nvidia-smi"], text=True, stderr=subprocess.STDOUT, timeout=15
            )
        except (OSError, subprocess.SubprocessError) as exc:
            box["nvidia_smi_error"] = str(exc)
    return box


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--power-source", choices=("ac", "server"), required=True)
    args = parser.parse_args(argv)
    console = Console()
    source = _frozen_source(ROOT)
    rule = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if args.layer not in rule["layers"] or args.cell not in rule["cells"]:
        parser.error("cell/layer is outside the committed protocol")
    # Fixed budget: one attempt for each (layer, cell). Never reuse/overwrite it.
    folder = args.session / f"layer-{args.layer:02d}-{args.cell}"
    folder.mkdir(parents=True, exist_ok=False)
    report = {
        "source": source, "protocol": rule, "layer": args.layer, "cell": args.cell,
        "power_source": f"operator-declared-{args.power_source}",
        "started_utc": datetime.now(timezone.utc).isoformat(), "verdict": "NO_VERDICT",
        "limitations": rule["limitations"],
    }
    try:
        import torch
        from transformers import OlmoeConfig

        report["box"] = _box()
        for name, version in rule["package_versions"].items():
            if report["box"][name].split("+")[0] != version:
                raise RuntimeError(f"package differs from the committed protocol: {name}")
        report["checkpoint_sha256"] = _checkpoint_identity(args.model_dir, rule)
        if args.cell.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA cell requested but no CUDA device is available")
        if args.cell.startswith("cuda") and (
            torch.cuda.get_device_capability(0)[0] < 8
            or not torch.cuda.is_bf16_supported(including_emulation=False)
        ):
            raise RuntimeError("native CUDA BF16 is unsupported on this device")
        torch.backends.cuda.matmul.allow_tf32 = False
        config = OlmoeConfig.from_pretrained(args.model_dir, local_files_only=True)
        tensors = _layer_tensors(args.model_dir, args.layer)
        report["result"] = _run_cell(
            args.cell, args.layer, config, tensors, folder, rule
        )
        report["verdict"] = report["result"]["verdict"]
    except (Exception, KeyboardInterrupt) as exc:
        # Exceptions do not pass acceptance. Retain the attempt before fixing or retrying.
        import traceback

        report["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        report["traceback"] = traceback.format_exc()
        if isinstance(exc, ValueError) and (
            str(exc).startswith("expert slot needs") or "above the weight budget" in str(exc)
        ):
            report["verdict"] = "FAIL"
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (folder / "result.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    console.print(f"{report['verdict']}: {folder / 'result.json'}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
