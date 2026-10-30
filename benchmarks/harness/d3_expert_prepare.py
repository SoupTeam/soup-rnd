"""D3 preparation: reuse B1.2 controls; never run a formal grouped-kernel gate.

CPU control uses small synthetic OLMoE experts. CUDA preflight additionally
checks native BF16 and exported device-kernel tracing. Neither mode downloads
weights, benchmarks throughput, changes training defaults or certifies D3.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "benchmarks/protocols/d3-grouped-experts/protocol-draft.json"
CASES = ("balanced", "skewed-unused", "all-experts", "empty")


def group_plan(active: list[int], *, width: int, slot_bytes: int, budget: int) -> dict:
    """Conservative replicated B1.2 slots, not a grouped allocator implementation.

    A future kernel must separately bound routing/activation/workspace storage.
    Shared code tables are conservatively counted once per slot in this design.
    """
    values = (width, slot_bytes, budget)
    if any(type(value) is not int or value < 1 for value in values):
        raise ValueError("width, slot bytes and budget must be positive integers")
    if any(type(expert) is not int or expert < 0 for expert in active):
        raise ValueError("active expert IDs must be nonnegative integers")
    if active != sorted(set(active)):
        raise ValueError("active experts must be unique and ascending")
    slots = min(width, len(active))
    required = slots * slot_bytes
    if required > budget:
        raise ValueError(f"group slots need {required} bytes, above weight budget {budget}")
    return {
        "groups": [active[start:start + width] for start in range(0, len(active), width)],
        "slots": slots, "weight_bytes_upper_bound": required,
        "weight_budget_bytes": budget,
        "scope": "replicated weight slots only; not total VRAM or an actual allocation",
    }


def preparation_plan() -> dict:
    rule = json.loads(DRAFT.read_text(encoding="utf-8"))
    if rule["status"] != "draft" or rule["formal_execution_enabled"] is not False:
        raise ValueError("this preparation tool accepts only a disabled draft protocol")
    return {
        "kind": "D3 preparation plan, not measured evidence",
        "formal_execution_enabled": False,
        "candidate_implemented": True,
        "candidate_acceptance_complete": False,
        "grouped_restore_component_implemented": True,
        "grouped_gemm_component_implemented": True,
        "next_gate": "operator source/protocol commit before formal paired layer performance",
        "reuse": rule["reuse"], "hardware": rule["hardware"],
        "correctness_matrix": rule["correctness_matrix"],
        "remaining_before_formal_run": rule["remaining_before_formal_run"],
    }


def _baseline() -> Any:
    # Reuse checkpoint decoding, resident construction and tensor hashing from
    # the audited B1.2 harness rather than duplicating the codec/reference.
    import importlib.util

    path = ROOT / "benchmarks/harness/b1_expert_forward_gate.py"
    spec = importlib.util.spec_from_file_location("d3_b12_baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture() -> tuple[Any, dict]:
    import torch
    from transformers import OlmoeConfig

    generator = torch.Generator().manual_seed(173)
    config = OlmoeConfig(
        hidden_size=32, intermediate_size=16, num_experts=8, num_experts_per_tok=2,
    )
    tensors = {}
    for expert in range(config.num_experts):
        for name, shape in (("gate", (16, 32)), ("up", (16, 32)), ("down", (32, 16))):
            tensors[f"mlp.experts.{expert}.{name}_proj.weight"] = (
                torch.randn(shape, generator=generator) * 0.2
            )
    return config, tensors


def routing_fixture(case: str, *, device: str, dtype: Any) -> tuple:
    """Nontrivial weights and repeatable, unequal traffic; zero-token case explicit."""
    import torch

    if case not in CASES:
        raise ValueError(f"unknown routing case: {case}")
    tokens = 0 if case == "empty" else 17
    topk = 8 if case == "all-experts" else 2
    generator = torch.Generator().manual_seed(29)
    hidden = torch.randn(tokens, 32, generator=generator).to(device=device, dtype=dtype)
    index = torch.arange(tokens * topk).reshape(tokens, topk) % 8
    if case == "skewed-unused":
        index[:, 0] = 0
        index[:, 1] = 1
        index[::4, 1] = 3
    weights = torch.rand(tokens, topk, generator=generator) * 0.8 + 0.1
    return hidden, index.to(device), weights.to(device=device, dtype=dtype)


def exact_control(expected: Any, repeat: Any, observed: Any, negative: Any) -> dict:
    import torch

    compatible = all(value.shape == expected.shape and value.dtype == expected.dtype
                     for value in (repeat, observed, negative))
    if not compatible:
        raise ValueError("control outputs differ in shape or dtype")
    finite = all(bool(torch.isfinite(value).all())
                 for value in (expected, repeat, observed, negative))
    empty = expected.numel() == 0
    nonzero = empty or bool(torch.count_nonzero(expected))
    equal = torch.equal(expected, observed)
    repeat_equal = torch.equal(expected, repeat)
    negative_differs = None if empty else not torch.equal(expected, negative)
    passed = finite and nonzero and equal and repeat_equal and (empty or negative_differs)
    return {
        "passed": bool(passed), "finite": finite, "non_vacuous": nonzero,
        "equal": equal, "repeat_equal": repeat_equal,
        "negative_differs": negative_differs,
        "empty_control": empty,
        "max_abs_diff": float((expected.float() - observed.float()).abs().max())
        if finite and not empty else (0.0 if empty else None),
    }


def baseline_control(folder: Path, *, device: str, codec: str, case: str) -> dict:
    """Native eager versus unchanged B1.2: this is NOT the D3 candidate."""
    import torch

    from soup_cli.utils.expert_shard import ExpertShardReader, write_expert_shard
    from soup_cli.utils.expert_stream import ExpertBufferPool, streamed_experts_class
    from soup_cli.utils.layer_shard import NF4_CODE_KEY, NF4_NESTED_CODE_KEY

    if codec not in ("dense", "nf4", "nf4-dq"):
        raise ValueError("unknown codec")
    if device not in ("cpu", "cuda") or (device == "cpu" and codec != "dense"):
        raise ValueError("CPU preparation uses dense fp32; NF4 preflight requires CUDA")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    config, tensors = _fixture()
    baseline = _baseline()
    module = baseline._experts_module(config, tensors, device, dtype)
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / "experts.safetensors"
    info = write_expert_shard(
        str(path), ((gate.detach(), down.detach()) for gate, down in zip(
            module.gate_up_proj, module.down_proj
        )), quantise=codec != "dense", double_quant=codec == "nf4-dq", device=device,
    )
    reader = ExpertShardReader(
        str(path), count=info.count, shapes=info.shapes, quant_specs=info.quant_specs,
        dtype=str(dtype).removeprefix("torch."), direct=False,
    )
    codes = {NF4_CODE_KEY: info.codes[0], NF4_NESTED_CODE_KEY: info.codes[1]}
    pool = None
    try:
        if codec != "dense":
            baseline._nf4_reference(module, reader, codes, device)
        pool = ExpertBufferPool(
            reader, device=device, dtype=reader.dtype, weight_budget_bytes=2**20, codes=codes,
        )
        streamed = streamed_experts_class()(reader, pool, module.act_fn, 0)
        args = routing_fixture(case, device=device, dtype=dtype)
        with torch.no_grad():
            expected = module(*args).cpu()
            repeat = module(*args).cpu()
            observed = streamed(*args).cpu()
            if args[1].numel():
                module.down_proj[int(args[1].min())].add_(0.125)
            negative = module(*args).cpu()
        result = exact_control(expected, repeat, observed, negative)
        active = sorted(set(args[1].cpu().flatten().tolist()))
        trace = streamed.last_trace
        loads_ok = (
            [row["expert"] for row in trace] == active
            and reader.fetches == len(active)
            and sum(row["token_routes"] for row in trace) == args[1].numel()
            and all(row["status"] == "ok" for row in trace)
        )
        result.update(
            kind="synthetic unchanged-B1.2 instrument control; not grouped D3",
            codec=codec, case=case, loads_ok=loads_ok,
            expected_sha256=baseline._tensor_hash(expected),
            observed_sha256=baseline._tensor_hash(observed),
            routing_indices_sha256=baseline._tensor_hash(args[1].cpu()),
            routing_weights_sha256=baseline._tensor_hash(args[2].cpu()),
            traffic=dict(sorted(Counter(args[1].cpu().flatten().tolist()).items())),
            trace=trace, pool=pool.stats(), physical_io_claim=False,
        )
        result["passed"] = bool(result["passed"] and loads_ok)
        return result
    finally:
        try:
            (folder / "progress.json").write_text(json.dumps({
                "fetches": reader.fetches, "logical_bytes": reader.bytes_read,
                "pool": pool.stats() if pool is not None else None,
            }, indent=2) + "\n", encoding="utf-8")
        finally:
            reader.close()
            if pool is not None:
                pool.close()


def kernel_trace_summary(trace: dict) -> dict:
    """Count exported CUDA kernels, never CPU operator/key_averages counts."""
    counts: Counter = Counter()
    for event in trace.get("traceEvents", []):
        categories = str(event.get("cat", "")).lower().split(",")
        if event.get("ph") == "X" and "kernel" in map(str.strip, categories):
            duration = event.get("dur")
            if (type(duration) not in (int, float) or not math.isfinite(duration)
                    or duration < 0):
                raise ValueError("kernel trace has an invalid duration")
            counts[str(event.get("name", "<unnamed>"))] += 1
    return {"cuda_kernel_events": sum(counts.values()), "kernels": dict(counts),
            "usable": bool(counts)}


def cuda_preflight(folder: Path) -> dict:
    import torch

    if sys.platform != "linux" or not torch.cuda.is_available():
        raise RuntimeError("D3 CUDA preflight requires Linux and an available CUDA GPU")
    if torch.cuda.get_device_capability(0)[0] < 8 or not torch.cuda.is_bf16_supported(
        including_emulation=False
    ):
        raise RuntimeError("D3 requires sm_80+ and native BF16")
    triton_version = importlib.metadata.version("triton")
    from torch.profiler import ProfilerActivity, profile

    tensor = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
    tensor @ tensor
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        tensor @ tensor
        torch.cuda.synchronize()
    path = folder / "cuda-preflight-trace.json"
    prof.export_chrome_trace(str(path))
    summary = kernel_trace_summary(json.loads(path.read_text(encoding="utf-8")))
    if not summary["usable"]:
        raise RuntimeError("no device kernels exported; CUPTI launch counting is unavailable")
    return {
        "gpu": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0),
        "total_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        "end_to_end_memory_requirement_met": (
            torch.cuda.get_device_properties(0).total_memory >= 16 * 10**9
        ),
        "triton_version": triton_version, "trace": summary,
        "triton_candidate_compiled": False,
        "scope": "BF16/profiler and B1.2 readiness only; no grouped kernel tested",
    }


def _sources() -> dict:
    names = (
        "benchmarks/harness/d3_expert_prepare.py",
        "benchmarks/protocols/d3-grouped-experts/protocol-draft.json",
        "benchmarks/harness/b1_expert_forward_gate.py",
        "src/soup_cli/utils/expert_stream.py", "src/soup_cli/utils/expert_shard.py",
        "src/soup_cli/utils/layer_shard.py", "src/soup_cli/utils/layer_stream_runtime.py",
        "src/soup_cli/utils/safetensors_reader.py",
    )
    return {
        "git_head": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_file_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names
        },
        "identity_scope": "working-file bytes; not a frozen formal-run commit",
    }


def main(argv: list[str] | None = None) -> int:
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("plan", "cpu-control", "cuda-preflight"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "not_ready", "d3_accepted": False, "formal_measurement": False,
        "started_utc": datetime.now(timezone.utc).isoformat(), "mode": args.mode,
    }
    exit_code = 1
    try:
        report["plan"] = preparation_plan()
        report["source"] = _sources()
        report["environment"] = {"python": platform.python_version(), "platform": sys.platform}
        if args.mode == "plan":
            report["status"] = "plan_only"
        else:
            import torch

            report["environment"]["torch"] = torch.__version__
            codecs = ("dense",)
            device = "cpu"
            if args.mode == "cuda-preflight":
                report["readiness"] = cuda_preflight(args.out)
                codecs, device = ("dense", "nf4", "nf4-dq"), "cuda"
            report["controls"] = []
            for codec in codecs:
                for case in CASES:
                    result = baseline_control(
                        args.out / f"{codec}-{case}", device=device, codec=codec, case=case,
                    )
                    report["controls"].append(result)
                    if not result["passed"]:
                        raise RuntimeError(f"unchanged B1.2 control failed: {codec}/{case}")
            report["status"] = "instrument_controls_passed"
        exit_code = 0
    except (Exception, KeyboardInterrupt) as exc:
        report["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        report["traceback"] = traceback.format_exc()
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        target = args.out / "preparation.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    Console().print(f"{report['status']}: {target}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
