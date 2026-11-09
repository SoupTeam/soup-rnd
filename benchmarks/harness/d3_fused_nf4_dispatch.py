"""Development proof of actual fused launches and primitive allocation bounds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def compute_kernel_names(activities: list[str]) -> list[str]:
    """CUPTI memcpy activities are GPU work but are not compute kernel launches."""
    return [name for name in activities if not name.startswith(("Memcpy ", "Memset "))]


def run(out: Path) -> dict:
    import bitsandbytes as bnb
    import torch

    from soup_cli.utils.expert_group_fused import (
        PackedNF4,
        fused_group_linear,
        validate_fused_group,
    )

    torch.manual_seed(271828)
    torch.backends.cuda.preferred_blas_library("cublaslt")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parents[2]
    sources = [Path(__file__), root / "src/soup_cli/utils/expert_group_fused.py",
               root / "src/soup_cli/utils/expert_group_fused_wmma.py",
               root / "src/soup_cli/utils/expert_group_gemm.py"]
    report = {"kind": "development primitive dispatch/memory, not whole-model VRAM",
              "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sources}, "cells": [], "status": "incomplete"}
    try:
        for shape in ((2048, 2048), (2048, 1024)):
            for nested in (False, True):
                raw = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                packed, state = bnb.functional.quantize_4bit(
                    raw, blocksize=64, compress_statistics=nested, quant_type="nf4")
                # Distinct storage per expert: never rely on pointer aliasing for qualification.
                weights = [PackedNF4(
                    packed.flatten().clone(), state.absmax.clone(), state.code.clone(), shape,
                    nested_scales=state.state2.absmax.clone() if nested else None,
                    nested_code=state.state2.code.clone() if nested else None,
                    offset=state.offset.reshape(1).clone() if nested else None,
                ) for _ in range(4)]
                for transpose in (False, True):
                    inputs = [torch.randn(m, shape[0 if transpose else 1],
                                          device="cuda", dtype=torch.bfloat16)
                              for m in (129, 481, 129, 512)]
                    tensors = inputs + [t for w in weights for t in
                                        (w.packed, w.scales, w.code, w.nested_scales,
                                         w.nested_code, w.offset) if t is not None]
                    saved = [t.clone() for t in tensors]
                    kwargs = dict(transpose=transpose, backend="hybrid_wmma",
                                  call_budget_bytes=64 * 2**20)
                    plan = validate_fused_group(inputs, weights, **kwargs)
                    warmup = fused_group_linear(inputs, weights, **kwargs)
                    torch.cuda.synchronize()
                    del warmup
                    baseline = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    measured = fused_group_linear(inputs, weights, **kwargs)
                    torch.cuda.synchronize()
                    delta = torch.cuda.max_memory_allocated() - baseline
                    del measured
                    activities_to_capture = [torch.profiler.ProfilerActivity.CPU,
                                             torch.profiler.ProfilerActivity.CUDA]
                    with torch.profiler.profile(activities=activities_to_capture) as p:
                        profiled = fused_group_linear(inputs, weights, **kwargs)
                        torch.cuda.synchronize()
                    activities = [event.name for event in p.events()
                                  if "CUDA" in str(event.device_type)]
                    kernels = compute_kernel_names(activities)
                    name = f"{shape[0]}-{shape[1]}-nested{int(nested)}-dx{int(transpose)}"
                    trace = out / (name + ".json")
                    p.export_chrome_trace(str(trace))
                    unchanged = all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                                    for a, b in zip(tensors, saved))
                    cell = {"name": name, "plan": plan, "allocator_delta_bytes": delta,
                            "input_state_unchanged": unchanged, "kernels": kernels,
                            "gpu_activities": activities,
                            "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest()}
                    cell["pass"] = (kernels == ["fused"] and unchanged
                                    and delta <= plan["call_bytes"]
                                    and plan["dense_weight_bytes"] == 0)
                    report["cells"].append(cell)
                    del profiled, inputs, saved, tensors
        report["status"] = ("development_dispatch_pass" if len(report["cells"]) == 8
                            and all(c["pass"] for c in report["cells"]) else "fail")
    except BaseException as exc:
        report["status"] = "error"
        report["error"] = repr(exc)
        raise
    finally:
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    result = run(parser.parse_args().out)
    Console().print(f"Status: {result['status']}; cells: {len(result['cells'])}")
    raise SystemExit(0 if result["status"] == "development_dispatch_pass" else 1)
