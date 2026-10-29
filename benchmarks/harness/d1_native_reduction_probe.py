"""Single native-default GEMM diagnosis; no training or performance claim."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from d1_rules import compare_tensors, digest, write_new_json
from rich.console import Console


def run(output: Path) -> dict[str, Any]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import bitsandbytes as bnb
    import bitsandbytes.functional as functional
    import torch
    from d1_wmma_nf4 import matmul

    if not torch.cuda.is_available():
        raise RuntimeError("physical CUDA required")
    torch.manual_seed(842)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    # The acceptance reference is native True throughout candidate comparisons.
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    m, n, k = 128, 32768, 12288
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") / k**0.5
    packed, state = functional.quantize_4bit(weight, quant_type="nf4", compress_statistics=True)
    del weight
    value = torch.randn(m, k, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    dy = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        native = bnb.matmul_4bit(value, packed.t(), quant_state=state)
        native.backward(dy)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(output / "native-trace.json"))
    trace = json.loads((output / "native-trace.json").read_text())
    kernels = [
        {"name": row["name"], "args": row.get("args", {})}
        for row in trace["traceEvents"]
        if row.get("cat") == "kernel"
    ]
    write_new_json(output / "native-kernels.json", kernels)
    saved = (packed, state.absmax, state.code, state.state2.absmax, state.state2.code, state.offset)
    meta = (n, k, state.blocksize, True, state.state2.blocksize, state.dtype)
    full = matmul(dy, saved, meta, True)
    comparison = compare_tensors({"dx": full}, {"dx": value.grad})
    dense = functional.dequantize_4bit(packed, state)
    direct = dy @ dense
    assert torch.equal(direct, value.grad), "direct GEMM does not reproduce native backward"
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    full_reference = dy @ dense
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.save(
        {
            "native_dx": value.grad.cpu(),
            "full_dx": full.cpu(),
            "full_reference_dx": full_reference.cpu(),
            "dy": dy.cpu(),
        },
        output / "witness.pt",
    )
    record = {
        "source_sha": os.environ.get("D1_SOURCE_SHA"),
        "shape": [m, n, k],
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "torch": torch.__version__,
        "bitsandbytes": bnb.__version__,
        "default_native_vs_full_candidate": comparison,
        "differing_dx": int(torch.count_nonzero(full != value.grad)),
        "full_mode_positive_control_exact": bool(torch.equal(full_reference, full)),
        "witness_sha256": digest(output / "witness.pt"),
        "native_trace_sha256": digest(output / "native-trace.json"),
        "reference_reduced_precision": True,
        "d1_complete": False,
        "training_run": False,
        "candidate_sha256": digest(Path(__file__).with_name("d1_wmma_nf4.py")),
    }
    write_new_json(output / "probe.json", record)
    Console().print(
        {
            "differing_dx": record["differing_dx"],
            "full_mode_control": record["full_mode_positive_control_exact"],
            "kernels": [
                {
                    "name": row["name"],
                    "grid": row["args"].get("grid"),
                    "block": row["args"].get("block"),
                }
                for row in kernels
            ],
        }
    )
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    run(args.output)
