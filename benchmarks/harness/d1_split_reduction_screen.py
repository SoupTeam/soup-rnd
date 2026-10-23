"""Native-default split-K diagnosis from the observed four-way CUDA launch."""

from __future__ import annotations

import argparse
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
    from d1_wmma_nf4 import compile_kernel, matmul

    torch.manual_seed(842)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    m, n, k = 128, 32768, 12288
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    packed, state = functional.quantize_4bit(weight, quant_type="nf4", compress_statistics=True)
    del weight
    value = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    native = bnb.matmul_4bit(value, packed.t(), quant_state=state)
    native.backward(dy)
    saved = (packed, state.absmax, state.code, state.state2.absmax, state.state2.code, state.offset)
    meta = (n, k, state.blocksize, True, state.state2.blocksize, state.dtype)
    rows = []
    witnesses = {"native_dx": value.grad.detach().cpu()}
    # Four native grid.z partitions -> contiguous K spans of 32768 / 4.
    # Three explicitly different epilogues, all tested against unchanged native True.
    for mode in (1, 2, 3):
        actual = matmul(
            dy, saved, meta, True, tile=32, reduction_tile=128, reduction_span=8192, round_mode=mode
        )
        row = {
            "round_mode": mode,
            "span": 8192,
            "comparison": compare_tensors({"dx": actual}, {"dx": value.grad}),
            "differing": int(torch.count_nonzero(actual != value.grad)),
        }
        rows.append(row)
        witnesses[f"mode{mode}_dx"] = actual.cpu()
        write_new_json(output / f"mode-{mode}.json", row)
        Console().print(row)
    # Same-input equivalence for faster staging, independent of default reduction diagnosis.
    same_order = []
    baseline = matmul(dy, saved, meta, True)
    for tile, reduction in ((32, 128), (64, 64)):
        actual = matmul(dy, saved, meta, True, tile=tile, reduction_tile=reduction)
        row = {
            "tile": tile,
            "reduction_tile": reduction,
            "same_order_exact": bool(torch.equal(actual, baseline)),
        }
        same_order.append(row)
        _driver, _module, _function, ptx, _log = compile_kernel(
            dy.dtype, 64, 256, tile=tile, reduction_tile=reduction
        )
        (output / f"tile-{tile}.ptx").write_bytes(ptx)
        Console().print(row)
    torch.save(witnesses, output / "witness.pt")
    result = {
        "source_sha": os.environ.get("D1_SOURCE_SHA"),
        "shape": [m, n, k],
        "reference_reduced_precision": True,
        "rounding_candidates": rows,
        "staging_equivalence": same_order,
        "witness_sha256": digest(output / "witness.pt"),
        "candidate_sha256": digest(Path(__file__).with_name("d1_wmma_nf4.py")),
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "d1_complete": False,
        "timing_run": False,
    }
    write_new_json(output / "split.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    run(args.output)
