"""Accuracy reference for the PR #72 A2 question: is row-wise GEMM more accurate than native?

One expert's GEMM is computed in float64 and compared with (a) native ``F.linear`` over portions
of ``portion_rows`` rows (what cuBLAS does at the grouped M) and (b) one call per row (M=1, the
invariant layer from ``p72_invariant_expert_gemm``). Max and mean absolute error against float64.
No pass/fail: invariance is constancy, not necessarily accuracy.

    python benchmarks/harness/p72_fp64_accuracy.py --synthetic --out report.json
    python benchmarks/harness/p72_fp64_accuracy.py --data expert.pt \
        --portion-rows 4096 --device cuda

``--data`` takes a torch file holding ``{"x": [rows, d_in], "w": [d_out, d_in]}``.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

NOISE_BAND = 0.10  # mean errors within 10% of each other count as noise


def classify(native_mean: float, invariant_mean: float) -> str:
    ratio = invariant_mean / native_mean if native_mean else float("inf")
    if ratio < 1 - NOISE_BAND:
        return "invariant точнее"
    if ratio > 1 + NOISE_BAND:
        return "invariant не точнее"
    return "в пределах шума"


def _stats(out: Any, reference: Any) -> Dict[str, float]:
    err = (out.double() - reference).abs()
    return {"max_abs": float(err.max()), "mean_abs": float(err.mean())}


def measure_gemm_accuracy(x: Any, w: Any, portion_rows: int) -> Dict[str, Dict[str, float]]:
    import torch
    import torch.nn.functional as functional

    if portion_rows < 1:
        raise ValueError("portion_rows must be >= 1")
    reference = x.double() @ w.double().T
    rows = x.shape[0]
    starts = range(0, rows, portion_rows)
    native = torch.cat([functional.linear(x[i : i + portion_rows], w) for i in starts])
    invariant = torch.cat([functional.linear(x[i : i + 1], w) for i in range(rows)])
    return {"native": _stats(native, reference), "invariant": _stats(invariant, reference)}


def run_report(x: Any, w: Any, portion_rows: int, source: str) -> Dict[str, Any]:
    import torch

    res = measure_gemm_accuracy(x, w, portion_rows)
    return {
        "source": source,
        "dtype": str(x.dtype),
        "device": str(x.device),
        "torch": torch.__version__,
        "rows": int(x.shape[0]),
        "portion_rows": portion_rows,
        **res,
        "verdict": classify(res["native"]["mean_abs"], res["invariant"]["mean_abs"]),
        "note": "reference only, no pass/fail",
    }


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true")
    src.add_argument("--data", help="torch file with {'x', 'w'} of one real expert")
    parser.add_argument("--portion-rows", type=int, default=64)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", help="write the JSON report here")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    if args.synthetic:
        gen = torch.Generator().manual_seed(0)
        x = torch.randn(512, 2048, generator=gen)
        w = torch.randn(1408, 2048, generator=gen) / 2048**0.5
        source = "synthetic"
    else:
        blob = torch.load(args.data, map_location="cpu", weights_only=True)
        x, w, source = blob["x"], blob["w"], args.data
    x, w = x.to(args.device, dtype), w.to(args.device, dtype)
    report = run_report(x, w, args.portion_rows, source)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
