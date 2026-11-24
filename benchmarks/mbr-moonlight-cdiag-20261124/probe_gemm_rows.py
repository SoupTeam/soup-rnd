"""Op-level probe of the expert projections behind ticket 14 (one card, strict deterministic).

Replays the two expert GEMMs of Moonlight-16B-A3B (``gate_up`` 2816x2048, ``down`` 2048x1408,
bf16) at the row counts the capture saw, prints the cuBLAS kernel behind every call, and measures
which row-count transitions change bits. Then it checks the candidate exact variants: per-row
calls, per-micro-batch calls, zero padding to a shared row count, and
``allow_bf16_reduced_precision_reduction`` off. Writes ``probe.md`` and ``probe.json``; every line
names model, card, window, arm, mode, torch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_AC_ONLINE = Path("/sys/class/power_supply/ACAD/online")
_MODEL = "Moonlight-16B-A3B"
_HIDDEN, _INTER, _GATE_UP = 2048, 1408, 2816
#: The pair the k=2 capture named: A0 called the expert once per micro-batch with M = 15 and 16,
#: C once with M = 31. Also the M=1 gemv case and a pair that lands in one kernel bucket.
_PAIRS = ((15, 31), (1, 2), (8, 16))
_SWEEP = (1, 2, 4, 8, 12, 15, 16, 17, 24, 31, 32, 33, 40, 48, 64)


def _diff(left: Any, right: Any) -> dict[str, Any]:
    import torch

    delta = (left.double() - right.double()).abs()
    return {
        "equal": torch.equal(left, right),
        "n_diff": int((left != right).count_nonzero().item()),
        "numel": int(left.numel()),
        "max_abs_diff": delta.max().item() if left.numel() else 0.0,
    }


def _kernels(weight: Any, inputs: Any, rows: int) -> list[str]:
    import torch
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as profiler:
        torch.nn.functional.linear(inputs[:rows], weight)
        torch.cuda.synchronize()
    return sorted({
        event.key for event in profiler.key_averages()
        if event.device_type == torch.autograd.DeviceType.CUDA
    })


def _padded(inputs: Any, weight: Any, pad: int, nonzero: int) -> Any:
    import torch

    block = torch.zeros(pad, weight.shape[1], dtype=weight.dtype, device=inputs.device)
    block[:nonzero] = inputs[:nonzero]
    return block


def _projection_probe(weight: Any, inputs: Any) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    def call(rows: int) -> Any:
        return functional.linear(inputs[:rows], weight)

    rows: dict[str, Any] = {
        "shape": list(weight.shape),
        "kernels": {str(m): _kernels(weight, inputs, m) for m in _SWEEP},
        "buckets": [],
        "pairs": [],
        "per_row": [],
        "per_micro_batch": [],
        "padded": [],
        "reduced_precision_off": [],
    }
    previous_kernel = None
    for m in _SWEEP:
        kernel = ",".join(rows["kernels"][str(m)])
        if kernel != previous_kernel:
            rows["buckets"].append({"first_m": m, "kernel": kernel})
            previous_kernel = kernel
    for small, large in _PAIRS:
        rows["pairs"].append({
            "m_small": small, "m_large": large, **_diff(call(small), call(large)[:small]),
        })
    for small, large in _PAIRS:
        joined = call(large)[:small]
        per_row = torch.cat([functional.linear(inputs[index : index + 1], weight)
                             for index in range(small)])
        rows["per_row"].append({
            "m_small": small, "m_large": large, **_diff(per_row, joined),
        })
    for small, total in ((15, 31),):
        joined = call(total)
        split = torch.cat([functional.linear(inputs[:small], weight),
                           functional.linear(inputs[small:total], weight)])
        rows["per_micro_batch"].append({
            "split": [small, total - small], "m_large": total, **_diff(split, joined),
        })
    for pad in (32, 512):
        rows["padded"].append({
            "pad": pad, **_diff(
                functional.linear(_padded(inputs, weight, pad, 1), weight)[:1],
                functional.linear(_padded(inputs, weight, pad, 2), weight)[:1],
            ),
        })
    previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    try:
        for small, large in _PAIRS[:2]:
            rows["reduced_precision_off"].append({
                "m_small": small, "m_large": large,
                **_diff(call(small), call(large)[:small]),
            })
        rows["reduced_precision_off_kernels"] = {
            str(m): _kernels(weight, inputs, m) for m in (1, 2, 15, 16, 31, 32)
        }
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous
    return rows


def probe(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.use_deterministic_algorithms(True)
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(0)
    weights = {
        "gate_up": torch.randn(_GATE_UP, _HIDDEN, generator=generator, device=device)
        .div(50).to(torch.bfloat16),
        "down": torch.randn(_HIDDEN, _INTER, generator=generator, device=device)
        .div(50).to(torch.bfloat16),
    }
    hidden = torch.randn(64, _HIDDEN, generator=generator, device=device).div(2).to(torch.bfloat16)
    activated = (
        torch.randn(64, _INTER, generator=generator, device=device).div(2).to(torch.bfloat16)
    )
    inputs = {"gate_up": hidden, "down": activated}
    return {
        "model": _MODEL,
        "card": torch.cuda.get_device_name(),
        "window": "op probe, no model",
        "arm": "operation",
        "mode": "bits (strict deterministic)",
        "torch_version": torch.__version__,
        "cuda": torch.version.cuda,
        "reduce_precision_reduction": bool(
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "projections": {
            projection: _projection_probe(weight, inputs[projection])
            for projection, weight in weights.items()
        },
    }


def _report(result: dict[str, Any]) -> str:
    lines = [
        f"# Op probe: expert GEMM row counts ({result['model']}, {result['card']})",
        "",
        f"Window: {result['window']}; arm: {result['arm']}; mode: {result['mode']}; "
        f"torch {result['torch_version']} (CUDA {result['cuda']}).",
        "",
        "## Kernel buckets",
        "",
        "The first row count of each cuBLAS kernel bucket, a kernel with another stage count "
        "being a different reduction order:",
        "",
    ]
    for projection, rows in result["projections"].items():
        lines.extend([
            f"### {projection} {tuple(rows['shape'])}",
            "",
            "| first M in bucket | kernel |",
            "|---|---|",
            *(f"| {bucket['first_m']} | {bucket['kernel']} |" for bucket in rows["buckets"]),
            "",
            "The first rows of a small call against the same rows inside a larger call:",
            "",
            "| M small | M large | equal | n_diff / numel | max abs diff |",
            "|---|---|---|---|---|",
            *(
                f"| {pair['m_small']} | {pair['m_large']} | {pair['equal']} | "
                f"{pair['n_diff']} / {pair['numel']} | {pair['max_abs_diff']:.6g} |"
                for pair in rows["pairs"]
            ),
            "",
            "Per-row calls against the joined call (the row-wise exact variant):",
            "",
            "| M small | M large | equal | n_diff / numel | max abs diff |",
            "|---|---|---|---|---|",
            *(
                f"| {pair['m_small']} | {pair['m_large']} | {pair['equal']} | "
                f"{pair['n_diff']} / {pair['numel']} | {pair['max_abs_diff']:.6g} |"
                for pair in rows["per_row"]
            ),
            "",
            "Per-micro-batch calls against the joined call (the segmented exact variant):",
            "",
            "| split | M large | equal | n_diff / numel | max abs diff |",
            "|---|---|---|---|---|",
            *(
                f"| {pair['split']} | {pair['m_large']} | {pair['equal']} | "
                f"{pair['n_diff']} / {pair['numel']} | {pair['max_abs_diff']:.6g} |"
                for pair in rows["per_micro_batch"]
            ),
            "",
            "Zero padding to a shared row count, one nonzero row against two:",
            "",
            "| pad | equal | n_diff / numel | max abs diff |",
            "|---|---|---|---|",
            *(
                f"| {pair['pad']} | {pair['equal']} | {pair['n_diff']} / {pair['numel']} | "
                f"{pair['max_abs_diff']:.6g} |"
                for pair in rows["padded"]
            ),
            "",
            "`allow_bf16_reduced_precision_reduction = False`:",
            "",
            "| pair | equal | n_diff / numel | max abs diff |",
            "|---|---|---|---|",
            *(
                f"| {pair['m_small']} vs {pair['m_large']} | {pair['equal']} | "
                f"{pair['n_diff']} / {pair['numel']} | {pair['max_abs_diff']:.6g} |"
                for pair in rows["reduced_precision_off"]
            ),
            "",
            f"Kernels with the flag off: `{rows['reduced_precision_off_kernels']}`",
            "",
        ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the probe and write its JSON and Markdown."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if int(_AC_ONLINE.read_text(encoding="ascii").strip()) != 1:
        print("refusing to run: AC power is not online", file=sys.stderr)
        return 2
    result = probe(args)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "probe.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.out / "probe.md").write_text(_report(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = (str(repo / "src"), str(repo))
    raise SystemExit(main())
