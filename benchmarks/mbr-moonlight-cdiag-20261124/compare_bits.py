"""Compare two saved bits runs of ticket 14 on the CPU: loss and all LoRA gradients.

A full step is bit-equal only when ``torch.equal`` holds for the loss and every gradient. The
report also gives the size of a difference: ``grad_rel_l2`` over all gradients, the largest
absolute difference, and the smallest cosine. Writes ``bits.json`` and ``bits.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cdiag


def _cosine(left: Any, right: Any) -> float:

    left64, right64 = left.double().flatten(), right.double().flatten()
    denominator = left64.norm().item() * right64.norm().item()
    return float((left64 @ right64).item() / denominator) if denominator else 1.0


def compare(paths: tuple[Path, Path], labels: tuple[str, str]) -> dict[str, Any]:
    import torch

    runs = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    left, right = runs
    if sorted(left["grads"]) != sorted(right["grads"]):
        raise ValueError("gradient names differ between the two runs")
    loss = cdiag.tensor_pair_diff(
        left["loss"].double(), right["loss"].double()
    )
    diffs = {
        name: {
            **cdiag.tensor_pair_diff(left["grads"][name], right["grads"][name]),
            "cos": _cosine(left["grads"][name], right["grads"][name]),
        }
        for name in sorted(left["grads"])
    }
    squared = {
        name: float((left["grads"][name].double() - right["grads"][name].double()).pow(2).sum())
        for name in diffs
    }
    reference = sum(float(left["grads"][name].double().pow(2).sum()) for name in diffs)
    differing = [name for name, diff in diffs.items() if not diff["bitwise_equal"]]
    return {
        "left": labels[0], "right": labels[1],
        "loss_bitwise_equal": loss["bitwise_equal"],
        "loss_left": float(left["loss"].sum()), "loss_right": float(right["loss"].sum()),
        "grad_total": len(diffs), "grad_differing": len(differing),
        "differing_names": differing,
        "max_abs_diff": max(diff["max_abs_diff"] for diff in diffs.values()),
        "max_abs": max(diff["max_abs"] for diff in diffs.values()),
        "grad_rel_l2": math.sqrt(sum(squared.values()) / reference) if reference else 0.0,
        "min_cos": min(diff["cos"] for diff in diffs.values()),
        "per_gradient": diffs,
    }


def main(argv: list[str] | None = None) -> int:
    """Compare two tensor files and write the JSON and Markdown reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--labels", default="left,right")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    labels = tuple(part.strip() for part in args.labels.split(","))
    if len(labels) != 2:
        parser.error("--labels takes two comma-separated names")
    report = compare((args.left, args.right), labels)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "bits.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    line = (
        f"| {labels[0]} vs {labels[1]} | {report['loss_bitwise_equal']} | "
        f"{report['grad_differing']} / {report['grad_total']} | "
        f"{report['max_abs_diff']:.3e} | {report['grad_rel_l2']:.3e} | "
        f"{report['min_cos']:.6f} |"
    )
    header = (
        "| pair | loss equal | grads differing | max abs diff | grad_rel_l2 | min cos |\n"
        "|---|---|---|---|---|---|\n"
    )
    (args.out / "bits.md").write_text(header + line + "\n", encoding="utf-8")
    print(line)
    return 0 if report["loss_bitwise_equal"] and not report["grad_differing"] else 1


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = (str(repo / "src"), str(repo), str(Path(__file__).resolve().parent))
    raise SystemExit(main())
