"""Compare saved Moonlight bits runs and optional attention captures on the CPU."""

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

from nondet_analysis import (
    SDPA_GRADS,
    grad_breakdown,
    projection_error,
    sdpa_capture_error,
    tensor_pair_diff,
)
from rich.console import Console

_MISSING = object()


def _cell(value: Any) -> str:
    return " ".join(str(value).splitlines()).replace("|", r"\|")


def _table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(_cell, row)) + " |" for row in rows),
        "",
    ]


def _diff(left: Any, right: Any) -> dict[str, Any]:
    if left is _MISSING or right is _MISSING:
        return {
            "equal": False, "bitwise_equal": False,
            "note": "missing A" if left is _MISSING else "missing B",
        }
    if left.shape != right.shape:
        return {
            "equal": False, "bitwise_equal": False,
            "note": f"shape {tuple(left.shape)} vs {tuple(right.shape)}",
        }
    return tensor_pair_diff(left, right)


def _label(path: Path, paths: list[Path]) -> str:
    return str(path) if sum(other.name == path.name for other in paths) > 1 else path.name


def _pair_report(
    left: dict[str, Any], right: dict[str, Any], label_a: str, label_b: str,
) -> tuple[tuple[Any, ...], list[str]]:
    loss = _diff(left["loss"], right["loss"])
    grads_a, grads_b = left["grads"], right["grads"]
    grad_diffs = {
        name: _diff(grads_a.get(name, _MISSING), grads_b.get(name, _MISSING))
        for name in sorted(grads_a.keys() | grads_b.keys())
    }
    breakdown = grad_breakdown(
        grad_diffs, {name: not diff["bitwise_equal"] for name, diff in grad_diffs.items()},
    )
    first_layer = breakdown["first_differing_layer_in_backward_order"]
    first_label = "none" if first_layer is None else str(first_layer)
    totals = breakdown["totals"]
    summary = (
        label_a, label_b, loss["bitwise_equal"], totals["differing"], totals["total"], first_label,
    )
    lines = [
        f"## {_cell(label_a)} vs {_cell(label_b)}",
        "",
        f"First differing layer in backward order: {first_label}.",
        "",
    ]
    for name in breakdown["first_differing_names"]:
        lines.append(f"- `{name}`")
    if breakdown["first_differing_names"]:
        lines.append("")

    groups = (("Layer", "per_layer"), ("Projection", "per_projection"), ("LoRA", "per_ab"))
    for label, key in groups:
        lines.extend(_table((label, "Differing", "Total"), [
            (group, counts["differing"], counts["total"])
            for group, counts in breakdown[key].items()
        ]))
    unparsed = breakdown["unparsed"]
    lines.extend([f"Unparsed: {unparsed['differing']} differing / {unparsed['total']} total.", ""])

    metrics = ("max_abs_diff", "rel_max_diff", "sum_diff", "rel_sum_diff")
    lines.extend(_table(
        ("Tensor", "Value equal", "Bitwise equal", "n_diff", "numel", *metrics, "Note"),
        [
            (name, diff["equal"], diff["bitwise_equal"], diff.get("n_diff", ""),
             diff.get("numel", ""), *(f"{diff[key]:.6g}" if key in diff else "" for key in metrics),
             diff.get("note", ""))
            for name, diff in [("loss", loss), *grad_diffs.items()]
            if name == "loss" or not diff["bitwise_equal"] or not diff["equal"]
        ],
    ))
    return summary, lines


def _digest_diff(left: str, right: str) -> tuple[str, str, str]:
    return left, right, "equal" if left == right else "different"


def _read_capture(path: Path) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    def read(name: str) -> Any:
        try:
            return json.loads((path / name).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"{path / name}: {error}") from error

    output, calls = read("output_grads.json"), read("sdpa_calls.json")
    if not isinstance(output, dict):
        raise ValueError(f"{path}: expected an object in output_grads.json")
    if error := projection_error(output):
        raise ValueError(f"{path}: {error}")
    if error := sdpa_capture_error(calls, path):
        raise ValueError(f"{path}: {error}")
    projections = {}
    for name, raw in output.items():
        legacy = isinstance(raw, dict)
        occurrences = [raw] if legacy else raw
        if not isinstance(occurrences, list) or not occurrences:
            raise ValueError(f"{path}: missing projection gradient occurrence for {name}")
        for entry in occurrences:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("sha256"), str)
                or not entry["sha256"]
                or not isinstance(entry.get("max_abs"), (float, int))
                or not isinstance(entry.get("numel"), int)
                or (not legacy and (not isinstance(entry.get("step"), int) or entry["step"] < 0))
            ):
                raise ValueError(f"{path}: invalid projection gradient occurrence for {name}")
        projections[name] = occurrences
    return projections, calls


def _capture_report(left: Path, right: Path) -> list[str]:
    out_a, calls_a = _read_capture(left)
    out_b, calls_b = _read_capture(right)
    if out_a.keys() != out_b.keys():
        raise ValueError("projection module names differ between captures")
    if len(calls_a) != len(calls_b):
        raise ValueError(f"SDPA call counts differ: {len(calls_a)} vs {len(calls_b)}")
    for index, (call_a, call_b) in enumerate(zip(calls_a, calls_b)):
        if call_a["q_shape"] != call_b["q_shape"]:
            raise ValueError(f"SDPA q_shape differs at call {index}")
    rows = []
    for name in sorted(out_a):
        entries_a, entries_b = out_a[name], out_b[name]
        if len(entries_a) != len(entries_b):
            raise ValueError(f"projection occurrence counts differ for {name}")
        if [entry.get("step") for entry in entries_a] != [entry.get("step") for entry in entries_b]:
            raise ValueError(f"projection occurrence steps differ for {name}")
        a, b, result = _digest_diff(entries_a[-1]["sha256"], entries_b[-1]["sha256"])
        differing = sum(
            entry_a["sha256"] != entry_b["sha256"]
            for entry_a, entry_b in zip(entries_a, entries_b)
        )
        rows.append((name, a, b, result, f"{differing}/{len(entries_a)}"))
    lines = [f"## Captures: {_cell(left)} vs {_cell(right)}", "", "### Projection gradients", ""]
    if any("step" not in entries[0] for entries in out_a.values()):
        lines.extend(["Legacy captures only recorded the first projection gradient.", ""])
    lines.extend(_table(
        ("Projection", "Last SHA-256 A", "Last SHA-256 B", "Last result", "Differing occurrences"),
        rows,
    ))
    rows = [
        (index, key, *_digest_diff(call_a[key], call_b[key]))
        for index, (call_a, call_b) in enumerate(zip(calls_a, calls_b))
        for key in (f"sha_{grad}" for grad in SDPA_GRADS)
    ]
    lines.extend(["### SDPA input gradients", ""])
    lines.extend(_table(("Call", "Key", "SHA-256 A", "SHA-256 B", "Result"), rows))
    return lines


def main(argv: list[str] | None = None) -> int:
    """Read saved tensors, compare each pair, and emit a Markdown report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tensors", type=Path, nargs="+", help="Two or more saved tensor files")
    parser.add_argument("--captures", type=Path, nargs=2, metavar=("DIR_A", "DIR_B"))
    parser.add_argument("--out", type=Path, help="Write compare.md here instead of to stdout")
    args = parser.parse_args(argv)
    if len(args.tensors) < 2:
        parser.error("provide at least two saved tensor files")

    console = Console(soft_wrap=True, markup=False, highlight=False)
    capture_lines = []
    if args.captures:
        try:
            capture_lines = _capture_report(*args.captures)
        except ValueError as error:
            message = f"Capture comparison incomplete: {error}"
            console.print(message)
            if args.out is not None:
                args.out.mkdir(parents=True, exist_ok=True)
                (args.out / "compare.md").write_text(message + "\n", encoding="utf-8")
            return 1

    import torch

    runs = [torch.load(path, map_location="cpu", weights_only=True) for path in args.tensors]
    labels = [_label(path, args.tensors) for path in args.tensors]
    summary, details = [], []
    for (label_a, left), (label_b, right) in combinations(zip(labels, runs), 2):
        row, lines = _pair_report(left, right, label_a, label_b)
        summary.append(row)
        details.extend(lines)
    header = ("Run A", "Run B", "Loss bitwise equal", "LoRA differing", "LoRA total", "First layer")
    report = "\n".join([
        "# Pairwise comparison", "", *_table(header, summary), *details, *capture_lines,
    ])
    if args.out is None:
        console.print(report, end="")
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "compare.md").write_text(report, encoding="utf-8")
        for left, right, equal, differs, total, layer in summary:
            console.print(
                f"{left} vs {right}: loss bitwise equal={equal}; LoRA {differs}/{total} differing; "
                f"first differing layer={layer}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
