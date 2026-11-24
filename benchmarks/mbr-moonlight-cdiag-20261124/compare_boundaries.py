"""Compare two boundary captures of ticket 14 and name the first divergent tensor.

Reads ``boundaries/<tag>.json`` of two arms, matches module boundaries by their input digest, the
expert linear calls and routed-expert node calls row by row, and writes ``compare.md`` plus
``compare.json``. The first divergent boundary is the lowest layer whose module output, routed
expert node row or expert projection row differs; everything in front of it is bitwise equal by
construction of the record digests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cdiag
from rich.console import Console

_FIRST_N = 12


def _cell(value: Any) -> str:
    return " ".join(str(value).splitlines()).replace("|", r"\|")


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("modules", "nodes", "linears"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"{path}: missing {key}")
    return payload


def _first_row_diff(
    left_calls: list[dict[str, Any]], right_calls: list[dict[str, Any]], record: dict[str, Any],
) -> dict[str, Any] | None:
    """Numeric diff of the first differing row, from the saved layer-0 tensors."""
    def rows(calls: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            cdiag.digest(call["input"][index]): call["output"][index]
            for call in calls
            for index in range(call["input"].shape[0])
            if (call["layer"], call["expert"], call["which"], call["phase"])
            == (record["layer"], record["expert"], record["which"], record["phase"])
        }

    reference, candidate = rows(left_calls), rows(right_calls)
    for in_sha, output in candidate.items():
        if in_sha not in reference:
            continue
        if cdiag.digest(output) != cdiag.digest(reference[in_sha]):
            diff = cdiag.tensor_pair_diff(reference[in_sha], output)
            diff["input_sha"] = in_sha
            return diff
    return None


def _trim(report: dict[str, Any]) -> dict[str, Any]:
    """Keep the JSON small: unmatched rows are counted, the first few are kept as samples."""
    trimmed = dict(report)
    trimmed["rows_missing_count"] = len(report["rows_missing"])
    trimmed["rows_missing"] = report["rows_missing"][:_FIRST_N]
    return trimmed


def _table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(_cell(cell) for cell in row) + " |" for row in rows),
        "",
    ]


def _row_counts(payload: dict[str, Any]) -> str:
    """Forward expert-call row counts, one bucket per observed M."""
    counts: dict[int, int] = {}
    for record in payload["linears"]:
        if record["phase"] == "forward":
            counts[record["rows"]] = counts.get(record["rows"], 0) + 1
    return ", ".join(f"M={m} {count}x" for m, count in sorted(counts.items()))


def _report(
    modules: dict[str, Any], experts: dict[str, Any], nodes: dict[str, Any],
    verdict: dict[str, Any], left: dict[str, Any], right: dict[str, Any],
    numeric: dict[str, Any] | None, labels: tuple[str, str],
) -> str:
    lines = [
        f"# Boundaries: {labels[0]} vs {labels[1]}",
        "",
        f"First divergent boundary: **{_cell(verdict)}**",
        "",
        "## Module boundaries",
        "",
        f"Matched {modules['left_count']} against {modules['right_count']} records by "
        f"(layer, role, input digest); equal: **{modules['equal']}**; mismatched: "
        f"{len(modules['mismatched'])}; inputs absent from the reference: "
        f"{len(modules['missing'])}.",
        "",
    ]
    if modules["mismatched"]:
        lines.extend(_table(
            ("Layer", "Role", "Micro-batch", "Left sha16", "Right sha16"),
            [
                (entry["layer"], entry["role"], entry["mb"], entry["left_sha"][:16],
                 entry["right_sha"][:16])
                for entry in modules["mismatched"][:_FIRST_N]
            ],
        ))

    lines.extend([
        "## Routed-expert node",
        "",
        f"Node calls: {nodes['calls']}; rows: {nodes['rows']}; matched by input row digest: "
        f"{nodes['rows_matched']}; unmatched: {nodes.get('rows_missing_count', 0)}; calls whose "
        f"output rows differ: {len(nodes['differing'])}.",
        "",
        "## Expert linear calls",
        "",
        f"Right-hand expert calls: {experts['calls']}; rows: {experts['rows']}; matched by row "
        f"input digest: {experts['rows_matched']}; unmatched inputs: "
        f"{experts.get('rows_missing_count', 0)}; calls with a differing output row: "
        f"{len(experts['differing'])}.",
        "",
        "### Forward call row counts",
        "",
        f"- {labels[0]}: {_row_counts(left)}",
        f"- {labels[1]}: {_row_counts(right)}",
        "",
    ])
    if experts["differing"]:
        lines.extend([
            "### Differing expert calls",
            "",
            *_table(
                ("Layer", "Expert", "Projection", "Phase", "Call", "Left M", "Right M", "Rows",
                 "Differing rows"),
                [
                    (entry["layer"], entry["expert"], entry["which"], entry["phase"],
                     entry["call"], sorted(set(entry["a_rows"])), entry["b_rows"],
                     entry["matched_rows"], entry["diff_rows"])
                    for entry in experts["differing"][:_FIRST_N]
                ],
            ),
        ])
    if numeric:
        lines.extend([
            "### First differing row, numbers",
            "",
            _cell(numeric),
            "",
        ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Compare the two captures and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True, help="A0 capture JSON")
    parser.add_argument("--right", type=Path, required=True, help="C capture JSON")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--labels", default=None,
                        help="two comma-separated arm labels; default: the capture tag stems")
    args = parser.parse_args(argv)
    if args.labels:
        labels = tuple(part.strip() for part in args.labels.split(","))
        if len(labels) != 2 or not all(labels):
            parser.error("--labels takes two comma-separated names")
    else:
        labels = (args.left.stem, args.right.stem)
    left, right = _load(args.left), _load(args.right)
    if left["step"] != right["step"]:
        raise SystemExit(f"step counts differ: {left['step']} vs {right['step']}")
    modules = cdiag.match_module_boundaries(left["modules"], right["modules"])
    experts = cdiag.match_expert_rows(left["linears"], right["linears"])
    nodes = cdiag.match_expert_rows(left["nodes"], right["nodes"])
    verdict = cdiag.boundary_verdict(modules, experts, nodes)
    numeric = None
    if verdict["kind"] == "expert_linear":
        import torch

        numeric = _first_row_diff(
            torch.load(args.left.with_name(f"{args.left.stem}-tensors.pt"), weights_only=False),
            torch.load(args.right.with_name(f"{args.right.stem}-tensors.pt"), weights_only=False),
            verdict,
        )
    payload = {
        "modules": modules, "experts": _trim(experts), "nodes": _trim(nodes),
        "verdict": verdict, "numeric": numeric,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "compare.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    (args.out / "compare.md").write_text(
        _report(modules, experts, nodes, verdict, left, right, numeric, labels),
        encoding="utf-8",
    )
    Console(soft_wrap=True, markup=False, highlight=False).print(json.dumps(verdict))
    return 0


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = (str(repo / "src"), str(repo), str(Path(__file__).resolve().parent))
    raise SystemExit(main())
