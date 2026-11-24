#!/usr/bin/env python3
"""Ticket 15: layout of one step (arms C against S) from the traced rows, the rows and py-spy.

Reads three pieces under ``--root`` and tolerates any of them missing:

- ``trace/*.json``: the traced step of a cell (``profile_step.py --trace``);
- ``runs/rows.jsonl``: the timed runs of the cells;
- ``pyspy/*.speedscope.json``: the ``py-spy`` snapshot of a cell.

Writes:

- ``analysis.json``: the per-window cells of C and S, the attribution of S's slower step, and the
  py-spy leaf shares;
- ``layout.md``: per window the C/S table (step and kernel seconds, busy share, exposed time,
  launches, expert-GEMM calls, seconds, rows, µs/row, py-spy shares), then the M histogram, the
  kernel table and the per-row table of each arm;
- ``rows.md``: one result line per measured cell, naming model, card, window, arm, mode and torch.

``attribute`` splits S's slower step into expert-GEMM kernels, other kernels and time outside the
kernels: at 0.6 of the gap or more the largest component names the verdict, else it is mixed.
Benchmark code, not shipped.

    python analyze.py --root benchmarks/mbr-moonlight-segprof-20261124
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

CAMPAIGN = Path(__file__).resolve().parent
if str(CAMPAIGN) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN))

from analyze_pyspy import classify  # noqa: E402

MODEL = "Moonlight-16B-A3B"
CARD = "RTX 3050 Laptop sm_86 4 GiB"
CELLS: Tuple[Tuple[int, int], ...] = ((8, 64), (64, 64))
ARMS = ("C", "S")
ARM_LABELS = {"C": "ordered backward", "S": "C + segmented"}
NAMED_AT = 0.6
HEADER = ("| arm | step s | kernel s | busy share | exposed s | launches | expert calls "
          "| expert s | expert rows | µs/row | py-spy expert | py-spy cuda | py-spy disk |")
RULE = "|" + "---|" * 13


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_jsons(folder: Path, suffix: str = ".json") -> Dict[str, Dict[str, Any]]:
    if not folder.is_dir():
        return {}
    return {path.name[: -len(suffix)]: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(folder.glob(f"*{suffix}"))}


def _torch(rows: Sequence[Dict[str, Any]]) -> str:
    versions = [r.get("environment", {}).get("versions", {}).get("torch")
                for r in rows if r.get("status") == "ok"]
    return next((version for version in versions if version), "unknown")


def attribute(cell_c: Dict[str, Any], cell_s: Dict[str, Any]) -> Dict[str, Any]:
    """Split S's slower step into expert-GEMM kernels, other kernels and non-kernel time."""
    delta = float(cell_s["step_seconds"]) - float(cell_c["step_seconds"])
    expert = float(cell_s["expert_gemm_seconds"]) - float(cell_c["expert_gemm_seconds"])
    other = ((float(cell_s["kernel_seconds"]) - float(cell_s["expert_gemm_seconds"]))
             - (float(cell_c["kernel_seconds"]) - float(cell_c["expert_gemm_seconds"])))
    non_kernel = delta - expert - other
    if not delta:
        return {"verdict": "mixed", "shares": {}}
    shares = {"expert_gemm": round(expert / delta, 4), "other_kernel": round(other / delta, 4),
              "non_kernel": round(non_kernel / delta, 4)}
    name, share = max(shares.items(), key=lambda item: item[1])
    if share < NAMED_AT:
        verdict = "mixed"
    else:
        verdict = "dispatch" if name == "non_kernel" else "kernels"
    return {"verdict": verdict, "shares": shares}


def _step_seconds(rows: Sequence[Dict[str, Any]], *, k: int, seq: int,
                  arm: str) -> Optional[float]:
    """Median of the measured step of the cell (an instrumented trace run never sets it)."""
    steps = [r["step_seconds"] for r in rows
             if (r.get("k"), r.get("seq"), r.get("arm")) == (k, seq, arm)
             and r.get("status") == "ok" and r.get("step_seconds") is not None]
    return statistics.median(steps) if steps else None


def cell(trace_entry: Optional[Dict[str, Any]], *, step_seconds: Optional[float],
         pyspy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """One arm in one window: the timed step plus the counters and timeline of its traced step."""
    entry = trace_entry or {}
    trace = entry.get("trace") or {}
    timeline = trace.get("timeline") or {}
    gemm = trace.get("expert_gemm") or {}
    if step_seconds is None:
        step_seconds = (entry.get("row") or {}).get("step_seconds")
    busy = float(timeline.get("busy_seconds") or 0.0)
    span = float(timeline.get("span_seconds") or 0.0)
    expert_seconds = float(gemm.get("seconds") or 0.0)
    rows_total = int(gemm.get("rows_total") or 0)
    return {
        "step_seconds": step_seconds,
        "kernel_seconds": float(timeline.get("kernel_seconds") or 0.0),
        "busy_share": busy / span if span else None,
        "exposed_seconds": float(timeline.get("exposed_seconds") or 0.0),
        "launches": int(timeline.get("kernels") or 0),
        "expert_calls": int(gemm.get("calls") or 0),
        "expert_gemm_seconds": expert_seconds,
        "expert_rows": rows_total,
        "us_per_row": expert_seconds * 1e6 / rows_total if rows_total else None,
        "kernel_table": gemm.get("kernels") or {},
        "by_rows": gemm.get("by_rows") or {},
        "pyspy": pyspy,
    }


def _seconds(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.2f}"


def _share(value: Optional[float]) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def _arm_row(arm: str, cell: Dict[str, Any]) -> str:
    pyspy = cell.get("pyspy") or {}
    busy = cell["busy_share"]
    per_row = cell["us_per_row"]
    fields = [arm, _seconds(cell["step_seconds"]), _seconds(cell["kernel_seconds"]),
              "-" if busy is None else f"{busy:.3f}", _seconds(cell["exposed_seconds"]),
              str(cell["launches"]), str(cell["expert_calls"]),
              _seconds(cell["expert_gemm_seconds"]), str(cell["expert_rows"]),
              "-" if per_row is None else f"{per_row:.1f}",
              _share(pyspy.get("expert_gemm_share")),
              _share(pyspy.get("cuda_runtime_share")), _share(pyspy.get("disk_share"))]
    return "| " + " | ".join(fields) + " |"


def _arm_sections(arm: str, window: str, cell: Dict[str, Any]) -> List[str]:
    """The M histogram, the kernel table and the per-row (M efficiency) table of one arm."""
    lines = [f"### M histogram, arm {arm}, window {window}", "", "| rows | calls |", "|---|---|"]
    lines += [f"| {rows} | {entry['calls']} |"
              for rows, entry in sorted((cell.get("by_rows") or {}).items())]
    lines += ["", f"### Kernel table, arm {arm}, window {window}", "",
              "| kernel | calls | seconds | rows |", "|---|---|---|---|"]
    lines += [f"| {name} | {entry['calls']} | {entry['seconds']:.2f} "
              f"| {entry['rows_min']}–{entry['rows_max']} |"
              for name, entry in sorted((cell.get("kernel_table") or {}).items())]
    lines += ["", f"### Per row, arm {arm}, window {window}", "",
              "| rows | calls | seconds | µs/row |", "|---|---|---|---|"]
    lines += [f"| {rows} | {entry['calls']} | {entry['seconds']:.2f} | {entry['us_per_row']:.1f} |"
              for rows, entry in sorted((cell.get("by_rows") or {}).items())]
    lines.append("")
    return lines


def layout_table(cells: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
    """The layout of one step per window, C and S side by side, then their per-arm tables."""
    lines = ["# Layout of one profiled step (ticket 15)", ""]
    for window, arms in cells.items():
        lines += [f"## window {window}", "", HEADER, RULE]
        lines += [_arm_row(arm, arms[arm]) for arm in ARMS if arm in arms]
        lines.append("")
        for arm in ARMS:
            if arm in arms:
                lines += _arm_sections(arm, window, arms[arm])
    return "\n".join(lines) + "\n"


def result_lines(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """One line per (arm, k, seq) cell, naming model, card, window, arm, mode and torch."""
    torch = _torch(rows)
    cells: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = {}
    for row in rows:
        cells.setdefault((row["arm"], row["k"], row["seq"]), []).append(row)
    order = {arm: position for position, arm in enumerate(ARMS)}
    lines = []
    for (arm, k, seq), group in sorted(cells.items(),
                                       key=lambda kv: (kv[0][1], kv[0][2],
                                                       order.get(kv[0][0], len(ARMS)))):
        status = next((r.get("status") for r in group if r.get("status") != "ok"), "ok")
        steps = [r["step_seconds"] for r in group
                 if r.get("status") == "ok" and r.get("step_seconds") is not None]
        text = (f"- {MODEL}, {CARD}, window ({k},{seq}), arm {arm} "
                f"({ARM_LABELS.get(arm, arm)}), mode speed (normal), torch {torch}: {status}")
        if steps:
            text += f", step {statistics.median(steps):.2f} s"
        lines.append(text)
    return lines


def analyze(root: Path) -> Dict[str, Any]:
    """Build the cells, the attribution and the three files under ``root``."""
    root = Path(root)
    traces = _read_jsons(root / "trace")
    rows = _read_jsonl(root / "runs" / "rows.jsonl")
    profiles = _read_jsons(root / "pyspy", ".speedscope.json")
    pyspy = {stem: classify(profile) for stem, profile in profiles.items()}
    windows: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for k, seq in CELLS:
        stem_prefix = f"-k{k}-s{seq}"
        for arm in ARMS:
            stem = f"{arm}{stem_prefix}"
            windows.setdefault(f"({k},{seq})", {})[arm] = cell(
                traces.get(stem), step_seconds=_step_seconds(rows, k=k, seq=seq, arm=arm),
                pyspy=pyspy.get(stem))
    attribution = {window: attribute(arms["C"], arms["S"])
                   for window, arms in windows.items()
                   if arms["C"]["step_seconds"] is not None
                   and arms["S"]["step_seconds"] is not None}
    analysis = {"model": MODEL, "card": CARD, "cells": windows, "attribution": attribution,
                "windows": [f"({k},{seq})" for k, seq in CELLS], "traces": len(traces),
                "rows": len(rows), "pyspy_profiles": len(pyspy)}
    (root / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    (root / "layout.md").write_text(layout_table(windows), encoding="utf-8")
    lines = ["# Result lines (ticket 15, segmented profile)", ""] + result_lines(rows)
    (root / "rows.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return analysis


def main(argv: object = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
