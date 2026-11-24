#!/usr/bin/env python3
"""Ticket 15: layout of one step (arms C against S) from the counters, the sweep and the trace.

The whole-step ``torch.profiler`` trace does not fit the host at (64,64) (OOM-killed at about 1.1M
kernel events), so the campaign splits its evidence:

- ``runs/rows.jsonl``: the ``--count`` runs of every cell, each with the routed-expert
  ``F.linear`` counter (calls per step, the M histogram, the phases) and the GPU utilisation;
- ``sweep.json``: the isolated per-M kernel cost of the two routed-expert projections, no model;
- ``trace/*.json``: the whole-step device timeline, (8,64) only;
- ``pyspy/*.speedscope.json``: the leaf shares of one cell.

Writes ``analysis.json`` (the cells of C and S, the attribution, the sweep), ``layout.md`` (per
window the C/S row, the M histogram of each arm, the sweep's per-M price, the py-spy shares and
the attribution) and ``rows.md`` (one line per measured cell naming model, card, window, arm,
mode, torch, expert calls and step seconds).

``attribute`` splits S's slower step into the sweep's expert model and the time outside it, and
adds the measured busy seconds when both cells are traced: at 0.6 of the gap or more the largest
component names the verdict, else it is mixed. An untraced cell's verdict rests on the model and
says ``measured: false``. Benchmark code, not shipped.

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
REPO = CAMPAIGN.parents[1]
for entry in (str(CAMPAIGN), str(REPO)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from analyze_pyspy import classify  # noqa: E402

from benchmarks.harness import mbr_count  # noqa: E402

MODEL = "Moonlight-16B-A3B"
CARD = "RTX 3050 Laptop sm_86 4 GiB"
CELLS: Tuple[Tuple[int, int], ...] = ((8, 64), (64, 64))
ARMS = ("C", "S")
ARM_LABELS = {"C": "ordered backward", "S": "C + segmented"}
NAMED_AT = 0.6
HEADER = ("| arm | step s | expert calls | expert s (model) | µs/call | kernel s | busy s "
          "| exposed s | launches | py-spy expert | py-spy cuda | py-spy disk | measured |")
RULE = "|" + "---|" * 13
NOT_TRACED = ("No whole-step trace for this window: the campaign traces (8,64) only, because the "
              "(64,64) trace was OOM-killed (about 1.1M kernel events). The verdict rests on the "
              "--count counters, the sweep's isolated per-M cost and the py-spy shares.")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsons(folder: Path, suffix: str = ".json") -> Dict[str, Dict[str, Any]]:
    if not folder.is_dir():
        return {}
    return {path.name[: -len(suffix)]: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(folder.glob(f"*{suffix}"))}


def _torch(rows: Sequence[Dict[str, Any]]) -> str:
    versions = [r.get("environment", {}).get("versions", {}).get("torch")
                for r in rows if r.get("status") == "ok"]
    return next((version for version in versions if version), "unknown")


def per_call_prices(sweep: Optional[Dict[str, Any]]) -> Dict[int, float]:
    """M -> mean microseconds of one expert GEMM over the gate_up and down projections.

    The model prices one call at M, not one row: ``sweep.json`` carries ``us_per_call`` per M for
    each projection, and the average of the two is the pair the routed expert runs (gate_up then
    down) at that M.
    """
    projections = (sweep or {}).get("projections") or {}
    prices: Dict[int, List[float]] = {}
    for entry in projections.values():
        for rows, price in (entry.get("rows") or {}).items():
            prices.setdefault(int(rows), []).append(float(price["us_per_call"]))
    return {rows: round(sum(values) / len(values), 4) for rows, values in sorted(prices.items())}


def attribute(cell_c: Dict[str, Any], cell_s: Dict[str, Any]) -> Dict[str, Any]:
    """Split S's slower step into the expert model, the measured busy time and the rest."""
    if (cell_c["step_seconds"] is None or cell_s["step_seconds"] is None
            or cell_c["expert_seconds_model"] is None or cell_s["expert_seconds_model"] is None):
        return {"verdict": "mixed", "shares": {}, "measured": False}
    delta = float(cell_s["step_seconds"]) - float(cell_c["step_seconds"])
    model = float(cell_s["expert_seconds_model"]) - float(cell_c["expert_seconds_model"])
    busy = None
    if cell_c.get("busy_seconds") is not None and cell_s.get("busy_seconds") is not None:
        busy = float(cell_s["busy_seconds"]) - float(cell_c["busy_seconds"])
    non_kernel = delta - model
    if not delta:
        return {"verdict": "mixed", "shares": {}, "measured": busy is not None}
    shares = {"expert_model_seconds": round(model / delta, 4),
              "non_kernel_seconds": round(non_kernel / delta, 4)}
    if busy is not None:
        shares["measured_busy_seconds"] = round(busy / delta, 4)
    name, share = max(shares.items(), key=lambda item: item[1])
    if share < NAMED_AT:
        verdict = "mixed"
    else:
        verdict = "dispatch" if name == "non_kernel_seconds" else "kernels"
    return {"verdict": verdict, "shares": shares, "measured": busy is not None}


def _step_seconds(rows: Sequence[Dict[str, Any]]) -> Optional[float]:
    """Median of the cell's ``--count`` runs; a cell without one keeps its ok rows."""
    counted = [r for r in rows if r.get("expert_gemm_calls")]
    pool = counted or list(rows)
    steps = [r["step_seconds"] for r in pool
             if r.get("status") == "ok" and r.get("step_seconds") is not None]
    return statistics.median(steps) if steps else None


def _expert_calls(rows: Sequence[Dict[str, Any]]) -> int:
    """Median of the per-step call counts of the ``--count`` rows."""
    medians = [int(r["expert_gemm_calls"]["median"]) for r in rows if r.get("expert_gemm_calls")]
    return int(statistics.median(medians)) if medians else 0


def _expert_rows(rows: Sequence[Dict[str, Any]]) -> Dict[int, int]:
    """Rows -> calls over the cell's ``--count`` rows, ascending."""
    merged: Dict[int, int] = {}
    for row in rows:
        histogram = (row.get("expert_gemm_calls") or {}).get("rows") or {}
        for row_count, calls in histogram.items():
            merged[int(row_count)] = merged.get(int(row_count), 0) + int(calls)
    return dict(sorted(merged.items()))


def cell(rows: Sequence[Dict[str, Any]], *, trace_entry: Optional[Dict[str, Any]] = None,
         pyspy: Optional[Dict[str, Any]] = None,
         prices: Optional[Dict[int, float]] = None) -> Dict[str, Any]:
    """One arm in one window: the counted calls, the sweep model, the trace and the py-spy shares.

    ``step_seconds`` and the counters come from the ``--count`` runs; the device timeline comes
    from the traced step and is None when the window was not traced.
    """
    prices = prices or {}
    timeline = ((trace_entry or {}).get("trace") or {}).get("timeline") or {}
    histogram = _expert_rows(rows)
    model = mbr_count.expert_seconds(histogram, prices) if histogram else None
    calls = _expert_calls(rows)
    return {
        "step_seconds": _step_seconds(rows),
        "expert_calls": calls,
        "expert_rows": histogram,
        "expert_seconds_model": model,
        "us_per_call": model * 1e6 / calls if model is not None and calls else None,
        "missing_rows": sorted(row_count for row_count in histogram if row_count not in prices),
        "measured": bool(timeline),
        "kernel_seconds": float(timeline["kernel_seconds"]) if timeline else None,
        "busy_seconds": float(timeline["busy_seconds"]) if timeline else None,
        "exposed_seconds": float(timeline["exposed_seconds"]) if timeline else None,
        "launches": int(timeline["kernels"]) if timeline else None,
        "pyspy": pyspy,
    }


def _seconds(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.2f}"


def _micros(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.1f}"


def _share(value: Optional[float]) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def _arm_row(arm: str, cell_row: Dict[str, Any]) -> str:
    pyspy = cell_row.get("pyspy") or {}
    launches = cell_row["launches"]
    fields = [arm, _seconds(cell_row["step_seconds"]), str(cell_row["expert_calls"]),
              _seconds(cell_row["expert_seconds_model"]), _micros(cell_row["us_per_call"]),
              _seconds(cell_row["kernel_seconds"]), _seconds(cell_row["busy_seconds"]),
              _seconds(cell_row["exposed_seconds"]),
              "-" if launches is None else str(launches),
              _share(pyspy.get("expert_gemm_share")),
              _share(pyspy.get("cuda_runtime_share")), _share(pyspy.get("disk_share")),
              "yes" if cell_row["measured"] else "no"]
    return "| " + " | ".join(fields) + " |"


def _histogram_section(arm: str, window: str, cell_row: Dict[str, Any]) -> List[str]:
    lines = [f"### M histogram, arm {arm}, window {window}", "", "| rows | calls |", "|---|---|"]
    lines += [f"| {rows} | {calls} |" for rows, calls in (cell_row["expert_rows"] or {}).items()]
    if cell_row["missing_rows"]:
        lines += ["", f"No sweep price for rows {cell_row['missing_rows']}: the model skips "
                      "the whole cell rather than guessing them."]
    lines.append("")
    return lines


def _sweep_section(window: str, sweep: Optional[Dict[str, Any]],
                   arms: Dict[str, Dict[str, Any]]) -> List[str]:
    """The sweep's per-M price, with the calls each arm made at that M."""
    title = f"### Per-M price (sweep, mean of gate_up and down), window {window}"
    if not sweep:
        return [title, "", "No sweep.json: no per-M price, so no expert seconds model.", ""]
    prices = per_call_prices(sweep)
    projections = sweep.get("projections") or {}

    def price(name: str, rows: int) -> str:
        entry = ((projections.get(name) or {}).get("rows") or {}).get(str(rows))
        return "-" if entry is None else f"{float(entry['us_per_call']):.1f}"

    lines = [title, "",
             "| rows | gate_up µs | down µs | mean µs | C calls | S calls |",
             "|---|---|---|---|---|---|"]
    for rows, mean in prices.items():
        calls = " | ".join(str((arms.get(arm) or {}).get("expert_rows", {}).get(rows, "-"))
                           for arm in ARMS)
        lines.append(f"| {rows} | {price('gate_up', rows)} | {price('down', rows)} "
                     f"| {mean:.1f} | {calls} |")
    lines.append("")
    return lines


def _pyspy_section(window: str, arms: Dict[str, Dict[str, Any]]) -> List[str]:
    lines = [f"### py-spy leaf shares, window {window}", "",
             "| arm | expert | cuda | disk | torch py | other py | native other |",
             "|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        profile = (arms.get(arm) or {}).get("pyspy") or {}
        lines.append(
            f"| {arm} | {_share(profile.get('expert_gemm_share'))} "
            f"| {_share(profile.get('cuda_runtime_share'))} "
            f"| {_share(profile.get('disk_share'))} "
            f"| {_share(profile.get('torch_python_share'))} "
            f"| {_share(profile.get('other_python_share'))} "
            f"| {_share(profile.get('native_other_share'))} |")
    lines.append("")
    return lines


def _attribution_section(window: str, attribution: Optional[Dict[str, Any]]) -> List[str]:
    lines = [f"### attribution, window {window}", ""]
    if not attribution:
        return lines + ["No attribution: a cell of this window has no step seconds.", ""]
    lines += [f"verdict: {attribution['verdict']} "
              f"(measured: {str(attribution['measured']).lower()})", "",
              "| component | share |", "|---|---|"]
    lines += [f"| {name} | {_share(share)} |"
              for name, share in attribution.get("shares", {}).items()]
    lines.append("")
    return lines


def layout_table(cells: Dict[str, Dict[str, Dict[str, Any]]], *,
                 sweep: Optional[Dict[str, Any]] = None,
                 attribution: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
    """One window per section: the C/S row, the M histograms, the sweep price, py-spy, verdict."""
    attribution = attribution or {}
    lines = ["# Layout of one profiled step (ticket 15)", ""]
    for window, arms in cells.items():
        lines += [f"## window {window}", ""]
        if not any(arms[arm]["measured"] for arm in arms):
            lines += [NOT_TRACED, ""]
        lines += [HEADER, RULE]
        lines += [_arm_row(arm, arms[arm]) for arm in ARMS if arm in arms]
        lines.append("")
        for arm in ARMS:
            if arm in arms:
                lines += _histogram_section(arm, window, arms[arm])
        lines += _sweep_section(window, sweep, arms)
        lines += _pyspy_section(window, arms)
        lines += _attribution_section(window, attribution.get(window))
    return "\n".join(lines) + "\n"


def result_lines(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """One line per (arm, k, seq) cell, naming model, card, window, arm, mode, torch and counts."""
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
        kind, mode = group[0].get("kind", "speed"), group[0].get("mode", "normal")
        step = _step_seconds(group)
        text = (f"- {MODEL}, {CARD}, window ({k},{seq}), arm {arm} "
                f"({ARM_LABELS.get(arm, arm)}), mode {kind} ({mode}), torch {torch}: {status}")
        if step is not None:
            text += f", step {step:.2f} s"
        calls = _expert_calls(group)
        if calls:
            text += f", expert calls {calls}"
        lines.append(text)
    return lines


def analyze(root: Path) -> Dict[str, Any]:
    """Build the cells, the attribution and the three files under ``root``."""
    root = Path(root)
    traces = _read_jsons(root / "trace")
    rows = _read_jsonl(root / "runs" / "rows.jsonl")
    profiles = _read_jsons(root / "pyspy", ".speedscope.json")
    pyspy = {stem: classify(profile) for stem, profile in profiles.items()}
    sweep = _read_json(root / "sweep.json")
    prices = per_call_prices(sweep)
    grouped: Dict[Tuple[int, int, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["k"]), int(row["seq"]), str(row["arm"])), []).append(row)
    windows: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for k, seq in CELLS:
        stem_prefix = f"-k{k}-s{seq}"
        for arm in ARMS:
            stem = f"{arm}{stem_prefix}"
            windows.setdefault(f"({k},{seq})", {})[arm] = cell(
                grouped.get((k, seq, arm), []), trace_entry=traces.get(stem),
                pyspy=pyspy.get(stem), prices=prices)
    attribution = {window: attribute(arms["C"], arms["S"])
                   for window, arms in windows.items()
                   if arms["C"]["step_seconds"] is not None
                   and arms["S"]["step_seconds"] is not None}
    traced = [window for window, arms in windows.items()
              if any(arms[arm]["measured"] for arm in arms)]
    analysis = {"model": MODEL, "card": CARD, "cells": windows, "attribution": attribution,
                "windows": [f"({k},{seq})" for k, seq in CELLS], "traced_windows": traced,
                "traces": len(traces), "rows": len(rows), "pyspy_profiles": len(pyspy),
                "sweep": ({"reps": sweep.get("reps"), "warmup": sweep.get("warmup"),
                           "torch": sweep.get("torch"), "per_m_us": prices}
                          if sweep else None)}
    (root / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    (root / "layout.md").write_text(
        layout_table(windows, sweep=sweep, attribution=attribution), encoding="utf-8")
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
