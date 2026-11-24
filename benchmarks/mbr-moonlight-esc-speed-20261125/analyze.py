#!/usr/bin/env python3
"""Ticket 16: the three candidate hands C, S and E in one campaign, and the best combination.

Reads the speed blocks written by ``run_speed.sh`` (``speed/rows.jsonl`` and
``speed/arm_records.jsonl`` under ``--root``) and writes:

- ``analysis.json``: per window the C/S/E step means, the three paired deltas (S-C, E-C, E-S)
  with a t over the five blocks, the memory peaks of every hand, the ArmWatch summary and the
  three candidate combinations;
- ``rows.md``: one result line per measured cell, each naming model, card, window, arm, mode and
  torch version (the reporting rule of the map).

The per-block number is the median of an arm's two runs in that block; the window number is the
mean of the per-block medians, as ``speed-rule.md`` says. The paired t uses the per-block gains
and the 5-block critical value 2.78. Benchmark code, not shipped.

    python analyze.py --root benchmarks/mbr-moonlight-esc-speed-20261125 \
        --bits benchmarks/mbr-moonlight-esc-speed-20261125/bits-cross.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
for entry in (str(REPO), str(REPO / "benchmarks")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from benchmarks.harness import mbr_measure as mm  # noqa: E402

MODEL = "Moonlight-16B-A3B"
CARD = "RTX 3050 Laptop sm_86 4 GiB"
SPEED_FOLDER = "speed"
ARMS = ("C", "S", "E")
ARM_LABELS = {"C": "ordered backward", "S": "C + segmented", "E": "C + lt-fused"}
DELTAS = (("S", "C"), ("E", "C"), ("E", "S"))
PEAK_KEYS = ("peak_allocated", "peak_reserved", "peak_nvml", "peak_ram")
BITS_EQUAL = "E == S == A0 (direct)"
BITS_NO_REFERENCE = "no A0 reference above k=8"
BITS_DIFFER = "E != S"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _torch(rows: Sequence[Dict[str, Any]]) -> str:
    versions = [r.get("environment", {}).get("versions", {}).get("torch")
                for r in rows if r.get("status") == "ok"]
    return next((v for v in versions if v), "unknown")


def _gib(value: Optional[int]) -> str:
    return "-" if value is None else f"{value / 2**30:.2f}"


def _result_lines(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """One line per (arm, k, seq) cell, naming model, card, window, arm, mode and torch."""
    cells: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = {}
    for row in rows:
        cells.setdefault((row["arm"], row["k"], row["seq"]), []).append(row)
    torch = _torch(rows)
    lines = []
    order = {arm: position for position, arm in enumerate(ARMS)}
    ordered = sorted(cells.items(),
                     key=lambda kv: (kv[0][1], kv[0][2], order.get(kv[0][0], len(ARMS))))
    for (arm, k, seq), group in ordered:
        status = next((r.get("status") for r in group if r.get("status") != "ok"), "ok")
        steps = [r["step_seconds"] for r in group
                 if r.get("status") == "ok" and r.get("step_seconds") is not None]
        text = (f"- {MODEL}, {CARD}, window ({k},{seq}), arm {arm} "
                f"({ARM_LABELS.get(arm, arm)}), mode speed (normal), torch {torch}: {status}")
        if steps:
            text += f", step {statistics.median(steps):.2f} s"
        for name, label in zip(PEAK_KEYS, ("allocated", "reserved", "NVML", "RAM")):
            values = [r[name] for r in group if r.get(name) is not None]
            peak = max(values) if values else None
            text += f", {label} {_gib(peak)} GiB"
        text += f", runs {len(group)}"
        lines.append(text)
    return lines


def _paired_t(gains: Sequence[float]) -> Optional[float]:
    """The paired t of the per-block gains, the same reading as ``speed-rule.md``."""
    if len(gains) < 2:
        return None
    spread = statistics.stdev(gains)
    if spread == 0:
        return float("inf") if statistics.mean(gains) > 0 else 0.0
    return statistics.mean(gains) / (spread / len(gains) ** 0.5)


def _arm_steps(rows: Sequence[Dict[str, Any]], *, k: int, seq: int,
               arm: str) -> Dict[int, float]:
    """Per block, the median of the arm's ok runs; blocks without a run are absent."""
    per_block: Dict[int, List[float]] = {}
    for row in rows:
        if (row["k"], row["seq"], row["arm"]) == (k, seq, arm) \
                and row.get("status") == "ok" and row.get("step_seconds") is not None:
            per_block.setdefault(row.get("block", 0), []).append(row["step_seconds"])
    return {block: statistics.median(values) for block, values in per_block.items()}


def _mean(values: Sequence[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def window_stats(rows: Sequence[Dict[str, Any]], *, k: int, seq: int) -> Dict[str, Any]:
    """The three hands' step means, all three deltas with their paired t, and the memory peaks."""
    steps = {arm: _arm_steps(rows, k=k, seq=seq, arm=arm) for arm in ARMS}
    means = {arm: _mean(list(values.values())) for arm, values in steps.items()}

    def gains(now: str, before: str) -> Optional[Dict[str, Any]]:
        pairs = [1 - steps[now][block] / steps[before][block]
                 for block in sorted(steps[now])
                 if block in steps[before] and steps[before][block]]
        if not pairs:
            return None
        return {"gains_pct": [round(g * 100, 2) for g in pairs],
                "mean_gain_pct": round(statistics.mean(pairs) * 100, 2),
                "t": None if (t := _paired_t(pairs)) is None else round(t, 2)}

    table = mm.summarize_rows(rows)
    per_block = {str(block): {arm: round(steps[arm][block], 3)
                              for arm in ARMS if block in steps[arm]}
                 for block in sorted(set().union(*steps.values()))}
    c, s, e = means["C"], means["S"], means["E"]
    return {
        "C_step_s": None if c is None else round(c, 2),
        "S_step_s": None if s is None else round(s, 2),
        "E_step_s": None if e is None else round(e, 2),
        "per_block": per_block,
        "deltas": {f"{now}_vs_{before}": gains(now, before) for now, before in DELTAS},
        "speedup_C_baseline": 1.0 if c else None,
        "speedup_S_over_C": round(c / s, 3) if c and s else None,
        "speedup_E_over_C": round(c / e, 3) if c and e else None,
        "memory": {arm: {name: table.get((arm, k, seq), {}).get(name) for name in PEAK_KEYS}
                   for arm in ARMS},
    }


# The three candidates the ticket asks for, as the hand each window runs on.
COMBINATIONS: Dict[str, Any] = {
    "C everywhere": lambda window: "C",
    "S everywhere": lambda window: "S",
    "S on k<=8 + E on k=64": lambda window: "E" if window.startswith("(64,") else "S",
    "E everywhere": lambda window: "E",
}


def _bits_status(bits: Optional[Mapping[str, Optional[bool]]], key: str) -> str:
    if bits is None or key not in bits:
        return "not measured"
    if bits[key] is None:
        return BITS_NO_REFERENCE
    return BITS_EQUAL if bits[key] else BITS_DIFFER


def combination_table(windows: Mapping[str, Mapping[str, Any]], *,
                      bits: Optional[Mapping[str, Optional[bool]]] = None) -> Dict[str, Any]:
    """Each candidate's arm per window, its step, its sum and its peaks against C everywhere.

    A window is keyed ``"(k,seq)"``; the step of an arm is its ``<arm>_step_s`` in that window.
    ``bits`` maps ``"k,seq"`` to ``True`` (E == S directly), ``False`` or ``None`` (no A0
    reference above k=8). The sum is the price of one step per window and is comparable across
    candidates because every window is measured for every hand.
    """
    base = sum(stats["C_step_s"] for stats in windows.values())
    table: Dict[str, Any] = {}
    for name, pick in COMBINATIONS.items():
        per_window: Dict[str, Dict[str, Any]] = {}
        peaks = {peak: 0 for peak in PEAK_KEYS}
        for window, stats in windows.items():
            arm = pick(window)
            step = stats[f"{arm}_step_s"]
            per_window[window] = {"arm": arm, "step_s": step}
            for peak in PEAK_KEYS:
                value = (stats.get("memory") or {}).get(arm, {}).get(peak) or 0
                peaks[peak] = max(peaks[peak], value)
        steps = [entry["step_s"] for entry in per_window.values()]
        table[name] = {
            "per_window": per_window,
            "step_sum_s": None if None in steps else round(sum(steps), 2),
            "steps_rel_C": {window: round(stats[f"{pick(window)}_step_s"]
                                          / stats["C_step_s"], 3)
                            for window, stats in windows.items()},
            "step_sum_rel_C": round(sum(steps) / base, 3) if base and None not in steps else None,
            "peaks": peaks,
            "bits": {window: _bits_status(bits, window.strip("()")) for window in windows},
        }
    return table


def _valid_runs(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """A void run is never counted; it is replaced under ``speed-rule.md``."""
    return [r for r in rows if r.get("outcome", "ok") == "ok"]


def _excluded_blocks(folder: Path) -> set:
    path = folder / "excluded_blocks.json"
    if not path.exists():
        return set()
    return {(item["k"], item["seq"], item["block"])
            for item in json.loads(path.read_text(encoding="utf-8"))}


def _speed_rows(folder: Path) -> List[Dict[str, Any]]:
    excluded = _excluded_blocks(folder)
    return [row for row in _valid_runs(_read_jsonl(folder / "rows.jsonl"))
            if (row["k"], row["seq"], row.get("block", 0)) not in excluded]


def _bits(path: Optional[Path]) -> Optional[Dict[str, Optional[bool]]]:
    """The per-cell bit verdicts of ``bits-cross.json``, keyed ``"k,seq"``."""
    if path is None or not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {f"{cell['k']},{cell['seq']}": bool(cell["equal"]) for cell in data.get("cells", [])}


def _speed(folder: Path, *, bits: Optional[Mapping[str, Optional[bool]]] = None) -> Dict[str, Any]:
    rows = _speed_rows(folder)
    records = _read_jsonl(folder / "arm_records.jsonl")
    windows = {f"({k},{seq})": window_stats(rows, k=k, seq=seq) for k, seq in mm.ESC_WINDOWS}
    outcomes = Counter(record["outcome"] for record in records)
    clock_max = [record["gpu_clock"]["sm_clock_mhz"]["max"] for record in records]
    failed = Counter(name for record in records for name, check in record["checks"].items()
                     if check["outcome"] != "ok")
    return {"folder": str(folder), "windows": windows,
            "combinations": combination_table(windows, bits=bits),
            "outcomes": dict(outcomes), "check_failures": dict(failed),
            "gpu_clock_max_min": min((c for c in clock_max if c is not None), default=None),
            "blocks": max((r.get("block", 0) for r in rows), default=0) + 1,
            "runs": len(rows), "records": len(records)}


def main(argv: object = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--bits", type=Path, default=None,
                        help="bits-cross.json of the same campaign, for the bits column")
    args = parser.parse_args(argv)

    folder = args.root / SPEED_FOLDER
    speed = _speed(folder, bits=_bits(args.bits))
    analysis = {"speed": speed, "windows": [f"({k},{seq})" for k, seq in mm.ESC_WINDOWS]}
    (args.root / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    lines = ["# Result lines (ticket 16, C-S-E-E-S-C speed)", ""]
    lines += _result_lines(_speed_rows(folder))
    (args.root / "rows.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({name: {"step_sum_s": combo["step_sum_s"],
                             "step_sum_rel_C": combo["step_sum_rel_C"]}
                      for name, combo in speed["combinations"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
