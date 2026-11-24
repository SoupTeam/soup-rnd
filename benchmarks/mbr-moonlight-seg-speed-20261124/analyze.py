#!/usr/bin/env python3
"""Ticket 14 tail: the price of the segmented expert GEMM (S) against C and A0.

Reads the speed blocks written by ``run_speed.sh`` (``speed/rows.jsonl`` and
``speed/arm_records.jsonl`` under ``--root``) and writes:

- ``analysis.json``: per window the A0/C/S step means, the S-against-C gain (the price of
  segmentation), the S-against-A0 gain and the remaining C/A0 and S/A0 speedups, the paired t
  over the five blocks, the memory peaks of every hand, and the ArmWatch summary;
- ``rows.md``: one result line per measured cell, each naming model, card, window, arm, mode
  and torch version (the reporting rule of the map).

The per-block number is the median of an arm's two runs in that block; the window number is the
mean of the per-block medians, as ``speed-rule.md`` says. The paired t uses the per-block gains
and the 5-block critical value 2.78. Benchmark code, not shipped.

    python analyze.py --root benchmarks/mbr-moonlight-seg-speed-20261124
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
for entry in (str(REPO), str(REPO / "benchmarks")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from benchmarks.harness import mbr_measure as mm  # noqa: E402

MODEL = "Moonlight-16B-A3B"
CARD = "RTX 3050 Laptop sm_86 4 GiB"
SPEED_FOLDER = "speed"
ARMS = ("A0", "C", "S")
ARM_LABELS = {"A0": "plain L2L", "C": "ordered backward", "S": "C + segmented"}
PEAK_KEYS = ("peak_allocated", "peak_reserved", "peak_nvml", "peak_ram")
# A loss up to 3% of C is "almost free"; 2x or more is a separate point on the scale.
ALMOST_FREE = 0.03
MULTIPLES = 0.50


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


def cost_reading(gain: float) -> Dict[str, Any]:
    """The ticket's scale for the S-against-C gain: 0 is free, -1 is twice as slow."""
    loss = -gain
    if loss <= ALMOST_FREE:
        kind = "almost_free"
    elif loss >= MULTIPLES:
        kind = "multiples"
    else:
        kind = "visible"
    return {"class": kind, "loss_pct": round(loss * 100, 2)}


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
    """The three hands' step means, both deltas with their paired t, and the memory peaks."""
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

    s_vs_c = gains("S", "C")
    a0, c, s = means["A0"], means["C"], means["S"]
    table = mm.summarize_rows(rows)
    per_block = {str(block): {arm: round(steps[arm][block], 3)
                              for arm in ARMS if block in steps[arm]}
                 for block in sorted(set().union(*steps.values()))}
    return {
        "A0_step_s": None if a0 is None else round(a0, 2),
        "C_step_s": None if c is None else round(c, 2),
        "S_step_s": None if s is None else round(s, 2),
        "per_block": per_block,
        "S_vs_C": s_vs_c,
        "S_vs_A0": gains("S", "A0") if 0 in steps["A0"] else None,
        "speedup_C_over_A0": round(a0 / c, 2) if a0 and c else None,
        "speedup_S_over_A0": round(a0 / s, 2) if a0 and s else None,
        "cost": None if s_vs_c is None else cost_reading(s_vs_c["mean_gain_pct"] / 100),
        "memory": {arm: {name: table.get((arm, k, seq), {}).get(name) for name in PEAK_KEYS}
                   for arm in ARMS},
    }


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


def _speed(folder: Path) -> Dict[str, Any]:
    rows = _speed_rows(folder)
    records = _read_jsonl(folder / "arm_records.jsonl")
    windows = {f"({k},{seq})": window_stats(rows, k=k, seq=seq)
               for k, seq in mm.SEGMENTED_WINDOWS}
    outcomes = Counter(record["outcome"] for record in records)
    clock_max = [record["gpu_clock"]["sm_clock_mhz"]["max"] for record in records]
    failed = Counter(name for record in records for name, check in record["checks"].items()
                     if check["outcome"] != "ok")
    return {"folder": str(folder), "windows": windows, "outcomes": dict(outcomes),
            "check_failures": dict(failed),
            "gpu_clock_max_min": min((c for c in clock_max if c is not None), default=None),
            "blocks": max((r.get("block", 0) for r in rows), default=0) + 1,
            "runs": len(rows), "records": len(records)}


def main(argv: object = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args(argv)

    folder = args.root / SPEED_FOLDER
    speed = _speed(folder)
    analysis = {"speed": speed, "windows": [f"({k},{seq})" for k, seq in mm.SEGMENTED_WINDOWS]}
    (args.root / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    lines = ["# Result lines (ticket 14 tail, segmented speed)", ""]
    lines += _result_lines(_speed_rows(folder))
    (args.root / "rows.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(speed, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
