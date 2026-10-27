#!/usr/bin/env python3
"""A5 ticket 10: tables for the report from one host's results directory.

``python l2l_ticket10_analyze.py RESULTS...`` where each RESULTS holds ``speed/`` and
``profile*/`` campaigns. Per run: tokens/s, the sensor flags re-judged from the saved samples
with the current ``judge_run`` (the campaign judged with the version of its day), the median
SM clock and the mean power while busy. Per profiled run: seconds per step of each part.
Benchmark harness code, not shipped.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import l2l_ticket10_profile as prof  # noqa: E402


def busy(samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Samples where the GPU was not idle (idle bit 0x1 clear)."""
    return [s for s in samples if s.get("gpu") and not (s["gpu"].get("throttle") or 0) & 1]


def run_row(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    samples = data.get("sensor_samples", [])
    verdict = prof.judge_run(samples)
    gpu = [s["gpu"] for s in busy(samples)]
    power = [g["power_w"] for g in gpu if g.get("power_w") is not None]
    row: Dict[str, Any] = {
        "run": path.stem,
        "arm": "GA" if "-A" in path.stem[-2:] else "L2L",
        "tokens_per_s": data["run"]["tokens_per_s"],
        "flags": verdict["flags"],
        "sm_clock_ratio": verdict["sensors"].get("sm_clock_ratio_median"),
        "power_w_mean": statistics.fmean(power) if power else None,
        "cpu_steal": verdict["sensors"].get("cpu_steal"),
        "layer_loads_per_step": data["run"]["layer_loads_per_step"],
    }
    if "parts_mean" in data:
        row["parts"] = data["parts_mean"]
    return row


def analyse(results: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for campaign in sorted(p for p in results.iterdir() if p.is_dir() and (p / "runs").exists()):
        rows = [run_row(f) for f in sorted((campaign / "runs").glob("*.json"))]
        summary = campaign / "summary.json"
        speed = json.loads(summary.read_text())["speed"] if summary.exists() else None
        out[campaign.name] = {"rows": rows, "speed": speed}
    return out


def main(argv: Sequence[str]) -> int:
    for arg in argv:
        print(json.dumps({arg: analyse(Path(arg))}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
