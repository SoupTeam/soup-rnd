#!/usr/bin/env python3
"""A5 ticket 09: tables for the report from one card's campaign results.

For every L2L run: per warm step, the layer loads the step makes besides its captures
(``layer_loads`` minus the loads the capture events measured) and the capture time as a
share of the step. The first must be one number on every step if the captures are the whole
extra cost. Benchmark harness code, not shipped.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

PAIRS = ("ga-vs-after", "before-vs-after", "before-vs-before-bucket")


def own_loads(steps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Loads per warm step without the captures', and the capture share of the step time."""
    rows = [s for s in steps[1:] if "l2l_window" in s]
    own = [s["layer_loads"] - s["l2l_window"]["capture_layer_loads"] for s in rows]
    share = [s["l2l_window"]["capture_seconds"] / s["step_s"] for s in rows]
    return {
        "steps": len(rows),
        "own_loads": sorted(set(own)),
        "capture_share_mean": statistics.fmean(share) if share else None,
        "new_shapes_equal_captures": all(
            s["l2l_window"]["new_shapes"] == s["l2l_window"]["captures"] for s in rows
        ),
        "captures_by_new_shapes": sorted({
            (s["l2l_window"]["new_shapes"], s["layer_loads"]) for s in rows
        }),
    }


def analyse(results: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for pair in PAIRS:
        runs = sorted((results / pair / "runs").glob("*.json"))
        arms: Dict[str, List[Dict[str, Any]]] = {}
        for path in runs:
            data = json.loads(path.read_text())
            arms.setdefault(path.stem.split("-")[-1], []).append(own_loads(data["steps"]))
        out[pair] = {
            arm: {
                "runs": len(items),
                "own_loads": sorted({v for item in items for v in item["own_loads"]}),
                "capture_share_mean": statistics.fmean(
                    i["capture_share_mean"] for i in items if i["capture_share_mean"] is not None
                ) if any(i["capture_share_mean"] is not None for i in items) else None,
                "new_shapes_equal_captures": all(i["new_shapes_equal_captures"] for i in items),
            }
            for arm, items in arms.items()
        }
    return out


if __name__ == "__main__":
    print(json.dumps(analyse(Path(sys.argv[1])), indent=1))
