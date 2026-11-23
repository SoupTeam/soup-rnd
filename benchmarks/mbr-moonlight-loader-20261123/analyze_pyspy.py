#!/usr/bin/env python3
"""Share of the Python loader controller inside a step, from a ``py-spy`` speedscope profile.

The profile is recorded with ``--native``, so a sample that is blocked inside the compiled
two-slot stream has an extension frame as its leaf, while a sample that really runs Python in
``expert_loader.py`` has that Python frame as its leaf. Three numbers per profile:

- ``step_seconds``: sampled time whose stack contains ``l2l_step.py`` (the timed step, warm-up
  included);
- ``controller_leaf_share``: samples inside the step whose leaf is Python in
  ``expert_loader.py``, the pure-Python controller;
- ``controller_any_share``: samples inside the step with any ``expert_loader.py`` frame, an
  upper bound that also counts time blocked in the extension;
- ``extension_share``: samples inside the step with an extension frame.

    python analyze_pyspy.py --profile H.speedscope.json --profile G.speedscope.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


def _is_file(frame: Dict[str, Any], suffix: str) -> bool:
    return str(frame.get("file") or "").endswith(suffix)


def analyze(path: Path) -> Dict[str, Any]:
    profile = json.loads(path.read_text(encoding="utf-8"))
    frames = profile["shared"]["frames"]
    step_total = controller_leaf = controller_any = extension = total = 0.0
    leaves: Counter = Counter()
    for block in profile["profiles"]:
        samples = block.get("samples") or []
        weights = block.get("weights") or [1.0] * len(samples)
        for stack, weight in zip(samples, weights):
            total += weight
            names = [frames[index] for index in stack]
            if not any(_is_file(frame, "l2l_step.py") for frame in names):
                continue
            step_total += weight
            if any(_is_file(frame, "expert_loader.py") for frame in names):
                controller_any += weight
            if names and _is_file(names[-1], "expert_loader.py"):
                controller_leaf += weight
                leaves[names[-1].get("name") or "?"] += weight
            if any(_is_file(frame, "native_stream_async.cpp")
                   or "_native_expert_stream_async" in str(frame.get("name") or "")
                   or "_native_expert_stream_async" in str(frame.get("file") or "")
                   for frame in names):
                extension += weight

    def share(part: float) -> Optional[float]:
        return round(part / step_total, 4) if step_total else None

    return {
        "profile": str(path),
        "run_seconds": round(total, 2),
        "step_seconds": round(step_total, 2),
        "controller_leaf_seconds": round(controller_leaf, 2),
        "controller_leaf_share": share(controller_leaf),
        "controller_any_share": share(controller_any),
        "extension_share": share(extension),
        "controller_leaf_frames": dict(leaves.most_common(8)),
    }


def main(argv: object = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    rows: List[Dict[str, Any]] = [analyze(path) for path in args.profile]
    text = json.dumps(rows, indent=2)
    if args.out is None:
        print(text)
    else:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
