#!/usr/bin/env python3
"""Leaf shares inside a step, from a ``py-spy`` speedscope profile (ticket 15 campaign).

The sampler is ``py-spy record --native``, so a stack can end in Python (``expert_gemm.py``,
``/torch/``) or in native code (cuBLAS launches, the disk read path). Samples whose stack contains
``l2l_step.py`` are the timed step; the leaf frame of each such sample is put in one class:

- ``expert_gemm``: Python in ``utils/expert_gemm.py``, the segmented GEMM dispatch;
- ``cuda_runtime``: a CUDA runtime launch, by name prefix or ``libcudart``/``libcuda``;
- ``disk``: a read path (``read``/``pread``/``posix``/``io_uring``);
- ``torch_python``: any Python frame under ``/torch/``;
- ``other_python``: any other Python file;
- ``native_other``: everything else.

The first matching class wins, in that order. Benchmark code, not shipped.

    python analyze_pyspy.py --profile C-k8-s64.speedscope.json --out leaf-shares.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

DISK_PARTS = ("read", "pread", "posix", "io_uring")


def _text(value: Any) -> str:
    return str(value or "")


def leaf_class(frame: Dict[str, Any]) -> str:
    """The class of one leaf frame, first match of the class order."""
    name, file = _text(frame.get("name")), _text(frame.get("file"))
    if file.endswith("utils/expert_gemm.py"):
        return "expert_gemm"
    if name.startswith(("cuda", "cu")) or "libcudart" in file or "libcuda" in file:
        return "cuda_runtime"
    if any(part in name or part in file for part in DISK_PARTS):
        return "disk"
    if "/torch/" in file:
        return "torch_python"
    if file.endswith(".py"):
        return "other_python"
    return "native_other"


def classify(profile: Dict[str, Any]) -> Dict[str, Any]:
    """The step's sampled seconds and the share of each leaf class (``None`` without a step)."""
    frames = profile["shared"]["frames"]
    step_total = 0.0
    seconds: Counter = Counter()
    leaves: Counter = Counter()
    for block in profile["profiles"]:
        samples = block.get("samples") or []
        weights = block.get("weights") or [1.0] * len(samples)
        for stack, weight in zip(samples, weights):
            names = [frames[index] for index in stack]
            if not any(_text(frame.get("file")).endswith("l2l_step.py") for frame in names):
                continue
            step_total += weight
            if not names:
                continue
            leaf = names[-1]
            kind = leaf_class(leaf)
            seconds[kind] += weight
            if kind == "expert_gemm":
                leaves[_text(leaf.get("name")) or "?"] += weight

    def share(part: float) -> Optional[float]:
        return round(part / step_total, 4) if step_total else None

    return {
        "step_seconds": round(step_total, 2),
        "expert_gemm_seconds": round(seconds["expert_gemm"], 2),
        "expert_gemm_share": share(seconds["expert_gemm"]),
        "cuda_runtime_share": share(seconds["cuda_runtime"]),
        "disk_share": share(seconds["disk"]),
        "torch_python_share": share(seconds["torch_python"]),
        "other_python_share": share(seconds["other_python"]),
        "native_other_share": share(seconds["native_other"]),
        "expert_gemm_leaf_frames": dict(leaves.most_common(8)),
    }


def analyze(path: Path) -> Dict[str, Any]:
    """``classify`` on the speedscope document written to ``path``, plus its name."""
    result = classify(json.loads(Path(path).read_text(encoding="utf-8")))
    result["profile"] = str(path)
    return result


def main(argv: object = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    rows: List[Dict[str, Any]] = [analyze(path) for path in args.profile]
    text = json.dumps(rows, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
