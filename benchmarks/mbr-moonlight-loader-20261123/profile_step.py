#!/usr/bin/env python3
"""One normal-mode loader run under ``ArmWatch``, in-process, for a ``py-spy`` profile.

``py-spy record`` wraps this script, so the sampled stacks contain the timed ``L2LStep.run``
frames and the loader controller. The row and the ArmWatch record are written next to the
profile. One window, one arm, no sweep: the caller decides how many steps. Benchmark code, not
shipped.

    py-spy record --native --format speedscope -o H.json -- python profile_step.py ...

Needs the model cache, CUDA and the compiled extension on ``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for entry in (str(REPO), str(REPO / "benchmarks")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from benchmarks.harness import mbr_measure as mm  # noqa: E402
from benchmarks.harness.arm_validity import ArmWatch  # noqa: E402


def main(argv: object = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="H")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--source", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lora-targets", type=mm.parse_lora_targets,
                        default=mm.DEFAULT_LORA_TARGETS)
    args = parser.parse_args(argv)
    run_args = argparse.Namespace(
        source=args.source, cache=args.cache, out=args.out, steps=args.steps, lora_r=16,
        lora_targets=args.lora_targets, tokenizer=None, device="auto", kind="speed",
        deterministic=False, math_attention=False, profile=False, arm=args.arm, k=args.k,
        seq=args.seq, label=mm.DEFAULT_LABEL)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with ArmWatch(**mm.arm_watch_kwargs(run_args, arm=args.arm, round_number=1, run=1)) as watch:
        row = mm.run_one(run_args)
    stem = f"{args.arm}-k{args.k}-s{args.seq}"
    (out / f"row-{stem}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (out / f"arm-{stem}.json").write_text(json.dumps(watch.record), encoding="utf-8")
    print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
