#!/usr/bin/env python3
"""Ticket 15: one normal-mode run of one cell under ``ArmWatch``, in-process.

``py-spy record`` wraps this script for the sampled profile, or ``--trace`` asks ``run_one`` for
one ``torch.profiler`` step. The row is appended to ``<out>/runs/rows.jsonl`` and the arm record to
``<out>/runs/arm_records.jsonl``; with ``--trace`` the traced step is also written to
``<out>/trace/<arm>-k<k>-s<seq>.json`` with the row and the arm record. One window, one arm, no
sweep: the caller decides the cells and the steps. Benchmark code, not shipped.

    py-spy record --native --format speedscope -o S-k64-s64.speedscope.json -- \
        python profile_step.py --arm S --k 64 --seq 64 --pyspy ...

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


def run_row(args: argparse.Namespace) -> "tuple[dict, dict]":
    """One run under ``ArmWatch``, its row and the arm record both appended under ``<out>/runs``."""
    out = Path(args.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    run_args = argparse.Namespace(
        source=args.source, cache=args.cache, out=args.out, steps=args.steps, lora_r=16,
        lora_targets=args.lora_targets, tokenizer=None, device="auto", kind="speed",
        deterministic=False, math_attention=False, profile=False, trace=bool(args.trace),
        arm=args.arm, k=args.k, seq=args.seq, label=mm.DEFAULT_LABEL)
    with ArmWatch(**mm.arm_watch_kwargs(run_args, arm=args.arm, round_number=1, run=1)) as watch:
        row = mm.run_one(run_args)
    if args.pyspy:  # the run is a py-spy snapshot; the seconds of such a row are not a speed
        row["pyspy"] = True
    mm.append_jsonl(out / "runs" / "rows.jsonl", row)
    mm.append_jsonl(out / "runs" / "arm_records.jsonl", watch.record)
    if args.trace and row.get("trace") is not None:
        (out / "trace").mkdir(parents=True, exist_ok=True)
        entry = {"row": {name: value for name, value in row.items() if name != "trace"},
                 "trace": row["trace"], "arm_record": watch.record}
        (out / "trace" / f"{args.arm}-k{args.k}-s{args.seq}.json").write_text(
            json.dumps(entry), encoding="utf-8")
    return row, watch.record


def main(argv: object = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="C")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--source", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lora-targets", type=mm.parse_lora_targets,
                        default=mm.DEFAULT_LORA_TARGETS)
    parser.add_argument("--trace", action="store_true",
                        help="add one torch-profiler step and write the trace next to the row")
    parser.add_argument("--pyspy", action="store_true",
                        help="mark the row as a py-spy snapshot; nothing else changes")
    args = parser.parse_args(argv)
    row, _ = run_row(args)
    print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
