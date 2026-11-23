#!/usr/bin/env python3
"""Re-run one block of the loader speed plan whose run was void, through the campaign path.

`replace_void.py` re-runs a *cell* (arm, k, seq, block) that has no valid run. `speed-rule.md`
says a *block* with a void run is not counted and is replaced, so a block whose one arm voided
stays counted with a single run of that arm. This script re-runs all four arms of the named
block, appending with the same block number; the void row and its record stay.

    python rerun_block.py --block 0 --k 64 --seq 64

Benchmark code, not shipped; one-off use is recorded in the campaign README.
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


def main(argv: object = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seq", type=int, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-targets", type=mm.parse_lora_targets,
                        default=mm.DEFAULT_LORA_TARGETS)
    parser.add_argument("--label", default=mm.DEFAULT_LABEL)
    parser.add_argument("--tokenizer", default=None)
    args = parser.parse_args(argv)

    folder = args.out
    records = [json.loads(line) for line in
               (folder / "arm_records.jsonl").read_text(encoding="utf-8").splitlines() if line]
    offset = len(records)
    plan = [{"arm": arm, "k": args.k, "seq": args.seq, "block": args.block,
             "index": offset + position, "kind": "speed", "deterministic": False}
            for position, arm in enumerate((mm.LOADER_PREVIOUS, mm.LOADER_ARM, mm.LOADER_ARM,
                                            mm.LOADER_PREVIOUS))]
    run_args = argparse.Namespace(
        source=args.source, cache=args.cache, out=str(folder), steps=args.steps,
        device="auto", lora_r=args.lora_r, lora_targets=args.lora_targets,
        tokenizer=args.tokenizer, label=args.label, math_attention=False, profile=False,
        deadline_hours=0)
    print(json.dumps({"block": args.block, "window": [args.k, args.seq], "runs": len(plan)}),
          flush=True)
    rows = mm.sweep(run_args, plan)
    print(json.dumps({"outcomes": [row.get("outcome") for row in rows]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
