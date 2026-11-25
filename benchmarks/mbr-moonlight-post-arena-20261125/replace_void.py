#!/usr/bin/env python3
"""Replace the void runs of the C-E-G-G-E-C speed plan (ticket 17).

``speed-rule.md`` and ``METHOD.md`` say a block with a void run does not count and is replaced:
its records are kept, and the cell runs again. ``mbr_measure.py ceg-speed`` launches the plan
once and keeps every record, so a contaminated arm (another process reading the model disk, a
suspend, a power blip) leaves a hole in the paired gains. This script reads the rows written so
far, re-runs only the cells with no valid run, and appends them with the same block number.

A cell is settled by any row whose ``outcome`` is ``ok``; its void rows stay in
``rows.jsonl``/``arm_records.jsonl`` and ``analyze.py`` ignores them.

    python replace_void.py --out benchmarks/mbr-moonlight-post-arena-20261125/speed \
        --source $HOME/moonlight-local/source --cache $HOME/moonlight-local/cache

Benchmark code, not shipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO = Path(__file__).resolve().parents[2]
for entry in (str(REPO), str(REPO / "benchmarks")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from benchmarks.harness import mbr_measure as mm  # noqa: E402

CELL_FIELDS = ("arm", "k", "seq", "block")
DEFAULT_GATE = Path(__file__).resolve().parent.parent / "mbr-moonlight-emem2-20261124" / "gate.py"


def _cell(row: Dict[str, Any]) -> tuple:
    return tuple(row[name] for name in CELL_FIELDS)


def pending_cells(plan: Sequence[Dict[str, Any]],
                  rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Plan items whose run has no valid row yet, in plan order.

    A void row does not settle a run; an ``ok`` row does. Every cell (arm, window, block) is
    planned twice, so one ok run settles only one of the two plan items: the second stays
    pending until it has its own valid row. A replacement that was itself replaced counts
    (extra ok rows are ignored)."""
    planned = Counter(_cell(item) for item in plan)
    settled = Counter(_cell(row) for row in rows if row.get("outcome", "ok") == "ok")
    missing = {cell: planned[cell] - settled[cell] for cell in planned}
    pending: List[Dict[str, Any]] = []
    seen: Counter = Counter()
    for item in plan:
        cell = _cell(item)
        seen[cell] += 1
        if seen[cell] > settled[cell] and missing[cell] > 0:
            pending.append(item)
            missing[cell] -= 1
    return pending


def sweep_args(args: argparse.Namespace, folder: Path, *,
               gate: Sequence[str] = ()) -> argparse.Namespace:
    """The namespace ``mbr_measure.sweep`` reads for one replacement run."""
    return argparse.Namespace(
        source=args.source, cache=args.cache, out=str(folder), steps=args.steps,
        device=args.device, lora_r=args.lora_r, lora_targets=args.lora_targets,
        tokenizer=args.tokenizer, label=args.label, math_attention=False, profile=False,
        deadline_hours=0, gate=list(gate))


def _arm_records(folder: Path) -> int:
    path = folder / "arm_records.jsonl"
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)


def main(argv: object = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--start-block", type=int, default=0,
                        help="first block number the campaign ran (a resumed run continues "
                             "its numbering)")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-targets", type=mm.parse_lora_targets,
                        default=mm.DEFAULT_LORA_TARGETS)
    parser.add_argument("--label", default=mm.DEFAULT_LABEL)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--gate", type=Path, default=None,
                        help="health gate checked before every replacement run")
    parser.add_argument("--max-rounds", type=int, default=3,
                        help="replacement rounds; a cell still void after the last round is "
                             "reported and the window gets no verdict")
    args = parser.parse_args(argv)

    folder = args.out
    text = (folder / "rows.jsonl").read_text(encoding="utf-8")
    rows: List[Dict[str, Any]] = [json.loads(line) for line in text.splitlines() if line]
    plan = mm.ceg_speed_plan(blocks=args.blocks, start_block=args.start_block)
    gate = [sys.executable, str(args.gate)] if args.gate else []
    run_args = sweep_args(args, folder, gate=gate)
    print(json.dumps({"pending_round_0": len(pending_cells(plan, rows))}))
    for round_number in range(args.max_rounds):
        pending = pending_cells(plan, rows)
        if not pending:
            return 0
        offset = _arm_records(folder)
        for position, item in enumerate(pending):
            item = dict(item, index=offset + position)
            pending[position] = item
        print(json.dumps({"round": round_number + 1, "runs": len(pending)}), flush=True)
        rows += mm.sweep(run_args, pending)
    pending = pending_cells(plan, rows)
    print(json.dumps({"pending_after_rounds": len(pending),
                      "cells": [{name: item[name] for name in CELL_FIELDS} for item in pending]}))
    return 0 if not pending else 1


if __name__ == "__main__":
    raise SystemExit(main())
