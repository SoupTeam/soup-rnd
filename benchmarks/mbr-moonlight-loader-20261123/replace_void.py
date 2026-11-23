#!/usr/bin/env python3
"""Replace the void runs of the loader speed plan (ticket 05, part 2).

``speed-rule.md`` and ``METHOD.md`` say a block with a void run does not count and is replaced:
its records are kept, and the cell runs again. ``mbr_measure.py loader-speed`` launches the plan
once and keeps every record, so a contaminated arm (another process reading the model disk, a
suspend, a power blip) leaves a hole in the paired gains. This script reads the rows written so
far, re-runs only the cells with no valid run, and appends them with the same block number.

A cell is settled by any row whose ``outcome`` is ``ok``; its void rows stay in
``rows.jsonl``/``arm_records.jsonl`` and ``analyze.py`` ignores them.

    python replace_void.py --out benchmarks/mbr-moonlight-loader-20261123/speed \
        --source $HOME/moonlight-local/source --cache $HOME/moonlight-local/cache

Benchmark code, not shipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO = Path(__file__).resolve().parents[2]
for entry in (str(REPO), str(REPO / "benchmarks")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from benchmarks.harness import mbr_measure as mm  # noqa: E402

CELL_FIELDS = ("arm", "k", "seq", "block")


def _cell(row: Dict[str, Any]) -> tuple:
    return tuple(row[name] for name in CELL_FIELDS)


def pending_cells(plan: Sequence[Dict[str, Any]],
                  rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Plan items whose cell has no valid run yet, in plan order.

    A void row does not settle its cell; an ``ok`` row does, whatever came before it. A cell
    with several ok rows (a replacement that was itself replaced) is settled."""
    settled = {_cell(row) for row in rows if row.get("outcome", "ok") == "ok"}
    return [item for item in plan if _cell(item) not in settled]


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
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-targets", type=mm.parse_lora_targets,
                        default=mm.DEFAULT_LORA_TARGETS)
    parser.add_argument("--label", default=mm.DEFAULT_LABEL)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--max-rounds", type=int, default=3,
                        help="replacement rounds; a cell still void after the last round is "
                             "reported and the window gets no verdict")
    args = parser.parse_args(argv)

    folder = args.out
    text = (folder / "rows.jsonl").read_text(encoding="utf-8")
    rows: List[Dict[str, Any]] = [json.loads(line) for line in text.splitlines() if line]
    plan = mm.loader_speed_plan(blocks=args.blocks)
    run_args = argparse.Namespace(
        source=args.source, cache=args.cache, out=str(folder), steps=args.steps,
        device=args.device, lora_r=args.lora_r, lora_targets=args.lora_targets,
        tokenizer=args.tokenizer, label=args.label, math_attention=False, profile=False)
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
