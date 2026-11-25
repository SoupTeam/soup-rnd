#!/usr/bin/env python3
"""Fill ticket 17's speed slots with complete, fresh C-E-G-G-E-C attempts.

The initial five slots per window have implicit IDs. Reserve every later attempt in
``attempts.jsonl`` before starting it. Even an attempt stopped before its first row
keeps its ID. Restarting this command can then replace it, never complete its tail.
Use the campaign's locked, sleep-inhibited launcher.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from rich.console import Console

if __package__:
    from .analyze import ATTEMPTS_FILE, _block_plan, mm, read_speed, speed_progress
else:
    from analyze import ATTEMPTS_FILE, _block_plan, mm, read_speed, speed_progress

DEFAULT_GATE = Path(__file__).resolve().parent.parent / "mbr-moonlight-emem2-20261124" / "gate.py"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-targets", type=mm.parse_lora_targets,
                        default=mm.DEFAULT_LORA_TARGETS)
    parser.add_argument("--label", default=mm.DEFAULT_LABEL)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--max-rounds", type=int, default=3,
                        help="attempt limit per outstanding slot in this invocation")
    args = parser.parse_args(argv)
    args.gate = [sys.executable, str(args.gate)]
    args.deadline_hours = 0
    args.command = "ceg-speed"
    args.require_complete = True
    identity = mm.campaign_identity(source=args.source, cache=args.cache,
                                    tokenizer=args.tokenizer, lora_r=args.lora_r,
                                    lora_targets=args.lora_targets, steps=args.steps)
    console = Console()
    attempted: Counter = Counter()
    while True:
        evidence = read_speed(args.out, identity)
        # An explicitly resumed pilot may have been refused before writing either file.
        fatal = [reason for reason in evidence.errors if not reason.startswith("missing:")]
        if fatal:
            console.print_json(data={"invalid_evidence": fatal})
            return 1
        foreign = {record.get("campaign_id") for record in evidence.rows + evidence.records}
        foreign -= {identity}
        if foreign:
            console.print_json(data={"foreign_campaign": sorted(map(str, foreign))})
            return 1
        progress = speed_progress(evidence)
        pending = progress["pending"]
        if not pending:
            extra = {window: slots for window, slots in progress["unexpected_slots"].items()
                     if slots}
            if extra or evidence.errors:
                console.print_json(data={"unexpected_slots": extra, "errors": evidence.errors})
                return 1
            console.print_json(data={"blocks_per_window": progress["final_target"]})
            return 0
        block = evidence.next_block
        for window, slot in pending:
            attempted[window, slot] += 1
            if attempted[window, slot] > args.max_rounds:
                console.print_json(data={"attempt_limit": {"window": window, "slot": slot}})
                return 1
            k, seq = window
            attempt = {"k": k, "seq": seq, "slot": slot, "block": block}
            # Flush the reservation before any subprocess starts. It survives an interrupted
            # launch and guarantees the next invocation cannot reuse this run range.
            with (args.out / ATTEMPTS_FILE).open("a", encoding="utf-8") as journal:
                journal.write(json.dumps(attempt) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
            console.print_json(data={"attempt": attempt})
            plan = _block_plan(window, block)
            if len(mm.sweep(args, plan)) != len(plan):
                console.print_json(data={"incomplete_sweep": attempt})
                return 1
            block += 1


if __name__ == "__main__":
    raise SystemExit(main())
