#!/usr/bin/env python3
"""A5 ticket 07: A-B-B-A blocks of fresh-process arms, watched by F1, until the repeats rule
asks for no more blocks. A is the code before the change (padded), B is the code after it.

``--before`` and ``--after`` are source trees (each with a ``src/``). Benchmark harness code,
not shipped.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arm_validity  # noqa: E402
import l2l_measure  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-blocks", type=int, default=5)
    parser.add_argument("--card", required=True)
    parsed = parser.parse_args()
    root = Path(parsed.out)
    root.mkdir(parents=True, exist_ok=False)
    trees = {"A": parsed.before, "B": parsed.after}
    blocks, runs = [], []
    arm_script = Path(__file__).resolve().parent / "l2l_ticket07_arm.py"
    while True:
        if len(blocks) >= arm_validity.MIN_BLOCKS and not arm_validity.blocks_to_add(runs):
            break
        if len(blocks) >= parsed.max_blocks:
            print("block cap reached; incomplete")
            break
        block, spans = [], []
        round_no = len(blocks) + 1
        for position, name in enumerate(arm_validity.BLOCK_ORDER):
            stem = f"b{round_no}-p{position}-{name}"
            out = root / f"{stem}.json"
            env = dict(os.environ, PYTHONPATH=str(Path(trees[name]) / "src"),
                       TOKENIZERS_PARALLELISM="false")
            with arm_validity.ArmWatch(arm=name, round=round_no, run=position,
                                       label=f"A5-07 {parsed.card}") as watch:
                with (root / f"{stem}.log").open("w") as log:
                    done = subprocess.run(
                        [sys.executable, str(arm_script), "--out", str(out)],
                        stdout=log, stderr=subprocess.STDOUT, env=env, timeout=1800,
                        check=False,
                    )
            if done.returncode:
                raise RuntimeError(f"{stem} exited {done.returncode}; see {stem}.log")
            data = json.loads(out.read_text())
            entry = l2l_measure.run_entry(
                data["run"], watch.record, arm=name, round=round_no, run=position
            )
            block.append(entry)
            spans.append(arm_validity.Run(
                data["run"]["tokens_per_s"],
                arm_validity.VOID if entry["spoiled"] else entry["outcome"],
            ))
            print(stem, f"{data['run']['tokens_per_s']:.1f} tokens/s", entry["outcome"], flush=True)
        blocks.append(block)
        runs.append(spans)
        (root / "blocks.json").write_text(json.dumps(blocks, indent=1, default=str))
    summary = l2l_measure.summarize(blocks)
    (root / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
