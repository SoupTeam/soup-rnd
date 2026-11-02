#!/usr/bin/env python3
"""A5 ticket 05: A-B-B-A blocks of L2L on the synthetic 70B, watched by F1's ``ArmWatch``.

A synthetic 70B has no gradient-accumulation arm (it does not fit one card), so A and B are
the same L2L config (``l2l_synth70b/l2l.yaml``, weights from disk, k = 4). The blocks give the
run-to-run spread of one arm and F1's verdict per run; they do not compare two arms. Blocks
are added by the repeats rule until it asks for no more or ``--max-blocks`` is reached; a
block with a void run is kept on record and replaced. The summary is speed and memory only:
the weights are random, so there is no loss verdict. Benchmark harness code, not shipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS))

import l2l_measure  # noqa: E402
import l2l_ticket04_campaign as camp04  # noqa: E402

CONFIGS = HARNESS / "l2l_synth70b"
MODEL = "synthetic 70B"


class Campaign(camp04.Campaign):
    arm_config = {"A": "l2l.yaml", "B": "l2l.yaml"}
    configs = CONFIGS
    label = f"A5-05 {MODEL} {{card}}"

    def __init__(self, work: Path, out: Path, card: str, base: str):
        super().__init__(work, out, card)
        self.model_path = base  # the synthetic checkpoint F1 watches, not an HF cache

    def write_summary(self) -> None:
        kept = [b for b in self.blocks if not any(e["spoiled"] for e in b)]
        summary: Dict[str, Any] = {"model": MODEL, "card": self.card, "speed": None}
        if self.blocks:
            summary["speed"] = l2l_measure.summarize([
                [{**e, "measurement": e["measurement"]["run"]} for e in block]
                for block in self.blocks
            ])
            summary["metrics"] = {
                name: camp04.metric_summary(kept, getter)
                for name, getter in camp04.METRICS.items()
            }
        (self.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, help="directory with data/ and the synthetic base")
    parser.add_argument("--out", required=True)
    parser.add_argument("--card", required=True)
    parser.add_argument("--base", required=True, help="the synthetic checkpoint directory")
    parser.add_argument("--max-blocks", type=int, default=5)
    parsed = parser.parse_args()
    campaign = Campaign(Path(parsed.work), Path(parsed.out), parsed.card, parsed.base)
    campaign.abba(parsed.max_blocks)
    print(json.dumps(json.loads((campaign.out / "summary.json").read_text()), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
