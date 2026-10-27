#!/usr/bin/env python3
"""A5 ticket 10: A-B-B-A campaigns of GA against L2L on one instance, with sensors and parts.

The ticket-09 campaign with the ticket-10 arm. ``--profile`` runs every arm with the step
split into parts (synchronised, so slower: its speeds are not speed results). A block whose
sensors flagged the host (``sensors.disturbed``) is listed in ``summary.json`` under
``disturbed``. Benchmark harness code, not shipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS))

import l2l_ticket09_campaign as camp09  # noqa: E402

ARM_SCRIPT = HARNESS / "l2l_ticket10_arm.py"


def disturbed_runs(blocks: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Runs whose sensors raised a flag: block, arm and the flags."""
    return [
        {"block": i + 1, "arm": e["arm"], "flags": e["measurement"]["sensors"]["flags"]}
        for i, block in enumerate(blocks) for e in block
        if e["measurement"].get("sensors", {}).get("disturbed")
    ]


class Campaign(camp09.Campaign):
    profile = False

    def run_arm(self, stem: str, spec: Dict[str, Any]):  # type: ignore[override]
        camp09.ARM_SCRIPT = ARM_SCRIPT
        extra = ["--profile"] if self.profile else []
        real_run = camp09.subprocess.run

        def run(cmd: List[str], **kwargs: Any):
            return real_run([*cmd, *extra], **kwargs)

        camp09.subprocess.run = run
        try:
            return super().run_arm(stem, spec)
        finally:
            camp09.subprocess.run = real_run

    def write_summary(self) -> None:
        super().write_summary()
        path = self.out / "summary.json"
        summary = json.loads(path.read_text())
        summary["profile"] = self.profile
        summary["disturbed"] = disturbed_runs(self.blocks)
        summary["host_bandwidth"] = [
            e["measurement"].get("host_bandwidth") for b in self.blocks for e in b[:1]
        ]
        path.write_text(json.dumps(summary, indent=1, default=str))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--card", required=True)
    parser.add_argument("--tree", required=True, help="source tree of the arms")
    parser.add_argument("--a", default="ga", choices=sorted(camp09.ARMS))
    parser.add_argument("--b", default="after", choices=sorted(camp09.ARMS))
    parser.add_argument("--max-blocks", type=int, default=4)
    parser.add_argument("--profile", action="store_true")
    parsed = parser.parse_args()
    campaign = Campaign(Path(parsed.work), Path(parsed.out), parsed.card,
                        {"before": parsed.tree, "after": parsed.tree}, parsed.a, parsed.b)
    campaign.profile = parsed.profile
    campaign.abba(parsed.max_blocks)
    print(json.dumps(json.loads((campaign.out / "summary.json").read_text()), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
