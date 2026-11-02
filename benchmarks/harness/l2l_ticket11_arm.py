#!/usr/bin/env python3
"""A5 ticket 11: the ticket-10 arm for runs with or without layer streaming.

Runs ``l2l_ticket10_arm.main`` and adds ``end_peaks`` to the record: the VRAM peaks the
process holds when the run ends, which for an out-of-memory run is the peak at the failure
(the per-step peaks are lost when a step does not finish). Benchmark harness code, not shipped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    out = argv[argv.index("--out") + 1]
    import l2l_ticket10_arm

    status = l2l_ticket10_arm.main(argv)
    import torch

    peaks = None
    if torch.cuda.is_available():
        peaks = {"allocated": int(torch.cuda.max_memory_allocated()),
                 "reserved": int(torch.cuda.max_memory_reserved())}
    path = Path(out)
    if path.exists():
        record = json.loads(path.read_text())
        record["end_peaks"] = peaks
        path.write_text(json.dumps(record, indent=1, default=str))
    return status


if __name__ == "__main__":
    sys.exit(main())
