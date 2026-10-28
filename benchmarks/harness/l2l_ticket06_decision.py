#!/usr/bin/env python3
"""A5 ticket 06: the readiness verdict for L2L SFT, applied to the raw records.

The rule is in ``benchmarks/l2l-readiness-rule-a5.md`` (written before this script). R1: loss
within the A3-07 tolerance in every ticket-04 rerun campaign. R2: the mixed-length forward and
backward checks of ticket 07 all pass. R3: L2L peak VRAM at most the GA peak in every campaign.
A criterion with no records fails. With all three met the verdict is "ready with limits" while
``LIMITS`` is not empty. The script reads files and runs nothing on a GPU. Benchmark harness
code, not shipped.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

RESULTS = Path(__file__).resolve().parents[1] / "results"

#: What the records do not cover. Not computed from the data; a limit leaves this list when
#: a measurement for it exists.
LIMITS = [
    "weight reads from disk not measured (page cache warm, counter near 0)",
    "activation spill to a separate disk not measured (A4 spill is not wired into the trainer)",
    "one model family measured (Mistral-7B v0.1 NF4); one card per set",
    "real 70B and 1T not measured (the 70B run uses random weights of its shape)",
]


def decide(evidence: Dict[str, List[Dict[str, Any]]], limits: Optional[List[str]] = None
           ) -> Dict[str, Any]:
    """Apply the rule to ``evidence`` (lists of records per criterion)."""
    limits = LIMITS if limits is None else limits
    criteria = {
        "R1": bool(evidence["r1"]) and all(item["pass"] for item in evidence["r1"]),
        "R2": bool(evidence["r2"]) and all(item["pass"] for item in evidence["r2"]),
        "R3": bool(evidence["r3"]) and all(item["l2l"] <= item["ga"] for item in evidence["r3"]),
    }
    if not all(criteria.values()):
        verdict = "не готов"
    elif limits:
        verdict = "готов с ограничениями"
    else:
        verdict = "готов"
    return {"verdict": verdict, "criteria": criteria, "limits": limits, "evidence": evidence}


def collect(results: Path = RESULTS) -> Dict[str, List[Dict[str, Any]]]:
    """The evidence from the committed raw records under ``results``."""
    r1, r3, r2 = [], [], []
    pattern = str(results / "l2l-mistral7b-ticket04-rerun-sft" / "*" / "results" / "campaign"
                  / "summary.json")
    for path in sorted(glob.glob(pattern)):
        summary = json.loads(Path(path).read_text())
        source = str(Path(path).relative_to(results))
        r1.append({"source": source, "pass": bool(summary["loss"]["pass"])})
        peak = summary["metrics"]["peak_vram_allocated_bytes"]
        r3.append({"source": source, "ga": peak["a"], "l2l": peak["b"]})
    pattern = str(results / "l2l-mistral7b-ticket07-nopad" / "*" / "mixed-*" / "records.json")
    for path in sorted(glob.glob(pattern)):
        record = json.loads(Path(path).read_text())
        r2.append({"source": str(Path(path).relative_to(results)),
                   "pass": bool(record["overall"]["pass"])})
    return {"r1": r1, "r2": r2, "r3": r3}


def main() -> int:
    print(json.dumps(decide(collect()), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
