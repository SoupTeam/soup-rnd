#!/usr/bin/env python3
"""A5 ticket 06: the readiness verdict for L2L SFT, applied to the raw records.

The rule is in ``benchmarks/l2l-readiness-rule-a5.md`` (written before this script). R1: loss
within the A3-07 tolerance in every ticket-04 rerun campaign. R2: the mixed-length forward and
backward checks of ticket 07, rerun on the final code (GA backward outside autocast), all
pass. R3: L2L peak VRAM at most the GA peak in every campaign.
A criterion with no records fails. R1 and R3 read arms, so every arm they read needs each
required validity check ``ok`` (``benchmarks/METHOD.md``): a required check that is ``unknown``
gives "no verdict" ("нет вердикта"), which is not "not ready". A ``void`` arm is excluded and
counted (the campaign leaves its block out of the summary and the repeats rule adds one).
With all three met the verdict is "ready with limits" while ``LIMITS`` is not
empty. The script reads files and runs nothing on a GPU. Benchmark harness code, not shipped.
"""

from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

RESULTS = Path(__file__).resolve().parents[1] / "results"

#: The only ticket-04 set the gate reads: measured where ``foreign_reads`` answers (a vast local
#: volume), see ``benchmarks/l2l-readiness-rule-a5.md``. The earlier sets are history.
VALID_SET = "l2l-ticket04-valid-5070ti"

#: What the records do not cover. Not computed from the data; a limit leaves this list when
#: a measurement for it exists.
LIMITS = [
    "weight reads from disk not measured (page cache warm, counter near 0)",
    "activation spill to a separate disk not measured (A4 spill is not wired into the trainer)",
    "one model family measured (Mistral-7B v0.1 NF4); the verdict covers one card and one host "
    "(RTX 5070 Ti 16 GB); earlier cards have foreign_reads unknown and are not read",
    "speed: the F1 repeats rule was not met at the 7-block cap on this host, so no speed claim",
    "real 70B and 1T not measured (the 70B run uses random weights of its shape)",
]


NO_VERDICT = "нет вердикта"


def decide(evidence: Dict[str, List[Dict[str, Any]]], limits: Optional[List[str]] = None
           ) -> Dict[str, Any]:
    """Apply the rule to ``evidence`` (lists of records per criterion)."""
    limits = LIMITS if limits is None else limits
    all_arms = evidence.get("arms", [])
    arms = [arm for arm in all_arms if arm["outcome"] != "void"]
    blocked = _blocked(arms)
    if not arms:
        blocked = {"no usable arm records": 1}
    criteria = {
        "R1": bool(evidence["r1"]) and all(item["pass"] for item in evidence["r1"]),
        "R2": bool(evidence["r2"]) and all(item["pass"] for item in evidence["r2"]),
        "R3": bool(evidence["r3"]) and all(item["l2l"] <= item["ga"] for item in evidence["r3"]),
    }
    if blocked:
        verdict = NO_VERDICT
    elif not all(criteria.values()):
        verdict = "не готов"
    elif limits:
        verdict = "готов с ограничениями"
    else:
        verdict = "готов"
    return {"verdict": verdict, "criteria": criteria, "limits": limits, "blocked_by": blocked,
            "arm_outcomes": dict(Counter(arm["outcome"] for arm in all_arms)),
            "excluded_void_arms": len(all_arms) - len(arms), "evidence": evidence}


def _blocked(arms: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count of arms per required check that is not ``ok`` (``check:outcome`` -> arms)."""
    count: Counter = Counter()
    for arm in arms:
        for name in arm["required"]:
            outcome = arm["checks"].get(name, "unknown")
            if outcome != "ok":
                count[f"{name}:{outcome}"] += 1
    return dict(count)


def collect(results: Path = RESULTS) -> Dict[str, List[Dict[str, Any]]]:
    """The evidence from the committed raw records under ``results``."""
    r1, r3, r2, arms = [], [], [], []
    pattern = str(results / VALID_SET / "results" / "campaign" / "summary.json")
    for path in sorted(glob.glob(pattern)):
        summary = json.loads(Path(path).read_text())
        source = str(Path(path).relative_to(results))
        r1.append({"source": source, "pass": bool(summary["loss"]["pass"])})
        peak = summary["metrics"]["peak_vram_allocated_bytes"]
        r3.append({"source": source, "ga": peak["a"], "l2l": peak["b"]})
        blocks = json.loads((Path(path).parent / "blocks.json").read_text())
        for entry in (arm for block in blocks for arm in block):
            record = entry["arm_record"]
            arms.append({
                "source": source, "arm": entry["arm"], "round": entry["round"],
                "run": entry["run"], "outcome": record["outcome"],
                "required": list(record["required"]),
                "checks": {name: check["outcome"] for name, check in record["checks"].items()},
            })
    pattern = str(results / "l2l-mistral7b-ticket07-rerun-final" / "*" / "mixed-*" / "records.json")
    for path in sorted(glob.glob(pattern)):
        record = json.loads(Path(path).read_text())
        r2.append({"source": str(Path(path).relative_to(results)),
                   "pass": bool(record["overall"]["pass"])})
    return {"r1": r1, "r2": r2, "r3": r3, "arms": arms}


def main() -> int:
    print(json.dumps(decide(collect()), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
