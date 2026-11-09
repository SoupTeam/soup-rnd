"""B5 ticket 03: acceptance verdict for the small real MoE run (Moonlight-16B-A3B).

Reads the journal of one run and answers pass or fail under
``benchmarks/b5-moonlight-acceptance-20261109.md``. The adapter questions (nonzero finite
gradients, the adapter moved, the digest after reload) belong to ``b5_judge.py`` and are asked
there. This module adds what the judge does not know: the number of steps, the card, whether
the peaks fit its memory, and the setup of the run.

Benchmark harness code, not shipped. No torch import.

    python benchmarks/harness/b5_acceptance.py --journal events.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from rich.console import Console

try:
    from benchmarks.harness import b5_judge
except ImportError:  # run as a script from benchmarks/harness
    import b5_judge  # type: ignore[no-redef]

console = Console()

MODEL = "moonshotai/Moonlight-16B-A3B"
REVISION = "476b36a473d4467f94469414bef6cee75c9c8172"
#: Optimizer steps of the rule. The judge needs two; the rule asks for exactly four.
STEPS = 4
#: Tokens of every step. Each text is cut to exactly this many tokens, so nothing is padded.
TOKENS_PER_STEP = 512
#: The 8 GB class: the card's ``total_memory`` in bytes, bounds included.
CARD_MIN_BYTES = int(7.5 * 2**30)
CARD_MAX_BYTES = int(8.6 * 2**30)
#: bf16 without emulation.
MIN_CAPABILITY = (8, 0)
#: What the run must say about its setup, field by field.
SETUP = {"quant": "nf4", "double_quant": False, "expert_layout": "per_expert", "tier": "disk",
         "trust_remote_code": False}


def _events(journal: Sequence[Mapping[str, Any]], name: str) -> List[Mapping[str, Any]]:
    return [row for row in journal if row.get("event") == name]


def _check_signature(journal: Sequence[Mapping[str, Any]], failures: List[str]) -> None:
    rows = _events(journal, "signature")
    if not rows:
        return  # the judge already reports a missing signature
    row = rows[0]
    if row.get("kind") != "real":
        failures.append(f"signature: kind {row.get('kind')!r}, a Moonlight report is 'real'")
    if row.get("model") != MODEL or row.get("revision") != REVISION:
        failures.append(f"signature: model {row.get('model')!r} revision {row.get('revision')!r}"
                        f" is not {MODEL} at {REVISION}")


def _check_environment(journal: Sequence[Mapping[str, Any]], failures: List[str]) -> float:
    """Checks the card and the setup. Returns the card memory, or 0 when it is unknown."""
    rows = _events(journal, "environment")
    if len(rows) != 1:
        failures.append(f"environment: {len(rows)} environment events, one is required")
        return 0
    row = rows[0]
    for key, want in SETUP.items():
        if row.get(key) != want:
            failures.append(f"environment: {key} is {row.get(key)!r}, the rule needs {want!r}")
    memory = row.get("vram_bytes")
    capability = row.get("compute_capability")
    if row.get("device_count") != 1:
        failures.append(f"card: device_count {row.get('device_count')!r}, one card is required")
    if not isinstance(memory, int) or not CARD_MIN_BYTES <= memory <= CARD_MAX_BYTES:
        failures.append(f"card: total_memory {memory!r} is outside the 8 GB class "
                        f"{CARD_MIN_BYTES} to {CARD_MAX_BYTES}")
        memory = 0
    if (not isinstance(capability, list) or len(capability) != 2
            or tuple(capability) < MIN_CAPABILITY):
        failures.append(f"card: compute capability {capability!r} is below {MIN_CAPABILITY}")
    return memory


def _check_steps(journal: Sequence[Mapping[str, Any]], card: float,
                 failures: List[str]) -> None:
    for row in _events(journal, "step"):
        label = f"step {row.get('step')}"
        if row.get("tokens") != TOKENS_PER_STEP:
            failures.append(f"padding: {label} has {row.get('tokens')!r} tokens, "
                            f"the rule has {TOKENS_PER_STEP} and no padding")
        for key, code in (("peak_allocated", "vram_allocated"), ("peak_reserved", "vram_reserved")):
            peak = row.get(key)
            if isinstance(peak, bool) or not isinstance(peak, (int, float)):
                failures.append(f"peak_missing: {label} has no numeric {key}")
            elif card and peak > card:
                failures.append(f"{code}: {label} {key} {peak:g} is above the card memory "
                                f"{card:g}")


def _throughput(journal: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Tokens per second without padding, or why it is not a measurement."""
    clocks = _events(journal, "training_clock")
    if not clocks or not clocks[0].get("samples"):
        return {"status": "not measured", "reason": "no clock samples were taken during the steps"}
    samples, throttled = clocks[0]["samples"], clocks[0].get("throttled_samples", 0)
    if throttled * 2 >= samples:
        return {"status": "not measured",
                "reason": f"measurement spoiled: the GPU was throttled in {throttled} of "
                          f"{samples} clock samples"}
    speeds = [row.get("tokens_per_second") for row in _events(journal, "step")]
    if len(speeds) < 2 or not all(isinstance(value, (int, float)) for value in speeds):
        return {"status": "not measured", "reason": "a step has no tokens_per_second"}
    return {"status": "measured", "first_step": speeds[0],
            "later_steps_median": statistics.median(speeds[1:])}


def _measured(journal: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    steps = _events(journal, "step")

    def peak(key: str) -> Optional[float]:
        values = [row[key] for row in steps if isinstance(row.get(key), (int, float))]
        return max(values) if values else None

    return {
        "peak_allocated_bytes": peak("peak_allocated"),
        "peak_reserved_bytes": peak("peak_reserved"),
        "expert_fetches_per_step": [row.get("expert_fetches") for row in steps],
        "expert_bytes_read_per_step": [row.get("expert_bytes_read") for row in steps],
        "tokens_per_second": _throughput(journal),
    }


def verdict(journal: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Pass or fail for one journal, with every reason for a fail."""
    judged = b5_judge.judge([journal])
    failures: List[str] = []
    if not judged["passed"]:
        failures += [f"judge: {reason}" for reason in judged["failures"]]
    _check_signature(journal, failures)
    count = len(_events(journal, "step"))
    if count != STEPS:
        failures.append(f"steps: {count} step events, the rule has exactly {STEPS}")
    _check_steps(journal, _check_environment(journal, failures), failures)
    if not _events(journal, "complete"):
        failures.append("incomplete: no complete event, the run did not reach the end")
    return {"passed": not failures, "failures": failures, "judge": judged,
            "measured": _measured(journal)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", required=True)
    args = parser.parse_args(argv)
    result = verdict(b5_judge.read_journal(Path(args.journal)))
    console.out(json.dumps(result, indent=1, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
