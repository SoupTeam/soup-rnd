"""B5 ticket 06: acceptance verdict for the synthetic 671B cold-disk run.

Reads the journal of one run and the record of the store write and answers pass or fail under
``benchmarks/b5-synthetic-671b-acceptance-20261109.md``. The adapter questions (nonzero finite
gradients, the adapter moved, the digest after reload) belong to ``b5_judge.py``. This module
adds what the judge does not know: the shape, the number of steps, the card, the store size
against its plan, the free disk, and whether the start was cold.

Every number below was fixed before the first rental and must not be tuned after a run. A rule
that turns out wrong gets a new date and a new file.

Benchmark harness code, not shipped. No torch import.

    python benchmarks/harness/b5_cold_acceptance.py --journal journal.jsonl --store-record r.json
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

#: Optimizer steps of the rule. The judge needs two; the rule asks for exactly three.
STEPS = 3
#: Tokens of every step: one batch of one random sequence, nothing padded.
TOKENS_PER_STEP = 128
#: The DeepSeek-V3 shape, field by field, as the ``shape`` event of the journal states it.
SHAPE = {"layers": 61, "moe_layers": 58, "experts": 256, "hidden": 7168, "expert_size": 2048,
         "parameters": 671_026_419_200}
#: The 8 GB class: the card's ``total_memory`` in bytes, bounds included.
CARD_MIN_BYTES = int(7.5 * 2**30)
CARD_MAX_BYTES = int(8.6 * 2**30)
#: bf16 without emulation.
MIN_CAPABILITY = (8, 0)
#: What the run must say about its setup, field by field.
SETUP = {"quant": "nf4", "double_quant": False, "expert_layout": "per_expert", "tier": "disk",
         "trust_remote_code": False}
#: Plan against actual store bytes, as a fraction of the plan (``b5_synth_store.SIZE_TOLERANCE``).
SIZE_TOLERANCE = 0.01
#: After the drop at most this fraction of the store bytes may sit in the page cache.
RESIDENT_MAX_FRACTION = 0.01
#: The first step must read at least this fraction of the logical expert bytes from the disk.
STEP0_READ_MIN_FRACTION = 0.9
#: Longest wait between the cache drop and the start of training, in seconds.
MAX_DROP_TO_TRAINING_SECONDS = 600.0


def _events(journal: Sequence[Mapping[str, Any]], name: str) -> List[Mapping[str, Any]]:
    return [row for row in journal if row.get("event") == name]


def _number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _check_signature(journal: Sequence[Mapping[str, Any]], failures: List[str]) -> None:
    rows = _events(journal, "signature")
    if rows and rows[0].get("kind") != "synthetic":
        failures.append(f"signature: kind {rows[0].get('kind')!r}, this report is 'synthetic'")


def _check_shape(journal: Sequence[Mapping[str, Any]], failures: List[str]) -> None:
    rows = _events(journal, "shape")
    if len(rows) != 1:
        failures.append(f"shape: {len(rows)} shape events, one is required")
        return
    for key, want in SHAPE.items():
        if rows[0].get(key) != want:
            failures.append(f"shape: {key} is {rows[0].get(key)!r}, DeepSeek-V3 has {want}")


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
    memory, capability = row.get("vram_bytes"), row.get("compute_capability")
    if not isinstance(row.get("gpu"), str) or not row["gpu"]:
        failures.append("card: the environment event does not name the card")
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
            if not _number(peak):
                failures.append(f"peak_missing: {label} has no numeric {key}")
            elif card and peak > card:
                failures.append(f"{code}: {label} {key} {peak:g} is above the card memory "
                                f"{card:g}")


def _check_store(record: Optional[Mapping[str, Any]], failures: List[str]) -> None:
    if not record:
        failures.append("store_record: no record of the store write was given")
        return
    planned, actual = record.get("planned_bytes"), record.get("actual_bytes")
    required, free = record.get("required_free_bytes"), record.get("free_before_bytes")
    if record.get("free_assumed") or record.get("failed"):
        failures.append("store_record: the record is of an assumed free space or of a failed "
                        "write, neither counts as a measured, finished store")
    if not all(_number(value) for value in (planned, actual, required, free)):
        failures.append("store_record: planned, actual, required free and free bytes are needed")
        return
    if abs(actual - planned) > SIZE_TOLERANCE * planned:
        failures.append(f"store_size: {actual:g} bytes on disk against {planned:g} planned, "
                        f"more than {SIZE_TOLERANCE:.0%} apart")
    if free < required:
        failures.append(f"disk_free: {free:g} bytes were free before the write, the plan "
                        f"required {required:g}")


def _check_cold(journal: Sequence[Mapping[str, Any]], failures: List[str]) -> None:
    drops = _events(journal, "cold_cache")
    if len(drops) != 1:
        failures.append(f"cold_cache: {len(drops)} cold_cache events, one is required")
        return
    drop = drops[0]
    if not isinstance(drop.get("method"), str) or not drop["method"]:
        failures.append("cold_cache: the event does not name the drop method")
    resident, total = drop.get("resident_after_bytes"), drop.get("store_bytes")
    if not _number(resident) or not _number(total) or total <= 0:
        failures.append("cold_cache: resident_after_bytes and store_bytes are needed")
    elif resident > RESIDENT_MAX_FRACTION * total:
        failures.append(f"cold_cache: {resident:g} of {total:g} store bytes were still in the "
                        f"page cache after the drop, the limit is {RESIDENT_MAX_FRACTION:.0%}")
    starts = _events(journal, "training_start")
    if not starts or not _number(drop.get("unix")) or not _number(starts[0].get("unix")):
        failures.append("cold_gap: the drop or the training start has no time stamp")
    else:
        gap = starts[0]["unix"] - drop["unix"]
        if not 0 <= gap <= MAX_DROP_TO_TRAINING_SECONDS:
            failures.append(f"cold_gap: training started {gap:g} s after the drop, the rule "
                            f"allows 0 to {MAX_DROP_TO_TRAINING_SECONDS:g}")
    first = next((row for row in _events(journal, "step") if row.get("step") == 0), None)
    if first is None:
        return
    read, logical = first.get("disk_read_bytes"), first.get("expert_bytes_read")
    # This check shows the storage layer was read in step 0, as the ticket asks. It does not tell
    # cold from warm: the experts bypass the page cache in every run. The residency check above
    # is what shows the cache was empty.
    if not _number(read) or not _number(logical) or logical <= 0:
        failures.append("cold_unconfirmed: step 0 has no disk_read_bytes or expert_bytes_read")
    elif read < STEP0_READ_MIN_FRACTION * logical:
        failures.append(f"cold_unconfirmed: step 0 read {read:g} bytes from the disk for "
                        f"{logical:g} logical expert bytes, the rule needs "
                        f"{STEP0_READ_MIN_FRACTION:.0%}")


def _check_warm(journal: Sequence[Mapping[str, Any]], failures: List[str]) -> None:
    """A warm repeat has no cache drop: a journal with one is a cold run with the wrong label."""
    if _events(journal, "cold_cache"):
        failures.append("warm: the journal has a cold_cache event, it is not a warm repeat")


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
    if len(speeds) < 2 or not all(_number(value) for value in speeds):
        return {"status": "not measured", "reason": "a step has no tokens_per_second"}
    return {"status": "measured", "first_step": speeds[0],
            "later_steps_median": statistics.median(speeds[1:])}


def _measured(journal: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    steps = _events(journal, "step")

    def peak(key: str) -> Optional[float]:
        values = [row[key] for row in steps if _number(row.get(key))]
        return max(values) if values else None

    return {
        "peak_allocated_bytes": peak("peak_allocated"),
        "peak_reserved_bytes": peak("peak_reserved"),
        "peak_rss_bytes": peak("peak_rss_bytes"),
        "expert_fetches_per_step": [row.get("expert_fetches") for row in steps],
        "expert_bytes_read_per_step": [row.get("expert_bytes_read") for row in steps],
        "disk_read_bytes_per_step": [row.get("disk_read_bytes") for row in steps],
        # Experts are read with O_DIRECT in every run. What the disk read beyond them is the
        # part a cold page cache changes, so it is shown beside the expert bytes.
        "non_expert_disk_read_bytes_per_step": [
            row["disk_read_bytes"] - row["expert_bytes_read"] for row in steps
            if _number(row.get("disk_read_bytes")) and _number(row.get("expert_bytes_read"))],
        "tokens_per_second": _throughput(journal),
    }


def verdict(journal: Sequence[Mapping[str, Any]], store_record: Optional[Mapping[str, Any]],
            *, warm: bool = False) -> Dict[str, Any]:
    """Pass or fail for one journal and one store record, with every reason for a fail.

    ``warm=True`` judges the optional warm repeat: every check of the cold run except the cache
    drop, which must be absent. Its numbers never replace those of the cold run.
    """
    judged = b5_judge.judge([journal])
    failures: List[str] = []
    if not judged["passed"]:
        failures += [f"judge: {reason}" for reason in judged["failures"]]
    _check_signature(journal, failures)
    _check_shape(journal, failures)
    count = len(_events(journal, "step"))
    if count != STEPS:
        failures.append(f"steps: {count} step events, the rule has exactly {STEPS}")
    _check_steps(journal, _check_environment(journal, failures), failures)
    _check_store(store_record, failures)
    if warm:
        _check_warm(journal, failures)
    else:
        _check_cold(journal, failures)
    if not _events(journal, "complete"):
        failures.append("incomplete: no complete event, the run did not reach the end")
    return {"passed": not failures, "failures": failures, "judge": judged,
            "mode": "warm" if warm else "cold", "measured": _measured(journal)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", required=True)
    parser.add_argument("--store-record", required=True)
    parser.add_argument("--warm", action="store_true", help="judge the warm repeat")
    args = parser.parse_args(argv)
    record = json.loads(Path(args.store_record).read_text(encoding="utf-8"))
    result = verdict(b5_judge.read_journal(Path(args.journal)), record, warm=args.warm)
    console.out(json.dumps(result, indent=1, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
