"""Prospective capacity classifications reject unrelated failures and missing proof."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks/harness"))
import f2_capacity_reference as capacity  # noqa: E402


def refused():
    return {"returncode": 1, "validity": {"outcome": "ok"},
            "audit": {"physical_vram_bytes": 8 * 1024**3, "steps": []}}


MESSAGE = "base exceeds the physical RAM safety ceiling"


def test_real_capacity_refusal_is_separate_from_a_working_run():
    assert capacity.capacity_verdict(refused(), MESSAGE)[0] == "EXPECTED_CAPACITY_REFUSAL"


@pytest.mark.parametrize("change", [
    {"returncode": 0}, {"returncode": None}, {"timeout": True},
    {"validity": {"outcome": "unknown"}},
    {"audit": {"physical_vram_bytes": 24 * 1024**3, "steps": []}},
    {"audit": {"physical_vram_bytes": 8 * 1024**3, "steps": [{"grad_norm": 1}]}},
])
def test_incomplete_or_contradictory_refusals_never_count(change):
    row = refused()
    row.update(change)
    assert capacity.capacity_verdict(row, MESSAGE)[0] != "EXPECTED_CAPACITY_REFUSAL"


@pytest.mark.parametrize("message", ["CUDA out of memory", "ModuleNotFoundError",
    "layer streaming needs NVMe or more RAM", "", "permission denied"])
def test_generic_errors_and_hdd_policy_are_not_capacity_evidence(message):
    assert capacity.capacity_verdict(refused(), message)[0] == "FAIL"


def test_schedule_keeps_five_complete_blocks_and_at_least_five_refusals():
    rows = []
    for block in range(5):
        rows.extend(capacity.block_schedule(block, len(rows)))
    assert [r["index"] for r in rows] == list(range(1, 151))
    for group, expected in (("ram", 10), ("overflow", 10), ("boundary", 5)):
        for length in capacity.f2.LENGTHS:
            for kind in ("train", "bench"):
                assert sum(r["group"] == group and r["length"] == length
                           and r["kind"] == kind for r in rows) == expected


def test_supplement_only_responds_to_missing_acceptance_counts():
    rows = []
    for block in range(5):
        rows.extend(capacity.block_schedule(block, len(rows)))
    for row in rows:
        row["status"] = "EXPECTED_CAPACITY_REFUSAL" if row["group"] == "boundary" else "PASS"
    assert not capacity.needs_supplement(rows)
    for row in rows:
        if row["group"] == "overflow" and row["length"] == 128 and row["kind"] == "bench":
            row["status"] = "NO_VERDICT"
    assert capacity.needs_supplement(rows)
