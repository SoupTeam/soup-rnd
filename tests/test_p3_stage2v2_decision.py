"""Informativeness vetoes preserve paired statistics and coverage precedence."""

from __future__ import annotations

from copy import deepcopy

import pytest

from benchmarks.harness.p3_stage2_statistics import summarize_pairs
from benchmarks.harness.p3_stage2v2_decision import apply_informativeness_gate

CHANCE = {"easy": 0.25, "hard": 0.25}


def _rows(
    benchmark: str,
    n: int,
    both_pass: int,
    *,
    seeds: tuple[int, ...] = (23,),
    lengths: tuple[int, ...] = (1024,),
) -> list[dict]:
    return [
        {
            "benchmark": benchmark,
            "length": length,
            "seed": seed,
            "task_id": f"{benchmark}-{index}",
            "baseline_pass": index < both_pass,
            "candidate_pass": index < both_pass,
        }
        for length in lengths
        for seed in seeds
        for index in range(n)
    ]


def _summary(spec: list[tuple[str, int, int]]) -> dict:
    rows, tasks = [], {}
    for benchmark, n, both_pass in spec:
        rows.extend(_rows(benchmark, n, both_pass))
        tasks[benchmark] = [f"{benchmark}-{index}" for index in range(n)]
    return summarize_pairs(rows, expected_tasks=tasks, seeds=[23], lengths=[1024])


def test_all_informative_noninferior_cells_keep_noninferior() -> None:
    summary = _summary([("easy", 4000, 2400), ("hard", 4000, 2000)])
    assert summary["decision"] == "NONINFERIOR"

    gated = apply_informativeness_gate(summary, chance=CHANCE)

    assert gated["statistical_decision"] == "NONINFERIOR"
    assert gated["decision"] == "NONINFERIOR"
    assert gated["floor_cells"] == []
    assert all(cell["informative"] is True for cell in gated["cells"])


def test_one_floor_cell_vetoes_noninferiority_without_removing_statistics() -> None:
    summary = _summary([("easy", 4000, 2400), ("hard", 4000, 400)])
    assert summary["decision"] == "NONINFERIOR"

    gated = apply_informativeness_gate(summary, chance=CHANCE)

    assert gated["statistical_decision"] == "NONINFERIOR"
    assert gated["decision"] == "INCONCLUSIVE_UNINFORMATIVE_FLOOR"
    assert gated["floor_cells"] == [
        {
            "benchmark": "hard",
            "length": 1024,
            "seed": 23,
            "baseline_accuracy": 0.1,
            "chance": 0.25,
        }
    ]
    assert len(gated["cells"]) == len(summary["cells"])
    for original, cell in zip(summary["cells"], gated["cells"]):
        assert {key: cell[key] for key in original} == original
        assert cell["informative"] is (cell["benchmark"] == "easy")
        assert cell["conditional_noninferiority"] is True
    for key, value in summary.items():
        if key not in {"decision", "cells"}:
            assert gated[key] == value


def test_all_floor_cells_never_yield_noninferior() -> None:
    summary = _summary([("easy", 4000, 0), ("hard", 4000, 0)])
    assert summary["decision"] == "NONINFERIOR"

    gated = apply_informativeness_gate(summary, chance=CHANCE)

    assert gated["statistical_decision"] == "NONINFERIOR"
    assert gated["decision"] == "INCONCLUSIVE_UNINFORMATIVE_FLOOR"
    assert [cell["benchmark"] for cell in gated["floor_cells"]] == ["easy", "hard"]
    assert all(cell["informative"] is False for cell in gated["cells"])


def test_excessive_degradation_survives_floor_cells() -> None:
    rows = _rows("hard", 4000, 0)
    rows.extend(
        {
            "benchmark": "easy",
            "length": 1024,
            "seed": 23,
            "task_id": f"easy-{index}",
            "baseline_pass": index < 2400,
            "candidate_pass": index < 2000,
        }
        for index in range(4000)
    )
    tasks = {name: [f"{name}-{index}" for index in range(4000)] for name in CHANCE}
    summary = summarize_pairs(rows, expected_tasks=tasks, seeds=[23], lengths=[1024])
    assert summary["decision"] == "EXCESSIVE_DEGRADATION"

    gated = apply_informativeness_gate(summary, chance=CHANCE)

    assert gated["statistical_decision"] == "EXCESSIVE_DEGRADATION"
    assert gated["decision"] == "EXCESSIVE_DEGRADATION"
    assert [cell["benchmark"] for cell in gated["floor_cells"]] == ["hard"]
    assert gated["cells"][0]["familywise_excessive_degradation"] is True


@pytest.mark.parametrize(
    ("both_pass", "expected_decision", "informative"),
    [(60, "INCONCLUSIVE", True), (10, "INCONCLUSIVE_UNINFORMATIVE_FLOOR", False)],
)
def test_inconclusive_statistics_only_change_for_a_floor(
    both_pass: int, expected_decision: str, informative: bool
) -> None:
    summary = _summary([("easy", 100, both_pass)])
    assert summary["decision"] == "INCONCLUSIVE"

    gated = apply_informativeness_gate(summary, chance=CHANCE)

    assert gated["statistical_decision"] == "INCONCLUSIVE"
    assert gated["decision"] == expected_decision
    assert gated["cells"][0]["informative"] is informative


@pytest.mark.parametrize(
    ("both_pass", "informative"), [(1199, False), (1200, True), (1201, True)]
)
def test_floor_comparison_is_strict_at_chance_plus_offset(
    both_pass: int, informative: bool
) -> None:
    summary = _summary([("easy", 4000, both_pass)])
    assert summary["decision"] == "NONINFERIOR"

    gated = apply_informativeness_gate(summary, chance=CHANCE)

    assert gated["cells"][0]["informative"] is informative
    assert gated["decision"] == (
        "NONINFERIOR" if informative else "INCONCLUSIVE_UNINFORMATIVE_FLOOR"
    )
    assert len(gated["floor_cells"]) == (0 if informative else 1)


@pytest.mark.parametrize("both_pass", [0, 200])
def test_zero_chance_is_a_valid_entry(both_pass: int) -> None:
    summary = _summary([("lambada", 4000, both_pass)])

    gated = apply_informativeness_gate(summary, chance={"lambada": 0.0})

    assert gated["cells"][0]["informative"] is (both_pass == 200)
    assert gated["decision"] == (
        "NONINFERIOR" if both_pass == 200 else "INCONCLUSIVE_UNINFORMATIVE_FLOOR"
    )


def test_floor_offset_argument_controls_the_threshold() -> None:
    summary = _summary([("easy", 4000, 1600)])

    default = apply_informativeness_gate(summary, chance=CHANCE)
    gated = apply_informativeness_gate(summary, chance=CHANCE, floor_offset=0.2)

    assert default["decision"] == "NONINFERIOR"
    assert gated["decision"] == "INCONCLUSIVE_UNINFORMATIVE_FLOOR"
    assert gated["floor_cells"][0]["chance"] == 0.25


def test_empty_primary_cells_are_refused() -> None:
    summary = _summary([("easy", 4000, 2400)])
    summary["cells"] = []

    with pytest.raises(ValueError, match="empty|at least one"):
        apply_informativeness_gate(summary, chance=CHANCE)


@pytest.mark.parametrize("chance", [{}, {"easy": 0.25}, {"other": 0.25}])
def test_missing_chance_for_any_required_benchmark_is_refused(chance: dict[str, float]) -> None:
    summary = _summary([("easy", 4000, 2400), ("hard", 4000, 400)])
    before = deepcopy(summary)

    with pytest.raises(ValueError, match="chance"):
        apply_informativeness_gate(summary, chance=chance)

    assert summary == before


@pytest.mark.parametrize("missing", ["task", "seed", "length", "benchmark"])
def test_missing_paired_coverage_raises_before_any_floor_verdict(missing: str) -> None:
    tasks = {name: [f"{name}-0", f"{name}-1"] for name in CHANCE}
    rows = [
        row
        for name in CHANCE
        for row in _rows(name, 2, 0, seeds=(23, 41), lengths=(1024, 4096))
    ]
    if missing == "task":
        rows.pop()
    elif missing == "seed":
        rows = [row for row in rows if row["seed"] != 41]
    elif missing == "length":
        rows = [row for row in rows if row["length"] != 4096]
    else:
        rows = [row for row in rows if row["benchmark"] != "hard"]

    with pytest.raises(ValueError, match="coverage"):
        summary = summarize_pairs(rows, expected_tasks=tasks, seeds=[23, 41], lengths=[1024, 4096])
        apply_informativeness_gate(summary, chance=CHANCE)


@pytest.mark.parametrize("blocked_marker", ["decision", "coverage"])
def test_blocked_coverage_is_not_reclassified_as_a_floor(blocked_marker: str) -> None:
    summary = _summary([("easy", 4000, 0)])
    if blocked_marker == "decision":
        summary["decision"] = "BLOCKED_INCOMPLETE_COVERAGE"
    else:
        summary["coverage"]["complete"] = False
    before = deepcopy(summary)

    gated = apply_informativeness_gate(summary, chance=CHANCE)

    assert gated["decision"] == "BLOCKED_INCOMPLETE_COVERAGE"
    assert gated["statistical_decision"] == summary["decision"]
    assert len(gated["floor_cells"]) == 1
    assert gated["cells"][0]["informative"] is False
    assert summary == before


def test_floor_cells_keep_each_required_length_and_seed() -> None:
    rows = _rows("easy", 1000, 600, seeds=(23, 41), lengths=(1024, 4096))
    for row in rows:
        if row["length"] == 4096 and row["seed"] == 41:
            passed = int(row["task_id"].rsplit("-", 1)[1]) < 100
            row["baseline_pass"] = row["candidate_pass"] = passed
    summary = summarize_pairs(
        rows,
        expected_tasks={"easy": [f"easy-{index}" for index in range(1000)]},
        seeds=[23, 41],
        lengths=[1024, 4096],
    )
    assert summary["decision"] == "NONINFERIOR"

    gated = apply_informativeness_gate(summary, chance=CHANCE)

    assert gated["decision"] == "INCONCLUSIVE_UNINFORMATIVE_FLOOR"
    assert gated["floor_cells"] == [
        {
            "benchmark": "easy",
            "length": 4096,
            "seed": 41,
            "baseline_accuracy": 0.1,
            "chance": 0.25,
        }
    ]
    assert len(gated["cells"]) == 4
    assert [cell["informative"] for cell in gated["cells"]].count(False) == 1


def test_gate_does_not_mutate_or_share_nested_statistics_with_input() -> None:
    summary = _summary([("easy", 4000, 2400), ("hard", 4000, 400)])
    before = deepcopy(summary)
    chance = dict(CHANCE)

    gated = apply_informativeness_gate(summary, chance=chance)

    assert summary == before
    assert chance == CHANCE
    assert gated is not summary
    gated["cells"][0]["pointwise_bounds"]["upper"] = 999
    gated["cells"][1]["simultaneous_bounds"]["lower"] = -999
    gated["coverage"]["seeds"].append(99)
    gated["endpoints"][0]["seed_results"][0]["baseline_accuracy"] = -1
    assert summary == before
