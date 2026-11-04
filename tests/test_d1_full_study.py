"""Complete grid and resume bookkeeping must not omit or overwrite comparisons."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks/harness"))


def test_complete_plan_has_96_distinct_projection_cases_and_unchanged_training():
    from d1_full_study import cases

    plan = cases()
    assert len(plan) == 98
    assert len({row["name"] for row in plan}) == 98
    projections = [row for row in plan if row["kind"] == "projection"]
    assert len(projections) == 96
    assert {row["m"] for row in projections} == {1, 8, 32, 128, 512, 2048}
    assert {row["dtype"] for row in projections} == {"float16", "bfloat16"}
    assert all(row["warmup_seconds"] == 5 and row["steps"] == 200 for row in projections)
    qa = plan[-1]
    assert qa["kind"] == "train" and qa["steps"] + qa["warmup"] == 276
    assert qa["preheat_steps"] == 200
    assert plan[-2]["kind"] == "ablation" and plan[-2]["warmup"] == 200
    assert all(row["kind"] == "projection" for row in plan[:-2])
    assert "preheat_steps" not in plan[0]
    assert "warmup_seconds" not in qa


def test_resume_skips_even_interrupted_block_ids(tmp_path):
    from d1_full_study import next_block

    assert next_block(tmp_path) == 0
    (tmp_path / "attempt-000/block-000-run-0-A").mkdir(parents=True)
    (tmp_path / "attempt-005/block-005-run-0-A").mkdir(parents=True)
    assert next_block(tmp_path) == 6


def test_empty_case_needs_the_five_counted_pilot_blocks():
    from d1_full_study import remaining_blocks

    assert remaining_blocks([]) == 5


def test_selected_followup_keeps_fixed_case_settings_and_is_not_complete_grid():
    from d1_full_study import cases, select_cases

    assert select_cases() == cases()
    selected = select_cases(["ablation"])
    assert selected == [cases()[-2]]
    assert len(selected) != 98


@pytest.mark.parametrize("names", [[], ["missing"], ["qa", "qa"]])
def test_followup_rejects_unknown_empty_or_duplicate_selection(names):
    from d1_full_study import select_cases

    with pytest.raises(ValueError, match="unique known"):
        select_cases(names)
