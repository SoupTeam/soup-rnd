"""An incomplete grid or an absent split timer cannot become full measurements."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks/harness"))


def test_empty_full_plan_is_explicitly_incomplete(tmp_path):
    from d1_full_report import build_index
    from d1_full_study import cases

    (tmp_path / "plan.json").write_text(json.dumps({"cases": cases(), "source_sha": "same"}))
    result = build_index(tmp_path)
    assert result["status"] == "INCOMPLETE"
    assert result["completed_cases"] == result["attempted_arms"] == 0
    assert result["expected_projection_cases"] == 96
    assert result["independent_acceptance"] is False


def test_reduced_plan_cannot_masquerade_as_complete(tmp_path):
    from d1_full_report import build_index
    from d1_full_study import cases

    (tmp_path / "plan.json").write_text(json.dumps({"cases": cases()[:-1], "source_sha": "same"}))
    with pytest.raises(ValueError, match="complete"):
        build_index(tmp_path)


def test_split_timing_contract_rejects_missing_events_and_changed_warmup():
    from d1_full_report import projection_contract
    from d1_full_study import cases
    from d1_rules import settings_digest

    case = cases()[2]
    settings = {key: case[key] for key in ("dtype", "m", "n", "k", "steps", "warmup")}
    settings.update(no_nested=False, warmup_seconds=15.0)
    row = {
        "settings_sha256": settings_digest(settings),
        "shape": [1, 4096, 4096],
        "actual_warmup_steps": 25,
        "actual_warmup_seconds": 15.01,
        **{
            field: [1.0] * 200
            for field in ("forward_cuda_ms", "backward_cuda_ms", "step_cuda_ms", "step_wall_ms")
        },
    }
    projection_contract(case, row)
    row["forward_cuda_ms"] = [1.0] * 199
    with pytest.raises(ValueError, match="array"):
        projection_contract(case, row)
    row["forward_cuda_ms"] = [1.0] * 200
    row["actual_warmup_seconds"] = 14.9
    with pytest.raises(ValueError, match="warm-up"):
        projection_contract(case, row)
