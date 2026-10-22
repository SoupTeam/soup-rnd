"""Fixed-budget and source-freeze contract for the D3 trained-step gate."""

from __future__ import annotations

import copy

import pytest
from d3_training_gate import main
from d3_training_protocol import read_rule, source_paths, validate_rule


def test_committed_protocol_pins_checkpoint_device_and_all_evidence() -> None:
    rule = read_rule()

    validate_rule(rule, require_execution=True)
    assert len(rule["checkpoint_files"]) == 4
    assert len(rule["bundle_sha256"]) == 3
    assert rule["expected_cell_count"] == 84
    assert rule["attempt_budget"]["total_cells"] == 84
    assert rule["schema_version"] == 4
    assert [attempt["status"] for attempt in rule["prior_attempts"]] == [
        "incomplete", "incomplete", "incomplete",
    ]
    assert rule["prior_attempts"][0]["completed_cells"] == 1
    assert rule["prior_attempts"][0]["no_verdict_cell"] == "correctness-dense-native"
    assert rule["prior_attempts"][1]["completed_cells"] == 0
    assert rule["prior_attempts"][1]["attempted_cell"] == "correctness-dense-native"
    assert rule["prior_attempts"][2]["completed_cells"] == 1
    assert rule["prior_attempts"][2]["no_verdict_cell"] == "correctness-dense-sequential"
    assert rule["gpu_uuid"].startswith("GPU-")
    assert "tests/test_d3_training_protocol.py" in source_paths()
    assert "tests/test_d3_full_model.py" in source_paths()
    assert any(path.endswith("attempt-01-no-verdict-a26f4948.zip")
               for path in source_paths())
    assert any(path.endswith("attempt-02-no-verdict-23bad807.zip")
               for path in source_paths())
    assert any(path.endswith("attempt-03-no-verdict-46937f49.zip")
               for path in source_paths())


@pytest.mark.parametrize(
    "change",
    [
        lambda rule: rule.update(replacements=1),
        lambda rule: rule.update(expected_cell_count=83),
        lambda rule: rule.update(tokens=1024),
        lambda rule: rule.update(formats=["dense"]),
        lambda rule: rule.update(scope="training quality benchmark"),
    ],
)
def test_protocol_rejects_unfrozen_recipe_or_budget(change) -> None:
    rule = copy.deepcopy(read_rule())
    change(rule)

    with pytest.raises(ValueError):
        validate_rule(rule)


def test_formal_run_requires_explicitly_enabled_protocol() -> None:
    rule = copy.deepcopy(read_rule())
    rule["execution_enabled"] = False

    with pytest.raises(ValueError, match="disabled"):
        validate_rule(rule, require_execution=True)


def test_child_cli_accepts_native_oracle_argument(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["child", "--help"])

    assert error.value.code == 0
    assert "--oracle ORACLE" in capsys.readouterr().out
