"""CPU checks for the prospective public SFT/L2L rule; no model or GPU execution."""

from __future__ import annotations

import copy
import json
from collections import Counter

import pytest

from benchmarks.harness.d3_trainer_protocol import (
    ENGINE_FILES,
    PREPARATION_FIELDS,
    TIMING_MODES,
    assert_launch_ready,
    cell_plan,
    contained,
    export_packet,
    file_hash,
    launch_blockers,
    main,
    pair_projection,
    read_rule,
    repeat_target,
    validate_rule,
)


def block(*, drift: float = 0.0, offset: float = 0.0) -> list[dict]:
    costs = {"sequential": 1.0, "grouped": 0.9, "triton": 0.85}
    return [
        {
            "mode": mode,
            "outcome": "ok",
            "seconds_per_input_token": costs[mode] + offset + drift * index,
        }
        for index, mode in enumerate(TIMING_MODES)
    ]


def test_live_rule_has_all_required_gates_but_empty_checkout_cannot_start(tmp_path):
    rule = read_rule()
    validate_rule(rule)
    assert set(rule["preparation"]) == set(PREPARATION_FIELDS)
    with pytest.raises(ValueError, match="formal execution refused"):
        assert_launch_ready(rule, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "change",
    [
        {"allow_allclose": True},
        {"allow_noise_floor_tolerance": True},
        {"allow_value_dependent_fallback": True},
        {"replacements": 1},
        {"adapter_dropout": 0.2},
        {"tokens": 1024},
        {"sessions_per_commit": 2},
        {"native_rows_from": 480},
        {"pilot_blocks": 2},
        {"max_blocks": 11},
        {"exactness": "loss_to_three_decimals"},
        {"physical_host_isolation_claim": True},
        {"disk_counter_source": "override"},
        {"moe_aux_loss_coeff": 0.01},
        {"scope_is_full_task_acceptance": True},
        {"execution_enabled": 1},
    ],
)
def test_rule_refuses_relaxed_numerics_silent_recipe_changes_or_unlimited_attempts(change):
    rule = copy.deepcopy(read_rule())
    rule.update(change)
    with pytest.raises(ValueError):
        validate_rule(rule)


def test_rule_refuses_a_replaced_checkpoint_or_a_missing_backward_control():
    rule = copy.deepcopy(read_rule())
    rule["checkpoint_files"]["config.json"] = "a" * 64
    with pytest.raises(ValueError, match="checkpoint"):
        validate_rule(rule)
    rule = copy.deepcopy(read_rule())
    del rule["numerical_gates"]["backward"]
    with pytest.raises(ValueError, match="numerical gates"):
        validate_rule(rule)


def test_pilot_and_capped_plans_are_complete_unique_and_use_fresh_arm_names():
    rule = read_rule()
    plan = cell_plan(rule)
    assert Counter(row["kind"] for row in plan) == {
        "correctness": 36,
        "timing": 180,
        "profile": 18,
    }
    assert len({row["name"] for row in plan}) == len(plan) == 234
    assert len(cell_plan(rule, blocks=10)) == 414
    for codec in rule["formats"]:
        for training in rule["training_modes"]:
            for index in range(1, 6):
                rows = [
                    row
                    for row in plan
                    if row["kind"] == "timing"
                    and row["codec"] == codec
                    and row["training"] == training
                    and row["block"] == index
                ]
                assert tuple(row["mode"] for row in rows) == TIMING_MODES
            correctness = [
                row["mode"]
                for row in plan
                if row["kind"] == "correctness"
                and row["codec"] == codec
                and row["training"] == training
            ]
            assert correctness == rule["correctness_modes"]


@pytest.mark.parametrize("count", [True, 4, 11, 5.0])
def test_planner_rejects_out_of_budget_or_noninteger_block_counts(count):
    with pytest.raises(ValueError, match="budget"):
        cell_plan(read_rule(), blocks=count)


@pytest.mark.parametrize(
    "contrast,left,right",
    [
        ("grouping", "sequential", "grouped"),
        ("activation", "grouped", "triton"),
    ],
)
def test_symmetric_projection_really_cancels_linear_drift(contrast, left, right):
    original = pair_projection(block(), contrast)
    drifting = pair_projection(block(drift=0.03, offset=0.2), contrast)
    assert [row["mode"] for row in original] == [left, right, right, left]

    def contrast_cost(rows):
        return (
            rows[1]["seconds_per_input_token"]
            + rows[2]["seconds_per_input_token"]
            - rows[0]["seconds_per_input_token"]
            - rows[3]["seconds_per_input_token"]
        ) / 2

    assert contrast_cost(drifting) == pytest.approx(contrast_cost(original), abs=1e-15)


@pytest.mark.parametrize("outcome", ["void", "unknown", "fail", None])
def test_invalid_runs_are_not_silently_dropped_or_replaced(outcome):
    rows = block()
    rows[2]["outcome"] = outcome
    with pytest.raises(ValueError, match="stop"):
        pair_projection(rows, "grouping")
    with pytest.raises(ValueError, match="stop"):
        repeat_target([block(), rows, block(), block(), block()])


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_nonpositive_or_nonfinite_timing_has_no_verdict(value):
    rows = block()
    rows[0]["seconds_per_input_token"] = value
    with pytest.raises(ValueError):
        pair_projection(rows, "activation")


def test_partial_or_shuffled_blocks_are_not_accepted_as_abba():
    with pytest.raises(ValueError, match="complete ordered"):
        pair_projection(block()[:-1], "grouping")
    rows = block()
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError, match="complete ordered"):
        pair_projection(rows, "activation")


def test_low_noise_stops_after_the_prospective_pilot():
    assert repeat_target([block() for _ in range(5)]) == {"status": "complete", "target": 5}


def test_high_noise_is_reported_without_exceeding_the_fixed_budget():
    rows = [block(offset=index * 10.0) for index in range(5)]
    result = repeat_target(rows)
    assert result["target"] > 10
    assert result["status"] == "no_verdict_precision_budget"
    with pytest.raises(ValueError, match="budget"):
        repeat_target([block() for _ in range(11)])


def test_joint_top_up_uses_the_fixed_first_prefix_once(monkeypatch):
    from benchmarks.harness import arm_validity

    calls = []

    def estimate(rows):
        from types import SimpleNamespace

        calls.append(len(rows))
        return SimpleNamespace(blocks_needed=7 if len(rows) == 5 else 9)

    monkeypatch.setattr(arm_validity, "repeats", estimate)
    assert repeat_target([block() for _ in range(5)])["target"] == 7
    assert repeat_target([block() for _ in range(7)]) == {
        "status": "additional_blocks_required",
        "target": 9,
    }
    calls.clear()
    assert repeat_target([block() for _ in range(9)]) == {"status": "complete", "target": 9}
    assert calls == [5, 5, 7, 7]


@pytest.mark.parametrize(
    "name", ["../escape", "C:/escape", "a/../escape", "a\\escape", "/escape", "a//b", "a/./b"]
)
def test_receipt_paths_are_contained_and_canonical(tmp_path, name):
    with pytest.raises(ValueError):
        contained(tmp_path, name)


def test_synthetic_fixture_report_cannot_stand_in_for_real_shard_controls(tmp_path):
    rule = copy.deepcopy(read_rule())
    rule["execution_enabled"] = True
    rule["preparation"] = dict.fromkeys(PREPARATION_FIELDS)
    folder = tmp_path / "benchmarks/protocols/d3-public-trainer"
    folder.mkdir(parents=True)
    receipt = folder / "actual-controls.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "controls_passed",
                "controls": [
                    {"case": "baseline", "expected": "ok", "observed": "ok"},
                    {"case": "own", "expected": "ok", "observed": "void"},
                    {"case": "foreign", "expected": "void", "observed": "void"},
                ],
            }
        )
    )
    rule["preparation"]["actual_shard_controls"] = {
        "path": "benchmarks/protocols/d3-public-trainer/actual-controls.json",
        "sha256": file_hash(receipt),
    }
    blockers = launch_blockers(rule, tmp_path, check_git=False)
    assert "actual-shard F1 controls have not all passed" in blockers
    assert any("unbound" in value for value in blockers)
    assert all(any(name in value for value in blockers) for name in ENGINE_FILES)


def test_changing_a_bound_receipt_is_a_blocker(tmp_path):
    rule = copy.deepcopy(read_rule())
    rule["execution_enabled"] = True
    folder = tmp_path / "benchmarks/protocols/d3-public-trainer"
    folder.mkdir(parents=True)
    receipt = folder / "software.json"
    receipt.write_text('{"old": true}')
    rule["preparation"]["software"] = {
        "path": "benchmarks/protocols/d3-public-trainer/software.json",
        "sha256": file_hash(receipt),
    }
    receipt.write_text('{"new": true}')
    assert "preparation evidence missing/changed: software" in launch_blockers(
        rule,
        tmp_path,
        check_git=False,
    )


def test_export_refuses_the_draft_without_creating_an_archive(tmp_path):
    folder = tmp_path / "benchmarks/protocols/d3-public-trainer"
    folder.mkdir(parents=True)
    rule = copy.deepcopy(read_rule())
    rule["execution_enabled"] = False
    (folder / "protocol-draft.json").write_text(json.dumps(rule))
    archive = tmp_path / "should-not-exist.zip"
    with pytest.raises(ValueError, match="formal execution refused"):
        export_packet(archive, tmp_path)
    assert not archive.exists()


def test_cli_check_is_read_only_and_reports_the_pending_merge_and_engine(tmp_path):
    output = tmp_path / "readiness.json"
    assert main(["check", "--out", str(output)]) == 1
    result = json.loads(output.read_text())
    assert result["formal_execution_started"] is False
    assert result["pilot_plan_cells"] == 234
    assert result["max_plan_cells"] == 414
    assert any("clean committed checkout" in row for row in result["launch_blockers"])


def test_final_launcher_receipt_requires_actual_success_and_current_runtime_bytes(tmp_path):
    import hashlib

    root = tmp_path
    rule = copy.deepcopy(read_rule())
    rule["execution_enabled"] = True
    folder = root / "benchmarks/protocols/d3-public-trainer"
    folder.mkdir(parents=True)
    for name in ENGINE_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic guard fixture\n")
    model = root / "src/synthetic.py"
    model.parent.mkdir()
    model.write_text("# synthetic runtime\n")
    source = {
        name: {
            "lf_sha256": hashlib.sha256(
                (root / name).read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
        }
        for name in [*ENGINE_FILES, "src/synthetic.py"]
    }
    for name in rule["preparation"]:
        path = folder / (name + ".json")
        value = {}
        if name == "actual_shard_controls":
            value = dict(
                status="controls_passed",
                controls=[
                    dict(case=case, expected=expected, observed=expected)
                    for case, expected in [("baseline", "ok"), ("own", "ok"), ("foreign", "void")]
                ],
            )
        if name == "launcher_self_test":
            value = dict(
                status="synthetic_launcher_checks_passed",
                numerical_count=36,
                profile_count=18,
                formal_execution=False,
                pretrained_model_executed=False,
                source_files=source,
            )
        path.write_text(json.dumps(value))
        rule["preparation"][name] = dict(
            path=path.relative_to(root).as_posix(), sha256=file_hash(path)
        )
    assert launch_blockers(rule, root, check_git=False) == []
    model.write_text("# altered runtime\n")
    assert "launcher self-test runtime source differs: src/synthetic.py" in launch_blockers(
        rule, root, check_git=False
    )
    path = folder / "launcher_self_test.json"
    value = json.loads(path.read_text())
    value["status"] = "failed"
    path.write_text(json.dumps(value))
    rule["preparation"]["launcher_self_test"]["sha256"] = file_hash(path)
    assert "new launcher synthetic controls/profiles have not all passed" in launch_blockers(
        rule, root, check_git=False
    )
