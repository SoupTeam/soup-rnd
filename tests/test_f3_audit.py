"""Tests for the F3 verdict driver (benchmarks/harness/f3_audit.py, F3 ticket 03).

Seam 1 of the spec: the driver with a fake runner and a fake arm watch. The runner
returns reports shaped like ``build_train_report``. No test starts ``soup bench
train``, touches a GPU or the network, or writes into benchmarks/results/.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"


@pytest.fixture(scope="module")
def audit():
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    name = "_f3_audit_test_module"
    spec = importlib.util.spec_from_file_location(name, HARNESS / "f3_audit.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def failure(check):
    return {"check": check, "message": f"{check} refused"}


def report(*failed, path="report.json", grad_norm="all reported steps finite and non-zero"):
    """A report shaped like build_train_report: valid unless a check is named in ``failed``."""
    return {
        "valid": not failed,
        "failures": [failure(check) for check in failed],
        "checks": {"trainable_parameters": 10, "grad_norm": grad_norm, "parameters_changed": True},
        "timing": {"median_seconds": 0.5},
        "report_path": path,
    }


GOOD = report()
BAD = report("grad_norm", "parameters_changed")


class FakeRunner:
    """Answers each (arm, process) from a table; an Exception value is raised."""

    def __init__(self, table):
        self.table = table
        self.calls = []

    def __call__(self, spec, process):
        self.calls.append((spec.arm, process))
        answer = self.table[spec.arm]
        if isinstance(answer, (list, tuple)):
            answer = answer[process - 1]
        if isinstance(answer, Exception):
            raise answer
        return answer


class FakeWatch:
    """Stands in for ArmWatch: a context manager whose record has a chosen outcome."""

    def __init__(self, outcomes=None):
        self.outcomes = outcomes or {}
        self.made = []

    def __call__(self, *, arm, round, run, **kwargs):
        self.made.append({"arm": arm, "round": round, "run": run, **kwargs})
        outcome = self.outcomes.get((arm, run), "ok")

        class Watched:
            record = None

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.record = {"arm": arm, "run": run, "outcome": outcome}

        return Watched()


def controls(audit, *, native=BAD, unsloth=BAD, weight_decay=0.0):
    return [
        audit.ControlRun("native", native, weight_decay),
        audit.ControlRun("unsloth", unsloth, weight_decay),
    ]


def spec(audit, arm="A", config="C1", **kwargs):
    return audit.ArmSpec(config=config, arm=arm, commit="abc123", **kwargs)


def run(audit, specs, table, *, control=None, watch=None):
    runner = FakeRunner(table)
    watch = watch or FakeWatch()
    result = audit.audit(
        specs, runner, watch, control if control is not None else controls(audit)
    )
    return result, runner, watch


def verdict_of(result, arm="A"):
    return next(v for v in result.verdicts if v.arm == arm)


# --- validity verdicts ---------------------------------------------------------


def test_pass_when_both_processes_are_valid_and_identical(audit):
    result, runner, _ = run(audit, [spec(audit)], {"A": GOOD})
    verdict = verdict_of(result)
    assert verdict.verdict == "pass"
    assert runner.calls == [("A", 1), ("A", 2)]
    assert result.refusals == []


def test_fail_when_both_processes_refuse_with_the_same_checks(audit):
    result, _, _ = run(audit, [spec(audit)], {"A": BAD})
    verdict = verdict_of(result)
    assert verdict.verdict == "fail"
    assert verdict.step == 2


def test_fail_when_only_one_check_refuses_in_both_processes(audit):
    result, _, _ = run(audit, [spec(audit)], {"A": report("grad_norm")})
    assert verdict_of(result).verdict == "fail"


def test_watch_is_made_per_process_with_no_required_checks(audit):
    _, _, watch = run(audit, [spec(audit)], {"A": GOOD})
    assert [(w["arm"], w["run"]) for w in watch.made] == [("A", 1), ("A", 2)]
    assert all(list(w["required"]) == [] for w in watch.made)


# --- the four reasons for no verdict -------------------------------------------


def test_no_verdict_when_the_arm_does_not_start(audit):
    result, _, _ = run(audit, [spec(audit)], {"A": RuntimeError("CUDA out of memory")})
    verdict = verdict_of(result)
    assert verdict.verdict == "no verdict"
    assert "did not start" in verdict.reason
    assert "CUDA out of memory" in verdict.reason


def test_no_verdict_when_the_second_process_does_not_start(audit):
    result, _, _ = run(audit, [spec(audit)], {"A": [BAD, RuntimeError("boom")]})
    assert verdict_of(result).verdict == "no verdict"


@pytest.mark.parametrize("outcome", ["void", "unknown"])
def test_no_verdict_when_a_validity_row_is_void_or_unknown(audit, outcome):
    watch = FakeWatch({("A", 2): outcome})
    result, _, _ = run(audit, [spec(audit)], {"A": BAD}, watch=watch)
    verdict = verdict_of(result)
    assert verdict.verdict == "no verdict"
    assert outcome in verdict.reason


def test_no_verdict_when_the_two_processes_disagree(audit):
    result, _, _ = run(audit, [spec(audit)], {"A": [GOOD, BAD]})
    verdict = verdict_of(result)
    assert verdict.verdict == "no verdict"
    assert "differ" in verdict.reason


def test_no_verdict_when_the_failing_checks_differ_between_processes(audit):
    table = {"A": [report("grad_norm"), report("parameters_changed")]}
    result, _, _ = run(audit, [spec(audit)], table)
    assert verdict_of(result).verdict == "no verdict"


def test_no_verdict_when_the_arm_did_not_reach_step_count(audit):
    result, _, _ = run(audit, [spec(audit)], {"A": report("step_count")})
    verdict = verdict_of(result)
    assert verdict.verdict == "no verdict"
    assert "step_count" in verdict.reason


def test_no_verdict_when_the_control_is_not_rejected(audit):
    control = controls(audit, unsloth=GOOD)
    result, runner, _ = run(audit, [spec(audit)], {"A": BAD}, control=control)
    verdict = verdict_of(result)
    assert verdict.verdict == "no verdict"
    assert "control" in verdict.reason and "unsloth" in verdict.reason
    assert runner.calls == []


# --- the control ---------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_control",
    [
        pytest.param(lambda a: controls(a, native=report("grad_norm")), id="one-check-only"),
        pytest.param(lambda a: controls(a, native=report("parameters_changed")), id="other-check"),
        pytest.param(lambda a: controls(a, native=GOOD), id="valid-true"),
        pytest.param(lambda a: controls(a, weight_decay=0.01), id="weight-decay"),
        pytest.param(lambda a: controls(a)[:1], id="unsloth-path-missing"),
        pytest.param(lambda a: [], id="no-control"),
    ],
)
def test_a_failed_control_turns_every_verdict_into_no_verdict(audit, bad_control):
    specs = [spec(audit, "A"), spec(audit, "B")]
    result, runner, _ = run(
        audit, specs, {"A": BAD, "B": GOOD}, control=bad_control(audit)
    )
    assert [v.verdict for v in result.verdicts] == ["no verdict", "no verdict"]
    assert all("control" in v.reason for v in result.verdicts)
    assert runner.calls == []


def test_a_rejected_control_leaves_verdicts_alone(audit):
    specs = [spec(audit, "A"), spec(audit, "B")]
    result, _, _ = run(audit, specs, {"A": GOOD, "B": BAD})
    assert [v.verdict for v in result.verdicts] == ["pass", "fail"]


# --- inference steps -----------------------------------------------------------


def test_step_3_needs_the_authors_log(audit):
    evidence = audit.Evidence(author_log=True)
    result, _, _ = run(audit, [spec(audit, evidence=evidence)], {"A": BAD})
    assert verdict_of(result).step == 3


def test_step_3_needs_an_exact_config_with_the_cause_confirmed(audit):
    evidence = audit.Evidence(exact_config_cause_confirmed=True)
    result, _, _ = run(audit, [spec(audit, evidence=evidence)], {"A": BAD})
    assert verdict_of(result).step == 3


def test_our_run_alone_never_reaches_step_3(audit):
    result, _, _ = run(audit, [spec(audit)], {"A": BAD})
    assert verdict_of(result).step == 2


def test_pass_and_no_verdict_stay_at_step_1(audit):
    result, _, _ = run(
        audit, [spec(audit, "A"), spec(audit, "B")], {"A": GOOD, "B": RuntimeError("x")}
    )
    assert [v.step for v in result.verdicts] == [1, 1]


def test_the_46000_run_is_not_audited_and_stays_at_step_1(audit):
    unfound = spec(audit, "S", config="46000", published_number=46000, config_found=False)
    result, runner, _ = run(audit, [unfound], {})
    verdict = verdict_of(result, "S")
    assert verdict.verdict == "no verdict"
    assert verdict.step == 1
    assert "config not found" in verdict.reason
    assert runner.calls == []


def test_step_3_is_rejected_for_the_46000_run(audit):
    with pytest.raises(ValueError, match="step 3"):
        spec(
            audit,
            "S",
            config="46000",
            config_found=False,
            evidence=audit.Evidence(author_log=True),
        )


# --- out of scope --------------------------------------------------------------


def test_an_arm_without_trainer_is_out_of_scope_with_its_reason(audit):
    custom = spec(audit, "B", uses_trainer=False, scope_reason="custom training loop")
    result, runner, _ = run(audit, [custom], {})
    verdict = verdict_of(result, "B")
    assert verdict.verdict == "out of scope"
    assert verdict.reason == "custom training loop"
    assert runner.calls == []


# --- refusal log ---------------------------------------------------------------


def test_every_refusal_is_a_row_with_all_fields(audit):
    result, _, _ = run(audit, [spec(audit)], {"A": BAD})
    assert len(result.refusals) == 2
    row = result.refusals[0]
    assert set(row) == {
        "config", "arm", "process", "check", "outcome", "reason", "commit", "report",
    }  # fmt: skip
    assert row["config"] == "C1" and row["arm"] == "A" and row["process"] == 1
    assert row["commit"] == "abc123"
    assert row["report"] == "report.json"
    assert row["check"] == "grad_norm, parameters_changed"
    assert "grad_norm refused" in row["reason"]
    assert [r["process"] for r in result.refusals] == [1, 2]


def test_an_arm_that_did_not_start_is_a_refusal_row(audit):
    result, _, _ = run(audit, [spec(audit)], {"A": RuntimeError("no GPU")})
    (row,) = result.refusals
    assert row["outcome"] == "did not start"
    assert row["process"] == 1
    assert row["reason"] == "no GPU"
    assert row["report"] is None


def test_valid_runs_leave_no_refusal_row(audit):
    result, _, _ = run(audit, [spec(audit)], {"A": GOOD})
    assert result.refusals == []


def test_refusals_are_written_as_jsonl(audit, tmp_path):
    result, _, _ = run(audit, [spec(audit)], {"A": BAD})
    path = tmp_path / "refusals.jsonl"
    audit.write_refusals(result.refusals, path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == result.refusals


# --- records -------------------------------------------------------------------


def test_the_audit_keeps_every_arm_record_and_report(audit):
    result, _, _ = run(audit, [spec(audit)], {"A": GOOD})
    assert [(r["arm"], r["process"]) for r in result.records] == [("A", 1), ("A", 2)]
    assert result.records[0]["watch"]["outcome"] == "ok"
    assert result.records[0]["report"] == GOOD


# --- module hygiene ------------------------------------------------------------


def test_module_does_not_import_torch_at_top_level(audit):
    source = (HARNESS / "f3_audit.py").read_text(encoding="utf-8")
    top_level = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    assert not any("torch" in line or "transformers" in line for line in top_level)
