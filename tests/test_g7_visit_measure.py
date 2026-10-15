"""Tests for the G7 driver (benchmarks/harness/visit_measure.py, ticket 07).

The driver runs the whole measurement in the committed order. These tests inject a fake arm
runner and a fake experiment, so no test starts a worker or reads a sensor.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"


@pytest.fixture(scope="module")
def measure():
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    name = "_visit_measure_test_module"
    spec = importlib.util.spec_from_file_location(name, HARNESS / "visit_measure.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def ok_record():
    return {"outcome": "ok", "checks": {}}


class FakeArms:
    """Records every call and returns numbers that pass every profile row."""

    def __init__(self, timer_host_ms=2.0):
        self.calls = []
        self.timer_host_ms = timer_host_ms

    def __call__(self, *, arm, seq, round, run, tag, out_dir):
        self.calls.append({"arm": arm, "seq": seq, "round": round, "run": run, "tag": tag})
        run_json = {"arm": arm, "seq": seq, "median_step_s": 1.0}
        if arm in ("timer", "control"):
            # The control's timer reads the profile arm's 2.4 ms plus the injections.
            host = 2.65 if arm == "control" else self.timer_host_ms
            run_json["timer"] = {"forward_visit_ms": {"median": host}}
        if arm in ("profile", "control"):
            extra = 1 if arm == "control" else 0
            run_json["profile"] = {
                "summary": {
                    "forward": {
                        "host_ms": 2.4 + 0.25 * extra,
                        "weight_prep_ms": 0.5 + 0.2 * extra,
                        "kernel_ms": 0.4,
                        "launch_ms": 0.6 + 0.05 * extra,
                        "launches_total": 100 + 50 * extra,
                    }
                }
            }
        return {"run": run_json, "arm": ok_record()}


def fake_experiment(verdict="fail"):
    def run():
        return {"gate": {"passed": True, "reason": "bit-equal"}, "blocks": [],
                "verdict": {"verdict": verdict, "reason": f"experiment {verdict}", "evidence": {}}}
    return run


# --- the order ----------------------------------------------------------------------------


def test_round_one_runs_bare_timer_profile_at_each_token_count_and_the_control_at_16(measure):
    plan = measure.round_plan(1)
    assert plan[:4] == [("bare", 16), ("timer", 16), ("profile", 16), ("control", 16)]
    assert plan[4:7] == [("bare", 32), ("timer", 32), ("profile", 32)]
    assert plan[-3:] == [("bare", 128), ("timer", 128), ("profile", 128)]
    assert len(plan) == 13


def test_round_two_is_round_one_reversed(measure):
    assert measure.round_plan(2) == list(reversed(measure.round_plan(1)))


def test_the_profile_runs_every_arm_once_per_round_in_that_order(measure, tmp_path):
    arms = FakeArms()
    measure.run_profile(arms, tmp_path)

    assert len(arms.calls) == 26
    assert [(c["arm"], c["seq"]) for c in arms.calls[:13]] == measure.round_plan(1)
    assert [(c["arm"], c["seq"]) for c in arms.calls[13:]] == measure.round_plan(2)
    assert [c["round"] for c in arms.calls] == [1] * 13 + [2] * 13
    assert [c["run"] for c in arms.calls] == list(range(1, 27))
    assert len({c["tag"] for c in arms.calls}) == 2  # one tag per round keeps the files apart


def test_the_dry_run_shows_the_order_and_runs_nothing(measure, capsys):
    lines = measure.dry_run()

    assert lines[0].startswith("profile round 1")
    assert "bare 16" in lines[0] and "control 16" in lines[0]
    assert lines[1].startswith("profile round 2")
    assert lines[1].index("control 16") < lines[1].index("bare 16")
    assert lines[2].startswith("experiment")
    assert "gate" in lines[2] and "A-B-B-A" in lines[2]


# --- evaluation ---------------------------------------------------------------------------


def test_the_profile_rows_agree_across_rounds_and_cover_every_token_count(measure, tmp_path):
    rounds = measure.run_profile(FakeArms(), tmp_path)
    rows = measure.evaluate_profile(rounds)

    assert set(rows) == {16, 32, 64, 128}
    assert rows[16]["control"]["verdict"] == "pass"
    assert rows[16]["cpu_bound"]["verdict"] == "pass"
    assert set(rows[32]) == {"timer_is_free", "profile_is_readable"}
    assert rows[32]["timer_is_free"]["verdict"] == "pass"


def test_the_whole_measurement_writes_a_summary_with_the_recommendation(measure, tmp_path):
    summary = measure.run_measurement(
        FakeArms(), fake_experiment("pass"), tmp_path, prototype=None
    )

    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved["recommendation"] == summary["recommendation"]
    assert "Continue: `tie_weights=False` upstream" in summary["recommendation"]
    assert summary["branches"]["cuda_graphs"]["row"] == "prototype"
    assert set(summary["profile_rows"]) == {"16", "32", "64", "128"}


def test_the_experiment_runs_after_the_profile(measure, tmp_path):
    order = []
    arms = FakeArms()

    def traced_arms(**kwargs):
        order.append("profile")
        return arms(**kwargs)

    def experiment():
        order.append("experiment")
        return fake_experiment()()

    measure.run_measurement(traced_arms, experiment, tmp_path, prototype=None)
    assert order == ["profile"] * 26 + ["experiment"]


def test_a_cpu_bound_visit_without_a_prototype_waits_for_the_prototype(measure, tmp_path):
    summary = measure.run_measurement(FakeArms(), fake_experiment(), tmp_path, prototype=None)
    assert summary["recommendation"] == "No verdict"
