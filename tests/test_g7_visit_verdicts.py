"""Tests for the G7 verdict evaluator (benchmarks/harness/visit_verdicts.py, G7 ticket 05).

Seam 2: arm records and run summaries in, validity rows, control and threshold out. All
numbers are fixed and synthetic. Times are in ms, like the run JSONs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"


@pytest.fixture(scope="module")
def verdicts():
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    name = "_visit_verdicts_test_module"
    spec = importlib.util.spec_from_file_location(name, HARNESS / "visit_verdicts.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def arm_record(outcome="ok", **bad):
    checks = {name: {"outcome": "ok", "reason": "fine"} for name in ("suspend", "power")}
    for name, value in bad.items():
        checks[name] = {"outcome": value, "reason": f"{name} is {value}"}
    return {"outcome": outcome, "checks": checks}


def bare(step_s=1.0, seq=16, outcome="ok"):
    return {"run": {"arm": "bare", "seq": seq, "median_step_s": step_s}, "arm": arm_record(outcome)}


def timer(step_s=1.0, host_ms=2.0, seq=16, outcome="ok"):
    run = {
        "arm": "timer",
        "seq": seq,
        "median_step_s": step_s,
        "timer": {"forward_visit_ms": {"median": host_ms}},
    }
    return {"run": run, "arm": arm_record(outcome)}


def parts(host_ms=2.4, weight_prep_ms=0.5, kernel_ms=0.4, launch_ms=0.6, launches=100):
    return {
        "summary": {
            "forward": {
                "host_ms": host_ms,
                "weight_prep_ms": weight_prep_ms,
                "kernel_ms": kernel_ms,
                "launch_ms": launch_ms,
                "launches_total": launches,
            }
        }
    }


def profile(seq=16, outcome="ok", **numbers):
    run = {"arm": "profile", "seq": seq, "profile": parts(**numbers)}
    return {"run": run, "arm": arm_record(outcome)}


def control(seq=16, outcome="ok", timer_host_ms=2.4, **numbers):
    run = {
        "arm": "control",
        "seq": seq,
        "profile": parts(**numbers),
        "timer": {"forward_visit_ms": {"median": timer_host_ms}},
    }
    return {"run": run, "arm": arm_record(outcome)}


def good_control(prof=None, delta_prep=0.2, delta_launches=50, delta_timer=None, launch_cost=0.05):
    """A control whose numbers are the profile arm's plus the injections."""
    base = (prof or profile())["run"]["profile"]["summary"]["forward"]
    if delta_timer is None:
        delta_timer = delta_prep + launch_cost
    return control(
        host_ms=base["host_ms"] + delta_prep + launch_cost,
        weight_prep_ms=base["weight_prep_ms"] + delta_prep,
        kernel_ms=base["kernel_ms"],
        launch_ms=base["launch_ms"] + launch_cost,
        launches=base["launches_total"] + delta_launches,
        timer_host_ms=base["host_ms"] + delta_timer,
    )


def evaluate(verdicts, **arms):
    defaults = {
        "bare": bare(),
        "timer": timer(),
        "profile": profile(),
        "control": good_control(),
    }
    defaults.update(arms)
    return verdicts.evaluate(**defaults)


# --- timer is free ------------------------------------------------------------------------


def test_timer_row_passes_at_1_9_percent(verdicts):
    result = evaluate(verdicts, timer=timer(step_s=1.019))
    assert result["timer_is_free"]["verdict"] == "pass"


def test_timer_row_fails_at_2_1_percent(verdicts):
    result = evaluate(verdicts, timer=timer(step_s=1.021))
    assert result["timer_is_free"]["verdict"] == "fail"
    assert result["timer_is_free"]["evidence"]["excess"] == pytest.approx(0.021)


def test_timer_row_passes_when_the_timer_step_is_shorter(verdicts):
    result = evaluate(verdicts, timer=timer(step_s=0.99))
    assert result["timer_is_free"]["verdict"] == "pass"


# --- profile is readable ------------------------------------------------------------------


def test_inflation_passes_at_1_4(verdicts):
    result = evaluate(verdicts, timer=timer(host_ms=2.0), profile=profile(host_ms=2.8))
    assert result["profile_is_readable"]["verdict"] == "pass"
    assert result["profile_is_readable"]["evidence"]["inflation"] == pytest.approx(1.4)


def test_inflation_fails_at_1_6(verdicts):
    result = evaluate(verdicts, timer=timer(host_ms=2.0), profile=profile(host_ms=3.2))
    assert result["profile_is_readable"]["verdict"] == "fail"


def test_parts_gates_get_no_verdict_when_the_profile_is_not_readable(verdicts):
    result = evaluate(verdicts, timer=timer(host_ms=2.0), profile=profile(host_ms=3.2))
    assert result["control"]["verdict"] == "no_verdict"
    assert result["cpu_bound"]["verdict"] == "no_verdict"
    assert "profile_is_readable" in result["cpu_bound"]["reason"]


# --- the positive control -----------------------------------------------------------------


def test_control_passes_at_plus_200_us(verdicts):
    result = evaluate(verdicts)
    assert result["control"]["verdict"] == "pass"


def test_control_fails_at_plus_150_us_of_weight_preparation(verdicts):
    prof = profile()
    result = evaluate(verdicts, control=good_control(prof, delta_prep=0.15, delta_timer=0.25))
    assert result["control"]["verdict"] == "fail"
    assert "weight preparation" in result["control"]["reason"]


def test_control_accepts_the_edges_of_plus_or_minus_20_percent(verdicts):
    for delta in (0.16, 0.24):
        control = good_control(delta_prep=delta, delta_timer=delta + 0.05)
        result = evaluate(verdicts, control=control)
        assert result["control"]["verdict"] == "pass", delta


def test_control_fails_at_plus_49_launches(verdicts):
    result = evaluate(verdicts, control=good_control(delta_launches=49))
    assert result["control"]["verdict"] == "fail"
    assert "launches" in result["control"]["reason"]


def test_control_fails_at_plus_51_launches(verdicts):
    result = evaluate(verdicts, control=good_control(delta_launches=51))
    assert result["control"]["verdict"] == "fail"


def test_control_fails_when_the_timer_does_not_see_the_delay(verdicts):
    result = evaluate(verdicts, control=good_control(delta_timer=0.0))
    assert result["control"]["verdict"] == "fail"
    assert "timer" in result["control"]["reason"]


def test_control_timer_check_adds_the_cost_of_50_launches_to_the_delay(verdicts):
    # 0.2 ms plus 0.1 ms of launches is expected; 0.2 ms alone is 0.1 ms short.
    low = evaluate(verdicts, control=good_control(launch_cost=0.1, delta_timer=0.2))
    right = evaluate(verdicts, control=good_control(launch_cost=0.1, delta_timer=0.3))
    assert low["control"]["verdict"] == "fail"
    assert right["control"]["verdict"] == "pass"


def test_failed_control_removes_the_verdict_from_the_threshold(verdicts):
    result = evaluate(verdicts, control=good_control(delta_prep=0.15, delta_timer=0.25))
    assert result["cpu_bound"]["verdict"] == "no_verdict"
    assert "control" in result["cpu_bound"]["reason"]


def test_control_must_run_at_16_tokens(verdicts):
    result = evaluate(verdicts, control=control(seq=32))
    assert result["control"]["verdict"] == "no_verdict"


# --- the CPU-bound threshold --------------------------------------------------------------


def threshold_arms(ratio, kernel_ms=0.4, weight_prep_ms=0.5, launch_ms=0.6):
    """Timer and profile arms where capturable host time is ``ratio`` times kernel time."""
    host = kernel_ms * ratio + weight_prep_ms
    prof = profile(
        host_ms=host, weight_prep_ms=weight_prep_ms, kernel_ms=kernel_ms, launch_ms=launch_ms
    )
    return {"timer": timer(host_ms=host), "profile": prof, "control": good_control(prof)}


def test_threshold_fires_at_2_1_times(verdicts):
    result = evaluate(verdicts, **threshold_arms(2.1))
    assert result["cpu_bound"]["verdict"] == "pass"
    assert result["cpu_bound"]["evidence"]["ratio"] == pytest.approx(2.1)


def test_threshold_does_not_fire_at_1_9_times(verdicts):
    result = evaluate(verdicts, **threshold_arms(1.9))
    assert result["cpu_bound"]["verdict"] == "fail"


def test_threshold_subtracts_weight_preparation_from_the_timer_host_time(verdicts):
    result = evaluate(verdicts, **threshold_arms(2.1, weight_prep_ms=1.0))
    evidence = result["cpu_bound"]["evidence"]
    assert evidence["capturable_ms"] == pytest.approx(0.84)


def test_threshold_reports_the_launch_share_and_does_not_decide_on_it(verdicts):
    low = evaluate(verdicts, **threshold_arms(2.1, launch_ms=0.01))
    high = evaluate(verdicts, **threshold_arms(2.1, launch_ms=2.0))
    assert low["cpu_bound"]["verdict"] == high["cpu_bound"]["verdict"] == "pass"
    assert "launch_share" in high["cpu_bound"]["evidence"]


def test_threshold_needs_16_tokens(verdicts):
    arms = threshold_arms(2.1)
    host_ms = arms["timer"]["run"]["timer"]["forward_visit_ms"]["median"]
    arms["timer"] = timer(host_ms=host_ms, seq=32)
    result = evaluate(verdicts, **arms)
    assert result["cpu_bound"]["verdict"] == "no_verdict"


def test_threshold_has_no_verdict_with_zero_kernel_time(verdicts):
    result = evaluate(verdicts, **threshold_arms(2.1, kernel_ms=0.0))
    assert result["cpu_bound"]["verdict"] == "no_verdict"


# --- void and unknown arms ----------------------------------------------------------------


@pytest.mark.parametrize("outcome", ["void", "unknown"])
def test_a_void_bare_arm_gives_the_timer_row_no_verdict(verdicts, outcome):
    result = evaluate(verdicts, bare=bare(outcome=outcome))
    row = result["timer_is_free"]
    assert row["verdict"] == "no_verdict"
    assert "bare" in row["reason"] and outcome in row["reason"]
    assert result["profile_is_readable"]["verdict"] == "pass"


def test_a_void_timer_arm_takes_the_timer_row_the_inflation_row_and_the_threshold(verdicts):
    result = evaluate(verdicts, timer=timer(outcome="void"))
    assert result["timer_is_free"]["verdict"] == "no_verdict"
    assert result["profile_is_readable"]["verdict"] == "no_verdict"
    assert result["cpu_bound"]["verdict"] == "no_verdict"
    assert "timer" in result["cpu_bound"]["reason"]


def test_a_void_profile_arm_takes_the_inflation_row_the_control_and_the_threshold(verdicts):
    result = evaluate(verdicts, profile=profile(outcome="void"))
    assert result["profile_is_readable"]["verdict"] == "no_verdict"
    assert result["control"]["verdict"] == "no_verdict"
    assert result["cpu_bound"]["verdict"] == "no_verdict"
    assert result["timer_is_free"]["verdict"] == "pass"


def test_a_void_control_arm_takes_the_control_and_the_threshold_only(verdicts):
    result = evaluate(verdicts, control=good_control() | {"arm": arm_record("void", swap="void")})
    assert result["control"]["verdict"] == "no_verdict"
    assert "swap" in result["control"]["reason"]
    assert result["cpu_bound"]["verdict"] == "no_verdict"
    assert result["timer_is_free"]["verdict"] == "pass"


def test_a_missing_arm_gives_no_verdict_and_names_it(verdicts):
    result = evaluate(verdicts, control=None)
    assert result["control"]["verdict"] == "no_verdict"
    assert "control" in result["control"]["reason"]
    assert result["cpu_bound"]["verdict"] == "no_verdict"


def test_a_missing_number_in_a_run_gives_no_verdict_not_an_error(verdicts):
    broken = timer()
    del broken["run"]["timer"]
    result = evaluate(verdicts, timer=broken)
    assert result["profile_is_readable"]["verdict"] == "no_verdict"
    assert "timer" in result["profile_is_readable"]["reason"]


def test_arms_of_different_token_counts_give_no_verdict_for_the_row(verdicts):
    result = evaluate(verdicts, bare=bare(seq=32))
    assert result["timer_is_free"]["verdict"] == "no_verdict"
