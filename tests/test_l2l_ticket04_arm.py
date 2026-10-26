"""CPU tests for the A5 ticket 04 arm harness: the loss-curve rule and the per-step meter."""

from __future__ import annotations

import math

import pytest

from tests.test_l2l_probe_harness import _load


@pytest.fixture()
def arm():
    return _load("l2l_ticket04_arm")


def test_identical_curves_pass_with_zero_floor_error(arm):
    ref = [2.0, 1.5, 1.2]
    verdict = arm.judge_loss_curve(ref, list(ref), [2.0, 1.5, 1.2])
    assert verdict["judged"] and verdict["pass"]


def test_error_above_one_and_a_half_floors_fails_at_that_step(arm):
    ref = [2.0, 1.5, 1.2]
    floor = [2.0, 1.5 * 1.001, 1.2]
    cand = [2.0, 1.5 * 1.002, 1.2]
    verdict = arm.judge_loss_curve(ref, cand, floor)
    assert verdict["judged"] and not verdict["pass"]
    assert [item["step"] for item in verdict["failing"]] == [1]


def test_error_within_one_and_a_half_floors_passes(arm):
    ref = [2.0, 1.5]
    floor = [2.0, 1.5 * 1.002]
    cand = [2.0, 1.5 * 1.003]
    assert arm.judge_loss_curve(ref, cand, floor)["pass"]


def test_no_floor_gives_no_verdict(arm):
    verdict = arm.judge_loss_curve([1.0], [1.0], None)
    assert not verdict["judged"] and not verdict["pass"] and verdict["problem"]


def test_curves_of_different_length_give_no_verdict(arm):
    verdict = arm.judge_loss_curve([1.0, 2.0], [1.0], [1.0, 2.0])
    assert not verdict["judged"] and "length" in verdict["problem"]


def test_missing_floor_value_gives_no_verdict(arm):
    verdict = arm.judge_loss_curve([1.0, 2.0], [1.0, 2.0], [1.0, None])
    assert not verdict["judged"] and "floor" in verdict["problem"]


def test_missing_candidate_value_fails(arm):
    verdict = arm.judge_loss_curve([1.0, 2.0], [1.0, None], [1.0, 2.1])
    assert verdict["judged"] and not verdict["pass"]


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def _meter(arm, clock, state):
    return arm.StepMeter({
        "counters": lambda: dict(state["counters"]),
        "io": lambda: dict(state["io"]),
        "peaks": lambda: (state["vram"], state["vram"] + 10),
        "reset_peaks": lambda: None,
        "reset_rss": lambda: state["rss_reset"],
        "rss": lambda: state["rss"],
        "clock": clock,
        "tokens": lambda: state["tokens"],
        "t0": 90.0,
    })


def _state():
    return {
        "counters": {"layer_loads": 0, "nf4_restores": 0, "large_loads": 0,
                     "pinned_store_bytes": 7},
        "io": {"read_bytes": 1000, "write_bytes": 5},
        "vram": 111, "rss": 222, "rss_reset": True, "tokens": 300,
    }


def _one_step(meter, clock, state, *, seconds, loads, read):
    meter.step_begin()
    clock.now += seconds
    state["counters"]["layer_loads"] += loads
    state["io"]["read_bytes"] += read
    return meter.step_end()


def test_step_holds_deltas_and_token_rate(arm):
    clock, state = _Clock(), _state()
    meter = _meter(arm, clock, state)
    step = _one_step(meter, clock, state, seconds=2.0, loads=64, read=40)
    assert step["step_s"] == 2.0
    assert step["tokens_per_s"] == 150.0
    assert step["layer_loads"] == 64
    assert step["os_read_bytes"] == 40
    assert step["peak_vram_allocated_bytes"] == 111
    assert step["peak_rss_bytes"] == 222


def test_rss_that_cannot_be_reset_is_null_with_a_reason(arm):
    clock, state = _Clock(), _state()
    state["rss_reset"] = False
    meter = _meter(arm, clock, state)
    step = _one_step(meter, clock, state, seconds=1.0, loads=1, read=0)
    assert step["peak_rss_bytes"] is None
    assert "peak_rss_bytes" in step["unavailable"]


def test_io_counter_going_backwards_is_null_with_a_reason(arm):
    clock, state = _Clock(), _state()
    meter = _meter(arm, clock, state)
    meter.step_begin()
    state["io"]["read_bytes"] = 1
    meter.step_end()
    assert meter.steps[0]["os_read_bytes"] is None
    assert "os_read_bytes" in meter.steps[0]["unavailable"]


def test_run_record_skips_step_zero_and_sums_the_rest(arm):
    clock, state = _Clock(), _state()
    meter = _meter(arm, clock, state)
    _one_step(meter, clock, state, seconds=10.0, loads=70, read=900)
    _one_step(meter, clock, state, seconds=2.0, loads=62, read=0)
    _one_step(meter, clock, state, seconds=2.0, loads=62, read=0)
    record = meter.run_record()
    assert record["steps"] == 2
    assert record["tokens_per_s"] == pytest.approx(150.0)
    assert record["layer_loads"] == 124
    assert record["layer_loads_per_step"] == 62
    assert record["os_read_bytes"] == 0


def test_run_record_needs_a_step_after_step_zero(arm):
    clock, state = _Clock(), _state()
    meter = _meter(arm, clock, state)
    _one_step(meter, clock, state, seconds=1.0, loads=1, read=0)
    with pytest.raises(ValueError):
        meter.run_record()


def test_end_without_begin_is_an_error(arm):
    clock, state = _Clock(), _state()
    with pytest.raises(RuntimeError):
        _meter(arm, clock, state).step_end()


def test_losses_are_kept_with_their_step(arm):
    clock, state = _Clock(), _state()
    meter = _meter(arm, clock, state)
    meter.log(1, {"loss": 2.5, "learning_rate": 1e-4})
    meter.log(1, {"grad_norm": 3.0})
    assert meter.losses == [{"step": 1, "loss": 2.5, "learning_rate": 1e-4}]


def test_relative_error_of_zero_reference_is_infinite(arm):
    assert math.isinf(arm.relative_errors([0.0], [1.0])[0])
