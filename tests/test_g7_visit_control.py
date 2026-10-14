"""Tests for the G7 positive control's injections (benchmarks/harness/visit_control.py, ticket 05).

CPU only. The launch is a counting fake and the clock is injectable, so no test needs a GPU.
The real-clock test checks the spin lasts at least 200 us.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

import visit_control  # noqa: E402


class StepClock:
    """A clock that moves ``step`` ns on every read."""

    def __init__(self, step):
        self.now = 0
        self.step = step
        self.reads = 0

    def __call__(self):
        self.reads += 1
        self.now += self.step
        return self.now


def test_the_defaults_are_a_200_us_spin_and_50_launches():
    assert visit_control.SPIN_NS == 200_000
    assert visit_control.LAUNCHES == 50


def test_the_spin_waits_on_the_clock_until_200_us_have_passed():
    clock = StepClock(step=30_000)
    wrappers = visit_control.control_wrappers(clock=clock, launch=lambda: None)

    wrappers["substituted_weights"](lambda *a: "weights")("self", "buffers")

    # One read to start, then reads until the clock has moved 200 us: 7 steps of 30 us.
    assert clock.reads == 1 + 7


def test_the_spin_runs_before_the_original_and_returns_its_result():
    log = []
    wrappers = visit_control.control_wrappers(clock=StepClock(100_000), launch=lambda: None)

    def original(*args):
        log.append(("original", args))
        return "weights"

    result = wrappers["substituted_weights"](original)("self", "buffers")

    assert result == "weights"
    assert log == [("original", ("self", "buffers"))]


def test_the_real_spin_lasts_at_least_200_us():
    wrappers = visit_control.control_wrappers(launch=lambda: None)
    spin = wrappers["substituted_weights"](lambda *a: None)

    started = time.perf_counter_ns()
    spin("self", "buffers")

    assert time.perf_counter_ns() - started >= 200_000


def test_functional_call_launches_50_times_then_calls_the_original():
    log = []
    wrappers = visit_control.control_wrappers(clock=StepClock(1), launch=lambda: log.append("k"))

    def original(*args, **kwargs):
        log.append("original")
        return "out"

    assert wrappers["functional_call"](original)("inner", {}, ()) == "out"
    assert log == ["k"] * 50 + ["original"]


def test_spin_and_launch_count_can_be_changed_for_the_tests():
    log = []
    wrappers = visit_control.control_wrappers(
        spin_ns=10, launches=3, clock=StepClock(10), launch=lambda: log.append(1)
    )
    wrappers["functional_call"](lambda *a, **k: None)()
    assert len(log) == 3


def test_it_wraps_exactly_the_substitution_and_functional_call_points():
    wrappers = visit_control.control_wrappers(launch=lambda: None)
    assert set(wrappers) == {"substituted_weights", "functional_call"}


def test_the_default_launch_is_one_tiny_cuda_kernel():
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    launch = visit_control.cuda_launch()
    launch()
    launch()
    torch.cuda.synchronize()
