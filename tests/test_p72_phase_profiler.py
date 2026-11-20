"""PR #72 phase profiler: exclusive timing, clean restore, bit-equal numbers, red-capable gate."""

from __future__ import annotations

import time

import pytest

from benchmarks.harness import p72_mbr_phase_profile as cli
from benchmarks.harness import p72_phase_profiler as profiler
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR

# The tiny step is ~30 ms, so its fixed glue time sits near 5%; real runs keep the 5% gate.
TINY = ["--blocks", "1", "--warmup", "1", "--kinds", "mbr", "--max-unattributed", "0.2"]


class ManualTime:
    """A clock that only moves when a test says so: no sleep, so a slow runner cannot stretch it."""

    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_clock_charges_each_phase_only_its_own_time():
    ticks = ManualTime()
    clock = profiler.PhaseClock("cpu", now=ticks)
    with clock.run():
        ticks.advance(0.01)
        with clock.frame("outer"):
            ticks.advance(0.01)
            with clock.frame("inner"):
                ticks.advance(0.03)
    assert clock.totals["inner"] == pytest.approx(0.03)
    assert clock.totals["outer"] == pytest.approx(0.01)
    assert clock.totals[profiler.ROOT] == pytest.approx(0.01)
    assert clock.counts["inner"] == 1


def test_worker_thread_frames_nest_under_a_blocked_autograd_backward():
    import threading

    ticks = ManualTime()
    clock = profiler.PhaseClock("cpu", now=ticks)

    def worker():
        with clock.frame("load"):
            with clock.frame("disk_read"):
                ticks.advance(0.02)

    with clock.run():
        with clock.frame("bwd/autograd_backward"):
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
    assert clock.counts["disk_read"] == 1
    assert clock.totals["disk_read"] == pytest.approx(0.02)
    assert not clock.background


def test_background_frames_use_the_injected_clock_too():
    import threading

    ticks = ManualTime()
    clock = profiler.PhaseClock("cpu", now=ticks)

    def prefetch():
        with clock.frame("read"):
            ticks.advance(0.04)

    with clock.run():
        thread = threading.Thread(target=prefetch, name="expert-prefetch_0")
        thread.start()
        thread.join()
    assert clock.background["read"] == pytest.approx(0.04)
    assert "read" not in clock.totals


def test_worker_thread_frames_outside_autograd_stay_in_background():
    import threading

    clock = profiler.PhaseClock("cpu")
    with clock.run():
        thread = threading.Thread(target=lambda: clock.frame("prefetch").__enter__())
        thread.start()
        thread.join()
    assert "prefetch" not in clock.totals


def test_prefetch_worker_frames_overlap_without_breaking_the_stack():
    import threading

    clock = profiler.PhaseClock("cpu")
    started, release = threading.Event(), threading.Event()

    def worker():
        with clock.frame("disk_read"):
            started.set()
            release.wait(2)

    thread = threading.Thread(target=worker, name="expert-prefetch_0")
    with clock.run():
        with clock.frame("bwd/autograd_backward"):
            thread.start()
            started.wait(2)
            with clock.frame("gemm"):
                time.sleep(0.01)
            release.set()
            thread.join()
    assert "disk_read" not in clock.totals and clock.background["disk_read"] > 0
    assert clock.counts["gemm"] == 1


def test_clock_rejects_frames_closed_out_of_order():
    clock = profiler.PhaseClock("cpu")
    first = clock.push("a")
    clock.push("b")
    with pytest.raises(RuntimeError, match="out of order"):
        clock.pop(first)


@_REQUIRES_TRAINING_FLOOR
def test_install_restores_every_patched_attribute():
    import torch

    from soup_cli.utils import expert_stream, l2l_step

    before = (
        expert_stream.ExpertBufferPool.lease,
        expert_stream.ExpertBufferPool._load,
        l2l_step.L2LStep._layer,
        torch.autograd.backward,
        torch.nn.functional.linear,
    )
    with profiler.install(profiler.PhaseClock("cpu")):
        assert l2l_step.L2LStep._layer is not before[2]
    after = (
        expert_stream.ExpertBufferPool.lease,
        expert_stream.ExpertBufferPool._load,
        l2l_step.L2LStep._layer,
        torch.autograd.backward,
        torch.nn.functional.linear,
    )
    assert after == before


@_REQUIRES_TRAINING_FLOOR
def test_tiny_step_is_green_and_bit_equal():
    assert cli.main(TINY) == 0


@_REQUIRES_TRAINING_FLOOR
def test_without_phases_the_coverage_gate_goes_red():
    assert cli.main([*TINY, "--coarse"]) == 1
