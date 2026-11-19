"""PR #72 phase profiler: exclusive timing, clean restore, bit-equal numbers, red-capable gate."""

from __future__ import annotations

import time

import pytest

from benchmarks.harness import p72_mbr_phase_profile as cli
from benchmarks.harness import p72_phase_profiler as profiler
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR

# The tiny step is ~30 ms, so its fixed glue time sits near 5%; real runs keep the 5% gate.
TINY = ["--blocks", "1", "--warmup", "1", "--kinds", "mbr", "--max-unattributed", "0.2"]


def test_clock_charges_each_phase_only_its_own_time():
    clock = profiler.PhaseClock("cpu")
    with clock.run():
        time.sleep(0.01)
        with clock.frame("outer"):
            time.sleep(0.01)
            with clock.frame("inner"):
                time.sleep(0.03)
    assert clock.totals["inner"] == pytest.approx(0.03, abs=0.015)
    assert clock.totals["outer"] == pytest.approx(0.01, abs=0.015)
    assert clock.totals[profiler.ROOT] == pytest.approx(0.01, abs=0.015)
    assert clock.counts["inner"] == 1


def test_worker_thread_frames_nest_under_a_blocked_autograd_backward():
    import threading

    clock = profiler.PhaseClock("cpu")

    def worker():
        with clock.frame("load"):
            with clock.frame("disk_read"):
                time.sleep(0.02)

    with clock.run():
        with clock.frame("bwd/autograd_backward"):
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
    assert clock.counts["disk_read"] == 1
    assert clock.totals["disk_read"] == pytest.approx(0.02, abs=0.015)
    assert not clock.background


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
