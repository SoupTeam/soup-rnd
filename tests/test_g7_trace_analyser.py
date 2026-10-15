"""Tests for the G7 trace analyser (benchmarks/harness/trace_analyser.py, G7 ticket 04).

Seam 1: synthetic profiler events with known durations (microseconds, as in the exported
chrome trace) go in, and per-visit parts, kernel time, launch counts and the reconciliation
come out. The events are labelled synthetic and no test reads a real trace.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"


@pytest.fixture(scope="module")
def analyser():
    name = "_trace_analyser_test_module"
    spec = importlib.util.spec_from_file_location(name, HARNESS / "trace_analyser.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _event(cat, name, ts, dur, tid=1, corr=None):
    args = {} if corr is None else {"correlation": corr}
    return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur, "tid": tid, "args": args}


def rng(name, ts, dur, tid=1):
    return _event("user_annotation", name, ts, dur, tid)


def api(name, ts, dur, corr, tid=1):
    return _event("cuda_runtime", name, ts, dur, tid, corr)


def kernel(ts, dur, corr):
    return _event("kernel", "some_kernel", ts, dur, tid=7, corr=corr)


def op(name, ts, dur, tid=1):
    return _event("cpu_op", name, ts, dur, tid)


# One forward visit of 1000 us at t=0: weight preparation 100 to 300 (200 us), functional_call
# 300 to 900 with the call into the layer 400 to 800.
def forward_visit(analyser, *, extra=()):
    n = analyser.RANGES
    return [
        rng(n.forward_visit, 0, 1000),
        rng(n.weight_prep, 100, 200),
        rng(n.functional_call, 300, 600),
        rng(n.layer_call, 400, 400),
        *extra,
    ]


def only(result, kind="forward"):
    return result["summary"][kind]


def test_a_launch_inside_the_pool_wait_counts_as_launch_time_not_weight_preparation(analyser):
    events = forward_visit(analyser, extra=[api("cudaLaunchKernel", 150, 20, 1)])

    visit = analyser.analyse_trace(events)["visits"][0]

    assert visit["launch_ms"] == pytest.approx(0.020)
    assert visit["weight_prep_ms"] == pytest.approx(0.180)


def test_the_four_parts_sum_to_the_visit(analyser):
    events = forward_visit(
        analyser,
        extra=[
            api("cudaLaunchKernel", 150, 20, 1),  # in weight preparation
            api("cudaMemcpyAsync", 320, 10, 2),  # in functional_call machinery
            api("cudaLaunchKernel", 500, 30, 3),  # in the call into the layer
            api("cudaEventRecordWithFlags", 950, 5, 4),  # plain Python and dispatcher
        ],
    )

    visit = analyser.analyse_trace(events)["visits"][0]

    assert visit["host_ms"] == pytest.approx(1.0)
    assert visit["launch_ms"] == pytest.approx(0.065)
    assert visit["weight_prep_ms"] == pytest.approx(0.180)
    assert visit["functional_call_ms"] == pytest.approx(0.190)  # 600 - 400 - 10
    assert visit["python_dispatcher_ms"] == pytest.approx(1.0 - 0.065 - 0.180 - 0.190)
    parts = ("launch_ms", "weight_prep_ms", "functional_call_ms", "python_dispatcher_ms")
    assert sum(visit[part] for part in parts) == pytest.approx(visit["host_ms"])


def test_all_three_kernel_launch_apis_and_the_other_queueing_calls_are_counted(analyser):
    events = forward_visit(
        analyser,
        extra=[
            api("cudaLaunchKernel", 410, 5, 1),
            api("cuLaunchKernel", 420, 5, 2),
            api("cudaLaunchKernelExC", 430, 5, 3),
            api("cudaMemcpyAsync", 440, 5, 4),
            api("cudaEventRecordWithFlags", 450, 5, 5),
            api("cudaStreamWaitEvent", 460, 5, 6),
            api("cudaMalloc", 470, 50, 7),  # does not queue work
        ],
    )

    visit = analyser.analyse_trace(events)["visits"][0]

    assert visit["launches"] == {
        "cudaLaunchKernel": 1,
        "cuLaunchKernel": 1,
        "cudaLaunchKernelExC": 1,
        "cudaMemcpyAsync": 1,
        "cudaEventRecordWithFlags": 1,
        "cudaStreamWaitEvent": 1,
    }
    assert visit["launch_ms"] == pytest.approx(0.030)


def test_kernel_time_comes_from_kernel_events_matched_by_correlation_id(analyser):
    events = forward_visit(
        analyser,
        extra=[
            api("cudaLaunchKernel", 410, 5, 11),
            api("cuLaunchKernel", 420, 5, 12),
            kernel(2000, 70, 11),
            kernel(2100, 30, 12),
            kernel(2200, 999, 13),  # launched outside the visit
        ],
    )

    visit = analyser.analyse_trace(events)["visits"][0]

    assert visit["kernel_ms"] == pytest.approx(0.100)


def test_forward_and_recompute_visits_are_told_apart(analyser):
    n = analyser.RANGES
    events = [
        rng(n.forward_visit, 0, 1000),
        rng(n.recompute_visit, 2000, 600),
        rng(n.recompute_visit, 3000, 800),
    ]

    result = analyser.analyse_trace(events)

    assert only(result)["count"] == 1
    assert only(result, "recompute")["count"] == 2
    assert only(result, "recompute")["host_ms"] == pytest.approx(0.7)


def test_launches_of_another_thread_are_not_the_visits_launch_time(analyser):
    events = forward_visit(analyser, extra=[api("cudaLaunchKernel", 500, 40, 1, tid=2)])

    visit = analyser.analyse_trace(events)["visits"][0]

    assert visit["launch_ms"] == 0
    assert visit["launches"] == {}


def test_layer_backward_leaves_out_the_recompute_visit_and_splits_launch_time(analyser):
    n = analyser.RANGES
    events = [
        rng(n.layer_backward, 0, 1000, tid=2),
        rng(n.recompute_visit, 100, 400, tid=2),
        api("cudaLaunchKernel", 200, 50, 1, tid=2),  # inside the recompute visit
        api("cudaLaunchKernel", 600, 30, 2, tid=2),
        api("cudaMemcpyAsync", 700, 20, 3, tid=2),
        kernel(5000, 40, 2),
    ]

    backward = analyser.analyse_trace(events)["layer_backwards"][0]

    assert backward["host_ms"] == pytest.approx(0.600)
    assert backward["launch_ms"] == pytest.approx(0.050)
    assert backward["rest_ms"] == pytest.approx(0.550)
    assert backward["kernel_ms"] == pytest.approx(0.040)
    assert backward["launches"] == {"cudaLaunchKernel": 1, "cudaMemcpyAsync": 1}


def test_summary_is_the_median_over_visits(analyser):
    n = analyser.RANGES
    events = [
        rng(n.forward_visit, 0, 1000),
        rng(n.forward_visit, 2000, 3000),
        rng(n.forward_visit, 6000, 2000),
    ]

    assert only(analyser.analyse_trace(events))["host_ms"] == pytest.approx(2.0)


# One step of 10 ms: a 3 ms visit, a 4 ms layer backward, a 1 ms final wait and work outside
# the layers. What the outside work leaves of the remaining 2 ms is the gap.
@pytest.mark.parametrize("gap_us, gap_fraction, ok", [(400, 0.04, True), (600, 0.06, False)])
def test_reconciliation_passes_at_4_percent_and_fails_at_6(analyser, gap_us, gap_fraction, ok):
    n = analyser.RANGES
    events = [
        rng(n.step, 0, 10_000),
        rng(n.forward_visit, 0, 3000),
        rng(n.layer_backward, 3000, 4000, tid=2),
        op("aten::embedding", 7000, 2000 - gap_us),
        api("cudaDeviceSynchronize", 9000, 1000, 9),
    ]

    reconciliation = analyser.analyse_trace(events)["reconciliation"]

    assert reconciliation["gap_fraction"] == pytest.approx(gap_fraction)
    assert reconciliation["ok"] is ok


def test_reconciliation_reports_every_part_and_the_gap(analyser):
    n = analyser.RANGES
    events = [
        rng(n.step, 0, 10_000),
        rng(n.forward_visit, 0, 3000),
        rng(n.recompute_visit, 3000, 500, tid=2),
        rng(n.layer_backward, 3000, 4000, tid=2),  # includes the recompute visit
        api("cudaDeviceSynchronize", 9000, 1000, 9),
        op("aten::embedding", 7000, 1500),
        op("aten::copy_", 7200, 100),  # nested in the embedding op, counted once
    ]

    r = analyser.analyse_trace(events)["reconciliation"]

    assert r["step_ms"] == pytest.approx(10.0)
    assert r["visits_ms"] == pytest.approx(3.5)
    assert r["layer_backward_ms"] == pytest.approx(3.5)
    assert r["final_wait_ms"] == pytest.approx(1.0)
    assert r["outside_ms"] == pytest.approx(1.5)
    assert r["gap_ms"] == pytest.approx(0.5)
    assert r["gap_fraction"] == pytest.approx(0.05)
    assert r["ok"] is True


def test_the_final_wait_is_the_last_synchronize_of_the_step(analyser):
    n = analyser.RANGES
    events = [
        rng(n.step, 0, 10_000),
        api("cudaDeviceSynchronize", 1000, 500, 1),  # in the middle: outside work
        api("cudaDeviceSynchronize", 9000, 1000, 2),
    ]

    r = analyser.analyse_trace(events)["reconciliation"]

    assert r["final_wait_ms"] == pytest.approx(1.0)
    assert r["outside_ms"] == pytest.approx(0.5)


def test_without_a_step_range_the_reconciliation_is_absent(analyser):
    result = analyser.analyse_trace(forward_visit(analyser))

    assert result["reconciliation"] is None


def test_device_to_device_copies_are_traced_back_to_the_ops_that_issued_them(analyser):
    copy = _event("gpu_memcpy", "Memcpy DtoD (Device -> Device)", 9000, 3, tid=7, corr=21)
    events = forward_visit(
        analyser,
        extra=[
            op("aten::copy_", 450, 20),
            op("aten::_to_copy", 440, 40),
            op("aten::to", 430, 60),
            api("cudaMemcpyAsync", 460, 5, 21),
            copy,
        ],
    )

    result = analyser.analyse_trace(events)

    assert result["memcpy_sources"] == {"aten::to > aten::_to_copy > aten::copy_": 1}
