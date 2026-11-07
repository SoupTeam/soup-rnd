"""B1.4 grouped-compute diagnostic: bucket routing, kernel attribution, one bucket on CPU.

Three seams of ``benchmarks/harness/b14_group_diag.py``: ``route_for_bucket``,
``attribute_kernels`` and ``run_bucket``. All run on CPU. The CUDA numbers (kernel counts
per phase, paired timing, grouped GEMM) come only from the run on the rented or Kaggle GPU.
"""

from __future__ import annotations

import pytest

from benchmarks.harness.b14_group_diag import BUCKETS, route_for_bucket


def test_buckets_are_the_agreed_row_counts() -> None:
    assert BUCKETS == (1, 2, 4, 8, 16, 32, 64, 128)


@pytest.mark.parametrize("rows", BUCKETS)
def test_each_visited_expert_gets_exactly_the_bucket_row_count(rows: int) -> None:
    import torch

    index, weights = route_for_bucket(rows, experts=8)

    assert index.dtype == torch.int64 and index.shape == (rows * 8, 1)
    counts = torch.bincount(index.flatten(), minlength=8).tolist()
    assert counts == [rows] * 8
    assert weights.shape == index.shape and bool((weights > 0).all())


def test_a_bucket_never_routes_to_an_expert_outside_the_layer() -> None:
    index, _ = route_for_bucket(3, experts=5)

    assert int(index.min()) == 0 and int(index.max()) == 4


@pytest.mark.parametrize("rows, experts", [(0, 8), (-1, 8), (4, 0)])
def test_a_bucket_needs_positive_rows_and_experts(rows: int, experts: int) -> None:
    with pytest.raises(ValueError):
        route_for_bucket(rows, experts=experts)


# --------------------------------------------------------------------------
# attribute_kernels: Chrome-trace events -> kernel launches per phase.
# --------------------------------------------------------------------------
def _phase(name: str, ts: int, dur: int, tid: int = 7) -> dict:
    return {"ph": "X", "cat": "user_annotation", "name": f"phase:{name}", "pid": 1, "tid": tid,
            "ts": ts, "dur": dur}


def _launch(correlation: int, ts: int, tid: int = 7, cat: str = "cuda_runtime") -> dict:
    return {"ph": "X", "cat": cat, "name": "cudaLaunchKernel", "pid": 1, "tid": tid, "ts": ts,
            "dur": 2, "args": {"correlation": correlation}}


def _kernel(correlation: int, ts: int = 900) -> dict:
    return {"ph": "X", "cat": "kernel", "name": "gemm", "pid": 0, "tid": "stream 7", "ts": ts,
            "dur": 3, "args": {"correlation": correlation}}


def test_a_kernel_counts_for_the_phase_around_its_launch() -> None:
    from benchmarks.harness.b14_group_diag import attribute_kernels

    events = [
        _phase("restore", 100, 50), _phase("gemm", 200, 50),
        _launch(1, 110), _kernel(1), _launch(2, 210), _kernel(2), _launch(3, 220), _kernel(3),
    ]

    assert attribute_kernels(events) == {"restore": 1, "gemm": 2}


def test_the_innermost_phase_wins_when_phases_nest() -> None:
    from benchmarks.harness.b14_group_diag import attribute_kernels

    events = [
        _phase("forward", 0, 1000), _phase("gemm", 200, 50),
        _launch(1, 210), _kernel(1), _launch(2, 500), _kernel(2),
    ]

    assert attribute_kernels(events) == {"gemm": 1, "forward": 1}


def test_a_launch_outside_every_phase_is_unattributed_not_dropped() -> None:
    from benchmarks.harness.b14_group_diag import attribute_kernels

    events = [_phase("gemm", 200, 50), _launch(1, 10), _kernel(1), _launch(2, 210), _kernel(2)]

    assert attribute_kernels(events) == {"unattributed": 1, "gemm": 1}


def test_a_kernel_without_its_launch_event_is_unattributed() -> None:
    from benchmarks.harness.b14_group_diag import attribute_kernels

    assert attribute_kernels([_phase("gemm", 0, 100), _kernel(99)]) == {"unattributed": 1}


def test_other_annotations_do_not_claim_the_launch() -> None:
    from benchmarks.harness.b14_group_diag import attribute_kernels

    plain = {"ph": "X", "cat": "user_annotation", "name": "aten::mm", "pid": 1, "tid": 7,
             "ts": 100, "dur": 50}

    assert attribute_kernels([plain, _launch(1, 110), _kernel(1)]) == {"unattributed": 1}


def test_a_phase_of_the_launching_thread_beats_a_phase_of_another_thread() -> None:
    from benchmarks.harness.b14_group_diag import attribute_kernels

    # autograd runs the backward on its own thread, inside the main thread's range
    events = [
        _phase("bwd", 0, 1000, tid=1), _phase("restore_bwd", 100, 50, tid=9),
        _launch(1, 110, tid=9), _kernel(1),
    ]

    assert attribute_kernels(events) == {"restore_bwd": 1}


def test_a_phase_of_another_thread_claims_a_launch_no_phase_of_its_thread_covers() -> None:
    from benchmarks.harness.b14_group_diag import attribute_kernels

    events = [
        _phase("bwd", 0, 1000, tid=1), _phase("restore_bwd", 100, 50, tid=9),
        _launch(1, 500, tid=9), _kernel(1),
    ]

    assert attribute_kernels(events) == {"bwd": 1}


def test_the_driver_api_launch_counts_like_the_runtime_launch() -> None:
    from benchmarks.harness.b14_group_diag import attribute_kernels

    events = [_phase("gemm", 0, 100), _launch(1, 10, cat="cuda_driver"), _kernel(1)]

    assert attribute_kernels(events) == {"gemm": 1}


def test_the_phase_counts_add_up_to_every_kernel_event() -> None:
    from benchmarks.harness.b14_group_diag import attribute_kernels

    events = [_phase("a", 0, 100), _phase("b", 200, 100)]
    for number in range(1, 8):
        events += [_launch(number, 10 if number % 2 else 210), _kernel(number)]
    events.append(_kernel(500))

    counts = attribute_kernels(events)

    assert sum(counts.values()) == sum(1 for e in events if e["cat"] == "kernel") == 8


def _copy(correlation: int, ts: int = 950) -> dict:
    return {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy HtoD", "pid": 0, "tid": "stream 7",
            "ts": ts, "dur": 5, "args": {"correlation": correlation}}


def test_copies_are_attributed_to_the_phase_that_launched_them() -> None:
    from benchmarks.harness.b14_group_diag import attribute_memcpy

    events = [
        _phase("restore_fwd", 0, 100), _phase("gemm", 200, 100),
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaMemcpyAsync", "pid": 1, "tid": 7,
         "ts": 10, "dur": 2, "args": {"correlation": 5}},
        _copy(5), _launch(1, 210), _kernel(1),
    ]

    assert attribute_memcpy(events) == {"restore_fwd": 1}


def test_gpu_time_is_the_sum_of_kernel_durations_per_phase() -> None:
    from benchmarks.harness.b14_group_diag import attribute_kernel_time_us

    events = [_phase("gemm", 0, 100), _phase("act", 200, 100)]
    for number, at in ((1, 10), (2, 20), (3, 210)):
        events += [_launch(number, at), _kernel(number)]
    events[3]["dur"] = 40  # kernel 1 runs 40 us, the others 3 us
    events.append(_kernel(77))  # no launch event, 3 us, unattributed

    assert attribute_kernel_time_us(events) == {"gemm": 43, "act": 3, "unattributed": 3}


# --------------------------------------------------------------------------
# run_bucket on a tiny CPU layer: sequential, grouped restore, grouped GEMM (refused on CPU).
# --------------------------------------------------------------------------
@pytest.fixture()
def layer(tmp_path):
    from benchmarks.harness.b14_group_diag import build_layer

    built = build_layer(
        tmp_path, hidden=32, intermediate=16, experts=8, device="cpu", dtype="float32",
    )
    yield built
    built.close()


def test_the_two_cpu_arms_equal_the_resident_layer_bit_for_bit(layer) -> None:
    from benchmarks.harness.b14_group_diag import run_bucket

    record = run_bucket(layer, rows=4, arms=("sequential", "grouped_restore"), repeats=2)

    for arm in ("sequential", "grouped_restore"):
        assert record["arms"][arm]["status"] == "ok"
        assert record["arms"][arm]["bitwise_equal_to_resident"] == {
            "output": True, "hidden_grad": True, "route_weight_grad": True,
        }


def test_the_record_names_the_bucket_and_the_visited_experts(layer) -> None:
    from benchmarks.harness.b14_group_diag import run_bucket

    record = run_bucket(layer, rows=4, arms=("sequential",), repeats=1)

    assert (record["rows"], record["experts"], record["tokens"]) == (4, 8, 32)


def test_each_expert_is_read_once_forward_and_once_backward(layer) -> None:
    from benchmarks.harness.b14_group_diag import run_bucket

    record = run_bucket(layer, rows=2, arms=("sequential", "grouped_restore"), repeats=1)

    assert record["arms"]["sequential"]["expert_fetches"] == 16
    assert record["arms"]["grouped_restore"]["expert_fetches"] == 16


def test_the_grouped_gemm_arm_is_refused_without_cuda_and_says_why(layer) -> None:
    from benchmarks.harness.b14_group_diag import run_bucket

    record = run_bucket(layer, rows=2, arms=("sequential", "grouped_gemm"), repeats=1)

    refused = record["arms"]["grouped_gemm"]
    assert refused["status"].startswith("refused") and "CUDA" in refused["status"]
    assert record["arms"]["sequential"]["status"] == "ok"


def test_without_cuda_there_are_no_kernel_counts_and_timing_is_recorded(layer) -> None:
    from benchmarks.harness.b14_group_diag import run_bucket

    record = run_bucket(layer, rows=2, arms=("sequential",), repeats=3)

    arm = record["arms"]["sequential"]
    assert arm["kernels"] is None
    assert arm["timing_ms"]["samples"] == 3 and arm["timing_ms"]["total_median"] > 0


def test_the_record_is_json_serialisable(layer) -> None:
    import json

    from benchmarks.harness.b14_group_diag import run_bucket

    record = run_bucket(layer, rows=1, arms=("sequential", "grouped_restore", "grouped_gemm"),
                        repeats=1)

    assert json.loads(json.dumps(record))["rows"] == 1
