"""Ticket 15: summaries of a torch profiler step, over fake ``FunctionEvent`` objects (no torch).

The routed experts of Moonlight-16B-A3B are selected by their weight shape, so the tests pin the
selection, the split-K kernel name, the histogram, the kernel table, the device timeline of the
traced step and the whole-step summary.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import pytest

from benchmarks.harness import mbr_trace as mt


class FakeKernel:
    """A kernel of a ``FunctionEvent``: a name and a duration in microseconds."""

    def __init__(self, name: str, duration_us: float) -> None:
        self.name = name
        self._duration_us = duration_us

    def duration_time(self) -> float:
        return self._duration_us


class FakeRange:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class FakeEvent:
    """Duck-typed ``FunctionEvent``: the fields the summarizers read, nothing else."""

    def __init__(self, name: str, *, device_type: str = "CPU",
                 input_shapes: Optional[Sequence[Sequence[int]]] = None,
                 kernels: Sequence[FakeKernel] = (), start: float = 0.0, end: float = 0.0,
                 self_cpu: float = 0.0) -> None:
        self.name = name
        self.device_type = device_type
        self.input_shapes = list(input_shapes or [])
        self.kernels = list(kernels)
        self.time_range = FakeRange(start, end)
        self.self_cpu_time_total = self_cpu


def expert_mm(rows: int, projection: str, *, kernels: Sequence[FakeKernel]) -> FakeEvent:
    """One routed-expert ``aten::mm``: ``F.linear`` hands it the weight transposed, ``(K, N)``."""
    weight = mt.EXPERT_WEIGHT_SHAPES[projection]
    return FakeEvent("aten::mm", input_shapes=((rows, weight[1]), weight[::-1]), kernels=kernels)


def test_the_weight_shapes_are_the_two_routed_expert_projections() -> None:
    assert mt.EXPERT_WEIGHT_SHAPES == {"gate_up": (2816, 2048), "down": (2048, 1408)}


def test_the_selection_reads_the_transposed_mm_shape_not_the_linear_weight_shape() -> None:
    """``aten::mm`` gets ``(K, N)``: the linear orientation must not select, the mm one must."""
    linear_orientation = FakeEvent(
        "aten::mm", input_shapes=((16, 2048), (2816, 2048)), kernels=[FakeKernel("k", 10.0)])
    assert mt.expert_calls([linear_orientation]) == []
    assert len(mt.expert_calls([expert_mm(16, "gate_up", kernels=[FakeKernel("k", 10.0)])])) == 1


def test_a_three_dimensional_input_counted_its_rows_without_the_feature_axis() -> None:
    call = mt.expert_calls([FakeEvent(
        "aten::mm", input_shapes=((1, 15, 2048), (2048, 2816)),
        kernels=[FakeKernel("k", 10.0)])])[0]
    assert call["rows"] == 15


def test_one_entry_per_routed_expert_gemm_call() -> None:
    events: List[Any] = [
        expert_mm(16, "gate_up", kernels=[FakeKernel("cutlass::gemm", 120.0)]),
        expert_mm(8, "down", kernels=[FakeKernel("cutlass::gemm", 40.0)]),
    ]
    calls = mt.expert_calls(events)
    assert calls == [
        {"rows": 16, "projection": "gate_up", "kernel": "cutlass::gemm", "seconds": 120e-6,
         "kernels": 1},
        {"rows": 8, "projection": "down", "kernel": "cutlass::gemm", "seconds": 40e-6,
         "kernels": 1},
    ]


def test_a_split_k_call_names_both_kernels_and_sums_their_time() -> None:
    call = mt.expert_calls([
        expert_mm(8, "down", kernels=[FakeKernel("k_a", 50.0), FakeKernel("k_b", 70.0)]),
    ])[0]
    assert call["kernel"] == "k_a | k_b"
    assert call["kernels"] == 2
    assert call["seconds"] == 120e-6


def test_the_shared_expert_attention_and_empty_kernel_lists_are_not_routed_experts() -> None:
    events = [
        # shared expert gate_up (5632, 2048) and down (2048, 2816): the mm shapes are (2048, 5632)
        # and (2816, 2048), the second the mirror of a routed gate_up, so orientation matters
        FakeEvent("aten::mm", input_shapes=((16, 2048), (2048, 5632)),
                  kernels=[FakeKernel("shared", 10.0)]),
        FakeEvent("aten::mm", input_shapes=((16, 2816), (2816, 2048)),
                  kernels=[FakeKernel("shared_down", 10.0)]),
        FakeEvent("aten::mm", input_shapes=((16, 2048), (2048, 2048)),
                  kernels=[FakeKernel("attention", 10.0)]),
        # the right weight, but the call has no kernels recorded (a CPU-only trace)
        expert_mm(16, "gate_up", kernels=[]),
        # not an aten::mm, and a shape list that cannot give rows and weight
        FakeEvent("aten::addmm", input_shapes=((16, 2048), (2816, 2048)),
                  kernels=[FakeKernel("x", 1.0)]),
        FakeEvent("aten::mm", input_shapes=((16, 2048),), kernels=[FakeKernel("y", 1.0)]),
    ]
    assert mt.expert_calls(events) == []


def test_the_row_histogram_counts_calls_per_row_count_ascending() -> None:
    calls = [{"rows": rows, "projection": "down", "kernel": "k", "seconds": 1e-6, "kernels": 1}
             for rows in (16, 16, 4, 8, 16)]
    assert mt.rows_histogram(calls) == {4: 1, 8: 1, 16: 3}
    assert list(mt.rows_histogram(calls)) == [4, 8, 16]


def test_the_kernel_table_groups_by_kernel_string_with_its_row_range() -> None:
    table = mt.kernel_table([
        {"rows": 16, "projection": "gate_up", "kernel": "k_a", "seconds": 1.5e-5, "kernels": 1},
        {"rows": 4, "projection": "down", "kernel": "k_a", "seconds": 0.5e-5, "kernels": 1},
        {"rows": 8, "projection": "down", "kernel": "k_a | k_b", "seconds": 2e-5,
         "kernels": 2},
    ])
    assert table["k_a"] == {"calls": 2, "seconds": 2e-5, "rows_min": 4, "rows_max": 16}
    assert table["k_a | k_b"] == {"calls": 1, "seconds": 2e-5, "rows_min": 8, "rows_max": 8}


def test_the_timeline_counts_copies_apart_and_overlaps_once() -> None:
    """k_a and k_b overlap (0-10 and 2-5), a memcpy follows at 10-12, a CPU op is ignored.

    Busy is the union 0-12 = 12 us, not the 25 us sum of the intervals. Copies are not kernel
    launches, and only kernels reach the top list.
    """
    events = [
        FakeEvent("k_a", device_type="CUDA", start=0.0, end=10.0),
        FakeEvent("k_b", device_type="CUDA", start=2.0, end=5.0),
        FakeEvent("Memcpy DtoH (Device -> Pageable)", device_type="CUDA", start=10.0, end=12.0),
        FakeEvent("aten::mm", device_type="CPU", start=0.0, end=1000.0),
    ]
    timeline = mt.kernel_timeline(events, wall_seconds=40e-6)
    assert timeline["kernels"] == 2
    assert timeline["copies"] == 1
    assert timeline["kernel_seconds"] == pytest.approx(13e-6)
    assert timeline["busy_seconds"] == pytest.approx(12e-6)
    assert timeline["span_seconds"] == pytest.approx(12e-6)
    assert timeline["gap_seconds"] == pytest.approx(0.0)
    assert timeline["exposed_seconds"] == pytest.approx(28e-6)
    assert timeline["top"] == [{"name": "k_a", "calls": 1, "seconds": pytest.approx(10e-6)},
                               {"name": "k_b", "calls": 1, "seconds": pytest.approx(3e-6)}]


def test_exposed_time_is_never_negative_when_the_wall_is_below_busy() -> None:
    events = [FakeEvent("k_a", device_type="CUDA", start=0.0, end=20.0)]
    timeline = mt.kernel_timeline(events, wall_seconds=5e-6)
    assert timeline["exposed_seconds"] == pytest.approx(0.0)
    assert timeline["gap_seconds"] == 0.0


def test_the_timeline_of_an_empty_trace_is_zeroed() -> None:
    timeline = mt.kernel_timeline([], wall_seconds=3.0)
    assert timeline == {"kernels": 0, "copies": 0, "kernel_seconds": 0.0, "busy_seconds": 0.0,
                        "span_seconds": 0.0, "gap_seconds": 0.0, "exposed_seconds": 3.0,
                        "top": []}


def test_a_step_summary_carries_the_counters_the_timeline_and_the_cpu_self_time() -> None:
    events: List[Any] = [
        expert_mm(16, "gate_up", kernels=[FakeKernel("cutlass", 100.0)]),
        expert_mm(8, "down", kernels=[FakeKernel("cutlass", 30.0)]),
        FakeEvent("aten::mm", device_type="CPU", self_cpu=2500.0, start=0.0, end=2000.0),
        FakeEvent("cutlass", device_type="CUDA", start=0.0, end=130.0),
    ]
    summary = mt.step_summary(events, wall_seconds=0.5)
    assert summary["wall_seconds"] == 0.5
    assert summary["cpu_self_seconds"] == pytest.approx(2500.0 / 1e6)
    gemm = summary["expert_gemm"]
    assert gemm["calls"] == 2
    assert gemm["rows_total"] == 24
    assert gemm["seconds"] == pytest.approx(130e-6)
    assert gemm["rows"] == {8: 1, 16: 1}
    assert gemm["kernels"] == {"cutlass": {"calls": 2, "seconds": pytest.approx(130e-6),
                                           "rows_min": 8, "rows_max": 16}}
    assert summary["timeline"]["kernels"] == 1
    assert summary["timeline"]["busy_seconds"] == pytest.approx(130e-6)


def test_a_step_without_expert_calls_has_zero_totals_and_empty_tables() -> None:
    summary = mt.step_summary([], wall_seconds=10.0)
    assert summary["cpu_self_seconds"] == 0.0
    assert summary["expert_gemm"] == {"calls": 0, "rows_total": 0, "seconds": 0.0,
                                      "rows": {}, "kernels": {}, "by_rows": {}}
    assert summary["timeline"]["kernels"] == 0 and summary["timeline"]["top"] == []


def test_the_row_table_gives_the_seconds_and_microseconds_per_row_of_each_m() -> None:
    """The M efficiency: two 16-row calls of 1.5e-5 s and 0.5e-5 s, one 4-row call of 0.5e-5 s."""
    table = mt.rows_table([
        {"rows": 16, "projection": "gate_up", "kernel": "k", "seconds": 1.5e-5, "kernels": 1},
        {"rows": 16, "projection": "down", "kernel": "k", "seconds": 0.5e-5, "kernels": 1},
        {"rows": 4, "projection": "down", "kernel": "k", "seconds": 0.5e-5, "kernels": 1},
    ])
    assert list(table) == [4, 16]
    assert table[16]["calls"] == 2
    assert table[16]["seconds"] == pytest.approx(20e-6)
    assert table[16]["us_per_row"] == pytest.approx(1.25)
    assert table[4]["us_per_row"] == pytest.approx(1.25)
