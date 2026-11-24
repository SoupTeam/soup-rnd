"""Ticket 15: summaries of a torch profiler step, over fake ``FunctionEvent`` objects (no torch).

The routed experts of Moonlight-16B-A3B are selected by their weight shape, so the tests pin the
selection, the split-K kernel name, the histogram, the kernel table, the device timeline of the
traced step and the whole-step summary.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

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
    weight = mt.EXPERT_WEIGHT_SHAPES[projection]
    return FakeEvent("aten::mm", input_shapes=((rows, weight[1]), weight), kernels=kernels)


def test_the_weight_shapes_are_the_two_routed_expert_projections() -> None:
    assert mt.EXPERT_WEIGHT_SHAPES == {"gate_up": (2816, 2048), "down": (2048, 1408)}


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
        # shared expert: (5632, 2048), attention: (2048, 2048)
        FakeEvent("aten::mm", input_shapes=((16, 2048), (5632, 2048)),
                  kernels=[FakeKernel("shared", 10.0)]),
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
