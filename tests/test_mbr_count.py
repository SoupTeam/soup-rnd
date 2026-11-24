"""Ticket 15: the routed-expert ``F.linear`` counter, over fake tensors and fake frames (no torch).

The counter replaces ``torch.nn.functional.linear`` for one run: a call from ``expert_mlp`` with
one of the two routed-expert weights is counted with its row count and its phase. The summarizers
turn the per-step snapshots into the call median, the merged M histogram, and the seconds model
that prices every call with the measured per-M table.
"""

from __future__ import annotations

from math import prod
from typing import Any, List

from benchmarks.harness import mbr_count as mc
from benchmarks.harness import mbr_trace as mt


class FakeTensor:
    """Only the two tensor reads the wrapper makes: the shape and the element count."""

    def __init__(self, shape: tuple) -> None:
        self.shape = tuple(shape)

    def numel(self) -> int:
        return prod(self.shape)


def test_the_expert_shapes_are_the_traced_ones() -> None:
    assert mc.EXPERT_ROWS == mt.EXPERT_WEIGHT_SHAPES


def test_a_call_from_expert_mlp_with_an_expert_weight_counts_the_rows() -> None:
    counter = mc.ExpertCounter()
    linear = counter.wrap(lambda input, weight, bias=None: "out")

    def expert_mlp() -> Any:
        return linear(FakeTensor((8, 2048)), FakeTensor((2816, 2048)))

    assert expert_mlp() == "out"
    assert counter.snapshot() == {"calls": 1, "rows": {8: 1}, "phases": {"?": 1}}


def test_a_three_dimensional_input_counts_its_rows_without_the_feature_axis() -> None:
    counter = mc.ExpertCounter()
    linear = counter.wrap(lambda input, weight, bias=None: None)

    def expert_mlp() -> None:
        linear(FakeTensor((1, 15, 2048)), FakeTensor((2048, 1408)))

    expert_mlp()
    assert counter.snapshot()["rows"] == {15: 1}


def test_a_weight_that_is_not_a_routed_expert_is_not_recorded() -> None:
    counter = mc.ExpertCounter()
    calls: List[Any] = []

    def original(*args: Any) -> str:
        calls.append(args)
        return "out"

    linear = counter.wrap(original)

    def expert_mlp() -> Any:
        return linear(FakeTensor((8, 2048)), FakeTensor((5632, 2048)))

    assert expert_mlp() == "out"
    assert counter.snapshot() == {"calls": 0, "rows": {}, "phases": {}}
    assert len(calls) == 1


def test_a_call_outside_expert_mlp_is_not_counted() -> None:
    counter = mc.ExpertCounter()
    linear = counter.wrap(lambda input, weight, bias=None: None)
    linear(FakeTensor((8, 2048)), FakeTensor((2816, 2048)))
    assert counter.snapshot()["calls"] == 0


def test_the_phase_is_the_nearest_enclosing_forward_or_backward_frame() -> None:
    counter = mc.ExpertCounter()
    linear = counter.wrap(lambda input, weight, bias=None: None)

    def expert_mlp() -> None:
        linear(FakeTensor((8, 2048)), FakeTensor((2048, 1408)))

    def forward() -> None:
        expert_mlp()

    def backward() -> None:
        expert_mlp()

    forward()
    backward()
    assert counter.snapshot()["phases"] == {"forward": 1, "backward": 1}


def test_the_frame_walk_stops_six_frames_above_the_call() -> None:
    """``expert_mlp`` sits seven frames up here, past the walk, so the call is not counted."""
    counter = mc.ExpertCounter()
    linear = counter.wrap(lambda input, weight, bias=None: None)

    def step_6() -> None:
        linear(FakeTensor((8, 2048)), FakeTensor((2816, 2048)))

    def step_5() -> None:
        step_6()

    def step_4() -> None:
        step_5()

    def step_3() -> None:
        step_4()

    def step_2() -> None:
        step_3()

    def step_1() -> None:
        step_2()

    def expert_mlp() -> None:
        step_1()

    expert_mlp()
    assert counter.snapshot() == {"calls": 0, "rows": {}, "phases": {}}


def test_the_snapshot_sorts_the_rows_and_reset_clears_everything() -> None:
    counter = mc.ExpertCounter()
    counter.record(rows=16, projection="gate_up", phase="forward")
    counter.record(rows=4, projection="down", phase="backward")
    counter.record(rows=16, projection="down", phase="forward")
    snapshot = counter.snapshot()
    assert snapshot == {"calls": 3, "rows": {4: 1, 16: 2},
                        "phases": {"forward": 2, "backward": 1}}
    assert list(snapshot["rows"]) == [4, 16]
    counter.reset()
    assert counter.snapshot() == {"calls": 0, "rows": {}, "phases": {}}


def test_the_call_median_of_the_steps() -> None:
    assert mc.median_calls([{"calls": 6}, {"calls": 8}, {"calls": 7}]) == 7
    assert mc.median_calls([]) == 0


def test_the_merged_rows_sum_the_steps_ascending() -> None:
    steps = [{"rows": {4: 1, 8: 2}}, {"rows": {4: 3}}]
    merged = mc.merged_rows(steps)
    assert merged == {4: 4, 8: 2}
    assert list(merged) == [4, 8]
    assert mc.merged_rows([]) == {}


def test_expert_seconds_prices_every_call_with_the_per_m_table() -> None:
    """100 calls of 8 rows at 40 µs and 50 calls of 16 rows at 90 µs: 0.0085 s."""
    seconds = mc.expert_seconds({8: 100, 16: 50}, {8: 40.0, 16: 90.0})
    assert seconds == 0.0085


def test_a_row_count_without_a_price_gives_no_seconds() -> None:
    assert mc.expert_seconds({24: 4}, {8: 40.0}) is None
    assert mc.expert_seconds({}, {8: 40.0}) == 0.0


def test_merged_calls_of_several_steps_are_divided_by_the_step_count() -> None:
    """The histogram of three steps priced once is a per-step number, not three steps of it."""
    merged = {8: 300}
    assert mc.expert_seconds(merged, {8: 40.0}, steps=3) == 0.004
    assert mc.expert_seconds(merged, {8: 40.0}) == 0.012
