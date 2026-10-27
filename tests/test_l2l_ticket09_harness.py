"""CPU tests for the pure parts of the ticket-09 harness (benchmarks/harness)."""

from __future__ import annotations

from tests.test_l2l_probe_harness import _load


def test_pad_to_multiple():
    arm = _load("l2l_ticket09_arm")
    assert [arm.pad_to_multiple(n, 64) for n in (1, 64, 65, 230)] == [64, 64, 128, 256]
    assert arm.pad_to_multiple(77, 0) == 77


def test_window_summary_counts_shapes_not_seen_before():
    arm = _load("l2l_ticket09_arm")
    seen = {(1, 5)}
    events = [{"layer_loads": 32, "seconds": 0.5}, {"layer_loads": 32, "seconds": 0.25}]
    out = arm.window_summary([(1, 5), (1, 6), (1, 6), (1, 7)], seen, events, 190)
    assert out["new_shapes"] == 2 and out["captures"] == 2
    assert out["capture_layer_loads"] == 64 and out["capture_seconds"] == 0.75
    assert seen == {(1, 5), (1, 6), (1, 7)}


def test_merge_windows_skips_step_zero_in_the_means():
    arm = _load("l2l_ticket09_arm")
    record = {"steps": [{}, {}, {}], "run": {"steps": 2}}
    windows = [
        {"new_shapes": 4, "captures": 4, "capture_layer_loads": 128, "capture_seconds": 4.0},
        {"new_shapes": 2, "captures": 2, "capture_layer_loads": 64, "capture_seconds": 2.0},
        {"new_shapes": 4, "captures": 4, "capture_layer_loads": 128, "capture_seconds": 6.0},
    ]
    arm.merge_windows(record, windows)
    assert record["steps"][2]["l2l_window"] is windows[2]
    assert record["run"]["new_shapes_per_step"] == 3
    assert record["run"]["capture_seconds_per_step"] == 4.0


def _step(i, new, loads):
    return {"step": i, "layer_loads": loads,
            "l2l_window": {"new_shapes": new, "captures": new}}


def test_consistency_agrees_when_loads_are_base_plus_32_per_capture():
    camp = _load("l2l_ticket09_campaign")
    steps = [_step(0, 4, 190)] + [_step(1, 2, 126), _step(2, 4, 190), _step(3, 3, 158)]
    out = camp.consistency(steps)
    assert out["agrees"] and out["residuals"] == [62]
    assert out["captures_equal_new_shapes"]


def test_consistency_disagrees_when_the_cause_is_not_the_capture():
    camp = _load("l2l_ticket09_campaign")
    out = camp.consistency([_step(0, 0, 62), _step(1, 2, 100), _step(2, 4, 190)])
    assert not out["agrees"]
