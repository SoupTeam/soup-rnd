"""CPU tests for the A5 ticket 04 campaign: the overflow size, the metric table, the verdicts."""

from __future__ import annotations

import pytest

from tests.test_l2l_probe_harness import _load


@pytest.fixture()
def campaign():
    _load("l2l_ticket04_arm")
    return _load("l2l_ticket04_campaign")


def test_overflow_store_is_above_the_trainers_limit(campaign):
    free = 100_000_000_000
    k = campaign.overflow_k(free, batch=4, seq=256)
    chunk = campaign.next_power_of_two(4 * 256 * 4096 * 2)
    assert campaign.LAYERS * k * chunk > free / 0.7 * 0.9 > free * 0.7


def test_next_power_of_two(campaign):
    assert [campaign.next_power_of_two(v) for v in (1, 2, 3, 17, 1024)] == [1, 2, 4, 32, 1024]


def _entry(arm, value):
    return {"arm": arm, "measurement": {"v": value}}


def _block(a1, b1, b2, a2):
    return [_entry("A", a1), _entry("B", b1), _entry("B", b2), _entry("A", a2)]


def test_metric_summary_means_and_difference(campaign):
    blocks = [_block(10, 6, 6, 10), _block(12, 8, 8, 12)]
    out = campaign.metric_summary(blocks, lambda m: m["v"])
    assert out["a"] == 11 and out["b"] == 7 and out["a_minus_b"] == 4
    assert out["interval_95"] == [4.0, 4.0]


def test_metric_summary_is_null_when_a_run_lacks_the_value(campaign):
    blocks = [_block(10, None, 6, 10), _block(12, 8, 8, 12)]
    out = campaign.metric_summary(blocks, lambda m: m["v"])
    assert out["a_minus_b"] is None and out["reason"]


def test_loss_verdict_passes_when_l2l_equals_ga_and_floor_is_finite(campaign):
    curves = {"ga": [[2.0, 1.0], [2.0, 1.0]], "l2l": [[2.0, 1.0]], "floor": [[2.0, 1.001]]}
    out = campaign.loss_verdicts(curves)
    assert out["pass"] and out["ga_repeat_error"] == [[0.0, 0.0]]


def test_loss_verdict_fails_when_one_l2l_curve_leaves_the_tolerance(campaign):
    curves = {"ga": [[2.0, 1.0]], "l2l": [[2.0, 1.0], [2.0, 1.01]], "floor": [[2.0, 1.001]]}
    assert not campaign.loss_verdicts(curves)["pass"]


def test_loss_verdict_without_floor_is_not_a_pass(campaign):
    out = campaign.loss_verdicts({"ga": [[2.0]], "l2l": [[2.0]], "floor": []})
    assert not out["pass"]
