"""Comparing two training runs: adapter digests per step, held-out NLL and a paired bootstrap."""

from __future__ import annotations

import math

import pytest

from benchmarks.harness import p72_train_compare as compare


def test_target_tokens_count_the_unmasked_labels_after_the_shift() -> None:
    labels = [[-100, -100, 5, 6, 2], [-100, 7, -100, -100, -100]]
    assert compare.target_tokens(labels) == 3 + 1       # the first label is never a target


def rows(digests: list[str]) -> list[dict]:
    return [{"step": i + 1, "adapter_sha256": d, "loss": 1.0} for i, d in enumerate(digests)]


def test_first_difference_names_the_first_step_whose_adapters_differ() -> None:
    same = compare.first_difference(rows(["a", "b", "c"]), rows(["a", "b", "c"]))
    assert same == {"steps_compared": 3, "equal": True, "first_difference_step": None}
    diff = compare.first_difference(rows(["a", "b", "c"]), rows(["a", "x", "y"]))
    assert diff == {"steps_compared": 3, "equal": False, "first_difference_step": 2}
    with pytest.raises(ValueError, match="different number"):
        compare.first_difference(rows(["a"]), rows(["a", "b"]))


def example(nll_sum: float, tokens: int, kind: str = "GSM_x") -> dict:
    return {"nll_sum": nll_sum, "target_tokens": tokens, "type": kind,
            "nll_token_mean": nll_sum / tokens}


def test_nll_summary_gives_the_example_mean_and_the_token_weighted_value() -> None:
    out = compare.nll_summary([example(2.0, 10), example(9.0, 30)])
    assert out["mean_per_example_nll"] == pytest.approx((0.2 + 0.3) / 2)
    assert out["token_weighted_nll"] == pytest.approx(11.0 / 40)


def test_paired_bootstrap_of_identical_values_is_exactly_zero() -> None:
    values = [0.1 * i for i in range(1, 9)]
    out = compare.paired_bootstrap_delta(values, list(values), seed=1, draws=500)
    assert out["mean_delta"] == 0.0 and out["ci95"] == [0.0, 0.0]


def test_paired_bootstrap_interval_contains_the_mean_difference() -> None:
    a = [0.30, 0.31, 0.29, 0.28, 0.33, 0.30, 0.27, 0.32]
    b = [x - 0.002 + 0.001 * (i % 3) for i, x in enumerate(a)]
    out = compare.paired_bootstrap_delta(a, b, seed=3, draws=2000)
    low, high = out["ci95"]
    assert low <= out["mean_delta"] <= high and not math.isnan(low)
    with pytest.raises(ValueError):
        compare.paired_bootstrap_delta([0.1], [0.1, 0.2], seed=1, draws=10)
