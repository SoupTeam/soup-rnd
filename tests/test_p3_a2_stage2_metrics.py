"""CPU checks for Stage 2 comparisons (no checkpoint or accelerator)."""

import pytest


def test_bitwise_gate_distinguishes_negative_zero_and_missing_parameters():
    import torch

    from benchmarks.harness.p3_route_preserving_mbr_soup_bridge import bitwise_tensors

    a = {"w": torch.tensor(-0.0, dtype=torch.bfloat16)}
    assert bitwise_tensors(a, a, {"w"})["all_equal"]
    assert not bitwise_tensors(a, {"w": -a["w"]}, {"w"})["all_equal"]
    assert not bitwise_tensors(a, a, {"w", "missing"})["all_equal"]
    assert not bitwise_tensors(a, {"w": a["w"].float()}, {"w"})["all_equal"]
    assert not bitwise_tensors(a, {"w": a["w"].view(1)}, {"w"})["all_equal"]


def test_bootstrap_resamples_pairs_not_arms_and_reports_sign():
    from benchmarks.harness.p3_a2_quality_common import _bootstrap_mean_delta

    # Independent arm resampling would produce a wide interval for these three pairs.
    paired = _bootstrap_mean_delta([1, 100, 2], [1.5, 100.5, 2.5])
    assert paired["mean_delta"] == 0.5
    assert paired["ci95"] == [0.5, 0.5]
    assert paired["replicates"] > 100
    with pytest.raises(ValueError):
        _bootstrap_mean_delta([0], [1, 2])


def test_batch_counts_shifted_unmasked_targets():
    from benchmarks.harness.p3_a2_quality_common import _make_batch

    ids, labels, count = _make_batch({"input_ids": [3, 4, 5], "labels": [3, -100, 5]}, "cpu")
    assert ids.tolist() == [[3, 4, 5]]
    assert labels.tolist() == [[3, -100, 5]]
    assert count == 1
