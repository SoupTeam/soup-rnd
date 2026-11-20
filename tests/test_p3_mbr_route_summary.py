"""The route report distinguishes slot permutations from changed expert sets."""

import torch

from benchmarks.harness.p3_route_preserving_mbr_soup_bridge import compare_route_artifacts


def test_route_summary_distinguishes_permutation_and_set_change():
    left = {"1": [{"ids": torch.tensor([[0, 1], [2, 3]]), "weights": torch.ones(2, 2)}]}
    right = {"1": [{"ids": torch.tensor([[1, 0], [2, 4]]), "weights": torch.ones(2, 2)}]}
    report = compare_route_artifacts(left, right)
    row = report["layers"][0]
    assert row["slot_mismatch_fraction"] == 0.75
    assert row["token_mismatch_fraction"] == 1.0
    assert row["token_expert_set_mismatch_fraction"] == 0.5
    assert report["token_expert_set_mismatch_fraction"] == 0.5
