"""Per-example NLL and paired percentile bootstrap for the Moonlight screen."""

from __future__ import annotations

import math
import random
from statistics import mean, quantiles


def _make_batch(example, device):
    import torch

    ids = torch.tensor([example["input_ids"]], dtype=torch.long, device=device)
    labels = torch.tensor([example["labels"]], dtype=torch.long, device=device)
    if ids.shape != labels.shape:
        raise ValueError("input and label lengths differ")
    count = sum(token != -100 for token in example["labels"][1:])
    if count < 1:
        raise ValueError("example has no shifted target")
    return ids, labels, count


def _bootstrap_mean_delta(control, candidate):
    if not control or len(control) != len(candidate):
        raise ValueError("expected nonempty paired samples of equal length")
    deltas = [b - a for a, b in zip(control, candidate)]
    if not all(map(math.isfinite, deltas)):
        raise ValueError("non-finite paired NLL difference")
    seed, replicates = 20261118, 10_000
    rng = random.Random(seed)
    n = len(deltas)
    draws = [sum(rng.choices(deltas, k=n)) / n for _ in range(replicates)]
    percentiles = quantiles(draws, n=1000, method="inclusive")
    return {
        "mean_delta": mean(deltas), "ci95": [percentiles[24], percentiles[974]],
        "sign": "candidate minus control", "method": "paired per-example percentile bootstrap",
        "n_pairs": n, "seed": seed, "replicates": replicates,
    }
