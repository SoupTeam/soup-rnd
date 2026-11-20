"""Comparing two training runs (harness only, pure Python): adapter digests, held-out NLL."""

from __future__ import annotations

import random
import statistics
from typing import Any, Dict, List, Sequence


def target_tokens(labels: Sequence[Sequence[int]]) -> int:
    """Unmasked labels after the causal shift (the first label is never predicted), over a window:
    HF's ``num_items_in_batch`` for masked labels."""
    return sum(sum(1 for x in row[1:] if x != -100) for row in labels)


def first_difference(a: Sequence[Dict[str, Any]], b: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Whether two runs have the same adapter digest at every step, and where they first differ."""
    if len(a) != len(b):
        raise ValueError(f"runs have a different number of steps: {len(a)} and {len(b)}")
    for left, right in zip(a, b):
        if left["adapter_sha256"] != right["adapter_sha256"]:
            return {"steps_compared": len(a), "equal": False,
                    "first_difference_step": int(left["step"])}
    return {"steps_compared": len(a), "equal": True, "first_difference_step": None}


def nll_summary(per_example: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    return {
        "mean_per_example_nll": sum(x["nll_token_mean"] for x in per_example) / len(per_example),
        "token_weighted_nll": sum(x["nll_sum"] for x in per_example)
        / sum(x["target_tokens"] for x in per_example),
    }


def paired_bootstrap_delta(a: Sequence[float], b: Sequence[float], *, seed: int,
                           draws: int = 10000) -> Dict[str, Any]:
    """Mean of ``b - a`` and a 95% percentile interval from paired resampling of the examples."""
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must have the same non-zero length")
    deltas = [y - x for x, y in zip(a, b)]
    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(draws):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        means.append(statistics.fmean(sample))
    means.sort()
    low = means[int(0.025 * (draws - 1))]
    high = means[int(0.975 * (draws - 1))]
    return {"n": len(deltas), "mean_delta": statistics.fmean(deltas), "ci95": [low, high],
            "draws": draws, "seed": seed}
