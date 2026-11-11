"""Conservative paired Stage 2 quality bounds and analytical precision planning.

Run inside the parent's Kaggle CPU verification job, from the checkout root:
    python -m benchmarks.harness.p3_stage2_statistics --out planning.json \
        --n 101 437 1299 5000 10000

No model, dataset, generated program, training job or Kaggle submission runs here.
The default one-percentage-point margin applies to every required cell, not to a
pooled score. Wider-task statements condition on the fixed trained model pairs
and on the independent, identically distributed task-pair model in the protocol.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from numbers import Real
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

_DEFAULT_SAMPLE_SIZES = (101, 437, 1299, 5000, 10000)
_BALANCED_DISCORDANCE_RATES = ((1, 100), (1, 20), (1, 10), (1, 4), (1, 2))
_REQUIRED_FIELDS = (
    "benchmark", "length", "seed", "task_id", "baseline_pass", "candidate_pass"
)


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python int, not a bool or coerced number")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _probability(value: Any, name: str, *, maximum: float = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number, not a bool")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result) or not 0.0 < result < maximum:
        raise ValueError(f"{name} must be finite and strictly between 0 and {maximum}")
    return result


def _alpha(value: Any) -> float:
    result = _probability(value, "one_sided_alpha", maximum=0.5)
    if result / 2 == 0:
        raise ValueError("one_sided_alpha is too small to represent its marginal tail")
    return result


def _sequence(values: Any, name: str) -> Sequence:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a non-string sequence")
    if not values:
        raise ValueError(f"{name} must not be empty")
    return values


def _distinct_integers(values: Any, name: str, *, minimum: int) -> list[int]:
    result = [_integer(value, name, minimum=minimum) for value in _sequence(values, name)]
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a nonempty string")
    if not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _scipy_beta() -> tuple[Any, str]:
    # SciPy is a planning/evidence dependency, never an eager CLI/training import.
    import scipy
    from scipy.stats import beta

    return beta, scipy.__version__


def _zero_upper(n: int, tail: float) -> float:
    # Avoid cancellation in 1 - tail**(1/n), including large zero-event cells.
    return -math.expm1(math.log(tail) / n)


def _cp_bounds(count: int, n: int, tail: float, beta: Any) -> tuple[float, float]:
    if count == 0:
        return 0.0, _zero_upper(n, tail)
    if count == n:
        return math.exp(math.log(tail) / n), 1.0
    lower = float(beta.ppf(tail, count, n - count + 1))
    # isf(tail) is the upper quantile without rounding 1 - tail to one.
    upper = float(beta.isf(tail, count + 1, n - count))
    if not (math.isfinite(lower) and math.isfinite(upper) and 0 <= lower <= upper <= 1):
        raise ValueError("SciPy returned invalid Clopper-Pearson quantiles")
    return lower, upper


def paired_bounds(
    harms: int, gains: int, n: int, *, one_sided_alpha: float = 0.025
) -> dict:
    """Bound P(BF16-only pass) - P(candidate-only pass) for one paired cell.

    Each difference bound has error at most ``one_sided_alpha``. The pair of
    bounds has coverage at least ``1 - 2 * one_sided_alpha`` under the task model.
    The marginal binomial counts need not be independent. Counts include every
    task pair, including both-fail pairs. Zero discordances do not imply certainty.
    """
    n = _integer(n, "n", minimum=1)
    harms = _integer(harms, "harms")
    gains = _integer(gains, "gains")
    if harms + gains > n:
        raise ValueError("harms + gains must not exceed n")
    alpha = _alpha(one_sided_alpha)
    try:
        finite_n = math.isfinite(float(n))
    except OverflowError as error:
        raise ValueError("n cannot be represented as a finite SciPy shape") from error
    if not finite_n:
        raise ValueError("n cannot be represented as a finite SciPy shape")
    beta, scipy_version = _scipy_beta()
    tail = alpha / 2
    harm_lower, harm_upper = _cp_bounds(harms, n, tail, beta)
    gain_lower, gain_upper = _cp_bounds(gains, n, tail, beta)
    return {
        "harms": harms,
        "gains": gains,
        "n": n,
        "discordant": harms + gains,
        "concordant": n - harms - gains,
        "degradation": (harms - gains) / n,
        "lower": harm_lower - gain_upper,
        "upper": harm_upper - gain_lower,
        "one_sided_alpha": alpha,
        "marginal_tail_alpha": tail,
        "one_sided_coverage_at_least": 1 - alpha,
        "two_sided_coverage_at_least": 1 - 2 * alpha,
        "method": "Clopper-Pearson marginal bounds plus union bound",
        "scipy_version": scipy_version,
    }


def minimum_zero_discordance_n(
    margin: float = 0.01, *, one_sided_alpha: float = 0.025
) -> int:
    """Smallest n for a strict U < margin when harms == gains == 0.

    This is an outcome-specific analytical precision threshold, not a power
    calculation or a promise that n tasks will produce zero discordances.
    """
    margin = _probability(margin, "margin")
    alpha = _alpha(one_sided_alpha)
    tail = alpha / 2
    threshold = math.log(tail) / math.log1p(-margin)
    if not math.isfinite(threshold):
        raise ValueError("margin is too small to represent its sample-size threshold")
    upper = max(1, math.floor(threshold) + 1)
    # Binary search also handles floating-point plateaus at very large n without
    # walking one integer at a time. Compare the reported stable bound strictly.
    while _zero_upper(upper, tail) >= margin:
        upper *= 2
    lower = 0
    while lower + 1 < upper:
        middle = (lower + upper) // 2
        if _zero_upper(middle, tail) < margin:
            upper = middle
        else:
            lower = middle
    return upper


def _scope() -> dict:
    return {
        "finite_set": "Exact paired scores on the complete frozen eligible task sets",
        "wider_tasks": "Conditional model-based uncertainty for benchmark-like populations",
        "task_model": (
            "Within each fixed benchmark/length/seed cell, task pairs are independent "
            "and identically distributed draws from its declared task population"
        ),
        "allowed_dependence": (
            "Arbitrary within-task arm dependence and arbitrary dependence across "
            "seeds, lengths and benchmarks; no independent seed-by-task-cell assumption"
        ),
        "eligibility": (
            "Eligibility, scorer, margin and required design must be frozen before outputs; "
            "all failures remain in the denominator"
        ),
        "coverage_is_conditional_on_assumptions": True,
        "future_seed_inference": False,
        "pooled_benchmark_inference": False,
        "seed_repetition_multiplies_independent_task_count": False,
        "selection_bias_or_contamination_repaired": False,
        "design_based_or_out_of_distribution_coverage": False,
    }


def precision_scenarios(
    sample_sizes: Sequence[int], *, margin: float = 0.01, one_sided_alpha: float = 0.025
) -> dict:
    """Deterministic hypothetical bounds at zero and balanced discordance counts.

    Rates are total discordance probabilities used only to choose example counts.
    Rounding is downward in each equal harm/gain count. These are neither measured
    model outcomes nor simulated rejection probabilities, and guarantee no power.
    """
    sizes = _distinct_integers(sample_sizes, "sample_sizes", minimum=1)
    margin = _probability(margin, "margin")
    alpha = _alpha(one_sided_alpha)
    scenarios = []
    for n in sizes:
        zero = paired_bounds(0, 0, n, one_sided_alpha=alpha)
        zero["noninferiority_bound_below_margin"] = zero["upper"] < margin
        balanced = []
        for numerator, denominator in _BALANCED_DISCORDANCE_RATES:
            count = n * numerator // (2 * denominator)
            bounds = paired_bounds(count, count, n, one_sided_alpha=alpha)
            balanced.append({
                **bounds,
                "target_total_discordance_rate": numerator / denominator,
                "hypothetical_total_discordance_rate": 2 * count / n,
                "noninferiority_bound_below_margin": bounds["upper"] < margin,
            })
        scenarios.append({"n": n, "zero_discordance": zero, "balanced_discordance": balanced})
    return {
        "schema": "p3-stage2-statistics-planning-v1",
        "kind": "ANALYTICAL_HYPOTHETICAL_NOT_MEASURED",
        "margin": margin,
        "units": "absolute probability; positive degradation means BF16 minus candidate",
        "one_sided_alpha": alpha,
        "minimum_zero_discordance_n": minimum_zero_discordance_n(
            margin, one_sided_alpha=alpha
        ),
        "minimum_n_scope": (
            "One cell and only the zero-discordance outcome; required cells must each "
            "pass their own bound. This is not an outcome-independent sample-size guarantee."
        ),
        "balanced_discordance_rounding": (
            "harms = gains = floor(n * target total discordance rate / 2); "
            "report the resulting hypothetical rate separately"
        ),
        "guaranteed_power": False,
        "power_estimate": None,
        "scipy_version": scenarios[0]["zero_discordance"]["scipy_version"],
        "scope": _scope(),
        "scenarios": scenarios,
    }


def _required_design(
    expected_tasks: dict[str, list[str]], seeds: Sequence[int], lengths: Sequence[int]
) -> tuple[dict[str, set[str]], list[int], list[int]]:
    if not isinstance(expected_tasks, dict):
        raise TypeError("expected_tasks must be a nonempty benchmark-to-task-list dict")
    if not expected_tasks:
        raise ValueError("expected_tasks must not be empty")
    tasks = {}
    for benchmark, task_ids in expected_tasks.items():
        benchmark = _identifier(benchmark, "expected benchmark")
        if not isinstance(task_ids, list):
            raise TypeError(f"expected_tasks[{benchmark!r}] must be a nonempty task-ID list")
        identifiers = [
            _identifier(task_id, "expected task_id")
            for task_id in _sequence(task_ids, f"expected_tasks[{benchmark!r}]")
        ]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"Duplicate expected task IDs for {benchmark!r}")
        tasks[benchmark] = set(identifiers)
    seed_ids = sorted(_distinct_integers(seeds, "seeds", minimum=0))
    length_ids = sorted(_distinct_integers(lengths, "lengths", minimum=1))
    return dict(sorted(tasks.items())), seed_ids, length_ids


def _cell_identity(benchmark: str, length: int, seed: int) -> dict:
    return {"benchmark": benchmark, "length": length, "seed": seed}


def summarize_pairs(
    rows: Sequence[dict],
    *,
    expected_tasks: dict[str, list[str]],
    seeds: Sequence[int],
    lengths: Sequence[int],
    margin: float = 0.01,
    one_sided_alpha: float = 0.025,
) -> dict:
    """Summarize a complete benchmark x length x fixed-seed paired study.

    Required task IDs apply independently to every length and seed of a benchmark.
    Invalid, duplicate, missing or unexpected pairs raise instead of allowing an
    admission decision. Metadata fields may accompany the six required row fields.
    """
    tasks, seed_ids, length_ids = _required_design(expected_tasks, seeds, lengths)
    margin = _probability(margin, "margin")
    alpha = _alpha(one_sided_alpha)
    if isinstance(rows, (str, bytes, bytearray)) or not isinstance(rows, Sequence):
        raise TypeError("rows must be a sequence of paired-result dicts")
    counts = {
        (benchmark, length, seed): {"n": 0, "harms": 0, "gains": 0, "both_pass": 0, "both_fail": 0}
        for benchmark in tasks
        for length in length_ids
        for seed in seed_ids
    }
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"rows[{index}] must be a paired-result dict")
        missing = [field for field in _REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"Missing required fields at rows[{index}]: {missing}")
        benchmark = _identifier(row["benchmark"], "benchmark")
        length = _integer(row["length"], "length", minimum=1)
        seed = _integer(row["seed"], "seed")
        task_id = _identifier(row["task_id"], "task_id")
        cell_key = (benchmark, length, seed)
        if cell_key not in counts or task_id not in tasks[benchmark]:
            raise ValueError(
                f"Unexpected paired-task coverage at rows[{index}]: {cell_key, task_id}"
            )
        for field in ("baseline_pass", "candidate_pass"):
            if type(row[field]) is not bool:
                raise TypeError(f"rows[{index}].{field} must be a Python bool")
        pair_key = (*cell_key, task_id)
        if pair_key in seen:
            raise ValueError(f"Duplicate paired task: {pair_key}")
        seen.add(pair_key)
        baseline, candidate = row["baseline_pass"], row["candidate_pass"]
        cell = counts[cell_key]
        cell["n"] += 1
        if baseline and not candidate:
            cell["harms"] += 1
        elif candidate and not baseline:
            cell["gains"] += 1
        elif baseline:
            cell["both_pass"] += 1
        else:
            cell["both_fail"] += 1
    incomplete = [
        {**_cell_identity(*key), "expected_n": len(tasks[key[0]]), "observed_n": cell["n"]}
        for key, cell in counts.items()
        if cell["n"] != len(tasks[key[0]])
    ]
    if incomplete:
        raise ValueError(f"Missing paired-task coverage: {incomplete[:8]}")

    comparison_count = len(counts)
    simultaneous_alpha = alpha / comparison_count
    _alpha(simultaneous_alpha)
    cells = []
    observed_breaches = []
    insufficient_precision = []
    for key, cell in counts.items():
        identity = _cell_identity(*key)
        n, harms, gains = cell["n"], cell["harms"], cell["gains"]
        pointwise = paired_bounds(harms, gains, n, one_sided_alpha=alpha)
        simultaneous = paired_bounds(harms, gains, n, one_sided_alpha=simultaneous_alpha)
        degradation = pointwise["degradation"]
        noninferiority = pointwise["upper"] < margin
        excessive = simultaneous["lower"] > margin
        breach = degradation >= margin
        result = {
            **identity,
            **cell,
            "concordant": cell["both_pass"] + cell["both_fail"],
            "baseline_accuracy": (cell["both_pass"] + harms) / n,
            "candidate_accuracy": (cell["both_pass"] + gains) / n,
            "degradation": degradation,
            "pointwise_bounds": pointwise,
            "simultaneous_bounds": simultaneous,
            "conditional_noninferiority": noninferiority,
            "familywise_excessive_degradation": excessive,
            "observed_margin_breach": breach,
        }
        cells.append(result)
        if breach:
            observed_breaches.append({**identity, "degradation": degradation, "n": n})
        if degradation < margin and not noninferiority:
            insufficient_precision.append(identity)

    endpoints = []
    for benchmark in tasks:
        for length in length_ids:
            endpoint_cells = [
                cell for cell in cells
                if cell["benchmark"] == benchmark and cell["length"] == length
            ]
            metrics = ("baseline_accuracy", "candidate_accuracy", "degradation")
            endpoints.append({
                "benchmark": benchmark,
                "length": length,
                "seeds": seed_ids,
                "seed_count": len(seed_ids),
                "tasks_per_seed": len(tasks[benchmark]),
                "seed_results": [
                    {"seed": cell["seed"], **{metric: cell[metric] for metric in metrics}}
                    for cell in endpoint_cells
                ],
                "mean": {
                    metric: math.fsum(cell[metric] for cell in endpoint_cells) / len(seed_ids)
                    for metric in metrics
                },
                "range": {
                    metric: [min(cell[metric] for cell in endpoint_cells),
                             max(cell[metric] for cell in endpoint_cells)]
                    for metric in metrics
                },
                "fixed_seed_mean_simultaneous_bounds": {
                    side: math.fsum(cell["simultaneous_bounds"][side] for cell in endpoint_cells)
                    / len(seed_ids)
                    for side in ("lower", "upper")
                },
                "fixed_seed_mean_scope": (
                    "Equal-weight mean over the listed fixed trained pairs, not a future-seed "
                    "mean; bounds average simultaneous cell limits, not pooled task counts"
                ),
            })
    noninferiority = all(cell["conditional_noninferiority"] for cell in cells)
    excessive = any(cell["familywise_excessive_degradation"] for cell in cells)
    if noninferiority:
        decision = "NONINFERIOR"
    elif excessive:
        decision = "EXCESSIVE_DEGRADATION"
    else:
        decision = "INCONCLUSIVE"
    expected_rows = (
        sum(len(task_ids) for task_ids in tasks.values()) * len(seed_ids) * len(length_ids)
    )
    return {
        "schema": "p3-stage2-paired-quality-summary-v1",
        "decision": decision,
        "joint_conditional_noninferiority": noninferiority,
        "familywise_excessive_degradation": excessive,
        "margin": margin,
        "units": "absolute probability; positive degradation means BF16 minus candidate",
        "comparison_count": comparison_count,
        "one_sided_alpha": alpha,
        "iut_joint_type_i_error_at_most": alpha,
        "pointwise_two_sided_coverage_at_least": 1 - 2 * alpha,
        "simultaneous_one_sided_alpha_per_cell": simultaneous_alpha,
        "simultaneous_two_sided_family_coverage_at_least": 1 - 2 * alpha,
        "any_excessive_degradation_false_claim_probability_at_most": alpha,
        "decision_rules": {
            "NONINFERIOR": "Every pointwise upper bound is strictly below the margin",
            "EXCESSIVE_DEGRADATION": (
                "At least one familywise-adjusted lower bound is strictly above the margin"
            ),
            "INCONCLUSIVE": "Neither conditional population criterion holds",
            "observed_margin_breach": "Finite-set degradation >= margin, reported separately",
        },
        "coverage": {
            "complete": True,
            "expected_paired_rows": expected_rows,
            "observed_paired_rows": len(rows),
            "benchmarks": list(tasks),
            "seeds": seed_ids,
            "lengths": length_ids,
            "tasks_per_benchmark": {
                benchmark: len(task_ids) for benchmark, task_ids in tasks.items()
            },
        },
        "observed_margin_breaches": observed_breaches,
        "within_margin_but_insufficient_precision_cells": insufficient_precision,
        "scope": _scope(),
        "scipy_version": cells[0]["pointwise_bounds"]["scipy_version"],
        "cells": cells,
        "endpoints": endpoints,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Write hypothetical precision JSON, never measured Stage 2 quality results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Destination planning JSON")
    parser.add_argument(
        "--n", type=int, nargs="+", default=_DEFAULT_SAMPLE_SIZES,
        help="Distinct positive hypothetical task counts; default: 101 437 1299 5000 10000",
    )
    args = parser.parse_args(argv)
    console = Console()
    try:
        report = precision_scenarios(args.n)
        serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized, encoding="utf-8")
    except (ImportError, OSError, TypeError, ValueError) as error:
        Console(stderr=True).print(f"Planning error: {error}", markup=False)
        return 2
    console.print("Analytical hypothetical precision only; not measured model results or power.")
    console.print(f"SciPy {report['scipy_version']}; strict one-percentage-point margin.")
    console.print(
        f"Minimum zero-discordance n for one cell: {report['minimum_zero_discordance_n']}"
    )
    table = Table(title="Hypothetical paired bounds, pointwise upper degradation")
    for heading in ("n", "Total discordance", "harms = gains", "Upper, pp", "U < 1pp"):
        table.add_column(heading, justify="right")
    for scenario in report["scenarios"]:
        zero = scenario["zero_discordance"]
        table.add_row(
            str(scenario["n"]), "0%", "0", f"{100 * zero['upper']:.6f}",
            str(zero["noninferiority_bound_below_margin"]),
        )
        for balanced in scenario["balanced_discordance"]:
            table.add_row(
                str(scenario["n"]),
                f"{100 * balanced['hypothetical_total_discordance_rate']:.4f}%",
                str(balanced["harms"]), f"{100 * balanced['upper']:.6f}",
                str(balanced["noninferiority_bound_below_margin"]),
            )
    console.print(table)
    console.print(f"Planning JSON: {args.out}", markup=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
