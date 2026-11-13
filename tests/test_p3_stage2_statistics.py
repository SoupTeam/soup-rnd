"""Paired coverage, conservative precision and all-required Stage 2 decisions."""

from __future__ import annotations

import builtins
import json
import math
import runpy
from pathlib import Path

import pytest

from benchmarks.harness.p3_stage2_statistics import (
    main,
    minimum_zero_discordance_n,
    paired_bounds,
    precision_scenarios,
    summarize_pairs,
)


def _concordant_rows(
    expected_tasks: dict[str, list[str]],
    seeds: tuple[int, ...] = (23,),
    lengths: tuple[int, ...] = (1024,),
) -> list[dict]:
    return [
        {
            "benchmark": benchmark,
            "length": length,
            "seed": seed,
            "task_id": task_id,
            "baseline_pass": True,
            "candidate_pass": True,
        }
        for benchmark, task_ids in expected_tasks.items()
        for length in lengths
        for seed in seeds
        for task_id in task_ids
    ]


def test_zero_discordance_has_nonzero_uncertainty_at_101_tasks() -> None:
    result = paired_bounds(0, 0, 101)
    assert result["upper"] == pytest.approx(0.0424587, abs=1e-7)
    assert result["lower"] == -result["upper"]
    assert result["degradation"] == 0
    assert result["harms"] == result["gains"] == 0
    assert result["concordant"] == result["n"] == 101
    assert result["upper"] > 0.01


def test_small_sample_can_pass_after_sufficient_candidate_only_successes() -> None:
    assert paired_bounds(0, 101, 101)["upper"] < 0.01


def test_strict_one_percentage_point_boundary_is_436_then_437() -> None:
    assert paired_bounds(0, 0, 436)["upper"] >= 0.01
    assert paired_bounds(0, 0, 437)["upper"] < 0.01
    assert minimum_zero_discordance_n() == 437


def test_minimum_sample_size_does_not_admit_equality() -> None:
    # At alpha=.25, beta=.125 and n=3, the zero-event upper bound is exactly .5.
    assert minimum_zero_discordance_n(0.5, one_sided_alpha=0.25) == 4


def test_1299_tasks_can_be_sufficient_but_do_not_guarantee_precision() -> None:
    assert paired_bounds(0, 0, 1299)["upper"] < 0.01
    assert paired_bounds(325, 325, 1299)["upper"] > 0.01


def test_harm_gain_exchange_reverses_sign_and_interval() -> None:
    harm = paired_bounds(9, 3, 101)
    gain = paired_bounds(3, 9, 101)
    assert harm["degradation"] == pytest.approx(6 / 101)
    assert gain["degradation"] == -harm["degradation"]
    assert gain["lower"] == pytest.approx(-harm["upper"], abs=1e-14)
    assert gain["upper"] == pytest.approx(-harm["lower"], abs=1e-14)


@pytest.mark.parametrize("n", [1, 101, 1299])
def test_clopper_pearson_endpoints_remain_finite(n: int) -> None:
    harm = paired_bounds(n, 0, n)
    gain = paired_bounds(0, n, n)
    # Closed forms are independent of SciPy's interior beta quantiles.
    expected_lower = 2 * math.exp(math.log(0.0125) / n) - 1
    assert harm["lower"] == pytest.approx(expected_lower, abs=1e-14)
    assert harm["upper"] == 1.0
    assert gain["lower"] == -1.0
    assert gain["upper"] == pytest.approx(-expected_lower, abs=1e-14)
    assert harm["degradation"] == 1.0
    assert gain["degradation"] == -1.0
    assert all(math.isfinite(harm[key]) for key in ("lower", "upper", "degradation"))


@pytest.mark.parametrize(
    ("harms", "gains", "n", "error"),
    [
        (True, 0, 1, TypeError),
        (0, False, 1, TypeError),
        (0, 0, True, TypeError),
        (1.0, 0, 2, TypeError),
        (0, "1", 2, TypeError),
        (0, 0, 2.0, TypeError),
        (float("nan"), 0, 2, TypeError),
        (0, float("inf"), 2, TypeError),
        (-1, 0, 2, ValueError),
        (0, -1, 2, ValueError),
        (0, 0, 0, ValueError),
        (0, 0, -1, ValueError),
        (3, 0, 2, ValueError),
        (1, 2, 2, ValueError),
        (0, 0, 10**400, ValueError),
    ],
)
def test_invalid_counts_cannot_produce_bounds(harms, gains, n, error) -> None:
    with pytest.raises(error):
        paired_bounds(harms, gains, n)


@pytest.mark.parametrize("alpha", [True, "0.025", 0, -0.1, 0.5, 1, math.nan, math.inf])
def test_invalid_alpha_is_rejected(alpha) -> None:
    with pytest.raises((TypeError, ValueError)):
        paired_bounds(0, 0, 101, one_sided_alpha=alpha)
    with pytest.raises((TypeError, ValueError)):
        minimum_zero_discordance_n(one_sided_alpha=alpha)


def test_interior_bounds_enclose_estimate_and_grow_with_confidence() -> None:
    pointwise = paired_bounds(9, 3, 101)
    adjusted = paired_bounds(9, 3, 101, one_sided_alpha=0.000625)
    assert pointwise["lower"] < pointwise["degradation"] < pointwise["upper"]
    assert adjusted["lower"] < pointwise["lower"]
    assert adjusted["upper"] > pointwise["upper"]


@pytest.mark.parametrize("margin", [True, "0.01", 0, -0.01, 1, math.nan, math.inf])
def test_invalid_margin_is_rejected(margin) -> None:
    with pytest.raises((TypeError, ValueError)):
        minimum_zero_discordance_n(margin)
    with pytest.raises((TypeError, ValueError)):
        precision_scenarios([101], margin=margin)


def test_counts_keep_both_failures_and_both_successes_in_denominator() -> None:
    tasks = {"fixture": ["both-pass", "harm", "gain", "both-fail"]}
    rows = _concordant_rows(tasks)
    rows[1]["candidate_pass"] = False
    rows[2]["baseline_pass"] = False
    rows[3]["baseline_pass"] = rows[3]["candidate_pass"] = False
    report = summarize_pairs(rows, expected_tasks=tasks, seeds=[23], lengths=[1024])
    cell = report["cells"][0]
    assert cell["n"] == 4
    assert cell["harms"] == cell["gains"] == 1
    assert cell["both_pass"] == cell["both_fail"] == 1
    assert cell["concordant"] == 2
    assert cell["baseline_accuracy"] == cell["candidate_accuracy"] == 0.5
    assert cell["degradation"] == 0.0
    assert cell["pointwise_bounds"]["upper"] > 0
    assert report == summarize_pairs(
        list(reversed(rows)), expected_tasks=tasks, seeds=[23], lengths=[1024]
    )


@pytest.mark.parametrize("missing", ["task", "seed", "length", "benchmark"])
def test_missing_paired_coverage_is_rejected(missing: str) -> None:
    tasks = {"gsm8k": ["g0", "g1"], "humaneval": ["h0", "h1"]}
    rows = _concordant_rows(tasks, seeds=(23, 41), lengths=(1024, 4096))
    if missing == "task":
        rows.pop()
    elif missing == "seed":
        rows = [row for row in rows if row["seed"] != 41]
    elif missing == "length":
        rows = [row for row in rows if row["length"] != 4096]
    else:
        rows = [row for row in rows if row["benchmark"] != "humaneval"]
    with pytest.raises(ValueError, match="coverage"):
        summarize_pairs(rows, expected_tasks=tasks, seeds=[23, 41], lengths=[1024, 4096])


def test_duplicate_paired_task_is_rejected_even_when_counts_look_complete() -> None:
    tasks = {"humaneval": ["h0", "h1"]}
    rows = _concordant_rows(tasks)
    rows[1] = dict(rows[0])
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_pairs(rows, expected_tasks=tasks, seeds=[23], lengths=[1024])


@pytest.mark.parametrize(
    ("field", "value"),
    [("benchmark", "extra"), ("seed", 67), ("length", 4096), ("task_id", "extra")],
)
def test_extra_coverage_is_rejected(field: str, value) -> None:
    tasks = {"humaneval": ["h0"]}
    rows = _concordant_rows(tasks)
    rows[0][field] = value
    with pytest.raises(ValueError, match="Unexpected"):
        summarize_pairs(rows, expected_tasks=tasks, seeds=[23], lengths=[1024])


@pytest.mark.parametrize(
    ("field", "value"),
    [("benchmark", 1), ("seed", True), ("length", 1024.0), ("task_id", None)],
)
def test_invalid_row_key_types_are_not_coerced(field: str, value) -> None:
    tasks = {"humaneval": ["h0"]}
    rows = _concordant_rows(tasks)
    rows[0][field] = value
    with pytest.raises(TypeError):
        summarize_pairs(rows, expected_tasks=tasks, seeds=[23], lengths=[1024])


@pytest.mark.parametrize("field", ["baseline_pass", "candidate_pass"])
@pytest.mark.parametrize("outcome", [0, 1, "false", None, math.nan])
def test_outcomes_must_be_python_bools(field: str, outcome) -> None:
    tasks = {"humaneval": ["h0"]}
    rows = _concordant_rows(tasks)
    rows[0][field] = outcome
    with pytest.raises(TypeError, match="bool"):
        summarize_pairs(rows, expected_tasks=tasks, seeds=[23], lengths=[1024])


@pytest.mark.parametrize(
    ("tasks", "seeds", "lengths"),
    [
        ({}, [23], [1024]),
        ({"humaneval": []}, [23], [1024]),
        ({"humaneval": ["h0", "h0"]}, [23], [1024]),
        ({"humaneval": [1]}, [23], [1024]),
        ({"humaneval": ["h0"]}, [], [1024]),
        ({"humaneval": ["h0"]}, [23, 23], [1024]),
        ({"humaneval": ["h0"]}, [True], [1024]),
        ({"humaneval": ["h0"]}, [23], []),
        ({"humaneval": ["h0"]}, [23], [1024, 1024]),
        ({"humaneval": ["h0"]}, [23], [1024.0]),
    ],
)
def test_invalid_required_design_cannot_be_admitted(tasks, seeds, lengths) -> None:
    with pytest.raises((TypeError, ValueError)):
        summarize_pairs([], expected_tasks=tasks, seeds=seeds, lengths=lengths)


def test_missing_outcome_field_is_not_discarded_or_treated_as_failure() -> None:
    tasks = {"humaneval": ["h0"]}
    rows = _concordant_rows(tasks)
    del rows[0]["candidate_pass"]
    with pytest.raises(ValueError, match="Missing"):
        summarize_pairs(rows, expected_tasks=tasks, seeds=[23], lengths=[1024])


def test_large_apps_cell_cannot_hide_inconclusive_humaneval() -> None:
    tasks = {
        "apps": [str(index) for index in range(5000)],
        "humaneval": [str(index) for index in range(101)],
    }
    report = summarize_pairs(
        _concordant_rows(tasks), expected_tasks=tasks, seeds=[23], lengths=[1024]
    )
    cells = {cell["benchmark"]: cell for cell in report["cells"]}
    assert cells["apps"]["conditional_noninferiority"] is True
    assert cells["humaneval"]["conditional_noninferiority"] is False
    assert report["decision"] == "INCONCLUSIVE"
    assert report["joint_conditional_noninferiority"] is False
    assert report["within_margin_but_insufficient_precision_cells"] == [
        {"benchmark": "humaneval", "length": 1024, "seed": 23}
    ]
    assert len(report["endpoints"]) == 2


def test_population_noninferiority_does_not_pass_at_upper_bound_equality() -> None:
    tasks = {"fixture": ["a", "b", "c"]}
    report = summarize_pairs(
        _concordant_rows(tasks), expected_tasks=tasks, seeds=[23], lengths=[1024],
        margin=0.5, one_sided_alpha=0.25,
    )
    assert report["cells"][0]["pointwise_bounds"]["upper"] == 0.5
    assert report["decision"] == "INCONCLUSIVE"
    assert report["joint_conditional_noninferiority"] is False


def test_all_required_iut_uses_pointwise_not_simultaneous_upper_bounds() -> None:
    tasks = {f"benchmark-{index}": [str(task) for task in range(437)] for index in range(4)}
    seeds = [23, 41, 67, 89, 113]
    lengths = [1024, 4096]
    report = summarize_pairs(
        _concordant_rows(tasks, tuple(seeds), tuple(lengths)),
        expected_tasks=tasks,
        seeds=seeds,
        lengths=lengths,
    )
    assert report["comparison_count"] == 40
    assert report["decision"] == "NONINFERIOR"
    assert report["joint_conditional_noninferiority"] is True
    assert report["simultaneous_one_sided_alpha_per_cell"] == 0.000625
    assert all(cell["simultaneous_bounds"]["upper"] > 0.01 for cell in report["cells"])
    assert report["coverage"]["complete"] is True
    assert report["coverage"]["expected_paired_rows"] == 40 * 437


@pytest.mark.parametrize(("n", "decision"), [(10, "INCONCLUSIVE"), (101, "EXCESSIVE_DEGRADATION")])
def test_any_harm_claim_uses_familywise_not_pointwise_lower_bounds(n, decision) -> None:
    tasks = {f"benchmark-{index}": [str(task) for task in range(n)] for index in range(4)}
    seeds = [23, 41, 67, 89, 113]
    lengths = [1024, 4096]
    rows = _concordant_rows(tasks, tuple(seeds), tuple(lengths))
    for row in rows:
        if (row["benchmark"], row["seed"], row["length"]) == ("benchmark-0", 23, 1024):
            row["candidate_pass"] = False
    report = summarize_pairs(
        rows, expected_tasks=tasks, seeds=seeds, lengths=lengths
    )
    harmful = report["cells"][0]
    assert harmful["pointwise_bounds"]["lower"] > 0.01
    assert report["decision"] == decision
    assert report["familywise_excessive_degradation"] is (n == 101)
    assert len(report["observed_margin_breaches"]) == 1
    assert report["observed_margin_breaches"][0]["degradation"] == 1


def test_finite_set_breach_at_equality_remains_separate_from_population_decision() -> None:
    tasks = {"fixture": [str(index) for index in range(100)]}
    rows = _concordant_rows(tasks)
    rows[0]["candidate_pass"] = False
    report = summarize_pairs(rows, expected_tasks=tasks, seeds=[23], lengths=[1024])
    assert report["decision"] == "INCONCLUSIVE"
    assert report["familywise_excessive_degradation"] is False
    assert report["cells"][0]["degradation"] == 0.01
    assert report["cells"][0]["observed_margin_breach"] is True
    assert len(report["observed_margin_breaches"]) == 1


def test_fixed_seed_mean_and_range_do_not_pool_seeds_as_independent_tasks() -> None:
    tasks = {"fixture": ["one-task"]}
    rows = _concordant_rows(tasks, seeds=(23, 41))
    rows[0]["baseline_pass"] = rows[0]["candidate_pass"] = False
    rows[1]["candidate_pass"] = False
    report = summarize_pairs(rows, expected_tasks=tasks, seeds=[23, 41], lengths=[1024])
    endpoint = report["endpoints"][0]
    assert endpoint["mean"]["degradation"] == 0.5
    assert endpoint["range"]["degradation"] == [0.0, 1.0]
    assert endpoint["mean"]["baseline_accuracy"] == 0.5
    assert endpoint["mean"]["candidate_accuracy"] == 0.0
    # With two cells, simultaneous marginal beta=.00625. These are averages
    # of the simultaneous cell bounds, not bounds based on a pooled n=2.
    bounds = endpoint["fixed_seed_mean_simultaneous_bounds"]
    assert bounds["lower"] == pytest.approx(-0.990625)
    assert bounds["upper"] == pytest.approx(0.996875)
    assert endpoint["tasks_per_seed"] == 1
    assert report["scope"]["future_seed_inference"] is False
    assert report["scope"]["pooled_benchmark_inference"] is False


def test_planning_scenarios_are_deterministic_hypotheticals_with_rounded_counts() -> None:
    sizes = [101, 437, 1299, 5000, 10000]
    report = precision_scenarios(sizes)
    assert report["kind"] == "ANALYTICAL_HYPOTHETICAL_NOT_MEASURED"
    assert report["guaranteed_power"] is False
    assert report["minimum_zero_discordance_n"] == 437
    assert [entry["n"] for entry in report["scenarios"]] == sizes
    first = report["scenarios"][0]
    assert first["zero_discordance"]["noninferiority_bound_below_margin"] is False
    assert report["scenarios"][2]["zero_discordance"][
        "noninferiority_bound_below_margin"
    ] is True
    balanced = first["balanced_discordance"]
    assert [entry["harms"] for entry in balanced] == [0, 2, 5, 12, 25]
    assert all(entry["gains"] == entry["harms"] for entry in balanced)
    assert balanced[2]["target_total_discordance_rate"] == 0.1
    assert balanced[2]["hypothetical_total_discordance_rate"] == pytest.approx(10 / 101)
    assert report == precision_scenarios(sizes)


@pytest.mark.parametrize("sizes", [[], [0], [101.0], [True], [101, 101], "101"])
def test_invalid_planning_sample_sizes_are_rejected(sizes) -> None:
    with pytest.raises((TypeError, ValueError)):
        precision_scenarios(sizes)


def test_module_import_does_not_require_scipy(monkeypatch) -> None:
    original_import = builtins.__import__

    def without_scipy(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("SciPy is unavailable in this import-only regression")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", without_scipy)
    source = Path(__file__).parents[1] / "benchmarks" / "harness" / "p3_stage2_statistics.py"
    namespace = runpy.run_path(str(source))
    assert namespace["minimum_zero_discordance_n"]() == 437
    with pytest.raises(ImportError, match="unavailable"):
        namespace["paired_bounds"](0, 0, 101)


def test_runtime_scipy_version_is_recorded() -> None:
    import scipy

    assert paired_bounds(0, 0, 101)["scipy_version"] == scipy.__version__
    assert precision_scenarios([101])["scipy_version"] == scipy.__version__
    tasks = {"fixture": ["one-task"]}
    report = summarize_pairs(
        _concordant_rows(tasks), expected_tasks=tasks, seeds=[23], lengths=[1024]
    )
    assert report["scipy_version"] == scipy.__version__


def test_cli_writes_requested_planning_json(tmp_path: Path, capsys) -> None:
    destination = tmp_path / "nested" / "planning.json"
    assert main(["--out", str(destination), "--n", "101", "437", "1299", "5000", "10000"]) == 0
    report = json.loads(destination.read_text(encoding="utf-8"))
    assert [entry["n"] for entry in report["scenarios"]] == [101, 437, 1299, 5000, 10000]
    assert report["kind"] == "ANALYTICAL_HYPOTHETICAL_NOT_MEASURED"
    assert report["scipy_version"]
    output = capsys.readouterr().out
    assert "hypothetical" in output.lower()
    assert "SciPy" in output


def test_cli_default_sample_sizes_include_all_required_planning_counts(tmp_path: Path) -> None:
    destination = tmp_path / "default.json"
    assert main(["--out", str(destination)]) == 0
    report = json.loads(destination.read_text(encoding="utf-8"))
    assert [entry["n"] for entry in report["scenarios"]] == [101, 437, 1299, 5000, 10000]


def test_invalid_cli_counts_do_not_write_a_plan(tmp_path: Path, capsys) -> None:
    destination = tmp_path / "invalid.json"
    assert main(["--out", str(destination), "--n", "0"]) == 2
    assert not destination.exists()
    assert "error" in capsys.readouterr().err.lower()
