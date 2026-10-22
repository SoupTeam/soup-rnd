"""G4 ticket 05: the quality verdict under benchmarks/gate-g4-nvfp4.md, from the run records.

Spec: .scratch/g4-nvfp4/issues/05-mistral-runs-and-verdict.md. Rule (binding, in Russian):
benchmarks/gate-g4-nvfp4.md. Run records come from ``g4_train.py``, one ``run.json`` per attempt.

    python benchmarks/harness/g4_verdict.py RUNS_DIR --out verdict.json

Every ``run.json`` under the given paths enters the verdict, spoiled ones included: the
one-repeat rule decides which attempt counts. Exit code 0 means the verdict was computed,
whatever it is. Task loss, the paired bootstrap and the step checks are C5's
(``c5_verdict.py``); what differs here is the arm names, the six validity rows and the
outcome table.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Sequence
from datetime import datetime
from fractions import Fraction
from typing import Any

from rich.console import Console

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import c5_verdict as c5  # noqa: E402

console = Console()

SEEDS = c5.SEEDS
GATE_ARMS = ("nf4", "nvfp4")
# Every run the rule asks for: NF4 and NVFP4 on three seeds, bf16 on seed 42.
REQUIRED_RUNS = tuple((arm, seed) for arm in GATE_ARMS for seed in SEEDS) + (("bf16", 42),)
MAX_ATTEMPTS = c5.MAX_ATTEMPTS
EXPECTED_STORAGE = {"nvfp4": "NVFP4Tensor", "nf4": "Params4bit", "bf16": "Parameter"}
QUANTIZED_LAYERS = 224
CONFIG_KEYS_THAT_MAY_DIFFER = frozenset({"base.format", "seed"})
TIME_FORMAT = "%Y-%m-%dT%H:%M:%S%z"

# Thresholds of the rule. Bounds are inclusive, and the mean ratio is an exact rational
# for the same reason as in C5.
MAX_MEAN_RATIO = Fraction("1.03")
MAX_UPPER_BOUND = 1.05
MAX_SUITE_DROP = Fraction(3)

#: What the outcome means for G4 (the rule's table, before the speed measurement of ticket 06).
G4_DECISION = {"pass": "measure_speed", "fail": "stop", "no_verdict": "repeat"}


def suite_results(selected: dict[tuple[str, int], dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Mean correct answers per gate suite over the seeds, NF4 against NVFP4.

    A suite where NF4 on average answers all questions or none is reported but does not
    gate (issue #1192).
    """
    results = {}
    for name, first in selected["nf4", SEEDS[0]]["gate_suites"].items():
        n = first["n"]
        mean = {
            arm: Fraction(sum(selected[arm, seed]["gate_suites"][name]["correct"]
                              for seed in SEEDS), len(SEEDS))
            for arm in GATE_ARMS
        }
        gating = mean["nf4"] not in (0, n)
        results[name] = {
            "n": n,
            "mean_correct": {arm: float(value) for arm, value in mean.items()},
            "mean_drop": float(mean["nf4"] - mean["nvfp4"]),
            "gating": gating,
            "fails": gating and mean["nf4"] - mean["nvfp4"] >= MAX_SUITE_DROP,
        }
    return results


def select_runs(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The run that counts for each arm and seed, under the one-repeat rule.

    A run breaking validity row 5 or 6 is kept as spoiled and may be repeated once with the
    same seed. Any other extra attempt (a third one, or a repeat of a clean run) is a re-roll
    the rule does not allow, and gives no verdict.
    """
    attempts: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for run in sorted(runs, key=lambda r: r["attempt"]):
        attempts.setdefault((run["arm"], run["seed"]), []).append(run)
    selected, spoiled, unexpected, unrepaired = {}, [], [], []
    missing = [{"arm": arm, "seed": seed} for arm, seed in REQUIRED_RUNS
               if (arm, seed) not in attempts]
    for key, tries in attempts.items():
        for i, run in enumerate(tries):
            where = {"arm": run["arm"], "seed": run["seed"], "attempt": run["attempt"]}
            if key not in REQUIRED_RUNS or key in selected or i >= MAX_ATTEMPTS:
                unexpected.append(where)
                continue
            problems = c5.run_problems(run)
            if problems:
                spoiled.append({**where, "problems": problems})
            else:
                selected[key] = run
        if key in REQUIRED_RUNS and key not in selected:
            unrepaired.extend(c5.run_problems(tries[min(len(tries), MAX_ATTEMPTS) - 1]))
    return {
        "selected": selected, "spoiled": spoiled, "unexpected": unexpected,
        "missing": missing, "unrepaired_problems": unrepaired,
    }


def config_diff(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Config keys, other than the base format and the seed, where a run differs from the first."""
    reference = c5._flatten(runs[0]["config"])
    diffs = []
    for run in runs[1:]:
        config = c5._flatten(run["config"])
        keys = sorted(
            key for key in reference.keys() | config.keys()
            if key not in CONFIG_KEYS_THAT_MAY_DIFFER and reference.get(key) != config.get(key)
        )
        if keys:
            diffs.append({"arm": run["arm"], "seed": run["seed"], "keys": keys})
    return diffs


def _rule_committed_first(run: dict[str, Any]) -> bool:
    """The rule's commit is clean and earlier than the run's ``started`` (rule, commit order)."""
    rule = run["rule"]
    if not rule.get("clean") or not rule.get("added_at"):
        return False
    return (datetime.strptime(rule["added_at"], TIME_FORMAT)
            < datetime.strptime(run["started"], TIME_FORMAT))


def _validity(
    chosen: Sequence[dict[str, Any]], selection: dict[str, Any],
) -> tuple[dict[str, bool], list[dict[str, Any]]]:
    """The six validity rows of the rule, plus the commit-order check."""
    diffs = config_diff(chosen) if chosen else []
    unrepaired = selection["unrepaired_problems"]
    fingerprints = [run["unquantized_fingerprints"] for run in chosen]
    data = {(run["data"]["heldout_sha256"], run["data"]["train_sha256"]) for run in chosen}
    return {
        "unquantized_fingerprints": all(f == fingerprints[0] for f in fingerprints),
        "data_hashes": len(data) <= 1,
        "configs_differ_only_in_base_format": not diffs,
        "all_layers_in_arm_format": all(
            run["storage_counts"] == {EXPECTED_STORAGE[run["arm"]]: QUANTIZED_LAYERS}
            for run in chosen
        ),
        "runs_finished_finite": not {"not_finished", "non_finite"} & set(unrepaired),
        "first_grad_norm_positive": "first_grad_norm" not in unrepaired,
        "rule_committed_first": all(_rule_committed_first(run) for run in chosen),
    }, diffs


def _quality(selected: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    ratios = {
        seed: c5.task_loss(selected["nvfp4", seed]) / c5.task_loss(selected["nf4", seed])
        for seed in SEEDS
    }
    mean_ratio = sum(ratios.values()) / len(ratios)
    intervals = {
        seed: c5.paired_bootstrap(
            selected["nf4", seed]["panel"]["per_example"],
            selected["nvfp4", seed]["panel"]["per_example"],
        )
        for seed in SEEDS
    }
    fail_reasons = []
    if mean_ratio > MAX_MEAN_RATIO:
        fail_reasons.append("mean_ratio")
    if any(high > MAX_UPPER_BOUND for _, high in intervals.values()):
        fail_reasons.append("interval")
    suites = suite_results(selected)
    fail_reasons += [f"suite:{name}" for name, suite in suites.items() if suite["fails"]]
    return {
        "fail_reasons": fail_reasons,
        "mean_ratio": float(mean_ratio),
        "ratios": {seed: float(ratio) for seed, ratio in ratios.items()},
        "intervals": intervals,
        "suites": suites,
    }


def verdict(runs: Sequence[dict[str, Any]], *, synthetic: bool = False) -> dict[str, Any]:
    """The rule applied in order: no verdict, then fail, then pass.

    A synthetic set (``synthetic=True``) is computed the same way and labelled; synthetic
    and real runs never mix.
    """
    if any(bool(run["synthetic"]) != synthetic for run in runs):
        raise ValueError(
            "synthetic runs never enter a real verdict, and real runs never enter "
            "a synthetic one"
        )
    selection = select_runs(runs)
    selected = selection["selected"]
    chosen = [selected[key] for key in REQUIRED_RUNS if key in selected]
    validity, diffs = _validity(chosen, selection)
    quality = (
        _quality(selected) if all((arm, seed) in selected for arm in GATE_ARMS for seed in SEEDS)
        else None
    )
    valid = all(validity.values()) and not selection["missing"] and not selection["unexpected"]
    if not valid or quality is None:
        outcome = "no_verdict"
    else:
        outcome = "fail" if quality["fail_reasons"] else "pass"
    return {
        "synthetic": synthetic,
        "outcome": outcome,
        "g4": G4_DECISION[outcome],
        "validity": validity,
        "config_diff": diffs,
        "missing": selection["missing"],
        "unexpected_attempts": selection["unexpected"],
        "spoiled": selection["spoiled"],
        "used": [
            {"arm": run["arm"], "seed": run["seed"], "attempt": run["attempt"]} for run in chosen
        ],
        "harness_sha256": sorted({run["harness"]["sha256"] for run in chosen}),
        **(quality or {"fail_reasons": [], "mean_ratio": None}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+", help="run.json files or directories holding them")
    parser.add_argument("--synthetic", action="store_true", help="the runs are synthetic")
    parser.add_argument("--out", required=True, help="where verdict.json is written")
    args = parser.parse_args(argv)

    files = []
    for path in map(pathlib.Path, args.paths):
        files += sorted(path.rglob("run.json")) if path.is_dir() else [path]
    runs = [json.loads(file.read_text(encoding="utf-8")) for file in files]
    result = verdict(runs, synthetic=args.synthetic)
    result["inputs"] = [str(file) for file in files]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(c5._json_ready(result), indent=1), encoding="utf-8")
    label = "SYNTHETIC " if args.synthetic else ""
    console.print(
        f"{label}verdict: {result['outcome']} (G4: {result['g4']}), "
        f"mean ratio {result['mean_ratio']}, fail reasons {result['fail_reasons'] or 'none'}, "
        f"validity {result['validity']} -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
