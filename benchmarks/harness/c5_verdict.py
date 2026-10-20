"""C5 ticket 04: the quality verdict under rule 03, from the run records.

Spec: .scratch/c5-q2-0-experts/issues/04-lora-nf4-vs-q2-0.md. Rule (binding,
in Russian): benchmarks/gate-c5-q2-0-experts.md. Run records come from
``c5_train.py``, one ``run.json`` per attempt.

    python benchmarks/harness/c5_verdict.py RUNS_DIR \
        --swap-record benchmarks/results/c5-olmoe-swap/nf4.json --out verdict.json

Every ``run.json`` under the given paths enters the verdict, spoiled ones
included: the one-repeat rule decides which attempt counts. Exit code 0 means
the verdict was computed, whatever it is.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
from collections.abc import Sequence
from fractions import Fraction
from typing import Any

from rich.console import Console

console = Console()

SEEDS = (42, 7, 123)
GATE_ARMS = ("nf4", "q2_0")
# Every run the rule asks for: NF4 and Q2_0 on three seeds, bf16 on seed 42.
REQUIRED_RUNS = tuple((arm, seed) for arm in GATE_ARMS for seed in SEEDS) + (("bf16", 42),)
# A run breaking validity row 4 or 5 is repeated once with the same seed.
MAX_ATTEMPTS = 2
CONFIG_KEYS_THAT_MAY_DIFFER = frozenset({"codec", "seed"})


# Thresholds of rule 03, as exact rationals: bounds are inclusive, and float
# averaging of three ratios of exactly 1.03 gives 1.0300000000000002.
MAX_MEAN_RATIO = Fraction("1.03")
# The interval's upper bound is a float from the percentile interpolation; 1.05
# itself passes.
MAX_UPPER_BOUND = 1.05
# A fail with a mean ratio up to this still allows G5 in mixed form.
MIXED_G5_RATIO = Fraction("1.10")
# A gating suite fails when Q2_0 answers this many fewer questions, on average.
MAX_SUITE_DROP = Fraction(3)
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 17


def task_loss(run: dict[str, Any]) -> Fraction:
    """NLL on response tokens over the panel, weighted by tokens, exactly."""
    examples = run["panel"]["per_example"]
    return Fraction(sum(Fraction(e["nll"]) for e in examples)) / sum(
        e["tokens"] for e in examples
    )


def _percentile(ordered: Sequence[float], q: float) -> float:
    """Linear interpolation between closest ranks (numpy's default method)."""
    h = (len(ordered) - 1) * q
    low = math.floor(h)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (h - low) * (ordered[high] - ordered[low])


def paired_bootstrap(
    nf4: Sequence[dict[str, Any]], q2_0: Sequence[dict[str, Any]], *,
    resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95% interval of task loss(Q2_0) / task loss(NF4) over panel examples.

    Each resample draws panel examples whole, with replacement, by
    ``random.Random(seed)``, and takes the same draw in both arms. A drawn
    example brings all its response tokens, so the ratio of the resample is
    its summed Q2_0 NLL over its summed NF4 NLL (both arms see the same
    tokens). Bounds are the 2.5th and 97.5th percentiles, interpolated.
    """
    if [(e["row"], e["tokens"]) for e in nf4] != [(e["row"], e["tokens"]) for e in q2_0]:
        raise ValueError("the two panels differ in their rows or token counts")
    rng = random.Random(seed)
    n = len(nf4)
    ratios = []
    for _ in range(resamples):
        draw = [rng.randrange(n) for _ in range(n)]
        ratios.append(sum(q2_0[i]["nll"] for i in draw) / sum(nf4[i]["nll"] for i in draw))
    ratios.sort()
    return _percentile(ratios, 0.025), _percentile(ratios, 0.975)


def suite_results(pairs: dict[tuple[str, int], dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Mean correct answers per gate suite over the seeds, NF4 against Q2_0.

    A suite where NF4 on average answers all questions or none is reported but
    does not gate (issue #1192).
    """
    results = {}
    for name, first in pairs["nf4", SEEDS[0]]["gate_suites"].items():
        n = first["n"]
        mean = {
            arm: Fraction(sum(pairs[arm, seed]["gate_suites"][name]["correct"]
                              for seed in SEEDS), len(SEEDS))
            for arm in GATE_ARMS
        }
        results[name] = {
            "n": n,
            "mean_correct": {arm: float(value) for arm, value in mean.items()},
            "mean_drop": float(mean["nf4"] - mean["q2_0"]),
            "gating": mean["nf4"] not in (0, n),
            "fails": mean["nf4"] not in (0, n) and mean["nf4"] - mean["q2_0"] >= MAX_SUITE_DROP,
        }
    return results


def run_problems(run: dict[str, Any]) -> list[str]:
    """Which of validity rows 4 and 5 one run breaks, by the trainer log."""
    log = run["train_log"]
    problems = []
    if run["status"] != "complete" or not log or log[-1]["step"] != run["total_steps"]:
        problems.append("not_finished")
    if not all(math.isfinite(step[key]) for step in log for key in ("loss", "grad_norm")):
        problems.append("non_finite")
    if log and not log[0]["grad_norm"] > 0:
        problems.append("first_grad_norm")
    return problems


def select_runs(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The run that counts for each arm and seed, under the one-repeat rule.

    A run breaking row 4 or 5 is kept as spoiled and may be repeated once with
    the same seed. Any other extra attempt (a third one, or a repeat of a clean
    run) is a re-roll the rule does not allow, and gives no verdict.
    """
    attempts: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for run in sorted(runs, key=lambda r: r["attempt"]):
        attempts.setdefault((run["arm"], run["seed"]), []).append(run)
    selected, spoiled, unexpected, missing, unrepaired = {}, [], [], [], []
    for key in REQUIRED_RUNS:
        if key not in attempts:
            missing.append({"arm": key[0], "seed": key[1]})
    for key, tries in attempts.items():
        for i, run in enumerate(tries):
            where = {"arm": run["arm"], "seed": run["seed"], "attempt": run["attempt"]}
            if key not in REQUIRED_RUNS or key in selected or i >= MAX_ATTEMPTS:
                unexpected.append(where)
                continue
            problems = run_problems(run)
            if problems:
                spoiled.append({**where, "problems": problems})
            else:
                selected[key] = run
        if key in REQUIRED_RUNS and key not in selected:
            unrepaired.extend(run_problems(tries[min(len(tries), MAX_ATTEMPTS) - 1]))
    return {
        "selected": selected, "spoiled": spoiled, "unexpected": unexpected,
        "missing": missing, "unrepaired_problems": unrepaired,
    }


def _flatten(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def config_diff(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Config keys, other than the codec and the seed, where a run differs from the first."""
    reference = _flatten(runs[0]["config"])
    diffs = []
    for run in runs[1:]:
        config = _flatten(run["config"])
        keys = sorted(
            key for key in reference.keys() | config.keys()
            if key not in CONFIG_KEYS_THAT_MAY_DIFFER and reference.get(key) != config.get(key)
        )
        if keys:
            diffs.append({"arm": run["arm"], "seed": run["seed"], "keys": keys})
    return diffs


def _validity(
    chosen: Sequence[dict[str, Any]], selection: dict[str, Any],
    swap_records: Sequence[dict[str, Any]],
) -> tuple[dict[str, bool], list[dict[str, Any]]]:
    fingerprints = [run["non_expert_fingerprints"] for run in chosen]
    fingerprints += [record["non_expert_fingerprints"] for record in swap_records]
    data = [(run["data"]["heldout_sha256"], run["data"]["train_sha256"]) for run in chosen]
    diffs = config_diff(chosen) if chosen else []
    unrepaired = selection["unrepaired_problems"]
    return {
        "non_expert_fingerprints": all(f == fingerprints[0] for f in fingerprints),
        "data_hashes": len(set(data)) <= 1,
        "configs_differ_only_in_codec": not diffs,
        "runs_finished_finite": not {"not_finished", "non_finite"} & set(unrepaired),
        "first_grad_norm_positive": "first_grad_norm" not in unrepaired,
    }, diffs


def _quality(selected: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    ratios = {
        seed: task_loss(selected["q2_0", seed]) / task_loss(selected["nf4", seed])
        for seed in SEEDS
    }
    mean_ratio = sum(ratios.values()) / len(ratios)
    intervals = {
        seed: paired_bootstrap(
            selected["nf4", seed]["panel"]["per_example"],
            selected["q2_0", seed]["panel"]["per_example"],
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
        "mean_ratio_exact": mean_ratio,
        "mean_ratio": float(mean_ratio),
        "ratios": {seed: float(ratio) for seed, ratio in ratios.items()},
        "intervals": intervals,
        "suites": suites,
    }


def g5_recommendation(outcome: str, mean_ratio: Fraction | None) -> str:
    """The rule's table: what the outcome means for G5."""
    if outcome == "no_verdict":
        return "repeat"
    if outcome == "pass":
        return "continue"
    return "mixed" if mean_ratio is not None and mean_ratio <= MIXED_G5_RATIO else "stop"


def verdict(
    runs: Sequence[dict[str, Any]], *, swap_records: Sequence[dict[str, Any]] = (),
    synthetic: bool = False,
) -> dict[str, Any]:
    """Rule 03 applied in order: no verdict, then fail, then pass.

    ``swap_records`` are ticket-02 JSONs whose non-expert fingerprints must also
    match. A synthetic set (``synthetic=True``) is computed the same way and
    labelled; synthetic and real runs never mix.
    """
    if any(bool(run["synthetic"]) != synthetic for run in runs):
        raise ValueError(
            "synthetic runs never enter a real verdict, and real runs never enter "
            "a synthetic one"
        )
    selection = select_runs(runs)
    selected = selection["selected"]
    chosen = [selected[key] for key in REQUIRED_RUNS if key in selected]
    validity, diffs = _validity(chosen, selection, swap_records)
    quality = (
        _quality(selected) if all((arm, seed) in selected for arm in GATE_ARMS for seed in SEEDS)
        else None
    )
    valid = all(validity.values()) and not selection["missing"] and not selection["unexpected"]
    if not valid or quality is None:
        outcome = "no_verdict"
    else:
        outcome = "fail" if quality["fail_reasons"] else "pass"
    mean_exact = quality.pop("mean_ratio_exact") if quality else None
    return {
        "synthetic": synthetic,
        "outcome": outcome,
        "g5": g5_recommendation(outcome, mean_exact),
        "validity": validity,
        "config_diff": diffs,
        "missing": selection["missing"],
        "unexpected_attempts": selection["unexpected"],
        "spoiled": selection["spoiled"],
        "used": [
            {"arm": run["arm"], "seed": run["seed"], "attempt": run["attempt"]} for run in chosen
        ],
        # The rule fixes the codec from the first arm on; more than one value here
        # means it changed. Reported, not a validity row.
        "codec_sha256": sorted({run["harness"]["codec_sha256"] for run in chosen}),
        **(quality or {"fail_reasons": [], "mean_ratio": None}),
    }


def _json_ready(value: Any) -> Any:
    """Seed keys as strings and intervals as lists, so the record is plain JSON."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+", help="run.json files or directories holding them")
    parser.add_argument(
        "--swap-record", action="append", default=[],
        help="a ticket-02 JSON whose non-expert fingerprints must match too",
    )
    parser.add_argument("--synthetic", action="store_true", help="the runs are synthetic")
    parser.add_argument("--out", required=True, help="where verdict.json is written")
    args = parser.parse_args(argv)
    if not args.synthetic and not args.swap_record:
        parser.error("a real verdict checks row 1 against the ticket-02 JSON: pass --swap-record")

    files = []
    for path in map(pathlib.Path, args.paths):
        files += sorted(path.rglob("run.json")) if path.is_dir() else [path]
    runs = [json.loads(file.read_text(encoding="utf-8")) for file in files]
    swaps = [json.loads(pathlib.Path(p).read_text(encoding="utf-8")) for p in args.swap_record]
    result = verdict(runs, swap_records=swaps, synthetic=args.synthetic)
    result["inputs"] = [str(file) for file in files] + list(args.swap_record)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_ready(result), indent=1), encoding="utf-8")
    if len(result["codec_sha256"]) > 1:
        console.print(f"[red]the codec changed between runs[/]: {result['codec_sha256']}")
    label = "SYNTHETIC " if args.synthetic else ""
    console.print(
        f"{label}verdict: {result['outcome']} (G5: {result['g5']}), "
        f"mean ratio {result['mean_ratio']}, fail reasons {result['fail_reasons'] or 'none'}, "
        f"validity {result['validity']} -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
