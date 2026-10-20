"""C5 ticket 04: the verdict under rule 03 (benchmarks/harness/c5_verdict.py).

Spec: .scratch/c5-q2-0-experts/issues/04-lora-nf4-vs-q2-0.md. Rule:
benchmarks/gate-c5-q2-0-experts.md. Every run record here is written by hand,
so nothing trains.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"

SUITE_SIZES = {
    "mini_mmlu": 26, "mini_common_sense": 24, "mini_instruction": 24, "mini_arithmetic": 40,
}
SEEDS = (42, 7, 123)


@pytest.fixture(scope="module")
def harness() -> Iterator[ModuleType]:
    name = "_c5_verdict_test_module"
    previous = sys.modules.get(name)
    spec = importlib.util.spec_from_file_location(name, HARNESS / "c5_verdict.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


BASE_NLL = [25.0, 25.0, 25.0, 25.0]


def _run(
    arm: str, seed: int, *, nll: list[float] | None = None,
    correct: dict[str, int] | None = None, attempt: int = 1,
) -> dict[str, Any]:
    """A clean, complete run record: 4 panel examples of 10 tokens, 3 train steps.

    With the default ``nll`` the panel's summed NLL is 100, so a Q2_0 arm with
    25.75 per example (exact in binary) has a ratio of exactly 1.03.
    """
    correct = correct or {"mini_mmlu": 10, "mini_common_sense": 12,
                          "mini_instruction": 8, "mini_arithmetic": 5}
    return {
        "synthetic": False,
        "arm": arm,
        "seed": seed,
        "attempt": attempt,
        "status": "complete",
        "non_expert_fingerprints": {"model.embed_tokens.weight": "aa", "lm_head.weight": "bb"},
        "data": {"heldout_sha256": "panel", "train_sha256": "train"},
        "config": {"codec": None if arm == "bf16" else arm, "seed": seed, "lr": 2e-4},
        "total_steps": 3,
        "train_log": [
            {"step": step, "loss": 2.0 - step / 10, "grad_norm": 0.5} for step in (1, 2, 3)
        ],
        "panel": {
            "per_example": [
                {"row": row, "tokens": 10, "nll": value}
                for row, value in enumerate(nll or BASE_NLL)
            ],
        },
        "gate_suites": {
            name: {"n": n, "correct": correct[name]} for name, n in SUITE_SIZES.items()
        },
    }


def _all_runs(**q2_kwargs: Any) -> list[dict[str, Any]]:
    runs = []
    for seed in SEEDS:
        runs.append(_run("nf4", seed))
        runs.append(_run("q2_0", seed, **q2_kwargs))
    runs.append(_run("bf16", 42))
    return runs


class TestOutcome:
    def test_identical_arms_pass(self, harness):
        result = harness.verdict(_all_runs())
        assert result["outcome"] == "pass"
        assert result["mean_ratio"] == pytest.approx(1.0)

    def test_a_mean_ratio_of_exactly_1_03_passes(self, harness):
        # Every example is 25.75 against 25, so every bootstrap resample is
        # also exactly 1.03 and the interval cannot fail the run.
        result = harness.verdict(_all_runs(nll=[25.75] * 4))
        assert result["mean_ratio"] == pytest.approx(1.03)
        assert result["outcome"] == "pass"

    def test_a_mean_ratio_above_1_03_fails(self, harness):
        result = harness.verdict(_all_runs(nll=[26.0] * 4))
        assert result["outcome"] == "fail"
        assert "mean_ratio" in result["fail_reasons"]


def _examples(nll: list[float], tokens: int = 10) -> list[dict[str, Any]]:
    return [{"row": row, "tokens": tokens, "nll": value} for row, value in enumerate(nll)]


class TestPairedBootstrap:
    def test_identical_arms_give_a_point_interval_even_with_spread(self, harness):
        # Unpaired resampling would spread this; paired resampling cannot.
        nll = [5.0, 40.0, 12.0, 90.0, 33.0]
        assert harness.paired_bootstrap(_examples(nll), _examples(nll)) == (1.0, 1.0)

    def test_two_example_panel_spans_its_three_possible_ratios(self, harness):
        # Resamples of 2 rows give ratio 1.0 (row 0 twice, p = 1/4), 1.1 (one
        # of each, 1/2) or 1.2 (row 1 twice, 1/4); 2.5% and 97.5% land on the ends.
        low, high = harness.paired_bootstrap(_examples([10.0, 10.0]), _examples([10.0, 12.0]))
        assert low == pytest.approx(1.0)
        assert high == pytest.approx(1.2)

    def test_the_interval_is_reproducible(self, harness):
        nf4, q2_0 = _examples([10.0, 20.0, 30.0]), _examples([11.0, 19.0, 35.0])
        assert harness.paired_bootstrap(nf4, q2_0) == harness.paired_bootstrap(nf4, q2_0)

    def test_panels_with_different_rows_are_refused(self, harness):
        nf4 = _examples([10.0, 20.0])
        q2_0 = _examples([10.0, 20.0])
        q2_0[1]["row"] = 99
        with pytest.raises(ValueError, match="row"):
            harness.paired_bootstrap(nf4, q2_0)


class TestIntervalRule:
    def test_an_upper_bound_above_1_05_in_one_seed_fails(self, harness):
        runs = _all_runs()
        q2_42 = next(r for r in runs if (r["arm"], r["seed"]) == ("q2_0", 42))
        q2_42["panel"]["per_example"] = _examples([25.0, 25.0, 25.0, 28.0])
        result = harness.verdict(runs)
        assert result["mean_ratio"] <= 1.03
        assert result["intervals"][42][1] > 1.05
        assert result["fail_reasons"] == ["interval"]

    def test_an_upper_bound_of_exactly_1_05_passes(self, harness):
        runs = _all_runs()
        q2_42 = next(r for r in runs if (r["arm"], r["seed"]) == ("q2_0", 42))
        q2_42["panel"]["per_example"] = _examples([26.25] * 4)
        result = harness.verdict(runs)
        assert result["intervals"][42] == (1.05, 1.05)
        assert result["outcome"] == "pass"


def _set_correct(runs, arm: str, suite: str, per_seed: dict[int, int]) -> None:
    for run in runs:
        if run["arm"] == arm and run["seed"] in per_seed:
            run["gate_suites"][suite]["correct"] = per_seed[run["seed"]]


class TestGateSuites:
    def test_q2_0_three_answers_worse_on_average_fails(self, harness):
        runs = _all_runs()
        _set_correct(runs, "q2_0", "mini_mmlu", {42: 7, 7: 7, 123: 7})  # NF4 has 10
        result = harness.verdict(runs)
        assert result["fail_reasons"] == ["suite:mini_mmlu"]
        assert result["suites"]["mini_mmlu"]["mean_drop"] == pytest.approx(3.0)

    def test_eight_thirds_of_an_answer_worse_passes(self, harness):
        runs = _all_runs()
        _set_correct(runs, "q2_0", "mini_mmlu", {42: 7, 7: 7, 123: 8})
        result = harness.verdict(runs)
        assert result["outcome"] == "pass"
        assert result["suites"]["mini_mmlu"]["mean_drop"] == pytest.approx(8 / 3)

    @pytest.mark.parametrize("nf4_correct", [0, 26])
    def test_a_suite_nf4_scores_all_or_none_on_is_reported_but_not_gated(
        self, harness, nf4_correct
    ):
        runs = _all_runs()
        _set_correct(runs, "nf4", "mini_mmlu", {42: nf4_correct, 7: nf4_correct,
                                                123: nf4_correct})
        _set_correct(runs, "q2_0", "mini_mmlu", {42: 13, 7: 13, 123: 13})
        result = harness.verdict(runs)
        assert result["suites"]["mini_mmlu"]["gating"] is False
        assert result["outcome"] == "pass"

    def test_nf4_all_correct_in_only_some_seeds_still_gates(self, harness):
        runs = _all_runs()
        _set_correct(runs, "nf4", "mini_mmlu", {42: 26, 7: 26, 123: 25})
        _set_correct(runs, "q2_0", "mini_mmlu", {42: 20, 7: 20, 123: 20})
        result = harness.verdict(runs)
        assert result["suites"]["mini_mmlu"]["gating"] is True
        assert result["fail_reasons"] == ["suite:mini_mmlu"]


def _find(runs, arm: str, seed: int, attempt: int = 1) -> dict[str, Any]:
    return next(r for r in runs if (r["arm"], r["seed"], r["attempt"]) == (arm, seed, attempt))


class TestValidityRows:
    def test_a_non_expert_fingerprint_differing_in_the_bf16_arm_gives_no_verdict(self, harness):
        runs = _all_runs()
        _find(runs, "bf16", 42)["non_expert_fingerprints"]["lm_head.weight"] = "cc"
        result = harness.verdict(runs)
        assert result["outcome"] == "no_verdict"
        assert result["validity"]["non_expert_fingerprints"] is False

    def test_fingerprints_differing_from_the_ticket_02_record_give_no_verdict(self, harness):
        swap = {"non_expert_fingerprints": {"model.embed_tokens.weight": "aa",
                                            "lm_head.weight": "zz"}}
        result = harness.verdict(_all_runs(), swap_records=[swap])
        assert result["validity"]["non_expert_fingerprints"] is False
        assert result["outcome"] == "no_verdict"

    def test_a_different_panel_hash_gives_no_verdict(self, harness):
        runs = _all_runs()
        _find(runs, "nf4", 7)["data"]["heldout_sha256"] = "other"
        assert harness.verdict(runs)["validity"]["data_hashes"] is False

    def test_configs_may_differ_in_codec_and_seed_only(self, harness):
        runs = _all_runs()
        assert harness.verdict(runs)["validity"]["configs_differ_only_in_codec"] is True
        _find(runs, "q2_0", 123)["config"]["lr"] = 3e-4
        result = harness.verdict(runs)
        assert result["validity"]["configs_differ_only_in_codec"] is False
        assert "lr" in result["config_diff"][0]["keys"]

    @pytest.mark.parametrize("field,value", [("loss", float("nan")), ("grad_norm", float("inf"))])
    def test_a_non_finite_step_spoils_the_run(self, harness, field, value):
        run = _run("nf4", 42)
        run["train_log"][1][field] = value
        assert harness.run_problems(run) == ["non_finite"]

    def test_a_run_that_stopped_early_is_spoiled(self, harness):
        run = _run("nf4", 42)
        run["train_log"].pop()
        run["status"] = "failed"
        assert harness.run_problems(run) == ["not_finished"]

    def test_zero_grad_norm_at_the_first_step_spoils_the_run(self, harness):
        run = _run("nf4", 42)
        run["train_log"][0]["grad_norm"] = 0.0
        assert harness.run_problems(run) == ["first_grad_norm"]


class TestRepeats:
    def test_a_spoiled_run_repeated_cleanly_is_replaced_and_kept_on_record(self, harness):
        runs = _all_runs()
        _find(runs, "q2_0", 7)["train_log"][0]["grad_norm"] = 0.0
        runs.append(_run("q2_0", 7, attempt=2))
        result = harness.verdict(runs)
        assert result["outcome"] == "pass"
        assert result["spoiled"] == [
            {"arm": "q2_0", "seed": 7, "attempt": 1, "problems": ["first_grad_norm"]}
        ]

    def test_a_repeat_that_also_fails_gives_no_verdict(self, harness):
        runs = _all_runs()
        _find(runs, "q2_0", 7)["train_log"][0]["grad_norm"] = 0.0
        repeat = _run("q2_0", 7, attempt=2)
        repeat["train_log"][2]["loss"] = float("nan")
        runs.append(repeat)
        result = harness.verdict(runs)
        assert result["outcome"] == "no_verdict"
        assert result["validity"]["runs_finished_finite"] is False

    def test_a_repeat_of_a_clean_run_gives_no_verdict(self, harness):
        # Only a spoiled run may be repeated; a second clean try is a re-roll.
        runs = _all_runs()
        runs.append(_run("q2_0", 7, attempt=2))
        result = harness.verdict(runs)
        assert result["outcome"] == "no_verdict"
        assert result["unexpected_attempts"] == [{"arm": "q2_0", "seed": 7, "attempt": 2}]

    def test_a_missing_arm_gives_no_verdict(self, harness):
        runs = [r for r in _all_runs() if (r["arm"], r["seed"]) != ("nf4", 123)]
        result = harness.verdict(runs)
        assert result["outcome"] == "no_verdict"
        assert result["missing"] == [{"arm": "nf4", "seed": 123}]


class TestSyntheticAndG5:
    def test_synthetic_runs_never_enter_a_real_verdict(self, harness):
        runs = _all_runs()
        runs[0]["synthetic"] = True
        with pytest.raises(ValueError, match="synthetic"):
            harness.verdict(runs)

    def test_an_all_synthetic_set_is_labelled_synthetic(self, harness):
        runs = _all_runs()
        for run in runs:
            run["synthetic"] = True
        assert harness.verdict(runs, synthetic=True)["synthetic"] is True

    @pytest.mark.parametrize("q2_nll,g5", [
        ([25.0] * 4, "continue"),
        ([27.5] * 4, "mixed"),          # ratio exactly 1.10: still mixed
        ([27.75] * 4, "stop"),
    ])
    def test_g5_recommendation_follows_the_outcome_table(self, harness, q2_nll, g5):
        assert harness.verdict(_all_runs(nll=q2_nll))["g5"] == g5

    def test_no_verdict_means_no_g5_decision(self, harness):
        runs = _all_runs()
        _find(runs, "nf4", 7)["data"]["train_sha256"] = "other"
        assert harness.verdict(runs)["g5"] == "repeat"


class TestCommandLine:
    def test_reads_run_records_under_a_directory_and_writes_the_verdict(self, harness, tmp_path):
        for run in _all_runs():
            run_dir = tmp_path / "runs" / f"{run['arm']}-s{run['seed']}-a{run['attempt']}"
            run_dir.mkdir(parents=True)
            (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
        out = tmp_path / "verdict.json"
        assert harness.main([str(tmp_path / "runs"), "--out", str(out)]) == 0
        written = json.loads(out.read_text(encoding="utf-8"))
        assert written["outcome"] == "pass"
        assert len(written["used"]) == 7
