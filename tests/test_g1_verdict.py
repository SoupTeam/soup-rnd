"""G1 ticket 05: the quality verdict under benchmarks/gate-g1-nvfp4.md (g1_verdict.py).

Every run record here is written by hand, so nothing trains.
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
STORAGE = {"nvfp4": "NVFP4Tensor", "nf4": "Params4bit", "bf16": "Parameter"}
BASE_NLL = [25.0, 25.0, 25.0, 25.0]
DEFAULT_CORRECT = {"mini_mmlu": 10, "mini_common_sense": 12,
                   "mini_instruction": 8, "mini_arithmetic": 5}


@pytest.fixture(scope="module")
def harness() -> Iterator[ModuleType]:
    name = "_g1_verdict_test_module"
    previous = sys.modules.get(name)
    sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location(name, HARNESS / "g1_verdict.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(HARNESS))
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def _run(
    arm: str, seed: int, *, nll: list[float] | None = None,
    correct: dict[str, int] | None = None, attempt: int = 1,
) -> dict[str, Any]:
    """A clean, complete run record: 4 panel examples of 10 tokens, 3 train steps.

    With the default ``nll`` the panel's summed NLL is 100, so an NVFP4 arm with
    25.75 per example (exact in binary) has a ratio of exactly 1.03.
    """
    return {
        "synthetic": False,
        "arm": arm,
        "seed": seed,
        "attempt": attempt,
        "status": "complete",
        "started": "2026-10-22T19:44:42+0000",
        "rule": {"added_commit": "4fddde39", "added_at": "2026-10-22T20:45:34+05:00",
                 "clean": True},
        "unquantized_fingerprints": {"model.embed_tokens.weight": "aa", "lm_head.weight": "bb"},
        "storage_counts": {STORAGE[arm]: 224},
        "data": {"heldout_sha256": "panel", "train_sha256": "train"},
        "config": {"seed": seed, "base": {"format": arm}, "training": {"micro_batch": 2}},
        "harness": {"sha256": "harness"},
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
            name: {"n": n, "correct": (correct or DEFAULT_CORRECT)[name]}
            for name, n in SUITE_SIZES.items()
        },
    }


def _all_runs(nf4_correct: dict[str, int] | None = None, **nvfp4_kwargs: Any) -> list[dict]:
    runs = []
    for seed in SEEDS:
        runs.append(_run("nf4", seed, correct=nf4_correct))
        runs.append(_run("nvfp4", seed, **nvfp4_kwargs))
    runs.append(_run("bf16", 42))
    return runs


def _replace(runs: list[dict], arm: str, seed: int, **fields: Any) -> list[dict]:
    return [{**run, **fields} if (run["arm"], run["seed"]) == (arm, seed) else run
            for run in runs]


class TestOutcome:
    def test_identical_arms_pass(self, harness):
        result = harness.verdict(_all_runs())
        assert result["outcome"] == "pass"
        assert result["g1"] == "measure_speed"
        assert result["mean_ratio"] == pytest.approx(1.0)

    def test_a_mean_ratio_of_exactly_1_03_passes(self, harness):
        result = harness.verdict(_all_runs(nll=[25.75] * 4))
        assert result["mean_ratio"] == pytest.approx(1.03)
        assert result["outcome"] == "pass"

    def test_a_mean_ratio_above_1_03_fails(self, harness):
        result = harness.verdict(_all_runs(nll=[26.0] * 4))
        assert result["outcome"] == "fail"
        assert result["g1"] == "stop"
        assert "mean_ratio" in result["fail_reasons"]

    def test_an_interval_above_1_05_in_one_seed_fails(self, harness):
        # Seed 7 alone has one example 30 times worse: the bootstrap upper bound passes 1.05
        # while the mean ratio stays below 1.03.
        runs = _all_runs()
        nll = [25.0, 25.0, 25.0, 25.0]
        nf4 = [25.0, 25.0, 25.0, 25.0]
        nll[0] = 30.0
        runs = _replace(runs, "nvfp4", 7, panel={"per_example": [
            {"row": i, "tokens": 10, "nll": v} for i, v in enumerate(nll)]})
        runs = _replace(runs, "nf4", 7, panel={"per_example": [
            {"row": i, "tokens": 10, "nll": v} for i, v in enumerate(nf4)]})
        result = harness.verdict(runs)
        assert result["mean_ratio"] < 1.03
        assert result["outcome"] == "fail"
        assert result["fail_reasons"] == ["interval"]

    def test_a_suite_drop_of_exactly_3_fails_and_2_passes(self, harness):
        low = {**DEFAULT_CORRECT, "mini_mmlu": 7}
        assert harness.verdict(_all_runs(correct=low))["outcome"] == "fail"
        near = {**DEFAULT_CORRECT, "mini_mmlu": 8}
        assert harness.verdict(_all_runs(correct=near))["outcome"] == "pass"

    def test_the_drop_is_the_mean_over_seeds(self, harness):
        runs = _all_runs()
        runs = _replace(runs, "nvfp4", 42, gate_suites={
            name: {"n": n, "correct": DEFAULT_CORRECT[name] - (9 if name == "mini_mmlu" else 0)}
            for name, n in SUITE_SIZES.items()})
        result = harness.verdict(runs)
        assert result["suites"]["mini_mmlu"]["mean_drop"] == pytest.approx(3.0)
        assert result["outcome"] == "fail"

    def test_a_suite_where_nf4_answers_nothing_does_not_gate(self, harness):
        zero = {**DEFAULT_CORRECT, "mini_arithmetic": 0}
        result = harness.verdict(_all_runs(nf4_correct=zero, correct={
            **DEFAULT_CORRECT, "mini_arithmetic": 0}))
        assert result["suites"]["mini_arithmetic"]["gating"] is False
        assert result["outcome"] == "pass"

    def test_bf16_does_not_enter_the_gate(self, harness):
        runs = _replace(_all_runs(), "bf16", 42, panel={"per_example": [
            {"row": i, "tokens": 10, "nll": 999.0} for i in range(4)]})
        assert harness.verdict(runs)["outcome"] == "pass"


class TestValidityRows:
    def test_a_fingerprint_mismatch_gives_no_verdict(self, harness):
        runs = _replace(_all_runs(), "nvfp4", 7,
                        unquantized_fingerprints={"model.embed_tokens.weight": "zz",
                                                  "lm_head.weight": "bb"})
        result = harness.verdict(runs)
        assert result["outcome"] == "no_verdict"
        assert result["g1"] == "repeat"
        assert result["validity"]["unquantized_fingerprints"] is False

    def test_a_data_hash_mismatch_gives_no_verdict(self, harness):
        runs = _replace(_all_runs(), "nf4", 42,
                        data={"heldout_sha256": "panel", "train_sha256": "other"})
        assert harness.verdict(runs)["validity"]["data_hashes"] is False

    def test_a_config_differing_beyond_the_base_format_gives_no_verdict(self, harness):
        runs = _replace(_all_runs(), "nvfp4", 42, config={
            "seed": 42, "base": {"format": "nvfp4"}, "training": {"micro_batch": 4}})
        result = harness.verdict(runs)
        assert result["validity"]["configs_differ_only_in_base_format"] is False
        assert result["config_diff"][0]["keys"] == ["training.micro_batch"]
        assert result["outcome"] == "no_verdict"

    def test_a_layer_outside_the_arm_format_gives_no_verdict_and_no_repeat(self, harness):
        runs = _replace(_all_runs(), "nvfp4", 42,
                        storage_counts={"NVFP4Tensor": 223, "Parameter": 1})
        result = harness.verdict(runs)
        assert result["validity"]["all_layers_in_arm_format"] is False
        assert result["outcome"] == "no_verdict"

    def test_a_wrong_layer_count_gives_no_verdict(self, harness):
        runs = _replace(_all_runs(), "nf4", 7, storage_counts={"Params4bit": 14})
        assert harness.verdict(runs)["validity"]["all_layers_in_arm_format"] is False

    def test_a_rule_committed_after_a_run_started_gives_no_verdict(self, harness):
        late = {"added_commit": "x", "added_at": "2026-10-22T21:00:00+0000", "clean": True}
        runs = _replace(_all_runs(), "nf4", 42, rule=late)
        result = harness.verdict(runs)
        assert result["validity"]["rule_committed_first"] is False
        assert result["outcome"] == "no_verdict"

    def test_a_dirty_rule_gives_no_verdict(self, harness):
        dirty = {"added_commit": "x", "added_at": "2026-10-22T10:00:00+0000", "clean": False}
        runs = _replace(_all_runs(), "nf4", 42, rule=dirty)
        assert harness.verdict(runs)["validity"]["rule_committed_first"] is False


class TestRepeatRule:
    def _spoiled(self, arm: str, seed: int, **fields: Any) -> dict[str, Any]:
        return {**_run(arm, seed), **fields}

    def test_a_non_finite_run_with_one_clean_repeat_counts_the_repeat(self, harness):
        bad = self._spoiled("nvfp4", 42, train_log=[
            {"step": 1, "loss": float("nan"), "grad_norm": 0.5}])
        runs = [bad if (r["arm"], r["seed"]) == ("nvfp4", 42) else r for r in _all_runs()]
        runs.append(_run("nvfp4", 42, attempt=2))
        result = harness.verdict(runs)
        assert result["outcome"] == "pass"
        assert result["spoiled"][0]["problems"] == ["not_finished", "non_finite"]
        assert {"arm": "nvfp4", "seed": 42, "attempt": 2} in result["used"]

    def test_two_spoiled_attempts_give_no_verdict(self, harness):
        bad = {"train_log": [{"step": 1, "loss": 1.0, "grad_norm": 0.0}]}
        runs = [self._spoiled("nvfp4", 42, **bad) if (r["arm"], r["seed"]) == ("nvfp4", 42)
                else r for r in _all_runs()]
        runs.append(self._spoiled("nvfp4", 42, attempt=2, **bad))
        result = harness.verdict(runs)
        assert result["outcome"] == "no_verdict"
        assert result["validity"]["first_grad_norm_positive"] is False

    def test_a_third_attempt_or_a_repeat_of_a_clean_run_gives_no_verdict(self, harness):
        runs = _all_runs() + [_run("nf4", 42, attempt=2)]
        result = harness.verdict(runs)
        assert result["outcome"] == "no_verdict"
        assert result["unexpected_attempts"] == [{"arm": "nf4", "seed": 42, "attempt": 2}]

    def test_a_missing_run_gives_no_verdict(self, harness):
        runs = [r for r in _all_runs() if (r["arm"], r["seed"]) != ("bf16", 42)]
        result = harness.verdict(runs)
        assert result["outcome"] == "no_verdict"
        assert result["missing"] == [{"arm": "bf16", "seed": 42}]


class TestInputs:
    def test_a_synthetic_run_never_enters_a_real_verdict(self, harness):
        runs = _replace(_all_runs(), "nf4", 42, synthetic=True)
        with pytest.raises(ValueError, match="synthetic"):
            harness.verdict(runs)

    def test_main_writes_the_verdict_json(self, harness, tmp_path):
        for run in _all_runs():
            folder = tmp_path / "runs" / f"{run['arm']}-s{run['seed']}-a{run['attempt']}"
            folder.mkdir(parents=True)
            (folder / "run.json").write_text(json.dumps(run), encoding="utf-8")
        out = tmp_path / "verdict.json"
        assert harness.main([str(tmp_path / "runs"), "--out", str(out)]) == 0
        written = json.loads(out.read_text(encoding="utf-8"))
        assert written["outcome"] == "pass"
        assert written["ratios"]["42"] == pytest.approx(1.0)
        assert written["intervals"]["42"] == [pytest.approx(1.0), pytest.approx(1.0)]
