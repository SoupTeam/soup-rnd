"""The section-5 rule of probe-moe-adapter-placement.md, on planted run records."""

import importlib.util
import sys
from pathlib import Path

import pytest

RULE = Path(__file__).parents[1] / "benchmarks" / "harness" / "moe_adapter_rule.py"
SUITES = ("mini_mmlu", "mini_safety")


@pytest.fixture(scope="module")
def rule():
    spec = importlib.util.spec_from_file_location("_moe_adapter_rule", RULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def _run(arm, seed, nll, gate=None, status="ok", model="m"):
    record = {"model": model, "arm": arm, "seed": seed, "status": status,
              "eval": {"nll": nll}, "gate": {"scores": dict(gate or {s: 0.5 for s in SUITES})},
              "task": {"accuracy": 0.5, "correct": [True, False]},
              "trainable_parameters": 10, "adapter_bytes_adamw_fp32": 160,
              "peak_cuda_bytes": 1, "train": {"log": [{"seconds": 1.0}, {"seconds": 2.0}]}}
    if arm != "base":
        record["v4"] = {"passed": True}
    return record


def _records(a_nll, ae_nll=(0.80, 0.81, 0.79), a_gate=None):
    runs = [_run("base", None, 1.00)]
    runs += [_run("AE", s, v) for s, v in enumerate(ae_nll)]
    runs += [_run("A", s, v, gate=a_gate) for s, v in enumerate(a_nll)]
    return runs


def test_small_gap_is_acceptable(rule):
    arms = rule.summarise(_records(a_nll=(0.81, 0.82, 0.80)))["m"]["arms"]
    assert arms["A"]["verdict"] == "ACCEPTABLE"
    assert arms["AE"]["verdict"] == "ACCEPTABLE"


def test_gap_above_floor_and_noise_costs_quality(rule):
    assert rule.summarise(_records(a_nll=(0.85, 0.86, 0.85)))["m"]["arms"]["A"]["verdict"] == \
        "COSTS QUALITY"


def test_gap_inside_seed_noise_is_not_material(rule):
    noisy = rule.summarise(_records(a_nll=(0.80, 0.90, 0.85), ae_nll=(0.75, 0.85, 0.80)))
    assert noisy["m"]["arms"]["A"]["verdict"] == "ACCEPTABLE"


def test_any_seed_forgetting_past_threshold_regresses(rule):
    gate = {"mini_mmlu": 0.44, "mini_safety": 0.5}  # 0.06 below the base on one suite
    summary = rule.summarise(_records(a_nll=(0.80, 0.80, 0.80), a_gate=gate))["m"]["arms"]["A"]
    assert summary["verdict"] == "REGRESSES"
    assert summary["leg2_failed"] == ["mini_mmlu"]


def test_drop_exactly_at_threshold_still_passes(rule):
    gate = {"mini_mmlu": 0.45, "mini_safety": 0.5}
    summary = rule.summarise(_records(a_nll=(0.80, 0.80, 0.80), a_gate=gate))["m"]["arms"]["A"]
    assert summary["verdict"] == "ACCEPTABLE"


def test_fullest_arm_that_learned_nothing_voids_the_model(rule):
    summary = rule.summarise(_records(a_nll=(0.99,), ae_nll=(0.97, 0.96)))["m"]
    assert summary["validity"]["V1"] is False
    assert summary["verdict"] == "NO VERDICT"


def test_edge_learning_rate_voids_the_model(rule):
    summary = rule.summarise(_records(a_nll=(0.8,)), edge_lrs={"A": True})["m"]
    assert summary["validity"]["V3"] is False and summary["verdict"] == "NO VERDICT"


def test_non_finite_run_voids_the_model(rule):
    records = _records(a_nll=(0.8,)) + [_run("A", 9, float("nan"), status="void_nonfinite")]
    assert rule.summarise(records)["m"]["verdict"] == "NO VERDICT"


def test_not_applicable_arm_is_listed_not_scored(rule):
    records = _records(a_nll=(0.8,)) + [_run("AS", 0, 0.0, status="not_applicable")]
    summary = rule.summarise(records)["m"]
    assert summary["not_applicable"] == ["AS"] and "AS" not in summary["arms"]


def test_fullest_arm_falls_back_when_all_experts_did_not_fit(rule):
    records = [_run("base", None, 1.0)] + [_run("AT", s, 0.8) for s in range(2)] + \
        [_run("A", s, 0.9) for s in range(2)]
    summary = rule.summarise(records)["m"]
    assert summary["fullest"] == "AT"
    assert summary["arms"]["A"]["verdict"] == "COSTS QUALITY"
