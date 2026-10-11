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


def test_a_run_that_failed_the_zero_adapter_check_fails_v4(rule):
    void = _run("A", 3, 0.8, status="void_v4")
    void["v4"] = {"passed": False}
    summary = rule.summarise(_records(a_nll=(0.80, 0.81, 0.79)) + [void])["m"]
    assert summary["validity"]["V4"] is False and summary["verdict"] == "NO VERDICT"


def test_per_seed_columns_follow_seed_order(rule):
    runs = [_run("base", None, 1.0)]
    for seed, accuracy in ((2, 0.2), (0, 0.0), (1, 0.1)):
        for arm in ("A", "AE"):
            run = _run(arm, seed, (0.80, 0.81, 0.82)[seed])
            run["task"]["accuracy"] = accuracy
            runs.append(run)
    row = rule.summarise(runs)["m"]["arms"]["A"]
    assert row["seeds"] == [0, 1, 2]
    assert row["accuracy"] == [0.0, 0.1, 0.2] and row["nll"] == [0.80, 0.81, 0.82]


def _lr_run(arm, lr, nll, seed=0, model="olmoe"):
    record = _run(arm, seed, nll, model=model)
    record["lr"] = lr
    return record


def _sweep(rule, a, ae=(0.50, 0.40, 0.45)):
    """OLMoE base and seed-0 sweep over the grid for A and AE (NLL in grid order)."""
    return [_run("base", None, 1.0, model="olmoe")] + [
        _lr_run(arm, lr, v) for arm, values in (("A", a), ("AE", ae))
        for lr, v in zip(rule.GRID, values)]


def test_sweep_winner_is_the_lowest_nll_inside_the_grid(rule):
    sweep = rule.lr_sweep(_sweep(rule, a=(0.70, 0.60, 0.65)))
    assert sweep["A"]["winner"] == 2e-4 and sweep["A"]["status"] == "ok"


def test_grid_edge_winner_withholds_the_verdict_until_its_extension_runs(rule):
    analysis = rule.analyse(_sweep(rule, a=(0.60, 0.65, 0.70)))
    assert analysis["sweep"]["A"]["status"] == "needs_extension"
    assert analysis["sweep"]["A"]["extension"] == 5e-5
    olmoe = analysis["models"]["olmoe"]
    assert olmoe["verdict"] == "NO VERDICT" and olmoe["unsettled_rates"] == ["A"]
    assert "arms" not in olmoe


@pytest.mark.parametrize(("extension_nll", "status", "winner"),
                         [(0.62, "ok", 1e-4), (0.55, "edge", 5e-5)])
def test_extension_decides_between_the_grid_edge_winner_and_v3(rule, extension_nll, status,
                                                               winner):
    records = _sweep(rule, a=(0.60, 0.65, 0.70)) + [_lr_run("A", 5e-5, extension_nll)]
    analysis = rule.analyse(records)
    assert (analysis["sweep"]["A"]["status"], analysis["sweep"]["A"]["winner"]) == \
        (status, winner)
    olmoe = analysis["models"]["olmoe"]
    assert olmoe["validity"]["V3"] is (status == "ok")
    assert olmoe["verdict"] == ("MEASURED" if status == "ok" else "NO VERDICT")


def test_only_winning_rate_runs_count_and_qwen_as_trains_at_the_a_rate(rule):
    records = _sweep(rule, a=(0.70, 0.60, 0.65)) + [_run("base", None, 1.2, model="qwen35")]
    records += [_lr_run(arm, lr, 0.9, seed=s, model="qwen35")
                for arm, lr in (("A", 2e-4), ("AS", 2e-4), ("AS", 4e-4), ("AE", 2e-4))
                for s in (0, 1)]
    analysis = rule.analyse(records)
    qwen = analysis["models"]["qwen35"]
    assert qwen["verdict"] == "MEASURED" and qwen["arms"]["AS"]["seeds"] == [0, 1]
    skipped = {(r["model"], r["arm"], r["lr"]) for r in analysis["not_counted"]}
    assert skipped == {("olmoe", "A", 1e-4), ("olmoe", "A", 4e-4), ("olmoe", "AE", 1e-4),
                       ("olmoe", "AE", 4e-4), ("qwen35", "AS", 4e-4)}


def test_records_from_another_split_or_hot_set_are_rejected_before_the_sweep(rule):
    records = _sweep(rule, a=(0.70, 0.60, 0.65))
    pilot = _lr_run("AE", 2e-4, 0.1)  # same arm and rate as a counted run
    pilot.update(split_sha256="old", _path="b/r/run-pilot.json")
    other_hot = _lr_run("AT", 2e-4, 0.5)
    other_hot.update(hot_sha256="x", _path="b/r/run-at.json")
    for record in records + [other_hot]:
        record.setdefault("split_sha256", "s")
        record.update(harness_sha256="h")
    pilot["harness_sha256"] = "h"
    expect = {"harness_sha256": "h", "split_sha256": "s", "hot_sha256": {"olmoe": "hot"}}
    analysis = rule.analyse(records + [pilot, other_hot], expect=expect)
    assert analysis["rejected"] == [{"source": "b/r/run-pilot.json", "problem": "split old"},
                                    {"source": "b/r/run-at.json", "problem": "hot set x"}]
    assert analysis["sweep"]["AE"]["nll"]["0.0002"] == 0.40 and "AT" not in analysis["sweep"]
