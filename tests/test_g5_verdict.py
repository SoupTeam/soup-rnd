"""G5: the verdict of each mixed arm against all-NF4 (benchmarks/harness/g5_verdict.py).

Rule: benchmarks/gate-g5-mixed-precision-experts.md. Every record here is
written by hand, so nothing trains. The scoring is C5's (``c5_verdict.py``),
which ``tests/test_c5_verdict.py`` covers; these tests cover what G5 adds: the
reproduction control, the baseline source, G5's validity rows and the decision.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"
NAMES = ("g5_verdict", "g5_select", "c5_verdict", "c5_train", "c5_expert_swap", "c5_data",
         "q2_0_codec", "l2l_box", "arm_validity")
SUITE_SIZES = {
    "mini_mmlu": 26, "mini_common_sense": 24, "mini_instruction": 24, "mini_arithmetic": 40,
}
SEEDS = (42, 7, 123)
BASE_NLL = [25.0, 25.0, 25.0, 25.0]
CORRECT = {"mini_mmlu": 10, "mini_common_sense": 12, "mini_instruction": 8, "mini_arithmetic": 5}
NEW_BOX = {"uuid": "GPU-new"}
C5_BOX = {"uuid": "GPU-c5"}


@pytest.fixture(scope="module")
def g5() -> Iterator[ModuleType]:
    original_path = sys.path[:]
    original = {name: sys.modules.get(name) for name in NAMES}
    spec = importlib.util.spec_from_file_location("g5_verdict", HARNESS / "g5_verdict.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["g5_verdict"] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = original_path
        for name, previous in original.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _run(arm: str, seed: int, *, nll: list[float] | None = None, gpu: dict | None = None,
         correct: dict[str, int] | None = None, codec_map: dict | None = None,
         attempt: int = 1) -> dict[str, Any]:
    """A clean run record as c5_train.py writes it (4 panel examples of 10 tokens)."""
    correct = correct or CORRECT
    codec = arm if arm in ("nf4", "q2_0") else None
    if codec_map is not None:
        codec = f"map:{codec_map['map_sha256']}"
    return {
        "synthetic": False, "arm": arm, "seed": seed, "attempt": attempt,
        "status": "complete",
        "gpu": gpu or C5_BOX,
        "versions": {"torch": "2.14.1"},
        "model": {"id": "allenai/OLMoE-1B-7B-0924", "revision": "rev"},
        "non_expert_fingerprints": {"model.embed_tokens.weight": "aa"},
        "expert_digest": f"experts-{arm}",
        "batch_order_sha256": f"order-{seed}",
        "data": {"heldout_sha256": "panel", "train_sha256": "train"},
        "config": {"codec": codec, "seed": seed, "lr": 2e-4},
        "harness": {"codec_sha256": "codec", "sha256": "train", "swap_sha256": "swap"},
        "total_steps": 3,
        "train_log": [{"step": s, "loss": 2.0, "grad_norm": 0.5} for s in (1, 2, 3)],
        "panel": {"per_example": [{"row": row, "tokens": 10, "nll": value}
                                  for row, value in enumerate(nll or BASE_NLL)]},
        "gate_suites": {name: {"n": n, "correct": correct[name]}
                        for name, n in SUITE_SIZES.items()},
        "swap": {"codec_map": codec_map} if codec_map else {"checks": {}},
    }


def _codes(arm: str) -> list[list[str]]:
    rule, fraction = arm.split("_")[1], int(arm[-2:]) / 100
    count = int(4 * fraction)
    chosen = range(count) if rule == "traffic" else range(4 - count, 4)
    return [["q2_0" if e in chosen else "nf4" for e in range(4)] for _ in range(2)]


@pytest.fixture(scope="module")
def selection(g5) -> dict[str, Any]:
    select = g5.g5_select
    maps = {}
    for arm in select.ARMS:
        codes = _codes(arm)
        maps[arm] = {"map_sha256": select.map_sha256(codes),
                     "q2_0_per_layer": [row.count("q2_0") for row in codes]}
    return {
        "synthetic": False,
        "harness": {
            "sha256": "g5-select", "c5_train_sha256": "train", "swap_sha256": "swap",
            "codec_sha256": "codec",
            "sources": {
                "c5_train.py": {"sha256": "train"},
                "c5_expert_swap.py": {"sha256": "swap"},
                "q2_0_codec.py": {"sha256": "codec"},
                "c5_data.py": {"sha256": "data"},
                "g5_select.py": {"sha256": "g5-select"},
                "g5_train.py": {"sha256": "g5-train"},
                "g5_bytes.py": {"sha256": "g5-bytes"},
                "g5_verdict.py": {"sha256": "g5-verdict"},
            },
        },
        "model": {"id": "allenai/OLMoE-1B-7B-0924", "revision": "rev"},
        "calibration": {"sha256": select.CALIBRATION_SHA256,
                        "train_sha256": next(iter(g5.c5_train.TRAIN_SHA256))},
        "checks": {"model_unchanged_by_selection": True},
        "stats_sha256": "stats",
        "maps": maps,
    }


def _bytes(selection: dict[str, Any], **override: int) -> dict[str, Any]:
    arms = {"all_nf4": {"ssd_bytes": 1000}, "all_q2_0": {"ssd_bytes": 600}}
    for arm, entry in selection["maps"].items():
        size = 900 if arm.endswith("f25") else 800
        arms[arm] = {"ssd_bytes": override.get(arm, size), "map_sha256": entry["map_sha256"]}
    # as g5_bytes.py writes it with --cache-json: C5's measured caches reproduced
    checks = dict.fromkeys(("all_nf4_layer_files_match_c5", "all_nf4_ssd_bytes_match_c5",
                            "all_q2_0_layer_files_match_c5", "all_q2_0_ssd_bytes_match_c5",
                            "mixed_arms_of_one_fraction_equal_bytes"), True)
    return {"arms": arms, "checks": checks}


def _map(selection: dict[str, Any], arm: str) -> dict[str, Any]:
    codes = _codes(arm)
    names = [f"model.layers.{layer}.mlp.experts.{key}" for layer in range(2)
             for key in ("gate_up_proj", "down_proj")]
    return {"arm": arm, "map_sha256": selection["maps"][arm]["map_sha256"],
            "stats_sha256": "stats", "calibration_sha256": selection["calibration"]["sha256"],
            "per_expert": {name: codes[int(name.split(".")[2])] for name in names},
            "harness": {"g5_train_sha256": "g5-train", "g5_select_sha256": "g5-select",
                        "sources": deepcopy(selection["harness"]["sources"])}}


def _baselines(train_data: str = "train") -> list[dict[str, Any]]:
    runs = []
    for seed in SEEDS:
        runs.append(_run("nf4", seed))
        runs.append(_run("q2_0", seed, nll=[27.5] * 4))
    runs.append(_run("bf16", 42))
    for run in runs:
        run["data"]["train_sha256"] = train_data
    return runs


def _mixed(selection, nll: dict[str, list[float]] | None = None, gpu=None) -> list[dict]:
    nll = nll or {}
    return [_run(arm, seed, nll=nll.get(arm), gpu=gpu or NEW_BOX,
                 codec_map=_map(selection, arm))
            for seed in SEEDS for arm in selection["maps"]]


def _verdict(g5, selection, *, mixed=None, baselines=None, control=None, bytes_record=None):
    train = selection["calibration"]["train_sha256"]
    mixed = mixed if mixed is not None else _mixed(selection)
    for run in mixed:
        run["data"]["train_sha256"] = train
    control = control or _run("nf4", 42, gpu=NEW_BOX)
    control["data"]["train_sha256"] = train
    reference = _run("nf4", 42)
    reference["data"]["train_sha256"] = train
    return g5.verdict(
        mixed, baselines if baselines is not None else _baselines(train),
        control=control, control_reference=reference, selection=selection,
        bytes_record=bytes_record or _bytes(selection),
    )


class TestReproductionControl:
    def test_a_control_within_tolerance_lets_c5s_baselines_be_reused(self, g5, selection):
        result = _verdict(g5, selection, control=_run("nf4", 42, gpu=NEW_BOX,
                                                      nll=[25.04] * 4))
        assert result["control"]["passed"]
        assert result["baselines"]["source"] == "reused"
        assert all(arm["validity"]["baselines_reproduced"] for arm in result["arms"].values())

    def test_a_control_outside_tolerance_gives_no_verdict_for_every_arm(self, g5, selection):
        # 0.3% off C5's task loss: above the 0.2% the rule allows.
        result = _verdict(g5, selection, control=_run("nf4", 42, gpu=NEW_BOX, nll=[25.075] * 4))
        assert not result["control"]["passed"]
        assert result["baselines"]["source"] == "not_reproduced"
        for arm in result["arms"].values():
            assert arm["outcome"] == "no_verdict"
            assert not arm["validity"]["baselines_reproduced"]
        assert result["decision"]["g5"] == "repeat"

    def test_a_control_from_another_box_than_the_mixed_arms_does_not_count(self, g5, selection):
        result = _verdict(g5, selection, control=_run("nf4", 42, gpu={"uuid": "GPU-other"}))
        assert result["control"]["passed"] and not result["control"]["on_mixed_box"]
        assert result["baselines"]["source"] == "not_reproduced"

    def test_baselines_other_than_the_controls_reference_are_not_reused(self, g5, selection):
        # A passing control vouches for C5's records, not for some other set of runs.
        train = selection["calibration"]["train_sha256"]
        other = _baselines(train)
        other[0]["batch_order_sha256"] = "another-order"
        result = _verdict(g5, selection, baselines=other)
        assert result["control"]["passed"]
        assert result["baselines"]["source"] == "not_reproduced"

    def test_a_failed_control_with_baselines_re_run_on_the_box_still_gives_a_verdict(
        self, g5, selection,
    ):
        train = selection["calibration"]["train_sha256"]
        rerun = _baselines(train)
        for run in rerun:
            run["gpu"] = NEW_BOX
        result = _verdict(g5, selection, baselines=rerun,
                          control=_run("nf4", 42, gpu=NEW_BOX, nll=[26.0] * 4))
        assert not result["control"]["passed"]
        assert result["baselines"]["source"] == "rerun"
        assert all(arm["outcome"] != "no_verdict" for arm in result["arms"].values())

    def test_a_gate_suite_two_answers_off_fails_the_control(self, g5):
        reference = _run("nf4", 42)
        control = _run("nf4", 42, gpu=NEW_BOX, correct={**CORRECT, "mini_mmlu": 12})
        result = g5.reproduction_control(control, reference)
        assert not result["checks"]["gate_suites_within_tolerance"]
        assert result["checks"]["task_loss_within_tolerance"]

    def test_a_control_with_other_expert_weights_fails(self, g5):
        control = {**_run("nf4", 42, gpu=NEW_BOX), "expert_digest": "different"}
        assert not g5.reproduction_control(control, _run("nf4", 42))["passed"]

    def test_a_spoiled_reference_cannot_vouch_for_baselines(self, g5):
        reference = _run("nf4", 42)
        reference["train_log"][0]["grad_norm"] = 0
        result = g5.reproduction_control(_run("nf4", 42, gpu=NEW_BOX), reference)
        assert result["passed"] is False
        assert result["checks"]["reference_clean"] is False

    def test_reused_baselines_must_all_come_from_the_reference_box(self, g5, selection):
        baselines = _baselines(selection["calibration"]["train_sha256"])
        baselines[2]["gpu"] = {"uuid": "GPU-unreproduced"}
        result = _verdict(g5, selection, baselines=baselines)
        assert result["baselines"]["source"] == "not_reproduced"
        assert all(arm["outcome"] == "no_verdict" for arm in result["arms"].values())
        assert result["decision"]["g5"] == "repeat"

    def test_an_incomplete_baseline_set_cannot_vouch_for_mixed_arms(self, g5, selection):
        baselines = [run for run in _baselines(selection["calibration"]["train_sha256"])
                     if (run["arm"], run["seed"]) != ("q2_0", 7)]
        result = _verdict(g5, selection, baselines=baselines)
        assert result["baselines"]["verdict"]["outcome"] == "no_verdict"
        assert all(arm["outcome"] == "no_verdict" for arm in result["arms"].values())
        assert result["decision"]["g5"] == "repeat"

    def test_a_counted_nf4_reference_must_match_the_control(self, g5, selection):
        baselines = _baselines(selection["calibration"]["train_sha256"])
        original = next(run for run in baselines if (run["arm"], run["seed"]) == ("nf4", 42))
        original["train_log"][0]["grad_norm"] = 0
        repeat = deepcopy(original)
        repeat.update(attempt=2, batch_order_sha256="different-control-order")
        repeat["train_log"][0]["grad_norm"] = 0.5
        baselines.append(repeat)
        result = _verdict(g5, selection, baselines=baselines)
        assert result["control"]["passed"] is True
        assert result["baselines"]["source"] == "not_reproduced"
        assert all(arm["outcome"] == "no_verdict" for arm in result["arms"].values())


class TestArmOutcome:
    def test_a_mixed_arm_as_good_as_nf4_passes_and_becomes_the_rule(self, g5, selection):
        nll = {arm: [27.5] * 4 for arm in selection["maps"]}
        nll["mixed_traffic_f25"] = [25.5] * 4  # ratio 1.02
        result = _verdict(g5, selection, mixed=_mixed(selection, nll))
        assert result["arms"]["mixed_traffic_f25"]["outcome"] == "pass"
        assert result["arms"]["mixed_traffic_f50"]["outcome"] == "fail"
        assert result["decision"] == {"g5": "rule", "arm": "mixed_traffic_f25",
                                      "rule": "traffic", "fraction_q2_0": 0.25,
                                      "unresolved": []}

    def test_the_larger_passing_fraction_wins(self, g5, selection):
        nll = {arm: [25.5] * 4 for arm in selection["maps"]}
        result = _verdict(g5, selection, mixed=_mixed(selection, nll))
        assert result["decision"]["fraction_q2_0"] == 0.5

    def test_every_mixed_arm_failing_stops_g5(self, g5, selection):
        nll = {arm: [26.0] * 4 for arm in selection["maps"]}  # ratio 1.04
        result = _verdict(g5, selection, mixed=_mixed(selection, nll))
        assert {arm["outcome"] for arm in result["arms"].values()} == {"fail"}
        assert result["decision"]["g5"] == "stop"

    def test_a_gate_suite_drop_of_three_fails_an_arm_with_a_good_ratio(self, g5, selection):
        mixed = _mixed(selection)
        for run in mixed:
            if run["arm"] == "mixed_sensitivity_f25":
                run["gate_suites"]["mini_mmlu"]["correct"] = 7
        result = _verdict(g5, selection, mixed=mixed)
        arm = result["arms"]["mixed_sensitivity_f25"]
        assert arm["outcome"] == "fail" and arm["fail_reasons"] == ["suite:mini_mmlu"]

    def test_the_ratio_is_against_all_nf4_not_all_q2_0(self, g5, selection):
        result = _verdict(g5, selection)
        assert result["arms"]["mixed_traffic_f25"]["mean_ratio"] == pytest.approx(1.0)
        assert result["baselines"]["verdict"]["mean_ratio"] == pytest.approx(1.1)
        used = result["arms"]["mixed_traffic_f25"]["used"]
        assert {entry["arm"] for entry in used} == {"nf4", "mixed_traffic_f25", "bf16"}

    def test_a_missing_seed_gives_no_verdict_for_that_arm_only(self, g5, selection):
        mixed = [run for run in _mixed(selection)
                 if not (run["arm"] == "mixed_traffic_f50" and run["seed"] == 7)]
        result = _verdict(g5, selection, mixed=mixed)
        assert result["arms"]["mixed_traffic_f50"]["outcome"] == "no_verdict"
        assert result["arms"]["mixed_traffic_f50"]["missing"] == [
            {"arm": "mixed_traffic_f50", "seed": 7}]
        assert result["arms"]["mixed_traffic_f25"]["outcome"] == "pass"

    def test_the_exact_mean_ratio_limit_still_passes(self, g5, selection):
        nll = {arm: [25.75] * 4 for arm in selection["maps"]}
        result = _verdict(g5, selection, mixed=_mixed(selection, nll))
        assert all(arm["outcome"] == "pass" for arm in result["arms"].values())
        assert result["decision"]["arm"] == "mixed_traffic_f50"

    def test_a_missing_arm_never_selects_the_preferred_fraction(self, g5, selection):
        mixed = [run for run in _mixed(selection) if run["arm"] != "mixed_traffic_f50"]
        result = _verdict(g5, selection, mixed=mixed)
        assert result["arms"]["mixed_traffic_f50"]["outcome"] == "no_verdict"
        assert result["decision"]["arm"] == "mixed_sensitivity_f50"
        assert result["decision"]["unresolved"] == ["mixed_traffic_f50"]

    def test_an_invalid_arm_precedes_quality_fail_and_requires_a_repeat(self, g5, selection):
        nll = {arm: [27.5] * 4 for arm in selection["maps"]}
        mixed = _mixed(selection, nll)
        next(run for run in mixed if run["arm"] == "mixed_traffic_f50")[
            "swap"]["codec_map"]["map_sha256"] = "wrong-map"
        result = _verdict(g5, selection, mixed=mixed)
        assert result["arms"]["mixed_traffic_f50"]["outcome"] == "no_verdict"
        assert result["arms"]["mixed_traffic_f50"]["fail_reasons"]
        assert result["decision"]["g5"] == "repeat"
        assert result["decision"]["arm"] is None


class TestG5ValidityRows:
    def test_a_run_over_another_map_than_selected_gives_no_verdict(self, g5, selection):
        mixed = _mixed(selection)
        for run in mixed:
            if run["arm"] == "mixed_sensitivity_f50":
                run["swap"]["codec_map"]["map_sha256"] = "other"
        result = _verdict(g5, selection, mixed=mixed)
        arm = result["arms"]["mixed_sensitivity_f50"]
        assert arm["outcome"] == "no_verdict" and not arm["validity"]["map_hash"]

    def test_codes_that_disagree_with_the_recorded_hash_give_no_verdict(self, g5, selection):
        mixed = _mixed(selection)
        for run in mixed:
            if run["arm"] == "mixed_traffic_f25":
                for codes in run["swap"]["codec_map"]["per_expert"].values():
                    codes[3] = "q2_0"
        result = _verdict(g5, selection, mixed=mixed)
        assert not result["arms"]["mixed_traffic_f25"]["validity"]["map_hash"]

    def test_unequal_bytes_between_the_two_rules_give_no_verdict(self, g5, selection):
        result = _verdict(g5, selection, bytes_record=_bytes(selection, mixed_traffic_f50=801))
        for arm in ("mixed_traffic_f50", "mixed_sensitivity_f50"):
            assert not result["arms"][arm]["validity"]["byte_budget_equal"]
            assert result["arms"][arm]["outcome"] == "no_verdict"
        assert result["arms"]["mixed_traffic_f25"]["validity"]["byte_budget_equal"]

    def test_selection_on_another_train_split_gives_no_verdict(self, g5, selection):
        other = {**selection, "calibration": {**selection["calibration"], "sha256": "other"}}
        result = _verdict(g5, other)
        assert not result["arms"]["mixed_traffic_f25"]["validity"]["selection_data_hash"]

    def test_selection_that_recorded_a_panel_hash_gives_no_verdict(self, g5, selection):
        other = {**selection, "calibration": {**selection["calibration"],
                                              "heldout_sha256": "panel"}}
        result = _verdict(g5, other)
        assert not result["arms"]["mixed_traffic_f25"]["validity"]["selection_data_hash"]

    def test_a_c5_harness_change_gives_no_verdict(self, g5, selection):
        mixed = _mixed(selection)
        mixed[0]["harness"] = {**mixed[0]["harness"], "sha256": "edited"}
        result = _verdict(g5, selection, mixed=mixed)
        assert not result["arms"][mixed[0]["arm"]]["validity"]["c5_code_unchanged"]

    def test_c5s_rows_still_apply(self, g5, selection):
        mixed = _mixed(selection)
        mixed[1]["non_expert_fingerprints"] = {"model.embed_tokens.weight": "changed"}
        result = _verdict(g5, selection, mixed=mixed)
        arm = result["arms"][mixed[1]["arm"]]
        assert arm["outcome"] == "no_verdict" and not arm["validity"]["non_expert_fingerprints"]

    def test_synthetic_and_real_records_never_mix(self, g5, selection):
        mixed = _mixed(selection)
        mixed[0]["synthetic"] = True
        with pytest.raises(ValueError, match="never mix"):
            _verdict(g5, selection, mixed=mixed)

    def test_synthetic_bytes_never_enter_a_real_verdict(self, g5, selection):
        with pytest.raises(ValueError, match="never mix"):
            _verdict(g5, selection, bytes_record={**_bytes(selection), "synthetic": True})

    def test_a_real_verdict_without_c5s_measured_bytes_gives_no_verdict(self, g5, selection):
        # g5_bytes.py without --cache-json writes no C5 checks: its sizes are unchecked
        unchecked = {**_bytes(selection), "checks": {
            "mixed_arms_of_one_fraction_equal_bytes": True}}
        result = _verdict(g5, selection, bytes_record=unchecked)
        for arm in result["arms"].values():
            assert arm["outcome"] == "no_verdict"
            assert not arm["validity"]["byte_budget_equal"]

    def test_missing_mixed_bytes_give_no_verdict_instead_of_selecting_that_arm(
        self, g5, selection,
    ):
        bytes_record = _bytes(selection)
        del bytes_record["arms"]["mixed_traffic_f50"]
        result = _verdict(g5, selection, bytes_record=bytes_record)
        for arm in ("mixed_traffic_f50", "mixed_sensitivity_f50"):
            assert result["arms"][arm]["outcome"] == "no_verdict"
            assert result["arms"][arm]["validity"]["byte_budget_equal"] is False
        assert result["decision"]["arm"] == "mixed_traffic_f25"
        assert "no_verdict" in g5.svg_plot(result, "missing bytes")

    def test_a_missing_g5_harness_never_counts_as_unchanged(self, g5, selection):
        mixed = _mixed(selection)
        for run in mixed:
            if run["arm"] == "mixed_traffic_f50":
                del run["swap"]["codec_map"]["harness"]
        result = _verdict(g5, selection, mixed=mixed)
        assert result["arms"]["mixed_traffic_f50"]["outcome"] == "no_verdict"
        assert not result["arms"]["mixed_traffic_f50"]["validity"]["g5_code_unchanged"]
        assert result["decision"]["arm"] == "mixed_sensitivity_f50"

    def test_each_arm_must_use_the_code_recorded_by_selection(self, g5, selection):
        mixed = _mixed(selection)
        for run in mixed:
            if run["arm"] == "mixed_traffic_f50":
                harness = run["swap"]["codec_map"]["harness"]
                harness["g5_train_sha256"] = "another-train"
                harness["sources"]["g5_train.py"]["sha256"] = "another-train"
        result = _verdict(g5, selection, mixed=mixed)
        assert result["arms"]["mixed_traffic_f50"]["outcome"] == "no_verdict"
        assert not result["arms"]["mixed_traffic_f50"]["validity"]["g5_code_unchanged"]
        assert result["decision"]["arm"] == "mixed_sensitivity_f50"

    def test_selection_must_use_the_same_c5_code_as_training(self, g5, selection):
        other = deepcopy(selection)
        other["harness"]["c5_train_sha256"] = "another-c5-train"
        result = _verdict(g5, other)
        assert all(arm["outcome"] == "no_verdict" for arm in result["arms"].values())
        assert all(not arm["validity"]["c5_code_unchanged"] for arm in result["arms"].values())
        assert result["decision"]["g5"] == "repeat"

    def test_same_seed_pairs_must_have_the_same_batch_order(self, g5, selection):
        mixed = _mixed(selection)
        next(run for run in mixed if run["arm"] == "mixed_traffic_f50")[
            "batch_order_sha256"] = "another-order"
        result = _verdict(g5, selection, mixed=mixed)
        assert result["arms"]["mixed_traffic_f50"]["outcome"] == "no_verdict"
        assert not result["arms"]["mixed_traffic_f50"]["validity"]["paired_batch_order"]
        assert result["decision"]["arm"] == "mixed_sensitivity_f50"


def _synthetic_inputs(g5, selection) -> dict[str, Any]:
    """Every record of a verdict labelled synthetic, the mixed arms as good as NF4."""
    train = selection["calibration"]["train_sha256"]
    records = {
        "mixed": _mixed(selection), "baselines": _baselines(train),
        "control": _run("nf4", 42, gpu=NEW_BOX), "reference": _run("nf4", 42),
    }
    for run in [*records["mixed"], *records["baselines"], records["control"],
                records["reference"]]:
        run["synthetic"] = True
        run["data"]["train_sha256"] = train
    records["selection"] = {**selection, "synthetic": True}
    records["bytes"] = {**_bytes(selection), "synthetic": True}
    return records


class TestSyntheticNeverReportable:
    def test_runs_that_pass_on_synthetic_records_give_no_verdict_and_no_decision(
        self, g5, selection,
    ):
        records = _synthetic_inputs(g5, selection)
        result = g5.verdict(
            records["mixed"], records["baselines"], control=records["control"],
            control_reference=records["reference"], selection=records["selection"],
            bytes_record=records["bytes"], synthetic=True)
        assert result["synthetic"] is True and result["reportable"] is False
        assert result["label"] == "SYNTHETIC / CPU PILOT: NO VERDICT"
        for arm in result["arms"].values():
            # the code ran the rule to the end, and the arm still has no verdict
            assert arm["pipeline_outcome"] == "pass"
            assert arm["outcome"] == "no_verdict"
            assert arm["synthetic"] is True
            assert arm["reportable"] is False
        assert result["baselines"]["verdict"]["outcome"] == "no_verdict"
        assert result["baselines"]["verdict"]["g5"] == "no_verdict"
        assert result["baselines"]["verdict"]["reportable"] is False
        assert result["decision"]["g5"] == "no_verdict" and result["decision"]["arm"] is None
        assert result["decision"]["pipeline"]["g5"] == "rule"
        assert {p["outcome"] for p in result["points"]} <= {"reference", "no_verdict"}

    def test_the_cli_says_no_verdict_in_the_json_the_chart_and_the_console(
        self, g5, selection, tmp_path, capsys,
    ):
        records = _synthetic_inputs(g5, selection)
        paths = {}
        for name in ("selection", "bytes", "control", "reference"):
            paths[name] = tmp_path / f"{name}.json"
            paths[name].write_text(json.dumps(records[name]), encoding="utf-8")
        for group in ("mixed", "baselines"):
            for run in records[group]:
                folder = tmp_path / group / f"{run['arm']}-s{run['seed']}-a1"
                folder.mkdir(parents=True)
                (folder / "run.json").write_text(json.dumps(run), encoding="utf-8")
        out = tmp_path / "out" / "verdict.json"
        assert g5.main([str(tmp_path / "mixed"), "--baselines", str(tmp_path / "baselines"),
                        "--control", str(paths["control"]),
                        "--control-reference", str(paths["reference"]),
                        "--selection", str(paths["selection"]), "--bytes", str(paths["bytes"]),
                        "--synthetic", "--out", str(out)]) == 0
        console = capsys.readouterr().out
        record = json.loads(out.read_text(encoding="utf-8"))
        assert record["reportable"] is False and record["decision"]["g5"] == "no_verdict"
        assert {arm["outcome"] for arm in record["arms"].values()} == {"no_verdict"}
        svg = (out.parent / "quality_vs_bytes.svg").read_text(encoding="utf-8")
        assert "SYNTHETIC / CPU PILOT: NO VERDICT" in svg
        assert ", pass" not in svg
        assert "NO VERDICT" in console and "G5 decision: no_verdict" in console

    def test_a_chart_from_a_verdict_json_written_before_the_relabel_says_no_verdict(
        self, g5, selection, tmp_path, capsys,
    ):
        # the 2026-10-21 pilot's verdict.json holds "pass" outcomes under synthetic: true
        result = _verdict(g5, selection)
        legacy = {**result, "synthetic": True}
        path = tmp_path / "verdict.json"
        path.write_text(json.dumps(g5.c5._json_ready(legacy)), encoding="utf-8")
        assert g5.main(["plot", str(path), "--out", str(tmp_path / "chart.svg")]) == 0
        svg = (tmp_path / "chart.svg").read_text(encoding="utf-8")
        assert "SYNTHETIC / CPU PILOT: NO VERDICT" in svg
        assert ", pass" not in svg and "no_verdict" in svg
        assert "SYNTHETIC / CPU PILOT: NO VERDICT" in capsys.readouterr().out

    def test_a_synthetic_control_never_vouches_for_a_real_reference(self, g5):
        control = {**_run("nf4", 42, gpu=NEW_BOX), "synthetic": True}
        with pytest.raises(ValueError, match="never mix"):
            g5.reproduction_control(control, _run("nf4", 42))

    def test_historical_synthetic_records_without_provenance_still_give_no_verdict(
        self, g5, selection,
    ):
        records = _synthetic_inputs(g5, selection)
        records["selection"] = deepcopy(records["selection"])
        del records["selection"]["harness"]["sources"]
        del records["bytes"]["synthetic"]
        for run in records["mixed"]:
            run["swap"]["codec_map"]["harness"].pop("sources")
        result = g5.verdict(
            records["mixed"], records["baselines"], control=records["control"],
            control_reference=records["reference"], selection=records["selection"],
            bytes_record=records["bytes"], synthetic=True)
        assert result["reportable"] is False
        assert result["decision"]["g5"] == "no_verdict"
        assert all(arm["outcome"] == "no_verdict" for arm in result["arms"].values())
        assert all(arm["pipeline_outcome"] == "no_verdict" for arm in result["arms"].values())

    def test_a_chart_is_nonreportable_when_only_the_reportable_flag_survives(
        self, g5, selection,
    ):
        result = _verdict(g5, selection)
        result.pop("synthetic")
        result["reportable"] = False
        svg = g5.svg_plot(result, "historical pilot")
        assert "SYNTHETIC / CPU PILOT: NO VERDICT" in svg
        assert ", pass" not in svg and ", fail" not in svg


def _digests(text: dict[str, bytes]) -> dict[str, dict[str, str]]:
    return {name: {"sha256": hashlib.sha256(data).hexdigest(),
                   "sha256_lf": hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()}
            for name, data in text.items()}


class TestProvenance:
    FILES = {"c5_train.py": b"a = 1\nb = 2\n", "c5_expert_swap.py": b"c = 3\n",
             "q2_0_codec.py": b"d = 4\n"}

    def _reference(self) -> dict[str, Any]:
        lf = _digests(self.FILES)
        run = _run("nf4", 42)
        run["harness"] = {"sha256": lf["c5_train.py"]["sha256"],
                          "swap_sha256": lf["c5_expert_swap.py"]["sha256"],
                          "codec_sha256": lf["q2_0_codec.py"]["sha256"]}
        return run

    def test_an_lf_checkout_matches_c5s_record(self, g5):
        result = g5.provenance(self._reference(), _digests(self.FILES))
        assert result["c5_code_matches"] and not result["eol_only"]

    def test_a_crlf_checkout_fails_and_is_named_a_line_ending_difference(self, g5):
        crlf = {name: data.replace(b"\n", b"\r\n") for name, data in self.FILES.items()}
        result = g5.provenance(self._reference(), _digests(crlf))
        assert not result["c5_code_matches"]
        assert result["eol_only"]
        assert result["crlf_files"] == sorted(self.FILES)

    def test_an_edited_c5_file_is_not_a_line_ending_difference(self, g5):
        edited = {**self.FILES, "c5_train.py": b"a = 1\nb = 3\r\n"}
        result = g5.provenance(self._reference(), _digests(edited))
        assert not result["c5_code_matches"] and not result["eol_only"]

    def test_the_cli_exits_1_when_c5s_code_differs(self, g5, tmp_path):
        reference = {**_run("nf4", 42), "harness": {"sha256": "x", "swap_sha256": "y",
                                                    "codec_sha256": "z"}}
        path = tmp_path / "reference.json"
        path.write_text(json.dumps(reference), encoding="utf-8")
        assert g5.main(["provenance", str(path), "--out", str(tmp_path / "p.json")]) == 1
        record = json.loads((tmp_path / "p.json").read_text(encoding="utf-8"))
        assert set(record["files"]) == {"c5_train.py", "c5_expert_swap.py", "q2_0_codec.py"}

    def test_the_reproduction_control_still_fails_on_crlf_hashes(self, g5):
        # the guard is not bypassed: other bytes of C5's code are another harness
        reference = self._reference()
        crlf = _digests({n: d.replace(b"\n", b"\r\n") for n, d in self.FILES.items()})
        control = {**_run("nf4", 42, gpu=NEW_BOX), "harness": {
            "sha256": crlf["c5_train.py"]["sha256"],
            "swap_sha256": crlf["c5_expert_swap.py"]["sha256"],
            "codec_sha256": crlf["q2_0_codec.py"]["sha256"]}}
        result = g5.reproduction_control(control, reference)
        assert not result["checks"]["same_c5_code"] and not result["passed"]


def test_the_plot_has_every_arm_and_c5s_points(g5, selection, tmp_path):
    result = _verdict(g5, selection)
    labels = [point["arm"] for point in result["points"]]
    assert labels[:2] == ["all_nf4", "all_q2_0"]
    assert set(labels[2:]) == set(selection["maps"])
    svg = g5.svg_plot(result, "test")
    assert svg.startswith("<svg") and svg.count('class="point"') == len(labels)
    assert svg.count('class="legend"') == len(labels)
    assert "gate: mean ratio 1.03" in svg


def test_the_cli_writes_the_verdict_and_the_plot(g5, selection, tmp_path):
    train = selection["calibration"]["train_sha256"]
    paths = {}
    for name, payload in (("selection", selection), ("bytes", _bytes(selection)),
                          ("swap", {"non_expert_fingerprints": {
                              "model.embed_tokens.weight": "aa"}})):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(payload), encoding="utf-8")
    for group, runs in (("mixed", _mixed(selection)), ("baselines", _baselines(train))):
        for run in runs:
            run["data"]["train_sha256"] = train
            folder = tmp_path / group / f"{run['arm']}-s{run['seed']}-a1"
            folder.mkdir(parents=True)
            (folder / "run.json").write_text(json.dumps(run), encoding="utf-8")
    control = _run("nf4", 42, gpu=NEW_BOX)
    control["data"]["train_sha256"] = train
    (tmp_path / "control.json").write_text(json.dumps(control), encoding="utf-8")
    reference = tmp_path / "baselines" / "nf4-s42-a1" / "run.json"
    out = tmp_path / "out" / "verdict.json"
    assert g5.main(["control", str(tmp_path / "control.json"), str(reference),
                    "--out", str(tmp_path / "out" / "control.json")]) == 0
    assert g5.main([str(tmp_path / "mixed"), "--baselines", str(tmp_path / "baselines"),
                    "--control", str(tmp_path / "control.json"),
                    "--control-reference", str(reference),
                    "--selection", str(paths["selection"]), "--bytes", str(paths["bytes"]),
                    "--swap-record", str(paths["swap"]), "--out", str(out)]) == 0
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["decision"]["g5"] == "rule"
    assert (out.parent / "quality_vs_bytes.svg").read_text(encoding="utf-8").startswith("<svg")


def test_the_control_command_exits_1_when_the_control_fails(g5, tmp_path):
    reference, control = _run("nf4", 42), _run("nf4", 42, gpu=NEW_BOX, nll=[30.0] * 4)
    for name, payload in (("reference", reference), ("control", control)):
        (tmp_path / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    assert g5.main(["control", str(tmp_path / "control.json"), str(tmp_path / "reference.json"),
                    "--out", str(tmp_path / "control-out.json")]) == 1


class TestAttemptsAndUnfinishedRecords:
    """Audit reproductions: an unfinished control raised KeyError: 'panel', and a kept
    failed mixed attempt without swap.codec_map blocked the verdict of its repeat."""

    def test_an_unfinished_control_is_a_failed_control_with_reasons(self, g5):
        control = _run("nf4", 42, gpu=NEW_BOX)
        control["status"] = "failed"
        for key in ("panel", "gate_suites"):
            del control[key]
        result = g5.reproduction_control(control, _run("nf4", 42))
        assert result["passed"] is False
        assert result["checks"] == {"control_complete": False, "reference_complete": True}
        assert "no panel" in result["incomplete"]["control"]
        assert result["incomplete"]["control"][0] == "status 'failed'"

    def test_the_control_command_exits_1_on_an_unfinished_control(self, g5, tmp_path):
        control = _run("nf4", 42, gpu=NEW_BOX)
        control["status"] = "running"
        del control["panel"]
        for name, payload in (("reference", _run("nf4", 42)), ("control", control)):
            (tmp_path / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
        out = tmp_path / "control-out.json"
        assert g5.main(["control", str(tmp_path / "control.json"),
                        str(tmp_path / "reference.json"), "--out", str(out)]) == 1
        assert json.loads(out.read_text())["checks"]["control_complete"] is False

    def test_a_verdict_with_an_unfinished_control_needs_rerun_baselines(self, g5, selection):
        control = _run("nf4", 42, gpu=NEW_BOX)
        control["status"] = "failed"
        del control["panel"]
        result = _verdict(g5, selection, control=control)
        assert result["control"]["passed"] is False
        assert result["baselines"]["source"] == "not_reproduced"
        assert all(arm["outcome"] == "no_verdict" for arm in result["arms"].values())

    def test_a_spoiled_attempt_without_a_map_does_not_block_its_repeat(self, g5, selection):
        arm = "mixed_traffic_f25"
        crashed = _run(arm, 42, gpu=NEW_BOX)
        crashed.update(status="failed", train_log=[], swap={})
        repeat = _run(arm, 42, gpu=NEW_BOX, codec_map=_map(selection, arm), attempt=2)
        mixed = [run for run in _mixed(selection)
                 if not (map_of_arm(run) == arm and run["seed"] == 42)] + [crashed, repeat]
        result = _verdict(g5, selection, mixed=mixed)
        entry = result["arms"][arm]
        assert {"arm": arm, "seed": 42, "attempt": 2} in entry["used"]
        assert any(s["attempt"] == 1 and s["seed"] == 42 for s in entry["spoiled"])
        assert entry["validity"]["map_hash"] is True
        assert entry["outcome"] == "pass"

    def test_a_finished_run_without_a_map_is_still_refused(self, g5, selection):
        stray = _run("mixed_traffic_f25", 42, gpu=NEW_BOX)
        stray["swap"] = {}
        with pytest.raises(ValueError, match="finished mixed run without a codec map"):
            _verdict(g5, selection, mixed=_mixed(selection) + [stray])

    def test_spoiled_baseline_provenance_does_not_block_a_clean_repeat(self, g5, selection):
        train = selection["calibration"]["train_sha256"]
        baselines = _baselines(train)
        spoiled = next(run for run in baselines if (run["arm"], run["seed"]) == ("nf4", 7))
        spoiled.update(status="failed", gpu={"uuid": "GPU-abandoned"})
        spoiled["harness"]["sha256"] = "abandoned-code"
        spoiled["data"]["train_sha256"] = "abandoned-data"
        repeat = _run("nf4", 7, attempt=2)
        repeat["data"]["train_sha256"] = train
        baselines.append(repeat)
        result = _verdict(g5, selection, baselines=baselines)
        assert result["baselines"]["source"] == "reused"
        assert all(arm["outcome"] == "pass" for arm in result["arms"].values())

    def test_a_spoiled_mixed_attempt_on_another_box_does_not_block_its_repeat(
        self, g5, selection,
    ):
        arm = "mixed_traffic_f50"
        mixed = _mixed(selection)
        spoiled = next(run for run in mixed if (run["arm"], run["seed"]) == (arm, 42))
        spoiled.update(status="failed", gpu={"uuid": "GPU-abandoned"})
        mixed.append(_run(arm, 42, attempt=2, gpu=NEW_BOX, codec_map=_map(selection, arm)))
        result = _verdict(g5, selection, mixed=mixed)
        assert result["arms"][arm]["outcome"] == "pass"
        assert result["decision"]["arm"] == arm

    @pytest.mark.parametrize("attempts", [(2,), (0,), (1, 3), (1, 1)])
    def test_missing_skipped_or_duplicate_attempt_numbers_never_select_a_winner(
        self, g5, selection, attempts,
    ):
        arm = "mixed_traffic_f50"
        mixed = [run for run in _mixed(selection)
                 if (run["arm"], run["seed"]) != (arm, 42)]
        for index, attempt in enumerate(attempts):
            run = _run(arm, 42, attempt=attempt, gpu=NEW_BOX, codec_map=_map(selection, arm))
            if index < len(attempts) - 1:
                run.update(status="failed", train_log=[])
            mixed.append(run)
        result = _verdict(g5, selection, mixed=mixed)
        assert result["arms"][arm]["outcome"] == "no_verdict"
        assert result["arms"][arm]["validity"]["attempt_sequence"] is False
        assert result["decision"]["arm"] == "mixed_sensitivity_f50"

    def test_baseline_repeats_need_the_kept_first_attempt(self, g5, selection):
        baselines = _baselines(selection["calibration"]["train_sha256"])
        next(run for run in baselines if (run["arm"], run["seed"]) == ("nf4", 7))[
            "attempt"] = 2
        result = _verdict(g5, selection, baselines=baselines)
        assert result["baselines"]["verdict"]["outcome"] == "no_verdict"
        assert all(arm["outcome"] == "no_verdict" for arm in result["arms"].values())
        assert result["decision"]["g5"] == "repeat"


def map_of_arm(run: dict[str, Any]) -> str | None:
    return ((run.get("swap") or {}).get("codec_map") or {}).get("arm")


class TestAnalyzerCrashesAreNotFailedChecks:
    """g5_run_all.sh reads control exit 1 as "re-run C5's arms" and provenance exit 1 as
    "C5's code differs". An analyzer that crashes on its input must exit 2 instead (an
    uncaught exception exits 1). These run the real CLI in a fresh process."""

    @staticmethod
    def _cli(*args: str) -> int:
        import subprocess

        done = subprocess.run([sys.executable, str(HARNESS / "g5_verdict.py"), *args],
                              capture_output=True, text=True, timeout=300, check=False)
        return done.returncode

    @pytest.mark.parametrize("payload", ['{"status": "complete", "arm": "nf4"', "[1, 2]", None])
    def test_an_unreadable_control_exits_2_not_1(self, tmp_path, payload):
        reference = tmp_path / "reference.json"
        reference.write_text(json.dumps(_run("nf4", 42)), encoding="utf-8")
        control = tmp_path / "control.json"
        if payload is not None:
            control.write_text(payload, encoding="utf-8")
        code = self._cli("control", str(control), str(reference), "--out",
                         str(tmp_path / "out.json"))
        assert code == 2
        assert not (tmp_path / "out.json").exists()

    def test_a_failed_control_still_exits_1(self, tmp_path):
        reference, control = _run("nf4", 42), _run("nf4", 42, gpu=NEW_BOX, nll=[30.0] * 4)
        for name, payload in (("reference", reference), ("control", control)):
            (tmp_path / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
        assert self._cli("control", str(tmp_path / "control.json"),
                         str(tmp_path / "reference.json"), "--out",
                         str(tmp_path / "out.json")) == 1

    def test_an_unreadable_provenance_reference_exits_2(self, tmp_path):
        bad = tmp_path / "run.json"
        bad.write_text('{"harness": ', encoding="utf-8")
        assert self._cli("provenance", str(bad), "--out", str(tmp_path / "p.json")) == 2


def test_the_chart_uses_the_attempts_the_verdict_counted(g5, selection):
    """Audit finding 6: bf16_point read attempt 1 of every baseline, so a spoiled nf4 seed 42
    first attempt (kept on disk, repeated once) reached the chart while the verdict used its
    repeat. The chart now reads the baseline verdict's ``used`` attempts."""
    train = selection["calibration"]["train_sha256"]
    baselines = _baselines(train)
    spoiled = next(run for run in baselines if (run["arm"], run["seed"]) == ("nf4", 42))
    spoiled["status"] = "failed"
    spoiled["panel"]["per_example"] = [{"row": r, "tokens": 10, "nll": 99.0} for r in range(4)]
    repeat = _run("nf4", 42, attempt=2)
    repeat["data"]["train_sha256"] = train
    baselines.append(repeat)
    result = _verdict(g5, selection, baselines=baselines)
    used = result["baselines"]["verdict"]["used"]
    assert {"arm": "nf4", "seed": 42, "attempt": 2} in used
    point = g5.bf16_point({"c5_bf16_ssd_bytes": 2000}, baselines, used)
    assert point["attempts"] == {"bf16": 1, "nf4": 2}
    # bf16 and the repeat both have the base NLL: the ratio is 1, not 25/99
    assert point["mean_ratio"] == pytest.approx(1.0)
