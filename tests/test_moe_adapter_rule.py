"""The section-5 rule of probe-moe-adapter-placement.md, on planted run records."""

import importlib.util
import sys
from pathlib import Path

import pytest

RULE = Path(__file__).parents[1] / "benchmarks" / "harness" / "moe_adapter_rule.py"
SUITES = ("mini_mmlu", "mini_safety")
M_PLAN = {"m": {"arms": ("A", "AE"), "seeds": (0, 1, 2)}}
OOM_LOG = ("Traceback (most recent call last):\n"
           "  File \"moe_adapter_placement.py\", line 685, in run\n"
           "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a "
           "total capacity of 94.97 GiB of which 1.10 GiB is free.\n")
# Logged once by the Qwen3.5 A pilot, which then finished: a retry, not a failure.
ALLOCATOR_RETRY_LOG = ("[W1011 03:30:00.968135379 CUDACachingAllocator.cpp:3934] memory "
                       "allocation failed with OOM on device 0 while trying to allocate "
                       "2032140288 bytes (free: 945553408, total: 101973950464).\n"
                       "run: wrote out.json status=ok\n")


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
              "peak_cuda_bytes": 1,
              "train": {"log": [{"seconds": 1.0, "loss": 2.0}, {"seconds": 2.0, "loss": 1.0}]}}
    if arm != "base":
        record["v4"] = {"passed": True}
    return record


def _records(a_nll, ae_nll=(0.80, 0.81, 0.79), a_gate=None):
    runs = [_run("base", None, 1.00)]
    runs += [_run("AE", s, v) for s, v in enumerate(ae_nll)]
    runs += [_run("A", s, v, gate=a_gate) for s, v in enumerate(a_nll)]
    return runs


def test_small_gap_is_acceptable(rule):
    arms = rule.summarise(_records(a_nll=(0.81, 0.82, 0.80)), plan=M_PLAN)["m"]["arms"]
    assert arms["A"]["verdict"] == "ACCEPTABLE"
    assert arms["AE"]["verdict"] == "ACCEPTABLE"


def test_gap_above_floor_and_noise_costs_quality(rule):
    summary = rule.summarise(_records(a_nll=(0.85, 0.86, 0.85)), plan=M_PLAN)
    assert summary["m"]["arms"]["A"]["verdict"] == "COSTS QUALITY"


def test_gap_inside_seed_noise_is_not_material(rule):
    noisy = rule.summarise(_records(a_nll=(0.80, 0.90, 0.85), ae_nll=(0.75, 0.85, 0.80)),
                           plan=M_PLAN)
    assert noisy["m"]["arms"]["A"]["verdict"] == "ACCEPTABLE"


def test_any_seed_forgetting_past_threshold_regresses(rule):
    gate = {"mini_mmlu": 0.44, "mini_safety": 0.5}  # 0.06 below the base on one suite
    summary = rule.summarise(_records(a_nll=(0.80, 0.80, 0.80), a_gate=gate),
                             plan=M_PLAN)["m"]["arms"]["A"]
    assert summary["verdict"] == "REGRESSES"
    assert summary["leg2_failed"] == ["mini_mmlu"]


def test_drop_exactly_at_threshold_still_passes(rule):
    gate = {"mini_mmlu": 0.45, "mini_safety": 0.5}
    summary = rule.summarise(_records(a_nll=(0.80, 0.80, 0.80), a_gate=gate),
                             plan=M_PLAN)["m"]["arms"]["A"]
    assert summary["verdict"] == "ACCEPTABLE"


def test_fullest_arm_that_learned_nothing_voids_the_model(rule):
    summary = rule.summarise(_records(a_nll=(0.99, 0.99, 0.99), ae_nll=(0.97, 0.96, 0.98)),
                             plan=M_PLAN)["m"]
    assert summary["validity"]["V1"] is False
    assert summary["verdict"] == "NO VERDICT"


def test_edge_learning_rate_voids_the_model(rule):
    summary = rule.summarise(_records(a_nll=(0.8, 0.8, 0.8)), plan=M_PLAN,
                             edge_lrs={"A": True})["m"]
    assert summary["validity"]["V3"] is False and summary["verdict"] == "NO VERDICT"


def test_non_finite_run_voids_the_model(rule):
    records = _records(a_nll=(0.8, 0.8, 0.8)) + [
        _run("A", 9, float("nan"), status="void_nonfinite")]
    summary = rule.summarise(records, plan=M_PLAN)["m"]
    assert summary["validity"]["V2"] is False and summary["verdict"] == "NO VERDICT"


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_eval_nll_on_an_ok_record_fails_v2(rule, bad):
    records = _records(a_nll=(0.8, 0.8, 0.8))
    next(r for r in records if r["arm"] == "A" and r["seed"] == 1)["eval"]["nll"] = bad
    summary = rule.summarise(records, plan=M_PLAN)["m"]
    assert summary["validity"]["V2"] is False and summary["verdict"] == "NO VERDICT"


@pytest.mark.parametrize("bad", [None, "0.8", "missing"])
def test_an_ok_record_without_a_numeric_eval_nll_is_refused_with_its_path(rule, bad):
    records = _records(a_nll=(0.8, 0.8, 0.8))
    record = next(r for r in records if r["arm"] == "A" and r["seed"] == 1)
    record["_path"] = "box/r/run-A-s1.json"
    if bad == "missing":
        del record["eval"]["nll"]
    else:
        record["eval"]["nll"] = bad
    with pytest.raises(ValueError, match="box/r/run-A-s1.json"):
        rule.summarise(records, plan=M_PLAN)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_training_loss_on_an_ok_record_fails_v2(rule, bad):
    records = _records(a_nll=(0.8, 0.8, 0.8))
    next(r for r in records if r["arm"] == "A" and r["seed"] == 2)["train"]["log"][1]["loss"] = bad
    summary = rule.summarise(records, plan=M_PLAN)["m"]
    assert summary["validity"]["V2"] is False and summary["verdict"] == "NO VERDICT"


def test_a_non_finite_sweep_nll_never_wins_the_rate(rule):
    sweep = rule.lr_sweep(_sweep(rule, a=(0.70, 0.60, 0.65), ae=(0.50, float("nan"), 0.45)))
    assert sweep["AE"]["winner"] is None and sweep["AE"]["status"] == "nonfinite"
    assert sweep["A"]["winner"] == 2e-4

def test_a_non_finite_eval_nll_on_an_uncounted_sweep_run_fails_v2(rule):
    records = _sweep(rule, a=(0.70, 0.60, 0.65))
    records.append(_lr_run("AE", 8e-4, float("nan")))
    analysis = rule.analyse(records, plan=SWEEP_PLAN)
    olmoe = analysis["models"]["olmoe"]
    assert olmoe["validity"]["V2"] is False and olmoe["verdict"] == "NO VERDICT"


def test_a_seed_outside_the_plan_is_not_counted(rule):
    records = _records(a_nll=(0.81, 0.82, 0.80)) + [_run("A", 7, 0.50)]
    summary = rule.summarise(records, plan=M_PLAN)["m"]
    assert summary["verdict"] == "MEASURED"
    assert summary["arms"]["A"]["seeds"] == [0, 1, 2]
    assert summary["not_counted"] == [{"arm": "A", "seed": 7, "source": None}]


def test_an_unplanned_seed_without_a_fullest_partner_raises_nothing(rule):
    plan = {"m": {"arms": ("A", "AE"), "seeds": (0, 1)}}
    records = [r for r in _records(a_nll=(0.81, 0.82, 0.80)) if r["seed"] in (None, 0, 1)]
    records.append(_run("A", 2, 0.80))
    summary = rule.summarise(records, plan=plan)["m"]
    assert summary["verdict"] == "MEASURED" and summary["arms"]["A"]["seeds"] == [0, 1]


def test_an_ok_ae_run_at_any_rate_proves_fit_against_an_oom_log(rule):
    records = _sweep(rule, a=(0.70, 0.60, 0.65)) + [_run("base", None, 1.2, model="qwen35")]
    records += [_lr_run("A", 2e-4, 0.9, seed=s, model="qwen35") for s in (0, 1)]
    records.append(_lr_run("AE", 1e-4, 0.8, model="qwen35"))  # ok, but AE's rate is 2e-4
    plan = {**SWEEP_PLAN, "qwen35": {"arms": ("A", "AE"), "seeds": (0, 1)}}
    qwen = rule.analyse(records, plan=plan, ae_oom_logs={"qwen35": OOM_LOG})["models"]["qwen35"]
    assert "ae_did_not_fit" not in qwen and qwen["verdict"] == "NO VERDICT"
    assert qwen["missing"] == ["AE seed 0", "AE seed 1"]


def test_summarise_keeps_ae_when_an_ae_run_fit_at_another_rate(rule):
    plan = {"m": {"arms": ("A", "AT", "AE"), "seeds": (0, 1)}}
    records = [_run("base", None, 1.0)] + [_run(a, s, 0.8) for a in ("A", "AT") for s in (0, 1)]
    summary = rule.summarise(records, plan=plan, ae_oom_logs={"m": OOM_LOG},
                             uncounted=[_run("AE", 0, 0.9)])["m"]
    assert "ae_did_not_fit" not in summary and summary["verdict"] == "NO VERDICT"
    assert summary["missing"] == ["AE seed 0", "AE seed 1"] and "reason" not in summary


def test_an_oom_log_is_ignored_once_every_planned_ae_run_completed(rule):
    summary = rule.summarise(_records(a_nll=(0.81, 0.82, 0.80)), plan=M_PLAN,
                             ae_oom_logs={"m": OOM_LOG})["m"]
    assert summary["verdict"] == "MEASURED" and summary["fullest"] == "AE"
    assert "reason" not in summary and "ae_did_not_fit" not in summary


def test_fullest_stays_ae_while_ae_is_planned_and_not_excluded(rule):
    records = [r for r in _records(a_nll=(0.81, 0.82, 0.80)) if r["arm"] != "AE"]
    summary = rule.summarise(records, plan=M_PLAN)["m"]
    assert summary["fullest"] == "AE" and summary["verdict"] == "NO VERDICT"
    assert summary["validity"]["V1"] is False and "arms" not in summary


def test_a_planned_model_without_any_record_is_listed_with_its_missing_runs(rule):
    plan = {**SWEEP_PLAN, "qwen35": {"arms": ("A", "AE"), "seeds": (0, 1)}}
    analysis = rule.analyse(_sweep(rule, a=(0.70, 0.60, 0.65)), plan=plan)
    qwen = analysis["models"]["qwen35"]
    assert qwen["verdict"] == "NO VERDICT"
    assert qwen["missing"] == ["base", "A seed 0", "A seed 1", "AE seed 0", "AE seed 1"]


def test_a_gap_exactly_at_the_floor_is_not_material(rule):
    assert rule.material_gap([0.81, 0.81, 0.81], [0.79, 0.79, 0.79])["material"] is False
    assert rule.material_gap([0.82, 0.82, 0.82], [0.79, 0.79, 0.79])["material"] is True


def test_learning_exactly_the_v1_floor_passes_v1(rule):
    records = [_run("base", None, 0.30)] + [_run(a, s, 0.25) for a in ("A", "AE")
                                             for s in (0, 1, 2)]
    assert rule.summarise(records, plan=M_PLAN)["m"]["validity"]["V1"] is True


def test_not_applicable_arm_is_listed_not_scored(rule):
    records = _records(a_nll=(0.8, 0.8, 0.8)) + [_run("AS", 0, 0.0, status="not_applicable")]
    summary = rule.summarise(records, plan=M_PLAN)["m"]
    assert summary["not_applicable"] == ["AS"] and "AS" not in summary["arms"]


def test_a_missing_seed_withholds_the_verdict_and_names_the_run(rule):
    records = [r for r in _records(a_nll=(0.81, 0.82, 0.80)) if (r["arm"], r["seed"]) != ("A", 2)]
    summary = rule.summarise(records, plan=M_PLAN)["m"]
    assert summary["verdict"] == "NO VERDICT" and "arms" not in summary
    assert summary["missing"] == ["A seed 2"]


def test_missing_runs_are_listed_even_when_no_planned_arm_has_a_run(rule):
    summary = rule.summarise([_run("base", None, 1.0)], plan=M_PLAN)["m"]
    assert summary["verdict"] == "NO VERDICT" and "arms" not in summary
    assert summary["missing"] == ["A seed 0", "A seed 1", "A seed 2",
                                  "AE seed 0", "AE seed 1", "AE seed 2"]


def test_two_ok_runs_for_one_planned_run_are_refused(rule):
    records = _records(a_nll=(0.81, 0.82, 0.80)) + [_run("A", 1, 0.70)]
    with pytest.raises(ValueError, match="A seed 1"):
        rule.summarise(records, plan=M_PLAN)


@pytest.mark.parametrize(
    ("ae_seeds", "ae_log", "missing", "reason"),
    [((), None, ["AE seed 0", "AE seed 1", "AE seed 2"], None),
     ((), ALLOCATOR_RETRY_LOG, ["AE seed 0", "AE seed 1", "AE seed 2"], "no CUDA out-of-memory"),
     ((0,), OOM_LOG, ["AE seed 1", "AE seed 2"], None)],
    ids=["no-flag", "allocator-retry-is-not-an-oom", "ae-fit-on-another-seed"])
def test_ae_leaves_the_fullest_arm_only_on_proof_that_it_did_not_fit(rule, ae_seeds, ae_log,
                                                                     missing, reason):
    records = [r for r in _records(a_nll=(0.81, 0.82, 0.80))
               if r["arm"] != "AE" or r["seed"] in ae_seeds]
    logs = None if ae_log is None else {"m": ae_log}
    summary = rule.summarise(records, plan=M_PLAN, ae_oom_logs=logs)["m"]
    assert summary["verdict"] == "NO VERDICT" and "arms" not in summary
    assert summary["missing"] == missing
    if reason is None:
        assert "reason" not in summary
    else:
        assert reason in summary["reason"]


def test_ae_with_oom_proof_hands_the_fullest_arm_to_at_with_the_evidence(rule):
    plan = {"m": {"arms": ("A", "AT", "AE"), "seeds": (0, 1)}}
    records = [_run("base", None, 1.0)] + [_run("AT", s, 0.8) for s in (0, 1)] + \
        [_run("A", s, 0.9) for s in (0, 1)]
    summary = rule.summarise(records, plan=plan, ae_oom_logs={"m": OOM_LOG})["m"]
    assert summary["verdict"] == "MEASURED" and summary["fullest"] == "AT"
    assert summary["ae_did_not_fit"].startswith("torch.OutOfMemoryError: CUDA out of memory.")
    assert summary["arms"]["A"]["verdict"] == "COSTS QUALITY"


def test_a_model_without_a_committed_plan_gets_no_verdict(rule):
    summary = rule.summarise(_records(a_nll=(0.81, 0.82, 0.80)), plan={})["m"]
    assert summary["verdict"] == "NO VERDICT" and "arms" not in summary


@pytest.mark.parametrize(("x_only", "f_only", "p"),
                         [(0, 6, 0.03125), (6, 0, 0.03125), (1, 9, 0.021484375), (3, 3, 1.0),
                          (0, 0, 1.0)])
def test_exact_mcnemar_p_value(rule, x_only, f_only, p):
    assert rule.mcnemar_exact(x_only, f_only) == pytest.approx(p)


def test_paired_accuracy_pairs_each_seed_with_the_same_seed_of_the_fullest_arm(rule):
    records = _records(a_nll=(0.81, 0.82, 0.80))
    for record in records:
        if record["arm"] == "A" and record["seed"] == 1:
            record["task"]["correct"] = [True, True, False, False]
        elif record["arm"] == "AE" and record["seed"] == 1:
            record["task"]["correct"] = [True, False, True, True]
        elif record["arm"] != "base":
            record["task"]["correct"] = [True, False, True, False]
    ae = {r["seed"]: r for r in records if r["arm"] == "AE"}
    records = [r for r in records if r["arm"] != "AE"] + [ae[1], ae[0], ae[2]]
    rows = rule.summarise(records, plan=M_PLAN)["m"]["arms"]
    assert rows["AE"]["paired_vs_fullest"] is None
    assert [(p["seed"], p["x_only"], p["f_only"]) for p in rows["A"]["paired_vs_fullest"]] == \
        [(0, 0, 0), (1, 1, 2), (2, 0, 0)]


def test_identical_seeds_do_not_narrow_the_accuracy_interval(rule):
    vector = [i % 5 != 0 for i in range(60)]
    records = _records(a_nll=(0.81, 0.82, 0.80))
    for record in records:
        if record["arm"] != "base":
            record["task"]["correct"] = list(vector)
    row = rule.summarise(records, plan=M_PLAN)["m"]["arms"]["A"]
    assert row["accuracy_ci95"] == rule.bootstrap_interval(vector)


@pytest.mark.parametrize(("x_only", "f_only"), [(0, 6), (1, 9), (3, 3), (7, 2)])
def test_question_clustered_sign_flip_on_one_seed_is_exact_mcnemar(rule, x_only, f_only):
    per_question = [1] * x_only + [-1] * f_only + [0] * 5
    assert rule.signflip_exact(per_question) == pytest.approx(rule.mcnemar_exact(x_only, f_only))


def test_duplicating_identical_seeds_leaves_the_clustered_p_unchanged(rule):
    one_seed = [1, -1, -1, 0, -1, 1, -1, -1]
    assert rule.signflip_exact([3 * d for d in one_seed]) == \
        pytest.approx(rule.signflip_exact(one_seed))


def test_paired_accuracy_is_clustered_by_question_across_seeds(rule):
    full = [True] * 4 + [False] * 4
    a_by_seed = {0: [False, True, True, True, False, False, False, False],
                 1: [False, True, True, True, False, False, False, False],
                 2: [True, False, True, True, False, False, False, True]}
    records = _records(a_nll=(0.81, 0.82, 0.80))
    for record in records:
        if record["arm"] == "AE":
            record["task"]["correct"] = list(full)
        elif record["arm"] == "A":
            record["task"]["correct"] = list(a_by_seed[record["seed"]])
    clustered = rule.summarise(records, plan=M_PLAN)["m"]["arms"]["A"]["paired_clustered"]
    # Per question, summed over seeds: q0 -2, q1 -1, q7 +1. Signs of (2, 1, 1): P(|S| >= 2) = 6/8.
    assert clustered["difference"] == pytest.approx(-2 / 24)
    assert (clustered["questions_x_better"], clustered["questions_f_better"]) == (1, 2)
    assert clustered["p"] == pytest.approx(0.75)


@pytest.mark.parametrize(("failing", "every"), [(("AE",), False), (("A", "AE"), True)])
def test_each_arm_meets_leg2_on_its_own(rule, failing, every):
    records = _records(a_nll=(0.81, 0.82, 0.80))
    for record in records:
        if record["arm"] in failing:
            record["gate"]["scores"]["mini_mmlu"] = 0.40  # 0.10 below the base
    summary = rule.summarise(records, plan=M_PLAN)["m"]
    assert summary["arms"]["AE"]["verdict"] == "REGRESSES"
    assert summary["arms"]["A"]["verdict"] == ("REGRESSES" if "A" in failing else "ACCEPTABLE")
    assert summary["leg2_failed_by_every_arm"] is every


@pytest.fixture()
def report():
    saved_path, saved_modules = list(sys.path), set(sys.modules)
    spec = importlib.util.spec_from_file_location("_moe_adapter_report",
                                                  RULE.parent / "moe_adapter_report.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = saved_path
        for name in set(sys.modules) - saved_modules:
            sys.modules.pop(name, None)


@pytest.mark.parametrize(("failing", "phrase"), [(("AE",), False), (("A", "AE"), True)])
def test_report_says_leg2_does_not_discriminate_only_when_every_arm_fails_it(report, failing,
                                                                             phrase):
    records = _records(a_nll=(0.81, 0.82, 0.80))
    for record in records:
        record["_path"] = f"box/r/{record['arm']}-{record['seed']}.json"
        if record["arm"] in failing:
            record["gate"]["scores"]["mini_mmlu"] = 0.40
    models = report.rule.summarise(records, plan=M_PLAN)
    no_flips = {suite: [0, 0] for suite in SUITES}
    detail = {"m": {"base_box": "box", "checks": {},
                    "arms": {arm: [no_flips] * 3 for arm in ("A", "AE")}}}
    text = report.tables({"sweep": {}, "models": models, "rejected": [], "not_counted": []},
                         detail)
    assert ("does not discriminate" in text) is phrase
    a_row = next(line for line in text.splitlines() if line.startswith("| `A` |"))
    assert a_row.endswith("**REGRESSES** |" if phrase else "**ACCEPTABLE** |")


def _report_inputs(report, records):
    for record in records:
        record["_path"] = f"box/r/{record['arm']}-{record['seed']}.json"
    no_flips = {suite: [0, 0] for suite in SUITES}
    detail = {"olmoe": {"base_box": "box", "checks": {},
                        "arms": {arm: [no_flips] for arm in ("A", "AE")}},
              "qwen35": {"base_box": "box", "checks": {},
                         "arms": {arm: [no_flips] * 2 for arm in ("A", "AE")}}}
    return detail


def test_tables_print_the_gate_status_once_and_each_model_its_rate_scope(report, rule):
    records = _sweep(rule, a=(0.70, 0.60, 0.65)) + [_run("base", None, 1.2, model="qwen35")]
    records += [_lr_run(arm, lr, 0.9, seed=s, model="qwen35")
                for arm, lr in (("A", 2e-4), ("AE", 2e-4)) for s in (0, 1)]
    plan = {**SWEEP_PLAN, "qwen35": {"arms": ("A", "AE"), "seeds": (0, 1)}}
    detail = _report_inputs(report, records)
    analysis = report.rule.analyse(records, plan=plan)
    text = report.tables(analysis, detail)
    head, models = text.split("### Learning-rate sweep", 1)
    for phrase in ("before the first run", "after the results", "this recipe only"):
        assert phrase in head
    olmoe, qwen = models.split("### olmoe:", 1)[1].split("### qwen35:", 1)
    assert "swept on this model" in olmoe.split("| Arm |", 1)[0]
    assert "transferred from OLMoE" in qwen.split("| Arm |", 1)[0]


def test_tables_list_a_seed_outside_the_plan_as_not_counted(report):
    records = _records(a_nll=(0.81, 0.82, 0.80)) + [_run("A", 7, 0.50)]
    for record in records:
        record["_path"] = f"box/r/{record['arm']}-{record['seed']}.json"
    models = report.rule.summarise(records, plan=M_PLAN)
    no_flips = {suite: [0, 0] for suite in SUITES}
    detail = {"m": {"base_box": "box", "checks": {},
                    "arms": {arm: [no_flips] * 3 for arm in ("A", "AE")}}}
    text = report.tables({"sweep": {}, "models": models, "rejected": [], "not_counted": []},
                         detail)
    assert "Seeds outside the plan, not counted: `A` seed 7" in text


def test_svg_gap_bars_span_one_sd_of_the_seed_matched_gap(report):
    full = {"nll": [0.80, 0.90, 0.85], "accuracy": [0.5] * 3, "accuracy_ci95": [0.4, 0.6]}
    row = {"nll": [0.83, 0.92, 0.88]}  # gaps 0.03, 0.02, 0.03: mean 0.02667, SD 0.00577
    low, mid, high = report._gap_band(row, full)
    assert mid == pytest.approx(0.08 / 3)
    assert high - mid == pytest.approx(mid - low) == pytest.approx(0.0057735, rel=1e-4)


def test_report_outputs_use_lf_and_end_with_a_newline(report, tmp_path, monkeypatch):
    records = _records(a_nll=(0.81, 0.82, 0.80))
    for record in records:
        record["_path"] = f"box/r/{record['arm']}-{record['seed']}.json"
    analysis = {"gate": report.rule.GATE, "sweep": {}, "rejected": [], "not_counted": [],
                "models": report.rule.summarise(records, plan=M_PLAN)}
    analysis["models"]["m"]["arms"]["A"]["accuracy_ci95"] = [0.4, 0.6]
    no_flips = {suite: [0, 0] for suite in SUITES}
    detail = {"m": {"base_box": "box", "checks": {},
                    "arms": {arm: [no_flips] * 3 for arm in ("A", "AE")}}}
    monkeypatch.setattr(report.rule, "load_runs", lambda *_a, **_k: records)
    monkeypatch.setattr(report.rule, "analyse", lambda *_a, **_k: analysis)
    monkeypatch.setattr(report, "leg2_detail", lambda *_a: detail)
    out = tmp_path / "out"
    report.main(["runs", str(out), "--harness-sha256", "h", "--split-sha256", "s",
                 "--hot", "m=x"])
    for name in ("summary.json", "tables.md", "curve-m.svg"):
        data = (out / name).read_bytes()
        assert b"\r" not in data and data.endswith(b"\n"), name


def test_a_run_that_failed_the_zero_adapter_check_fails_v4(rule):
    void = _run("A", 3, 0.8, status="void_v4")
    void["v4"] = {"passed": False}
    summary = rule.summarise(_records(a_nll=(0.80, 0.81, 0.79)) + [void], plan=M_PLAN)["m"]
    assert summary["validity"]["V4"] is False and summary["verdict"] == "NO VERDICT"


def test_per_seed_columns_follow_seed_order(rule):
    runs = [_run("base", None, 1.0)]
    for seed, accuracy in ((2, 0.2), (0, 0.0), (1, 0.1)):
        for arm in ("A", "AE"):
            run = _run(arm, seed, (0.80, 0.81, 0.82)[seed])
            run["task"]["accuracy"] = accuracy
            runs.append(run)
    row = rule.summarise(runs, plan=M_PLAN)["m"]["arms"]["A"]
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


SWEEP_PLAN = {"olmoe": {"arms": ("A", "AE"), "seeds": (0,)}}


def test_sweep_winner_is_the_lowest_nll_inside_the_grid(rule):
    sweep = rule.lr_sweep(_sweep(rule, a=(0.70, 0.60, 0.65)))
    assert sweep["A"]["winner"] == 2e-4 and sweep["A"]["status"] == "ok"


def test_grid_edge_winner_withholds_the_verdict_until_its_extension_runs(rule):
    analysis = rule.analyse(_sweep(rule, a=(0.60, 0.65, 0.70)), plan=SWEEP_PLAN)
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
    analysis = rule.analyse(records, plan=SWEEP_PLAN)
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
    plan = {**SWEEP_PLAN, "qwen35": {"arms": ("A", "AS", "AE"), "seeds": (0, 1)}}
    analysis = rule.analyse(records, plan=plan)
    qwen = analysis["models"]["qwen35"]
    assert qwen["verdict"] == "MEASURED" and qwen["arms"]["AS"]["seeds"] == [0, 1]
    skipped = {(r["model"], r["arm"], r["lr"]) for r in analysis["not_counted"]}
    assert skipped == {("olmoe", "A", 1e-4), ("olmoe", "A", 4e-4), ("olmoe", "AE", 1e-4),
                       ("olmoe", "AE", 4e-4), ("qwen35", "AS", 4e-4)}


def test_the_committed_plan_withholds_a_qwen_verdict_until_every_arm_and_seed_ran(rule):
    records = _sweep(rule, a=(0.70, 0.60, 0.65)) + [_run("base", None, 1.2, model="qwen35")]
    records += [_lr_run(arm, 2e-4, 0.9, seed=s, model="qwen35")
                for arm in ("A", "AS", "AE") for s in (0, 1)]
    qwen = rule.analyse(records)["models"]["qwen35"]
    assert qwen["verdict"] == "NO VERDICT" and qwen["missing"] == ["AT seed 0", "AT seed 1"]


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
