"""The E1 confirmation rule (probe-moe-adapter-placement.md section 11), on planted records."""

import importlib.util
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).parents[1] / "benchmarks" / "harness"
CONFIRM_RULE = HARNESS / "moe_adapter_confirm_rule.py"
SUITES = ("mini_mmlu", "mini_safety")
ROWS = 4
OLMOE_RATES = {"A": 2e-4, "AT": 1e-4, "AE": 1e-4}


@pytest.fixture(scope="module")
def cr():
    spec = importlib.util.spec_from_file_location("_moe_adapter_confirm_rule", CONFIRM_RULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def _rows(nll, tokens=(10,) * ROWS):
    """Per-row block whose token-weighted NLL is exactly ``nll``."""
    return {"nll_sum": [nll * t for t in tokens], "tokens": list(tokens)}


def _record(arm, seed, e_nll, c_nll, *, model="olmoe", lr=None, gate=None, status="ok",
            correct_e=None, correct_c=None, outputs=None):
    if lr is None:
        lr = 0.0 if arm == "base" else OLMOE_RATES.get(arm, 2e-4)
    correct_e = [True, False] * (ROWS // 2) if correct_e is None else correct_e
    correct_c = [True, False] * (ROWS // 2) if correct_c is None else correct_c
    record = {
        "model": model, "arm": arm, "lr": lr, "seed": seed, "status": status,
        "_path": f"confirm-{model}-{arm}-lr{lr:g}-s{seed}.json",
        "eval": {"nll": e_nll, "tokens": 10 * ROWS},
        "task": {"accuracy": sum(correct_e) / ROWS, "correct": correct_e,
                 "outputs": list(outputs or ["e"] * ROWS)},
        "gate": {"scores": dict(gate or {s: 0.5 for s in SUITES}),
                 "outputs": {"p": "answer"}},
        "trainable_parameters": 10, "adapter_bytes_adamw_fp32": 160, "peak_cuda_bytes": 1,
        "confirm": {"split_sha256": "c", "wrapper_sha256": "w",
                    "eval": {"nll": c_nll, "tokens": 10 * ROWS},
                    "per_row": {"E": _rows(e_nll), "C": _rows(c_nll)},
                    "task": {"accuracy": sum(correct_c) / ROWS, "correct": correct_c,
                             "outputs": ["c"] * ROWS},
                    "seconds": 1.0},
    }
    if arm != "base":
        record["v4"] = {"passed": True}
        record["train"] = {"steps": 2, "log": [
            {"step": 0, "loss": 2.0, "grad_norm": 1.0, "seconds": 1.0, "tokens": 5},
            {"step": 1, "loss": 1.0, "grad_norm": 1.0, "seconds": 2.0, "tokens": 5}]}
    return record


def _olmoe(a_e, a_c, ae_e=(0.80, 0.81, 0.79), ae_c=(0.80, 0.81, 0.79)):
    runs = [_record("base", 0, 1.0, 1.0)]
    for s in range(3):
        runs.append(_record("AE", s, ae_e[s], ae_c[s]))
        runs.append(_record("AT", s, ae_e[s], ae_c[s]))
        runs.append(_record("A", s, a_e[s], a_c[s]))
    return runs


def test_c_verdict_reads_c_nll(cr):
    runs = _olmoe(a_e=(0.81, 0.82, 0.80), a_c=(0.86, 0.87, 0.86))
    e, c = cr.verdicts(runs, "olmoe", "E"), cr.verdicts(runs, "olmoe", "C")
    assert e["arms"]["A"]["verdict"] == "ACCEPTABLE"
    assert c["arms"]["A"]["verdict"] == "COSTS QUALITY"
    comparison = cr.compare(e, c)["arms"]["A"]
    assert comparison["verdict_holds"] is False
    assert comparison["gap_holds"] is False


def test_non_frozen_rate_never_enters_a_verdict(cr):
    runs = _olmoe(a_e=(0.81, 0.82, 0.80), a_c=(0.81, 0.82, 0.80))
    runs.append(_record("A", 0, 5.0, 5.0, lr=4e-4))  # would cost quality if counted
    runs.append(_record("AT", 1, 0.8, 0.8, lr=2e-4, status="void_oom"))
    for s in ("E", "C"):
        v = cr.verdicts(runs, "olmoe", s)
        assert v["verdict"] == "MEASURED"
        assert v["arms"]["A"]["nll"] == [0.81, 0.82, 0.80]
        assert v["arms"]["A"]["verdict"] == "ACCEPTABLE"
        assert v["void"] == []


def test_gap_holds_catches_sign_flip_and_materiality_change_separately(cr):
    flip = _olmoe(a_e=(0.81, 0.82, 0.80), a_c=(0.79, 0.80, 0.78))
    a = cr.compare(cr.verdicts(flip, "olmoe", "E"), cr.verdicts(flip, "olmoe", "C"))["arms"]["A"]
    assert a["verdict_holds"] is True and a["gap_holds"] is False
    assert a["seed_sign_agrees"] == {"0": False, "1": False, "2": False}
    material = _olmoe(a_e=(0.81, 0.82, 0.80), a_c=(0.85, 0.86, 0.85))
    b = cr.compare(cr.verdicts(material, "olmoe", "E"),
                   cr.verdicts(material, "olmoe", "C"))["arms"]["A"]
    assert b["task_gap_E"]["gap"] > 0 and b["task_gap_C"]["gap"] > 0
    assert b["gap_holds"] is False
    same = _olmoe(a_e=(0.81, 0.82, 0.80), a_c=(0.815, 0.825, 0.805))
    c = cr.compare(cr.verdicts(same, "olmoe", "E"), cr.verdicts(same, "olmoe", "C"))["arms"]["A"]
    assert c["gap_holds"] is True


def test_no_verdict_side_gives_no_holds(cr):
    runs = [r for r in _olmoe(a_e=(0.81, 0.82, 0.80), a_c=(0.81, 0.82, 0.80))
            if not (r["arm"] == "A" and r["seed"] == 2)]
    comparison = cr.compare(cr.verdicts(runs, "olmoe", "E"), cr.verdicts(runs, "olmoe", "C"))
    assert comparison["verdict"] == "NO VERDICT"
    assert "arms" not in comparison
    assert "A seed 2" in comparison["C"]["missing"]


def _three_row(arm, e_sums, e_tokens, c_sums, c_tokens, correct_e, correct_c, seed=0):
    r = _record(arm, seed, sum(e_sums) / sum(e_tokens), sum(c_sums) / sum(c_tokens),
                correct_e=correct_e, correct_c=correct_c)
    r["eval"]["tokens"] = sum(e_tokens)
    r["confirm"]["eval"]["tokens"] = sum(c_tokens)
    r["confirm"]["per_row"] = {"E": {"nll_sum": e_sums, "tokens": e_tokens},
                               "C": {"nll_sum": c_sums, "tokens": c_tokens}}
    r["task"]["correct"], r["confirm"]["task"]["correct"] = correct_e, correct_c
    return r


def test_contrast_token_weighted_point_estimates(cr):
    tokens_e, tokens_c = [1, 2, 7], [2, 3, 5]
    base = _three_row("base", [2.0, 2.0, 7.0], tokens_e, [1.0, 3.0, 6.0], tokens_c,
                      [True, False, False], [True, True, False])
    arm = _three_row("AE", [1.0, 1.0, 5.0], tokens_e, [1.0, 2.0, 3.0], tokens_c,
                     [True, True, False], [True, True, True])
    out = cr.contrast([base, arm], draws=200)
    b, a = out["base"], out["arms"]["AE"]
    assert b["nll_E"] == pytest.approx(11 / 10) and b["nll_C"] == pytest.approx(10 / 10)
    assert a["nll_E"] == pytest.approx(7 / 10) and a["nll_C"] == pytest.approx(6 / 10)
    assert a["contrast_nll"] == pytest.approx((0.7 - 0.6) - (1.1 - 1.0))
    assert a["accuracy_E"] == pytest.approx(2 / 3) and a["accuracy_C"] == pytest.approx(1.0)
    assert a["contrast_accuracy"] == pytest.approx((2 / 3 - 1.0) - (1 / 3 - 2 / 3))


def _shifted_pair(shift_arm, shift_base, n=12):
    import random

    rng = random.Random(7)
    tokens = [rng.randrange(5, 15) for _ in range(n)]
    base_e = [t * rng.uniform(0.9, 1.1) for t in tokens]
    arm_e = [t * rng.uniform(0.5, 0.9) for t in tokens]
    correct = [rng.random() < 0.5 for _ in range(n)]
    correct_arm = [rng.random() < 0.7 for _ in range(n)]
    base = _three_row("base", base_e, tokens, [v + shift_base * t for v, t in zip(base_e, tokens)],
                      tokens, correct, correct)
    arm = _three_row("AE", arm_e, tokens, [v + shift_arm * t for v, t in zip(arm_e, tokens)],
                     tokens, correct_arm, [not v for v in correct_arm])
    return base, arm


def test_identical_shift_gives_zero_contrast(cr):
    base, arm = _shifted_pair(0.1, 0.1)
    row = cr.contrast([base, arm], draws=200)["arms"]["AE"]
    assert row["contrast_nll"] == pytest.approx(0.0, abs=1e-12)


def test_contrast_interval_deterministic_and_brackets_point(cr):
    base, arm = _shifted_pair(0.3, 0.0)
    first = cr.contrast([base, arm], draws=500, seed=0)["arms"]["AE"]
    second = cr.contrast([base, arm], draws=500, seed=0)["arms"]["AE"]
    assert first["contrast_nll_ci95"] == second["contrast_nll_ci95"]
    low, high = first["contrast_nll_ci95"]
    assert low < first["contrast_nll"] < high
    low, high = first["contrast_accuracy_ci95"]
    assert low < first["contrast_accuracy"] < high


def test_gap_difference_against_fullest(cr):
    base, ae = _shifted_pair(0.0, 0.0)
    e_rows, c_rows = ae["confirm"]["per_row"]["E"], ae["confirm"]["per_row"]["C"]
    a = _three_row("A", [v + 0.2 * t for v, t in zip(e_rows["nll_sum"], e_rows["tokens"])],
                   e_rows["tokens"], c_rows["nll_sum"], c_rows["tokens"], ae["task"]["correct"],
                   ae["confirm"]["task"]["correct"])
    row = cr.contrast([base, ae, a], draws=100)["arms"]["A"]
    assert row["gap_difference_nll"] == pytest.approx(0.2)
    assert "gap_difference_nll" not in cr.contrast([base, ae, a], draws=10)["arms"]["AE"]


def test_per_row_check_refuses_mismatch(cr):
    record = _record("A", 0, 0.8, 0.9)
    cr.check_per_row(record)
    record["confirm"]["per_row"]["C"]["nll_sum"][0] += 1.0
    with pytest.raises(ValueError, match="C per-row NLL"):
        cr.check_per_row(record)
    record = _record("A", 0, 0.8, 0.9)
    record["confirm"]["per_row"]["E"]["tokens"][0] += 1
    with pytest.raises(ValueError, match="E per-row tokens"):
        cr.check_per_row(record)


def test_reproduction(cr):
    published = _record("A", 0, 0.8, 0.9)
    new = _record("A", 0, 0.8, 0.9)
    new["train"]["log"][1]["seconds"] = 99.0
    assert cr.reproduce(new, published)["reproduced"] is True
    loss = _record("A", 0, 0.8, 0.9)
    loss["train"]["log"][1]["loss"] = 1.0001
    assert cr.reproduce(loss, published) == {**cr.reproduce(loss, published),
                                             "reproduced": False, "differ": ["train.log"]}
    gate = _record("A", 0, 0.8, 0.9)
    gate["gate"]["outputs"]["p"] = "other"
    assert cr.reproduce(gate, published)["differ"] == ["gate.outputs"]
    rows = cr.reproduction([new, _record("A", 1, 0.8, 0.9)], [published])
    assert rows[0]["reproduced"] is True and rows[1]["reproduced"] is None


def _ram(arm, seed, peak, params, bpp=16, e_nll=0.80, c_nll=0.80, gate=None):
    r = _record(arm, seed, e_nll, c_nll, model="qwen35", lr=2e-4, gate=gate)
    r["regime"] = "ram_tier"
    r["ram_tier"] = {"peak_cuda_bytes": peak, "trainable_parameters": params,
                     "bytes_per_parameter": bpp}
    return r


def test_ram_memory_rule_tolerance(cr):
    extra = 1_000_000
    within = cr.MEMORY_TOLERANCE_BYTES - 1
    ram = [_ram("A", s, 10**9, 5 * extra) for s in (0, 1)]
    ram += [_ram("AS", s, 10**9 + 16 * extra + within, 6 * extra) for s in (0, 1)]
    mem = cr.ram_memory(ram)
    assert mem["predicted_e1"] == 16 * extra and mem["holds_e1_rule"] is True
    assert mem["residual_e1"] == {"0": within, "1": within}
    over = [r for r in ram if r["arm"] == "A"]
    over += [_ram("AS", s, 10**9 + 16 * extra + cr.MEMORY_TOLERANCE_BYTES + 1, 6 * extra)
             for s in (0, 1)]
    assert cr.ram_memory(over)["holds_e1_rule"] is False
    big = 10 * extra  # the 16-byte rule misses this pair by 6 * big bytes
    dtype = [_ram("A", s, 10**9, 5 * big, bpp=10) for s in (0, 1)]
    dtype += [_ram("AS", s, 10**9 + 10 * big, 6 * big, bpp=10) for s in (0, 1)]
    mem = cr.ram_memory(dtype)
    assert mem["holds_dtype_rule"] is True and mem["holds_e1_rule"] is False


def test_ram_memory_refuses_mismatched_bytes_per_parameter(cr):
    ram = [_ram("A", 0, 10, 5, bpp=16), _ram("AS", 0, 26, 6, bpp=12)]
    with pytest.raises(ValueError, match="bytes_per_parameter"):
        cr.ram_memory(ram)


def _resident():
    runs = [_record("base", 0, 1.0, 1.0, model="qwen35")]
    runs += [_record("AE", s, 0.80 + 0.01 * s, 0.80 + 0.01 * s, model="qwen35", lr=1e-4)
             for s in (0, 1)]
    runs += [_record(a, s, 0.80, 0.80, model="qwen35", lr=2e-4) for a in ("A", "AS")
             for s in (0, 1)]
    return runs


def test_ram_verdicts(cr):
    forgetting = {"mini_mmlu": 0.44, "mini_safety": 0.5}
    ram = [_ram("A", s, 1, 1, gate=forgetting) for s in (0, 1)]
    ram += [_ram("AS", s, 1, 1, e_nll=0.86, c_nll=0.80) for s in (0, 1)]
    e = cr.ram_verdicts(ram, _resident(), "E")
    assert e["verdict"] == "MEASURED"
    assert e["arms"]["A"]["verdict"] == "REGRESSES"
    assert e["arms"]["AS"]["verdict"] == "COSTS QUALITY"
    c = cr.ram_verdicts(ram, _resident(), "C")
    assert c["arms"]["AS"]["verdict"] == "ACCEPTABLE"


EXPECT = {"harness_sha256": None, "split_sha256": None, "hot_sha256": {},
          "confirm_split_sha256": "c"}
PARITY_FAILED = {"status": "not_passed", "passed": False, "_path": "parity-qwen35.json",
                 "quant": "nf4", "tier": "ram", "device": "cuda:0", "harness_sha256": "ab" * 32,
                 "bit_exact": {"losses": False}, "rule": {"grad_relative_l2": 0.01},
                 "steps": [{"step": s, "streamed": 1.0, "resident": 1.1, "relative": 0.09}
                           for s in range(4)],
                 "gradient_step0": {"global_relative_l2": 0.5, "min_cosine": 0.2, "max_abs": 1.0}}
PARITY_PASSED = {
    **PARITY_FAILED, "status": "ok", "passed": True, "_path": "synthetic-passing-parity.json",
    "bit_exact": {"losses": True},
    "steps": [{"step": s, "streamed": 1.0, "resident": 1.0, "relative": 0.0} for s in range(4)],
    "gradient_step0": {"global_relative_l2": 0.0, "min_cosine": 1.0, "max_abs": 0.0},
}


def _ram_analysis(cr, parity):
    ram = [_ram("A", s, 10**9, 50, gate={"mini_mmlu": 0.44, "mini_safety": 0.5}) for s in (0, 1)]
    ram += [_ram("AS", s, 10**9 + 16, 51, e_nll=0.86) for s in (0, 1)]
    return cr.analyse(_resident(), [], ram, expect=EXPECT, parity=parity, draws=10)["ram_tier"]


def test_failed_parity_withholds_ram_verdicts(cr):
    ram = _ram_analysis(cr, PARITY_FAILED)
    assert ram["parity"]["counted"] is False
    for s in ("E", "C"):
        assert ram[s]["verdict"] == "NOT COUNTED"
        assert ram[s]["verdict_if_counted"] == "MEASURED"
        assert ram[s]["arms"]["A"]["verdict"] == "REGRESSES"
    assert ram["E"]["arms"]["AS"]["verdict"] == "COSTS QUALITY"
    assert ram["memory"]["predicted_e1"] == 16 and ram["memory"]["parity_counted"] is False


def test_passing_parity_counts_ram_verdicts(cr):
    ram = _ram_analysis(cr, PARITY_PASSED)
    assert ram["parity"]["counted"] is True
    assert ram["E"]["verdict"] == ram["C"]["verdict"] == "MEASURED"
    assert "verdict_if_counted" not in ram["E"]
    assert ram["memory"]["parity_counted"] is True


@pytest.mark.parametrize("missing_pairs", [
    [("A", 0)], [("A", 1)], [("AS", 0)], [("AS", 1)], [("A", 1), ("AS", 1)],
])
def test_passing_parity_requires_every_ram_arm_seed(cr, missing_pairs):
    ram = [_ram(a, s, 1, 1) for a in ("A", "AS") for s in (0, 1)
           if (a, s) not in missing_pairs]
    result = cr.analyse(_resident(), [], ram, expect=EXPECT, parity=PARITY_PASSED, draws=10)
    assert result["ram_tier"]["parity"]["counted"] is True
    for task_set in ("E", "C"):
        verdict = result["ram_tier"][task_set]
        assert verdict["verdict"] == "NO VERDICT"
        assert verdict["missing"] == [f"RAM-tier {a} seed {s}" for a, s in missing_pairs]
        assert "arms" not in verdict


@pytest.mark.parametrize(("arm", "replacement"), [
    ("A", {"lr": 1e-4}), ("AS", {"lr": 1e-4}),
    ("A", {"seed": 2}), ("AS", {"status": "void_oom"}),
])
def test_unplanned_or_void_record_cannot_fill_ram_seed(cr, arm, replacement):
    ram = [_ram(a, s, 1, 1) for a in ("A", "AS") for s in (0, 1)]
    next(r for r in ram if r["arm"] == arm and r["seed"] == 1).update(replacement)
    result = cr.analyse(_resident(), [], ram, expect=EXPECT, parity=PARITY_PASSED, draws=10)
    assert result["ram_tier"]["parity"]["counted"] is True
    for task_set in ("E", "C"):
        verdict = result["ram_tier"][task_set]
        assert verdict["verdict"] == "NO VERDICT"
        assert verdict["missing"] == [f"RAM-tier {arm} seed 1"]
        assert "arms" not in verdict


def test_ram_verdicts_exclude_unplanned_rates_and_seeds(cr):
    ram = [_ram(a, s, 1, 1) for a in ("A", "AS") for s in (0, 1)]
    ram += [{**_ram("A", 0, 1, 1, e_nll=5.0, c_nll=5.0), "lr": 1e-4},
            _ram("AS", 2, 1, 1, e_nll=5.0, c_nll=5.0)]
    for task_set in ("E", "C"):
        verdict = cr.ram_verdicts(ram, _resident(), task_set)
        assert verdict["verdict"] == "MEASURED"
        for arm in ("A", "AS"):
            assert verdict["arms"][arm]["verdict"] == "ACCEPTABLE"
            assert verdict["arms"][arm]["seeds"] == [0, 1]
            assert verdict["arms"][arm]["nll"] == [0.8, 0.8]


def test_smoke_parity_is_not_counted(cr):
    smoke = {**PARITY_PASSED, "quant": "none", "device": "cpu", "steps": PARITY_PASSED["steps"][:3]}
    ram = _ram_analysis(cr, smoke)
    assert ram["parity"]["counted"] is False
    assert "quant" in ram["parity"]["reason"]
    assert ram["E"]["verdict"] == ram["C"]["verdict"] == "NOT COUNTED"


def test_missing_parity_record_withholds_ram_verdicts(cr):
    ram = _ram_analysis(cr, None)
    assert ram["parity"] == {"counted": False, "reason": "no parity record"}
    assert ram["E"]["verdict"] == ram["C"]["verdict"] == "NOT COUNTED"
    assert ram["C"]["gate_reason"] == "no parity record"


def test_per_row_check_runs_before_provenance(cr):
    bad = _record("A", 0, 0.8, 0.9)
    bad["harness_sha256"] = "another harness"
    bad["confirm"]["per_row"]["E"]["nll_sum"][0] += 1.0
    with pytest.raises(ValueError, match="E per-row NLL"):
        cr.analyse([bad], [], expect=EXPECT, draws=10)
