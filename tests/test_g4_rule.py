"""The G4 verdict (benchmarks/harness/g4_rule.py) on synthetic records: every gate's
pass and fail, and every validity row's no-verdict path."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tests.test_l2l_probe_harness import _load

N_LAYERS = 4
STEPS = 8
N_SETS = 2
K = 4
CHUNK = [1, 8, 16]
ROWS_PER_LAYER = N_SETS * K * 8
TENSORS = 8


@pytest.fixture()
def rule():
    return _load("g4_rule")


def _layer_rows(restore=0.01, worst=0.05, element=0.003, branch=0.02):
    return [
        {"layer": layer,
         "restore": {"rms_row_rel": restore, "worst_row_rel": worst,
                     "max_abs_over_absmax": element, "rows": ROWS_PER_LAYER},
         "branch": {"rms_row_rel": branch, "rows": ROWS_PER_LAYER},
         "output": {"rms_row_rel": restore, "rows": ROWS_PER_LAYER}}
        for layer in range(N_LAYERS)
    ]


def _cmp(equal=True):
    return {"tensors": TENSORS, "unequal": [] if equal else ["x"], "all_zero": [],
            "equal": equal}


def _grad(global_rel_l2, min_cosine):
    """A gradient agreement as g4_activation_codec.grad_agreement writes it: the per-tensor
    table covers every tensor and its smallest cosine is ``min_cosine``."""
    per = {f"t{i}": {"cosine": 1.0 if i else min_cosine, "rel_l2": global_rel_l2}
           for i in range(TENSORS)}
    return {"tensors": TENSORS, "global_rel_l2": global_rel_l2, "min_cosine": min_cosine,
            "per_tensor": per}


def _error(**over):
    codecs = {
        "int8": {"losses_equal": True, "grad": _grad(0.01, 0.9995)},
        "fp8_e4m3": {"losses_equal": True, "grad": _grad(0.01, 0.9995)},
        "int4": {"losses_equal": True, "grad": _grad(0.6, 0.3)},
    }
    sets = [{"kind": "error_set", "set": index, "ga_vs_l2l": _cmp(), "a_vs_a": _cmp(),
             "null_control": _cmp(), "ga_vs_l2l_losses_equal": True,
             "a_vs_a_losses_equal": True, "null_control_losses_equal": True,
             "codecs": copy.deepcopy(codecs)} for index in range(N_SETS)]
    layers = {"kind": "layers", "codecs": {
        "int8": _layer_rows(), "fp8_e4m3": _layer_rows(element=0.03),
        "int4": _layer_rows(restore=0.3, worst=0.6, element=0.07, branch=0.5)}}
    payload = {"meta": {"device": "cuda", "verdict_eligible": True, "label": "box",
                        **copy.deepcopy(_SHARED_META),
                        "args": {"error_sets": N_SETS}, "n_layers": N_LAYERS,
                        "chunk_shape": CHUNK,
                        "config": {"codec": "bf16", "seed": 1, "steps": 0, "lr": 2e-4, "k": K,
                                   "model_revision": "60b99dc"}},
               "records": sets + [layers]}
    for key, value in over.items():
        payload[key] = value
    return payload


_SHARED_META = {
    "data": {"sha256": "d", "train_ids_sha256": "t", "panel_ids_sha256": "p",
             "split_sha256": {"heldout": "h", "train": "s"}},
    "harness_sha256": {"g4_probe.py": "x"}, "git_head": "abc", "versions": {"torch": "2"},
    "gpu": "RTX", "model_files_sha256": {"config.json": "c"},
}


def _arm(codec, seed, *, task_loss=1.50, curve=None, per_example=None):
    curve = curve or [2.0 - 0.05 * step for step in range(STEPS)]
    per_example = per_example or [
        {"row": row, "nll": task_loss * 10.0, "tokens": 10} for row in range(20)]
    log = [{"step": step, "loss": loss, "grad_norm": 0.5, "lr": 1e-4}
           for step, loss in enumerate(curve)]
    log[0]["mb_losses"] = [2.1, 1.9, 2.0, 2.0] if seed == 1 else [2.2, 1.8, 2.0, 2.0]
    return {
        "adapter": {"dir": "x", "sha256": {"adapter_model.safetensors": f"{codec}{seed}"}},
        "meta": {
            "device": "cuda", "verdict_eligible": True, "label": "box",
            **copy.deepcopy(_SHARED_META),
            "config": {"codec": codec, "seed": seed, "steps": STEPS, "lr": 2e-4, "k": 4,
                       "model_revision": "60b99dc"},
        },
        "log": log,
        "final": {"task_loss": task_loss, "examples": len(per_example),
                  "tokens": sum(ex["tokens"] for ex in per_example), "per_example": per_example},
        "memory": {"store": {"measured_bytes_per_token_per_layer": 4100.0,
                             "arithmetic_bytes_per_token_per_layer": 4100.0}},
    }


SUITE_N = {"mini_mmlu": 26, "mini_common_sense": 24, "mini_instruction": 24,
           "mini_arithmetic": 40}


TASK_LOSS = {("bf16", 1): 1.50, ("int8", 1): 1.503, ("fp8_e4m3", 1): 1.502, ("bf16", 2): 1.51}
PARITY_ROWS = 4


def _suites(codec, seed, *, parity_loss=None, **correct):
    """A suites record. Its resident parity rows mirror rows 0-3 of its training arm's
    panel (filled in by ``_write`` from the arm actually written) unless ``parity_loss``
    sets them explicitly."""
    scores = {"mini_mmlu": 15, "mini_common_sense": 18, "mini_instruction": 12,
              "mini_arithmetic": 30, **correct}
    rows = None if parity_loss is None else [
        {"row": row, "nll": parity_loss * 10.0, "tokens": 10} for row in range(PARITY_ROWS)]
    return {
        "meta": {"device": "cuda", "verdict_eligible": True, "label": "box",
                 "adapter_sha256": {"adapter_model.safetensors": f"{codec}{seed}"},
                 "max_new_tokens": 64, "quant": "nf4"},
        "base_parity": {"adapter": {"tensors": 8, "mismatch": [], "mismatched": 0},
                        "per_example_resident": rows},
        "suites": {name: {"n": SUITE_N[name], "correct": scores[name]} for name in SUITE_N},
    }


def _write(tmp_path: Path, error=None, arms=None, timing=None, drop=(), suites=None):
    files = {"error.json": error or _error()}
    default_arms = {
        "train-bf16-seed1.json": _arm("bf16", 1, task_loss=1.50),
        "train-int8-seed1.json": _arm("int8", 1, task_loss=1.503),
        "train-fp8_e4m3-seed1.json": _arm("fp8_e4m3", 1, task_loss=1.502),
        "train-bf16-seed2.json": _arm("bf16", 2, task_loss=1.51,
                                      curve=[2.01 - 0.05 * s for s in range(STEPS)]),
    }
    default_arms.update(arms or {})
    files.update(default_arms)
    default_suites = {
        "suites-bf16-seed1.json": _suites("bf16", 1),
        "suites-int8-seed1.json": _suites("int8", 1),
        "suites-fp8_e4m3-seed1.json": _suites("fp8_e4m3", 1),
        "suites-bf16-seed2.json": _suites("bf16", 2),
    }
    default_suites.update(suites or {})
    for name, record in default_suites.items():
        parity = (record or {}).get("base_parity") or {}
        arm = default_arms.get(name.replace("suites-", "train-"))
        if "per_example_resident" in parity and parity["per_example_resident"] is None:
            parity["per_example_resident"] = copy.deepcopy(
                (arm or {}).get("final", {}).get("per_example", [])[:PARITY_ROWS])
    files.update(default_suites)
    if timing is not None:
        files["timing.json"] = timing
    for name, payload in files.items():
        if name not in drop and payload is not None:
            (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def _status(out, codec, gate):
    return out["codecs"][codec]["gates"][gate]["status"]


def test_everything_passes(rule, tmp_path):
    out = rule.evaluate(_write(tmp_path))
    assert out["verdict"] == "PASS: int8, fp8_e4m3"
    assert all(row["ok"] for row in out["validity"].values()), out["validity"]
    for codec in ("int8", "fp8_e4m3"):
        gates = ("E1", "E2", "E3", "Q1", "Q2", "Q3")
        assert [_status(out, codec, g) for g in gates] == ["pass"] * 6
    assert out["noise_floor"]["task_loss_margin"] == pytest.approx(0.01)


def test_a_suite_drop_beyond_the_margin_fails_q3(rule, tmp_path):
    suites = {"suites-int8-seed1.json": _suites("int8", 1, mini_arithmetic=27)}
    out = rule.evaluate(_write(tmp_path, suites=suites))
    assert _status(out, "int8", "Q3") == "fail"
    assert out["codecs"]["int8"]["gates"]["Q3"]["failing"] == ["mini_arithmetic"]
    assert out["codecs"]["int8"]["verdict"] == "fail"


def test_the_suite_margin_widens_with_the_seed_to_seed_difference(rule, tmp_path):
    suites = {"suites-int8-seed1.json": _suites("int8", 1, mini_arithmetic=26),
              "suites-bf16-seed2.json": _suites("bf16", 2, mini_arithmetic=26)}
    out = rule.evaluate(_write(tmp_path, suites=suites))
    row = out["codecs"]["int8"]["gates"]["Q3"]["suites"]["mini_arithmetic"]
    assert row["margin"] == 4 and row["drop"] == 4
    assert _status(out, "int8", "Q3") == "pass"


def test_a_saturated_suite_is_reported_but_not_gated(rule, tmp_path):
    suites = {name: _suites(*key, mini_mmlu=26 if key[0] == "bf16" else 10)
              for name, key in (("suites-bf16-seed1.json", ("bf16", 1)),
                                ("suites-bf16-seed2.json", ("bf16", 2)),
                                ("suites-int8-seed1.json", ("int8", 1)))}
    out = rule.evaluate(_write(tmp_path, suites=suites))
    assert out["codecs"]["int8"]["gates"]["Q3"]["suites"]["mini_mmlu"]["gating"] is False
    assert _status(out, "int8", "Q3") == "pass"


def test_restore_error_over_the_bound_on_one_layer_fails_e1(rule, tmp_path):
    error = _error()
    error["records"][-1]["codecs"]["int8"][2]["restore"]["rms_row_rel"] = 0.051
    out = rule.evaluate(_write(tmp_path, error=error))
    assert _status(out, "int8", "E1") == "fail"
    assert out["codecs"]["int8"]["gates"]["E1"]["layers_over"] == [2]
    assert out["codecs"]["int8"]["verdict"] == "fail"
    assert out["verdict"] == "PASS: fp8_e4m3"


def test_one_bad_token_fails_e1(rule, tmp_path):
    error = _error()
    error["records"][-1]["codecs"]["fp8_e4m3"][0]["restore"]["worst_row_rel"] = 0.26
    out = rule.evaluate(_write(tmp_path, error=error))
    assert _status(out, "fp8_e4m3", "E1") == "fail"


def test_the_bounds_are_inclusive(rule, tmp_path):
    error = _error()
    rows = error["records"][-1]["codecs"]["int8"]
    rows[0]["restore"]["rms_row_rel"] = rule.RESTORE_RMS_BOUND
    rows[0]["branch"]["rms_row_rel"] = rule.BRANCH_RMS_BOUND
    for rec in error["records"][:2]:
        rec["codecs"]["int8"]["grad"] = _grad(rule.GRAD_GLOBAL_BOUND, rule.GRAD_MIN_COSINE)
    out = rule.evaluate(_write(tmp_path, error=error))
    assert [_status(out, "int8", g) for g in ("E1", "E2", "E3")] == ["pass"] * 3


def test_recompute_amplification_fails_e2(rule, tmp_path):
    error = _error()
    error["records"][-1]["codecs"]["int8"][3]["branch"]["rms_row_rel"] = 0.12
    out = rule.evaluate(_write(tmp_path, error=error))
    assert _status(out, "int8", "E2") == "fail"
    assert _status(out, "int8", "E1") == "pass"


@pytest.mark.parametrize("grad", [(0.06, 0.999), (0.01, 0.98)])
def test_gradient_disagreement_in_any_set_fails_e3(rule, tmp_path, grad):
    error = _error()
    error["records"][1]["codecs"]["fp8_e4m3"]["grad"] = _grad(*grad)
    out = rule.evaluate(_write(tmp_path, error=error))
    assert _status(out, "fp8_e4m3", "E3") == "fail"


# --------------------------------------------------------------------------
# V15: an incomplete or non-finite error experiment is never a pass
# --------------------------------------------------------------------------
def _fp8_grad(error, set_index):
    return error["records"][set_index]["codecs"]["fp8_e4m3"]["grad"]


def _fp8_layers(error):
    return error["records"][-1]["codecs"]["fp8_e4m3"]


def _drop_tensor(grad):
    grad["per_tensor"].pop(sorted(grad["per_tensor"])[-1])


def _remove_set(error, set_id):
    error["records"] = [rec for rec in error["records"]
                        if not (rec["kind"] == "error_set" and rec["set"] == set_id)]


INCOMPLETE_CASES = {
    # E3 took max/min over sets: a NaN that is not first was skipped and the codec passed.
    "NaN global rel L2, first set": lambda e: _fp8_grad(e, 0).__setitem__("global_rel_l2", NAN),
    "NaN global rel L2, later set": lambda e: _fp8_grad(e, 1).__setitem__("global_rel_l2", NAN),
    "NaN min cosine, later set": lambda e: _fp8_grad(e, 1).__setitem__("min_cosine", NAN),
    "Inf global rel L2": lambda e: _fp8_grad(e, 1).__setitem__("global_rel_l2", INF),
    "-Inf min cosine": lambda e: _fp8_grad(e, 0).__setitem__("min_cosine", -INF),
    "missing min cosine": lambda e: _fp8_grad(e, 1).pop("min_cosine"),
    "missing gradient agreement": lambda e: e["records"][1]["codecs"]["fp8_e4m3"].pop("grad"),
    "NaN branch error on one layer": lambda e: _fp8_layers(e)[2]["branch"].__setitem__(
        "rms_row_rel", NAN),
    "Inf restore error on one layer": lambda e: _fp8_layers(e)[3]["restore"].__setitem__(
        "worst_row_rel", INF),
    "missing restore metric": lambda e: _fp8_layers(e)[1]["restore"].pop("rms_row_rel"),
    "a removed set": lambda e: _remove_set(e, 1),
    "a set id twice": lambda e: e["records"][1].__setitem__("set", 0),
    "a missing layer": lambda e: _fp8_layers(e).pop(2),
    "a duplicated layer": lambda e: _fp8_layers(e).__setitem__(3, copy.deepcopy(
        _fp8_layers(e)[2])),
    "a layer merged over too few rows": lambda e: _fp8_layers(e)[0]["branch"].__setitem__(
        "rows", ROWS_PER_LAYER // N_SETS),
    "an incomplete gradient table": lambda e: _drop_tensor(_fp8_grad(e, 1)),
    "a gradient count off the control": lambda e: _fp8_grad(e, 0).__setitem__("tensors",
                                                                              TENSORS - 1),
    "a min cosine not from its table": lambda e: _fp8_grad(e, 1).__setitem__("min_cosine",
                                                                             0.9999),
    "the protocol not recorded": lambda e: e["meta"]["args"].pop("error_sets"),
}
NAN, INF = float("nan"), float("inf")


@pytest.mark.parametrize("case", sorted(INCOMPLETE_CASES))
def test_an_incomplete_or_non_finite_error_experiment_is_never_a_pass(rule, tmp_path, case):
    error = _error()
    INCOMPLETE_CASES[case](error)
    out = rule.evaluate(_write(tmp_path, error=error))
    assert out["validity"]["V15_error_complete"]["ok"] is False, case
    assert out["validity"]["V15_error_complete"]["detail"]  # names the problem
    for codec in ("int8", "fp8_e4m3"):
        assert [_status(out, codec, g) for g in ("E1", "E2", "E3")] == ["no verdict"] * 3
        assert out["codecs"][codec]["verdict"] == "no verdict"
    assert out["verdict"] == "NO VERDICT"


def test_the_expected_counts_come_from_the_recorded_protocol(rule, tmp_path):
    """Three sets of six layers, all present: complete, whatever the defaults are."""
    error = _error()
    error["meta"]["args"]["error_sets"] = 3
    error["meta"]["n_layers"] = 6
    error["records"] = [{**copy.deepcopy(error["records"][0]), "set": index}
                        for index in range(3)] + [error["records"][-1]]
    for codec, rows in error["records"][-1]["codecs"].items():
        extra = [{**copy.deepcopy(rows[0]), "layer": layer} for layer in range(N_LAYERS, 6)]
        for row in rows + extra:
            for part in ("restore", "branch", "output"):
                row[part]["rows"] = 3 * K * 8
        rows.extend(extra)
    out = rule.evaluate(_write(tmp_path, error=error))
    assert out["validity"]["V15_error_complete"]["ok"] is True, out["validity"]
    assert "3 sets" in out["validity"]["V15_error_complete"]["detail"]


def test_finite_complete_data_keeps_its_decision(rule, tmp_path):
    """V15 only removes verdicts: complete, finite data decides exactly as before."""
    error = _error()
    error["records"][1]["codecs"]["int8"]["grad"] = _grad(0.06, 0.999)
    out = rule.evaluate(_write(tmp_path, error=error))
    assert out["validity"]["V15_error_complete"]["ok"] is True
    assert _status(out, "int8", "E3") == "fail" and out["codecs"]["int8"]["verdict"] == "fail"
    assert out["verdict"] == "PASS: fp8_e4m3"


def test_an_error_gate_fails_the_codec_even_without_training_arms(rule, tmp_path):
    error = _error()
    error["records"][-1]["codecs"]["int8"][0]["restore"]["rms_row_rel"] = 0.2
    out = rule.evaluate(_write(tmp_path, error=error, drop=("train-bf16-seed2.json",)))
    assert out["codecs"]["int8"]["verdict"] == "fail"
    assert out["codecs"]["fp8_e4m3"]["verdict"] == "no verdict"
    assert out["verdict"] == "NO VERDICT"


def test_both_codecs_failing_is_fail(rule, tmp_path):
    arms = {"train-int8-seed1.json": _arm("int8", 1, task_loss=1.6),
            "train-fp8_e4m3-seed1.json": _arm("fp8_e4m3", 1, task_loss=1.6)}
    out = rule.evaluate(_write(tmp_path, arms=arms))
    assert out["verdict"] == "FAIL"


def test_task_loss_over_the_noise_margin_fails_q1(rule, tmp_path):
    arms = {"train-int8-seed1.json": _arm("int8", 1, task_loss=1.5101)}
    out = rule.evaluate(_write(tmp_path, arms=arms))
    assert _status(out, "int8", "Q1") == "fail"
    assert out["codecs"]["int8"]["gates"]["Q1"]["delta"] == pytest.approx(0.0101)


def test_a_better_task_loss_passes_q1(rule, tmp_path):
    arms = {"train-int8-seed1.json": _arm("int8", 1, task_loss=1.40)}
    out = rule.evaluate(_write(tmp_path, arms=arms))
    assert _status(out, "int8", "Q1") == "pass"


def test_the_margin_has_a_floor_when_the_seeds_agree(rule, tmp_path):
    arms = {"train-bf16-seed2.json": _arm("bf16", 2, task_loss=1.50,
                                          curve=[2.0 - 0.05 * s for s in range(STEPS)])}
    out = rule.evaluate(_write(tmp_path, arms=arms))
    floor = out["noise_floor"]
    assert floor["task_loss_noise"] == 0.0
    assert floor["task_loss_margin"] == pytest.approx(rule.QUALITY_FLOOR_REL * 1.50)
    assert _status(out, "int8", "Q1") == "pass"  # delta 0.003 <= 0.0075
    assert _status(out, "fp8_e4m3", "Q1") == "pass"


def test_a_loss_curve_that_drifts_up_at_the_end_fails_q2(rule, tmp_path):
    curve = [2.0 - 0.05 * s for s in range(STEPS)]
    curve[-2:] = [value + 0.05 for value in curve[-2:]]
    arms = {"train-fp8_e4m3-seed1.json": _arm("fp8_e4m3", 1, task_loss=1.502, curve=curve)}
    out = rule.evaluate(_write(tmp_path, arms=arms))
    assert _status(out, "fp8_e4m3", "Q2") == "fail"
    assert _status(out, "fp8_e4m3", "Q1") == "pass"


def test_paired_bootstrap_is_reported_and_reproducible(rule, tmp_path):
    exact = [{"row": r, "nll": 10.0 + (r % 3), "tokens": 10} for r in range(30)]
    worse = [{"row": r, "nll": 10.5 + (r % 3), "tokens": 10} for r in range(30)]
    first = rule.paired_bootstrap(exact, worse)
    assert first == rule.paired_bootstrap(exact, worse)
    assert first["low"] == pytest.approx(0.05) and first["high"] == pytest.approx(0.05)
    assert rule.paired_bootstrap(exact, worse[:10]) is None


# --------------------------------------------------------------------------
# no-verdict paths, one per validity row
# --------------------------------------------------------------------------
def _cpu_error():
    error = _error()
    error["meta"] = {"device": "cpu", "verdict_eligible": False,
                     "label": "SYNTHETIC/CPU PILOT smoke"}
    return error


def _null_broken():
    error = _error()
    error["records"][0]["null_control"] = _cmp(equal=False)
    return error


def _forward_leak():
    error = _error()
    error["records"][1]["codecs"]["int8"]["losses_equal"] = False
    return error


def _codec_nonconforming():
    error = _error()
    error["records"][-1]["codecs"]["int8"][1]["restore"]["max_abs_over_absmax"] = 0.02
    return error


def _control_passes():
    error = _error()
    error["records"][-1]["codecs"]["int4"] = _layer_rows(element=0.07)
    return error


def _arms(**changes):
    return changes


def _nonfinite():
    arm = _arm("int8", 1, task_loss=1.503)
    arm["log"][3]["loss"] = float("nan")
    return {"train-int8-seed1.json": arm}


def _short():
    arm = _arm("fp8_e4m3", 1, task_loss=1.502)
    arm["log"] = arm["log"][:5]
    return {"train-fp8_e4m3-seed1.json": arm}


def _zero_first_grad():
    arm = _arm("bf16", 1)
    arm["log"][0]["grad_norm"] = 0.0
    return {"train-bf16-seed1.json": arm}


def _other_data():
    arm = _arm("int8", 1, task_loss=1.503)
    arm["meta"]["data"]["train_ids_sha256"] = "other"
    return {"train-int8-seed1.json": arm}


def _other_commit():
    arm = _arm("fp8_e4m3", 1, task_loss=1.502)
    arm["meta"]["git_head"] = "def"
    return {"train-fp8_e4m3-seed1.json": arm}


def _other_lr():
    arm = _arm("int8", 1, task_loss=1.503)
    arm["meta"]["config"]["lr"] = 1e-4
    return {"train-int8-seed1.json": arm}


def _noise_same_seed():
    arm = _arm("bf16", 2, task_loss=1.51)
    arm["meta"]["config"]["seed"] = 1
    return {"train-bf16-seed2.json": arm}


def _bf16_saved_as_int8():
    """The audit's reproduction: a bf16 run under the int8 arm's name."""
    arm = _arm("bf16", 1, task_loss=1.503)
    arm["adapter"]["sha256"] = {"adapter_model.safetensors": "int81"}
    return {"train-int8-seed1.json": arm}


def _error_other_commit():
    error = _error()
    error["meta"]["git_head"] = "def"
    return error


def _error_other_model():
    error = _error()
    error["meta"]["model_files_sha256"] = {"config.json": "other"}
    return error


def _error_other_setup():
    error = _error()
    error["meta"]["config"]["k"] = 2
    return error


def _error_no_config():
    error = _error()
    del error["meta"]["config"]
    return error


def _first_step_differs():
    arm = _arm("int8", 1, task_loss=1.503)
    arm["log"][0]["mb_losses"] = [2.1, 1.9, 2.0, 2.0001]
    return {"train-int8-seed1.json": arm}


def _cpu_arm():
    arm = _arm("bf16", 2, task_loss=1.51)
    arm["meta"]["device"] = "cpu"
    return {"train-bf16-seed2.json": arm}


def _too_noisy():
    return {"train-bf16-seed2.json": _arm("bf16", 2, task_loss=1.60)}


def _suites_wrong_adapter():
    record = _suites("int8", 1)
    record["meta"]["adapter_sha256"] = {"adapter_model.safetensors": "stale"}
    return {"suites-int8-seed1.json": record}


def _suites_other_budget():
    record = _suites("fp8_e4m3", 1)
    record["meta"]["max_new_tokens"] = 128
    return {"suites-fp8_e4m3-seed1.json": record}


def _suites_cpu():
    record = _suites("bf16", 2)
    record["meta"]["device"] = "cpu"
    return {"suites-bf16-seed2.json": record}


def _suites_missing():
    return {"suites-bf16-seed1.json": None}


def _parity_off():
    """Resident base 0.3% above the streamed loss: over the 0.25% bound."""
    return {"suites-int8-seed1.json": _suites("int8", 1, parity_loss=1.503 * 1.003)}


def _parity_missing():
    record = _suites("fp8_e4m3", 1)
    del record["base_parity"]
    return {"suites-fp8_e4m3-seed1.json": record}


def _parity_adapter_mismatch():
    record = _suites("bf16", 1)
    record["base_parity"]["adapter"] = {"tensors": 8, "mismatch": ["x"], "mismatched": 1}
    return {"suites-bf16-seed1.json": record}


def _parity_tokens_differ():
    record = _suites("bf16", 2, parity_loss=1.51)
    record["base_parity"]["per_example_resident"][0]["tokens"] = 11
    record["base_parity"]["per_example_resident"][0]["nll"] = 1.51 * 11.0
    return {"suites-bf16-seed2.json": record}


def _parity_unknown_row():
    record = _suites("bf16", 1, parity_loss=1.50)
    record["base_parity"]["per_example_resident"][0]["row"] = 999
    return {"suites-bf16-seed1.json": record}


NO_VERDICT_CASES = {
    "V1_error_eligible": dict(error=_cpu_error),
    "V2_harness_exact": dict(error=_null_broken),
    "V3_forward_exact": dict(error=_forward_leak),
    "V4_codec_conforms": dict(error=_codec_nonconforming),
    "V5_control_discriminates": dict(error=_control_passes),
    "V6_training_eligible": dict(arms=_cpu_arm),
    "V7_arms_complete": dict(arms=_nonfinite),
    "V7_arms_complete#short": dict(arms=_short),
    "V7_arms_complete#grad": dict(arms=_zero_first_grad),
    "V8_same_inputs": dict(arms=_other_data),
    "V8_same_inputs#commit": dict(arms=_other_commit),
    "V9_configs": dict(arms=_other_lr),
    "V9_configs#seed": dict(arms=_noise_same_seed),
    "V9_configs#identity": dict(arms=_bf16_saved_as_int8),
    "V10_first_step_forward_exact": dict(arms=_first_step_differs),
    "V11_noise_floor_usable": dict(arms=_too_noisy),
    "V12_suites_records": dict(suites=_suites_wrong_adapter),
    "V12_suites_records#budget": dict(suites=_suites_other_budget),
    "V12_suites_records#cpu": dict(suites=_suites_cpu),
    "V12_suites_records#missing": dict(suites=_suites_missing),
    "V13_base_parity": dict(suites=_parity_off),
    "V13_base_parity#missing": dict(suites=_parity_missing),
    "V13_base_parity#adapter": dict(suites=_parity_adapter_mismatch),
    "V13_base_parity#tokens": dict(suites=_parity_tokens_differ),
    "V13_base_parity#row": dict(suites=_parity_unknown_row),
    "V14_error_linked": dict(error=_error_other_commit),
    "V14_error_linked#model": dict(error=_error_other_model),
    "V14_error_linked#setup": dict(error=_error_other_setup),
    "V14_error_linked#config": dict(error=_error_no_config),
}


@pytest.mark.parametrize("case", sorted(NO_VERDICT_CASES))
def test_each_validity_row_removes_the_verdict(rule, tmp_path, case):
    spec = NO_VERDICT_CASES[case]
    kwargs = {key: make() for key, make in spec.items()}
    out = rule.evaluate(_write(tmp_path, **kwargs))
    row = case.split("#")[0]
    assert out["validity"][row]["ok"] is False, out["validity"][row]
    assert out["verdict"] == "NO VERDICT"
    for codec in ("int8", "fp8_e4m3"):
        assert out["codecs"][codec]["verdict"] == "no verdict"
        assert _status(out, codec, "Q1") == "no verdict"


def test_a_missing_noise_arm_is_no_verdict(rule, tmp_path):
    out = rule.evaluate(_write(tmp_path, drop=("train-bf16-seed2.json",)))
    assert out["validity"]["V6_training_eligible"]["ok"] is False
    assert out["noise_floor"] is None
    assert out["verdict"] == "NO VERDICT"


def test_a_missing_error_record_is_no_verdict(rule, tmp_path):
    out = rule.evaluate(_write(tmp_path, drop=("error.json",)))
    assert out["validity"]["V2_harness_exact"]["ok"] is False
    assert out["verdict"] == "NO VERDICT"
    assert _status(out, "int8", "E1") == "no verdict"


def test_an_empty_results_directory_is_no_verdict(rule, tmp_path):
    out = rule.evaluate(tmp_path)
    assert out["verdict"] == "NO VERDICT"
    assert all(row["ok"] is False for row in out["validity"].values()), out["validity"]


def test_base_parity_inside_the_bound_passes(rule, tmp_path):
    """0.2% off on one arm, under the 0.25% bound: V13 holds and the verdict stands."""
    suites = {"suites-int8-seed1.json": _suites("int8", 1, parity_loss=1.503 * 1.002)}
    out = rule.evaluate(_write(tmp_path, suites=suites))
    assert out["validity"]["V13_base_parity"]["ok"] is True, out["validity"]["V13_base_parity"]
    assert out["bounds"]["base_parity_rel"] == rule.QUALITY_FLOOR_REL / 2
    assert out["verdict"] == "PASS: int8, fp8_e4m3"


def _truncate(path: Path) -> None:
    path.write_text(path.read_text(encoding="utf-8")[:40], encoding="utf-8")


def _set(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


MALFORMED_CASES = {
    "truncated error": ("error.json", _truncate),
    "truncated arm": ("train-int8-seed1.json", _truncate),
    "truncated suites": ("suites-fp8_e4m3-seed1.json", _truncate),
    "list instead of object": ("train-bf16-seed1.json",
                               lambda p: p.write_text("[1, 2]", encoding="utf-8")),
    "layer row without restore": ("error.json", lambda p: _set(
        p, lambda d: d["records"][-1]["codecs"]["int8"][0].pop("restore"))),
    "arm without final": ("train-fp8_e4m3-seed1.json", lambda p: _set(p, lambda d: d.pop("final"))),
    "noise arm task loss is a string": ("train-bf16-seed2.json", lambda p: _set(
        p, lambda d: d["final"].__setitem__("task_loss", "1.5"))),
    "suites without counts": ("suites-bf16-seed1.json", lambda p: _set(
        p, lambda d: d["suites"]["mini_mmlu"].pop("correct"))),
    "parity rows not a list": ("suites-int8-seed1.json", lambda p: _set(
        p, lambda d: d["base_parity"].__setitem__("per_example_resident", 3))),
}


@pytest.mark.parametrize("case", sorted(MALFORMED_CASES))
def test_a_malformed_input_is_no_verdict_and_never_raises(rule, tmp_path, case):
    name, damage = MALFORMED_CASES[case]
    results = _write(tmp_path)
    damage(results / name)
    out = rule.evaluate(results)
    assert out["verdict"] == "NO VERDICT", case
    assert not all(row["ok"] for row in out["validity"].values())
    for codec in ("int8", "fp8_e4m3"):
        assert out["codecs"][codec]["verdict"] in ("no verdict", "fail")
        assert out["codecs"][codec]["verdict"] != "pass"


def test_cli_on_a_malformed_directory_prints_no_verdict(rule, tmp_path, capsys):
    results = _write(tmp_path)
    _truncate(results / "error.json")
    assert rule.main([str(results)]) == 0
    assert "Verdict: NO VERDICT" in capsys.readouterr().out


# --------------------------------------------------------------------------
# informational sections
# --------------------------------------------------------------------------
def test_timing_is_a_ratio_and_never_changes_the_verdict(rule, tmp_path):
    def block(a, b, outcome="ok"):
        return [{"arm": arm, "value": value, "outcome": outcome}
                for arm, value in (("A", a), ("B", b), ("B", b), ("A", a))]

    timing = {"comparisons": [{
        "codec": "int8", "blocks_needed": 2, "stopped": None,
        "blocks": [block(10.0, 11.0), block(10.0, 11.0), block(1.0, 99.0, outcome="void")]}]}
    out = rule.evaluate(_write(tmp_path, timing=timing))
    info = out["timing"]["comparisons"]["int8"]
    assert info["valid_blocks"] == 2
    assert info["step_time_ratio_codec_over_bf16"] == pytest.approx(1.1)
    assert info["repeats_rule_met"] is True
    assert out["verdict"] == "PASS: int8, fp8_e4m3"


def test_cli_prints_and_writes_the_verdict(rule, tmp_path, capsys):
    results = _write(tmp_path)
    target = tmp_path / "verdict.json"
    assert rule.main([str(results), "--json", str(target)]) == 0
    assert "Verdict: PASS: int8, fp8_e4m3" in capsys.readouterr().out
    assert json.loads(target.read_text(encoding="utf-8"))["verdict"] == "PASS: int8, fp8_e4m3"
