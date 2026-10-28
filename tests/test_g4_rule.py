"""The G4 verdict (benchmarks/harness/g4_rule.py) on synthetic records: every gate's
pass and fail, and every validity row's no-verdict path."""

from __future__ import annotations

import copy
import json
import math
import shutil
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


#: The held-out panel every fixture arm scores, as (row id, response tokens): row ids are
#: not positions, and two rows score no token (a prompt that fills the window), as in the
#: GPU run.
PANEL = [(1000 + 37 * index, 0 if index in (5, 13) else 6 + index % 5) for index in range(20)]


def _task_loss(per_example):
    """What g4_probe.task_loss records: summed NLL over summed scored tokens."""
    tokens = sum(ex["tokens"] for ex in per_example)
    return sum(ex["nll"] for ex in per_example) / tokens if tokens else None


def _final(per_example):
    """``final`` as g4_probe.mode_train writes it from the panel rows."""
    return {"task_loss": _task_loss(per_example), "examples": len(per_example),
            "tokens": sum(ex["tokens"] for ex in per_example), "per_example": per_example}


def _arm(codec, seed, *, task_loss=1.50, curve=None, panel=PANEL):
    curve = curve or [2.0 - 0.05 * step for step in range(STEPS)]
    per_example = [{"row": row, "nll": task_loss * tokens, "tokens": tokens}
                   for row, tokens in panel]
    log = [{"step": step, "loss": loss, "grad_norm": 0.5, "lr": 1e-4}
           for step, loss in enumerate(curve)]
    log[0]["mb_losses"] = [2.1, 1.9, 2.0, 2.0] if seed == 1 else [2.2, 1.8, 2.0, 2.0]
    return {
        "adapter": {"dir": "x", "sha256": {"adapter_model.safetensors": f"{codec}{seed}"}},
        "meta": {
            "device": "cuda", "verdict_eligible": True, "label": "box",
            **copy.deepcopy(_SHARED_META),
            # As g4_probe.load_data records the panel it encoded.
            "data": {**copy.deepcopy(_SHARED_META["data"]), "panel_examples": len(panel),
                     "panel_response_tokens": sum(tokens for _row, tokens in panel)},
            "config": {"codec": codec, "seed": seed, "steps": STEPS, "lr": 2e-4, "k": 4,
                       "model_revision": "60b99dc"},
        },
        "log": log,
        "final": _final(per_example),
        "memory": {"store": {"measured_bytes_per_token_per_layer": 4100.0,
                             "arithmetic_bytes_per_token_per_layer": 4100.0}},
    }


SUITE_N = {"mini_mmlu": 26, "mini_common_sense": 24, "mini_instruction": 24,
           "mini_arithmetic": 40}


TASK_LOSS = {("bf16", 1): 1.50, ("int8", 1): 1.503, ("fp8_e4m3", 1): 1.502, ("bf16", 2): 1.51}
PARITY_ROWS = 4


def _items(suite, n, correct):
    """A suite's item table as g4_probe.score_suites writes it: the first ``correct`` of its
    ``n`` questions answered right."""
    return [{"question": f"{suite}: question {index}?", "answer": f"answer {index}",
             "output": f"answer {index}" if index < correct else "no idea",
             "correct": index < correct} for index in range(n)]


def _suites(codec, seed, *, parity_loss=None, sizes=None, **correct):
    """A suites record. Its resident parity rows mirror the first rows of its training arm's
    panel (filled in by ``_write`` from the arm actually written) unless ``parity_loss``
    sets them explicitly. ``n`` and ``correct`` agree with the item tables."""
    sizes = {**SUITE_N, **(sizes or {})}
    scores = {"mini_mmlu": 15, "mini_common_sense": 18, "mini_instruction": 12,
              "mini_arithmetic": 30, **correct}
    rows = None if parity_loss is None else [
        {"row": row, "nll": parity_loss * tokens, "tokens": tokens}
        for row, tokens in PANEL[:PARITY_ROWS]]
    return {
        "meta": {"device": "cuda", "verdict_eligible": True, "label": "box",
                 "adapter_sha256": {"adapter_model.safetensors": f"{codec}{seed}"},
                 "max_new_tokens": 64, "quant": "nf4"},
        "base_parity": {"adapter": {"tensors": 8, "mismatch": [], "mismatched": 0},
                        "per_example_resident": rows},
        "suites": {name: {"n": sizes[name], "correct": scores[name],
                          "items": _items(name, sizes[name], scores[name])}
                   for name in SUITE_N},
    }


def _write(tmp_path: Path, error=None, arms=None, timing=None, drop=(), suites=None,
           panel=PANEL):
    files = {"error.json": error or _error()}
    default_arms = {
        "train-bf16-seed1.json": _arm("bf16", 1, task_loss=1.50, panel=panel),
        "train-int8-seed1.json": _arm("int8", 1, task_loss=1.503, panel=panel),
        "train-fp8_e4m3-seed1.json": _arm("fp8_e4m3", 1, task_loss=1.502, panel=panel),
        "train-bf16-seed2.json": _arm("bf16", 2, task_loss=1.51, panel=panel,
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
# V16: Q1 and its noise floor read the whole recorded panel, recomputed
# --------------------------------------------------------------------------
ARMS = ("train-bf16-seed1.json", "train-int8-seed1.json", "train-fp8_e4m3-seed1.json",
        "train-bf16-seed2.json")


def _rows(arm):
    return arm["final"]["per_example"]


def _restate(arm):
    """Rewrite ``final`` from the rows it holds: a self-consistent record of other rows."""
    arm["final"] = _final(_rows(arm))


def _keep_first(arm, rows=PARITY_ROWS):
    """The review's first reproduction: only the parity rows survive, aggregates kept."""
    arm["final"]["per_example"] = _rows(arm)[:rows]


def _one_nat_per_token_more(arm, rows=PARITY_ROWS):
    """The review's second reproduction: +1 nat per token outside the parity rows, the
    recorded task loss kept."""
    for ex in _rows(arm)[rows:]:
        ex["nll"] += ex["tokens"]


def _drop_row(arm):
    _rows(arm).pop(10)
    _restate(arm)


def _repeat_row(arm):
    rows = _rows(arm)
    rows[19] = copy.deepcopy(rows[14])  # same token count: only the row id repeats
    _restate(arm)


def _foreign_row(arm):
    _rows(arm)[10]["row"] = 99999


def _move_a_token(arm):
    """Rows 8 and 9 trade a token: the same total, but not the other arms' rows."""
    rows = _rows(arm)
    for ex, step in ((rows[8], 1), (rows[9], -1)):
        ex["nll"] = ex["nll"] / ex["tokens"] * (ex["tokens"] + step)
        ex["tokens"] += step
    _restate(arm)


def _field(key, value):
    return lambda arm: _rows(arm)[10].__setitem__(key, value)


def _final_field(key, change):
    return lambda arm: arm["final"].__setitem__(key, change(arm["final"][key]))


PANEL_CASES = {
    "a missing row, aggregates restated": ("train-fp8_e4m3-seed1.json", _drop_row),
    "a repeated row id, aggregates restated": ("train-bf16-seed2.json", _repeat_row),
    "a row of another panel": ("train-int8-seed1.json", _foreign_row),
    "a token moved between rows": ("train-bf16-seed2.json", _move_a_token),
    "row id a string": ("train-bf16-seed1.json", _field("row", "1370")),
    "row id a bool": ("train-fp8_e4m3-seed1.json", _field("row", True)),
    "row id negative": ("train-int8-seed1.json", _field("row", -1)),
    "token count a bool": ("train-bf16-seed2.json", _field("tokens", True)),
    "token count a float": ("train-fp8_e4m3-seed1.json", _field("tokens", 6.0)),
    "token count negative": ("train-bf16-seed1.json", _field("tokens", -6)),
    "NaN NLL": ("train-bf16-seed2.json", _field("nll", float("nan"))),
    "Inf NLL": ("train-fp8_e4m3-seed1.json", _field("nll", float("inf"))),
    "NLL a string": ("train-int8-seed1.json", _field("nll", "9.0")),
    "NLL negative": ("train-bf16-seed1.json", _field("nll", -9.0)),
    "NLL on a row that scores no token": (
        "train-bf16-seed2.json", lambda arm: _rows(arm)[5].__setitem__("nll", 0.5)),
    "final.examples off its rows": (
        "train-fp8_e4m3-seed1.json", _final_field("examples", lambda n: n + 1)),
    "final.tokens off its rows": ("train-bf16-seed2.json", _final_field("tokens", lambda n: n - 1)),
    "final.task_loss stale": (
        "train-bf16-seed1.json", _final_field("task_loss", lambda t: t * 1.5)),
    "final.task_loss missing": ("train-int8-seed1.json", _final_field("task_loss", lambda t: None)),
    "panel protocol not recorded": (
        "train-bf16-seed2.json", lambda arm: arm["meta"]["data"].pop("panel_examples")),
    "no per-example table": (
        "train-fp8_e4m3-seed1.json", lambda arm: arm["final"].__setitem__("per_example", {})),
}


def _panel_refused(out):
    assert out["verdict"] == "NO VERDICT"
    for codec in ("int8", "fp8_e4m3"):
        assert out["codecs"][codec]["verdict"] == "no verdict"
        assert _status(out, codec, "Q1") == "no verdict"
    assert out["validity"]["V16_panel_complete"]["ok"] is False, out["validity"]


@pytest.mark.parametrize("name", ARMS)
@pytest.mark.parametrize("damage", [_keep_first, _one_nat_per_token_more],
                         ids=["only the parity rows", "stale task loss"])
def test_the_reviews_panel_reproductions_are_no_verdict_in_every_arm(rule, tmp_path, name,
                                                                      damage):
    results = _write(tmp_path)
    _set(results / name, damage)
    out = rule.evaluate(results)
    _panel_refused(out)
    # The noise floor is never computed from an exact arm's unverified rows.
    assert (out["noise_floor"] is None) == name.startswith("train-bf16")


@pytest.mark.parametrize("case", sorted(PANEL_CASES))
def test_an_incomplete_or_inconsistent_panel_is_no_verdict(rule, tmp_path, case):
    name, damage = PANEL_CASES[case]
    results = _write(tmp_path)
    _set(results / name, damage)
    _panel_refused(rule.evaluate(results))


@pytest.mark.parametrize("size", [1, 7, 64])
def test_a_complete_panel_of_any_recorded_size_keeps_its_decision(rule, tmp_path, size):
    """Completeness follows each arm's recorded protocol, not a fixed panel size; rows
    that score no token are valid and weigh nothing."""
    panel = [(5 * index, 0 if index % 6 == 5 else 3 + index % 4) for index in range(size)]
    out = rule.evaluate(_write(tmp_path, panel=panel))
    assert out["validity"]["V16_panel_complete"]["ok"] is True, out["validity"]
    assert out["verdict"] == "PASS: int8, fp8_e4m3"
    assert out["codecs"]["fp8_e4m3"]["gates"]["Q1"]["delta"] == pytest.approx(0.002)


def test_a_panel_that_scores_no_token_is_no_verdict(rule, tmp_path):
    _panel_refused(rule.evaluate(_write(tmp_path, panel=[(row, 0) for row in range(6)])))


def test_the_task_loss_check_allows_rounding_and_nothing_more(rule, tmp_path):
    """A recorded task loss one ulp off its recomputation is rounding; 1e-9 relative,
    far inside the Q1 margin, is not."""
    results = _write(tmp_path)
    arm = results / "train-fp8_e4m3-seed1.json"
    _set(arm, _final_field("task_loss", lambda t: math.nextafter(t, math.inf)))
    assert rule.evaluate(results)["validity"]["V16_panel_complete"]["ok"] is True
    _set(arm, _final_field("task_loss", lambda t: t * (1 + 1e-9)))
    _panel_refused(rule.evaluate(results))


def test_paired_bootstrap_never_pairs_a_repeated_row(rule):
    exact = [{"row": r, "nll": 10.0 + (r % 3), "tokens": 10} for r in range(30)]
    repeated = copy.deepcopy(exact)
    repeated[-1] = copy.deepcopy(repeated[0])
    assert rule.paired_bootstrap(exact, repeated) is None
    assert rule.paired_bootstrap(repeated, exact) is None


# --------------------------------------------------------------------------
# V17: Q3 compares the same questions out of the same n
# --------------------------------------------------------------------------
SUITES_RECORDS = tuple(name.replace("train-", "suites-") for name in ARMS)


def _mmlu(record):
    return record["suites"]["mini_mmlu"]


def _restate_counts(entry):
    entry["n"] = len(entry["items"])
    entry["correct"] = sum(item["correct"] for item in entry["items"])


def _extra_wrong_items(record):
    """The review's reproduction: 26 more mini_mmlu items, all wrong, n 26 -> 52, the
    correct count unchanged and consistent."""
    entry = _mmlu(record)
    entry["items"] += [{"question": f"extra {index}?", "answer": "A", "output": "",
                        "correct": False} for index in range(26)]
    entry["n"] = len(entry["items"])


def _items_twice(record):
    """Every mini_mmlu item scored a second time, wrong; n doubled, correct unchanged."""
    entry = _mmlu(record)
    entry["items"] += [{**copy.deepcopy(item), "output": "", "correct": False}
                       for item in entry["items"]]
    entry["n"] = len(entry["items"])


def _item(index, key, value):
    return lambda record: _mmlu(record)["items"][index].__setitem__(key, value)


def _entry(change):
    def damage(record):
        change(_mmlu(record))
    return damage


SUITE_CASES = {
    "every item scored twice": ("suites-bf16-seed2.json", _items_twice),
    "a missing item, counts restated": ("suites-fp8_e4m3-seed1.json", _entry(
        lambda e: (e["items"].pop(20), _restate_counts(e)))),
    "a missing item, n kept": ("suites-bf16-seed1.json", _entry(lambda e: e["items"].pop(20))),
    "an extra item, counts restated": ("suites-int8-seed1.json", _entry(lambda e: (
        e["items"].append({"question": "new?", "answer": "B", "output": "B", "correct": True}),
        _restate_counts(e)))),
    "an item replaced by a repeat": ("suites-bf16-seed2.json", _entry(
        lambda e: e["items"].__setitem__(20, copy.deepcopy(e["items"][21])))),
    "a substituted question, same n": ("suites-fp8_e4m3-seed1.json",
                                       _item(20, "question", "another question?")),
    "a substituted answer, same n": ("suites-bf16-seed2.json", _item(20, "answer", "another")),
    "n off its table": ("suites-int8-seed1.json", _entry(lambda e: e.__setitem__("n", 27))),
    "n a string": ("suites-bf16-seed1.json", _entry(lambda e: e.__setitem__("n", "26"))),
    "correct off its table": ("suites-fp8_e4m3-seed1.json",
                              _entry(lambda e: e.__setitem__("correct", e["correct"] - 1))),
    "no item table": ("suites-bf16-seed2.json", _entry(lambda e: e.pop("items"))),
    "an empty item table": ("suites-int8-seed1.json", _entry(
        lambda e: e.update(items=[], n=0, correct=0))),
    "items not a list": ("suites-fp8_e4m3-seed1.json",
                         _entry(lambda e: e.__setitem__("items", {}))),
    "an item's score not a bool": ("suites-bf16-seed1.json", _item(0, "correct", 1)),
    "an item without its question": ("suites-bf16-seed2.json",
                                     lambda r: _mmlu(r)["items"][0].pop("question")),
    "a suite missing": ("suites-fp8_e4m3-seed1.json",
                        lambda r: r["suites"].pop("mini_arithmetic")),
}


def _suites_refused(out):
    assert out["verdict"] == "NO VERDICT"
    for codec in ("int8", "fp8_e4m3"):
        assert out["codecs"][codec]["verdict"] == "no verdict"
        assert _status(out, codec, "Q3") == "no verdict"
    assert out["validity"]["V17_suite_items"]["ok"] is False, out["validity"]


@pytest.mark.parametrize("name", SUITES_RECORDS)
def test_the_reviews_extra_suite_items_are_no_verdict_in_every_record(rule, tmp_path, name):
    results = _write(tmp_path)
    _set(results / name, _extra_wrong_items)
    _suites_refused(rule.evaluate(results))


@pytest.mark.parametrize("case", sorted(SUITE_CASES))
def test_suite_tables_that_differ_or_disagree_with_their_counts_are_no_verdict(rule, tmp_path,
                                                                               case):
    name, damage = SUITE_CASES[case]
    results = _write(tmp_path)
    _set(results / name, damage)
    _suites_refused(rule.evaluate(results))


def test_outputs_scores_and_order_may_differ_between_arms(rule, tmp_path):
    """An item is its recorded question and answer: the model's output and score differ
    between arms by design."""
    record = _suites("fp8_e4m3", 1, mini_mmlu=14)
    _mmlu(record)["items"].reverse()
    out = rule.evaluate(_write(tmp_path, suites={"suites-fp8_e4m3-seed1.json": record}))
    assert out["validity"]["V17_suite_items"]["ok"] is True, out["validity"]
    assert _status(out, "fp8_e4m3", "Q3") == "pass"
    assert out["codecs"]["fp8_e4m3"]["gates"]["Q3"]["suites"]["mini_mmlu"]["drop"] == 1


def test_suites_of_another_size_keep_their_decision(rule, tmp_path):
    sizes = {"mini_mmlu": 5, "mini_common_sense": 3, "mini_instruction": 9, "mini_arithmetic": 7}
    scores = {"mini_mmlu": 3, "mini_common_sense": 2, "mini_instruction": 4, "mini_arithmetic": 5}
    suites = {name: _suites(*key, sizes=sizes, **scores) for name, key in (
        ("suites-bf16-seed1.json", ("bf16", 1)), ("suites-int8-seed1.json", ("int8", 1)),
        ("suites-fp8_e4m3-seed1.json", ("fp8_e4m3", 1)), ("suites-bf16-seed2.json", ("bf16", 2)))}
    out = rule.evaluate(_write(tmp_path, suites=suites))
    assert out["validity"]["V17_suite_items"]["ok"] is True, out["validity"]
    assert out["verdict"] == "PASS: int8, fp8_e4m3"
    rows = out["codecs"]["fp8_e4m3"]["gates"]["Q3"]["suites"]
    assert {name: row["n"] for name, row in rows.items()} == sizes


@pytest.mark.parametrize("record, damage", [
    ("train-fp8_e4m3-seed1.json", _keep_first),
    ("suites-fp8_e4m3-seed1.json", _extra_wrong_items),
], ids=["panel", "suite items"])
def test_a_proven_error_fail_survives_a_damaged_quality_record(rule, tmp_path, record, damage):
    """int8 fails E2 on a complete error experiment: it stays FAIL whatever a damaged
    quality record shows, and fp8, which needs that record, has no verdict."""
    error = _error()
    error["records"][-1]["codecs"]["int8"][3]["branch"]["rms_row_rel"] = 0.12
    results = _write(tmp_path, error=error)
    _set(results / record, damage)
    out = rule.evaluate(results)
    assert out["codecs"]["int8"]["verdict"] == "fail"
    assert out["codecs"]["fp8_e4m3"]["verdict"] == "no verdict"
    assert out["verdict"] == "NO VERDICT"


# --------------------------------------------------------------------------
# the published GPU run (benchmarks/results/g4-rtx3090-20261024)
# --------------------------------------------------------------------------
PUBLISHED = Path(__file__).resolve().parents[1] / "benchmarks" / "results" / "g4-rtx3090-20261024"
#: The parity rows of the GPU run (g4_probe.py --parity-rows, default 16).
RUN_PARITY_ROWS = 16


def test_the_published_run_replays_with_every_number_unchanged(rule):
    """V16 and V17 hold on the run's own inputs and change nothing else in the replay."""
    out = json.loads(json.dumps(rule.evaluate(PUBLISHED / "inputs")))
    added = {name: out["validity"].pop(name) for name in ("V16_panel_complete",
                                                          "V17_suite_items")}
    assert all(row["ok"] for row in added.values()), added
    assert out["verdict"] == "PASS: fp8_e4m3"
    replay = json.loads((PUBLISHED / "verdict-v15-replay.json").read_text(encoding="utf-8"))
    assert out == replay


PUBLISHED_CASES = {
    "only the parity rows": ("train-fp8_e4m3-seed1.json",
                             lambda arm: _keep_first(arm, RUN_PARITY_ROWS), "V16_panel_complete"),
    "stale task loss": ("train-fp8_e4m3-seed1.json",
                        lambda arm: _one_nat_per_token_more(arm, RUN_PARITY_ROWS),
                        "V16_panel_complete"),
    "26 extra wrong mini_mmlu items": ("suites-fp8_e4m3-seed1.json", _extra_wrong_items,
                                       "V17_suite_items"),
    "mini_mmlu scored twice": ("suites-fp8_e4m3-seed1.json", _items_twice, "V17_suite_items"),
}


@pytest.mark.parametrize("case", sorted(PUBLISHED_CASES))
def test_the_reviews_reproductions_on_the_published_run_are_no_verdict(rule, tmp_path, case):
    name, damage, row = PUBLISHED_CASES[case]
    results = Path(shutil.copytree(PUBLISHED / "inputs", tmp_path / "inputs"))
    _set(results / name, damage)
    out = rule.evaluate(results)
    assert out["verdict"] == "NO VERDICT"
    assert out["codecs"]["fp8_e4m3"]["verdict"] == "no verdict"
    assert out["codecs"]["int8"]["verdict"] == "fail"  # E2 and E3 still fail it
    assert out["validity"][row]["ok"] is False, out["validity"][row]

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
