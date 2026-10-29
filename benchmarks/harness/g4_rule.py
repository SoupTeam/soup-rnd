#!/usr/bin/env python3
"""The G4 verdict, recomputed from the raw JSON alone.

Implements the decision rule of benchmarks/probe-g4-l2l-activation-compression.md §3,
fixed before the GPU run; to be committed before the first GPU arm. Standard library
only (plus the pure byte arithmetic of g4_activation_codec). A gate whose inputs fail a
validity row reports "no verdict", never a pass. A missing, unreadable or malformed
input file fails its validity rows and never raises: the CLI prints NO VERDICT.

Inputs, in one results directory:

* ``error.json``: ``g4_probe.py --mode error`` (restore error, one-recompute
  propagation, gradient agreement, the controls).
* ``train-<codec>-seed<seed>.json``: ``g4_probe.py --mode train``. The rule needs
  ``bf16``, ``int8`` and ``fp8_e4m3`` at the main seed, and ``bf16`` at the noise seed.
* ``suites-<codec>-seed<seed>.json``: ``g4_probe.py --mode suites`` on each of those
  four arms' saved adapters, each with its ``base_parity`` record (V13).
* ``timing.json`` (optional): ``g4_timing.py``. Informational: it never changes a verdict.

Usage: python benchmarks/harness/g4_rule.py <results dir> [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import g4_activation_codec as codecs  # noqa: E402

PASS, FAIL, NONE = "pass", "fail", "no verdict"
CANDIDATES = codecs.CANDIDATES
CONTROL = codecs.COARSE
EXACT = codecs.IDENTITY
MAIN_SEED = 1
NOISE_SEED = 2
CPU_LABEL = "SYNTHETIC/CPU PILOT"

# -- the error bounds (§3.2) -------------------------------------------------
#: E1: per-token relative L2 of the restored boundary, RMS over tokens, every layer.
RESTORE_RMS_BOUND = 0.05
#: E1: the worst single token's relative L2, every layer.
RESTORE_WORST_ROW_BOUND = 0.25
#: E2: the layer's residual branch (output - input) recomputed from the restored
#: boundary, per-token relative L2, RMS over tokens, every layer.
BRANCH_RMS_BOUND = 0.10
#: E3: LoRA gradients of compressed L2L against exact L2L at k = 4, every set.
GRAD_GLOBAL_BOUND = 0.05
GRAD_MIN_COSINE = 0.99
#: V4: the largest element error over its row's absmax that each codec's definition
#: allows, bf16 rounding of the restored value included.
ELEMENT_BOUND = {
    codecs.INT8: 1 / 254 + 2.0**-8,
    codecs.FP8: 2.0**-4 + 2.0**-8,
    codecs.COARSE: 1 / 14 + 2.0**-8,
}

# -- the quality gate (§3.3) ---------------------------------------------------
#: The task-loss margin never drops below this fraction of the exact arm's task loss.
QUALITY_FLOOR_REL = 0.005
#: A seed-to-seed spread above this fraction of the exact task loss is too wide to gate on.
NOISE_CAP_REL = 0.03
#: The loss-curve comparison reads the last quarter of the steps.
CURVE_TAIL = 0.25
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 17
#: Q3: Soup's four mini gate suites (C5's set), greedy, 64 new tokens.
SUITES = ("mini_mmlu", "mini_common_sense", "mini_instruction", "mini_arithmetic")
#: Q3: the smallest allowed drop in correct answers per suite, whatever the seeds show.
SUITE_MIN_MARGIN = 2

# -- base parity (§3.4, V13; added in the 2026-10-22 review, before any parity number) --
#: |TL(resident base + adapter) - TL(streamed model as trained)| over the parity rows,
#: both recomputed from the per-example NLL, must not exceed this fraction of the
#: streamed TL: half the Q1 floor, so the model the suites score is indistinguishable
#: from the trained one at the resolution the quality gates themselves use.
BASE_PARITY_REL = QUALITY_FLOOR_REL / 2

#: What a malformed record raises when a field is missing or of the wrong type.
MALFORMED = (KeyError, TypeError, ValueError, IndexError, AttributeError, ZeroDivisionError)


def _load(path: Path) -> Optional[Dict[str, Any]]:
    """The JSON object in ``path``; None when the file is absent. A file that is not a
    readable JSON object comes back as ``{"meta": {}, "unreadable": reason}``, which no
    validity row accepts."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"meta": {}, "unreadable": f"{path.name}: {exc}"}
    if not isinstance(payload, dict) or not isinstance(payload.get("meta", {}), dict):
        return {"meta": {}, "unreadable": f"{path.name}: not a JSON object with a meta object"}
    return payload


def _row(ok: Optional[bool], detail: str) -> Dict[str, Any]:
    return {"ok": ok, "detail": detail}


def _gate(status: str, detail: str, **values: Any) -> Dict[str, Any]:
    return {"status": status, "detail": detail, **values}


def _records(payload: Optional[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
    if not payload:
        return []
    return [rec for rec in payload.get("records", []) if rec.get("kind") == kind]


def _eligible(payload: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    if payload is None:
        return False, "missing"
    if payload.get("unreadable"):
        return False, f"unreadable ({payload['unreadable']})"
    meta = payload.get("meta", {})
    label = str(meta.get("label", ""))
    if meta.get("device") != "cuda" or meta.get("verdict_eligible") is not True:
        return False, f"device {meta.get('device')!r} is not a verdict run"
    if CPU_LABEL in label:
        return False, f"labelled {CPU_LABEL}"
    return True, "cuda"


def _rows_or_fail(names: Sequence[str], body: Any, *args: Any) -> Dict[str, Dict[str, Any]]:
    """``body(*args)``'s validity rows; every one of ``names`` fails if it raises."""
    try:
        return body(*args)
    except MALFORMED as exc:
        return {name: _row(False, f"malformed input: {exc!r}") for name in names}


def _gates_or_none(names: Sequence[str], body: Any, *args: Any) -> Dict[str, Dict[str, Any]]:
    """``body(*args)``'s gates; every one of ``names`` reads no verdict if it raises."""
    try:
        return body(*args)
    except MALFORMED as exc:
        return {name: _gate(NONE, f"malformed input: {exc!r}") for name in names}


# ==========================================================================
# the error experiment
# ==========================================================================
def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _count(value: Any) -> bool:
    """A non-negative integer; a bool is not a count."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _listed(problems: Sequence[str]) -> str:
    """The first eight problems, then how many more."""
    shown = list(problems[:8]) + ([f"... {len(problems) - 8} more"] if len(problems) > 8 else [])
    return "; ".join(shown)


#: The per-layer metrics E1, E2 and V4 read, per codec and layer.
LAYER_METRICS = (("restore", "rms_row_rel"), ("restore", "worst_row_rel"),
                 ("restore", "max_abs_over_absmax"), ("branch", "rms_row_rel"),
                 ("output", "rms_row_rel"))
#: The per-set gradient-agreement metrics E3 reads, per codec.
GRAD_METRICS = ("global_rel_l2", "min_cosine")
#: The codecs every error set and the layer table must hold.
ERROR_CODECS = CANDIDATES + (CONTROL,)


def error_expectations(error: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """What a complete error experiment must hold, from the run's own recorded protocol
    (``meta.args.error_sets``, ``meta.n_layers``, ``meta.chunk_shape`` and ``meta.config.k``),
    never from the length of whatever arrays survived. Raises MALFORMED on a missing field."""
    meta = (error or {})["meta"]
    sets = int(meta["args"]["error_sets"])
    n_layers = int(meta["n_layers"])
    k = int(meta["config"]["k"])
    chunk = [int(x) for x in meta["chunk_shape"]]
    tokens_per_chunk = math.prod(chunk[:-1])
    if sets < 1 or n_layers < 1 or k < 1 or tokens_per_chunk < 1:
        raise ValueError(f"error.json records a degenerate protocol: sets {sets}, layers "
                         f"{n_layers}, k {k}, chunk {chunk}")
    # Every layer of every micro-batch of every set adds one chunk's rows to each layer's merge.
    return {"sets": sets, "n_layers": n_layers, "k": k,
            "rows_per_layer": sets * k * tokens_per_chunk}


def completeness_row(error: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """V15: the error experiment is complete and every metric E1-E3 and V4 read is finite.

    Sets: exactly the recorded ``error_sets``, ids 0..n-1, each once. Layers: per codec,
    exactly the decoder layers 0..n_layers-1, each once, each merged over every set's k
    chunks (rows = sets x k x tokens per chunk). Gradients: per set, the three exact
    controls compare the same number of tensors, and every codec's agreement covers that
    many tensors, with the same tensor names in every set and codec and a ``min_cosine``
    equal to the minimum of its per-tensor cosines. A missing, non-numeric, NaN or infinite
    metric names its place. Any problem fails the row, so E1-E3 read no verdict and the
    codec verdict is NO VERDICT, never a pass."""
    if error is None or error.get("unreadable"):
        return _row(False, "error.json missing or unreadable")
    try:
        want = error_expectations(error)
    except MALFORMED as exc:
        return _row(False, f"error.json does not record its protocol ({exc!r})")
    problems: List[str] = []
    sets = _records(error, "error_set")
    ids = [rec.get("set") for rec in sets]
    if ids != list(range(want["sets"])):
        problems.append(f"error sets {ids}, expected ids 0..{want['sets'] - 1} once each")
    names: Optional[set] = None
    for rec in sets:
        sid = rec.get("set")
        counts = {key: (rec.get(key) or {}).get("tensors")
                  for key in ("ga_vs_l2l", "a_vs_a", "null_control")}
        control = counts["ga_vs_l2l"]
        if not isinstance(control, int) or control < 1 or len(set(counts.values())) != 1:
            problems.append(f"set {sid}: exact controls compare {counts} tensors")
            continue
        for codec in ERROR_CODECS:
            grad = ((rec.get("codecs") or {}).get(codec) or {}).get("grad")
            if not isinstance(grad, dict):
                problems.append(f"set {sid} {codec}: no gradient agreement")
                continue
            for metric in GRAD_METRICS:
                if not _finite(grad.get(metric)):
                    problems.append(f"set {sid} {codec}: grad.{metric} = {grad.get(metric)!r}")
            per = grad.get("per_tensor")
            if grad.get("tensors") != control or not isinstance(per, dict) or len(per) != control:
                problems.append(f"set {sid} {codec}: {grad.get('tensors')!r} gradient tensors "
                                f"({len(per) if isinstance(per, dict) else 'no'} per-tensor "
                                f"rows), exact control has {control}")
                continue
            cosines = [(entry or {}).get("cosine") for entry in per.values()]
            if not all(_finite(c) for c in cosines):
                problems.append(f"set {sid} {codec}: a per-tensor cosine is not finite")
            elif _finite(grad.get("min_cosine")) and min(cosines) != grad["min_cosine"]:
                problems.append(f"set {sid} {codec}: min_cosine {grad['min_cosine']} is not the "
                                f"minimum of its per-tensor cosines ({min(cosines)})")
            if names is None:
                names = set(per)
            elif set(per) != names:
                problems.append(f"set {sid} {codec}: per-tensor names differ from set 0")
    layers = _records(error, "layers")
    if len(layers) != 1:
        problems.append(f"{len(layers)} layers records, expected 1")
    table = (layers[-1].get("codecs") or {}) if layers else {}
    for codec in ERROR_CODECS:
        rows = table.get(codec)
        if not isinstance(rows, list):
            problems.append(f"layers {codec}: missing")
            continue
        seen = [row.get("layer") for row in rows]
        if seen != list(range(want["n_layers"])):
            problems.append(f"layers {codec}: {len(seen)} rows, layers {_span(seen)}, expected "
                            f"0..{want['n_layers'] - 1} once each in order")
        for row in rows:
            for part, metric in LAYER_METRICS:
                value = (row.get(part) or {}).get(metric)
                if not _finite(value):
                    problems.append(f"layers {codec} layer {row.get('layer')}: "
                                    f"{part}.{metric} = {value!r}")
            for part in ("restore", "branch", "output"):
                count = (row.get(part) or {}).get("rows")
                if count != want["rows_per_layer"]:
                    problems.append(f"layers {codec} layer {row.get('layer')}: {part} merged "
                                    f"{count!r} rows, expected {want['rows_per_layer']}")
    if problems:
        return _row(False, _listed(problems))
    return _row(True, f"{want['sets']} sets x {len(ERROR_CODECS)} codecs, {want['n_layers']} "
                      f"layers each, {want['rows_per_layer']} rows per layer, gradient tensor "
                      f"counts equal to the exact control, every metric finite")


def _span(values: Sequence[Any]) -> str:
    ints = [v for v in values if isinstance(v, int)]
    dup = sorted({v for v in ints if ints.count(v) > 1})
    span = f"{min(ints)}..{max(ints)}" if ints else "none"
    return span + (f", duplicated {dup}" if dup else "")


def error_validity(error: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    sets = _records(error, "error_set")
    layers = _records(error, "layers")
    rows: Dict[str, Dict[str, Any]] = {}
    ok, why = _eligible(error)
    rows["V1_error_eligible"] = _row(ok, f"error.json: {why}")
    if not sets or not layers:
        detail = "error.json has no error_set or layers record"
        for name in ("V2_harness_exact", "V3_forward_exact", "V4_codec_conforms"):
            rows[name] = _row(False, detail)
        return rows
    problems = []
    for rec in sets:
        for key in ("ga_vs_l2l", "a_vs_a", "null_control"):
            cmp = rec.get(key) or {}
            if not cmp.get("equal") or cmp.get("all_zero"):
                problems.append(f"set {rec.get('set')}: {key} not exact or has all-zero grads")
            if rec.get(f"{key}_losses_equal") is not True:
                problems.append(f"set {rec.get('set')}: {key} losses differ")
    rows["V2_harness_exact"] = _row(
        not problems,
        "; ".join(problems) or f"GA == L2L, recording == plain, identity == plain in "
        f"{len(sets)} sets (grads and losses torch.equal)")
    leaks = [f"set {rec.get('set')}: {name}" for rec in sets
             for name in CANDIDATES + (CONTROL,)
             if (rec.get("codecs", {}).get(name) or {}).get("losses_equal") is not True]
    rows["V3_forward_exact"] = _row(
        not leaks, "per-micro-batch losses differ: " + ", ".join(leaks) if leaks
        else "every codec's per-micro-batch losses torch.equal to exact L2L")
    table = layers[-1].get("codecs", {})
    broken = []
    for name, bound in ELEMENT_BOUND.items():
        per_layer = table.get(name)
        if not per_layer:
            broken.append(f"{name}: missing")
            continue
        values = [row["restore"]["max_abs_over_absmax"] for row in per_layer]
        bad = [v for v in values if not _finite(v)]
        if bad:  # max() would skip a NaN that is not first; V15 names its layer
            broken.append(f"{name}: non-finite element error {bad[:3]}")
            continue
        worst = max(values)
        if not worst <= bound:
            broken.append(f"{name}: {worst} > {bound:.5f}")
    rows["V4_codec_conforms"] = _row(
        not broken, "; ".join(broken) or "every codec's element error within its definition")
    return rows


def error_gates(error: Optional[Dict[str, Any]], codec: str) -> Dict[str, Dict[str, Any]]:
    """E1-E3 for ``codec``. A gate whose inputs are incomplete or not finite reads no verdict
    with the reason; ``max``/``min`` never see a NaN (they would skip one that is not first)."""
    names = ("E1", "E2", "E3")
    complete = completeness_row(error)
    if not complete["ok"]:
        return {name: _gate(NONE, f"V15: {complete['detail']}") for name in names}
    sets = _records(error, "error_set")
    layers = _records(error, "layers")
    table = (layers[-1].get("codecs", {}) if layers else {}).get(codec)
    if not table or not sets:
        return {name: _gate(NONE, "no data") for name in names}
    rms = max(row["restore"]["rms_row_rel"] for row in table)
    worst_row = max(row["restore"]["worst_row_rel"] for row in table)
    over = [row["layer"] for row in table
            if not (row["restore"]["rms_row_rel"] <= RESTORE_RMS_BOUND
                    and row["restore"]["worst_row_rel"] <= RESTORE_WORST_ROW_BOUND)]
    e1 = _gate(PASS if not over else FAIL,
               f"max over layers: rms row rel {rms:.4g} (bound {RESTORE_RMS_BOUND}), "
               f"worst row rel {worst_row:.4g} (bound {RESTORE_WORST_ROW_BOUND}); "
               f"layers over: {over}", max_rms_row_rel=rms, max_worst_row_rel=worst_row,
               layers_over=over)
    branch = max(row["branch"]["rms_row_rel"] for row in table)
    over2 = [row["layer"] for row in table
             if not row["branch"]["rms_row_rel"] <= BRANCH_RMS_BOUND]
    output = max(row["output"]["rms_row_rel"] for row in table)
    e2 = _gate(PASS if not over2 else FAIL,
               f"max branch rms row rel {branch:.4g} (bound {BRANCH_RMS_BOUND}); "
               f"layers over: {over2}; layer output (residual included) {output:.4g}",
               max_branch_rms_row_rel=branch, max_output_rms_row_rel=output, layers_over=over2)
    grads = [rec["codecs"][codec]["grad"] for rec in sets]  # V15: present and finite
    global_rel = max(grad["global_rel_l2"] for grad in grads)
    min_cos = min(grad["min_cosine"] for grad in grads)
    ok = global_rel <= GRAD_GLOBAL_BOUND and min_cos >= GRAD_MIN_COSINE
    e3 = _gate(PASS if ok else FAIL,
               f"worst set: global rel L2 {global_rel:.4g} (bound {GRAD_GLOBAL_BOUND}), "
               f"min tensor cosine {min_cos:.6f} (bound {GRAD_MIN_COSINE})",
               max_global_rel_l2=global_rel, min_cosine=min_cos)
    return {"E1": e1, "E2": e2, "E3": e3}


def control_row(error: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """V5: the coarse control must FAIL E1, E2 and E3, or that metric does not discriminate."""
    gates = error_gates(error, CONTROL)
    passing = [name for name, gate in gates.items() if gate["status"] != FAIL]
    return _row(not passing, f"{CONTROL} must fail every error gate; "
                + (f"it does not fail {passing}" if passing else "it fails E1, E2 and E3"))


# ==========================================================================
# the training arms
# ==========================================================================
def _arm_name(codec: str, seed: int) -> str:
    return f"train-{codec}-seed{seed}.json"


#: The four training arms the rule reads, and the (codec, seed) each file must have run.
ARM_IDENTITY: Dict[str, Tuple[str, int]] = {
    **{_arm_name(codec, MAIN_SEED): (codec, MAIN_SEED) for codec in (EXACT,) + CANDIDATES},
    _arm_name(EXACT, NOISE_SEED): (EXACT, NOISE_SEED),
}
#: What error.json and every training arm must share: the model, the code, the
#: dependencies, the data file and the streamed setup the codecs are measured in.
LINKED_META = ("harness_sha256", "git_head", "versions", "gpu", "model_files_sha256")
LINKED_DATA = ("sha256", "split_sha256")
LINKED_CONFIG = ("k", "seq", "lora_r", "lora_alpha", "lora_dropout", "lora_targets", "quant",
                 "tier", "dtype", "deterministic", "model_revision", "device")


def _config_diff(left: Dict[str, Any], right: Dict[str, Any]) -> List[str]:
    return sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))


def linkage_row(error: Optional[Dict[str, Any]],
                arms: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """V14: error.json measured the same model, code, dependencies, data file and
    streamed setup as the training arms, so E1-E3 and Q1-Q3 are about one experiment."""
    if error is None or error.get("unreadable"):
        return _row(False, "error.json missing or unreadable")
    present = [(name, payload) for name, payload in arms.items() if payload is not None]
    if not present:
        return _row(False, "no training arm to link error.json to")
    meta = error.get("meta", {})
    config = meta.get("config")
    if not isinstance(config, dict):
        return _row(False, "error.json records no experiment config")
    diffs = []
    for name, payload in present:
        other = payload.get("meta", {})
        diffs += [f"{name}: {key}" for key in LINKED_META if meta.get(key) != other.get(key)]
        diffs += [f"{name}: data.{key}" for key in LINKED_DATA
                  if meta.get("data", {}).get(key) != other.get("data", {}).get(key)]
        diffs += [f"{name}: config.{key}" for key in LINKED_CONFIG
                  if config.get(key) != other.get("config", {}).get(key)]
    return _row(not diffs, "; ".join(diffs) or
                "error.json and every arm: same model, code, dependencies, data and setup")


def training_validity(arms: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    missing = [name for name, payload in arms.items() if payload is None]
    ineligible = [f"{name}: {_eligible(p)[1]}" for name, p in arms.items()
                  if p is not None and not _eligible(p)[0]]
    rows["V6_training_eligible"] = _row(
        not missing and not ineligible,
        "; ".join([f"missing {missing}"] * bool(missing) + ineligible) or "four cuda arms")
    present = {name: p for name, p in arms.items() if p is not None}
    problems = []
    for name, payload in present.items():
        config = payload.get("meta", {}).get("config", {})
        log = payload.get("log", [])
        final = (payload.get("final") or {}).get("task_loss")
        if payload.get("stopped") or len(log) != config.get("steps") or not config.get("steps"):
            problems.append(f"{name}: {len(log)} of {config.get('steps')} steps "
                            f"({payload.get('stopped')})")
        if any(not (_finite(e.get("loss")) and _finite(e.get("grad_norm"))) for e in log):
            problems.append(f"{name}: non-finite loss or grad norm")
        if log and not (_finite(log[0].get("grad_norm")) and log[0]["grad_norm"] > 0):
            problems.append(f"{name}: first-step grad norm is not > 0")
        if not _finite(final):
            problems.append(f"{name}: final task loss {final}")
    rows["V7_arms_complete"] = _row(
        bool(present) and not problems and not missing,
        "; ".join(problems) or ("arms missing" if missing else "every arm ran every step, finite"))
    same_keys = ("sha256", "train_ids_sha256", "panel_ids_sha256", "split_sha256")
    meta_keys = ("harness_sha256", "git_head", "versions", "gpu", "model_files_sha256")
    diffs = []
    reference = next(iter(present.values()), None)
    for name, payload in present.items():
        data, ref_data = payload["meta"].get("data", {}), reference["meta"].get("data", {})
        diffs += [f"{name}: data.{key}" for key in same_keys if data.get(key) != ref_data.get(key)]
        diffs += [f"{name}: {key}" for key in meta_keys
                  if payload["meta"].get(key) != reference["meta"].get(key)]
        revision = payload["meta"].get("config", {}).get("model_revision")
        if not revision:
            diffs.append(f"{name}: no model revision recorded")
    rows["V8_same_inputs"] = _row(
        bool(present) and not diffs,
        "; ".join(diffs) or "same data, panel, harness, commit, versions, GPU and model")
    exact = arms.get(_arm_name(EXACT, MAIN_SEED))
    config_problems = []
    for name, payload in present.items():
        # The file name is the claim; the config is what the process ran. A bf16 record
        # saved as train-int8-seed1.json must not stand in for the int8 arm.
        codec, seed = ARM_IDENTITY[name]
        config = payload["meta"].get("config", {})
        if config.get("codec") != codec or config.get("seed") != seed:
            config_problems.append(f"{name} ran codec {config.get('codec')!r} seed "
                                   f"{config.get('seed')!r}, not {codec!r} seed {seed}")
    if exact is not None:
        base = exact["meta"].get("config", {})
        for name, payload in present.items():
            diff = _config_diff(base, payload["meta"].get("config", {}))
            allowed = {"codec"} if name.endswith(f"seed{MAIN_SEED}.json") else {"seed"}
            if set(diff) - allowed:
                config_problems.append(f"{name} differs in {sorted(set(diff) - allowed)}")
    rows["V9_configs"] = _row(
        exact is not None and not config_problems,
        "; ".join(config_problems) or ("exact arm missing" if exact is None else
                                       "every arm ran its own (codec, seed); codec arms "
                                       "differ only in codec, noise arm only in seed"))
    first = {}
    for codec in (EXACT,) + CANDIDATES:
        payload = arms.get(_arm_name(codec, MAIN_SEED))
        if payload and payload.get("log"):
            first[codec] = payload["log"][0].get("mb_losses")
    values = list(first.values())
    rows["V10_first_step_forward_exact"] = _row(
        len(values) == 1 + len(CANDIDATES) and all(v == values[0] for v in values)
        and values[0] is not None,
        f"first-step per-micro-batch losses of the main-seed arms: {first}")
    return rows


def _tail(log: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    start = int(math.floor(len(log) * (1 - CURVE_TAIL)))
    return list(log[start:])


# -- the held-out panel (V16; added 2026-10-29, after the GPU run) ---------------
def _task_loss(per_example: Sequence[Dict[str, Any]]) -> Optional[float]:
    tokens = sum(int(ex["tokens"]) for ex in per_example)
    return sum(float(ex["nll"]) for ex in per_example) / tokens if tokens else None


def _rounding_bound(rows: int) -> float:
    """The relative difference allowed between a recorded task loss and the rule's
    recomputation of it over ``rows`` examples. Float arithmetic, not a quality margin: in
    any order, a recursive float64 sum of n non-negative terms has a relative error of at
    most (n - 1) u to first order (u = eps / 2), and the division adds u, so two evaluations
    that differ only in summation order differ by less than (n + 1) eps, relative. The rule
    sums in the record's own order with the probe's own expression, so a faithful record
    reproduces its value exactly; the bound only admits another order. 512 rows: 1.1e-13,
    against the 5e-3 Q1 floor, which it never touches."""
    return (rows + 1) * sys.float_info.epsilon


def _panel_entry(ex: Any) -> bool:
    """A panel row as panel_nll writes it: an integer row id, a non-negative integer token
    count and a finite non-negative NLL, 0 when no token is scored (a prompt that fills the
    window scores none, and g4_data keeps such rows)."""
    return (isinstance(ex, dict) and _count(ex.get("row")) and _count(ex.get("tokens"))
            and _finite(ex.get("nll")) and ex["nll"] >= 0
            and (ex["tokens"] > 0 or ex["nll"] == 0))


def _panel(name: str, payload: Optional[Dict[str, Any]]
           ) -> Tuple[List[str], Optional[List[Dict[str, Any]]]]:
    """One arm's held-out panel against the arm's own recorded protocol
    (``meta.data.panel_examples`` and ``panel_response_tokens``, written when the panel was
    encoded), never against the length of whatever rows survived: the problems, and the
    rows when there are none."""
    if payload is None or payload.get("unreadable"):
        return [f"{name}: missing or unreadable"], None
    data = (payload.get("meta") or {}).get("data")
    data = data if isinstance(data, dict) else {}
    examples, tokens = data.get("panel_examples"), data.get("panel_response_tokens")
    if not (_count(examples) and examples > 0 and _count(tokens) and tokens > 0):
        return [f"{name}: the panel protocol is not recorded or empty "
                f"(meta.data.panel_examples {examples!r}, panel_response_tokens {tokens!r})"], None
    final = payload.get("final")
    rows = final.get("per_example") if isinstance(final, dict) else None
    if not isinstance(rows, list):
        return [f"{name}: no final.per_example table"], None
    problems = []
    if len(rows) != examples:
        problems.append(f"{name}: {len(rows)} panel rows, the protocol has {examples}")
    for key, want in (("examples", examples), ("tokens", tokens)):
        if not (_count(final.get(key)) and final[key] == want):
            problems.append(f"{name}: final.{key} {final.get(key)!r}, the protocol has {want}")
    bad = [index for index, ex in enumerate(rows) if not _panel_entry(ex)]
    if bad:
        problems.append(f"{name}: {len(bad)} malformed panel rows, first {rows[bad[0]]!r:.80}")
        return problems, None
    repeated = sorted(row for row, seen in Counter(ex["row"] for ex in rows).items() if seen > 1)
    if repeated:
        problems.append(f"{name}: repeated panel row ids {repeated[:4]}")
    scored = sum(ex["tokens"] for ex in rows)
    if scored != tokens:
        problems.append(f"{name}: the rows score {scored} tokens, the protocol has {tokens}")
    if scored:
        recomputed = _task_loss(rows)
        recorded = final.get("task_loss")
        if not (_finite(recorded)
                and abs(recorded - recomputed) <= _rounding_bound(len(rows)) * recomputed):
            problems.append(f"{name}: final.task_loss {recorded!r} is not its rows' "
                            f"token-weighted NLL {recomputed!r}")
    return problems, (None if problems else rows)


def verified_panels(arms: Dict[str, Optional[Dict[str, Any]]]
                    ) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
    """Every training arm's panel checked on its own (``_panel``), then against the exact
    arm's (or the first sound one's): the same row ids with the same token counts. Returns
    the rows of each arm that passes both, and the problems. Only these rows reach the noise
    floor, Q1 and the paired bootstrap, which therefore never pair a subset of rows or a
    row a repeated id overwrote."""
    own: Dict[str, List[Dict[str, Any]]] = {}
    problems: List[str] = []
    for name in ARM_IDENTITY:
        found, rows = _panel(name, arms.get(name))
        problems += found
        if rows is not None:
            own[name] = rows
    verified: Dict[str, List[Dict[str, Any]]] = {}
    reference = next(iter(own), None)
    if reference is not None:
        want = {ex["row"]: ex["tokens"] for ex in own[reference]}
        for name, rows in own.items():
            have = {ex["row"]: ex["tokens"] for ex in rows}
            if have == want:
                verified[name] = rows
                continue
            ids = len(set(have) ^ set(want))
            counts = sum(1 for row in set(have) & set(want) if have[row] != want[row])
            problems.append(f"{name}: panel differs from {reference} ({ids} row ids not in "
                            f"both, {counts} rows with other token counts)")
    return verified, problems


def panel_row(arms: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """V16: every arm's held-out panel is complete and consistent (``verified_panels``), so
    Q1 and its noise floor read the token-weighted NLL of the whole recorded panel. A row
    count, row id, token count, NLL, ``final.examples`` / ``final.tokens`` or
    ``final.task_loss`` that disagrees with the recorded protocol or with the other arms
    fails the row: no verdict, never a pass, and never silently repaired."""
    verified, problems = verified_panels(arms)
    if problems:
        return _row(False, _listed(problems))
    rows = next(iter(verified.values()))
    empty = sum(1 for ex in rows if not ex["tokens"])
    return _row(True, f"{len(verified)} arms x {len(rows)} panel rows as recorded, unique ids, "
                      f"the same rows and token counts in every arm "
                      f"({sum(ex['tokens'] for ex in rows)} scored tokens; {empty} rows score "
                      f"none); every final.task_loss is its rows' token-weighted NLL")


def noise_floor(arms: Dict[str, Optional[Dict[str, Any]]]) -> Optional[Dict[str, float]]:
    exact = arms.get(_arm_name(EXACT, MAIN_SEED))
    noise = arms.get(_arm_name(EXACT, NOISE_SEED))
    if not exact or not noise:
        return None
    panels, _problems = verified_panels(arms)
    exact_rows = panels.get(_arm_name(EXACT, MAIN_SEED))
    noise_rows = panels.get(_arm_name(EXACT, NOISE_SEED))
    if exact_rows is None or noise_rows is None:
        return None
    tl_exact, tl_noise = _task_loss(exact_rows), _task_loss(noise_rows)
    tail_a, tail_b = _tail(exact["log"]), _tail(noise["log"])
    if not tail_a or len(tail_a) != len(tail_b):
        return None
    curve = statistics.fmean(abs(a["loss"] - b["loss"]) for a, b in zip(tail_a, tail_b))
    tail_mean = statistics.fmean(a["loss"] for a in tail_a)
    return {
        "task_loss_exact": tl_exact, "task_loss_noise_seed": tl_noise,
        "task_loss_noise": abs(tl_noise - tl_exact),
        "task_loss_margin": max(abs(tl_noise - tl_exact), QUALITY_FLOOR_REL * tl_exact),
        "curve_noise": curve, "curve_tail_mean": tail_mean,
        "curve_margin": max(curve, QUALITY_FLOOR_REL * tail_mean),
    }


def paired_bootstrap(exact: Sequence[Dict[str, Any]], other: Sequence[Dict[str, Any]]
                     ) -> Optional[Dict[str, float]]:
    """95% interval of task loss(other) - task loss(exact), resampling panel examples
    whole (seed 17, 2000 resamples). Informational: panel sampling, not training noise."""
    by_row = {ex["row"]: ex for ex in exact}
    if (len(by_row) != len(exact) or len(other) != len(exact)
            or {ex["row"] for ex in other} != set(by_row)):
        return None
    pairs = [(by_row[ex["row"]], ex) for ex in other]
    if not pairs:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(pairs)
    deltas = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        tokens = sum(a["tokens"] for a, _b in sample)
        if not tokens:
            continue
        deltas.append((sum(b["nll"] for _a, b in sample) - sum(a["nll"] for a, _b in sample))
                      / tokens)
    deltas.sort()
    if not deltas:
        return None
    return {"low": deltas[int(0.025 * len(deltas))], "high": deltas[int(0.975 * len(deltas)) - 1]}


def quality_gates(arms: Dict[str, Optional[Dict[str, Any]]], codec: str,
                  floor: Optional[Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
    exact_name, cand_name = _arm_name(EXACT, MAIN_SEED), _arm_name(codec, MAIN_SEED)
    exact, cand = arms.get(exact_name), arms.get(cand_name)
    if not exact or not cand or floor is None:
        return {"Q1": _gate(NONE, "arm or noise floor missing"),
                "Q2": _gate(NONE, "arm or noise floor missing")}
    panels, _problems = verified_panels(arms)
    unverified = [name for name in (exact_name, cand_name) if name not in panels]
    if unverified:
        q1 = _gate(NONE, f"V16: the panel of {unverified} is not verified")
    else:
        tl_exact, tl_cand = _task_loss(panels[exact_name]), _task_loss(panels[cand_name])
        delta = tl_cand - tl_exact
        interval = paired_bootstrap(panels[exact_name], panels[cand_name])
        q1 = _gate(PASS if delta <= floor["task_loss_margin"] else FAIL,
                   f"task loss {tl_cand:.6f} vs exact {tl_exact:.6f}: delta {delta:+.6f}, "
                   f"margin {floor['task_loss_margin']:.6f}", delta=delta,
                   margin=floor["task_loss_margin"], panel_bootstrap_95=interval)
    tail_e, tail_c = _tail(exact["log"]), _tail(cand["log"])
    if not tail_e or len(tail_e) != len(tail_c):
        q2 = _gate(NONE, "curves of different length")
    else:
        gap = statistics.fmean(c["loss"] - e["loss"] for e, c in zip(tail_e, tail_c))
        q2 = _gate(PASS if gap <= floor["curve_margin"] else FAIL,
                   f"mean train-loss gap over the last {len(tail_e)} steps {gap:+.6f}, "
                   f"margin {floor['curve_margin']:.6f}", gap=gap, margin=floor["curve_margin"])
    return {"Q1": q1, "Q2": q2}


def _suites_name(codec: str, seed: int) -> str:
    return f"suites-{codec}-seed{seed}.json"


def suites_validity(arms: Dict[str, Optional[Dict[str, Any]]],
                    suites: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """V12: one CUDA suites record per training arm, scoring that arm's exact adapter
    with the same generation settings."""
    problems = []
    settings = set()
    for arm_name, payload in suites.items():
        train = arms.get(arm_name.replace("suites-", "train-"))
        if payload is None:
            problems.append(f"{arm_name} missing")
            continue
        ok, why = _eligible(payload)
        if not ok:
            problems.append(f"{arm_name}: {why}")
        meta = payload.get("meta", {})
        settings.add((meta.get("max_new_tokens"), meta.get("quant")))
        if train is None or meta.get("adapter_sha256") != (train.get("adapter") or {}).get(
                "sha256"):
            problems.append(f"{arm_name}: adapter differs from its training arm's")
        scored = payload.get("suites", {})
        if set(scored) != set(SUITES):
            problems.append(f"{arm_name}: suites {sorted(scored)}")
        bad = [name for name, entry in scored.items()
               if not (isinstance(entry, dict) and isinstance(entry.get("n"), int)
                       and isinstance(entry.get("correct"), int)
                       and 0 <= entry["correct"] <= entry["n"])]
        if bad:
            problems.append(f"{arm_name}: malformed counts in {sorted(bad)}")
    if len(settings) > 1:
        problems.append(f"generation settings differ: {sorted(settings, key=str)}")
    return _row(not problems, "; ".join(problems) or "four suites records, adapters match")


# -- the suites' item tables (V17; added 2026-10-29, after the GPU run) -----------
#: The four suites records, in the order of their training arms.
SUITES_RECORDS = tuple(name.replace("train-", "suites-") for name in ARM_IDENTITY)


def _suite_item(item: Any) -> bool:
    """An item as score_suites writes it: the question and expected answer it scored, the
    model's output, and Soup's ``score_answer`` verdict, a bool."""
    return (isinstance(item, dict) and isinstance(item.get("question"), str)
            and isinstance(item.get("answer"), str) and isinstance(item.get("output"), str)
            and isinstance(item.get("correct"), bool))


def _suite_table(where: str, entry: Any) -> Tuple[List[str], Optional[frozenset]]:
    """One suite of one record against its own item table: the problems, and the item
    identities (recorded question, expected answer) when there are none. ``n`` must be the
    number of items and ``correct`` the number the items mark correct."""
    items = entry.get("items") if isinstance(entry, dict) else None
    if not isinstance(items, list) or not items:
        return [f"{where}: no item table"], None
    bad = [index for index, item in enumerate(items) if not _suite_item(item)]
    if bad:
        return [f"{where}: {len(bad)} malformed items, first {items[bad[0]]!r:.80}"], None
    problems = []
    keys = [(item["question"], item["answer"]) for item in items]
    repeated = sum(1 for seen in Counter(keys).values() if seen > 1)
    if repeated:
        problems.append(f"{where}: {repeated} items scored more than once")
    n, correct = entry.get("n"), entry.get("correct")
    right = sum(item["correct"] for item in items)
    if not (_count(n) and n == len(items)):
        problems.append(f"{where}: n {n!r}, the table has {len(items)} items")
    if not (_count(correct) and correct == right):
        problems.append(f"{where}: correct {correct!r}, the table marks {right} correct")
    return problems, (None if problems else frozenset(keys))


def verified_suites(suites: Dict[str, Optional[Dict[str, Any]]]
                    ) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Every suites record's four gating suites checked against their own item tables
    (``_suite_table``), then against the exact arm's record (or the first sound one's): the
    same questions with the same expected answers, hence the same sample counts. Item
    identity is the recorded question and answer, never the output or its score, which may
    differ between arms; the record is compared with the other arms, never with the
    suites of today's code. Returns each record that passes both as {suite: entry}, and the
    problems. Only these reach Q3."""
    own: Dict[str, Dict[str, Any]] = {}
    keys: Dict[str, Dict[str, frozenset]] = {}
    problems: List[str] = []
    for name in SUITES_RECORDS:
        payload = suites.get(name)
        if payload is None or payload.get("unreadable"):
            problems.append(f"{name}: missing or unreadable")
            continue
        scored = payload.get("suites")
        scored = scored if isinstance(scored, dict) else {}
        found, tables = [], {}
        for suite in SUITES:
            more, table = _suite_table(f"{name} {suite}", scored.get(suite))
            found += more
            tables[suite] = table
        problems += found
        if not found:
            own[name] = {suite: scored[suite] for suite in SUITES}
            keys[name] = tables
    verified: Dict[str, Dict[str, Any]] = {}
    reference = next(iter(own), None)
    for name in own:
        differ = []
        for suite in SUITES:
            want, have = keys[reference][suite], keys[name][suite]
            if have != want:
                differ.append(f"{suite} ({len(have - want)} items not in {reference}, "
                              f"{len(want - have)} of its items absent; n {len(have)} vs "
                              f"{len(want)})")
        if differ:
            problems.append(f"{name}: other items than {reference}: " + ", ".join(differ))
        else:
            verified[name] = own[name]
    return verified, problems


def suite_items_row(suites: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """V17: every suites record scored the same complete item tables (``verified_suites``),
    so Q3 compares correct counts of the same questions out of the same n."""
    verified, problems = verified_suites(suites)
    if problems:
        return _row(False, _listed(problems))
    sizes = ", ".join(f"{suite} {entry['n']}"
                      for suite, entry in next(iter(verified.values())).items())
    return _row(True, f"{len(verified)} records score the same questions and answers ({sizes}); "
                      f"n and correct agree with every item table")


def base_parity_validity(arms: Dict[str, Optional[Dict[str, Any]]],
                         suites: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """V13: the resident base the suites score reproduces the streamed model each arm
    trained on. Per suites record: the live adapter equals its saved file, and over the
    parity rows the resident response NLL (recorded by the suites mode) is within
    ``BASE_PARITY_REL`` of the streamed NLL the training arm recorded for the same rows,
    token counts equal. Both task losses are recomputed here from the per-example data."""
    problems, worst = [], 0.0
    for name, payload in suites.items():
        train = arms.get(name.replace("suites-", "train-"))
        if payload is None or train is None:
            problems.append(f"{name}: suites or training record missing")
            continue
        parity = payload.get("base_parity")
        if not parity:
            problems.append(f"{name}: no base_parity record")
            continue
        adapter = parity.get("adapter") or {}
        if adapter.get("mismatched") != 0 or not adapter.get("tensors"):
            problems.append(f"{name}: live adapter differs from its file ({adapter})")
        resident = parity["per_example_resident"]
        by_row = {ex["row"]: ex for ex in train["final"]["per_example"]}
        streamed = [by_row.get(ex["row"]) for ex in resident]
        if not resident or any(ex is None for ex in streamed):
            problems.append(f"{name}: parity rows missing from the training record")
            continue
        if any(int(a["tokens"]) != int(b["tokens"]) for a, b in zip(resident, streamed)):
            problems.append(f"{name}: token counts differ between resident and streamed")
        tl_res, tl_str = _task_loss(resident), _task_loss(streamed)
        if not (_finite(tl_res) and _finite(tl_str) and tl_str > 0):
            problems.append(f"{name}: task loss resident {tl_res}, streamed {tl_str}")
            continue
        rel = abs(tl_res - tl_str) / tl_str
        worst = max(worst, rel)
        if rel > BASE_PARITY_REL:
            problems.append(f"{name}: resident {tl_res:.6f} vs streamed {tl_str:.6f} "
                            f"(rel {rel:.3g} > {BASE_PARITY_REL})")
    return _row(not problems, "; ".join(problems) or (
        f"resident base reproduces the streamed model on every arm's parity rows (worst rel "
        f"{worst:.3g} <= {BASE_PARITY_REL}); live adapters equal their files"))


def suite_gate(suites: Dict[str, Optional[Dict[str, Any]]], codec: str) -> Dict[str, Any]:
    """Q3: in every gating suite the codec arm answers no more than ``margin`` fewer
    items correctly than exact, with margin = max(SUITE_MIN_MARGIN, seed-to-seed
    difference of the exact arm in that suite). A suite the exact arm answers all
    right or all wrong is reported and not gated (C5, issue #1192)."""
    names = (_suites_name(EXACT, MAIN_SEED), _suites_name(EXACT, NOISE_SEED),
             _suites_name(codec, MAIN_SEED))
    if not all(suites.get(name) for name in names):
        return _gate(NONE, "suites record missing")
    verified, _problems = verified_suites(suites)
    unverified = [name for name in names if name not in verified]
    if unverified:
        return _gate(NONE, f"V17: the item tables of {unverified} are not verified")
    rows, failing = {}, []
    for name in SUITES:
        e, n, c = (verified[record][name] for record in names)
        margin = max(SUITE_MIN_MARGIN, abs(n["correct"] - e["correct"]))
        gating = 0 < e["correct"] < e["n"]
        drop = e["correct"] - c["correct"]
        # V17: the three tables hold the same items, so one n is every arm's n.
        rows[name] = {"exact": e["correct"], "noise_seed": n["correct"], "codec": c["correct"],
                      "n": e["n"], "drop": drop, "margin": margin, "gating": gating}
        if gating and drop > margin:
            failing.append(name)
    detail = ", ".join(f"{name} {row['codec']}/{row['n']} vs {row['exact']} (margin "
                       f"{row['margin']}{'' if row['gating'] else ', not gating'})"
                       for name, row in rows.items())
    return _gate(FAIL if failing else PASS, detail, suites=rows, failing=failing)


# ==========================================================================
# memory and timing (informational)
# ==========================================================================
def memory_info(arms: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, payload in arms.items():
        if not payload:
            continue
        memory = payload.get("memory") or {}
        store = memory.get("store") or {}
        out[name] = {
            "measured_bytes_per_token_per_layer": store.get("measured_bytes_per_token_per_layer"),
            "arithmetic_bytes_per_token_per_layer":
                store.get("arithmetic_bytes_per_token_per_layer"),
            "store_host_bytes": store.get("host_bytes"),
            "peak_alloc_gb": memory.get("peak_alloc_gb"),
            "peak_reserved_gb": memory.get("peak_reserved_gb"),
            "compress_s": (payload.get("codec_stats") or {}).get("compress_s"),
            "restore_s": (payload.get("codec_stats") or {}).get("restore_s"),
            "train_s": (payload.get("durations") or {}).get("train_s"),
        }
    return out


def timing_info(timing: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not timing:
        return {"present": False}
    out: Dict[str, Any] = {"present": True, "comparisons": {}}
    for comp in timing.get("comparisons", []):
        valid = [block for block in comp.get("blocks", [])
                 if all(run.get("outcome") != "void" for run in block)]
        a = [run["value"] for block in valid for run in block if run["arm"] == "A"]
        b = [run["value"] for block in valid for run in block if run["arm"] == "B"]
        needed = comp.get("blocks_needed")
        ratio = statistics.fmean(b) / statistics.fmean(a) if a and b else None
        out["comparisons"][comp.get("codec")] = {
            "valid_blocks": len(valid), "blocks_needed": needed,
            "repeats_rule_met": needed is not None and len(valid) >= needed
            and not comp.get("stopped"),
            "mean_step_s_bf16": statistics.fmean(a) if a else None,
            "mean_step_s_codec": statistics.fmean(b) if b else None,
            "step_time_ratio_codec_over_bf16": ratio,
            "note": "a time ratio, not a speed-up from the memory saving",
        }
    return out


# ==========================================================================
# the verdict
# ==========================================================================
def evaluate(results: Path) -> Dict[str, Any]:
    error = _load(results / "error.json")
    arms = {_arm_name(codec, MAIN_SEED): _load(results / _arm_name(codec, MAIN_SEED))
            for codec in (EXACT,) + CANDIDATES}
    arms[_arm_name(EXACT, NOISE_SEED)] = _load(results / _arm_name(EXACT, NOISE_SEED))
    error_rows = ("V1_error_eligible", "V2_harness_exact", "V3_forward_exact",
                  "V4_codec_conforms")
    validity = _rows_or_fail(error_rows, error_validity, error)
    validity["V5_control_discriminates"] = _rows_or_fail(
        ("V5",), lambda e: {"V5": control_row(e)}, error)["V5"]
    validity["V15_error_complete"] = _rows_or_fail(
        ("V15",), lambda e: {"V15": completeness_row(e)}, error)["V15"]
    error_ok = all(row["ok"] for row in validity.values())
    validity.update(_rows_or_fail(
        ("V6_training_eligible", "V7_arms_complete", "V8_same_inputs", "V9_configs",
         "V10_first_step_forward_exact"), training_validity, arms))
    try:
        floor = noise_floor(arms)
    except MALFORMED:
        floor = None
    cap_ok = floor is not None and floor["task_loss_noise"] <= NOISE_CAP_REL * floor[
        "task_loss_exact"]
    validity["V11_noise_floor_usable"] = _row(
        cap_ok, f"seed-to-seed task-loss spread {floor and floor['task_loss_noise']} must be "
        f"<= {NOISE_CAP_REL} x exact task loss" if floor else "noise floor not computable")
    suites = {_suites_name(codec, MAIN_SEED): _load(results / _suites_name(codec, MAIN_SEED))
              for codec in (EXACT,) + CANDIDATES}
    suites[_suites_name(EXACT, NOISE_SEED)] = _load(results / _suites_name(EXACT, NOISE_SEED))
    validity["V12_suites_records"] = _rows_or_fail(
        ("V12",), lambda a, s: {"V12": suites_validity(a, s)}, arms, suites)["V12"]
    validity["V13_base_parity"] = _rows_or_fail(
        ("V13",), lambda a, s: {"V13": base_parity_validity(a, s)}, arms, suites)["V13"]
    validity["V14_error_linked"] = _rows_or_fail(
        ("V14",), lambda e, a: {"V14": linkage_row(e, a)}, error, arms)["V14"]
    validity["V16_panel_complete"] = _rows_or_fail(
        ("V16",), lambda a: {"V16": panel_row(a)}, arms)["V16"]
    validity["V17_suite_items"] = _rows_or_fail(
        ("V17",), lambda s: {"V17": suite_items_row(s)}, suites)["V17"]
    all_ok = all(row["ok"] for row in validity.values())
    codecs_out: Dict[str, Any] = {}
    for codec in CANDIDATES:
        gates = _gates_or_none(("E1", "E2", "E3"), error_gates, error, codec)
        gates.update(_gates_or_none(("Q1", "Q2"), quality_gates, arms, codec, floor))
        gates["Q3"] = _gates_or_none(
            ("Q3",), lambda s, c: {"Q3": suite_gate(s, c)}, suites, codec)["Q3"]
        error_failed = error_ok and any(gates[g]["status"] == FAIL for g in ("E1", "E2", "E3"))
        if error_failed:
            verdict = FAIL
        elif not all_ok:
            verdict = NONE
        elif any(gate["status"] == FAIL for gate in gates.values()):
            verdict = FAIL
        elif all(gate["status"] == PASS for gate in gates.values()):
            verdict = PASS
        else:
            verdict = NONE
        if not error_ok:
            for name in ("E1", "E2", "E3"):
                gates[name] = {**gates[name], "status": NONE,
                               "detail": "error validity failed; " + gates[name]["detail"]}
        if not all_ok:
            for name in ("Q1", "Q2", "Q3"):
                gates[name] = {**gates[name], "status": NONE,
                               "detail": "validity failed; " + gates[name]["detail"]}
        codecs_out[codec] = {"verdict": verdict, "gates": gates}
    verdicts = [entry["verdict"] for entry in codecs_out.values()]
    passing = [codec for codec, entry in codecs_out.items() if entry["verdict"] == PASS]
    if passing:
        overall = "PASS: " + ", ".join(passing)
    elif all(v == FAIL for v in verdicts):
        overall = "FAIL"
    else:
        overall = "NO VERDICT"
    try:
        memory = memory_info(arms)
    except MALFORMED as exc:
        memory = {"error": f"malformed input: {exc!r}"}
    try:
        timing = timing_info(_load(results / "timing.json"))
    except MALFORMED as exc:
        timing = {"present": True, "error": f"malformed input: {exc!r}"}
    return {
        "verdict": overall, "codecs": codecs_out, "validity": validity,
        "noise_floor": floor, "memory": memory, "timing": timing,
        "per_token_arithmetic": codecs.per_token_table(),
        "bounds": {
            "restore_rms_row_rel": RESTORE_RMS_BOUND,
            "restore_worst_row_rel": RESTORE_WORST_ROW_BOUND,
            "branch_rms_row_rel": BRANCH_RMS_BOUND, "grad_global_rel_l2": GRAD_GLOBAL_BOUND,
            "grad_min_cosine": GRAD_MIN_COSINE, "element_bound": ELEMENT_BOUND,
            "quality_floor_rel": QUALITY_FLOOR_REL, "noise_cap_rel": NOISE_CAP_REL,
            "curve_tail": CURVE_TAIL, "suites": SUITES, "suite_min_margin": SUITE_MIN_MARGIN,
            "base_parity_rel": BASE_PARITY_REL,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("--json", type=Path, default=None, help="also write the verdict here")
    args = parser.parse_args(argv)
    out = evaluate(args.results)
    for name, row in out["validity"].items():
        print(f"{name:<32} {'ok' if row['ok'] else 'FAILS'}  {row['detail']}")
    for codec, entry in out["codecs"].items():
        print(f"\n{codec}: {entry['verdict']}")
        for gate, value in entry["gates"].items():
            print(f"  {gate} {value['status']:<10} {value['detail']}")
    print(f"\nVerdict: {out['verdict']}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
