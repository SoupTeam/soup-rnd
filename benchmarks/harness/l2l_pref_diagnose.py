#!/usr/bin/env python3
"""Ticket A6-08: where the L2L and gradient-accumulation gap of DPO on CUDA comes from.

The rule is ``benchmarks/l2l-preference-diagnosis-rule-a6-ticket08.md``, committed before the first
GPU run. This module holds the judges (pure functions over recorded arms) and the driver that
records the arms through the real DPO wrapper. Benchmark code, not shipped. torch is imported
inside functions.

An arm is ``{"grads": [{name: tensor}, ...], "losses": [float, ...]}``, as in ``l2l_pref_check``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import l2l_pref_check  # noqa: E402

F32 = "torch.float32"
#: Fields of a dtype record and the value each must have for the float32 patch to count as taken.
CONFIRMED = {
    "stream_dtype": "float32",
    "train_bf16": False,
    "train_fp16": False,
    "mixed_precision": "no",
    "base_weight": F32,
    "lora_weight": F32,
    "layer_in": F32,
    "layer_out": F32,
}


def _type_problems(arm: str, record: Optional[Dict[str, Any]]) -> List[str]:
    """Why the dtype record of ``arm`` does not confirm float32 with autocast off."""
    if record is None:
        return [f"{arm}: no type record"]
    out = [
        f"{arm}: {field} is {record.get(field)!r}, float32 needs {wanted!r}"
        for field, wanted in CONFIRMED.items()
        if record.get(field) != wanted
    ]
    grads = record.get("grads")
    if not grads or any(kind != F32 for kind in grads):
        out.append(f"{arm}: grads is {grads!r}, float32 needs only {F32!r}")
    return out


def judge_float32(
    ga1: Optional[Dict[str, Any]],
    ga2: Optional[Dict[str, Any]],
    l2l: Optional[Dict[str, Any]],
    dtypes: Dict[str, Any],
) -> Dict[str, Any]:
    """Control 1: float32 L2L against float32 accumulation, bit for bit (rule, control 1).

    ``unavailable`` when an arm has no valid record, a dtype record does not confirm float32
    with autocast off, or the accumulation does not repeat itself bit for bit.
    """
    arms = {"f32_ga1": ga1, "f32_ga2": ga2, "f32_l2l": l2l}
    unavailable = [f"{name}: no valid arm" for name, arm in arms.items() if arm is None]
    for name in arms:
        unavailable += _type_problems(name, dtypes.get(name))
    if not unavailable:
        repeat = l2l_pref_check._differences(ga1, ga2)
        if repeat:
            unavailable.append(f"f32_ga1 against f32_ga2 is not equal: {repeat[0]}")
    if unavailable:
        return {"state": "unavailable", "problems": unavailable}
    problems = l2l_pref_check._differences(ga1, l2l)
    return {"state": "not_equal" if problems else "equal", "problems": problems}


def _first_step(arm: Dict[str, Any]) -> Dict[str, Any]:
    return {"grads": arm["grads"][:1], "losses": arm["losses"][:1]}


def judge_floor(
    ga1: Optional[Dict[str, Any]], floor: Optional[Dict[str, Any]], l2l: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Control 2: L2L against accumulation, with the second SDPA backend as the noise floor.

    The A3-07 rule (``l2l_pref_check._within_floor``): per step and tensor the gap of L2L may be
    1.5 x the gap of ``floor``, and a floor of 0 leaves no room. ``state`` is ``within`` when that
    holds on every step; ``step0_within`` is the same test on step 0 alone, where all arms start
    from the same adapters. ``no_data`` when an arm is missing.
    """
    arms = {"native_ga1": ga1, "native_floor": floor, "native_l2l": l2l}
    missing = [f"{name}: no valid arm" for name, arm in arms.items() if arm is None]
    if missing:
        return {"state": "no_data", "step0_within": None, "problems": missing}
    problems = l2l_pref_check._within_floor(ga1, floor, l2l)
    step0 = l2l_pref_check._within_floor(_first_step(ga1), _first_step(floor), _first_step(l2l))
    return {
        "state": "outside" if problems else "within",
        "step0_within": not step0,
        "problems": problems,
        "max_floor": l2l_pref_check.max_abs_diff(ga1, floor)["grad"],
        "max_gap": l2l_pref_check.max_abs_diff(ga1, l2l)["grad"],
    }


#: Row of the outcome table -> what it means (rule, "Таблица исходов"). The report quotes the rule.
CONCLUSIONS = {
    1: "rounding, no bug; the readiness criterion of the preference methods moves to the A3-07 "
       "tolerance by a new rule before a new run (not in this ticket)",
    2: "rounding larger than the accumulation noise; the tolerance is a user decides matter",
    3: "a bug in the shared L2L path is likely; control 3 names the layer; "
       "fixed in a separate ticket",
    4: "the gap is of the order of the backend noise; precision not separated, a stated limit",
    5: "cause not established",
}


def judge_outcome(float32: str, floor: str) -> Dict[str, Any]:
    """The row of the ticket's outcome table from the states of controls 1 and 2.

    ``trace_needed`` and ``trace_dtype`` say whether control 3 runs and in which type: float32
    when control 1 could speak and found a gap, the native type when it was unavailable.
    """
    trace = {"equal": (False, None), "not_equal": (True, "float32")}.get(float32, (True, "native"))
    out: Dict[str, Any] = {"trace_needed": trace[0], "trace_dtype": trace[1]}
    if floor == "no_data":
        return {**out, "row": None, "conclusion": "undefined: control 2 has no valid data"}
    if float32 == "not_equal":
        row = 3
    elif float32 == "equal":
        row = 1 if floor == "within" else 2
    else:
        row = 4 if floor == "within" else 5
    return {**out, "row": row, "conclusion": CONCLUSIONS[row]}


def first_diverging_layer(
    ga: Optional[Dict[str, Any]], l2l: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Control 3: the first output, in order embedding, layer 0, layer 1, ..., that differs bitwise.

    A trace is ``{"embed": tensor, "layers": [tensor, ...]}`` recorded at step 0. ``first`` is
    ``"embed"`` or a layer index; ``differing`` lists every layer index that differs.
    """
    import torch

    if ga is None or l2l is None:
        return {"state": "no_data", "first": None, "reason": "a trace is missing"}
    pairs = [("embed", ga["embed"], l2l["embed"])] + [
        (i, a, b) for i, (a, b) in enumerate(zip(ga["layers"], l2l["layers"]))
    ]
    if len(ga["layers"]) != len(l2l["layers"]) or any(a.shape != b.shape for _, a, b in pairs):
        reason = "recorded shapes or layer counts differ"
        return {"state": "no_data", "first": None, "reason": reason}
    differing = [name for name, a, b in pairs if not torch.equal(a, b)]
    if not differing:
        return {"state": "equal", "first": None, "differing": []}
    first = differing[0]
    a, b = next((a, b) for name, a, b in pairs if name == first)
    return {
        "state": "differs",
        "first": first,
        "differing": [name for name in differing if name != "embed"],
        "embed_differs": "embed" in differing,
        "max_abs_at_first": float((a.double() - b.double()).abs().max()),
    }
