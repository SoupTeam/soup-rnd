#!/usr/bin/env python3
"""Ticket A6-06: L2L against gradient accumulation for DPO, ORPO, SimPO and KTO.

The rule is ``benchmarks/l2l-preference-check-rule-a6.md``, committed before the first GPU run.
This module holds the judging code (pure functions over recorded arms) and the driver that
records the arms through the real trainer wrappers. Benchmark code, not shipped. torch is
imported inside functions.

An arm is ``{"grads": [{name: tensor}, ...], "losses": [float, ...]}``: the LoRA gradients
before every optimizer step and the logged loss of every step.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def _differences(ref: Dict[str, Any], cand: Dict[str, Any]) -> List[str]:
    """What differs bit for bit between two arms, one sentence each; empty when equal."""
    import torch

    out: List[str] = []
    if len(ref["grads"]) != len(cand["grads"]):
        out.append(f"step count {len(ref['grads'])} against {len(cand['grads'])}")
        return out
    for step, (g_ref, g_cand) in enumerate(zip(ref["grads"], cand["grads"])):
        if g_ref.keys() != g_cand.keys():
            out.append(f"step {step}: gradient keys differ")
            continue
        for name in sorted(g_ref):
            if not torch.equal(g_ref[name], g_cand[name]):
                out.append(f"step {step}: {name} differs")
    for step, (a, b) in enumerate(zip(ref["losses"], cand["losses"])):
        if a != b:
            out.append(f"step {step}: loss {a!r} against {b!r}")
    return out


#: The A3-07 factor: a candidate may differ from the reference by 1.5 x the noise floor.
FLOOR_FACTOR = 1.5


def _within_floor(ref: Dict[str, Any], floor: Dict[str, Any], cand: Dict[str, Any]) -> List[str]:
    """The A3-07 rule with A against A as the floor, per tensor and per step, and on the loss.

    ``floor`` is the second accumulation run. A floor of 0 on a tensor leaves no room: the
    candidate has to equal the reference there. Arms of another shape of record never pass.
    """
    out: List[str] = []
    shapes = [len(arm["grads"]) for arm in (ref, floor, cand)]
    if len(set(shapes)) != 1 or any(
        a["grads"][i].keys() != ref["grads"][i].keys()
        for a in (floor, cand)
        for i in range(shapes[0])
    ):
        return ["the arms do not have the same steps and gradient keys"]
    for step in range(shapes[0]):
        for name in sorted(ref["grads"][step]):
            base = ref["grads"][step][name].double()
            noise = float((floor["grads"][step][name].double() - base).abs().max())
            gap = float((cand["grads"][step][name].double() - base).abs().max())
            if not gap <= FLOOR_FACTOR * noise:
                out.append(
                    f"step {step}: {name} differs by {gap:.3e}, limit {FLOOR_FACTOR * noise:.3e}"
                )
    for step, (a, f, c) in enumerate(zip(ref["losses"], floor["losses"], cand["losses"])):
        noise, gap = abs(f - a), abs(c - a)
        if not gap <= FLOOR_FACTOR * noise:
            out.append(f"step {step}: loss differs by {gap:.3e}, limit {FLOOR_FACTOR * noise:.3e}")
    return out


def _not_worth_comparing(arm: Dict[str, Any], windows: int) -> List[str]:
    """Why an arm cannot back a pass even when it equals the reference: it must be non-vacuous."""
    import math

    import torch

    out: List[str] = []
    if len(arm["grads"]) != windows:
        out.append(f"{len(arm['grads'])} optimizer steps, but the data makes {windows} windows")
    if not all(math.isfinite(loss) for loss in arm["losses"]):
        out.append("a loss is not finite")
    tensors = [t for step in arm["grads"] for t in step.values()]
    if not all(bool(torch.isfinite(t).all()) for t in tensors):
        out.append("a gradient is not finite")
    elif not any(bool((t != 0).any()) for t in tensors):
        out.append("every gradient is zero")
    names = sorted({name for step in arm["grads"] for name in step if ".ref." in name})
    if names:
        out.append(f"gradient on the reference adapter: {names[0]} (.ref.)")
    return out


def judge_cell(
    ga1: Dict[str, Any], ga2: Dict[str, Any], l2l: Dict[str, Any], *, windows: int
) -> Dict[str, Any]:
    """One cell (method, k): ``l2l`` against ``ga1``, with ``ga2`` as the A-against-A control.

    Tier 1 is bit equality. Tier 2 applies only when ``ga2`` differs from ``ga1``: the card
    shows run-to-run noise, and the noise is the floor for the A3-07 factor. Either tier also
    needs the L2L arm to be non-vacuous (``windows`` steps, finite, not all zero, no ``.ref.``).
    """
    a_vs_a_equal = not _differences(ga1, ga2)
    problems = _differences(ga1, l2l)
    tier: Optional[int] = 1
    if problems and not a_vs_a_equal:
        problems = _within_floor(ga1, ga2, l2l)
        tier = 2
    problems = problems + _not_worth_comparing(l2l, windows)
    return {
        "pass": not problems,
        "tier": tier if not problems else None,
        "a_vs_a_equal": a_vs_a_equal,
        "problems": problems,
    }


def _per_step(rows: Sequence[Dict[str, Any]], key: str) -> List[int]:
    """Per-step amounts from the running totals of a training log."""
    totals = [int(row.get(key, 0)) for row in rows]
    return [b - a for a, b in zip([0] + totals, totals)]


def judge_loads(
    method_logs: Sequence[Dict[str, Any]], sft_logs: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """The method's layer loads per step against L2L SFT with the same k (runtime counters).

    Layer loads must be equal at every step, large loads and weight restores not larger.
    Capture loads depend on the micro-batch shape, which differs by method; they are reported.
    """
    problems: List[str] = []
    layer = {"method": _per_step(method_logs, "l2l_layer_loads"),
             "sft": _per_step(sft_logs, "l2l_layer_loads")}
    out: Dict[str, Any] = {
        "layer_loads": layer,
        "capture_layer_loads": {
            "method": int(method_logs[-1].get("l2l_capture_layer_loads", 0)) if method_logs else 0,
            "sft": int(sft_logs[-1].get("l2l_capture_layer_loads", 0)) if sft_logs else 0,
        },
    }
    if len(method_logs) != len(sft_logs):
        problems.append(f"{len(method_logs)} steps against {len(sft_logs)} steps of SFT")
    else:
        for step, (m, s) in enumerate(zip(layer["method"], layer["sft"])):
            if m != s:
                problems.append(f"step {step}: {m} layer loads against {s} of SFT")
        for key, label in (
            ("l2l_large_loads", "large loads"),
            ("l2l_weight_restores", "weight restores"),
        ):
            m_steps, s_steps = _per_step(method_logs, key), _per_step(sft_logs, key)
            out[key] = {"method": m_steps, "sft": s_steps}
            for step, (m, s) in enumerate(zip(m_steps, s_steps)):
                if m > s:
                    problems.append(f"step {step}: {m} {label} against {s} of SFT")
    out.update(problems=problems, **{"pass": not problems})
    return out


KS = (1, 2, 4)
#: Attempts per cell: the first and at most two reruns after a void arm (rule A6).
MAX_ATTEMPTS = 3


def _judge_attempt(attempt: Dict[str, Any]) -> Dict[str, Any]:
    """``excluded`` with a reason, ``no_verdict`` on an unknown required check, or the verdict."""
    if attempt.get("crashed"):
        return {"state": "excluded", "reason": f"crashed: {attempt['crashed']}"}
    for arm, outcome in attempt["arms"].items():
        if outcome == "void":
            return {"state": "excluded", "reason": f"arm {arm} is void"}
    for arm, outcome in attempt["arms"].items():
        if outcome != "ok":
            reason = f"arm {arm} is {outcome}: unknown required check"
            return {"state": "no_verdict", "reason": reason}
    passed = bool(attempt["equality"]["pass"]) and bool(attempt["loads"]["pass"])
    return {"state": "pass" if passed else "fail", "reason": ""}


def _judge_cell_attempts(cell: Dict[str, Any]) -> Dict[str, Any]:
    excluded: List[Dict[str, Any]] = []
    for index, attempt in enumerate(cell["attempts"][:MAX_ATTEMPTS]):
        judged = _judge_attempt(attempt)
        if judged["state"] == "excluded":
            excluded.append({"attempt": index, "reason": judged["reason"]})
            continue
        return {
            "k": cell["k"],
            "verdict": judged["state"],
            "used_attempt": index,
            "excluded": excluded,
            "reason": judged["reason"],
        }
    return {
        "k": cell["k"], "verdict": "no_verdict", "used_attempt": None, "excluded": excluded,
        "reason": f"no valid attempt in {len(cell['attempts'][:MAX_ATTEMPTS])} tries",
    }


def judge_method(cells: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The verdict of one method from its cells: pass, fail or no_verdict.

    A fail on a valid attempt stands, whatever the other cells say. A method with all cells
    passing, valid and present for every k passes; any other case is no verdict.
    """
    judged = [_judge_cell_attempts(cell) for cell in cells]
    missing = [k for k in KS if k not in {cell["k"] for cell in cells}]
    reasons = [f"no cell for k={k}" for k in missing]
    reasons += [
        f"k={cell['k']}: {cell['reason']}" for cell in judged if cell["verdict"] == "no_verdict"
    ]
    if any(cell["verdict"] == "fail" for cell in judged):
        verdict = "fail"
    elif reasons:
        verdict = "no_verdict"
    else:
        verdict = "pass"
    return {"verdict": verdict, "cells": judged, "reason": "; ".join(reasons)}
