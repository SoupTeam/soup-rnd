"""Verdict evaluator for G7: the profile's validity rows, the positive control and the
CPU-bound threshold (ADR 0005).

``evaluate`` is a pure function. Each arm is the dict ``visit_profile.run_arm_process``
returns, ``{"run": <run JSON>, "arm": <arm record>}``, for one token count. A missing arm is
``None``. Every verdict is ``{"verdict", "reason", "evidence"}`` with ``verdict`` one of
``pass``, ``fail`` and ``no_verdict``. For the threshold ``pass`` means the visit is CPU-bound.

- ``timer_is_free``: the timer arm's step is at most 2% longer than the bare arm's.
- ``profile_is_readable``: the profile arm's host time per visit is at most 1.5 times the
  timer arm's (the inflation factor).
- ``control``: at 16 tokens the control arm found both injections. Weight preparation grew
  by 200 us +/- 20% and launches per visit by exactly 50, both against the profile arm, and
  the timer's host time per visit grew by 200 us +/- 20% plus the cost of the 50 launches.
  The timer's baseline is the profile arm's host time per visit, not the timer arm's: the
  control runs the timer and the profiler together, so only a profiled baseline leaves the
  profiler's own stretch out of the difference. The cost of the 50 launches is the growth
  of launch time in the control's profile.
- ``cpu_bound``: at 16 tokens, the timer arm's host time per visit minus weight preparation
  from the profile arm is at least 2 times the GPU kernel time per visit. The launch share
  is evidence and decides nothing.

All medians are over forward visits. A row gets ``no_verdict``, with the cause named, when
an arm it reads is missing, void or unknown, when a number it reads is missing, when the
arms disagree on the token count, or when a row it depends on has no pass. The control and
the threshold also need a readable profile, and the threshold needs a passing control.

Benchmark harness code, not shipped.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import visit_control

PASS = "pass"
FAIL = "fail"
NO_VERDICT = "no_verdict"

#: The timer arm's step may be at most this much longer than the bare arm's.
TIMER_EXCESS_LIMIT = 0.02
#: The profile arm's host time per visit may be at most this many times the timer arm's.
INFLATION_LIMIT = 1.5
#: The control's injections, and how far a measured delay may stray from the spin.
CONTROL_SPIN_MS = visit_control.SPIN_NS / 1e6
CONTROL_TOLERANCE = 0.2
CONTROL_LAUNCHES = visit_control.LAUNCHES
CONTROL_TOKENS = visit_control.CONTROL_TOKENS
#: CPU-bound: capturable host time per visit over GPU kernel time per visit.
CPU_BOUND_RATIO = 2.0

#: Slack for float comparison at an edge, so 2.0 times counts as 2.0 times.
_EPS = 1e-9

Arm = Optional[Mapping[str, Any]]


class _MissingError(Exception):
    """An input the row reads is absent. The text becomes the row's reason."""


def _row(verdict: str, reason: str, **evidence: Any) -> Dict[str, Any]:
    return {"verdict": verdict, "reason": reason, "evidence": evidence}


def _no_verdict(reason: str) -> Dict[str, Any]:
    return _row(NO_VERDICT, reason)


def _usable(name: str, arm: Arm) -> Mapping[str, Any]:
    """The arm's run JSON, or ``_MissingError`` naming why the arm cannot be read."""
    if arm is None:
        raise _MissingError(f"the {name} arm was not run")
    outcome = arm["arm"]["outcome"]
    if outcome != "ok":
        bad = [
            f"{check}: {result.get('reason', result['outcome'])}"
            for check, result in arm["arm"].get("checks", {}).items()
            if result["outcome"] != "ok"
        ]
        detail = f" ({'; '.join(bad)})" if bad else ""
        raise _MissingError(f"the {name} arm is {outcome}{detail}")
    return arm["run"]


def _number(name: str, run: Mapping[str, Any], *path: str) -> float:
    node: Any = run
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            raise _MissingError(f"the {name} arm has no {'.'.join(path)}")
        node = node[key]
    if not isinstance(node, (int, float)):
        raise _MissingError(f"the {name} arm has no {'.'.join(path)}")
    return float(node)


def _same_tokens(
    runs: Sequence[Tuple[str, Mapping[str, Any]]], required: Optional[int] = None
) -> None:
    seqs = {name: run.get("seq") for name, run in runs}
    if len(set(seqs.values())) != 1:
        raise _MissingError(f"the arms ran at different token counts: {seqs}")
    seq = next(iter(seqs.values()))
    if required is not None and seq != required:
        raise _MissingError(f"this row needs {required} tokens, the arms ran at {seq}")


_FORWARD = ("profile", "summary", "forward")


def _timer_is_free(bare: Arm, timer: Arm) -> Dict[str, Any]:
    try:
        bare_run, timer_run = _usable("bare", bare), _usable("timer", timer)
        _same_tokens([("bare", bare_run), ("timer", timer_run)])
        bare_s = _number("bare", bare_run, "median_step_s")
        timer_s = _number("timer", timer_run, "median_step_s")
    except _MissingError as missing:
        return _no_verdict(str(missing))
    if bare_s <= 0:
        return _no_verdict("the bare arm's median step is not positive")
    excess = timer_s / bare_s - 1.0
    ok = excess <= TIMER_EXCESS_LIMIT + _EPS
    return _row(
        PASS if ok else FAIL,
        f"the timer arm's step is {excess:+.1%} against the bare arm's "
        f"(limit {TIMER_EXCESS_LIMIT:+.0%})",
        excess=excess,
        bare_step_s=bare_s,
        timer_step_s=timer_s,
    )


def _profile_is_readable(timer: Arm, profile: Arm) -> Dict[str, Any]:
    try:
        timer_run, profile_run = _usable("timer", timer), _usable("profile", profile)
        _same_tokens([("timer", timer_run), ("profile", profile_run)])
        timer_ms = _number("timer", timer_run, "timer", "forward_visit_ms", "median")
        profile_ms = _number("profile", profile_run, *_FORWARD, "host_ms")
    except _MissingError as missing:
        return _no_verdict(str(missing))
    if timer_ms <= 0:
        return _no_verdict("the timer arm's host time per visit is not positive")
    inflation = profile_ms / timer_ms
    ok = inflation <= INFLATION_LIMIT + _EPS
    return _row(
        PASS if ok else FAIL,
        f"the profile's host time per visit is {inflation:.2f} times the timer's "
        f"(limit {INFLATION_LIMIT})",
        inflation=inflation,
        timer_host_ms=timer_ms,
        profile_host_ms=profile_ms,
    )


def _control(profile: Arm, control: Arm) -> Dict[str, Any]:
    try:
        profile_run, control_run = _usable("profile", profile), _usable("control", control)
        _same_tokens([("profile", profile_run), ("control", control_run)], CONTROL_TOKENS)
        base = {
            key: _number("profile", profile_run, *_FORWARD, key)
            for key in ("host_ms", "weight_prep_ms", "launch_ms", "launches_total")
        }
        with_injection = {
            key: _number("control", control_run, *_FORWARD, key)
            for key in ("weight_prep_ms", "launch_ms", "launches_total")
        }
        timer_ms = _number("control", control_run, "timer", "forward_visit_ms", "median")
    except _MissingError as missing:
        return _no_verdict(str(missing))
    low, high = CONTROL_SPIN_MS * (1 - CONTROL_TOLERANCE), CONTROL_SPIN_MS * (1 + CONTROL_TOLERANCE)
    prep_delta = with_injection["weight_prep_ms"] - base["weight_prep_ms"]
    launch_delta = with_injection["launches_total"] - base["launches_total"]
    launch_cost = with_injection["launch_ms"] - base["launch_ms"]
    timer_delta = timer_ms - base["host_ms"]
    failures: List[str] = []
    if not low - _EPS <= prep_delta <= high + _EPS:
        failures.append(
            f"weight preparation grew by {prep_delta * 1000:.0f} us, "
            f"expected {low * 1000:.0f} to {high * 1000:.0f} us"
        )
    if abs(launch_delta - CONTROL_LAUNCHES) > _EPS:
        failures.append(
            f"launches per visit grew by {launch_delta:g}, expected exactly {CONTROL_LAUNCHES}"
        )
    if not low + launch_cost - _EPS <= timer_delta <= high + launch_cost + _EPS:
        failures.append(
            f"the timer's host time per visit grew by {timer_delta * 1000:.0f} us, expected "
            f"{(low + launch_cost) * 1000:.0f} to {(high + launch_cost) * 1000:.0f} us "
            f"(200 us +/- 20% plus {launch_cost * 1000:.0f} us for the launches)"
        )
    evidence = {
        "weight_prep_delta_ms": prep_delta,
        "launch_delta": launch_delta,
        "launch_cost_ms": launch_cost,
        "timer_delta_ms": timer_delta,
    }
    if failures:
        return _row(FAIL, "; ".join(failures), **evidence)
    return _row(PASS, "the control found both injections", **evidence)


def _cpu_bound(timer: Arm, profile: Arm) -> Dict[str, Any]:
    try:
        timer_run, profile_run = _usable("timer", timer), _usable("profile", profile)
        _same_tokens([("timer", timer_run), ("profile", profile_run)], CONTROL_TOKENS)
        host_ms = _number("timer", timer_run, "timer", "forward_visit_ms", "median")
        prep_ms = _number("profile", profile_run, *_FORWARD, "weight_prep_ms")
        kernel_ms = _number("profile", profile_run, *_FORWARD, "kernel_ms")
        launch_ms = _number("profile", profile_run, *_FORWARD, "launch_ms")
        profile_host_ms = _number("profile", profile_run, *_FORWARD, "host_ms")
    except _MissingError as missing:
        return _no_verdict(str(missing))
    if kernel_ms <= 0:
        return _no_verdict("the profile arm found no GPU kernel time per visit")
    capturable = host_ms - prep_ms
    ratio = capturable / kernel_ms
    ok = ratio >= CPU_BOUND_RATIO - _EPS
    share = launch_ms / profile_host_ms if profile_host_ms > 0 else None
    return _row(
        PASS if ok else FAIL,
        f"capturable host time is {ratio:.2f} times the GPU kernel time per visit "
        f"(threshold {CPU_BOUND_RATIO:g})",
        capturable_ms=capturable,
        kernel_ms=kernel_ms,
        ratio=ratio,
        launch_share=share,
    )


def _needing(row: Dict[str, Any], **required: Dict[str, Any]) -> Dict[str, Any]:
    """``row`` unless a row it depends on has no pass, then a no-verdict naming that row."""
    for name, dependency in required.items():
        if dependency["verdict"] != PASS:
            return _no_verdict(
                f"{name} is {dependency['verdict'].replace('_', ' ')}: {dependency['reason']}"
            )
    return row


def evaluate(*, bare: Arm, timer: Arm, profile: Arm, control: Arm) -> Dict[str, Dict[str, Any]]:
    """The validity rows, the control and the CPU-bound threshold for one token count."""
    timer_is_free = _timer_is_free(bare, timer)
    readable = _profile_is_readable(timer, profile)
    control_row = _needing(_control(profile, control), profile_is_readable=readable)
    threshold = _cpu_bound(timer, profile)
    return {
        "timer_is_free": timer_is_free,
        "profile_is_readable": readable,
        "control": control_row,
        "cpu_bound": _needing(threshold, profile_is_readable=readable, control=control_row),
    }
