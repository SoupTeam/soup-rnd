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

- ``evaluate_experiment``: the ``tie_weights=False`` experiment. The bit-exact gate must have
  passed. The test is a two-sided t-test on the per-block difference (mean of the A runs minus
  mean of the B runs) at alpha 0.05, over the blocks without a void run. ``pass`` means B is
  faster, ``fail`` means no gain, and ``no_verdict`` means the gate failed or never ran, a
  run is void or unknown, or the repeats rule asks for more valid blocks.

All medians are over forward visits. A row gets ``no_verdict``, with the cause named, when
an arm it reads is missing, void or unknown, when a number it reads is missing, when the
arms disagree on the token count, or when a row it depends on has no pass. The control and
the threshold also need a readable profile, and the threshold needs a passing control.

Benchmark harness code, not shipped.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import visit_control
from arm_validity import VOID, Run, blocks_to_add, valid_blocks

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


# --- the tie_weights experiment ---------------------------------------------------------

#: Two-sided significance level of the experiment's t-test.
ALPHA = 0.05
#: What the experiment can detect, as written in the verdict when it finds nothing.
DETECTION_LIMIT = "effect below the detection limit, about 10% of a step"


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        for term in (
            m * (b - m) * x / ((a + m2 - 1.0) * (a + m2)),
            -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0)),
        ):
            d = 1.0 + term * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + term / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-14:
            break
    return h


def _incomplete_beta(a: float, b: float, x: float) -> float:
    """The regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _betacf(a, b, x) / a
    return 1.0 - math.exp(log_front) * _betacf(b, a, 1.0 - x) / b


def _two_sided_p(t: float, df: int) -> float:
    """P(|T| >= |t|) for Student's t with ``df`` degrees of freedom."""
    return _incomplete_beta(df / 2.0, 0.5, df / (df + t * t))


def _t_critical(df: int, alpha: float = ALPHA) -> float:
    """The two-sided critical value of Student's t, found by bisection on ``_two_sided_p``."""
    low, high = 0.0, 1000.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if _two_sided_p(mid, df) > alpha:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def _block_runs(blocks: Sequence[Sequence[Mapping[str, Any]]]) -> List[List[Run]]:
    return [[Run(float(run["value"]), str(run["outcome"])) for run in block] for block in blocks]


def evaluate_experiment(
    gate: Optional[Mapping[str, Any]], blocks: Sequence[Sequence[Mapping[str, Any]]]
) -> Dict[str, Any]:
    """The verdict of the ``tie_weights=False`` experiment.

    ``gate`` is the bit-exact gate's result (``passed`` and ``reason``), or ``None`` if it
    never ran. ``blocks`` are every block run in A, B, B, A order, void ones included, each run
    a dict with ``value`` (seconds per step) and ``outcome``.
    """
    if gate is None:
        return _no_verdict("the bit-exact gate was not run")
    if not gate.get("passed"):
        return _no_verdict(f"the bit-exact gate failed: {gate.get('reason', 'no reason given')}")
    runs = _block_runs(blocks)
    valid = valid_blocks(runs)
    voids = sum(1 for block in runs if any(run.outcome == VOID for run in block))
    unknown = [
        f"run {index} of block {number}"
        for number, block in enumerate(blocks, start=1)
        for index, run in enumerate(block, start=1)
        if run["outcome"] not in ("ok", "void")
    ]
    if unknown:
        return _no_verdict(
            f"{', '.join(unknown)} has an unknown outcome, so its block cannot be trusted"
        )
    missing = blocks_to_add(runs) if len(valid) >= 2 else 2 - len(valid)
    if missing:
        return _no_verdict(
            f"{voids} void blocks left out; the repeats rule asks for {missing} more valid "
            f"block{'s' if missing != 1 else ''} before the test can run"
        )
    diffs = [
        statistics.fmean([block[0].value, block[3].value])
        - statistics.fmean([block[1].value, block[2].value])
        for block in valid
    ]
    mean_a = statistics.fmean(
        value for block in valid for value in (block[0].value, block[3].value)
    )
    n = len(diffs)
    gain = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    se = sd / math.sqrt(n)
    if se == 0:
        p_value = 0.0 if gain != 0 else 1.0
        half_width = 0.0
    else:
        p_value = _two_sided_p(gain / se, n - 1)
        half_width = _t_critical(n - 1) * se
    evidence = {
        "blocks": n,
        "gain_s": gain,
        "gain_fraction": gain / mean_a,
        "ci_low_s": gain - half_width,
        "ci_high_s": gain + half_width,
        "p_value": p_value,
        "mean_a_s": mean_a,
    }
    interval = f"95% interval {evidence['ci_low_s']:+.4f} to {evidence['ci_high_s']:+.4f} s"
    if p_value < ALPHA and gain > 0:
        return _row(
            PASS,
            f"B is faster by {gain:.4f} s per step ({gain / mean_a:.1%}), {interval}, "
            f"p = {p_value:.4f}",
            **evidence,
        )
    if p_value < ALPHA:
        return _row(
            FAIL,
            f"B is slower by {-gain:.4f} s per step ({-gain / mean_a:.1%}), {interval}, "
            f"p = {p_value:.4f}",
            **evidence,
        )
    return _row(FAIL, f"{DETECTION_LIMIT} (p = {p_value:.3f}, {interval})", **evidence)
