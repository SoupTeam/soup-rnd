#!/usr/bin/env python3
"""Verdict driver for the F3 audit of published fine-tuning benchmarks.

``audit`` takes arm specs (config, arm, commit, versions, evidence), a runner and
an arm watch factory. It runs each arm in two fresh processes and returns the arm
records, one verdict per arm and the refusal log. The decision rule is the one
committed in ``benchmarks/gate-f3-benchmark-audit.md``.

The runner takes ``(spec, process)``, runs the arm in a fresh process and returns a
report shaped like ``build_train_report``. It raises if the arm does not start. The
watch factory is called as ``watch(arm=, round=, run=, required=())`` and returns a
context manager whose ``record`` holds the arm record. ``required`` is empty because
the validity verdict does not wait on timing checks (ADR 0006), but a void check still
voids the arm.

A control that is not rejected in every path turns each verdict into no verdict, and
no arm runs.

``speed_audit`` is the second half: A-B-B-A blocks of one config on one card, only when both
arms passed the validity verdict. Block count and void replacement follow ``arm_validity``.

Benchmark harness code, not shipped. Standard library only; no torch.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ContextManager, Dict, List, Mapping, Optional, Sequence, Tuple

from arm_validity import UNKNOWN, VOID, Run, block_order, blocks_to_add, valid_blocks
from visit_verdicts import _t_critical, _two_sided_p

PASS = "pass"
FAIL = "fail"
NO_VERDICT = "no verdict"
OUT_OF_SCOPE = "out of scope"

#: Inference steps: not reproduced, our run invalid, published run invalid.
NOT_REPRODUCED = 1
OUR_RUN_INVALID = 2
PUBLISHED_RUN_INVALID = 3

#: The paths the zero-gradient control must be rejected in, and the checks that must refuse.
CONTROL_PATHS = ("native", "unsloth")
CONTROL_CHECKS = ("grad_norm", "parameters_changed")

REFUSAL_FIELDS = ("config", "arm", "process", "check", "outcome", "reason", "commit", "report")

#: A report as ``build_train_report`` returns it, with an optional ``report_path``.
Report = Mapping[str, Any]
Runner = Callable[["ArmSpec", int], Report]
WatchFactory = Callable[..., ContextManager[Any]]


@dataclass(frozen=True)
class Evidence:
    """What makes step 3 possible: the author's log with the same failure, or an exact
    config and version with the cause confirmed in code."""

    author_log: bool = False
    exact_config_cause_confirmed: bool = False

    @property
    def supports_step_3(self) -> bool:
        return self.author_log or self.exact_config_cause_confirmed


@dataclass(frozen=True)
class ArmSpec:
    """One published arm: a framework at one version on one public config."""

    config: str
    arm: str
    commit: str
    published_number: Optional[float] = None
    versions: Mapping[str, str] = field(default_factory=dict)
    #: False when the foreign loop builds no ``Trainer`` (custom loop): out of scope.
    uses_trainer: bool = True
    scope_reason: str = ""
    #: False when the publication names no recoverable config (the 46,000 tok/s run).
    config_found: bool = True
    evidence: Evidence = field(default_factory=Evidence)

    def __post_init__(self) -> None:
        if not self.config_found and self.evidence.supports_step_3:
            raise ValueError(
                f"{self.config}/{self.arm}: step 3 is not allowed without a found config"
            )


@dataclass(frozen=True)
class Verdict:
    config: str
    arm: str
    verdict: str
    reason: str
    step: Optional[int]


@dataclass
class Audit:
    records: List[Dict[str, Any]] = field(default_factory=list)
    verdicts: List[Verdict] = field(default_factory=list)
    refusals: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ControlRun:
    """One run of the zero-gradient control in ``path`` (native or unsloth)."""

    path: str
    report: Report
    weight_decay: float


def _failed_checks(report: Report) -> List[str]:
    return [str(item["check"]) for item in report.get("failures", [])]


def control_problems(runs: Sequence[ControlRun]) -> List[str]:
    """Why the control does not count as rejected in every path; empty if it does."""
    problems: List[str] = []
    by_path = {run.path: run for run in runs}
    for path in CONTROL_PATHS:
        run = by_path.get(path)
        if run is None:
            problems.append(f"{path} path has no control run")
            continue
        if run.weight_decay != 0:
            problems.append(f"{path} control has weight_decay {run.weight_decay}, not 0")
        if run.report.get("valid"):
            problems.append(f"{path} control reported valid: true")
            continue
        missing = [name for name in CONTROL_CHECKS if name not in _failed_checks(run.report)]
        if missing:
            problems.append(f"{path} control was not refused by {', '.join(missing)}")
    return problems


def _signature(report: Report) -> Tuple[Any, ...]:
    """What two processes must agree on: validity, refusing checks and the check values."""
    checks = report.get("checks", {})
    return (bool(report.get("valid")), tuple(sorted(_failed_checks(report))), repr(checks))


def _refusal(spec: ArmSpec, process: int, report: Report) -> Dict[str, Any]:
    failures = report.get("failures", [])
    return {
        "config": spec.config,
        "arm": spec.arm,
        "process": process,
        "check": ", ".join(str(item["check"]) for item in failures),
        "outcome": "refused",
        "reason": "; ".join(str(item["message"]) for item in failures),
        "commit": spec.commit,
        "report": report.get("report_path"),
    }


def _not_started(spec: ArmSpec, process: int, error: Exception) -> Dict[str, Any]:
    return {
        "config": spec.config,
        "arm": spec.arm,
        "process": process,
        "check": "start",
        "outcome": "did not start",
        "reason": str(error),
        "commit": spec.commit,
        "report": None,
    }


def _audit_arm(spec: ArmSpec, runner: Runner, watch: WatchFactory, result: Audit) -> Verdict:
    def no_verdict(reason: str) -> Verdict:
        return Verdict(spec.config, spec.arm, NO_VERDICT, reason, NOT_REPRODUCED)

    reports: List[Report] = []
    for process in (1, 2):
        try:
            with watch(arm=spec.arm, round=process, run=process, required=()) as watched:
                report = runner(spec, process)
        except Exception as error:  # the arm did not start; the reason goes in the log
            result.refusals.append(_not_started(spec, process, error))
            return no_verdict(f"arm did not start in process {process}: {error}")
        record = watched.record
        result.records.append(
            {"config": spec.config, "arm": spec.arm, "process": process,
             "watch": record, "report": report}
        )  # fmt: skip
        if not report.get("valid"):
            result.refusals.append(_refusal(spec, process, report))
        reports.append(report)
        outcome = record["outcome"] if record else UNKNOWN
        if outcome in (VOID, UNKNOWN):
            return no_verdict(f"validity row {outcome} in process {process}")

    if _signature(reports[0]) != _signature(reports[1]):
        return no_verdict("the two processes' reports differ")
    if all(report.get("valid") for report in reports):
        return Verdict(spec.config, spec.arm, PASS, "", NOT_REPRODUCED)
    failed = _failed_checks(reports[0])
    if "step_count" in failed:
        return no_verdict("arm did not reach step_count")
    step = PUBLISHED_RUN_INVALID if spec.evidence.supports_step_3 else OUR_RUN_INVALID
    return Verdict(spec.config, spec.arm, FAIL, "refused: " + ", ".join(failed), step)


def audit(
    specs: Sequence[ArmSpec],
    runner: Runner,
    watch: WatchFactory,
    control: Sequence[ControlRun],
) -> Audit:
    """Run every spec twice in fresh processes and return records, verdicts and refusals."""
    result = Audit()
    problems = control_problems(control)
    for spec in specs:
        if not spec.config_found:
            verdict = Verdict(
                spec.config, spec.arm, NO_VERDICT, "config not found, not audited", NOT_REPRODUCED
            )
        elif not spec.uses_trainer:
            verdict = Verdict(spec.config, spec.arm, OUT_OF_SCOPE, spec.scope_reason, None)
        elif problems:
            verdict = Verdict(
                spec.config, spec.arm, NO_VERDICT,
                "control not rejected: " + "; ".join(problems), NOT_REPRODUCED,
            )  # fmt: skip
        else:
            verdict = _audit_arm(spec, runner, watch, result)
        result.verdicts.append(verdict)
    return result


def write_refusals(refusals: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Write the refusal log: one JSON object per line, with the fields of ``REFUSAL_FIELDS``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({name: row[name] for name in REFUSAL_FIELDS}) for row in refusals]
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


# --- speed verdict ---------------------------------------------------------------

#: ``ArmWatch`` checks the speed verdict requires. ``foreign_reads`` is left out (ADR 0006).
SPEED_REQUIRED = ("suspend", "power")
#: A void block is replaced at most this many times per config.
MAX_VOID_REPLACEMENTS = 2
#: Two-sided significance level of the t-test on the per-block difference.
SPEED_ALPHA = 0.05
MEASURED = "measured"
FOREIGN_READS_NOTE = "neighbour disk reads were not checked"
DETECTION_LIMIT = "effect below the detection limit, about 10% of the mean of A"

#: Runs one speed run of the arm labelled "A" or "B" and returns a ``build_train_report`` report.
SpeedRunner = Callable[[str, int, int], Report]


@dataclass
class SpeedAudit:
    """The speed row of one config: ``measured`` or ``no verdict``, with every block run."""

    config: str
    verdict: str
    reason: str = ""
    blocks: List[List[Dict[str, Any]]] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    gpu_clock: List[Dict[str, Any]] = field(default_factory=list)


def _as_runs(blocks: Sequence[Sequence[Mapping[str, Any]]]) -> List[List[Run]]:
    return [[Run(run["value"], run["outcome"]) for run in block] for block in blocks]


def _token_notes(blocks: Sequence[Sequence[Mapping[str, Any]]]) -> List[str]:
    notes = []
    for label in ("A", "B"):
        tokens = [run["tokens"] for block in blocks for run in block if run["arm"] == label]
        useful = sum(item.get("useful", 0) for item in tokens)
        total = sum(item.get("total", 0) for item in tokens)
        if total and useful != total:
            notes.append(
                f"arm {label}: useful tokens are {useful / total:.0%} of total tokens; "
                f"tokens per second use the total count"
            )
    return notes


def _speed_statistics(blocks: Sequence[Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    """Two-sided t-test on the per-block difference: mean of the A runs minus mean of the B runs."""
    runs = valid_blocks(_as_runs(blocks))
    diffs = [
        statistics.fmean([block[0].value, block[3].value])
        - statistics.fmean([block[1].value, block[2].value])
        for block in runs
    ]
    mean_a = statistics.fmean(v for block in runs for v in (block[0].value, block[3].value))
    mean_b = statistics.fmean(v for block in runs for v in (block[1].value, block[2].value))
    n = len(diffs)
    diff = statistics.fmean(diffs)
    se = statistics.stdev(diffs) / math.sqrt(n)
    if se == 0:
        p_value, half_width = (0.0 if diff != 0 else 1.0), 0.0
    else:
        p_value, half_width = _two_sided_p(diff / se, n - 1), _t_critical(n - 1) * se
    significant = p_value < SPEED_ALPHA
    return {
        "blocks": n,
        "mean_a_s": mean_a,
        "mean_b_s": mean_b,
        "diff_s": diff,
        "diff_fraction": diff / mean_a,
        "ci_low_s": diff - half_width,
        "ci_high_s": diff + half_width,
        "p_value": p_value,
        "significant": significant,
        # Values are seconds per step, so the smaller mean is the faster arm.
        "faster": ("A" if diff < 0 else "B") if significant else None,
        "detection_limit": DETECTION_LIMIT,
    }


def speed_audit(
    config: str,
    arms: Tuple[str, str],
    validity: Sequence[Verdict],
    runner: SpeedRunner,
    watch: WatchFactory,
) -> SpeedAudit:
    """Compare two arms of ``config`` in A-B-B-A blocks. ``arms`` names the validity verdicts
    of A and B; the runner is called with the label "A" or "B".

    Runs nothing unless both arms passed. The pilot, the top-up and the replacement of void
    blocks come from ``arm_validity.blocks_to_add``; more than ``MAX_VOID_REPLACEMENTS`` void
    blocks, an unknown outcome, or a run that does not start gives no verdict.
    """
    result = SpeedAudit(config, NO_VERDICT, notes=[FOREIGN_READS_NOTE])
    by_arm = {v.arm: v for v in validity if v.config == config}
    not_passed = [
        f"arm {name} {by_arm[name].verdict if name in by_arm else 'has no validity verdict'}"
        for name in arms
        if name not in by_arm or by_arm[name].verdict != PASS
    ]
    if not_passed:
        result.reason = "speed blocks not run: " + "; ".join(not_passed)
        return result

    while blocks_to_add(_as_runs(result.blocks)) > 0:
        voids = len(result.blocks) - len(valid_blocks(_as_runs(result.blocks)))
        if voids > MAX_VOID_REPLACEMENTS:
            result.reason = (
                f"{voids} void blocks, more than the {MAX_VOID_REPLACEMENTS} replacements allowed"
            )
            return result
        index = len(result.blocks)
        block: List[Dict[str, Any]] = []
        result.blocks.append(block)
        for position, label in enumerate(block_order(1)[0]):
            round, run = 2 * index + 1 + position // 2, 4 * index + position + 1
            try:
                with watch(arm=label, round=round, run=run, required=SPEED_REQUIRED) as watched:
                    report = runner(label, round, run)
            except Exception as error:
                result.reason = f"run {run} (arm {label}) did not start: {error}"
                return result
            record = watched.record or {}
            block.append(
                {"arm": label, "round": round, "run": run, "record": record,
                 "value": float(report["timing"]["median_seconds"]),
                 "outcome": str(record.get("outcome", UNKNOWN)),
                 "tokens": report.get("tokens", {}), "report": report}
            )  # fmt: skip
            result.gpu_clock.append(
                {"arm": label, "round": round, "run": run, "gpu_clock": record.get("gpu_clock")}
            )
        unknown = [run["run"] for run in block if run["outcome"] == UNKNOWN]
        if unknown:
            result.reason = (
                f"run {unknown[0]} has an unknown outcome, so block {index + 1} cannot be trusted"
            )
            return result

    result.evidence = _speed_statistics(result.blocks)
    result.notes += _token_notes(result.blocks)
    result.verdict = MEASURED
    return result
