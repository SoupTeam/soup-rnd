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

Benchmark harness code, not shipped. Standard library only; no torch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ContextManager, Dict, List, Mapping, Optional, Sequence, Tuple

from arm_validity import UNKNOWN, VOID

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
