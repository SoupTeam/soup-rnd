#!/usr/bin/env python3
"""Validity checks for one benchmark arm (F1).

Wrap an arm in ``ArmWatch`` and read ``watch.record`` afterwards: the arm's
identity, one entry per check under ``checks`` and the arm's overall
``outcome``. Each check ends in ok, void (the sensor answered and the arm is
contaminated) or unknown (the sensor gave no answer). A sensor that raises or
returns None makes its check unknown, never ok (ADR 0001).

Sensors are callables passed in by the caller, with real Linux defaults that
need no root and no third-party packages.

Benchmark harness code, not shipped. Standard library only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Collection, Dict, List, Optional, Tuple, TypeVar

_T = TypeVar("_T")

OK = "ok"
VOID = "void"
UNKNOWN = "unknown"

#: Boot-time minus monotonic time grows by the time spent suspended. Growth past
#: this many seconds during an arm means the machine slept.
SLEEP_OFFSET_LIMIT_S = 1.0

SUSPEND_SUCCESS_PATH = "/sys/power/suspend_stats/success"

#: Every check an arm record carries, in record order.
CHECKS = ("suspend",)


def sleep_offset() -> Optional[float]:
    """Boot-time minus monotonic time in seconds; None where the OS has no boot-time clock."""
    boottime = getattr(time, "CLOCK_BOOTTIME", None)
    if boottime is None:
        return None
    return time.clock_gettime(boottime) - time.clock_gettime(time.CLOCK_MONOTONIC)


def suspend_count() -> Optional[int]:
    """The kernel's count of successful suspends since boot."""
    with open(SUSPEND_SUCCESS_PATH, encoding="ascii") as handle:
        return int(handle.read().strip())


def _read(name: str, sensor: Callable[[], Optional[_T]]) -> Tuple[Optional[_T], Optional[str]]:
    """``(value, problem)``: the reading, or None and why there is none."""
    try:
        value = sensor()
    except Exception as exc:  # noqa: BLE001 - any sensor failure is an unknown
        return None, f"{name} sensor raised {type(exc).__name__}: {exc}"
    if value is None:
        return None, f"{name} sensor returned None"
    return value, None


def _check(outcome: str, reason: str, evidence: Dict[str, object]) -> Dict[str, object]:
    return {"outcome": outcome, "reason": reason, "evidence": evidence}


@dataclass
class SleepReadings:
    """One reading of both sleep sensors, and why either gave nothing."""

    offset: Optional[float]
    count: Optional[int]
    problems: List[str]

    @classmethod
    def take(
        cls,
        offset_sensor: Callable[[], Optional[float]],
        count_sensor: Callable[[], Optional[int]],
    ) -> "SleepReadings":
        offset, offset_problem = _read("boot-time offset", offset_sensor)
        count, count_problem = _read("suspend count", count_sensor)
        return cls(offset, count, [p for p in (offset_problem, count_problem) if p])


def suspend_check(start: SleepReadings, end: SleepReadings) -> Dict[str, object]:
    """Void if boot-time offset grew past the limit or the suspend count rose.

    Either sensor seeing sleep is enough to void. Otherwise a silent sensor
    makes the check unknown. There is no wall-clock gap rule on Linux.
    """
    evidence: Dict[str, object] = {
        "sleep_offset_start_s": start.offset,
        "sleep_offset_end_s": end.offset,
        "suspend_count_start": start.count,
        "suspend_count_end": end.count,
    }
    slept = []
    if start.offset is not None and end.offset is not None:
        growth = end.offset - start.offset
        if growth > SLEEP_OFFSET_LIMIT_S:
            slept.append(f"boot-time offset grew {growth:.3f} s (limit {SLEEP_OFFSET_LIMIT_S} s)")
    if start.count is not None and end.count is not None and end.count > start.count:
        slept.append(f"suspend count rose from {start.count} to {end.count}")
    if slept:
        return _check(VOID, "; ".join(slept), evidence)
    problems = start.problems + end.problems
    if problems:
        return _check(UNKNOWN, "; ".join(problems), evidence)
    return _check(OK, "boot-time offset and suspend count unchanged", evidence)


def arm_outcome(checks: Dict[str, Dict[str, object]], required: Collection[str]) -> str:
    """Void if any check is void, else unknown if any required check is unknown, else ok."""
    if any(check["outcome"] == VOID for check in checks.values()):
        return VOID
    if any(checks[name]["outcome"] == UNKNOWN for name in required):
        return UNKNOWN
    return OK


class ArmWatch:
    """Watches one arm. Use as a context manager, then read ``record``.

    ``required`` names the checks whose unknown makes the arm unknown; it
    defaults to every check. A void check voids the arm whether required or not.
    """

    def __init__(
        self,
        *,
        arm: str,
        round: int,
        run: int,
        label: Optional[str] = None,
        required: Optional[Collection[str]] = None,
        sleep_offset: Callable[[], Optional[float]] = sleep_offset,
        suspend_count: Callable[[], Optional[int]] = suspend_count,
    ):
        self.arm = arm
        self.round = round
        self.run = run
        self.label = label
        self.required = list(CHECKS if required is None else required)
        unknown_names = sorted(set(self.required) - set(CHECKS))
        if unknown_names:
            raise ValueError(f"no such checks: {', '.join(unknown_names)}")
        self._sleep_offset = sleep_offset
        self._suspend_count = suspend_count
        self._start: Optional[SleepReadings] = None
        self.record: Optional[Dict[str, object]] = None

    def start(self) -> None:
        self._start = SleepReadings.take(self._sleep_offset, self._suspend_count)

    def stop(self) -> Dict[str, object]:
        if self._start is None:
            raise RuntimeError("stop() called before start()")
        end = SleepReadings.take(self._sleep_offset, self._suspend_count)
        checks = {"suspend": suspend_check(self._start, end)}
        self.record = {
            "arm": self.arm,
            "round": self.round,
            "run": self.run,
            "label": self.label,
            "required": self.required,
            "outcome": arm_outcome(checks, self.required),
            "checks": checks,
        }
        return self.record

    def __enter__(self) -> "ArmWatch":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
