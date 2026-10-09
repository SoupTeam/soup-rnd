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

import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Collection, Dict, List, NamedTuple, Optional, Tuple, TypedDict, TypeVar

_T = TypeVar("_T")

OK = "ok"
VOID = "void"
UNKNOWN = "unknown"

#: Boot-time minus monotonic time grows by the time spent suspended. Growth past
#: this many seconds during an arm means the machine slept.
SLEEP_OFFSET_LIMIT_S = 1.0

SUSPEND_SUCCESS_PATH = "/sys/power/suspend_stats/success"

POWER_SUPPLY_DIR = "/sys/class/power_supply"

#: Seconds between power samples during an arm.
POWER_INTERVAL_S = 2.0

#: Every check an arm record carries, in record order.
CHECKS = ("suspend", "power")


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


class PowerReading(TypedDict):
    """One read of the power supplies: whether a battery is present, and which
    non-battery supplies are online."""

    battery: bool
    online: Dict[str, bool]


def _supply_attribute(supply: str, name: str) -> Optional[str]:
    """One sysfs attribute of a power supply, or None if the supply lacks it."""
    try:
        with open(os.path.join(supply, name), encoding="ascii") as handle:
            return handle.read().strip()
    except FileNotFoundError:
        return None


def power_supplies() -> PowerReading:
    """A ``PowerReading`` from the kernel's power supplies.

    Follows the kernel's ``power_supply_is_system_supplied()``: supplies with
    scope "Device" (a mouse battery, say) are skipped, and ``online`` covers
    every remaining supply that is not a battery and has an ``online`` value.
    ``battery`` says whether any remaining supply is a battery.
    """
    battery = False
    online: Dict[str, bool] = {}
    for name in sorted(os.listdir(POWER_SUPPLY_DIR)):
        supply = os.path.join(POWER_SUPPLY_DIR, name)
        if _supply_attribute(supply, "scope") == "Device":
            continue
        kind = _supply_attribute(supply, "type")
        if kind == "Battery":
            battery = True
            continue
        state = _supply_attribute(supply, "online")
        if kind is not None and state is not None:
            online[name] = int(state) != 0
    return {"battery": battery, "online": online}


def _read(name: str, sensor: Callable[[], Optional[_T]]) -> Tuple[Optional[_T], Optional[str]]:
    """``(value, problem)``: the reading, or None and why there is none."""
    try:
        value = sensor()
    except Exception as exc:  # noqa: BLE001 - any sensor failure is an unknown
        return None, f"{name} sensor raised {type(exc).__name__}: {exc}"
    if value is None:
        return None, f"{name} sensor returned None"
    return value, None


def _check_result(outcome: str, reason: str, evidence: Dict[str, object]) -> Dict[str, object]:
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


def suspend_check_result(start: SleepReadings, end: SleepReadings) -> Dict[str, object]:
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
    growth = None
    if start.offset is not None and end.offset is not None:
        growth = end.offset - start.offset
        if growth > SLEEP_OFFSET_LIMIT_S:
            slept.append(f"boot-time offset grew {growth:.3f} s (limit {SLEEP_OFFSET_LIMIT_S} s)")
    if start.count is not None and end.count is not None and end.count > start.count:
        slept.append(f"suspend count rose from {start.count} to {end.count}")
    if slept:
        return _check_result(VOID, "; ".join(slept), evidence)
    problems = start.problems + end.problems
    if problems:
        return _check_result(UNKNOWN, "; ".join(problems), evidence)
    return _check_result(
        OK,
        f"boot-time offset grew {growth:.3f} s (limit {SLEEP_OFFSET_LIMIT_S} s); "
        f"suspend count stayed at {end.count}",
        evidence,
    )


class PowerSample(NamedTuple):
    """One power sample. ``problem`` says why ``reading`` is None."""

    at_s: float
    reading: Optional[PowerReading]
    problem: Optional[str]


class PowerSampler:
    """Reads the power sensor at start, every ``interval_s`` seconds on a thread, and at stop."""

    def __init__(self, sensor: Callable[[], Optional[PowerReading]], interval_s: float) -> None:
        self._sensor = sensor
        self._interval_s = interval_s
        self._stopped = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0
        self.samples: List[PowerSample] = []

    def _sample(self) -> None:
        reading, problem = _read("power", self._sensor)
        self.samples.append(PowerSample(time.monotonic() - self._t0, reading, problem))

    def _loop(self) -> None:
        while not self._stopped.wait(self._interval_s):
            self._sample()

    def start(self) -> None:
        self._t0 = time.monotonic()
        self._sample()
        self._thread = threading.Thread(target=self._loop, name="arm-power-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()


def _on_mains(reading: PowerReading) -> bool:
    """A machine with no battery runs on mains; otherwise some non-battery supply must be online."""
    return not reading["battery"] or any(reading["online"].values())


def power_check_result(samples: List[PowerSample]) -> Dict[str, object]:
    """Void if any sample shows the machine off mains.

    Otherwise a sample with no reading makes the check unknown.
    """
    off_mains = [
        {"sample": index, "at_s": round(sample.at_s, 3), "online": sample.reading["online"]}
        for index, sample in enumerate(samples)
        if sample.reading is not None and not _on_mains(sample.reading)
    ]
    readings = [sample.reading for sample in samples if sample.reading is not None]
    problems = [sample.problem for sample in samples if sample.problem]
    evidence: Dict[str, object] = {
        "samples": len(samples),
        "off_mains_samples": off_mains,
        "battery_present": any(reading["battery"] for reading in readings),
        "unknown_samples": len(problems),
        "last_reading": readings[-1] if readings else None,
    }
    if off_mains:
        return _check_result(
            VOID, f"{len(off_mains)} of {len(samples)} power samples off mains", evidence
        )
    if problems:
        return _check_result(
            UNKNOWN,
            f"{len(problems)} of {len(samples)} power samples gave no reading; "
            f"first: {problems[0]}",
            evidence,
        )
    if not evidence["battery_present"]:
        return _check_result(
            OK, f"no battery in any of {len(samples)} power samples, so on mains", evidence
        )
    return _check_result(OK, f"on mains in all {len(samples)} power samples", evidence)


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
        power_supplies: Callable[[], Optional[PowerReading]] = power_supplies,
        power_interval_s: float = POWER_INTERVAL_S,
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
        self._power = PowerSampler(power_supplies, power_interval_s)
        self.record: Optional[Dict[str, object]] = None

    def start(self) -> None:
        self._start = SleepReadings.take(self._sleep_offset, self._suspend_count)
        self._power.start()

    def stop(self) -> Dict[str, object]:
        if self._start is None:
            raise RuntimeError("stop() called before start()")
        self._power.stop()
        end = SleepReadings.take(self._sleep_offset, self._suspend_count)
        checks = {
            "suspend": suspend_check_result(self._start, end),
            "power": power_check_result(self._power.samples),
        }
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
