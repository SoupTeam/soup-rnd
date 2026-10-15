#!/usr/bin/env python3
"""Two extra arm checks for G7, and an arm watch that adds them to F1's three.

``gpu_power_limit`` reads the GPU's "Current Power Limit" from the full power
query before and after the arm. ``swap`` reads the kernel's swap-in and swap-out
page counters before and after it. Each ends in ok, void or unknown with the same
meaning as an F1 check (ADR 0001), and returns the same shape: outcome, reason
and evidence.

``G7ArmWatch`` runs F1's ``ArmWatch`` and these two checks around one arm, then
combines all five through F1's ``arm_outcome``. It changes no F1 file: F1's watch
knows only its own three checks, so this module merges the two new ones into the
finished record and recomputes the outcome.

Benchmark harness code, not shipped. Standard library only.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from arm_validity import (
    CHECKS,
    OK,
    UNKNOWN,
    VOID,
    ArmWatch,
    SensorUnavailableError,
    _check_result,
    _nvidia_smi,
    _read,
    arm_outcome,
)

#: The power limits a G7 arm may run under, in watts: the card's default and its maximum.
#: Under load the limit rises from 80 W to about 90 W (Dynamic Boost), which is healthy. A
#: card stuck at 10 or 15 W gives tight, clean-looking numbers that describe the cap.
POWER_LIMIT_MIN_W = 80.0
POWER_LIMIT_MAX_W = 95.0
POWER_LIMIT_RANGE = f"{POWER_LIMIT_MIN_W:.2f} to {POWER_LIMIT_MAX_W:.2f} W"

#: The field of the full power query the check reads, and the section that holds it.
POWER_LIMIT_FIELD = "Current Power Limit"
POWER_SECTION = "GPU Power Readings"

VMSTAT_PATH = "/proc/vmstat"

#: The /proc/vmstat counters of pages swapped in and out since boot.
SWAP_COUNTERS = ("pswpin", "pswpout")

#: The two checks this module adds, in record order.
EXTRA_CHECKS = ("gpu_power_limit", "swap")

#: Every check a G7 arm requires: F1's three, then the two above.
G7_REQUIRED = CHECKS + EXTRA_CHECKS

PowerQuery = Callable[[], Optional[str]]
SwapCounters = Callable[[], Optional[Dict[str, Optional[int]]]]


def gpu_power_query() -> str:
    """The full ``nvidia-smi -q -d POWER`` text for the workload's GPU.

    The per-field query (``--query-gpu=power.limit``) returns N/A on this card, so the
    full query is the source.
    """
    return "\n".join(_nvidia_smi("-q", "-d", "POWER"))


def parse_power_limit(query: str) -> float:
    """The "Current Power Limit" in watts under "GPU Power Readings".

    Raises SensorUnavailableError if the field is missing or reads N/A. The module
    section repeats the field, so a section-blind search could read the wrong one.
    """
    section = ""
    for line in query.splitlines():
        key, separator, value = line.partition(" : ")
        if not separator:
            section = line.strip()
            continue
        if section == POWER_SECTION and key.strip() == POWER_LIMIT_FIELD:
            number = value.split()[0] if value.split() else ""
            try:
                return float(number)
            except ValueError:
                raise SensorUnavailableError(
                    f"{POWER_LIMIT_FIELD} reads {value.strip()!r}, not a number of watts"
                ) from None
    raise SensorUnavailableError(f"no {POWER_LIMIT_FIELD} field under {POWER_SECTION}")


def swap_counters() -> Dict[str, Optional[int]]:
    """Pages swapped in and out since boot, from /proc/vmstat.

    Raises ValueError if either counter is missing.
    """
    found: Dict[str, Optional[int]] = {}
    with open(VMSTAT_PATH, encoding="ascii") as handle:
        for line in handle:
            name, _, value = line.partition(" ")
            if name in SWAP_COUNTERS:
                found[name] = int(value)
    missing = [name for name in SWAP_COUNTERS if name not in found]
    if missing:
        raise ValueError(f"{VMSTAT_PATH} has no {', '.join(missing)} line")
    return found


def gpu_power_limit_check_result(
    start: Tuple[Optional[float], Optional[str]], end: Tuple[Optional[float], Optional[str]]
) -> Dict[str, object]:
    """Void if the limit read outside 80.00 to 95.00 W at either end of the arm.

    Each argument is ``(watts, problem)``. A value the sensor gave decides, even when
    the other end gave nothing. Otherwise a silent end makes the check unknown.
    """
    (before, before_problem), (after, after_problem) = start, end
    evidence: Dict[str, object] = {
        "power_limit_w_start": before,
        "power_limit_w_end": after,
        "expected_w": [POWER_LIMIT_MIN_W, POWER_LIMIT_MAX_W],
    }
    wrong = [
        f"{when} the arm the limit was {watts:.2f} W"
        for when, watts in (("before", before), ("after", after))
        if watts is not None and not POWER_LIMIT_MIN_W <= watts <= POWER_LIMIT_MAX_W
    ]
    if wrong:
        return _check_result(
            VOID, f"{'; '.join(wrong)} (expected {POWER_LIMIT_RANGE})", evidence
        )
    problems = [problem for problem in (before_problem, after_problem) if problem]
    if problems:
        return _check_result(UNKNOWN, "; ".join(problems), evidence)
    return _check_result(
        OK,
        f"limit {before:.2f} W before and {after:.2f} W after the arm (within {POWER_LIMIT_RANGE})",
        evidence,
    )


def _pages(reading: Optional[Dict[str, Optional[int]]], name: str) -> Optional[int]:
    return None if reading is None else reading.get(name)


def swap_check_result(
    start: Tuple[Optional[Dict[str, Optional[int]]], Optional[str]],
    end: Tuple[Optional[Dict[str, Optional[int]]], Optional[str]],
) -> Dict[str, object]:
    """Void if either swap counter rose during the arm, with the page counts as evidence.

    Each argument is ``(counters, problem)``. A counter that rose decides, even when
    the other one gave nothing. Otherwise a counter with no reading at both ends makes
    the check unknown.
    """
    (before, before_problem), (after, after_problem) = start, end
    moved: Dict[str, Optional[int]] = {}
    for name in SWAP_COUNTERS:
        first, last = _pages(before, name), _pages(after, name)
        moved[name] = None if first is None or last is None else last - first
    evidence: Dict[str, object] = {
        "swapped_in_pages": moved["pswpin"],
        "swapped_out_pages": moved["pswpout"],
        "counters_start": before,
        "counters_end": after,
    }
    rose = [
        f"{label} {moved[name]} pages"
        for name, label in (("pswpin", "swapped in"), ("pswpout", "swapped out"))
        if (moved[name] or 0) > 0
    ]
    if rose:
        return _check_result(VOID, f"the machine {' and '.join(rose)} during the arm", evidence)
    problems: List[str] = [problem for problem in (before_problem, after_problem) if problem]
    if not problems:
        problems = [f"{name} gave no reading" for name in SWAP_COUNTERS if moved[name] is None]
    if problems:
        return _check_result(UNKNOWN, "; ".join(problems), evidence)
    return _check_result(OK, "no swap-in or swap-out during the arm", evidence)


class ExtraChecks:
    """Reads the two sensors at the start and the end of an arm and judges them."""

    def __init__(
        self,
        power_query: PowerQuery = gpu_power_query,
        swap_counters: SwapCounters = swap_counters,
    ) -> None:
        self._power_query = power_query
        self._swap_counters = swap_counters
        self._power: List[Tuple[Optional[float], Optional[str]]] = []
        self._swap: List[Tuple[Optional[Dict[str, Optional[int]]], Optional[str]]] = []

    def _take(self) -> None:
        self._power.append(
            _read("power limit", lambda: parse_power_limit(self._power_query() or ""))
        )
        self._swap.append(_read("swap counters", self._swap_counters))

    def start(self) -> None:
        self._take()

    def stop(self) -> None:
        self._take()

    def results(self) -> Dict[str, Dict[str, object]]:
        return {
            "gpu_power_limit": gpu_power_limit_check_result(self._power[0], self._power[-1]),
            "swap": swap_check_result(self._swap[0], self._swap[-1]),
        }


class G7ArmWatch:
    """F1's ``ArmWatch`` plus the two G7 checks. Use as a context manager, then read ``record``.

    ``required`` defaults to all five checks. The keyword arguments reach ``ArmWatch``
    unchanged, so its sensors can be replaced the same way.
    """

    def __init__(
        self,
        *,
        power_query: PowerQuery = gpu_power_query,
        swap_counters: SwapCounters = swap_counters,
        required: Optional[Sequence[str]] = None,
        **watch_kwargs: object,
    ) -> None:
        self.required = list(G7_REQUIRED if required is None else required)
        unknown_names = sorted(set(self.required) - set(G7_REQUIRED))
        if unknown_names:
            raise ValueError(f"no such checks: {', '.join(unknown_names)}")
        self._watch = ArmWatch(
            required=[name for name in self.required if name in CHECKS],
            **watch_kwargs,  # type: ignore[arg-type]
        )
        self._extra = ExtraChecks(power_query, swap_counters)
        self.record: Optional[Dict[str, object]] = None

    def start(self) -> None:
        self._extra.start()
        self._watch.start()

    def stop(self) -> Dict[str, object]:
        record = self._watch.stop()
        self._extra.stop()
        checks = {**record["checks"], **self._extra.results()}  # type: ignore[dict-item]
        self.record = {
            **record,
            "required": self.required,
            "outcome": arm_outcome(checks, self.required),
            "checks": checks,
        }
        return self.record

    def __enter__(self) -> "G7ArmWatch":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
