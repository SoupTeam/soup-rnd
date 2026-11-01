#!/usr/bin/env python3
"""A5 ticket 10: the pure parts of the step decomposition and the host sensors.

``Buckets`` collects seconds per named part of one optimizer step. ``decompose`` turns them
and the step's wall time into the parts the ticket asks for (transfer of activations, wait for
a layer, optimizer, compute). The sensor readers parse what a container can still read
(``/proc/stat``, ``/proc/loadavg``, ``nvidia-smi --query-gpu``) and ``judge_run`` marks a run
whose host was busy or whose GPU was slowed. Benchmark harness code, not shipped.
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence

#: Parts that are not compute. Everything left of the step's wall time is ``compute``.
PARTS = ("activation_put", "activation_get", "layer_wait", "optimizer")

#: CPU wall seconds spent in a call, reported next to the parts and not taken off the step:
#: the work is part of ``compute`` (the CPU issues it while the stream runs).
CPU_PARTS = ("cpu_rebind", "cpu_load_issue")

#: ``judge_run`` limits: a run is flagged when any of them is crossed.
STEAL_LIMIT = 0.05
OUTSIDE_CPU_LIMIT = 0.25
CLOCK_FLOOR = 0.90

#: NVML throttle bits that flag a run: hardware slowdown (0x8), software thermal (0x20),
#: hardware thermal (0x40), power brake (0x80). The software power cap (0x4) is a property of
#: the host (a rented card runs at its cap all the time), so it is reported as
#: ``power_cap_share`` and does not flag a run.
THROTTLE_MASK = 0x8 | 0x20 | 0x40 | 0x80
POWER_CAP = 0x4
#: NVML bit for a card with nothing to do (the model loading, not the run).
IDLE = 0x1


class Buckets:
    """Seconds per name, summed over a step. ``timed`` adds the time of a ``with`` block."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter,
                 sync: Optional[Callable[[], None]] = None):
        self.clock, self.sync = clock, sync
        self.seconds: Dict[str, float] = {}
        self.calls: Dict[str, int] = {}

    @contextmanager
    def timed(self, name: str) -> Iterator[None]:
        if self.sync:
            self.sync()
        start = self.clock()
        try:
            yield
        finally:
            if self.sync:
                self.sync()
            self.add(name, self.clock() - start)

    def add(self, name: str, seconds: float) -> None:
        """Add seconds measured elsewhere (CUDA events) to a bucket."""
        self.seconds[name] = self.seconds.get(name, 0.0) + seconds
        self.calls[name] = self.calls.get(name, 0) + 1

    def take(self) -> Dict[str, float]:
        """The sums so far; the buckets start again from zero."""
        out = {name: self.seconds.get(name, 0.0) for name in (*PARTS, *CPU_PARTS)}
        out.update({k: v for k, v in self.seconds.items() if k not in out})
        self.seconds, self.calls = {}, {}
        return out


def decompose(step_s: float, parts: Mapping[str, float]) -> Dict[str, float]:
    """The step as seconds per part. ``compute`` is what the named parts leave of the step."""
    named = {name: float(parts.get(name, 0.0)) for name in PARTS}
    compute = step_s - sum(named.values())
    cpu = {name: float(parts.get(name, 0.0)) for name in CPU_PARTS}
    return {"step_s": step_s, **named, "compute": compute, **cpu,
            "overhead_negative": compute < 0}


def mean_parts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    """Mean seconds per step of each part over ``rows`` (the warm steps)."""
    keys = ("step_s", *PARTS, "compute", *CPU_PARTS)
    return {k: statistics.fmean(r[k] for r in rows) for k in keys} if rows else {}


# ----------------------------------------------------------------------
# sensors
# ----------------------------------------------------------------------
def parse_proc_stat(text: str) -> Optional[Dict[str, int]]:
    """The aggregate ``cpu`` line of ``/proc/stat`` as jiffies per field, or None."""
    for line in text.splitlines():
        if line.startswith("cpu "):
            names = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")
            values = [int(v) for v in line.split()[1:9]]
            return dict(zip(names, values))
    return None


def cpu_fractions(before: Mapping[str, int], after: Mapping[str, int]) -> Dict[str, float]:
    """Share of the host's CPU time per kind between two ``/proc/stat`` readings."""
    delta = {k: after[k] - before[k] for k in after}
    total = sum(delta.values())
    if total <= 0:
        return {"busy": 0.0, "steal": 0.0, "iowait": 0.0}
    idle = delta["idle"] + delta["iowait"]
    return {"busy": 1 - idle / total, "steal": delta["steal"] / total,
            "iowait": delta["iowait"] / total}


def parse_gpu_query(line: str) -> Optional[Dict[str, Any]]:
    """One ``nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw,
    clocks_throttle_reasons.active --format=csv,noheader,nounits`` line."""
    cells = [c.strip() for c in line.split(",")]
    if len(cells) != 5:
        return None

    def number(cell: str) -> Optional[float]:
        try:
            return float(cell)
        except ValueError:
            return None

    reasons = None
    try:
        reasons = int(cells[4], 16)
    except ValueError:
        pass
    return {"sm_mhz": number(cells[0]), "max_sm_mhz": number(cells[1]),
            "temp_c": number(cells[2]), "power_w": number(cells[3]), "throttle": reasons}


def busy_gpu(samples: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """GPU readings of the samples where the card was not idle (idle bit clear)."""
    gpus = [s["gpu"] for s in samples if s.get("gpu")]
    return [g for g in gpus if not (g.get("throttle") or 0) & IDLE]


def judge_run(samples: Sequence[Mapping[str, Any]], own_cpu_share: Optional[float] = None
              ) -> Dict[str, Any]:
    """Flags for one run from its sensor samples.

    A sample holds ``cpu`` (from ``cpu_fractions``) and ``gpu`` (``parse_gpu_query`` or None).
    ``own_cpu_share`` is the process tree's share of the host CPU, taken off the busy share
    so that the neighbour's load is what is left. A flag is raised on steal, on the outside
    CPU load, on a median SM clock below ``CLOCK_FLOOR`` x the card's maximum, and on a
    throttle reason active in any sample. No samples means no flags and ``sensed`` False.
    """
    out: Dict[str, Any] = {"samples": len(samples), "sensed": False, "flags": [],
                           "sensors": {}}
    cpu = [s["cpu"] for s in samples if s.get("cpu")]
    # A sample with the idle bit (0x1) is the model loading, not the run: its clock is low
    # because the card has nothing to do.
    gpu = busy_gpu(samples)
    if cpu:
        out["sensed"] = True
        steal = statistics.fmean(c["steal"] for c in cpu)
        busy = statistics.fmean(c["busy"] for c in cpu)
        outside = max(0.0, busy - (own_cpu_share or 0.0))
        out["sensors"].update(cpu_steal=steal, cpu_busy=busy, cpu_outside=outside)
        if steal > STEAL_LIMIT:
            out["flags"].append(f"cpu steal {steal:.1%}")
        if own_cpu_share is not None and outside > OUTSIDE_CPU_LIMIT:
            out["flags"].append(f"outside cpu load {outside:.1%}")
    clocks = [
        (g["sm_mhz"], g["max_sm_mhz"]) for g in gpu if g.get("sm_mhz") and g.get("max_sm_mhz")
    ]
    if clocks:
        out["sensed"] = True
        ratio = statistics.median(c / m for c, m in clocks)
        out["sensors"]["sm_clock_ratio_median"] = ratio
        out["sensors"]["sm_mhz_min"] = min(c for c, _ in clocks)
        if ratio < CLOCK_FLOOR:
            out["flags"].append(f"sm clock {ratio:.0%} of max")
    reasons = [g["throttle"] for g in gpu if g.get("throttle") is not None]
    if reasons:
        hit = 0
        for value in reasons:
            hit |= value & THROTTLE_MASK
        out["sensors"]["throttle_bits"] = hit
        out["sensors"]["power_cap_share"] = sum(1 for v in reasons if v & POWER_CAP) / len(reasons)
        if hit:
            out["flags"].append(f"gpu throttle bits {hit:#x}")
    out["disturbed"] = bool(out["flags"])
    return out


def per_layer_seconds(step: Mapping[str, Any]) -> Optional[float]:
    """Wait for a layer per layer load of the step."""
    loads = step.get("layer_loads")
    return step["layer_wait"] / loads if loads else None


def spread(values: Sequence[float]) -> Dict[str, float]:
    """Min, max and max over min of a list (the GA speed spread across runs)."""
    return {"min": min(values), "max": max(values), "ratio": max(values) / min(values)}
