"""Trace analyser for G7: the parts of a layer visit, from exported PyTorch profiler events.

``analyse_trace`` is a pure function. It takes the ``traceEvents`` list of an exported chrome
trace (times in microseconds) and returns, as plain dicts:

- ``visits``: for each forward and recompute visit, the host time split into four parts that
  sum to it, the GPU kernel time and the launches per API;
- ``layer_backwards``: for each layer backward, the host time split into launch time and the
  rest, the kernel time and the launches per API;
- ``summary``: the medians of the two lists, in ms;
- ``reconciliation``: for the profiled steps, the visits, layer backwards, work outside the
  layers and the final GPU wait against the step time, with the gap. Work outside the layers
  is the host time covered by any CPU op, CUDA runtime call or other annotation on the step's
  threads that no visit, layer backward or final wait already covers. The gap is therefore
  the step time with no traced event at all, so a small gap shows the trace has no hole, not
  that each part was timed by an independent clock;
- ``memcpy_sources``: the chain of PyTorch ops that issued each device-to-device copy in a
  visit, with the median count per visit.

The ranges come from ``visit_ranges.py`` and carry the names in ``RANGES``. The analyser
cuts the four parts of a visit in this order, so no time counts twice and python_dispatcher
is the remainder:

1. launch time: every queueing CUDA API call on the visit's thread (``QUEUEING_APIS``),
   wherever it falls, even inside weight preparation;
2. weight preparation: the ``weight_prep`` ranges, minus launch time;
3. ``functional_call``: its machinery, meaning the ``functional_call`` ranges minus the call
   into the layer, minus launch time and weight preparation;
4. Python and dispatcher: the rest of the visit.

Benchmark harness code, not shipped. Needs no torch.
"""

from __future__ import annotations

import statistics
from bisect import bisect_left
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

Interval = Tuple[float, float]
Intervals = List[Interval]


class _Ranges:
    """The names of the profiler ranges the profile arm records."""

    step = "g7::step"
    forward_visit = "g7::forward_visit"
    recompute_visit = "g7::recompute_visit"
    weight_prep = "g7::weight_prep"
    functional_call = "g7::functional_call"
    layer_call = "g7::layer_call"
    layer_backward = "g7::layer_backward"
    prefix = "g7::"


RANGES = _Ranges()

#: Kernel launch APIs.
KERNEL_LAUNCH_APIS = ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC")

#: Every CUDA API call that queues work: the three kernel launch APIs, asynchronous copies and
#: fills, event records and event waits.
QUEUEING_APIS = frozenset(
    KERNEL_LAUNCH_APIS
    + (
        "cudaMemcpyAsync",
        "cudaMemcpy2DAsync",
        "cuMemcpyAsync",
        "cuMemcpyHtoDAsync",
        "cuMemcpyDtoHAsync",
        "cuMemcpyDtoDAsync",
        "cudaMemsetAsync",
        "cudaEventRecord",
        "cudaEventRecordWithFlags",
        "cuEventRecord",
        "cudaStreamWaitEvent",
        "cuStreamWaitEvent",
    )
)

#: Calls that block the host until the GPU is done.
SYNC_APIS = frozenset(("cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaEventSynchronize"))

RECONCILIATION_TOLERANCE = 0.05

#: Microseconds per millisecond. Event times are microseconds, results are milliseconds.
_US_PER_MS = 1000.0


# --- intervals -------------------------------------------------------------------------


def _merge(intervals: Iterable[Interval]) -> Intervals:
    merged: Intervals = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _clip(intervals: Iterable[Interval], lo: float, hi: float) -> Intervals:
    return _merge((max(s, lo), min(e, hi)) for s, e in intervals)


def _subtract(a: Intervals, b: Intervals) -> Intervals:
    """The parts of ``a`` outside ``b``. Both are merged and sorted."""
    out: Intervals = []
    for start, end in a:
        cursor = start
        for b_start, b_end in b:
            if b_end <= cursor or b_start >= end:
                continue
            if b_start > cursor:
                out.append((cursor, b_start))
            cursor = max(cursor, b_end)
            if cursor >= end:
                break
        if cursor < end:
            out.append((cursor, end))
    return out


def _intersect(a: Intervals, b: Intervals) -> Intervals:
    return _subtract(a, _subtract(a, b))


def _length(intervals: Intervals) -> float:
    return sum(end - start for start, end in intervals)


# --- event index -----------------------------------------------------------------------


class _Spans:
    """Spans (start, end, payload) of one kind per thread, sorted by start."""

    def __init__(self) -> None:
        self._items: Dict[Any, List[Tuple[float, float, Any]]] = defaultdict(list)
        self._starts: Dict[Any, List[float]] = {}

    def add(self, tid: Any, start: float, end: float, payload: Any = None) -> None:
        self._items[tid].append((start, end, payload))

    def freeze(self) -> None:
        for tid, items in self._items.items():
            items.sort(key=lambda item: (item[0], -item[1]))
            self._starts[tid] = [item[0] for item in items]

    def all(self) -> List[Tuple[Any, float, float]]:
        """Every span as (tid, start, end)."""
        return [(tid, s, e) for tid, items in self._items.items() for s, e, _ in items]

    def starting_in(self, tid: Any, lo: float, hi: float) -> List[Tuple[float, float, Any]]:
        """Spans of ``tid`` whose start is in [lo, hi)."""
        items = self._items.get(tid, [])
        starts = self._starts.get(tid, [])
        out = []
        for index in range(bisect_left(starts, lo), len(items)):
            if items[index][0] >= hi:
                break
            out.append(items[index])
        return out

    def overlapping(self, tid: Any, lo: float, hi: float) -> List[Tuple[float, float, Any]]:
        """Spans of ``tid`` that overlap (lo, hi). Spans of one kind do not nest, so the one
        span that starts before ``lo`` and reaches it is the one just before the first start
        at or after ``lo``."""
        items = self._items.get(tid, [])
        starts = self._starts.get(tid, [])
        first = bisect_left(starts, lo)
        out = []
        if first > 0 and items[first - 1][1] > lo:
            out.append(items[first - 1])
        for index in range(first, len(items)):
            if items[index][0] >= hi:
                break
            out.append(items[index])
        return out


class _Index:
    def __init__(self, events: Sequence[Mapping[str, Any]]) -> None:
        self.ranges: Dict[str, _Spans] = defaultdict(_Spans)
        self.launches = _Spans()
        self.ops = _Spans()
        self.others: List[Tuple[Any, float, float]] = []
        self.syncs: List[Tuple[Any, float, float]] = []
        self.kernel_us: Dict[Any, float] = defaultdict(float)
        self.dtod: set = set()
        for event in events:
            if event.get("ph") != "X" or "ts" not in event or "dur" not in event:
                continue
            cat, name = event.get("cat"), event.get("name", "")
            tid, start = event.get("tid"), float(event["ts"])
            end = start + float(event["dur"])
            corr = (event.get("args") or {}).get("correlation")
            if cat == "user_annotation" and name.startswith(RANGES.prefix):
                self.ranges[name].add(tid, start, end)
            elif cat == "cuda_runtime":
                if name in QUEUEING_APIS:
                    self.launches.add(tid, start, end, (name, corr))
                if name in SYNC_APIS:
                    self.syncs.append((tid, start, end))
                self.others.append((tid, start, end))
            elif cat == "cpu_op":
                self.ops.add(tid, start, end, name)
                self.others.append((tid, start, end))
            elif cat == "user_annotation":
                self.others.append((tid, start, end))
            elif cat == "kernel" and corr is not None:
                self.kernel_us[corr] += end - start
            elif cat == "gpu_memcpy" and corr is not None and "DtoD" in name:
                self.dtod.add(corr)
        for spans in self.ranges.values():
            spans.freeze()
        self.launches.freeze()
        self.ops.freeze()

    def span_list(self, name: str) -> List[Tuple[Any, float, float]]:
        spans = self.ranges.get(name)
        if spans is None:
            return []
        return spans.all()

    def intervals(self, name: str, tid: Any, lo: float, hi: float) -> Intervals:
        spans = self.ranges.get(name)
        if spans is None:
            return []
        return _clip(((s, e) for s, e, _ in spans.overlapping(tid, lo, hi)), lo, hi)


# --- visits and layer backwards ---------------------------------------------------------


def _launch_facts(
    index: _Index, tid: Any, region: Intervals, lo: float, hi: float
) -> Tuple[float, float, Dict[str, int]]:
    """Launch time (us), kernel time (us) and launches per API of the launches whose start is
    in ``region`` (merged intervals inside [lo, hi))."""
    spans: Intervals = []
    counts: Counter = Counter()
    kernel = 0.0
    for start, end, (name, corr) in index.launches.starting_in(tid, lo, hi):
        if not any(r_start <= start < r_end for r_start, r_end in region):
            continue
        spans.append((start, end))
        counts[name] += 1
        kernel += index.kernel_us.get(corr, 0.0)
    launch = _length(_intersect(_merge(spans), region))
    return launch, kernel, dict(counts)


def _visit(index: _Index, kind: str, tid: Any, lo: float, hi: float) -> Dict[str, Any]:
    whole: Intervals = [(lo, hi)]
    launch_us, kernel_us, counts = _launch_facts(index, tid, whole, lo, hi)
    launch_set = _intersect(
        _merge((s, e) for s, e, _ in index.launches.starting_in(tid, lo, hi)), whole
    )
    prep = _subtract(index.intervals(RANGES.weight_prep, tid, lo, hi), launch_set)
    machinery = _subtract(
        index.intervals(RANGES.functional_call, tid, lo, hi),
        index.intervals(RANGES.layer_call, tid, lo, hi),
    )
    machinery = _subtract(_subtract(machinery, launch_set), prep)
    host = hi - lo
    parts = {
        "launch_ms": launch_us,
        "weight_prep_ms": _length(prep),
        "functional_call_ms": _length(machinery),
    }
    parts["python_dispatcher_ms"] = host - sum(parts.values())
    result: Dict[str, Any] = {
        "kind": kind,
        "tid": tid,
        "start_us": lo,
        "host_ms": host / _US_PER_MS,
    }
    result.update({name: value / _US_PER_MS for name, value in parts.items()})
    result["kernel_ms"] = kernel_us / _US_PER_MS
    result["launches"] = counts
    return result


def _layer_backward(index: _Index, tid: Any, lo: float, hi: float) -> Dict[str, Any]:
    recompute = _clip(
        ((s, e) for s, e, _ in index.ranges[RANGES.recompute_visit].overlapping(tid, lo, hi)),
        lo,
        hi,
    )
    region = _subtract([(lo, hi)], recompute)
    launch_us, kernel_us, counts = _launch_facts(index, tid, region, lo, hi)
    host = _length(region)
    return {
        "tid": tid,
        "start_us": lo,
        "host_ms": host / _US_PER_MS,
        "launch_ms": launch_us / _US_PER_MS,
        "rest_ms": (host - launch_us) / _US_PER_MS,
        "kernel_ms": kernel_us / _US_PER_MS,
        "launches": counts,
    }


def _median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else 0.0


def _summarise(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> Dict[str, Any]:
    apis = sorted({api for row in rows for api in row["launches"]})
    summary: Dict[str, Any] = {"count": len(rows)}
    for field in fields:
        summary[field] = _median([row[field] for row in rows])
    summary["launches_per_visit"] = {
        api: _median([row["launches"].get(api, 0) for row in rows]) for api in apis
    }
    summary["launches_total"] = _median([sum(row["launches"].values()) for row in rows])
    return summary


# --- reconciliation ---------------------------------------------------------------------


def _reconcile(index: _Index, tolerance: float) -> Optional[Dict[str, Any]]:
    steps = sorted(index.span_list(RANGES.step), key=lambda item: item[1])
    if not steps:
        return None
    visits = [
        (tid, s, e)
        for name in (RANGES.forward_visit, RANGES.recompute_visit)
        for tid, s, e in index.span_list(name)
    ]
    windows = index.span_list(RANGES.layer_backward)
    recomputes = index.span_list(RANGES.recompute_visit)
    totals: Counter = Counter()
    per_step = []
    for step_tid, lo, hi in steps:
        inside = lambda item: lo <= item[1] and item[2] <= hi  # noqa: E731
        step_visits = [v for v in visits if inside(v)]
        step_windows = [w for w in windows if inside(w)]
        recompute_in = [r for r in recomputes if inside(r)]
        visits_us = sum(e - s for _, s, e in step_visits)
        backward_set = _merge((s, e) for _, s, e in step_windows)
        backward_set = _subtract(backward_set, _merge((s, e) for _, s, e in recompute_in))
        backward_us = _length(backward_set)
        covered = _merge([(s, e) for _, s, e in step_visits + step_windows])
        waits = [(s, e) for tid, s, e in index.syncs if tid == step_tid and lo <= s and e <= hi]
        final_wait = [max(waits)] if waits else []
        final_wait = _subtract(_merge(final_wait), covered)
        threads = {step_tid} | {tid for tid, _, _ in step_visits + step_windows}
        elsewhere = _clip(
            ((s, e) for tid, s, e in index.others if tid in threads), lo, hi
        )
        outside = _subtract(_subtract(elsewhere, covered), final_wait)
        step_us = hi - lo
        row = {
            "step_ms": step_us,
            "visits_ms": visits_us,
            "layer_backward_ms": backward_us,
            "outside_ms": _length(outside),
            "final_wait_ms": _length(final_wait),
        }
        row["gap_ms"] = step_us - sum(
            row[k] for k in ("visits_ms", "layer_backward_ms", "outside_ms", "final_wait_ms")
        )
        row["gap_fraction"] = row["gap_ms"] / step_us if step_us else 0.0
        per_step.append({key: value / _US_PER_MS if key.endswith("_ms") else value
                         for key, value in row.items()})
        totals.update(row)
    gap_fraction = totals["gap_ms"] / totals["step_ms"] if totals["step_ms"] else 0.0
    result: Dict[str, Any] = {
        key: value / _US_PER_MS for key, value in totals.items()
    }
    result.update(
        {
            "steps": per_step,
            "gap_fraction": gap_fraction,
            "tolerance": tolerance,
            "ok": abs(gap_fraction) <= tolerance + 1e-9,
        }
    )
    return result


# --- device-to-device copies ------------------------------------------------------------


def _memcpy_sources(index: _Index, visits: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    """The ops enclosing each device-to-device copy launch in a visit, outermost first, with
    the median number of such copies per visit."""
    per_visit: List[Counter] = []
    for visit in visits:
        tid, lo = visit["tid"], visit["start_us"]
        hi = lo + visit["host_ms"] * _US_PER_MS
        ops = index.ops.starting_in(tid, lo, hi)
        chains: Counter = Counter()
        for start, _end, (name, corr) in index.launches.starting_in(tid, lo, hi):
            if name != "cudaMemcpyAsync" or corr not in index.dtod:
                continue
            chain = [op[2] for op in ops if op[0] <= start <= op[1]]
            chains[" > ".join(chain) if chain else "(no enclosing op)"] += 1
        per_visit.append(chains)
    keys = sorted({key for chains in per_visit for key in chains})
    return {key: _median([chains.get(key, 0) for chains in per_visit]) for key in keys}


# --- entry point ------------------------------------------------------------------------


def analyse_trace(
    events: Sequence[Mapping[str, Any]], *, tolerance: float = RECONCILIATION_TOLERANCE
) -> Dict[str, Any]:
    """Cut the profiler ``events`` into per-visit parts. See the module docstring."""
    index = _Index(events)
    visits = [
        _visit(index, kind, tid, lo, hi)
        for kind, name in (("forward", RANGES.forward_visit), ("recompute", RANGES.recompute_visit))
        for tid, lo, hi in sorted(index.span_list(name), key=lambda item: item[1])
    ]
    backwards = [
        _layer_backward(index, tid, lo, hi)
        for tid, lo, hi in sorted(index.span_list(RANGES.layer_backward), key=lambda i: i[1])
    ]
    visit_fields = (
        "host_ms",
        "launch_ms",
        "weight_prep_ms",
        "functional_call_ms",
        "python_dispatcher_ms",
        "kernel_ms",
    )
    summary = {
        kind: _summarise([v for v in visits if v["kind"] == kind], visit_fields)
        for kind in ("forward", "recompute")
    }
    summary["layer_backward"] = _summarise(
        backwards, ("host_ms", "launch_ms", "rest_ms", "kernel_ms")
    )
    return {
        "visits": visits,
        "layer_backwards": backwards,
        "summary": summary,
        "reconciliation": _reconcile(index, tolerance),
        "memcpy_sources": _memcpy_sources(index, visits),
    }
