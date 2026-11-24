"""Summaries of one profiled ``L2LStep.run`` step for the ticket 15 phase profile (harness only).

Everything here is pure and duck-typed over ``torch.profiler`` ``FunctionEvent``-like objects, so
the tests need no torch and no GPU. A routed-expert GEMM is selected by its weight shape: the
Moonlight-16B-A3B routed experts are ``(2816, 2048)`` and ``(2048, 1408)``, while the shared expert
is ``(5632, 2048)`` / ``(2048, 2816)`` and attention is ``(2048, 2048)``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

# Weight shapes of one routed expert: gate/up fused, then down. Moonlight-16B-A3B only.
EXPERT_WEIGHT_SHAPES: Dict[str, tuple] = {"gate_up": (2816, 2048), "down": (2048, 1408)}
_PROJECTION_BY_WEIGHT = {shape: name for name, shape in EXPERT_WEIGHT_SHAPES.items()}


def _weight_shape(event: Any) -> tuple:
    """The second input shape of the event, when there is one."""
    shapes = list(getattr(event, "input_shapes", ()) or ())
    if len(shapes) < 2:
        return ()
    return tuple(shapes[1])


def expert_calls(events: Iterable[Any]) -> List[Dict[str, Any]]:
    """One entry per routed-expert GEMM: rows, projection, kernel string, seconds, kernel count."""
    calls: List[Dict[str, Any]] = []
    for event in events:
        if getattr(event, "name", None) != "aten::mm":
            continue
        weight = _weight_shape(event)
        projection = _PROJECTION_BY_WEIGHT.get(weight)
        if projection is None:
            continue
        kernels = list(getattr(event, "kernels", ()) or ())
        if not kernels:
            continue
        calls.append({
            "rows": int(list(event.input_shapes)[0][0]),
            "projection": projection,
            "kernel": " | ".join(str(k.name) for k in kernels),
            "seconds": sum(float(k.duration_time()) for k in kernels) / 1e6,
            "kernels": len(kernels),
        })
    return calls


def rows_histogram(calls: Sequence[Dict[str, Any]]) -> Dict[int, int]:
    """Rows -> number of expert GEMM calls, ascending by row count."""
    counts: Dict[int, int] = {}
    for call in calls:
        rows = int(call["rows"])
        counts[rows] = counts.get(rows, 0) + 1
    return dict(sorted(counts.items()))


def kernel_timeline(events: Iterable[Any], wall_seconds: float) -> Dict[str, Any]:
    """Device timeline of the step: kernel launches, copies, busy span and the exposed gap.

    Kernel and copy names come from ``Memcpy*`` / ``Memset*`` prefixes; every other device event is
    a kernel. Intervals are the ``time_range`` of the event in microseconds, and ``busy_seconds``
    counts overlapping intervals once.
    """
    kernels: List[tuple] = []
    copy_intervals: List[tuple] = []
    counts: Dict[str, int] = {}
    seconds: Dict[str, float] = {}
    intervals: List[tuple] = []
    for event in events:
        if not str(getattr(event, "device_type", "")).endswith("CUDA"):
            continue
        span = getattr(event, "time_range", None)
        if span is None:
            continue
        start, end = float(span.start), float(span.end)
        intervals.append((start, end))
        name = str(getattr(event, "name", "?"))
        if name.startswith(("Memcpy", "Memset")):
            copy_intervals.append((start, end))
            continue
        kernels.append((start, end))
        counts[name] = counts.get(name, 0) + 1
        seconds[name] = seconds.get(name, 0.0) + (end - start)

    def union(items: Sequence[tuple]) -> float:
        total, right = 0.0, None
        for start, end in sorted(items):
            if right is None or start > right:
                total += end - start
                right = end
            elif end > right:
                total += end - right
                right = end
        return total

    busy = union(intervals) / 1e6
    span = ((max(finish for _, finish in intervals) - min(begin for begin, _ in intervals)) / 1e6
            if intervals else 0.0)
    top = sorted(({"name": name, "calls": counts[name], "seconds": seconds[name] / 1e6}
                  for name in counts), key=lambda entry: entry["seconds"], reverse=True)
    return {"kernels": len(kernels), "copies": len(copy_intervals),
            "kernel_seconds": sum(seconds.values()) / 1e6, "busy_seconds": busy,
            "span_seconds": span, "gap_seconds": span - busy,
            "exposed_seconds": max(0.0, float(wall_seconds) - busy), "top": top[:10]}


def kernel_table(calls: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Kernel string -> calls, summed seconds, and the row range it was launched for."""
    table: Dict[str, Dict[str, Any]] = {}
    for call in calls:
        name = str(call["kernel"])
        entry = table.get(name)
        if entry is None:
            table[name] = {"calls": 1, "seconds": float(call["seconds"]),
                           "rows_min": int(call["rows"]), "rows_max": int(call["rows"])}
            continue
        entry["calls"] += 1
        entry["seconds"] += float(call["seconds"])
        entry["rows_min"] = min(entry["rows_min"], int(call["rows"]))
        entry["rows_max"] = max(entry["rows_max"], int(call["rows"]))
    return table


def rows_table(calls: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Rows -> calls, summed seconds and microseconds per row: the M efficiency of the GEMMs."""
    table: Dict[int, Dict[str, Any]] = {}
    for call in calls:
        rows = int(call["rows"])
        entry = table.setdefault(rows, {"calls": 0, "seconds": 0.0})
        entry["calls"] += 1
        entry["seconds"] += float(call["seconds"])
    for rows, entry in table.items():
        entry["us_per_row"] = entry["seconds"] * 1e6 / rows
    return dict(sorted(table.items()))


def step_summary(events: Iterable[Any], wall_seconds: float) -> Dict[str, Any]:
    """Whole-step summary: wall, CPU self time, the routed-expert GEMM counters and the timeline."""
    events = list(events)
    calls = expert_calls(events)
    return {
        "wall_seconds": float(wall_seconds),
        "cpu_self_seconds": sum(
            float(getattr(event, "self_cpu_time_total", 0.0) or 0.0)
            for event in events if str(getattr(event, "device_type", "")).endswith("CPU")) / 1e6,
        "expert_gemm": {
            "calls": len(calls),
            "rows_total": sum(int(call["rows"]) for call in calls),
            "seconds": sum(float(call["seconds"]) for call in calls),
            "rows": rows_histogram(calls),
            "kernels": kernel_table(calls),
            "by_rows": rows_table(calls),
        },
        "timeline": kernel_timeline(events, wall_seconds),
    }
