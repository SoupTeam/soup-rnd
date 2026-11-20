"""GPU-side diagnosis of one step (harness only): what the GPU did while the step ran.

``maybe_profile()`` wraps the stepper call in the bridge when ``SOUP_GPU_PROFILE_JSON`` names a
file prefix: a ``torch.profiler`` CUDA trace of the step is reduced by ``summarize_kernels`` to
total kernel time, the real busy time of the GPU (overlapping kernels of several streams counted
once), idle time inside the step, a split by kind of work (GEMM, elementwise, copies) and the
heaviest kernels. The same summary on two hosts says whether the step is limited by the GPU or
by the host feeding it. Profiling slows the CPU side down; kernel durations are unaffected, so
the wall time of a profiled step is not a result, the GPU totals are.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterator, List

_GEMM_MARKS = ("gemm", "gemv", "cutlass", "cublas", "splitk", "s16816", "sgemm")
_COPY_MARKS = ("memcpy", "memset")
_ELEMENTWISE_MARKS = ("elementwise", "vectorized", "unrolled", "reduce", "index", "cat", "copy_")
TOP_KERNELS = 15
_RUN_COUNTER = [0]


# (upper limit in us, label) of the idle-gap buckets: gaps between kernels of a step
GAP_BUCKETS = [(20, "<20us"), (100, "20-100us"), (1000, "100us-1ms"), (10000, "1-10ms"),
               (float("inf"), ">=10ms")]
TOP_GAPS = 12
TOP_CPU_OPS = 30
TOP_THREADS = 12


def category_of(name: str) -> str:
    """``gemm``, ``memcpy``, ``elementwise`` or ``other`` for a kernel or copy name."""
    low = name.lower()
    if any(mark in low for mark in _COPY_MARKS):
        return "memcpy"
    if any(mark in low for mark in _GEMM_MARKS):
        return "gemm"
    if any(mark in low for mark in _ELEMENTWISE_MARKS):
        return "elementwise"
    return "other"


def summarize_kernels(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Totals of ``rows`` (``name``, ``start_us``, ``dur_us``) for one profiled step."""
    if not rows:
        return {
            "kernels": 0, "sum_dur_us": 0, "busy_us": 0, "span_us": 0, "idle_us": 0,
            "busy_fraction": 0.0, "by_category": {}, "top": [], "gaps": {}, "long_gaps": [],
        }
    ordered = sorted(rows, key=lambda r: r["start_us"])
    intervals = [(r["start_us"], r["start_us"] + r["dur_us"]) for r in ordered]
    busy = 0.0
    current_start, current_end = intervals[0]
    last_name = ordered[0]["name"]            # the kernel that ends the busy interval so far
    gap_stats = {label: {"count": 0, "us": 0} for _, label in GAP_BUCKETS}
    long_gaps: List[Dict[str, Any]] = []
    for r, (start, end) in zip(ordered[1:], intervals[1:]):
        if start > current_end:
            gap = start - current_end
            for limit, label in GAP_BUCKETS:
                if gap < limit:
                    gap_stats[label]["count"] += 1
                    gap_stats[label]["us"] += gap
                    break
            long_gaps.append({"us": gap, "before": last_name, "after": r["name"]})
            busy += current_end - current_start
            current_start, current_end, last_name = start, end, r["name"]
        elif end > current_end:
            current_end, last_name = end, r["name"]
    busy += current_end - current_start
    span = max(end for _, end in intervals) - intervals[0][0]
    by_category: Dict[str, Dict[str, float]] = defaultdict(lambda: {"kernels": 0, "us": 0})
    by_name: Dict[str, Dict[str, float]] = defaultdict(lambda: {"kernels": 0, "us": 0})
    for r in rows:
        cat = by_category[category_of(r["name"])]
        cat["kernels"] += 1
        cat["us"] += r["dur_us"]
        item = by_name[r["name"]]
        item["kernels"] += 1
        item["us"] += r["dur_us"]
    top = sorted(by_name.items(), key=lambda kv: -kv[1]["us"])[:TOP_KERNELS]
    return {
        "kernels": len(rows),
        "sum_dur_us": sum(r["dur_us"] for r in rows),
        "busy_us": busy,
        "span_us": span,
        "idle_us": span - busy,
        "busy_fraction": busy / span if span else 0.0,
        "by_category": {k: dict(v) for k, v in by_category.items()},
        "gaps": gap_stats,
        "long_gaps": sorted(long_gaps, key=lambda g: -g["us"])[:TOP_GAPS],
        "top": [{"name": n, "kernels": v["kernels"], "us": v["us"]} for n, v in top],
    }


def _merged(intervals: List[Any]) -> List[Any]:
    out: List[List[float]] = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return [(a, b) for a, b in out]


def _coverage(intervals: List[Any], start_us: float, window_us: float, count: int) -> List[float]:
    """Busy fraction of each of ``count`` consecutive windows for merged ``intervals``."""
    busy = [0.0] * count
    for a, b in _merged(intervals):
        first = max(int((a - start_us) // window_us), 0)
        last = min(int((b - start_us) // window_us), count - 1)
        for w in range(first, last + 1):
            lo = start_us + w * window_us
            busy[w] += max(0.0, min(b, lo + window_us) - max(a, lo))
    return [value / window_us for value in busy]


def classify_windows(gpu: List[Any], cpu_by_thread: Dict[Any, List[Any]], window_us: float,
                     start_us: float, end_us: float) -> Dict[str, Any]:
    """Class of every window of a step: ``gpu`` (the GPU is busy 80% or more), ``cpu`` (some
    thread runs ops 80% or more of it), ``mixed`` (either side 30% or more), else ``stall`` (both
    mostly idle: the step waits on a copy, a lock or a sync). ``classes`` give the windows, their
    share of the step and the mean GPU and busiest-thread fractions per class."""
    count = max(int((end_us - start_us) // window_us), 1)
    gpu_busy = _coverage(gpu, start_us, window_us, count)
    cpu_busy = [0.0] * count
    for intervals in cpu_by_thread.values():
        for w, value in enumerate(_coverage(intervals, start_us, window_us, count)):
            cpu_busy[w] = max(cpu_busy[w], value)
    classes: Dict[str, Dict[str, float]] = {
        name: {"windows": 0, "share": 0.0, "gpu_busy": 0.0, "cpu_busy": 0.0}
        for name in ("gpu", "cpu", "mixed", "stall")
    }
    for g, c in zip(gpu_busy, cpu_busy):
        name = ("gpu" if g >= 0.8 else "cpu" if c >= 0.8 else "mixed" if max(g, c) >= 0.3
                else "stall")
        classes[name]["windows"] += 1
        classes[name]["gpu_busy"] += g
        classes[name]["cpu_busy"] += c
    for entry in classes.values():
        n = entry["windows"]
        entry["share"] = n / count
        entry["gpu_busy"] = entry["gpu_busy"] / n if n else 0.0
        entry["cpu_busy"] = entry["cpu_busy"] / n if n else 0.0
    return {"windows": count, "window_us": window_us, "classes": classes}


def classify_chrome_trace(trace: Dict[str, Any], window_us: float = 50_000.0) -> Dict[str, Any]:
    """``classify_windows`` of a ``torch.profiler`` Chrome trace: GPU kernels, copies and memsets
    against CPU ops (``cpu_op`` events) per thread, over the span of the GPU work."""
    gpu: List[Any] = []
    cpu: Dict[Any, List[Any]] = defaultdict(list)
    ops: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0, "us": 0})
    names: Dict[Any, str] = {}
    for event in trace.get("traceEvents", []):
        if event.get("ph") == "M" and event.get("name") == "thread_name":
            names[event.get("tid")] = str(event.get("args", {}).get("name", ""))
            continue
        if event.get("ph") != "X" or "ts" not in event:
            continue
        interval = (float(event["ts"]), float(event["ts"]) + float(event.get("dur", 0.0)))
        category = event.get("cat")
        if category in ("kernel", "gpu_memcpy", "gpu_memset"):
            gpu.append(interval)
        elif category == "cpu_op":
            cpu[event.get("tid")].append(interval)
            row = ops[event.get("name", "?")]
            row["count"] += 1
            row["us"] += interval[1] - interval[0]
    if not gpu:
        return {"windows": 0, "window_us": window_us, "classes": {}}
    out = classify_windows(gpu, cpu, window_us, min(a for a, _ in gpu), max(b for _, b in gpu))
    span = max(b for _, b in gpu) - min(a for a, _ in gpu)
    threads = []
    for tid, intervals in cpu.items():
        busy = sum(b - a for a, b in _merged(intervals))
        threads.append({"tid": tid, "name": names.get(tid, ""), "busy_us": busy,
                        "busy_share": busy / span if span else 0.0, "ops": len(intervals)})
    out["threads"] = sorted(threads, key=lambda row: -row["busy_us"])[:TOP_THREADS]
    out["cpu_ops"] = [{"name": n, "count": v["count"], "us": v["us"]}
                      for n, v in sorted(ops.items(), key=lambda kv: -kv[1]["us"])[:TOP_CPU_OPS]]
    return out


def rows_from_profile(prof: Any) -> List[Dict[str, Any]]:
    """CUDA kernel, copy and memset rows of a finished ``torch.profiler`` trace."""
    import torch

    rows: List[Dict[str, Any]] = []
    for event in prof.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        rows.append({
            "name": event.name,
            "start_us": float(event.time_range.start),
            "dur_us": float(event.time_range.elapsed_us()),
        })
    return rows


@contextlib.contextmanager
def maybe_profile() -> Iterator[None]:
    """Profile the enclosed step when ``SOUP_GPU_PROFILE_JSON`` is set; otherwise do nothing."""
    prefix = os.environ.get("SOUP_GPU_PROFILE_JSON")
    if not prefix:
        yield
        return
    import torch
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        yield
        torch.cuda.synchronize()
    summary = summarize_kernels(rows_from_profile(prof))
    _RUN_COUNTER[0] += 1
    if os.environ.get("SOUP_GPU_TRACE_WINDOWS"):     # busy classes of 50 ms windows, from the trace
        trace_path = f"{prefix}.{_RUN_COUNTER[0]}.trace.json"
        prof.export_chrome_trace(trace_path)
        with open(trace_path) as handle:
            summary["windows"] = classify_chrome_trace(json.load(handle))
        os.remove(trace_path)
    with open(f"{prefix}.{_RUN_COUNTER[0]}.json", "w") as handle:
        json.dump(summary, handle, indent=1)
