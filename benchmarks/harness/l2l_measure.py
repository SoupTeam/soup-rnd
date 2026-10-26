#!/usr/bin/env python3
"""A5 measurement record: speed, GPU and RAM, and I/O in one record per step and per run.

``step_record`` holds one step: tokens without padding per second, the step time,
the VRAM peaks (allocated and reserved), the process RSS peak, the pinned bytes of
the activation store, the bytes read from the weights' disk, the bytes the spill
wrote and read, the layer loads and the NF4 restores. ``run_record`` folds the
steps of one run into the same fields.

A value whose source is missing is ``None`` with its reason under ``unavailable``,
never zero. A ``None`` without a reason, or a reason for a value that is there, is
refused, so a record cannot hide a gap.

Benchmark harness code, not shipped. Standard library only.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

#: Fields that hold a peak: a run takes the largest step value.
PEAK_FIELDS = (
    "peak_vram_allocated_bytes",
    "peak_vram_reserved_bytes",
    "peak_rss_bytes",
    "pinned_store_bytes",
)

#: Fields that count work or bytes moved: a run sums its steps.
COUNT_FIELDS = (
    "weight_read_bytes",
    "spill_written_bytes",
    "spill_read_bytes",
    "layer_loads",
    "nf4_restores",
)

MEASURED_FIELDS = PEAK_FIELDS + COUNT_FIELDS


def real_tokens(attention_mask: Any) -> int:
    """Tokens that are not padding: the ones in an attention mask (rows of 0 and 1)."""
    total = getattr(attention_mask, "sum", None)
    if callable(total):
        return int(total())
    return sum(int(value) for row in attention_mask for value in row)


def _check_reasons(values: Mapping[str, Optional[int]], unavailable: Mapping[str, str]) -> None:
    for name, value in values.items():
        if value is None and name not in unavailable:
            raise ValueError(f"{name} is null without a reason in 'unavailable'")
        if value is not None and name in unavailable:
            raise ValueError(f"{name} has a value and a reason for being unavailable")
    unknown = sorted(set(unavailable) - set(values))
    if unknown:
        raise ValueError(f"no such measured fields: {', '.join(unknown)}")


def step_record(
    *,
    step_s: float,
    real_tokens: int,
    layer_loads: Optional[int],
    nf4_restores: Optional[int],
    spill_written_bytes: Optional[int],
    spill_read_bytes: Optional[int],
    peak_vram_allocated_bytes: Optional[int],
    peak_vram_reserved_bytes: Optional[int],
    peak_rss_bytes: Optional[int],
    pinned_store_bytes: Optional[int],
    weight_read_bytes: Optional[int],
    unavailable: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """One step. Tokens are the ones without padding; bytes are bytes, time is seconds."""
    if not step_s > 0:
        raise ValueError(f"step_s must be positive, got {step_s}")
    if real_tokens <= 0:
        raise ValueError(f"real_tokens must be positive, got {real_tokens}")
    values = {
        "peak_vram_allocated_bytes": peak_vram_allocated_bytes,
        "peak_vram_reserved_bytes": peak_vram_reserved_bytes,
        "peak_rss_bytes": peak_rss_bytes,
        "pinned_store_bytes": pinned_store_bytes,
        "weight_read_bytes": weight_read_bytes,
        "spill_written_bytes": spill_written_bytes,
        "spill_read_bytes": spill_read_bytes,
        "layer_loads": layer_loads,
        "nf4_restores": nf4_restores,
    }
    reasons = dict(unavailable or {})
    _check_reasons(values, reasons)
    return {
        "step_s": float(step_s),
        "real_tokens": int(real_tokens),
        "tokens_per_s": real_tokens / step_s,
        **values,
        "unavailable": reasons,
    }


def run_record(steps: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """One run from its steps: tokens over time, the largest peaks, summed counters.

    A field is ``None`` when any step lacks it, with the steps' reasons joined.
    """
    if not steps:
        raise ValueError("a run needs at least one step")
    total_s = sum(step["step_s"] for step in steps)
    total_tokens = sum(step["real_tokens"] for step in steps)
    record: Dict[str, Any] = {
        "steps": len(steps),
        "step_s_mean": total_s / len(steps),
        "real_tokens": total_tokens,
        "tokens_per_s": total_tokens / total_s,
    }
    reasons: Dict[str, str] = {}
    for name in MEASURED_FIELDS:
        column: List[Optional[int]] = [step[name] for step in steps]
        if any(value is None for value in column):
            record[name] = None
            found = (step["unavailable"][name] for step in steps if name in step["unavailable"])
            reasons[name] = "; ".join(dict.fromkeys(found))
        elif name in PEAK_FIELDS:
            record[name] = max(column)  # type: ignore[type-var]
        else:
            record[name] = sum(column)  # type: ignore[arg-type]
    record["unavailable"] = reasons
    return record


#: ``counters()`` fields that only grow: a step reports the difference.
CUMULATIVE_COUNTERS = ("layer_loads", "nf4_restores", "spill_written_bytes", "spill_read_bytes")

NO_IO_COUNTER = "no operating-system read counter was supplied"


def _peak_rss() -> Optional[int]:
    from a4_box import process_memory

    return process_memory()["peak_rss"]


def _cuda_peaks(device: str) -> Dict[str, Optional[int]]:
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        return {"allocated": None, "reserved": None}
    torch.cuda.synchronize()
    return {
        "allocated": int(torch.cuda.max_memory_allocated()),
        "reserved": int(torch.cuda.max_memory_reserved()),
    }


def measure_step(
    body: Callable[[], Any],
    *,
    attention_mask: Any,
    device: str,
    counters: Callable[[], Mapping[str, int]],
    read_bytes: Optional[Callable[[], Optional[int]]] = None,
) -> Dict[str, Any]:
    """Run one step and return its ``step_record``.

    ``counters()`` returns ``layer_loads``, ``nf4_restores``, ``spill_written_bytes`` and
    ``spill_read_bytes`` as running totals, and ``pinned_store_bytes`` as a level.
    ``read_bytes()`` is the operating system's running total of bytes the process read
    from the weights' disk; without it the field is null. VRAM peaks are null off CUDA,
    and the RSS peak is null where the platform gives none.
    """
    import torch

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    before = dict(counters())
    read_before = read_bytes() if read_bytes is not None else None
    started = time.perf_counter()
    body()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    step_s = time.perf_counter() - started
    after = dict(counters())
    read_after = read_bytes() if read_bytes is not None else None

    reasons: Dict[str, str] = {}
    cuda = _cuda_peaks(device)
    for name, key in (("peak_vram_allocated_bytes", "allocated"),
                      ("peak_vram_reserved_bytes", "reserved")):
        if cuda[key] is None:
            reasons[name] = "no CUDA device"
    peak_rss = _peak_rss()
    if peak_rss is None:
        reasons["peak_rss_bytes"] = "the platform gives no peak RSS"
    weight_read = None if read_before is None or read_after is None else read_after - read_before
    if weight_read is None:
        reasons["weight_read_bytes"] = NO_IO_COUNTER
    delta = {name: after[name] - before[name] for name in CUMULATIVE_COUNTERS}
    return step_record(
        step_s=step_s,
        real_tokens=real_tokens(attention_mask),
        pinned_store_bytes=after["pinned_store_bytes"],
        peak_vram_allocated_bytes=cuda["allocated"],
        peak_vram_reserved_bytes=cuda["reserved"],
        peak_rss_bytes=peak_rss,
        weight_read_bytes=weight_read,
        unavailable=reasons,
        **delta,
    )
