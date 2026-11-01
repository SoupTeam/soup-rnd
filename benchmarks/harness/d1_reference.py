"""Validate a separate, frequency-matched dense reference without CUDA imports."""

from __future__ import annotations

import math
from typing import Any

from d1_rules import CLOCK_SPREAD


def _number(value: Any, minimum: float = 0) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= minimum
    )


def validate_reference(primary: dict, reference: dict) -> dict:
    """No ceiling verdict from unsampled work, another device or changed clocks.

    Do not trust the precomputed clock lists: derive samples from the actual
    recorded phase boundaries and check that the exported aggregates agree.
    The reference needs at least three samples over >=1 second, >=200 calls
    and >=1 second/20 calls of warm-up. Primary update counts stay unchanged.
    """
    reasons, samples = [], []
    identity = primary.get("gpu_uuid")
    if not identity or reference.get("gpu_uuid") != identity:
        reasons.append("missing or mixed CUDA GPU UUID")
    for name, row in (("primary", primary), ("reference", reference)):
        start, end = row.get("timed_start_ns"), row.get("timed_end_ns")
        if not _number(start, 1) or not _number(end, 1) or end <= start:
            reasons.append(f"{name}: invalid timed window")
            continue
        window = [
            sample
            for sample in row.get("telemetry", [])
            if _number(sample.get("time_ns"), 1) and start <= sample["time_ns"] <= end
        ]
        samples.extend(window)
        if len(window) < (3 if name == "reference" else 1):
            reasons.append(f"{name}: insufficient in-window telemetry")
        if row.get("telemetry_errors") or row.get("telemetry_gpu_matches_cuda") is not True:
            reasons.append(f"{name}: failed telemetry")
        for sample in window:
            if sample.get("available") is not True or sample.get("uuid") != identity:
                reasons.append(f"{name}: unavailable or foreign GPU sample")
            if not _number(sample.get("sm_mhz"), 1):
                reasons.append(f"{name}: missing/nonfinite SM clock")
            if not _number(sample.get("power_limit_mw"), 1):
                reasons.append(f"{name}: missing/nonfinite power limit")
            if not _number(sample.get("memory_util_pct")) or sample["memory_util_pct"] > 100:
                reasons.append(f"{name}: invalid memory-controller utilization")
        for field, source in (
            ("clock_sm_mhz", "sm_mhz"),
            ("power_limit_mw", "power_limit_mw"),
            ("memory_util_pct", "memory_util_pct"),
        ):
            if row.get(field) != [sample.get(source) for sample in window]:
                reasons.append(f"{name}: aggregate differs from sampled window: {field}")
        events = row.get("step_cuda_ms", [])
        if (
            not events
            or any(not _number(value, 1e-12) for value in events)
            or row.get("actual_timed_steps") != len(events)
        ):
            reasons.append(f"{name}: missing/nonfinite event timings or count mismatch")
    if (
        not _number(reference.get("actual_timed_seconds"), 1)
        or not _number(reference.get("actual_timed_steps"), 200)
        or not _number(reference.get("actual_warmup_steps"), 20)
        or not _number(reference.get("actual_warmup_seconds"), 1)
    ):
        reasons.append("reference: insufficient timed work or warm-up")
    start, end = reference.get("timed_start_ns"), reference.get("timed_end_ns")
    if _number(start, 1) and _number(end, 1):
        seconds = (end - start) / 1e9
        if seconds < 1 or reference.get("actual_timed_seconds") != seconds:
            reasons.append("reference: duration differs from actual window")
        primary_end = primary.get("timed_end_ns")
        if not _number(primary_end, 1) or start < primary_end:
            reasons.append("reference: window overlaps or precedes primary work")
    clocks = [sample.get("sm_mhz") for sample in samples]
    if (
        not clocks
        or any(not _number(value, 1) for value in clocks)
        or max(clocks) > min(clocks) * (1 + CLOCK_SPREAD)
    ):
        reasons.append("primary/reference: missing clock or >5% combined clock spread")
    limits = [sample.get("power_limit_mw") for sample in samples]
    if not limits or any(not _number(value, 1) for value in limits) or len(set(limits)) != 1:
        reasons.append("primary/reference: missing or changed power limit")
    return {"passed": not reasons, "reasons": sorted(set(reasons))}
