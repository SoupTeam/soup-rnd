"""Setup-only admission of immutable expert caches against measured whole-model peaks.

The ceilings are payload limits, not predictions of a useful working set or speed.
Both probes run a complete forward/backward without stepping an optimizer. The
candidate gate adds the *full* device ceiling even when the probe already warmed
some cache entries. This intentional double charge protects cold or different
expert routes from spending unmeasured VRAM later.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from soup_cli.utils.stream_ram_cache import available_memory, cache_budget

EXPERT_CACHE_RESERVE_BYTES = 64 * 2**20


@dataclass(frozen=True)
class ExpertCachePlan:
    host_cache_bytes: int = 0
    device_cache_bytes: int = 0
    host_available_bytes: int = 0
    host_cap_bytes: int = 0
    host_reserve_bytes: int = 0
    cap_bytes: int = 0
    device_available_bytes: int = 0
    baseline_peak_allocated_bytes: int = 0
    baseline_peak_reserved_bytes: int = 0
    candidate_peak_allocated_bytes: int = 0
    candidate_peak_reserved_bytes: int = 0
    reserve_bytes: int = 0
    optimizer_reserve_bytes: int = 0
    setup_seconds: float = 0.0
    reason: str = "disabled"

    @property
    def enabled(self) -> bool:
        return bool(self.host_cache_bytes or self.device_cache_bytes)


def expert_device_reserve_bytes(runtime: Any) -> int:
    """Hold back the full lazy device ceiling from other CUDA capacity planners."""
    if not getattr(getattr(runtime, "pool", None), "is_cuda", False):
        return 0
    fetcher = getattr(runtime, "expert_fetcher", None)
    if fetcher is None:
        return 0
    value = fetcher.cache_stats().get("device", {}).get("budget_bytes", 0)
    if type(value) is not int or value < 0:
        raise ValueError("expert device cache ceiling must be a non-negative integer")
    return value


def configure_stream_expert_cache(
    model: Any,
    runtime: Any,
    *,
    host_cache_bytes: int = 0,
    device_cache_bytes: int = 0,
    rows: int,
    seq_len: int,
    vocab_size: int,
    override_bytes: Optional[int] = None,
    reserve_bytes: int = EXPERT_CACHE_RESERVE_BYTES,
    optimizer_reserve_bytes: Optional[int] = None,
    memory: Any = None,
    measure: Optional[Callable[..., Any]] = None,
    host_available: Optional[int] = None,
) -> ExpertCachePlan:
    """Admit explicit ceilings, then verify them with a full candidate step.

    Production use requires CUDA. A CPU target is a correctness control only
    when both memory and measurement instruments are supplied explicitly; its
    host and device cache ceilings are both charged against host RAM. Existing
    decoder pools, source hotsets and expert staging are present in the baseline
    and in the subsequent live host-availability sample. Optimizer state is
    conservatively estimated at twelve bytes per trainable parameter.

    This is a pristine setup operation: existing expert budgets or entries and
    accumulated gradients are refused. Failed admission leaves both budgets at
    zero; non-OOM probe errors propagate with their original meaning.
    """
    for name, value in (
        ("host_cache_bytes", host_cache_bytes), ("device_cache_bytes", device_cache_bytes),
        ("reserve_bytes", reserve_bytes),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if optimizer_reserve_bytes is not None and (
        type(optimizer_reserve_bytes) is not int or optimizer_reserve_bytes < 0
    ):
        raise ValueError("optimizer_reserve_bytes must be a non-negative integer or None")
    if override_bytes is not None and (type(override_bytes) is not int or override_bytes < 1):
        raise ValueError("override_bytes must be a positive integer or None")
    if host_available is not None and (type(host_available) is not int or host_available < 0):
        raise ValueError("host_available must be a non-negative integer or None")
    if not (host_cache_bytes or device_cache_bytes):
        return ExpertCachePlan()
    if any(type(value) is not int or value < 1 for value in (rows, seq_len, vocab_size)):
        raise ValueError("expert cache planning needs positive batch/context/vocabulary sizes")

    started = time.perf_counter()
    pool = runtime.pool
    fetcher = getattr(runtime, "expert_fetcher", None)
    if fetcher is None or not getattr(runtime, "expert_readers", ()):
        raise ValueError("expert cache planning requires per-expert streaming")
    if not getattr(pool, "is_cuda", False) and (memory is None or measure is None):
        raise ValueError("expert cache planning requires a CUDA pool")
    initial = fetcher.cache_stats()
    if any(
        initial.get(tier, {}).get(field, 0)
        for tier in ("host", "device") for field in ("budget_bytes", "payload_bytes")
    ):
        raise ValueError("expert cache planning requires disabled, empty expert caches")

    import torch

    from soup_cli.utils.layer_stream_runtime import (
        _is_out_of_memory,
        clear_stream_substitution_cache,
        measure_step_peak_bytes,
    )
    from soup_cli.utils.stream_residency import (
        OPTIMIZER_RESERVE_BYTES_PER_PARAM,
        _CudaMemory,
        _measure_preserving_state,
        residency_budget,
    )

    target_type = torch.device(pool.device).type
    if target_type not in {"cuda", "cpu"}:
        raise ValueError("expert cache planning requires CUDA or an injected CPU control")
    target_is_cpu = target_type == "cpu"
    if target_is_cpu and (memory is None or measure is None):
        raise ValueError("CPU expert cache controls require memory and measure instruments")
    memory = _CudaMemory(pool.device) if memory is None else memory
    measure = measure_step_peak_bytes if measure is None else measure
    if optimizer_reserve_bytes is None:
        optimizer_reserve_bytes = OPTIMIZER_RESERVE_BYTES_PER_PARAM * sum(
            int(param.numel()) for param in model.parameters() if param.requires_grad
        )

    def probe(label: str) -> Any:
        peak = _measure_preserving_state(
            model, measure, rows=rows, seq_len=seq_len, vocab_size=vocab_size,
            device=pool.device,
        )
        if peak is None:
            raise ValueError(f"expert cache {label} peak could not be measured")
        if getattr(peak, "failed", False):
            raise RuntimeError(f"expert cache {label} probe failed: {peak.error}")
        if getattr(peak, "oom", False):
            raise ValueError(f"expert cache {label} probe ran out of memory")
        if any(
            type(value) is not int or value < 0
            for value in (peak.peak_bytes, peak.reserved_bytes)
        ):
            raise ValueError(f"expert cache {label} returned invalid peak byte counts")
        return peak

    def rollback() -> None:
        try:
            memory.synchronize()
        finally:
            fetcher.configure_cache(host_cache_bytes=0, device_cache_bytes=0)
            clear_stream_substitution_cache(model)
            memory.release()

    try:
        baseline = probe("baseline")
        memory.synchronize()
        budget = residency_budget(
            total_bytes=memory.total(), free_bytes=memory.free(), reserved_bytes=memory.reserved(),
            peak_reserved_bytes=baseline.reserved_bytes, reserve_bytes=reserve_bytes,
            optimizer_reserve_bytes=optimizer_reserve_bytes, override_bytes=override_bytes,
        )
        baseline_required = baseline.reserved_bytes + reserve_bytes + optimizer_reserve_bytes
        if baseline_required > budget.cap_bytes:
            raise ValueError("expert cache baseline exceeds the shared cap including reserves")
        if device_cache_bytes > budget.available_bytes:
            raise ValueError("full expert device cache ceiling exceeds the measured spare budget")

        host_report = available_memory(host_available=host_available)
        host_available_bytes = int(host_report["available_bytes"])
        # On a CPU correctness control the dense tier consumes the same RAM as
        # the packed tier. Existing staging and source arenas are already live.
        host_charge = host_cache_bytes + (device_cache_bytes if target_is_cpu else 0)
        host_cap = cache_budget(host_charge, host_available_bytes)
        if host_charge > host_cap:
            raise ValueError("full expert host cache ceiling exceeds available RAM admission")
        clear_stream_substitution_cache(model)
        fetcher.configure_cache(
            host_cache_bytes=host_cache_bytes, device_cache_bytes=device_cache_bytes,
        )
        candidate = probe("candidate")
        memory.synchronize()
        # Another process can spend device memory while the candidate runs.
        # Resample live capacity, retaining the baseline cap as an upper bound.
        current_budget = residency_budget(
            total_bytes=memory.total(), free_bytes=memory.free(), reserved_bytes=memory.reserved(),
            peak_reserved_bytes=candidate.reserved_bytes, reserve_bytes=reserve_bytes,
            optimizer_reserve_bytes=optimizer_reserve_bytes,
            override_bytes=budget.cap_bytes if budget.cap_bytes else override_bytes,
        )
        cap_bytes = min(budget.cap_bytes, current_budget.cap_bytes)
        # Add the FULL lazy ceiling, including entries that this very probe
        # already warmed. A different route must fit without borrowing slack.
        required = (
            candidate.reserved_bytes + reserve_bytes + optimizer_reserve_bytes
            + device_cache_bytes
        )
        if required > cap_bytes:
            raise ValueError("expert cache candidate and full device ceiling exceed the shared cap")
    except Exception as exc:
        try:
            rollback()
        except Exception as cleanup_error:
            # A poisoned CUDA context can fail again while synchronizing, but
            # rollback's finally still removes both budgets. Keep the original
            # probe error rather than reclassifying it as that cleanup failure.
            exc.add_note(f"expert cache cleanup also failed: {cleanup_error!r}")
        if _is_out_of_memory(exc):
            raise ValueError("expert cache allocation/probe ran out of memory") from exc
        raise

    source = getattr(runtime, "source", None)
    if callable(getattr(source, "configure_ram_cache", None)):
        # A later SSD hotset rebuild must hold back this full lazy host growth.
        # Warm entries can therefore be conservatively charged twice there too.
        source._expert_host_cache_reserve_bytes = host_charge
    return ExpertCachePlan(
        host_cache_bytes=host_cache_bytes, device_cache_bytes=device_cache_bytes,
        host_available_bytes=host_available_bytes, host_cap_bytes=host_cap,
        host_reserve_bytes=host_charge, cap_bytes=cap_bytes,
        device_available_bytes=current_budget.available_bytes,
        baseline_peak_allocated_bytes=baseline.peak_bytes,
        baseline_peak_reserved_bytes=baseline.reserved_bytes,
        candidate_peak_allocated_bytes=candidate.peak_bytes,
        candidate_peak_reserved_bytes=candidate.reserved_bytes,
        reserve_bytes=reserve_bytes, optimizer_reserve_bytes=optimizer_reserve_bytes,
        setup_seconds=time.perf_counter() - started,
        reason="measured expert cache candidate and full ceilings fit the shared caps",
    )
