"""C3: bounded, immutable decoder-layer residency beside the existing disk reader.

The budget covers owned tensor allocations and one mapped layer during loading.
It is not a process-RSS cap: the caller must reserve activations, other buffers,
Python/allocator overhead and safety headroom, including A4's live allocations.
"""

import os
from dataclasses import dataclass
from typing import Any


def _bytes(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be non-negative integer bytes")


@dataclass(frozen=True)
class PartialRamBudget:
    """Explicit shared-host envelope; cache_bytes is an allocation cap, not a target."""

    limit_bytes: int
    reserved_bytes: int
    cache_bytes: int

    def __post_init__(self) -> None:
        for name in ("limit_bytes", "reserved_bytes", "cache_bytes"):
            _bytes(getattr(self, name), name)
        if self.limit_bytes == 0 or self.reserved_bytes > self.limit_bytes:
            raise ValueError("RAM budget must be positive and cover all reservations")


class PartialRamSource:
    """One consumer, immutable whole-layer cache, unchanged async disk source.

    Only nonresident paths are handed to AsyncDiskSource, so lookahead cannot
    accidentally read cached layers. Global indices are mapped in both get and
    release. Cache loading reads each selected shard once before training.
    """

    def __init__(
        self,
        shard_dir: str,
        n_layers: int,
        spec: Any,
        *,
        budget: PartialRamBudget,
        pin: bool = True,
        shard_paths: Any = None,
        read_ahead: int = 2,
        layer_roots: Any = None,
        cacheable_layers: int | None = None,
        host_budget: Any = None,
    ) -> None:
        from soup_cli.utils.async_disk_source import AsyncDiskSource
        from soup_cli.utils.host_memory_budget import HostMemoryBudget
        from soup_cli.utils.layer_stream_runtime import RamSource, _spec_bytes, plan_pinned_arenas

        if not isinstance(budget, PartialRamBudget):
            raise TypeError("An explicit PartialRamBudget is required")
        if type(n_layers) is not int or n_layers < 1:
            raise ValueError("n_layers must be positive")
        count = n_layers if cacheable_layers is None else cacheable_layers
        if type(count) is not int or not 0 <= count <= n_layers:
            raise ValueError("cacheable_layers must name a decoder prefix")
        specs = RamSource._normalize_layer_specs(spec, n_layers)
        paths = RamSource._normalize_shard_paths(shard_dir, n_layers, shard_paths)
        roots = AsyncDiskSource._normalize_layer_roots(layer_roots, n_layers)
        sizes = [_spec_bytes(entry) for entry in specs]
        self._closed = False
        self._disk = None
        self._ram = None
        self._events: list[Any] = []
        self._lease = None
        self.n_layers = n_layers
        self.pinned = bool(pin)
        self.read_ahead = read_ahead
        self.budget = budget
        self.host_budget = host_budget or HostMemoryBudget(
            budget.limit_bytes, budget.reserved_bytes
        )
        if (
            not isinstance(self.host_budget, HostMemoryBudget)
            or self.host_budget.limit_bytes != budget.limit_bytes
            or self.host_budget.reserved_bytes != budget.reserved_bytes
        ):
            raise ValueError(
                "shared host budget must match the explicit limit and fixed reservation"
            )
        self.resident_tensor_bytes_served = 0
        chosen = []
        tensor_sizes: list[int] = []
        allocation = 0
        # Stable first-fit in decoder order. Whole layers only, including NF4 sidecars.
        for idx in range(count):
            candidate = tensor_sizes + [
                _spec_bytes({key: value}) for key, value in specs[idx].items()
            ]
            amount = plan_pinned_arenas(candidate).pinned_bytes if pin else sum(candidate)
            if sizes[idx] > 0 and amount <= budget.cache_bytes:
                chosen.append(idx)
                tensor_sizes = candidate
                allocation = amount
        self.resident_indices = tuple(chosen)
        self.disk_indices = tuple(idx for idx in range(n_layers) if idx not in chosen)
        self._ram_index = {idx: local for local, idx in enumerate(self.resident_indices)}
        self._disk_index = {idx: local for local, idx in enumerate(self.disk_indices)}
        self.resident_bytes = sum(sizes[idx] for idx in chosen)
        # safe_open can map the source while copying into the final RAM store.
        transient = max((os.path.getsize(paths[idx]) for idx in chosen), default=0)
        try:
            # Claim the currently available envelope atomically before any allocation;
            # shrink to owned buffers after loading. Other consumers cannot race the check.
            self._lease = self.host_budget.reserve("c3.weights", self.host_budget.available_bytes)
            remaining = self._lease.nbytes - allocation - transient
            if remaining < 0:
                raise ValueError(
                    "RAM budget cannot cover cache plus loading transient/reservations"
                )
            if self.disk_indices:
                self._disk = AsyncDiskSource(
                    shard_dir,
                    len(self.disk_indices),
                    [specs[idx] for idx in self.disk_indices],
                    shard_paths=[paths[idx] for idx in self.disk_indices],
                    pin=pin,
                    read_ahead=read_ahead,
                    layer_roots=[roots[idx] for idx in self.disk_indices],
                    max_host_bytes=remaining,
                )
            if chosen:
                self._ram = RamSource(
                    shard_dir,
                    len(chosen),
                    [specs[idx] for idx in chosen],
                    pin=pin,
                    shard_paths=[paths[idx] for idx in chosen],
                )
            disk_allocated = getattr(self._disk, "host_allocated_bytes", 0)
            self.host_allocated_bytes = allocation + disk_allocated
            self.accounted_peak_bytes = (
                self.host_budget.used_bytes
                - self._lease.nbytes
                + self.host_allocated_bytes
                + transient
            )
            self.pinned_bytes = getattr(self._ram, "pinned_bytes", 0) + getattr(
                self._disk, "pinned_bytes", 0
            )
            self.nbytes = self.resident_bytes + getattr(self._disk, "nbytes", 0)
            self.staging_bytes = getattr(self._disk, "staging_bytes", 0)
            self.disk_bytes = sum(sizes[idx] for idx in self.disk_indices)
            self._lease.resize(self.host_allocated_bytes)
        except BaseException:
            self.close()
            raise

    def _check(self, idx: int) -> None:
        if self._closed:
            raise RuntimeError("partial RAM source is closed")
        if type(idx) is not int:
            raise TypeError("layer index must be an integer")
        if not 0 <= idx < self.n_layers:
            raise IndexError(idx)

    def get(self, idx: int, name: str) -> Any:
        self._check(idx)
        if idx in self._ram_index:
            tensor = self._ram.get(self._ram_index[idx], name)
            self.resident_tensor_bytes_served += tensor.numel() * tensor.element_size()
            return tensor
        return self._disk.get(self._disk_index[idx], name)

    def release(self, idx: int, event: Any = None) -> None:
        self._check(idx)
        if idx in self._disk_index:
            self._disk.release(self._disk_index[idx], event)
        elif event is not None:
            # Keep pinned source storage alive until every outstanding H2D completes.
            self._events = [held for held in self._events if not held.query()]
            self._events.append(event)

    def close(self) -> None:
        if self._closed:
            return
        if self._disk is not None:
            self._disk.close()
        for event in self._events:
            event.synchronize()
        self._events.clear()
        self._ram = None
        self._disk = None
        self.nbytes = 0
        self.host_allocated_bytes = 0
        self.pinned_bytes = 0
        if self._lease is not None:
            self._lease.close()
        self._closed = True
