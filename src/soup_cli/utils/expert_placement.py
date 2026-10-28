"""Bounded, setup-only expert placement from observed logical requests.

This experimental greedy policy ranks frequency * packed-transfer bytes per
dense-GPU byte. That is a traffic proxy, not measured latency or an optimal
knapsack solution. It allocates no tensors and never reads or mutates a cache.
The caller admits the GPU ceiling against complete training-memory peaks.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from fractions import Fraction


def _integer(name: str, value: int, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class ExpertUsageRecord:
    key: Hashable
    requests: int
    transfer_bytes: int
    dense_bytes: int


@dataclass(frozen=True)
class ExpertPlacementPlan:
    keys: tuple[Hashable, ...]
    payload_bytes: int
    budget_bytes: int
    candidate_count: int
    total_requests: int
    selected_requests: int
    saved_transfer_bytes_proxy: int
    policy: str = "frequency_transfer_bytes_per_dense_byte"
    _keyset: frozenset[Hashable] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _integer("budget_bytes", self.budget_bytes, 0)
        _integer("payload_bytes", self.payload_bytes, 0)
        if type(self.keys) is not tuple or self.payload_bytes > self.budget_bytes:
            raise ValueError("placement needs immutable keys within the exact payload budget")
        members = frozenset(self.keys)
        if len(members) != len(self.keys):
            raise ValueError("placement must not contain duplicate keys")
        object.__setattr__(self, "_keyset", members)

    def contains(self, key: Hashable) -> bool:
        """An O(1) immutable membership check; no admission or recency changes."""
        return key in self._keyset


class ExpertUsageProfile:
    """Observe actual requests, never speculative prefetches or cache probes.

    The fetcher calls :meth:`record_request` once per real logical request,
    including RAM/GPU hits. ``count`` restores an already-observed aggregated
    trace; it is not a prediction. Keys must be immutable hashable logical IDs,
    such as ``(realpath, expert_index)``. Metadata must stay fixed per key.

    Snapshot records and returned plans are immutable. Equal ranking scores use
    first-observed order, so a fixed ordered trace produces a stable plan.
    The metadata entry ceiling fails explicitly instead of dropping observations.
    This observer is sequential and is not thread-safe.
    """

    def __init__(self, max_entries: int = 65536) -> None:
        _integer("max_entries", max_entries, 1)
        self._max_entries = max_entries
        self._records: dict[Hashable, ExpertUsageRecord] = {}
        self._total_requests = 0

    def record_request(
        self, key: Hashable, *, transfer_bytes: int, dense_bytes: int, count: int = 1,
    ) -> None:
        """Record real demand only; every refusal leaves the profile unchanged."""
        hash(key)
        _integer("count", count, 1)
        _integer("transfer_bytes", transfer_bytes, 1)
        _integer("dense_bytes", dense_bytes, 1)
        previous = self._records.get(key)
        if previous is not None:
            if (previous.transfer_bytes, previous.dense_bytes) != (transfer_bytes, dense_bytes):
                raise ValueError("expert byte metadata changed within one usage profile")
            requests = previous.requests + count
        else:
            if len(self._records) >= self._max_entries:
                raise OverflowError("expert usage profile reached its metadata entry ceiling")
            requests = count
        self._records[key] = ExpertUsageRecord(key, requests, transfer_bytes, dense_bytes)
        self._total_requests += count

    def snapshot(self) -> tuple[ExpertUsageRecord, ...]:
        """Frozen, portable records in first-observed order, with no live dict exposed."""
        return tuple(self._records.values())

    def stats(self) -> dict[str, int]:
        return {"entries": len(self._records), "max_entries": self._max_entries,
                "total_requests": self._total_requests}

    def plan(self, device_budget_bytes: int) -> ExpertPlacementPlan:
        """Greedily fit whole experts; skip oversized entries and keep considering later ones.

        Exact rational ranking avoids float ties and rounding of large byte/count
        values. Metadata/Python overhead is not charged as GPU tensor payload.
        Planning does not consume, clear, or update observations or any cache.
        """
        _integer("device_budget_bytes", device_budget_bytes, 0)
        ranked = sorted(enumerate(self._records.values()), key=lambda item: (
            -Fraction(item[1].requests * item[1].transfer_bytes, item[1].dense_bytes), item[0],
        ))
        chosen, used, requests, saved = [], 0, 0, 0
        for _position, record in ranked:
            if record.dense_bytes > device_budget_bytes - used:
                continue
            chosen.append(record.key)
            used += record.dense_bytes
            requests += record.requests
            saved += record.requests * record.transfer_bytes
        return ExpertPlacementPlan(
            keys=tuple(chosen), payload_bytes=used, budget_bytes=device_budget_bytes,
            candidate_count=len(self._records), total_requests=self._total_requests,
            selected_requests=requests, saved_transfer_bytes_proxy=saved,
        )
