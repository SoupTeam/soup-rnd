"""Byte-bounded, immutable tensor entries for the experimental expert caches.

The bound counts tensor payload (``numel() * element_size()``), not allocator
arenas, Python objects, staging, or tensors still held by an evicted entry's
consumer. It is not a whole-process RAM or VRAM cap. Admission against those
resources belongs to the caller. This cache is not thread-safe.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


class TensorLRU:
    """An LRU of complete named tensor sets under a payload-byte ceiling.

    ``copy_on_insert=True`` clones every tensor off reusable host staging.
    ``False`` retains the supplied tensors by reference; the caller must supply
    owning tensors whose storage will never be overwritten. Stored tensors and
    tensors returned by :meth:`get` must be treated as immutable in both modes.
    The returned mapping itself is read-only.

    Every unsuccessful ``get`` counts as a miss, including with a zero budget.
    A zero budget disables insertion. ``clear`` releases entries and payload
    accounting, retaining the lifetime hit/miss/eviction counters.
    """

    def __init__(self, budget_bytes: int, *, copy_on_insert: bool) -> None:
        if type(budget_bytes) is not int or budget_bytes < 0:
            raise ValueError("budget_bytes must be a non-negative integer byte ceiling")
        if type(copy_on_insert) is not bool:
            raise ValueError("copy_on_insert must be a bool")
        self._budget_bytes = budget_bytes
        self.copy_on_insert = copy_on_insert
        self._entries: OrderedDict[Any, tuple[Mapping[str, Any], int]] = OrderedDict()
        self._payload_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: Any) -> Mapping[str, Any] | None:
        """Return the complete entry and refresh recency, or count one miss."""
        entry = self._entries.get(key)
        if entry is None:
            self._misses += 1
            return None
        self._entries.move_to_end(key)
        self._hits += 1
        return entry[0]

    def put(self, key: Any, tensors: Mapping[str, Any]) -> bool:
        """Admit one complete entry, returning False when its payload cannot fit.

        Oversized entries bypass without tensor clones or eviction. Otherwise,
        entries are removed *before* cloning so cache-owned payload never rises
        above the ceiling during insertion. If a clone fails, no partial entry
        is published; entries already removed stay evicted. Replacing a key
        removes its previous value but does not count a capacity eviction.
        """
        if self._budget_bytes == 0:
            return False
        import torch

        hash(key)
        if not isinstance(tensors, Mapping):
            raise TypeError("an expert cache entry must be a mapping of named tensors")
        items = tuple(tensors.items())
        for name, tensor in items:
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                raise TypeError("an expert cache entry must map string names to tensors")
        payload = sum(int(tensor.numel() * tensor.element_size()) for _, tensor in items)
        if payload > self._budget_bytes:
            return False
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._payload_bytes -= previous[1]
            del previous
        while self._payload_bytes + payload > self._budget_bytes:
            removed = self._entries.popitem(last=False)[1]
            self._payload_bytes -= removed[1]
            self._evictions += 1
            del removed
        if self.copy_on_insert:
            owned = {name: tensor.detach().clone() for name, tensor in items}
        else:
            owned = dict(items)
        self._entries[key] = (MappingProxyType(owned), payload)
        self._payload_bytes += payload
        return True

    def clear(self) -> None:
        """Release the cache's references, preserving lifetime lookup counters."""
        self._entries.clear()
        self._payload_bytes = 0

    def discard(self, key: Any) -> None:
        """Release one entry, if present, without changing any lookup counters."""
        removed = self._entries.pop(key, None)
        if removed is not None:
            self._payload_bytes -= removed[1]

    def stats(self) -> dict[str, int]:
        """A snapshot of payload accounting and lifetime lookup/eviction counts."""
        return {
            "budget_bytes": self._budget_bytes,
            "payload_bytes": self._payload_bytes,
            "entries": len(self._entries),
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
        }
