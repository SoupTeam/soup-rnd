"""Experimental source adapter for C2 timing; never installed by shipped Soup.

Only pending-read admission/bookkeeping changes. Allocation, workers, direct I/O,
consumer ownership, release/drain events, completion and teardown stay inherited.
"""
import hashlib
import time
from pathlib import Path
from typing import Any

from soup_cli.utils import async_disk_source as shipped
from soup_cli.utils.async_disk_source import AsyncDiskSource, _PendingRead

SHIPPED_SHA256 = "17bca116e3f5935d4d916e3f3fde425a9544ca22890cda713b7892b56b1451df"
# Reviewed RAM/GPU-cache source, in addition to the original measured baseline.
# Historical timing evidence remains bound to its original source/adapter hashes.
RAM_CACHE_SHIPPED_SHA256 = "3c9b81077924a2b75a5ea07dfe6af785d25c554b527e27d7c1a7f8e0612d92f7"


def depth_source(depth: int) -> type:
    """Create an explicitly named benchmark-only source at depth 1 or 2 per root."""
    if type(depth) is not int or depth not in (1, 2):
        raise ValueError("experimental depth must be 1 or 2")
    actual = hashlib.sha256(Path(shipped.__file__).read_text(encoding="utf-8").encode()).hexdigest()
    if actual not in (SHIPPED_SHA256, RAM_CACHE_SHIPPED_SHA256):
        raise RuntimeError("shipped source drifted; review the C2 adapter before measuring")

    class C2DepthSource(AsyncDiskSource):
        experimental_read_depth = depth
        last_admission = None

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            type(self).last_admission = {
                "read_depth": depth, "read_ahead": self.read_ahead,
                "read_ranges": self.read_ranges, "staging_bytes": self.staging_bytes,
                "pinned": self.pinned, "direct_io": self.direct_io,
                "workers_per_root": {str(root): len(pool._threads)
                                     for root, pool in self._pools.items()},
            }

        def _claim_reads(self) -> list[tuple[_PendingRead, Any]]:
            if depth == 1:
                return super()._claim_reads()
            # Unlike the shipped map's root keys, experimental keys are layer IDs.
            # Every inherited consumer of this map uses values, except pool checks,
            # which are overridden below to recover the physical placement root.
            started = []
            position = 0
            while position < len(self._queue):
                idx = self._queue[position]
                if (idx in getattr(self, "_cache", ()) or idx in getattr(self, "_resident", ())
                        or idx in self._slot_of or idx in self._in_flight_batch):
                    self._queue.pop(position)
                    continue
                root = self._root_of[idx]
                count = sum(self._root_of[read.idx] == root for read in self._pending.values())
                if count >= depth:
                    position += 1
                    continue
                busy = frozenset(read.slot for read in self._pending.values())
                slot = self._claim_slot(idx, exclude=busy)
                if slot is None:
                    break
                self._queue.pop(position)
                draining = self._drain[slot]
                self._drain[slot] = None
                read = _PendingRead(
                    idx=idx, slot=slot, region=self._regions[slot], started_at=time.monotonic()
                )
                self._pending[idx] = read
                self._sync_in_flight()
                started.append((read, draining))
            return started

        def _check_pools(self) -> None:
            if depth == 1:
                return super()._check_pools()
            for read in self._pending.values():
                if read.outcomes is not None and not read.done():
                    self._pools[self._root_of[read.idx]]._check_workers()

    C2DepthSource.__name__ = f"C2Depth{depth}Source"
    return C2DepthSource
