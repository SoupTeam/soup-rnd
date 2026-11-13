"""B5 ticket 06: drop the page cache before a cold run, and count what is left.

``drop_page_cache`` flushes dirty pages, then empties the cache of the store's files and says
how: ``drop_caches`` (write 3 to ``/proc/sys/vm/drop_caches``, needs root, empties the cache of
the whole host) or ``fadvise`` (``POSIX_FADV_DONTNEED`` on each file, no root, clean pages of
these files only). When neither works the method is ``none`` and nothing was dropped. It
measures the residency of the store's files with ``mincore`` before and after, so a drop that did
not work shows in the numbers. The residency after the drop is what the acceptance rule reads.

``process_read_bytes`` is the ``read_bytes`` counter of ``/proc/self/io``: bytes this process had
the storage layer read. The experts are read with ``O_DIRECT`` and count there too.

Benchmark harness code, not shipped. No torch import.
"""

from __future__ import annotations

import ctypes
import mmap
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: The methods ``drop_page_cache`` reports.
METHODS = ("drop_caches", "fadvise", "none")
DROP_CACHES_PATH = "/proc/sys/vm/drop_caches"
#: Translate table that keeps only the lowest bit of each byte (the mincore residency flag).
_LOW_BIT_TABLE = bytes(value & 1 for value in range(256))


def _files(store: Path) -> List[Path]:
    return sorted(item for item in Path(store).rglob("*") if item.is_file())


def _file_residency(path: Path, libc: Any) -> Tuple[int, int]:
    size = path.stat().st_size
    if size == 0:
        return 0, 0
    page = mmap.PAGESIZE
    pages = -(-size // page)
    vector = ctypes.create_string_buffer(pages)
    with open(path, "rb") as handle:
        mapped = mmap.mmap(handle.fileno(), size, access=mmap.ACCESS_COPY)
        try:
            anchor = ctypes.c_char.from_buffer(mapped)
            try:
                libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
                if libc.mincore(ctypes.addressof(anchor), size, vector) != 0:
                    raise OSError(ctypes.get_errno(), "mincore failed")
            finally:
                del anchor
        finally:
            mapped.close()
    return min(size, vector.raw.translate(_LOW_BIT_TABLE).count(1) * page), size


def store_residency(store: Path) -> Tuple[int, int]:
    """(bytes of the store's files in the page cache, total bytes of those files)."""
    libc = ctypes.CDLL(None, use_errno=True)
    resident = total = 0
    for path in _files(store):
        file_resident, file_total = _file_residency(path, libc)
        resident += file_resident
        total += file_total
    return resident, total


def _fadvise_all(store: Path) -> bool:
    advise = getattr(os, "posix_fadvise", None)
    if advise is None:
        return False
    for path in _files(store):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            advise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(descriptor)
    return True


def drop_page_cache(store: Path) -> Dict[str, Any]:
    """Empty the page cache of ``store`` and report the method and the residency around it."""
    started = time.perf_counter()
    before, total = store_residency(store)
    method = "none"
    if hasattr(os, "sync"):
        os.sync()
    try:
        with open(DROP_CACHES_PATH, "w", encoding="ascii") as handle:
            handle.write("3")
        method = "drop_caches"
    except OSError:
        try:
            method = "fadvise" if _fadvise_all(store) else "none"
        except OSError:
            method = "none"
    after, _unused_total = store_residency(store)
    return {"method": method, "store_bytes": total, "resident_before_bytes": before,
            "resident_after_bytes": after, "seconds": time.perf_counter() - started}


def process_read_bytes(proc_root: str = "/proc") -> Optional[int]:
    """Bytes this process had read from the storage layer, or None where the counter is absent."""
    path = Path(proc_root) / "self" / "io"
    try:
        for line in path.read_text().splitlines():
            if line.startswith("read_bytes:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        return None
    return None
