#!/usr/bin/env python3
"""Host readings for the measurements: the RAM limit the run is under and the process's memory.

``ram_limit()`` reads the cgroup memory limit the run is under (v2 ``memory.max`` or v1
``memory.limit_in_bytes``), and ``process_memory()`` the process's resident bytes, split the
way gate-655 measured them: page-locked memory shows up as ``RssShmem``, pageable as
``RssAnon``. ``reset_peak_rss()`` resets the peak so a step's own peak can be read.

Benchmark harness code, not shipped. Standard library only (psutil off Linux, if present).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_text(path: str) -> Optional[str]:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def parse_status(text: str) -> Dict[str, int]:
    """Bytes of the ``Vm*``/``Rss*`` lines of /proc/<pid>/status."""
    out: Dict[str, int] = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        parts = value.split()
        if key.startswith(("Vm", "Rss")) and len(parts) == 2 and parts[1] == "kB":
            out[key] = int(parts[0]) * 1024
    return out


def _hierarchy_paths(proc_self_cgroup: str, *, v2: bool) -> Optional[List[str]]:
    """The cgroup path components of this process from /proc/self/cgroup text: the
    unified (v2, ``0::/path``) line, or the v1 line whose controllers include ``memory``
    (``4:memory:/path``, ``9:cpuset,memory:/path``). None when there is no such line."""
    for line in proc_self_cgroup.splitlines():
        hierarchy, _, rest = line.partition(":")
        controllers, _, path = rest.partition(":")
        if v2 and hierarchy == "0" and controllers == "":
            return [p for p in path.split("/") if p]
        if not v2 and "memory" in controllers.split(","):
            return [p for p in path.split("/") if p]
    return None


def _walk(parts: List[str], root: str) -> List[str]:
    return [root + "".join("/" + p for p in parts[:depth]) for depth in range(len(parts), -1, -1)]


def cgroup_dirs(proc_self_cgroup: str, root: str = "/sys/fs/cgroup") -> List[str]:
    """The cgroup v2 directories from this process's own up to the mount root, innermost
    first. ``systemd-run --scope -p MemoryMax=`` and ``docker --memory`` put the limit on
    the process's own cgroup or an ancestor, not necessarily on the mount root. Inside a
    container with a private cgroup namespace the path is ``/`` and the root is the
    container's own cgroup."""
    parts = _hierarchy_paths(proc_self_cgroup, v2=True)
    return _walk(parts or [], root)


def cgroup_v1_dirs(proc_self_cgroup: str, root: str = "/sys/fs/cgroup/memory") -> List[str]:
    """The cgroup v1 memory-controller directories, innermost first, as for v2. A container
    sees its host path here but has its own cgroup mounted at the root, so the inner
    directories do not exist there and the root is what is read."""
    parts = _hierarchy_paths(proc_self_cgroup, v2=False)
    return _walk(parts or [], root)


def _limit(raw: Optional[str]) -> Optional[int]:
    """A cgroup byte limit, or None for unlimited (``max``; v1's PAGE_COUNTER_MAX, which
    reads 9223372036854771712 on x86-64) or unreadable."""
    if raw is None or raw == "max" or not raw.isdigit() or int(raw) >= 1 << 60:
        return None
    return int(raw)


def _counter(raw: Optional[str]) -> Optional[int]:
    return int(raw) if raw and raw.isdigit() else None


def _tightest(values: List[Tuple[str, Optional[int]]]) -> Tuple[Optional[str], Optional[int]]:
    """``(source, bytes)`` of the smallest limit among (source, limit) pairs; None: unlimited."""
    found = [(limit, source) for source, limit in values if limit is not None]
    if not found:
        return None, None
    limit, source = min(found)
    return source, limit


def _swap_total(meminfo: Optional[str]) -> Optional[int]:
    for line in (meminfo or "").splitlines():
        key, _, value = line.partition(":")
        parts = value.split()
        if key == "SwapTotal" and len(parts) == 2 and parts[1] == "kB" and parts[0].isdigit():
            return int(parts[0]) * 1024
    return None


def ram_limit(*, proc_root: str = "/proc", v2_root: str = "/sys/fs/cgroup",
              v1_root: str = "/sys/fs/cgroup/memory") -> Dict[str, Any]:
    """The tightest cgroup memory limit over this process's cgroup and its ancestors: v2
    ``memory.max``, else the v1 memory controller's ``memory.limit_in_bytes``, with the
    usage and peak of the cgroup that sets it, and the swap the run may use on top.

    ``limit_bytes`` None means unlimited or unknown. Page-locked (pinned) host memory is
    charged to this limit as shmem (gate-655), so a pinned store counts against it.
    ``swap_max_bytes`` is the tightest v2 ``memory.swap.max`` (v1: ``memory.memsw.limit_in_
    bytes`` minus the memory limit of the same cgroup); None means the cgroup does not
    cap swap, and ``swap_total_bytes`` (/proc/meminfo ``SwapTotal``) says whether the box
    has any. Swap lets pageable memory leave RAM while RSS stays under the limit, so the
    runbook sets it to 0 (``MemorySwapMax=0``, ``--memory-swap`` = ``--memory``).
    The roots are parameters so the arithmetic can be tested on fixture trees.
    """
    cgroup_text = _read_text(f"{proc_root}/self/cgroup") or ""
    swap_total = _swap_total(_read_text(f"{proc_root}/meminfo"))
    unknown: Dict[str, Any] = {"source": None, "limit_bytes": None, "current_bytes": None,
                               "peak_bytes": None, "swap_source": None, "swap_max_bytes": None,
                               "swap_total_bytes": swap_total, "version": None}
    if _hierarchy_paths(cgroup_text, v2=True) is not None:
        dirs = cgroup_dirs(cgroup_text, v2_root)
        source, limit = _tightest([(f"{d}/memory.max", _limit(_read_text(f"{d}/memory.max")))
                                   for d in dirs])
        swap_source, swap = _tightest([(f"{d}/memory.swap.max",
                                        _limit(_read_text(f"{d}/memory.swap.max")))
                                       for d in dirs])
        if source is not None:
            directory = source.rsplit("/", 1)[0]
            return {**unknown, "version": 2, "source": source, "limit_bytes": limit,
                    "current_bytes": _counter(_read_text(f"{directory}/memory.current")),
                    "peak_bytes": _counter(_read_text(f"{directory}/memory.peak")),
                    "swap_source": swap_source, "swap_max_bytes": swap}
    rows = []
    for d in cgroup_v1_dirs(cgroup_text, v1_root):
        limit = _limit(_read_text(f"{d}/memory.limit_in_bytes"))
        if limit is not None:
            both = _limit(_read_text(f"{d}/memory.memsw.limit_in_bytes"))
            rows.append((limit, d, None if both is None else max(0, both - limit)))
    if not rows:
        return unknown
    limit, directory, swap = min(rows, key=lambda row: row[0])
    return {**unknown, "version": 1, "source": f"{directory}/memory.limit_in_bytes",
            "limit_bytes": limit,
            "current_bytes": _counter(_read_text(f"{directory}/memory.usage_in_bytes")),
            "peak_bytes": _counter(_read_text(f"{directory}/memory.max_usage_in_bytes")),
            "swap_source": None if swap is None else f"{directory}/memory.memsw.limit_in_bytes",
            "swap_max_bytes": swap}


def process_memory(*, proc_root: str = "/proc") -> Dict[str, Optional[int]]:
    """Resident bytes of this process: total, anonymous (pageable), shmem (page-locked
    CUDA host memory on Linux, gate-655), the peak (VmHWM) and swapped-out (VmSwap)."""
    if sys.platform.startswith("linux"):
        status = parse_status(_read_text(f"{proc_root}/self/status") or "")
        return {"rss": status.get("VmRSS"), "rss_anon": status.get("RssAnon"),
                "rss_shmem": status.get("RssShmem"), "peak_rss": status.get("VmHWM"),
                "locked": status.get("VmLck"), "swap": status.get("VmSwap")}
    try:
        import psutil

        info = psutil.Process().memory_info()
    except (ImportError, OSError):
        return {"rss": None, "rss_anon": None, "rss_shmem": None, "peak_rss": None,
                "locked": None, "swap": None}
    return {"rss": int(info.rss), "rss_anon": None, "rss_shmem": None,
            "peak_rss": int(getattr(info, "peak_wset", 0)) or None, "locked": None,
            "swap": None}


def reset_peak_rss() -> bool:
    """Reset VmHWM to the current RSS (Linux >= 4.0: write 5 to clear_refs)."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        with open("/proc/self/clear_refs", "w", encoding="ascii") as handle:
            handle.write("5")
        return True
    except OSError:
        return False
