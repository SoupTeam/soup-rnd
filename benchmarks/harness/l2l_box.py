#!/usr/bin/env python3
"""Box stamps for the L2L probe, and a watch for sleep and battery.

The probe runs beside other Claude sessions on a laptop. Every block records the
memory, power and GPU state it ran in (rule V3), and every arm runs under a
``SuspendWatch`` that samples the wall clock and the AC line every 2 s (rule
V5): a gap over 30 s means the box slept, and a battery sample means the clocks
were not the ones the gates assume. On 2026-10-01 a peer's probe died exactly
this way when the laptop was unplugged.

Rule V6 is the foreign reader: R2's attempt 1 lost an arm to SearchIndexer.exe
reading 4.94 GB from the drive under test. ``process_read_bytes`` snapshots every
process's read counter (psutil; ``None`` without it) and ``foreign_readers``
names the other processes that read ``threshold`` bytes or more between two
snapshots.

Benchmark harness code, not shipped. Standard library only.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Collection, Dict, List, Optional

_GB = 1e9
FOREIGN_READ_BYTES = 1_000_000_000
_GPU_OWNER_PID: Optional[int] = None
_GPU_OWNER_EVIDENCE: Optional[dict] = None


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_ubyte),
        ("BatteryFlag", ctypes.c_ubyte),
        ("BatteryLifePercent", ctypes.c_ubyte),
        ("SystemStatusFlag", ctypes.c_ubyte),
        ("BatteryLifeTime", ctypes.c_ulong),
        ("BatteryFullLifeTime", ctypes.c_ulong),
    ]


def linux_memory_status(text: str) -> Dict[str, Optional[float]]:
    """Parse /proc/meminfo. Linux commit accounting is not Windows page-lock charge."""
    values = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        parts = value.split()
        if parts and parts[0].isdigit():
            values[key] = int(parts[0]) * 1024
    return {
        "total_phys_gb": values["MemTotal"] / _GB if "MemTotal" in values else None,
        "avail_phys_gb": values["MemAvailable"] / _GB if "MemAvailable" in values else None,
        "commit_limit_gb": None,
        "commit_avail_gb": None,
    }


def memory_status() -> Dict[str, Optional[float]]:
    keys = ("total_phys_gb", "avail_phys_gb", "commit_limit_gb", "commit_avail_gb")
    if sys.platform == "linux":
        try:
            result = linux_memory_status(Path("/proc/meminfo").read_text())
            # A container can see host RAM in /proc. Honour exposed cgroup limits too.
            for limit_path, used_path, cache_key in (
                ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current", "inactive_file"),
                ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
                 "/sys/fs/cgroup/memory/memory.usage_in_bytes", "total_inactive_file"),
            ):
                try:
                    limit = int(Path(limit_path).read_text().strip())
                    used = int(Path(used_path).read_text().strip())
                except (OSError, ValueError):
                    continue
                # Like Docker's working set, exclude only inactive file cache.
                # It can be reclaimed for pinned allocations; treating all charged
                # page cache as resident would reject a freshly generated fixture.
                inactive_file = 0
                try:
                    stat = Path(limit_path).parent / "memory.stat"
                    counters = dict(line.split() for line in stat.read_text().splitlines())
                    inactive_file = max(0, int(counters.get(cache_key, "0")))
                except (OSError, ValueError):
                    pass  # Missing cache details retain the conservative estimate.
                working_set = max(0, used - inactive_file)
                if result["total_phys_gb"] is not None:
                    result["total_phys_gb"] = min(result["total_phys_gb"], limit / _GB)
                if result["avail_phys_gb"] is not None:
                    result["avail_phys_gb"] = min(
                        result["avail_phys_gb"], max(0, limit - working_set) / _GB)
            return result
        except OSError:
            return dict.fromkeys(keys)
    if sys.platform != "win32":
        return dict.fromkeys(keys)
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return dict.fromkeys(keys)
    return {
        "total_phys_gb": status.ullTotalPhys / _GB,
        "avail_phys_gb": status.ullAvailPhys / _GB,
        "commit_limit_gb": status.ullTotalPageFile / _GB,
        "commit_avail_gb": status.ullAvailPageFile / _GB,
    }


def on_ac_power() -> Optional[bool]:
    if sys.platform == "linux":
        online_values = []
        discharging = False
        try:
            for supply in Path("/sys/class/power_supply").iterdir():
                online = supply / "online"
                if online.exists():
                    value = online.read_text().strip()
                    if value in ("0", "1"):
                        online_values.append(value == "1")
                status = supply / "status"
                if status.exists() and status.read_text().strip() == "Discharging":
                    discharging = True
        except OSError:
            pass
        if discharging:
            return False
        if online_values:
            return any(online_values)
        # A cloud server has no AC sensor. This is an explicit operator declaration,
        # stamped in every arm; it is never inferred from a missing sensor.
        if os.environ.get("SOUP_L2L_POWER_SOURCE") == "server":
            return True
        return None
    if sys.platform != "win32":
        return None
    status = _SystemPowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return None
    return {0: False, 1: True}.get(int(status.ACLineStatus))


def _nvidia_smi(query: List[str]) -> Optional[str]:
    tool = shutil.which("nvidia-smi")
    if tool is None:
        return None
    try:
        out = subprocess.run(
            [tool, *query],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def gpu_compute_pids() -> Optional[List[int]]:
    text = _nvidia_smi(["--query-compute-apps=pid", "--format=csv,noheader"])
    if text is None:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if any(not line.isdigit() for line in lines):
        return None
    return [int(line) for line in lines]


def gpu_memory_used_mib() -> Optional[int]:
    text = _nvidia_smi(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    if not text or not text.strip().splitlines()[0].strip().isdigit():
        return None
    return int(text.strip().splitlines()[0].strip())


def own_process_ids() -> List[int]:
    """Include an exposed host PID: nvidia-smi can report host PIDs inside containers."""
    pids = {os.getpid()}
    if sys.platform == "linux":
        try:
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("NSpid:"):
                    pids.update(int(value) for value in line.split()[1:] if value.isdigit())
        except OSError:
            pass
    return sorted(pids)


def confirm_gpu_owner(before: Optional[List[int]], *, context_active: bool,
                      single_device: bool) -> Dict[str, object]:
    """Attribute a host PID only across an exclusive local CUDA initialization.

    NVIDIA reports host PIDs that a container's /proc may not expose. A confirmed
    live local context and exactly one reported compute process on the sole visible
    GPU identify that process. Unknown/busy starts or multiple processes never do.
    This alias applies only to GPU telemetry, not namespace-local I/O counters.
    """
    global _GPU_OWNER_PID, _GPU_OWNER_EVIDENCE
    _GPU_OWNER_PID = None
    after = gpu_compute_pids()
    evidence = {"before": before, "after": after, "context_active": context_active,
                "single_device": single_device, "method": "unresolved"}
    if (sys.platform == "linux" and before == [] and context_active and single_device
            and after is not None and len(after) == 1):
        _GPU_OWNER_PID = after[0]
        evidence["method"] = "exclusive-local-CUDA-initialization"
    _GPU_OWNER_EVIDENCE = evidence
    return evidence


def own_gpu_process_ids() -> List[int]:
    pids = set(own_process_ids())
    if _GPU_OWNER_PID is not None:
        pids.add(_GPU_OWNER_PID)
    return sorted(pids)


def python_process_count() -> Optional[int]:
    if sys.platform != "win32":
        return None
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",  # tasklist prints in the console's OEM code page
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return sum(1 for line in out.stdout.splitlines() if line.startswith('"python.exe"'))


def box_stamp() -> Dict[str, object]:
    return {
        "pid": os.getpid(),
        "own_pids": own_process_ids(),
        "gpu_own_pids": own_gpu_process_ids(),
        "gpu_pid_attribution": _GPU_OWNER_EVIDENCE,
        "unix": time.time(),
        **memory_status(),
        "ac": on_ac_power(),
        "gpu_pids": gpu_compute_pids(),
        "gpu_mem_used_mib": gpu_memory_used_mib(),
        "python_processes": python_process_count(),
        "power_source": ("operator-declared-server"
                         if os.environ.get("SOUP_L2L_POWER_SOURCE") == "server" else "sensor"),
    }


class SuspendWatch:
    """Samples (wall clock, AC line) every ``interval`` s on a daemon thread."""

    def __init__(
        self,
        interval: float = 2.0,
        gap_limit: float = 30.0,
        clock: Callable[[], float] = time.time,
        power: Callable[[], Optional[bool]] = on_ac_power,
    ):
        self.interval = float(interval)
        self.gap_limit = float(gap_limit)
        self._clock = clock
        self._power = power
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last: Optional[float] = None
        self.samples = 0
        self.max_gap_s = 0.0
        self.battery_samples = 0
        self.unknown_power_samples = 0

    def sample(self) -> None:
        now = self._clock()
        ac = self._power()
        with self._lock:
            if self._last is not None:
                self.max_gap_s = max(self.max_gap_s, now - self._last)
            self._last = now
            self.samples += 1
            if ac is False:
                self.battery_samples += 1
            elif ac is None:
                self.unknown_power_samples += 1

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample()

    def start(self) -> None:
        self.sample()
        self._thread = threading.Thread(target=self._loop, name="l2l-suspend-watch", daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, object]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self.sample()
        return self.report()

    def report(self) -> Dict[str, object]:
        with self._lock:
            return {
                "samples": self.samples,
                "max_gap_s": self.max_gap_s,
                "battery_samples": self.battery_samples,
                "unknown_power_samples": self.unknown_power_samples,
                "void": self.max_gap_s > self.gap_limit or self.battery_samples > 0,
            }


def process_read_bytes() -> Optional[Dict[int, Dict[str, object]]]:
    """``{pid: {"name", "read"}}`` for every process whose counters can be read."""
    try:
        import psutil
    except ImportError:
        return None
    out: Dict[int, Dict[str, object]] = {}
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            counters = proc.io_counters()
        except (psutil.Error, OSError):
            continue
        out[int(proc.info["pid"])] = {"name": proc.info["name"], "read": int(counters.read_bytes)}
    return out if os.getpid() in out else None


def foreign_readers(
    before: Optional[Dict[int, Dict[str, object]]],
    after: Optional[Dict[int, Dict[str, object]]],
    *,
    own_pids: Collection[int],
    threshold: int = FOREIGN_READ_BYTES,
) -> Optional[List[Dict[str, object]]]:
    """Other processes that read ``threshold`` bytes or more between two snapshots,
    heaviest first. A process absent from ``before`` counts its whole counter.
    ``None`` when either snapshot is missing: unknown, not clean."""
    if before is None or after is None:
        return None
    heavy = []
    for pid, row in after.items():
        if pid in own_pids:
            continue
        read = int(row["read"]) - int((before.get(pid) or {"read": 0})["read"])
        if read >= threshold:
            heavy.append({"pid": pid, "name": row["name"], "read_bytes": read})
    return sorted(heavy, key=lambda item: item["read_bytes"], reverse=True)
