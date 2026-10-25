#!/usr/bin/env python3
"""A4 box readings: what the spill volume is, and how much host RAM the run may use.

``spill_volume(path)`` says which filesystem and drive a spill directory sits
on, and refuses the ones a spill must not use:

* ``tmpfs``/``ramfs``: RAM again, so a "spill" frees nothing;
* ``overlay`` (Vast-style container roots) and other stacked filesystems:
  what drive the bytes reach, and whether ``O_DIRECT`` reaches it, depends on
  the layer below; a container must spill to a bind-mounted local volume;
* network filesystems (NFS, SMB/CIFS, 9p, virtiofs, FUSE, Ceph, ...): a cloud
  network volume is not a local SSD; refused unless the caller labels the run
  as network-volume scope, which keeps it out of the local-NVMe verdict;
* a volume where an aligned unbuffered write and read round trip fails.

A virtual block device (QEMU/virtio, model strings like ``QEMU HARDDISK``) is
not refused: it is labelled ``virtual`` and the record says that the guest's
disk, not a physical NVMe, was measured.

``ram_limit()`` reads the cgroup memory limit the run is under (v2
``memory.max`` or v1 ``memory.limit_in_bytes``), and ``process_memory()`` the
process's resident bytes, split the way gate-655 measured them: page-locked
memory shows up as ``RssShmem``, pageable as ``RssAnon``.

Benchmark harness code, not shipped. Standard library only at the top level.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REFUSED_FS = {
    "tmpfs": "tmpfs is RAM: a spill there frees no memory",
    "ramfs": "ramfs is RAM: a spill there frees no memory",
    "devtmpfs": "devtmpfs is RAM",
    "overlay": "overlayfs: the drive below and its O_DIRECT path are the upper layer's, "
               "not this mount's; spill to a bind-mounted local volume",
    "aufs": "aufs is a stacked filesystem; spill to a bind-mounted local volume",
    "squashfs": "squashfs is read-only",
}
NETWORK_FS_PREFIXES = ("nfs", "cifs", "smb", "smb3", "9p", "virtiofs", "fuse", "ceph",
                       "glusterfs", "lustre", "beegfs", "gpfs", "afs", "davfs", "sshfs")
VIRTUAL_MODEL_MARKERS = ("qemu", "virtual", "vbox", "vmware", "virtio", "msft virtual",
                         "google persistent", "amazon elastic", "ec2", "xen")
_ROUND_TRIP_BYTES = 1 << 20

LOCAL = "local"
VIRTUAL = "virtual"
NETWORK = "network"
REFUSED = "refused"


def parse_mountinfo(text: str, path: str) -> Optional[Dict[str, str]]:
    """The mount holding ``path`` (already resolved), from /proc/self/mountinfo text:
    the longest mount point that is a prefix of it."""
    best: Optional[Dict[str, str]] = None
    for line in text.splitlines():
        left, sep, right = line.partition(" - ")
        if not sep:
            continue
        fields, tail = left.split(), right.split()
        if len(fields) < 5 or len(tail) < 2:
            continue
        mount_point = fields[4].replace("\\040", " ")
        inside = path == mount_point or path.startswith(mount_point.rstrip("/") + "/")
        if inside and (best is None or len(mount_point) >= len(best["mount_point"])):
            best = {"mount_point": mount_point, "fstype": tail[0], "source": tail[1]}
    return best


def classify_fs(fstype: str, model: Optional[str], network_drive: bool = False
                ) -> Tuple[str, str]:
    """``(scope, reason)``: LOCAL, VIRTUAL, NETWORK or REFUSED."""
    name = fstype.lower()
    if name in REFUSED_FS:
        return REFUSED, REFUSED_FS[name]
    if network_drive or name.startswith(NETWORK_FS_PREFIXES):
        return NETWORK, (f"{fstype} is a network filesystem: a cloud network volume "
                         f"is not a local SSD")
    text = (model or "").lower()
    if any(marker in text for marker in VIRTUAL_MODEL_MARKERS):
        return VIRTUAL, f"the drive reports model {model!r}: a virtual disk, not a physical NVMe"
    return LOCAL, f"{fstype} on {model or 'an unnamed drive'}"


def _read_text(path: str) -> Optional[str]:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _linux_drive(path: str) -> Dict[str, Any]:
    from arm_validity import disk_device

    try:
        device = disk_device(path)
    except (OSError, ValueError) as exc:
        return {"device": None, "device_problem": str(exc)}
    if device is None:
        return {"device": None, "device_problem": "no block device under this filesystem"}
    base = f"/sys/block/{device}"
    return {
        "device": device,
        "model": _read_text(f"{base}/device/model"),
        "vendor": _read_text(f"{base}/device/vendor"),
        "rotational": _read_text(f"{base}/queue/rotational"),
        "logical_block_size": _read_text(f"{base}/queue/logical_block_size"),
    }


def _windows_disk_number(path: str) -> Dict[str, Any]:
    """The physical disk number behind a path's drive letter (Get-Partition)."""
    import subprocess

    letter = os.path.splitdrive(os.path.realpath(path))[0][:1]
    if not letter.isalpha():
        return {"device": None, "device_problem": "no drive letter"}
    script = f"(Get-Partition -DriveLetter {letter} -ErrorAction Stop).DiskNumber"
    try:
        done = subprocess.run(["powershell.exe", "-NoProfile", "-Command", script],
                              capture_output=True, text=True, timeout=60, check=False)
        if done.returncode:
            raise RuntimeError(done.stderr.strip() or f"exit {done.returncode}")
        return {"device": f"PhysicalDrive{int(done.stdout.strip())}"}
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        return {"device": None, "device_problem": f"{type(exc).__name__}: {exc}"}


def placement(path: str) -> Dict[str, Any]:
    """Where ``path`` really lives: resolved path (symlinks followed), mount point,
    filesystem source and the whole physical disk under it, as the OS reports it.

    A path that does not exist yet (a shard cache the first child creates) is placed by
    its nearest existing ancestor, named in ``probed``; nothing is created."""
    resolved = os.path.realpath(path)
    probed = resolved
    while not os.path.exists(probed) and os.path.dirname(probed) != probed:
        probed = os.path.dirname(probed)
    resolved = probed
    record: Dict[str, Any] = {"path": os.path.realpath(path), "probed": probed,
                              "platform": sys.platform}
    if sys.platform.startswith("linux"):
        record.update(parse_mountinfo(_read_text("/proc/self/mountinfo") or "", resolved) or {})
        record["drive"] = _linux_drive(resolved)
    elif sys.platform == "win32":
        record["mount_point"] = os.path.splitdrive(resolved)[0] + "\\"
        record["drive"] = _windows_disk_number(resolved)
    else:
        record["drive"] = {"device": None, "device_problem": f"not read on {sys.platform}"}
    return record


def same_physical_disk(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]
                       ) -> Tuple[Optional[bool], str]:
    """``(True|False|None, why)``: whether two ``placement`` records share one physical
    disk. None when either disk is unknown: two directories, a shared parent, a mount
    point or matching direct-I/O flags never stand in for that evidence."""
    left = ((a or {}).get("drive") or {}).get("device")
    right = ((b or {}).get("drive") or {}).get("device")
    if left is None or right is None:
        problems = [str(((x or {}).get("drive") or {}).get("device_problem"))
                    for x in (a, b) if not ((x or {}).get("drive") or {}).get("device")]
        return None, "physical disk unknown: " + "; ".join(problems)
    if left == right:
        return True, f"both on {left}"
    return False, f"{left} and {right}"


def _windows_volume(path: str) -> Dict[str, Any]:
    root = os.path.splitdrive(os.path.abspath(path))[0] + "\\"
    kernel32 = getattr(ctypes, "windll").kernel32
    fs_name = ctypes.create_unicode_buffer(64)
    ok = kernel32.GetVolumeInformationW(ctypes.c_wchar_p(root), None, 0, None, None, None,
                                        fs_name, len(fs_name))
    drive_type = int(kernel32.GetDriveTypeW(ctypes.c_wchar_p(root)))
    return {
        "mount_point": root, "fstype": fs_name.value if ok else "unknown",
        "source": root, "drive_type": drive_type,
        # GetDriveTypeW: 4 DRIVE_REMOTE, 6 DRIVE_RAMDISK.
        "network_drive": drive_type == 4, "ram_disk": drive_type == 6,
    }


def round_trip(directory: str) -> Tuple[bool, str]:
    """An aligned unbuffered write then read of 1 MiB on the volume, compared byte for byte."""
    import l2l_activations
    import torch

    path = os.path.join(directory, f"a4-direct-probe-{os.getpid()}.bin")
    try:
        spill = l2l_activations.DirectSpillFile(path, _ROUND_TRIP_BYTES)
    except (OSError, ValueError) as exc:
        return False, f"unbuffered open or fill failed: {exc}"
    try:
        sent = l2l_activations.aligned_buffer(_ROUND_TRIP_BYTES, pin=False)
        sent.copy_(torch.arange(_ROUND_TRIP_BYTES, dtype=torch.int64).remainder(251)
                   .to(torch.uint8))
        got = l2l_activations.aligned_buffer(_ROUND_TRIP_BYTES, pin=False)
        spill.write(0, sent)
        spill.read(0, got)
        if not torch.equal(sent, got):
            return False, "the unbuffered round trip returned other bytes"
        return True, "aligned 1 MiB unbuffered write and read returned the same bytes"
    except OSError as exc:
        return False, f"unbuffered write or read failed: {exc}"
    finally:
        spill.close(remove=True)


def spill_volume(directory: str, *, accept_network: bool = False) -> Dict[str, Any]:
    """Preflight of a spill directory. ``ok`` False means: do not spill there."""
    os.makedirs(directory, exist_ok=True)
    resolved = os.path.realpath(directory)
    record: Dict[str, Any] = {"path": resolved, "platform": sys.platform}
    if sys.platform.startswith("linux"):
        mount = parse_mountinfo(_read_text("/proc/self/mountinfo") or "", resolved) or {}
        record.update(mount)
        record["drive"] = _linux_drive(resolved)
        network = False
    elif sys.platform == "win32":
        volume = _windows_volume(resolved)
        record.update(volume)
        record["drive"] = {"model": None, "device_problem": "drive model not read on Windows"}
        network = bool(volume["network_drive"])
        if volume["ram_disk"]:
            record["fstype"] = "ramfs"
    else:
        record.update({"fstype": "unknown", "drive": {"model": None}})
        network = False
    usage = shutil.disk_usage(resolved)
    record.update(total_bytes=int(usage.total), free_bytes=int(usage.free))
    scope, reason = classify_fs(str(record.get("fstype") or "unknown"),
                                record["drive"].get("model"), network)
    record.update(scope=scope, scope_reason=reason)
    direct_ok, direct_reason = (False, "not tried: the filesystem is refused")
    if scope != REFUSED:
        direct_ok, direct_reason = round_trip(resolved)
    record.update(direct_io=direct_ok, direct_io_reason=direct_reason)
    refusals: List[str] = []
    if scope == REFUSED:
        refusals.append(reason)
    if scope == NETWORK and not accept_network:
        refusals.append(reason + "; pass --accept-network to run it labelled as network scope")
    if scope != REFUSED and not direct_ok:
        refusals.append(f"unbuffered I/O is not usable here: {direct_reason}")
    record["refusals"] = refusals
    record["ok"] = not refusals
    return record


# ---------------------------------------------------------------------------
# host memory
# ---------------------------------------------------------------------------
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


def non_activation_bytes(mem: Dict[str, Optional[int]], weight_pinned_bytes: int
                         ) -> Optional[int]:
    """What the process holds besides the activations and the weight source's page-locked
    bytes, for priority step 2 of the split (``split_host_ram(process_bytes=)``).

    Linux: ``RssAnon`` (pageable) plus whatever page-locked memory ``RssShmem`` shows
    beyond the weight source's own figure (gate-655: pinned host memory is shmem), so a
    pinned buffer the runtime does not account for is still charged, and the weights are
    never charged twice. Elsewhere: the resident set minus the weights. None when the
    process's memory cannot be read: a split must then refuse rather than plan blind.
    """
    weights = int(weight_pinned_bytes)
    if mem.get("rss_anon") is not None:
        return int(mem["rss_anon"]) + max(0, int(mem.get("rss_shmem") or 0) - weights)
    if mem.get("rss") is not None:
        return max(0, int(mem["rss"]) - weights)
    return None


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
