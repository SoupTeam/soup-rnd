#!/usr/bin/env python3
"""G2 sustained spill-drive I/O: write, read, the G2 step pattern, and that pattern
while a weight stream reads the drive (the C4 method).

G2 keeps each decoder layer's inner saved activations in host RAM or on NVMe instead
of recomputing them in backward. On the disk tier the spill competes with the streamed
weight reads, so the break-even needs the SUSTAINED write and read rate of the spill
target, alone and while a weight stream reads, not a short warm-cache probe.

Modes (``run --mode``; the CLI runs each arm in a fresh process):

* ``write``: sequential unbuffered ``--chunk-bytes`` writes into a ``--spill-bytes``
  file, wrapping (the pass number grows by one per wrap);
* ``read``: unbuffered reads of a spill file this process filled first (untimed);
* ``cycle``: the G2 step: layers 0..L-1 written, then read back L-1..0, repeated;
* ``weights``: the reference stream, unbuffered sequential reads of a certified weight
  fixture (``prepare-weights``), optional ``--weight-rate`` pacing, wrapping;
* ``cycle+weights``: both at once on two threads, measured separately.

Method (C4 ``origin/c4-ssd-research`` runner): sector-aligned buffers, per-window JSONL
samples with per-disk OS counters and RAM available, the first ``--warmup`` seconds
excluded, every full window kept (no outlier removal), mean / min / P10 and the first
and last five windows per direction, F1 ``ArmWatch``, AC power, source hashes, fixture
certificates. Each direction reports two rates: ``wall`` (bytes per window second) and
``busy`` (bytes per second spent inside that direction's I/O calls). ``cycle`` alternates
write and read passes, so its wall rates share the clock; ``busy`` is the per-byte cost
the break-even charges and the rate ``export_rates`` exports.

Every chunk written carries a stamp in each 4 KiB sector (magic, absolute offset, pass)
over a seeded random body. Reads check the stamps of every chunk, and the body of every
chunk (``--verify full``) or of every ``--verify-every``-th chunk (``sample``); a
mismatch fails the arm (``G2_SPILL_INTEGRITY``). A full drive is ``G2_SPILL_ENOSPC``.
The spill file is removed on every exit, error and Ctrl-C included, unless ``--keep``.
Two paths are not two disks: each arm records the volume and physical-disk mapping the
OS reported, and says "unknown" when it could not tell.

A ``--smoke`` arm is labelled ``SMOKE, excluded`` and is never evidence. Buffered I/O
(``--buffered``) is allowed only with ``--smoke``. Invalid and interrupted arms are
written, never deleted.

Benchmark harness code, not shipped. Standard library at the top level; torch, psutil
and soup_cli are imported inside functions.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import glob
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arm_validity  # noqa: E402
import l2l_box  # noqa: E402

from soup_cli.utils import l2l_activations  # noqa: E402

HARNESS = Path(__file__).resolve().parent
ROOT = HARNESS.parents[1]
KIB, MIB, GIB = 1 << 10, 1 << 20, 1 << 30
SECTOR = l2l_activations.SECTOR_BYTES
_write_all = l2l_activations._write_all
_read_all = l2l_activations._read_all

MODES = ("write", "read", "cycle", "weights", "cycle+weights")
DIRECTIONS = ("write", "read", "weight")
MODE_DIRECTIONS = {
    "write": ("write",),
    "read": ("read",),
    "cycle": ("write", "read"),
    "weights": ("weight",),
    "cycle+weights": ("write", "read", "weight"),
}
SPILL_MODES = ("write", "read", "cycle", "cycle+weights")
WEIGHT_MODES = ("weights", "cycle+weights")

#: b"G2SPILL1" as a big-endian int64: the first word of every stamped sector.
MAGIC = 0x4732_5350_494C_4C31
#: int64 words at the head of each sector: magic, absolute byte offset, pass number.
STAMP_WORDS = 3
#: A sample interval longer than this many windows (and 5 s) invalidates the arm (C4).
MAX_GAP_WINDOWS = 5
#: C4's sustained protocol, used only to label an arm "not sustained"; the G2 rule's V9
#: row decides what the break-even may use.
SUSTAINED_WARMUP_S = 600.0
SUSTAINED_MEASURED_S = 1800.0
#: Windows error codes that mean the volume is full.
_WIN_DISK_FULL = (112, 39)
_FILE_ATTRIBUTE_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_FILE_ATTRIBUTE_SPARSE_FILE = getattr(stat, "FILE_ATTRIBUTE_SPARSE_FILE", 0x200)
_FILE_ATTRIBUTE_COMPRESSED = getattr(stat, "FILE_ATTRIBUTE_COMPRESSED", 0x800)

REFUSED_FS = {
    "tmpfs": "tmpfs is RAM: a spill there is not a drive",
    "ramfs": "ramfs is RAM: a spill there is not a drive",
    "devtmpfs": "devtmpfs is RAM",
    "overlay": "overlayfs: the drive below and its direct-I/O path belong to the upper layer",
    "aufs": "aufs is a stacked filesystem",
    "squashfs": "squashfs is read-only",
}
NETWORK_FS_PREFIXES = ("nfs", "cifs", "smb", "9p", "virtiofs", "fuse", "ceph", "glusterfs",
                       "lustre", "beegfs", "gpfs", "afs", "davfs", "sshfs")
#: A virtual block device is labelled, never refused: a KVM guest's virtio disk (PCI vendor
#: 0x1af4; vast.ai VMs report no model string at all), Xen and virtio device names, and the
#: model strings hypervisors give their disks. G4 and the break-even then describe that
#: virtual disk, not a physical NVMe.
VIRTIO_VENDOR = "0x1af4"
VIRTUAL_DEVICE_PREFIXES = ("vd", "xvd")
VIRTUAL_MODEL_MARKERS = ("qemu", "virtual", "vbox", "vmware", "virtio", "xen")

#: Files whose SHA-256 every arm records: this module, the harness modules it imports,
#: and Soup's own direct-I/O reader the weight stream uses.
SOURCE_FILES = (
    "benchmarks/harness/g2_disk.py",
    "benchmarks/harness/l2l_activations.py",
    "benchmarks/harness/l2l_box.py",
    "benchmarks/harness/arm_validity.py",
    "src/soup_cli/utils/l2l_activations.py",
    "src/soup_cli/utils/safetensors_reader.py",
)


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
class G2DiskError(RuntimeError):
    """A refusal or failure with a stable code at the head of its message."""

    code = "G2_DISK"

    def __init__(self, message: str):
        super().__init__(f"{self.code}: {message}")


class SpillDiskFullError(G2DiskError):
    code = "G2_SPILL_ENOSPC"


class SpillDiskShortError(G2DiskError):
    code = "G2_SPILL_DISK_SHORT"


class IntegrityError(G2DiskError):
    code = "G2_SPILL_INTEGRITY"


class FixtureError(G2DiskError):
    code = "G2_FIXTURE"


class BackendRefusedError(G2DiskError):
    code = "G2_BUFFERED_REFUSED"


class ScopeRefusedError(G2DiskError):
    code = "G2_SCOPE_REFUSED"


def is_enospc(exc: BaseException) -> bool:
    """True when an OSError says the volume is full (POSIX ENOSPC or its Windows codes)."""
    return isinstance(exc, OSError) and (
        exc.errno == errno.ENOSPC or getattr(exc, "winerror", None) in _WIN_DISK_FULL)


# ---------------------------------------------------------------------------
# pure analysis
# ---------------------------------------------------------------------------
def percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolation percentile (C4 ``analyze.percentile``)."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of no values")
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def validate_rows(rows: Sequence[Dict[str, Any]]) -> None:
    """Raise ValueError unless time strictly increases and every cumulative counter
    never decreases: a summary of broken samples is not a summary."""
    previous_t = 0.0
    previous: Dict[str, float] = {}
    for row in rows:
        elapsed = float(row["elapsed_s"])
        if not math.isfinite(elapsed) or elapsed <= previous_t:
            raise ValueError(f"sample time does not increase at elapsed_s={elapsed}")
        for direction in DIRECTIONS:
            for key in (f"total_{direction}_bytes", f"{direction}_busy_s"):
                value = float(row[key])
                if not math.isfinite(value) or value < previous.get(key, 0.0):
                    raise ValueError(f"{key} decreased or is not finite at {elapsed} s")
                previous[key] = value
        previous_t = elapsed


def _interpolate(times: Sequence[float], values: Sequence[float], t: float) -> float:
    lo, hi = 0, len(times) - 1
    while lo < hi:  # first index with times[i] >= t
        mid = (lo + hi) // 2
        if times[mid] < t:
            lo = mid + 1
        else:
            hi = mid
    if lo == 0 or times[lo] == t:
        return values[lo]
    t0, t1 = times[lo - 1], times[lo]
    return values[lo - 1] + (t - t0) / (t1 - t0) * (values[lo] - values[lo - 1])


def window_deltas(rows: Sequence[Dict[str, Any]], key: str, *, start_s: float,
                  width_s: float, end_s: float) -> List[float]:
    """The growth of cumulative ``key`` over every FULL window ``[start + i*w, start +
    (i+1)*w)`` that ends by ``end_s``, interpolating linearly between samples (C4
    ``window_rates``). A trailing partial window is dropped; nothing else is."""
    if width_s <= 0:
        raise ValueError("window width must be positive")
    times = [0.0] + [float(r["elapsed_s"]) for r in rows]
    values = [0.0] + [float(r[key]) for r in rows]
    end = min(end_s, times[-1])
    out: List[float] = []
    index = 0
    while start_s + (index + 1) * width_s <= end + 1e-9:
        a = start_s + index * width_s
        out.append(_interpolate(times, values, a + width_s) - _interpolate(times, values, a))
        index += 1
    return out


def _stats(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"n_windows": 0, "mean_Bps": None, "min_Bps": None, "p10_Bps": None,
                "first_5_Bps": [], "last_5_Bps": [], "first_5_mean_Bps": None,
                "last_5_mean_Bps": None, "windows_Bps": []}
    first, last = list(values[:5]), list(values[-5:])
    return {"n_windows": len(values), "mean_Bps": sum(values) / len(values),
            "min_Bps": min(values), "p10_Bps": percentile(values, 0.1),
            "first_5_Bps": first, "last_5_Bps": last,
            "first_5_mean_Bps": sum(first) / len(first),
            "last_5_mean_Bps": sum(last) / len(last), "windows_Bps": list(values)}


def summarize(rows: Sequence[Dict[str, Any]], *, warmup_s: float, window_s: float,
              directions: Sequence[str] = DIRECTIONS,
              end_s: Optional[float] = None) -> Dict[str, Any]:
    """Per-direction rates over every full window after ``warmup_s``.

    ``wall``: bytes moved per window second. ``busy``: per window, bytes over seconds
    spent inside that direction's I/O calls (windows where it issued none have no busy
    rate); its ``mean_Bps`` is all bytes over all busy seconds of the full windows.
    A direction not in ``directions`` is None. Nothing is removed as an outlier.
    """
    validate_rows(rows)
    last = float(rows[-1]["elapsed_s"]) if rows else 0.0
    end = last if end_s is None else min(last, float(end_s))
    out: Dict[str, Any] = {
        "warmup_s": float(warmup_s), "window_s": float(window_s), "end_s": end,
        "method": "every full window after warmup; no outlier removal; trailing partial "
                  "window dropped; P10 by linear interpolation",
    }
    n_windows = 0
    for direction in DIRECTIONS:
        if direction not in directions:
            out[direction] = None
            continue
        moved = window_deltas(rows, f"total_{direction}_bytes", start_s=warmup_s,
                              width_s=window_s, end_s=end)
        busy = window_deltas(rows, f"{direction}_busy_s", start_s=warmup_s,
                             width_s=window_s, end_s=end)
        n_windows = len(moved)
        total_bytes, total_busy = sum(moved), sum(busy)
        busy_stats = _stats([b / s for b, s in zip(moved, busy) if s > 0])
        busy_stats["per_window_mean_Bps"] = busy_stats["mean_Bps"]
        busy_stats["mean_Bps"] = total_bytes / total_busy if total_busy > 0 else None
        out[direction] = {"bytes": total_bytes, "busy_s": total_busy,
                          "wall": _stats([b / window_s for b in moved]), "busy": busy_stats}
    out["n_windows"] = n_windows
    out["measured_s"] = n_windows * float(window_s)
    return out


def _rate(summary: Optional[Dict[str, Any]], direction: str, kind: str = "busy"
          ) -> Optional[float]:
    if summary and "summary" in summary:
        summary = summary["summary"]
    part = (summary or {}).get(direction)
    return None if not part else part[kind]["mean_Bps"]


def contention(reference_weights_summary: Dict[str, Any],
               concurrent_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Weight-read slowdown from a ``weights`` arm (reference) to a ``cycle+weights`` arm,
    and the spill rates measured under that contention. Accepts summaries or whole arm
    records. ``weight_ratio`` is concurrent busy mean / reference busy mean; None when
    either side has no rate."""
    ref, con = _rate(reference_weights_summary, "weight"), _rate(concurrent_summary, "weight")
    ref_wall = _rate(reference_weights_summary, "weight", "wall")
    con_wall = _rate(concurrent_summary, "weight", "wall")
    return {
        "weight_ref_Bps": ref, "weight_with_spill_Bps": con,
        "weight_ratio": con / ref if ref and con is not None else None,
        "weight_ref_wall_Bps": ref_wall, "weight_with_spill_wall_Bps": con_wall,
        "weight_wall_ratio": con_wall / ref_wall if ref_wall and con_wall is not None else None,
        "spill_write_Bps": _rate(concurrent_summary, "write"),
        "spill_read_Bps": _rate(concurrent_summary, "read"),
        "spill_write_wall_Bps": _rate(concurrent_summary, "write", "wall"),
        "spill_read_wall_Bps": _rate(concurrent_summary, "read", "wall"),
    }


def decide_status(*, smoke: bool, direct: bool, error: Optional[str],
                  f1_outcome: Optional[str], n_windows: int, ac_power: Optional[bool],
                  sources_unchanged: bool, scope_override: bool,
                  problems: Sequence[str] = ()) -> Tuple[str, List[str]]:
    """``(status, reasons)``: ``candidate`` only for a non-smoke, unbuffered, error-free
    arm with F1 ok, AC power, unchanged sources, a local scope and >= 1 full window;
    ``smoke_passed`` for a smoke arm that meets the rest; else ``invalid``."""
    reasons = list(problems)
    if error:
        reasons.append(f"error: {error}")
    if not direct and not smoke:
        reasons.append(f"{BackendRefusedError.code}: buffered I/O is allowed only in --smoke")
    if scope_override and not smoke:
        reasons.append("non-local volume accepted by override: not local-drive evidence")
    if f1_outcome != arm_validity.OK:
        reasons.append(f"F1 outcome {f1_outcome}")
    if n_windows < 1:
        reasons.append("no full window after warmup")
    if ac_power is not True:
        reasons.append(f"AC power {ac_power}")
    if not sources_unchanged:
        reasons.append("sources changed during the arm")
    if reasons:
        return "invalid", reasons
    return ("smoke_passed" if smoke else "candidate"), []


def check_backend(*, smoke: bool, buffered: bool) -> None:
    """Refuse buffered I/O outside a smoke arm, before any byte moves."""
    if buffered and not smoke:
        raise BackendRefusedError("buffered I/O measures the page cache; it is allowed "
                                  "only with --smoke and is never a candidate")


# ---------------------------------------------------------------------------
# sizes
# ---------------------------------------------------------------------------
_UNITS = {"": 1, "B": 1, "K": KIB, "KB": 1000, "KIB": KIB, "M": MIB, "MB": 1000 ** 2,
          "MIB": MIB, "G": GIB, "GB": 1000 ** 3, "GIB": GIB, "T": GIB * KIB,
          "TB": 1000 ** 4, "TIB": GIB * KIB}


def parse_bytes(text: str) -> int:
    """``"512MiB"``, ``"8M"`` (binary), ``"2GB"`` (decimal) or a plain integer; > 0."""
    value = parse_bytes_or_zero(text)
    if value == 0:
        raise argparse.ArgumentTypeError(f"not a positive byte size: {text!r}")
    return value


def parse_bytes_or_zero(text: str) -> int:
    """As ``parse_bytes``, but 0 is allowed (a reserve may be none)."""
    raw = str(text).strip().replace("_", "")
    digits = raw.rstrip("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    unit = raw[len(digits):].strip().upper()
    if unit not in _UNITS or not digits:
        raise argparse.ArgumentTypeError(f"not a byte size: {text!r}")
    try:
        value = float(digits) * _UNITS[unit]
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a byte size: {text!r}") from None
    if value < 0 or value != int(value):
        raise argparse.ArgumentTypeError(f"not a whole, non-negative byte size: {text!r}")
    return int(value)


def check_geometry(*, spill_bytes: int, chunk_bytes: int, layers: int) -> Dict[str, int]:
    """Every spill request a whole, sector-aligned chunk; the spill split evenly into
    ``layers`` layers of whole chunks."""
    if chunk_bytes <= 0 or chunk_bytes % SECTOR:
        raise ValueError(f"--chunk-bytes must be a positive multiple of {SECTOR}")
    if layers < 1:
        raise ValueError("--layers must be >= 1")
    if spill_bytes <= 0 or spill_bytes % (layers * chunk_bytes):
        raise ValueError(f"--spill-bytes must be a positive multiple of layers x chunk = "
                         f"{layers * chunk_bytes}; got {spill_bytes}")
    return {"spill_bytes": spill_bytes, "chunk_bytes": chunk_bytes, "layers": layers,
            "layer_bytes": spill_bytes // layers,
            "chunks_per_layer": spill_bytes // layers // chunk_bytes}


def check_weight_geometry(weight_bytes: int, weight_chunk_bytes: int) -> None:
    """The weight fixture holds whole sector-aligned ``--weight-chunk-bytes`` requests."""
    if weight_chunk_bytes <= 0 or weight_chunk_bytes % SECTOR:
        raise ValueError(f"--weight-chunk-bytes must be a positive multiple of {SECTOR}")
    if weight_bytes < weight_chunk_bytes or weight_bytes % weight_chunk_bytes:
        raise ValueError(f"the weight fixture ({weight_bytes} bytes) must hold whole "
                         f"--weight-chunk-bytes ({weight_chunk_bytes}) requests")


# ---------------------------------------------------------------------------
# the verifiable chunk pattern
# ---------------------------------------------------------------------------
class ChunkPattern:
    """A seeded random chunk whose every 4 KiB sector starts with (MAGIC, absolute byte
    offset, pass). Stamps make a misplaced, stale or torn sector visible; the seeded body
    makes corrupted bytes visible."""

    def __init__(self, chunk_bytes: int, seed: int):
        import torch

        if chunk_bytes % SECTOR:
            raise ValueError(f"chunk must be a multiple of {SECTOR}")
        self.chunk_bytes = int(chunk_bytes)
        self.template = l2l_activations.aligned_buffer(self.chunk_bytes, pin=False)
        generator = torch.Generator().manual_seed(int(seed))
        self.template.random_(0, 256, generator=generator)
        words = self.words(self.template)
        words[:, 0] = MAGIC
        words[:, 1:STAMP_WORDS] = 0
        self.sector_offsets = torch.arange(words.shape[0], dtype=torch.int64) * SECTOR
        self.body_checks = 0
        self.stamp_checks = 0

    @staticmethod
    def words(buf: Any) -> Any:
        import torch

        return buf.view(torch.int64).view(-1, SECTOR // 8)

    def buffer(self) -> Any:
        buf = l2l_activations.aligned_buffer(self.chunk_bytes, pin=False)
        buf.copy_(self.template)
        return buf

    def stamp(self, buf: Any, offset: int, pass_no: int) -> None:
        words = self.words(buf)
        words[:, 1].copy_(self.sector_offsets + int(offset))
        words[:, 2].fill_(int(pass_no))

    def verify(self, buf: Any, offset: int, pass_no: int, *, body: bool) -> None:
        """Raise IntegrityError naming the first bad sector of a chunk read at ``offset``."""
        import torch

        words, expected = self.words(buf), self.words(self.template)
        bad = ((words[:, 0] != MAGIC) | (words[:, 1] != self.sector_offsets + int(offset))
               | (words[:, 2] != int(pass_no)))
        self.stamp_checks += 1
        if body:
            self.body_checks += 1
            bad |= (words[:, STAMP_WORDS:] != expected[:, STAMP_WORDS:]).any(dim=1)
        if bool(bad.any()):
            sector = int(torch.nonzero(bad)[0, 0])
            row = words[sector, :STAMP_WORDS].tolist()
            raise IntegrityError(
                f"chunk at byte {offset}, sector {sector}: read (magic {row[0]:#x}, offset "
                f"{row[1]}, pass {row[2]}), expected (magic {MAGIC:#x}, offset "
                f"{offset + sector * SECTOR}, pass {pass_no}){' or body differs' if body else ''}")


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------
def _norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def check_no_links(path: str) -> None:
    """Refuse a path that is, or sits under, a symlink, junction or other reparse point:
    its realpath must equal its absolute path, and neither it nor its parent may carry a
    link attribute."""
    absolute = os.path.abspath(path)
    if _norm(os.path.realpath(absolute)) != _norm(absolute):
        raise FixtureError(f"{absolute} resolves to {os.path.realpath(absolute)}: links and "
                           f"junctions are refused")
    for item in (absolute, os.path.dirname(absolute)):
        if not os.path.lexists(item):
            continue
        info = os.lstat(item)
        attributes = getattr(info, "st_file_attributes", 0)
        if stat.S_ISLNK(info.st_mode) or attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
            raise FixtureError(f"{item} is a link or reparse point: refused")


def _contained(path: str, directory: str) -> bool:
    real_dir = os.path.realpath(directory)
    return os.path.commonpath([os.path.realpath(path), real_dir]) == real_dir


def spill_name(tag: str) -> str:
    """The spill file name for ``tag``; anything but [A-Za-z0-9._-] becomes ``_``."""
    safe = "".join(c if c.isascii() and (c.isalnum() or c in "._-") else "_" for c in tag)
    return f"g2-spill-{safe.lstrip('.') or 'x'}.bin"


def _create_exclusive(path: str) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    os.close(os.open(path, flags, 0o600))


def free_space_check(directory: str, needed_bytes: int, reserve_bytes: int) -> Dict[str, int]:
    import shutil

    free = int(shutil.disk_usage(directory).free)
    if free < needed_bytes + reserve_bytes:
        raise SpillDiskShortError(
            f"{directory} has {free} bytes free; needs {needed_bytes} + reserve "
            f"{reserve_bytes} = {needed_bytes + reserve_bytes}")
    return {"free_bytes": free, "needed_bytes": needed_bytes, "reserve_bytes": reserve_bytes}


class SpillFile:
    """One exclusively created spill file with one positional read-write handle.

    Unbuffered (``l2l_activations._open_unbuffered``) unless ``direct`` is False. A full
    volume raises ``SpillDiskFullError``. ``close`` removes the file unless ``keep``; on
    POSIX it is unlinked right after opening, so even a killed process leaves nothing.
    """

    def __init__(self, path: str, size: int, *, direct: bool, keep: bool = False):
        self.path, self.size = str(path), int(size)
        self.direct, self.keep = bool(direct), bool(keep)
        self.handle: Optional[io.FileIO] = None
        self.removed = False
        self.left_behind: Optional[str] = None
        self.unlinked_early = False
        _create_exclusive(self.path)
        try:
            if self.direct:
                self.handle = l2l_activations._open_unbuffered(self.path)
            else:
                self.handle = io.FileIO(self.path, "r+b")
            if not self.keep and os.name == "posix":
                os.unlink(self.path)
                self.unlinked_early = True
        except BaseException:
            self.close()
            raise

    def write(self, offset: int, buf: Any) -> None:
        try:
            _write_all(self.handle, offset, buf)
        except OSError as exc:
            if is_enospc(exc):
                raise SpillDiskFullError(
                    f"the spill volume is full writing {buf.numel()} bytes at {offset} of "
                    f"{self.path}: {exc}") from exc
            raise

    def read(self, offset: int, buf: Any) -> None:
        _read_all(self.handle, offset, buf)

    def sync(self) -> None:
        os.fsync(self.handle.fileno())

    def close(self) -> Dict[str, Any]:
        if self.handle is not None:
            try:
                self.handle.close()
            finally:
                self.handle = None
        if not self.keep:
            if self.unlinked_early or not os.path.exists(self.path):
                self.removed = True
            else:
                try:
                    os.remove(self.path)
                    self.removed = True
                except OSError:
                    self.left_behind = self.path
        return {"path": self.path, "size_bytes": self.size, "direct": self.direct,
                "kept": self.keep, "removed": self.removed, "left_behind": self.left_behind,
                "unlinked_after_open": self.unlinked_early}

    def __enter__(self) -> "SpillFile":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def prefill(spill: SpillFile, pattern: ChunkPattern, buf: Any,
            stop: Optional[threading.Event] = None) -> float:
    """Write pass 0 over the whole file (untimed): the read mode's data, and no timed
    write ever extends the file or lands past its valid-data length."""
    start = time.perf_counter()
    for offset in range(0, spill.size, pattern.chunk_bytes):
        if stop is not None and stop.is_set():
            raise KeyboardInterrupt("stopped during prefill")
        pattern.stamp(buf, offset, 0)
        spill.write(offset, buf)
    spill.sync()
    return time.perf_counter() - start


def certificate_path(fixture: str) -> str:
    return fixture + ".cert.json"


def file_identity(path: str) -> Dict[str, int]:
    info = os.stat(path)
    return {"size": int(info.st_size), "mtime_ns": int(info.st_mtime_ns),
            "dev": int(info.st_dev), "ino": int(info.st_ino)}


def sha256_file(path: str, block: int = 16 * MIB) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: str, data: Any, *, exclusive: bool = False) -> None:
    text = json.dumps(data, indent=2, allow_nan=False) + "\n"
    with open(path, "x" if exclusive else "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def prepare_weights(path: str, size: int, *, reserve_bytes: int,
                    block: int = 4 * MIB) -> Dict[str, Any]:
    """Create the weight-stream fixture: random bytes, exclusive creation, fsync, then a
    certificate (size, SHA-256, mtime, identity). The certificate is written ``incomplete``
    first, so an interrupted preparation is never mistaken for a fixture."""
    import psutil

    path = os.path.abspath(path)
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    check_no_links(directory)
    if size <= 0 or size % block:
        raise ValueError(f"fixture size must be a positive multiple of {block}")
    space = free_space_check(directory, size, reserve_bytes)
    cert: Dict[str, Any] = {
        "status": "incomplete", "kind": "synthetic random-byte weight-stream fixture",
        "path": path, "size_bytes": size, "created_unix_s": time.time(),
        "ram_total_bytes": int(psutil.virtual_memory().total), "space": space,
    }
    _write_json(certificate_path(path), cert, exclusive=True)
    digest = hashlib.sha256()
    created = False
    try:
        with open(path, "xb", buffering=0) as out:
            created = True
            for _ in range(size // block):
                data = os.urandom(block)
                view = memoryview(data)
                while view:
                    done = out.write(view)
                    if not done:
                        raise OSError("short fixture write")
                    view = view[done:]
                digest.update(data)
            os.fsync(out.fileno())
        info = os.stat(path)
        attributes = getattr(info, "st_file_attributes", 0)
        blocks = getattr(info, "st_blocks", None)
        if attributes & (_FILE_ATTRIBUTE_SPARSE_FILE | _FILE_ATTRIBUTE_COMPRESSED) or (
                blocks is not None and blocks * 512 < size):
            raise FixtureError(f"{path} is sparse or compressed: refused")
    except BaseException as exc:
        # The partial fixture goes; the certificate stays, saying why.
        if created and os.path.exists(path):
            os.remove(path)
        cert.update(status="failed", error=repr(exc),
                    error_code=SpillDiskFullError.code if is_enospc(exc) else None)
        _write_json(certificate_path(path), cert)
        if is_enospc(exc):
            raise SpillDiskFullError(f"the volume filled while preparing {path}: {exc}") from exc
        raise
    cert.update(status="complete", sha256=digest.hexdigest(), identity=file_identity(path),
                exceeds_ram=size > cert["ram_total_bytes"])
    _write_json(certificate_path(path), cert)
    return cert


def check_fixture(path: str, *, weight_chunk_bytes: int, rehash: bool = False
                  ) -> Dict[str, Any]:
    """The certified weight fixture, unchanged since preparation, or FixtureError."""
    path = os.path.abspath(path)
    check_no_links(path)
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode):
        raise FixtureError(f"{path} is not a regular file")
    try:
        with open(certificate_path(path), encoding="utf-8") as handle:
            cert = json.load(handle)
    except (OSError, ValueError) as exc:
        raise FixtureError(f"no readable certificate for {path}: {exc}") from exc
    if cert.get("status") != "complete":
        raise FixtureError(f"{path}: certificate status {cert.get('status')!r}")
    if _norm(cert.get("path", "")) != _norm(path):
        raise FixtureError(f"{path}: certificate names {cert.get('path')}")
    if file_identity(path) != cert.get("identity"):
        raise FixtureError(f"{path} changed after preparation (size/mtime/identity)")
    check_weight_geometry(int(cert["size_bytes"]), weight_chunk_bytes)
    out = dict(cert)
    out["rehash"] = None
    if rehash:
        actual = sha256_file(path)
        if actual != cert["sha256"]:
            raise FixtureError(f"{path}: SHA-256 {actual} differs from certificate")
        out["rehash"] = actual
    return out


def open_weights(path: str, *, direct: bool) -> io.FileIO:
    """Soup's own direct-I/O reader (``safetensors_reader.open_direct``), as the streamed
    weight source opens shards; buffered only for a smoke arm."""
    if not direct:
        return io.FileIO(path, "rb")
    from soup_cli.utils.safetensors_reader import open_direct

    return open_direct(path)


# ---------------------------------------------------------------------------
# volumes and physical disks
# ---------------------------------------------------------------------------
def parse_mountinfo(text: str, path: str) -> Optional[Dict[str, str]]:
    """The mount holding resolved ``path``: the longest mount point that prefixes it."""
    best: Optional[Dict[str, str]] = None
    for line in text.splitlines():
        left, sep, right = line.partition(" - ")
        fields, tail = left.split(), right.split()
        if not sep or len(fields) < 5 or len(tail) < 2:
            continue
        mount_point = fields[4].replace("\\040", " ")
        inside = path == mount_point or path.startswith(mount_point.rstrip("/") + "/")
        if inside and (best is None or len(mount_point) >= len(best["mount_point"])):
            best = {"mount_point": mount_point, "fstype": tail[0], "source": tail[1]}
    return best


def classify_scope(fstype: str, *, network: bool = False, ram_disk: bool = False
                   ) -> Tuple[str, str]:
    """``(scope, reason)``: ``local``, ``network``, ``unknown`` or ``refused``. Only
    ``local`` runs without ``--accept-nonlocal``; an unreadable filesystem is not local."""
    name = (fstype or "unknown").lower()
    if ram_disk:
        return "refused", "a RAM disk: a spill there is not a drive"
    if name in REFUSED_FS:
        return "refused", REFUSED_FS[name]
    if network or name.startswith(NETWORK_FS_PREFIXES):
        return "network", f"{fstype} is a network volume, not a local drive"
    if name == "unknown":
        return "unknown", "the filesystem type could not be read"
    return "local", f"{fstype}"


_WINDOWS_DISKS: Dict[str, Dict[str, Any]] = {}


def _windows_disk(letter: str) -> Dict[str, Any]:
    if letter in _WINDOWS_DISKS:
        return _WINDOWS_DISKS[letter]
    script = (f"$p = Get-Partition -DriveLetter {letter} -ErrorAction Stop; "
              f"$d = Get-PhysicalDisk | Where-Object DeviceId -eq $p.DiskNumber; "
              f"@{{number = $p.DiskNumber; model = $d.FriendlyName; "
              f"media = [string]$d.MediaType; bus = [string]$d.BusType}} | ConvertTo-Json")
    try:
        done = subprocess.run(["powershell.exe", "-NoProfile", "-Command", script],
                              capture_output=True, text=True, timeout=60, check=False)
        if done.returncode:
            raise RuntimeError(done.stderr.strip() or f"exit {done.returncode}")
        data = json.loads(done.stdout)
        out = {"key": f"PhysicalDrive{int(data['number'])}", "id": f"disk{data['number']}",
               "model": data.get("model"), "media": data.get("media"),
               "bus": data.get("bus"), "problem": None}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError,
            subprocess.SubprocessError) as exc:
        out = {"key": None, "id": None, "model": None, "problem": f"{type(exc).__name__}: {exc}"}
    _WINDOWS_DISKS[letter] = out
    return out


def virtual_disk(device: Optional[str], model: Optional[str], vendor: Optional[str]
                 ) -> Tuple[Optional[bool], str]:
    """``(virtual, why)`` from the whole disk's name, model and vendor as /sys shows them:
    True for a hypervisor's disk, False for a named non-virtual one, None when nothing names
    it. A label for the record (G4 and the break-even describe that disk), never a refusal."""
    if vendor and vendor.strip().lower() == VIRTIO_VENDOR:
        return True, f"vendor {vendor.strip()} (virtio)"
    text = (model or "").lower()
    marker = next((m for m in VIRTUAL_MODEL_MARKERS if m in text), None)
    if marker:
        return True, f"model {model!r}"
    if device and device.startswith(VIRTUAL_DEVICE_PREFIXES):
        return True, f"device name {device}"
    if model or vendor:
        return False, f"model {model!r}, vendor {vendor!r}"
    return None, "the disk names no model or vendor"


def _linux_disk(path: str) -> Dict[str, Any]:
    try:
        device = arm_validity.disk_device(path)
    except (OSError, ValueError) as exc:
        return {"key": None, "id": None, "model": None, "problem": str(exc)}
    if device is None:
        return {"key": None, "id": None, "model": None,
                "problem": "no block device under this filesystem"}
    model, vendor = (_read_first_line(f"/sys/block/{device}/device/{name}")
                     for name in ("model", "vendor"))
    virtual, why = virtual_disk(device, model, vendor)
    return {"key": device, "id": device, "model": model, "vendor": vendor,
            "virtual": virtual, "virtual_reason": why, "problem": None}


def volume_info(path: str) -> Dict[str, Any]:
    """Filesystem, volume and physical disk of ``path`` as the OS reports them."""
    resolved = os.path.realpath(path)
    record: Dict[str, Any] = {"path": resolved, "platform": sys.platform}
    network = ram_disk = False
    if sys.platform.startswith("linux"):
        try:
            text = Path("/proc/self/mountinfo").read_text()
        except OSError:
            text = ""
        record.update(parse_mountinfo(text, resolved) or {"fstype": "unknown"})
        record["st_dev"] = int(os.stat(resolved).st_dev)
        record["disk"] = _linux_disk(resolved)
    elif sys.platform == "win32":
        root = os.path.splitdrive(resolved)[0] + "\\"
        kernel32 = getattr(ctypes, "windll").kernel32
        fs_name = ctypes.create_unicode_buffer(64)
        serial = ctypes.c_ulong(0)
        ok = kernel32.GetVolumeInformationW(ctypes.c_wchar_p(root), None, 0,
                                            ctypes.byref(serial), None, None, fs_name,
                                            len(fs_name))
        drive_type = int(kernel32.GetDriveTypeW(ctypes.c_wchar_p(root)))
        network, ram_disk = drive_type == 4, drive_type == 6
        record.update(mount_point=root, fstype=fs_name.value if ok else "unknown",
                      volume_serial=f"{serial.value:08X}" if ok else None,
                      drive_type=drive_type)
        record["disk"] = (_windows_disk(root[0]) if root[:1].isalpha() else
                          {"key": None, "id": None, "model": None, "problem": "no drive letter"})
    else:
        record.update(fstype="unknown", disk={"key": None, "id": None, "model": None,
                                              "problem": f"not implemented on {sys.platform}"})
    scope, reason = classify_scope(str(record.get("fstype")), network=network,
                                   ram_disk=ram_disk)
    record.update(scope=scope, scope_reason=reason)
    return record


def check_scope(info: Dict[str, Any], *, accept_nonlocal: bool) -> bool:
    """Raise ScopeRefusedError for a refused or network volume unless overridden; return
    True when the override was what let it through."""
    if info["scope"] == "local":
        return False
    if not accept_nonlocal:
        raise ScopeRefusedError(f"{info['path']}: {info['scope_reason']}; pass "
                                f"--accept-nonlocal to run it labelled (never a candidate)")
    return True



def _read_first_line(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.readline().strip() or None
    except OSError:
        return None


def _windows_machine_guid() -> Optional[str]:
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SOFTWARE\Microsoft\Cryptography") as key:
            return str(winreg.QueryValueEx(key, "MachineGuid")[0])
    except (ImportError, OSError):
        return None


def host_identity() -> Dict[str, Any]:
    """Which machine (and which boot of it) a record came from, as the OS reports it.

    ``machine_id`` is /etc/machine-id (systemd; one per installed system) on Linux and the
    MachineGuid on Windows; ``boot_id`` is /proc/sys/kernel/random/boot_id (Linux only).
    Missing parts are None: two records are only "the same host" when both name one."""
    if sys.platform.startswith("linux"):
        machine = _read_first_line("/etc/machine-id") or _read_first_line(
            "/var/lib/dbus/machine-id")
        boot = _read_first_line("/proc/sys/kernel/random/boot_id")
    elif sys.platform == "win32":
        machine, boot = _windows_machine_guid(), None
    else:
        machine, boot = None, None
    return {"hostname": platform.node() or None, "machine_id": machine, "boot_id": boot,
            "platform": sys.platform}


def same_host(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]
              ) -> Tuple[Optional[bool], str]:
    """True/False when both records name a machine id, else None (never "same")."""
    left, right = (a or {}).get("machine_id"), (b or {}).get("machine_id")
    if not left or not right:
        return None, "machine id unknown on " + ", ".join(
            side for side, value in (("one side", left), ("the other", right)) if not value)
    if left != right:
        return False, f"machine {left[:12]} vs {right[:12]}"
    return True, f"machine {left[:12]}"


def same_disk(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Optional[bool]:
    """True/False when both physical disks are known, else None: two paths alone are
    never taken for two disks."""
    if not a or not b:
        return None
    left, right = a.get("disk", {}).get("id"), b.get("disk", {}).get("id")
    return None if left is None or right is None else left == right


def drive_temperature(disk_key: Optional[str]) -> Optional[float]:
    """Hottest hwmon sensor of the NVMe controller behind ``disk_key`` (Linux), else None."""
    if not disk_key or not sys.platform.startswith("linux"):
        return None
    controller = os.path.realpath(f"/sys/block/{disk_key}/device")
    values = []
    for sensor in glob.glob("/sys/class/hwmon/hwmon*/temp*_input"):
        real = os.path.realpath(os.path.dirname(sensor))
        if real.startswith(controller.rstrip("/") + "/"):
            try:
                values.append(int(Path(sensor).read_text().strip()) / 1000.0)
            except (OSError, ValueError):
                continue
    return max(values) if values else None


def disk_counters(keys: Sequence[str]) -> Dict[str, Dict[str, int]]:
    """psutil per-disk counters for ``keys`` (all disks when none is known)."""
    import psutil

    counters = psutil.disk_io_counters(perdisk=True) or {}
    chosen = [k for k in keys if k in counters] or sorted(counters)
    return {k: {f: int(v) for f, v in counters[k]._asdict().items()} for k in chosen}


# ---------------------------------------------------------------------------
# the I/O loops
# ---------------------------------------------------------------------------
class Meter:
    """Cumulative completed bytes, seconds inside I/O calls and requests, per direction."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.bytes = dict.fromkeys(DIRECTIONS, 0)
        self.busy = dict.fromkeys(DIRECTIONS, 0.0)
        self.requests = dict.fromkeys(DIRECTIONS, 0)
        self.cycles = 0

    def add(self, direction: str, nbytes: int, seconds: float) -> None:
        with self._lock:
            self.bytes[direction] += int(nbytes)
            self.busy[direction] += float(seconds)
            self.requests[direction] += 1

    def cycle_done(self) -> None:
        with self._lock:
            self.cycles += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {"bytes": dict(self.bytes), "busy": dict(self.busy),
                    "requests": dict(self.requests), "cycles": self.cycles}


def _timed(meter: Meter, direction: str, nbytes: int, call: Callable[[], None]) -> None:
    start = time.perf_counter()
    call()
    meter.add(direction, nbytes, time.perf_counter() - start)


def write_loop(spill: SpillFile, pattern: ChunkPattern, buf: Any, meter: Meter,
               stop: threading.Event, state: Dict[str, int]) -> None:
    """Sequential writes, wrapping; the pass number grows by one per wrap (prefill is 0).
    ``state`` holds the next offset and its pass, for the read-back after the arm."""
    size = pattern.chunk_bytes
    state.update(offset=0, pass_no=1)
    while not stop.is_set():
        offset, pass_no = state["offset"], state["pass_no"]
        pattern.stamp(buf, offset, pass_no)
        _timed(meter, "write", size, lambda: spill.write(offset, buf))
        if offset + size >= spill.size:
            state.update(offset=0, pass_no=pass_no + 1)
        else:
            state["offset"] = offset + size


def verify_written(spill: SpillFile, pattern: ChunkPattern, buf: Any, state: Dict[str, int],
                   *, body_every: int) -> int:
    """Untimed read-back after a ``write`` arm: chunks before ``state["offset"]`` carry
    ``pass_no``, the rest ``pass_no - 1``. Returns the chunks checked."""
    checked = 0
    for index, offset in enumerate(range(0, spill.size, pattern.chunk_bytes)):
        spill.read(offset, buf)
        expected = state["pass_no"] - (0 if offset < state["offset"] else 1)
        pattern.verify(buf, offset, expected, body=index % body_every == 0)
        checked += 1
    return checked


def read_loop(spill: SpillFile, pattern: ChunkPattern, buf: Any, meter: Meter,
              stop: threading.Event, *, body_every: int) -> None:
    """Sequential reads of the prefilled file (pass 0), wrapping, every chunk verified."""
    offset, index, size = 0, 0, pattern.chunk_bytes
    while not stop.is_set():
        _timed(meter, "read", size, lambda: spill.read(offset, buf))
        pattern.verify(buf, offset, 0, body=index % body_every == 0)
        offset, index = offset + size, index + 1
        if offset >= spill.size:
            offset = 0


def cycle_loop(spill: SpillFile, pattern: ChunkPattern, wbuf: Any, rbuf: Any, meter: Meter,
               stop: threading.Event, *, layers: int, body_every: int) -> None:
    """The G2 step: layers 0..L-1 written (pass c), then read back L-1..0 and verified."""
    size = pattern.chunk_bytes
    per_layer = spill.size // layers // size
    pass_no, index = 0, 0
    while not stop.is_set():
        pass_no += 1
        for layer in range(layers):
            for j in range(per_layer):
                if stop.is_set():
                    return
                offset = (layer * per_layer + j) * size
                pattern.stamp(wbuf, offset, pass_no)
                _timed(meter, "write", size, lambda: spill.write(offset, wbuf))
        for layer in reversed(range(layers)):
            for j in range(per_layer):
                if stop.is_set():
                    return
                offset = (layer * per_layer + j) * size
                _timed(meter, "read", size, lambda: spill.read(offset, rbuf))
                pattern.verify(rbuf, offset, pass_no, body=index % body_every == 0)
                index += 1
        meter.cycle_done()


def weights_loop(handle: io.FileIO, size: int, buf: Any, meter: Meter,
                 stop: threading.Event, *, rate_Bps: Optional[float] = None) -> None:  # noqa: N803
    """Sequential reads of the weight fixture, wrapping, optionally paced to ``rate_Bps``."""
    chunk = buf.numel()
    offset, done, start = 0, 0, time.monotonic()
    while not stop.is_set():
        _timed(meter, "weight", chunk, lambda: _read_all(handle, offset, buf))
        offset, done = offset + chunk, done + chunk
        if offset >= size:
            offset = 0
        if rate_Bps:
            delay = start + done / rate_Bps - time.monotonic()
            if delay > 0:
                stop.wait(delay)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def source_hashes() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for name in SOURCE_FILES:
        path = ROOT / name
        out[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return out


def versions() -> Dict[str, str]:
    out = {"python": sys.version, "platform": platform.platform()}
    for package in ("torch", "psutil"):
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = "not installed"
    return out


def git_head() -> Optional[str]:
    try:
        done = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def soup_cli_file() -> Optional[str]:
    try:
        import soup_cli
    except ImportError:
        return None
    return os.path.realpath(soup_cli.__file__)


# ---------------------------------------------------------------------------
# one arm
# ---------------------------------------------------------------------------
def _verify_every(args: argparse.Namespace) -> int:
    return 1 if args.verify == "full" else max(1, int(args.verify_every))


def _sample_row(start: float, previous: float, meter: Meter, disk_keys: Sequence[str]
                ) -> Dict[str, Any]:
    import psutil

    now = time.monotonic()
    snap = meter.snapshot()
    row: Dict[str, Any] = {"elapsed_s": now - start, "interval_s": now - previous,
                           "unix_s": time.time()}
    for direction in DIRECTIONS:
        row[f"total_{direction}_bytes"] = snap["bytes"][direction]
        row[f"{direction}_busy_s"] = snap["busy"][direction]
        row[f"total_{direction}_requests"] = snap["requests"][direction]
    row["cycles"] = snap["cycles"]
    row["disk"] = disk_counters(disk_keys)
    row["ram_available_bytes"] = int(psutil.virtual_memory().available)
    row["temperature_c"] = {key: drive_temperature(key) for key in disk_keys}
    return row


def _add_deltas(row: Dict[str, Any], before: Optional[Dict[str, Any]]) -> None:
    """Per-window bytes and requests: the JSONL is readable without differencing."""
    for direction in DIRECTIONS:
        prior = before[f"total_{direction}_bytes"] if before else 0
        row[f"{direction}_bytes"] = row[f"total_{direction}_bytes"] - prior
    total = sum(row[f"total_{d}_requests"] for d in DIRECTIONS)
    prior_total = sum(before[f"total_{d}_requests"] for d in DIRECTIONS) if before else 0
    row["requests"] = total - prior_total


def _arm_watch(label: str, model_path: Optional[str]) -> arm_validity.ArmWatch:
    return arm_validity.ArmWatch(arm=label, round=0, run=0, model_path=model_path,
                                 gpu_clock=lambda: None, gpu_memory_used=lambda: None,
                                 gpu_processes=lambda: None)


def run_arm(args: argparse.Namespace, out_dir: str, *,
            watch_factory: Callable[[str, Optional[str]], Any] = _arm_watch,
            ac_power: Callable[[], Optional[bool]] = l2l_box.on_ac_power) -> Dict[str, Any]:
    """One measured arm into ``out_dir`` (new): ``samples.jsonl`` and ``arm.json``.

    The record is written whatever happens, Ctrl-C included, and the spill file is
    removed on every path unless ``--keep``.
    """
    import psutil

    os.makedirs(out_dir, exist_ok=False)
    mode, smoke, direct = args.mode, bool(args.smoke), not args.buffered
    record: Dict[str, Any] = {
        "kind": "SMOKE, excluded" if smoke else "sustained arm",
        "status": "incomplete", "mode": mode, "smoke": smoke, "direct": direct,
        "backend": "unbuffered" if direct else "buffered (smoke only)",
        "label": args.label, "started_unix_s": time.time(),
        "config": {k: v for k, v in sorted(vars(args).items()) if k != "func"},
        "sources": source_hashes(), "versions": versions(), "git_head": git_head(),
        "soup_cli_file": soup_cli_file(), "ram_total_bytes": int(psutil.virtual_memory().total),
        "box_before": l2l_box.box_stamp(), "host": host_identity(), "error": None,
        "error_code": None,
    }
    record["platform"] = record["versions"]["platform"]
    meter, stop = Meter(), threading.Event()
    spill: Optional[SpillFile] = None
    weight_handle: Optional[io.FileIO] = None
    threads: List[threading.Thread] = []
    thread_errors: List[BaseException] = []
    watch = None
    rows: List[Dict[str, Any]] = []
    pattern: Optional[ChunkPattern] = None
    write_state: Dict[str, int] = {}
    try:
        check_backend(smoke=smoke, buffered=args.buffered)
        uses_spill, uses_weights = mode in SPILL_MODES, mode in WEIGHT_MODES
        overrides = []
        geometry: Dict[str, int] = {}
        if uses_spill:
            geometry.update(check_geometry(spill_bytes=args.spill_bytes,
                                           chunk_bytes=args.chunk_bytes, layers=args.layers))
        if uses_weights:
            geometry["weight_chunk_bytes"] = args.weight_chunk_bytes
        record["geometry"] = geometry
        spill_volume = weight_volume = None
        if uses_spill:
            os.makedirs(args.spill_dir, exist_ok=True)
            check_no_links(args.spill_dir)
            spill_volume = volume_info(args.spill_dir)
            if check_scope(spill_volume, accept_nonlocal=args.accept_nonlocal):
                overrides.append(f"spill volume {spill_volume['scope']} accepted by override")
            record["spill_space"] = free_space_check(args.spill_dir, args.spill_bytes,
                                                     args.reserve_bytes)
        if uses_weights:
            if not args.weights:
                raise FixtureError(f"--mode {mode} needs --weights (see prepare-weights)")
            cert = check_fixture(args.weights, weight_chunk_bytes=args.weight_chunk_bytes,
                                 rehash=args.rehash)
            record["weight_fixture"] = cert
            weight_volume = volume_info(os.path.dirname(os.path.abspath(args.weights)))
            if check_scope(weight_volume, accept_nonlocal=args.accept_nonlocal):
                overrides.append(f"weight volume {weight_volume['scope']} accepted by override")
        record["volumes"] = {
            "spill": spill_volume, "weights": weight_volume,
            "same_physical_disk": same_disk(spill_volume, weight_volume),
            "note": "physical disks as the OS reported them; None means unknown, and two "
                    "paths are never taken for two disks",
        }
        record["scope_overrides"] = overrides
        record["stop_temperature_c"] = args.stop_temperature
        disk_keys = sorted({v["disk"]["key"] for v in (spill_volume, weight_volume)
                            if v and v["disk"].get("key")})
        record["disk_keys"] = disk_keys
        record["fixture_bytes"] = {"spill": args.spill_bytes if uses_spill else None,
                                   "weights": (record["weight_fixture"]["size_bytes"]
                                               if uses_weights else None)}
        sizes = [s for s in record["fixture_bytes"].values() if s is not None]
        record["fixture_exceeds_ram"] = bool(sizes) and all(
            s > record["ram_total_bytes"] for s in sizes)
        if uses_spill:
            pattern = ChunkPattern(args.chunk_bytes, args.seed)
            wbuf, rbuf = pattern.buffer(), pattern.buffer()
            spill_path = os.path.join(os.path.realpath(args.spill_dir),
                                      spill_name(args.spill_tag or uuid.uuid4().hex))
            if not _contained(spill_path, args.spill_dir):
                raise FixtureError(f"{spill_path} escapes {args.spill_dir}")
            spill = SpillFile(spill_path, args.spill_bytes, direct=direct, keep=args.keep)
            record["spill_file"] = {"path": spill_path, "size_bytes": args.spill_bytes}
            record["prefill_s"] = prefill(spill, pattern, wbuf, stop)
        if uses_weights:
            weight_handle = open_weights(os.path.abspath(args.weights), direct=direct)
            weight_buf = l2l_activations.aligned_buffer(args.weight_chunk_bytes, pin=False)
        body_every = _verify_every(args)
        record["verify"] = {"mode": args.verify, "body_every_chunks": body_every,
                            "stamps": "every chunk read"}

        def worker(target: Callable[[], None]) -> None:
            try:
                target()
            except BaseException as exc:  # noqa: BLE001 - the arm fails with it
                thread_errors.append(exc)
                stop.set()

        targets: List[Callable[[], None]] = []
        if mode == "write":
            targets.append(lambda: write_loop(spill, pattern, wbuf, meter, stop, write_state))
        elif mode == "read":
            targets.append(lambda: read_loop(spill, pattern, rbuf, meter, stop,
                                             body_every=body_every))
        elif mode in ("cycle", "cycle+weights"):
            targets.append(lambda: cycle_loop(spill, pattern, wbuf, rbuf, meter, stop,
                                              layers=args.layers, body_every=body_every))
        if uses_weights:
            targets.append(lambda: weights_loop(weight_handle,
                                                record["weight_fixture"]["size_bytes"],
                                                weight_buf, meter, stop,
                                                rate_Bps=args.weight_rate))
        model_path = (os.path.dirname(os.path.abspath(args.weights)) if uses_weights
                      else os.path.realpath(args.spill_dir))
        record["temperature_sensors"] = {key: drive_temperature(key) is not None
                                         for key in disk_keys}
        watch = watch_factory(f"g2-disk-{mode}", model_path)
        watch.start()
        record["disk_before"] = disk_counters(disk_keys)
        record["ac_power_samples"] = [ac_power()]
        samples_path = os.path.join(out_dir, "samples.jsonl")
        with open(samples_path, "x", encoding="utf-8", newline="\n") as log:
            start = previous = time.monotonic()
            for target in targets:
                thread = threading.Thread(target=worker, args=(target,), daemon=True)
                threads.append(thread)
                thread.start()
            deadline = start + args.warmup + args.duration
            while not stop.is_set():
                # A failed I/O thread sets ``stop``: one last sample, then the loop ends.
                stop.wait(min(args.window, max(0.0, deadline - time.monotonic())))
                row = _sample_row(start, previous, meter, disk_keys)
                _add_deltas(row, rows[-1] if rows else None)
                rows.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
                if row["interval_s"] > max(5.0, MAX_GAP_WINDOWS * args.window):
                    raise RuntimeError(f"sample gap {row['interval_s']:.1f} s")
                hottest = max((t for t in row["temperature_c"].values() if t is not None),
                              default=None)
                if hottest is not None and hottest >= args.stop_temperature:
                    raise RuntimeError(f"drive temperature {hottest} C reached the stop "
                                       f"{args.stop_temperature} C")
                previous = start + row["elapsed_s"]
                if len(rows) % max(1, int(10 / args.window)) == 0:
                    record["ac_power_samples"].append(ac_power())
                # Decided on the row's own time: a clock read after the sample could pass
                # the deadline while the row ends a hair before it, losing the last window.
                if row["elapsed_s"] >= args.warmup + args.duration:
                    break
            stop.set()
            for thread in threads:
                thread.join(timeout=30)
            if any(t.is_alive() for t in threads):
                raise RuntimeError("an I/O thread did not stop within 30 s")
        if thread_errors:
            raise thread_errors[0]
        record["ac_power_samples"].append(ac_power())
        record["disk_after"] = disk_counters(disk_keys)
        if mode == "write":
            record["verify"]["write_readback_chunks"] = verify_written(
                spill, pattern, rbuf, write_state, body_every=body_every)
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - kept as evidence
        stop.set()
        record["error"] = repr(exc)
        record["error_code"] = getattr(exc, "code", None)
        if is_enospc(exc) and not isinstance(exc, SpillDiskFullError):
            record["error_code"] = SpillDiskFullError.code
        record["interrupted"] = isinstance(exc, KeyboardInterrupt)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=30)
        if watch is not None:
            try:
                watch.stop()
            except Exception as exc:  # noqa: BLE001 - an F1 failure invalidates, not crashes
                record["error"] = record["error"] or f"F1 stop failed: {exc!r}"
        record["F1"] = getattr(watch, "record", None)
        if weight_handle is not None:
            weight_handle.close()
        if spill is not None:
            record["spill_file"] = spill.close()
        if pattern is not None:
            record.setdefault("verify", {}).update(stamp_checks=pattern.stamp_checks,
                                                   body_checks=pattern.body_checks)
        record["box_after"] = l2l_box.box_stamp()
        record["ended_unix_s"] = time.time()
        final = meter.snapshot()
        record["requests"], record["cycles"] = final["requests"], final["cycles"]
        try:
            # A late final sample still brackets the deadline for interpolation.
            summary = summarize(rows, warmup_s=args.warmup, window_s=args.window,
                                directions=MODE_DIRECTIONS[mode],
                                end_s=args.warmup + args.duration) if rows else None
        except ValueError as exc:
            summary = None
            record["error"] = record["error"] or f"summary: {exc}"
        record["summary"] = summary
        record["measured_s"] = summary["measured_s"] if summary else 0.0
        power = record.get("ac_power_samples") or [ac_power()]
        ac_ok: Optional[bool] = (True if power and all(p is True for p in power)
                                 else (False if any(p is False for p in power) else None))
        record["ac_power"] = ac_ok
        f1 = record["F1"] or {}
        left = (record.get("spill_file") or {}).get("left_behind")
        status, reasons = decide_status(
            smoke=smoke, direct=direct, error=record["error"], f1_outcome=f1.get("outcome"),
            n_windows=summary["n_windows"] if summary else 0, ac_power=ac_ok,
            sources_unchanged=source_hashes() == record["sources"],
            scope_override=bool(record.get("scope_overrides")),
            problems=[f"spill file left behind: {left}"] if left else [])
        record["status"], record["invalid_reasons"] = status, reasons
        record["arm_path"] = os.path.realpath(os.path.join(out_dir, "arm.json"))
        record["sustained"] = bool(
            status == "candidate" and record.get("fixture_exceeds_ram")
            and args.warmup >= SUSTAINED_WARMUP_S
            and record["measured_s"] >= SUSTAINED_MEASURED_S)
        if not record["sustained"]:
            record["not_sustained_because"] = (
                "smoke arm" if smoke else
                f"needs candidate status, fixtures larger than RAM, warmup >= "
                f"{SUSTAINED_WARMUP_S:.0f} s and >= {SUSTAINED_MEASURED_S:.0f} s measured")
        _write_json(os.path.join(out_dir, "arm.json"), record)
    return record


# ---------------------------------------------------------------------------
# rates for the break-even (g2_rule V9 / analyse_breakeven)
# ---------------------------------------------------------------------------
def export_rates(record: Dict[str, Any], source: Optional[str] = None) -> Dict[str, Any]:
    """One arm record as the dict ``g2_rule`` consumes. Rates are busy means in bytes/s
    (bytes over seconds inside that direction's I/O calls, full post-warmup windows).

    ``weight_ref_Bps`` comes from a ``weights`` arm, ``weight_with_spill_Bps`` from a
    ``cycle+weights`` arm; each is None for every other mode. ``fixture_exceeds_ram`` is
    True only when every fixture the mode uses (spill file, weight fixture) is larger
    than the box's physical RAM.
    """
    mode = record.get("mode")
    summary = record.get("summary")
    weight = _rate(summary, "weight")
    sizes = record.get("fixture_bytes") or {}
    return {
        "write_Bps": _rate(summary, "write"),
        "read_Bps": _rate(summary, "read"),
        "weight_ref_Bps": weight if mode == "weights" else None,
        "weight_with_spill_Bps": weight if mode == "cycle+weights" else None,
        "source": source or record.get("arm_path"),
        "mode": mode,
        "status": record.get("status"),
        "direct": bool(record.get("direct")),
        "measured_s": float(record.get("measured_s") or 0.0),
        "fixture_bytes": {"spill": sizes.get("spill"), "weights": sizes.get("weights")},
        "ram_total_bytes": record.get("ram_total_bytes"),
        "fixture_exceeds_ram": bool(record.get("fixture_exceeds_ram")),
        # where the rates were measured: g2_rule binds them to the model profile by these
        "host": record.get("host"),
        "spill_disk": (((record.get("volumes") or {}).get("spill") or {}).get("disk") or {})
        .get("id"),
        "weight_disk": (((record.get("volumes") or {}).get("weights") or {}).get("disk") or {})
        .get("id"),
    }


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def export_session(session_dir: str) -> Dict[str, Any]:
    """Rates for the break-even from a finished ``session`` directory.

    ``write_Bps``/``read_Bps``: the mean over the ``cycle+weights`` arms' spill rates (the
    spill under weight-read contention) when any exist, else ``write`` / ``read`` arms.
    ``weight_ref_Bps``: mean over ``weights`` arms; ``weight_with_spill_Bps``: mean over
    ``cycle+weights`` arms. Means use only arms that did not end ``invalid`` (a failed
    arm's partial rate is never an input); ``means_from`` names them. Every arm in the
    manifest stays under ``arms`` with its own status, so g2_rule V9 sees the failures.
    """
    directory = os.path.realpath(session_dir)
    manifest_path = os.path.join(directory, "manifest.json")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    arms: List[Dict[str, Any]] = []
    for entry in manifest.get("arms", []):
        path = os.path.realpath(os.path.join(directory, entry["dir"], "arm.json"))
        if os.path.commonpath([path, directory]) != directory:
            raise ValueError(f"manifest arm {entry['dir']!r} escapes {directory}")
        if not os.path.exists(path):
            arms.append({"write_Bps": None, "read_Bps": None, "weight_ref_Bps": None,
                         "weight_with_spill_Bps": None, "source": path,
                         "mode": entry.get("mode"), "status": "invalid", "direct": False,
                         "measured_s": 0.0, "fixture_bytes": {"spill": None, "weights": None},
                         "ram_total_bytes": None, "fixture_exceeds_ram": False,
                         "missing": "no arm.json"})
            continue
        with open(path, encoding="utf-8") as handle:
            exported = export_rates(json.load(handle), source=path)
        failed = entry.get("orchestrator_error") or entry.get("exit_code") not in (None, 0)
        if failed and exported["status"] != "invalid":
            # A child that was killed or exited non-zero is never an input, whatever
            # its own record says.
            exported["status"] = "invalid"
            exported["session_problem"] = (entry.get("orchestrator_error")
                                           or f"child exit code {entry.get('exit_code')}")
        arms.append(exported)

    def of(mode: str) -> List[Dict[str, Any]]:
        return [arm for arm in arms if arm["mode"] == mode and arm["status"] != "invalid"]

    contended = of("cycle+weights")
    write_from = contended or of("write")
    read_from = contended or of("read")
    return {
        "write_Bps": _mean([a["write_Bps"] for a in write_from]),
        "read_Bps": _mean([a["read_Bps"] for a in read_from]),
        "weight_ref_Bps": _mean([a["weight_ref_Bps"] for a in of("weights")]),
        "weight_with_spill_Bps": _mean([a["weight_with_spill_Bps"] for a in contended]),
        "spill_rates_from": "cycle+weights" if contended else "write/read",
        "means_from": {"write_Bps": [a["source"] for a in write_from],
                       "read_Bps": [a["source"] for a in read_from],
                       "weight_ref_Bps": [a["source"] for a in of("weights")],
                       "weight_with_spill_Bps": [a["source"] for a in contended]},
        "rate_kind": "busy: bytes over seconds inside the direction's I/O calls, full "
                     "windows after warmup",
        "arms": arms,
        "source": manifest_path,
    }


# ---------------------------------------------------------------------------
# ABBA session: fresh process per arm
# ---------------------------------------------------------------------------
def abba_order(modes: Sequence[str]) -> List[str]:
    """A-B-B-A over ``modes``: the list, then the list reversed (C4 / METHOD.md block)."""
    if not modes:
        raise ValueError("no modes")
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise ValueError(f"unknown modes: {unknown}")
    return list(modes) + list(reversed(modes))


def run_session(modes: Sequence[str], out: str, arm_args: Sequence[str], *,
                timeout_s: float, python: str = sys.executable, script: Optional[str] = None,
                spill_dir: Optional[str] = None, keep: bool = False) -> Dict[str, Any]:
    """Every mode A-B-B-A as fresh ``run`` subprocesses of ``script`` (this file) into one
    new directory, with a manifest written before the first child and after each one. A
    failed child does not stop the session: its record stays, marked, and the next arm
    runs. A child killed before its own cleanup (timeout) leaves its spill file; the
    session removes it."""
    order = abba_order(modes)
    os.makedirs(out, exist_ok=False)
    manifest: Dict[str, Any] = {
        "kind": "g2_disk ABBA session", "created_unix_s": time.time(),
        "modes": list(modes), "order": order, "arm_args": list(arm_args),
        "python": python, "sources": source_hashes(), "versions": versions(),
        "git_head": git_head(), "soup_cli_file": soup_cli_file(), "arms": [],
        "status": "running",
    }
    path = os.path.join(out, "manifest.json")
    _write_json(path, manifest, exclusive=True)
    try:
        for position, mode in enumerate(order):
            name = f"p{position}_{mode.replace('+', '_')}"
            arm_dir = os.path.join(out, name)
            # The tag, not the pid: a venv launcher's pid is not the interpreter's.
            tag = f"{os.path.basename(os.path.realpath(out))}-{name}-{uuid.uuid4().hex[:8]}"
            command = [python, script or os.path.realpath(__file__), "run", "--mode", mode,
                       "--out", arm_dir, "--spill-tag", tag, *arm_args]
            entry: Dict[str, Any] = {"position": position, "mode": mode, "dir": name,
                                     "command": command, "started_unix_s": time.time()}
            manifest["arms"].append(entry)
            _write_json(path, manifest)
            log_path = os.path.join(out, f"{name}.log")
            with open(log_path, "x", encoding="utf-8") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    entry["exit_code"] = child.wait(timeout=timeout_s)
                except BaseException as exc:
                    child.terminate()
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=30)
                    entry["exit_code"] = child.returncode
                    entry["orchestrator_error"] = repr(exc)
                    if not isinstance(exc, subprocess.TimeoutExpired):
                        raise
            entry["ended_unix_s"] = time.time()
            entry["log"] = os.path.basename(log_path)
            if spill_dir and not keep:
                leftover = os.path.join(os.path.realpath(spill_dir), spill_name(tag))
                if os.path.exists(leftover):
                    try:
                        os.remove(leftover)
                        entry["spill_removed_by_session"] = leftover
                    except OSError as exc:
                        entry["spill_left_behind"] = f"{leftover}: {exc}"
            arm_json = os.path.join(arm_dir, "arm.json")
            if os.path.exists(arm_json):
                with open(arm_json, encoding="utf-8") as handle:
                    entry["status"] = json.load(handle).get("status")
            else:
                entry["status"] = "invalid"
                entry["missing"] = "no arm.json"
            _write_json(path, manifest)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "interrupted"
        manifest["error"] = repr(exc)
        raise
    finally:
        manifest["ended_unix_s"] = time.time()
        try:
            _write_json(os.path.join(out, "rates.json"), export_session(out))
        except (OSError, ValueError, KeyError) as exc:
            manifest["rates_error"] = repr(exc)
        _write_json(path, manifest)
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _add_arm_args(parser: argparse.ArgumentParser) -> List[argparse.Action]:
    """The arm options shared by ``run`` and ``session``; returns their actions so a
    session can pass them to every child unchanged."""
    actions: List[argparse.Action] = []

    def add(*flags: str, **options: Any) -> None:
        actions.append(parser.add_argument(*flags, **options))

    add("--spill-dir", help="directory for the spill file (created)")
    add("--spill-bytes", type=parse_bytes, default=512 * MIB,
        help="spill file size; > host RAM for a sustained arm")
    add("--chunk-bytes", type=parse_bytes, default=8 * MIB,
        help="spill request size (sector multiple)")
    add("--layers", type=int, default=32, help="cycle mode: layers the spill is split into")
    add("--weights", help="weight fixture from prepare-weights")
    add("--weight-chunk-bytes", type=parse_bytes, default=8 * MIB)
    add("--weight-rate", type=parse_bytes, default=None,
        help="pace the weight stream to this many bytes/s (default: unpaced)")
    add("--duration", type=float, default=1800.0, help="measured seconds after warmup")
    add("--warmup", type=float, default=600.0, help="excluded seconds")
    add("--window", type=float, default=1.0, help="sample window seconds")
    add("--verify", choices=("sample", "full"), default="sample",
        help="stamps of every chunk read are checked; bodies of every chunk (full) or "
             "every --verify-every-th (sample)")
    add("--verify-every", type=int, default=16)
    add("--seed", type=int, default=20261022)
    add("--reserve-bytes", type=parse_bytes_or_zero, default=20 * GIB,
        help="free space that must remain beside the spill file")
    add("--keep", action="store_true", help="keep the spill file")
    add("--smoke", action="store_true", help="label the arm 'SMOKE, excluded'; never evidence")
    add("--buffered", action="store_true", help="page-cache I/O; only with --smoke")
    add("--accept-nonlocal", action="store_true",
        help="run on a network/refused volume, labelled, never a candidate")
    add("--rehash", action="store_true", help="re-hash the whole weight fixture before the arm")
    add("--label", default="")
    add("--stop-temperature", type=float, default=70.0,
        help="stop the arm (invalid) when a mapped drive sensor reaches this many C "
             "(C4 Linux stop); no mapped sensor records temperature as unknown")
    return actions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare-weights", help="create a certified weight fixture")
    prep.add_argument("--path", required=True, help="new fixture file (never overwritten)")
    prep.add_argument("--bytes", type=parse_bytes, required=True,
                      help="fixture size; > host RAM for a sustained arm")
    prep.add_argument("--reserve-bytes", type=parse_bytes_or_zero, default=20 * GIB)
    run = sub.add_parser("run", help="one arm in this process")
    run.add_argument("--mode", choices=MODES, required=True)
    run.add_argument("--out", required=True, help="new arm directory")
    run.add_argument("--spill-tag", default=None,
                     help="spill file name tag (a session sets it, to clean up after a kill)")
    _add_arm_args(run)
    session = sub.add_parser("session", help="modes A-B-B-A, fresh process per arm")
    session.add_argument("--modes", required=True,
                         help="comma list, e.g. write,read,cycle,weights,cycle+weights")
    session.add_argument("--out", required=True, help="new session directory")
    session.add_argument("--arm-timeout", type=float, default=None,
                         help="seconds per child (default: warmup + duration + 900)")
    _add_arm_args(session)
    export = sub.add_parser("export", help="print export_session of a session directory")
    export.add_argument("session_dir")
    return parser


def _arm_argv(args: argparse.Namespace) -> List[str]:
    """The session's arm options, passed through to every child unchanged."""
    argv: List[str] = []
    for action in _add_arm_args(argparse.ArgumentParser(add_help=False)):
        value = getattr(args, action.dest)
        flag = action.option_strings[0]
        if action.nargs == 0:  # store_true
            if value:
                argv.append(flag)
        elif value is not None:
            argv += [flag, str(value)]
    return argv


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-weights":
        try:
            cert = prepare_weights(args.path, args.bytes, reserve_bytes=args.reserve_bytes)
        except (OSError, ValueError, G2DiskError) as exc:
            print(f"prepare-weights failed: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(cert, indent=2))
        return 0
    if args.command == "export":
        print(json.dumps(export_session(args.session_dir), indent=2))
        return 0
    if args.command == "session":
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        timeout = args.arm_timeout or (args.warmup + args.duration + 900.0)
        manifest = run_session(modes, args.out, _arm_argv(args), timeout_s=timeout,
                               spill_dir=args.spill_dir, keep=args.keep)
        statuses = [arm.get("status") for arm in manifest["arms"]]
        print(json.dumps({"session": os.path.realpath(args.out), "order": manifest["order"],
                          "statuses": statuses}, indent=2))
        good = {"smoke_passed"} if args.smoke else {"candidate"}
        return 0 if all(s in good for s in statuses) else 1
    if args.mode in SPILL_MODES and not args.spill_dir:
        print(f"--mode {args.mode} needs --spill-dir", file=sys.stderr)
        return 2
    record = run_arm(args, args.out)
    summary = record.get("summary") or {}
    brief = {d: (summary.get(d) or {}).get("busy", {}).get("mean_Bps") for d in DIRECTIONS
             if summary.get(d)}
    print(json.dumps({"status": record["status"], "kind": record["kind"],
                      "busy_mean_Bps": brief, "measured_s": record["measured_s"],
                      "error": record["error"], "invalid_reasons": record["invalid_reasons"],
                      "arm": record["arm_path"]}, indent=2))
    return 0 if record["status"] in ("smoke_passed", "candidate") else 1


if __name__ == "__main__":
    def _interrupted(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(f"signal {signum}")

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _interrupted)
    raise SystemExit(main())
