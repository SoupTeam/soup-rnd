#!/usr/bin/env python3
"""A3 stage 2 machine admission: does the spill volume keep up with this session's spill?

Before a session (benchmarks/a3-spill-gpu-check.md §5) the spill volume runs the store's
I/O pattern through the store's own unbuffered ``DirectSpillFile``, on a schedule set by
two files measured on the same box:

* ``--plan``: the calibration child's JSON (``a3_spill_stand.py child --arm calibration``).
  Its ``spill_plan`` gives n_layers, k and slot_bytes.
* ``--a``: a no-spill cost A child (``a3_spill_stand.py child --arm A``, 2 warm-up and 5
  timed optimizer steps). Its ``value`` is the median timed step.

One CYCLE is one spill file of the session's step: ``spill_layers`` (the stand's
``TARGET_SPILL``, 16) x k chunks of slot_bytes. It is written in the forward's order and
read back in the backward's (layers descending). Then:

* cycle_bytes = spill_layers x k x slot_bytes
* D = cycle_bytes / A.value  (spill bytes written per step / the A step)
* V = cycle_bytes x (7 steps x 2 B arms x 12 blocks + 3 steps x 4 arms)
  (cost B arms at 2 warm-up + 5 timed steps, at most 12 A-B-B-A blocks; l2l_spill,
  sigint, kill9 and kill9_next at the correctness rows' 3 optimizer steps each)
* period = cycle_bytes / (2 x D)
* N = max(20, ceil(V / cycle_bytes)),  duration = N x period

Exactly N cycles are scheduled. Cycle i starts at max(t0 + i x period, the end of cycle
i - 1). Its lateness is its end minus t0 + (i + 1) x period. The machine is admitted iff
all N cycles ran, no cycle ended more than 5 s late, every chunk read back is the one its
cycle wrote there (each 4 KiB sector carries its cycle and file offset; then every byte),
and no test file is left.

The decision reads those numbers only. The record carries D, V, N, the period, the cycle
bytes and their inputs: the A record's value and SHA-256, and the plan's numbers and
SHA-256. Description only, never part of the decision: the per-second log, 10 s windows
against their median, and the zero fill of the file when it is created (one more cycle of
writes). ``a3_spill_rule`` reads nothing of the admission.

Copied from A4's ``a4_admission.py`` and ``a4_box.py`` at d6f7f733 (the A4 PR branch), with
its constants turned into the parameters above. Nothing of A4 is imported or changed.

Usage::

    python benchmarks/harness/a3_admission.py --spill-dir /mnt/nvme/a3-spill/admission \\
        --plan .soup/a3-prep.json --a .soup/a3-demand-A.json \\
        --out .soup/a3-admission.json --cycles .soup/a3-admission-cycles.csv

Exit 0 admitted, 4 refused, 3 the volume itself refused (tmpfs, overlay, network, no direct
I/O), 2 refused inputs or an existing output. torch is imported inside functions.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import a3_spill_rule as rule  # noqa: E402
import a3_spill_stand as stand  # noqa: E402
import arm_validity  # noqa: E402

#: Fixed before any A3 stage-2 box (the owner's protocol change, 2026-10-27).
PROTOCOL = "a3-admission-v1"
MAX_LATENESS_S = 5.0
#: The schedule writes, and reads back, each cycle at this multiple of the demand D.
MARGIN = 2.0
MIN_CYCLES = 20
#: The session's spill outside the cost blocks: these arms, at the correctness rows'
#: optimizer steps each (10 rows at k 4: windows of 4, 4 and 2).
SPILL_ARMS = ("l2l_spill", "sigint", "kill9", "kill9_next")
SPILL_ARM_STEPS = -(-stand.ROWS // stand.K)
#: A cost B arm's optimizer steps, and the B arms of one A-B-B-A block.
COST_ARM_STEPS = stand.COST_WARMUP + stand.COST_STEPS
B_ARMS_PER_BLOCK = arm_validity.BLOCK_ORDER.count("B")
#: Distinct random chunks rotated through the writes. With the per-sector stamps no two
#: sectors written in one run are equal, so a deduplicating volume cannot skip the work.
DISTINCT_CHUNKS = 16
#: The stamp: (cycle, file offset) in the first two int64 words of every sector.
SECTOR_BYTES = 4096
#: Description only: the windows of the stability summary and the fraction of their median.
WINDOW_S = 10.0
FRACTION = 0.5

Op = Tuple[str, float, float, int]  # (kind "write"|"read", start, end, bytes)
Cycle = Tuple[float, float, float, float, int]  # planned, start, end, lateness, bad chunks


class InputError(ValueError):
    """``--plan`` or ``--a`` cannot set the schedule; the message says why."""


# ---------------------------------------------------------------------------
# the inputs and the schedule (pure)
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _record(path: Path, what: str) -> Dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InputError(f"{what} {path} cannot be read: {exc}") from exc
    if not isinstance(data, dict):
        raise InputError(f"{what} {path} is not a JSON object")
    return data


def _positive_int(plan: Dict[str, Any], key: str, what: str) -> int:
    value = plan.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise InputError(f"{what}: spill_plan.{key} is {value!r}, not a positive integer")
    return value


def load_inputs(plan_path: Path, a_path: Path) -> Dict[str, Any]:
    """The plan's numbers and the A arm's step, each with its file and SHA-256. Refuses a
    CPU record, a failed child, an A arm that spilled or did not time its 5 steps, and an
    A arm whose store had another shape than the plan's."""
    plan_rec, a_rec = _record(plan_path, "--plan"), _record(a_path, "--a")
    for what, rec in (("--plan", plan_rec), ("--a", a_rec)):
        if rec.get("device") != "cuda":
            raise InputError(f"{what}: device {rec.get('device')!r}, not cuda")
        if rec.get("exit") != "normal":
            raise InputError(f"{what}: the child exited {rec.get('exit')!r} "
                             f"({rec.get('error')})")
        if not isinstance(rec.get("spill_plan"), dict):
            raise InputError(f"{what}: no spill_plan")
    plan = plan_rec["spill_plan"]
    numbers = {key: _positive_int(plan, key, "--plan")
               for key in ("n_layers", "k", "slot_bytes", "chunk_bytes")}
    if numbers["n_layers"] < stand.TARGET_SPILL:
        raise InputError(f"--plan: {numbers['n_layers']} layers, fewer than the "
                         f"{stand.TARGET_SPILL} the session spills")
    if a_rec.get("kind") != "cost" or a_rec.get("role") != "A":
        raise InputError(f"--a: arm {a_rec.get('arm')!r} is not a cost A arm")
    a_plan = a_rec["spill_plan"]
    if a_plan.get("spill_layers") != 0:
        raise InputError(f"--a: the A arm spilled {a_plan.get('spill_layers')!r} layers; D "
                         f"needs the step without spill")
    for key in ("k", "slot_bytes"):
        if a_plan.get(key) != numbers[key]:
            raise InputError(f"--a: its store has {key} {a_plan.get(key)!r}, the plan's "
                             f"{numbers[key]}")
    timed = a_rec.get("timed_step_s") or []
    value = a_rec.get("value")
    if len(timed) != stand.COST_STEPS or not isinstance(value, (int, float)) or value <= 0:
        raise InputError(f"--a: value {value!r} over {len(timed)} timed steps; "
                         f"{stand.COST_STEPS} timed steps and a positive median are needed")
    return {
        "plan": {**numbers, "pin": plan.get("pin"), "file": str(plan_path),
                 "sha256": sha256_file(plan_path)},
        "a": {"value_s": float(value), "timed_step_s": list(timed), "k": a_plan["k"],
              "slot_bytes": a_plan["slot_bytes"], "file": str(a_path),
              "sha256": sha256_file(a_path),
              "gpu": ((a_rec.get("env") or {}).get("gpu_facts") or {}).get("gpu")},
    }


def session_volume(cycle_bytes: int) -> Dict[str, Any]:
    """V: the most the session's spill arms write, in cycles of one step's spill file."""
    cost = COST_ARM_STEPS * B_ARMS_PER_BLOCK * rule.MAX_BLOCKS
    other = SPILL_ARM_STEPS * len(SPILL_ARMS)
    return {"formula": (f"cycle_bytes x ({COST_ARM_STEPS} steps x {B_ARMS_PER_BLOCK} B arms x "
                        f"{rule.MAX_BLOCKS} blocks + {SPILL_ARM_STEPS} steps x "
                        f"{len(SPILL_ARMS)} arms {list(SPILL_ARMS)})"),
            "cost_cycles": cost, "other_cycles": other,
            "bytes": int(cycle_bytes) * (cost + other)}


def schedule(*, cycle_bytes: int, a_value_s: float, volume_bytes: float) -> Dict[str, Any]:
    """D, the period, N and the duration from the cycle, the A step and V."""
    demand = cycle_bytes / a_value_s
    period = cycle_bytes / (MARGIN * demand)
    n_cycles = max(MIN_CYCLES, math.ceil(volume_bytes / cycle_bytes))
    return {"demand_bytes_per_s": demand, "period_s": period, "n_cycles": n_cycles,
            "duration_s": n_cycles * period, "margin": MARGIN, "min_cycles": MIN_CYCLES}


def admission_plan(inputs: Dict[str, Any], *, spill_layers: int = stand.TARGET_SPILL
                   ) -> Dict[str, Any]:
    plan = inputs["plan"]
    cycle_bytes = spill_layers * plan["k"] * plan["slot_bytes"]
    volume = session_volume(cycle_bytes)
    return {"spill_layers": spill_layers, "k": plan["k"], "slot_bytes": plan["slot_bytes"],
            "cycle_bytes": cycle_bytes, "volume": volume,
            **schedule(cycle_bytes=cycle_bytes, a_value_s=inputs["a"]["value_s"],
                       volume_bytes=volume["bytes"]),
            "formulas": {
                "cycle_bytes": "spill_layers x k x slot_bytes",
                "demand_bytes_per_s": "cycle_bytes / A.value",
                "volume_bytes": volume["formula"],
                "period_s": f"cycle_bytes / ({MARGIN:g} x D)",
                "n_cycles": f"max({MIN_CYCLES}, ceil(V / cycle_bytes))",
                "duration_s": "n_cycles x period_s",
            }}


def spill_order(spill_layers: int, k: int, chunk: int) -> Tuple[List[int], List[int]]:
    """Offsets in the order ``ActivationStore`` writes them in the forward (layer, then
    micro-batch, ascending) and reads them in the backward (layers descending)."""
    write = [(layer * k + mb) * chunk for layer in range(spill_layers) for mb in range(k)]
    read = [(layer * k + mb) * chunk for layer in range(spill_layers - 1, -1, -1)
            for mb in range(k)]
    return write, read


# ---------------------------------------------------------------------------
# description only
# ---------------------------------------------------------------------------
def bins(ops: Iterable[Op], start: float, width: float, count: int) -> List[Dict[str, Any]]:
    """Per bin and direction: bytes and busy seconds, each operation shared out by its
    overlap with the bin. ``*_gb_per_s`` is bytes over busy time, None with no busy time."""
    out = [{"index": i, "t0": i * width, "write_bytes": 0.0, "write_busy_s": 0.0,
            "read_bytes": 0.0, "read_busy_s": 0.0} for i in range(count)]
    for kind, t0, t1, nbytes in ops:
        span = t1 - t0
        if span <= 0:
            continue
        first = max(0, int((t0 - start) // width))
        last = min(count - 1, int((t1 - start) // width))
        for i in range(first, last + 1):
            lo = max(t0, start + i * width)
            hi = min(t1, start + (i + 1) * width)
            if hi <= lo:
                continue
            share = (hi - lo) / span
            out[i][f"{kind}_bytes"] += nbytes * share
            out[i][f"{kind}_busy_s"] += hi - lo
    for row in out:
        for kind in ("write", "read"):
            busy = row[f"{kind}_busy_s"]
            row[f"{kind}_gb_per_s"] = row[f"{kind}_bytes"] / busy / 1e9 if busy > 0 else None
    return out


def stability(windows: Sequence[Dict[str, Any]], fraction: float = FRACTION) -> Dict[str, Any]:
    """DESCRIPTION ONLY: per direction, the median window busy rate and the windows below
    ``fraction`` of it."""
    out: Dict[str, Any] = {"description_only": True, "fraction": fraction}
    for kind in ("write", "read"):
        rates = [w[f"{kind}_gb_per_s"] for w in windows]
        measured = [r for r in rates if r is not None]
        median = statistics.median(measured) if measured else None
        out[kind] = {"median_gb_per_s": median,
                     "min_gb_per_s": min(measured) if measured else None,
                     "max_gb_per_s": max(measured) if measured else None,
                     "low": [{"index": w["index"], "gb_per_s": r}
                             for w, r in zip(windows, rates)
                             if r is not None and r < fraction * median]}
    return out


# ---------------------------------------------------------------------------
# the exercise
# ---------------------------------------------------------------------------
def _sectors(buf: Any) -> Any:
    import torch

    return buf.view(torch.int64).view(-1, SECTOR_BYTES // 8)


def stamp(buf: Any, cycle: int, offset: int) -> None:
    """Mark every sector of ``buf`` with the cycle and the file offset it is written to."""
    words = _sectors(buf)
    words[:, 0] = cycle
    words[:, 1] = offset


def matches(sink: Any, source: Any, cycle: int, offset: int) -> bool:
    """Whether ``sink``, read back from ``offset``, is what ``cycle`` wrote there: every
    sector's stamp, then every other byte against ``source``. Overwrites the sink's stamps."""
    import torch

    got, want = _sectors(sink), _sectors(source)
    if not (bool((got[:, 0] == cycle).all()) and bool((got[:, 1] == offset).all())):
        return False
    got[:, :2] = want[:, :2]
    return torch.equal(got, want)


def _default_backend(path: str, size: int) -> Any:
    import l2l_activations

    return l2l_activations.DirectSpillFile(path, size)


def _random_chunks(chunk: int, count: int) -> List[Any]:
    import l2l_activations
    import torch

    generator = torch.Generator().manual_seed(4)
    chunks = []
    for _ in range(count):
        buf = l2l_activations.aligned_buffer(chunk, pin=False)
        buf.copy_(torch.randint(0, 256, (chunk,), dtype=torch.uint8, generator=generator))
        chunks.append(buf)
    return chunks


def _aligned(chunk: int) -> Any:
    import l2l_activations

    return l2l_activations.aligned_buffer(chunk, pin=False)


def exercise(directory: str, *, n_cycles: int, period_s: float, spill_layers: int, k: int,
             chunk: int, clock: Callable[[], float] = time.perf_counter,
             sleep: Callable[[float], None] = time.sleep,
             backend: Callable[[str, int], Any] = _default_backend,
             chunks: Optional[List[Any]] = None) -> Dict[str, Any]:
    """Exactly ``n_cycles`` write-then-read cycles on the schedule.

    Cycle i waits for t0 + i x period_s (or starts at once if cycle i - 1 ended later),
    writes one spill file in the forward's order and reads it back in the backward's, every
    chunk stamped before its write and checked after its read. Every operation is timed."""
    write_offsets, read_offsets = spill_order(spill_layers, k, chunk)
    size = spill_layers * k * chunk
    path = os.path.join(directory, f"a3-admission-{os.getpid()}.bin")
    sources = chunks if chunks is not None else _random_chunks(chunk, DISTINCT_CHUNKS)
    sink = chunks[0].clone() if chunks is not None else _aligned(chunk)
    ops: List[Op] = []
    cycles: List[Cycle] = []
    mismatched = 0
    spill = backend(path, size)
    try:
        start = clock()
        for cycle in range(n_cycles):
            planned = start + cycle * period_s
            wait = planned - clock()
            if wait > 0:
                sleep(wait)
            begin = clock()
            for index, offset in enumerate(write_offsets):
                source = sources[index % len(sources)]
                stamp(source, cycle, offset)
                t0 = clock()
                spill.write(offset, source)
                ops.append(("write", t0, clock(), chunk))
            bad = 0
            for offset in read_offsets:
                t0 = clock()
                spill.read(offset, sink)
                ops.append(("read", t0, clock(), chunk))
                if not matches(sink, sources[(offset // chunk) % len(sources)], cycle, offset):
                    bad += 1
            end = clock()
            cycles.append((planned - start, begin - start, end - start,
                           end - (planned + period_s), bad))
            mismatched += bad
        finish = clock()
    finally:
        spill.close(remove=True)
    return {"start": start, "end": finish, "period_s": period_s, "ops": ops, "cycles": cycles,
            "mismatched_chunks": mismatched, "chunks_read": len(cycles) * len(read_offsets),
            "file_bytes": size, "path": path, "left_behind": os.path.exists(path)}


def judgement(run: Dict[str, Any], *, n_cycles: int,
              limit_s: float = MAX_LATENESS_S) -> Dict[str, Any]:
    """Admitted iff all ``n_cycles`` ran, none ended more than ``limit_s`` behind its
    schedule, no chunk read back other bytes than its cycle wrote, and the file is gone."""
    lateness = [row[3] for row in run["cycles"]]
    late = [i for i, value in enumerate(lateness) if value > limit_s]
    problems: List[str] = []
    if len(lateness) != n_cycles:
        problems.append(f"{len(lateness)} of {n_cycles} scheduled cycles ran")
    if late:
        problems.append(f"{len(late)} of {len(lateness)} cycle(s) ended more than {limit_s:g} s "
                        f"behind the schedule (worst {max(lateness):.2f} s; first: cycle "
                        f"{late[0]})")
    if run["mismatched_chunks"]:
        problems.append(f"{run['mismatched_chunks']} chunk(s) read back other bytes than "
                        f"their cycle wrote")
    if run["left_behind"]:
        problems.append(f"the test file was left at {run['path']}")
    return {"admitted": not problems, "limit_s": limit_s,
            "max_lateness_s": max(lateness) if lateness else None,
            "late_cycles": len(late), "first_late_cycle": late[0] if late else None,
            "problems": problems}


def assess(run: Dict[str, Any], *, n_cycles: int, window_s: float = WINDOW_S,
           limit_s: float = MAX_LATENESS_S) -> Dict[str, Any]:
    """The admission record of one ``exercise``; the log's bins cover the run."""
    elapsed = run["end"] - run["start"]
    windows = bins(run["ops"], run["start"], window_s, max(1, math.ceil(elapsed / window_s)))
    totals = {kind: sum(n for k_, _, _, n in run["ops"] if k_ == kind)
              for kind in ("write", "read")}
    return {"judgement": judgement(run, n_cycles=n_cycles, limit_s=limit_s),
            "cycles": len(run["cycles"]), "scheduled_cycles": n_cycles,
            "bytes": totals, "elapsed_s": elapsed,
            "wall_gb_per_s": {kind: totals[kind] / elapsed / 1e9 if elapsed > 0 else None
                              for kind in totals},
            "mismatched_chunks": run["mismatched_chunks"], "chunks_read": run["chunks_read"],
            "file_bytes": run["file_bytes"], "left_behind": run["left_behind"],
            "description_only": {"windows": windows, "stability": stability(windows),
                                 "zero_fill_bytes": run["file_bytes"]},
            "per_second": bins(run["ops"], run["start"], 1.0, max(1, math.ceil(elapsed))),
            "cycle_rows": run["cycles"]}


def run_admission(directory: str, inputs: Dict[str, Any], **exercise_kw: Any) -> Dict[str, Any]:
    """The schedule from ``inputs``, its exercise on ``directory``, and the decision."""
    planned = admission_plan(inputs)
    run = exercise(directory, n_cycles=planned["n_cycles"], period_s=planned["period_s"],
                   spill_layers=planned["spill_layers"], k=planned["k"],
                   chunk=planned["slot_bytes"], **exercise_kw)
    record = {"kind": "admission", "protocol": PROTOCOL, "inputs": inputs,
              "schedule": planned, "max_lateness_limit_s": MAX_LATENESS_S}
    record.update(assess(run, n_cycles=planned["n_cycles"]))
    judged = record["judgement"]
    record["admitted"] = bool(judged["admitted"])
    record["reason"] = "admitted" if record["admitted"] else "; ".join(judged["problems"])
    return record


def write_per_second(path: Path, seconds: Sequence[Dict[str, Any]]) -> None:
    lines = ["second,write_bytes,write_busy_s,write_gb_per_s,read_bytes,read_busy_s,"
             "read_gb_per_s"]
    for row in seconds:
        def num(value: Optional[float]) -> str:
            return "" if value is None else f"{value:.6f}"
        lines.append(f"{row['index']},{row['write_bytes']:.0f},{row['write_busy_s']:.6f},"
                     f"{num(row['write_gb_per_s'])},{row['read_bytes']:.0f},"
                     f"{row['read_busy_s']:.6f},{num(row['read_gb_per_s'])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_cycles(path: Path, cycles: Sequence[Cycle]) -> None:
    lines = ["cycle,planned_s,start_s,end_s,lateness_s,mismatched_chunks"]
    lines += [f"{i},{planned:.6f},{begin:.6f},{end:.6f},{late:.6f},{bad}"
              for i, (planned, begin, end, late, bad) in enumerate(cycles)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------
# the volume (a4_box.py at d6f7f733; a network volume is refused, the product refuses it)
# ---------------------------------------------------------------------------
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
LOCAL, VIRTUAL, NETWORK, REFUSED = "local", "virtual", "network", "refused"


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
    try:
        device = arm_validity.disk_device(path)
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

    path = os.path.join(directory, f"a3-direct-probe-{os.getpid()}.bin")
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


def spill_volume(directory: str) -> Dict[str, Any]:
    """Preflight of a spill directory. ``ok`` False means: do not spill there."""
    os.makedirs(directory, exist_ok=True)
    resolved = os.path.realpath(directory)
    record: Dict[str, Any] = {"path": resolved, "platform": sys.platform}
    if sys.platform.startswith("linux"):
        record.update(parse_mountinfo(_read_text("/proc/self/mountinfo") or "", resolved) or {})
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
    if scope not in (REFUSED, NETWORK):
        direct_ok, direct_reason = round_trip(resolved)
    record.update(direct_io=direct_ok, direct_io_reason=direct_reason)
    refusals: List[str] = []
    if scope in (REFUSED, NETWORK):
        refusals.append(reason)
    elif not direct_ok:
        refusals.append(f"unbuffered I/O is not usable here: {direct_reason}")
    record["refusals"] = refusals
    record["ok"] = not refusals
    return record


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--spill-dir", required=True, help="the session's spill volume")
    parser.add_argument("--plan", required=True,
                        help="the calibration child's JSON (its spill_plan: n_layers, k, "
                             "slot_bytes)")
    parser.add_argument("--a", required=True, help="a no-spill cost A child's JSON (its value)")
    parser.add_argument("--out", required=True, help="output JSON (a new file)")
    parser.add_argument("--cycles", required=True, help="per-cycle CSV (a new file)")
    parser.add_argument("--log", default=None,
                        help="per-second CSV (a new file); description only")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    for target in [out] + [Path(p) for p in (args.cycles, args.log) if p]:
        if target.exists():
            print(f"refusing to overwrite {target}")
            return 2
    try:
        inputs = load_inputs(Path(args.plan), Path(args.a))
    except InputError as exc:
        print(f"REFUSED INPUTS: {exc}")
        return 2
    directory = os.path.abspath(args.spill_dir)
    volume = spill_volume(directory)
    if not volume["ok"]:
        record = {"kind": "admission", "protocol": PROTOCOL, "inputs": inputs,
                  "schedule": admission_plan(inputs), "volume": volume, "admitted": False,
                  "reason": "volume refused: " + "; ".join(volume["refusals"])}
        write_json(out, record)
        print(record["reason"])
        return 3
    planned = admission_plan(inputs)
    print(f"D {planned['demand_bytes_per_s'] / 1e9:.4f} GB/s, cycle "
          f"{planned['cycle_bytes']:,} B, V {planned['volume']['bytes']:,} B, N "
          f"{planned['n_cycles']} cycles every {planned['period_s']:.4f} s "
          f"({planned['duration_s']:.0f} s)")
    started = time.time()
    record = run_admission(directory, inputs)
    record.update(volume=volume, started_unix=started, ended_unix=time.time())
    seconds = record.pop("per_second")
    if args.log:
        write_per_second(Path(args.log), seconds)
        record["per_second_log"] = str(args.log)
    write_cycles(Path(args.cycles), record.pop("cycle_rows"))
    record["cycle_log"] = str(args.cycles)
    write_json(out, record)
    judged = record["judgement"]
    print(f"{record['cycles']} of {record['scheduled_cycles']} scheduled cycles; max lateness "
          f"{judged['max_lateness_s']:.3f} s (limit {MAX_LATENESS_S:g} s); "
          f"{record['mismatched_chunks']} of {record['chunks_read']} chunks read back wrong")
    print(f"{'ADMITTED' if record['admitted'] else 'REFUSED'}: {record['reason']}")
    return 0 if record["admitted"] else 4


if __name__ == "__main__":
    sys.exit(main())
