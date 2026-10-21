#!/usr/bin/env python3
"""C2 depth 1/2 preparation and Windows physical-NVMe gate; see the frozen protocol."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Sequence

PROTOCOL = Path(__file__).parents[1] / "c2-per-drive-depth-20261018.md"
SPAN_BYTES = 400 * 1024 * 1024
WORKERS = 4
ROUNDS = 6
SPANS = 24


class ByteMismatchError(RuntimeError):
    """A full staging region differs from its immutable file reference."""


def pipeline(
    items: Sequence[Any],
    depth: int,
    workers: int,
    make_works: Callable[[Any, int], Sequence[Callable[[], None]]],
    verify: Callable[[Any, int], None] | None = None,
) -> dict[str, Any]:
    """Bound queued spans; never reuse a slot or return while its writers are running.

    The caller owns exactly two arenas at either depth. ``verify`` is for the
    untimed correctness pass only: it blocks slot reuse, simulating a consumer.
    Native synchronous I/O is not forcibly cancelled on error. Executor teardown
    drains every submitted writer, including a partly submitted span.
    """
    if depth not in (1, 2) or workers < 1 or not items:
        raise ValueError("depth must be 1/2, workers positive, and items nonempty")
    pending: collections.deque = collections.deque()
    result = {"completed": 0, "max_pending_spans": 0, "max_pending_ranges": 0}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="c2-range") as pool:
        started = time.perf_counter()

        def complete() -> None:
            item, slot, futures = pending.popleft()
            for future in futures:
                future.result()
            if verify is not None:
                verify(item, slot)
            result["completed"] += 1

        for index, item in enumerate(items):
            if len(pending) == depth:
                complete()
            slot = index % depth
            works = make_works(item, slot)
            if len(works) != workers:
                raise ValueError("one range per fixed worker is required")
            futures = [pool.submit(work) for work in works]
            pending.append((item, slot, futures))
            result["max_pending_spans"] = max(result["max_pending_spans"], len(pending))
            result["max_pending_ranges"] = max(
                result["max_pending_ranges"], sum(len(entry[2]) for entry in pending)
            )
        while pending:
            complete()
        result["seconds"] = time.perf_counter() - started
    return result


def counter_reasons(
    samples: Sequence[tuple[float, dict]], start: float, end: float, gb_s: float
) -> list[str]:
    """Apply the frozen per-arm PDH/power validity rows; unavailable is not zero."""
    rows = [(stamp, row) for stamp, row in samples if start < stamp <= end]
    reasons = []
    if len(rows) < 4 or any(
        key not in row or not math.isfinite(row[key]) or row[key] < 0
        for _, row in rows for key in ("read", "write", "ac")
    ):
        return ["missing counter/power samples"]
    stamps = [start, *(stamp for stamp, _ in rows), end]
    if any(right - left > 2 for left, right in zip(stamps, stamps[1:])):
        reasons.append("counter gap/suspend >2 s")
    if any(row["ac"] != 1 for _, row in rows):
        reasons.append("AC lost or unknown")
    if statistics.fmean(row["write"] for _, row in rows) > 2**20:
        reasons.append("foreign writes >1 MiB/s")
    if statistics.fmean(row["read"] for _, row in rows) > gb_s * 1e9 * 1.20:
        reasons.append("foreign reads >20% allowance")
    return reasons


def verdict(record: dict, reviewed: bool = False) -> dict:
    """Judge saved data only after explicit host-state review; no rounded ratios."""
    if record.get("correctness_ok") is False:
        return {"verdict": "FAIL correctness"}
    rounds = record.get("rounds", [])
    if (not reviewed or record.get("correctness_ok") is not True
            or record.get("complete") is not True
            or record.get("invalid_reasons") != [] or len(rounds) != ROUNDS):
        return {"verdict": "NO VERDICT", "reason": "invalid, incomplete or unreviewed evidence"}
    ratios = []
    try:
        for index, entry in enumerate(rounds):
            if entry["order"] != ([1, 2] if index % 2 == 0 else [2, 1]):
                raise ValueError("arm order changed")
            rates = [entry["arms"][str(depth)]["gb_s"] for depth in (1, 2)]
            if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
                raise ValueError("invalid rate")
            ratios.append(rates[1] / rates[0])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return {"verdict": "NO VERDICT", "reason": "malformed paired arms"}
    answer = "NO VERDICT"
    if all(ratio >= 1.10 for ratio in ratios):
        answer = "PASS primitive"
    elif all(0.95 <= ratio <= 1.05 for ratio in ratios):
        answer = "FAIL benefit"
    elif all(ratio < 0.95 for ratio in ratios):
        answer = "FAIL regression"
    return {"verdict": answer, "ratios": ratios, "median_ratio": statistics.median(ratios)}


def save(path: Path, record: dict) -> None:
    """Keep partial arms on failure; replace only this sequence's JSON atomically."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def physical_drive(root: Path) -> dict:
    """Resolve a normal local volume to a disk; reject ambiguous/reparse paths."""
    import stat

    for part in (root, *root.parents):
        if part.stat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError(f"reparse/junction path is outside this protocol: {part}")
    if len(root.drive) != 2 or root.drive[1] != ":":
        raise ValueError("ordinary local drive path required")
    letter = root.drive[0]
    # letter is taken from a validated local drive, never arbitrary shell input.
    script = (
        f"Get-Partition -DriveLetter '{letter}' | Get-Disk | "
        "Select-Object Number,FriendlyName,SerialNumber,BusType,Size | ConvertTo-Json -Compress"
    )
    done = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        check=True, capture_output=True, text=True, timeout=30,
    )
    disk = json.loads(done.stdout)
    if not isinstance(disk, dict) or disk.get("BusType") != 17:  # MSFT_Disk NVMe
        raise ValueError(f"physical NVMe not established: {disk}")
    return disk


def cpu_check(path: Path) -> int:
    """Portable synthetic bytes/lifetime exercise; never produce a throughput verdict."""
    if path.exists():
        raise SystemExit("output already exists; preserve earlier evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    slots = [bytearray(64 * 1024), bytearray(64 * 1024)]
    payloads = [hashlib.sha256(str(index).encode()).digest() * 2048 for index in range(19)]
    record: dict = {
        "kind": "synthetic_cpu_correctness", "python": sys.version,
        "executable": sys.executable, "hardware_measured": False,
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROTOCOL.parent, text=True
        ).strip(),
        "allocated_bytes": sum(map(len, slots)), "workers": WORKERS, "arms": [],
        "decision": {"verdict": "NO VERDICT", "reason": "synthetic CPU correctness only"},
    }
    for depth in (1, 2):
        checks = []

        def works(index: int, slot: int) -> list:
            def write(start: int) -> None:
                slots[slot][start : start + 16384] = payloads[index][start : start + 16384]

            return [lambda start=start: write(start) for start in range(0, 65536, 16384)]

        def verify(index: int, slot: int) -> None:
            if slots[slot] != payloads[index]:
                raise ByteMismatchError(f"synthetic CPU item {index}")
            checks.append({"item": index, "slot": slot,
                           "sha256": hashlib.sha256(slots[slot]).hexdigest()})

        result = pipeline(list(range(19)), depth, WORKERS, works, verify)
        record["arms"].append({"depth": depth, **result, "full_byte_checks": checks})
    record["correctness_ok"] = True
    save(path, record)
    return 0


def run(args: argparse.Namespace) -> int:
    """Execute only on a pre-arranged physical stand, with the old #974 primitive."""
    if sys.platform != "win32":
        raise SystemExit("physical C2 measurement requires Windows")
    from rich.console import Console

    console = Console()
    root = Path(os.path.abspath(args.drive))
    out = Path(args.out)
    if out.exists():
        raise SystemExit("output already exists: keep earlier sequences; choose a new filename")
    out.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "window": args.window, "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version, "executable": sys.executable, "rounds": [],
        "invalid_reasons": [], "correctness_ok": None, "complete": False,
        "kind": "hardware_direct_io_primitive", "workers": WORKERS,
        "span_bytes": SPAN_BYTES, "allocated_slots": 2, "pinned": True,
    }
    sampler = None
    observer = None
    stop = threading.Event()
    ac_samples: list = []
    arenas = []  # keep strong references until every pipeline executor has drained
    try:
        record["disk"] = physical_drive(root)
        import torch
        from issue974_unbuffered import SECTOR
        from two_drive_aspm_probe import on_ac_power
        from two_drive_read import (
            CounterSampler,
            _read_range,
            box_state,
            collect_spans,
            span_ranges,
        )

        import soup_cli

        record.update(torch=torch.__version__, soup_cli_file=soup_cli.__file__)
        record["git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROTOCOL.parent, text=True
        ).strip()
        record["box_start"] = box_state()

        def boundary() -> dict:
            state = box_state()
            if not on_ac_power():
                raise RuntimeError("AC lost or unknown")
            if (state["avail_phys_gb"] < 4 or state["commit_limit_gb"] <= 0
                    or state["commit_used_gb"] / state["commit_limit_gb"] >= 0.85):
                raise RuntimeError("host memory validity row fired")
            return state

        boundary()  # refuse before CUDA initialization or any fixture read
        record["fixture"] = fixture = {
            str(path): [path.stat().st_size, path.stat().st_mtime_ns]
            for path in sorted(root.iterdir()) if path.is_file()
        }
        if not fixture:
            raise ValueError("empty fixture")
        for path in fixture:
            if Path(path).is_symlink() or Path(path).stat().st_file_attributes & 0x400:
                raise ValueError(f"fixture reparse path: {path}")
        import ctypes

        from two_drive_read import _MemoryStatusEx

        memory = _MemoryStatusEx()
        memory.dwLength = ctypes.sizeof(memory)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise RuntimeError("physical RAM size unavailable")
        record["physical_ram_bytes"] = memory.ullTotalPhys
        if sum(size for size, _ in fixture.values()) <= memory.ullTotalPhys:
            raise ValueError("fixture must exceed physical RAM")
        spans = collect_spans([str(root)], SPAN_BYTES)[:SPANS]
        if len(spans) != SPANS:
            raise ValueError("need 24 complete 400 MiB spans")
        record["spans"] = [vars(span) for span in spans]
        torch.cuda.init()
        arenas = [torch.empty(SPAN_BYTES, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
        for arena in arenas:
            arena.zero_()
            if not arena.is_pinned() or arena.data_ptr() % SECTOR:
                raise ValueError("staging must be pinned and sector aligned")

        def works(span: Any, slot: int) -> list:
            base = arenas[slot].data_ptr()
            return [
                lambda start=start, end=end: _read_range(
                    span.path, start, base + start - span.offset, end - start
                )
                for start, end in span_ranges(span, WORKERS)
            ]

        def verify(span: Any, slot: int) -> None:
            # Chunked full comparison keeps reference memory constant at both depths.
            with open(span.path, "rb") as handle:
                handle.seek(span.offset)
                for offset in range(0, span.nbytes, 2**20):
                    count = min(2**20, span.nbytes - offset)
                    expected = handle.read(count)
                    got = arenas[slot][offset : offset + count].numpy().tobytes()
                    if len(expected) != count or got != expected:
                        raise ByteMismatchError(f"{span.path}:{span.offset + offset} slot {slot}")

        record["correctness_passes"] = []
        save(out, record)
        for depth in (1, 2):
            result = pipeline(spans, depth, WORKERS, works, verify)
            record["correctness_passes"].append({"depth": depth, **result})
            save(out, record)
        record["correctness_ok"] = True
        instance = f"{record['disk']['Number']} {root.drive}"
        sampler = CounterSampler({
            "read": rf"\PhysicalDisk({instance})\Disk Read Bytes/sec",
            "write": rf"\PhysicalDisk({instance})\Disk Write Bytes/sec",
        })

        def observe() -> None:
            while not stop.is_set():
                ac_samples.append((time.time(), int(on_ac_power())))
                stop.wait(0.25)

        observer = threading.Thread(target=observe, name="c2-power", daemon=True)
        observer.start()
        sampler.start()
        time.sleep(0.5)

        for index in range(ROUNDS):
            entry = {"order": [1, 2] if index % 2 == 0 else [2, 1], "arms": {}}
            record["rounds"].append(entry)
            for depth in entry["order"]:
                arm: dict = {"box_before": boundary()}
                entry["arms"][str(depth)] = arm
                save(out, record)
                pipeline(spans[:1], depth, WORKERS, works)  # untimed I/O warm-up
                arm["wall_start"] = time.time()
                result = pipeline(spans, depth, WORKERS, works)
                arm.update(result)
                arm["wall_end"] = time.time()
                arm["gb_s"] = sum(span.nbytes for span in spans) / result["seconds"] / 1e9
                # Each of the last depth spans still owns its slot.
                for item_index in range(len(spans) - depth, len(spans)):
                    verify(spans[item_index], item_index % depth)
                arm["retained_bytes_ok"] = True
                arm["box_after"] = boundary()
                combined = [
                    (stamp, dict(row, ac=min(
                        (value for at, value in ac_samples
                         if arm["wall_start"] - 0.25 <= at <= stamp),
                        default=0,
                    ))) for stamp, row in sampler.samples
                ]
                record["invalid_reasons"].extend(counter_reasons(
                    combined, arm["wall_start"], arm["wall_end"], arm["gb_s"]
                ))
                record["invalid_reasons"].extend(sampler.errors)
                save(out, record)
                console.print(f"round {index} depth {depth}: {arm['gb_s']:.3f} GB/s")
                if record["invalid_reasons"]:
                    raise RuntimeError("counter validity row fired; preserve sequence and stop")
        record["complete"] = True
    except ByteMismatchError as exc:
        record["correctness_ok"] = False
        record["invalid_reasons"].append(repr(exc))
    except BaseException as exc:  # preserve partial evidence, including Ctrl-C
        record["invalid_reasons"].append(repr(exc))
    finally:
        stop.set()
        if observer is not None:
            observer.join()
        if sampler is not None:
            sampler.stop()
            record["counter_samples"] = sampler.samples
            record["counter_errors"] = sampler.errors
            record["invalid_reasons"].extend(sampler.errors)
        record["ac_samples"] = ac_samples
        if "fixture" in record:
            try:
                after = {path: [os.stat(path).st_size, os.stat(path).st_mtime_ns]
                         for path in record["fixture"]}
                record["fixture_after"] = after
                if after != record["fixture"]:
                    record["invalid_reasons"].append("fixture changed")
                record["box_end"] = box_state()
            except BaseException as exc:
                record["invalid_reasons"].append(f"final state unavailable: {exc!r}")
        record["decision"] = verdict(record)
        save(out, record)
    console.print(record["decision"])
    return 0 if record["complete"] and not record["invalid_reasons"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", type=Path, help="evaluate saved JSON without hardware")
    parser.add_argument("--cpu-check", action="store_true", help="synthetic byte/lifetime check")
    parser.add_argument("--host-reviewed", action="store_true", help="host-state review completed")
    parser.add_argument("--drive", help="immutable flat fixture directory on one physical NVMe")
    parser.add_argument("--window", help="identifier of the user-agreed quiet resource window")
    parser.add_argument("--out", help="new JSON path; existing sequences are never overwritten")
    args = parser.parse_args()
    if args.cpu_check:
        if not args.out:
            parser.error("--cpu-check requires --out")
        return cpu_check(Path(args.out))
    if args.evaluate:
        from rich.console import Console

        Console().print(verdict(json.loads(args.evaluate.read_text()), args.host_reviewed))
        return 0
    if not all((args.drive, args.window, args.out)):
        parser.error("physical run requires --drive, --window and --out")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
