#!/usr/bin/env python3
"""A4 machine admission: is the spill volume steady enough to time spill arms on?

Before a v2 session (benchmarks/probe-a4-l2l-ram-budget-spill.md §3.1) the spill volume
runs ``DURATION_S`` of unbuffered I/O in 4 MiB chunks in the store's own pattern: each
cycle writes one k = 4 minimum-budget spill file (32 layers x k chunks) in the forward's
order, then reads it back in the backward's order (layers descending). Every operation's
bytes and busy seconds are spread over the wall-clock windows it overlaps, so a stall
longer than a window still lands in it. The machine is admitted iff every
``WINDOW_S`` window's write rate and read rate is >= ``FRACTION`` of that direction's
median window rate. The thresholds were fixed before any v2 box was rented.

This is a machine acceptance, NOT a gate: ``a4_rule`` never reads it. ``slow_write_arms``
lists, for description only, the session's spill arms whose write rate fell below
FRACTION of the admission's median write rate; it drops no arm and changes no verdict.
Passing the admission does not mean the volume cannot dip during the session.

Usage::

    python benchmarks/harness/a4_admission.py --spill-dir /root/a4/spill \\
        --out admission.json --log admission-per-second.csv

Exit 0 admitted, 4 refused, 3 the volume itself refused (tmpfs, overlay, no direct I/O).
torch is imported inside functions (``aligned_buffer``).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: Fixed before the first v2 box (the owner's protocol v2, 2026-10-26).
DURATION_S = 600.0
WINDOW_S = 10.0
FRACTION = 0.5
CHUNK_BYTES = 4 << 20
N_LAYERS = 32
K = 4
#: Distinct random chunks rotated through the writes, so a deduplicating or compressing
#: virtual disk cannot turn the test into a no-op.
DISTINCT_CHUNKS = 16

Op = Tuple[str, float, float, int]  # (kind "write"|"read", start, end, bytes)


def spill_order(n_layers: int, k: int, chunk: int) -> Tuple[List[int], List[int]]:
    """Offsets in the order ``ActivationStore`` writes them in the forward (layer, then
    micro-batch, ascending) and reads them in the backward (layers descending)."""
    write = [(layer * k + mb) * chunk for layer in range(n_layers) for mb in range(k)]
    read = [(layer * k + mb) * chunk for layer in range(n_layers - 1, -1, -1)
            for mb in range(k)]
    return write, read


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


def judge(windows: Sequence[Dict[str, Any]], fraction: float = FRACTION) -> Dict[str, Any]:
    """Admitted iff, per direction, every window's rate >= fraction x the median window
    rate. A window with no busy time in a direction (the other direction held the whole
    window) is listed and does not count; a direction with no measured window refuses."""
    result: Dict[str, Any] = {"fraction": fraction, "admitted": True, "problems": []}
    for kind in ("write", "read"):
        rates = [w[f"{kind}_gb_per_s"] for w in windows]
        measured = [r for r in rates if r is not None]
        idle = [w["index"] for w, r in zip(windows, rates) if r is None]
        if not measured:
            result["admitted"] = False
            result["problems"].append(f"{kind}: no window measured")
            result[kind] = {"median_gb_per_s": None, "floor_gb_per_s": None, "low": [],
                            "idle": idle}
            continue
        median = statistics.median(measured)
        floor = fraction * median
        low = [{"index": w["index"], "gb_per_s": r} for w, r in zip(windows, rates)
               if r is not None and r < floor]
        if low:
            result["admitted"] = False
            result["problems"].append(f"{kind}: {len(low)} window(s) below {fraction:.0%} of "
                                      f"the median {median:.3f} GB/s")
        result[kind] = {"median_gb_per_s": median, "floor_gb_per_s": floor,
                        "min_gb_per_s": min(measured), "max_gb_per_s": max(measured),
                        "low": low, "idle": idle}
    return result


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


def exercise(directory: str, *, duration_s: float = DURATION_S, chunk: int = CHUNK_BYTES,
             n_layers: int = N_LAYERS, k: int = K,
             clock: Callable[[], float] = time.perf_counter,
             backend: Callable[[str, int], Any] = _default_backend,
             chunks: Optional[List[Any]] = None) -> Dict[str, Any]:
    """Write-then-read cycles until ``duration_s`` has passed; every operation timed.

    The last chunk of each cycle is read back and compared with what was written, so a
    volume that returns other bytes is caught (``mismatches``)."""
    import torch

    write_offsets, read_offsets = spill_order(n_layers, k, chunk)
    size = n_layers * k * chunk
    path = os.path.join(directory, f"a4-admission-{os.getpid()}.bin")
    sources = chunks if chunks is not None else _random_chunks(chunk, DISTINCT_CHUNKS)
    sink = chunks[0].clone() if chunks is not None else _random_chunks(chunk, 1)[0]
    ops: List[Op] = []
    cycles = mismatches = 0
    spill = backend(path, size)
    try:
        start = clock()
        while clock() - start < duration_s:
            for index, offset in enumerate(write_offsets):
                t0 = clock()
                spill.write(offset, sources[index % len(sources)])
                ops.append(("write", t0, clock(), chunk))
            for offset in read_offsets:
                t0 = clock()
                spill.read(offset, sink)
                ops.append(("read", t0, clock(), chunk))
            # read_offsets ends at layer 0, mb k - 1: the (k - 1)-th write of the cycle.
            if not torch.equal(sink, sources[(k - 1) % len(sources)]):
                mismatches += 1
            cycles += 1
        end = clock()
    finally:
        spill.close(remove=True)
    return {"start": start, "end": end, "ops": ops, "cycles": cycles,
            "mismatches": mismatches, "file_bytes": size, "path": path,
            "left_behind": os.path.exists(path)}


def assess(run: Dict[str, Any], *, duration_s: float = DURATION_S,
           window_s: float = WINDOW_S, fraction: float = FRACTION) -> Dict[str, Any]:
    """The admission record from one ``exercise``: windows over [0, duration_s) only."""
    count = int(duration_s // window_s)
    windows = bins(run["ops"], run["start"], window_s, count)
    seconds = bins(run["ops"], run["start"], 1.0, int(duration_s))
    verdict = judge(windows, fraction)
    if run["mismatches"]:
        verdict["admitted"] = False
        verdict["problems"].append(f"{run['mismatches']} cycle(s) read back other bytes")
    if run["left_behind"]:
        verdict["admitted"] = False
        verdict["problems"].append(f"the test file was left at {run['path']}")
    totals = {kind: sum(n for k_, _, _, n in run["ops"] if k_ == kind)
              for kind in ("write", "read")}
    return {"windows": windows, "per_second": seconds, "judgement": verdict,
            "cycles": run["cycles"], "bytes": totals, "elapsed_s": run["end"] - run["start"],
            "mismatches": run["mismatches"], "file_bytes": run["file_bytes"]}


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


def slow_write_arms(results_dir: str, admission: Dict[str, Any],
                    fraction: float = FRACTION) -> Dict[str, Any]:
    """DESCRIPTION ONLY: spill arms whose ``spill_write_gb_per_s`` is below ``fraction`` x
    the admission's median write rate. Disk-tier arms share the drive with the weight
    reads by design, so they are listed per comparison, not mixed into one count."""
    median = admission["judgement"]["write"]["median_gb_per_s"]
    floor = fraction * median
    out: Dict[str, Any] = {"description_only": True, "admission_write_median_gb_per_s": median,
                           "floor_gb_per_s": floor, "comparisons": {}}
    for path in sorted(Path(results_dir, "arms").glob("*/b*_p*_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("records", []):
            if row.get("kind") != "arm" or row.get("skipped"):
                continue
            if not (row.get("plan") or {}).get("spill_layers"):
                continue
            entry = out["comparisons"].setdefault(path.parent.name,
                                                  {"spill_arms": 0, "below": []})
            entry["spill_arms"] += 1
            rate = row.get("spill_write_gb_per_s")
            if rate is not None and rate < floor:
                entry["below"].append({"arm": f"{path.parent.name}/{path.name}",
                                       "block": row.get("block"), "write_gb_per_s": rate})
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--spill-dir", help="run the admission on this spill directory")
    parser.add_argument("--describe", metavar="SESSION_DIR",
                        help="instead: list a finished session's slow-write spill arms against "
                             "--admission (description only)")
    parser.add_argument("--admission", help="the admission record, with --describe")
    parser.add_argument("--out", required=True, help="output JSON (a new file)")
    parser.add_argument("--log", help="per-second CSV (a new file), with --spill-dir")
    parser.add_argument("--accept-network", action="store_true")
    parser.add_argument("--duration-s", type=float, default=DURATION_S,
                        help="protocol value 600; other values are recorded as a smoke")
    parser.add_argument("--window-s", type=float, default=WINDOW_S,
                        help="protocol value 10; other values are recorded as a smoke")
    args = parser.parse_args(argv)
    if bool(args.spill_dir) == bool(args.describe):
        parser.error("give exactly one of --spill-dir and --describe")
    if args.describe and not args.admission:
        parser.error("--describe needs --admission")
    if args.spill_dir and not args.log:
        parser.error("--spill-dir needs --log")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    import a4_box

    args = parse_args(argv)
    out = Path(args.out)
    targets = [out] + ([Path(args.log)] if args.log else [])
    for target in targets:
        if target.exists():
            print(f"refusing to overwrite {target}")
            return 2
    if args.describe:
        admission = json.loads(Path(args.admission).read_text(encoding="utf-8"))
        listing = slow_write_arms(args.describe, admission)
        out.write_text(json.dumps(listing, indent=2) + "\n", encoding="utf-8", newline="\n")
        for name, entry in listing["comparisons"].items():
            print(f"{name}: {len(entry['below'])} of {entry['spill_arms']} spill arms below "
                  f"{listing['floor_gb_per_s']:.3f} GB/s (description only)")
        return 0
    log = Path(args.log)
    directory = os.path.abspath(args.spill_dir)
    protocol = args.duration_s == DURATION_S and args.window_s == WINDOW_S
    record: Dict[str, Any] = {
        "kind": "admission",
        "settings": {"duration_s": args.duration_s, "window_s": args.window_s,
                     "fraction": FRACTION, "chunk_bytes": CHUNK_BYTES, "n_layers": N_LAYERS,
                     "k": K, "distinct_chunks": DISTINCT_CHUNKS},
        "settings_are_protocol": protocol,
        "volume": a4_box.spill_volume(directory, accept_network=args.accept_network),
        "placement": a4_box.placement(directory), "started_unix": time.time(),
    }
    if not record["volume"]["ok"]:
        record.update(admitted=False, reason="volume refused: "
                      + "; ".join(record["volume"]["refusals"]))
        out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(record["reason"])
        return 3
    run = exercise(directory, duration_s=args.duration_s)
    record.update(assess(run, duration_s=args.duration_s, window_s=args.window_s))
    record["admitted"] = bool(record["judgement"]["admitted"] and protocol)
    record["reason"] = ("admitted" if record["admitted"] else
                        "; ".join(record["judgement"]["problems"]
                                  + ([] if protocol else ["not the protocol's settings"])))
    per_second = record.pop("per_second")
    write_per_second(log, per_second)
    record["per_second_log"] = str(log)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    judgement = record["judgement"]
    for kind in ("write", "read"):
        row = judgement[kind]
        print(f"{kind}: median {row['median_gb_per_s']:.3f} GB/s, min {row['min_gb_per_s']:.3f}, "
              f"floor {row['floor_gb_per_s']:.3f}, {len(row['low'])} low window(s)")
    print(f"{'ADMITTED' if record['admitted'] else 'REFUSED'}: {record['reason']}")
    return 0 if record["admitted"] else 4


if __name__ == "__main__":
    sys.exit(main())
