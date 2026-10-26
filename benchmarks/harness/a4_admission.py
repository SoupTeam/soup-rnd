#!/usr/bin/env python3
"""A4 machine admission, protocol v3: does the spill volume keep up with the session?

Before a session (benchmarks/probe-a4-l2l-ram-budget-spill.md §3.2) the spill volume runs
the store's I/O pattern through the store's own unbuffered ``DirectSpillFile`` on a fixed
schedule at ``MARGIN`` x the session's spill demand: every ``PERIOD_S`` one k = 4
minimum-budget spill file (32 layers x 4 chunks of 4 MiB = 512 MiB) is written in the
forward's order and read back in the backward's order (layers descending). Cycle i starts
at max(t0 + i x PERIOD_S, the end of cycle i - 1); its lateness is its end minus
t0 + (i + 1) x PERIOD_S. The machine is admitted iff over ``DURATION_S`` no cycle ends more
than ``MAX_LATENESS_S`` late, every chunk read back is the one its cycle wrote there (each
4 KiB sector carries its cycle and file offset), and no test file is left. Everything here
was fixed before the first v3 box was rented.

``DEMAND_BYTES_PER_S`` is v1's (one RTX 3090 VM, 2026-10-25): the largest, over the
session's spill arms, of the arm's median over its timed steps of spill_written / step_s.
``v1_demand`` recomputes it from the published v1 records.

Description only, never part of the decision: per-second bytes and busy time, 10 s windows
against their median (v2's criterion, which judged the volume against itself), and, after
a session, each spill arm's write rate against the demand (``describe_arms``). Passing shows
the volume held twice v1's spill rate for 30 minutes without falling more than 5 s behind;
it does not show that the volume cannot dip during the session. ``a4_rule`` reads none of it.

Usage::

    python benchmarks/harness/a4_admission.py --spill-dir /root/a4/spill \\
        --out admission.json --log admission-per-second.csv --cycles admission-cycles.csv
    python benchmarks/harness/a4_admission.py --describe <session dir> --out arms.json

Exit 0 admitted, 4 refused, 3 the volume itself refused (tmpfs, overlay, no direct I/O).
torch is imported inside functions.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: Fixed before the first v3 box (the owner's protocol v3, 2026-10-26).
PROTOCOL = "v3"
DURATION_S = 1800.0
MAX_LATENESS_S = 5.0
MARGIN = 2.0
CHUNK_BYTES = 4 << 20
N_LAYERS = 32
K = 4
#: Distinct random chunks rotated through the writes. With the per-sector stamps no two
#: sectors written in one run are equal, so a deduplicating volume cannot skip the work.
DISTINCT_CHUNKS = 16
#: v1's largest spill demand: ``ram_k3_min/b05_p1_B.json`` wrote 402,653,184 B per timed
#: step at a median step of 1.729781039000045 s (``v1_demand`` on
#: ``benchmarks/results/a4-rtx3090-20261025``).
DEMAND_BYTES_PER_S = 402_653_184 / 1.729781039000045
CYCLE_BYTES = N_LAYERS * K * CHUNK_BYTES
#: Each period writes CYCLE_BYTES and reads them back, each at MARGIN x the demand: 1.153 s.
PERIOD_S = CYCLE_BYTES / (MARGIN * DEMAND_BYTES_PER_S)
#: The stamp: (cycle, file offset) in the first two int64 words of every sector.
SECTOR_BYTES = 4096
#: Description only: the log's windows and v2's fraction of their median.
WINDOW_S = 10.0
FRACTION = 0.5

Op = Tuple[str, float, float, int]  # (kind "write"|"read", start, end, bytes)
Cycle = Tuple[float, float, float, float, int]  # planned, start, end, lateness, bad chunks


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


def stability(windows: Sequence[Dict[str, Any]], fraction: float = FRACTION) -> Dict[str, Any]:
    """DESCRIPTION ONLY (v2's criterion, which judged the volume against itself): per
    direction, the median window busy rate and the windows below ``fraction`` of it."""
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


def exercise(directory: str, *, duration_s: float = DURATION_S, period_s: float = PERIOD_S,
             chunk: int = CHUNK_BYTES, n_layers: int = N_LAYERS, k: int = K,
             clock: Callable[[], float] = time.perf_counter,
             sleep: Callable[[float], None] = time.sleep,
             backend: Callable[[str, int], Any] = _default_backend,
             chunks: Optional[List[Any]] = None) -> Dict[str, Any]:
    """Write-then-read cycles on the schedule; no cycle starts after ``duration_s``.

    Cycle i waits for t0 + i x period_s (or starts at once if cycle i - 1 ended later),
    writes one spill file in the forward's order and reads it back in the backward's, every
    chunk stamped before its write and checked after its read. Every operation is timed."""
    write_offsets, read_offsets = spill_order(n_layers, k, chunk)
    size = n_layers * k * chunk
    path = os.path.join(directory, f"a4-admission-{os.getpid()}.bin")
    sources = chunks if chunks is not None else _random_chunks(chunk, DISTINCT_CHUNKS)
    sink = chunks[0].clone() if chunks is not None else _aligned(chunk)
    ops: List[Op] = []
    cycles: List[Cycle] = []
    mismatched = 0
    spill = backend(path, size)
    try:
        start = clock()
        for cycle in itertools.count():
            planned = start + cycle * period_s
            if max(planned, clock()) - start >= duration_s:
                break
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


def judgement(run: Dict[str, Any], limit_s: float = MAX_LATENESS_S) -> Dict[str, Any]:
    """Admitted iff no cycle ended more than ``limit_s`` behind its schedule, no chunk read
    back other bytes than its cycle wrote, and the test file is gone."""
    lateness = [row[3] for row in run["cycles"]]
    late = [i for i, value in enumerate(lateness) if value > limit_s]
    problems: List[str] = []
    if not lateness:
        problems.append("no cycle ran")
    elif late:
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


def assess(run: Dict[str, Any], *, duration_s: float = DURATION_S, window_s: float = WINDOW_S,
           limit_s: float = MAX_LATENESS_S) -> Dict[str, Any]:
    """The admission record of one ``exercise``; the log's bins cover [0, duration_s)."""
    windows = bins(run["ops"], run["start"], window_s, int(duration_s // window_s))
    totals = {kind: sum(n for k_, _, _, n in run["ops"] if k_ == kind)
              for kind in ("write", "read")}
    elapsed = run["end"] - run["start"]
    return {"judgement": judgement(run, limit_s), "cycles": len(run["cycles"]),
            "scheduled_cycles": math.ceil(duration_s / run["period_s"]),
            "bytes": totals, "elapsed_s": elapsed,
            "wall_gb_per_s": {kind: totals[kind] / elapsed / 1e9 if elapsed > 0 else None
                              for kind in totals},
            "mismatched_chunks": run["mismatched_chunks"], "chunks_read": run["chunks_read"],
            "file_bytes": run["file_bytes"], "left_behind": run["left_behind"],
            "description_only": {"windows": windows, "stability": stability(windows)},
            "per_second": bins(run["ops"], run["start"], 1.0, int(duration_s)),
            "cycle_rows": run["cycles"]}


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


def _spill_arms(results_dir: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
    for path in sorted(Path(results_dir, "arms").glob("*/b*_p*_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("records", []):
            if (row.get("kind") == "arm" and not row.get("skipped")
                    and (row.get("plan") or {}).get("spill_layers")):
                yield f"{path.parent.name}/{path.name}", row


def arm_demand(row: Dict[str, Any]) -> Optional[float]:
    """What a spill arm asked of the volume: the median over its timed steps of
    spill_written / step_s, in bytes per second."""
    rates = [step["spill_written"] / step["step_s"] for step in row.get("records") or []
             if step.get("step_s")]
    return statistics.median(rates) if rates else None


def v1_demand(results_dir: str) -> Dict[str, Any]:
    """The protocol's demand, recomputed from a session: its largest spill-arm demand."""
    rate, arm = max((demand, name) for name, row in _spill_arms(results_dir)
                    for demand in [arm_demand(row)] if demand is not None)
    return {"bytes_per_s": rate, "arm": arm}


def describe_arms(results_dir: str, demand: float = DEMAND_BYTES_PER_S,
                  margin: float = MARGIN) -> Dict[str, Any]:
    """DESCRIPTION ONLY: each spill arm's write rate (bytes over write-busy seconds, as the
    probe records it) against the admission's demand; the arms that wrote slower than the
    demand, and slower than the rate the admission held (margin x demand); and the arms that
    asked for more than the demand. Disk-tier arms share the drive with the weight reads by
    design, so everything is per comparison. It drops no arm and changes no verdict."""
    tested = margin * demand
    out: Dict[str, Any] = {"description_only": True, "demand_gb_per_s": demand / 1e9,
                           "tested_gb_per_s": tested / 1e9, "comparisons": {}}
    for name, row in _spill_arms(results_dir):
        entry = out["comparisons"].setdefault(name.split("/")[0], {
            "spill_arms": 0, "below_demand": [], "below_tested_rate": [], "demand_above": [],
            "arms": []})
        rate = row.get("spill_write_gb_per_s")
        asked = arm_demand(row)
        entry["spill_arms"] += 1
        entry["arms"].append({"arm": name, "block": row.get("block"), "write_gb_per_s": rate,
                              "write_to_demand": rate * 1e9 / demand if rate else None,
                              "demand_gb_per_s": asked / 1e9 if asked else None})
        if rate is not None and rate * 1e9 < demand:
            entry["below_demand"].append(name)
        if rate is not None and rate * 1e9 < tested:
            entry["below_tested_rate"].append(name)
        if asked is not None and asked > demand:
            entry["demand_above"].append(name)
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--spill-dir", help="run the admission on this spill directory")
    parser.add_argument("--describe", metavar="SESSION_DIR",
                        help="instead: describe a finished session's spill arms against the "
                             "demand (description only)")
    parser.add_argument("--out", required=True, help="output JSON (a new file)")
    parser.add_argument("--log", help="per-second CSV (a new file), with --spill-dir")
    parser.add_argument("--cycles", help="per-cycle CSV (a new file), with --spill-dir")
    parser.add_argument("--accept-network", action="store_true")
    parser.add_argument("--duration-s", type=float, default=DURATION_S,
                        help="protocol value 1800; any other value is a smoke and never admits")
    parser.add_argument("--window-s", type=float, default=WINDOW_S,
                        help="the description's windows (10 s); not part of the decision")
    args = parser.parse_args(argv)
    if bool(args.spill_dir) == bool(args.describe):
        parser.error("give exactly one of --spill-dir and --describe")
    if args.spill_dir and not (args.log and args.cycles):
        parser.error("--spill-dir needs --log and --cycles")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    import a4_box

    args = parse_args(argv)
    out = Path(args.out)
    for target in [out] + [Path(p) for p in (args.log, args.cycles) if p]:
        if target.exists():
            print(f"refusing to overwrite {target}")
            return 2
    if args.describe:
        listing = describe_arms(args.describe)
        out.write_text(json.dumps(listing, indent=2) + "\n", encoding="utf-8", newline="\n")
        for name, entry in listing["comparisons"].items():
            print(f"{name}: of {entry['spill_arms']} spill arms, "
                  f"{len(entry['below_demand'])} wrote below the demand "
                  f"{listing['demand_gb_per_s']:.3f} GB/s and "
                  f"{len(entry['below_tested_rate'])} below the tested "
                  f"{listing['tested_gb_per_s']:.3f} GB/s; {len(entry['demand_above'])} asked "
                  f"for more than the demand (description only)")
        return 0
    directory = os.path.abspath(args.spill_dir)
    protocol = args.duration_s == DURATION_S
    record: Dict[str, Any] = {
        "kind": "admission", "protocol": PROTOCOL,
        "settings": {"duration_s": args.duration_s, "period_s": PERIOD_S,
                     "demand_bytes_per_s": DEMAND_BYTES_PER_S, "margin": MARGIN,
                     "max_lateness_s": MAX_LATENESS_S, "chunk_bytes": CHUNK_BYTES,
                     "n_layers": N_LAYERS, "k": K, "distinct_chunks": DISTINCT_CHUNKS,
                     "window_s_description_only": args.window_s},
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
    write_per_second(Path(args.log), record.pop("per_second"))
    write_cycles(Path(args.cycles), record.pop("cycle_rows"))
    record["per_second_log"], record["cycle_log"] = str(args.log), str(args.cycles)
    judged = record["judgement"]
    record["admitted"] = bool(judged["admitted"] and protocol)
    record["reason"] = ("admitted" if record["admitted"] else
                        "; ".join(judged["problems"]
                                  + ([] if protocol else ["not the protocol's settings"])))
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{record['cycles']} of {record['scheduled_cycles']} scheduled cycles, period "
          f"{PERIOD_S:.4f} s; max lateness {judged['max_lateness_s']:.3f} s (limit "
          f"{MAX_LATENESS_S:g} s); {record['mismatched_chunks']} of {record['chunks_read']} "
          f"chunks read back wrong")
    described = record["description_only"]["stability"]["write"]
    print(f"description only: wall write {record['wall_gb_per_s']['write']:.3f} GB/s; busy "
          f"write median {described['median_gb_per_s'] or 0:.3f} GB/s, {len(described['low'])} "
          f"window(s) below half of it")
    print(f"{'ADMITTED' if record['admitted'] else 'REFUSED'}: {record['reason']}")
    return 0 if record["admitted"] else 4


if __name__ == "__main__":
    sys.exit(main())
