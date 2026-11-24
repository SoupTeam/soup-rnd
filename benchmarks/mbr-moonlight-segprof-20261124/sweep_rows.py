#!/usr/bin/env python3
"""Per-M kernel cost of the two routed-expert GEMMs, isolated (ticket 15 campaign).

No model and no arm: one bf16 weight and one bf16 input per (projection, M), ``--warmup`` calls of
``F.linear``, then one CUDA-only ``torch.profiler`` around ``--reps`` calls. The report gives the
kernel names with their call count and mean duration (device time / reps), plus µs per call and
µs per row. This is the table that prices the M histogram the campaign counts; it stays
memory-light (two weight matrices) because the whole-step trace does not fit the host at (64,64).

    python sweep_rows.py --out benchmarks/mbr-moonlight-segprof-20261124

Refuses to run when AC power is not online: a sweep on the battery reports a clock-limited card.
Benchmark code, not shipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO = Path(__file__).resolve().parents[2]
for entry in (str(REPO), str(REPO / "benchmarks")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from benchmarks.harness.mbr_trace import EXPERT_WEIGHT_SHAPES  # noqa: E402

MODEL = "Moonlight-16B-A3B"
# Every row count the campaign's counters can report, so the model prices each call exactly: C
# groups up to 481 rows in one call, S never exceeds one micro-batch. 640 is the next power-ish
# step above the largest group, kept as the tail bound.
ROWS = (*range(1, 513), 640)
# The rows the Markdown table shows: the bucket edges of the cuBLAS kernels and a few sizes inside
# them. sweep.json keeps every measured M, and that is what the analyzer prices the calls with.
REPORT_ROWS = (1, 2, 4, 6, 8, 12, 16, 17, 24, 31, 32, 40, 48, 64, 96, 128, 192, 256, 384, 512)
AC_ONLINE = Path("/sys/class/power_supply/ACAD/online")


def ac_online(path: Path = AC_ONLINE) -> bool:
    """The host is on AC power; a missing or unreadable file counts as offline."""
    try:
        return Path(path).read_text(encoding="utf-8").strip() == "1"
    except OSError:
        return False


def _card() -> str:
    """The card as ``name sm_XY N GiB``, from torch (imported by the caller)."""
    import torch

    major, minor = torch.cuda.get_device_capability(0)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    return f"{torch.cuda.get_device_name(0)} sm_{major}{minor} {total:.0f} GiB"


def _kernel_record(events: Iterable[Any], *, reps: int, rows: int) -> Dict[str, Any]:
    """Kernel name -> calls, µs per call and µs per row, over the device events of ``reps`` calls.

    ``us_per_call`` is the device time of that kernel divided by the number of ``F.linear`` calls,
    so one GEMM with a split-k pair sums its kernels. ``Memcpy``/``Memset`` are transfers, not
    kernels, and are left out.
    """
    kernels: Dict[str, Dict[str, float]] = {}
    total_us = 0.0
    for event in events:
        if not str(getattr(event, "device_type", "")).endswith("CUDA"):
            continue
        span = getattr(event, "time_range", None)
        if span is None:
            continue
        name = str(getattr(event, "name", "?"))
        if name.startswith(("Memcpy", "Memset")):
            continue
        micros = float(span.end) - float(span.start)
        total_us += micros
        entry = kernels.setdefault(name, {"calls": 0, "us": 0.0})
        entry["calls"] += 1
        entry["us"] += micros
    per_call = total_us / reps
    return {
        "calls": reps, "us_per_call": per_call, "us_per_row": per_call / rows,
        "kernels": {name: {"calls": int(entry["calls"]),
                           "calls_per_rep": entry["calls"] / reps,
                           "us_per_call": entry["us"] / reps,
                           "us_per_row": entry["us"] / reps / rows}
                    for name, entry in kernels.items()},
    }


def run_sweep(*, reps: int, warmup: int) -> Dict[str, Any]:  # pragma: no cover - needs the card
    """The per-M table of both projections, from CUDA-only profiles of ``reps`` calls each."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    document: Dict[str, Any] = {"model": MODEL, "card": _card(), "torch": str(torch.__version__),
                                "reps": reps, "warmup": warmup, "rows": list(ROWS),
                                "projections": {}}
    for name, (out_features, in_features) in EXPERT_WEIGHT_SHAPES.items():
        weight = torch.randn((out_features, in_features), device="cuda", dtype=torch.bfloat16)
        entry: Dict[str, Any] = {"weight": [out_features, in_features], "rows": {}}
        for rows in ROWS:
            x = torch.randn((rows, in_features), device="cuda", dtype=torch.bfloat16)
            for _ in range(warmup):
                torch.nn.functional.linear(x, weight)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as profiler:
                for _ in range(reps):
                    torch.nn.functional.linear(x, weight)
                torch.cuda.synchronize()
            entry["rows"][str(rows)] = _kernel_record(profiler.events(), reps=reps, rows=rows)
            del x
        document["projections"][name] = entry
        del weight
    return document


def report_lines(sweep: Dict[str, Any]) -> List[str]:
    """One table per projection: M, kernel, calls/reps, µs/call, µs/row."""
    reps = int(sweep["reps"])
    lines = [f"# Per-M kernel cost of the routed-expert GEMMs ({sweep['model']})", "",
             f"{sweep['model']} on {sweep['card']}, torch {sweep['torch']}: {reps} calls per M "
             f"after {sweep['warmup']} warm-up calls, CUDA-only profile, no model.", ""]
    for name, entry in sweep["projections"].items():
        out_features, in_features = entry["weight"]
        lines += [f"## {name}: {sweep['model']} weight ({out_features}, {in_features}) on "
                  f"{sweep['card']}, torch {sweep['torch']}", "",
                  "| M | kernel | calls/reps | µs/call | µs/row |", "|---|---|---|---|---|"]
        shown = [str(rows) for rows in REPORT_ROWS]
        for rows, record in ((key, entry["rows"][key]) for key in shown if key in entry["rows"]):
            lines.append(f"| {rows} | total | {record['calls'] / reps:.1f} "
                         f"| {record['us_per_call']:.3f} | {record['us_per_row']:.3f} |")
            for kernel, kernel_record in (record.get("kernels") or {}).items():
                lines.append(f"| {rows} | {kernel} | {kernel_record['calls_per_rep']:.1f} "
                             f"| {kernel_record['us_per_call']:.3f} "
                             f"| {kernel_record['us_per_row']:.3f} |")
        lines.append("")
    return lines


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path,
                        help="campaign folder: sweep.json and sweep.md are written here")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args(argv)
    if not ac_online():
        print(f"Refusing the sweep: AC power is not online ({AC_ONLINE}).", file=sys.stderr)
        return 1
    sweep = run_sweep(reps=args.reps, warmup=args.warmup)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sweep.json").write_text(json.dumps(sweep, indent=2), encoding="utf-8")
    (args.out / "sweep.md").write_text("\n".join(report_lines(sweep)) + "\n", encoding="utf-8")
    print(f"sweep written to {args.out}/sweep.json and {args.out}/sweep.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
