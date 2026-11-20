#!/usr/bin/env python3
"""A-B-C-C-B-A blocks: native L2L A0, B0 (route-preserving MBR) and the final exact MBR.

Arm A is the native L2L step (A0). Arm B is stock B0: MBR with packed transfer, prefetch and the
host cache. Arm C is the final exact MBR: B plus ``--expert-gemm lt-fused --mbr-ordered-backward
--expert-fast-gather --expert-async-lease`` (A0 bits). Each run is a fresh process of
``p3_route_preserving_mbr_soup_bridge.py --phase-profile-only --phase-profile-blocks 0`` watched
by ``arm_validity.ArmWatch``; its time is the median of its timed steps. In every block each arm
sits at mean position 3.5 (A at 1 and 6, B at 2 and 5, C at 3 and 4), so a steady drift of the
host cancels in any pair. The table for the PR reads from the summary: step time, logical expert
fetches per step and logical expert bytes per step (fetches times the packed expert size), per arm,
plus the paired differences B-A, C-A and C-B over blocks.

    python p72_abc_blocks.py --root /workspace/p3-baseline --output results/abc

Benchmark harness code, not shipped. Needs the CUDA host of the bridge.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

HARNESS = Path(__file__).resolve().parent
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

from arm_validity import VOID, ArmWatch  # noqa: E402
from p72_f1_blocks import t_critical  # noqa: E402
from rich.console import Console  # noqa: E402

console = Console()
ORDER = ("A", "B", "C", "C", "B", "A")
PAIRS = (("B", "A"), ("C", "A"), ("C", "B"))      # (numerator arm, reference arm)


def arm_flags(*, host_cache_mib: int) -> Dict[str, List[str]]:
    """Bridge flags of the three arms."""
    complete = ["--expert-transfer", "packed", "--expert-prefetch",
                "--expert-host-cache-mib", str(host_cache_mib)]
    return {
        "A": ["--profile-kinds", "l2l", "--expert-transfer", "packed",
              "--expert-host-cache-mib", str(host_cache_mib)],
        "B": list(complete),
        "C": complete + ["--expert-gemm", "lt-fused", "--mbr-ordered-backward",
                         "--expert-fast-gather", "--expert-async-lease"],
    }


def step_metrics(completed_runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Time, fetches and logical bytes per step of one process from the bridge's progress record.

    ``completed_runs[0]`` is the warm run, the rest are the timed steps. The pool's fetch counter is
    cumulative, so a step's fetches are the difference from the run before it. Logical bytes are the
    fetches times the packed expert size (what the CLI logs; host cache hits count too)."""
    steps = list(completed_runs)
    seconds, fetches = [], []
    packed = 0
    for before, after in zip(steps, steps[1:]):
        seconds.append(after["model_step_seconds"])
        now = after["expert_pool_stats_after_run"]
        fetches.append(now["fetches"] - before["expert_pool_stats_after_run"]["fetches"])
        packed = now["plan"]["packed_region_bytes"]
    return {"step_seconds": statistics.median(seconds), "fetches": int(statistics.median(fetches)),
            "logical_bytes": int(statistics.median(fetches)) * packed, "timed_steps": seconds}


def _mean_arm(block: Sequence[Dict[str, Any]], arm: str, key: str) -> float:
    return statistics.mean(run[key] for run in block if run["arm"] == arm)


def summarize(blocks: Sequence[Sequence[Dict[str, Any]]]) -> Dict[str, Any]:
    """Arm means and the paired differences over blocks without a void run."""
    valid = [b for b in blocks if all(run["outcome"] != VOID for run in b)]
    summary: Dict[str, Any] = {"valid_blocks": len(valid), "void_blocks": len(blocks) - len(valid),
                               "arms": {}, "comparisons": {}}
    for arm in "ABC":
        summary["arms"][arm] = {
            key: statistics.mean(_mean_arm(b, arm, key) for b in valid)
            for key in ("value", "fetches", "logical_bytes")
        }
        summary["arms"][arm]["step_seconds"] = summary["arms"][arm].pop("value")
    for top, ref in PAIRS:
        diffs = [_mean_arm(b, top, "value") - _mean_arm(b, ref, "value") for b in valid]
        relative = [(_mean_arm(b, top, "value") / _mean_arm(b, ref, "value") - 1) * 100
                    for b in valid]
        entry: Dict[str, Any] = {"block_diff_seconds": diffs, "block_diff_percent": relative,
                                 "mean_diff_seconds": statistics.mean(diffs),
                                 "mean_diff_percent": statistics.mean(relative)}
        if len(diffs) >= 2:
            sd, mean = statistics.stdev(diffs), statistics.mean(diffs)
            t = (mean / (sd / math.sqrt(len(diffs)))) if sd else (math.inf if mean else 0.0)
            critical = t_critical(len(diffs) - 1)
            entry.update(t=t, df=len(diffs) - 1, t_critical_5pct=critical,
                         significant_5pct=abs(t) > critical)
        summary["comparisons"][f"{top}-{ref}"] = entry
    return summary


def run_blocks(
    run_arm: Callable[[str, int], Dict[str, Any]],
    *,
    label: str,
    required: Sequence[str],
    model_path: Optional[str],
    blocks: int,
    max_blocks: int,
    watch_factory: Callable[..., Any] = ArmWatch,
    on_record: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run ``blocks`` valid blocks; a block with a void run is kept on record and replaced."""
    done: List[List[Dict[str, Any]]] = []
    records: List[Dict[str, Any]] = []
    attempts = 0
    while sum(all(r["outcome"] != VOID for r in b) for b in done) < blocks:
        if attempts >= max_blocks:
            raise RuntimeError(f"{max_blocks} blocks attempted and fewer than {blocks} are valid")
        block: List[Dict[str, Any]] = []
        for position, arm in enumerate(ORDER):
            run = 6 * attempts + position + 1
            with watch_factory(arm=arm, round=attempts + 1, run=run, label=label,
                               required=required, model_path=model_path) as watch:
                result = run_arm(arm, run)
            row = {"arm": arm, "run": run, "block": attempts + 1, "value": result["value"],
                   "fetches": result["fetches"], "logical_bytes": result["logical_bytes"],
                   "outcome": watch.record["outcome"]}
            block.append(row)
            record = dict(row, extra=result.get("extra"), arm_record=watch.record)
            records.append(record)
            if on_record is not None:
                on_record(record)
        done.append(block)
        attempts += 1
    return {"blocks": done, "records": records}


def make_runner(args: argparse.Namespace, out_dir: Path) -> Callable[[str, int], Dict[str, Any]]:
    bridge = HARNESS / "p3_route_preserving_mbr_soup_bridge.py"
    flags = arm_flags(host_cache_mib=args.host_cache_mib)

    def run_arm(arm: str, run: int) -> Dict[str, Any]:
        out = out_dir / f"run{run:02d}_{arm}"
        command = [sys.executable, str(bridge), "--root", args.root, "--output", str(out),
                   "--mbr-group-size", "8", "--phase-profile-only", "--profile-kinds", "mbr",
                   "--phase-profile-blocks", "0", "--timed-steps", str(args.timed_steps),
                   *flags[arm]]
        with open(f"{out}.log", "w", encoding="utf-8") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        result = json.loads((out / "result_phase_profile_only.json").read_text(encoding="utf-8"))
        progress = json.loads((out / "progress.json").read_text(encoding="utf-8"))
        metrics = step_metrics(progress["completed_runs"])
        return {"value": metrics["step_seconds"], "fetches": metrics["fetches"],
                "logical_bytes": metrics["logical_bytes"],
                "extra": {"timed_steps": metrics["timed_steps"],
                          "grad_sha256": result["grad_sha256"], "arm": result["arm"]}}

    return run_arm


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--root", default="/workspace/p3-baseline")
    parser.add_argument("--output", required=True)
    parser.add_argument("--timed-steps", type=int, default=3)
    parser.add_argument("--host-cache-mib", type=int, default=12288)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--max-blocks", type=int, default=8)
    parser.add_argument("--label", default="RTX 3060 (vast.ai), Moonlight, A0 / B0 / exact MBR")
    parser.add_argument("--required", default="suspend,power")
    args = parser.parse_args(argv)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=False)
    required = [name for name in args.required.split(",") if name]
    records_path = out_dir / "abc_records.jsonl"

    def keep(record: Dict[str, Any]) -> None:
        with open(records_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    outcome = run_blocks(
        make_runner(args, out_dir), label=args.label, required=required,
        model_path=str(Path(args.root) / "cache"), blocks=args.blocks,
        max_blocks=args.max_blocks, on_record=keep,
    )
    summary = summarize(outcome["blocks"])
    report = {"label": args.label, "settings": vars(args), "required": required,
              "summary": summary, "runs": outcome["records"]}
    (out_dir / "abc_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    console.out(json.dumps(summary, indent=2))
    digests: Dict[str, set] = {}
    for record in outcome["records"]:
        digests.setdefault(record["arm"], set()).add(tuple(record["extra"]["grad_sha256"]))
    for arm, seen in sorted(digests.items()):
        verdict = "identical" if len(seen) == 1 else str(seen)
        console.out(f"arm {arm} gradient digests: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
