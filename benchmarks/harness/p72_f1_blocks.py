#!/usr/bin/env python3
"""F1 blocks for PR #72: the default expert path against the combined fast path.

Arm A is ``--expert-transfer sync``. Arm B is ``--expert-transfer packed --expert-prefetch
--expert-host-cache-mib N``. Each run is a fresh process of
``p3_route_preserving_mbr_soup_bridge.py --phase-profile-only --phase-profile-blocks 0``,
watched by ``arm_validity.ArmWatch``, and its number is the median of its timed steps
(``--timed-steps``). Block order, the 5-block pilot and the top-up come from
``arm_validity`` (``benchmarks/METHOD.md``); a block with a void run is kept on record and
replaced. The rule it follows is committed beside it: ``benchmarks/p72-f1-rule-20261119.md``.

    python p72_f1_blocks.py --root /workspace/p3-baseline --output results/f1

Benchmark harness code, not shipped. Needs the CUDA host of the bridge.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

HARNESS = Path(__file__).resolve().parent
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

from arm_validity import (  # noqa: E402
    ArmWatch,
    Run,
    block_order,
    blocks_to_add,
    repeats,
    valid_blocks,
)

#: Two-sided 5% critical values of Student's t by degrees of freedom (the next lower df is used).
T_CRITICAL = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
              9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}


def t_critical(df: int) -> float:
    return T_CRITICAL[max(key for key in T_CRITICAL if key <= max(df, 1))]


def run_blocks(
    run_arm: Callable[[str, int], Dict[str, Any]],
    *,
    label: str,
    required: Sequence[str],
    model_path: Optional[str],
    max_blocks: int,
    watch_factory: Callable[..., Any] = ArmWatch,
) -> Dict[str, Any]:
    """Run blocks until ``blocks_to_add`` says stop. ``run_arm(arm, run)`` returns ``value``."""
    blocks: List[List[Run]] = []
    records: List[Dict[str, Any]] = []
    while blocks_to_add(blocks) > 0:
        if len(blocks) >= max_blocks:
            raise RuntimeError(f"{max_blocks} blocks attempted and the rule still wants more")
        number = len(blocks)
        block: List[Run] = []
        for position, arm in enumerate(block_order(1)[0]):
            round_, run = 2 * number + 1 + position // 2, 4 * number + position + 1
            with watch_factory(
                arm=arm, round=round_, run=run, label=label, required=required,
                model_path=model_path,
            ) as watch:
                result = run_arm(arm, run)  # a fresh process; the median of its timed steps
            block.append(Run(result["value"], watch.record["outcome"]))
            records.append({"block": number + 1, "arm": arm, "run": run,
                            "value": result["value"], "extra": result.get("extra"),
                            "arm_record": watch.record})
        blocks.append(block)
    return {"blocks": blocks, "records": records}


def summarize(blocks: Sequence[Sequence[Run]]) -> Dict[str, Any]:
    """Per-block difference B minus A, the paired t statistic and the repeats figures."""
    valid = valid_blocks(blocks)
    diffs, relative = [], []
    for block in valid:
        pairs = list(zip(block, ("A", "B", "B", "A")))
        a = statistics.mean(run.value for run, arm in pairs if arm == "A")
        b = statistics.mean(run.value for run, arm in pairs if arm == "B")
        diffs.append(b - a)
        relative.append((b / a - 1) * 100)
    summary: Dict[str, Any] = {
        "valid_blocks": len(valid), "void_blocks": len(blocks) - len(valid),
        "block_diff_seconds_B_minus_A": diffs, "block_diff_percent": relative,
    }
    if len(diffs) >= 2:
        sd, mean = statistics.stdev(diffs), statistics.mean(diffs)
        if sd:
            t = mean / (sd / math.sqrt(len(diffs)))
        else:  # identical differences: a certain effect, or none at all
            t = math.inf if mean else 0.0
        summary.update(
            t=t, df=len(diffs) - 1, t_critical_5pct=t_critical(len(diffs) - 1),
            significant_5pct=abs(t) > t_critical(len(diffs) - 1),
            mean_diff_percent=statistics.mean(relative),
        )
        try:
            plan = repeats(blocks)
            summary["repeats"] = {"sigma": plan.sigma, "mean_a": plan.mean_a,
                                     "blocks_needed": plan.blocks_needed}
        except ValueError as error:  # fewer than the minimum valid blocks
            summary["repeats"] = {"error": str(error)}
    return summary


def make_runner(args: argparse.Namespace, out_dir: Path) -> Callable[[str, int], Dict[str, Any]]:
    bridge = HARNESS / "p3_route_preserving_mbr_soup_bridge.py"
    arms = {
        "A": ["--expert-transfer", "sync"],
        "B": ["--expert-transfer", "packed", "--expert-prefetch",
              "--expert-host-cache-mib", str(args.host_cache_mib)],
    }

    def run_arm(arm: str, run: int) -> Dict[str, Any]:
        out = out_dir / f"run{run:02d}_{arm}"
        command = [sys.executable, str(bridge), "--root", args.root, "--output", str(out),
                   "--mbr-group-size", "8", "--phase-profile-only", "--profile-kinds", "mbr",
                   "--phase-profile-blocks", "0", "--timed-steps", str(args.timed_steps),
                   *arms[arm]]
        with open(f"{out}.log", "w", encoding="utf-8") as log:
            subprocess.run(
                command, check=True, stdout=log, stderr=subprocess.STDOUT, env=os.environ
            )
        result = json.loads((out / "result_phase_profile_only.json").read_text(encoding="utf-8"))
        steps = [row["model_step_seconds"] for row in result["measured_uninstrumented"]]
        progress = json.loads((out / "progress.json").read_text(encoding="utf-8"))
        return {"value": statistics.median(steps),
                "extra": {"timed_steps": steps, "grad_sha256": result["grad_sha256"],
                          "cold_first_step": progress["completed_runs"][0]["model_step_seconds"],
                          "arm": result["arm"]}}

    return run_arm


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--root", default="/workspace/p3-baseline")
    parser.add_argument("--output", required=True)
    parser.add_argument("--timed-steps", type=int, default=3)
    parser.add_argument("--host-cache-mib", type=int, default=12288)
    parser.add_argument("--max-blocks", type=int, default=10)
    parser.add_argument("--label", default="RTX 3060 (vast.ai), Moonlight MBR")
    parser.add_argument("--required", default="suspend,power",
                        help="checks whose unknown voids the verdict (the rule says why)")
    args = parser.parse_args(argv)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=False)
    required = [name for name in args.required.split(",") if name]
    outcome = run_blocks(
        make_runner(args, out_dir), label=args.label, required=required,
        model_path=str(Path(args.root) / "cache"), max_blocks=args.max_blocks,
    )
    summary = summarize(outcome["blocks"])
    report = {"label": args.label, "settings": vars(args), "required": required,
              "summary": summary, "runs": outcome["records"]}
    (out_dir / "f1_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    digests = {tuple(r["extra"]["grad_sha256"]) for r in outcome["records"]}
    print("gradient digests across all runs:", "identical" if len(digests) == 1 else digests)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
