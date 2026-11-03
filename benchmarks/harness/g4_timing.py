#!/usr/bin/env python3
"""G4 timing: the bf16 store against each compressed store, A-B-B-A, fresh processes.

One comparison per candidate codec: arm A is the exact bf16 store, arm B the codec's
store. Every run is ``g4_probe.py --mode timing`` in a fresh process under an
``arm_validity.ArmWatch``; the value of a run is the median of its timed steps. Blocks
run until ``arm_validity.blocks_to_add`` returns 0 (the METHOD.md repeats rule: the
5-block pilot, then up to N, then one top-up), capped at ``--max-blocks``. A block with
a void run is kept on record and replaced.

``suspend`` and ``power`` are required checks. ``foreign_reads`` is recorded but not
required: the RAM tier reads no disk inside the timed steps, so the arm has no model
path for the check to watch (it reports unknown, which does not void).

Benchmark harness code, not shipped.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arm_validity import ArmWatch, Run, block_order, blocks_to_add, repeats  # noqa: E402

HARNESS = Path(__file__).resolve().parent
REQUIRED = ("suspend", "power")
DEFAULT_MAX_BLOCKS = 12
ARMS = {"A": "bf16"}

Runner = Callable[[str, str, int, int], Dict[str, Any]]


def _as_runs(blocks: Sequence[Sequence[Dict[str, Any]]]) -> List[List[Run]]:
    return [[Run(run["value"], run["outcome"]) for run in block] for block in blocks]


def run_comparison(codec: str, runner: Runner, watch: Callable[..., Any], *,
                   max_blocks: int = DEFAULT_MAX_BLOCKS,
                   first_run: int = 1) -> Dict[str, Any]:
    """Blocks of A (bf16), B (``codec``), B, A until the repeats rule is satisfied."""
    codec_of = {"A": ARMS["A"], "B": codec}
    blocks: List[List[Dict[str, Any]]] = []
    run_number = first_run
    stopped: Optional[str] = None
    while True:
        try:
            missing = blocks_to_add(_as_runs(blocks))
        except ValueError as exc:  # sigma undefined, e.g. a zero mean
            stopped = f"repeats rule failed: {exc}"
            break
        if missing == 0:
            break
        if len(blocks) >= max_blocks:
            stopped = f"max blocks ({max_blocks}) reached with {missing} still needed"
            break
        arms = block_order(1)[0]
        block: List[Dict[str, Any]] = []
        for position, arm in enumerate(arms):
            rnd = 2 * len(blocks) + position // 2
            with watch(arm=arm, round=rnd, run=run_number) as watched:
                result = runner(arm, codec_of[arm], rnd, run_number)
            record = watched.record
            block.append({"arm": arm, "codec": codec_of[arm], "round": rnd, "run": run_number,
                          "value": float(result["value"]), "outcome": str(record["outcome"]),
                          "record": record, "report": result["report"]})
            run_number += 1
        blocks.append(block)
    out: Dict[str, Any] = {"codec": codec, "blocks": blocks, "stopped": stopped,
                           "next_run": run_number}
    try:
        rep = repeats(_as_runs(blocks))
        out.update({"sigma": rep.sigma, "mean_a": rep.mean_a, "blocks_needed": rep.blocks_needed})
    except ValueError as exc:
        out["repeats_error"] = str(exc)
    return out


class ProbeRunner:
    """Runs ``g4_probe.py --mode timing`` in a fresh process for each run."""

    def __init__(self, probe_args: Sequence[str], runs_dir: Path):
        self.probe_args = list(probe_args)
        self.runs_dir = runs_dir

    def __call__(self, arm: str, codec: str, rnd: int, run: int) -> Dict[str, Any]:
        report = self.runs_dir / f"run{run:03d}-{arm}-{codec}.json"
        log = report.with_suffix(".log")
        command = [sys.executable, str(HARNESS / "g4_probe.py"), "--mode", "timing",
                   "--codec", codec, "--out", str(report), "--label", f"{arm} r{rnd}",
                   *self.probe_args]
        # "x": a run never writes over an earlier run's log; its report path is new too.
        if report.exists():
            raise FileExistsError(f"{report} exists; timing runs never overwrite one another")
        with log.open("x", encoding="utf-8") as handle:
            subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)
        payload = json.loads(report.read_text(encoding="utf-8"))
        return {"value": payload["timing"]["median_seconds"], "report": str(report)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--codecs", default="int8,fp8_e4m3")
    parser.add_argument("--out", required=True, help="timing.json; runs/ goes next to it")
    parser.add_argument("--label", default="")
    parser.add_argument("--max-blocks", type=int, default=DEFAULT_MAX_BLOCKS)
    args, probe_args = parser.parse_known_args(argv)
    out = Path(args.out)
    if out.exists():
        parser.error(f"{out} exists; a timing attempt never replaces an earlier one "
                     f"(rename it, or pass a new --out)")
    runs_root = out.parent / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    attempt = 1 + sum(1 for path in runs_root.glob("attempt-*") if path.is_dir())
    runs_dir = runs_root / f"attempt-{attempt:02d}"
    runs_dir.mkdir()  # raises if a concurrent attempt took the number
    runner = ProbeRunner(probe_args, runs_dir)

    def watch(*, arm: str, round: int, run: int) -> ArmWatch:
        return ArmWatch(arm=arm, round=round, run=run, label=args.label, required=REQUIRED)

    payload: Dict[str, Any] = {"label": args.label, "required_checks": list(REQUIRED),
                               "probe_args": probe_args, "runs_dir": str(runs_dir),
                               "attempt": attempt, "comparisons": []}
    next_run = 1
    for codec in [part.strip() for part in args.codecs.split(",") if part.strip()]:
        result = run_comparison(codec, runner, watch, max_blocks=args.max_blocks,
                                first_run=next_run)
        next_run = result["next_run"]
        payload["comparisons"].append(result)
        out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        print(f"{codec}: {len(result['blocks'])} blocks, sigma {result.get('sigma')}, "
              f"stopped {result['stopped']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
