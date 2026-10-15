"""Resume the preregistered complete D1 cohort; no cloud API or credentials."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from d1_nf4_study import ROOT, provenance
from d1_rules import summarize, write_new_json
from rich.console import Console

SHAPES = ((4096, 4096), (1024, 4096), (14336, 4096), (4096, 14336))
ROWS = (1, 8, 32, 128, 512, 2048)


def cases() -> list[dict]:
    result = [
        {
            "name": "ablation",
            "kind": "ablation",
            "pair": "baseline-nodequant",
            "dtype": "bfloat16",
            "seq": 512,
            "steps": 100,
            "warmup": 20,
        },
        {
            "name": "qa",
            "kind": "train",
            "pair": "native-forced",
            "dtype": "bfloat16",
            "seq": 1024,
            "steps": 256,
            "warmup": 20,
        },
    ]
    for dtype in ("float16", "bfloat16"):
        for columns, inner in SHAPES:
            for rows in ROWS:
                for pair in ("dense-native", "native-forced"):
                    result.append(
                        {
                            "name": f"projection-{dtype}-{rows}x{columns}x{inner}-{pair}",
                            "kind": "projection",
                            "pair": pair,
                            "dtype": dtype,
                            "m": rows,
                            "n": columns,
                            "k": inner,
                            "steps": 200,
                            "warmup": 20,
                            "warmup_seconds": 5,
                        }
                    )
    # Diagnose the full projection grid before committing more real-task repeats
    # on a new physical GPU. The plan still contains every one of the 98 cases.
    return [result[0], *result[2:], result[1]]


def next_block(root: Path) -> int:
    numbers = [
        int(path.name.split("-")[1]) for path in root.rglob("block-*-run-*") if path.is_dir()
    ]
    return max(numbers, default=-1) + 1


def records(root: Path) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(root.rglob("watched.json"))]


def remaining_blocks(rows: list[dict]) -> int:
    if not rows:
        return 5
    comparisons = summarize(rows)["comparisons"]
    if len(comparisons) != 1:
        raise ValueError("one case directory contains multiple comparisons")
    result = comparisons[0]
    if result["accepted_blocks"] < 5:
        return 5 - result["accepted_blocks"]
    return result["blocks_to_add_for_10_percent_effect"]


def execute(command: list[str], log: Path, deadline: float) -> int:
    remaining = deadline - time.time()
    if remaining < 60:
        raise TimeoutError("reserved evidence/export deadline reached")
    write_new_json(log.with_suffix(".command.json"), command)
    with log.open("x", encoding="utf-8") as handle:
        child = subprocess.Popen(
            command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            return child.wait(timeout=remaining - 30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            raise


def run(args: argparse.Namespace) -> dict:
    source = provenance(True)
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {
        "source_sha": source["source_sha"],
        "cases": cases(),
        "weights": str(args.weights),
        "data": str(args.data),
        "clock_policy": "Recorded A100 clock policy; unchanged <=5% guard per comparison",
        "thermal_policy": ">=20 steps and 5 seconds for projections; exactly 20 QA warm-up updates",
    }
    plan_path = args.output / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("resume plan/source differs from the frozen cohort")
    else:
        write_new_json(plan_path, plan)
    harness = str(ROOT / "benchmarks/harness/d1_nf4_study.py")
    python = sys.executable
    # New physical GPU and dependency cohort: re-establish correctness before timings.
    for dtype in ("float16", "bfloat16"):
        output = args.output / ("correctness-" + dtype)
        result = output / "result.json"
        if not result.exists():
            code = execute(
                [
                    python,
                    harness,
                    "correctness",
                    "--device",
                    "cuda",
                    "--dtype",
                    dtype,
                    "--output",
                    str(output),
                ],
                args.output / ("correctness-" + dtype + ".log"),
                args.deadline,
            )
            if code:
                raise RuntimeError("new-cohort CUDA correctness failed; timings not started")
        if json.loads(result.read_text())["status"] != "PASS":
            raise RuntimeError("stored new-cohort correctness is not PASS")
    regression = args.output / "gpu-tests.json"
    if not regression.exists():
        code = execute(
            [
                python,
                "-m",
                "pytest",
                "tests/test_issue842_checkpoint_visible_nf4.py",
                "tests/test_issue968_bnb_nf4_path_divergence.py",
                "-m",
                "gpu",
                "--no-cov",
                "-q",
            ],
            args.output / "gpu-tests.log",
            args.deadline,
        )
        write_new_json(regression, {"exit_code": code})
    if json.loads(regression.read_text())["exit_code"]:
        raise RuntimeError("new-cohort GPU regression failed")
    completed = []
    for case in cases():
        root = args.output / case["name"]
        root.mkdir(exist_ok=True)
        while True:
            rows = records(root)
            due = remaining_blocks(rows)
            if not due:
                break
            # Identity/clock violations affect the comparison, not just a disposable block.
            if rows and any(
                any("clock" in reason or "mixed identity" in reason for reason in block["reasons"])
                for block in summarize(rows)["comparisons"][0]["rejected_blocks"]
            ):
                raise RuntimeError(f"{case['name']}: cohort-wide identity/clock violation")
            if any(row.get("returncode") for row in rows):
                raise RuntimeError(f"{case['name']}: child failed; explicit review required")
            first = next_block(root)
            output = root / f"attempt-{first:03d}"
            command = [
                python,
                harness,
                "run",
                "--blocks",
                str(due),
                "--block-start",
                str(first),
                "--output",
                str(output),
            ]
            for name, value in case.items():
                if name != "name":
                    command.extend(["--" + name.replace("_", "-"), str(value)])
            if case["kind"] != "projection":
                command.extend(["--weights", str(args.weights)])
            if case["kind"] == "train":
                command.extend(["--data", str(args.data)])
            code = execute(command, root / f"attempt-{first:03d}.log", args.deadline)
            if code not in (0, 2):
                raise RuntimeError(f"{case['name']}: controller failed with exit {code}")
            attempt = json.loads((output / "result.json").read_text())
            if attempt["status"] == "FAIL" or not records(output):
                raise RuntimeError(f"{case['name']}: no usable arm records")
        result = summarize(records(root))
        snapshot = root / f"summary-{next_block(root):03d}.json"
        if not snapshot.exists():
            write_new_json(snapshot, result)
        completed.append({"name": case["name"], **result})
        Console().print(
            f"{len(completed)}/{len(cases())}: {case['name']} {result['comparisons'][0]['status']}"
        )
        # Mutable progress is explicitly separate from immutable raw arms and snapshots.
        (args.output / "progress.json").write_text(
            json.dumps(
                {
                    "status": "RUNNING",
                    "completed_cases": len(completed),
                    "total_cases": len(cases()),
                    "cases": completed,
                    "updated_unix": time.time(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return {
        "status": "COMPLETE_MEASUREMENTS_REVIEW_REQUIRED",
        "d1_complete": False,
        "cases": completed,
        "source": source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    result = run(args)
    write_new_json(args.output / "complete.json", result)
    Console().print(result["status"])


if __name__ == "__main__":
    main()
