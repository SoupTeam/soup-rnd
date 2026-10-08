
"""Reproducible A-B-B-A benchmark for E2."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CONFIGS = {
    "A": "experiments/e2_long_baseline.yaml",
    "B": "experiments/e2_long_cached.yaml",
}


def command_output(command):
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return {
            "returncode": None,
            "stdout": "",
            "stderr": f"Command not available: {command[0]}",
        }

    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def machine_state():
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "cpu": platform.processor(),
        "nvidia_smi": command_output([
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,memory.used,"
            "temperature.gpu,clocks.sm,utilization.gpu",
            "--format=csv,noheader",
        ]),
        "mac_hardware": (
            command_output([
                "system_profiler",
                "SPHardwareDataType",
            ])
            if platform.system() == "Darwin"
            else None
        ),
    }


def run_benchmark(mode, index, directory, steps, warmup):
    report_path = directory / f"{index:02d}_{mode}.json"

    command = [
        sys.executable,
        "-m",
        "soup_cli",
        "bench",
        "train",
        "--config",
        CONFIGS[mode],
        "--steps",
        str(steps),
        "--warmup",
        str(warmup),
        "--output",
        str(report_path),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    before = machine_state()

    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    after = machine_state()

    (directory / f"{index:02d}_{mode}.log").write_text(
        result.stdout + "\n" + result.stderr,
        encoding="utf-8",
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Benchmark {mode} failed. "
            f"See {index:02d}_{mode}.log"
        )

    with report_path.open() as file:
        report = json.load(file)

    return {
        "mode": mode,
        "index": index,
        "report": report,
        "machine_before": before,
        "machine_after": after,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=4)
    args = parser.parse_args()

    if args.cycles < 1 or args.steps <= args.warmup:
        parser.error("Require cycles >= 1 and steps > warmup")

    directory = ROOT / "benchmarks" / "e2" / "abba_results"
    directory.mkdir(parents=True, exist_ok=True)

    order = ["A", "B", "B", "A"] * args.cycles
    results = []

    for index, mode in enumerate(order, start=1):
        print(
            f"[{index}/{len(order)}] Running {mode}",
            flush=True,
        )
        result = run_benchmark(
            mode,
            index,
            directory,
            args.steps,
            args.warmup,
        )
        results.append(result)
        report = result["report"]

        if not report["valid"]:
            raise RuntimeError(
                f"Invalid benchmark: run {index}"
            )

        if report["steps_measured"] != args.steps:
            raise RuntimeError(
                f"Unexpected step count: run {index}"
            )

        if not report.get("epoch_timing"):
            raise RuntimeError(
                f"Epoch timing missing: run {index}"
            )
        (directory / "manifest.json").write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )
    print("\n=== E2 EPOCH ANALYSIS ===")

    for cycle in range(args.cycles):
        cycle_runs = results[cycle * 4:(cycle + 1) * 4]

        print(f"\nCycle {cycle + 1}")

        for mode in ("A", "B"):
            selected = [r for r in cycle_runs if r["mode"] == mode]

            first_epoch = []
            second_epoch = []

            for run in selected:
                report = run["report"]
                epochs = report.get("epoch_timing", [])

                if len(epochs) != 2:
                    raise RuntimeError(
                        f"Expected 2 epochs, got {len(epochs)}"
                    )

                first_epoch.append(epochs[0]["duration_seconds"])
                second_epoch.append(epochs[1]["duration_seconds"])

            print(
                f"{mode}: "
                f"epoch1={statistics.median(first_epoch) * 1000:.3f} ms, "
                f"epoch2={statistics.median(second_epoch) * 1000:.3f} ms"
            )
    baseline = [
        r["report"]["timing"]["median_seconds"]
        for r in results if r["mode"] == "A"
    ]
    cached = [
        r["report"]["timing"]["median_seconds"]
        for r in results if r["mode"] == "B"
    ]
    token_counts = [
        r["report"]["tokens"]["useful"]
        for r in results
    ]

    if len(set(token_counts)) != 1:
        raise RuntimeError(
            f"Supervised token counts differ: {token_counts}"
        )

    print(f"\nSupervised tokens equal: {token_counts[0]}")
    speedup = statistics.median(baseline) / statistics.median(cached)

    print("\n=== E2 A-B-B-A RESULTS ===")
    print(f"Baseline median: {statistics.median(baseline)*1000:.3f} ms")
    print(f"E2 median:       {statistics.median(cached)*1000:.3f} ms")
    print(f"Observed ratio:  {speedup:.3f}x")
    print("Speed verdict: NO VERDICT (protocol thresholds pending)")
    print(f"Manifest: {directory / 'manifest.json'}")


if __name__ == "__main__":
    main()

