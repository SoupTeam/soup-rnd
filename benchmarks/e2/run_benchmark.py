
"""Repeated E2 versus Top-K LoRA benchmark."""

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "e2" / "repeated"
RESULTS.mkdir(parents=True, exist_ok=True)

CONFIGS = {
    "baseline": "experiments/e2_baseline.yaml",
    "cached": "experiments/e2_smoke_fresh.yaml",
}

RUN_ORDER = [
    "baseline", "cached",
    "cached", "baseline",
    "baseline", "cached",
]

results = {"baseline": [], "cached": []}

for index, mode in enumerate(RUN_ORDER, start=1):
    report_path = RESULTS / f"{index:02d}_{mode}.json"

    command = [
        sys.executable, "-m", "soup_cli",
        "bench", "train",
        "--config", CONFIGS[mode],
        "--steps", "8",
        "--warmup", "2",
        "--output", str(report_path),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    print(f"\n[{index}/{len(RUN_ORDER)}] Running {mode}", flush=True)

    subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
    )

    with report_path.open() as file:
        report = json.load(file)

    if not report["valid"]:
        raise RuntimeError(f"Invalid benchmark: {mode}")

    results[mode].append(report["timing"]["median_seconds"])

baseline = statistics.median(results["baseline"])
cached = statistics.median(results["cached"])

print("\n=== E2 BENCHMARK RESULTS ===")
print(f"Baseline median: {baseline * 1000:.3f} ms")
print(f"E2 median:       {cached * 1000:.3f} ms")
print(f"Speedup:         {baseline / cached:.3f}x")

