
"""Repeated baseline versus E2 frozen-prefix-cache benchmark."""

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "e2" / "long_repeated"
RESULTS.mkdir(parents=True, exist_ok=True)

CONFIGS = {
    "baseline": "experiments/e2_long_baseline.yaml",
    "cached": "experiments/e2_long_cached.yaml",
}

# Three independent runs per mode, balanced order.
RUN_ORDER = [
    "baseline", "cached",
    "cached", "baseline",
    "baseline", "cached",
]

STEPS = 64
WARMUP = 8


def median(values):
    return statistics.median(values) if values else None


def run_one(index, mode):
    path = RESULTS / f"{index:02d}_{mode}.json"

    command = [
        sys.executable, "-m", "soup_cli",
        "bench", "train",
        "--config", CONFIGS[mode],
        "--steps", str(STEPS),
        "--warmup", str(WARMUP),
        "--output", str(path),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    print(f"\n[{index}/{len(RUN_ORDER)}] {mode}", flush=True)

    subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
    )

    report = json.loads(path.read_text())

    if not report.get("valid", False):
        raise RuntimeError(
            f"Invalid benchmark: {mode}, "
            f"failures={report.get('failures')}"
        )

    timing = report["timing"]
    throughput = report.get("throughput", {})
    memory = report.get("memory", {})
    epochs = report.get("epoch_timing", [])

    hits = sum(
        e["cache_hits"]
        for e in epochs
        if e.get("cache_hits") is not None
    )
    misses = sum(
        e["cache_misses"]
        for e in epochs
        if e.get("cache_misses") is not None
    )

    return {
        "run": index,
        "mode": mode,
        "config_hash": report.get("config_hash"),
        "steps": timing["counted_steps"],
        "median_step_ms": timing["median_seconds"] * 1000,
        "p95_step_ms": timing["p95_seconds"] * 1000,
        "tokens_per_second": throughput.get(
            "useful_tokens_per_second"
        ),
        "peak_allocated_bytes": memory.get(
            "max_memory_allocated_bytes"
        ),
        "peak_reserved_bytes": memory.get(
            "max_memory_reserved_bytes"
        ),
        "cache_hits": hits,
        "cache_misses": misses,
        "epoch_timing": epochs,
        "report_path": str(path.relative_to(ROOT)),
    }


def summarize(rows):
    grouped = {
        mode: [r for r in rows if r["mode"] == mode]
        for mode in CONFIGS
    }

    summary = {}

    for mode, runs in grouped.items():
        summary[mode] = {
            "runs": len(runs),
            "median_step_ms": median([
                r["median_step_ms"] for r in runs
            ]),
            "median_p95_ms": median([
                r["p95_step_ms"] for r in runs
            ]),
            "median_tokens_per_second": median([
                r["tokens_per_second"]
                for r in runs
                if r["tokens_per_second"] is not None
            ]),
            "median_peak_allocated_bytes": median([
                r["peak_allocated_bytes"]
                for r in runs
                if r["peak_allocated_bytes"] is not None
            ]),
            "total_cache_hits": sum(
                r["cache_hits"] for r in runs
            ),
            "total_cache_misses": sum(
                r["cache_misses"] for r in runs
            ),
        }

    a = summary["baseline"]["median_step_ms"]
    b = summary["cached"]["median_step_ms"]

    summary["speedup"] = a / b if b > 0 else None
    return summary


def main():
    rows = []

    for index, mode in enumerate(RUN_ORDER, 1):
        rows.append(run_one(index, mode))

    summary = summarize(rows)

    output = {
        "steps": STEPS,
        "warmup": WARMUP,
        "run_order": RUN_ORDER,
        "runs": rows,
        "summary": summary,
    }

    path = RESULTS / "summary.json"
    path.write_text(json.dumps(output, indent=2))

    print("\n=== FINAL E2 BENCHMARK ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
