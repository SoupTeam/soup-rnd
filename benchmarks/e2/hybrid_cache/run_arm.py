"""Run one run of the E2 hybrid-cache series on Kaggle (hybrid_cache/PROTOCOL.txt 3).

  python benchmarks/e2/hybrid_cache/run_arm.py --arm B_c1_2 --base SNAPSHOT --results DIR

Runs: C (hybrid K=8, no cache), then A_c<n>_1, B_c<n>_1, B_c<n>_2, A_c<n>_2 for n = 1..3.
Training only, 6,000 steps, seed 42. Run from the soup-rnd root.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "recipe_check"))
import run_arm as recipe  # noqa: E402

RUNNER = "benchmarks/e2/hybrid/run_hybrid.py"
STEPS = 6000
KEEP_TOP = 8
CACHE_ROOT = Path("/tmp/e2_hybrid_cache")
ORDER = ("C",) + tuple(
    f"{kind}_c{cycle}_{index}"
    for cycle in (1, 2, 3)
    for kind, index in (("A", 1), ("B", 1), ("B", 2), ("A", 2))
)


def gpu_state() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=index,temperature.gpu,clocks.sm,clocks.mem,"
             "power.draw,clocks_throttle_reasons.active,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {type(exc).__name__}"


def command(arm: str, config: Path, quality: Path, cache_dir: Path) -> list[str]:
    out = [sys.executable, RUNNER, "--config", str(config), "--steps", str(STEPS),
           "--output", str(quality), "--seed", "42"]
    if arm == "C" or arm.startswith("B"):
        out += ["--keep-top", str(KEEP_TOP)]
    if arm.startswith("B"):
        out += ["--cache-dir", str(cache_dir)]
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ORDER)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    args = parser.parse_args(argv)

    root = Path.cwd().resolve()
    results = args.results.resolve()
    results.relative_to(root)
    arm_dir = results / args.arm
    arm_dir.mkdir(parents=True, exist_ok=False)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    config = arm_dir / "config.yaml"
    config.write_text(recipe.render("W32", args.base, arm_dir / "adapter"), encoding="utf-8")
    quality = arm_dir / "quality.json"
    cache_dir = CACHE_ROOT / args.arm
    if cache_dir.exists():
        raise RuntimeError(f"cache directory already exists: {cache_dir}")
    train_cmd = command(args.arm, config, quality, cache_dir)
    record = {
        "schema": "e2-hybrid-cache-run-v1",
        "arm": args.arm,
        "position": ORDER.index(args.arm),
        "training_command": train_cmd[1:],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "gpus": recipe.gpu_snapshot(),
        "gpu_state_before": gpu_state(),
        "template_sha256": recipe.sha256(recipe.HERE / "configs" / "W32.yaml"),
        "config_sha256": recipe.sha256(config),
    }
    started = time.time()
    train = recipe.step("train", train_cmd, arm_dir / "train.log",
                        started + recipe.ARM_WALL_LIMIT_S)
    record["process_wall_seconds"] = time.time() - started
    record["steps"] = [train]
    record["gpu_state_after"] = gpu_state()
    if cache_dir.exists():
        files = [p for p in cache_dir.iterdir() if p.is_file()]
        record["cache_dir_at_end"] = {
            "files": len(files), "bytes": sum(p.stat().st_size for p in files),
        }
        shutil.rmtree(cache_dir)
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    record["ok"] = train["exit"] == 0
    (arm_dir / "arm.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
