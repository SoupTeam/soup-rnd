"""Run one E2 recipe-check arm on Kaggle: train (run_quality.py), then gate (run_gate.py).

  python benchmarks/e2/recipe_check/run_arm.py --arm W32 --base SNAPSHOT_DIR --results DIR

Run from the soup-rnd root (the unpacked kit). DIR must be inside the root and the arm
folder must not exist yet: an arm is never run twice into the same place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ("W32", "W24", "S32", "S24", "S16")
STEPS = 4000
ARM_WALL_LIMIT_S = 5 * 3600


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(arm: str, base: Path, output: Path) -> str:
    template = (HERE / "configs" / f"{arm}.yaml").read_text(encoding="utf-8")
    if template.count("__BASE__") != 1 or template.count("__OUTPUT__") != 1:
        raise ValueError(f"{arm}: template must name __BASE__ and __OUTPUT__ once")
    return template.replace("__BASE__", str(base)).replace("__OUTPUT__", str(output))


def gpu_snapshot() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {type(exc).__name__}"


def step(name: str, command: list[str], log: Path, deadline: float) -> dict:
    started = time.time()
    with open(log, "x", encoding="utf-8") as handle:
        try:
            done = subprocess.run(
                command, stdout=handle, stderr=subprocess.STDOUT,
                timeout=max(1.0, deadline - started), env=os.environ.copy(),
            )
            code = done.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    return {"step": name, "exit": code, "seconds": round(time.time() - started, 1)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
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
    config.write_text(render(args.arm, args.base, arm_dir / "adapter"), encoding="utf-8")
    quality = arm_dir / "quality.json"
    record = {
        "schema": "e2-recipe-arm-v1",
        "arm": args.arm,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "gpus": gpu_snapshot(),
        "template_sha256": sha256(HERE / "configs" / f"{args.arm}.yaml"),
        "config_sha256": sha256(config),
        "steps": [],
    }
    deadline = time.time() + ARM_WALL_LIMIT_S
    train = step(
        "train",
        [sys.executable, "benchmarks/e2/quality/run_quality.py", "--config", str(config),
         "--steps", str(STEPS), "--output", str(quality)],
        arm_dir / "train.log", deadline,
    )
    record["steps"].append(train)
    if train["exit"] == 0:
        record["steps"].append(step(
            "gate",
            [sys.executable, "benchmarks/e2/recipe_check/run_gate.py",
             "--training-config", str(config), "--quality-report", str(quality),
             "--output-dir", str(arm_dir / "gate")],
            arm_dir / "gate.log", deadline,
        ))
    adapter = arm_dir / "adapter" / "adapter_model.safetensors"
    if adapter.is_file():
        record["adapter_sha256"] = sha256(adapter)
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    record["ok"] = all(s["exit"] == 0 for s in record["steps"]) and len(record["steps"]) == 2
    (arm_dir / "arm.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
