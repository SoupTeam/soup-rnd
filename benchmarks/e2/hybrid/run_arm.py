"""Run one E2 hybrid arm on Kaggle (hybrid/PROTOCOL.txt section 3): train, then gate.

  python benchmarks/e2/hybrid/run_arm.py --arm H24 --base SNAPSHOT_DIR --results DIR

Run from the soup-rnd root (the unpacked kit). The arm folder must not exist yet.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "recipe_check"))
import run_arm as recipe  # noqa: E402

# arm -> (training script, extra arguments); every arm uses the W32 template.
ARMS = {
    "H24": ("benchmarks/e2/hybrid/run_hybrid.py", ["--keep-top", "24"]),
    "H16": ("benchmarks/e2/hybrid/run_hybrid.py", ["--keep-top", "16"]),
    "W32": ("benchmarks/e2/quality/run_quality.py", []),
}
TEMPLATE = "W32"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    args = parser.parse_args(argv)

    root = Path.cwd().resolve()
    results = args.results.resolve()
    results.relative_to(root)
    arm_dir = results / args.arm
    arm_dir.mkdir(parents=True, exist_ok=False)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    script, extra = ARMS[args.arm]
    config = arm_dir / "config.yaml"
    config.write_text(recipe.render(TEMPLATE, args.base, arm_dir / "adapter"), encoding="utf-8")
    quality = arm_dir / "quality.json"
    record = {
        "schema": "e2-hybrid-arm-v1",
        "arm": args.arm,
        "training_script": script,
        "training_args": extra,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "gpus": recipe.gpu_snapshot(),
        "template": TEMPLATE,
        "template_sha256": recipe.sha256(recipe.HERE / "configs" / f"{TEMPLATE}.yaml"),
        "config_sha256": recipe.sha256(config),
        "steps": [],
    }
    deadline = time.time() + recipe.ARM_WALL_LIMIT_S
    train = recipe.step(
        "train",
        [sys.executable, script, "--config", str(config), "--steps", str(recipe.STEPS),
         "--output", str(quality), *extra],
        arm_dir / "train.log", deadline,
    )
    record["steps"].append(train)
    if train["exit"] == 0:
        record["steps"].append(recipe.step(
            "gate",
            [sys.executable, "benchmarks/e2/recipe_check/run_gate.py",
             "--training-config", str(config), "--quality-report", str(quality),
             "--output-dir", str(arm_dir / "gate")],
            arm_dir / "gate.log", deadline,
        ))
    adapter = arm_dir / "adapter" / "adapter_model.safetensors"
    if adapter.is_file():
        record["adapter_sha256"] = recipe.sha256(adapter)
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    record["ok"] = all(s["exit"] == 0 for s in record["steps"]) and len(record["steps"]) == 2
    (arm_dir / "arm.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
