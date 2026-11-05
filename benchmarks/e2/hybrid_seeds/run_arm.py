"""Run one arm of the E2 hybrid seed series on Kaggle (hybrid_seeds/PROTOCOL.txt 3).

  python benchmarks/e2/hybrid_seeds/run_arm.py --arm H16_s901 --base SNAPSHOT --results DIR

Arm names are <KIND>_s<SEED>: KIND is W32 (all layers), H24/H16/H8 (hybrid) or ANCHOR
(W32 through the extended runner, training only). Run from the soup-rnd root.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "recipe_check"))
import run_arm as recipe  # noqa: E402

RUNNER = "benchmarks/e2/hybrid/run_hybrid.py"
KINDS = {"W32": None, "ANCHOR": None, "H24": 24, "H16": 16, "H8": 8}
SEEDS = (42, 901, 1234)
ARMS = tuple(
    f"{kind}_s{seed}" for seed in SEEDS for kind in KINDS
    if not (seed == 42 and kind in {"W32", "H24", "H16"})  # reused from 20261105
    and not (seed != 42 and kind == "ANCHOR")
)
TEMPLATE = "W32"


def parse(arm: str) -> tuple[str, int]:
    match = re.fullmatch(r"(W32|ANCHOR|H24|H16|H8)_s(\d+)", arm)
    if match is None or arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    return match.group(1), int(match.group(2))


def render(seed: int, base: Path, output: Path) -> str:
    text = recipe.render(TEMPLATE, base, output)
    for key in ("seed", "data_seed"):
        line = f"  {key}: 42\n"
        if text.count(line) != 1:
            raise ValueError(f"template must hold '{line.strip()}' once")
        text = text.replace(line, f"  {key}: {seed}\n")
    return text


def command(kind: str, seed: int, config: Path, quality: Path) -> list[str]:
    out = [sys.executable, RUNNER, "--config", str(config), "--steps", str(recipe.STEPS),
           "--output", str(quality), "--seed", str(seed)]
    if KINDS[kind] is not None:
        out += ["--keep-top", str(KINDS[kind])]
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    args = parser.parse_args(argv)
    kind, seed = parse(args.arm)

    root = Path.cwd().resolve()
    results = args.results.resolve()
    results.relative_to(root)
    arm_dir = results / args.arm
    arm_dir.mkdir(parents=True, exist_ok=False)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    config = arm_dir / "config.yaml"
    config.write_text(render(seed, args.base, arm_dir / "adapter"), encoding="utf-8")
    quality = arm_dir / "quality.json"
    train_cmd = command(kind, seed, config, quality)
    record = {
        "schema": "e2-hybrid-seeds-arm-v1",
        "arm": args.arm, "kind": kind, "seed": seed, "training_command": train_cmd[1:],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "gpus": recipe.gpu_snapshot(),
        "template": TEMPLATE,
        "template_sha256": recipe.sha256(recipe.HERE / "configs" / f"{TEMPLATE}.yaml"),
        "config_sha256": recipe.sha256(config),
        "steps": [],
    }
    deadline = time.time() + recipe.ARM_WALL_LIMIT_S
    train = recipe.step("train", train_cmd, arm_dir / "train.log", deadline)
    record["steps"].append(train)
    wants_gate = kind != "ANCHOR"
    if train["exit"] == 0 and wants_gate:
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
    expected = 2 if wants_gate else 1
    record["ok"] = (
        all(s["exit"] == 0 for s in record["steps"]) and len(record["steps"]) == expected
    )
    (arm_dir / "arm.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
