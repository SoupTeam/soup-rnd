"""Ten-minute feasibility smoke for OLMoE on Kaggle (not part of the protocol, no result).

  python benchmarks/e2/olmoe_cache/smoke.py --base SNAPSHOT --out DIR

Runs 60 training steps twice with the freeze at step 20: hybrid K=4 without and with the
cache. Checks that the model loads and fits on two T4s, the step speed, the cache install
time, and that both runs end bit-identical (fingerprint and validation CE).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_arm as arm  # noqa: E402


def run(name: str, base: Path, out: Path, cache: bool) -> dict:
    d = out / name
    d.mkdir(parents=True)
    config = d / "config.yaml"
    config.write_text(arm.render(base, d / "adapter"), encoding="utf-8")
    cmd = [sys.executable, arm.RUNNER, "--config", str(config), "--steps", "60",
           "--output", str(d / "quality.json"), "--seed", "42", "--keep-top", str(arm.KEEP_TOP),
           "--freeze-at-step", "20"]
    cache_dir = Path("/tmp/e2_olmoe_smoke_cache") / name
    if cache:
        shutil.rmtree(cache_dir, ignore_errors=True)
        cmd += ["--cache-dir", str(cache_dir)]
    started = time.time()
    with open(d / "train.log", "x", encoding="utf-8") as log:
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    seconds = round(time.time() - started, 1)
    quality = json.loads((d / "quality.json").read_text()) if code == 0 else {}
    hybrid = quality.get("hybrid") or {}
    return {
        "name": name, "exit": code, "seconds": seconds,
        "validation_loss": quality.get("validation_loss"),
        "fingerprint": quality.get("trainable_fingerprint"),
        "cache": hybrid.get("cache"),
        "bottom_unchanged_after_freeze": hybrid.get("bottom_unchanged_after_freeze"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    results = [run("smoke_nocache", args.base, args.out, False),
               run("smoke_cache", args.base, args.out, True)]
    summary = {
        "runs": results,
        "gpu": arm.gpu_state(),
        "parity": all(r["exit"] == 0 for r in results)
        and results[0]["fingerprint"] == results[1]["fingerprint"]
        and results[0]["validation_loss"] == results[1]["validation_loss"],
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary["parity"] else 1


if __name__ == "__main__":
    sys.exit(main())
