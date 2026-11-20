"""Kaggle entry: Kanana-2-30B-A3B on one T4 with the mbr_measure driver (harness only).

Stages, from ``MBR_STAGES`` (comma list, default ``probe``):

  probe    download the pinned checkpoint to /tmp, build the NF4 ``per_expert`` cache, run one
           A0 window (k=8, S=64) and time every step. Answers: does it fit, how long does each
           part take, does the model load at all.
  bits     deterministic ladder k=1,2,4,8 for A0, B0 and C at S=64.
  speed    A-B-C-C-B-A blocks over the four windows (``MBR_BLOCKS``, default 5).
  profile  one phase-profiled run per arm and window.

Also the entry of the Vast run (``mbr_measure_vast.sh`` sets ``MBR_SRC``, ``MBR_WORK``,
``MBR_OUT``, ``MBR_SKIP_PIP``, ``MBR_DEADLINE_HOURS``).

Needs the repo source as a dataset (``/kaggle/input/.../{src,benchmarks}``). /kaggle/working has
only about 20 GiB, so the checkpoint and cache live in /tmp (gone after the run); only the small
JSON reports go to /kaggle/working, written after every stage.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # exactly one device
os.environ.setdefault("TMPDIR", "/tmp")             # the shard writer accepts $TMPDIR paths
REPO = "kakaocorp/kanana-2-30b-a3b-base-2601"
REVISION = "0d31365acf656e5f9a593cacc4913e9c0d36cb1c"
OUT = Path(os.environ.get("MBR_OUT", "/kaggle/working"))
WORK = Path(os.environ.get("MBR_WORK", "/tmp/mbr"))
PIP = ["transformers==5.20.2", "peft==0.21.2", "huggingface_hub", "safetensors", "rich",
       "pydantic", "bitsandbytes", "pyyaml", "typer", "psutil", "datasets", "tiktoken"]
T0 = time.time()
REPORT: dict = {"stages": {}, "timings": {}}


def log(message: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {message}", flush=True)


def sh(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, **kwargs)


def save() -> None:
    (OUT / "kanana-report.json").write_text(json.dumps(REPORT, indent=2, default=str))


def timed(name: str, fn: Callable[[], T]) -> T:
    started = time.time()
    try:
        return fn()
    finally:
        REPORT["timings"][name] = time.time() - started
        log(f"{name}: {REPORT['timings'][name]:.1f}s")
        save()


def main() -> int:
    if not os.environ.get("MBR_SKIP_PIP"):
        log(sh([sys.executable, "-m", "pip", "install", "-q", *PIP]).stderr[-300:])
        sh([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
    root = Path(glob.glob(f"{os.environ.get('MBR_SRC', '/kaggle/input')}/**/src/soup_cli",
                          recursive=True)[0]).parent.parent
    for sub in ("src", "benchmarks/harness", "."):
        sys.path.insert(0, str(root / sub))
    harness = root / "benchmarks/harness"
    for name in ("mbr_measure.py", "mbr_measure_kanana_kaggle.py"):
        log(f"source {name} {hashlib.sha256((harness / name).read_bytes()).hexdigest()[:12]}")
    import oc_arm_run
    from huggingface_hub import snapshot_download

    source, cache = WORK / "kanana", WORK / "cache"
    stages = os.environ.get("MBR_STAGES", "probe").split(",")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        str(root / s) for s in ("src", ".", "benchmarks/harness")))
    REPORT["disk_before"] = shutil.disk_usage("/tmp")._asdict()
    timed("download", lambda: snapshot_download(REPO, revision=REVISION, local_dir=str(source),
                                                max_workers=8))
    from argparse import Namespace
    timed("prepare_nf4", lambda: oc_arm_run.prepare(Namespace(
        source=str(source), cache=str(cache), device="cuda", double_quant=False)))
    REPORT["disk_after_prepare"] = shutil.disk_usage("/tmp")._asdict()
    save()
    common = ["--source", str(source), "--cache", str(cache), "--steps", "3"]
    script = str(harness / "mbr_measure.py")

    def run(command: list[str], label: str) -> dict:
        done = timed(label, lambda: sh([sys.executable, script, *command], env=env))
        return {"returncode": done.returncode, "stdout": done.stdout[-3000:],
                "stderr": done.stderr[-3000:]}

    for stage in stages:
        if stage == "probe":
            REPORT["stages"]["probe"] = run(
                ["run", "--arm", "A0", "--k", "8", "--seq", "64", "--kind", "speed",
                 "--out", str(OUT / "probe"), *common], "probe_run")
        else:
            extra = {"bits": ["--seq", "64"], "speed": ["--blocks",
                                                        os.environ.get("MBR_BLOCKS", "5")],
                     "profile": [], "lever-bits": ["--seq", "64"],
                     "lever-speed": ["--blocks", os.environ.get("MBR_BLOCKS", "1")]}[stage]
            if os.environ.get("MBR_DEADLINE_HOURS"):
                extra += ["--deadline-hours", os.environ["MBR_DEADLINE_HOURS"]]
            REPORT["stages"][stage] = run([stage, "--out", str(OUT / stage), *common, *extra],
                                          stage)
        save()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:  # noqa: BLE001
        REPORT["error"] = traceback.format_exc()
        save()
        raise
