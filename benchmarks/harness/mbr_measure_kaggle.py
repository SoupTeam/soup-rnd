"""Kaggle entry for the mbr_measure driver: smoke on a tiny synthetic DeepSeek-V3 MoE
(CPU, or one GPU when there is one).

Checks the plumbing (MBPP-train text and hash, every arm, bit ladder verdicts, OOM-free rows,
report) with no Kanana download. It says nothing about CUDA, NF4, sm_89 or speed. Needs the repo
source attached as a dataset (``/kaggle/input/<dataset>/{src,benchmarks}``); writes
``/kaggle/working/smoke-rows.jsonl`` and ``smoke-result.json``.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # the runs need exactly one device
OUT = Path(os.environ.get("MBR_OUT", "/kaggle/working"))
WORK = Path(os.environ.get("MBR_WORK", "/tmp/mbr"))
PIP = ["transformers==5.20.2", "peft==0.21.2", "huggingface_hub", "safetensors", "rich",
       "pydantic", "bitsandbytes", "pyyaml", "typer", "psutil", "datasets"]


def sh(command: list[str]) -> str:
    done = subprocess.run(command, capture_output=True, text=True)
    return (done.stdout + done.stderr).strip()


def main() -> int:
    if not os.environ.get("MBR_SKIP_PIP"):
        print(sh([sys.executable, "-m", "pip", "install", "-q", *PIP])[-400:], flush=True)
        print(sh([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])[-200:], flush=True)


    try:
        import bitsandbytes
        import transformers.utils as tu
        print("bitsandbytes", bitsandbytes.__version__, "available:",
              tu.is_bitsandbytes_available(), flush=True)
    except Exception:  # noqa: BLE001 - diagnostic only
        print("bitsandbytes import failed:", traceback.format_exc()[-1500:], flush=True)
    roots = glob.glob(f"{os.environ.get('MBR_SRC', '/kaggle/input')}/**/src/soup_cli",
                      recursive=True)
    root = Path(roots[0]).parent.parent
    for sub in ("src", "benchmarks/harness", "."):
        sys.path.insert(0, str(root / sub))
    import hashlib

    import b13_moe_fixture as fixture
    import oc_arm_run
    for name in ("mbr_measure.py", "mbr_measure_kaggle.py"):
        digest = hashlib.sha256((root / "benchmarks/harness" / name).read_bytes()).hexdigest()
        print("source", name, digest[:12], flush=True)
    from transformers import AutoTokenizer

    vocab = len(AutoTokenizer.from_pretrained("gpt2"))
    weights = fixture.moe_weights_dir(WORK / "tiny", fixture.moe_config(
        vocab_size=vocab, max_position_embeddings=256))
    from argparse import Namespace

    import torch
    device = "cuda" if torch.cuda.is_available() and not os.environ.get("MBR_CPU") else "cpu"
    print("device", device, torch.cuda.get_device_name(0) if device == "cuda" else "", flush=True)
    oc_arm_run.prepare(Namespace(source=str(weights), cache=str(WORK / "cache"), device=device,
                                 double_quant=False))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        str(root / s) for s in ("src", ".", "benchmarks/harness")))
    result: dict = {}
    for name, extra in (("bits", ["--seq", "64"]), ("speed", ["--blocks", "1"]), ("profile", [])):
        out = OUT / f"smoke-{name}"
        done = subprocess.run(
            [sys.executable, str(root / "benchmarks/harness/mbr_measure.py"), name,
             "--source", str(weights), "--cache", str(WORK / "cache"), "--out", str(out),
             "--tokenizer", "gpt2", "--device", "cpu" if device == "cpu" else "auto",
             "--steps", "2", *extra],
            capture_output=True, text=True, env=env)
        result[name] = {"returncode": done.returncode, "stdout": done.stdout[-2000:],
                        "stderr": done.stderr[-2000:]}
        rows = out / "rows.jsonl"
        result[name]["rows"] = ([json.loads(x) for x in rows.read_text().splitlines()]
                                if rows.exists() else [])
        print(name, done.returncode, len(result[name]["rows"]), flush=True)
    (OUT / "smoke-result.json").write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:  # noqa: BLE001
        (OUT / "smoke-error.txt").write_text(traceback.format_exc())
        raise
