"""Write e2_recipe_check.ipynb (the Kaggle notebook) next to this file."""

import json
from pathlib import Path

INTRO = """# E2 recipe check (Kaggle, GPU T4 x2)

Protocol: `benchmarks/e2/recipe_check/PROTOCOL.txt`. One session runs the arms in `ARMS`.

Before running: Settings -> Accelerator **GPU T4 x2**, Internet **on**; Add-ons -> Secrets ->
`HF_TOKEN` (a Hugging Face read token whose account accepted the Mistral-7B-v0.3 terms);
Add data -> the uploaded dataset with `e2_recipe_kit.zip`.
Then **Save Version -> Save & Run All (Commit)**. Download `e2rc_results_*.zip` from Output."""

PARAMS = """# Session 1: ["W32", "W24"]   Session 2: ["S32", "S24"]   Session 3 (optional): ["S16"]
ARMS = ["W32", "W24"]"""

SETUP = r'''import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

gpus = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                      capture_output=True, text=True).stdout.split("\n")
gpus = [g for g in gpus if g.strip()]
print(gpus)
assert len(gpus) == 2 and all("T4" in g for g in gpus), "Set the accelerator to GPU T4 x2"

kits = glob.glob("/kaggle/input/**/e2_recipe_kit.zip", recursive=True)
assert len(kits) == 1, f"expected one e2_recipe_kit.zip in the inputs, found {kits}"
kit_sha = hashlib.sha256(Path(kits[0]).read_bytes()).hexdigest()
print("kit", kits[0], kit_sha)
ROOT = Path("/kaggle/working/soup-rnd")
if ROOT.exists():
    shutil.rmtree(ROOT)
with zipfile.ZipFile(kits[0]) as z:
    z.extractall(ROOT)
os.chdir(ROOT)
print(subprocess.check_output([sys.executable, "benchmarks/e2/recipe_check/kit.py",
                               "verify", str(ROOT)], text=True))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[train]"], check=True)
freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)'''

MODEL = r'''from huggingface_hub import snapshot_download
from kaggle_secrets import UserSecretsClient

REVISION = "caa1feb0e54d415e2df31207e5f4e273e33509b1"
BASE = Path(snapshot_download(
    "mistralai/Mistral-7B-v0.3", revision=REVISION, cache_dir="/tmp/hf",
    allow_patterns=["*.json", "model-*.safetensors", "tokenizer*"],
    token=UserSecretsClient().get_secret("HF_TOKEN"),
))
assert BASE.name == REVISION, BASE
print(BASE, sorted(p.name for p in BASE.iterdir()))'''

RUN = r'''stamp = time.strftime("%Y%m%d_%H%M%S")
RESULTS = ROOT / "e2rc_results" / stamp
RESULTS.mkdir(parents=True)
(RESULTS / "session.json").write_text(json.dumps({
    "arms": ARMS, "kit_sha256": kit_sha, "gpus": gpus, "pip_freeze": freeze.splitlines(),
}, indent=1))
for arm in ARMS:
    print("=== arm", arm, time.strftime("%H:%M:%S"), flush=True)
    done = subprocess.run([sys.executable, "benchmarks/e2/recipe_check/run_arm.py",
                           "--arm", arm, "--base", str(BASE), "--results", str(RESULTS)])
    print("arm", arm, "exit", done.returncode, flush=True)
    if done.returncode != 0:
        print("Stopping: an arm failed. Do not rerun; send the results zip.")
        break'''

PACK = r'''out = Path(f"/kaggle/working/e2rc_results_{stamp}.zip")
with zipfile.ZipFile(out, "x", zipfile.ZIP_DEFLATED) as z:
    for path in sorted(RESULTS.rglob("*")):
        if path.is_file() and not path.name.endswith((".safetensors", ".weights.pt", ".bin")):
            z.write(path, Path(stamp) / path.relative_to(RESULTS))
print(out, out.stat().st_size, hashlib.sha256(out.read_bytes()).hexdigest())'''


def cell(kind, text):
    body = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        body.update({"execution_count": None, "outputs": []})
    return body


notebook = {
    "cells": [cell("markdown", INTRO), cell("code", PARAMS), cell("code", SETUP),
              cell("code", MODEL), cell("code", RUN), cell("code", PACK)],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                "name": "python3"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = Path(__file__).with_name("e2_recipe_check.ipynb")
out.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(out)
