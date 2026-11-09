"""Write the two Kaggle notebooks of the OLMoE cache series next to this file."""

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "recipe_make_notebook", HERE.parent / "recipe_check" / "make_notebook.py"
)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

MODEL = r'''from huggingface_hub import snapshot_download
from kaggle_secrets import UserSecretsClient

REVISION = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
BASE = Path(snapshot_download(
    "allenai/OLMoE-1B-7B-0924", revision=REVISION, cache_dir="/tmp/hf",
    allow_patterns=["*.json", "model-*.safetensors", "tokenizer*"],
    token=UserSecretsClient().get_secret("HF_TOKEN"),
))
assert BASE.name == REVISION, BASE
print(BASE, sorted(p.name for p in BASE.iterdir()))'''

INTRO = """# E2 OLMoE cache ABBA (Kaggle, GPU T4 x2)

Protocol: `benchmarks/e2/olmoe_cache/PROTOCOL.txt`. One session runs the arms in `ARMS`.

Before running: Settings -> Accelerator **GPU T4 x2**, Internet **on**; Add-ons -> Secrets ->
`HF_TOKEN` (any Hugging Face read token); Add data -> the uploaded dataset with
`e2_recipe_kit.zip`. Then **Save Version -> Save & Run All (Commit)**; do not run cells
in the editor. Download `e2oc_results_*.zip` from the version's Output."""

SMOKE_INTRO = """# E2 OLMoE smoke (Kaggle, GPU T4 x2) - feasibility only, no result

Runs 60 steps with and without the cache to check that OLMoE fits and the cache installs.
Same setup as the series notebook. **Save Version -> Save & Run All (Commit)**; read the log."""

ARMS = (
    "# One session, the protocol order (about 5 hours). Do not change.\n"
    'ARMS = ["C",\n'
    '        "A_c1_1", "B_c1_1", "B_c1_2", "A_c1_2",\n'
    '        "A_c2_1", "B_c2_1", "B_c2_2", "A_c2_2",\n'
    '        "A_c3_1", "B_c3_1", "B_c3_2", "A_c3_2"]'
)

RUN = base.RUN.replace(
    "benchmarks/e2/recipe_check/run_arm.py", "benchmarks/e2/olmoe_cache/run_arm.py"
).replace('"e2rc_results"', '"e2oc_results"')
PACK = base.PACK.replace("e2rc_results_", "e2oc_results_")

SMOKE_RUN = r'''stamp = time.strftime("%Y%m%d_%H%M%S")
OUT = ROOT / "e2oc_smoke" / stamp
OUT.mkdir(parents=True)
done = subprocess.run([sys.executable, "benchmarks/e2/olmoe_cache/smoke.py",
                       "--base", str(BASE), "--out", str(OUT)])
print("smoke exit", done.returncode)'''
SMOKE_PACK = r'''out = Path(f"/kaggle/working/e2oc_smoke_{stamp}.zip")
with zipfile.ZipFile(out, "x", zipfile.ZIP_DEFLATED) as z:
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and not path.name.endswith((".safetensors", ".weights.pt", ".bin")):
            z.write(path, Path(stamp) / path.relative_to(OUT))
print(out, out.stat().st_size)'''


def write(name, intro, params, run, pack):
    notebook = {
        "cells": [base.cell("markdown", intro), base.cell("code", params),
                  base.cell("code", base.SETUP), base.cell("code", MODEL),
                  base.cell("code", run), base.cell("code", pack)],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    name.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(name)


write(HERE / "e2_olmoe_cache.ipynb", INTRO, ARMS, RUN, PACK)
write(
    HERE / "e2_olmoe_smoke.ipynb",
    SMOKE_INTRO,
    "# Feasibility smoke, not part of the protocol.\nARMS = []",
    SMOKE_RUN,
    SMOKE_PACK,
)
