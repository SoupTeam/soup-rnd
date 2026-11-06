"""B1.4 ticket 08: Kaggle notebook entry for the real small MoE correctness run.

Runs as a Kaggle script kernel with the repository source attached as a dataset
(``/kaggle/input/<dataset>/{src,benchmarks}``). Two stages:

  probe  environment, installs, import chain, and the harness on the tiny synthetic
         checkpoint (float32 and bfloat16). No model download.
  run    download the pinned checkpoint, check the ticket 08 definition, run
         ``b14_real_moe_correctness.run_correctness`` in bfloat16 on CPU.

The stage comes from the environment variable ``B14_STAGE``; the kernel's own script is a
three-line wrapper that sets it and runs this file from the dataset. Evidence goes to
``/kaggle/working/result-<stage>.json``.
"""

from __future__ import annotations

import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = "ByteDance-Seed/academic-ds-9B"
REVISION = "2a205727f616995ad4aba57806f8e2d0d2f28743"
OUT = Path(os.environ.get("B14_OUT", "/kaggle/working"))
WORK = Path("/tmp/b14")
PIP = ["transformers==5.20.2", "peft==0.21.2", "huggingface_hub", "safetensors", "rich", "pydantic",
       "pyyaml", "typer", "psutil"]
TEXTS = [
    "The capital of France is Paris, and the river Seine runs through the city.",
    "Mixture of experts models route each token to a small subset of experts.",
]

T0 = time.time()
EVIDENCE: dict = {"stages": {}}


def log(message: str) -> None:
    # bare print on purpose: this entry runs before `rich` is installed (see PIP), and a Kaggle
    # kernel keeps only stdout in its log.
    print(f"[{time.time() - T0:7.1f}s] {message}", flush=True)


def sh(command: list[str]) -> str:
    done = subprocess.run(command, capture_output=True, text=True)
    return (done.stdout + done.stderr).strip()


def attach_source() -> Path:
    base = os.environ.get("B14_SRC", "/kaggle/input")
    roots = glob.glob(f"{base}/**/src/soup_cli", recursive=True)
    if not roots:
        raise RuntimeError(f"repository source not found: no src/soup_cli under {base}")
    root = Path(roots[0]).parent.parent
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "benchmarks" / "harness"))
    sys.path.insert(0, str(root))
    return root


def environment() -> dict:
    import psutil

    memory = psutil.virtual_memory()
    return {
        "platform": platform.platform(), "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(), "ram_total_bytes": memory.total,
        "ram_available_bytes": memory.available,
        "disk_tmp": shutil.disk_usage("/tmp")._asdict(),
        "disk_working": shutil.disk_usage(str(OUT))._asdict(),
        "cpu_model": next(
            (line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo")
             if line.startswith("model name")), ""),
        "gpu_note": "CPU-only run; any GPU on the host is not used",
    }


def versions() -> dict:
    import importlib.metadata as md

    return {name: md.version(name) for name in
            ("torch", "transformers", "peft", "safetensors", "huggingface_hub")}


def synthetic_selftest(root: Path) -> dict:
    """The harness on the tiny synthetic checkpoint, in the environment of this kernel."""
    import b13_moe_fixture as fixture
    import b14_real_moe_correctness as harness

    weights = fixture.moe_weights_dir(WORK / "tiny")
    out = {}
    for dtype in ("float32", "bfloat16"):
        out[dtype] = harness.run_correctness(weights, WORK / f"tiny-work-{dtype}", dtype=dtype)
        log(f"synthetic {dtype}: {out[dtype]['verdict']} control={out[dtype]['control_detected']}")
    return out


def stage_probe(root: Path) -> None:
    from huggingface_hub import HfApi

    EVIDENCE["environment"] = environment()
    EVIDENCE["versions"] = versions()
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    EVIDENCE["hf"] = {
        "sha": info.sha,
        "safetensors_bytes": sum(s.size or 0 for s in info.siblings
                                 if s.rfilename.endswith(".safetensors")),
    }
    log(f"hf reachable: {EVIDENCE['hf']}")
    EVIDENCE["synthetic"] = synthetic_selftest(root)


def expert_and_router_std(model_dir: Path, config: dict) -> tuple[float, float]:
    from safetensors import safe_open

    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    layer = config["first_k_dense_replace"]
    stds = []
    for expert in range(3):
        key = f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"
        with safe_open(str(model_dir / index[key]), framework="pt") as handle:
            stds.append(float(handle.get_tensor(key).float().std()))
    key = f"model.layers.{layer}.mlp.gate.weight"
    with safe_open(str(model_dir / index[key]), framework="pt") as handle:
        router = float(handle.get_tensor(key).float().std())
    return sum(stds) / len(stds), router


def stage_run(root: Path) -> None:
    import b14_real_moe_correctness as harness
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer

    EVIDENCE["environment"] = environment()
    EVIDENCE["versions"] = versions()
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise RuntimeError(f"revision mismatch: {info.sha}")
    size = sum(s.size or 0 for s in info.siblings if s.rfilename.endswith(".safetensors"))
    model_dir = Path(snapshot_download(
        REPO, revision=REVISION, local_dir=str(WORK / "model"),
        allow_patterns=["*.json", "*.safetensors", "*.model", "LICENSE*", "*.txt"],
    ))
    log(f"downloaded to {model_dir}")
    config = json.loads((model_dir / "config.json").read_text())
    names = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    expert_std, router_std = expert_and_router_std(model_dir, config)
    failed = harness.check_definition(config, names, size, expert_std)
    EVIDENCE["definition"] = {
        "failed_points": failed, "safetensors_bytes": size, "expert_std": expert_std,
        "router_std": router_std, "repo": REPO, "revision": REVISION,
    }
    log(f"definition: failed={failed} expert_std={expert_std:.4f} router_std={router_std:.4f}")
    if failed:
        raise RuntimeError(f"checkpoint fails the definition: {failed}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=False)
    encoded = [tokenizer(text, return_tensors="pt")["input_ids"][0] for text in TEXTS]
    length = min(len(row) for row in encoded)
    ids = torch.stack([row[:length] for row in encoded])
    batch = {"input_ids": ids, "labels": ids.clone()}
    batch["labels"][0, :1] = -100
    EVIDENCE["batch"] = {"rows": int(ids.shape[0]), "length": int(length), "texts": TEXTS}

    started = time.time()
    result = harness.run_correctness(
        model_dir, WORK / "work", dtype="bfloat16", batches=[batch],
    )
    result["seconds"] = time.time() - started
    EVIDENCE["correctness"] = result
    log(f"correctness: {result['verdict']} bitwise={result['bitwise']} "
        f"control={result['control_detected']} seconds={result['seconds']:.0f}")


def main() -> None:
    stage = os.environ.get("B14_STAGE", "probe")
    EVIDENCE["stage"] = stage
    try:
        log(f"pip install {PIP}")
        log(sh([sys.executable, "-m", "pip", "install", "-q", *PIP])[-500:])
        # The image ships torchao 0.10; peft 0.21 refuses it when it wraps a LoRA layer.
        log(sh([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])[-200:])
        root = attach_source()
        EVIDENCE["source_root"] = str(root)
        EVIDENCE["source_sha"] = (root / "SOURCE_SHA").read_text().strip() if (
            root / "SOURCE_SHA").exists() else None
        {"probe": stage_probe, "run": stage_run}[stage](root)
        EVIDENCE["ok"] = True
    except BaseException:
        EVIDENCE["ok"] = False
        EVIDENCE["traceback"] = traceback.format_exc()
        log(EVIDENCE["traceback"])
    finally:
        EVIDENCE["seconds_total"] = time.time() - T0
        (OUT / f"result-{stage}.json").write_text(json.dumps(EVIDENCE, indent=2, default=str))


if __name__ == "__main__":
    main()
