"""B1.4 ticket 09: remote entry that measures the per_expert path on a GPU host.

Runs on a Kaggle script kernel (``B14_PIP=1``, source attached as a dataset) or on a rented
host (source uploaded, dependencies installed by hand). The stage comes from ``B14_STAGE``:

  tiny       the 8-expert synthetic checkpoint of ``b13_moe_fixture``: checks that the counters
             and the profiler work on this GPU before anything large is written or downloaded.
  oversized  a synthetic MoE whose routed experts of one layer are larger than 8 GiB, written
             tensor by tensor (``b14_synthetic_checkpoint``). Random weights, no language.
  real       the pinned ``academic-ds-9B`` checkpoint of ticket 08, same definition check.

Other environment variables: ``B14_SRC`` (repository root or its parent), ``B14_OUT`` (result
folder), ``B14_WORK`` (scratch, needs about 2x the checkpoint size), ``B14_SEQS`` (comma list),
``B14_TIER`` (``ram`` or ``disk``), ``B14_DEVICE`` (``cuda`` or ``cpu``).
Evidence goes to ``<B14_OUT>/result-<stage>.json``.
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
WORK = Path(os.environ.get("B14_WORK", "/tmp/b14"))
PIP = ["transformers==5.20.2", "peft==0.21.2", "huggingface_hub", "safetensors", "rich",
       "pydantic", "pyyaml", "typer", "psutil"]

#: Routed experts of one layer: 640 x 3 x 2048 x 1408 x 2 bytes = 11.07 GB, above 8 GiB.
OVERSIZED = dict(
    vocab_size=32000, hidden_size=2048, intermediate_size=4096, moe_intermediate_size=1408,
    num_hidden_layers=3, first_k_dense_replace=1, n_routed_experts=640, n_group=8,
    topk_group=4, num_experts_per_tok=8, n_shared_experts=1, q_lora_rank=512,
    kv_lora_rank=256, qk_rope_head_dim=64, qk_nope_head_dim=128, v_head_dim=128,
    num_attention_heads=16, num_key_value_heads=16, max_position_embeddings=4096,
)

T0 = time.time()
EVIDENCE: dict = {}


def log(message: str) -> None:
    # bare print on purpose: this entry runs before `rich` is installed on Kaggle, and a
    # Kaggle kernel keeps only stdout in its log.
    print(f"[{time.time() - T0:7.1f}s] {message}", flush=True)


def sh(command: list[str]) -> str:
    done = subprocess.run(command, capture_output=True, text=True)
    return (done.stdout + done.stderr).strip()


def attach_source() -> Path:
    base = os.environ.get("B14_SRC", "/kaggle/input")
    roots = glob.glob(f"{base}/**/src/soup_cli", recursive=True) or glob.glob(
        f"{base}/src/soup_cli")
    if not roots:
        raise RuntimeError(f"repository source not found: no src/soup_cli under {base}")
    root = Path(roots[0]).parent.parent
    for entry in (root / "src", root / "benchmarks" / "harness", root):
        sys.path.insert(0, str(entry))
    return root


def environment() -> dict:
    import importlib.metadata as md

    import psutil
    import torch

    memory = psutil.virtual_memory()
    cpuinfo = Path("/proc/cpuinfo")
    cpu = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")), "") if cpuinfo.exists() else platform.processor()
    gpu = sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,pci.bus_id",
              "--format=csv,noheader"]) if shutil.which("nvidia-smi") else None
    return {
        "platform": platform.platform(), "python": sys.version.split()[0], "cpu_model": cpu,
        "cpu_count": os.cpu_count(), "ram_total_bytes": memory.total,
        "ram_available_bytes": memory.available,
        "cgroup_memory_max": next((Path(p).read_text().strip() for p in (
            "/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes")
            if Path(p).exists()), None),
        "disk_work": shutil.disk_usage(str(WORK))._asdict(),
        "uptime": sh(["uptime"]), "nvidia_smi": gpu,
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "gpu_total_bytes": (
            torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available()
            else None),
        "gpu_bf16_supported": (
            torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None),
        "versions": {name: md.version(name) for name in
                     ("transformers", "peft", "safetensors", "huggingface_hub", "psutil")},
    }


def measure(weights: Path, work: Path, *, dtype: str) -> dict:
    import b14_path_report as report

    seqs = tuple(int(item) for item in os.environ.get("B14_SEQS", "16,512,2048").split(","))
    device = os.environ.get("B14_DEVICE", "cuda")
    tier = os.environ.get("B14_TIER", "ram")
    log(f"measure device={device} dtype={dtype} tier={tier} seqs={seqs}")
    return report.measure_path(weights, work, device=device, dtype=dtype, tier=tier, seqs=seqs)


def stage_tiny() -> None:
    import b13_moe_fixture as fixture

    weights = fixture.moe_weights_dir(WORK / "tiny")
    EVIDENCE["measurement"] = measure(weights, WORK / "tiny-work", dtype="float32")


def stage_oversized() -> None:
    import b13_moe_fixture as fixture
    import b14_synthetic_checkpoint as synthetic

    config = fixture.moe_config(**OVERSIZED)
    layer = synthetic.expert_layer_bytes(config, dtype="bfloat16")
    EVIDENCE["checkpoint"] = {
        "kind": "synthetic, random weights", "config": OVERSIZED, "dtype": "bfloat16",
        "expert_layer_bytes": layer, "moe_layers": 2,
    }
    log(f"writing synthetic checkpoint, expert layer {layer} bytes")
    weights = synthetic.write_checkpoint(WORK / "synthetic", config, dtype="bfloat16", seed=3)
    EVIDENCE["checkpoint"]["written_bytes"] = sum(
        f.stat().st_size for f in weights.glob("*.safetensors"))
    log(f"written {EVIDENCE['checkpoint']['written_bytes']} bytes")
    EVIDENCE["measurement"] = measure(weights, WORK / "synthetic-work", dtype="bfloat16")


def stage_real() -> None:
    import b14_real_moe_correctness as correctness
    import b14_real_moe_kaggle as ticket08
    from huggingface_hub import HfApi, snapshot_download

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
    expert_std, router_std = ticket08.expert_and_router_std(model_dir, config)
    failed = correctness.check_definition(config, names, size, expert_std)
    EVIDENCE["checkpoint"] = {
        "kind": "real, published", "repo": REPO, "revision": REVISION, "failed_points": failed,
        "safetensors_bytes": size, "expert_std": expert_std, "router_std": router_std,
    }
    if failed:
        raise RuntimeError(f"checkpoint fails the definition: {failed}")
    EVIDENCE["measurement"] = measure(model_dir, WORK / "real-work", dtype="bfloat16")


def main() -> None:
    stage = os.environ.get("B14_STAGE", "tiny")
    EVIDENCE["stage"] = stage
    try:
        WORK.mkdir(parents=True, exist_ok=True)
        OUT.mkdir(parents=True, exist_ok=True)
        if os.environ.get("B14_PIP") == "1":
            log(sh([sys.executable, "-m", "pip", "install", "-q", *PIP])[-500:])
            # The Kaggle image ships torchao 0.10; peft 0.21 refuses it.
            log(sh([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])[-200:])
        root = attach_source()
        EVIDENCE["source_root"] = str(root)
        sha_file = root / "SOURCE_SHA"
        EVIDENCE["source_sha"] = sha_file.read_text().strip() if sha_file.exists() else None
        EVIDENCE["environment"] = environment()
        log(f"environment: {json.dumps(EVIDENCE['environment'], default=str)[:1500]}")
        {"tiny": stage_tiny, "oversized": stage_oversized, "real": stage_real}[stage]()
        EVIDENCE["ok"] = True
    except BaseException:
        EVIDENCE["ok"] = False
        EVIDENCE["traceback"] = traceback.format_exc()
        log(EVIDENCE["traceback"])
    finally:
        (OUT / f"result-{stage}.json").write_text(json.dumps(EVIDENCE, indent=2, default=str))
        log(f"result written to {OUT / f'result-{stage}.json'}, ok={EVIDENCE.get('ok')}")


if __name__ == "__main__":
    main()
