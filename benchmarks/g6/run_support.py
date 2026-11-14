"""Offline provenance, strict admission and fail-closed single-GPU ownership."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from checkpoint_eval import sha256

ROOT = Path(__file__).parent
REPO = ROOT.parents[1]


def harness() -> object:
    directory = str(REPO / "benchmarks/harness")
    if directory not in sys.path:
        sys.path.insert(0, directory)
    return importlib.import_module("arm_validity")


def contained(root: Path, name: str) -> Path:
    location = os.path.realpath(root / name)
    if os.path.commonpath([os.path.realpath(root), location]) != os.path.realpath(root):
        raise ValueError("Path escapes declared root")
    return Path(location)


def contract(path: Path, expected: str) -> dict:
    if sha256(path) != expected:
        raise ValueError("Contract SHA differs from explicit launch pin")
    spec = json.loads(path.read_text())
    for name, digest in spec["source_sha256"].items():
        if sha256(contained(REPO, name)) != digest:
            raise ValueError(f"Frozen source changed: {name}")
    for name, digest in spec["artifact_sha256"].items():
        if sha256(contained(ROOT, name)) != digest:
            raise ValueError(f"Frozen artifact changed: {name}")
    return spec


def verify_model(root: Path, files: list[dict], guard: object) -> dict:
    verified = {}
    allowed = {row["name"] for row in files}
    for path in root.iterdir():
        if (
            path.is_file()
            and path.name not in allowed
            and (
                path.suffix in {".safetensors", ".bin", ".pt", ".json"}
                or path.name == "adapter_config.json"
            )
        ):
            raise ValueError(f"Unverified loadable model artifact: {path.name}")
    for row in files:
        path = contained(root, row["name"])
        if path.stat().st_size != row["size"]:
            raise ValueError(f"Model size mismatch: {row['name']}")
        digest = hashlib.sha256() if row["algorithm"] == "sha256" else hashlib.sha1()
        if row["algorithm"] == "git_blob_sha1":
            digest.update(f"blob {row['size']}\0".encode())
        with path.open("rb") as handle:
            while block := handle.read(16 * 1024 * 1024):
                guard()
                digest.update(block)
        if digest.hexdigest() != row["digest"]:
            raise ValueError(f"Pinned model bytes differ: {row['name']}")
        verified[row["name"]] = row["digest"]
    return verified


def admit(receipt: dict, spec: dict, mode: str, now: float | None = None) -> None:
    now = time.time() if now is None else now
    if receipt.get("account") != "soup-rnd" or receipt.get("mode") != mode:
        raise ValueError("Wrong account or stage")
    if receipt.get("contract_sha256") != spec["launch_contract_sha256"]:
        raise ValueError("Admission binds a different contract")
    stamp = receipt["observed_unix_s"]
    if type(stamp) not in (int, float) or not math.isfinite(stamp):
        raise ValueError("Invalid Billing timestamp")
    age = now - stamp
    if age < 0 or age > 300:
        raise ValueError("Fresh actual Billing (<=300s) required")
    if not receipt.get("instance_id") or not receipt.get("gpu_uuid", "").startswith("GPU-"):
        raise ValueError("Explicit instance and GPU UUID required")
    balance, rate = receipt["balance_usd"], receipt["total_account_rate_usd_h"]
    if (
        not all(type(value) in (int, float) and math.isfinite(value) for value in (balance, rate))
        or rate <= 0
    ):
        raise ValueError("Invalid Billing numbers")
    bound = spec["limits"][mode + "_total_s"]
    budget = spec.get("budget", {})
    if budget.get("mode") == "user_session_spend_cap":
        if mode not in budget["authorized_modes"]:
            raise ValueError("This spending authorization excludes full training")
        opening, cap = budget["opening_balance_usd"], budget["spend_cap_usd"]
        if (
            not all(
                type(value) in (int, float) and math.isfinite(value) for value in (opening, cap)
            )
            or not 0 < cap <= 5
        ):
            raise ValueError("Invalid session spending cap")
        if balance > opening + 0.01:
            raise ValueError("Balance increased; do not silently expand the fixed session budget")
        floor, reserve = opening - cap, budget["reserve_usd"]
    else:
        floor, reserve = 42, 0.25
    required = 1.25 * rate * ((bound + age) / 3600) + reserve
    if balance - floor <= required:
        raise ValueError("Bounded stage does not fit within the authorized remaining budget")
    if receipt.get("authorized_for_this_stage") is not True:
        raise ValueError("No authorization recorded for paid stage")


class Guard:
    def __init__(self, out: Path, deadline_s: int, *, gpu: bool, uuid: str | None = None):
        self.out, self.deadline, self.gpu, self.uuid = out, time.monotonic() + deadline_s, gpu, uuid
        self.last_scan = -math.inf
        self.stage_deadline = self.deadline

    def stage(self, seconds: int) -> None:
        self()
        self.stage_deadline = min(self.deadline, time.monotonic() + seconds)

    def __call__(self) -> None:
        if time.monotonic() >= min(self.deadline, self.stage_deadline):
            raise TimeoutError("Prospective wall-time bound reached; no automatic retry")
        if shutil.disk_usage(self.out).free < 1024**3:
            raise ValueError("Less than1GiB output disk reserve")
        if self.gpu and time.monotonic() - self.last_scan >= 2:
            rows = harness().gpu_processes()
            if rows is None or any(row["pid"] != os.getpid() for row in rows):
                raise ValueError("Foreign or unmapped GPU PID; namespace ambiguity is fatal")
            reading = harness().gpu_clock()
            if reading is None:
                raise ValueError("GPU clocks unavailable")
            with (self.out / "ownership_clocks.jsonl").open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "monotonic_s": time.monotonic(),
                            "own_pid": os.getpid(),
                            "processes": rows,
                            "clock": reading,
                            "gpu_uuid": self.uuid,
                        }
                    )
                    + "\n"
                )
            self.last_scan = time.monotonic()


def gpu_inventory(uuid: str) -> dict:
    import torch

    if sys.platform != "linux" or os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
        raise ValueError("Linux and CUDA_VISIBLE_DEVICES=approved single GPU UUID required")
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability()[0] < 8:
        raise ValueError("One visible Ampere-or-newer GPU required")
    result = subprocess.run(
        [
            "nvidia-smi",
            "--id=" + uuid,
            "--query-gpu=uuid,name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    if result.stdout.split(",")[0].strip() != uuid:
        raise ValueError("GPU identity mismatch")
    if torch.cuda.get_device_properties(0).total_memory < 23 * 1024**3:
        raise ValueError("Prospective FP16 recipe requires >=23GiB physical VRAM before fit pilot")
    return {
        "nvidia_smi": result.stdout.strip(),
        "torch_cuda": torch.version.cuda,
        "compute_capability": list(torch.cuda.get_device_capability()),
        "uuid": uuid,
        "namespace_mapping_not_assumed": True,
    }


def runtime() -> dict:
    return {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "peft", "safetensors", "tokenizers", "accelerate")
    }


def preceding_initial(claims: Path, spec: dict, mode: str, arm: str) -> str | None:
    """Enforce preregistered sequential order and refuse any repeated arm claim."""
    if (claims / f"{mode}_{arm}.json").exists():
        raise ValueError("An existing claim forbids redeployment")
    previous = spec["arm_order"][: spec["arm_order"].index(arm)]
    initial = set()
    for name in previous:
        claim = json.loads((claims / f"{mode}_{name}.json").read_text())
        out = Path(claim["out"])
        if (out / "failure.json").exists() or not (out / "files_sha256.json").exists():
            raise ValueError("Preceding arm is not preserved and complete")
        receipt = json.loads((out / name / "receipt.json").read_text())
        if receipt["status"] != "ARM_COMPLETED_REVIEW_PENDING":
            raise ValueError("Preceding arm failed")
        initial.add(receipt["initial_state_sha256"])
    if len(initial) > 1:
        raise ValueError("Preceding arms have different initial states")
    return next(iter(initial), None)
