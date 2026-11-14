"""Offline evidence gates for the prospective fp64 CE series; no execution or spending."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from h2_ce_ref64 import GRID, REAL_VOCAB, SEEDS_PER_LEVEL, perturbation_seed


def identity(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_series_authority(root: Path, contract: dict, evidence: dict) -> None:
    """Reuse the sealed-child gates and bind all children to the frozen series authority.

    Do not call live preflight: offline review must work after the lease has expired and
    after evidence has been moved to another machine.
    """
    import h2_ce_confirmation as runner

    if root.is_symlink() or not isinstance(contract, dict) or contract.get("series") != "ref64":
        raise ValueError("Expected regular ref64 series evidence")
    pins = contract.get("artifacts_sha256", {})
    if set(pins) != set(runner.ARTIFACTS) or any(
        runner.sha_file(runner.REPO / name) != pins[name] for name in runner.ARTIFACTS
    ):
        raise ValueError("Offline reviewer/source differs from captured authority")
    study = json.loads((runner.SPEC / "operator_study.json").read_text())
    for key in ("model_revision", "model_files_sha256", "versions", "numerical_policy_sha256"):
        if contract.get(key) != study[key]:
            raise ValueError("Frozen study pin differs: " + key)
    for key in ("source_commit", "gpu_uuid", "boot_id"):
        if not isinstance(contract.get(key), str) or not contract[key]:
            raise ValueError("Missing series identity: " + key)
    for phase in runner.REF64_PHASES:
        child = root / phase / "runner"
        data = runner.read_child(child, phase)  # inventory, SHA, F1, ownership, bindings, trace
        if data != evidence[phase]:
            raise ValueError("Child evidence changed during review")
        if json.loads((child / "contract.json").read_text()) != contract:
            raise ValueError("Child contract differs from whole-series authority")
        for key in ("source_commit", "gpu_uuid", "boot_id"):
            if data.get(key) != contract[key]:
                raise ValueError("Child identity differs from authority: " + key)
        if data.get("initial_identity") != study["reference_identity"]["before_sha256"]:
            raise ValueError("Child differs from historical initial state")
        if not isinstance(data.get("determinism_receipt"), dict) or not data["determinism_receipt"]:
            raise ValueError("Missing determinism receipt")
        if phase.startswith("ref64") and data.get("ref64", {}).get("config") != "real":
            raise ValueError("CPU test configuration is not GPU evidence")


def validate_probe(path: Path, data: dict, values: dict) -> None:
    """Bind actual input tensors and model identity to the declared probe, including Rp."""
    import h2_ce_confirmation as runner

    phase, key = path.parents[2].name, path.parent.name
    if phase in runner.PERTURBED_PROBE:
        index = runner.PERTURBED_PROBE[phase]
    else:
        if key not in {f"probe_{i:02d}" for i in range(4)}:
            raise ValueError("Unknown probe key")
        index = int(key[-2:])
    work = json.loads((runner.SPEC / "ce_confirmation_workset.json").read_text())["probes"]
    study = json.loads((runner.SPEC / "operator_study.json").read_text())
    if data.get("compute_policy") != "bf16-autocast-cuda":
        raise ValueError("GPU probe review requires BF16 CUDA evidence")
    if identity(data["before"]) != study["reference_identity"]["before_sha256"]:
        raise ValueError("Probe tensors start from a different state")
    if {
        k.removeprefix("input."): v.tolist() for k, v in values.items() if k.startswith("input.")
    } != work[index]:
        raise ValueError("Probe actual input differs from frozen workset")
    names = set(data["before"]["trainable"])
    if {k.removeprefix("gradient.") for k in values if k.startswith("gradient.")} != names:
        raise ValueError("Probe gradient inventory differs from initial state")


def read_hooks(root: Path, phase: str) -> dict:
    """Missing hashes must not compare equal as None; every recorded pass is required."""
    keys = (
        {f"pass_{level}_{seed}" for level in range(len(GRID)) for seed in range(SEEDS_PER_LEVEL)}
        if "perturbed" in phase
        else {f"probe_{i:02d}" for i in range(4)}
    )
    shas = json.loads((root / phase / "runner/hooks/lm_head_sha256.json").read_text())
    if set(shas) != keys or any(
        not isinstance(v, str) or re.fullmatch(r"[0-9a-f]{64}", v) is None for v in shas.values()
    ):
        raise ValueError("Missing or malformed lm_head hashes")
    return shas


def validate_dlogits(tensor: Any, labels: Any) -> None:
    import torch

    if (
        tensor.dtype != torch.bfloat16
        or tuple(tensor.shape) != (*labels.shape, REAL_VOCAB)
        or not bool(torch.isfinite(tensor).all())
    ):
        raise ValueError("Invalid bf16 dlogits shape/dtype/values")
    shifted = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
    if bool(tensor[shifted == -100].any()):
        raise ValueError("Non-zero dlogits on ignored rows")


def validate_flips(record: dict, active_elements: int) -> None:
    flips = record.get("flips")
    if not isinstance(flips, list) or len(flips) != len(GRID) * SEEDS_PER_LEVEL:
        raise ValueError("Missing perturbation pass parameters")
    for i, item in enumerate(flips):
        level, seed = divmod(i, SEEDS_PER_LEVEL)
        count = max(1, round(GRID[level] * active_elements))
        if (
            item.get("level") != GRID[level]
            or item.get("seed") != perturbation_seed(level, seed)
            or item.get("flipped") != count
            or item.get("eligible_total") != active_elements
            or type(item.get("eligible_usable")) is not int
            or not count <= item["eligible_usable"] <= active_elements
            or not isinstance(item.get("positions_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}:[0-9a-f]{64}", item["positions_sha256"]) is None
        ):
            raise ValueError("Perturbation parameters differ from fixed grid/seed/count")
