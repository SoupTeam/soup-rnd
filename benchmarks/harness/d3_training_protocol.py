"""Prospective D3 training protocol and clean-Git source packet checks."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

from d3_full_model import contained, file_hash
from d3_training_model import MODES

ROOT = Path(__file__).resolve().parents[2]
RELATIVE_PROTOCOL = "benchmarks/protocols/d3-trained-step/protocol.json"
CODECS = ("dense", "nf4", "nf4-dq")
HELPERS = (
    "d3_training_model.py", "d3_training_protocol.py", "d3_training_gate.py",
    "d3_training_series.py", "d3_training_analysis.py", "d3_full_model.py",
    "d3_full_model_gate.py", "d3_grouped_restore_check.py", "d3_cublas_grouped_probe.py",
    "b1_expert_forward_gate.py", "d3_f1_mapping_probe.py", "d3_f1_controls.py", "arm_validity.py",
)
TESTS = (
    "tests/test_b1_expert_forward.py",
    "tests/test_d3_grouped_expert.py",
    "tests/test_d3_grouped_compute.py",
    "tests/test_d3_training_model.py", "tests/test_d3_training_protocol.py",
    "tests/test_d3_training_analysis.py",
    "tests/test_d3_full_model.py",
    "tests/test_d3_expert_activation.py", "tests/test_d3_expert_derivatives.py",
    "tests/test_d3_f1_mapping.py",
)
SUPERSEDED_ATTEMPT_FILES = (
    "benchmarks/results/d3-trained-step-attempt-01-20261022/.gitattributes",
    "benchmarks/results/d3-trained-step-attempt-01-20261022/SHA256SUMS.json",
    "benchmarks/results/d3-trained-step-attempt-01-20261022/attempt-01-no-verdict-a26f4948.zip",
    "benchmarks/results/d3-trained-step-attempt-01-20261022/protocol.json",
    "benchmarks/results/d3-trained-step-attempt-01-20261022/report.md",
)


def read_rule(root: Path = ROOT) -> dict:
    rule = json.loads(contained(root, RELATIVE_PROTOCOL).read_text(encoding="utf-8"))
    validate_rule(rule)
    return rule


def validate_rule(rule: dict, *, require_execution: bool = False) -> None:
    supersedes = {
        "commit": "a26f49484f88522ad7e34f6e3cb13b6893d45328",
        "protocol_sha256": "1aaf1cd0f8d37a058d322c95c299a18351f25472b3623c31b74cab0bde1e8e46",
        "session_id": "7ecbb5b9d3544bc0b9f7f68b041fc108",
        "attempt_archive_sha256": "1e30d95875f51dfe2544a23c051be38e185eb01bb2d1140c5d87dbe3a86fb444",
        "status": "incomplete",
        "completed_cells": 1,
        "no_verdict_cell": "correctness-dense-native",
        "cause": "tensor_hash failed on the scalar loss after the full native optimizer step; F1 was ok",
        "treatment": "Retain this attempt separately; the corrected source gets a new commit and a fresh frozen 84-cell series, not a replacement within the prior protocol.",
    }
    if rule.get("schema_version") != 2 or rule.get("supersedes") != supersedes:
        raise ValueError("new series must identify the retained first no-verdict attempt")
    expected = {"model_id": "allenai/OLMoE-1B-7B-0924",
                "model_revision": "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
                "batch": 1, "tokens": 512, "input_seed": 1234,
                "adapter_seed": 25, "adapter_rank": 4, "learning_rate": 0.5,
                "steps": 4, "warmup_steps": 1, "timed_steps": 3,
                "measurement_blocks_per_codec": 5,
                "sessions_per_commit": 1, "replacements": 0,
                "weight_budget_bytes": 64 * 2**20,
                "gemm_call_budget_bytes": 64 * 2**20,
                "activation_budget_bytes": 256 * 2**20, "native_rows_max": 128}
    if any(rule.get(key) != value for key, value in expected.items()):
        raise ValueError("training protocol recipe differs from the validated design")
    if tuple(rule.get("modes", ())) != MODES or tuple(rule.get("formats", ())) != CODECS:
        raise ValueError("all trained formats and resident/streamed modes required")
    if rule.get("timing_pair") != ["grouped", "triton"]:
        raise ValueError("Triton timing control must hold grouped GEMM and pool width constant")
    budget = rule.get("attempt_budget", {})
    if (rule.get("expected_cell_count") != 84
            or budget != {"correctness_cells": 18, "timing_cells": 60,
                          "profile_cells": 6, "total_cells": 84}
            or rule.get("scope") != ("trained OLMoE optimizer-step numerical and descriptive "
                                     "GPU comparison; synthetic token IDs, no quality claim")):
        raise ValueError("fixed complete-model training attempt budget/scope is incomplete")
    if rule.get("execution_enabled") not in (True, False):
        raise ValueError("execution_enabled must be an explicit boolean")
    if require_execution and rule.get("execution_enabled") is not True:
        raise ValueError("formal execution disabled until preparation and operator commit")
    if not rule["execution_enabled"]:
        return
    for key in ("mapping_sha256", "controls_sha256", "setup_archive_sha256",
                "cublas_library_sha256"):
        if not re.fullmatch("[0-9a-f]{64}", rule.get(key, "")):
            raise ValueError(f"missing preparation identity: {key}")
    if set(rule.get("bundle_sha256", {})) != set(CODECS) or any(
        not re.fullmatch("[0-9a-f]{64}", digest) for digest in rule["bundle_sha256"].values()
    ):
        raise ValueError("pin all three prepared bundle identities")
    if not rule.get("gpu_uuid") or not rule.get("software"):
        raise ValueError("pin the actual execution interpreter and GPU")
    archive = contained(ROOT, "benchmarks/protocols/d3-trained-step/setup-evidence.zip")
    if file_hash(archive) != rule["setup_archive_sha256"]:
        raise ValueError("committed setup-evidence archive differs from its frozen identity")
    prior = contained(ROOT, SUPERSEDED_ATTEMPT_FILES[2])
    if file_hash(prior) != rule["supersedes"]["attempt_archive_sha256"]:
        raise ValueError("superseded no-verdict attempt archive differs from its frozen identity")


def source_paths(root: Path = ROOT) -> list[str]:
    paths = {p.relative_to(root).as_posix() for p in (root / "src").rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    paths.update("benchmarks/harness/" + name for name in HELPERS)
    paths.update(TESTS)
    paths.update((RELATIVE_PROTOCOL, "benchmarks/protocols/d3-trained-step/setup-evidence.zip",
                  "benchmarks/protocols/b1.2-expert-forward/protocol.json"))
    paths.update(SUPERSEDED_ATTEMPT_FILES)
    return sorted(paths)


def export_packet(output: Path, root: Path = ROOT) -> dict:
    """Export actual committed blobs locally; the remote host needs no Git token.

    This is the only source-packet creation path. Retained commit/blob identities
    are independently comparable with GitHub after the operator's push.
    """
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)

    if git("status", "--porcelain", "--untracked-files=normal").strip():
        raise ValueError("commit the exact prepared code/protocol in a clean checkout first")
    rule = read_rule(root)
    validate_rule(rule, require_execution=True)
    data = {name: git("show", "HEAD:" + name) for name in source_paths(root)}
    if json.loads(data[RELATIVE_PROTOCOL]) != rule:
        raise ValueError("working protocol differs from its committed blob")
    setup = "benchmarks/protocols/d3-trained-step/setup-evidence.zip"
    if hashlib.sha256(data[setup]).hexdigest() != rule["setup_archive_sha256"]:
        raise ValueError("committed setup evidence differs from the prospective pin")
    receipt = {"kind": "clean-Git export of canonical committed blobs",
               "git_head": git("rev-parse", "HEAD").decode().strip(),
               "git_blob_sha256": {name: hashlib.sha256(value).hexdigest()
                                   for name, value in data.items()}}
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in data.items():
            archive.writestr(name, value)
        archive.writestr("FROZEN-D3-SOURCE.json", json.dumps(receipt, indent=2) + "\n")
    return {"source": receipt, "packet_sha256": file_hash(output)}


def frozen_source(root: Path = ROOT) -> dict:
    """Verify every received source blob against the operator's clean-Git export."""
    rule = read_rule(root)
    validate_rule(rule, require_execution=True)
    receipt = json.loads((root / "FROZEN-D3-SOURCE.json").read_text(encoding="utf-8"))
    if (receipt.get("kind") != "clean-Git export of canonical committed blobs"
            or not re.fullmatch("[0-9a-f]{40}", receipt.get("git_head", ""))
            or set(receipt.get("git_blob_sha256", {})) != set(source_paths(root))):
        raise ValueError("incomplete clean-commit source receipt")
    for name, expected in receipt["git_blob_sha256"].items():
        if file_hash(contained(root, name)) != expected:
            raise ValueError(f"executed source differs from exported committed bytes: {name}")
    return receipt
