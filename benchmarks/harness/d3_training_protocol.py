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
    "benchmarks/results/d3-trained-step-attempt-02-20261022/.gitattributes",
    "benchmarks/results/d3-trained-step-attempt-02-20261022/SHA256SUMS.json",
    "benchmarks/results/d3-trained-step-attempt-02-20261022/attempt-02-no-verdict-23bad807.zip",
    "benchmarks/results/d3-trained-step-attempt-02-20261022/protocol.json",
    "benchmarks/results/d3-trained-step-attempt-02-20261022/report.md",
    "benchmarks/results/d3-trained-step-attempt-03-20261022/.gitattributes",
    "benchmarks/results/d3-trained-step-attempt-03-20261022/SHA256SUMS.json",
    "benchmarks/results/d3-trained-step-attempt-03-20261022/attempt-03-no-verdict-46937f49.zip",
    "benchmarks/results/d3-trained-step-attempt-03-20261022/protocol.json",
    "benchmarks/results/d3-trained-step-attempt-03-20261022/report.md",
    "benchmarks/results/d3-trained-step-attempt-04-20261022/.gitattributes",
    "benchmarks/results/d3-trained-step-attempt-04-20261022/SHA256SUMS.json",
    "benchmarks/results/d3-trained-step-attempt-04-20261022/attempt-04-no-verdict-3492ed83.zip",
    "benchmarks/results/d3-trained-step-attempt-04-20261022/protocol.json",
    "benchmarks/results/d3-trained-step-attempt-04-20261022/report.md",
)


def read_rule(root: Path = ROOT) -> dict:
    rule = json.loads(contained(root, RELATIVE_PROTOCOL).read_text(encoding="utf-8"))
    validate_rule(rule)
    return rule


def validate_rule(rule: dict, *, require_execution: bool = False) -> None:
    prior_attempt_01 = {
        "commit": "a26f49484f88522ad7e34f6e3cb13b6893d45328",
        "protocol_sha256": "1aaf1cd0f8d37a058d322c95c299a18351f25472b3623c31b74cab0bde1e8e46",
        "session_id": "7ecbb5b9d3544bc0b9f7f68b041fc108",
        "attempt_archive_sha256": (
            "1e30d95875f51dfe2544a23c051be38e185eb01bb2d1140c5d87dbe3a86fb444"
        ),
        "status": "incomplete",
        "completed_cells": 1,
        "no_verdict_cell": "correctness-dense-native",
        "cause": ("tensor_hash failed on the scalar loss after the full native optimizer step; "
                  "F1 was ok"),
        "treatment": ("Retain as a separate no-verdict attempt; do not count as a replacement "
                      "or passing result."),
    }
    prior_attempt_02 = {
        "commit": "23bad80765ae3a02c19887e0ba5c3bc8550d225e",
        "protocol_sha256": "2e7a2709eeaaa564223914a88e8d34ebb97056ebd26567c9818bbb714f6738f0",
        "session_id": "730ef9ec393745068b3c776dd198ddce",
        "attempt_archive_sha256": (
            "09fba45f4cb84dbb7880c40fb0d4982c4a0a65c8e186617d9a034dd48d2774a0"
        ),
        "status": "incomplete",
        "completed_cells": 0,
        "attempted_cell": "correctness-dense-native",
        "cause": ("operator-created supervisor wrapper had CRLF line endings; the cuBLAS path "
                  "received a trailing carriage return and failed identity validation before "
                  "model construction"),
        "treatment": ("Retain as a separate no-verdict attempt; no model step/F1/measurement "
                      "ran. Create launch files with LF and require a fresh frozen commit/"
                      "protocol."),
    }
    prior_attempt_03 = {
        "commit": "46937f49994edd5667769ba4ec4f161d2154997f",
        "protocol_sha256": "83af5ce45389f7f1796380459bcce3b55b08ff26d7d55e617e95d50fb49ac8a1",
        "session_id": "d1238f4a055a4f189e5e69304f87b29e",
        "attempt_archive_sha256": (
            "611e81b81abae294ce5661b778ce738ede665b82877efea56b3d00b38bfe9b9c"
        ),
        "status": "incomplete",
        "completed_cells": 1,
        "passed_cell": "correctness-dense-native",
        "no_verdict_cell": "correctness-dense-sequential",
        "cause": ("series passed --oracle to the child CLI, but the frozen child parser did not "
                  "accept that argument; the next cell was rejected before model construction"),
        "treatment": ("Retain the native cell and failed invocation separately; do not treat this "
                      "as a comparison or replacement. Add the missing child argument, verify "
                      "native-to-candidate dispatch, then use a new frozen commit/protocol."),
    }
    prior_attempt_04 = {
        "commit": "3492ed83c2cd122ec0ad6318a2f40af6295ebcbc",
        "protocol_sha256": "8a253f6c6f968d49e80bd235fa6eb9c0ac8f8f137c6384d239c5517e04fe180f",
        "session_id": "bb01c3a357ee4f1aaefda153734762e5",
        "attempt_archive_sha256": (
            "80afca9d960093b826772ca2372d135eab2d8648e36c20a794a86f03a6fba0b3"
        ),
        "status": "incomplete",
        "completed_cells": 0,
        "attempted_cell": "correctness-dense-native",
        "cause": ("after a container restart, the live shard st_dev changed from 46 to 49; "
                  "the frozen mapping was correctly rejected before model construction"),
        "treatment": ("Retain as a no-verdict attempt with no replacement under revision 4. "
                      "Revision 5 refreshes mapping and independently passing F1 controls; "
                      "require a new frozen commit before any formal cell."),
    }
    if (rule.get("schema_version") != 5
            or rule.get("prior_attempts") != [
                prior_attempt_01, prior_attempt_02, prior_attempt_03, prior_attempt_04,
            ]):
        raise ValueError("fresh series must identify all retained no-verdict attempts")
    if rule.get("container_restart_policy") != (
            "Any container or instance restart invalidates the prior mapping. Rediscover all "
            "shards and pass independent baseline/own/foreign F1 controls before freezing; "
            "identity mismatch is no-verdict and never an automatic replacement."):
        raise ValueError("container restart must invalidate mapping and require fresh F1 controls")
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
    with zipfile.ZipFile(archive) as evidence:
        for name in ("mapping", "controls"):
            payload = evidence.read(name + ".json")
            if hashlib.sha256(payload).hexdigest() != rule[name + "_sha256"]:
                raise ValueError(f"setup archive {name} differs from its frozen identity")
        controls = json.loads(evidence.read("controls.json"))
    if (controls.get("status") != "controls_passed"
            or [row.get("case") for row in controls.get("controls", ())]
            != ["baseline", "own", "foreign"]
            or [row.get("observed") for row in controls.get("controls", ())]
            != ["ok", "ok", "void"]):
        raise ValueError(
            "frozen setup archive must retain passing baseline/own/foreign F1 controls"
        )
    archive_paths = (SUPERSEDED_ATTEMPT_FILES[2], SUPERSEDED_ATTEMPT_FILES[7],
                     SUPERSEDED_ATTEMPT_FILES[12], SUPERSEDED_ATTEMPT_FILES[17])
    for prior, archive_path in zip(
        (prior_attempt_01, prior_attempt_02, prior_attempt_03, prior_attempt_04),
        archive_paths, strict=True,
    ):
        if file_hash(contained(ROOT, archive_path)) != prior["attempt_archive_sha256"]:
            raise ValueError(
                "a retained no-verdict attempt archive differs from its frozen identity"
            )


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
