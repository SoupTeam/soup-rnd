"""Admission and numerical boundaries for the offline, two-layer Moonlight verifier."""

from __future__ import annotations

import hashlib
import json
import os
import py_compile
import sys
from pathlib import Path
from typing import Any

import pytest

from benchmarks.harness import p3_moonlight_reference_parity as parity

_REPO = Path(__file__).parents[1]
_RESULTS = _REPO / "benchmarks/results/p3-moonlight-quality"
_CONTRACT_SHA256 = "12e9a1f548d3192035f5781e0d568642480a94a383d48fb1f5373a8f2e178cd0"
_AUX_REPORT = _RESULTS / "prelaunch-review-v1/aux-report-v2.json"
_AUX_SHA256 = "3b56a594a381822bd33c11cea01d6ff5b1d2ef9508522924f148b1431f62f27a"


def _json_file(path: Path, value: Any) -> str:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_aux_admission_accepts_only_the_retained_approved_proof() -> None:
    admitted = parity.admit_aux_report(_AUX_REPORT, _AUX_SHA256)
    assert admitted["report"]["sha256"] == _AUX_SHA256
    assert admitted["contract"]["sha256"] == _CONTRACT_SHA256
    assert admitted["normalization"] == [
        {"field": "aux_loss_alpha", "source_value": 0.001, "runtime": "absent",
         "reason": ("no numerical consumer in pinned native deepseek_v3; "
                    "generic config validation read only")},
        {"field": "seq_aux", "source_value": True, "runtime": "absent",
         "reason": ("no numerical consumer in pinned native deepseek_v3; "
                    "generic config validation read only")},
    ]


def test_self_asserted_aux_pass_is_refused_even_with_matching_supplied_hash(tmp_path: Path) -> None:
    report = json.loads(_AUX_REPORT.read_text(encoding="utf-8"))
    report["comparisons"]["float32"]["every_named_gradient"] = {}
    path = tmp_path / "aux.json"
    digest = _json_file(path, report)
    with pytest.raises(ValueError, match="approved"):
        parity.admit_aux_report(path, digest)


def test_aux_report_hash_is_checked_before_trusting_pass(tmp_path: Path) -> None:
    path = tmp_path / "aux.json"
    payload = _AUX_REPORT.read_bytes() + b"\n"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="hash"):
        parity.admit_aux_report(path, _AUX_SHA256)


def test_reference_executes_verified_source_not_valid_stale_bytecode(tmp_path, monkeypatch):
    sources = {
        "configuration_deepseek.py": b"value = 'right'\n",
        "modeling_deepseek.py": (
            b"from .configuration_deepseek import value as config_value\n"
            b"value = 'right'\nobserved = (config_value, value)\n"
        ),
    }
    for name, payload in sources.items():
        path = tmp_path / name
        path.write_bytes(payload.replace(b"right", b"stale"))
        py_compile.compile(str(path), doraise=True)
        stat = path.stat()
        path.write_bytes(payload)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    monkeypatch.setattr(parity, "_REFERENCE_BLOBS", {
        name: parity._git_blob(payload) for name, payload in sources.items()
    })
    original_version = parity.importlib.metadata.version
    monkeypatch.setattr(parity.importlib.metadata, "version", lambda name: (
        parity.REFERENCE_TRANSFORMERS if name == "transformers" else original_version(name)
    ))
    package = "_soup_pinned_moonlight_reference"
    for suffix in ("", ".configuration_deepseek", ".modeling_deepseek"):
        monkeypatch.setitem(sys.modules, package + suffix, None)
    module = parity._reference_module(tmp_path)
    assert module.observed == ("right", "right")


def _input_bundle(root: Path) -> tuple[Path, str]:
    root.mkdir()
    ids = list(range(1024))
    windows = [{"window_id": "math-0000", "micro_batches": [{"input_ids": [ids]}]}]
    layout = root / "windows-L1024.json"
    layout_sha256 = _json_file(layout, windows)
    layout_descriptor = {"path": layout.name, "bytes": layout.stat().st_size,
                         "sha256": layout_sha256, "format": "json_array", "rows": 1,
                         "length": 1024, "batch_size": 1, "micro_batches_per_window": 4}
    manifest = {
        "schema": "p3-deepseek-quality-training-pack-v1",
        "policy": "p3-deepseek-quality-training-pack-v1",
        "status": "PAIRED_DATA_PACKED_RUNTIME_AND_PRECISION_PENDING",
        "profile_id": "moonlight-16b-a3b", "test_only": False,
        "training_sources_sha256": (
            "8ab0bdfdefde056387835fcaf38914fe2c6879cde1cf89a8f6169629630675a7"),
        "tokenizer": {"repo_id": "moonshotai/Moonlight-16B-A3B",
                      "revision": "476b36a473d4467f94469414bef6cee75c9c8172",
                      "vocabulary_size": 163840},
        "files": {"windows_L1024": layout_descriptor},
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    manifest["manifest_payload_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    manifest_path = root / "manifest.json"
    manifest_sha256 = _json_file(manifest_path, manifest)
    evidence = {
        "schema": "p3-moonlight-reference-input-v1", "profile_id": "moonlight-16b-a3b",
        "model_id": "moonshotai/Moonlight-16B-A3B",
        "revision": "476b36a473d4467f94469414bef6cee75c9c8172",
        "training_pack_manifest_sha256": manifest_sha256,
        "manifest": {"path": "manifest.json", "bytes": manifest_path.stat().st_size,
                     "sha256": manifest_sha256},
        "windows": layout_descriptor,
        "selection": {"length": 1024, "window_index": 0, "micro_batch_index": 0,
                      "token_offset": 9, "token_count": 64},
        "input_ids": list(range(9, 73)),
    }
    evidence_path = root / "input-evidence.json"
    _json_file(evidence_path, evidence)
    return evidence_path, manifest_sha256


def test_explicit_input_is_verified_against_hashed_final_pack_slice(tmp_path, monkeypatch):
    evidence, pack_hash = _input_bundle(tmp_path / "input")
    monkeypatch.setattr(parity, "PACK_MANIFEST_SHA256", pack_hash)
    admitted = parity.load_input(input_evidence=evidence, pack_manifest_sha256=pack_hash)
    assert admitted["input_ids"] == list(range(9, 73))
    assert admitted["training_pack_manifest_sha256"] == pack_hash
    assert admitted["selection"]["token_offset"] == 9


@pytest.mark.parametrize("mutation", ["invented_ids", "short_input", "foreign_pack", "escape"])
def test_input_evidence_rejects_unbound_or_incomplete_tokens(
    tmp_path: Path, mutation: str, monkeypatch,
) -> None:
    path, pack_hash = _input_bundle(tmp_path / "input")
    monkeypatch.setattr(parity, "PACK_MANIFEST_SHA256", pack_hash)
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "invented_ids":
        evidence["input_ids"][0] = 999
    elif mutation == "short_input":
        evidence["input_ids"] = evidence["input_ids"][:-1]
    elif mutation == "foreign_pack":
        evidence["training_pack_manifest_sha256"] = "0" * 64
    else:
        evidence["windows"]["path"] = "../windows-L1024.json"
    _json_file(path, evidence)
    with pytest.raises(ValueError):
        parity.load_input(input_evidence=path, pack_manifest_sha256=pack_hash)


def test_detached_input_cannot_self_declare_an_unapproved_production_pack(tmp_path):
    path, pack_hash = _input_bundle(tmp_path / "input")
    with pytest.raises(ValueError, match="approved"):
        parity.load_input(input_evidence=path, pack_manifest_sha256=pack_hash)


def test_input_requires_independently_supplied_manifest_hash(tmp_path: Path) -> None:
    path, _ = _input_bundle(tmp_path / "input")
    with pytest.raises(ValueError, match="manifest"):
        parity.load_input(input_evidence=path, pack_manifest_sha256="")


def test_changed_hashed_windows_file_refuses_detached_input(tmp_path, monkeypatch):
    path, pack_hash = _input_bundle(tmp_path / "input")
    monkeypatch.setattr(parity, "PACK_MANIFEST_SHA256", pack_hash)
    windows_path = path.parent / "windows-L1024.json"
    windows = json.loads(windows_path.read_text(encoding="utf-8"))
    windows[0]["micro_batches"][0]["input_ids"][0][9] = 999
    _json_file(windows_path, windows)
    with pytest.raises(ValueError, match="receipt"):
        parity.load_input(input_evidence=path, pack_manifest_sha256=pack_hash)


def test_duplicate_json_admission_keys_are_not_last_value_wins(tmp_path: Path) -> None:
    path = tmp_path / "aux.json"
    path.write_text('{"status":"FAIL","status":"PASS"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        parity._json(path)


def test_missing_correction_bias_refuses_prefix_before_materialization() -> None:
    metadata = json.loads((_RESULTS / "source-metadata-admission-v1.json").read_text())
    names = metadata["reference_prefix_tensor_names"]
    weight_map = {name: "model-1-of-27.safetensors" for name in names}
    weight_map[names[-1]] = "model-2-of-27.safetensors"
    del weight_map["model.layers.1.mlp.gate.e_score_correction_bias"]
    with pytest.raises(ValueError, match="prefix"):
        parity.validate_prefix_map(weight_map, metadata)


@pytest.mark.parametrize("mutation", ["extra_prefix_key", "third_shard"])
def test_prefix_coverage_refuses_extra_weights_or_an_unreviewed_shard(mutation: str) -> None:
    metadata = json.loads((_RESULTS / "source-metadata-admission-v1.json").read_text())
    names = metadata["reference_prefix_tensor_names"]
    weight_map = {name: "model-1-of-27.safetensors" for name in names}
    weight_map[names[-1]] = "model-2-of-27.safetensors"
    if mutation == "extra_prefix_key":
        weight_map["model.layers.1.mlp.unused.weight"] = "model-1-of-27.safetensors"
    else:
        weight_map[names[0]] = "model-3-of-27.safetensors"
    with pytest.raises(ValueError, match="prefix"):
        parity.validate_prefix_map(weight_map, metadata)


def _comparison_tensors() -> tuple[Any, Any, Any, Any]:
    torch = pytest.importorskip("torch")
    reference = torch.full((1, 64, 2048), 10000.0, dtype=torch.float32)
    routes = torch.tensor([[0, 1, 2, 3, 4, 5]] * 64, dtype=torch.int64)
    return reference.clone(), reference, routes.clone(), routes


def test_max_abs_boundary_is_inclusive_without_additive_floor() -> None:
    native, reference, native_ids, reference_ids = _comparison_tensors()
    native[0, 0, 0] += 1.0
    result = parity.compare_outputs(native, reference, native_ids, reference_ids)
    assert result["status"] == "PASS"
    assert result["max_abs_error"] == result["allowed_max_abs_error"] == 1.0
    assert result["conditions"] == {"identical_top6_expert_ids": True,
                                     "layer1_max_abs_error_within_limit": True}


def test_one_fp32_increment_beyond_boundary_fails() -> None:
    torch = pytest.importorskip("torch")
    native, reference, native_ids, reference_ids = _comparison_tensors()
    native[0, 0, 0] = torch.nextafter(torch.tensor(10001.0), torch.tensor(float("inf")))
    result = parity.compare_outputs(native, reference, native_ids, reference_ids)
    assert result["status"] == "FAIL"
    assert result["max_abs_error"] > 1.0


def test_same_expert_set_in_different_top6_order_does_not_pass() -> None:
    native, reference, native_ids, reference_ids = _comparison_tensors()
    native_ids[17, :2] = native_ids[17, :2].flip(0)
    result = parity.compare_outputs(native, reference, native_ids, reference_ids)
    assert result["status"] == "FAIL"
    assert result["route_mismatch_tokens"] == [17]
    assert result["conditions"]["layer1_max_abs_error_within_limit"] is True


def test_zero_reference_requires_exact_zero_error() -> None:
    native, reference, native_ids, reference_ids = _comparison_tensors()
    native.zero_()
    reference.zero_()
    assert parity.compare_outputs(native, reference, native_ids, reference_ids)["status"] == "PASS"
    native[0, 0, 0] = 1e-30
    result = parity.compare_outputs(native, reference, native_ids, reference_ids)
    assert result["status"] == "FAIL"
    assert result["allowed_max_abs_error"] == 0.0
    assert result["relative_max_abs_error"] is None


@pytest.mark.parametrize("mutation", [
    "empty_output", "wrong_shape", "nonfinite", "bf16", "duplicate_experts", "empty_routes",
    "out_of_range_expert",
])
def test_comparison_refuses_malformed_numeric_evidence(mutation: str) -> None:
    torch = pytest.importorskip("torch")
    native, reference, native_ids, reference_ids = _comparison_tensors()
    if mutation == "empty_output":
        native = torch.empty(0, dtype=torch.float32)
    elif mutation == "wrong_shape":
        native = native[:, :-1]
    elif mutation == "nonfinite":
        reference[0, 0, 0] = float("nan")
    elif mutation == "bf16":
        native = native.bfloat16()
    elif mutation == "duplicate_experts":
        native_ids[0, 1] = native_ids[0, 0]
    elif mutation == "empty_routes":
        native_ids = torch.empty((0, 6), dtype=torch.int64)
    else:
        native_ids[0, 0] = 64
    with pytest.raises(ValueError):
        parity.compare_outputs(native, reference, native_ids, reference_ids)
