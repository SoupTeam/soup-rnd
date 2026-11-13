"""Offline Moonlight embedding/dense-0/MoE-1 FP32 native/reference correctness evidence.

Run in the pinned native venv. A second, distinct venv executes only the two
blob-pinned reference sources. Neither process downloads anything, loads the full
model, attaches Oracle, changes routing, or retries a failed comparison. See
reference-resume-report.md for environment preparation and detached input format.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
import traceback
import types
from contextlib import ExitStack
from pathlib import Path
from typing import Any, TextIO

MODEL_ID = "moonshotai/Moonlight-16B-A3B"
REVISION = "476b36a473d4467f94469414bef6cee75c9c8172"
PROFILE_ID = "moonlight-16b-a3b"
SCHEMA = "p3-moonlight-reference-parity-v1"
INVENTORY_SHA256 = "372baec885e52e89d26895f79759758b9600b67c518f80761c437e0f5e08a5ee"
METADATA_SHA256 = "2c22f9f346d2c58aa82445f8208bf48e12f8bd27f84447b2697f2d0fccc4a180"
AUX_CONTRACT_SHA256 = "12e9a1f548d3192035f5781e0d568642480a94a383d48fb1f5373a8f2e178cd0"
AUX_REPORT_SHA256 = "3b56a594a381822bd33c11cea01d6ff5b1d2ef9508522924f148b1431f62f27a"
PACK_MANIFEST_SHA256 = "2d142c8f1641bd55173910311744455a79c4be5c9152cf181673d3965247cee4"
ORACLE_SHA256 = "c47fc3bea0a6034bdbaa39e6ac1da7f1b527451f0f5273ee93d1197481ff831b"
NATIVE_TRANSFORMERS = "5.19.0"
REFERENCE_TRANSFORMERS = "4.46.3"
COMMON_VERSIONS = {"torch": "2.11.0", "numpy": "2.1.3", "safetensors": "0.8.0",
                   "rich": "13.9.4"}
_TRANSFORMERS_DEPENDENCIES = {
    "native": {"tokenizers": "0.23.3", "huggingface_hub": "2.4.0"},
    "reference": {"tokenizers": "0.20.3", "huggingface_hub": "0.26.2"},
}
SHARD_BYTES = 2006757520
WORKER_SECONDS = 1800
_FIELDS = {"aux_loss_alpha": 0.001, "seq_aux": True}
_NORMALIZATION_REASON = (
    "no numerical consumer in pinned native deepseek_v3; generic config validation read only")
_RESULTS = Path(__file__).parents[1] / "results/p3-moonlight-quality"
_REPO = Path(__file__).parents[2]
_METADATA_FILES = {
    "config.json": (1375, "884b0b151c22df50797e9392f1f1f968abfa0019",
                    "64513a4aa6ffd2337215ea80c880491ad54a180ffb66a53dd9e178c6f8e9f76c"),
    "model.safetensors.index.json": (
        445693, "5a1b46de3d2e062948cd2883d56960711a1a16d3",
        "b21dee4e168d9967b4407bdf945cf382a5449ce5be4f60163119942b36c06aef"),
}
_REFERENCE_BLOBS = {
    "modeling_deepseek.py": "a72d8b7d904ebc28aab13be12bbb5f0469a41be7",
    "configuration_deepseek.py": "79b86a084abd1013a46244394f876aa0375b88ca",
}
_NATIVE_SOURCE_PINS = {
    "models/deepseek_v3/modeling_deepseek_v3.py": (
        "1ef325ceed5ac2f01a2fa435aa00f9a11c6fc804bdd5431d044e30a1b506ba41"),
    "models/deepseek_v3/configuration_deepseek_v3.py": (
        "1f0eba8a53a5f72681dc5d6ba5fb4506e690dc6252e31ff6fe29d6063893b04f"),
}
_SHAPE = {
    "model_type": "deepseek_v3", "hidden_size": 2048, "vocab_size": 163840,
    "num_hidden_layers": 27, "num_attention_heads": 16, "num_key_value_heads": 16,
    "q_lora_rank": None, "kv_lora_rank": 512, "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64, "v_head_dim": 128, "intermediate_size": 11264,
    "moe_intermediate_size": 1408, "first_k_dense_replace": 1, "moe_layer_freq": 1,
    "n_shared_experts": 2, "n_routed_experts": 64, "num_experts_per_tok": 6,
    "n_group": 1, "topk_group": 1, "routed_scaling_factor": 2.446,
    "scoring_func": "sigmoid", "topk_method": "noaux_tc", "norm_topk_prob": True,
    "ep_size": 1, "rope_theta": 50000.0, "max_position_embeddings": 8192,
    "rms_norm_eps": 1e-5, "attention_bias": False, "attention_dropout": 0.0,
    "hidden_act": "silu", "num_nextn_predict_layers": 0,
}


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _hex(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number: {value}")


def _json(path: Path) -> Any:
    value = json.loads(path.read_bytes(), object_pairs_hook=_unique,
                       parse_constant=_reject_constant)
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, float):
            _require(math.isfinite(item), "nonfinite JSON number")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_canonical(value) + b"\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _descriptor(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required file missing: {path}")
    return {"path": os.path.realpath(path), "bytes": path.stat().st_size,
            "sha256": _sha256(path)}


def _contained(path: Path, root: Path) -> bool:
    try:
        parent, child = os.path.realpath(root), os.path.realpath(path)
        return os.path.commonpath((parent, child)) == parent
    except ValueError:
        return False


def _child(root: Path, name: Any) -> Path:
    _require(isinstance(name, str) and bool(name) and "\\" not in name and ":" not in name
             and not Path(name).is_absolute()
             and all(part not in {"", ".", ".."} for part in name.split("/")),
             "invalid relative artifact path")
    result = root / name
    _require(_contained(result, root), "artifact path escapes its directory")
    return result


def _artifact(root: Path, descriptor: Any) -> Path:
    _require(isinstance(descriptor, dict), "file descriptor missing")
    path = _child(root, descriptor.get("path"))
    _require(type(descriptor.get("bytes")) is int and descriptor["bytes"] > 0
             and _hex(descriptor.get("sha256")), "invalid file receipt")
    _require(path.is_file() and path.stat().st_size == descriptor["bytes"],
             f"file receipt size mismatch: {path}")
    _require(_sha256(path) == descriptor["sha256"], f"file receipt hash mismatch: {path}")
    return path


def _git_blob(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()


def admit_aux_report(report_path: Path, report_sha256: str) -> dict[str, Any]:
    """Admit only the amended, hash-bound, nonvacuous FP32 and BF16 aux proof.

    The caller must retrieve and validate the full aux evidence archive before
    invoking this verifier. This consumer validates the shared v2 report contract;
    it does not reinterpret the static audit or repeat the aux experiment.
    """
    _require(report_sha256 == AUX_REPORT_SHA256, "approved aux report SHA256 required")
    report_path = Path(report_path)
    report_descriptor = _descriptor(report_path)
    _require(report_descriptor["sha256"] == report_sha256, "aux report hash mismatch")
    contract_path = _RESULTS / "aux-admission-contract-v2.json"
    contract_descriptor = _descriptor(contract_path)
    _require(contract_descriptor["sha256"] == AUX_CONTRACT_SHA256,
             "amended aux contract hash mismatch")
    contract, report = _json(contract_path), _json(report_path)
    _require(isinstance(contract, dict)
             and contract.get("schema") == "p3-moonlight-aux-admission-contract-v2"
             and contract.get("transformers_version") == NATIVE_TRANSFORMERS,
             "amended aux contract identity mismatch")
    _require(isinstance(report, dict)
             and report.get("schema") == "p3-moonlight-native-aux-admission-v2"
             and report.get("status") == "PASS" and report.get("aux_fields_admitted") is True
             and report.get("evidence_complete") is True
             and report.get("admission_contract_sha256") == AUX_CONTRACT_SHA256,
             "complete amended aux admission PASS required")
    comparisons = report.get("comparisons")
    _require(isinstance(comparisons, dict) and set(comparisons) == {"float32", "bfloat16"},
             "both aux precision proofs required")
    for precision, comparison in comparisons.items():
        _require(isinstance(comparison, dict) and all(comparison.get(name) is True for name in (
            "loss_bitwise_equal", "logits_bitwise_equal", "gradient_names_complete",
            "identical_initial_state", "identical_initial_buffers")),
            f"aux equality failure: {precision}")
        count, gradients = (comparison.get("expected_named_gradient_count"),
                            comparison.get("every_named_gradient"))
        _require(type(count) is int and count > 0 and isinstance(gradients, dict)
                 and len(gradients) == count and all(
                     isinstance(name, str) and bool(name) and isinstance(entry, dict)
                     and all(entry.get(field) is True for field in (
                         "both_present", "bitwise_equal", "both_finite"))
                     for name, entry in gradients.items()),
                 f"incomplete or nonfinite named aux gradients: {precision}")
        routes = comparison.get("route_results_bitwise_equal")
        _require(isinstance(routes, dict) and bool(routes) and all(
            isinstance(name, str) and bool(name) and equal is True
            for name, equal in routes.items()), f"missing aux route equality: {precision}")
    audit, guard = report.get("source_audit"), report.get("oracle_guard_check")
    _require(isinstance(audit, dict) and audit.get("status") == "PASS"
             and audit.get("numerical_consumers") == [], "aux numerical-consumer audit failed")
    _require(isinstance(guard, dict) and guard.get("status") == "PASS"
             and guard.get("oracle_source_sha256") == ORACLE_SHA256 and all(
                 guard.get(name) is True for name in ("normalized_config_accepted",
                                                     "normalized_prehook_accepted",
                                                     "original_nonzero_rejected")),
             "unchanged Oracle guard admission missing")
    return {"report": report_descriptor, "contract": contract_descriptor,
            "source_audit": audit, "oracle_guard_check": guard,
            "source_provenance": {name: report[name] for name in (
                "packages", "sources", "source_tree_sha256", "repository_sources")
                if name in report},
            "normalization": [{"field": name, "source_value": value, "runtime": "absent",
                               "reason": _NORMALIZATION_REASON}
                              for name, value in _FIELDS.items()]}


def prefix_shapes() -> dict[str, tuple[int, ...]]:
    """Exact checkpoint layout, independent of either model's initialized state."""
    result = {"model.embed_tokens.weight": (163840, 2048)}
    for index in (0, 1):
        root = f"model.layers.{index}."
        result.update({root + "input_layernorm.weight": (2048,),
                       root + "post_attention_layernorm.weight": (2048,)})
        result.update({root + "self_attn." + name: shape for name, shape in {
            "q_proj.weight": (3072, 2048), "kv_a_proj_with_mqa.weight": (576, 2048),
            "kv_a_layernorm.weight": (512,), "kv_b_proj.weight": (4096, 512),
            "o_proj.weight": (2048, 2048), "rotary_emb.inv_freq": (64,),
        }.items()})
        if index == 0:
            intermediate = 11264
            for name in ("gate_proj", "up_proj", "down_proj"):
                shape = (2048, intermediate) if name == "down_proj" else (intermediate, 2048)
                result[root + f"mlp.{name}.weight"] = shape
        else:
            result[root + "mlp.gate.weight"] = (64, 2048)
            result[root + "mlp.gate.e_score_correction_bias"] = (64,)
            for name in ("gate_proj", "up_proj", "down_proj"):
                shared = (2048, 2816) if name == "down_proj" else (2816, 2048)
                expert = (2048, 1408) if name == "down_proj" else (1408, 2048)
                result[root + f"mlp.shared_experts.{name}.weight"] = shared
                for expert_id in range(64):
                    result[root + f"mlp.experts.{expert_id}.{name}.weight"] = expert
    return result


def validate_prefix_map(weight_map: Any, metadata: dict[str, Any]) -> list[str]:
    """Refuse missing, extra or third-shard requested prefix keys before allocation."""
    _require(isinstance(weight_map, dict) and bool(weight_map) and all(
        isinstance(name, str) and isinstance(shard, str) for name, shard in weight_map.items()),
        "nonempty checkpoint weight map required")
    expected = set(prefix_shapes())
    declared = metadata.get("reference_prefix_tensor_names")
    _require(isinstance(declared, list) and all(isinstance(name, str) for name in declared)
             and len(declared) == len(set(declared)) and set(declared) == expected,
             "reviewed prefix key coverage differs")
    observed = {name for name in weight_map if name == "model.embed_tokens.weight"
                or name.startswith(("model.layers.0.", "model.layers.1."))}
    _require(observed == expected, "checkpoint prefix key coverage differs")
    shards = sorted({weight_map[name] for name in observed})
    _require(shards == ["model-1-of-27.safetensors", "model-2-of-27.safetensors"]
             and shards == metadata.get("reference_prefix_weight_files"),
             "prefix must use exactly the two reviewed complete shards")
    return shards


def _safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        _require(len(prefix) == 8, "truncated safetensors header")
        size = struct.unpack("<Q", prefix)[0]
        _require(1 <= size <= 64 * 1024**2 and size + 8 < path.stat().st_size,
                 "invalid safetensors header length")
        raw = stream.read(size)
    _require(len(raw) == size, "truncated safetensors metadata")
    value = json.loads(raw, object_pairs_hook=_unique, parse_constant=_reject_constant)
    _require(isinstance(value, dict), "safetensors header must be an object")
    return {name: entry for name, entry in value.items() if name != "__metadata__"}


def validate_sources(source_dir: Path) -> dict[str, Any]:
    """Hash complete local shards and both executable reference files, never fetch."""
    source_dir = Path(source_dir)
    inventory_path, metadata_path = (_RESULTS / "source-inventory-v1.json",
                                     _RESULTS / "source-metadata-admission-v1.json")
    _require(_sha256(inventory_path) == INVENTORY_SHA256, "reviewed inventory hash mismatch")
    _require(_sha256(metadata_path) == METADATA_SHA256, "reviewed source metadata hash mismatch")
    inventory, metadata = _json(inventory_path), _json(metadata_path)
    for value in (inventory, metadata):
        _require(value.get("model_id") == MODEL_ID and value.get("revision") == REVISION,
                 "source metadata model/revision mismatch")
    _require(metadata.get("inventory_sha256") == INVENTORY_SHA256,
             "source metadata inventory binding mismatch")
    files: dict[str, Any] = {}
    for name, (size, blob, digest) in _METADATA_FILES.items():
        path = _child(source_dir, name)
        receipt = _descriptor(path)
        _require(receipt["bytes"] == size and receipt["sha256"] == digest
                 and _git_blob(path.read_bytes()) == blob, f"pinned metadata mismatch: {name}")
        _require(inventory["files"][name]["bytes"] == size
                 and inventory["files"][name]["hub_etag"] == blob,
                 "metadata inventory mismatch")
        files[name] = {**receipt, "git_blob_sha1": blob}
    config = _json(_child(source_dir, "config.json"))
    _require(all(config.get(name) == value and type(config.get(name)) is type(value)
                 for name, value in {**_SHAPE, **_FIELDS}.items()),
             "pinned Moonlight configuration dimensions or routing differ")
    _require("quantization_config" not in config, "BF16 source must not declare quantization")
    index = _json(_child(source_dir, "model.safetensors.index.json"))
    weight_map = index.get("weight_map")
    shards = validate_prefix_map(weight_map, metadata)
    _require(set(weight_map.values()) == {f"model-{index}-of-27.safetensors"
                                         for index in range(1, 28)}
             and len(weight_map) == metadata.get("tensor_names") == 5344,
             "checkpoint index is not the complete pinned inventory")
    _require(not any("weight_scale_inv" in name for name in weight_map),
             "quantized source tensors are forbidden")
    tensor_metadata: dict[str, Any] = {}
    complete_headers: dict[str, Any] = {}
    header_counts: dict[str, Any] = {}
    expected_shapes = prefix_shapes()
    for name in shards:
        path = _child(source_dir, name)
        expected = inventory["files"][name]
        receipt = _descriptor(path)
        _require(receipt["bytes"] == expected["bytes"]
                 and receipt["sha256"] == expected["hub_sha256"],
                 f"complete shard size/SHA256 mismatch: {name}")
        header = _safetensors_header(path)
        _require(set(header) == {key for key, shard in weight_map.items() if shard == name},
                 f"complete shard/index key coverage mismatch: {name}")
        for key, entry in header.items():
            _require(isinstance(entry, dict) and entry.get("dtype") in {"BF16", "F32"},
                     f"BF16/FP32 source tensor required: {key}")
            shape, offsets = entry.get("shape"), entry.get("data_offsets")
            _require(isinstance(shape, list) and all(type(dim) is int and dim > 0 for dim in shape)
                     and isinstance(offsets, list) and len(offsets) == 2
                     and all(type(offset) is int for offset in offsets)
                     and 0 <= offsets[0] <= offsets[1], f"invalid tensor header: {key}")
            elements = math.prod(shape)
            item_size = 2 if entry["dtype"] == "BF16" else 4
            _require(offsets[1] - offsets[0] == elements * item_size,
                     f"tensor header byte count differs: {key}")
            classification = ("derived_nonpersistent_rotary_buffer"
                              if key.endswith(".rotary_emb.inv_freq") else
                              "native_persistent_router_buffer_reference_parameter"
                              if key.endswith(".e_score_correction_bias") else "parameter")
            complete_headers[key] = {"shard": name, **entry, "numel": elements,
                                     "payload_bytes": elements * item_size,
                                     "classification": classification}
            counts = header_counts.setdefault(name, {
                "checkpoint_tensor_names": 0, "parameter_tensor_names_excluding_inv_freq": 0,
                "derived_buffer_tensor_names": 0, "native_persistent_router_buffer_names": 0,
                "parameter_numel_excluding_inv_freq": 0, "derived_buffer_numel": 0,
                "tensor_payload_bytes": 0, "payload_bytes_by_dtype": {},
            })
            derived = classification == "derived_nonpersistent_rotary_buffer"
            counts["checkpoint_tensor_names"] += 1
            counts["parameter_tensor_names_excluding_inv_freq"] += int(not derived)
            counts["derived_buffer_tensor_names"] += int(derived)
            counts["native_persistent_router_buffer_names"] += int(
                classification == "native_persistent_router_buffer_reference_parameter")
            count_key = "derived_buffer_numel" if derived else "parameter_numel_excluding_inv_freq"
            counts[count_key] += elements
            counts["tensor_payload_bytes"] += elements * item_size
            dtype_counts = counts["payload_bytes_by_dtype"]
            dtype_counts[entry["dtype"]] = (
                dtype_counts.get(entry["dtype"], 0) + elements * item_size)
            if key not in expected_shapes:
                continue
            _require(entry.get("shape") == list(expected_shapes[key]),
                     f"source prefix shape mismatch: {key}")
            constant = key.endswith(("rotary_emb.inv_freq", "e_score_correction_bias"))
            _require(constant or entry["dtype"] == "BF16", f"BF16 source weight required: {key}")
            tensor_metadata[key] = complete_headers[key]
        files[name] = receipt
    _require(set(tensor_metadata) == set(expected_shapes)
             and sum(files[name]["bytes"] for name in shards) == SHARD_BYTES,
             "complete prefix coverage or byte total mismatch")
    for name, blob in _REFERENCE_BLOBS.items():
        path = _child(source_dir, name)
        receipt = _descriptor(path)
        _require(0 < receipt["bytes"] <= 512 * 1024 and _git_blob(path.read_bytes()) == blob,
                 f"pinned reference source blob mismatch: {name}")
        files[name] = {**receipt, "git_blob_sha1": blob}
    return {"model_id": MODEL_ID, "revision": REVISION, "profile_id": PROFILE_ID,
            "source_dir": os.path.realpath(source_dir), "files": files,
            "inventory": _descriptor(inventory_path), "metadata": _descriptor(metadata_path),
            "weight_map": {name: weight_map[name] for name in sorted(expected_shapes)},
            "tensor_metadata": tensor_metadata, "weight_shards": shards,
            "complete_shard_tensor_headers": complete_headers,
            "complete_shard_header_counts": header_counts,
            "parameter_accounting_scope": (
                "two downloaded shard headers only; all checkpoint parameters except inv_freq; "
                "router correction bias is a native persistent buffer and reference parameter"),
            "index_declared_tensor_payload_bytes": index.get("metadata", {}).get("total_size"),
            "index_tensor_name_classification": {
                "all_tensor_names": len(weight_map),
                "derived_inv_freq_names": sum(key.endswith(".rotary_emb.inv_freq")
                                              for key in weight_map),
                "other_tensor_names": sum(not key.endswith(".rotary_emb.inv_freq")
                                          for key in weight_map),
                "all_shapes_materialized": False,
            },
            "complete_shard_bytes_hashed": SHARD_BYTES, "config": config}


def _valid_ids(ids: Any) -> bool:
    return (isinstance(ids, list) and len(ids) == 64
            and all(type(value) is int and 0 <= value < 163840 for value in ids))


def _manifest_contract(path: Path, expected_hash: str) -> dict[str, Any]:
    _require(_hex(expected_hash) and _sha256(path) == expected_hash,
             "final training pack manifest hash mismatch")
    manifest = _json(path)
    _require(isinstance(manifest, dict), "training pack manifest must be an object")
    keys = [key for key in ("manifest_payload_sha256", "manifest_sha256") if key in manifest]
    _require(len(keys) == 1, "training pack manifest payload hash missing or ambiguous")
    payload = {name: value for name, value in manifest.items() if name != keys[0]}
    _require(manifest[keys[0]] == hashlib.sha256(_canonical(payload)).hexdigest(),
             "training pack manifest payload hash mismatch")
    _require(manifest.get("profile_id") == PROFILE_ID and manifest.get("test_only") is False
             and manifest.get("schema") == "p3-deepseek-quality-training-pack-v1"
             and manifest.get("policy") == "p3-deepseek-quality-training-pack-v1"
             and manifest.get("status") == "PAIRED_DATA_PACKED_RUNTIME_AND_PRECISION_PENDING"
             and manifest.get("training_sources_sha256") == (
                 "8ab0bdfdefde056387835fcaf38914fe2c6879cde1cf89a8f6169629630675a7"),
             "final production Moonlight pack required")
    tokenizer = manifest.get("tokenizer")
    _require(isinstance(tokenizer, dict) and tokenizer.get("repo_id") == MODEL_ID
             and tokenizer.get("revision") == REVISION
             and tokenizer.get("vocabulary_size") == 163840, "training pack tokenizer differs")
    _require(isinstance(manifest.get("files"), dict) and bool(manifest["files"]),
             "training pack file descriptors missing")
    return manifest


def _select(windows: Any, selection: dict[str, Any]) -> list[int]:
    _require(isinstance(selection, dict) and set(selection) == {
        "length", "window_index", "micro_batch_index", "token_offset", "token_count"},
        "explicit pack slice selection required")
    _require(all(type(value) is int for value in selection.values())
             and selection["length"] in (1024, 4096) and selection["token_count"] == 64
             and all(selection[name] >= 0 for name in (
                 "window_index", "micro_batch_index", "token_offset")),
             "invalid exact 64-token pack slice")
    _require(isinstance(windows, list) and 0 <= selection["window_index"] < len(windows),
             "selected pack window absent")
    window = windows[selection["window_index"]]
    _require(isinstance(window, dict) and isinstance(window.get("micro_batches"), list)
             and selection["micro_batch_index"] < len(window["micro_batches"]),
             "selected micro-batch absent")
    batch = window["micro_batches"][selection["micro_batch_index"]]
    ids = batch.get("input_ids") if isinstance(batch, dict) else None
    _require(isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list)
             and len(ids[0]) == selection["length"], "one full pack micro-batch required")
    offset = selection["token_offset"]
    result = ids[0][offset:offset + 64]
    _require(_valid_ids(result), "exactly 64 in-vocabulary integer pack tokens required")
    return result


def load_input(*, pack_manifest_sha256: str, pack: Path | None = None,
               input_evidence: Path | None = None, length: int = 1024, window_index: int = 0,
               micro_batch_index: int = 0, token_offset: int = 0) -> dict[str, Any]:
    """Read a verified final pack or a detached manifest/windows/slice evidence bundle.

    Detached evidence includes the exact final manifest and the full windows file
    it hashes. A bare JSON array of token IDs or a self-asserted source hash is not
    evidence. The pack manifest hash is a separate, required caller input.
    """
    _require(pack_manifest_sha256 == PACK_MANIFEST_SHA256,
             "approved training pack manifest SHA256 required")
    _require((pack is None) != (input_evidence is None),
             "choose pack or source-bound input evidence")
    files: dict[str, Any] = {}
    if pack is not None:
        from benchmarks.harness import p3_deepseek_quality_data as data

        root = Path(pack)
        manifest_path = _child(root, "manifest.json")
        manifest = _manifest_contract(manifest_path, pack_manifest_sha256)
        data.load_pack(root, pack_manifest_sha256, length=length, steps=64, profile_id=PROFILE_ID)
        selection = {"length": length, "window_index": window_index,
                     "micro_batch_index": micro_batch_index, "token_offset": token_offset,
                     "token_count": 64}
        descriptor = manifest["files"][f"windows_L{length}"]
        windows_path = _artifact(root, descriptor)
        files.update({key: _descriptor(_artifact(root, value))
                      for key, value in manifest["files"].items()})
        mode = "verified_final_training_pack"
    else:
        evidence_path = Path(input_evidence)
        evidence = _json(evidence_path)
        _require(isinstance(evidence, dict)
                 and evidence.get("schema") == "p3-moonlight-reference-input-v1"
                 and evidence.get("profile_id") == PROFILE_ID
                 and evidence.get("model_id") == MODEL_ID
                 and evidence.get("revision") == REVISION
                 and evidence.get("training_pack_manifest_sha256") == pack_manifest_sha256,
                 "source-bound input identity or final pack hash mismatch")
        root = evidence_path.parent
        manifest_path = _artifact(root, evidence.get("manifest"))
        manifest = _manifest_contract(manifest_path, pack_manifest_sha256)
        selection = evidence.get("selection")
        _require(isinstance(selection, dict) and selection.get("length") in (1024, 4096),
                 "input evidence slice selection missing")
        descriptor = evidence.get("windows")
        _require(descriptor == manifest.get("files", {}).get(f"windows_L{selection['length']}"),
                 "input windows are not bound to the final manifest")
        windows_path = _artifact(root, descriptor)
        files["input_evidence"] = _descriptor(evidence_path)
        mode = "source_bound_final_pack_slice"
    _require(isinstance(descriptor, dict) and descriptor.get("format") == "json_array"
             and type(descriptor.get("rows")) is int and descriptor["rows"] > 0
             and descriptor.get("length") == selection["length"]
             and descriptor.get("batch_size") == 1, "hashed windows descriptor is incomplete")
    windows = _json(windows_path)
    _require(isinstance(windows, list) and len(windows) == descriptor["rows"],
             "hashed windows row count differs from manifest")
    ids = _select(windows, selection)
    if input_evidence is not None:
        _require(_valid_ids(evidence.get("input_ids")) and evidence["input_ids"] == ids,
                 "input evidence tokens differ from the hashed final pack slice")
    files.update(manifest=_descriptor(manifest_path), windows=_descriptor(windows_path))
    return {"schema": "p3-moonlight-reference-input-v1", "mode": mode,
            "model_id": MODEL_ID, "revision": REVISION, "profile_id": PROFILE_ID,
            "training_pack_manifest_sha256": pack_manifest_sha256, "selection": selection,
            "input_ids": ids, "input_ids_json_sha256": hashlib.sha256(_canonical(ids)).hexdigest(),
            "files": files, "attention_mask": "64 unpadded tokens, causal, positions 0..63",
            "original_pack_position_offset": selection["token_offset"]}


def _check_output(value: Any, name: str) -> None:
    import torch

    _require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
             and value.dtype == torch.float32 and value.layout == torch.strided
             and tuple(value.shape) == (1, 64, 2048) and bool(torch.isfinite(value).all()),
             f"{name} must be nonempty finite CPU FP32 [1,64,2048]")


def _check_routes(value: Any, name: str) -> None:
    import torch

    _require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
             and value.dtype == torch.int64 and tuple(value.shape) == (64, 6)
             and bool(((value >= 0) & (value < 64)).all()),
             f"{name} must be CPU int64 top6 IDs for every one of 64 tokens")
    sorted_ids = value.sort(dim=-1).values
    _require(bool((sorted_ids[:, 1:] != sorted_ids[:, :-1]).all()),
             f"{name} contains repeated experts")


def _residual_statistics(native: Any, reference: Any) -> dict[str, Any]:
    residual = native.double() - reference.double()
    absolute = residual.abs()
    ref_max = reference.double().abs().max().item()
    maximum = absolute.max().item()
    return {"reference_max_abs": ref_max, "native_max_abs": native.double().abs().max().item(),
            "max_abs_error": maximum, "mean_abs_error": absolute.mean().item(),
            "rms_error": residual.square().mean().sqrt().item(),
            "signed_mean_error": residual.mean().item(), "residual_l2": residual.norm().item(),
            "relative_max_abs_error": (
                maximum / ref_max if ref_max else (0.0 if not maximum else None)),
            "per_token_max_abs_error": absolute.reshape(64, -1).max(dim=-1).values.tolist()}


def compare_outputs(native: Any, reference: Any, native_ids: Any,
                    reference_ids: Any) -> dict[str, Any]:
    """Apply the fixed acceptance rule once, preserving native top-k order.

    The exact rule is max_abs(native-reference) <= 1e-4*max_abs(reference).
    There is no additive floor, epsilon, sorting of route IDs or relaxed fallback.
    A zero reference therefore permits only zero error. Malformed evidence raises.
    """
    _check_output(native, "native layer1 output")
    _check_output(reference, "reference layer1 output")
    _check_routes(native_ids, "native selected experts")
    _check_routes(reference_ids, "reference selected experts")
    statistics = _residual_statistics(native, reference)
    mismatches = (native_ids != reference_ids).any(dim=-1).nonzero().reshape(-1).tolist()
    limit = 1e-4 * statistics["reference_max_abs"]
    conditions = {"identical_top6_expert_ids": not mismatches,
                  "layer1_max_abs_error_within_limit": statistics["max_abs_error"] <= limit}
    return {"status": "PASS" if all(conditions.values()) else "FAIL", "conditions": conditions,
            "route_mismatch_tokens": mismatches, "compared_tokens": 64, "experts_per_token": 6,
            "relative_tolerance": 1e-4, "absolute_tolerance": 0.0,
            "allowed_max_abs_error": limit, "measurement": "layer1 output before final norm",
            **statistics}


class _Provenance:
    """Retain executed installed Python sources and check their wheel RECORDs."""

    def __init__(self, output: Path, role: str) -> None:
        self.output = output
        self.packages: dict[str, Any] = {}
        self.sources: dict[str, Any] = {}
        self.called: set[str] = set()
        self.roots: dict[str, Path] = {}
        self.entries: dict[str, Any] = {}
        expected = {**COMMON_VERSIONS, **_TRANSFORMERS_DEPENDENCIES[role], "transformers": (
            NATIVE_TRANSFORMERS if role == "native" else REFERENCE_TRANSFORMERS)}
        for name, required in expected.items():
            distribution = importlib.metadata.distribution(name)
            observed_version = (distribution.version.split("+")[0] if name == "torch"
                                else distribution.version)
            _require(observed_version == required, f"exactly {name}=={required} required")
            root = Path(os.path.realpath(distribution.locate_file(name)))
            _require(_contained(root, Path(sys.prefix)), f"{name} is outside the isolated venv")
            self.roots[name] = root
            entries = {str(item).replace("\\", "/"): item for item in distribution.files or ()}
            self.entries[name] = entries
            metadata = {}
            for label in ("METADATA", "RECORD"):
                candidates = [item for key, item in entries.items()
                              if key.endswith(f".dist-info/{label}")]
                _require(len(candidates) == 1, f"missing unique {name} wheel {label}")
                original = Path(distribution.locate_file(candidates[0]))
                target = output / "packages" / name / label
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, target)
                metadata[label] = _descriptor(target)
            self.packages[name] = {"version": distribution.version, "root": str(root),
                                   "metadata": metadata, "requires": distribution.requires or []}
        self.initial_metadata = {
            name: dict(value["metadata"]) for name, value in self.packages.items()}
        transformer_root = self.roots["transformers"]
        required_sources = ["configuration_utils.py", "modeling_utils.py"]
        if role == "native":
            required_sources.extend(_NATIVE_SOURCE_PINS)
            required_sources.extend(["modeling_rope_utils.py", "integrations/moe.py"])
            for relative, expected_hash in _NATIVE_SOURCE_PINS.items():
                _require(_sha256(_child(transformer_root, relative)) == expected_hash,
                         f"native source differs from reviewed 5.19.0 wheel: {relative}")
        self.called.update(os.path.realpath(_child(transformer_root, relative))
                           for relative in required_sources)
        for relative, key in (
            ("src/soup_cli/utils/layer_shard.py", "repository/src/soup_cli/utils/layer_shard.py"),
            ("src/soup_cli/utils/fp8_source.py", "repository/src/soup_cli/utils/fp8_source.py"),
        ):
            if role == "native":
                original = _child(_REPO, relative)
                target = output / key
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, target)
                self.sources[key] = {"installed": _descriptor(original),
                                     "retained": _descriptor(target),
                                     "record_hash": None, "record_hash_matches": None,
                                     "provenance": "repository source bytes"}

    def _profile(self, frame: Any, event: str, _arg: Any) -> None:
        if event == "call" and frame.f_code.co_filename.endswith(".py"):
            self.called.add(os.path.realpath(frame.f_code.co_filename))

    def start(self) -> None:
        _require(sys.getprofile() is None, "an existing profiler would invalidate source coverage")
        sys.setprofile(self._profile)

    def finish(self) -> dict[str, Any]:
        sys.setprofile(None)
        for module in tuple(sys.modules.values()):
            filename = getattr(module, "__file__", None)
            if isinstance(filename, str) and filename.endswith((".so", ".pyd", ".dll")):
                self.called.add(os.path.realpath(filename))
        for filename in sorted(self.called):
            path = Path(filename)
            for name, root in self.roots.items():
                if not _contained(path, root):
                    continue
                relative = Path(os.path.relpath(path, root)).as_posix()
                key = f"{name}/{relative}"
                item = self.entries[name].get(key)
                _require(item is not None and item.hash is not None and item.hash.mode == "sha256",
                         f"executed source/binary lacks wheel RECORD SHA256: {key}")
                observed = _descriptor(path)
                encoded = base64.urlsafe_b64encode(bytes.fromhex(observed["sha256"]))
                _require(encoded.decode("ascii").rstrip("=") == item.hash.value
                         and item.size == observed["bytes"],
                         f"executed source/binary differs from wheel RECORD: {key}")
                retained = None
                if path.suffix == ".py":
                    _require(len(self.sources) < 4096, "executed source coverage limit exceeded")
                    target = self.output / "sources" / key
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
                    retained = _descriptor(target)
                self.sources[key] = {"installed": observed, "retained": retained,
                                     "record_hash": item.hash.value, "record_hash_matches": True}
                break
        for filename in sorted(self.called):
            path = Path(filename)
            if (not _contained(path, _REPO / "src/soup_cli") or path.suffix != ".py"):
                continue
            relative = Path(os.path.relpath(path, _REPO)).as_posix()
            key = f"repository/{relative}"
            if key in self.sources:
                continue
            target = self.output / key
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            self.sources[key] = {"installed": _descriptor(path), "retained": _descriptor(target),
                                 "record_hash": None, "record_hash_matches": None,
                                 "provenance": "executed repository Python source bytes"}
        for entry in self.sources.values():
            installed = entry["installed"]
            _require(_descriptor(Path(installed["path"])) == installed,
                     "installed source changed during provenance retention")
            retained = entry.get("retained")
            if retained is not None:
                _require(retained["sha256"] == installed["sha256"]
                         and retained["bytes"] == installed["bytes"],
                         "retained source bytes differ from installed source")
        for name, metadata in self.initial_metadata.items():
            for label, original in metadata.items():
                key = next(key for key in self.entries[name]
                           if key.endswith(f".dist-info/{label}"))
                installed = Path(importlib.metadata.distribution(name).locate_file(key))
                _require(installed.stat().st_size == original["bytes"]
                         and _sha256(installed) == original["sha256"],
                         f"installed {name} {label} changed during execution")
        _require(any("modeling_deepseek" in key for key in self.sources)
                 or "transformers/modeling_utils.py" in self.sources,
                 "executed Transformers source evidence is empty")
        return {"packages": self.packages, "executed_sources": self.sources,
                "source_tree_sha256": hashlib.sha256(_canonical({
                    key: value["installed"]["sha256"]
                    for key, value in sorted(self.sources.items())})).hexdigest(),
                "coverage": "profiled Python calls and loaded package extension binaries",
                "binary_payload_retained": False}


def _runtime(role: str) -> dict[str, Any]:
    import numpy
    import safetensors
    import torch
    import transformers

    _require(sys.prefix != sys.base_prefix, "worker must run in a virtual environment")
    expected = NATIVE_TRANSFORMERS if role == "native" else REFERENCE_TRANSFORMERS
    _require(transformers.__version__ == expected, "worker Transformers import/version mismatch")
    _require(torch.__version__.split("+")[0] == COMMON_VERSIONS["torch"],
             "worker torch import/version mismatch")
    _require(numpy.__version__ == COMMON_VERSIONS["numpy"]
             and safetensors.__version__ == COMMON_VERSIONS["safetensors"],
             "worker numerical package import/version mismatch")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.mkldnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_default_dtype(torch.float32)
    _require(not torch.is_autocast_enabled("cpu"), "CPU autocast is forbidden")
    return {"role": role, "python": sys.version, "executable": sys.executable,
            "interpreter_binary": _descriptor(Path(sys.executable)),
            "prefix": os.path.realpath(sys.prefix),
            "base_prefix": os.path.realpath(sys.base_prefix),
            "platform": platform.platform(), "transformers": transformers.__version__,
            "torch": torch.__version__, "numpy": numpy.__version__,
            "safetensors": safetensors.__version__, "device": "cpu", "dtype": "torch.float32",
            "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "mkldnn": torch.backends.mkldnn.enabled, "autocast": False,
            "attention_implementation": "eager", "experts_implementation": "eager",
            "weights_downloaded": False, "oracle_attached": False, "training": False}


def _reference_module(source_dir: Path) -> Any:
    """Execute only two blob-verified files, without Hub auto-loading or other code."""
    _require(importlib.metadata.version("transformers") == REFERENCE_TRANSFORMERS,
             "reference code may execute only in its pinned isolated environment")
    package_name = "_soup_pinned_moonlight_reference"
    package = types.ModuleType(package_name)
    package.__path__ = [os.path.realpath(source_dir)]
    sys.modules[package_name] = package
    loaded = None
    for name in ("configuration_deepseek.py", "modeling_deepseek.py"):
        path = _child(source_dir, name)
        payload = path.read_bytes()
        _require(_git_blob(payload) == _REFERENCE_BLOBS[name],
                 "reference executable source changed before import")
        module_name = f"{package_name}.{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        _require(spec is not None and spec.loader is not None, "reference import spec missing")
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = loaded
        exec(compile(payload, str(path), "exec"), loaded.__dict__)
    return loaded


def _native_module() -> Any:
    from transformers.models.deepseek_v3 import modeling_deepseek_v3

    _require(
        modeling_deepseek_v3.__name__ == "transformers.models.deepseek_v3.modeling_deepseek_v3",
        "native production class must not execute Hub remote code")
    return modeling_deepseek_v3


def _read_tensor(handle: Any, key: str, shapes: dict[str, tuple[int, ...]]) -> Any:
    import torch

    value = handle.get_tensor(key)
    _require(tuple(value.shape) == shapes[key] and value.dtype in (torch.bfloat16, torch.float32),
             f"loaded prefix tensor shape/dtype differs: {key}")
    result = value.to(dtype=torch.float32, device="cpu").contiguous()
    if result is value:
        result = result.clone()
    _require(bool(torch.isfinite(result).all()), f"nonfinite pinned source tensor: {key}")
    return result


def _load_prefix(role: str, module: Any, source: dict[str, Any], output: Path,
                 config: Any) -> tuple[Any, list[Any], dict[str, Any]]:
    import torch
    from safetensors import safe_open

    with torch.device("meta"):
        prefix = torch.nn.Module()
        prefix.embed_tokens = torch.nn.Embedding(163840, 2048)
        prefix.layers = torch.nn.ModuleList([
            module.DeepseekV3DecoderLayer(config, index) for index in (0, 1)])
    shapes = prefix_shapes()
    state_names, source_keys = {}, set()
    rotary_modules, inverse_frequencies = [], []
    rotary_evidence: dict[str, Any] = {}
    conversion = None
    if role == "native":
        # Reuse Soup's source-validated gate/up concatenation and integer expert ordering.
        sys.path.insert(0, str(_REPO / "src"))
        from soup_cli.utils.layer_shard import _stack_experts

        conversion_file = Path(inspect.getfile(_stack_experts))
        _require(os.path.realpath(conversion_file) == os.path.realpath(
            _REPO / "src/soup_cli/utils/layer_shard.py"),
            "expert conversion must use this repository's verified helper")
        target = output / "repository" / "src/soup_cli/utils/layer_shard.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(conversion_file, target)
        conversion = {"function": "soup_cli.utils.layer_shard._stack_experts",
                      "source": _descriptor(conversion_file), "retained": _descriptor(target)}
    with ExitStack() as stack:
        handles = {name: stack.enter_context(safe_open(
            str(_child(Path(source["source_dir"]), name)), framework="pt", device="cpu"))
            for name in source["weight_shards"]}

        def read(key: str) -> Any:
            _require(key not in source_keys, f"duplicate source weight consumption: {key}")
            source_keys.add(key)
            return _read_tensor(handles[source["weight_map"][key]], key, shapes)

        embedded = {"weight": read("model.embed_tokens.weight")}
        prefix.embed_tokens.load_state_dict(embedded, strict=True, assign=True)
        state_names["embed_tokens"] = ["weight"]
        del embedded
        for index, layer in enumerate(prefix.layers):
            root = f"model.layers.{index}."
            frequency_key = root + "self_attn.rotary_emb.inv_freq"
            frequency = read(frequency_key)
            _require(bool((frequency > 0).all()), "rotary frequencies must be finite positive")
            inverse_frequencies.append(frequency)
            rotary_evidence[f"layer{index}.checkpoint_inv_freq_raw"] = handles[
                source["weight_map"][frequency_key]].get_tensor(frequency_key).clone().contiguous()
            keys = sorted(key for key in shapes if key.startswith(root) and key != frequency_key)
            layer_state = {}
            if role == "native" and index == 1:
                expert_keys = {key for key in keys if key.startswith(root + "mlp.experts.")}
                experts: dict[str, dict[int, tuple[str, str]]] = {
                    name: {} for name in ("gate_proj", "up_proj", "down_proj")}
                for key in expert_keys:
                    suffix = key[len(root + "mlp.experts."):].split(".")
                    expert_id, projection = int(suffix[0]), suffix[1]
                    experts[projection][expert_id] = (source["weight_map"][key], key)
                _require(len(expert_keys) == 192, "all 64 gate/up/down expert weights required")
                # Shapes/dtypes came from the complete-shard admission; this helper reads
                # only those exact tensors and performs the existing verified conversion.
                count = _stack_experts(index, experts, layer_state, handles, "float32")
                _require(count == sum(math.prod(shapes[key]) for key in expert_keys),
                         "native expert conversion parameter coverage differs")
                source_keys.update(expert_keys)
                keys = [key for key in keys if key not in expert_keys]
            for key in keys:
                layer_state[key[len(root):]] = read(key)
            layer.load_state_dict(layer_state, strict=True, assign=True)
            state_names[str(index)] = sorted(layer_state)
            del layer_state
            if role == "native":
                rotary = module.DeepseekV3RotaryEmbedding(config).to(device="cpu")
            else:
                # The meta skeleton's nonpersistent constants are derived by the
                # pinned constructor on CPU, just as in a normal reference load.
                rotary = module.DeepseekV3RotaryEmbedding(
                    config.qk_rope_head_dim,
                    max_position_embeddings=config.max_position_embeddings,
                    base=config.rope_theta, device=torch.device("cpu"))
                layer.self_attn.rotary_emb = rotary
            _require(tuple(rotary.inv_freq.shape) == (32,) and not rotary.inv_freq.is_meta,
                     "derived rotary dimensions/device differ")
            rotary_evidence[f"layer{index}.checkpoint_inv_freq"] = frequency
            rotary_evidence[f"layer{index}.derived_inv_freq"] = rotary.inv_freq.detach().clone()
            rotary_modules.append(rotary)
    _require(source_keys == set(shapes), "not every requested prefix source tensor was consumed")
    from safetensors.torch import save_file

    rotary_path = output / "rotary-constants.safetensors"
    save_file(rotary_evidence, str(rotary_path))
    rotary_diagnostics = {}
    for index in (0, 1):
        checkpoint = rotary_evidence[f"layer{index}.checkpoint_inv_freq"]
        derived = rotary_evidence[f"layer{index}.derived_inv_freq"]
        rotary_diagnostics[str(index)] = {
            "checkpoint_values_fp32": checkpoint.tolist(), "derived_values_fp32": derived.tolist(),
            "checkpoint_stored_dtype": source["tensor_metadata"][
                f"model.layers.{index}.self_attn.rotary_emb.inv_freq"]["dtype"],
            "checkpoint_shape": list(checkpoint.shape), "checkpoint_numel": checkpoint.numel(),
            "derived_shape": list(derived.shape),
            "same_shape_diagnostic_only": checkpoint.shape == derived.shape,
            "bitwise_equal_diagnostic_only": bool(torch.equal(checkpoint, derived)),
            "max_abs_difference_diagnostic_only": (
                (checkpoint.double() - derived.double()).abs().max().item()
                if checkpoint.shape == derived.shape else None),
            "acceptance_gate": False,
        }
    for name, value in list(prefix.named_parameters()) + list(prefix.named_buffers()):
        _require(not value.is_meta and value.device.type == "cpu" and value.dtype == torch.float32
                 and bool(torch.isfinite(value).all()),
                 f"unloaded/nonfinite/non-FP32 tensor: {name}")
    if conversion is not None:
        _require(_descriptor(Path(conversion["source"]["path"])) == conversion["source"],
                 "expert conversion source changed while loading")
    prefix.requires_grad_(False)
    prefix.eval()
    for rotary in rotary_modules:
        rotary.eval()
    return prefix, rotary_modules, {
        "strict_load": True, "missing_keys": [], "unexpected_keys": [],
        "source_keys_consumed": sorted(source_keys), "native_state_keys": state_names,
        "rotary_constants": _descriptor(rotary_path),
        "source_rotary_inv_freq_classified_without_override": True,
        "nonpersistent_checkpoint_keys": [
            f"model.layers.{index}.self_attn.rotary_emb.inv_freq" for index in (0, 1)],
        "rotary_diagnostics": rotary_diagnostics,
        "checkpoint_layer_frequencies_equal_diagnostic_only": bool(
            torch.equal(*inverse_frequencies)),
        "nonpersistent_classification_sources": (
            [{"path": "transformers/models/deepseek_v3/modeling_deepseek_v3.py",
              "lines": [71, 72], "reason": "inv_freq and original_inv_freq are nonpersistent"},
             {"path": "transformers/modeling_utils.py", "lines": [4821, 4825],
              "reason": "legacy per-layer rotary inv_freq unexpected keys are ignored"},
             {"path": "src/soup_cli/utils/layer_shard.py", "lines": [1731, 1732],
              "reason": "production sharder skips nonpersistent rotary constants"}]
            if role == "native" else
            [{"path": "modeling_deepseek.py", "git_blob_sha1": _REFERENCE_BLOBS[
                "modeling_deepseek.py"], "lines": [118, 118],
              "reason": "inv_freq registered with persistent=False"},
             {"path": "transformers/modeling_utils.py", "lines": [4428, 4435],
              "reason": "nonpersistent buffers removed from unexpected keys"},
             {"path": "transformers/modeling_utils.py", "lines": [909, 911],
              "reason": "state entries absent from expected_keys are not loaded"}]),
        "expert_conversion": conversion, "random_parameters_remaining": False,
        "constructed_modules": ["embedding", "dense layer0", "MoE layer1"],
        "full_model_constructed": False, "final_norm_constructed": False,
    }


def _capture_forward(role: str, prefix: Any, rotary: list[Any], ids: list[int]) -> dict[str, Any]:
    import torch

    saved: dict[str, Any] = {}
    counts: dict[str, int] = {}
    handles = []

    def capture(name: str, tuple_output: bool = False) -> Any:
        def hook(_module: Any, _args: Any, result: Any) -> None:
            counts[name] = counts.get(name, 0) + 1
            value = result[0] if tuple_output else result
            _require(isinstance(value, torch.Tensor), f"observed output missing: {name}")
            saved[name] = value.detach().contiguous().clone()
        return hook

    def gate_hook(_module: Any, _args: Any, result: Any) -> None:
        _require(isinstance(result, tuple) and len(result) == (3 if role == "native" else 2),
                 "native/reference router output contract differs")
        counts["layer1.router"] = counts.get("layer1.router", 0) + 1
        if role == "native":
            logits, weights, selected = result
            saved["layer1.router_logits"] = logits.detach().contiguous().clone()
        else:
            selected, weights = result
        saved["layer1.selected_experts"] = selected.detach().contiguous().clone()
        saved["layer1.router_weights"] = weights.detach().contiguous().clone()

    parameter_stamps = {name: (id(value), value._version)
                        for name, value in prefix.named_parameters()}
    try:
        for index, layer in enumerate(prefix.layers):
            for suffix, module, tuple_output in (
                ("attention_output", layer.self_attn, True),
                ("mlp_input", layer.post_attention_layernorm, False),
                ("mlp_output", layer.mlp, False),
            ):
                handles.append(module.register_forward_hook(
                    capture(f"layer{index}.{suffix}", tuple_output)))
        handles.append(prefix.layers[1].mlp.gate.register_forward_hook(gate_hook))
        with torch.inference_mode():
            input_ids = torch.tensor([ids], device="cpu", dtype=torch.int64)
            positions = torch.arange(64, device="cpu", dtype=torch.int64).unsqueeze(0)
            mask = torch.zeros((1, 1, 64, 64), device="cpu", dtype=torch.float32)
            future = torch.ones((64, 64), device="cpu", dtype=torch.bool).triu(1)
            mask.masked_fill_(future, torch.finfo(torch.float32).min)
            hidden = prefix.embed_tokens(input_ids)
            saved["embedding_output"] = hidden.detach().contiguous().clone()
            for index, layer in enumerate(prefix.layers):
                _require(not layer.training, "reference noaux_tc requires eval mode")
                kwargs = {"attention_mask": mask, "position_ids": positions, "use_cache": False}
                if role == "native":
                    kwargs["position_embeddings"] = rotary[index](hidden, positions)
                    hidden = layer(hidden, **kwargs)
                else:
                    result = layer(hidden, **kwargs)
                    _require(isinstance(result, tuple) and len(result) == 1,
                             "reference decoder output contract differs")
                    hidden = result[0]
                _check_output(hidden, f"{role} layer{index}")
                saved[f"layer{index}.output"] = hidden.detach().contiguous().clone()
            saved["input_ids"] = input_ids
            saved["position_ids"] = positions
            saved["causal_attention_mask"] = mask
    finally:
        for handle in handles:
            handle.remove()
    _require(len(counts) == 7 and all(value == 1 for value in counts.values()),
             "every observed dense/MoE/attention/router call must execute exactly once")
    _require(parameter_stamps == {name: (id(value), value._version)
                                  for name, value in prefix.named_parameters()},
             "forward mutated or replaced a loaded parameter")
    _check_routes(saved.get("layer1.selected_experts"), "observed selected experts")
    weights = saved.get("layer1.router_weights")
    _require(isinstance(weights, torch.Tensor) and weights.dtype == torch.float32
             and tuple(weights.shape) == (64, 6) and bool(torch.isfinite(weights).all())
             and bool((weights >= 0).all()), "router weights missing, nonfinite or negative")
    for name, value in saved.items():
        _require(value.device.type == "cpu" and not value.is_meta and value.numel() > 0
                 and bool(torch.isfinite(value).all()), f"empty/nonfinite retained output: {name}")
    return {"tensors": saved, "observed_calls": counts}


def _config(role: str, module: Any, raw: dict[str, Any]) -> Any:
    runtime = dict(raw)
    if role == "native":
        for name, value in _FIELDS.items():
            _require(type(runtime.get(name)) is type(value) and runtime[name] == value,
                     "source auxiliary fields differ")
            del runtime[name]
    config = module.DeepseekV3Config(**runtime)
    config._attn_implementation = "eager"
    if role == "native":
        config._experts_implementation = "eager"
        _require(not any(name in vars(config) or hasattr(config, name) for name in _FIELDS),
                 "native aux fields must be absent before model construction")
        _require(config.rope_interleave is True and config.head_dim == 64,
                 "native checkpoint interleaved rotary layout differs")
    else:
        _require(all(type(getattr(config, name, None)) is type(value)
                     and getattr(config, name) == value for name, value in _FIELDS.items()),
                 "reference must retain original auxiliary fields")
    return config


def _unchanged(descriptors: dict[str, Any]) -> None:
    for name, original in descriptors.items():
        observed = _descriptor(Path(original["path"]))
        _require(all(observed[key] == original.get(key) for key in ("path", "bytes", "sha256")),
                 f"input/source changed during verification: {name}")


def _worker(role: str, request_path: Path, output: Path) -> int:
    started = time.monotonic()
    report: dict[str, Any] = {"schema": SCHEMA, "role": role, "status": "FAIL",
                              "attempt": 1, "evidence_complete": False}
    provenance = None
    try:
        _require(os.environ.get("SOUP_MOONLIGHT_REFERENCE_WORKER") == role,
                 "private worker requires coordinator invocation")
        request = _json(request_path)
        _require(request.get("script_sha256") == _sha256(Path(__file__)),
                 "worker script differs from coordinator")
        _require(request.get("role") == role, "worker role mismatch")
        native_prefix = request.get("native_prefix")
        _require(isinstance(native_prefix, str), "native venv identity missing")
        same_environment = os.path.realpath(sys.prefix) == native_prefix
        _require(same_environment if role == "native" else not same_environment,
                 "reference code requires a venv distinct from the native coordinator")
        report["request"] = _descriptor(request_path)
        report["script"] = _descriptor(Path(__file__))
        aux = admit_aux_report(Path(request["aux_report"]), request["aux_report_sha256"])
        report["aux_admission"] = aux
        source = validate_sources(Path(request["source_dir"]))
        _require(source == request["source_identity"], "worker/coordinator source identity differs")
        report["source_identity"] = source
        input_path = Path(request["input_receipt"])
        _require(_sha256(input_path) == request["input_receipt_sha256"],
                 "worker input hash mismatch")
        inputs = _json(input_path)
        _require(_valid_ids(inputs.get("input_ids")) and _hex(
            inputs.get("training_pack_manifest_sha256")), "worker pack input evidence missing")
        _unchanged(inputs["files"])
        report["inputs"] = _descriptor(input_path)
        provenance = _Provenance(output, role)
        provenance.start()
        report["runtime"] = _runtime(role)
        if role == "native":
            admitted_sources = aux["source_provenance"].get("sources", {})
            admitted_packages = aux["source_provenance"].get("packages", {})
            native_expectations = {**COMMON_VERSIONS, **_TRANSFORMERS_DEPENDENCIES[role],
                                   "transformers": NATIVE_TRANSFORMERS}
            for package, expected_version in native_expectations.items():
                if package not in admitted_packages:
                    continue
                version = admitted_packages[package].get("version")
                _require(isinstance(version, str) and (version.split("+")[0] if package == "torch"
                         else version) == expected_version,
                         f"aux proof/native package version mismatch: {package}")
            for relative, expected_hash in _NATIVE_SOURCE_PINS.items():
                entry = admitted_sources.get(f"transformers/{relative}", {})
                _require(entry.get("sha256") == expected_hash
                         and entry.get("record_hash_matches") is True,
                         f"aux proof/native source identity mismatch: {relative}")
        module = (_native_module() if role == "native"
                  else _reference_module(Path(source["source_dir"])))
        config = _config(role, module, source["config"])
        _write_json(output / "runtime-config.json", config.to_dict())
        prefix, rotary, loading = _load_prefix(role, module, source, output, config)
        report["config_class"] = f"{type(config).__module__}.{type(config).__name__}"
        report["decoder_class"] = (
            f"{module.DeepseekV3DecoderLayer.__module__}.DeepseekV3DecoderLayer")
        report["loading"] = loading
        report["normalization"] = aux["normalization"] if role == "native" else []
        observed = _capture_forward(role, prefix, rotary, inputs["input_ids"])
        from safetensors.torch import save_file

        save_file(observed["tensors"], str(output / "outputs.safetensors"))
        report["outputs"] = _descriptor(output / "outputs.safetensors")
        report["observed_calls"] = observed["observed_calls"]
        report["tensor_metadata"] = {
            name: {"shape": list(value.shape), "dtype": str(value.dtype),
                   "finite": True, "device": str(value.device)}
            for name, value in observed["tensors"].items()}
        report["observed_tensor_statistics"] = {
            name: {"max_abs": value.double().abs().max().item(),
                   "mean": value.double().mean().item(),
                   "rms": value.double().square().mean().sqrt().item()}
            for name, value in observed["tensors"].items() if value.is_floating_point()}
        _unchanged(inputs["files"])
        _unchanged(source["files"])
        report.update(status="PASS", evidence_complete=True)
    except BaseException as error:
        report.update(status="FAIL", evidence_complete=False, error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
    finally:
        if provenance is not None:
            try:
                report["provenance"] = provenance.finish()
            except BaseException as error:
                report.update(status="FAIL", evidence_complete=False,
                              provenance_error=f"{type(error).__name__}: {error}",
                              provenance_traceback=traceback.format_exc())
        report["wall_seconds"] = time.monotonic() - started
        _write_json(output / "report.json", report)
    return 0 if report["status"] == "PASS" else 1


def _run_worker(
    role: str, executable: Path, request: dict[str, Any], output: Path,
) -> dict[str, Any]:
    output.mkdir()
    request_path = output / "request.json"
    _write_json(request_path, {**request, "role": role})
    environment = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        environment.pop(key, None)
    environment.update({"SOUP_MOONLIGHT_REFERENCE_WORKER": role, "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                        "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1"})
    # Do not realpath a venv interpreter symlink into the base interpreter.
    command = [os.path.abspath(executable), "-I", "-B", os.path.realpath(__file__),
               "--_worker", role, "--_request", os.path.realpath(request_path),
               "--output", os.path.realpath(output)]
    stdout, stderr, exit_code = b"", b"", None
    try:
        completed = subprocess.run(command, env=environment, cwd=output, capture_output=True,
                                   timeout=WORKER_SECONDS, check=False)
        stdout, stderr, exit_code = completed.stdout, completed.stderr, completed.returncode
    except subprocess.TimeoutExpired as error:
        stdout, stderr = error.stdout or b"", error.stderr or b""
        raise ValueError(
            f"{role} worker exceeded fixed {WORKER_SECONDS}s limit; no retry") from error
    finally:
        (output / "stdout.log").write_bytes(stdout)
        (output / "stderr.log").write_bytes(stderr)
        _write_json(output / "process.json", {"command": command, "exit_code": exit_code,
                                               "timeout_seconds": WORKER_SECONDS, "attempt": 1})
    path = output / "report.json"
    _require(path.is_file(), f"{role} worker emitted no evidence report")
    result = _json(path)
    _require(exit_code == 0 and result.get("status") == "PASS"
             and result.get("evidence_complete") is True,
             f"{role} worker failed; inspect {path}; no retry")
    artifact = result.get("outputs", {})
    _require(isinstance(artifact, dict) and isinstance(artifact.get("path"), str)
             and _contained(Path(artifact["path"]), output)
             and _descriptor(Path(artifact["path"])) == artifact,
             f"{role} worker output receipt mismatch")
    return result


def verify(*, source_dir: Path, reference_python: Path, output: Path, aux_report: Path,
           aux_report_sha256: str, pack_manifest_sha256: str, pack: Path | None = None,
           input_evidence: Path | None = None, length: int = 1024, window_index: int = 0,
           micro_batch_index: int = 0, token_offset: int = 0) -> dict[str, Any]:
    """Publish fresh, hash-bound evidence, with PASS only for both fixed conditions.

    Admission/execution failures produce report.json and raise ValueError. A valid
    numerical disagreement returns a FAIL report, retaining both outputs and raw
    residual tensors. The CLI returns nonzero for every non-PASS outcome.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {"schema": SCHEMA, "status": "FAIL", "evidence_complete": False,
                              "correctness_evidence_only": True,
                              "baseline_quality_evaluation": False,
                              "attempt": 1, "weights_downloaded_bytes": 0,
                              "model_id": MODEL_ID, "revision": REVISION, "profile_id": PROFILE_ID,
                              "script": _descriptor(Path(__file__))}
    error_to_raise = None
    started = time.monotonic()
    try:
        _require(sys.prefix != sys.base_prefix, "coordinator must run in the native venv")
        _require(importlib.metadata.version("transformers") == NATIVE_TRANSFORMERS,
                 f"coordinator requires transformers=={NATIVE_TRANSFORMERS}")
        _require(Path(reference_python).is_file(), "separate reference Python executable missing")
        aux = admit_aux_report(Path(aux_report), aux_report_sha256)
        report["aux_admission"] = aux
        inputs = load_input(pack=pack, input_evidence=input_evidence,
                            pack_manifest_sha256=pack_manifest_sha256, length=length,
                            window_index=window_index, micro_batch_index=micro_batch_index,
                            token_offset=token_offset)
        inputs_dir = output / "inputs"
        inputs_dir.mkdir()
        for name in ("manifest", "windows", "input_evidence"):
            if name in inputs["files"]:
                shutil.copyfile(inputs["files"][name]["path"], inputs_dir / f"{name}.json")
        input_receipt = output / "input-receipt.json"
        _write_json(input_receipt, inputs)
        report["inputs"] = inputs
        report["input_receipt"] = _descriptor(input_receipt)
        source = validate_sources(Path(source_dir))
        report["source_identity"] = source
        retained = output / "pinned-source"
        retained.mkdir()
        for name in (*_METADATA_FILES, *_REFERENCE_BLOBS):
            shutil.copyfile(source["files"][name]["path"], retained / name)
        for name in ("source-inventory-v1.json", "source-metadata-admission-v1.json",
                     "aux-admission-contract-v2.json"):
            shutil.copyfile(_RESULTS / name, retained / name)
        shutil.copyfile(aux["report"]["path"], retained / "aux-report.json")
        request = {"script_sha256": report["script"]["sha256"], "source_dir": source["source_dir"],
                   "source_identity": source, "input_receipt": os.path.realpath(input_receipt),
                   "input_receipt_sha256": _sha256(input_receipt),
                   "aux_report": aux["report"]["path"], "aux_report_sha256": aux_report_sha256,
                   "native_prefix": os.path.realpath(sys.prefix)}
        native = _run_worker("native", Path(sys.executable), request, output / "native")
        report["native"] = _descriptor(output / "native/report.json")
        reference = _run_worker("reference", Path(reference_python), request, output / "reference")
        report["reference"] = _descriptor(output / "reference/report.json")
        _require(native["runtime"]["prefix"] == os.path.realpath(sys.prefix)
                 and native["runtime"]["prefix"] != reference["runtime"]["prefix"]
                 and reference["runtime"]["prefix"] != reference["runtime"]["base_prefix"],
                 "native and reference must execute in separate isolated venvs")
        _require(native["runtime"]["torch"] == reference["runtime"]["torch"],
                 "native/reference torch builds must match exactly")
        _require(native["runtime"]["python"] == reference["runtime"]["python"],
                 "native/reference Python builds must match exactly")
        import torch
        from safetensors.torch import load_file, save_file

        native_tensors = load_file(native["outputs"]["path"], device="cpu")
        reference_tensors = load_file(reference["outputs"]["path"], device="cpu")
        for name in ("input_ids", "position_ids", "causal_attention_mask"):
            _require(name in native_tensors and name in reference_tensors
                     and torch.equal(native_tensors[name], reference_tensors[name]),
                     f"native/reference execution inputs differ: {name}")
        comparison = compare_outputs(
            native_tensors["layer1.output"], reference_tensors["layer1.output"],
            native_tensors["layer1.selected_experts"], reference_tensors["layer1.selected_experts"])
        report["comparison"] = comparison
        _write_json(output / "comparison.json", comparison)
        paired = sorted((native_tensors.keys() & reference_tensors.keys()) - {
            "layer1.selected_experts", "input_ids", "position_ids", "causal_attention_mask"})
        statistics, residuals = {}, {}
        for name in paired:
            first, second = native_tensors[name], reference_tensors[name]
            _require(first.shape == second.shape, f"retained output shape mismatch: {name}")
            if tuple(first.shape) == (1, 64, 2048):
                _check_output(first, name)
                _check_output(second, name)
                statistics[name] = _residual_statistics(first, second)
            residuals[name] = (first.double() - second.double()).contiguous()
        save_file(residuals, str(output / "residuals.safetensors"))
        _write_json(output / "residual-statistics.json", statistics)
        _unchanged(inputs["files"])
        _unchanged(source["files"])
        _require(_descriptor(Path(aux_report)) == aux["report"], "aux report changed during run")
        _require(_descriptor(Path(__file__)) == report["script"],
                 "verifier script changed during run")
        report.update(status=comparison["status"], evidence_complete=True,
                      accepted=comparison["status"] == "PASS", retry_performed=False)
    except BaseException as error:
        report.update(status="FAIL", accepted=False, evidence_complete=False,
                      error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc(), retry_performed=False)
        error_to_raise = error
    finally:
        report["wall_seconds"] = time.monotonic() - started
        files = {path.relative_to(output).as_posix(): _descriptor(path)
                 for path in sorted(output.rglob("*")) if path.is_file()}
        _write_json(output / "artifacts.json", {"schema": "p3-moonlight-reference-artifacts-v1",
                                                "files": files, "report_excluded": True})
        report["artifacts"] = _descriptor(output / "artifacts.json")
        _write_json(output / "report.json", report)
    if error_to_raise is not None:
        raise ValueError(
            f"Moonlight reference admission/execution failed: {output / 'report.json'}; "
            f"{type(error_to_raise).__name__}: {error_to_raise}") from error_to_raise
    return report


class _Parser(argparse.ArgumentParser):
    def _print_message(self, message: str, file: TextIO | None = None) -> None:
        if message:
            from rich.console import Console

            Console(file=file or sys.stderr, markup=False).print(message, end="", highlight=False)


def main(argv: list[str] | None = None) -> int:
    parser = _Parser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory")
    parser.add_argument("--_worker", choices=("native", "reference"), help=argparse.SUPPRESS)
    parser.add_argument("--_request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--reference-python", type=Path)
    parser.add_argument("--aux-report", type=Path)
    parser.add_argument("--aux-report-sha256")
    parser.add_argument("--pack-manifest-sha256")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--pack", type=Path)
    source.add_argument("--input-evidence", type=Path)
    parser.add_argument("--length", type=int, choices=(1024, 4096), default=1024)
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--micro-batch-index", type=int, default=0)
    parser.add_argument("--token-offset", type=int, default=0)
    args = parser.parse_args(argv)
    if args._worker is not None:
        _require(args._request is not None and args.output.is_dir(),
                 "private worker request missing")
        return _worker(args._worker, args._request, args.output)
    for name in ("source_dir", "reference_python", "aux_report", "aux_report_sha256",
                 "pack_manifest_sha256"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    if args.pack is None and args.input_evidence is None:
        parser.error("one of --pack or --input-evidence is required")
    from rich.console import Console

    console = Console(stderr=True, markup=False)
    try:
        result = verify(source_dir=args.source_dir, reference_python=args.reference_python,
                        output=args.output, aux_report=args.aux_report,
                        aux_report_sha256=args.aux_report_sha256,
                        pack_manifest_sha256=args.pack_manifest_sha256, pack=args.pack,
                        input_evidence=args.input_evidence, length=args.length,
                        window_index=args.window_index, micro_batch_index=args.micro_batch_index,
                        token_offset=args.token_offset)
    except (OSError, ValueError) as error:
        console.print(f"FAIL: {error}")
        return 1
    console.print(f"{result['status']}: {args.output / 'report.json'}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
