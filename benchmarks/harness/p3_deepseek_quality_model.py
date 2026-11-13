"""Native DeepSeek-family NF4 disk sessions and strict paired adapter identity.

Tiny checkpoints use the full checkpoint -> sharder -> streamed runtime path.
They do not qualify as either pinned production checkpoint.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import distribution, version
from pathlib import Path
from typing import Any, Iterator

from benchmarks.harness.p3_deepseek_quality_profiles import get_profile

_DEFAULT_PROFILE = get_profile("deepseek-v3")
MODEL_ID = _DEFAULT_PROFILE.model_id
REVISION = _DEFAULT_PROFILE.revision
SOURCE_BYTES = _DEFAULT_PROFILE.weight_bytes
SOURCE_PARAMETERS = _DEFAULT_PROFILE.parameters
TARGETS = _DEFAULT_PROFILE.lora_targets
_SOURCE_VERIFICATION = "hub_commit_etag_and_size_plus_cache_source_fingerprint"
SOURCE_INVENTORY_SHA256 = _DEFAULT_PROFILE.source_inventory_sha256
_FULL_SHAPE = dict(_DEFAULT_PROFILE.shape)
_AUX_CONTRACT = "benchmarks/results/p3-moonlight-quality/aux-admission-contract-v2.json"
_AUX_CONTRACT_SHA256 = "12e9a1f548d3192035f5781e0d568642480a94a383d48fb1f5373a8f2e178cd0"
_AUX_REPORT = "benchmarks/results/p3-moonlight-quality/prelaunch-review-v1/aux-report-v2.json"
_AUX_REPORT_SHA256 = "3b56a594a381822bd33c11cea01d6ff5b1d2ef9508522924f148b1431f62f27a"
_AUX_CONTINUITY = "benchmarks/results/p3-moonlight-quality/aux-source-continuity-v1.json"
_AUX_CONTINUITY_SHA256 = "055426bb633c04cd81933a4c6d3ad5b2bacbccba7d1e76cb6cde5c2349828999"
_SOURCE_AUX_FIELDS = {"aux_loss_alpha": 0.001, "seq_aux": True}
_NORMALIZATION_REASON = (
    "no numerical consumer in pinned native deepseek_v3; generic config validation read only"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hex(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _child(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("invalid relative artifact path")
    if Path(name).is_absolute() or any(part in {"", ".", ".."} for part in name.split("/")):
        raise ValueError("artifact path must remain inside its directory")
    parent, child = os.path.realpath(root), os.path.realpath(root / name)
    if os.path.commonpath((parent, child)) != parent:
        raise ValueError("artifact path escapes its directory")
    return Path(child)


def source_inventory(profile_id: str = "deepseek-v3") -> dict[str, dict[str, Any]]:
    profile = get_profile(profile_id)
    result_dir = ("p3-deepseek-quality" if profile_id == "deepseek-v3"
                  else "p3-moonlight-quality")
    path = (Path(__file__).parents[2] / "benchmarks/results"
            / result_dir / "source-inventory-v1.json")
    if sha256(path) != profile.source_inventory_sha256:
        raise ValueError("reviewed source inventory hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("model_id") != profile.model_id or value.get("revision") != profile.revision:
        raise ValueError("reviewed source inventory model/revision mismatch")
    return value["files"]


def _require_source_provenance(
    declared: dict[str, Any], *, profile_id: str = "deepseek-v3",
) -> None:
    profile = get_profile(profile_id)
    pinned = source_inventory() if profile_id == "deepseek-v3" else source_inventory(profile_id)
    if set(declared) != set(pinned):
        raise ValueError("source inventory differs from the reviewed Hub pin")
    for name, expected in pinned.items():
        observed = declared[name]
        weight = name.endswith(".safetensors")
        origin = ("hub_lfs_content_identity_not_locally_hashed" if weight
                  else "locally_measured_metadata")
        if (not isinstance(observed, dict)
                or any(observed.get(key) != value for key, value in expected.items())
                or observed.get("hub_commit") != profile.revision
                or observed.get("sha256_origin") != origin
                or not _hex(observed.get("sha256"))
                or (weight and observed["sha256"] != expected["hub_sha256"])):
            raise ValueError(f"source provenance differs from the reviewed Hub pin: {name}")


def _audit_aux_repository_sources(root: Path, sources: dict, continuity: dict) -> None:
    """Check the complete reviewed repository partition independently of installed packages."""
    unchanged = continuity["unchanged_sources"]
    reviewed_deltas = set(continuity["reviewed_wrapper_deltas"])
    repository_sources = {name.removeprefix("repository/") for name in sources
                          if name.startswith("repository/")}
    if (continuity["aux_report_sha256"] != _AUX_REPORT_SHA256
            or repository_sources != set(unchanged) | reviewed_deltas
            or set(unchanged) & reviewed_deltas):
        raise ValueError("Moonlight repository source continuity coverage differs")
    for relative in repository_sources - reviewed_deltas:
        expected = unchanged[relative]
        payload = _child(root, relative).read_bytes()
        normalized_sha = hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest()
        if (expected["source_sha256"] != sources[f"repository/{relative}"]["sha256"]
                or normalized_sha != expected["normalized_text_sha256"]):
            raise ValueError(f"unchanged audited repository source differs: {relative}")


def native_aux_admission(profile_id: str = "deepseek-v3") -> dict[str, Any] | None:
    """Bind the accepted proof and the installed native/PEFT execution sources."""
    get_profile(profile_id)
    if profile_id == "deepseek-v3":
        return None
    root = Path(__file__).parents[2]
    binding = {
        "contract": {"path": _AUX_CONTRACT, "sha256": _AUX_CONTRACT_SHA256},
        "evidence": {"path": _AUX_REPORT, "sha256": _AUX_REPORT_SHA256},
    }
    for item in binding.values():
        if sha256(_child(root, item["path"])) != item["sha256"]:
            raise ValueError("Moonlight native aux admission file changed")
    # The fixed evidence hash binds the complete FP32/BF16 equality and guard checks.
    report = json.loads((root / _AUX_REPORT).read_text(encoding="utf-8"))
    continuity_path = _child(root, _AUX_CONTINUITY)
    if sha256(continuity_path) != _AUX_CONTINUITY_SHA256:
        raise ValueError("Moonlight admitted source continuity manifest changed")
    continuity = json.loads(continuity_path.read_text(encoding="utf-8"))
    # Only the seven explicitly reviewed wrappers differ from historical proof.
    # Their current bytes are bound by the final packet and study protocol.
    packages = {}
    for name, evidence in report["packages"].items():
        expected = evidence["version"]
        if version(name) != expected:
            raise ValueError(f"Moonlight native aux proof requires {name}=={expected}")
        packages[name] = distribution(name)
    _audit_aux_repository_sources(root, report["sources"], continuity)
    for name, item in report["sources"].items():
        package = name.split("/", 1)[0]
        if package == "repository":
            continue
        if package in packages:
            installed = Path(packages[package].locate_file(name))
            if sha256(installed) != item["sha256"] or item.get("record_hash_matches") is not True:
                raise ValueError(f"installed source differs from admitted wheel RECORD: {name}")
        else:
            raise ValueError(f"unclassified aux-proof execution source: {name}")
    return binding


def _source_aux_metadata(
    config: dict[str, Any], *, profile_id: str = "deepseek-v3",
) -> dict[str, Any]:
    admission = native_aux_admission(profile_id)
    if admission is None:
        return {}
    present = _SOURCE_AUX_FIELDS.keys() & config.keys()
    if present and present != _SOURCE_AUX_FIELDS.keys():
        raise ValueError("Moonlight source has only part of the admitted aux field pair")
    changes = []
    for field_name, expected in _SOURCE_AUX_FIELDS.items():
        if field_name in config:
            value = config[field_name]
            if type(value) is not type(expected) or value != expected:
                raise ValueError(
                    f"Moonlight source aux value differs from admitted input: {field_name}")
            changes.append({
                "field": field_name, "source_value": value, "runtime": "absent",
                "reason": _NORMALIZATION_REASON,
            })
    return {"aux_admission": admission, "runtime_config_normalization": changes}


def _normalized_runtime_source(
    source: Path, config: dict[str, Any], metadata: dict[str, Any], *,
    profile_id: str = "deepseek-v3",
) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """Keep source bytes untouched and own normalized config until session close."""
    if profile_id == "deepseek-v3":
        return source, None
    normalized = dict(config)
    for change in metadata["runtime_config_normalization"]:
        del normalized[change["field"]]
    directory = tempfile.TemporaryDirectory(prefix="p3-moonlight-runtime-config-")
    try:
        path = Path(directory.name)
        with (path / "config.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(normalized, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        metadata["runtime_config_sha256"] = sha256(path / "config.json")
        return path, directory
    except BaseException:
        directory.cleanup()
        raise


def _moonlight_source_shapes(config: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Enumerate the pinned checkpoint layout, before native expert stacking."""
    if "quantization_config" in config or config.get("q_lora_rank") is not None:
        raise ValueError("Moonlight requires BF16 source weights and the single q_proj path")
    hidden, vocabulary = config["hidden_size"], config["vocab_size"]
    heads, kv_rank = config["num_attention_heads"], config["kv_lora_rank"]
    rope, nope = config["qk_rope_head_dim"], config["qk_nope_head_dim"]
    value_dim = config["v_head_dim"]
    shapes = {
        "model.embed_tokens.weight": (vocabulary, hidden),
        "model.norm.weight": (hidden,),
        "lm_head.weight": (vocabulary, hidden),
    }
    attention = {
        "q_proj.weight": (heads * (nope + rope), hidden),
        "kv_a_proj_with_mqa.weight": (kv_rank + rope, hidden),
        "kv_a_layernorm.weight": (kv_rank,),
        "kv_b_proj.weight": (heads * (nope + value_dim), kv_rank),
        "o_proj.weight": (hidden, heads * value_dim),
        # The pinned checkpoint stores full-width legacy buffers; runtime derives half-width.
        "rotary_emb.inv_freq": (rope,),
    }
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        shapes[prefix + "input_layernorm.weight"] = (hidden,)
        shapes[prefix + "post_attention_layernorm.weight"] = (hidden,)
        shapes.update({prefix + "self_attn." + name: shape for name, shape in attention.items()})
        if layer < config["first_k_dense_replace"]:
            projections = {prefix + "mlp.": config["intermediate_size"]}
        else:
            experts = config["n_routed_experts"]
            intermediate = config["moe_intermediate_size"]
            shapes[prefix + "mlp.gate.weight"] = (experts, hidden)
            shapes[prefix + "mlp.gate.e_score_correction_bias"] = (experts,)
            projections = {
                prefix + f"mlp.experts.{expert}.": intermediate for expert in range(experts)
            }
            projections[prefix + "mlp.shared_experts."] = (
                config["n_shared_experts"] * intermediate)
        for name, width in projections.items():
            shapes[name + "gate_proj.weight"] = (width, hidden)
            shapes[name + "up_proj.weight"] = (width, hidden)
            shapes[name + "down_proj.weight"] = (hidden, width)
    return shapes


def _validate_moonlight_index(
    config: dict[str, Any], weight_map: dict[str, str],
) -> dict[str, tuple[int, ...]]:
    shapes = _moonlight_source_shapes(config)
    if set(weight_map) != set(shapes):
        missing, unexpected = sorted(shapes.keys() - weight_map.keys()), sorted(
            weight_map.keys() - shapes.keys())
        raise ValueError(
            f"Moonlight source keys differ: missing={missing}, unexpected={unexpected}")
    return shapes


def _validate_moonlight_headers(
    source: Path, config: dict[str, Any], weight_map: dict[str, str],
) -> dict[str, Any]:
    from safetensors import safe_open

    shapes = _validate_moonlight_index(config, weight_map)
    per_file: dict[str, set[str]] = {}
    for name, shard in weight_map.items():
        per_file.setdefault(shard, set()).add(name)
    elements, rotary_elements, dtypes = 0, 0, Counter()
    for shard, expected_names in sorted(per_file.items()):
        with safe_open(str(_child(source, shard)), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != expected_names:
                raise ValueError(f"Moonlight shard header/index key coverage differs: {shard}")
            for name in sorted(expected_names):
                tensor = handle.get_slice(name)
                shape, dtype = tuple(tensor.get_shape()), tensor.get_dtype()
                rotary = name.endswith(".rotary_emb.inv_freq")
                allowed = {"BF16", "F32"} if rotary or name.endswith(
                    ".e_score_correction_bias") else {"BF16"}
                if shape != shapes[name] or dtype not in allowed:
                    raise ValueError(f"Moonlight tensor shape or BF16 source dtype differs: {name}")
                dtypes[dtype] += 1
                if rotary:
                    rotary_elements += math.prod(shape)
                else:
                    elements += math.prod(shape)
    if elements != get_profile("moonlight-16b-a3b").parameters:
        raise ValueError("Moonlight source tensor elements differ from the native cache contract")
    return {
        "tensor_count": len(shapes), "cached_tensor_elements": elements,
        "regenerated_rotary_elements": rotary_elements, "dtype_tensor_counts": dict(dtypes),
        "all_shard_headers_match_index": True,
        "rotary_policy": "native and reference constructors regenerate nonpersistent inv_freq",
    }


def _source_identity(
    source: Path, config: dict[str, Any], *, profile_id: str = "deepseek-v3",
) -> dict[str, Any]:
    profile = get_profile(profile_id)
    weights = sorted(source.glob("*.safetensors"))
    if not weights:
        raise ValueError("source contains no safetensors checkpoint")
    files = {path.name: {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
             for path in weights}
    result: dict[str, Any] = {
        "profile_id": profile_id,
        "path": os.path.realpath(source), "config_sha256": sha256(source / "config.json"),
        "files": files, "weight_files": len(weights),
        "weight_bytes": sum(item["bytes"] for item in files.values()),
        "shape": {name: config.get(name) for name in profile.shape},
        "model_id": None, "revision": None, "production_qualified": False,
        "source_verification": "source_stat_fingerprint_and_config_sha256",
    }
    result.update(_source_aux_metadata(config, profile_id=profile_id))
    index = source / "model.safetensors.index.json"
    if index.is_file():
        result["checkpoint_index_sha256"] = sha256(index)
    receipt_path = source / "checkpoint-identity.json"
    if not receipt_path.is_file():
        return result
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("checkpoint receipt must be a JSON object")
    if receipt.get("profile_id", "deepseek-v3") != profile_id:
        raise ValueError("checkpoint receipt profile differs")
    if receipt.get("model_id") != profile.model_id or receipt.get("revision") != profile.revision:
        raise ValueError("checkpoint receipt model_id or revision differs from production pin")
    declared = receipt.get("source_files")
    if not isinstance(declared, dict):
        raise ValueError("checkpoint receipt requires source_files")
    if (receipt.get("schema") != "p3-deepseek-quality-checkpoint-identity-v1"
            or receipt.get("source_verification") != _SOURCE_VERIFICATION
            or receipt.get("inventory_sha256") != profile.source_inventory_sha256
            or receipt.get("weight_content_sha256_measured_this_session") is not False):
        raise ValueError("checkpoint receipt schema/verification/inventory pin differs")
    _require_source_provenance(declared, profile_id=profile_id)
    for name, descriptor in declared.items():
        path = _child(source, name)
        if (not isinstance(descriptor, dict) or type(descriptor.get("bytes")) is not int
                or descriptor["bytes"] < 1 or not _hex(descriptor.get("sha256"))):
            raise ValueError("invalid checkpoint receipt file descriptor")
        if not path.is_file() or path.stat().st_size != descriptor["bytes"]:
            raise ValueError(f"checkpoint receipt size mismatch: {name}")
        if not name.endswith(".safetensors") and sha256(path) != descriptor["sha256"]:
            raise ValueError(f"checkpoint metadata content hash mismatch: {name}")
        if not name.endswith(".safetensors"):
            payload = path.read_bytes()
            content_identity = (hashlib.sha256(payload).hexdigest()
                                if descriptor["hub_sha256"] is not None else hashlib.sha1(
                                    b"blob " + str(len(payload)).encode("ascii") + b"\0"
                                    + payload).hexdigest())
            if content_identity != descriptor["hub_etag"]:
                raise ValueError(f"checkpoint metadata differs from reviewed Hub blob: {name}")
    expected_weights = profile.weight_files
    required = expected_weights | {"config.json", "model.safetensors.index.json"}
    if not required.issubset(declared) or set(files) != expected_weights:
        raise ValueError(
            "production checkpoint receipt requires exactly the profile's weight shards")
    if result["weight_bytes"] != profile.weight_bytes or result["shape"] != dict(profile.shape):
        raise ValueError("production checkpoint dimensions or total bytes differ from profile")
    checkpoint_index = json.loads(index.read_text(encoding="utf-8"))
    weight_map = checkpoint_index.get("weight_map", {})
    if not isinstance(weight_map, dict) or set(weight_map.values()) != expected_weights:
        raise ValueError("production checkpoint index does not cover all pinned source shards")
    if profile_id == "moonlight-16b-a3b":
        for key in ("aux_admission", "runtime_config_normalization"):
            if receipt.get(key) != result[key]:
                raise ValueError("Moonlight source receipt lacks admitted aux normalization")
        result["tensor_header_admission"] = _validate_moonlight_headers(
            source, config, weight_map)
    result.update(
        model_id=profile.model_id, revision=profile.revision, production_qualified=True,
        receipt_sha256=sha256(receipt_path), expected_source_files=declared,
        source_verification=_SOURCE_VERIFICATION,
        inventory_sha256=profile.source_inventory_sha256,
        weight_content_sha256_measured_this_session=False,
    )
    return result


def require_production_source(
    source_metadata: dict[str, Any], *, profile_id: str = "deepseek-v3",
) -> None:
    """Refuse a tiny, truncated, foreign-profile or unpinned production session."""
    profile = get_profile(profile_id)
    if source_metadata.get("profile_id", "deepseek-v3") != profile_id:
        raise ValueError("production source profile differs")
    if (source_metadata.get("production_qualified") is not True
            or source_metadata.get("model_id") != profile.model_id
            or source_metadata.get("revision") != profile.revision
            or source_metadata.get("shape") != dict(profile.shape)
            or source_metadata.get("weight_files") != len(profile.weight_files)
            or source_metadata.get("weight_bytes") != profile.weight_bytes
            or source_metadata.get("source_verification") != _SOURCE_VERIFICATION
            or not _hex(source_metadata.get("receipt_sha256"))):
        raise ValueError("production training requires the pinned full profile source receipt")
    expected = profile.weight_files
    files = source_metadata.get("files", {})
    declared = source_metadata.get("expected_source_files", {})
    if (not isinstance(files, dict) or not isinstance(declared, dict)
            or set(files) != expected or not expected.issubset(declared)
            or not {"config.json", "model.safetensors.index.json"}.issubset(declared)
            or not _hex(source_metadata.get("config_sha256"))
            or not _hex(source_metadata.get("checkpoint_index_sha256"))
            or not _hex(source_metadata.get("cache_source_fingerprint"))):
        raise ValueError("production source identity lacks complete checkpoint/cache inventory")
    if source_metadata.get("inventory_sha256") != profile.source_inventory_sha256:
        raise ValueError("production source inventory pin differs")
    _require_source_provenance(declared, profile_id=profile_id)
    for name in expected:
        descriptor = files[name]
        provenance = declared[name]
        if (not isinstance(descriptor, dict) or not isinstance(provenance, dict)
                or type(descriptor.get("bytes")) is not int or descriptor["bytes"] < 1
                or descriptor["bytes"] != provenance.get("bytes")
                or not _hex(provenance.get("sha256"))):
            raise ValueError("production checkpoint inventory/provenance differs")
    if sum(item["bytes"] for item in files.values()) != profile.weight_bytes:
        raise ValueError("production checkpoint inventory total differs")
    if profile_id == "moonlight-16b-a3b":
        expected_aux = _source_aux_metadata(_SOURCE_AUX_FIELDS, profile_id=profile_id)
        if any(source_metadata.get(key) != value for key, value in expected_aux.items()):
            raise ValueError("Moonlight source aux admission or normalization receipt differs")


@contextmanager
def _meta_construction() -> Iterator[None]:
    import torch

    previous = torch.utils.deterministic.fill_uninitialized_memory
    try:
        torch.utils.deterministic.fill_uninitialized_memory = False
        yield
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = previous


@dataclass
class ModelSession:
    model: Any
    runtime: Any
    config: Any
    metadata: dict[str, Any]
    closed: bool = False
    retained_resources: list[Any] = field(default_factory=list, repr=False)
    runtime_config_directory: tempfile.TemporaryDirectory | None = field(default=None, repr=False)

    def close(self) -> None:
        if self.retained_resources:
            raise RuntimeError(
                "window cleanup retained live leases; runtime cannot close underneath")
        if not self.closed:
            self.runtime.close()
            if self.runtime_config_directory is not None:
                self.runtime_config_directory.cleanup()
            self.closed = True

    def __enter__(self) -> ModelSession:
        if self.closed:
            raise RuntimeError("model session is closed")
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def trainable_parameters(model: Any) -> dict[str, Any]:
    """Exact canonical names used in raw adapter and gradient safetensors files."""
    from soup_cli.utils.layer_stream_runtime import canonical_named_parameters

    result = {}
    for name, value in canonical_named_parameters(model):
        if value.requires_grad:
            if name in result:
                raise ValueError(f"duplicate canonical trainable name: {name}")
            result[name] = value
    if not result:
        raise ValueError("no trainable attention adapters")
    return dict(sorted(result.items()))


def _parameter_guard(
    model: Any, layers: int, *, profile_id: str = "deepseek-v3",
) -> dict[str, Any]:
    import torch

    params = trainable_parameters(model)
    expected = {(layer, target, factor) for layer in range(layers)
                for target in get_profile(profile_id).lora_targets for factor in ("A", "B")}
    observed = set()
    for name, value in params.items():
        match = re.search(
            r"\.layers\.(\d+)\.self_attn\.([^.]+)\.lora_([AB])\.default\.weight$", name,
        )
        if match is None or value.is_meta or value.dtype != torch.float32:
            raise ValueError(f"unexpected trainable attention adapter: {name}")
        layer, target, factor = int(match[1]), match[2], match[3]
        if value.ndim != 2 or value.shape[0 if factor == "A" else 1] != 2:
            raise ValueError(f"adapter rank differs from two: {name}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"nonfinite adapter: {name}")
        observed.add((layer, target, factor))
    if observed != expected or len(params) != len(expected):
        raise ValueError("every decoder layer requires exactly the profile's rank-two targets")
    return params


def open_session(
    source: Path, cache: Path, *, device: str = "cuda", dtype: str = "bfloat16",
    prepare: bool = False, profile_id: str = "deepseek-v3",
) -> ModelSession:
    """Build native sessions while preserving the original checkpoint fingerprint.

    Both skeletons use ``trust_remote_code=False``. Only admitted Moonlight
    sessions remove the two numerically unused aux fields, in a separate config
    directory before construction. DeepSeek retains its original config.
    """
    import torch
    from peft import LoraConfig, TaskType

    from soup_cli.utils.layer_shard import (
        checkpoint_source_components,
        fingerprint_source_files,
        inspect_shard_cache,
        shard_checkpoint,
        source_identity_includes_config,
    )
    from soup_cli.utils.layer_stream_runtime import (
        build_meta_skeleton,
        build_streamed_model,
        quantised_layer_suffixes,
        shard_layout_of,
    )
    from soup_cli.utils.peft_wiring import _moe_targets

    profile = get_profile(profile_id)
    source, cache = Path(os.path.realpath(source)), Path(os.path.realpath(cache))
    if dtype not in {"bfloat16", "float32"} or device not in {"cpu", "cuda"}:
        raise ValueError("research sessions require CPU/CUDA and BF16/FP32 compute")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA session requested without an available CUDA device")
    try:
        cache_inside_source = os.path.commonpath((str(source), str(cache))) == str(source)
    except ValueError:
        cache_inside_source = False
    if cache_inside_source:
        raise ValueError("cache must be separate from the immutable source directory")
    config_json = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config_json.get("model_type") != "deepseek_v3":
        raise ValueError("only native DeepSeek-V3 is admitted")
    if config_json.get("attention_dropout", 0.0) != 0:
        raise ValueError(
            "quality sessions require zero attention dropout for capture without draws")
    source_identity = _source_identity(source, config_json, profile_id=profile_id)
    runtime_source, runtime_directory = _normalized_runtime_source(
        source, config_json, source_identity, profile_id=profile_id)
    runtime = None
    try:
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices), _meta_construction():
            torch.manual_seed(23)
            skeleton = build_meta_skeleton(
                str(runtime_source), dtype=dtype, quant="nf4", double_quant=False)
            if set(_moe_targets("deepseek_v3", skeleton)) != set(profile.lora_targets):
                raise ValueError("native attention targets differ from the frozen profile")
            if profile_id == "moonlight-16b-a3b" and any(
                name in vars(skeleton.config) for name in _SOURCE_AUX_FIELDS
            ):
                raise ValueError("native skeleton reintroduced normalized aux fields")
            suffixes, layout = quantised_layer_suffixes(skeleton), shard_layout_of(skeleton)
            del skeleton
            gc.collect()
            components = tuple((name, item["bytes"], item["mtime_ns"])
                               for name, item in source_identity["files"].items())
            components = checkpoint_source_components(
                str(source), components,
                include_config=source_identity_includes_config(str(source), "deepseek_v3"),
            )
            fingerprint = fingerprint_source_files(components)
            if not (cache / "index.json").is_file():
                if not prepare:
                    raise ValueError(
                        "valid cache is required; use prepare=True to build a missing cache")
                if cache.exists() and any(cache.iterdir()):
                    raise ValueError(
                        "incomplete nonempty cache cannot be replaced or silently resumed")
                shard_checkpoint(
                    str(source), str(cache), dtype=dtype, arch="deepseek_v3", layout=layout,
                    quant="nf4", quant_suffixes=suffixes, double_quant=False,
                    quant_device=device, expert_layout="per_expert",
                )
            index, reason = inspect_shard_cache(
                str(cache), dtype, fingerprint, components, "nf4", False, device,
                layout=layout, expert_layout="per_expert",
            )
            if index is None:
                raise ValueError(f"cache validation failed: {reason}")
            specs = (*index.quant_specs.values(), *index.expert_quant_specs.values())
            if (not index.quant_specs or not index.expert_quant_specs
                    or any(spec.quant_type != "nf4" or spec.blocksize != 64 or spec.nested
                           for spec in specs)):
                raise ValueError("cache requires block64 NF4 without double quantization")
            if source_identity["production_qualified"] and index.total_params != profile.parameters:
                raise ValueError("full production source parameter count differs from profile")
            lora = LoraConfig(
                r=2, lora_alpha=4, lora_dropout=0.0, bias="none",
                target_modules=list(profile.lora_targets), task_type=TaskType.CAUSAL_LM,
            )
            model, runtime = build_streamed_model(
                model_id=str(runtime_source), shard_dir=str(cache), index=index, lora_config=lora,
                device=device, dtype=dtype, buffers=2, pin=False, seed=23, quant="nf4",
                double_quant=False, tier="disk", read_ahead=1, trust_remote_code=False,
            )
            if profile_id == "moonlight-16b-a3b" and any(
                name in vars(model.config) for name in _SOURCE_AUX_FIELDS
            ):
                raise ValueError("streamed model reintroduced normalized aux fields")
            for value in trainable_parameters(model).values():
                value.data = value.data.to(dtype=torch.float32)
            model.set_attn_implementation("sdpa")
            if getattr(model.config, "_attn_implementation", None) != "sdpa":
                raise ValueError("native DeepSeek SDPA attention was not installed")
            model.config.use_cache = False
            params = _parameter_guard(
                model, int(model.config.num_hidden_layers), profile_id=profile_id)
            model._p3_quality_initial_parameter_sha256 = _canonical_initial_digest(params)
            if runtime.expert_pool is None or runtime.expert_pool.dtype != dtype:
                raise ValueError("expected per-expert streaming pool at requested compute dtype")
            if runtime.stats()["tier"] != "disk":
                raise ValueError("quality sessions require disk-tier weights")
            model.train()
            source_identity["cache_source_fingerprint"] = fingerprint
            metadata = {
                "profile_id": profile_id, "source": source_identity,
                "cache": {
                    "profile_id": profile_id,
                    "path": str(cache), "index_sha256": sha256(cache / "index.json"),
                    "source_fingerprint": index.source_fingerprint, "dtype": index.dtype,
                    "quant": index.quant, "double_quant": index.double_quant,
                    "quant_device": index.quant_device, "expert_layout": index.expert_layout,
                    "source_parameters": index.total_params, "n_layers": index.n_layers,
                    "tier": "disk", "buffers": 2, "read_ahead": 1,
                },
                "precision": {
                    "compute_dtype": dtype, "trainable_dtype": "float32", "autocast": False,
                    "attention_implementation": "sdpa", "trainable_tensors": len(params),
                    "trainable_elements": sum(value.numel() for value in params.values()),
                    "base_parameter_dtype_counts": dict(Counter(
                        str(value.dtype) for value in model.parameters() if not value.requires_grad
                    )),
                    "lora": {"rank": 2, "alpha": 4, "dropout": 0.0,
                             "targets": list(profile.lora_targets)},
                },
            }
            return ModelSession(
                model, runtime, model.config, metadata,
                runtime_config_directory=runtime_directory)
    except BaseException:
        try:
            if runtime is not None:
                runtime.close()
        finally:
            if runtime_directory is not None:
                runtime_directory.cleanup()
        raise


def adapter_digest(parameters: dict[str, Any]) -> tuple[str, dict[str, dict[str, Any]]]:
    import torch

    combined = hashlib.sha256()
    tensors = {}
    for name, parameter in sorted(parameters.items()):
        value = parameter.detach().cpu().contiguous()
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        tensors[name] = {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": digest}
        combined.update(json.dumps([name, list(value.shape), str(value.dtype), digest]).encode())
    return combined.hexdigest(), tensors


def _load_adapter_tensors(model: Any, path: Path) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    params = trainable_parameters(model)
    loaded = load_file(str(path), device="cpu")
    if loaded.keys() != params.keys():
        raise ValueError("adapter names do not exactly match all trainable parameters")
    for name, value in loaded.items():
        if (value.dtype != torch.float32 or params[name].dtype != torch.float32
                or value.shape != params[name].shape or not bool(torch.isfinite(value).all())):
            raise ValueError(f"invalid FP32 adapter shape/dtype/value: {name}")
    return loaded


def _canonical_initial_digest(parameters: dict[str, Any]) -> str:
    import torch

    generator = torch.Generator().manual_seed(23)
    canonical = {}
    for name, parameter in sorted(parameters.items()):
        canonical[name] = (
            torch.randn(parameter.shape, generator=generator) * 0.02
            if "lora_B" in name else parameter.detach().cpu().contiguous()
        )
    return adapter_digest(canonical)[0]


def _adapter_metadata(path: Path) -> dict[str, str]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        return handle.metadata() or {}


def load_initial_adapter(
    model: Any, path: Path, *, profile_id: str = "deepseek-v3",
) -> str:
    """Admit canonical seed-23 initializer metadata and all tensors before any copy."""
    import torch

    path = Path(path)
    before = sha256(path)
    metadata = _adapter_metadata(path)
    get_profile(profile_id)
    if metadata.get("profile_id", "deepseek-v3") != profile_id:
        raise ValueError("initial adapter profile differs")
    required = {
        "schema": "p3-deepseek-quality-initial-v1", "seed": "23",
        "nonzero_B_scale": "0.02",
        "initializer": "benchmarks.harness.l2l_schedule.make_non_vacuous",
    }
    if any(metadata.get(name) != value for name, value in required.items()):
        raise ValueError("initial adapter schema/seed/scale/initializer metadata differs")
    loaded = _load_adapter_tensors(model, path)
    digest = adapter_digest(loaded)[0]
    canonical = getattr(model, "_p3_quality_initial_parameter_sha256", None)
    if (not _hex(metadata.get("parameter_sha256")) or metadata["parameter_sha256"] != digest
            or not _hex(canonical) or digest != canonical):
        raise ValueError(
            "initial adapter parameter digest differs from canonical seeded initializer"
        )
    if sha256(path) != before:
        raise ValueError("adapter file changed during admission")
    with torch.no_grad():
        for name, value in trainable_parameters(model).items():
            value.copy_(loaded[name])
    return before


def load_checkpoint_adapter(
    model: Any, path: Path, *, profile_id: str = "deepseek-v3",
) -> str:
    """Restore a trained checkpoint only after its caller admits the checkpoint manifest."""
    import torch

    path = Path(path)
    before = sha256(path)
    metadata = _adapter_metadata(path)
    get_profile(profile_id)
    if metadata.get("profile_id", "deepseek-v3") != profile_id:
        raise ValueError("checkpoint adapter profile differs")
    if metadata.get("schema") != "quality-checkpoint-adapter-v1":
        raise ValueError("checkpoint adapter schema differs")
    loaded = _load_adapter_tensors(model, path)
    if sha256(path) != before:
        raise ValueError("checkpoint adapter changed during admission")
    with torch.no_grad():
        for name, value in trainable_parameters(model).items():
            value.copy_(loaded[name])
    return before


def write_initial_adapter(
    model: Any, path: Path, *, seed: int = 23, profile_id: str = "deepseek-v3",
) -> dict[str, Any]:
    """Write one exclusive FP32 snapshot with the frozen seeded nonzero-B initializer."""
    from safetensors.torch import save

    from benchmarks.harness.l2l_schedule import make_non_vacuous

    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if type(seed) is not int or seed != 23:
        raise ValueError("the paired study initial adapter requires seed 23")
    params = _parameter_guard(model, int(model.config.num_hidden_layers), profile_id=profile_id)
    make_non_vacuous(model, seed=seed, scale=0.02)
    tensors = {name: value.detach().cpu().contiguous() for name, value in params.items()}
    digest, identity = adapter_digest(tensors)
    canonical = getattr(model, "_p3_quality_initial_parameter_sha256", None)
    if digest != canonical:
        raise ValueError("initial adapter A factors differ from the canonical fresh session")
    if any(not math.isfinite(float(value.float().norm())) for value in tensors.values()):
        raise ValueError("nonfinite initial adapter")
    payload = save(tensors, metadata={
        "schema": "p3-deepseek-quality-initial-v1", "seed": str(seed),
        "profile_id": profile_id,
        "nonzero_B_scale": "0.02", "parameter_sha256": digest,
        "initializer": "benchmarks.harness.l2l_schedule.make_non_vacuous",
    })
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return {"profile_id": profile_id, "sha256": sha256(path),
            "parameter_sha256": digest, "parameters": identity,
            "seed": seed, "nonzero_B_scale": 0.02}
