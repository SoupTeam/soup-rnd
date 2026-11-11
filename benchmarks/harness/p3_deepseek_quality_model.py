"""Shared native DeepSeek-V3 NF4 disk session and strict paired adapter identity.

Tiny checkpoints deliberately use the full checkpoint -> sharder -> streamed runtime path.
Opening a research session does not qualify it as the pinned 671B production checkpoint.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

MODEL_ID = "deepseek-ai/DeepSeek-V3"
REVISION = "e815299b0bcbac849fa540c768ef21845365c9eb"
SOURCE_BYTES = 688586727753
SOURCE_PARAMETERS = 671026419200
TARGETS = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")
_SOURCE_VERIFICATION = "hub_commit_etag_and_size_plus_cache_source_fingerprint"
SOURCE_INVENTORY_SHA256 = "5677f32a218d96414334a1536d66a93d0bb87ac98786585faef41329f7ae7e1c"
_FULL_SHAPE = {
    "model_type": "deepseek_v3", "hidden_size": 7168, "num_hidden_layers": 61,
    "n_routed_experts": 256, "num_experts_per_tok": 8, "first_k_dense_replace": 3,
    "vocab_size": 129280, "q_lora_rank": 1536, "kv_lora_rank": 512,
}


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


def source_inventory() -> dict[str, dict[str, Any]]:
    path = Path(__file__).parents[2] / (
        "benchmarks/results/p3-deepseek-quality/source-inventory-v1.json")
    if sha256(path) != SOURCE_INVENTORY_SHA256:
        raise ValueError("reviewed source inventory hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("model_id") != MODEL_ID or value.get("revision") != REVISION:
        raise ValueError("reviewed source inventory model/revision mismatch")
    return value["files"]


def _require_source_provenance(declared: dict[str, Any]) -> None:
    pinned = source_inventory()
    if set(declared) != set(pinned):
        raise ValueError("source inventory differs from the reviewed Hub pin")
    for name, expected in pinned.items():
        observed = declared[name]
        weight = name.endswith(".safetensors")
        origin = ("hub_lfs_content_identity_not_locally_hashed" if weight
                  else "locally_measured_metadata")
        if (not isinstance(observed, dict)
                or any(observed.get(key) != value for key, value in expected.items())
                or observed.get("hub_commit") != REVISION
                or observed.get("sha256_origin") != origin
                or not _hex(observed.get("sha256"))
                or (weight and observed["sha256"] != expected["hub_sha256"])):
            raise ValueError(f"source provenance differs from the reviewed Hub pin: {name}")


def _source_identity(source: Path, config: dict[str, Any]) -> dict[str, Any]:
    weights = sorted(source.glob("*.safetensors"))
    if not weights:
        raise ValueError("source contains no safetensors checkpoint")
    files = {path.name: {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
             for path in weights}
    result: dict[str, Any] = {
        "path": os.path.realpath(source), "config_sha256": sha256(source / "config.json"),
        "files": files, "weight_files": len(weights),
        "weight_bytes": sum(item["bytes"] for item in files.values()),
        "shape": {name: config.get(name) for name in _FULL_SHAPE},
        "model_id": None, "revision": None, "production_qualified": False,
        "source_verification": "source_stat_fingerprint_and_config_sha256",
    }
    index = source / "model.safetensors.index.json"
    if index.is_file():
        result["checkpoint_index_sha256"] = sha256(index)
    receipt_path = source / "checkpoint-identity.json"
    if not receipt_path.is_file():
        return result
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("checkpoint receipt must be a JSON object")
    if receipt.get("model_id") != MODEL_ID or receipt.get("revision") != REVISION:
        raise ValueError("checkpoint receipt model_id or revision differs from production pin")
    declared = receipt.get("source_files")
    if not isinstance(declared, dict):
        raise ValueError("checkpoint receipt requires source_files")
    if (receipt.get("schema") != "p3-deepseek-quality-checkpoint-identity-v1"
            or receipt.get("source_verification") != _SOURCE_VERIFICATION
            or receipt.get("inventory_sha256") != SOURCE_INVENTORY_SHA256
            or receipt.get("weight_content_sha256_measured_this_session") is not False):
        raise ValueError("checkpoint receipt schema/verification/inventory pin differs")
    _require_source_provenance(declared)
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
    expected_weights = {f"model-{number:05d}-of-000163.safetensors"
                        for number in range(1, 164)}
    required = expected_weights | {"config.json", "model.safetensors.index.json"}
    if not required.issubset(declared) or set(files) != expected_weights:
        raise ValueError("production checkpoint receipt requires exactly 163 source weight shards")
    if result["weight_bytes"] != SOURCE_BYTES or result["shape"] != _FULL_SHAPE:
        raise ValueError("production checkpoint dimensions or total bytes differ from pinned V3")
    checkpoint_index = json.loads(index.read_text(encoding="utf-8"))
    weight_map = checkpoint_index.get("weight_map", {})
    if not isinstance(weight_map, dict) or set(weight_map.values()) != expected_weights:
        raise ValueError("production checkpoint index does not cover all pinned source shards")
    result.update(
        model_id=MODEL_ID, revision=REVISION, production_qualified=True,
        receipt_sha256=sha256(receipt_path), expected_source_files=declared,
        source_verification=_SOURCE_VERIFICATION,
        inventory_sha256=SOURCE_INVENTORY_SHA256,
        weight_content_sha256_measured_this_session=False,
    )
    return result


def require_production_source(source_metadata: dict[str, Any]) -> None:
    """Refuse a tiny, truncated or unpinned session at a production CLI boundary."""
    if (source_metadata.get("production_qualified") is not True
            or source_metadata.get("model_id") != MODEL_ID
            or source_metadata.get("revision") != REVISION
            or source_metadata.get("shape") != _FULL_SHAPE
            or source_metadata.get("weight_files") != 163
            or source_metadata.get("weight_bytes") != SOURCE_BYTES
            or source_metadata.get("source_verification") != _SOURCE_VERIFICATION
            or not _hex(source_metadata.get("receipt_sha256"))):
        raise ValueError("production training requires the pinned full DeepSeek-V3 source receipt")
    expected = {f"model-{number:05d}-of-000163.safetensors" for number in range(1, 164)}
    files = source_metadata.get("files", {})
    declared = source_metadata.get("expected_source_files", {})
    if (not isinstance(files, dict) or not isinstance(declared, dict)
            or set(files) != expected or not expected.issubset(declared)
            or not {"config.json", "model.safetensors.index.json"}.issubset(declared)
            or not _hex(source_metadata.get("config_sha256"))
            or not _hex(source_metadata.get("checkpoint_index_sha256"))
            or not _hex(source_metadata.get("cache_source_fingerprint"))):
        raise ValueError("production source identity lacks complete checkpoint/cache inventory")
    if source_metadata.get("inventory_sha256") != SOURCE_INVENTORY_SHA256:
        raise ValueError("production source inventory pin differs")
    _require_source_provenance(declared)
    for name in expected:
        descriptor = files[name]
        provenance = declared[name]
        if (not isinstance(descriptor, dict) or not isinstance(provenance, dict)
                or type(descriptor.get("bytes")) is not int or descriptor["bytes"] < 1
                or descriptor["bytes"] != provenance.get("bytes")
                or not _hex(provenance.get("sha256"))):
            raise ValueError("production checkpoint inventory/provenance differs")
    if sum(item["bytes"] for item in files.values()) != SOURCE_BYTES:
        raise ValueError("production checkpoint inventory total differs")


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

    def close(self) -> None:
        if self.retained_resources:
            raise RuntimeError(
                "window cleanup retained live leases; runtime cannot close underneath")
        if not self.closed:
            self.runtime.close()
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


def _parameter_guard(model: Any, layers: int) -> dict[str, Any]:
    import torch

    params = trainable_parameters(model)
    expected = {(layer, target, factor) for layer in range(layers)
                for target in TARGETS for factor in ("A", "B")}
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
        raise ValueError("every decoder layer requires exactly the five rank-two attention targets")
    return params


def open_session(
    source: Path, cache: Path, *, device: str = "cuda", dtype: str = "bfloat16",
    prepare: bool = False,
) -> ModelSession:
    """Build missing caches from the unchanged source config and native Transformers classes.

    Both skeleton constructions use ``trust_remote_code=False``. A checkpoint's
    ``auto_map`` is provenance, not permission to load its Hub configuration/model code.
    Source-declared auxiliary coefficients remain on the native config, so the frozen
    Oracle can refuse them rather than silently changing either arm's objective.
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
    from soup_cli.utils.peft_wiring import MOE_TEXT_LORA_TARGETS

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
    source_identity = _source_identity(source, config_json)
    if set(MOE_TEXT_LORA_TARGETS["deepseek_v3"]) != set(TARGETS):
        raise ValueError("production DeepSeek attention targets differ from the frozen study")
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    runtime = None
    with torch.random.fork_rng(devices=devices), _meta_construction():
        torch.manual_seed(23)
        skeleton = build_meta_skeleton(str(source), dtype=dtype, quant="nf4", double_quant=False)
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
                raise ValueError("incomplete nonempty cache cannot be replaced or silently resumed")
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
        if source_identity["production_qualified"] and index.total_params != SOURCE_PARAMETERS:
            raise ValueError(
                "full production source parameter count differs from pinned checkpoint")
        lora = LoraConfig(
            r=2, lora_alpha=4, lora_dropout=0.0, bias="none", target_modules=list(TARGETS),
            task_type=TaskType.CAUSAL_LM,
        )
        model, runtime = build_streamed_model(
            model_id=str(source), shard_dir=str(cache), index=index, lora_config=lora,
            device=device, dtype=dtype, buffers=2, pin=False, seed=23, quant="nf4",
            double_quant=False, tier="disk", read_ahead=1, trust_remote_code=False,
        )
        try:
            for value in trainable_parameters(model).values():
                value.data = value.data.to(dtype=torch.float32)
            model.set_attn_implementation("sdpa")
            if getattr(model.config, "_attn_implementation", None) != "sdpa":
                raise ValueError("native DeepSeek SDPA attention was not installed")
            model.config.use_cache = False
            params = _parameter_guard(model, int(model.config.num_hidden_layers))
            model._p3_quality_initial_parameter_sha256 = _canonical_initial_digest(params)
            if runtime.expert_pool is None or runtime.expert_pool.dtype != dtype:
                raise ValueError("expected per-expert streaming pool at requested compute dtype")
            if runtime.stats()["tier"] != "disk":
                raise ValueError("quality sessions require disk-tier weights")
            model.train()
            source_identity["cache_source_fingerprint"] = fingerprint
            metadata = {
                "source": source_identity,
                "cache": {
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
                    "lora": {"rank": 2, "alpha": 4, "dropout": 0.0, "targets": list(TARGETS)},
                },
            }
            return ModelSession(model, runtime, model.config, metadata)
        except BaseException:
            runtime.close()
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


def load_initial_adapter(model: Any, path: Path) -> str:
    """Admit canonical seed-23 initializer metadata and all tensors before any copy."""
    import torch

    path = Path(path)
    before = sha256(path)
    metadata = _adapter_metadata(path)
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


def load_checkpoint_adapter(model: Any, path: Path) -> str:
    """Restore a trained checkpoint only after its caller admits the checkpoint manifest."""
    import torch

    path = Path(path)
    before = sha256(path)
    if _adapter_metadata(path).get("schema") != "quality-checkpoint-adapter-v1":
        raise ValueError("checkpoint adapter schema differs")
    loaded = _load_adapter_tensors(model, path)
    if sha256(path) != before:
        raise ValueError("checkpoint adapter changed during admission")
    with torch.no_grad():
        for name, value in trainable_parameters(model).items():
            value.copy_(loaded[name])
    return before


def write_initial_adapter(model: Any, path: Path, *, seed: int = 23) -> dict[str, Any]:
    """Write one exclusive FP32 snapshot with the frozen seeded nonzero-B initializer."""
    from safetensors.torch import save

    from benchmarks.harness.l2l_schedule import make_non_vacuous

    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if type(seed) is not int or seed != 23:
        raise ValueError("the paired study initial adapter requires seed 23")
    params = _parameter_guard(model, int(model.config.num_hidden_layers))
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
        "nonzero_B_scale": "0.02", "parameter_sha256": digest,
        "initializer": "benchmarks.harness.l2l_schedule.make_non_vacuous",
    })
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return {"sha256": sha256(path), "parameter_sha256": digest, "parameters": identity,
            "seed": seed, "nonzero_B_scale": 0.02}
