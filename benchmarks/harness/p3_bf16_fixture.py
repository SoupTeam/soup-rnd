"""BF16 execution of the existing tiny DeepSeek checkpoint and adapter distribution.

NF4 resident projections retain both the stored bytes and Soup's Linear4bit arithmetic.
A dense dequantized Linear is not an oracle for nonzero BF16 PEFT adapters.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    from .b13_moe_fixture import (
        ALL_TARGETS,
        TARGETS,
        copy_adapter_weights,
        moe_config,
        moe_lora_config,
        moe_weights_dir,
        randomize_adapters,
        resident_adapter,
        resident_base,
        trainables,
    )
except ImportError:
    from b13_moe_fixture import (
        ALL_TARGETS,
        TARGETS,
        copy_adapter_weights,
        moe_config,
        moe_lora_config,
        moe_weights_dir,
        randomize_adapters,
        resident_adapter,
        resident_base,
        trainables,
    )


class OracleMismatchError(RuntimeError):
    """The reference does not represent the shard values or execution arithmetic."""


def _admit(device: str) -> dict[str, Any]:
    """Bound this fixed tiny fixture before constructing any weight tensors."""
    import psutil
    import torch

    host_bound = 512 * 1024**2
    device_bound = 256 * 1024**2 if device == "cuda" else 0
    available = int(psutil.virtual_memory().available)
    if available < host_bound:
        raise MemoryError(f"tiny fixture needs {host_bound} free host bytes, has {available}")
    free_device = None
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        free_device = int(torch.cuda.mem_get_info()[0])
        if free_device < device_bound:
            raise MemoryError("tiny fixture's fixed CUDA admission bound does not fit")
    return {
        "kind": "fixed conservative admission, not measured tensor attribution",
        "host_bound_bytes": host_bound,
        "host_available_before_bytes": available,
        "device_bound_bytes": device_bound,
        "device_available_before_bytes": free_device,
        "includes": "tiny checkpoint/twins, input schedule, decoder diagnostic clones and scratch",
    }


def _tensor_evidence(tensor: Any) -> dict[str, Any]:
    import torch

    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(tensor.shape), "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _stored_nf4(base: Any, shards: Path, index: Any, device: str) -> dict[str, Any]:
    """Use stored NF4 modules for projections, stored dequantizations for frozen experts."""
    import bitsandbytes as bnb
    import torch
    from bitsandbytes.functional import dequantize_4bit
    from safetensors.torch import load_file

    from soup_cli.utils.expert_shard import EXPERT_PARTS, expert_key
    from soup_cli.utils.layer_shard import (
        NF4_CODE_KEY,
        expert_paths,
        extras_shard_path,
        layer_paths,
    )
    from soup_cli.utils.layer_stream_runtime import (
        _stamp_4bit_markers,
        build_nf4_config,
        install_dequant_forward,
        rebuild_params4bit,
        rebuild_quant_state,
    )

    specs = (*index.quant_specs.values(), *index.expert_quant_specs.values())
    if not specs or any(
        spec.quant_type != "nf4" or spec.blocksize != 64 or spec.nested for spec in specs
    ) or index.double_quant:
        raise OracleMismatchError("NF4 control requires block64, double_quant=false throughout")
    extras = load_file(extras_shard_path(str(shards)), device=device)
    codes = {NF4_CODE_KEY: extras[NF4_CODE_KEY]}
    projections = 0
    expert_matrices = 0
    states: dict[str, Any] = {}
    with torch.no_grad():
        for layer_index, path in enumerate(layer_paths(str(shards), index)):
            stored = load_file(path, device=device)
            layer = base.model.layers[layer_index]
            for key, spec in index.quant_specs.items():
                if key not in stored:
                    continue
                module_name, _, leaf = key.rpartition(".")
                if leaf != "weight":
                    raise OracleMismatchError(f"NF4 key is not a projection weight: {key}")
                original = layer.get_submodule(module_name)
                if not isinstance(original, torch.nn.Linear):
                    raise OracleMismatchError(
                        f"NF4 oracle cannot replace {type(original).__name__}"
                    )
                replacement = bnb.nn.Linear4bit(
                    original.in_features, original.out_features,
                    bias=original.bias is not None, compute_dtype=torch.bfloat16,
                    compress_statistics=False, quant_type="nf4", device=device,
                )
                replacement.weight = rebuild_params4bit(key, stored, spec, codes)
                replacement.quant_state = replacement.weight.quant_state
                if original.bias is not None:
                    replacement.bias = torch.nn.Parameter(
                        original.bias.detach().clone(), requires_grad=False,
                    )
                parent_name, _, name = module_name.rpartition(".")
                parent = layer.get_submodule(parent_name) if parent_name else layer
                setattr(parent, name, replacement)
                observed = dequantize_4bit(
                    replacement.weight, quant_state=replacement.weight.quant_state,
                )
                expected = dequantize_4bit(
                    stored[key], quant_state=rebuild_quant_state(key, stored, spec, codes),
                )
                if not torch.equal(observed, expected):
                    raise OracleMismatchError(f"resident NF4 projection differs from shard: {key}")
                states[f"layer{layer_index}/{key}"] = {
                    "spec": spec.to_json(), "packed": _tensor_evidence(stored[key]),
                    "absmax": _tensor_evidence(replacement.weight.quant_state.absmax),
                    "code": _tensor_evidence(replacement.weight.quant_state.code),
                    "decoded_dtype": str(observed.dtype),
                }
                projections += 1
        for layer_index, path in expert_paths(str(shards), index).items():
            stored = load_file(path, device=device)
            experts = base.model.layers[layer_index].mlp.experts
            for expert in range(index.expert_count):
                for part, target in zip(EXPERT_PARTS, (experts.gate_up_proj, experts.down_proj)):
                    key = expert_key(expert, part)
                    state = rebuild_quant_state(key, stored, index.expert_quant_specs[part], codes)
                    value = dequantize_4bit(stored[key], quant_state=state)
                    target[expert].copy_(value)
                    if target.dtype != torch.bfloat16 or not torch.equal(target[expert], value):
                        raise OracleMismatchError(
                            f"resident expert differs from stored BF16: {key}"
                        )
                    states[f"layer{layer_index}/{key}"] = {
                        "spec": index.expert_quant_specs[part].to_json(),
                        "packed": _tensor_evidence(stored[key]),
                        "absmax": _tensor_evidence(state.absmax),
                        "code": _tensor_evidence(state.code),
                        "decoded_dtype": str(value.dtype),
                    }
                    expert_matrices += 1
    patched = install_dequant_forward(base)
    if patched != projections or not expert_matrices:
        raise OracleMismatchError("NF4 resident did not replace every stored projection/expert")
    _stamp_4bit_markers(base, build_nf4_config("bfloat16", double_quant=False))
    return {
        "projection_modules": projections,
        "product_nf4_forward_modules": patched,
        "frozen_expert_matrices": expert_matrices,
        "projection_arithmetic": "product install_dequant_forward on stored Params4bit",
        "expert_arithmetic": "native eager with exact stored per-expert BF16 dequantizations",
        "quant_states": states,
        "dense_dequantized_projection_fallback": False,
    }


def _attention(model: Any) -> dict[str, Any]:
    error = None
    previous = getattr(model.config, "_attn_implementation", None)
    try:
        model.set_attn_implementation("sdpa")
    except (AttributeError, ValueError, RuntimeError) as failure:
        error = f"{type(failure).__name__}: {failure}"
        model.config._attn_implementation = previous
    return {
        "requested": "sdpa",
        "request_error": error,
        "previous_native_config": previous,
        "on_request_failure": "restore previous native attention config, record error",
        "config_resolved": getattr(model.config, "_attn_implementation", None),
        "module_classes": sorted({
            type(module).__name__ for name, module in model.named_modules()
            if name.endswith("self_attn")
        }),
        "executed_path": "not inferred from config; driver counts SDPA calls",
    }


def _parameter_evidence(model: Any) -> dict[str, Any]:
    from peft.tuners.tuners_utils import BaseTunerLayer

    adapters = trainables(model)
    return {
        "parameter_dtypes": dict(Counter(str(parameter.dtype) for parameter in model.parameters())),
        "trainable_dtypes": dict(Counter(str(parameter.dtype) for parameter in adapters.values())),
        "trainable_classes": {
            name.replace(".inner.", "."): type(module).__module__ + "." + type(module).__name__
            for name, module in model.named_modules()
            if isinstance(module, BaseTunerLayer)
        },
        "router_parameters": {
            name: {"dtype": str(value.dtype), "device": str(value.device)}
            for name, value in model.named_parameters()
            if ".mlp.gate." in name
        },
        "trainable_tensors": len(adapters),
        "trainable_elements": sum(parameter.numel() for parameter in adapters.values()),
        "all_adapters_nonzero": all(bool(parameter.detach().ne(0).any())
                                    for parameter in adapters.values()),
    }


@dataclass
class BF16Fixture:
    config: Any
    streamed: Any
    resident: Any
    runtime: Any
    precision_evidence: dict[str, Any]

    def close(self) -> None:
        self.runtime.close()


def build_bf16_fixture(
    root: Path, *, device: str, targets: Sequence[str], weight_format: str = "dense",
) -> BF16Fixture:
    """Build independent checkpoint-loaded resident and streamed BF16 native models."""
    from safetensors import safe_open

    from soup_cli.utils.layer_shard import (
        expert_paths,
        extras_shard_path,
        layer_paths,
        shard_checkpoint,
    )
    from soup_cli.utils.layer_stream import stream_arch_of
    from soup_cli.utils.layer_stream_runtime import (
        build_meta_skeleton,
        build_streamed_model,
        quantised_layer_suffixes,
    )

    if device not in ("cpu", "cuda") or weight_format not in ("dense", "nf4"):
        raise ValueError("fixture requires device cpu/cuda and weight_format dense/nf4")
    if tuple(targets) not in (tuple(TARGETS), tuple(ALL_TARGETS)):
        raise ValueError("fixture supports exact P3 TARGETS or broad ALL_TARGETS only")
    admission = _admit(device)
    root.mkdir(parents=True, exist_ok=True)
    config = moe_config()
    weights = moe_weights_dir(root, config, seed=61)
    shards = root / "shards"
    quant_args: dict[str, Any] = {}
    if weight_format == "nf4":
        skeleton = build_meta_skeleton(
            str(weights), dtype="bfloat16", quant="nf4", double_quant=False,
        )
        quant_args = {
            "quant": "nf4", "quant_suffixes": quantised_layer_suffixes(skeleton),
            "quant_device": device, "double_quant": False,
        }
        del skeleton
    index = shard_checkpoint(
        str(weights), str(shards), dtype="bfloat16", arch=stream_arch_of(config),
        expert_layout="per_expert", **quant_args,
    )
    lora = moe_lora_config(dropout=0.0, targets=targets)
    streamed, runtime = build_streamed_model(
        model_id=str(weights), shard_dir=str(shards), index=index, lora_config=lora,
        device=device, dtype="bfloat16", buffers=2, pin=False, seed=71,
        quant="nf4" if weight_format == "nf4" else "none", double_quant=False,
    )
    try:
        randomize_adapters(streamed, seed=67)
        base = resident_base(weights, dtype="bfloat16").to(device)
        nf4 = _stored_nf4(base, shards, index, device) if weight_format == "nf4" else None
        resident = resident_adapter(base, lora)
        copy_adapter_weights(streamed, resident)
        streamed.train()
        resident.train()
        resident_evidence = _parameter_evidence(resident)
        streamed_evidence = _parameter_evidence(streamed)
        if resident_evidence["trainable_classes"] != streamed_evidence["trainable_classes"]:
            raise OracleMismatchError("resident/streamed PEFT arithmetic module classes differ")
        cache_paths = [
            weights / "model.safetensors", Path(extras_shard_path(str(shards))),
            *(Path(path) for path in layer_paths(str(shards), index)),
            *(Path(path) for path in expert_paths(str(shards), index).values()),
        ]
        with safe_open(str(weights / "model.safetensors"), framework="pt") as checkpoint:
            checkpoint_dtypes = dict(Counter(
                checkpoint.get_slice(key).get_dtype() for key in checkpoint.keys()
            ))
        evidence = {
            "weight_format": weight_format, "checkpoint_source_dtype": "float32",
            "base_storage_compute_dtype": "bfloat16",
            "source_checkpoint_policy": "unchanged b13_moe_fixture.moe_weights_dir(seed=61)",
            "checkpoint_tensor_dtypes_observed": checkpoint_dtypes,
            "weights_seed": 61, "adapter_seed": 67, "rank": 2, "targets": list(targets),
            "adapter_materialization_seed": 71,
            "adapter_initialization_policy": (
                "Existing streamed PEFT initialization, then unchanged "
                "b13_moe_fixture.randomize_adapters(seed=67), B and embedding factors only"
            ),
            "admission": admission, "autocast_enabled": False,
            "shard_dtype": index.dtype, "expert_pool_dtype": runtime.expert_pool.dtype,
            "pool_tensor_dtypes": dict(Counter(
                str(value.dtype) for slot in runtime.pool.buffers for value in slot.values()
            )),
            "expert_pool_tensor_dtypes": dict(Counter(
                str(value.dtype) for value in runtime.expert_pool._buffers.values()
            )),
            "quant_device": index.quant_device,
            "double_quant": index.double_quant,
            "quant_specs": {key: spec.to_json() for key, spec in index.quant_specs.items()},
            "expert_quant_specs": {
                key: spec.to_json() for key, spec in index.expert_quant_specs.items()
            },
            "checkpoint_and_cache_sha256": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in cache_paths
            },
            "nf4_oracle": nf4,
            "resident": resident_evidence,
            "streamed": streamed_evidence,
            "attention": {"resident": _attention(resident), "streamed": _attention(streamed)},
        }
        if not evidence["resident"]["all_adapters_nonzero"]:
            raise OracleMismatchError("nonzero resident adapters were not preserved")
        return BF16Fixture(config, streamed, resident, runtime, evidence)
    except BaseException:
        runtime.close()
        raise
