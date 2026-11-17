"""Bounded attached-weight/router and full-dimension activation-storage probes.

These probes never allocate a DeepSeek model. Embedding rows are not decoder-layer
activations, a router-only result does not certify 671B gradients, and allocating a
61-layer activation store does not demonstrate an 8 GB whole-model training fit.
Heavy dependencies stay inside the functions that use them.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import os
from pathlib import Path
from typing import Any, Sequence

try:
    from .p3_fp8_smoke import (
        KS,
        MemoryObservation,
        _codec_metrics,
        _error,
        _finite_tree,
        store_evidence,
        thresholds,
    )
except ImportError:
    from p3_fp8_smoke import (
        KS,
        MemoryObservation,
        _codec_metrics,
        _error,
        _finite_tree,
        store_evidence,
        thresholds,
    )


MAX_TENSOR_READ_BYTES = 32 * 1024 * 1024
MAX_STORE_RESERVATION_BYTES = 3 * 1024**3


def _tensor_identity(tensor: Any) -> dict[str, Any]:
    import torch

    contiguous = tensor.detach().cpu().contiguous()
    raw = contiguous.view(torch.uint8).numpy().tobytes()
    values = contiguous.float()
    return {
        "shape": list(tensor.shape), "dtype": str(tensor.dtype),
        "elements": tensor.numel(), "storage_bytes": tensor.numel() * tensor.element_size(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "finite": bool(torch.isfinite(values).all()),
        "minimum": float(values.min()), "maximum": float(values.max()),
    }


def _local_shard(root: Path, model_dir: Path, weight_map: dict[str, str], key: str) -> Path:
    if key not in weight_map:
        raise ValueError(f"attached index has no {key}")
    shard = Path(os.path.realpath(model_dir / weight_map[key]))
    if os.path.commonpath((str(root), str(shard))) != str(root):
        raise ValueError(f"attached shard escapes supplied model input: {key}")
    if not shard.is_file():
        raise FileNotFoundError(f"attached shard missing for {key}: {shard}")
    return shard


def _bounded_tensor(
    root: Path, model_dir: Path, weight_map: dict[str, str], key: str,
    expected_shape: Sequence[int], *, row_ids: Sequence[int] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Read only explicitly bounded shapes; never handle.get_tensor for the embedding."""
    import torch
    from safetensors import safe_open

    shard = _local_shard(root, model_dir, weight_map, key)
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        view = handle.get_slice(key)
        shape = list(view.get_shape())
        if shape != list(expected_shape):
            raise ValueError(f"{key}: expected {list(expected_shape)}, found {shape}")
        count = math.prod(shape) if row_ids is None else len(row_ids) * shape[1]
        # At most 8 bytes per element. Validate before asking safetensors for storage.
        if count * 8 > MAX_TENSOR_READ_BYTES:
            raise ValueError(f"{key}: bounded read exceeds {MAX_TENSOR_READ_BYTES} bytes")
        if row_ids is None:
            tensor = view[:]
        else:
            if any(row < 0 or row >= shape[0] for row in row_ids):
                raise ValueError("embedding probe row is outside attached vocabulary")
            tensor = torch.cat([view[row:row + 1, :] for row in row_ids], dim=0)
    identity = _tensor_identity(tensor)
    if identity["storage_bytes"] > MAX_TENSOR_READ_BYTES:
        raise AssertionError("bounded tensor read exceeded its hard byte limit")
    identity.update({
        "tensor": key, "shard": str(shard), "full_shape": shape,
        "read_kind": "complete bounded router tensor" if row_ids is None else "embedding rows",
        "row_ids": list(row_ids) if row_ids is not None else None,
    })
    return tensor, identity


def _router_result(output: Any, top_k: int) -> tuple[Any, Any]:
    """Require the installed native router's returned expert indices, not reimplemented top-k."""
    import torch

    if not isinstance(output, (tuple, list)):
        raise TypeError("native router no longer returns a tuple/list; probe needs explicit update")
    indices = [item for item in output if isinstance(item, torch.Tensor)
               and item.dtype in (torch.int32, torch.int64) and item.shape[-1] == top_k]
    weights = [item for item in output if isinstance(item, torch.Tensor)
               and item.is_floating_point() and item.shape[-1] == top_k]
    if len(indices) != 1 or len(weights) != 1:
        raise ValueError("native router return contract has no unique expert indices/weights")
    return indices[0], weights[0]


def embedding_router_probe(
    *, root: Path, model_dir: Path, config_dict: dict[str, Any],
    weight_map: dict[str, str], device: str,
) -> dict[str, Any]:
    """Attached embedding rows through an attached native router and the production store."""
    import torch
    from transformers import DeepseekV3Config
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3TopkRouter

    from soup_cli.utils.fp8_activations import FP8ActivationStore, fp8_activation_allocation
    from soup_cli.utils.host_memory_budget import HostMemoryBudget

    result: dict[str, Any] = {
        "status": "FAIL",
        "kind": "attached embedding-row probe, NOT decoder-layer activations or training",
        "scope": (
            "Eight fixed vocabulary embedding rows fed directly to the first routed layer's "
            "native router. Does not run the preceding dense decoder layers, attention, "
            "normalization, experts, backward or optimizer. Does not certify 671B training."
        ),
        "whole_model_allocated": False, "training_671b_steps": 0,
        "read_limit_bytes_per_tensor": MAX_TENSOR_READ_BYTES,
        "strict_routing_policy": "changed expert IDs are FP8 refusal, never forced unchanged",
    }
    store = None
    observed = None
    budget = HostMemoryBudget(64 * 1024 * 1024)
    try:
        hidden = int(config_dict["hidden_size"])
        experts = int(config_dict["n_routed_experts"])
        first_layer = int(config_dict["first_k_dense_replace"])
        row_ids = (0, 1, 2, 3, 127, 1024, 4096, 32000)
        prefix = f"model.layers.{first_layer}.mlp.gate"
        rows, embedding_identity = _bounded_tensor(
            root, model_dir, weight_map, "model.embed_tokens.weight",
            (int(config_dict["vocab_size"]), hidden), row_ids=row_ids,
        )
        weight, weight_identity = _bounded_tensor(
            root, model_dir, weight_map, f"{prefix}.weight", (experts, hidden),
        )
        bias, bias_identity = _bounded_tensor(
            root, model_dir, weight_map, f"{prefix}.e_score_correction_bias", (experts,),
        )
        result["attached_weight_reads"] = [embedding_identity, weight_identity, bias_identity]
        if not all(item["finite"] for item in result["attached_weight_reads"]):
            raise ValueError("nonfinite attached embedding/router tensor")
        if not weight.is_floating_point() or weight.element_size() < 2:
            raise ValueError(
                "router weight is quantized; bounded probe requires its native fp32/bf16"
            )
        config = DeepseekV3Config.from_dict(config_dict)
        router = DeepseekV3TopkRouter(config).requires_grad_(False).to(device)
        if list(router.weight.shape) != [experts, hidden]:
            raise ValueError("installed native router configuration disagrees with attached shape")
        with torch.no_grad():
            router.weight.copy_(weight.to(router.weight))
            router.e_score_correction_bias.copy_(bias.to(router.e_score_correction_bias))
        del weight, bias
        result["native_router"] = {
            "class": f"{type(router).__module__}.{type(router).__name__}",
            "forward_source_sha256": hashlib.sha256(
                inspect.getsource(type(router).forward).encode("utf-8")
            ).hexdigest(),
            "layer_index": first_layer, "experts": experts, "top_k": config.num_experts_per_tok,
            "n_group": config.n_group, "topk_group": config.topk_group,
            "router_parameters": sum(param.numel() for param in router.parameters()),
        }
        # Preserve attached embedding dtype; the native router casts its linear math to fp32.
        original = rows.unsqueeze(0).to(device)
        del rows
        allocation = fp8_activation_allocation(
            n_layers=1, mb_shapes=[tuple(original.shape)], pin=device == "cuda",
            compute_dtype=str(original.dtype).removeprefix("torch."), device_type=device,
        )
        result["store_allocation_estimate"] = allocation.as_dict()
        with MemoryObservation(device, scope=(
            "One bounded native router, attached embedding rows, production FP8 store and "
            "router comparison tensors. No resident or streamed full model, no backward."
        )) as observed:
            store = FP8ActivationStore(
                host_budget=budget, n_layers=1, k=1, chunk_shape=tuple(original.shape),
                mb_shapes=[tuple(original.shape)], dtype=original.dtype,
                device=device, pin=device == "cuda",
            )
            store.put(0, 0, original)
            store.begin_backward()
            restored = store.get(0, 0)
            with torch.no_grad():
                reference_output = router(original)
                candidate_output = router(restored)
                reference_ids, reference_weights = _router_result(
                    reference_output, config.num_experts_per_tok,
                )
                candidate_ids, candidate_weights = _router_result(
                    candidate_output, config.num_experts_per_tok,
                )
                reference_logits = torch.nn.functional.linear(original.float(), router.weight)
                candidate_logits = torch.nn.functional.linear(restored.float(), router.weight)
            metrics = _codec_metrics()
            result["restore_e1"] = metrics.restore_error(original, restored)
            result["router_logit_error"] = metrics.restore_error(reference_logits, candidate_logits)
            result["route_weight_error"] = metrics.restore_error(
                reference_weights, candidate_weights,
            )
            changed = (reference_ids != candidate_ids).any(dim=-1)
            set_changed = (
                reference_ids.sort(dim=-1).values != candidate_ids.sort(dim=-1).values
            ).any(dim=-1)
            result["routing"] = {
                "kind": "observed native router output; ordered IDs match product guard policy",
                "tokens": int(changed.numel()), "changed_ordered_tokens": int(changed.sum()),
                "changed_expert_set_tokens": int(set_changed.sum()),
                "ordered_disagreement_fraction": float(changed.float().mean()),
                "reference_experts": reference_ids.detach().cpu().tolist(),
                "fp8_restored_experts": candidate_ids.detach().cpu().tolist(),
            }
            result["store_observed"] = store_evidence(store, budget)
            store.end_step()
            del restored
            store.close()
        result["memory_observed"] = observed.result
        bounds = thresholds()
        finite = all(_finite_tree(result[key]) for key in (
            "restore_e1", "router_logit_error", "route_weight_error", "routing",
        ))
        restore_ok = (
            result["restore_e1"]["rms_row_rel"] <= bounds["restore_rms_row_rel_max"]
            and result["restore_e1"]["worst_row_rel"] <= bounds["restore_worst_row_rel_max"]
        )
        result["finite"] = finite
        result["status"] = "PASS" if finite and restore_ok else "FAIL"
        if finite and restore_ok and result["routing"]["changed_ordered_tokens"]:
            result["status"] = "REFUSED"
            result["reason"] = "product-equivalent strict policy refuses changed native expert IDs"
    except Exception as error:
        result["error"] = _error(error)
    finally:
        if observed is not None:
            result["memory_observed"] = observed.result
        if store is not None:
            try:
                store.close()
            except Exception as error:
                result["close_error"] = _error(error)
                result["status"] = "FAIL"
        result["budget_used_after_close_bytes"] = budget.used_bytes
        if budget.used_bytes:
            result["status"] = "FAIL"
    return result


def full_shape_store_probes(config: dict[str, Any], *, device: str) -> dict[str, Any]:
    """Allocate/fill/restore every slot sequentially at attached 61x7168 dimensions."""
    import torch

    from soup_cli.utils.fp8_activations import FP8ActivationStore, fp8_activation_allocation
    from soup_cli.utils.host_memory_budget import HostMemoryBudget

    layers = int(config["num_hidden_layers"])
    hidden = int(config["hidden_size"])
    sequence = 64
    if (layers, hidden) != (61, 7168):
        raise ValueError("full-shape store probe requires attached 61-layer, hidden-7168 config")
    result: dict[str, Any] = {
        "status": "FAIL", "cases": [],
        "kind": "activation-store shape evidence, NOT 671B training or an 8 GB whole-model fit",
        "shape_source": "attached config.num_hidden_layers and config.hidden_size",
        "layers": layers, "hidden_size": hidden, "sequence_length": sequence,
        "batch_rows": 1, "compute_dtype": "bfloat16", "k": list(KS),
        "whole_model_allocated": False, "training_671b_steps": 0,
        "source_values": (
            "Fixed synthetic bf16 normal tensor seed123, per-slot scaled/offset. "
            "No decoder activations or giant-model forward implied."
        ),
        "peak_reservation_hard_limit_bytes": MAX_STORE_RESERVATION_BYTES,
        "allocation_lifetime": "one store at a time; close drains before next k",
    }
    metrics = _codec_metrics()
    bounds = thresholds()
    shape = (1, sequence, hidden)
    for k in KS:
        row: dict[str, Any] = {"k": k, "status": "FAIL"}
        result["cases"].append(row)
        budget = HostMemoryBudget(MAX_STORE_RESERVATION_BYTES)
        store = None
        observed = None
        try:
            estimate = fp8_activation_allocation(
                n_layers=layers, mb_shapes=[shape] * k, pin=device == "cuda",
                compute_dtype="bfloat16", device_type=device,
            )
            row["allocation_estimate"] = estimate.as_dict()
            if estimate.host_bytes > MAX_STORE_RESERVATION_BYTES:
                raise MemoryError("product estimate exceeds full-shape probe hard memory limit")
            generator = torch.Generator(device="cpu").manual_seed(123)
            source = torch.randn(shape, generator=generator).to(torch.bfloat16).to(device)

            def chunk(source: Any, layer: int, mb: int) -> Any:
                return source * ((layer + 1) / layers) + (mb % 7) * 0.03125

            error_parts: dict[int, list[dict[str, float]]] = {layer: [] for layer in range(layers)}
            finite = True
            puts = gets = 0
            with MemoryObservation(device, scope=(
                "One full-dimension activation store plus one synthetic input/restored chunk, "
                "G4 metric scratch and Python process baseline. No model weights or training."
            )) as observed:
                store = FP8ActivationStore(
                    host_budget=budget, n_layers=layers, k=k, chunk_shape=shape,
                    mb_shapes=[shape] * k, dtype=torch.bfloat16,
                    device=device, pin=device == "cuda",
                )
                for layer in range(layers):
                    for mb in range(k):
                        value = chunk(source, layer, mb)
                        store.put(layer, mb, value)
                        puts += 1
                        del value
                store.begin_backward()
                for layer in reversed(range(layers)):
                    for mb in range(k):
                        restored = store.get(layer, mb)
                        value = chunk(source, layer, mb)
                        finite = finite and bool(torch.isfinite(restored).all())
                        error_parts[layer].append(metrics.restore_error(value, restored))
                        gets += 1
                        del restored, value
                store.end_step()
                row["store_observed"] = store_evidence(store, budget)
                row["reservation_matches_estimate"] = budget.used_bytes == estimate.host_bytes
                storage = row["store_observed"]["storage_observed"]
                # Independent shape arithmetic, not a second read of estimator output.
                payload_formula = layers * k * sequence * hidden
                scales_formula = layers * k * sequence * 4
                persistent = payload_formula + scales_formula
                row["estimator_payload_scale_match_shape_formula"] = (
                    estimate.payload_bytes == payload_formula
                    and estimate.scale_bytes == scales_formula
                )
                row["host_reservation_breakdown_matches"] = estimate.host_bytes == (
                    persistent + estimate.pinned_overhead_bytes + estimate.host_transient_bytes
                )
                row["host_accounting_excludes_device_transients"] = (
                    row["host_reservation_breakdown_matches"]
                )
                row["owned_persistent_storage_matches_payload_scales"] = (
                    storage["owned_host_storage_bytes"] == persistent
                )
                row["payload_scale_storage_formula_bytes"] = persistent
                row["fp32_uncompressed_payload_estimate_bytes"] = layers * k * math.prod(shape) * 4
                row["bf16_uncompressed_payload_estimate_bytes"] = layers * k * math.prod(shape) * 2
                store.close()
            row["memory_observed"] = observed.result
            row["puts"] = puts
            row["gets"] = gets
            row["expected_chunks_each_direction"] = layers * k
            row["restore_e1_layers"] = {
                str(layer): metrics.merge_errors(parts) for layer, parts in error_parts.items()
            }
            row["finite"] = finite and _finite_tree(row["restore_e1_layers"])
            restore_ok = all(
                item["rms_row_rel"] <= bounds["restore_rms_row_rel_max"]
                and item["worst_row_rel"] <= bounds["restore_worst_row_rel_max"]
                for item in row["restore_e1_layers"].values()
            )
            row["status"] = "PASS" if (
                row["finite"] and restore_ok and row["reservation_matches_estimate"]
                and row["owned_persistent_storage_matches_payload_scales"]
                and row["estimator_payload_scale_match_shape_formula"]
                and row["host_reservation_breakdown_matches"]
                and puts == gets == layers * k
            ) else "FAIL"
            del source, error_parts
        except Exception as error:
            row["error"] = _error(error)
        finally:
            if observed is not None:
                row["memory_observed"] = observed.result
            if store is not None:
                try:
                    store.close()
                except Exception as error:
                    row["close_error"] = _error(error)
                    row["status"] = "FAIL"
            row["budget_used_after_close_bytes"] = budget.used_bytes
            if budget.used_bytes:
                row["status"] = "FAIL"
        # Continue all declared k even if one allocation or roundtrip fails.
    result["status"] = "PASS" if all(row["status"] == "PASS" for row in result["cases"]) else "FAIL"
    return result
