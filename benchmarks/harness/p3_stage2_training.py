"""Bounded Stage2 training of the exact Oracle plus block64 E4M3 experiment.

This module launches nothing. The caller loads the pinned native Granite model,
freezes data/protocol admission, and runs one arm in a fresh process. Cross-arm
BF16/VJP comparisons remain mandatory and cannot be inferred from forced IDs.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import time
import traceback
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

if __package__:
    from . import p3_realdata_baseline as baseline
    from . import p3_realdata_model as native
    from .p3_fp8_smoke import MemoryObservation, thresholds
    from .p3_realdata_codec import CodecPolicy, allocation, make_store
    from .p3_realdata_routing import RouteProbe, routing_allocation_bound
else:
    import p3_realdata_baseline as baseline
    import p3_realdata_model as native
    from p3_fp8_smoke import MemoryObservation, thresholds
    from p3_realdata_codec import CodecPolicy, allocation, make_store
    from p3_realdata_routing import RouteProbe, routing_allocation_bound

HOST_LIMIT = 2 * 1024**3
DEVICE_SCRATCH_LIMIT = 128 * 1024**2
_COMMON_CAPTURE_LIMIT = 1024**3
_COMMON_TRAINER_DEVICE_LIMIT = 2 * 1024**3
_EVIDENCE_LIMIT = 256 * 1024**2
_STEPS = 64
_SEEDS = (23, 41, 67, 89, 113)
_ARMS = {
    "bf16_ga": ("ga", False, CodecPolicy("bf16", "none")),
    "bf16_l2l": ("l2l", False, CodecPolicy("bf16", "none")),
    "bf16_oracle": ("l2l", True, CodecPolicy("bf16_oracle", "none")),
    "oracle_block64_e4m3": (
        "l2l", True, CodecPolicy("oracle_block64_e4m3", "e4m3", group_size=64),
    ),
}


class _GuardError(RuntimeError):
    """A mandatory admission condition failed; no permissive fallback exists."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise _GuardError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _error(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _hex_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _ids_hash(rows: Sequence[Sequence[int]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        for token in row:
            digest.update(struct.pack("<q", token))
    return digest.hexdigest()


def _inspect_window(window: dict) -> dict[str, Any]:
    """Validate frozen JSON and hash flattened int64 streams before allocating tensors."""
    if not isinstance(window, dict):
        raise ValueError("window must be a JSON dictionary")
    encoded = json.dumps(window, sort_keys=True, allow_nan=False, separators=(",", ":"))
    frozen = json.loads(encoded)
    if not isinstance(frozen.get("window_id"), str) or not frozen["window_id"]:
        raise ValueError("window_id must be a nonempty string")
    batches = frozen.get("micro_batches")
    if not isinstance(batches, list) or not batches:
        raise ValueError("micro_batches must be a nonempty list")
    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    shapes = []
    valid = []
    for mb in batches:
        if not isinstance(mb, dict):
            raise ValueError("each microbatch must be a JSON dictionary")
        ids, labels = mb.get("input_ids"), mb.get("labels")
        if not isinstance(ids, list) or not ids or not isinstance(labels, list):
            raise ValueError("input_ids/labels must be nonempty rectangular BxL lists")
        if not isinstance(ids[0], list) or len(ids[0]) < 2 or len(labels) != len(ids):
            raise ValueError("input_ids/labels must share a BxL shape with L >= 2")
        length = len(ids[0])
        count = 0
        for row, label in zip(ids, labels):
            if not isinstance(row, list) or not isinstance(label, list):
                raise ValueError("input_ids/labels rows must be lists")
            if len(row) != length or len(label) != length:
                raise ValueError("input_ids/labels must be rectangular and shape-matched")
            for token, target in zip(row, label):
                if type(token) is not int or not 0 <= token < 2**63:
                    raise ValueError("input IDs must be nonnegative signed-int64 integers")
                if type(target) is not int or target != -100 and target != token:
                    raise ValueError("response labels must equal their input ID or -100")
            count += sum(target != -100 for target in label[1:])
        for key, rows in (("input_sha256", ids), ("labels_sha256", labels)):
            if key in mb and mb[key] != _ids_hash(rows):
                raise ValueError(f"microbatch {key} checksum mismatch")
        shapes.append([len(ids), length])
        valid.append(count)
        input_rows.extend(ids)
        label_rows.extend(labels)
    count = sum(valid)
    if type(frozen.get("valid_target_count")) is not int or count < 1:
        raise ValueError("valid_target_count must be a positive integer")
    if frozen["valid_target_count"] != count:
        raise ValueError("valid_target_count differs from complete-window shifted labels")
    inputs, labels = _ids_hash(input_rows), _ids_hash(label_rows)
    paired = hashlib.sha256(
        b"p3-stage2-pair-v1\0" + bytes.fromhex(inputs) + bytes.fromhex(labels)
        + struct.pack("<q", count)
    ).hexdigest()
    hashes = {"input_sha256": inputs, "labels_sha256": labels, "pairing_sha256": paired}
    for key, expected in hashes.items():
        if key in frozen and frozen[key] != expected:
            raise ValueError(f"window {key} checksum mismatch")
    return {
        "window": frozen, "window_id": frozen["window_id"], "k": len(batches),
        "micro_batch_shapes": shapes, "micro_batch_valid_target_counts": valid,
        "valid_target_count": count, "input_token_count": sum(map(len, input_rows)),
        "window_json_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(), **hashes,
    }


def _study_windows(windows: Sequence[dict], *, preflight: bool) -> list[dict[str, Any]]:
    if not isinstance(windows, Sequence) or isinstance(windows, (str, bytes)):
        raise ValueError("windows must be a frozen sequence")
    if preflight and len(windows) < 2:
        raise ValueError("preflight requires at least two frozen windows")
    if not preflight and len(windows) != _STEPS:
        raise ValueError("training requires exactly 64 frozen windows")
    selected = windows[:2] if preflight else windows
    infos = [_inspect_window(window) for window in selected]
    seen = set()
    layout = None
    for info in infos:
        if info["window_id"] in seen:
            raise ValueError("duplicate frozen window_id")
        seen.add(info["window_id"])
        shapes = info["micro_batch_shapes"]
        current = (len(shapes), tuple(tuple(shape) for shape in shapes))
        if info["input_token_count"] != 4096 or shapes not in (
            [[1, 1024]] * 4, [[1, 4096]],
        ):
            raise ValueError("each window must contain 4096 inputs, B1 L1024/k4 or L4096/k1")
        if layout is not None and current != layout:
            raise ValueError("an arm cannot mix training-length layouts")
        layout = current
        offset = 0
        for mb in info["window"]["micro_batches"]:
            for label in mb["labels"]:
                for index, target in enumerate(label):
                    if (offset + index) % 1024 == 0 and target != -100:
                        raise ValueError("every common 1024-token boundary must be ignored")
                offset += len(label)
    return infos


def _new_optimizer(params: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    from transformers import get_cosine_schedule_with_warmup

    optimizer = torch.optim.AdamW(
        list(params.values()), lr=2e-4, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.01, foreach=False, fused=False,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=2, num_training_steps=_STEPS,
    )
    return optimizer, scheduler


def _optimizer_guard(optimizer: Any, scheduler: Any, params: dict[str, Any]) -> None:
    import torch

    if optimizer is None:
        _require(scheduler is None, "a scheduler requires an optimizer")
        return
    _require(isinstance(optimizer, torch.optim.AdamW), "the declared optimizer is AdamW")
    observed = [param for group in optimizer.param_groups for param in group["params"]]
    _require(len(observed) == len(params) and {id(param) for param in observed}
             == {id(param) for param in params.values()},
             "optimizer parameter set must equal all and only trainable LoRA parameters")
    for group in optimizer.param_groups:
        _require(group["betas"] == (0.9, 0.999) and group["eps"] == 1e-8
                 and group["weight_decay"] == 0.01 and group.get("foreach") is False
                 and group.get("fused") is False and not group.get("amsgrad", False)
                 and not group.get("maximize", False) and not group.get("capturable", False)
                 and not group.get("differentiable", False),
                 "AdamW settings differ from the approved protocol")
        rate = group["lr"]
        _require(type(rate) in (int, float) and math.isfinite(rate) and 0 <= rate <= 2e-4,
                 "learning rate is outside the declared schedule")
    if scheduler is not None:
        _require(scheduler.optimizer is optimizer, "scheduler belongs to another optimizer")


def _frozen_signatures(model: Any) -> dict[str, tuple[Any, ...]]:
    return {
        name: (id(param), param.data_ptr(), param._version, tuple(param.shape),
               str(param.dtype), str(param.device))
        for name, param in model.named_parameters() if not param.requires_grad
    }


def _check_frozen(model: Any, signatures: dict[str, tuple[Any, ...]]) -> None:
    _require(_frozen_signatures(model) == signatures, "frozen base identity/version changed")
    _require(all(param.grad is None for param in model.parameters() if not param.requires_grad),
             "a frozen base parameter received a gradient")


def _parameter_hashes(params: dict[str, Any]) -> dict[str, str]:
    return {name: baseline.tensor_hash(param) for name, param in params.items()}


def _gradient_evidence(
    params: dict[str, Any], *, capture: bool,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    manifests: dict[str, Any] = {}
    gradients: dict[str, Any] = {}
    for name, param in params.items():
        _require(param.grad is not None, f"missing gradient: {name}")
        _require(not param.grad.is_sparse and param.grad.device == param.device,
                 f"incorrect gradient layout/device: {name}")
        _require(param.grad.shape == param.shape and param.grad.dtype == param.dtype,
                 f"incorrect gradient shape/dtype: {name}")
        value = param.grad.detach().to(device="cpu", copy=True).contiguous()
        manifest = baseline.gradient_manifest({name: value})[name]
        _require(manifest["finite"], f"nonfinite gradient: {name}")
        _require(manifest["nonzero"] > 0, f"zero gradient tensor: {name}")
        manifests[name] = manifest
        if capture:
            gradients[name] = value
    norm = math.sqrt(math.fsum(item["l2"] ** 2 for item in manifests.values()))
    _require(math.isfinite(norm) and norm > 0, "nonfinite or zero raw gradient norm")
    return manifests, gradients, norm


def _device_sync(device: Any) -> None:
    if device.type == "cuda":
        import torch

        if not torch.cuda.is_available():
            return

        torch.cuda.synchronize(device)


def _policy_plan(
    model: Any, info: dict[str, Any], *, method: str, policy: CodecPolicy,
    oracle: bool, device_type: str,
) -> dict[str, Any]:
    shapes = [(*shape, model.config.hidden_size) for shape in info["micro_batch_shapes"]]
    route = routing_allocation_bound(
        n_layers=model.config.num_hidden_layers, mb_shapes=shapes,
        experts=model.config.num_local_experts, topk=model.config.num_experts_per_tok,
        oracle=oracle, device_type=device_type,
    )
    store = allocation(
        policy, n_layers=model.config.num_hidden_layers, mb_shapes=shapes,
        dtype="bfloat16", pin=device_type == "cuda", device_type=device_type,
    ) if method == "l2l" else None
    return {
        "route": route, "store": store,
        "host_bytes": sum(route["host"].values())
        + (store["charged_host_bytes"] if store else 0),
        "device_scratch_bytes": route["device_buffer_bound_bytes"]
        + (store["device_scratch_bytes"] if store else 0),
        "device_persistent_bytes": store["device_persistent_bytes"] if store else 0,
    }


def _common_plan(
    model: Any, info: dict[str, Any], params: dict[str, Any], *, method: str,
    device_type: str, capture: bool, optimizer: Any,
) -> dict[str, Any]:
    elements = sum(param.numel() for param in params.values())
    maximum = max(param.numel() for param in params.values())
    tokens = info["input_token_count"]
    shapes = info["micro_batch_shapes"]
    # Native eager capture retains one BF16 mask, RoPE pair and positions per microbatch.
    rope_width = model.config.hidden_size // model.config.num_attention_heads
    capture_persistent = sum(
        batch * length * length * 2 + 2 * batch * length * rope_width * 2 + length * 8
        for batch, length in shapes
    ) if method == "l2l" else 0
    # Conservative allowance for mask/RoPE construction, separate from policy scratch.
    capture_scratch = max(
        32 * batch * length * length + 16 * batch * length * model.config.hidden_size
        for batch, length in shapes
    ) if method == "l2l" else 0
    optimizer_state = 8 * elements + 8 * len(params) if optimizer is not None else 0
    optimizer_scratch = 32 * maximum if optimizer is not None else 0
    l2l_buffers = 4 * tokens * model.config.hidden_size if method == "l2l" else 0
    cpu_common = (
        16 * tokens + capture_persistent + capture_scratch + l2l_buffers
        + 4 * elements + optimizer_state + optimizer_scratch
    ) if device_type == "cpu" else 8 * max(batch * length for batch, length in shapes)
    return {
        "input_tensor_bytes": 16 * tokens,
        "input_host_staging_bytes": 8 * max(batch * length for batch, length in shapes),
        "raw_gradient_export_bytes": 4 * elements if capture else 0,
        "host_diagnostic_scratch_bound_bytes": 40 * maximum + 1024**2,
        "device_diagnostic_scratch_bound_bytes": 16 * maximum + 64 * len(params),
        "native_call_capture_persistent_bound_bytes": capture_persistent,
        "native_call_capture_scratch_bound_bytes": capture_scratch,
        "l2l_hidden_upstream_bound_bytes": l2l_buffers,
        "trainable_gradient_bytes": 4 * elements,
        "optimizer_state_bound_bytes": optimizer_state,
        "optimizer_operation_scratch_bound_bytes": optimizer_scratch,
        "host_reservation_bytes": (4 * elements if capture else 0)
        + 40 * maximum + 1024**2 + cpu_common,
        "scope": "common caller/trainer tensors, not activation-policy saving",
        "native_decoder_autograd_and_kernel_workspaces": "observed, not tensor-ledger bounded",
    }


def _captured_storage(calls: Any) -> int:
    import torch

    seen = set()

    def visit(value: Any) -> int:
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            identity = (str(value.device), storage.data_ptr())
            if identity in seen:
                return 0
            seen.add(identity)
            return storage.nbytes()
        if isinstance(value, dict):
            return sum(visit(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(visit(item) for item in value)
        return 0

    return visit(calls)


def _memory_accounting(plan: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    host = plan["host_bytes"]
    device = plan["device_scratch_bytes"] + plan["device_persistent_bytes"]
    ref_host = reference["host_bytes"]
    ref_device = reference["device_scratch_bytes"] + reference["device_persistent_bytes"]
    return {
        "kind": "conservative tensor reservation arithmetic, not measured peaks",
        "policy_host_bound_bytes": host,
        "policy_device_scratch_bound_bytes": plan["device_scratch_bytes"],
        "policy_device_persistent_bound_bytes": plan["device_persistent_bytes"],
        "reference_bf16_l2l_host_bound_bytes": ref_host,
        "reference_bf16_l2l_device_bound_bytes": ref_device,
        "host_reservation_saving_bytes": ref_host - host,
        "device_reservation_saving_bytes": ref_device - device,
        "cross_resource_reservation_saving_bytes": ref_host + ref_device - host - device,
        "positive_fully_accounted_reservation_saving": host + device < ref_host + ref_device,
        "reservation_sum_is_measured_peak": False,
        "stage2_minimum_saving_fraction": None,
        "historical_25_percent_gate_replaced_for_stage2_only": True,
        "historical_evidence_modified": False,
        "allocator_caches_python_objects_kernel_workspaces": "observed separately, not bounded RSS",
        "lease_release_proves_os_rss_release": False,
        "fresh_process_per_arm_required": True,
    }


def run_window(
    model: Any, window: dict, *, method: str, policy: CodecPolicy, oracle: bool,
    device: str, optimizer: Any = None, scheduler: Any = None,
    capture_gradients: bool = False, host_limit: int = HOST_LIMIT,
    device_scratch_limit: int = DEVICE_SCRATCH_LIMIT,
) -> tuple[dict, dict]:
    """Run one weighted native window, optional clip/update, then drop owned resources.

    The second return owns raw pre-clip CPU FP32 gradients when requested. A failed
    record is never admission and never permits an arm to continue to later windows.
    """
    import torch

    from soup_cli.utils.host_memory_budget import HostMemoryBudget

    started = time.perf_counter()
    result: dict[str, Any] = {
        "status": "STARTED", "method": method, "oracle": oracle,
        "policy": vars(policy) if isinstance(policy, CodecPolicy) else None,
        "optimizer_step": False, "optimizer_steps_completed": 0,
        "updated_parameter_count": 0, "mandatory_admission_failure": False,
        "forward_backward_complete": False,
        "host_limit_bytes": host_limit, "device_scratch_limit_bytes": device_scratch_limit,
        "numerical_admission": {"status": "EXTERNAL_COMPARISON_REQUIRED",
                                "forced_agreement_is_vjp_proof": False},
        "old_numerical_gates": {"thresholds": thresholds(), "status": "NOT_COMPARED",
                                "finite_failure_is_not_stage2_approximate_experiment_veto": True,
                                "finite_failure_is_numerical_admission": False},
    }
    budget = lease = probe = store = calls = memory = context = target = None
    params: dict[str, Any] = {}
    gradients: dict[str, Any] = {}
    batches: list[Any] = []
    labels: list[Any] = []
    cleanup_errors: list[dict[str, str]] = []
    host_tensor = None
    memory_entered = context_entered = False
    cleared_gradients = False
    fatal = None
    phase = "admission"
    signatures = None

    def close_policy() -> None:
        nonlocal store, probe
        for name, resource in (("routing", probe), ("store", store)):
            if resource is None:
                continue
            try:
                resource.close()
                cleaned = resource.report()
                _require(cleaned["owned_host_storage_bytes"] == 0
                         and cleaned["owned_device_storage_bytes"] == 0,
                         f"{name} retained owned tensor storage after close")
                if name == "routing":
                    _require(cleaned["closed"] and cleaned["pending_oracle_replays"] == 0
                             and cleaned["cleanup"]["hooks_removed"]
                             and cleaned["cleanup"]["original_forwards_restored"],
                             "routing interception or VJP survived cleanup")
                else:
                    _require(cleaned["cleanup"]["closed"]
                             and cleaned["cleanup"]["host_lease_released"],
                             "codec host lease survived cleanup")
                result[f"{name}_after_cleanup"] = resource.report()
            except BaseException as error:
                cleanup_errors.append({"scope": name, **_error(error)})
                traceback.clear_frames(error.__traceback__)
            else:
                if name == "routing":
                    probe = None
                else:
                    store = None

    try:
        model.zero_grad(set_to_none=True)
        cleared_gradients = True
        if type(host_limit) is not int or host_limit < 1:
            raise ValueError("host_limit must be a finite positive integer byte limit")
        if type(device_scratch_limit) is not int or device_scratch_limit < 1:
            raise ValueError("device_scratch_limit must be a finite positive integer byte limit")
        if type(oracle) is not bool or type(capture_gradients) is not bool:
            raise ValueError("oracle and capture_gradients must be bool")
        if method not in ("ga", "l2l") or not isinstance(policy, CodecPolicy):
            raise ValueError("method must be ga/l2l and policy must be a CodecPolicy")
        allowed = policy.format == "none" and not policy.smooth and policy.group_size is None
        combo = policy.format == "e4m3" and policy.group_size == 64 and not policy.smooth
        if not allowed and not (combo and oracle and method == "l2l"):
            raise ValueError("Stage2 FP8 is only the exact Oracle+block64+E4M3 combination")
        if method == "ga" and oracle:
            raise ValueError("native GA cannot use Oracle routing")
        info = _inspect_window(window)
        result.update(info)
        target = torch.device(device)
        _require(target.type in ("cpu", "cuda"), "only CPU/CUDA are declared")
        if target.type == "cuda":
            _require(torch.cuda.is_available(), "CUDA is unavailable")
            target = torch.device("cuda", target.index if target.index is not None
                                  else torch.cuda.current_device())
        context = torch.cuda.device(target) if target.type == "cuda" else nullcontext()
        context.__enter__()
        context_entered = True
        native._parts(model)
        result["parameter_dtype_observations"] = native._adapter_guard(model)
        params = baseline.trainables(model)
        _require(bool(params), "no trainable adapters")
        _require(all(param.device == target for param in model.parameters()),
                 "model parameters are not on the declared resident device")
        _optimizer_guard(optimizer, scheduler, params)
        signatures = _frozen_signatures(model)
        plan = _policy_plan(model, info, method=method, policy=policy,
                            oracle=oracle, device_type=target.type)
        common = _common_plan(model, info, params, method=method, device_type=target.type,
                              capture=capture_gradients, optimizer=optimizer)
        reference = _policy_plan(
            model, info, method="l2l", policy=CodecPolicy("bf16", "none"),
            oracle=False, device_type=target.type,
        )
        result["memory_accounting"] = _memory_accounting(plan, reference)
        result["memory_accounting"]["policy_plan"] = plan
        result["memory_accounting"]["common_trainer_plan"] = common
        if plan["device_scratch_bytes"] + plan["device_persistent_bytes"] > device_scratch_limit:
            raise MemoryError("combined codec/router scratch and persistent bound exceeds cap")
        if common["native_call_capture_persistent_bound_bytes"] + common[
            "native_call_capture_scratch_bound_bytes"
        ] > _COMMON_CAPTURE_LIMIT:
            raise MemoryError("common native call-capture bound exceeds its separate 1GiB cap")
        budget = HostMemoryBudget(host_limit)
        common_device = sum(common[key] for key in (
            "input_tensor_bytes", "native_call_capture_persistent_bound_bytes",
            "native_call_capture_scratch_bound_bytes", "l2l_hidden_upstream_bound_bytes",
            "trainable_gradient_bytes", "optimizer_state_bound_bytes",
            "optimizer_operation_scratch_bound_bytes", "device_diagnostic_scratch_bound_bytes",
        )) if target.type == "cuda" else 0
        result["memory_accounting"]["common_trainer_device_bound_bytes"] = common_device
        result["memory_accounting"]["common_trainer_device_limit_bytes"] = (
            _COMMON_TRAINER_DEVICE_LIMIT
        )
        if common_device > _COMMON_TRAINER_DEVICE_LIMIT:
            raise MemoryError("common trainer tensor bound exceeds its separate 2GiB device cap")
        if common["host_reservation_bytes"] + plan["host_bytes"] > budget.available_bytes:
            raise MemoryError("complete common/codec/router host reservation exceeds host_limit")
        lease = budget.reserve("stage2.common-tensors", common["host_reservation_bytes"])
        memory = MemoryObservation(target.type, scope=(
            "One resident model and supplied initial snapshot. Includes input tensor creation/H2D, "
            "admission/allocation, native mask/RoPE capture, policy transfers, forward/backward, "
            "raw pre-clip CPU gradient evidence/hashing, clip, optimizer/scheduler and cleanup. "
            "Excludes model loading, entry snapshot/RNG restoration, data-manifest preparation "
            "and later safetensors/JSON serialization. RSS is sampled; allocator caches remain."
        ))
        memory.__enter__()
        memory_entered = True
        before = _parameter_hashes(params)
        result["adapter_hashes_before"] = before
        for mb in info["window"]["micro_batches"]:
            for key, destination in (("input_ids", batches), ("labels", labels)):
                host_tensor = torch.tensor(mb[key], dtype=torch.long, device="cpu")
                _require(key == "labels" or bool((host_tensor < model.config.vocab_size).all()),
                         "input ID exceeds the unchanged native vocabulary")
                destination.append(host_tensor.to(target, non_blocking=False))
                host_tensor = None
        shapes = [(*batch.shape, model.config.hidden_size) for batch in batches]
        capture_started = time.perf_counter()
        calls = native.capture_calls(model, batches) if method == "l2l" else None
        captured_bytes = _captured_storage(calls)
        result["native_call_capture_owned_storage_bytes"] = captured_bytes
        _require(captured_bytes <= common["native_call_capture_persistent_bound_bytes"],
                 "captured native call storage exceeded its bound")
        _device_sync(target)
        result["native_call_capture_seconds"] = time.perf_counter() - capture_started
        probe = RouteProbe(model, shapes, budget, oracle=oracle, strict=True)
        if method == "l2l":
            store = make_store(
                policy, host_budget=budget, n_layers=model.config.num_hidden_layers,
                mb_shapes=shapes, dtype=torch.bfloat16, device=target, pin=target.type == "cuda",
            )
        _require(probe.report()["allocation"] == plan["route"], "routing allocation plan drift")
        if store is not None:
            _require(store.report()["allocation"] == plan["store"], "codec allocation plan drift")
        result["charged_host_bytes"] = plan["host_bytes"]
        result["total_host_reservation_bytes"] = budget.used_bytes
        phase = "forward_backward"
        forward_started = time.perf_counter()
        if method == "ga":
            losses = native.ga_step(model, batches, labels, probe=probe)
        else:
            losses = native.l2l_step(model, batches, labels, calls, store, probe=probe)
        _device_sync(target)
        result["forward_backward_seconds"] = time.perf_counter() - forward_started
        result["losses"] = losses
        _require(len(losses) == len(batches) and all(math.isfinite(loss) for loss in losses),
                 "missing/nonfinite complete-window loss")
        result["forward_backward_complete"] = True
        result["window_loss"] = math.fsum(losses)
        _require(math.isfinite(result["window_loss"]), "nonfinite accumulated window loss")
        result["routing"] = probe.report()
        route = result["routing"]
        _require(route["routing_pass"] and route["complete"] and route["admissible"]
                 and not route["pending_oracle_replays"], "incomplete or refused routing/VJP")
        if oracle:
            _require(route["natural_rmr"] is None and route["forced_rmr"] == 1.0
                     and route["oracle_vjp_calls"] == route["expected_pairs"],
                     "Oracle requires every saved-router input VJP, not only forced indices")
        else:
            _require(route["natural_admissible"] and route["natural_rmr"] == 1.0,
                     "BF16 controls require strict natural ordered routing")
        if store is not None:
            result["store"] = store.report()
            _require(result["store"]["put_calls"] == result["store"]["get_calls"]
                     == model.config.num_hidden_layers * len(batches),
                     "incomplete codec boundary replay")
            _require(result["store"]["owned_host_storage_bytes"] <= plan["store"][
                "charged_host_bytes"
            ], "codec owned host storage exceeds reservation")
            store.end_step()
        _require(route["owned_host_storage_bytes"] <= sum(plan["route"]["host"].values()),
                 "routing owned host storage exceeds reservation")
        phase = "gradient_evidence"
        manifest, gradients, raw_norm = _gradient_evidence(params, capture=capture_gradients)
        result.update(gradient_manifest=manifest, raw_gradient_l2=raw_norm,
                      missing_gradient_tensors=[], zero_gradient_tensors=[], gradients_finite=True,
                      gradient_capture_scope="raw pre-clip CPU FP32; full-window target weighting")
        _check_frozen(model, signatures)
        # No parameter update is allowed after a policy cleanup failure.
        close_policy()
        _require(not cleanup_errors, "policy cleanup failed before optimizer update")
        result["learning_rates_before"] = (
            [group["lr"] for group in optimizer.param_groups] if optimizer is not None else []
        )
        if optimizer is not None:
            phase = "optimizer"
            clipped = torch.nn.utils.clip_grad_norm_(
                list(params.values()), 1.0, error_if_nonfinite=True, foreach=False,
            )
            result["clip_returned_norm"] = float(clipped.detach().cpu())
            result["clip_max_norm"] = 1.0
            result["post_clip_gradient_l2"] = math.sqrt(math.fsum(
                float(param.grad.detach().double().norm().cpu()) ** 2 for param in params.values()
            ))
            _require(math.isfinite(result["post_clip_gradient_l2"])
                     and result["post_clip_gradient_l2"] <= 1.00001,
                     "clipped gradient norm exceeds the declared finite bound")
            result["optimizer_step_attempted"] = True
            optimizer.step()
            result["optimizer_step_returned"] = True
            result["optimizer_step"] = True
            result["optimizer_steps_completed"] = 1
            if scheduler is not None:
                scheduler.step()
            _device_sync(target)
        after = _parameter_hashes(params)
        result["adapter_hashes_after"] = after
        result["updated_parameter_count"] = sum(after[name] != before[name] for name in params)
        result["learning_rates_after"] = (
            [group["lr"] for group in optimizer.param_groups] if optimizer is not None else []
        )
        for name, param in params.items():
            _require(bool(torch.isfinite(param).all()), f"nonfinite updated adapter: {name}")
            if optimizer is not None:
                state = optimizer.state[param]
                _require(set(state) == {"step", "exp_avg", "exp_avg_sq"},
                         f"incomplete/unexpected AdamW state: {name}")
                _require(all(bool(torch.isfinite(value).all()) for value in state.values()),
                         f"nonfinite AdamW state: {name}")
        _check_frozen(model, signatures)
        _require(optimizer is not None or after == before, "gradient-only window changed adapters")
        result["frozen_base_invariants"] = {
            "pass": True, "tensors": len(signatures), "requires_grad": False,
            "no_gradients": True, "identity_storage_version_unchanged": True,
            "proof_scope": "tensor identity, storage pointer and mutation version; no base copy",
        }
        result["status"] = "MEASURED"
    except BaseException as error:
        result["status"] = "REFUSED" if phase == "admission" else "ERROR"
        result["mandatory_admission_failure"] = True
        result["failure_phase"] = phase
        result["error"] = _error(error)
        result["traceback"] = traceback.format_exc()
        traceback.clear_frames(error.__traceback__)
        if params and "adapter_hashes_before" in result:
            try:
                after_failure = _parameter_hashes(params)
                result["adapter_hashes_after_failure"] = after_failure
                result["updated_parameter_count"] = sum(
                    after_failure[name] != result["adapter_hashes_before"][name] for name in params
                )
            except BaseException as evidence_error:
                result["partial_update_evidence_error"] = _error(evidence_error)
                traceback.clear_frames(evidence_error.__traceback__)
        result["partial_optimizer_update_possible"] = bool(
            result.get("optimizer_step_attempted") and not result.get("optimizer_step_returned")
        )
        if not isinstance(error, Exception):
            fatal = error
    finally:
        cleanup_started = time.perf_counter()
        close_policy()
        for scope, check in (
            ("gradients", lambda: model.zero_grad(set_to_none=True) if cleared_gradients else None),
            ("frozen-base", lambda: _check_frozen(model, signatures)
             if signatures is not None else None),
        ):
            try:
                check()
            except BaseException as error:
                cleanup_errors.append({"scope": scope, **_error(error)})
                traceback.clear_frames(error.__traceback__)
        # A failed invariant check must not skip independent tensor release or drains.
        batches.clear()
        labels.clear()
        calls = host_tensor = None
        drained = False
        try:
            if target is not None:
                _device_sync(target)
            drained = True
        except BaseException as error:
            cleanup_errors.append({"scope": "device-drain", **_error(error)})
            traceback.clear_frames(error.__traceback__)
        if lease is not None and drained:
            try:
                lease.close()
            except BaseException as error:
                cleanup_errors.append({"scope": "common-lease", **_error(error)})
                traceback.clear_frames(error.__traceback__)
        used = budget.used_bytes if budget is not None else 0
        result["cleanup"] = {
            "complete": used == 0 and not cleanup_errors,
            "host_used_bytes": used, "errors": cleanup_errors,
            "returned_gradient_evidence_bytes": sum(
                value.numel() * value.element_size() for value in gradients.values()
            ),
            "returned_evidence_ownership": "caller; export reservation ends at ownership transfer",
            "lease_release_proves_os_rss_release": False,
            "seconds": time.perf_counter() - cleanup_started,
        }
        if used or cleanup_errors:
            result["status"] = "ERROR"
            result["mandatory_admission_failure"] = True
        if memory is not None and memory_entered:
            try:
                memory.__exit__(None, None, None)
            except BaseException as error:
                result["memory_observation_error"] = _error(error)
                result["status"] = "ERROR"
                result["mandatory_admission_failure"] = True
            result["memory_observed"] = memory.result
        if context is not None and context_entered:
            try:
                context.__exit__(None, None, None)
            except BaseException as error:
                result["device_context_cleanup_error"] = _error(error)
                result["status"] = "ERROR"
                result["mandatory_admission_failure"] = True
        result["full_step_wall_seconds"] = time.perf_counter() - started
        result["timing_scope"] = {
            "includes": ["validation", "input tensor creation/H2D", "allocation/admission",
                         "native call capture", "policy transfers", "forward", "backward",
                         "raw-gradient CPU evidence and hashes", "clip", "optimizer", "scheduler",
                         "policy/common cleanup", "synchronization"],
            "excludes": ["model loading", "entry snapshot/RNG restoration",
                         "data preparation", "later safetensors/JSON serialization"],
            "diagnostics_included": True, "data_transfers_included": True,
        }
    if fatal is not None:
        fatal.p3_stage2_record = result
        raise fatal
    return result, gradients


def _snapshot_guard(model: Any, initial: dict[str, Any]) -> None:
    import torch

    params = baseline.trainables(model)
    _require(params.keys() == initial.keys(), "paired initial snapshot parameter set changed")
    for name, param in params.items():
        value = initial[name]
        _require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                 and value.dtype == torch.float32 and value.shape == param.shape
                 and not value.requires_grad and bool(torch.isfinite(value).all()),
                 f"invalid paired initial FP32 CPU snapshot: {name}")


def _save_tensors(
    path: Path, tensors: dict[str, Any], *, metadata: dict[str, str],
) -> dict[str, Any]:
    from safetensors.torch import save_file

    _require(bool(tensors) and not path.exists(), "empty evidence or existing evidence destination")
    manifest = baseline.gradient_manifest(tensors)
    _require(all(item["finite"] for item in manifest.values()), "nonfinite checkpoint/evidence")
    save_file(tensors, str(path), metadata=metadata)
    return {"file": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size,
            "tensor_manifest": manifest}


def _optimizer_evidence(
    params: dict[str, Any], optimizer: Any, step: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    names = {id(param): name for name, param in params.items()}
    tensors = {}
    for name, param in params.items():
        state = optimizer.state.get(param)
        _require(state is not None and set(state) == {"step", "exp_avg", "exp_avg_sq"},
                 f"missing AdamW evidence: {name}")
        _require(state["step"].item() == step, f"AdamW step count differs: {name}")
        for key, value in state.items():
            _require(isinstance(value, torch.Tensor), "AdamW state must contain only tensors")
            tensors[f"{name}::{key}"] = value.detach().to(device="cpu", copy=True).contiguous()
    groups = []
    for group in optimizer.param_groups:
        copied = {key: value for key, value in group.items() if key != "params"}
        copied["params"] = [names[id(param)] for param in group["params"]]
        groups.append(copied)
    return tensors, groups


def _save_step(
    output: Path, *, params: dict[str, Any], optimizer: Any, scheduler: Any,
    gradients: dict[str, Any], step: int, arm: str, seed: int, protocol_sha256: str,
) -> dict[str, Any]:
    from soup_cli.utils.host_memory_budget import HostMemoryBudget

    elements = sum(param.numel() for param in params.values())
    maximum = max(param.numel() for param in params.values())
    # Incoming raw gradients, adapter/state copies, manifest temporaries and serializer copies.
    bound = 96 * elements + 40 * maximum + 4 * 1024**2
    budget = HostMemoryBudget(_EVIDENCE_LIMIT)
    lease = budget.reserve("stage2.safe-evidence", bound)
    adapters = state = None
    try:
        adapters = {name: param.detach().to(device="cpu", copy=True).contiguous()
                    for name, param in params.items()}
        state, groups = _optimizer_evidence(params, optimizer, step)
        metadata = {"arm": arm, "seed": str(seed), "optimizer_step": str(step),
                    "protocol_sha256": protocol_sha256}
        prefix = f"step-{step:03d}"
        result = {
            "gradients": _save_tensors(
                output / f"{prefix}.gradients.safetensors", gradients,
                metadata={**metadata, "kind": "raw-pre-clip-gradients"},
            ),
            "adapters": _save_tensors(
                output / f"{prefix}.adapters.safetensors", adapters,
                metadata={**metadata, "kind": "updated-adapter-evaluation-eligibility-in-arm-json"},
            ),
            "optimizer": _save_tensors(
                output / f"{prefix}.optimizer.safetensors", state,
                metadata={**metadata, "kind": "named-AdamW-state"},
            ),
            "scheduler_state": scheduler.state_dict(), "evaluation_eligible": False,
            "host_export_bound_bytes": bound, "host_export_limit_bytes": _EVIDENCE_LIMIT,
        }
        result["optimizer"]["param_groups"] = groups
        result["cleanup"] = {"host_used_bytes_after_export": 0, "complete": True}
        return result
    except BaseException as error:
        traceback.clear_frames(error.__traceback__)
        raise
    finally:
        adapters = state = None
        lease.close()
        _require(budget.used_bytes == 0, "safe evidence export leaked a host reservation")


def _protocol(arm: str) -> dict[str, Any]:
    return {
        "schema": "p3-stage2-training-v1", "arm": arm, "experimental_only": True,
        "base_dtype": "torch.bfloat16", "adapter_dtype": "torch.float32",
        "optimizer_step": True, "optimizer": {"name": "AdamW", "lr": 2e-4,
            "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01,
            "foreach": False, "fused": False, "clip_max_norm": 1.0},
        "scheduler": {"name": "transformers.get_cosine_schedule_with_warmup",
                      "num_warmup_steps": 2, "num_training_steps": _STEPS},
        "optimizer_steps_planned": _STEPS, "input_tokens_per_training_arm": 262144,
        "window_input_tokens": 4096, "loss": "native CE; complete-window shifted target count",
        "fp16_cast": False, "inference_backend_changed": False,
        "stored_dtype_proves_t4_arithmetic_backend": False,
        "t4_kernel_accumulation_evidence": "parent-owned external prerequisite",
        "oracle_forced_agreement_is_natural_rmr": False,
        "oracle_forced_agreement_is_vjp_proof": False,
        "production_numerical_routing_guards_changed": False,
    }


def _run_arm(
    model: Any, windows: Sequence[dict], initial: dict, *, seed: int, arm: str,
    output: Path, device: str, protocol_sha256: str, preflight: bool,
) -> dict[str, Any]:
    output = Path(output)
    result: dict[str, Any] = {
        "schema": "p3-stage2-arm-v1", "status": "STARTED", "arm": arm, "seed": seed,
        "preflight": preflight, "protocol_sha256": protocol_sha256,
        "supplied_window_count": len(windows) if isinstance(windows, Sequence) else None,
        "consumed_window_count": 0, "optimizer_steps_completed": 0, "windows": [],
        "optimizer_step": False,
        "final_checkpoint_evaluation_eligible": False,
        "frozen_base_invariants_pass": None,
        "optimizer_state_initially_empty": None, "adapter_update_proved_after_warmup": False,
        "cross_arm_admission": {"status": "EXTERNAL_COMPARISON_REQUIRED",
            "required": ["matched BF16 GA/L2L/Oracle losses", "raw pre-clip gradients",
                         "updated adapters", "optimizer states", "router input VJP"],
            "finite_fp8_old_gate_failure_is_not_numerical_admission": True},
        "source_domain_overlap_admission": "external frozen manifest bound by protocol_sha256",
        "production_admission": False, "quality_admission": False,
    }
    optimizer = scheduler = None
    params: dict[str, Any] = {}
    gradients: dict[str, Any] = {}
    signatures = None
    own_output = False
    cleanup_errors = []
    fatal = None
    started = time.perf_counter()
    try:
        if output.exists() and any(output.iterdir()):
            raise ValueError("output must be a fresh empty arm directory; evidence is immutable")
        output.mkdir(parents=True, exist_ok=True)
        own_output = True
        infos = _study_windows(windows, preflight=preflight)
        if arm not in _ARMS or not preflight and arm not in (
            "bf16_l2l", "oracle_block64_e4m3",
        ):
            raise ValueError("unapproved training/preflight arm")
        if type(seed) is not int or seed not in _SEEDS:
            raise ValueError("seed must be one of the five approved paired seeds")
        if not _hex_digest(protocol_sha256):
            raise ValueError("protocol_sha256 must be a lowercase SHA-256 digest")
        native._parts(model)
        native._adapter_guard(model)
        _snapshot_guard(model, initial)
        import torch

        target = torch.device(device)
        _require(target.type in ("cpu", "cuda"), "only CPU/CUDA are declared")
        if target.type == "cuda":
            _require(torch.cuda.is_available(), "CUDA is unavailable")
            target = torch.device("cuda", target.index if target.index is not None
                                  else torch.cuda.current_device())
        _require(all(param.device == target for param in model.parameters()),
                 "arm device differs from the supplied resident model")
        _require(all(token < model.config.vocab_size for info in infos
                     for mb in info["window"]["micro_batches"] for row in mb["input_ids"]
                     for token in row), "frozen window exceeds the unchanged native vocabulary")
        arm_export_bound = (
            96 * sum(param.numel() for param in baseline.trainables(model).values())
            + 40 * max(param.numel() for param in baseline.trainables(model).values())
            + 4 * 1024**2
        )
        if arm_export_bound > _EVIDENCE_LIMIT:
            raise MemoryError("safe evidence export bound exceeds the fixed 256MiB cap")
        signatures = _frozen_signatures(model)
        _check_frozen(model, signatures)
        params = baseline.trainables(model)
        baseline.restore(model, initial, seed)
        result["entry_snapshot_restored"] = True
        result["entry_rng_reset_to_seed"] = seed
        result["initial_adapter_manifest"] = baseline.gradient_manifest(initial)
        expected_initial = {
            name: item["sha256"] for name, item in result["initial_adapter_manifest"].items()
        }
        _require(_parameter_hashes(params) == expected_initial,
                 "initial snapshot restore byte mismatch")
        result["protocol"] = _protocol(arm)
        result["planned_window_count"] = len(infos)
        result["input_token_count_planned"] = sum(info["input_token_count"] for info in infos)
        result["valid_target_count_planned"] = sum(info["valid_target_count"] for info in infos)
        result["ordered_window_pairing_sha256"] = hashlib.sha256(b"".join(
            bytes.fromhex(info["pairing_sha256"]) for info in infos
        )).hexdigest()
        metadata = getattr(model, "_p3_realdata_metadata", {})
        result["model_provenance"] = {key: metadata[key] for key in (
            "repo_id", "revision", "source", "library_versions", "lora",
            "parameter_dtype_observations",
        ) if key in metadata}
        optimizer, scheduler = _new_optimizer(params)
        _require(not optimizer.state, "fresh arm inherited optimizer states")
        result["optimizer_state_initially_empty"] = True
        method, oracle, policy = _ARMS[arm]
        for index, info in enumerate(infos, 1):
            record, gradients = run_window(
                model, info["window"], method=method, policy=policy, oracle=oracle,
                device=device, optimizer=optimizer, scheduler=scheduler, capture_gradients=True,
            )
            record["optimizer_step_index"] = index
            result["windows"].append(record)
            result["optimizer_steps_completed"] += record["optimizer_steps_completed"]
            if record["status"] != "MEASURED" or not record["cleanup"]["complete"]:
                baseline.write_json(output / f"step-{index:03d}.json", record)
                raise _GuardError(
                    f"window {index} failed or refused: {record.get('error', record['status'])}"
                )
            _require(record["pairing_sha256"] == info["pairing_sha256"]
                     and record["window_json_sha256"] == info["window_json_sha256"],
                     "frozen input/snapshot pairing changed during consumption")
            _require(record["optimizer_step"] and scheduler.last_epoch == index,
                     "optimizer/scheduler update count differs from the frozen schedule")
            serialization_started = time.perf_counter()
            record["evidence"] = _save_step(
                output, params=params, optimizer=optimizer, scheduler=scheduler,
                gradients=gradients, step=index, arm=arm, seed=seed,
                protocol_sha256=protocol_sha256,
            )
            record["evidence_serialization_wall_seconds"] = (
                time.perf_counter() - serialization_started
            )
            gradients.clear()
            baseline.write_json(output / f"step-{index:03d}.json", record)
            if index == 2:
                _require(any(rate > 0 for rate in record["learning_rates_before"])
                         and record["updated_parameter_count"] > 0,
                         "second warmup-aware optimizer step failed to change adapter bytes")
                result["adapter_update_proved_after_warmup"] = True
        _require(result["adapter_update_proved_after_warmup"],
                 "warmup-only preflight did not prove a parameter update")
        _check_frozen(model, signatures)
        result["input_token_count_consumed"] = sum(
            record["input_token_count"] for record in result["windows"]
        )
        result["valid_target_count_consumed"] = sum(
            record["valid_target_count"] for record in result["windows"]
        )
        _require(preflight or result["input_token_count_consumed"] == 262144,
                 "training did not consume exactly 262144 frozen inputs")
        final = result["windows"][-1]["evidence"]["adapters"]
        # Only this final-checkpoint pointer is eligible for final generation.
        result["final_adapter_file"] = final["file"]
        result["final_adapter_file_sha256"] = final["sha256"]
        result["final_adapter_manifest"] = final["tensor_manifest"]
        result["final_checkpoint_optimizer_step"] = result["optimizer_steps_completed"]
        result["final_optimizer_evidence"] = result["windows"][-1]["evidence"]["optimizer"]
        result["final_scheduler_state"] = scheduler.state_dict()
        result["frozen_base_invariants_pass"] = True
        result["status"] = "PREFLIGHT_EVIDENCE_READY" if preflight else "TRAINED_CHECKPOINT_READY"
    except BaseException as error:
        result["status"] = "ERROR" if result["windows"] else "REFUSED"
        result["error"] = _error(error)
        result["mandatory_admission_failure"] = True
        result["traceback"] = traceback.format_exc()
        if hasattr(error, "p3_stage2_record"):
            record = error.p3_stage2_record
            result["windows"].append(record)
            result["optimizer_steps_completed"] += record["optimizer_steps_completed"]
        traceback.clear_frames(error.__traceback__)
        if not isinstance(error, Exception):
            fatal = error
    finally:
        gradients.clear()
        if params:
            try:
                _device_sync(next(iter(params.values())).device)
            except BaseException as error:
                cleanup_errors.append({"scope": "arm-device-drain", **_error(error)})
                traceback.clear_frames(error.__traceback__)
            try:
                for param in model.parameters():
                    param.grad = None
                _check_frozen(model, signatures)
            except BaseException as error:
                cleanup_errors.append({"scope": "frozen-base/gradients", **_error(error)})
                result["frozen_base_invariants_pass"] = False
                traceback.clear_frames(error.__traceback__)
        if optimizer is not None:
            optimizer.state.clear()
            optimizer.param_groups.clear()
        optimizer = scheduler = None
        result["cleanup"] = {
            "complete": not cleanup_errors and all(
                record.get("cleanup", {}).get("complete", False) for record in result["windows"]
            ),
            "errors": cleanup_errors, "optimizer_owned_states_dropped": True,
            "initial_snapshot_restored_at_exit": False, "rng_reset_at_exit": False,
            "live_model_retains_measured_updated_adapters": bool(params),
            "failed_partial_updates_rolled_back": False,
            "partial_optimizer_update_possible": any(
                record.get("partial_optimizer_update_possible", False)
                for record in result["windows"]
            ),
            "lease_release_proves_os_rss_release": False,
        }
        if not result["cleanup"]["complete"]:
            result["status"] = "ERROR"
            result["mandatory_admission_failure"] = True
        result["arm_wall_seconds_with_serialization"] = time.perf_counter() - started
        result["consumed_window_count"] = sum(
            bool(record.get("forward_backward_complete")) for record in result["windows"]
        )
        result["attempted_window_count"] = len(result["windows"])
        result["optimizer_step"] = result["optimizer_steps_completed"] > 0
        result["input_token_count_consumed"] = sum(
            record.get("input_token_count", 0) for record in result["windows"]
            if record.get("forward_backward_complete")
        )
        result["valid_target_count_consumed"] = sum(
            record.get("valid_target_count", 0) for record in result["windows"]
            if record.get("forward_backward_complete")
        )
        result["completed_window_count"] = sum(
            record.get("status") == "MEASURED" for record in result["windows"]
        )
        result["final_checkpoint_evaluation_eligible"] = (
            not preflight and result["status"] == "TRAINED_CHECKPOINT_READY"
            and result["optimizer_steps_completed"] == _STEPS
            and result["completed_window_count"] == _STEPS
            and result["cleanup"]["complete"]
            and result["frozen_base_invariants_pass"] is True
        )
        result["full_training_step_wall_seconds"] = [
            record["full_step_wall_seconds"] for record in result["windows"]
            if "full_step_wall_seconds" in record
        ]
        if own_output:
            baseline.write_json(output / "arm.json", result)
    if fatal is not None:
        raise fatal
    return result


def run_preflight_arm(
    model: Any, windows: Sequence[dict], initial: dict, *, seed: int, arm: str,
    output: Path, device: str, protocol_sha256: str,
) -> dict:
    """Consume the first two windows with fresh states and the full 64-step scheduler."""
    return _run_arm(model, windows, initial, seed=seed, arm=arm, output=output,
                    device=device, protocol_sha256=protocol_sha256, preflight=True)


def train_arm(
    model: Any, windows: Sequence[dict], initial: dict, *, seed: int, arm: str,
    output: Path, device: str, protocol_sha256: str,
) -> dict:
    """Consume exactly 64 frozen windows and retain the final live adapter weights."""
    return _run_arm(model, windows, initial, seed=seed, arm=arm, output=output,
                    device=device, protocol_sha256=protocol_sha256, preflight=False)
