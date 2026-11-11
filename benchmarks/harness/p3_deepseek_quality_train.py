"""Paired DeepSeek-V3 BF16 L2L versus frozen block64 E4M3 plus Oracle training.

Production CLI requires the pinned source receipt, a frozen pack, one shared initial
adapter, a passing hash-admitted gradient gate and a frozen protocol. Example commands::

    python -m benchmarks.harness.p3_deepseek_quality_train initial-adapter \
        --source SOURCE --cache CACHE --output initial.safetensors --prepare
    python -m benchmarks.harness.p3_deepseek_quality_train \
        --source SOURCE --cache CACHE --pack PACK --pack-sha256 HEX \
        --initial-adapter initial.safetensors --arm bf16_l2l --length 1024 --steps 64 \
        --gate GATE/manifest.json --gate-sha256 HEX \
        --protocol protocol.json --protocol-sha256 HEX --output NEW_DIRECTORY

``--pilot-prefreeze`` replaces the protocol arguments and measures one LR0 warmup
step, never endpoint quality. Only externally documented instance loss may resume;
scientific, memory, deadline and budget failures are terminal. ``--resume PRIOR_RUN``
requires ``--infrastructure-event EVENT.json`` in a fresh output directory. The event
contains event=instance_loss, run_identity_sha256, provider, instance_id, detected_at
and description. Checkpoints and final adapters commit their manifest last.
The final adapter uses the unchanged production save guard. Data retain the frozen
Granite domain cycling order, identity/cycle records and final document cutoff.
Every step records loss. ``adapter-delta.json`` reports initial-to-final global
Frobenius L2, the initial norm and relative change with file/parameter hash bindings.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import random
import re
import signal
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Sequence

from rich.console import Console

from benchmarks.harness import p3_deepseek_quality_attention as attention
from benchmarks.harness import p3_deepseek_quality_model as model_api
from benchmarks.harness.p3_deepseek_quality_profiles import get_profile

ARMS = ("bf16_l2l", "oracle_block64_e4m3")
OPTIMIZER = {
    "name": "AdamW", "lr": 2e-4, "betas": [0.9, 0.999], "eps": 1e-8,
    "weight_decay": 0.01, "clip_max_norm": 1.0,
}
_CHECKPOINT = re.compile(r"step-([0-9]{6})")


class WindowFailureError(RuntimeError):
    """A failed window retains detached JSON evidence, never permission to retry."""

    def __init__(self, record: dict[str, Any]) -> None:
        self.record = record
        error = record.get("error", {})
        super().__init__(f"{error.get('type', 'WindowFailureError')}: {error.get('message', '')}")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def _object_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _require_profile(value: dict, profile_id: str, context: str) -> None:
    get_profile(profile_id)
    if not isinstance(value, dict) or value.get("profile_id", "deepseek-v3") != profile_id:
        raise ValueError(f"{context} profile differs from selected profile")


def _require_moonlight_budget(value: dict) -> None:
    budget = value.get("budget")
    if not isinstance(budget, dict):
        raise ValueError("Moonlight frozen budget record required")

    def finite_cost(cost: Any) -> bool:
        return type(cost) in (int, float) and math.isfinite(cost) and cost >= 0

    upper = budget.get("upper_bound_total_usd")
    adjustments = budget.get("adjustments")
    if (any(type(budget.get(name)) is not int for name in (
                "limit_usd", "compute_stop_usd", "closeout_reserve_usd"))
            or budget.get("limit_usd") != 15 or budget.get("compute_stop_usd") != 14
            or budget.get("closeout_reserve_usd") != 1
            or not finite_cost(upper) or upper > 15 or not isinstance(adjustments, list)):
        raise ValueError(
            "Moonlight frozen budget must retain the $15 total and $1 closeout reserve")
    lengths = value.get("training", {}).get("lengths")
    if lengths not in ([1024, 4096], [1024]):
        raise ValueError("Moonlight protocol must retain L1024 and only budget-omit the L4096 pair")
    expected_actions = (["hellaswag_prefix", "omit_L4096_pair"] if lengths == [1024]
                        else ["hellaswag_prefix"] if adjustments else [])
    if [item.get("action") if isinstance(item, dict) else None
        for item in adjustments] != expected_actions:
        raise ValueError(
            "Moonlight budget cuts must reduce HellaSwag before omitting the L4096 pair")
    previous = None
    for item in adjustments:
        before, after = item.get("upper_bound_before_usd"), item.get("upper_bound_after_usd")
        if (not finite_cost(before) or not finite_cost(after) or before <= 15
                or after > before or (previous is not None and before != previous)):
            raise ValueError("Moonlight budget adjustment estimates are missing or inconsistent")
        if item["action"] == "hellaswag_prefix":
            count = item.get("after_n")
            if (item.get("before_n") != 10042 or type(count) is not int or not 0 < count < 10042
                    or value.get("loglik", {}).get("hellaswag_n") != count):
                raise ValueError("Moonlight HellaSwag prefix cut differs from frozen scoring count")
        elif after > 15:
            raise ValueError("Moonlight omitted-length budget must fit the approved total")
        previous = after
    if previous is not None and upper != previous:
        raise ValueError("Moonlight final budget does not match its last ordered adjustment")
    if not adjustments and value.get("loglik", {}).get("hellaswag_n") != 10042:
        raise ValueError("Moonlight HellaSwag reduction requires a frozen budget adjustment")


def require_study_profile(value: dict, *, profile_id: str = "deepseek-v3") -> None:
    """Keep Moonlight's diagnostic gate and fixed paired study independent of V3 defaults."""
    profile = get_profile(profile_id)
    _require_profile(value, profile_id, "frozen protocol")
    if profile_id == "deepseek-v3":
        return
    training = value.get("training", {})
    if (value.get("numerical_gate_blocks_quality") is not False
            or value.get("arms") != list(ARMS)
            or not isinstance(training, dict)
            or training.get("seed") != 23 or type(training.get("steps")) is not int
            or training["steps"] != 64
            or training.get("input_tokens_per_step") != 4096
            or type(training.get("micro_batch_size")) is not int
            or training.get("micro_batch_size") != 1
            or training.get("optimizer") != OPTIMIZER
            or training.get("scheduler") != {
                "name": "cosine", "warmup_steps": 2, "horizon": 64}
            or value.get("model") != {"model_id": profile.model_id,
                                      "revision": profile.revision}):
        raise ValueError("Moonlight protocol requires both diagnostic-gate arms and 64 fixed steps")
    _require_moonlight_budget(value)
    admission = model_api.native_aux_admission(profile_id)
    if admission is None or value.get("aux_admission") != admission:
        raise ValueError("Moonlight protocol lacks the approved auxiliary admission binding")
    for name in ("source", "cache", "tokenizer"):
        _require_profile(value.get(name), profile_id, f"Moonlight protocol {name}")
    if value["source"].get("aux_admission") != admission:
        raise ValueError("Moonlight protocol source auxiliary admission differs")


def _gate_admission(result: dict, digest: str, *, profile_id: str = "deepseek-v3") -> dict:
    """Preserve the verdict returned by load_gate, never turn a diagnostic failure into PASS."""
    _require_profile(result, profile_id, "gradient gate")
    verdict = result.get("verdict")
    allowed = ("PASS",) if profile_id == "deepseek-v3" else ("PASS", "FAIL", "NO_VERDICT")
    if verdict not in allowed:
        raise ValueError("gradient gate numerical verdict is not admitted for this profile")
    return {"profile_id": profile_id, "manifest_sha256": digest, "overall": verdict,
            "environment": result["environment"], "attention_policy": result["attention_policy"]}


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(_canonical(row).decode() + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _save_tensors(
    path: Path, values: dict[str, Any], *, schema: str, profile_id: str = "deepseek-v3",
) -> dict[str, str]:
    from safetensors.torch import save

    tensors = {name: value.detach().cpu().contiguous() for name, value in values.items()}
    payload = save(tensors, metadata={"schema": schema, "profile_id": profile_id})
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return {"profile_id": profile_id, "path": path.name, "sha256": model_api.sha256(path)}


def _error(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _finite(value: Any, context: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite {context}")
    return result


def adapter_delta(model: Any, initial_adapter: Path) -> dict[str, Any]:
    """Global Frobenius norms over every strict FP32 attention trainable, in FP64."""
    initial_adapter = Path(initial_adapter)
    initial_file_hash = model_api.sha256(initial_adapter)
    initial = model_api._load_adapter_tensors(model, initial_adapter)
    if model_api.sha256(initial_adapter) != initial_file_hash:
        raise ValueError("initial adapter changed during final delta measurement")
    parameters = model_api.trainable_parameters(model)
    difference_squared, initial_squared = [], []
    for name, parameter in parameters.items():
        final = parameter.detach().cpu().double()
        before = initial[name].double()
        difference_squared.append(_finite((final - before).square().sum(), "adapter delta"))
        initial_squared.append(_finite(before.square().sum(), "initial adapter norm"))
    delta = math.sqrt(math.fsum(difference_squared))
    initial_norm = math.sqrt(math.fsum(initial_squared))
    return {
        "metric": "global_frobenius_l2", "accumulation_dtype": "float64",
        "delta_l2": delta, "initial_l2": initial_norm,
        "relative_l2": delta / initial_norm if initial_norm else None,
        "initial_adapter_sha256": initial_file_hash,
        "initial_parameter_sha256": model_api.adapter_digest(initial)[0],
        "final_parameter_sha256": model_api.adapter_digest(parameters)[0],
        "tensor_count": len(parameters),
        "element_count": sum(value.numel() for value in parameters.values()),
    }


def new_optimizer(parameters: dict[str, Any], steps: int) -> tuple[Any, Any]:
    """Frozen Granite AdamW and HF cosine schedule, including the initial LR0 update."""
    import torch
    from transformers import get_cosine_schedule_with_warmup

    if type(steps) is not int or steps < 1 or not parameters:
        raise ValueError("optimizer requires positive scheduler horizon and trainable parameters")
    optimizer = torch.optim.AdamW(
        list(parameters.values()), lr=OPTIMIZER["lr"], betas=tuple(OPTIMIZER["betas"]),
        eps=OPTIMIZER["eps"], weight_decay=OPTIMIZER["weight_decay"],
        foreach=False, fused=False,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=2, num_training_steps=steps,
    )
    return optimizer, scheduler


def _gradients(
    parameters: dict[str, Any], capture: bool, *, profile_id: str = "deepseek-v3",
) -> tuple[dict, dict, float]:
    import torch
    diagnostic_zeros = get_profile(profile_id).profile_id == "moonlight-16b-a3b"

    evidence, copies = {}, {}
    squared = []
    for name, parameter in parameters.items():
        gradient = parameter.grad
        if (gradient is None or gradient.is_sparse or gradient.dtype != torch.float32
                or gradient.shape != parameter.shape or gradient.device != parameter.device):
            raise ValueError(f"missing or invalid FP32 gradient: {name}")
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError(f"nonfinite gradient: {name}")
        nonzero = int(torch.count_nonzero(gradient))
        if nonzero == 0 and not diagnostic_zeros:
            raise ValueError(f"zero attention adapter gradient: {name}")
        value = gradient.detach().cpu().contiguous()
        _, identity = model_api.adapter_digest({name: value})
        norm_squared = _finite(value.double().square().sum(), f"gradient norm: {name}")
        squared.append(norm_squared)
        evidence[name] = {**identity[name], "nonzero": nonzero,
                          "l2": math.sqrt(norm_squared), "finite": True}
        if capture:
            copies[name] = value.clone()
    norm = math.sqrt(math.fsum(squared))
    if not math.isfinite(norm) or (norm <= 0 and not diagnostic_zeros):
        raise ValueError("nonfinite or zero global gradient norm")
    return evidence, copies, norm


def _cgroup_memory() -> dict[str, Any]:
    from benchmarks.harness.host_box import cgroup_dirs, cgroup_v1_dirs, ram_limit

    result = ram_limit()
    proc = Path("/proc/self/cgroup")
    peaks = []
    if proc.is_file():
        text = proc.read_text(encoding="utf-8")
        for directory in cgroup_dirs(text):
            path = Path(directory) / "memory.peak"
            if path.is_file():
                raw = path.read_text(encoding="utf-8").strip()
                if raw.isdigit():
                    peaks.append({"path": str(path), "bytes": int(raw)})
        if not peaks:
            for directory in cgroup_v1_dirs(text):
                path = Path(directory) / "memory.max_usage_in_bytes"
                if path.is_file():
                    raw = path.read_text(encoding="utf-8").strip()
                    if raw.isdigit():
                        peaks.append({"path": str(path), "bytes": int(raw)})
    return {**result, "observed_hierarchy_peaks": peaks,
            "peak_scope": "cgroup lifetime, not reset or added to RSS/CUDA"}


def _available_host_bytes() -> int:
    import psutil

    from benchmarks.harness.host_box import ram_limit

    available = int(psutil.virtual_memory().available)
    limit = ram_limit()
    if limit["limit_bytes"] is not None and limit["current_bytes"] is not None:
        available = min(available, max(0, limit["limit_bytes"] - limit["current_bytes"]))
    if available < 1:
        raise MemoryError("no available host memory for boundary and routing stores")
    return available


def _activation_storage_evidence(storage: dict, oracle: dict | None) -> dict:
    """Separate observed tensor capacities from reservation and temporary-buffer bounds."""
    components = {
        name: {"host_bytes": 0, "device_bytes": 0, "pinned_host_bytes": 0}
        for name in ("payload_including_padding", "scales", "calibration_vectors")
    }
    for observed in storage["actual_owned_storages"]:
        name = observed["name"]
        component = ("payload_including_padding" if name.startswith("payload.")
                     else "scales" if name == "scales" else "calibration_vectors")
        location = "host_bytes" if observed["device"] == "cpu" else "device_bytes"
        components[component][location] += observed["storage_bytes"]
        if observed["device"] == "cpu" and observed["pinned"]:
            components[component]["pinned_host_bytes"] += observed["storage_bytes"]
    expected = {
        "payload_including_padding": storage["payload_bytes"] + storage["padding_bytes"],
        "scales": storage["scale_bytes"],
    }
    for name, wanted in expected.items():
        observed = components[name]["host_bytes"] + components[name]["device_bytes"]
        if observed != wanted:
            raise ValueError(f"observed activation storage disagrees with allocation: {name}")
    return {
        "measurement": "unique tensor untyped_storage.nbytes; views counted once",
        "scope": "fully populated persistent decoder-boundary store before close",
        "components": components,
        "host_tensor_storage_bytes": storage["owned_host_storage_bytes"],
        "device_tensor_storage_bytes": storage["owned_device_storage_bytes"],
        "host_reservation_upper_bound_bytes": storage["charged_host_bytes"],
        "pinned_allocator_rounding_is_estimate": True,
        "codec_temporary_upper_bounds": {
            "host_bytes": storage["host_scratch_bytes"],
            "device_bytes": storage["device_scratch_bytes"],
            "decoded_output_bytes_included": storage["decoded_output_peak_bytes"],
            "measured_peak": False,
        },
        "oracle_owned_storage_after_backward": None if oracle is None else {
            "host_bytes": oracle["owned_host_storage_bytes"],
            "device_bytes": oracle["owned_device_storage_bytes"],
            "host_reservation_upper_bound_bytes": oracle["host_reservation_bound_bytes"],
            "device_scratch_upper_bound_bytes": oracle["device_scratch_bound_bytes"],
        },
        "total_activation_peak_measured": False,
        "excluded": [
            "captured call metadata, live hidden states and activation gradients",
            "decoder/head autograd intermediates and kernel workspaces",
            "weights, optimizer, routing guard and allocator caches",
        ],
        "whole_process_RSS_and_CUDA_peaks_are_not_activation_attributed": True,
    }


def run_window(
    session: model_api.ModelSession, window: dict, *, arm: str, optimizer: Any = None,
    scheduler: Any = None, capture_gradients: bool = False, host_limit_bytes: int | None = None,
    profile_id: str = "deepseek-v3",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One real production L2L forward/backward and optional validated AdamW update.

    Arbitrary microbatch lengths are admitted here for explicit research qualification.
    ``run_training`` enforces the study's complete 4096-token layouts. Both arms pass
    identical response labels and the complete-window shifted-target denominator.
    """
    import torch

    from benchmarks.harness import p3_deepseek_oracle as oracle_module
    from benchmarks.harness import p3_realdata_codec as codec
    from benchmarks.harness.p3_deepseek_fp8_smoke import _stepper
    from benchmarks.harness.p3_fp8_smoke import MemoryObservation
    from benchmarks.harness.p3_stage2_training import _inspect_window
    from soup_cli.utils.host_memory_budget import HostMemoryBudget
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call, model_parts

    record: dict[str, Any] = {
        "profile_id": profile_id, "status": "FAILED", "arm": arm, "optimizer_applied": False,
        "attention_backend": attention.MAIN_BACKEND, "autocast": False,
        "routing_guard": {"enabled": True, "implementation": "production unchanged",
                          "outcome": "NOT_COMPLETED", "natural_routing_measured": False},
    }
    gradients: dict[str, Any] = {}
    store = probe = lease = budget = memory = None
    observation = None
    cleanup_errors = []
    failure = None
    started = time.perf_counter()
    step = None
    ids: list[Any] = []
    labels: list[Any] = []
    calls: list[Any] = []
    try:
        _require_profile(session.metadata, profile_id, "training session")
        if arm not in ARMS:
            raise ValueError("unknown quality arm")
        if session.closed or session.metadata["precision"]["compute_dtype"] != "bfloat16":
            raise ValueError("paired training requires an open BF16-compute session")
        target = str(session.runtime.device)
        record["attention_policy"] = attention.policy_identity(profile_id)
        record["hardware"] = attention.require_ampere_cuda(target)
        if scheduler is not None and optimizer is None:
            raise ValueError("scheduler requires its optimizer")
        if scheduler is not None and scheduler.optimizer is not optimizer:
            raise ValueError("scheduler belongs to a different optimizer")
        parameters = model_api.trainable_parameters(session.model)
        if optimizer is not None:
            owned = [value for group in optimizer.param_groups for value in group["params"]]
            if len(owned) != len(parameters) or {id(value) for value in owned} != {
                id(value) for value in parameters.values()
            }:
                raise ValueError("optimizer must own exactly the FP32 attention trainables")
            if len(optimizer.param_groups) != 1 or optimizer.param_groups[0]["lr"] < 0:
                raise ValueError("window optimizer requires one nonnegative-LR parameter group")
        info = _inspect_window(window)
        record.update({key: value for key, value in info.items() if key != "window"})
        device = torch.device(target).type
        _, owner = model_parts(session.model)
        layers = len(owner.layers)
        moe_layers = layers - int(session.config.first_k_dense_replace)
        if moe_layers < 1:
            raise ValueError("quality training requires at least one native MoE layer")
        shapes = [(*shape, int(session.config.hidden_size))
                  for shape in info["micro_batch_shapes"]]
        policy = (codec.CodecPolicy("bf16", "none") if arm == "bf16_l2l" else
                  codec.CodecPolicy("block64_e4m3", "e4m3", group_size=64))
        record["activation_codec"] = asdict(policy)
        budget = HostMemoryBudget(
            _available_host_bytes() if host_limit_bytes is None else host_limit_bytes,
        )
        topk = int(session.config.num_experts_per_tok)
        record_bytes = moe_layers * sum(shape[0] * shape[1] for shape in shapes) * topk * 8
        comparison_bytes = max(shape[0] * shape[1] for shape in shapes) * topk * 8
        lease = budget.reserve("quality.production-routing-guard", record_bytes + comparison_bytes)
        record["routing_guard"].update(record_bytes=record_bytes,
                                       comparison_bytes=comparison_bytes)
        session.model.train()
        session.model.zero_grad(set_to_none=True)
        memory = MemoryObservation(device, scope=(
            "One streamed model; inputs, capture, boundary store, routing, Oracle, "
            "forward/backward, "
            "raw gradient evidence, clipping, optimizer and scheduler. Excludes model/cache "
            "construction, window teardown, checkpoint serialization and final adapter save."
        ))
        with (memory, attention.attention_execution(
            attention.MAIN_BACKEND, device=target, profile_id=profile_id,
        ) as observation, torch.autocast(device_type=device, enabled=False)):
            ids = [torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
                   for batch in info["window"]["micro_batches"]]
            labels = [torch.tensor(batch["labels"], dtype=torch.long, device=device)
                      for batch in info["window"]["micro_batches"]]
            if any(bool((value >= session.config.vocab_size).any()) for value in ids):
                raise ValueError("pack input token outside checkpoint vocabulary")
            calls = [capture_layer_call(session.model, value, draws=False) for value in ids]
            store = codec.make_store(
                policy, host_budget=budget, n_layers=layers, mb_shapes=shapes,
                dtype=torch.bfloat16, device=device, pin=device == "cuda",
            )
            store.k = len(shapes)
            if arm == "oracle_block64_e4m3":
                bound = oracle_module._allocation_bound(
                    shapes, moe_layers, int(session.config.n_routed_experts), topk, device,
                )
                probe = oracle_module.DeepseekOracleProbe(
                    session.model, shapes, budget,
                    device_scratch_limit=bound["device_scratch_bound_bytes"],
                )
            step = (_stepper(session.model, session.runtime, calls, probe) if probe is not None
                    else L2LStep(session.model, session.runtime, calls[0]))
            record["adapter_before_sha256"] = model_api.adapter_digest(parameters)[0]
            losses = step.run(
                ids, store, labels=labels, n_items=info["valid_target_count"], calls=calls,
                routing_budget=(record_bytes, comparison_bytes),
            )
            record["attention"] = observation.report()
            observation.require_complete(dtype="bfloat16", backward=True,
                                         min_calls=layers * len(ids))
            expected_attention = {
                (batch, int(session.config.num_attention_heads), length,
                 int(session.config.qk_nope_head_dim) + int(session.config.qk_rope_head_dim),
                 int(session.config.v_head_dim))
                for batch, length in info["micro_batch_shapes"]
            }
            observed_attention = {
                (call["query_shape"][0], call["query_shape"][1], call["query_shape"][2],
                 call["query_shape"][3], call["value_shape"][3])
                for call in record["attention"]["unique_calls"]
            }
            if observed_attention != expected_attention or any(
                call["query_shape"] != call["key_shape"]
                or call["value_shape"][:3] != call["query_shape"][:3]
                for call in record["attention"]["unique_calls"]
            ):
                raise ValueError("executed attention Q/K/V differs from the native model/layout")
            record["routing_guard"]["outcome"] = "PASSED"
            record["routing_guard"]["natural_routing_measured"] = probe is None
            loss_values = [_finite(value.detach(), "training loss") for value in losses]
            if len(loss_values) != info["k"]:
                raise ValueError("incomplete microbatch training losses")
            record["micro_batch_losses"] = loss_values
            record["loss"] = _finite(math.fsum(loss_values), "complete-window loss")
            record["runtime_load_deltas"] = asdict(step.counters)
            if any(record["runtime_load_deltas"][name] <= 0 for name in (
                "layer_loads", "expert_fetches", "expert_bytes_read",
            )):
                raise ValueError("no production layer/expert I/O observed in training window")
            storage = store.report()
            record["store_execution"] = storage
            if (storage["put_calls"] != layers * len(ids)
                    or storage["get_calls"] != layers * len(ids)):
                raise ValueError("not every decoder boundary used the selected activation store")
            if probe is not None:
                record["oracle_before_update"] = probe.report()
                probe.require_complete()
            record["activation_storage"] = _activation_storage_evidence(
                storage, record.get("oracle_before_update"))
            gradient_evidence, gradients, norm = _gradients(
                parameters, capture_gradients, profile_id=profile_id)
            record["gradients"] = gradient_evidence
            record["gradient_norm_before_clip"] = norm
            record["learning_rate_used"] = (
                _finite(optimizer.param_groups[0]["lr"], "learning rate")
                if optimizer is not None else None
            )
            if any(not bool(torch.isfinite(value).all()) for value in parameters.values()):
                raise ValueError("nonfinite trainable before update")
            if optimizer is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(parameters.values()), max_norm=1.0, error_if_nonfinite=True, foreach=False,
                )
                optimizer.step()
                record["optimizer_applied"] = True
                if scheduler is not None:
                    scheduler.step()
            if any(not bool(torch.isfinite(value).all()) for value in parameters.values()):
                raise ValueError("nonfinite trainable after optimizer update")
            record["adapter_after_sha256"] = model_api.adapter_digest(parameters)[0]
            changed = record["adapter_before_sha256"] != record["adapter_after_sha256"]
            record["adapter_changed"] = changed
            if (profile_id == "deepseek-v3" and optimizer is not None
                    and record["learning_rate_used"] > 0 and not changed):
                raise ValueError("positive-LR update did not change any adapter")
            if optimizer is not None and record["learning_rate_used"] == 0 and changed:
                raise ValueError("LR0 warmup update unexpectedly changed the adapter")
            if any(value.grad is not None for value in session.model.parameters()
                   if not value.requires_grad):
                raise ValueError("frozen base received a gradient")
            if device == "cuda":
                torch.cuda.synchronize()
            record["boundary_store_host_bytes"] = storage["owned_host_storage_bytes"]
            record["boundary_store_payload_bytes"] = storage["payload_bytes"]
            record["boundary_store_scale_bytes"] = storage["scale_bytes"]
            record["status"] = "COMPLETE"
    except BaseException as error:
        failure = error
        record["error"] = _error(error)
        record["status"] = "FAILED"
        if "routing guard" in str(error).lower():
            record["routing_guard"]["outcome"] = "REFUSED"
        traceback.clear_frames(error.__traceback__)
    finally:
        if observation is not None:
            record["attention"] = observation.report()
        for name, resource in (("oracle", probe), ("store", store)):
            if resource is None:
                continue
            try:
                record[name + "_before_close"] = resource.report()
                resource.close()
                record[name + "_after_close"] = resource.report()
                closed = record[name + "_after_close"]
                if name == "oracle" and (not closed["closed"] or closed["records_remaining"]
                                         or closed["pending_replays"]
                                         or closed["host_reserved_bytes"]):
                    raise RuntimeError("Oracle retained replay state after close")
                if name == "store" and (not closed["cleanup"]["closed"]
                                        or closed["owned_host_storage_bytes"]
                                        or closed["owned_device_storage_bytes"]
                                        or closed["charged_host_bytes"]):
                    raise RuntimeError("boundary store retained owned state after close")
            except BaseException as error:
                cleanup_errors.append({"resource": name, **_error(error)})
                session.retained_resources.append(resource)
        if lease is not None:
            if cleanup_errors:
                session.retained_resources.append(lease)
            else:
                lease.close()
        ids.clear()
        labels.clear()
        calls.clear()
        step = store = probe = None
        record["step_seconds"] = time.perf_counter() - started
        if record.get("input_token_count"):
            record["tokens_per_second"] = record["input_token_count"] / record["step_seconds"]
            record["target_tokens_per_second"] = (
                record["valid_target_count"] / record["step_seconds"]
            )
        if budget is not None:
            record["host_budget_after_close_bytes"] = budget.used_bytes
            if budget.used_bytes:
                cleanup_errors.append({"resource": "budget", "message": "owned leases retained"})
        if memory is not None and memory.result:
            measured = memory.result
            if measured.get("sampling_errors") or not measured.get("process_rss_samples"):
                cleanup_errors.append({"resource": "rss_sampler",
                                       "message": "required RSS observation incomplete"})
            record["resources"] = {
                "sampled_rss_peak_bytes": measured.get("process_rss_sampled_peak_bytes"),
                "cuda_allocated_peak_bytes": measured.get("cuda_allocated_peak_bytes"),
                "cuda_reserved_peak_bytes": measured.get("cuda_reserved_peak_bytes"),
                "boundary_store_host_bytes": record.get("boundary_store_host_bytes"),
                "cgroup": _cgroup_memory(), "observation": measured,
                "resources_are_not_summed": True,
            }
        record["cleanup_errors"] = cleanup_errors
        if cleanup_errors:
            record["status"] = "FAILED"
            record.setdefault("error", {"type": "CleanupFailure", "message": str(cleanup_errors)})
        if record["status"] == "FAILED":
            gradients.clear()
    if record["status"] != "COMPLETE":
        raise WindowFailureError(record) from failure
    return record, gradients


@contextmanager
def _terminal_signals() -> Iterator[None]:
    if threading.current_thread() is not threading.main_thread():
        raise ValueError("training must run on the main thread for terminal signal handling")
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(signum: int, _frame: Any) -> None:
        raise RuntimeError(f"terminal process signal {signum}; intentional stops cannot resume")

    signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


@contextmanager
def _deterministic_run() -> Iterator[None]:
    import torch

    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in (None, ":4096:8"):
        raise ValueError("training requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    if workspace is None:
        if torch.cuda.is_initialized():
            raise ValueError("training requires CUBLAS_WORKSPACE_CONFIG before CUDA initialization")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    previous = {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }
    python_state = random.getstate()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(23)
            torch.manual_seed(23)
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            yield
    finally:
        random.setstate(python_state)
        torch.use_deterministic_algorithms(
            previous["deterministic"], warn_only=previous["warn_only"],
        )
        torch.backends.cuda.matmul.allow_tf32 = previous["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = previous["cudnn_tf32"]
        torch.backends.cudnn.benchmark = previous["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = previous["cudnn_deterministic"]
        if workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


def _rng_state() -> dict[str, Any]:
    import torch

    return {"python": random.getstate(), "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state: dict[str, Any]) -> None:
    import torch

    if set(state) != {"python", "torch_cpu", "torch_cuda"}:
        raise ValueError("incomplete checkpoint RNG state")
    random.setstate(state["python"])
    torch.random.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG device count differs")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _execution_identity(device: str, *, profile_id: str = "deepseek-v3") -> dict[str, Any]:
    import torch

    hardware = attention.require_ampere_cuda(device)
    attention_policy = attention.policy_identity(profile_id)

    modules = (
        "benchmarks.harness.p3_deepseek_quality_attention",
        "benchmarks.harness.p3_deepseek_quality_model",
        "benchmarks.harness.p3_deepseek_quality_train",
        "benchmarks.harness.p3_deepseek_quality_data",
        "benchmarks.harness.p3_deepseek_quality_profiles",
        "benchmarks.harness.p3_deepseek_fp8_smoke", "benchmarks.harness.p3_deepseek_oracle",
        "benchmarks.harness.p3_realdata_codec", "benchmarks.harness.p3_stage2_training",
        "benchmarks.harness.p3_fp8_smoke", "benchmarks.harness.l2l_schedule",
        "benchmarks.harness.l2l_mistral_stand",
        "soup_cli.utils.l2l_step", "soup_cli.utils.layer_shard",
        "soup_cli.utils.layer_stream_runtime", "soup_cli.utils.expert_stream",
        "soup_cli.utils.budgeted_activations", "soup_cli.utils.host_memory_budget",
    )
    sources = {}
    for name in modules:
        module = importlib.import_module(name)
        filename = inspect.getsourcefile(module)
        if filename is None:
            raise ValueError(f"missing implementation source identity: {name}")
        sources[name] = model_api.sha256(Path(filename))
    sources[attention_policy["path"]] = attention_policy["sha256"]
    packages = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes", "safetensors", "accelerate"):
        packages[name] = importlib.metadata.version(name)
    environment = {
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "packages": {name: packages[name] for name in (
            "transformers", "peft", "bitsandbytes", "accelerate", "safetensors",
        )},
    }
    environment["gpu"] = {name: hardware[name] for name in (
        "name", "capability", "bf16_supported", "native_bf16",
    )}
    return {
        "profile_id": profile_id,
        "source_files": sources, "packages": packages, "python": sys.version,
        "gate_environment": environment, "hardware": hardware,
        "device": torch.device(device).type, "cuda_version": torch.version.cuda,
        "gpu": hardware["name"], "cuda_capability": hardware["capability"],
        "torch_threads": torch.get_num_threads(), "deterministic_algorithms": True,
        "warn_only": False, "attention_backend": attention.MAIN_BACKEND, "autocast": False,
        "attention_policy": attention_policy,
        "allow_tf32": False, "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def require_gate_environment(gate_environment: dict, live_environment: dict) -> None:
    if not isinstance(gate_environment, dict) or gate_environment != live_environment:
        raise ValueError("gradient gate software/CUDA/GPU environment differs from training")


def require_gate_binding(
    result: dict, *, model_metadata: dict, initial_sha256: str, execution_identity: dict,
    profile_id: str = "deepseek-v3",
) -> None:
    """Bind an already raw-verified full gate to this live execution."""
    _require_profile(result, profile_id, "gradient gate")
    _require_profile(execution_identity, profile_id, "execution")
    for name, value in (("model", model_metadata), ("source", model_metadata.get("source")),
                        ("cache", model_metadata.get("cache"))):
        _require_profile(value, profile_id, name)
    if result.get("attention_policy") != execution_identity["attention_policy"]:
        raise ValueError("gradient gate attention policy differs from live execution")
    if (result.get("model") != model_metadata
            or result.get("initial_adapter_sha256") != initial_sha256):
        raise ValueError("gate source/cache/precision/initial adapter mismatch")
    require_gate_environment(result.get("environment"), execution_identity["gate_environment"])


def _require_gate_workload(result: dict, parameters: dict[str, Any], window: dict) -> None:
    import torch

    from benchmarks.harness.p3_deepseek_quality_gate import gate_prefix

    if result.get("trainable_names") != sorted(parameters):
        raise ValueError("gradient gate trainable names differ from the live adapter")
    for name in ("ga", "ga_repeat", "bf16_l2l", "ga_floor", "fp8_oracle", "int4_oracle"):
        checks = result.get("passes", {}).get(name, {}).get("gradient_checks", {})
        if set(checks) != set(parameters) or any(
            parameter.dtype != torch.float32
            or checks[key].get("shape") != list(parameter.shape)
            for key, parameter in parameters.items()
        ):
            raise ValueError("gradient gate tensor inventory differs from the live adapter")
    _, _, expected_prefix = gate_prefix(window)
    if result.get("input") != expected_prefix:
        raise ValueError("gradient gate prefix differs from the admitted training pack")


def _protocol(
    path: Path | None, digest: str | None, *, steps: int, length: int, pack_sha256: str,
    initial_sha256: str, pilot: bool, test_fixture: bool,
    profile_id: str = "deepseek-v3",
) -> tuple[dict | None, str | None]:
    profile = get_profile(profile_id)
    attention_policy = attention.policy_identity(profile_id)
    if pilot:
        if steps != 1 or path is not None or digest is not None:
            raise ValueError("prefreeze pilot admits exactly one step and no frozen training claim")
        return None, None
    if profile_id == "moonlight-16b-a3b" and steps != 64:
        raise ValueError("Moonlight training requires exactly 64 steps per retained arm")
    if test_fixture and path is None and digest is None:
        return None, None
    if path is None or not model_api._hex(digest) or model_api.sha256(path) != digest:
        raise ValueError("full training requires a matching frozen protocol path and SHA256")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("protocol must be an object")
    _require_profile(value, profile_id, "frozen protocol")
    if value.get("attention_policy") != attention_policy:
        raise ValueError("frozen protocol attention policy differs from the approved contract")
    if (value.get("schema") != "p3-deepseek-quality-protocol-v1"
            or value.get("status") != "FROZEN"
            or value.get("include_base_model_evaluation") is not False):
        raise ValueError("explicit frozen study protocol forbidding base evaluation required")
    training = value.get("training", {})
    if not isinstance(training, dict):
        raise ValueError("protocol training must be an object")
    lengths = training.get("lengths")
    if (not isinstance(lengths, list) or not lengths or len(set(lengths)) != len(lengths)
            or any(type(item) is not int or item not in (1024, 4096) for item in lengths)):
        raise ValueError("protocol lengths must be a unique nonempty study-length list")
    expected = {"seed": 23, "steps": steps, "input_tokens_per_step": 4096,
                "micro_batch_size": 1, "optimizer": OPTIMIZER,
                "scheduler": {"name": "cosine", "warmup_steps": 2, "horizon": steps}}
    if (any(training.get(key) != item for key, item in expected.items())
            or length not in lengths
            or value.get("pack_sha256") != pack_sha256
            or value.get("initial_adapter_sha256") != initial_sha256
            or value.get("model") != {"model_id": profile.model_id,
                                      "revision": profile.revision}):
        raise ValueError("frozen protocol training/model/data/adapter identity differs")
    require_study_profile(value, profile_id=profile_id)
    return value, digest


def _chain(previous: str, row: dict) -> str:
    payload = {"step": row["step"], "window_json_sha256": row["window_json_sha256"],
               "adapter_before_sha256": row["adapter_before_sha256"],
               "adapter_after_sha256": row["adapter_after_sha256"],
               "input_sha256": row["input_sha256"], "labels_sha256": row["labels_sha256"],
               "valid_target_count": row["valid_target_count"],
               "learning_rate_used": row["learning_rate_used"], "loss": row["loss"],
               "attention_policy": row["attention_policy"], "attention": row["attention"]}
    return hashlib.sha256(bytes.fromhex(previous) + _canonical(payload)).hexdigest()


def require_resume_identity(saved: dict, proposed: dict) -> None:
    if saved != proposed:
        raise ValueError("resume identity differs from the original scientific/execution identity")


def _save_checkpoint(
    root: Path, *, parameters: dict, optimizer: Any, scheduler: Any, identity: dict,
    completed_steps: int, rows: list[dict], digest_chain: str,
) -> dict[str, Any]:
    import torch

    folder = model_api._child(root, f"checkpoints/step-{completed_steps:06d}")
    folder.mkdir(parents=True, exist_ok=False)
    _save_tensors(
        folder / "adapter.safetensors", parameters, schema="quality-checkpoint-adapter-v1",
        profile_id=identity.get("profile_id", "deepseek-v3"))
    state = {
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "rng": _rng_state(), "completed_steps": completed_steps, "pack_cursor": completed_steps,
        "run_identity_sha256": _object_hash(identity), "digest_chain": digest_chain,
        "parameter_names": list(parameters),
    }
    with (folder / "training-state.pt").open("xb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    manifest = {
        "schema": "p3-deepseek-quality-checkpoint-v1", "status": "COMPLETE",
        "profile_id": identity.get("profile_id", "deepseek-v3"),
        "completed_steps": completed_steps, "pack_cursor": completed_steps,
        "identity": identity, "run_identity_sha256": _object_hash(identity),
        "adapter_sha256": model_api.adapter_digest(parameters)[0], "digest_chain": digest_chain,
        "rows": rows, "files": {name: model_api.sha256(folder / name)
                                  for name in ("adapter.safetensors", "training-state.pt")},
        "step_1_gradients": {
            name: model_api.sha256(root / name)
            for name in ("step-1-gradients.safetensors", "step-1-gradients.json")
        },
    }
    pending = folder / ".manifest.pending.json"
    _write_json(pending, manifest)
    os.rename(pending, folder / "manifest.json")
    return manifest


def _journal_rows(path: Path, committed: list[dict]) -> dict[str, Any]:
    """Allow an absent/torn last write, but never discard a complete conflicting row."""
    if not path.is_file():
        if committed:
            raise ValueError("resume journal missing committed preceding rows")
        return {"trailing_partial_line_ignored": False}
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    complete = []
    ignored = False
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                ignored = True
                break
            raise ValueError("resume journal contains a corrupt complete row") from None
        if not isinstance(row, dict):
            raise ValueError("resume journal contains a non-object row")
        complete.append(row)
    if len(complete) < max(0, len(committed) - 1):
        raise ValueError("resume journal is missing preceding committed steps")
    for index, row in enumerate(complete):
        if index >= len(committed) or row != committed[index]:
            raise ValueError("resume journal has a complete conflicting or uncommitted row")
    return {"trailing_partial_line_ignored": ignored,
            "last_committed_row_recovered_from_checkpoint": len(complete) < len(committed)}


def load_resume(
    root: Path, *, identity: dict, infrastructure_event: Path,
    profile_id: str = "deepseek-v3",
) -> tuple[Path, dict, dict]:
    """Admit only an unterminated attempt and its newest complete validated checkpoint."""
    _require_profile(identity, profile_id, "resume identity")
    root = Path(os.path.realpath(root))
    if (root / "manifest.json").exists() or (root / "FAILED.json").exists():
        raise ValueError("terminal COMPLETE/FAILED attempts cannot resume")
    event_file_sha256 = model_api.sha256(infrastructure_event)
    event = json.loads(Path(infrastructure_event).read_text(encoding="utf-8"))
    if event.get("event") != "instance_loss":
        raise ValueError("resume requires an explicit instance_loss infrastructure event")
    for name in ("provider", "instance_id", "detected_at", "description"):
        if not isinstance(event.get(name), str) or not event[name].strip():
            raise ValueError(f"infrastructure event lacks {name}")
    running = json.loads((root / "RUNNING.json").read_text(encoding="utf-8"))
    _require_profile(running, profile_id, "running receipt")
    require_resume_identity(running.get("identity"), identity)
    digest = _object_hash(identity)
    if running.get("run_identity_sha256") != digest or event.get("run_identity_sha256") != digest:
        raise ValueError("infrastructure event or RUNNING identity hash mismatch")
    complete = []
    checkpoint_root = root / "checkpoints"
    if checkpoint_root.is_dir():
        for folder in checkpoint_root.iterdir():
            match = _CHECKPOINT.fullmatch(folder.name)
            if match and folder.is_dir() and (folder / "manifest.json").is_file():
                complete.append((int(match[1]), folder))
    if not complete:
        raise ValueError("instance-loss resume requires at least one complete checkpoint")
    step, folder = max(complete)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    _require_profile(manifest, profile_id, "checkpoint")
    require_resume_identity(manifest.get("identity"), identity)
    if (manifest.get("schema") != "p3-deepseek-quality-checkpoint-v1"
            or manifest.get("status") != "COMPLETE" or manifest.get("completed_steps") != step
            or manifest.get("pack_cursor") != step or manifest.get("run_identity_sha256") != digest
            or set(manifest.get("files", {})) != {"adapter.safetensors", "training-state.pt"}):
        raise ValueError("invalid committed checkpoint manifest")
    for name, expected in manifest["files"].items():
        if model_api.sha256(model_api._child(folder, name)) != expected:
            raise ValueError("checkpoint content hash mismatch")
    gradient_files = manifest.get("step_1_gradients", {})
    if set(gradient_files) != {"step-1-gradients.safetensors", "step-1-gradients.json"}:
        raise ValueError("checkpoint lacks complete step-1 gradient identity")
    for name, expected in gradient_files.items():
        path = model_api._child(root, name)
        if not path.is_file() or model_api.sha256(path) != expected:
            raise ValueError("checkpoint step-1 gradient evidence hash mismatch")
    gradient_receipt = json.loads((root / "step-1-gradients.json").read_text(encoding="utf-8"))
    _require_profile(gradient_receipt, profile_id, "step-1 gradients")
    if gradient_receipt.get("sha256") != gradient_files["step-1-gradients.safetensors"]:
        raise ValueError("step-1 gradient descriptor differs from committed file digest")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != step:
        raise ValueError("checkpoint lacks complete preceding step rows")
    chain = identity["initial_parameter_sha256"]
    previous_adapter = identity["initial_parameter_sha256"]
    for index, row in enumerate(rows, 1):
        _require_profile(row, profile_id, "checkpoint step")
        if (row.get("step") != index or row.get("status") != "COMPLETE"
                or row.get("adapter_before_sha256") != previous_adapter
                or row.get("previous_digest_chain") != chain):
            raise ValueError("checkpoint step or adapter digest chain is broken")
        if (row.get("attention_backend") != attention.MAIN_BACKEND
                or row.get("attention_policy") != attention.policy_identity(profile_id)):
            raise ValueError("checkpoint attention policy/backend differs from training")
        attention.require_execution_evidence(
            row.get("attention"), backend=attention.MAIN_BACKEND, dtype="bfloat16",
            min_calls=row.get("k", 1),
            profile_id=profile_id,
        )
        chain = _chain(chain, row)
        if row.get("digest_chain") != chain:
            raise ValueError("checkpoint digest chain differs from preceding rows")
        previous_adapter = row["adapter_after_sha256"]
    if manifest.get("digest_chain") != chain or manifest.get("adapter_sha256") != previous_adapter:
        raise ValueError("checkpoint final adapter/digest chain mismatch")
    journal = _journal_rows(root / "steps.jsonl", rows)
    if model_api.sha256(infrastructure_event) != event_file_sha256:
        raise ValueError("infrastructure event changed during resume admission")
    return folder, manifest, {
        "event": event, "event_sha256": event_file_sha256,
        "previous_output": str(root), "checkpoint_manifest_sha256": model_api.sha256(
            folder / "manifest.json"), **journal,
    }


def _restore_checkpoint(
    session: model_api.ModelSession, folder: Path, manifest: dict, optimizer: Any, scheduler: Any,
) -> None:
    import torch

    for name, expected in manifest["files"].items():
        if model_api.sha256(model_api._child(folder, name)) != expected:
            raise ValueError("checkpoint changed between admission and restoration")

    model_api.load_checkpoint_adapter(
        session.model, folder / "adapter.safetensors",
        profile_id=manifest.get("profile_id", "deepseek-v3"))
    if model_api.adapter_digest(model_api.trainable_parameters(session.model))[0] != manifest[
        "adapter_sha256"
    ]:
        raise ValueError("restored checkpoint adapter digest differs")
    state = torch.load(folder / "training-state.pt", map_location="cpu", weights_only=True)
    if (set(state) != {"optimizer", "scheduler", "rng", "completed_steps", "pack_cursor",
                      "run_identity_sha256", "digest_chain", "parameter_names"}
            or state["completed_steps"] != manifest["completed_steps"]
            or state["pack_cursor"] != manifest["pack_cursor"]
            or state["run_identity_sha256"] != manifest["run_identity_sha256"]
            or state["digest_chain"] != manifest["digest_chain"]
            or state["parameter_names"] != list(model_api.trainable_parameters(session.model))):
        raise ValueError("checkpoint optimizer/scheduler/RNG/cursor identity mismatch")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if scheduler.last_epoch != manifest["completed_steps"]:
        raise ValueError("checkpoint scheduler cursor mismatch")
    parameters = model_api.trainable_parameters(session.model)
    if len(optimizer.state) != len(parameters):
        raise ValueError("checkpoint lacks optimizer state for every attention trainable")
    for parameter in parameters.values():
        saved = optimizer.state.get(parameter, {})
        if int(saved.get("step", -1)) != manifest["completed_steps"]:
            raise ValueError("checkpoint optimizer cursor mismatch")
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = saved.get(name)
            if (tensor is None or tensor.dtype != torch.float32 or tensor.shape != parameter.shape
                    or not bool(torch.isfinite(tensor).all())):
                raise ValueError("invalid checkpoint FP32 optimizer state")
    _restore_rng(state["rng"])


def _save_final_adapter(session: model_api.ModelSession, root: Path, identity: dict,
                        completed_steps: int) -> dict:
    from soup_cli.utils.layer_stream_runtime import assert_streamed_adapter_saved

    folder = root / "final_adapter"
    folder.mkdir(exist_ok=False)
    session.model.save_pretrained(folder, safe_serialization=True, save_embedding_layers=False)
    assert_streamed_adapter_saved(session.model, str(folder))
    files = {path.relative_to(folder).as_posix(): model_api.sha256(path)
             for path in sorted(folder.rglob("*")) if path.is_file()}
    value = {
        "schema": "p3-deepseek-quality-adapter-v1", "files": files,
        "profile_id": identity.get("profile_id", "deepseek-v3"),
        "identity": {"profile_id": identity.get("profile_id", "deepseek-v3"),
                     "source": session.metadata["source"], "cache": session.metadata["cache"],
                     "length": identity["length"], "seed": 23, "arm": identity["arm"],
                     "initial_adapter_sha256": identity["initial_adapter_sha256"],
                     "pack_sha256": identity["pack_sha256"],
                     "protocol_sha256": identity["protocol_sha256"],
                     "attention_policy": identity["attention_policy"],
                     "completed_steps": completed_steps,
                     **({"tokenizer": identity["tokenizer"]}
                        if "tokenizer" in identity else {})},
        "save_guard": "assert_streamed_adapter_saved passed",
        "parameter_sha256": model_api.adapter_digest(
            model_api.trainable_parameters(session.model))[0],
    }
    _write_json(folder / "manifest.json", value)
    return {"path": "final_adapter", "manifest_sha256": model_api.sha256(folder / "manifest.json"),
            **value}


def run_training(
    *, source: Path, cache: Path, pack: Path, pack_sha256: str, initial_adapter: Path,
    arm: str, length: int, steps: int, output: Path, device: str = "cuda", prepare: bool = False,
    resume: Path | None = None, infrastructure_event: Path | None = None,
    protocol: Path | None = None, protocol_sha256: str | None = None,
    gate: Path | None = None, gate_sha256: str | None = None,
    pilot_prefreeze: bool = False, test_fixture: bool = False,
    host_limit_bytes: int | None = None,
    profile_id: str = "deepseek-v3",
) -> dict[str, Any]:
    """Run one immutable study arm; Python-only fixture admission never reaches the CLI."""
    from benchmarks.harness.p3_deepseek_quality_data import load_pack

    output = Path(os.path.realpath(output))
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {"schema": "p3-deepseek-quality-run-v1", "status": "FAILED",
                              "profile_id": profile_id, "arm": arm, "length": length,
                              "seed": 23, "completed_steps": 0}
    rows: list[dict] = []
    session = None
    failure = None
    try:
        get_profile(profile_id)
        if profile_id == "moonlight-16b-a3b" and not pilot_prefreeze and steps != 64:
            raise ValueError("Moonlight training requires exactly 64 steps per retained arm")
        if arm not in ARMS or length not in (1024, 4096) or type(steps) is not int or steps < 1:
            raise ValueError("requires a named arm, L1024 or L4096 and positive step count")
        if device != "cuda":
            raise ValueError("training requires native Ampere CUDA BF16, including tiny fixtures")
        if (resume is None) != (infrastructure_event is None):
            raise ValueError("resume requires both prior output and infrastructure event")
        windows, pack_manifest = load_pack(
            pack, pack_sha256, length=length, steps=steps, profile_id=profile_id)
        _require_profile(pack_manifest, profile_id, "training pack")
        if pack_manifest.get("test_only") and not test_fixture:
            raise ValueError("production training refuses test-only tokenizer packs")
        if pack_manifest.get("exhaustion_policy") != (
            "cycle_fixed_order_record_identity_cycle_and_final_cutoff"
        ):
            raise ValueError("training pack must use the user-approved frozen Granite cycling")
        initial_sha256 = model_api.sha256(initial_adapter)
        frozen, protocol_sha256 = _protocol(
            protocol, protocol_sha256, steps=steps, length=length, pack_sha256=pack_sha256,
            initial_sha256=initial_sha256, pilot=pilot_prefreeze, test_fixture=test_fixture,
            profile_id=profile_id,
        )
        with _terminal_signals(), _deterministic_run():
            report["attention_policy"] = attention.policy_identity(profile_id)
            report["hardware"] = attention.require_ampere_cuda(device)
            session = model_api.open_session(
                source, cache, device=device, prepare=prepare, profile_id=profile_id)
            _require_profile(session.metadata, profile_id, "training session")
            if not test_fixture:
                model_api.require_production_source(
                    session.metadata["source"], profile_id=profile_id)
            execution_identity = _execution_identity(device, profile_id=profile_id)
            gate_admission = None
            if not test_fixture:
                from benchmarks.harness.p3_deepseek_quality_gate import load_gate

                if gate is None or not model_api._hex(gate_sha256):
                    raise ValueError("production/pilot training requires a hash-admitted gate")
                gate_result = load_gate(gate, gate_sha256, profile_id=profile_id)
                require_gate_binding(
                    gate_result, model_metadata=session.metadata, initial_sha256=initial_sha256,
                    execution_identity=execution_identity, profile_id=profile_id,
                )
                if gate_result.get("pack", {}).get("manifest_sha256") != pack_sha256:
                    raise ValueError("gate training pack mismatch")
                gate_windows = (windows if length == 1024 else load_pack(
                    pack, pack_sha256, length=1024, steps=1, profile_id=profile_id)[0])
                _require_gate_workload(
                    gate_result, model_api.trainable_parameters(session.model), gate_windows[0])
                gate_admission = _gate_admission(gate_result, gate_sha256, profile_id=profile_id)
            if frozen is not None:
                for name in ("source", "cache"):
                    if name in frozen and frozen[name] != session.metadata[name]:
                        raise ValueError(f"frozen protocol {name} identity differs")
                if profile_id != "deepseek-v3":
                    if (frozen["tokenizer"] != pack_manifest.get("tokenizer")
                            or frozen["aux_admission"]
                            != session.metadata["source"].get("aux_admission")):
                        raise ValueError("frozen protocol tokenizer/auxiliary admission differs")
            loaded_sha = model_api.load_initial_adapter(
                session.model, initial_adapter, profile_id=profile_id)
            if loaded_sha != initial_sha256:
                raise ValueError("shared initial adapter changed during admission")
            parameters = model_api.trainable_parameters(session.model)
            initial_parameters = model_api.adapter_digest(parameters)[0]
            report["model"] = session.metadata
            if not pilot_prefreeze and steps < 2:
                raise ValueError("a non-pilot arm must include a positive-LR optimizer update")
            identity = {
                "profile_id": profile_id,
                "tokenizer": pack_manifest.get("tokenizer"),
                "source": session.metadata["source"], "cache": session.metadata["cache"],
                "precision": session.metadata["precision"], "pack_sha256": pack_sha256,
                "initial_adapter_sha256": initial_sha256,
                "initial_parameter_sha256": initial_parameters, "arm": arm,
                "length": length, "k": 4096 // length, "micro_batch_size": 1,
                "input_tokens_per_step": 4096, "seed": 23, "steps": steps,
                "optimizer": OPTIMIZER, "scheduler": {"name": "cosine", "warmup_steps": 2,
                                                       "horizon": steps},
                "protocol_sha256": protocol_sha256, "pilot_prefreeze": pilot_prefreeze,
                "attention_policy": execution_identity["attention_policy"],
                "gradient_gate": gate_admission,
                "host_limit_bytes": host_limit_bytes,
                "host_limit_policy": "explicit" if host_limit_bytes is not None else
                "available_psutil_cgroup_min_before_window",
                "test_fixture": test_fixture, "execution": execution_identity,
            }
            optimizer, scheduler = new_optimizer(parameters, steps)
            cursor, chain = 0, initial_parameters
            resume_evidence = None
            if resume is not None:
                folder, checkpoint, resume_evidence = load_resume(
                    resume, identity=identity, infrastructure_event=infrastructure_event,
                    profile_id=profile_id,
                )
                _restore_checkpoint(session, folder, checkpoint, optimizer, scheduler)
                cursor, chain = checkpoint["completed_steps"], checkpoint["digest_chain"]
                if cursor > steps:
                    raise ValueError("checkpoint cursor exceeds frozen training horizon")
                rows = checkpoint["rows"]
                gradient_file = Path(resume) / "step-1-gradients.safetensors"
                gradient_manifest = Path(resume) / "step-1-gradients.json"
                for path in (gradient_file, gradient_manifest):
                    expected = checkpoint.get("step_1_gradients", {}).get(path.name)
                    if not path.is_file() or model_api.sha256(path) != expected:
                        raise ValueError("resume lacks hash-bound original step-1 gradients")
                    with (
                        path.open("rb") as source_stream,
                        (output / path.name).open("xb") as target,
                    ):
                        while chunk := source_stream.read(1024**2):
                            target.write(chunk)
                gradient_receipt = json.loads(gradient_manifest.read_text(encoding="utf-8"))
                if gradient_receipt["sha256"] != model_api.sha256(output / gradient_file.name):
                    raise ValueError("resumed step-1 gradient evidence hash mismatch")
                for row in rows:
                    _append_row(output / "steps.jsonl", row)
                _save_checkpoint(
                    output, parameters=parameters, optimizer=optimizer, scheduler=scheduler,
                    identity=identity, completed_steps=cursor, rows=rows, digest_chain=chain,
                )
            running = {
                "schema": "p3-deepseek-quality-running-v1", "identity": identity,
                "profile_id": profile_id,
                "run_identity_sha256": _object_hash(identity), "resumed_from_step": cursor,
                "infrastructure_loss": resume_evidence,
            }
            if resume_evidence is not None:
                with Path(infrastructure_event).open("rb") as source_stream:
                    with (output / "infrastructure-event.json").open("xb") as target:
                        target.write(source_stream.read())
                        target.flush()
                        os.fsync(target.fileno())
                if model_api.sha256(output / "infrastructure-event.json") != resume_evidence[
                    "event_sha256"
                ]:
                    raise ValueError("infrastructure event changed during preservation")
            _write_json(output / "RUNNING.json", running)
            report.update(
                identity=identity, run_identity_sha256=_object_hash(identity),
                initial_adapter_sha256=initial_sha256, resumed_from_step=cursor,
                infrastructure_loss=resume_evidence,
            )
            for index in range(cursor, steps):
                expected_shapes = [[1, length]] * (4096 // length)
                if (len(windows[index]["micro_batches"]) != len(expected_shapes)
                        or any(len(batch["input_ids"]) != 1
                               or len(batch["input_ids"][0]) != length
                               for batch in windows[index]["micro_batches"])):
                    raise ValueError("training window differs from the frozen study layout")
                record, gradients = run_window(
                    session, windows[index], arm=arm, optimizer=optimizer, scheduler=scheduler,
                    capture_gradients=index == 0, host_limit_bytes=host_limit_bytes,
                    profile_id=profile_id,
                )
                record["step"] = index + 1
                record["previous_digest_chain"] = chain
                chain = _chain(chain, record)
                record["digest_chain"] = chain
                if index == 0:
                    gradient_receipt = _save_tensors(
                        output / "step-1-gradients.safetensors", gradients,
                        schema="p3-deepseek-quality-preclip-gradients-v1",
                        profile_id=profile_id,
                    )
                    gradient_receipt.update(profile_id=profile_id, parameters=record["gradients"],
                                            window_json_sha256=record["window_json_sha256"])
                    _write_json(output / "step-1-gradients.json", gradient_receipt)
                gradients.clear()
                rows.append(record)
                _save_checkpoint(
                    output, parameters=parameters, optimizer=optimizer, scheduler=scheduler,
                    identity=identity, completed_steps=index + 1, rows=rows, digest_chain=chain,
                )
                _append_row(output / "steps.jsonl", record)
                report["completed_steps"] = index + 1
                Console().print({"step": index + 1, "arm": arm, "loss": record["loss"],
                                 "lr": record["learning_rate_used"],
                                 "tokens_per_second": record["tokens_per_second"]})
            final = _save_final_adapter(session, output, identity, steps)
            delta = adapter_delta(session.model, initial_adapter)
            if delta["initial_adapter_sha256"] != initial_sha256:
                raise ValueError("shared initial adapter changed before final delta measurement")
            if (delta["initial_parameter_sha256"] != initial_parameters
                    or delta["final_parameter_sha256"] != final["parameter_sha256"]):
                raise ValueError("initial/final adapter delta parameter identities differ")
            delta["final_adapter_manifest_sha256"] = final["manifest_sha256"]
            delta["profile_id"] = profile_id
            _write_json(output / "adapter-delta.json", delta)
            report.update(
                status="COMPLETE", completed_steps=steps, digest_chain=chain,
                final_parameter_sha256=final["parameter_sha256"], final_adapter=final,
                input_token_count=sum(row["input_token_count"] for row in rows),
                valid_target_count=sum(row["valid_target_count"] for row in rows),
                runtime=session.runtime.stats(), pilot_prefreeze=pilot_prefreeze,
                quality_evaluated=False,
                adapter_delta=delta,
                adapter_update_proved=any(row["learning_rate_used"] > 0
                                          and row["adapter_changed"] for row in rows),
            )
            optimizer.zero_grad(set_to_none=True)
            optimizer.state.clear()
    except BaseException as error:
        failure = error
        report["error"] = _error(error)
        report["status"] = "FAILED"
        report["resumable"] = False
        _write_json(output / "FAILED.json", {
            "profile_id": profile_id, "error": _error(error), "resumable": False,
        })
        if isinstance(error, WindowFailureError):
            _write_json(output / "failed-window.json", error.record)
            report["failed_window"] = error.record
        traceback.clear_frames(error.__traceback__)
    finally:
        if session is not None:
            try:
                session.close()
            except BaseException as error:
                report["status"] = "FAILED"
                report["cleanup_error"] = _error(error)
                report["resumable"] = False
                if not (output / "FAILED.json").exists():
                    _write_json(output / "FAILED.json", {
                        "profile_id": profile_id, "error": _error(error), "resumable": False,
                    })
                if failure is None:
                    failure = error
        report["files"] = {
            path.relative_to(output).as_posix(): model_api.sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path != output / "manifest.json"
        }
        _write_json(output / "manifest.json", report)
    if failure is not None:
        raise RuntimeError(f"training FAILED; evidence preserved at {output}") from failure
    return report


def _initial_command(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or receipt_path.exists():
        raise FileExistsError("initial adapter or its immutable receipt already exists")
    if args.device != "cuda":
        raise ValueError("initial adapter production admission requires native Ampere CUDA")
    with _deterministic_run():
        attention.require_ampere_cuda(args.device)
        with model_api.open_session(
            Path(args.source), Path(args.cache), device=args.device, prepare=args.prepare,
            profile_id=args.profile_id,
        ) as session:
            model_api.require_production_source(
                session.metadata["source"], profile_id=args.profile_id)
            receipt = model_api.write_initial_adapter(
                session.model, output, profile_id=args.profile_id)
            receipt.update(profile_id=args.profile_id, source=session.metadata["source"],
                           cache=session.metadata["cache"],
                           attention_policy=attention.policy_identity(args.profile_id))
            _write_json(receipt_path, receipt)
            return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    arguments = list(sys.argv[1:] if argv is None else argv)
    initial = bool(arguments and arguments[0] == "initial-adapter")
    if initial:
        arguments = arguments[1:]
    parser.add_argument("--profile", dest="profile_id",
                        choices=("deepseek-v3", "moonlight-16b-a3b"), default="deepseek-v3")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--prepare", action="store_true")
    if not initial:
        parser.add_argument("--pack", type=Path, required=True)
        parser.add_argument("--pack-sha256", required=True)
        parser.add_argument("--initial-adapter", type=Path, required=True)
        parser.add_argument("--arm", choices=ARMS, required=True)
        parser.add_argument("--length", type=int, choices=(1024, 4096), required=True)
        parser.add_argument("--steps", type=int, required=True)
        parser.add_argument("--resume", type=Path)
        parser.add_argument("--infrastructure-event", type=Path)
        parser.add_argument("--protocol", type=Path)
        parser.add_argument("--protocol-sha256")
        parser.add_argument("--gate", type=Path)
        parser.add_argument("--gate-sha256")
        parser.add_argument("--pilot-prefreeze", action="store_true")
        parser.add_argument("--host-limit-bytes", type=int)
    args = parser.parse_args(arguments)
    try:
        result = _initial_command(args) if initial else run_training(**vars(args))
        Console().print_json(json.dumps(result, allow_nan=False))
        return 0
    except Exception as error:
        Console(stderr=True).print(f"[red]{type(error).__name__}: {error}[/red]")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
