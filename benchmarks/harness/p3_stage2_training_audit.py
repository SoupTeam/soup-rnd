"""Read-only CPU audit of the frozen Stage2 training artifacts, never quality admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from functools import partial
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rich.console import Console

from . import p3_realdata_baseline as baseline
from . import p3_stage2_dataset as data
from . import p3_stage2_training as training

FROZEN_PROTOCOL_SHA256 = "170c0b700c137c1f6e0c6788376927e4aaa4e99866b990bc83d0afc42a62bc3c"
VERIFIED = "TRAINING_EVIDENCE_VERIFIED_QUALITY_NOT_MEASURED"
_ARM_VERIFIED = "TRAINING_ARM_EVIDENCE_VERIFIED"
_BLOCKED = "BLOCKED_INCOMPLETE_EVIDENCE"
_HEADER_ALLOWANCE = 2 * 1024**2
_KINDS = {
    "gradients": "raw-pre-clip-gradients",
    "adapters": "updated-adapter-evaluation-eligibility-in-arm-json",
    "optimizer": "named-AdamW-state",
}
_COMMON_DEVICE_KEYS = (
    "input_tensor_bytes",
    "native_call_capture_persistent_bound_bytes",
    "native_call_capture_scratch_bound_bytes",
    "l2l_hidden_upstream_bound_bytes",
    "trainable_gradient_bytes",
    "optimizer_state_bound_bytes",
    "optimizer_operation_scratch_bound_bytes",
    "device_diagnostic_scratch_bound_bytes",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _error(error: Exception, *, scope: str | None = None) -> dict:
    value = {"type": type(error).__name__, "message": str(error)}
    if scope is not None:
        value["scope"] = scope
    return value


def _read_json(path: Path) -> Any:
    return data._json(path.read_text(encoding="utf-8"))


def _manifest_identity(folder: Path, filename: str) -> dict:
    path = data._relative(folder, filename)
    return {"file": filename, "bytes": path.stat().st_size, "sha256": training._sha256(path)}


def _number(value: Any, *, positive: bool = False) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0)
    )


def _bytes(value: Any) -> bool:
    return type(value) is int and value >= 0


def _load_protocol(path: Path) -> tuple[dict, str, list[dict]]:
    digest = training._sha256(path)
    _require(digest == FROZEN_PROTOCOL_SHA256, "Frozen protocol file SHA-256 mismatch")
    protocol = _read_json(path)
    _require(
        protocol["schema"] == "p3-stage2-frozen-paired-study-v1",
        "Unsupported frozen study protocol",
    )
    configurations = []
    for key in ("lengths", "seeds", "arms"):
        values = protocol[key]
        _require(
            isinstance(values, list) and values and len(set(values)) == len(values),
            "Duplicate or absent protocol configurations",
        )
    for length, seed, arm in product(protocol["lengths"], protocol["seeds"], protocol["arms"]):
        _require(
            type(length) is int and type(seed) is int and arm in training._ARMS,
            "Invalid frozen protocol configuration",
        )
        configurations.append(
            {"length": length, "seed": seed, "arm": arm, "directory": f"L{length}-seed{seed}-{arm}"}
        )
    _require(len(configurations) == 20, "Frozen study must require all 20 configurations")
    return protocol, digest, configurations


def _source_windows(manifest: dict, infos: list[dict], documents: list[dict]) -> None:
    """Join each window's provenance to the admitted pack's document-span evidence."""
    by_domain = {domain: [] for domain in ("math", "code")}
    for document in documents:
        _require(document["domain"] in by_domain, "Unknown pack source domain")
        by_domain[document["domain"]].append(document)
    for domain, spans in by_domain.items():
        order = manifest["domain_source_order"][domain]
        _require(order and len(order) == len(set(order)), "Invalid frozen source order")
        cursor = 0
        for number, span in enumerate(spans):
            _require(
                span["stream_start"] == cursor
                and type(span["stream_end"]) is int
                and span["stream_end"] > cursor
                and span["domain_document_number"] == number
                and span["source_ref"] == order[number % len(order)]
                and span["source_order_position"] == number % len(order)
                and span["cycle"] == number // len(order)
                and training._hex_digest(span["source_row_sha256"])
                and span["document_token_start"] == 0
                and span["document_token_end"] == span["stream_end"] - cursor
                and span["document_token_end"] <= span["document_token_count"],
                "Pack document source/order/span join mismatch",
            )
            cursor = span["stream_end"]
        _require(
            cursor == manifest["domain_input_token_counts"][domain]
            and spans[-1] == manifest["final_cutoffs"][domain],
            "Pack source coverage/final cutoff mismatch",
        )
    counts = {"math": 0, "code": 0}
    targets = {"math": 0, "code": 0}
    for index, info in enumerate(infos):
        window = info["window"]
        domain = "math" if index % 2 == 0 else "code"
        start = index // 2 * 4096
        _require(
            window["domain"] == domain
            and window["domain_stream_start"] == start
            and window["domain_stream_end"] == start + 4096
            and window["common_boundary_mask_interval"] == 1024,
            "Frozen source window order/domain mismatch",
        )
        expected = []
        for span in by_domain[domain]:
            left, right = max(start, span["stream_start"]), min(start + 4096, span["stream_end"])
            if left < right:
                expected.append(
                    {
                        "source_ref": span["source_ref"],
                        "cycle": span["cycle"],
                        "domain_document_number": span["domain_document_number"],
                        "window_token_start": left - start,
                        "window_token_end": right - start,
                        "document_token_start": left - span["stream_start"],
                        "document_token_end": right - span["stream_start"],
                    }
                )
        _require(window["document_spans"] == expected, "Frozen source/window span join mismatch")
        counts[domain] += 1
        targets[domain] += info["valid_target_count"]
    _require(
        counts == manifest["domain_windows"]
        and targets == manifest["domain_trained_target_counts"],
        "Pack source domain window/target totals mismatch",
    )


def _load_pack(root: Path, digest: str, protocol: dict) -> tuple[dict, dict[int, list[dict]]]:
    _require(
        digest == protocol["data"]["training_pack_manifest_file_sha256"]
        and training._sha256(data._relative(root, "manifest.json")) == digest,
        "Frozen pack manifest file SHA-256/protocol join mismatch",
    )
    manifest = data._read_manifest(root)
    _require(
        manifest["schema"] == data.PACKING_POLICY
        and manifest["status"] == "PAIRED_DATA_PACKED_RUNTIME_AND_PRECISION_PENDING"
        and manifest["dataset_manifest_file"]["sha256"]
        == protocol["data"]["dataset_manifest_file_sha256"]
        and training._hex_digest(manifest["dataset_manifest_sha256"])
        and manifest["seeds"] == protocol["seeds"]
        and manifest["seed_changes_data"] is False
        and manifest["common_boundary_mask_interval"] == 1024
        and manifest["tokenizer"]["repo_id"] == protocol["model"]["id"]
        and manifest["tokenizer"]["revision"] == protocol["model"]["revision"],
        "Frozen pack source/dataset/model/protocol join mismatch",
    )
    contract = protocol["training"]
    _require(
        manifest["input_token_count"] == contract["input_tokens_per_arm"] == 262144
        and manifest["trained_target_count"] == contract["supervised_targets_per_arm"] == 179326
        and manifest["window_count_per_length"] == contract["windows_per_arm"] == 64
        and manifest["domain_input_token_counts"]
        == {
            "math": contract["input_tokens_per_domain_per_arm"],
            "code": contract["input_tokens_per_domain_per_arm"],
        },
        "Frozen pack token/target/window totals mismatch",
    )
    documents = list(data._jsonl(data._artifact(root, manifest["files"]["document_spans"])))
    infos_by_length = {}
    paired_flat = None
    for length in protocol["lengths"]:
        descriptor = manifest["files"][f"windows_L{length}"]
        _require(
            descriptor["rows"] == 64
            and descriptor["length"] == length
            and descriptor["batch_size"] == protocol["micro_batch_size"] == 1
            and descriptor["micro_batches_per_window"] == protocol["effective_k"][str(length)],
            "Frozen pack layout descriptor mismatch",
        )
        windows = _read_json(data._artifact(root, descriptor))
        infos = training._study_windows(windows, preflight=False)
        _require(
            all(info["micro_batch_shapes"] == [[1, length]] * (4096 // length) for info in infos),
            "Frozen pack layout mismatch",
        )
        _require(
            sum(info["input_token_count"] for info in infos) == 262144
            and sum(info["valid_target_count"] for info in infos) == 179326,
            "Recomputed frozen pack token/target totals mismatch",
        )
        inputs, labels = hashlib.sha256(), hashlib.sha256()
        for info in infos:
            for batch in info["window"]["micro_batches"]:
                for rows, stream in ((batch["input_ids"], inputs), (batch["labels"], labels)):
                    for row in rows:
                        stream.update(struct.pack(f"<{len(row)}q", *row))
        input_hash, label_hash = inputs.hexdigest(), labels.hexdigest()
        flat = {
            "input_sha256": input_hash,
            "labels_sha256": label_hash,
            "valid_target_count": 179326,
            "pairing_sha256": data._pairing(input_hash, label_hash, 179326),
            "ordered_window_pairing_sha256": hashlib.sha256(
                b"".join(bytes.fromhex(info["pairing_sha256"]) for info in infos)
            ).hexdigest(),
        }
        _require(
            manifest["flat_pairing"][str(length)] == flat,
            "Recomputed ordered pack pairing mismatch",
        )
        comparable = [
            {key: value for key, value in info["window"].items() if key != "micro_batches"}
            for info in infos
        ]
        if paired_flat is not None:
            _require(paired_flat == (flat, comparable), "Cross-length flat/source pairing mismatch")
        paired_flat = flat, comparable
        _source_windows(manifest, infos, documents)
        infos_by_length[length] = infos
    return manifest, infos_by_length


def _adapter_shapes() -> dict[str, list[int]]:
    config = training.native._EXPECTED_CONFIG
    hidden = config["hidden_size"]
    key_value = hidden * config["num_key_value_heads"] // config["num_attention_heads"]
    return {
        f"model.layers.{layer}.self_attn.{target}.lora_{side}.default.weight": (
            [2, hidden]
            if side == "A"
            else [key_value if target in ("k_proj", "v_proj") else hidden, 2]
        )
        for layer in range(config["num_hidden_layers"])
        for target in training.native.TARGETS
        for side in ("A", "B")
    }


def _check_adapter_manifest(manifest: dict, shapes: dict, *, nonzero: bool) -> None:
    _require(
        set(manifest) == set(shapes) and len(manifest) == 192,
        "Incomplete 192-tensor named adapter/gradient coverage",
    )
    for name, shape in shapes.items():
        item = manifest[name]
        _require(
            item["shape"] == shape
            and item["dtype"] == "torch.float32"
            and item["finite"] is True
            and training._hex_digest(item["sha256"])
            and _number(item["l2"])
            and type(item["nonzero"]) is int
            and 0 <= item["nonzero"] <= math.prod(shape)
            and (not nonzero or item["nonzero"] > 0),
            f"Invalid finite FP32 tensor manifest: {name}",
        )


def _tensor_manifest(
    folder: Path,
    descriptor: dict,
    *,
    filename: str,
    metadata: dict,
    shapes: dict[str, list[int]],
    optimizer_step: int | None = None,
) -> dict:
    """Hash files in chunks and inspect one CPU tensor at a time, including named AdamW state."""
    from safetensors import safe_open

    _require(descriptor["file"] == filename, "Unexpected or unsafe evidence filename")
    path = data._relative(folder, filename)
    expected_bytes = 4 * sum(math.prod(shape) for shape in shapes.values())
    _require(
        type(descriptor["bytes"]) is int
        and 8 < descriptor["bytes"] <= expected_bytes + _HEADER_ALLOWANCE
        and path.stat().st_size == descriptor["bytes"]
        and training._sha256(path) == descriptor["sha256"],
        "Evidence file bytes/SHA-256 mismatch",
    )
    manifest = {}
    with safe_open(str(path), framework="pt", device="cpu") as stream:
        _require(stream.metadata() == metadata, "Safetensors header identity mismatch")
        _require(set(stream.keys()) == set(shapes), "Safetensors named tensor coverage mismatch")
        for name, shape in shapes.items():
            _require(
                stream.get_slice(name).get_shape() == shape,
                f"Safetensors tensor shape mismatch: {name}",
            )
            tensor = stream.get_tensor(name)
            try:
                item = baseline.gradient_manifest({name: tensor})[name]
                _require(
                    item["dtype"] == "torch.float32" and item["finite"] is True,
                    f"Nonfinite or non-FP32 saved tensor: {name}",
                )
                if optimizer_step is not None:
                    if name.endswith("::step"):
                        _require(tensor.item() == optimizer_step, "Named AdamW step count mismatch")
                    elif name.endswith("::exp_avg_sq"):
                        _require(bool((tensor >= 0).all()), "Negative AdamW second moment")
                manifest[name] = item
            finally:
                del tensor
    _require(manifest == descriptor["tensor_manifest"], "Evidence tensor manifest mismatch")
    return manifest


def _expected_memory(info: dict, shapes: dict, arm: str) -> dict:
    """Reuse only existing tensor-free reservation planners, never construct a model."""
    spec = SimpleNamespace(config=SimpleNamespace(**training.native._EXPECTED_CONFIG))
    method, oracle, policy = training._ARMS[arm]
    plan = training._policy_plan(
        spec, info, method=method, oracle=oracle, policy=policy, device_type="cuda"
    )
    ref_method, ref_oracle, ref_policy = training._ARMS["bf16_l2l"]
    reference = training._policy_plan(
        spec, info, method=ref_method, oracle=ref_oracle, policy=ref_policy, device_type="cuda"
    )
    counts = {
        name: SimpleNamespace(numel=partial(math.prod, shape)) for name, shape in shapes.items()
    }
    common = training._common_plan(
        spec, info, counts, method=method, device_type="cuda", capture=True, optimizer=True
    )
    return {
        **training._memory_accounting(plan, reference),
        "policy_plan": plan,
        "common_trainer_plan": common,
        "common_trainer_device_bound_bytes": sum(common[key] for key in _COMMON_DEVICE_KEYS),
        "common_trainer_device_limit_bytes": training._COMMON_TRAINER_DEVICE_LIMIT,
    }


def _dtype_observations(shapes: dict) -> dict:
    return {
        "frozen_base": {
            "parameter_elements_by_dtype": {"torch.bfloat16": 1334625280},
            "tensors": 218,
        },
        "trainable_lora": {
            "parameter_elements_by_dtype": {
                "torch.float32": sum(math.prod(shape) for shape in shapes.values())
            },
            "tensors": 192,
        },
    }


def _learning_rate(index: int, protocol: dict) -> float:
    schedule = protocol["scheduler"]
    warmup, horizon = schedule["warmup_steps"], schedule["horizon"]
    if index < warmup:
        factor = float(index) / float(max(1, warmup))
    else:
        progress = float(index - warmup) / float(max(1, horizon - warmup))
        factor = max(0.0, 0.5 * (1.0 + math.cos(math.pi * 0.5 * 2.0 * progress)))
    return protocol["optimizer"]["lr"] * factor


def _settings(window: dict, *, step: int, shapes: dict, protocol: dict) -> None:
    before, after = _learning_rate(step - 1, protocol), _learning_rate(step, protocol)
    _require(
        window["learning_rates_before"] == [before]
        and window["learning_rates_after"] == [after]
        and all(
            _number(rate)
            for key in ("learning_rates_before", "learning_rates_after")
            for rate in window[key]
        ),
        "Warmup/cosine learning rate mismatch",
    )
    scheduler = window["evidence"]["scheduler_state"]
    _require(
        scheduler["last_epoch"] == step
        and scheduler["_step_count"] == step + 1
        and scheduler["base_lrs"] == [protocol["optimizer"]["lr"]]
        and scheduler["_last_lr"] == [after]
        and scheduler["lr_lambdas"] == [{}],
        "Saved scheduler state/horizon progression mismatch",
    )
    groups = window["evidence"]["optimizer"]["param_groups"]
    _require(
        len(groups) == 1
        and len(groups[0]["params"]) == len(shapes)
        and set(groups[0]["params"]) == set(shapes),
        "Named AdamW parameter groups mismatch",
    )
    expected = {
        key: value
        for key, value in protocol["optimizer"].items()
        if key not in ("type", "clip_max_norm")
    }
    expected.update(
        lr=after,
        initial_lr=protocol["optimizer"]["lr"],
        amsgrad=False,
        maximize=False,
        capturable=False,
        differentiable=False,
    )
    _require(
        all(groups[0].get(key) == value for key, value in expected.items())
        and all(
            groups[0].get(key) is False
            for key in ("foreach", "fused", "amsgrad", "maximize", "capturable", "differentiable")
        )
        and _number(groups[0]["lr"]),
        "Saved AdamW settings mismatch",
    )
    _require(
        groups[0].get("decoupled_weight_decay", True) is True,
        "Saved optimizer is not decoupled AdamW",
    )


def _routing(window: dict, *, length: int, arm: str, protocol: dict) -> None:
    oracle = arm == "oracle_block64_e4m3"
    route = window["routing"]
    pairs = 24 * (4096 // length)
    _require(
        all(route[key] is True for key in ("strict", "routing_pass", "complete", "admissible"))
        and route["oracle"] is oracle
        and route["diagnostic_only"] is False
        and route["oracle_output_only"] is False
        and route["vjp_required"] is oracle
        and route["pending_oracle_replays"] == 0
        and route["expected_pairs"]
        == route["forward_records"]
        == route["recomputed_records"]
        == pairs,
        "Incomplete strict routing/pair/VJP evidence",
    )
    expected_counts = {
        "native_forward": pairs,
        "native_recompute": 0 if oracle else pairs,
        "oracle_forward": pairs if oracle else 0,
        "oracle_vjp": pairs if oracle else 0,
        "oracle_vjp_required": pairs if oracle else 0,
        "oracle_diagnostic_forward": 0,
        "probability_reconstruction": pairs if oracle else 0,
        "unobserved_native_calls": 0,
    }
    _require(
        all(route["counts"].get(key) == value for key, value in expected_counts.items())
        and route["native_forward_calls"] == (pairs if oracle else 2 * pairs)
        and route["oracle_forward_calls"] == (pairs if oracle else 0)
        and route["oracle_vjp_calls"] == (pairs if oracle else 0)
        and route["oracle_vjp_required_calls"] == (pairs if oracle else 0)
        and route["oracle_diagnostic_forward_calls"] == 0,
        "Native versus Oracle execution/VJP count mismatch",
    )
    if oracle:
        _require(
            route["natural_rmr"] is None
            and route["natural_admissible"] is False
            and route["forced_rmr"] == 1.0
            and route["forced_routing"] is True
            and all(
                route.get(key) is None
                for key in (
                    "ordered_topk_rmr",
                    "observed_natural_rmr",
                    "natural_slot_equality",
                    "natural_set_rmr",
                )
            ),
            "Oracle forced routing must not claim natural RMR",
        )
    else:
        _require(
            route["natural_rmr"] == 1.0
            and route["natural_admissible"] is True
            and route["forced_rmr"] is None
            and route["forced_routing"] is False,
            "BF16 native ordered routing mismatch",
        )
    layers = route["layers"]
    _require(len(layers) == 24, "Missing native router layers")
    for index, layer in enumerate(layers):
        _require(
            layer["layer"] == index
            and layer["complete"] is True
            and layer["forward_calls"] == layer["recompute_calls"] == 4096 // length
            and layer["forward_tokens"] == layer["compared_tokens"] == 4096
            and layer["mismatched_tokens"]
            == layer["mismatched_slots"]
            == layer["set_mismatched_tokens"]
            == 0,
            "Native router layer coverage/count mismatch",
        )
    source = route["source_hashes"]["benchmark"]
    _require(
        source["available"] is True
        and source["sha256"]
        == protocol["source_files_sha256"]["benchmarks/harness/p3_realdata_routing.py"],
        "Routing source/protocol join mismatch",
    )


def _cleanup_and_memory(window: dict, *, memory: dict, shapes: dict, arm: str) -> None:
    elements = sum(math.prod(shape) for shape in shapes.values())
    maximum = max(math.prod(shape) for shape in shapes.values())
    cleanup, evidence = window["cleanup"], window["evidence"]
    _require(
        cleanup["complete"] is True
        and cleanup["errors"] == []
        and cleanup["host_used_bytes"] == 0
        and cleanup["returned_gradient_evidence_bytes"] == 4 * elements
        and cleanup["lease_release_proves_os_rss_release"] is False
        and evidence["cleanup"]["complete"] is True
        and evidence["cleanup"]["host_used_bytes_after_export"] == 0
        and evidence["host_export_bound_bytes"] == 96 * elements + 40 * maximum + 4 * 1024**2
        and evidence["host_export_bound_bytes"]
        <= evidence["host_export_limit_bytes"]
        == training._EVIDENCE_LIMIT
        and evidence["evaluation_eligible"] is False,
        "Window/export cleanup, leases or eligibility mismatch",
    )
    route, store = window["routing_after_cleanup"], window["store_after_cleanup"]
    _require(
        route["closed"] is True
        and route["pending_oracle_replays"] == 0
        and route["cleanup"]["hooks_removed"] is True
        and route["cleanup"]["original_forwards_restored"] is True
        and route["cleanup"]["owned_buffers_dropped"] is True
        and store["cleanup"]["closed"] is True
        and store["cleanup"]["host_lease_released"] is True
        and all(
            resource[key] == 0
            for resource in (route, store)
            for key in ("owned_host_storage_bytes", "owned_device_storage_bytes")
        ),
        "Policy cleanup left storage, hooks, leases or pending Oracle replays",
    )
    _require(window["memory_accounting"] == memory, "Tensor reservation plan/accounting mismatch")
    plan, common = memory["policy_plan"], memory["common_trainer_plan"]
    _require(
        window["host_limit_bytes"] == training.HOST_LIMIT
        and window["device_scratch_limit_bytes"] == training.DEVICE_SCRATCH_LIMIT
        and window["charged_host_bytes"] == plan["host_bytes"]
        and window["total_host_reservation_bytes"]
        == plan["host_bytes"] + common["host_reservation_bytes"]
        <= training.HOST_LIMIT
        and plan["device_scratch_bytes"] + plan["device_persistent_bytes"]
        <= training.DEVICE_SCRATCH_LIMIT
        and memory["common_trainer_device_bound_bytes"] <= training._COMMON_TRAINER_DEVICE_LIMIT
        and common["native_call_capture_persistent_bound_bytes"]
        + common["native_call_capture_scratch_bound_bytes"]
        <= training._COMMON_CAPTURE_LIMIT
        and _bytes(window["native_call_capture_owned_storage_bytes"])
        and window["native_call_capture_owned_storage_bytes"]
        <= common["native_call_capture_persistent_bound_bytes"],
        "Common/policy host or device reservation/capture bound mismatch",
    )
    for label, allocation, host_bound, device_bound in (
        (
            "routing",
            plan["route"],
            sum(plan["route"]["host"].values()),
            plan["route"]["device_buffer_bound_bytes"],
        ),
        (
            "store",
            plan["store"],
            plan["store"]["charged_host_bytes"],
            plan["store"]["device_persistent_bytes"] + plan["store"]["device_scratch_bytes"],
        ),
    ):
        observed = window[label]
        _require(
            observed["allocation"] == allocation
            and _bytes(observed["owned_host_storage_bytes"])
            and observed["owned_host_storage_bytes"] <= host_bound
            and _bytes(observed["owned_device_storage_bytes"])
            and observed["owned_device_storage_bytes"] <= device_bound,
            "Observed policy tensor storage exceeds declared reservation",
        )
    _require(
        window["store"]["put_calls"]
        == window["store"]["get_calls"]
        == window["routing"]["expected_pairs"],
        "Incomplete codec boundary replay",
    )
    if arm == "oracle_block64_e4m3":
        _require(
            memory["positive_fully_accounted_reservation_saving"] is True,
            "Candidate lacks approved positive fully accounted reservation saving",
        )
    observed = window["memory_observed"]
    for key in (
        "process_rss_before_bytes",
        "process_rss_after_bytes",
        "process_rss_sampled_peak_bytes",
        "cuda_allocated_peak_bytes",
        "cuda_reserved_peak_bytes",
    ):
        _require(_bytes(observed[key]), "Missing/nonfinite separate RAM/VRAM observation")
    _require(
        observed["process_lifetime_rss_high_water_bytes"] is None
        or _bytes(observed["process_lifetime_rss_high_water_bytes"]),
        "Invalid process-lifetime RSS high water",
    )
    _require(
        _bytes(observed["process_rss_samples"]) and observed["process_rss_samples"] > 0,
        "No process RSS samples recorded",
    )


def _validate_step(
    window: dict,
    info: dict,
    *,
    folder: Path,
    arm: str,
    seed: int,
    length: int,
    step: int,
    initial: dict,
    before: dict,
    memory: dict,
    protocol: dict,
    protocol_sha256: str,
    shapes: dict,
) -> None:
    _require(
        all(window.get(key) == value for key, value in info.items()),
        "Consumed window JSON/provenance differs from frozen source",
    )
    method, oracle, policy = training._ARMS[arm]
    _require(
        window["status"] == "MEASURED"
        and window["method"] == method
        and window["oracle"] is oracle
        and window["policy"] == vars(policy)
        and window["mandatory_admission_failure"] is False
        and all(
            window[key] is True
            for key in (
                "forward_backward_complete",
                "optimizer_step",
                "optimizer_step_attempted",
                "optimizer_step_returned",
            )
        )
        and window["optimizer_steps_completed"] == 1
        and window["optimizer_step_index"] == step,
        "Incomplete successful optimizer/forward/backward step evidence",
    )
    _require(
        window["parameter_dtype_observations"] == _dtype_observations(shapes),
        "Frozen-base/adapter precision observation mismatch",
    )
    frozen = window["frozen_base_invariants"]
    _require(
        frozen["pass"] is True
        and frozen["tensors"] == 218
        and frozen["requires_grad"] is False
        and frozen["no_gradients"] is True
        and frozen["identity_storage_version_unchanged"] is True,
        "Frozen base invariant evidence failed or missing",
    )
    _routing(window, length=length, arm=arm, protocol=protocol)
    _cleanup_and_memory(window, memory=memory, shapes=shapes, arm=arm)
    _settings(window, step=step, shapes=shapes, protocol=protocol)
    _require(window["adapter_hashes_before"] == before, "Adapter trajectory hash chain mismatch")
    manifests = {}
    state_shapes = {
        f"{name}::{key}": [] if key == "step" else shape
        for name, shape in shapes.items()
        for key in ("step", "exp_avg", "exp_avg_sq")
    }
    for kind, label in _KINDS.items():
        manifests[kind] = _tensor_manifest(
            folder,
            window["evidence"][kind],
            filename=f"step-{step:03d}.{kind}.safetensors",
            metadata={
                "arm": arm,
                "seed": str(seed),
                "optimizer_step": str(step),
                "protocol_sha256": protocol_sha256,
                "kind": label,
            },
            shapes=state_shapes if kind == "optimizer" else shapes,
            optimizer_step=step if kind == "optimizer" else None,
        )
    for kind in ("gradients", "adapters"):
        _check_adapter_manifest(manifests[kind], shapes, nonzero=kind == "gradients")
    gradients = manifests["gradients"]
    _require(
        gradients == window["gradient_manifest"]
        and window["gradients_finite"] is True
        and window["missing_gradient_tensors"] == []
        and window["zero_gradient_tensors"] == []
        and window["raw_gradient_l2"]
        == math.sqrt(math.fsum(item["l2"] ** 2 for item in gradients.values())),
        "Raw finite/nonzero gradient artifact/window evidence mismatch",
    )
    after = {name: item["sha256"] for name, item in manifests["adapters"].items()}
    changed = sum(after[name] != before[name] for name in shapes)
    _require(
        window["adapter_hashes_after"] == after and window["updated_parameter_count"] == changed,
        "Updated adapter artifact/hash/count mismatch",
    )
    if step == 1:
        _require(
            after == {name: item["sha256"] for name, item in initial.items()},
            "Zero-LR first warmup call changed paired adapter bytes",
        )
    if step == 2:
        _require(
            window["learning_rates_before"][0] > 0 and changed > 0,
            "Second warmup-aware optimizer call did not change adapter bytes",
        )
    _require(
        window["clip_max_norm"] == protocol["optimizer"]["clip_max_norm"] == 1.0
        and _number(window["clip_returned_norm"], positive=True)
        and _number(window["post_clip_gradient_l2"])
        and window["post_clip_gradient_l2"] <= 1.00001,
        "Missing/nonfinite or out-of-bound clipping evidence",
    )
    losses = window["losses"]
    _require(
        len(losses) == 4096 // length
        and all(_number(value) for value in losses)
        and window["window_loss"] == math.fsum(losses),
        "Nonfinite/incomplete window loss",
    )
    _require(
        _number(window["full_step_wall_seconds"], positive=True)
        and _number(window["evidence_serialization_wall_seconds"]),
        "Missing/nonfinite full-step or serialization timing",
    )


def _validate_run(
    wrapper: dict,
    record: dict,
    *,
    length: int,
    seed: int,
    arm: str,
    protocol: dict,
    protocol_sha256: str,
    pack_sha256: str,
    manifest: dict,
    shapes: dict,
) -> None:
    _require(
        wrapper["schema"] == "p3-stage2-pretrained-arm-v1"
        and record["schema"] == "p3-stage2-arm-v1"
        and all(
            value["arm"] == arm
            and value["seed"] == seed
            and value["preflight"] is False
            and value["protocol_sha256"] == protocol_sha256
            for value in (wrapper, record)
        )
        and wrapper["length"] == length
        and wrapper["pack_manifest_file_sha256"] == pack_sha256
        and wrapper["pack_manifest_payload_sha256"] == manifest["manifest_payload_sha256"]
        and wrapper["windows_file"] == manifest["files"][f"windows_L{length}"]
        and record["protocol"] == training._protocol(arm),
        "Requested configuration/protocol/pack identity mismatch",
    )
    environment = wrapper["environment"]
    flags = environment["precision_flags"]
    _require(
        environment["gpu"] == "Tesla T4"
        and environment["compute_capability"] == [7, 5]
        and environment["native_bf16_support"] is False
        and environment["emulated_bf16_tensor_support"] is True
        and environment["weight_format"] == "dense BF16, not P3 NF4"
        and flags == wrapper["precision_flags_after_arm"]
        and flags["cuda_matmul_allow_tf32"] is protocol["model"]["cuda_matmul_allow_tf32"] is False
        and flags["cudnn_allow_tf32"] is protocol["model"]["cudnn_allow_tf32"] is True
        and flags["autocast_enabled_cuda"] is False
        and flags["autocast_enabled_cpu"] is False,
        "Recorded hardware/precision permissions changed or differ from frozen protocol",
    )
    _require(
        all(
            type(flags[key]) is bool
            for key in (
                "allow_fp16_reduced_precision_reduction",
                "allow_bf16_reduced_precision_reduction",
            )
        ),
        "Missing recorded reduction-precision permissions",
    )
    provenance = record["model_provenance"]
    _require(
        provenance["repo_id"] == protocol["model"]["id"]
        and provenance["revision"] == protocol["model"]["revision"]
        and provenance["parameter_dtype_observations"] == _dtype_observations(shapes)
        and provenance["source"]["benchmark_module"]["available"] is True
        and provenance["source"]["benchmark_module"]["sha256"]
        == protocol["source_files_sha256"]["benchmarks/harness/p3_realdata_model.py"],
        "Pinned model/source/precision provenance join mismatch",
    )
    lora = provenance["lora"]
    _require(
        all(
            lora.get(key) == value
            for key, value in {
                "rank": 2,
                "alpha": 4,
                "dropout": 0.0,
                "seed": seed,
                "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "nonzero_B_scale": 0.02,
            }.items()
        ),
        "Pinned paired adapter initializer/settings mismatch",
    )
    _require(
        record["entry_snapshot_restored"] is True
        and record["entry_rng_reset_to_seed"] == seed
        and record["optimizer_state_initially_empty"] is True
        and wrapper["caller_host_reservation_after_cleanup"] == 0,
        "Paired snapshot/RNG/fresh optimizer or caller lease cleanup mismatch",
    )
    _require(
        wrapper["quality_admitted"] is False
        and wrapper["production_admitted"] is False
        and wrapper["final_task_ids_used"] == []
        and record["quality_admission"] is False
        and record["production_admission"] is False,
        "Training evidence made an unauthorized quality/production claim",
    )


def _final_checkpoint(wrapper: dict, record: dict, infos: list[dict], initial: dict) -> None:
    _require(
        wrapper["status"] == wrapper["arm_status"] == record["status"] == "TRAINED_CHECKPOINT_READY"
        and wrapper["optimizer_steps_completed"] == 64
        and record["optimizer_step"] is True
        and all(
            record[key] == 64
            for key in (
                "supplied_window_count",
                "planned_window_count",
                "attempted_window_count",
                "consumed_window_count",
                "completed_window_count",
                "optimizer_steps_completed",
            )
        )
        and len(record["windows"]) == 64
        and record["adapter_update_proved_after_warmup"] is True
        and record["frozen_base_invariants_pass"] is True
        and record["final_checkpoint_evaluation_eligible"] is True
        and not record.get("mandatory_admission_failure", False),
        "Incomplete training or ineligible final step64 checkpoint",
    )
    cleanup = record["cleanup"]
    _require(
        cleanup["complete"] is True
        and cleanup["errors"] == []
        and cleanup["optimizer_owned_states_dropped"] is True
        and cleanup["partial_optimizer_update_possible"] is False
        and cleanup["lease_release_proves_os_rss_release"] is False,
        "Final arm cleanup/optimizer state release failed",
    )
    _require(
        record["initial_adapter_manifest"] == initial
        and all(
            record[key] == 262144
            for key in ("input_token_count_planned", "input_token_count_consumed")
        )
        and all(
            record[key] == 179326
            for key in ("valid_target_count_planned", "valid_target_count_consumed")
        )
        and record["ordered_window_pairing_sha256"]
        == hashlib.sha256(
            b"".join(bytes.fromhex(info["pairing_sha256"]) for info in infos)
        ).hexdigest(),
        "Final ordered pairing/input/target totals mismatch",
    )
    evidence = record["windows"][63]["evidence"]
    final = evidence["adapters"]
    _require(
        record["final_checkpoint_optimizer_step"] == 64
        and record["final_adapter_file"] == final["file"] == "step-064.adapters.safetensors"
        and record["final_adapter_file_sha256"] == final["sha256"]
        and record["final_adapter_manifest"] == final["tensor_manifest"]
        and record["final_optimizer_evidence"] == evidence["optimizer"]
        and record["final_scheduler_state"] == evidence["scheduler_state"],
        "Final checkpoint pointer is not the cleaned final step64 artifacts",
    )
    _require(
        record["full_training_step_wall_seconds"]
        == [window["full_step_wall_seconds"] for window in record["windows"]]
        and _number(record["arm_wall_seconds_with_serialization"], positive=True)
        and _number(wrapper["wall_seconds_including_loading_serialization"], positive=True),
        "Incomplete arm/full-step timing evidence",
    )


def _run_summary(report: dict, wrapper: dict, record: dict) -> None:
    wrapper = wrapper if isinstance(wrapper, dict) else {}
    record = record if isinstance(record, dict) else {}
    steps = report["steps"]
    verified = [step for step in steps if step["verified"]]
    report["verified_optimizer_steps"] = len(verified)
    report["verified_input_tokens"] = sum(step["input_tokens"] for step in verified)
    report["verified_supervised_targets"] = sum(step["supervised_targets"] for step in verified)
    report["verified_contiguous_prefix_steps"] = next(
        (step["step"] - 1 for step in steps if not step["verified"]), len(steps)
    )
    report["timing"] = {
        "full_steps_seconds": math.fsum(step["full_step_wall_seconds"] for step in verified),
        "serialization_seconds": math.fsum(step["serialization_wall_seconds"] for step in verified),
        "declared_arm_seconds_including_serialization": record.get(
            "arm_wall_seconds_with_serialization"
        ),
        "declared_wrapper_seconds_including_loading_serialization": wrapper.get(
            "wall_seconds_including_loading_serialization"
        ),
        "totals_scope": "individually verified steps; arm/wrapper scopes reported separately",
    }
    report["loss"] = {
        "verified_window_losses": [
            {"step": step["step"], "loss": step["window_loss"]} for step in verified
        ],
        "target_weighted_mean_training_loss": (
            math.fsum(step["window_loss"] * step["supervised_targets"] for step in verified)
            / report["verified_supervised_targets"]
            if verified
            else None
        ),
        "scope": "training trajectory observations, not held-out quality",
    }
    observations = [step["memory_observed"] for step in verified]
    high_water = [
        item["process_lifetime_rss_high_water_bytes"]
        for item in observations
        if item["process_lifetime_rss_high_water_bytes"] is not None
    ]
    report["memory"] = {
        "ram": {
            "max_window_sampled_rss_bytes": max(
                (item["process_rss_sampled_peak_bytes"] for item in observations), default=None
            ),
            "max_process_lifetime_rss_high_water_bytes": max(high_water, default=None),
        },
        "vram": {
            "max_window_cuda_allocated_peak_bytes": max(
                (item["cuda_allocated_peak_bytes"] for item in observations), default=None
            ),
            "max_window_cuda_reserved_peak_bytes": max(
                (item["cuda_reserved_peak_bytes"] for item in observations), default=None
            ),
        },
        "scope": "separate per-arm window maxima; not simultaneous or global peak",
        "rss_sampling_scope": "sampled lower bound; lifetime high water includes earlier work",
        "lease_release_proves_os_rss_release": False,
    }


def _audit_run(
    *,
    folder: Path,
    length: int,
    seed: int,
    arm: str,
    protocol: dict,
    protocol_sha256: str,
    pack_sha256: str,
    manifest: dict,
    infos: list[dict],
) -> tuple[dict, dict | None]:
    report = {
        "length": length,
        "seed": seed,
        "arm": arm,
        "directory": folder.name,
        "status": _BLOCKED,
        "errors": [],
        "steps": [],
        "final_checkpoint_evaluation_eligible": False,
        "verified_optimizer_steps": 0,
        "verified_input_tokens": 0,
        "verified_supervised_targets": 0,
    }
    wrapper, record, initial = {}, {}, None
    try:
        _require(
            folder.name == f"L{length}-seed{seed}-{arm}",
            "Run directory does not name the requested configuration",
        )
        run_identity = _manifest_identity(folder, "run.json")
        wrapper = _read_json(data._relative(folder, "run.json"))
        report["declared"] = {"wrapper_status": wrapper.get("status")}
        arm_identity = _manifest_identity(folder, "arm/arm.json")
        record = _read_json(data._relative(folder, "arm/arm.json"))
        report["declared"] = {
            "wrapper_status": wrapper.get("status"),
            "arm_status": record.get("status"),
            **{
                key: record.get(key)
                for key in (
                    "optimizer_steps_completed",
                    "attempted_window_count",
                    "completed_window_count",
                    "input_token_count_consumed",
                    "valid_target_count_consumed",
                )
            },
        }
        shapes = _adapter_shapes()
        try:
            _validate_run(
                wrapper,
                record,
                length=length,
                seed=seed,
                arm=arm,
                protocol=protocol,
                protocol_sha256=protocol_sha256,
                pack_sha256=pack_sha256,
                manifest=manifest,
                shapes=shapes,
            )
        except Exception as error:
            report["errors"].append(_error(error, scope="run_identity_and_cleanup"))
        initial = _tensor_manifest(
            folder,
            wrapper["initial_snapshot"],
            filename="initial.adapters.safetensors",
            metadata={
                "kind": "paired-initial-adapters",
                "seed": str(seed),
                "protocol_sha256": protocol_sha256,
            },
            shapes=shapes,
        )
        _check_adapter_manifest(initial, shapes, nonzero=True)
        _require(
            initial == record["initial_adapter_manifest"],
            "Initial snapshot artifact/arm manifest mismatch",
        )
        elements = sum(math.prod(shape) for shape in shapes.values())
        snapshot = wrapper["initial_snapshot"]
        _require(
            snapshot["owned_host_tensor_bytes"] == 4 * elements
            and snapshot["host_reservation_bytes"] == 16 * elements + 2**20
            and snapshot["reservation_is_measured_peak"] is False,
            "Caller initial snapshot tensor/reservation accounting mismatch",
        )
        memory = _expected_memory(infos[0], shapes, arm)
        report["reservations"] = memory
        report["reservations_are_global_measured_peaks"] = False
        before = {name: item["sha256"] for name, item in initial.items()}
        windows = record["windows"]
        for step, info in enumerate(infos, 1):
            item = {"step": step, "verified": False}
            report["steps"].append(item)
            window = None
            try:
                _require(step <= len(windows), f"Missing arm window at step {step}")
                window = windows[step - 1]
                saved = _read_json(data._relative(folder, f"arm/step-{step:03d}.json"))
                _require(
                    saved == {**window, "nonfinite_metric_paths": []},
                    "Standalone step JSON/embedded arm window mismatch",
                )
                _validate_step(
                    window,
                    info,
                    folder=data._relative(folder, "arm"),
                    arm=arm,
                    seed=seed,
                    length=length,
                    step=step,
                    initial=initial,
                    before=before,
                    memory=memory,
                    protocol=protocol,
                    protocol_sha256=protocol_sha256,
                    shapes=shapes,
                )
                item.update(
                    verified=True,
                    input_tokens=info["input_token_count"],
                    supervised_targets=info["valid_target_count"],
                    window_loss=window["window_loss"],
                    full_step_wall_seconds=window["full_step_wall_seconds"],
                    serialization_wall_seconds=window["evidence_serialization_wall_seconds"],
                    memory_observed=window["memory_observed"],
                )
            except Exception as error:
                item["error"] = _error(error)
                report["errors"].append(_error(error, scope=f"step-{step:03d}"))
            finally:
                # Only hashes survive the step. No tensor or whole safe file is retained.
                if window is not None:
                    before = window.get("adapter_hashes_after", {})
        try:
            _final_checkpoint(wrapper, record, infos, initial)
        except Exception as error:
            report["errors"].append(_error(error, scope="final_checkpoint"))
        if not report["errors"]:
            _require(
                run_identity == _manifest_identity(folder, "run.json")
                and arm_identity == _manifest_identity(folder, "arm/arm.json"),
                "Training manifests changed during the audit",
            )
            report["status"] = _ARM_VERIFIED
            report["final_checkpoint_evaluation_eligible"] = True
            report["final_checkpoint"] = {
                "file": f"arm/{record['final_adapter_file']}",
                "sha256": record["final_adapter_file_sha256"],
                "optimizer_step": 64,
            }
            report["checkpoint_identity"] = {
                "directory": folder.name,
                "length": length,
                "seed": seed,
                "arm": arm,
                "protocol_sha256": protocol_sha256,
                "arm_manifest_file": f"{folder.name}/arm/arm.json",
                "arm_manifest_file_sha256": arm_identity["sha256"],
                "arm_manifest_bytes": arm_identity["bytes"],
                "run_manifest_file": f"{folder.name}/run.json",
                "run_manifest_file_sha256": run_identity["sha256"],
                "run_manifest_bytes": run_identity["bytes"],
                "final_step64": {
                    **record["windows"][63]["evidence"]["adapters"],
                    "path": f"{folder.name}/arm/step-064.adapters.safetensors",
                    "metadata": {
                        "arm": arm,
                        "seed": str(seed),
                        "optimizer_step": "64",
                        "protocol_sha256": protocol_sha256,
                        "kind": _KINDS["adapters"],
                    },
                },
                "initial_snapshot": {
                    **snapshot,
                    "path": f"{folder.name}/initial.adapters.safetensors",
                    "metadata": {
                        "kind": "paired-initial-adapters",
                        "seed": str(seed),
                        "protocol_sha256": protocol_sha256,
                    },
                },
                "initializer": record["model_provenance"]["lora"],
                "data": {
                    "pack_manifest_file_sha256": pack_sha256,
                    "pack_manifest_payload_sha256": manifest["manifest_payload_sha256"],
                    "dataset_manifest_file_sha256": protocol["data"][
                        "dataset_manifest_file_sha256"
                    ],
                    "windows_file": manifest["files"][f"windows_L{length}"],
                    "ordered_window_pairing_sha256": record["ordered_window_pairing_sha256"],
                    "input_token_count_consumed": 262144,
                    "valid_target_count_consumed": 179326,
                },
                "evaluation_eligible": True,
            }
    except Exception as error:
        report["status"] = _BLOCKED
        report["final_checkpoint_evaluation_eligible"] = False
        report.pop("checkpoint_identity", None)
        report["errors"].append(_error(error, scope="required_files_or_initial_snapshot"))
    _run_summary(report, wrapper, record)
    return report, initial


def _gradient_diagnostic(
    left: Path, right: Path, *, step: int, same_parameters_before: bool
) -> dict:
    """Tensorwise CPU diagnostics without old G4 thresholds or a quality verdict."""
    from safetensors import safe_open

    reference_squared, delta_squared, cosines = [], [], []
    exact = True
    with (
        safe_open(str(left), framework="pt", device="cpu") as reference,
        safe_open(
            str(right),
            framework="pt",
            device="cpu",
        ) as candidate,
    ):
        names = list(reference.keys())
        _require(
            set(names) == set(candidate.keys()) and names, "Diagnostic gradient coverage mismatch"
        )
        for name in names:
            ref_tensor, got_tensor = reference.get_tensor(name), candidate.get_tensor(name)
            ref = got = None
            try:
                _require(ref_tensor.shape == got_tensor.shape, "Diagnostic gradient shape mismatch")
                exact = exact and baseline.tensor_hash(ref_tensor) == baseline.tensor_hash(
                    got_tensor
                )
                ref, got = ref_tensor.double().reshape(-1), got_tensor.double().reshape(-1)
                norm, got_norm = float(ref.norm()), float(got.norm())
                _require(
                    norm > 0 and got_norm > 0 and math.isfinite(norm) and math.isfinite(got_norm),
                    "Nonfinite/vacuous diagnostic gradients",
                )
                delta_squared.append(float((got - ref).square().sum()))
                reference_squared.append(norm**2)
                cosines.append(float(ref.dot(got)) / (norm * got_norm))
            finally:
                # Double temporaries are bounded by one named adapter tensor pair.
                del ref_tensor, got_tensor, ref, got
    return {
        "step": step,
        "tensor_count": len(names),
        "gradient_byte_parity": exact,
        "global_relative_l2": math.sqrt(math.fsum(delta_squared) / math.fsum(reference_squared)),
        "minimum_tensor_cosine": min(cosines),
        "same_adapter_parameters_before": same_parameters_before,
        "scope": (
            "POST_UPDATE_DIVERGENT_TRAJECTORY_DIAGNOSTIC"
            if step >= 3
            else "PAIRED_INITIAL_PARAMETER_POINT_DIAGNOSTIC"
        ),
        "pure_compression_error": False,
        "numerical_acceptance": "NOT_ASSESSED",
        "quality_acceptance": "NOT_MEASURED",
        "natural_candidate_rmr": None,
    }


def _paired_diagnostics(root: Path, left: dict, right: dict) -> list[dict]:
    records = []
    for step in range(1, 65):
        left_folder = data._relative(root, left["directory"])
        right_folder = data._relative(root, right["directory"])
        ref = _read_json(data._relative(left_folder, f"arm/step-{step:03d}.json"))
        got = _read_json(data._relative(right_folder, f"arm/step-{step:03d}.json"))
        # Recheck gradient file bytes immediately before reading diagnostics.
        for folder, window in ((left_folder, ref), (right_folder, got)):
            descriptor = window["evidence"]["gradients"]
            path = data._relative(folder, f"arm/step-{step:03d}.gradients.safetensors")
            _require(
                path.stat().st_size == descriptor["bytes"]
                and training._sha256(path) == descriptor["sha256"],
                "Gradient evidence changed before paired diagnostics",
            )
        diagnostic = _gradient_diagnostic(
            data._relative(left_folder, f"arm/step-{step:03d}.gradients.safetensors"),
            data._relative(right_folder, f"arm/step-{step:03d}.gradients.safetensors"),
            step=step,
            same_parameters_before=ref["adapter_hashes_before"] == got["adapter_hashes_before"],
        )
        diagnostic.update(
            length=left["length"],
            seed=left["seed"],
            bf16_training_loss=ref["window_loss"],
            candidate_training_loss=got["window_loss"],
            training_loss_difference=got["window_loss"] - ref["window_loss"],
        )
        records.append(diagnostic)
    return records


def _pair_initials(entries: list[tuple[dict, dict]], protocol: dict) -> list[dict]:
    """Veto every member of a mismatched seed pair, not just the last one read."""
    summaries = []
    required = len(protocol["lengths"]) * len(protocol["arms"])
    for seed in protocol["seeds"]:
        members = [(report, initial) for report, initial in entries if report["seed"] == seed]
        matched = bool(members) and all(initial == members[0][1] for _, initial in members)
        summaries.append(
            {
                "seed": seed,
                "required_snapshot_count": required,
                "validated_snapshot_count": len(members),
                "all_available_snapshots_equal": matched if members else None,
                "complete_pairing": len(members) == required and matched,
            }
        )
        if members and not matched:
            for report, _ in members:
                report["errors"].append(
                    {
                        "type": "ValueError",
                        "scope": "seed_pairing",
                        "message": "Within-seed cross-arm/length initial snapshot mismatch",
                    }
                )
                report["status"] = _BLOCKED
                report["final_checkpoint_evaluation_eligible"] = False
    return summaries


def audit_training(
    *, root: Path, pack_root: Path, pack_sha256: str, protocol_path: Path, output: Path
) -> dict:
    """Audit every protocol-required run. Write a fresh report even when evidence is incomplete."""
    if os.path.lexists(output):
        raise ValueError("Existing training audit evidence is immutable; output must not exist")
    result = {
        "schema": "p3-stage2-training-audit-v1",
        "status": _BLOCKED,
        "errors": [],
        "runs": [],
        "paired_gradient_diagnostics": [],
        "verified_final_checkpoints": {},
        "required_run_count": None,
        "verified_run_count": 0,
        "quality_admitted": False,
        "production_admitted": False,
        "quality_status": "NOT_MEASURED",
        "natural_fp8_rmr_measured": False,
        "old_G4_acceptance_assessed": False,
        "gradient_scope": "Post-positive-update gradients compare divergent training trajectories",
        "memory_scope": "RAM, VRAM and tensor reservations separate; no global peak claim",
        "frozen_base_proof_scope": (
            "recorded identity/storage/version/no-gradient guards, not base bytes"
        ),
        "source_scope": (
            "frozen pack bytes and source/window joins, not a new tokenizer/overlap audit"
        ),
        "execution_scope": "saved CPU artifacts only; no model, optimizer or training execution",
    }
    try:
        protocol, protocol_sha256, configurations = _load_protocol(protocol_path)
        result.update(
            protocol_sha256=protocol_sha256,
            pack_manifest_file_sha256=pack_sha256,
            required_run_count=len(configurations),
        )
        result["runs"] = [
            {
                **config,
                "status": _BLOCKED,
                "errors": [],
                "steps": [],
                "verified_optimizer_steps": 0,
                "final_checkpoint_evaluation_eligible": False,
            }
            for config in configurations
        ]
        manifest, infos = _load_pack(pack_root, pack_sha256, protocol)
        result["pack_source"] = {
            "manifest_payload_sha256": manifest["manifest_payload_sha256"],
            "dataset_manifest_file_sha256": manifest["dataset_manifest_file"]["sha256"],
            "window_count_per_length": 64,
            "input_tokens_per_arm": 262144,
            "supervised_targets_per_arm": 179326,
            "ordered_window_pairing_sha256": manifest["flat_pairing"][str(protocol["lengths"][0])][
                "ordered_window_pairing_sha256"
            ],
        }
        paired_initial = []
        for index, config in enumerate(configurations):
            try:
                folder = data._relative(root, config["directory"])
                report, initial = _audit_run(
                    folder=folder,
                    length=config["length"],
                    seed=config["seed"],
                    arm=config["arm"],
                    protocol=protocol,
                    protocol_sha256=protocol_sha256,
                    pack_sha256=pack_sha256,
                    manifest=manifest,
                    infos=infos[config["length"]],
                )
                result["runs"][index] = report
                if initial is not None:
                    paired_initial.append((report, initial))
            except Exception as error:
                result["runs"][index]["errors"].append(_error(error, scope="run_directory"))
        result["initial_snapshot_pairing"] = _pair_initials(paired_initial, protocol)
        verified = [report for report in result["runs"] if report["status"] == _ARM_VERIFIED]
        result["verified_run_count"] = len(verified)
        for report in result["runs"]:
            identity = report.pop("checkpoint_identity", None)
            if report["status"] == _ARM_VERIFIED and identity is not None:
                result["verified_final_checkpoints"][report["directory"]] = identity
        for length, seed in product(protocol["lengths"], protocol["seeds"]):
            pair = {
                report["arm"]: report
                for report in verified
                if report["length"] == length and report["seed"] == seed
            }
            if set(pair) == set(protocol["arms"]):
                try:
                    result["paired_gradient_diagnostics"].extend(
                        _paired_diagnostics(
                            root,
                            pair["bf16_l2l"],
                            pair["oracle_block64_e4m3"],
                        )
                    )
                except Exception as error:
                    result["errors"].append(
                        _error(error, scope=f"L{length}-seed{seed}-diagnostics")
                    )
        if (
            len(verified) == len(configurations)
            and len(result["verified_final_checkpoints"]) == len(configurations)
            and all(item["complete_pairing"] for item in result["initial_snapshot_pairing"])
            and not result["errors"]
        ):
            result["status"] = VERIFIED
    except Exception as error:
        result["errors"].append(_error(error, scope="protocol_or_pack_prerequisite"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--pack-root", required=True, type=Path)
    parser.add_argument("--pack-sha256", required=True)
    parser.add_argument("--protocol-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit_training(
        root=args.root,
        pack_root=args.pack_root,
        pack_sha256=args.pack_sha256,
        protocol_path=args.protocol_path,
        output=args.output,
    )
    Console().print(
        {
            "status": result["status"],
            "verified_runs": result["verified_run_count"],
            "required_runs": result["required_run_count"],
            "output": str(args.output),
        }
    )
    raise SystemExit(0 if result["status"] == VERIFIED else 1)


if __name__ == "__main__":
    main()
