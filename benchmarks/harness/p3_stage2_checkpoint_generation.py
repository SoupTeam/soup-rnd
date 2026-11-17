"""One frozen, batch-aligned Stage2 shard from an audited final step64 checkpoint.

Run only in a parent-authorized fresh Kaggle T4 process. This module does not
schedule jobs, score answers, retry requests or establish full-study coverage.
"""

from __future__ import annotations

import argparse
import copy
import gc
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from rich.console import Console

from . import p3_realdata_baseline as baseline
from . import p3_realdata_model as native
from . import p3_stage2_dataset as data
from . import p3_stage2_generation as generation
from . import p3_stage2_preflight as preflight
from . import p3_stage2_training_audit as audit
from .p3_fp8_smoke import _lifetime_rss_peak, _rss

_BENCHMARKS = ("gsm8k", "humaneval", "apps", "math")
_DATA_STATUS = "LEXICAL_AUDIT_COMPLETE_RUNTIME_AND_PRECISION_PENDING"
_FINAL_FILE = "step-064.adapters.safetensors"
_FINAL_KIND = "updated-adapter-evaluation-eligibility-in-arm-json"


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _read_json(path: Path) -> dict:
    value = data._json(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "json_object_required")
    return value


def _descriptor(path: Path, relative: str | None = None) -> dict:
    return {
        "path": relative if relative is not None else str(path),
        "bytes": path.stat().st_size,
        "sha256": generation._file_hash(path),
    }


def _manifest(path: Path) -> dict:
    """Validate the payload identity without opening scoring/reference artifacts."""
    value = _read_json(path)
    _require(
        value.get("manifest_payload_sha256")
        == generation._object_hash(
            {key: item for key, item in value.items() if key != "manifest_payload_sha256"}
        ),
        "manifest_payload_sha256_mismatch",
    )
    _require(isinstance(value.get("files"), dict), "manifest_files_required")
    return value


def _admit_protocol(
    path: Path, expected: str, arm: str, seed: int, length: int, benchmark: str
) -> dict:
    _require(
        expected == audit.FROZEN_PROTOCOL_SHA256 and generation._file_hash(path) == expected,
        "frozen_protocol_file_sha256_mismatch",
    )
    protocol = _read_json(path)
    _require(
        protocol.get("schema") == "p3-stage2-frozen-paired-study-v1",
        "frozen_protocol_schema_required",
    )
    _require(
        type(seed) is int
        and seed in protocol["seeds"]
        and type(length) is int
        and length in protocol["lengths"]
        and arm in protocol["arms"]
        and benchmark in _BENCHMARKS,
        "unapproved_frozen_configuration",
    )
    _require(
        protocol["generation"]["batch_size"] == generation.BATCH_SIZE == 8
        and protocol["model"]["id"] == native.MODEL_ID
        and protocol["model"]["revision"] == native.REVISION,
        "pinned_model_or_batch_policy_mismatch",
    )
    root = Path(__file__).parents[2]
    for module in (native, baseline, data, generation, preflight, audit.training):
        relative = "benchmarks/harness/" + Path(module.__file__).name
        _require(
            protocol["source_files_sha256"].get(relative) == generation._file_hash(root / relative),
            "frozen_dependency_source_changed",
        )
    return protocol


def _admit_dataset(
    root: Path, protocol: dict, benchmark: str, start: int, stop: int
) -> tuple[list[dict], dict]:
    _require(benchmark in _BENCHMARKS, "unsupported_benchmark")
    _require(
        generation._file_hash(root / "manifest.json")
        == protocol["data"]["dataset_manifest_file_sha256"],
        "frozen_dataset_manifest_file_sha256_mismatch",
    )
    manifest = _manifest(root / "manifest.json")
    _require(
        manifest.get("schema") == data.POLICY
        and manifest.get("policy") == data.POLICY
        and manifest.get("status") == _DATA_STATUS
        and set(manifest.get("evaluations", {})) == set(_BENCHMARKS),
        "complete_frozen_dataset_required",
    )
    counts = protocol["data"]["retained_evaluation_tasks"]
    _require(set(counts) == set(_BENCHMARKS), "four_frozen_benchmark_counts_required")
    total = counts[benchmark]
    _require(
        type(start) is int
        and type(stop) is int
        and 0 <= start < stop <= total
        and start % generation.BATCH_SIZE == 0
        and (stop % generation.BATCH_SIZE == 0 or stop == total),
        "shard_must_preserve_original_batch8_boundaries",
    )
    selected = []
    selected_descriptor = None
    for name in _BENCHMARKS:
        folder = data._relative(root, name)
        endpoint_path = folder / "manifest.json"
        _require(
            manifest["files"].get(f"{name}_manifest")
            == _descriptor(endpoint_path, f"{name}/manifest.json"),
            "root_endpoint_manifest_file_binding_mismatch",
        )
        endpoint = _manifest(endpoint_path)
        _require(
            endpoint == manifest["evaluations"][name]
            and endpoint.get("schema") == "p3-stage2-filtered-evaluation-v1"
            and endpoint.get("policy") == data.POLICY
            and endpoint.get("status") == _DATA_STATUS
            and endpoint.get("benchmark") == name
            and endpoint.get("generation_policy") == generation._frozen_policy(name),
            "frozen_endpoint_identity_mismatch",
        )
        expected_count = counts[name]
        _require(
            type(expected_count) is int
            and expected_count > 0
            and endpoint.get("retained_row_count") == expected_count
            and manifest["counts"]["evaluation_retained"][name] == expected_count,
            "frozen_complete_benchmark_count_mismatch",
        )
        descriptor = endpoint["files"]["generation"]
        _require(
            descriptor.get("path") == "generation.jsonl"
            and descriptor.get("rows") == expected_count
            and manifest["files"].get(f"{name}_generation")
            == {
                **descriptor,
                "path": f"{name}/generation.jsonl",
            },
            "generation_file_descriptor_join_mismatch",
        )
        path = data._artifact(folder, descriptor)
        seen = {
            key: set() for key in ("task_id", "source_id", "source_row_sha256", "prompt_sha256")
        }
        ordered = []
        for ordinal, row in enumerate(data._jsonl(path)):
            admitted = generation._validated_records([row], name)[0]
            for field, identities in seen.items():
                identity = generation._compact(admitted[field])
                _require(identity not in identities, "duplicate_complete_endpoint_identity")
                identities.add(identity)
            ordered.append(admitted["task_id"])
            if name == benchmark and start <= ordinal < stop:
                selected.append(admitted)
        _require(
            len(ordered) == expected_count and ordered == endpoint["retained_task_ids"],
            "complete_frozen_endpoint_order_required",
        )
        _require(
            _descriptor(path, "generation.jsonl")
            == {field: descriptor[field] for field in ("path", "bytes", "sha256")},
            "generation_file_changed_during_admission",
        )
        if name == benchmark:
            selected_descriptor = {**descriptor, "path": f"{name}/generation.jsonl"}
    _require(len(selected) == stop - start, "complete_selected_shard_required")
    return selected, {
        "dataset_manifest_file_sha256": protocol["data"]["dataset_manifest_file_sha256"],
        "dataset_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "benchmark": benchmark,
        "complete_benchmark_row_count": total,
        "start_index": start,
        "stop_index": stop,
        "expected_task_ids": [row["task_id"] for row in selected],
        "selected_source_rows_sha256": generation._object_hash(selected),
        "generation_file": selected_descriptor,
    }


def _admit_checkpoint(
    matrix_root: Path,
    audit_path: Path,
    audit_sha256: str,
    protocol: dict,
    protocol_sha256: str,
    arm: str,
    seed: int,
    length: int,
) -> dict:
    _require(
        isinstance(audit_sha256, str)
        and generation._SHA256.fullmatch(audit_sha256) is not None
        and generation._file_hash(audit_path) == audit_sha256,
        "externally_frozen_training_audit_file_sha256_mismatch",
    )
    report = _read_json(audit_path)
    expected_keys = {
        f"L{size}-seed{value}-{variant}"
        for size in protocol["lengths"]
        for value in protocol["seeds"]
        for variant in protocol["arms"]
    }
    _require(
        report.get("schema") == "p3-stage2-training-audit-v1"
        and report.get("status") == audit.VERIFIED
        and report.get("required_run_count") == report.get("verified_run_count") == 20
        and report.get("errors") == []
        and report.get("protocol_sha256") == protocol_sha256
        and report.get("pack_manifest_file_sha256")
        == protocol["data"]["training_pack_manifest_file_sha256"]
        and len(expected_keys) == 20
        and set(report.get("verified_final_checkpoints", {})) == expected_keys,
        "full20_verified_training_evidence_required",
    )
    runs = report.get("runs", [])
    _require(
        len(runs) == 20
        and {item.get("directory") for item in runs} == expected_keys
        and all(
            item.get("status") == "TRAINING_ARM_EVIDENCE_VERIFIED"
            and item.get("errors") == []
            and item.get("verified_optimizer_steps") == 64
            and item.get("final_checkpoint_evaluation_eligible") is True
            for item in runs
        ),
        "all20_clean_step64_runs_required",
    )
    key = f"L{length}-seed{seed}-{arm}"
    entry = report["verified_final_checkpoints"][key]
    _require(
        all(
            entry.get(field) == value
            for field, value in {
                "directory": key,
                "length": length,
                "seed": seed,
                "arm": arm,
                "protocol_sha256": protocol_sha256,
                "evaluation_eligible": True,
            }.items()
        ),
        "audited_checkpoint_scope_identity_mismatch",
    )
    paths = {}
    for prefix, relative in (("arm", f"{key}/arm/arm.json"), ("run", f"{key}/run.json")):
        _require(
            entry.get(f"{prefix}_manifest_file") == relative,
            "training_manifest_path_identity_mismatch",
        )
        path = data._relative(matrix_root, relative)
        _require(
            path.stat().st_size == entry[f"{prefix}_manifest_bytes"]
            and generation._file_hash(path) == entry[f"{prefix}_manifest_file_sha256"],
            "audited_training_manifest_bytes_or_hash_mismatch",
        )
        paths[prefix] = path
    wrapper, record = _read_json(paths["run"]), _read_json(paths["arm"])
    _require(
        wrapper.get("schema") == "p3-stage2-pretrained-arm-v1"
        and record.get("schema") == "p3-stage2-arm-v1"
        and all(
            item.get("arm") == arm
            and item.get("seed") == seed
            and item.get("preflight") is False
            and item.get("protocol_sha256") == protocol_sha256
            and item.get("status") == "TRAINED_CHECKPOINT_READY"
            and item.get("optimizer_steps_completed") == 64
            for item in (wrapper, record)
        )
        and wrapper.get("length") == length
        and wrapper.get("arm_status") == "TRAINED_CHECKPOINT_READY",
        "clean_final_training_scope_required",
    )
    cleanup = record.get("cleanup", {})
    _require(
        record.get("final_checkpoint_evaluation_eligible") is True
        and record.get("final_checkpoint_optimizer_step") == 64
        and record.get("frozen_base_invariants_pass") is True
        and record.get("adapter_update_proved_after_warmup") is True
        and not record.get("mandatory_admission_failure", False)
        and cleanup.get("complete") is True
        and cleanup.get("errors") == []
        and cleanup.get("optimizer_owned_states_dropped") is True
        and cleanup.get("partial_optimizer_update_possible") is False
        and wrapper.get("caller_host_reservation_after_cleanup") == 0,
        "final_training_eligibility_or_cleanup_failed",
    )
    source = entry["data"]
    _require(
        source.get("dataset_manifest_file_sha256")
        == protocol["data"]["dataset_manifest_file_sha256"]
        and source.get("pack_manifest_file_sha256")
        == wrapper.get("pack_manifest_file_sha256")
        == protocol["data"]["training_pack_manifest_file_sha256"]
        and source.get("pack_manifest_payload_sha256")
        == wrapper.get("pack_manifest_payload_sha256")
        and source.get("windows_file") == wrapper.get("windows_file")
        and source.get("ordered_window_pairing_sha256")
        == record.get("ordered_window_pairing_sha256")
        and source.get("input_token_count_consumed")
        == record.get("input_token_count_consumed")
        == 262144
        and source.get("valid_target_count_consumed")
        == record.get("valid_target_count_consumed")
        == 179326,
        "audited_training_data_or_pairing_join_mismatch",
    )
    initializer = entry["initializer"]
    _require(
        initializer == record["model_provenance"]["lora"]
        and all(
            initializer.get(field) == value
            for field, value in {
                "rank": 2,
                "alpha": 4,
                "dropout": 0.0,
                "seed": seed,
                "targets": list(native.TARGETS),
                "nonzero_B_scale": 0.02,
            }.items()
        ),
        "audited_lora_initializer_mismatch",
    )
    final = entry["final_step64"]
    expected_metadata = {
        "arm": arm,
        "seed": str(seed),
        "optimizer_step": "64",
        "protocol_sha256": protocol_sha256,
        "kind": _FINAL_KIND,
    }
    _require(
        final.get("file") == record.get("final_adapter_file") == _FINAL_FILE
        and final.get("path") == f"{key}/arm/{_FINAL_FILE}"
        and final.get("metadata") == expected_metadata
        and final.get("sha256") == record.get("final_adapter_file_sha256")
        and final.get("tensor_manifest") == record.get("final_adapter_manifest")
        and len(record.get("windows", [])) == 64
        and {field: final[field] for field in ("file", "bytes", "sha256", "tensor_manifest")}
        == record["windows"][63]["evidence"]["adapters"],
        "final_pointer_must_name_audited_updated_step64",
    )
    initial = entry["initial_snapshot"]
    _require(
        initial.get("path") == f"{key}/initial.adapters.safetensors"
        and initial.get("tensor_manifest")
        == record.get("initial_adapter_manifest")
        == wrapper["initial_snapshot"]["tensor_manifest"]
        and all(
            initial.get(field) == wrapper["initial_snapshot"].get(field)
            for field in ("file", "bytes", "sha256")
        )
        and final["sha256"] != initial["sha256"]
        and any(
            item["sha256"] != initial["tensor_manifest"][name]["sha256"]
            for name, item in final["tensor_manifest"].items()
        ),
        "final_checkpoint_cannot_be_initial_adapter_snapshot",
    )
    # Revalidate bytes, safetensors metadata, all192 names/shapes and finite FP32 contents.
    shapes = audit._adapter_shapes()
    observed = audit._tensor_manifest(
        data._relative(matrix_root, f"{key}/arm"),
        final,
        filename=_FINAL_FILE,
        metadata=expected_metadata,
        shapes=shapes,
    )
    audit._check_adapter_manifest(observed, shapes, nonzero=False)
    _require(
        generation._file_hash(audit_path) == audit_sha256
        and all(
            generation._file_hash(paths[prefix]) == entry[f"{prefix}_manifest_file_sha256"]
            for prefix in paths
        ),
        "training_evidence_changed_during_admission",
    )
    return copy.deepcopy(entry)


def _loaded_adapter_manifest(model: Any) -> dict:
    # Saved manifests use CPU reductions. Hash and describe one CPU tensor at a
    # time so cross-device FP64 norm reduction order is not an identity guard.
    result = {}
    for name, parameter in baseline.trainables(model).items():
        value = parameter.detach().cpu()
        try:
            result.update(baseline.gradient_manifest({name: value}))
        finally:
            del value
    return result


def _restore_final(model: Any, tensors: dict, descriptor: dict, seed: int) -> dict:
    _require(
        baseline.gradient_manifest(tensors) == descriptor["tensor_manifest"],
        "final_adapter_tensor_manifest_changed_before_restore",
    )
    baseline.restore(model, tensors, seed)
    loaded = _loaded_adapter_manifest(model)
    _require(
        loaded == descriptor["tensor_manifest"], "loaded_final_adapter_tensor_manifest_mismatch"
    )
    return loaded


def _admit_runtime() -> tuple[Any, dict]:
    _require(
        sys.platform.startswith("linux") and Path("/kaggle/working").is_dir(),
        "fresh_kaggle_cuda_process_required",
    )
    _require("torch" not in sys.modules, "fresh_process_before_torch_import_required")
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") != "-1", "cuda_explicitly_disabled_no_fallback")
    _require(
        not any(
            os.environ.get(key)
            for key in (
                "CUDA_INJECTION64_PATH",
                "NVTX_INJECTION64_PATH",
                "NSYS_PROFILING_SESSION_ID",
            )
        ),
        "external_profiler_injection_not_admitted",
    )
    _require(
        not any(
            os.environ.get(key, "0") not in ("", "0")
            for key in (
                "CUBLAS_LOGINFO_DBG",
                "CUBLAS_LOGTRACE_DBG",
                "CUBLAS_LOGWARN_DBG",
                "CUBLASLT_LOG_LEVEL",
                "CUBLASLT_LOG_MASK",
                "CUDA_PROFILE",
                "COMPUTE_PROFILE",
            )
        ),
        "logging_or_profiling_must_not_change_measured_cost",
    )
    import torch

    _require(torch.cuda.is_available(), "cuda_required_no_cpu_fallback")
    _require(
        torch.cuda.get_device_name(0) == "Tesla T4"
        and tuple(torch.cuda.get_device_capability(0)) == (7, 5)
        and not torch.cuda.is_bf16_supported(including_emulation=False)
        and torch.cuda.is_bf16_supported(),
        "frozen_t4_bf16_environment_required",
    )
    _require(torch.cuda.memory_allocated(0) == 0, "fresh_process_has_existing_cuda_allocations")
    flags = generation._precision_flags()
    _require(
        flags["cuda_matmul_allow_tf32"] is False
        and flags["cudnn_allow_tf32"] is True
        and flags["autocast_enabled_cuda"] is False
        and flags["autocast_enabled_cpu"] is False
        and os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") != "1"
        and os.environ.get("NVIDIA_TF32_OVERRIDE") != "1"
        and not os.environ.get("CUBLAS_EMULATE_SINGLE_PRECISION")
        and not os.environ.get("CUBLAS_EMULATE_DOUBLE_PRECISION"),
        "frozen_precision_permissions_required_no_silent_override",
    )
    return torch, {
        "gpu": "Tesla T4",
        "device": "cuda:0",
        "compute_capability": [7, 5],
        "native_bf16_support": False,
        "emulated_bf16_tensor_support": True,
        "precision_flags": flags,
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "weight_format": "dense BF16 base; FP32 LoRA; native eager inference",
        "native_bf16_tensor_cores_claimed": False,
    }


def _wrapper_measurement(started: float, rss_before: int | None) -> dict:
    return {
        "scope": (
            "Whole wrapper admission, checkpoint/base loading, native generation and artifact "
            "writes before final teardown/run.json export"
        ),
        "wall_seconds_before_cleanup": time.perf_counter() - started,
        "process_rss_before_bytes": rss_before,
        "process_rss_after_bytes": _rss(),
        "process_lifetime_rss_high_water_bytes": _lifetime_rss_peak(),
        "process_lifetime_rss_scope": "Process lifetime, not an isolated wrapper peak",
        "whole_wrapper_cuda_peak_bytes": None,
        "whole_wrapper_cuda_peak_status": "NOT_ESTABLISHED_PER_CALL_PEAK_COUNTER_RESETS",
        "per_call_measurements": "generation/batches.jsonl",
        "reservation_sum_is_peak": False,
        "model_or_workspace_cleanup_in_measured_cost": False,
    }


def _cleanup_cuda(torch: Any, before: int) -> dict:
    torch.cuda.synchronize()
    after_model = torch.cuda.memory_allocated()
    # Pinned PyTorch cuBLAS workspaces survive tensor deletion. Release them only
    # after recording measurements, as in the existing evaluation preflight.
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated()
    if allocated > before:
        raise RuntimeError("cuda_allocations_remain_after_model_and_workspace_cleanup")
    return {
        "complete": True,
        "cuda_allocated_before_model_loading_bytes": before,
        "cuda_allocated_after_model_release_bytes": after_model,
        "cuda_allocated_after_workspace_release_bytes": allocated,
        "cuda_reserved_after_empty_cache_bytes": torch.cuda.memory_reserved(),
        "cublas_workspace_cache_explicitly_released": True,
        "workspace_release_after_measurements_only": True,
        "python_model_references_released": True,
        "os_rss_release_proven": False,
    }


def _complete_generation(folder: Path, result: dict, admission: dict) -> None:
    saved = _read_json(folder / "manifest.json")
    request = _read_json(folder / "request.json")
    expected = admission["expected_task_ids"]
    _require(
        saved == result
        and saved.get("status") == "COMPLETE"
        and saved.get("selected_shard_complete") is True
        and saved.get("requested_rows") == saved.get("completed_rows") == len(expected)
        and saved.get("records_sha256")
        == request.get("records_sha256")
        == admission["selected_source_rows_sha256"]
        and request.get("ordered_task_ids") == expected
        and [row["task_id"] for row in data._jsonl(folder / "generations.jsonl")] == expected,
        "generation_manifest_and_complete_ordered_selected_shard_required",
    )


def _fresh_output(output: Path, inputs: Sequence[Path], cache: Path) -> Path:
    if os.path.lexists(output):
        raise FileExistsError("existing_output_preserved")
    out = os.path.realpath(output)
    cache_name = os.path.realpath(cache)
    for source in inputs:
        source_name = os.path.realpath(source)
        for destination in (out, cache_name):
            try:
                overlaps = os.path.commonpath((source_name, destination)) in (
                    source_name,
                    destination,
                )
            except ValueError:
                overlaps = False
            _require(not overlaps, "output_and_cache_must_not_overlap_frozen_inputs")
    try:
        overlap = os.path.commonpath((out, cache_name)) in (out, cache_name)
    except ValueError:
        overlap = False
    _require(not overlap, "output_and_cache_must_be_separate")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=False)
    return root


def _failure(root: Path, result: dict, stage: str, error: BaseException) -> None:
    # Keep bounded diagnostics in private evidence, never echo source text to the console.
    # Retain each failure without overwriting earlier evidence.
    failure = {
        "stage": stage,
        "error_type": type(error).__name__,
        "reason": str(error)[:4096],
        "retry_attempted": False,
    }
    failures = result.setdefault("failures", [])
    generation._write_json(root / f"failure-{len(failures):02d}.json", failure)
    failures.append(failure)
    result.update(status="FAILED", selected_shard_complete=False)


def run_checkpoint_shard(
    *,
    matrix_root: Path,
    training_audit_path: Path,
    training_audit_sha256: str,
    dataset_root: Path,
    protocol_path: Path,
    protocol_sha256: str,
    arm: str,
    seed: int,
    length: int,
    benchmark: str,
    start_index: int,
    stop_index: int,
    cache_dir: Path,
    output_dir: Path,
) -> dict:
    """Load the exact audited final adapter and execute the selected shard once."""
    root = _fresh_output(
        output_dir, (matrix_root, dataset_root, training_audit_path, protocol_path), cache_dir
    )
    started = time.perf_counter()
    result = {
        "schema": "p3-stage2-checkpoint-generation-v1",
        "status": "STARTED",
        "arm": arm,
        "seed": seed,
        "length": length,
        "benchmark": benchmark,
        "selected_shard_complete": False,
        "global_required_coverage_verified": False,
        "quality_admitted": False,
        "production_admitted": False,
        "generated_code_executed": False,
        "retries": 0,
        "fallback_precision": False,
        "implementation": {
            Path(module.__file__).name: generation._file_hash(Path(module.__file__))
            for module in (
                sys.modules[__name__],
                audit,
                generation,
                preflight,
                native,
                baseline,
                data,
            )
        },
    }
    stage = "frozen_protocol_and_complete_dataset_admission"
    torch = model = tokenizer = tensors = metadata = rng = None
    original_flags = None
    allocator_before = None
    rss_before = _rss()
    measured = None
    try:
        protocol = _admit_protocol(protocol_path, protocol_sha256, arm, seed, length, benchmark)
        records, data_binding = _admit_dataset(
            dataset_root, protocol, benchmark, start_index, stop_index
        )
        stage = "fresh_kaggle_t4_precision_admission"
        torch, environment = _admit_runtime()
        result["environment"] = environment
        original_flags = environment["precision_flags"]
        allocator_before = torch.cuda.memory_allocated(0)
        import numpy as np

        rng = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state_all(),
        )
        stage = "audited_final_step64_checkpoint_admission"
        entry = _admit_checkpoint(
            matrix_root,
            training_audit_path,
            training_audit_sha256,
            protocol,
            protocol_sha256,
            arm,
            seed,
            length,
        )
        admission = {
            "schema": "p3-stage2-checkpoint-generation-admission-v1",
            **data_binding,
            "protocol_file_sha256": protocol_sha256,
            "training_audit_file_sha256": training_audit_sha256,
            "training_arm_manifest_file_sha256": entry["arm_manifest_file_sha256"],
            "checkpoint_file_sha256": entry["final_step64"]["sha256"],
            "arm": arm,
            "seed": seed,
            "length": length,
            "training_run_key": entry["directory"],
            "training_audit": _descriptor(training_audit_path),
            "training_arm_manifest": {
                "path": entry["arm_manifest_file"],
                "bytes": entry["arm_manifest_bytes"],
                "sha256": entry["arm_manifest_file_sha256"],
            },
            "training_run_manifest": {
                "path": entry["run_manifest_file"],
                "bytes": entry["run_manifest_bytes"],
                "sha256": entry["run_manifest_file_sha256"],
            },
            "checkpoint": copy.deepcopy(entry["final_step64"]),
            "training_data": copy.deepcopy(entry["data"]),
            "checkpoint_optimizer_step": 64,
            "model": native.MODEL_ID,
            "model_revision": native.REVISION,
        }
        generation._write_json(root / "admission.json", admission)
        result.update(
            admission=admission,
            admission_file_sha256=generation._file_hash(root / "admission.json"),
        )
        stage = "final_checkpoint_tensor_loading"
        tensors = preflight._tensors(
            data._relative(matrix_root, f"{entry['directory']}/arm"),
            entry["final_step64"],
            entry["final_step64"]["metadata"],
        )
        stage = "pinned_native_base_loading"
        model, tokenizer, metadata = native.load_model(
            cache_dir=cache_dir, device="cuda:0", seed=seed
        )
        _require(
            metadata["lora"] == entry["initializer"], "loaded_lora_initializer_settings_mismatch"
        )
        stage = "final_not_initial_adapter_restore"
        result["loaded_final_adapter_tensor_manifest"] = _restore_final(
            model,
            tensors,
            entry["final_step64"],
            seed,
        )
        tensors.clear()
        tensors = None
        native._parts(model)
        native._adapter_guard(model)
        result["native_model_admission"] = generation._native_guard(model, tokenizer)
        _require(model.config.use_cache is False, "native_model_config_cache_must_remain_false")
        _require(
            generation._precision_flags() == original_flags,
            "checkpoint_or_model_loading_changed_precision_permissions",
        )
        generation._write_json(root / "model.json", metadata)
        stage = "unchanged_native_run_generation"
        measured = generation.run_generation(
            model,
            tokenizer,
            records,
            output_dir=root / "generation",
            benchmark=benchmark,
            seed=seed,
            length=length,
            arm=arm,
            protocol_sha256=protocol_sha256,
            checkpoint_sha256=entry["final_step64"]["sha256"],
        )
        stage = "generation_completion_and_loaded_weight_revalidation"
        result["generation_status"] = measured["status"]
        _complete_generation(root / "generation", measured, admission)
        _require(
            _loaded_adapter_manifest(model) == entry["final_step64"]["tensor_manifest"],
            "inference_changed_loaded_final_adapter_tensors",
        )
        _require(
            generation._precision_flags() == original_flags,
            "generation_changed_precision_permissions",
        )
        generation._no_runtime_hooks(model)
        _require(
            generation._file_hash(root / "admission.json") == result["admission_file_sha256"]
            and _read_json(root / "admission.json") == admission,
            "immutable_admission_changed_during_execution",
        )
        result.update(
            status="COMPLETE",
            selected_shard_complete=True,
            loaded_final_adapter_hashes_verified_after_generation=True,
        )
    except BaseException as error:
        _failure(root, result, stage, error)
        if not isinstance(error, Exception):
            result["interrupted"] = True
            raise
    finally:
        try:
            result["whole_wrapper_measurement"] = _wrapper_measurement(started, rss_before)
        except BaseException as error:
            _failure(root, result, "whole_wrapper_measurement", error)
        try:
            if torch is not None and rng is not None:
                import numpy as np

                random.setstate(rng[0])
                np.random.set_state(rng[1])
                torch.set_rng_state(rng[2])
                torch.cuda.set_rng_state_all(rng[3])
                result["caller_rng_state_restored"] = True
            if original_flags is not None:
                changed = generation._precision_flags() != original_flags
                torch.set_float32_matmul_precision(original_flags["float32_matmul_precision"])
                torch.backends.cuda.matmul.allow_tf32 = original_flags["cuda_matmul_allow_tf32"]
                torch.backends.cudnn.allow_tf32 = original_flags["cudnn_allow_tf32"]
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = original_flags[
                    "allow_fp16_reduced_precision_reduction"
                ]
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = original_flags[
                    "allow_bf16_reduced_precision_reduction"
                ]
                _require(
                    generation._precision_flags() == original_flags,
                    "caller_precision_flags_restoration_failed",
                )
                result["caller_precision_flags_restored"] = True
                _require(not changed, "owned_execution_changed_precision_permissions")
        except BaseException as error:
            _failure(root, result, "owned_state_restoration", error)
        finally:
            if tensors is not None:
                tensors.clear()
            model = tokenizer = tensors = metadata = measured = rng = None
            gc.collect()
            try:
                if torch is not None and allocator_before is not None:
                    result["cleanup"] = _cleanup_cuda(torch, allocator_before)
                else:
                    result["cleanup"] = {"complete": True, "model_loading_not_started": True}
            except BaseException as error:
                result["cleanup"] = {"complete": False}
                _failure(root, result, "model_and_cuda_cleanup", error)
        if (root / "generation/manifest.json").is_file():
            result["generation_manifest_file"] = _descriptor(
                root / "generation/manifest.json",
                "generation/manifest.json",
            )
        result["wall_seconds_including_admission_loading_and_cleanup"] = (
            time.perf_counter() - started
        )
        result["files"] = {**generation._files(root)}
        generation._write_json(root / "run.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "matrix-root",
        "training-audit-path",
        "dataset-root",
        "protocol-path",
        "cache-dir",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--training-audit-sha256", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--arm", required=True, choices=("bf16_l2l", "oracle_block64_e4m3"))
    parser.add_argument("--seed", required=True, type=int, choices=(23, 41, 67, 89, 113))
    parser.add_argument("--length", required=True, type=int, choices=(1024, 4096))
    parser.add_argument("--benchmark", required=True, choices=_BENCHMARKS)
    parser.add_argument("--start-index", required=True, type=int)
    parser.add_argument("--stop-index", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        result = run_checkpoint_shard(**vars(args))
    except Exception as error:
        Console(stderr=True).print(
            {
                "status": "FAILED",
                "error_type": type(error).__name__,
                "output_dir": str(args.output_dir),
            }
        )
        return 1
    Console().print(
        {"status": result["status"], "output_dir": str(args.output_dir), "quality_admitted": False}
    )
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
