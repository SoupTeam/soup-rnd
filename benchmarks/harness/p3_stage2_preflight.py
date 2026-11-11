"""Paired Stage2 pretrained arms; each run requires a fresh Kaggle CUDA process."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

from rich.console import Console

from . import p3_realdata_baseline as baseline
from . import p3_realdata_model as native
from . import p3_stage2_dataset as data
from . import p3_stage2_generation as generation
from . import p3_stage2_training as training

ARMS = ("bf16_ga", "bf16_l2l", "bf16_oracle", "oracle_block64_e4m3")


def run_arm(*, pack_root: Path, pack_sha256: str, length: int, arm: str,
            cache_dir: Path, output_dir: Path, protocol_sha256: str,
            seed: int, preflight: bool) -> dict:
    """Run the two-call preflight or all 64 frozen windows in a fresh CUDA process."""
    import torch
    from safetensors.torch import save_file

    from soup_cli.utils.host_memory_budget import HostMemoryBudget

    output_dir.mkdir(parents=True, exist_ok=False)
    result = {"schema": "p3-stage2-pretrained-arm-v1", "status": "STARTED",
              "length": length, "arm": arm, "seed": seed, "preflight": preflight,
              "protocol_sha256": protocol_sha256, "final_task_ids_used": [],
              "quality_admitted": False, "production_admitted": False}
    started = time.perf_counter()
    snapshot_budget = HostMemoryBudget(256 * 1024**2)
    snapshot_lease = None
    initial = {}
    metadata = None
    try:
        if not Path("/kaggle/working").is_dir() or not torch.cuda.is_available():
            raise RuntimeError("Kaggle CUDA required; no host or CPU fallback")
        if (length not in (1024, 4096) or arm not in ARMS or type(preflight) is not bool
                or type(seed) is not int or seed not in training._SEEDS
                or preflight and seed != 23
                or not preflight and arm not in ("bf16_l2l", "oracle_block64_e4m3")):
            raise ValueError("Unapproved paired-study configuration")
        manifest_path = pack_root / "manifest.json"
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != pack_sha256:
            raise ValueError("Frozen pack file SHA-256 mismatch")
        manifest = data._read_manifest(pack_root)
        if manifest["status"] != "PAIRED_DATA_PACKED_RUNTIME_AND_PRECISION_PENDING":
            raise ValueError("Paired pack admission missing")
        descriptor = manifest["files"][f"windows_L{length}"]
        windows_path = data._artifact(pack_root, descriptor)
        windows = json.loads(windows_path.read_text(encoding="utf-8"))
        if len(windows) != 64 or descriptor["rows"] != 64:
            raise ValueError("Complete frozen 64-window pack required")
        result["pack_manifest_file_sha256"] = pack_sha256
        result["pack_manifest_payload_sha256"] = manifest["manifest_payload_sha256"]
        result["windows_file"] = descriptor
        result["environment"] = {
            "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "native_bf16_support": torch.cuda.is_bf16_supported(including_emulation=False),
            "emulated_bf16_tensor_support": torch.cuda.is_bf16_supported(),
            "weight_format": "dense BF16, not P3 NF4",
            "precision_flags": generation._precision_flags(),
        }
        if "T4" not in result["environment"]["gpu"]:
            raise RuntimeError("Requested T4 not allocated; no hardware substitution")
        flags = result["environment"]["precision_flags"]
        if (flags["cuda_matmul_allow_tf32"] or flags["autocast_enabled_cuda"]
                or flags["autocast_enabled_cpu"]):
            raise RuntimeError("Frozen no-TF32/no-autocast contract required; no silent override")
        model, _, metadata = native.load_model(cache_dir=cache_dir, device="cuda:0", seed=seed)
        baseline.write_json(output_dir / "model.json", metadata)
        elements = sum(param.numel() for param in baseline.trainables(model).values())
        snapshot_lease = snapshot_budget.reserve("paired-initial-snapshot", 16 * elements + 2**20)
        initial = baseline.snapshot(model)
        initial_path = output_dir / "initial.adapters.safetensors"
        save_file(initial, str(initial_path),
                  metadata={"kind": "paired-initial-adapters", "seed": str(seed),
                            "protocol_sha256": protocol_sha256})
        result["initial_snapshot"] = {
            "file": initial_path.name, "bytes": initial_path.stat().st_size,
            "sha256": hashlib.sha256(initial_path.read_bytes()).hexdigest(),
            "tensor_manifest": baseline.gradient_manifest(initial),
            "owned_host_tensor_bytes": 4 * elements,
            "host_reservation_bytes": snapshot_budget.used_bytes,
            "scope": "caller-owned snapshot and serialization, outside window policy ledger",
            "reservation_is_measured_peak": False,
        }
        execute = training.run_preflight_arm if preflight else training.train_arm
        measured = execute(
            model, windows, initial, seed=seed, arm=arm, output=output_dir / "arm",
            device="cuda:0", protocol_sha256=protocol_sha256,
        )
        result["precision_flags_after_arm"] = generation._precision_flags()
        if result["precision_flags_after_arm"] != flags:
            raise RuntimeError("Training changed the caller's compute-precision permissions")
        result["arm_status"] = measured["status"]
        result["status"] = measured["status"]
        result["optimizer_steps_completed"] = measured["optimizer_steps_completed"]
    except Exception as error:
        result.update(status="FAILED", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    finally:
        initial.clear()
        if snapshot_lease is not None:
            snapshot_lease.close()
        if metadata is not None:
            metadata["stage2_execution"] = {
                "preflight": preflight, "seed": seed, "status": result["status"],
                "optimizer_steps_completed": result.get("optimizer_steps_completed"),
                "evidence": "arm/arm.json",
                "loader_protocol_is_not_optimizer_evidence": True,
            }
            baseline.write_json(output_dir / "model.final.json", metadata)
        result["caller_host_reservation_after_cleanup"] = snapshot_budget.used_bytes
        result["wall_seconds_including_loading_serialization"] = time.perf_counter() - started
        baseline.write_json(output_dir / "run.json", result)
    return result


def _tensors(folder: Path, descriptor: dict, expected_metadata: dict) -> dict:
    from safetensors import safe_open

    path = folder / descriptor["file"]
    if path.name != descriptor["file"]:
        raise ValueError("Unsafe evidence filename")
    if (path.stat().st_size != descriptor["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != descriptor["sha256"]):
        raise ValueError("Evidence file bytes/SHA-256 mismatch")
    with safe_open(str(path), framework="pt", device="cpu") as stream:
        if stream.metadata() != expected_metadata:
            raise ValueError("Evidence header identity mismatch")
        tensors = {name: stream.get_tensor(name) for name in stream.keys()}
    if baseline.gradient_manifest(tensors) != descriptor["tensor_manifest"]:
        raise ValueError("Evidence tensor manifest mismatch")
    return tensors


def _step_artifacts(folder: Path, window: dict, initial_manifest: dict, *,
                    arm: str, step: int, protocol_sha256: str) -> dict:
    kinds = {
        "gradients": "raw-pre-clip-gradients",
        "adapters": "updated-adapter-evaluation-eligibility-in-arm-json",
        "optimizer": "named-AdamW-state",
    }
    values = {}
    for kind, label in kinds.items():
        values[kind] = _tensors(folder, window["evidence"][kind], {
            "arm": arm, "seed": "23", "optimizer_step": str(step),
            "protocol_sha256": protocol_sha256, "kind": label,
        })
    manifests = {kind: baseline.gradient_manifest(value) for kind, value in values.items()}
    if manifests["gradients"] != window["gradient_manifest"]:
        raise ValueError("Raw gradient artifact/window manifest mismatch")
    if any(not item["finite"] or item["nonzero"] == 0
           for item in manifests["gradients"].values()):
        raise ValueError("Nonfinite or vacuous raw gradient evidence")
    for kind in ("gradients", "adapters"):
        if manifests[kind].keys() != initial_manifest.keys() or any(
            item["shape"] != initial_manifest[name]["shape"] or item["dtype"] != "torch.float32"
            for name, item in manifests[kind].items()
        ):
            raise ValueError("Artifact differs from paired FP32 adapter set")
    if {name: item["sha256"] for name, item in manifests["adapters"].items()} != (
        window["adapter_hashes_after"]
    ):
        raise ValueError("Updated adapter artifact/window hash mismatch")
    expected_states = {f"{name}::{key}" for name in initial_manifest
                       for key in ("step", "exp_avg", "exp_avg_sq")}
    if values["optimizer"].keys() != expected_states:
        raise ValueError("Named AdamW state coverage mismatch")
    for name, tensor in values["optimizer"].items():
        parameter, kind = name.rsplit("::", 1)
        if kind == "step":
            if tensor.numel() != 1 or tensor.item() != step:
                raise ValueError("AdamW step count mismatch")
        elif list(tensor.shape) != initial_manifest[parameter]["shape"]:
            raise ValueError("AdamW state shape mismatch")
    return values


def _validate_arm(wrapper: dict, record: dict, *, length: int, arm: str,
                  manifest: dict, pack_sha256: str, protocol_sha256: str,
                  infos: list[dict], folder: Path) -> tuple[dict, list[dict]]:
    if (wrapper["arm"] != arm or record["arm"] != arm or wrapper["length"] != length
            or wrapper["seed"] != 23 or record["seed"] != 23
            or record["preflight"] is not True
            or wrapper["protocol_sha256"] != protocol_sha256
            or record["protocol_sha256"] != protocol_sha256
            or wrapper["pack_manifest_file_sha256"] != pack_sha256
            or wrapper["pack_manifest_payload_sha256"] != manifest["manifest_payload_sha256"]
            or wrapper["windows_file"] != manifest["files"][f"windows_L{length}"]
            or record["protocol"] != training._protocol(arm)):
        raise ValueError("Requested configuration/protocol/pack identity mismatch")
    if (wrapper["status"] != "PREFLIGHT_EVIDENCE_READY"
            or record["status"] != "PREFLIGHT_EVIDENCE_READY"
            or wrapper["optimizer_steps_completed"] != 2
            or record["optimizer_steps_completed"] != 2 or len(record["windows"]) != 2
            or not record["adapter_update_proved_after_warmup"]
            or not record["frozen_base_invariants_pass"] or not record["cleanup"]["complete"]
            or not record["entry_snapshot_restored"] or record["entry_rng_reset_to_seed"] != 23
            or not record["optimizer_state_initially_empty"]
            or record["final_checkpoint_evaluation_eligible"]
            or wrapper["caller_host_reservation_after_cleanup"] != 0):
        raise ValueError("Incomplete or failed preflight")
    initial = _tensors(folder, wrapper["initial_snapshot"], {
        "kind": "paired-initial-adapters", "seed": "23", "protocol_sha256": protocol_sha256,
    })
    initial_manifest = baseline.gradient_manifest(initial)
    if initial_manifest != record["initial_adapter_manifest"]:
        raise ValueError("Initial snapshot artifact/arm manifest mismatch")
    del initial
    if record["input_token_count_consumed"] != sum(info["input_token_count"] for info in infos):
        raise ValueError("Consumed input count mismatch")
    if record["valid_target_count_consumed"] != sum(info["valid_target_count"] for info in infos):
        raise ValueError("Consumed target count mismatch")
    ordered = hashlib.sha256(b"".join(bytes.fromhex(info["pairing_sha256"])
                                     for info in infos)).hexdigest()
    if record["ordered_window_pairing_sha256"] != ordered:
        raise ValueError("Ordered frozen-window pairing mismatch")
    method, oracle, policy = training._ARMS[arm]
    artifacts = []
    initial_hashes = {name: item["sha256"] for name, item in initial_manifest.items()}
    for step, (window, info) in enumerate(zip(record["windows"], infos, strict=True), 1):
        if any(window[key] != value for key, value in info.items()):
            raise ValueError("Consumed window/provenance differs from frozen input")
        if (window["status"] != "MEASURED" or not window["cleanup"]["complete"]
                or not window["optimizer_step"] or window["optimizer_step_index"] != step
                or window["method"] != method or window["oracle"] is not oracle
                or window["policy"] != vars(policy)):
            raise ValueError("Requested measured method/codec/update mismatch")
        route = window["routing"]
        if not all(route[key] for key in ("routing_pass", "complete", "admissible")):
            raise ValueError("Routing evidence incomplete or refused")
        if route["pending_oracle_replays"] != 0:
            raise ValueError("Oracle replay remained pending")
        if oracle:
            if (route["natural_rmr"] is not None or route["forced_rmr"] != 1.0
                    or route["oracle_vjp_calls"] != route["expected_pairs"]
                    or route["expected_pairs"] != 24 * (4096 // length)):
                raise ValueError("Oracle forced-routing/VJP coverage mismatch")
        elif route["natural_rmr"] != 1.0 or not route["natural_admissible"]:
            raise ValueError("Natural BF16 routing mismatch")
        if (window["learning_rates_before"] != [0.0 if step == 1 else 0.0001]
                or window["learning_rates_after"] != [0.0001 if step == 1 else 0.0002]
                or window["adapter_hashes_before"] != initial_hashes
                or (step == 1 and window["adapter_hashes_after"] != initial_hashes)
                or (step == 2 and window["adapter_hashes_after"] == initial_hashes)):
            raise ValueError("Warmup-aware real update evidence mismatch")
        values = _step_artifacts(folder / "arm", window, initial_manifest,
                                 arm=arm, step=step, protocol_sha256=protocol_sha256)
        scheduler = window["evidence"]["scheduler_state"]
        groups = window["evidence"]["optimizer"]["param_groups"]
        if (scheduler["last_epoch"] != step or scheduler["base_lrs"] != [0.0002]
                or scheduler["_last_lr"] != window["learning_rates_after"]
                or len(groups) != 1 or set(groups[0]["params"]) != set(initial_manifest)
                or len(groups[0]["params"]) != len(initial_manifest)):
            raise ValueError("Scheduler/named optimizer groups mismatch")
        expected = {"lr": window["learning_rates_after"][0], "initial_lr": 0.0002,
                    "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01,
                    "foreach": False, "fused": False, "maximize": False, "amsgrad": False}
        if any(groups[0].get(key) != value for key, value in expected.items()):
            raise ValueError("Declared AdamW settings mismatch")
        artifacts.append(values)
    return initial_manifest, artifacts


def summarize(*, root: Path, output: Path, pack_root: Path, pack_sha256: str,
              protocol_sha256: str) -> dict:
    """Compare saved measurements, never promote forced routes to natural RMR."""
    if output.exists():
        raise ValueError("Existing comparison evidence is immutable")
    result = {
        "schema": "p3-stage2-preflight-comparison-v1", "status": "STARTED", "lengths": {},
        "old_thresholds": baseline.thresholds(), "quality_admitted": False,
        "production_admitted": False, "natural_fp8_rmr_measured": False,
        "gradient_scope": "first two forwards share initial adapters; step one LR is zero",
        "optimizer_scope": "FP8 AdamW moments may differ after step one; no rollback",
        "memory_scope": "RAM and VRAM observations separate; reservation sums are not peaks",
    }
    try:
        if hashlib.sha256((pack_root / "manifest.json").read_bytes()).hexdigest() != pack_sha256:
            raise ValueError("Expected frozen pack SHA-256 mismatch")
        manifest = data._read_manifest(pack_root)
        paired_initial = paired_flat = None
        complete = True
        for length in (1024, 4096):
            records = {}
            artifacts = {}
            windows = json.loads(data._artifact(
                pack_root, manifest["files"][f"windows_L{length}"]
            ).read_text(encoding="utf-8"))
            infos = training._study_windows(windows, preflight=True)
            flat = [{key: info[key] for key in (
                "window_id", "input_sha256", "labels_sha256", "pairing_sha256",
                "input_token_count", "valid_target_count",
            )} for info in infos]
            if paired_flat is not None and flat != paired_flat:
                raise ValueError("Cross-length flat-window pairing mismatch")
            paired_flat = flat
            for arm in ARMS:
                folder = root / f"L{length}" / arm
                wrapper = json.loads((folder / "run.json").read_text())
                record = json.loads((folder / "arm/arm.json").read_text())
                initial_manifest, artifacts[arm] = _validate_arm(
                    wrapper, record, length=length, arm=arm, manifest=manifest,
                    pack_sha256=pack_sha256, protocol_sha256=protocol_sha256,
                    infos=infos, folder=folder,
                )
                if paired_initial is not None and initial_manifest != paired_initial:
                    raise ValueError("Cross-arm/cross-length initial adapter mismatch")
                paired_initial = initial_manifest
                records[arm] = record
            reference = records["bf16_l2l"]
            for record in records.values():
                for key in ("initial_adapter_manifest", "ordered_window_pairing_sha256",
                            "valid_target_count_consumed", "input_token_count_consumed"):
                    if record[key] != reference[key]:
                        raise ValueError("Cross-arm pairing mismatch: " + key)
            compared = {"controls": [], "candidate": []}
            for index in range(2):
                ref_window = reference["windows"][index]
                ref_grads = artifacts["bf16_l2l"][index]["gradients"]
                for arm in ("bf16_ga", "bf16_oracle", "oracle_block64_e4m3"):
                    window = records[arm]["windows"][index]
                    gradients = artifacts[arm][index]["gradients"]
                    comparison = baseline.compare_control(ref_window, window, ref_grads, gradients)
                    if (comparison["gradient_byte_parity"]
                            and not comparison["gradient"]["exact"]):
                        raise ValueError("Bitwise verdict contradicts loaded gradient tensors")
                    evidence_parity = {}
                    for kind in ("adapters", "optimizer"):
                        left = artifacts["bf16_l2l"][index][kind]
                        right = artifacts[arm][index][kind]
                        evidence_parity[kind] = baseline.gradient_manifest(left) == (
                            baseline.gradient_manifest(right))
                    scheduler_parity = ref_window["evidence"]["scheduler_state"] == (
                        window["evidence"]["scheduler_state"])
                    groups_parity = ref_window["evidence"]["optimizer"]["param_groups"] == (
                        window["evidence"]["optimizer"]["param_groups"])
                    item = {"arm": arm, "step": index + 1, "comparison": comparison,
                            "state_byte_parity": evidence_parity,
                            "scheduler_parity": scheduler_parity,
                            "optimizer_groups_parity": groups_parity,
                            "memory_accounting": window["memory_accounting"],
                            "bf16_l2l_memory_accounting": ref_window["memory_accounting"],
                            "evidence_serialization_seconds": window[
                                "evidence_serialization_wall_seconds"],
                            "bf16_l2l_evidence_serialization_seconds": ref_window[
                                "evidence_serialization_wall_seconds"],
                            "routing": window["routing"],
                            "whole_step_seconds": window["full_step_wall_seconds"],
                            "bf16_l2l_whole_step_seconds": ref_window["full_step_wall_seconds"],
                            "whole_step_time_ratio": window["full_step_wall_seconds"] / (
                                ref_window["full_step_wall_seconds"]),
                            "memory_observed": window["memory_observed"],
                            "bf16_l2l_memory_observed": ref_window["memory_observed"]}
                    if arm == "oracle_block64_e4m3":
                        item["old_bitwise_control_status_is_not_stage2_quality_verdict"] = True
                        item["natural_rmr"] = None
                        item["forced_rmr"] = window["routing"]["forced_rmr"]
                        compared["candidate"].append(item)
                    else:
                        passed = (comparison["status"] == "PASS" and all(evidence_parity.values())
                                  and scheduler_parity and groups_parity)
                        item["matched_bf16_control_pass"] = passed
                        complete = complete and passed
                        compared["controls"].append(item)
            result["lengths"][str(length)] = compared
        result["protocol_sha256"] = protocol_sha256
        result["pack_manifest_file_sha256"] = pack_sha256
        result["status"] = ("BF16_CONTROLS_VERIFIED_CANDIDATE_MEASURED_QUALITY_UNRESOLVED"
                            if complete else "BLOCKED_BF16_CONTROL_MISMATCH")
    except Exception as error:
        result.update(status="BLOCKED_INCOMPLETE_EVIDENCE", error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
    baseline.write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("arm", "train"):
        arm = commands.add_parser(command)
        arm.add_argument("--pack-root", required=True, type=Path)
        arm.add_argument("--pack-sha256", required=True)
        arm.add_argument("--length", required=True, type=int, choices=(1024, 4096))
        choices = ARMS if command == "arm" else ("bf16_l2l", "oracle_block64_e4m3")
        arm.add_argument("--arm", required=True, choices=choices)
        arm.add_argument("--cache-dir", required=True, type=Path)
        arm.add_argument("--output", required=True, type=Path)
        arm.add_argument("--protocol-sha256", required=True)
        if command == "train":
            arm.add_argument("--seed", required=True, type=int, choices=training._SEEDS)
    combined = commands.add_parser("summarize")
    combined.add_argument("--root", required=True, type=Path)
    combined.add_argument("--output", required=True, type=Path)
    combined.add_argument("--pack-root", required=True, type=Path)
    combined.add_argument("--pack-sha256", required=True)
    combined.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    if args.command in ("arm", "train"):
        preflight = args.command == "arm"
        result = run_arm(pack_root=args.pack_root, pack_sha256=args.pack_sha256,
                         length=args.length, arm=args.arm, cache_dir=args.cache_dir,
                         output_dir=args.output, protocol_sha256=args.protocol_sha256,
                         seed=23 if preflight else args.seed, preflight=preflight)
        success = result["status"] == (
            "PREFLIGHT_EVIDENCE_READY" if preflight else "TRAINED_CHECKPOINT_READY")
    else:
        result = summarize(root=args.root, output=args.output, pack_root=args.pack_root,
                           pack_sha256=args.pack_sha256, protocol_sha256=args.protocol_sha256)
        success = result["status"] == (
            "BF16_CONTROLS_VERIFIED_CANDIDATE_MEASURED_QUALITY_UNRESOLVED")
    Console().print({"status": result["status"], "output": str(args.output)})
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
