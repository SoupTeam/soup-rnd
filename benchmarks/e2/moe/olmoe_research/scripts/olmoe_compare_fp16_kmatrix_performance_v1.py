#!/usr/bin/env python3
"""Compare FP16+GradScaler OLMoE K4/K8/K12 timing reports; fail closed on mismatch."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, choices=(4, 8, 12), required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--cached", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    a = json.loads(args.baseline.read_text())
    b = json.loads(args.cached.read_text())

    if a["mode"] != "baseline" or b["mode"] != "cached":
        raise RuntimeError("Expected baseline and cached reports in that order")
    fields = ("kind", "model_id", "revision", "precision", "lora", "optimizer",
              "k", "frozen_layers", "model_layers",
              "batch_size", "seq_len", "examples", "synthetic_data_sha256",
              "adapter_sha256_before", "precision_flags")
    for key in fields:
        if a[key] != b[key]:
            raise RuntimeError(f"Unmatched experimental condition: {key}")
    if a["k"] != args.k or b["k"] != args.k:
        raise RuntimeError("Requested K differs from report K")
    if a["frozen_layers"] != 16 - args.k or a["model_layers"] != 16:
        raise RuntimeError("Invalid frozen-prefix cutoff")
    if a["lora"]["upper_layers"] != args.k:
        raise RuntimeError("LoRA layers mismatch")
    for key in ("pilot_sha256", "verified_cache_sha256", "guard_cache_sha256"):
        if a["code"][key] != b["code"][key]:
            raise RuntimeError(f"Different source versions: {key}")
    for key in ("torch", "transformers", "peft", "gpu", "compute_capability"):
        if a["versions"][key] != b["versions"][key]:
            raise RuntimeError(f"Different environment: {key}")
    if len(a["epochs"]) != 2 or len(b["epochs"]) != 2:
        raise RuntimeError("Expected two epochs in each arm")
    steps_per_epoch = math.ceil(a["examples"] / a["batch_size"])
    if any(e.get("optimizer_steps") != steps_per_epoch for report in (a,b) for e in report["epochs"]):
        raise RuntimeError("Incomplete optimizer-step totals per epoch")
    if len(a["steps"]) != 2 * steps_per_epoch or len(b["steps"]) != 2 * steps_per_epoch:
        raise RuntimeError("Unexpected training step count")
    if b["cache"]["hits"] != steps_per_epoch or b["cache"]["misses"] != steps_per_epoch:
        raise RuntimeError("Unexpected cache hit/miss counts")
    if b["cache"]["skipped_frozen_layer_calls"] != steps_per_epoch * a["frozen_layers"]:
        raise RuntimeError("Frozen layers were not skipped on all hits")
    if b["cache"]["enabled"] is not True:
        raise RuntimeError("Cached arm must enable cache")
    if any(a["cache"].values()):
        raise RuntimeError("Baseline unexpectedly used a cache")

    for label, arm in (("baseline", a), ("cached", b)):
        flags = arm.get("precision_flags")
        scaler = arm.get("grad_scaler", {})
        if flags != {"fp16": True, "bf16": False} or not scaler.get("enabled"):
            raise RuntimeError(f"{label}: actual FP16/scaler provenance missing")
        if scaler.get("initial_scale") != 1024.0 or scaler.get("skipped_optimizer_steps") != 0:
            raise RuntimeError(f"{label}: unexpected GradScaler behavior")
        if scaler.get("completed_optimizer_steps") != 2 * steps_per_epoch:
            raise RuntimeError(f"{label}: optimizer steps not complete")
        if len(arm["steps"]) != 2 * steps_per_epoch:
            raise RuntimeError(f"{label}: missing per-step scaler records")
        if any(
            not x.get("optimizer_step_verified")
            or not math.isfinite(x.get("scaler_before", float("nan")))
            or not math.isfinite(x.get("scaler_after", float("nan")))
            or x["scaler_after"] < x["scaler_before"]
            for x in arm["steps"]
        ):
            raise RuntimeError(f"{label}: invalid/possibly skipped FP16 training step")
        if any(
            abs(float(x["scaler_before"]) - float(y["scaler_before"])) > 0
            or abs(float(x["scaler_after"]) - float(y["scaler_after"])) > 0
            for x, y in zip(a["steps"], b["steps"])
        ):
            raise RuntimeError("Baseline/cached GradScaler evolution diverged")
    loss_abs_differences = []
    for x, y in zip(a["steps"], b["steps"]):
        if (x["epoch"], x["batch"], x["batch_size"]) != (y["epoch"], y["batch"], y["batch_size"]):
            raise RuntimeError("Training batch order mismatch")
        for value in ("loss", "aux_loss"):
            loss_abs_differences.append(abs(x[value] - y[value]))
    max_loss_difference = max(loss_abs_differences, default=float("inf"))
    adapter_match = a["adapter_sha256_after"] == b["adapter_sha256_after"]
    if not adapter_match or max_loss_difference > 2e-3:
        raise RuntimeError(
            "Training diverged between baseline and cached processes; "
            f"final adapter SHA match={adapter_match}, max loss delta={max_loss_difference}. "
            "Do not interpret a timing speedup before resolving this discrepancy."
        )

    def div(x, y):
        return x / y if y > 0 else float("nan")

    if b.get("guard_profile", {}).get("full_sha256_recomputes") != 1:
        raise RuntimeError("Cached run must hash the frozen prefix exactly once")
    if b.get("guard_profile", {}).get("metadata_calls") != 2 * steps_per_epoch:
        raise RuntimeError("Cached run did not perform the expected guarded lookups")

    for label, report in (("baseline", a), ("cached", b)):
        if report["training_wall_seconds"] <= 0 or not math.isfinite(report["training_wall_seconds"]):
            raise RuntimeError(f"{label}: invalid training wall time")
        for e in report["epochs"]:
            if not math.isfinite(e["step_sum_seconds"]) or e["step_sum_seconds"] <= 0:
                raise RuntimeError(f"{label}: invalid epoch timing")

    summary = {
        "kind": "olmoe_kmatrix_pretrained_fp16_gradscaler_research_performance_comparison_v1",
        "k": args.k,
        "frozen_layers": a["frozen_layers"],
        "model_layers": a["model_layers"],
        "precision": a["precision"],
        "hardware": a["versions"]["gpu"],
        "quality_evaluation": False,
        "not_abba": True,
        "research_acceptance": "NO VERDICT",
        "steps_per_epoch": steps_per_epoch,
        "max_loss_or_aux_delta": max_loss_difference,
        "final_adapters_exact_sha_match": adapter_match,
        "fp16_gradscaler_steps_verified": True,
        "cold_epoch_step_sum_seconds": {
            "baseline": a["epochs"][0]["step_sum_seconds"],
            "cached": b["epochs"][0]["step_sum_seconds"],
        },
        "warm_epoch_step_sum_seconds": {
            "baseline": a["epochs"][1]["step_sum_seconds"],
            "cached": b["epochs"][1]["step_sum_seconds"],
        },
        "training_wall_seconds_including_cache_build": {
            "baseline": a["training_wall_seconds"],
            "cached": b["training_wall_seconds"],
        },
        "warm_speed_ratio_baseline_over_cache": div(
            a["epochs"][1]["step_sum_seconds"], b["epochs"][1]["step_sum_seconds"]
        ),
        "two_epoch_training_wall_ratio_baseline_over_cache": div(
            a["training_wall_seconds"], b["training_wall_seconds"]
        ),
        "cache": b["cache"],
        "guard_profile": b["guard_profile"],
        "code": b["code"],
        "caveat": (
            "Preliminary two-process Turing FP16+GradScaler synthetic-data timing only. "
            "Not repeated ABBA, no continuous GPU-peer monitor, not real-dataset quality, not performance acceptance. "
            "Version-based frozen-state guard assumes no unannounced .data/raw-storage mutations."
        ),
    }
    print(json.dumps(summary, indent=2))
    if args.output is not None:
        if args.output.exists():
            raise RuntimeError("Comparison output exists; refusing overwrite")
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        print("COMPARISON REPORT:", args.output)
    print(f"OLMOE K{args.k} FP16 EXPLORATORY PERFORMANCE COMPARISON COMPLETE")


if __name__ == "__main__":
    main()
