#!/usr/bin/env python3
"""Matched BF16/FP8 store accounting at the attached R1 dimensions, not model training."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rich.console import Console

try:
    from .p3_fp8_smoke import (
        KS,
        MemoryObservation,
        _codec_metrics,
        _error,
        _json_safe,
        store_evidence,
        thresholds,
    )
except ImportError:
    from p3_fp8_smoke import (
        KS,
        MemoryObservation,
        _codec_metrics,
        _error,
        _json_safe,
        store_evidence,
        thresholds,
    )


HOST_LIMIT = 5 * 1024**3
MINIMUM_HOST_SAVING = 0.25


def attached_config(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require the complete indexed Input, without loading model tensors."""
    root = Path(os.path.realpath(root))
    config_file = root / "config.json"
    index_file = root / "model.safetensors.index.json"
    config = json.loads(config_file.read_text())
    index = json.loads(index_file.read_text())
    expected = (61, 7168, 256, 8)
    observed = tuple(config[key] for key in (
        "num_hidden_layers", "hidden_size", "n_routed_experts", "num_experts_per_tok",
    ))
    if observed != expected or config["model_type"] != "deepseek_v3":
        raise ValueError(f"Input is not the expected DeepSeek-R1 671B configuration: {observed}")
    shards = sorted(set(index["weight_map"].values()))
    for name in shards:
        path = Path(os.path.realpath(root / name))
        if os.path.commonpath((str(root), str(path))) != str(root) or not path.is_file():
            raise ValueError(f"Missing or escaping Input shard: {name}")
    return config, {
        "root": str(root), "config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256(index_file.read_bytes()).hexdigest(),
        "referenced_shards_present": len(shards), "whole_model_allocated": False,
        "training_671b_steps": 0,
    }


def run_case(config: dict[str, Any], *, codec: str, k: int, device: str) -> dict[str, Any]:
    import torch

    from soup_cli.trainer.l2l_trainer import L2LState, _store_for
    from soup_cli.utils.host_memory_budget import HostMemoryBudget
    from soup_cli.utils.moe_l2l_budget import plan_moe_l2l

    layers, hidden = config["num_hidden_layers"], config["hidden_size"]
    shape = (1, 64, hidden)
    budget = HostMemoryBudget(HOST_LIMIT)
    row: dict[str, Any] = {"codec": codec, "k": k, "status": "FAIL"}
    state = observation = None
    try:
        plan = plan_moe_l2l(
            budget, n_layers=layers, n_moe_layers=layers - config["first_k_dense_replace"],
            max_shape=shape, experts_per_token=config["num_experts_per_tok"], requested_k=k,
            dtype="bfloat16", pin=device == "cuda", codec=codec, device_type=device,
        )
        row["resolved_plan"] = {**asdict(plan), "host_bytes": plan.host_bytes}
        if plan.effective_k != k:
            raise MemoryError("The fixed-k probe refuses reduced admission")
        state = L2LState(
            runtime=SimpleNamespace(tier="ram"), dtype=torch.bfloat16,
            activation_codec=codec, host_budget=budget, memory_plan=plan,
        )
        # This exercises the production store-selection/accounting path only. No fake
        # model forward or routing result is substituted for a training measurement.
        stepper = SimpleNamespace(
            causal=SimpleNamespace(config=SimpleNamespace(hidden_size=hidden)),
            layers=[None] * layers,
        )
        ids = [torch.zeros((1, 64), dtype=torch.long, device=device) for _ in range(k)]
        source = torch.randn(shape, generator=torch.Generator().manual_seed(123))
        source = source.to(dtype=torch.bfloat16, device=device)
        parts = []
        exact = True
        puts = gets = 0
        metrics, bounds = _codec_metrics(), thresholds()
        with MemoryObservation(device, scope=(
            "Storage-only allocation, put/get, G4 comparison and process baseline; no weights, "
            "optimizer or expert graphs. Routing bytes are leased, not filled route tensors."
        )) as observation:
            store = _store_for(state, stepper, ids, torch.device(device))
            reserved = budget.used_bytes
            row["reservation_matches_resolved_plan"] = reserved == plan.host_bytes
            row["store_before"] = store_evidence(store, budget)
            for layer in range(layers):
                for mb in range(k):
                    value = source * ((layer + 1) / layers) + (mb % 7) * 0.03125
                    store.put(layer, mb, value)
                    puts += 1
                    del value
            store.begin_backward()
            for layer in reversed(range(layers)):
                for mb in range(k):
                    restored = store.get(layer, mb)
                    value = source * ((layer + 1) / layers) + (mb % 7) * 0.03125
                    exact = exact and torch.equal(restored, value)
                    parts.append(metrics.restore_error(value, restored))
                    gets += 1
                    del restored, value
            store.end_step()
            row["store_after"] = store_evidence(store, budget)
            row["reservation_unchanged_through_roundtrip"] = budget.used_bytes == reserved
            state.close()
        error = metrics.merge_errors(parts)
        row.update({
            "puts": puts, "gets": gets, "restore_e1": error, "roundtrip_bitwise": exact,
            "route_reservation_bytes": plan.routing_record_bytes + plan.routing_comparison_bytes,
            "routing_storage_observed": False,
            "host_tensor_payload_formula_bytes": layers * k * 64 * hidden
            * (2 if codec == "none" else 1) + (0 if codec == "none" else layers * k * 64 * 4),
            "memory_observed": observation.result,
        })
        storage_bytes = row["store_after"]["storage_observed"]["owned_host_storage_bytes"]
        row["persistent_storage_matches_shape_formula"] = (
            storage_bytes == row["host_tensor_payload_formula_bytes"]
        )
        gate = exact if codec == "none" else (
            error["rms_row_rel"] <= bounds["restore_rms_row_rel_max"]
            and error["worst_row_rel"] <= bounds["restore_worst_row_rel_max"]
        )
        row["status"] = "PASS" if (
            gate and puts == gets == layers * k
            and row["reservation_matches_resolved_plan"]
            and row["reservation_unchanged_through_roundtrip"]
            and row["persistent_storage_matches_shape_formula"]
        ) else "FAIL"
        del ids, source, parts
    except Exception as error:
        row["error"] = _error(error)
    finally:
        if observation is not None:
            row["memory_observed"] = observation.result
        if state is not None:
            try:
                state.close()
            except Exception as error:
                row["close_error"] = _error(error)
                row["status"] = "FAIL"
        row["budget_used_after_close_bytes"] = budget.used_bytes
        if budget.used_bytes:
            row["status"] = "FAIL"
    return row


def screen_moe_retention(config: dict[str, Any], device: str) -> dict[str, Any]:
    """Reject an unhelpful policy using a generous reservation lower bound, not a codec."""
    from soup_cli.utils.host_memory_budget import HostMemoryBudget
    from soup_cli.utils.moe_l2l_budget import plan_moe_l2l

    layers = config["num_hidden_layers"]
    dense = config["first_k_dense_replace"]
    plan = plan_moe_l2l(
        HostMemoryBudget(HOST_LIMIT), n_layers=layers, n_moe_layers=layers - dense,
        max_shape=(1, 64, config["hidden_size"]),
        experts_per_token=config["num_experts_per_tok"], requested_k=64,
        dtype="bfloat16", pin=device == "cuda", codec="none", device_type=device,
    )
    if plan.effective_k != 64 or plan.activation_bytes % layers:
        raise ValueError("The declared fixed-shape policy bound is not applicable")
    # Even deleting every dense-layer activation for free cannot save enough. Real
    # FP8 payloads, scales, metadata and scratch can only increase this lower bound.
    impossible_free_dense_bytes = plan.activation_bytes // layers * dense
    lower_bound = plan.host_bytes - impossible_free_dense_bytes
    upper_saving = impossible_free_dense_bytes / plan.host_bytes
    return {
        "policy": "retain all MoE decoder boundaries in BF16; compress dense boundaries",
        "diagnosis": "native decoder routing changes in both MoE layers, not just embeddings",
        "status": ("REJECTED_BEFORE_IMPLEMENTATION" if upper_saving < MINIMUM_HOST_SAVING
                   else "INCONCLUSIVE"),
        "baseline_plan": {**asdict(plan), "host_bytes": plan.host_bytes},
        "retained_bf16_layers": layers - dense, "compressed_dense_layers": dense,
        "candidate_host_charge_lower_bound_bytes": lower_bound,
        "maximum_possible_saving_fraction": upper_saving,
        "minimum_useful_saving_fraction": MINIMUM_HOST_SAVING,
        "bound_assumption": "Dense boundaries cost zero; routes and retained slabs stay charged",
        "omitted_costs": "Candidate payload, scales, metadata, scratch, extra compute: all >= 0",
        "implemented": False, "numerically_validated": False,
        "whole_step_policy_overhead": None,
        "separate_input_policy_validation": "not applicable: memory screen rejects selection",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--model-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, choices=KS)
    parser.add_argument("--codec", choices=("none", "fp8_e4m3"))
    parser.add_argument("--screen-only", action="store_true")
    args = parser.parse_args()
    if (args.k is None) != (args.codec is None):
        parser.error("--k and --codec must be supplied together for an isolated child")
    result: dict[str, Any] = {
        "status": "FAIL", "cases": [], "minimum_useful_host_saving_fraction": MINIMUM_HOST_SAVING,
        "hard_host_reservation_limit_bytes": HOST_LIMIT,
        "minimum_useful_shape": {"layers": 61, "hidden": 7168, "tokens_per_mb": 64, "k": 64},
        "scope": "BF16/FP8 store comparison only, not training acceptance or 8 GB fit",
        "excluded_consumers": [
            "model and optimizer", "expert arena/graphs", "caller-retained hidden states/gradients",
            "allocator caches and Python metadata", "physical routing tensors",
        ],
    }
    console = Console()
    try:
        config, input_evidence = attached_config(args.model_input)
        result["attached_input"] = input_evidence
        if args.screen_only:
            result["policy_screen"] = screen_moe_retention(config, args.device)
            result["status"] = "SCREENED"
            console.print(result["policy_screen"])
            return 0
        if args.k is None:
            # Caching pinned allocations survive close(). Each comparison arm must
            # start in a fresh process before its RSS can be interpreted independently.
            args.output.parent.mkdir(parents=True, exist_ok=True)
            for k in KS:
                for codec in ("none", "fp8_e4m3"):
                    child_path = args.output.with_name(f"{args.output.stem}-k{k}-{codec}.json")
                    child = subprocess.run([
                        sys.executable, __file__, "--device", args.device,
                        "--model-input", str(args.model_input), "--output", str(child_path),
                        "--k", str(k), "--codec", codec,
                    ], check=False, timeout=300)
                    if child_path.is_file():
                        child_record = json.loads(child_path.read_text(encoding="utf-8"))
                        row = child_record["cases"][0]
                    else:
                        row = {"k": k, "codec": codec, "status": "FAIL",
                               "error": {"message": "Isolated child produced no record"}}
                    row["child_returncode"] = child.returncode
                    if child.returncode:
                        row["status"] = "FAIL"
                    result["cases"].append(row)
        else:
            import torch

            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            row = run_case(config, codec=args.codec, k=args.k, device=args.device)
            result["cases"].append(row)
            console.print(f"accounting k={args.k} codec={args.codec}: {row['status']}")
            gc.collect()
        result["process_isolation"] = "One fresh child per codec/k; no preceding store cache"
        final = {row["codec"]: row for row in result["cases"] if row["k"] == 64}
        if set(final) == {"none", "fp8_e4m3"} and all(
            "resolved_plan" in row for row in final.values()
        ):
            bf16 = final["none"]["resolved_plan"]["host_bytes"]
            fp8 = final["fp8_e4m3"]["resolved_plan"]["host_bytes"]
            saving = 1 - fp8 / bf16
            result["minimum_memory_gate"] = {
                "status": "PASS" if saving >= MINIMUM_HOST_SAVING else "FAIL",
                "bf16_host_bytes": bf16, "fp8_host_bytes": fp8, "saving_fraction": saving,
                "criterion": "charged activation plus routing reservation, not raw payload",
            }
        result["status"] = "PASS" if all(
            row["status"] == "PASS" for row in result["cases"]
        ) and (
            args.k is not None or result.get("minimum_memory_gate", {}).get("status") == "PASS"
        ) else "FAIL"
    except Exception as error:
        result["error"] = _error(error)
    finally:
        problems: list[str] = []
        safe = _json_safe(result, problems)
        safe["nonfinite_metric_paths"] = problems
        if problems:
            safe["status"] = result["status"] = "FAIL"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(safe, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    console.print(f"Storage accounting: {result['status']}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
