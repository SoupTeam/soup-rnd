"""Measured native-GA, resident-L2L and saved-Jacobian Oracle BF16 controls.

Run only through the Kaggle experiment runner. Every configuration is preserved,
including failures. This is a trained Granite proxy, not DeepSeek/P3 acceptance.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import time
import traceback
from pathlib import Path
from typing import Any

from rich.console import Console

try:
    from . import p3_realdata_model as native
    from .p3_fp8_smoke import (
        MemoryObservation,
        _json_safe,
        gradient_metrics,
        loss_metrics,
        thresholds,
    )
    from .p3_realdata_codec import CodecPolicy, make_store
    from .p3_realdata_corpus import build_corpus
    from .p3_realdata_routing import RouteProbe
except ImportError:
    import p3_realdata_model as native
    from p3_fp8_smoke import (
        MemoryObservation,
        _json_safe,
        gradient_metrics,
        loss_metrics,
        thresholds,
    )
    from p3_realdata_codec import CodecPolicy, make_store
    from p3_realdata_corpus import build_corpus
    from p3_realdata_routing import RouteProbe

HOST_LIMIT = 2 * 1024**3
DEVICE_SCRATCH_LIMIT = 128 * 1024**2


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    serializable = _json_safe(value, problems)
    if isinstance(serializable, dict):
        serializable["nonfinite_metric_paths"] = problems
    path.write_text(
        json.dumps(serializable, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )


def tensor_hash(value: Any) -> str:
    import torch

    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def trainables(model: Any) -> dict[str, Any]:
    return {name: param for name, param in model.named_parameters() if param.requires_grad}


def snapshot(model: Any) -> dict[str, Any]:
    return {name: param.detach().cpu().clone() for name, param in trainables(model).items()}


def restore(model: Any, values: dict[str, Any], seed: int) -> None:
    import numpy as np
    import torch

    params = trainables(model)
    if params.keys() != values.keys():
        raise ValueError("The paired adapter snapshot changed its parameter set")
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(values[name].to(param.device))
            param.grad = None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def case_tensors(case: dict[str, Any], device: str) -> tuple[list[Any], list[Any]]:
    import torch

    ids = torch.tensor([case["input_ids"]], dtype=torch.long)
    labels = torch.tensor([case["labels"]], dtype=torch.long)
    if tensor_hash(ids) != case["input_sha256"] or tensor_hash(labels) != case["labels_sha256"]:
        raise ValueError("Paired corpus tensor checksum mismatch")
    if ids.shape != (1, case["sequence_length"]):
        raise ValueError("Corpus length is not its declared exact real-token length")
    if int((labels[:, 1:] != -100).sum()) != case["valid_target_count"]:
        raise ValueError("Corpus valid-target denominator changed")
    return [ids.to(device)], [labels.to(device)]


def gradient_manifest(values: dict[str, Any]) -> dict[str, Any]:
    import torch

    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype),
               "sha256": tensor_hash(value), "finite": bool(torch.isfinite(value).all()),
               "l2": float(value.double().norm()), "nonzero": int(value.ne(0).sum())}
        for name, value in values.items()
    }


def run_arm(
    model: Any, case: dict[str, Any], initial: dict[str, Any], *,
    method: str, policy: CodecPolicy, oracle: bool, strict: bool,
    device: str, output: Path, name: str, channel_scales: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure a complete gradient window without optimizer updates or weight training."""
    import torch
    from safetensors.torch import save_file

    from soup_cli.utils.host_memory_budget import HostMemoryBudget

    if method not in ("ga", "l2l") or method == "ga" and oracle:
        raise ValueError("Oracle is a resident-L2L experiment, not a replacement native GA")
    output.mkdir(parents=True, exist_ok=True)
    restore(model, initial, case["paired_rng_seed"])
    batches, labels = case_tensors(case, device)
    shapes = [(*batch.shape, model.config.hidden_size) for batch in batches]
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    budget = HostMemoryBudget(HOST_LIMIT)
    probe = store = calls = None
    params = trainables(model)
    grads: dict[str, Any] = {}
    result: dict[str, Any] = {
        "name": name, "case_id": case["case_id"], "method": method,
        "policy": vars(policy), "oracle": oracle, "strict": strict,
        "diagnostic_only": not strict, "optimizer_step": False, "k": len(batches),
        "input_sha256": case["input_sha256"], "labels_sha256": case["labels_sha256"],
        "sequence_length": case["sequence_length"], "domain": case["domain"],
        "partition": case["partition"], "paired_rng_seed": case["paired_rng_seed"],
        "valid_target_count": case["valid_target_count"], "status": "STARTED",
        "host_limit_bytes": HOST_LIMIT, "device_scratch_limit_bytes": DEVICE_SCRATCH_LIMIT,
    }
    memory = MemoryObservation(device, scope=(
        "One resident trained Granite, common adapter snapshot and input tensors; includes "
        "native call capture, store/route allocation and full forward/backward. No optimizer "
        "update. Excludes model loading, adapter reset, corpus preparation, gradient-file "
        "copies and later E1/E2 diagnostics. Process RSS can include pinned allocator caches "
        "from earlier arms; the owned-host ledger, not RSS subtraction, defines "
        "activation-memory saving."
    ))
    error_text = None
    window_started = time.perf_counter()
    try:
        with memory:
            capture_started = time.perf_counter()
            calls = native.capture_calls(model, batches) if method == "l2l" else None
            if device == "cuda":
                torch.cuda.synchronize()
            result["native_call_capture_seconds"] = time.perf_counter() - capture_started
            probe = RouteProbe(model, shapes, budget, oracle=oracle, strict=strict)
            if method == "l2l":
                store = make_store(
                    policy, host_budget=budget, n_layers=model.config.num_hidden_layers,
                    mb_shapes=shapes, dtype=torch.bfloat16, device=device, pin=device == "cuda",
                    channel_scales=channel_scales,
                )
            route_allocation = probe.report()
            store_allocation = store.report() if store is not None else None
            scratch = route_allocation["device_scratch_bytes"] + (
                store_allocation["device_scratch_bytes"] if store_allocation else 0
            )
            if scratch > DEVICE_SCRATCH_LIMIT:
                raise MemoryError("Combined codec/router scratch exceeds the declared device bound")
            result["charged_host_bytes"] = budget.used_bytes
            result["combined_device_scratch_bound_bytes"] = scratch
            result["store"] = store_allocation
            if device == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                if method == "ga":
                    result["losses"] = native.ga_step(model, batches, labels, probe=probe)
                else:
                    result["losses"] = native.l2l_step(
                        model, batches, labels, calls, store, probe=probe,
                    )
                    result["store_after_replay"] = store.report()
                    store.end_step()
            finally:
                if device == "cuda":
                    torch.cuda.synchronize()
                result["forward_backward_window_seconds"] = time.perf_counter() - started
        result["status"] = "MEASURED"
    except Exception as error:
        error_text = traceback.format_exc()
        result["status"] = "ERROR"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        traceback.clear_frames(error.__traceback__)
    finally:
        result["memory_observed"] = memory.result
        cleanup_started = time.perf_counter()
        cleanup_errors = []
        if probe is not None:
            result["routing"] = probe.report()
            try:
                probe.close()
                result["routing_after_cleanup"] = probe.report()
            except Exception as error:
                cleanup_errors.append({"scope": "routing", "type": type(error).__name__,
                                       "message": str(error)})
        if store is not None:
            try:
                store.close()
                result["store_after_cleanup"] = store.report()
            except Exception as error:
                cleanup_errors.append({"scope": "store", "type": type(error).__name__,
                                       "message": str(error)})
        result["cleanup"] = {
            "host_used_bytes": budget.used_bytes, "errors": cleanup_errors,
            "seconds": time.perf_counter() - cleanup_started,
        }
        result["window_with_capture_admission_cleanup_seconds"] = (
            time.perf_counter() - window_started
        )
        if budget.used_bytes or cleanup_errors:
            result["status"] = "ERROR"
        result["adapter_parameter_bytes_unchanged"] = all(
            tensor_hash(param.detach().cpu()) == tensor_hash(initial[key])
            for key, param in params.items()
        )
        if not result["adapter_parameter_bytes_unchanged"]:
            result["status"] = "ERROR"
            result.setdefault("error", {
                "type": "UnexpectedParameterMutation",
                "message": "Numerical replay changed adapters",
            })
        missing = [key for key, param in params.items() if param.grad is None]
        result["missing_gradient_tensors"] = missing
        grads = {key: param.grad.detach().cpu().clone() for key, param in params.items()
                 if param.grad is not None}
        for param in params.values():
            param.grad = None
        calls = batches = labels = store = probe = None
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    result["gradient_manifest"] = gradient_manifest(grads)
    result["zero_gradient_tensors"] = [key for key, item in result["gradient_manifest"].items()
                                        if item["nonzero"] == 0]
    result["gradients_finite"] = bool(grads) and all(
        item["finite"] for item in result["gradient_manifest"].values()
    )
    if grads:
        gradient_path = output / f"{name}.gradients.safetensors"
        save_file(grads, str(gradient_path))
        result["gradient_file"] = gradient_path.name
        result["gradient_file_sha256"] = hashlib.sha256(gradient_path.read_bytes()).hexdigest()
    if error_text is not None:
        (output / f"{name}.error.log").write_text(error_text, encoding="utf-8")
    write_json(output / f"{name}.json", result)
    return result, grads


def compare_control(
    reference: dict[str, Any], candidate: dict[str, Any],
    ref_grads: dict[str, Any], candidate_grads: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "FAIL"}
    if reference["status"] != "MEASURED" or candidate["status"] != "MEASURED":
        result["reason"] = "missing complete measured control"
        return result
    result["gradient"] = gradient_metrics(ref_grads, candidate_grads)
    result["loss"] = loss_metrics(reference["losses"], candidate["losses"])
    gates = thresholds()
    gradient = result["gradient"]
    result["g4_gradient_pass"] = bool(
        gradient.get("finite")
        and gradient.get("global_rel_l2", float("inf")) <= gates["gradient_global_rel_l2_max"]
        and gradient.get("min_cosine", 0) >= gates["gradient_min_cosine_min"]
    )
    ref_manifest = reference.get("gradient_manifest", {})
    got_manifest = candidate.get("gradient_manifest", {})
    result["gradient_byte_parity"] = bool(ref_manifest) and (
        ref_manifest.keys() == got_manifest.keys() == ref_grads.keys() == candidate_grads.keys()
        and all(ref_manifest[name]["sha256"] == got_manifest[name]["sha256"]
                for name in ref_manifest)
    )
    result["finite_losses"] = result["loss"]["finite"]
    result["bitwise_parity"] = bool(
        result["gradient_byte_parity"] and result["finite_losses"] and result["loss"]["exact"]
    )
    result["nonvacuous"] = not any(
        arm["missing_gradient_tensors"] or arm["zero_gradient_tensors"]
        or not arm["gradients_finite"] for arm in (reference, candidate)
    )
    result["routing_pass"] = bool(candidate.get("routing", {}).get("routing_pass"))
    result["status"] = "PASS" if (
        result["bitwise_parity"] and result["nonvacuous"] and result["routing_pass"]
    ) else "FAIL"
    return result


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--lengths", default="256,1024,4096")
    parser.add_argument("--partitions", default="discovery")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    console = Console()
    lengths = {int(item) for item in args.lengths.split(",")}
    partitions = set(args.partitions.split(","))
    if not lengths <= {256, 1024, 4096} or not lengths:
        parser.error("Lengths must be a nonempty subset of 256,1024,4096")
    if not partitions <= {"calibration", "discovery", "holdout"} or not partitions:
        parser.error("Unknown corpus partition")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    summary: dict[str, Any] = {
        "status": "STARTED", "scope": "trained Granite BF16 controls; no P3 acceptance",
        "thresholds": thresholds(), "cases": [], "optimizer_step": False,
        "performance_scope": "raw complete forward/backward windows; not optimizer throughput",
        "independent_a3_floor": "not established by this matched-control benchmark",
    }
    write_json(output / "baseline.json", summary)
    model, tokenizer, metadata = native.load_model(cache_dir=args.cache, device=args.device)
    write_json(output / "model.json", metadata)
    corpus = build_corpus(
        tokenizer, tokenizer_spec=metadata["tokenizer_spec"], cache_dir=args.cache,
        output_dir=output / "corpus", blocks=1,
    )
    initial = snapshot(model)
    summary["initial_adapters"] = gradient_manifest(initial)
    selected = [case for case in corpus["cases"]
                if case["sequence_length"] in lengths and case["partition"] in partitions]
    for case in selected:
        case_output = output / case["partition"] / case["domain"] / f"L{case['sequence_length']}"
        row: dict[str, Any] = {"case_id": case["case_id"], "arms": {}, "controls": {}}
        ref = ref_grads = None
        for name, method, oracle in (
            ("ga1", "ga", False), ("ga2", "ga", False),
            ("l2l_none", "l2l", False), ("l2l_none_oracle", "l2l", True),
        ):
            console.print({"case": case["case_id"], "arm": name, "state": "STARTED"})
            result, grads = run_arm(
                model, case, initial, method=method, policy=CodecPolicy("none", "none"),
                oracle=oracle, strict=True, device=args.device, output=case_output, name=name,
            )
            row["arms"][name] = result
            if name == "ga1":
                ref, ref_grads = result, grads
            else:
                row["controls"][name] = compare_control(ref, result, ref_grads, grads)
                grads.clear()
            write_json(case_output / "controls.json", row)
            console.print({"case": case["case_id"], "arm": name, "status": result["status"],
                           "seconds": result.get("forward_backward_window_seconds")})
        row["status"] = "PASS" if all(
            item["status"] == "PASS" for item in row["controls"].values()
        ) else "FAIL"
        ref_grads.clear()
        summary["cases"].append(row)
        write_json(output / "baseline.json", summary)
    summary["status"] = "PASS" if summary["cases"] and all(
        item["status"] == "PASS" for item in summary["cases"]
    ) else "FAIL"
    write_json(output / "model.json", metadata)
    write_json(output / "baseline.json", summary)
    console.print({"status": summary["status"], "cases": len(summary["cases"])})


if __name__ == "__main__":
    main()
