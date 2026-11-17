#!/usr/bin/env python3
"""Bounded BF16 GA/L2L baseline and separate FP8/L2L storage investigation.

Run on the submitted Kaggle checkout, with src on PYTHONPATH. Every window starts all
arms at the same adapters; only GA1 advances the next window's common starting point.
Raw tiny gradients/updates are disk-backed, never retained in the measured arm scope.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from rich.console import Console

try:
    from . import l2l_mistral_stand as stand
    from .b13_moe_fixture import ALL_TARGETS, TARGETS, moe_batch, snapshot, trainables
    from .p3_bf16_fixture import build_bf16_fixture
    from .p3_fp8_smoke import (
        MemoryObservation,
        _json_safe,
        gradient_metrics,
        loss_metrics,
        source_identity,
        store_evidence,
        thresholds,
    )
except ImportError:
    import l2l_mistral_stand as stand
    from b13_moe_fixture import ALL_TARGETS, TARGETS, moe_batch, snapshot, trainables
    from p3_bf16_fixture import build_bf16_fixture
    from p3_fp8_smoke import (
        MemoryObservation,
        _json_safe,
        gradient_metrics,
        loss_metrics,
        source_identity,
        store_evidence,
        thresholds,
    )

KS = (1, 4, 16, 64)
HOST_LIMIT = 256 * 1024**2
DEVICE_SCRATCH_LIMIT = 64 * 1024**2
ARMS = ("ga1", "ga2", "ga_floor", "l2l_none", "l2l_fp8")


def _error(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _gpu_evidence(device: str) -> dict[str, Any]:
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        return {"device": device, "cuda_available": torch.cuda.is_available()}
    native = None
    native_error = None
    try:
        native = bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError as error:
        native_error = f"{type(error).__name__}: {error}"
    return {
        "name": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "bf16_supported_including_emulation": bool(torch.cuda.is_bf16_supported()),
        "bf16_hardware_supported_excluding_emulation": native,
        "native_support_api_error": native_error,
        "throughput_claim": "No native-BF16 hardware throughput claim; inspect observed dtypes",
    }


def _digest(tensor: Any) -> str:
    import torch

    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _tensor_manifest(values: dict[str, Any]) -> dict[str, Any]:
    return {
        name: None if value is None else {
            "shape": list(value.shape), "dtype": str(value.dtype), "sha256": _digest(value),
            "l2": float(value.double().norm()), "nonzero": int(value.ne(0).sum()),
        }
        for name, value in values.items()
    }


def make_windows(
    config: Any, k: int, *, device: str, input_set: str,
) -> list[tuple[list[Any], list[Any]]]:
    """Two full windows and a short final window, never pad a variable shape."""
    offset = {"discovery": 0, "holdout": 10000}[input_set]
    windows = []
    for window, count in enumerate((k, k, max(1, k // 2))):
        ids, labels = [], []
        for microbatch in range(count):
            data_index = offset + window * k + microbatch
            batch = moe_batch(
                config, seed=data_index, rows=1 + microbatch % 2,
                seq=(7, 4, 3)[window] + microbatch % 3,
            )
            batch["labels"][0, :1 + microbatch % 2] = -100
            ids.append(batch["input_ids"].to(device))
            labels.append(batch["labels"].to(device))
        windows.append((ids, labels))
    return windows


class AttentionCalls:
    """Count executed SDPA operations; a requested config is not a backend floor."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __enter__(self) -> AttentionCalls:
        import torch.nn.functional as functional

        self.original = functional.scaled_dot_product_attention

        def observed(query: Any, key: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
            self.calls.append({
                "query_shape": list(query.shape), "dtype": str(query.dtype),
                "device": str(query.device),
            })
            return self.original(query, key, value, *args, **kwargs)

        functional.scaled_dot_product_attention = observed
        return self

    def __exit__(self, *_error: Any) -> None:
        import torch.nn.functional as functional

        functional.scaled_dot_product_attention = self.original


class Routes:
    """Record native router outputs and boundary dtypes without changing a decision."""

    def __init__(self, model: Any, *, l2l: bool) -> None:
        from soup_cli.utils.l2l_step import model_parts

        self.l2l = l2l
        self.events: dict[str, dict[str, list[dict[str, Any]]]] = {}
        self.boundary_dtypes: dict[str, set[str]] = {}
        self.router_dtypes: dict[str, set[str]] = {}
        self.handles = []
        for index, layer in enumerate(model_parts(model)[1].layers):
            key = str(index)
            self.boundary_dtypes[key] = set()

            def boundary(
                _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], *, name: str = key,
            ) -> None:
                value = args[0] if args else kwargs["hidden_states"]
                self.boundary_dtypes[name].add(str(value.dtype))

            self.handles.append(layer.register_forward_pre_hook(boundary, with_kwargs=True))
            native = getattr(layer, "inner", layer)
            router = getattr(getattr(native, "mlp", None), "gate", None)
            if router is None:
                continue
            self.events[key] = {"forward": [], "recompute": []}
            self.router_dtypes[key] = set()

            def route(
                _module: Any, _args: Any, output: Any, *, name: str = key,
            ) -> None:
                import torch

                selected = output[2]
                phase = "recompute" if self.l2l and torch.is_grad_enabled() else "forward"
                self.events[name][phase].append({
                    "shape": list(selected.shape), "sha256": _digest(selected),
                })
                self.router_dtypes[name].update(str(value.dtype) for value in output
                                               if isinstance(value, torch.Tensor))

            self.handles.append(router.register_forward_hook(route))
        if not self.events:
            raise RuntimeError("no native DeepSeek routers found for strict routing evidence")

    def evidence(self, k: int, expected: dict[str, Any] | None) -> dict[str, Any]:
        forward = {name: events["forward"] for name, events in self.events.items()}
        complete = all(len(events) == k for events in forward.values())
        recompute_equal = not self.l2l or all(
            events["forward"] == events["recompute"] for events in self.events.values()
        )
        matched = expected is None or forward == expected
        return {
            "forward": forward,
            "recompute": {name: events["recompute"] for name, events in self.events.items()},
            "forward_complete": complete, "forward_reference_equal": matched,
            "forward_recompute_equal": recompute_equal,
            "strict_ok": complete and recompute_equal and matched,
            "boundary_dtypes": {name: sorted(values)
                                for name, values in self.boundary_dtypes.items()},
            "router_output_dtypes": {name: sorted(values)
                                     for name, values in self.router_dtypes.items()},
            "product_guard": "unchanged L2LStep RoutingGuard" if self.l2l else None,
            "forced_routes": False,
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _plans(k: int, device: str) -> dict[str, Any]:
    """Resolve both fixed-k budgets and scratch bound before any fixture/input tensors."""
    from soup_cli.utils.fp8_activations import fp8_activation_allocation
    from soup_cli.utils.host_memory_budget import HostMemoryBudget
    from soup_cli.utils.moe_l2l_budget import plan_moe_l2l

    plans = {}
    for codec in ("none", "fp8_e4m3"):
        plan = plan_moe_l2l(
            HostMemoryBudget(HOST_LIMIT), n_layers=3, n_moe_layers=2,
            max_shape=(2, 9, 32), experts_per_token=2, requested_k=k,
            dtype="bfloat16", pin=device == "cuda", codec=codec, device_type=device,
        )
        if plan.effective_k != k:
            raise MemoryError("fixed matrix refuses planner-reduced k")
        plans[codec] = plan
    scratch = fp8_activation_allocation(
        n_layers=3, mb_shapes=[(2, 9, 32)] * k, pin=device == "cuda",
        compute_dtype="bfloat16", device_type=device,
    ).device_transient_bytes
    if scratch > DEVICE_SCRATCH_LIMIT:
        raise MemoryError("FP8 device scratch exceeds predeclared tiny-fixture bound")
    return {"plans": plans, "fp8_device_scratch_bytes": scratch}


def _restore(model: Any, path: Path) -> None:
    import torch

    values = torch.load(path, map_location="cpu", weights_only=True)
    parameters = trainables(model)
    if values.keys() != parameters.keys():
        raise RuntimeError("canonical adapter names differ from arm")
    with torch.no_grad():
        for name, parameter in parameters.items():
            if parameter.dtype != values[name].dtype:
                raise RuntimeError(f"adapter dtype differs from canonical: {name}")
            parameter.copy_(values[name])
            if not torch.equal(parameter.detach().cpu(), values[name]):
                raise RuntimeError(f"adapter restore differs from canonical: {name}")
    del values


def _diagnose(model: Any, store: Any, ids: list[Any], device: str, input_set: str) -> Any:
    try:
        from .p3_bf16_diagnostics import diagnose_decoder
    except ImportError:
        from p3_bf16_diagnostics import diagnose_decoder

    return diagnose_decoder(model, store, ids, device=device, input_set=input_set)


def _run_arm(
    fixture: Any, *, name: str, ids: list[Any], labels: list[Any], device: str,
    canonical: Path, artifact: Path, plan: Any, expected_routes: dict[str, Any] | None,
    input_set: str, diagnostics: bool,
) -> dict[str, Any]:
    import torch

    from soup_cli.trainer.l2l_trainer import L2LState, _store_for, run_window
    from soup_cli.utils.host_memory_budget import HostMemoryBudget
    from soup_cli.utils.l2l_step import model_parts

    l2l = name.startswith("l2l_")
    codec = "fp8_e4m3" if name == "l2l_fp8" else "none"
    backend = "EFFICIENT_ATTENTION" if name == "ga_floor" else "MATH"
    model = fixture.streamed if l2l else fixture.resident
    row: dict[str, Any] = {
        "status": "FAIL", "name": name, "backend_requested": backend,
        "codec": codec if l2l else None, "optimizer_applied": False,
        "optimizer": {"class": "SGD", "lr": 0.05, "momentum": 0.0},
        "loss_normalization": "sum causal unmasked token losses / window label token count",
        "optimizer_policy": (
            "Provisional diagnostic SGD after strict route/finite checks; numerical acceptance "
            "is judged separately. Every next arm restores the canonical adapters."
        ),
    }
    state = routes = memory = attention = None
    budget = HostMemoryBudget(HOST_LIMIT)
    causal = model_parts(model)[0]
    original_loss = causal.loss_function
    losses: list[float] = []
    loss_dtypes: set[str] = set()

    def observed_loss(*args: Any, **kwargs: Any) -> Any:
        value = original_loss(*args, **kwargs)
        losses.append(float(value.detach().cpu()))
        loss_dtypes.add(str(value.dtype))
        return value

    try:
        _restore(model, canonical)
        for twin in (fixture.streamed, fixture.resident):
            twin.zero_grad(set_to_none=True)
        optimizer = torch.optim.SGD(trainables(model).values(), lr=0.05, momentum=0.0)
        if l2l:
            state = L2LState(
                runtime=fixture.runtime, dtype=torch.bfloat16, activation_codec=codec,
                host_budget=budget, has_dropout=False, memory_plan=plan,
            )
        causal.loss_function = observed_loss
        routes = Routes(model, l2l=l2l)
        total = sum(int((label[:, 1:] != -100).sum()) for label in labels)
        gc.collect()
        with MemoryObservation(device, scope=(
            "Whole forward/backward plus SGD and strict-route/finite checks, synchronized. "
            "Includes live fixture twins and input tensors, route digest instrumentation, "
            "trainer allocations and store leases. Reference gradient/update/adapter clones "
            "are on disk, absent from this interval. Diagnostics and artifact copies excluded."
        )) as memory, AttentionCalls() as attention, stand.pinned_sdpa(backend):
            try:
                if l2l:
                    returned, counters = run_window(state, model, ids, labels, total, ids[0].device)
                    row["load_counters"] = asdict(counters)
                    del returned
                else:
                    for batch, label in zip(ids, labels):
                        loss = model(
                            input_ids=batch, labels=label, num_items_in_batch=total,
                        ).loss
                        loss.backward()
                    del loss
                routing = routes.evidence(len(ids), expected_routes)
                if not routing["strict_ok"]:
                    raise RuntimeError(
                        "strict routing observation changed or is incomplete; no SGD"
                    )
                if len(losses) != len(ids) or any(not math.isfinite(value) for value in losses):
                    raise RuntimeError("missing/nonfinite per-microbatch loss; no SGD")
                parameters = trainables(model)
                if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
                       for parameter in parameters.values()):
                    raise RuntimeError("missing/nonfinite adapter gradient; no SGD")
                optimizer.step()
                row["optimizer_applied"] = True
                row["status"] = "OBSERVED"
            except Exception as error:
                row["error"] = _error(error)
                row["status"] = "REFUSED" if "rout" in str(error).lower() else "FAIL"
                model.zero_grad(set_to_none=True)
        row["memory_and_timing"] = memory.result
        row["memory_and_timing"]["timing_kind"] = (
            "completed forward/backward+SGD; numerical acceptance is separate"
            if row["optimizer_applied"]
            else "partial failed/refused step; not comparable successful-step latency"
        )
        row["routing"] = routes.evidence(len(ids), expected_routes)
        row["attention"] = {
            "config_resolved": getattr(model.config, "_attn_implementation", None),
            "sdpa_call_count": len(attention.calls),
            "unique_calls": [json.loads(value) for value in sorted({
                json.dumps(call, sort_keys=True) for call in attention.calls
            })],
            "observed_path": "sdpa" if attention.calls else "no executed SDPA observed",
            "valid_floor": bool(name == "ga_floor" and attention.calls
                                and row["optimizer_applied"]),
        }
        if name == "ga_floor":
            row["floor_status"] = "observed_sdpa_floor" if (
                row["attention"]["valid_floor"]
            ) else "no_floor"
            row["floor_unavailable_reason"] = None if row["attention"]["valid_floor"] else (
                row.get("error", {}).get("message") or "No executed SDPA operation observed"
            )
        row["per_microbatch_losses"] = losses
        row["loss_dtypes"] = sorted(loss_dtypes)
        if row["optimizer_applied"]:
            gradients = snapshot(model, gradients=True)
            after = snapshot(model)
            before = torch.load(canonical, map_location="cpu", weights_only=True)
            delta = {key: after[key] - before[key] for key in before}
            torch.save({"gradients": gradients, "updates": delta, "parameters": after}, artifact)
            row["raw_tensor_artifact"] = artifact.name
            row["gradient_manifest"] = _tensor_manifest(gradients)
            row["update_manifest"] = _tensor_manifest(delta)
            del gradients, after, before, delta
        else:
            row["partial_gradients_discarded"] = True
        if state is not None and state.store is not None:
            row["store"] = store_evidence(state.store, budget)
            row["store_configured_compute_dtype"] = str(state.store.dtype)
            if diagnostics:
                try:
                    _restore(fixture.resident, canonical)
                    state.close_store()
                    diagnostic_store = _store_for(state, state.stepper, ids, ids[0].device)
                    with stand.pinned_sdpa("MATH"):
                        diagnostic = _diagnose(
                            fixture.resident, diagnostic_store, ids, device, input_set,
                        )
                    diagnostic_path = artifact.with_suffix(".diagnostics.json")
                    nonfinite: list[str] = []
                    safe_diagnostic = _json_safe(diagnostic, nonfinite)
                    safe_diagnostic["nonfinite_metric_paths"] = nonfinite
                    diagnostic_path.write_text(
                        json.dumps(safe_diagnostic, indent=2), encoding="utf-8",
                    )
                    row["decoder_diagnostics"] = {
                        "status": diagnostic["status"], "complete": diagnostic["complete"],
                        "artifact": diagnostic_path.name,
                        "e1": diagnostic.get("e1"), "e2": diagnostic.get("e2"),
                        "decoded_dtypes_observed": sorted({
                            chunk["boundary_restored"]["dtype"]
                            for chunk in diagnostic.get("chunks", [])
                        }),
                    }
                    del diagnostic, safe_diagnostic, diagnostic_store
                except Exception as error:
                    row["decoder_diagnostics"] = {"status": "FAIL", "error": _error(error)}
    except Exception as error:
        row["status"] = "FAIL"
        row["error"] = _error(error)
        row["traceback"] = traceback.format_exc()
    finally:
        causal.loss_function = original_loss
        if memory is not None:
            row.setdefault("memory_and_timing", memory.result)
            row["memory_and_timing"].setdefault(
                "timing_kind", "partial failed step; not comparable successful-step latency",
            )
        if routes is not None:
            routes.close()
        if state is not None:
            try:
                state.close()
            except Exception as error:
                row["close_error"] = _error(error)
                row["status"] = "FAIL"
        row["budget_used_after_close_bytes"] = budget.used_bytes
        if budget.used_bytes:
            row["status"] = "FAIL"
        if name == "ga_floor":
            row.setdefault("floor_status", "no_floor")
            row.setdefault(
                "floor_unavailable_reason",
                row.get("error", {}).get("message") or "No successful SDPA floor execution",
            )
        for twin in (fixture.streamed, fixture.resident):
            twin.zero_grad(set_to_none=True)
    return row


def _record(row: dict[str, Any], payload: dict[str, Any], *, l2l: bool) -> Any:
    return stand.RunRecord(
        logs=[{"loss": sum(row["per_microbatch_losses"]), "learning_rate": 0.05}],
        rates=[0.05], grads=[payload["gradients"]], steps=1, l2l=l2l,
    )


def _compare(window: dict[str, Any], raw: Path) -> None:
    import torch

    arms = window["arms"]
    payloads = {
        name: torch.load(raw / row["raw_tensor_artifact"], map_location="cpu", weights_only=True)
        for name, row in arms.items() if row.get("optimizer_applied")
        and "raw_tensor_artifact" in row
    }
    needed = {"ga1", "ga2", "l2l_none"}
    baseline: dict[str, Any] = {"status": "BLOCKED", "pass": False}
    if needed <= payloads.keys():
        floor = None
        floor_reason = "EFFICIENT_ATTENTION arm failed or refused"
        main_sdpa = arms["ga1"]["attention"]["sdpa_call_count"] > 0
        floor_sdpa = arms["ga_floor"].get("attention", {}).get("valid_floor", False)
        if "ga_floor" in payloads and main_sdpa and floor_sdpa:
            floor = _record(arms["ga_floor"], payloads["ga_floor"], l2l=False)
            floor_reason = None
        elif not main_sdpa:
            floor_reason = "native GA did not execute SDPA; requested backend is not a floor"
        baseline = stand.decide(
            _record(arms["ga1"], payloads["ga1"], l2l=False),
            _record(arms["ga2"], payloads["ga2"], l2l=False),
            _record(arms["l2l_none"], payloads["l2l_none"], l2l=True),
            mode="bf16", floor=floor,
        )
        baseline["floor_unavailable_reason"] = floor_reason
        baseline["ga1_ga2_updates"] = gradient_metrics(
            payloads["ga1"]["updates"], payloads["ga2"]["updates"],
        )
        baseline["gradients"] = gradient_metrics(
            payloads["ga1"]["gradients"], payloads["l2l_none"]["gradients"],
        )
        baseline["updates"] = gradient_metrics(
            payloads["ga1"]["updates"], payloads["l2l_none"]["updates"],
        )
        baseline["losses"] = loss_metrics(
            arms["ga1"]["per_microbatch_losses"], arms["l2l_none"]["per_microbatch_losses"],
        )
        baseline["update_tolerance_note"] = (
            "A3 is unchanged and judges gradients/losses, update errors are separate observations"
        )
        if floor is not None:
            update_floor = stand.RunRecord(
                logs=floor.logs, rates=floor.rates,
                grads=[payloads["ga_floor"]["updates"]], steps=1,
            )
            update_ref = stand.RunRecord(
                logs=floor.logs, rates=floor.rates, grads=[payloads["ga1"]["updates"]], steps=1,
            )
            update_cand = stand.RunRecord(
                logs=floor.logs, rates=floor.rates,
                grads=[payloads["l2l_none"]["updates"]], steps=1,
            )
            baseline["update_floor_check"] = stand.judge_tolerance(
                update_ref, update_cand, mode="bf16", floor=update_floor,
            )
    else:
        baseline["missing_successful_arms"] = sorted(needed - payloads.keys())
    window["resident_ga_vs_uncompressed_l2l"] = baseline
    fp8: dict[str, Any] = {"status": "BLOCKED", "comparison": "l2l_fp8 vs l2l_none only"}
    if {"l2l_none", "l2l_fp8"} <= payloads.keys():
        reference, candidate = payloads["l2l_none"], payloads["l2l_fp8"]
        gradient = gradient_metrics(reference["gradients"], candidate["gradients"])
        updates = gradient_metrics(reference["updates"], candidate["updates"])
        bounds = thresholds()

        def gate(metrics: dict[str, Any]) -> bool:
            return bool(metrics.get("finite")) and (
                metrics.get("global_rel_l2", math.inf) <= bounds["gradient_global_rel_l2_max"]
                and metrics.get("min_cosine", -math.inf) >= bounds["gradient_min_cosine_min"]
            )

        fp8.update({
            "status": "PASS" if gate(gradient) and gate(updates) else "FAIL",
            "gradient": gradient, "updates": updates, "thresholds": bounds,
            "losses": loss_metrics(
                arms["l2l_none"]["per_microbatch_losses"],
                arms["l2l_fp8"]["per_microbatch_losses"],
            ),
            "baseline_a3_status": baseline["status"],
            "baseline_established": bool(baseline.get("pass")),
            "note": "G4 storage comparison does not establish or excuse the GA/L2L baseline",
        })
    else:
        fp8["missing_successful_arms"] = sorted({"l2l_none", "l2l_fp8"} - payloads.keys())
    window["fp8_vs_uncompressed_l2l"] = fp8


def run_case(
    root: Path, raw: Path, *, scope: str, weight_format: str, k: int, device: str,
    input_set: str,
) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {
        "adapter_scope": scope, "weight_format": weight_format, "k": k, "windows": [],
        "status": "BLOCKED", "input_set": input_set,
        "comparison_scope": (
            "Per window, all arms restore identical adapters and share identical inputs/labels. "
            "GA2 duplicates GA1 before any update. Only successful GA1 advances the next common "
            "window start; no cross-arm divergent optimizer trajectories are compared."
        ),
    }
    fixture = None
    try:
        admission = _plans(k, device)
        result["memory_admission"] = {
            "host_limit_bytes": HOST_LIMIT, "device_scratch_limit_bytes": DEVICE_SCRATCH_LIMIT,
            "fp8_device_scratch_bytes": admission["fp8_device_scratch_bytes"],
            "plans": {key: asdict(value) for key, value in admission["plans"].items()},
            "before_fixture_and_input_allocation": True,
        }
        fixture = build_bf16_fixture(
            root, device=device, targets=TARGETS if scope == "p3" else ALL_TARGETS,
            weight_format=weight_format,
        )
        result["precision_evidence"] = fixture.precision_evidence
        result["model_config"] = fixture.config.to_dict()
        canonical = root / "canonical.pt"
        torch.save(snapshot(fixture.streamed), canonical)
        windows = make_windows(fixture.config, k, device=device, input_set=input_set)
        for index, (ids, labels) in enumerate(windows):
            window: dict[str, Any] = {
                "index": index, "microbatches": len(ids), "short_final": len(ids) < k,
                "input_shapes": [list(batch.shape) for batch in ids],
                "input_hashes": [_digest(batch) for batch in ids],
                "label_hashes": [_digest(label) for label in labels],
                "label_tokens": sum(int((label[:, 1:] != -100).sum()) for label in labels),
                "canonical_adapter_file": f"{scope}-{weight_format}-k{k}-w{index}-start.pt",
                "arms": {},
            }
            start = torch.load(canonical, map_location="cpu", weights_only=True)
            torch.save(start, raw / window["canonical_adapter_file"])
            del start
            result["windows"].append(window)
            for name in ARMS:
                expected = None
                reference_name = "l2l_none" if name == "l2l_fp8" else "ga1"
                reference = window["arms"].get(reference_name)
                if reference is not None and reference.get("routing", {}).get("forward_complete"):
                    expected = reference.get("routing", {}).get("forward")
                artifact = raw / f"{scope}-{weight_format}-k{k}-w{index}-{name}.pt"
                codec = "fp8_e4m3" if name == "l2l_fp8" else "none"
                window["arms"][name] = _run_arm(
                    fixture, name=name, ids=ids, labels=labels, device=device,
                    canonical=canonical, artifact=artifact, plan=admission["plans"][codec],
                    expected_routes=expected, input_set=input_set, diagnostics=index == 0,
                )
            _compare(window, raw)
            ga1 = window["arms"]["ga1"]
            if ga1.get("optimizer_applied") and "raw_tensor_artifact" in ga1:
                payload = torch.load(raw / ga1["raw_tensor_artifact"], weights_only=True)
                torch.save(payload["parameters"], canonical)
                del payload
        result["status"] = "OBSERVED"
        result["baseline_updates_established_all_windows"] = all(
            window["resident_ga_vs_uncompressed_l2l"].get(
                "update_floor_check", {},
            ).get("pass", False)
            and window["resident_ga_vs_uncompressed_l2l"].get(
                "ga1_ga2_updates", {},
            ).get("exact", False)
            for window in result["windows"]
        )
        result["baseline_established_all_windows"] = all(
            window["resident_ga_vs_uncompressed_l2l"].get("pass", False)
            for window in result["windows"]
        )
        diagnostics = result["windows"][0]["arms"]["l2l_fp8"].get("decoder_diagnostics", {})
        result["fp8_decoder_gates"] = {
            "complete": bool(diagnostics.get("complete")),
            "e1": diagnostics.get("e1", {}).get("status", "MISSING"),
            "e2": diagnostics.get("e2", {}).get("status", "MISSING"),
            "scope": "first full window, including maximum sequence length; not all windows",
        }
        result["fp8_storage_gate_all_windows"] = (
            result["fp8_decoder_gates"]["complete"]
            and result["fp8_decoder_gates"]["e1"] == "PASS"
            and result["fp8_decoder_gates"]["e2"] == "PASS"
            and all(
                window["fp8_vs_uncompressed_l2l"]["status"] == "PASS"
                for window in result["windows"]
            )
        )
        result["short_final_note"] = (
            "k=1 has no nonempty shorter final window" if k == 1 else "short final exercised"
        )
    except Exception as error:
        result["error"] = _error(error)
        result["traceback"] = traceback.format_exc()
        for index, count in enumerate((k, k, max(1, k // 2))):
            if index < len(result["windows"]):
                window = result["windows"][index]
            else:
                window = {"index": index, "microbatches": count, "arms": {}}
                result["windows"].append(window)
            for name in ARMS:
                window["arms"].setdefault(name, {
                    "status": "BLOCKED", "optimizer_applied": False,
                    "reason": "fixture or case prerequisite failed", "error": result["error"],
                })
    finally:
        if fixture is not None:
            try:
                fixture.close()
            except Exception as error:
                result["runtime_close_error"] = _error(error)
                result["status"] = "FAIL"
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-scope", choices=("p3", "broad", "all"), default="all")
    parser.add_argument("--weight-format", choices=("dense", "nf4", "all"), default="all")
    parser.add_argument("--input-set", choices=("discovery", "holdout"), default="discovery")
    parser.add_argument("--ks", default=",".join(map(str, KS)))
    args = parser.parse_args(argv)
    try:
        ks = tuple(int(value) for value in args.ks.split(","))
    except ValueError:
        parser.error("--ks must be a comma-separated integer list")
    if not ks or any(k not in KS for k in ks) or len(set(ks)) != len(ks):
        parser.error("--ks must contain distinct values from 1,4,16,64")
    stand.set_deterministic_env()
    import torch

    stand.enable_deterministic()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(61)
    console = Console()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = args.output.with_suffix(".tensors")
    raw.mkdir(parents=True, exist_ok=True)
    scopes = ("p3", "broad") if args.adapter_scope == "all" else (args.adapter_scope,)
    formats = ("dense", "nf4") if args.weight_format == "all" else (args.weight_format,)
    source = source_identity()
    source["bf16_harness_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path(__file__), Path(__file__).with_name("p3_bf16_fixture.py"),
            Path(__file__).with_name("p3_bf16_diagnostics.py"),
            Path(__file__).with_name("l2l_mistral_stand.py"),
        ) if path.is_file()
    }
    versions = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "psutil"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report: dict[str, Any] = {
        "kind": "synthetic tiny native DeepSeekV3 BF16 investigation, not 671B training",
        "command": sys.argv if argv is None else list(argv), "source_identity": source,
        "platform": platform.platform(), "packages": versions, "cases": [],
        "device": args.device, "raw_tensor_directory": raw.name,
        "gpu": _gpu_evidence(args.device),
        "input_set": args.input_set, "data_offset": 10000 if args.input_set == "holdout" else 0,
        "determinism": {
            "strict": torch.are_deterministic_algorithms_enabled(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "autocast": False, "backward_outside_autocast": True,
            "weight_seed": 61, "adapter_seed": 67, "dropout": 0.0,
        },
        "g4_thresholds": thresholds(),
        "a3_rule": "unchanged l2l_mistral_stand.decide(mode='bf16')",
        "scope_note": (
            "No forced routes, no candidates selected, baseline and storage gates separate"
        ),
    }
    temporary_root = os.environ.get("TMPDIR") or tempfile.gettempdir()
    with tempfile.TemporaryDirectory(prefix="p3-bf16-", dir=temporary_root) as directory:
        for scope in scopes:
            for weight_format in formats:
                for k in ks:
                    console.print(f"BF16 {scope} {weight_format} k={k} {args.input_set}")
                    report["cases"].append(run_case(
                        Path(directory) / f"{scope}-{weight_format}-k{k}", raw,
                        scope=scope, weight_format=weight_format, k=k, device=args.device,
                        input_set=args.input_set,
                    ))
                    problems: list[str] = []
                    encoded = _json_safe(report, problems)
                    encoded["nonfinite_metric_paths"] = problems
                    args.output.write_text(json.dumps(encoded, indent=2), encoding="utf-8")
                    gc.collect()
                    if args.device == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
    complete = len(report["cases"]) == len(scopes) * len(formats) * len(ks)
    report["baseline_accepted"] = complete and all(
        case.get("baseline_established_all_windows", False)
        and case.get("baseline_updates_established_all_windows", False)
        for case in report["cases"]
    )
    report["fp8_accepted"] = report["baseline_accepted"] and all(
        case.get("fp8_storage_gate_all_windows", False) for case in report["cases"]
    )
    report["verdict"] = "PASS" if report["fp8_accepted"] else "BLOCKED_OR_FAILED"
    problems = []
    encoded = _json_safe(report, problems)
    encoded["nonfinite_metric_paths"] = problems
    args.output.write_text(json.dumps(encoded, indent=2, allow_nan=False), encoding="utf-8")
    console.print(f"Recorded {len(report['cases'])} cases: {report['verdict']} ({args.output})")
    return 0 if report["fp8_accepted"] and not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
