#!/usr/bin/env python3
"""P3 phase-0 FP8 boundary-storage experiment, with no downloads or giant-model load.

Run from the checkout root with src on PYTHONPATH:
    python benchmarks/harness/p3_fp8_smoke.py --device cuda --output p3.json

The tiny native DeepSeek fixture is synthetic, not DeepSeek-R1 671B training. Optional
--model-input adds local metadata, bounded attached-weight/native-router probes and
sequential activation-store roundtrips at the attached full model dimensions.
Every natural FP8 routing refusal fails the training case and remains in the report.
The independent negative control must refuse a deliberately changed recompute route.
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
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from rich.console import Console

try:
    from .b13_moe_fixture import (
        ALL_TARGETS,
        build_fixture,
        copy_adapter_weights,
        moe_batch,
        snapshot,
        trainables,
    )
except ImportError:
    from b13_moe_fixture import (
        ALL_TARGETS,
        build_fixture,
        copy_adapter_weights,
        moe_batch,
        snapshot,
        trainables,
    )


KS = (1, 4, 16, 64)
HOST_LIMIT = 256 * 1024 * 1024


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity() -> dict[str, Any]:
    """Hash the measured implementation even when a Kaggle archive has no .git directory."""
    import soup_cli

    package = Path(os.path.realpath(soup_cli.__file__))
    root = Path(os.path.realpath(__file__)).parents[2]
    product_root = package.parent
    files = {
        "harness/p3_fp8_smoke.py": Path(__file__),
        "harness/b13_moe_fixture.py": Path(sys.modules[build_fixture.__module__].__file__),
        "harness/g4_activation_codec.py": Path(__file__).with_name("g4_activation_codec.py"),
        "harness/g4_rule.py": Path(__file__).with_name("g4_rule.py"),
        "harness/p3_fp8_memory.py": Path(__file__).with_name("p3_fp8_memory.py"),
    }
    for relative in (
        "trainer/l2l_trainer.py", "utils/fp8_activations.py", "utils/activation_codec.py",
        "utils/l2l_step.py", "utils/l2l_activations.py", "utils/host_memory_budget.py",
        "utils/layer_stream_runtime.py", "utils/expert_stream.py",
        "utils/moe_l2l_budget.py", "utils/budgeted_activations.py", "utils/expert_shard.py",
        "utils/kimi_k2.py", "utils/peft_wiring.py",
    ):
        files[f"product/{relative}"] = product_root / relative
    result: dict[str, Any] = {
        "package_file": str(package), "harness_file": os.path.realpath(__file__),
        "sha256_manifest": {
            name: _sha256(path) if path.is_file() else None for name, path in files.items()
        },
        "launcher_revision": os.environ.get("SOUP_SOURCE_REVISION"),
    }
    try:
        def git(*args: str) -> str:
            return subprocess.run(
                ["git", *args], cwd=root, capture_output=True, check=True,
                text=True, timeout=10,
            ).stdout.strip()

        result.update({
            "revision": git("rev-parse", "HEAD"), "revision_kind": "observed git HEAD",
            "git_status_porcelain": git("status", "--porcelain"),
        })
    except (OSError, subprocess.SubprocessError) as error:
        result.update({
            "revision": result["launcher_revision"],
            "revision_kind": "launcher assertion; source hashes are observations",
            "git_unavailable": f"{type(error).__name__}: {error}",
        })
    return result


def _error(error: BaseException) -> dict[str, Any]:
    return {"type": type(error).__name__, "message": str(error)}


def _finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_tree(item) for item in value)
    return True


def _json_safe(value: Any, problems: list[str], path: str = "$") -> Any:
    """Write strict JSON; a missing/nonfinite metric never becomes a passing number."""
    if isinstance(value, float) and not math.isfinite(value):
        problems.append(path)
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item, problems, f"{path}.{key}")
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, problems, f"{path}[{index}]")
                for index, item in enumerate(value)]
    return value


def thresholds() -> dict[str, float]:
    try:
        from . import g4_rule
    except ImportError:
        import g4_rule

    return {
        "restore_rms_row_rel_max": g4_rule.RESTORE_RMS_BOUND,
        "restore_worst_row_rel_max": g4_rule.RESTORE_WORST_ROW_BOUND,
        "gradient_global_rel_l2_max": g4_rule.GRAD_GLOBAL_BOUND,
        "gradient_min_cosine_min": g4_rule.GRAD_MIN_COSINE,
        "branch_rms_row_rel_max": g4_rule.BRANCH_RMS_BOUND,
    }


def _codec_metrics() -> Any:
    try:
        from . import g4_activation_codec
    except ImportError:
        import g4_activation_codec
    return g4_activation_codec


def gradient_metrics(want: dict[str, Any], got: dict[str, Any]) -> dict[str, Any]:
    import torch

    missing = sorted(name for name in want.keys() | got.keys()
                     if want.get(name) is None or got.get(name) is None)
    if missing:
        return {"status": "FAIL", "finite": False, "missing_gradients": missing}
    finite = all(bool(torch.isfinite(value).all()) for value in (*want.values(), *got.values()))
    if not finite:
        return {"status": "FAIL", "finite": False, "reason": "nonfinite gradient"}
    metrics = _codec_metrics().grad_agreement(want, got)
    return {
        "status": "OBSERVED", "finite": _finite_tree(metrics),
        "exact": all(torch.equal(want[name], got[name]) for name in want), **metrics,
    }


def loss_metrics(want: Sequence[float], got: Sequence[float]) -> dict[str, Any]:
    if len(want) != len(got) or not want:
        return {"status": "FAIL", "finite": False, "reference": list(want),
                "candidate": list(got), "reason": "missing per-microbatch loss"}
    finite = all(math.isfinite(item) for item in (*want, *got))
    differences = [abs(left - right) for left, right in zip(want, got)]
    return {
        "status": "OBSERVED", "finite": finite, "reference": list(want),
        "candidate": list(got), "exact": list(want) == list(got),
        "max_abs_error": max(differences),
        "global_rel_l2": math.sqrt(sum(item * item for item in differences)
                                   / max(sum(item * item for item in want), 1e-300)),
        "reference_window_loss": sum(want), "candidate_window_loss": sum(got),
    }


def _rss() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        if sys.platform.startswith("linux"):
            with open("/proc/self/statm", encoding="ascii") as handle:
                return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    return None


def _lifetime_rss_peak() -> int | None:
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak if sys.platform == "darwin" else peak * 1024
    except ImportError:
        return None


class MemoryObservation:
    """Observed whole-process peaks, not activation-attributed allocator estimates."""

    def __init__(self, device: str, *, scope: str | None = None) -> None:
        self.device = device
        self.scope = scope
        self.stop = threading.Event()
        self.peak: int | None = None
        self.samples = 0
        self.errors: list[str] = []
        self.thread: threading.Thread | None = None
        self.result: dict[str, Any] = {}

    def _sample(self) -> None:
        try:
            rss = _rss()
            if rss is not None:
                self.peak = max(self.peak or 0, rss)
                self.samples += 1
        except (OSError, ValueError) as error:
            self.errors.append(str(error))

    def _loop(self) -> None:
        while not self.stop.wait(0.01):
            self._sample()

    def __enter__(self) -> "MemoryObservation":
        import torch

        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.result = {
            "kind": "observed process/allocator bytes; not activation-only memory",
            "process_rss_before_bytes": _rss(),
            "cuda_allocated_before_bytes": (
                torch.cuda.memory_allocated() if self.device == "cuda" else None
            ),
            "cuda_reserved_before_bytes": (
                torch.cuda.memory_reserved() if self.device == "cuda" else None
            ),
            "scope": (
                "Includes resident and streamed twins, resident gradients/CPU gradient copy, "
                "initial adapter snapshot, inputs, runtime, route digests and trainer. Excludes "
                "E1/E2 diagnostic clones and candidate gradient snapshots. Pinned caching "
                "and process memory from earlier cases can remain in the baseline."
            ),
        }
        if self.scope is not None:
            self.result["scope"] = self.scope
        self._sample()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.started = time.perf_counter()
        return self

    def __exit__(self, *_exc: Any) -> None:
        import torch

        try:
            if self.device == "cuda":
                torch.cuda.synchronize()
            self._sample()
            self.result.update({
                "wall_seconds": time.perf_counter() - self.started,
                "process_rss_after_bytes": _rss(),
                "process_rss_sampled_peak_bytes": self.peak,
                "process_rss_samples": self.samples,
                "process_rss_sampling_interval_seconds": 0.01,
                "process_rss_peak_kind": "sampled lower bound, short-lived peaks may be missed",
                "process_lifetime_rss_high_water_bytes": _lifetime_rss_peak(),
                "cuda_allocated_peak_bytes": (
                    torch.cuda.max_memory_allocated() if self.device == "cuda" else None
                ),
                "cuda_reserved_peak_bytes": (
                    torch.cuda.max_memory_reserved() if self.device == "cuda" else None
                ),
                "cuda_allocated_after_bytes": (
                    torch.cuda.memory_allocated() if self.device == "cuda" else None
                ),
                "cuda_reserved_after_bytes": (
                    torch.cuda.memory_reserved() if self.device == "cuda" else None
                ),
            })
        finally:
            self.stop.set()
            if self.thread is not None:
                self.thread.join()
            self.result["sampling_errors"] = self.errors


def make_windows(config: Any, k: int, device: str) -> list[tuple[list[Any], list[Any]]]:
    """Two full windows, then an unpadded short final window when k > 1."""
    result = []
    for step, count in enumerate((k, k, max(1, k // 2))):
        ids, labels = [], []
        for mb in range(count):
            seq = (7, 4, 3)[step] + mb % 3
            rows = 1 + mb % 2
            batch = moe_batch(config, seed=step * k + mb, rows=rows, seq=seq)
            lab = batch["labels"]
            lab[0, :1 + mb % 2] = -100
            ids.append(batch["input_ids"].to(device))
            labels.append(lab.to(device))
        result.append((ids, labels))
    return result


class RouteObservation:
    """Observe routed IDs without changing them or replacing the product RoutingGuard."""

    def __init__(self, model: Any) -> None:
        from soup_cli.utils.expert_stream import streamed_experts_class

        self.events: list[dict[str, Any]] = []
        self.handles = []
        for module in model.modules():
            if isinstance(module, streamed_experts_class()):
                self.handles.append(module.register_forward_pre_hook(
                    self._watch(module.layer_idx), with_kwargs=True,
                ))
        if not self.handles:
            raise AssertionError("native DeepSeek fixture has no streamed experts to guard")

    def _watch(self, layer: int) -> Callable[..., None]:
        def watch(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            import torch

            routed = kwargs["top_k_index"] if "top_k_index" in kwargs else args[1]
            raw = routed.detach().cpu().contiguous().numpy().tobytes()
            self.events.append({
                "layer": layer, "recompute": torch.is_grad_enabled(),
                "shape": list(routed.shape), "sha256": hashlib.sha256(raw).hexdigest(),
            })
        return watch

    def reset(self) -> None:
        self.events.clear()

    def evidence(self, k: int, *, completed: bool) -> dict[str, Any]:
        grouped: dict[int, dict[str, list[str]]] = {}
        for event in self.events:
            group = grouped.setdefault(event["layer"], {"forward": [], "recompute": []})
            group["recompute" if event["recompute"] else "forward"].append(event["sha256"])
        layers = {
            str(layer): {
                "forward_visits": len(visits["forward"]),
                "recompute_visits": len(visits["recompute"]),
                "ordered_route_digest_equal": visits["forward"] == visits["recompute"],
                **visits,
            }
            for layer, visits in grouped.items()
        }
        observed_equal = bool(layers) and all(
            row["forward_visits"] == row["recompute_visits"] == k
            and row["ordered_route_digest_equal"] for row in layers.values()
        )
        return {
            "product_guard": "L2LStep.run RoutingGuard, unchanged and always enabled",
            "completed": completed, "all_observed_routes_equal": observed_equal,
            "expected_moe_layers": len(self.handles), "layers": layers,
            "complete": len(layers) == len(self.handles) and observed_equal,
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def store_evidence(store: Any, budget: Any) -> dict[str, Any]:
    """Keep estimates and tensor-storage observations in separate fields."""
    measured = None
    if hasattr(store, "host_bytes"):
        measured = dict(store.host_bytes())
    else:
        storages = {}
        for tensor in store._resident.values():
            storage = tensor.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
        measured = {"owned_host_storage_bytes": sum(storages.values())}
    allocation = getattr(store, "allocation", None)
    stats = store.stats
    stats_dict = asdict(stats) if is_dataclass(stats) else dict(vars(stats))
    return {
        "kind": "tensor storage and budget observations, not process RAM",
        "class": type(store).__name__, "mb_capacity_shapes": list(store.mb_shapes),
        "n_layers": store.n_layers, "capacity_k": store.k,
        "storage_observed": measured,
        "allocation_estimate": allocation.as_dict() if allocation is not None else None,
        "allocation_estimate_kind": (
            "conservative product estimator; includes allocator rounding and one output peak; "
            "device transient is not host memory"
        ),
        "budget": {
            "kind": "observed accounting reservation; not an RSS limit",
            "limit_bytes": budget.limit_bytes, "fixed_reserved_bytes": budget.reserved_bytes,
            "live_used_bytes": budget.used_bytes, "available_bytes": budget.available_bytes,
        },
        "stats": stats_dict,
    }


def diagnose_restore(model: Any, store: Any, ids: Sequence[Any]) -> dict[str, Any]:
    """E1/E2 on native eager twin boundaries; diagnostic clones are outside memory peaks."""
    import torch

    from soup_cli.utils.l2l_step import model_parts

    metrics = _codec_metrics()
    layers = model_parts(model)[1].layers
    originals: dict[tuple[int, int], Any] = {}
    outputs: dict[tuple[int, int], Any] = {}
    calls: dict[tuple[int, int], Any] = {}
    handles = []
    current_mb = 0
    for index, layer in enumerate(layers):
        def capture(
            _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any],
            *, layer_index: int = index,
        ) -> None:
            boundary = args[0]
            originals[(layer_index, current_mb)] = boundary.detach().cpu().clone()
            calls[(layer_index, current_mb)] = (args[1:], kwargs.copy())
            store.put(layer_index, current_mb, boundary)

        def capture_output(
            _module: Any, _args: Any, output: Any, *, layer_index: int = index,
        ) -> None:
            value = output[0] if isinstance(output, tuple) else output
            outputs[(layer_index, current_mb)] = value.detach().cpu().clone()

        handles.append(layer.register_forward_pre_hook(capture, with_kwargs=True))
        handles.append(layer.register_forward_hook(capture_output))
    try:
        with torch.no_grad():
            for current_mb, batch in enumerate(ids):
                model(input_ids=batch)
    finally:
        for handle in handles:
            handle.remove()
    parts: dict[int, list[dict[str, float]]] = {index: [] for index in range(len(layers))}
    branches: dict[int, list[dict[str, float]]] = {index: [] for index in range(len(layers))}
    try:
        store.begin_backward()
        with torch.no_grad():
            for layer_index in reversed(range(len(layers))):
                for mb in range(len(ids)):
                    restored = store.get(layer_index, mb)
                    original = originals.pop((layer_index, mb))
                    parts[layer_index].append(metrics.restore_error(
                        original, restored.detach().cpu(),
                    ))
                    args, kwargs = calls.pop((layer_index, mb))
                    output = layers[layer_index](restored, *args, **kwargs)
                    output = output[0] if isinstance(output, tuple) else output
                    branches[layer_index].append(metrics.restore_error(
                        outputs.pop((layer_index, mb)) - original,
                        (output - restored).detach().cpu(),
                    ))
        store.end_step()
    finally:
        originals.clear()
        outputs.clear()
        calls.clear()
    bounds = thresholds()
    rows = {str(index): metrics.merge_errors(values) for index, values in parts.items()}
    branch_rows = {str(index): metrics.merge_errors(values)
                   for index, values in branches.items()}
    finite = _finite_tree(rows)
    passed = finite and all(
        row["rms_row_rel"] <= bounds["restore_rms_row_rel_max"]
        and row["worst_row_rel"] <= bounds["restore_worst_row_rel_max"]
        for row in rows.values()
    )
    branch_ok = _finite_tree(branch_rows) and all(
        row["rms_row_rel"] <= bounds["branch_rms_row_rel_max"]
        for row in branch_rows.values()
    )
    return {
        "status": "PASS" if passed else "FAIL", "finite": finite,
        "kind": "observed native eager-twin boundary -> production store -> restore, G4 E1",
        "microbatches": len(ids), "expected_chunks": len(layers) * len(ids),
        "observed_chunks": sum(len(values) for values in parts.values()), "layers": rows,
        "branch_e2": {
            "status": "PASS" if branch_ok else "FAIL", "finite": _finite_tree(branch_rows),
            "kind": "native eager layer residual branch output-input, restored vs original",
            "layers": branch_rows, "rms_row_rel_bound": bounds["branch_rms_row_rel_max"],
        },
        "memory_scope": "diagnostic CPU boundary/output clones excluded from training peaks",
    }


def _runtime_delta(before: dict[str, Any], after: dict[str, Any], k: int) -> dict[str, Any]:
    keys = ("expert_fetches", "expert_bytes_read")
    missing = [key for key in keys if key not in before or key not in after]
    if missing:
        raise ValueError(f"runtime omitted observed expert counters: {missing}")
    values = {key: after[key] - before[key] for key in keys}
    return {
        "kind": "observed runtime counters; expert bytes are logical decoded bytes, not disk I/O",
        **values, "microbatches": k,
        "expert_fetches_per_microbatch": values["expert_fetches"] / k,
        "logical_expert_bytes_per_microbatch": values["expert_bytes_read"] / k,
    }


def _negative_control(
    fixture: Any, state: Any, ids: list[Any], labels: list[Any],
) -> dict[str, Any]:
    """Change expert IDs only in recompute; the existing guard must stop the step."""
    import torch

    from soup_cli.trainer.l2l_trainer import run_window
    from soup_cli.utils.expert_stream import streamed_experts_class

    changed = 0
    handles = []

    def corrupt(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        nonlocal changed
        if not torch.is_grad_enabled():
            return None
        changed += 1
        if "top_k_index" in kwargs:
            return args, {**kwargs, "top_k_index": (kwargs["top_k_index"] + 1)
                          % fixture.config.n_routed_experts}
        return (args[0], (args[1] + 1) % fixture.config.n_routed_experts, *args[2:]), kwargs

    fixture.streamed.zero_grad(set_to_none=True)
    for module in fixture.streamed.modules():
        if isinstance(module, streamed_experts_class()):
            handles.append(
                module.register_forward_pre_hook(corrupt, with_kwargs=True, prepend=True)
            )
    result: dict[str, Any] = {
        "status": "FAIL", "control": "uncompressed store, changed recompute expert IDs",
        "optimizer_steps": 0, "product_guard_bypassed": False,
    }
    try:
        total = sum(int((label[:, 1:] != -100).sum()) for label in labels)
        run_window(state, fixture.streamed, ids, labels, total, ids[0].device)
        result["reason"] = "guard accepted an injected route change"
    except RuntimeError as error:
        result["error"] = _error(error)
        detected = "routing guard:" in str(error) and "routed differently" in str(error)
        result["status"] = "PASS" if detected and changed > 0 else "FAIL"
    except Exception as error:
        result["error"] = _error(error)
    finally:
        for handle in handles:
            handle.remove()
        fixture.streamed.zero_grad(set_to_none=True)
    result["corrupted_recompute_calls"] = changed
    return result


def run_arm(root: Path, *, k: int, device: str, codec: str) -> dict[str, Any]:
    import torch

    from soup_cli.trainer.l2l_trainer import (
        L2LState,
        _l2l_step_for,
        _store_for,
        run_window,
    )
    from soup_cli.utils.host_memory_budget import HostMemoryBudget
    from soup_cli.utils.l2l_step import model_parts
    from soup_cli.utils.moe_l2l_budget import plan_moe_l2l

    result: dict[str, Any] = {
        "codec": codec, "k": k, "status": "FAIL", "windows": [],
        "optimizer_steps": 0,
        "accepted_optimizer_steps": 0, "routing_refusals": 0,
        "comparison_scope": (
            "Independent FP32 native resident-eager GA at identical current adapter weights; "
            "resident adapters copied from streamed before each window. SGD has no momentum. "
            "No router weights, correction biases, input seeds or routes are modified."
        ),
        "exactness_claim": "none for FP8; uncompressed control requires bitwise equality",
    }
    fixture = state = observer = None
    budget = HostMemoryBudget(HOST_LIMIT)
    try:
        fixture = build_fixture(root, device=device, targets=ALL_TARGETS, dropout=0.0)
        windows = make_windows(fixture.config, k, device)
        max_shape = (
            max(batch.shape[0] for ids, _ in windows for batch in ids),
            max(batch.shape[1] for ids, _ in windows for batch in ids),
            fixture.config.hidden_size,
        )
        memory_plan = plan_moe_l2l(
            budget, n_layers=fixture.config.num_hidden_layers,
            n_moe_layers=fixture.config.num_hidden_layers - fixture.config.first_k_dense_replace,
            max_shape=max_shape, experts_per_token=fixture.config.num_experts_per_tok,
            requested_k=k, dtype="float32", pin=device == "cuda",
            codec=codec, device_type=device,
        )
        result["resolved_memory_plan"] = asdict(memory_plan)
        if memory_plan.effective_k != k:
            raise MemoryError("fixed-k experiment refuses a planner-reduced accumulation window")
        result["model"] = {
            "kind": "synthetic tiny native DeepSeek-V3, random local checkpoint",
            "class": type(fixture.resident.get_base_model()).__name__,
            "config": fixture.config.to_dict(),
            "trainable_tensors": len(trainables(fixture.streamed)),
            "trainable_elements": sum(
                param.numel() for param in trainables(fixture.streamed).values()
            ),
            "lora_targets": list(ALL_TARGETS), "dtype": "float32", "seed": 61,
            "dropout": 0.0, "giant_model_training": False,
        }
        head_losses: list[float] = []

        causal = model_parts(fixture.streamed)[0]
        original_loss = causal.loss_function

        def observed_loss(*args: Any, **kwargs: Any) -> Any:
            loss = original_loss(*args, **kwargs)
            head_losses.append(float(loss.detach().cpu()))
            return loss

        causal.loss_function = observed_loss

        state = L2LState(
            runtime=fixture.runtime, dtype=torch.float32, activation_codec=codec,
            host_budget=budget, has_dropout=False, memory_plan=memory_plan,
        )
        # Allocate through the product trainer, not a benchmark-only store replacement.
        stepper, _ = _l2l_step_for(state, fixture.streamed, windows[0][0])
        store = _store_for(state, stepper, windows[0][0], windows[0][0][0].device)
        result["store_before"] = store_evidence(store, budget)
        result["restore_e1"] = diagnose_restore(fixture.resident, store, windows[0][0])
        gc.collect()
        observer = RouteObservation(fixture.streamed)
        optimizers = [torch.optim.SGD(trainables(model).values(), lr=0.05)
                      for model in (fixture.resident, fixture.streamed)]
        initial_adapters = snapshot(fixture.streamed)
        bounds = thresholds()
        for index, (ids, labels) in enumerate(windows):
            row: dict[str, Any] = {
                "index": index, "microbatches": len(ids),
                "input_shapes": [list(batch.shape) for batch in ids],
                "short_final_window": len(ids) < k,
                "status": "FAIL", "optimizer_applied": False,
            }
            result["windows"].append(row)
            copy_adapter_weights(fixture.streamed, fixture.resident)
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            total = sum(int((label[:, 1:] != -100).sum()) for label in labels)
            row["label_tokens"] = total
            reference_losses = []
            with MemoryObservation(device) as resident_memory:
                for batch, label in zip(ids, labels):
                    loss = fixture.resident(
                        input_ids=batch, labels=label, num_items_in_batch=total,
                    ).loss
                    loss.backward()
                    reference_losses.append(float(loss.detach().cpu()))
                del loss
            row["resident_eager_memory"] = resident_memory.result
            reference_grads = snapshot(fixture.resident, gradients=True)
            observer.reset()
            head_losses.clear()
            before = fixture.runtime.stats()
            failure = None
            with MemoryObservation(device) as streamed_memory:
                try:
                    losses, counters = run_window(
                        state, fixture.streamed, ids, labels, total, ids[0].device,
                    )
                    row["load_counters"] = asdict(counters)
                    del losses
                except Exception as error:
                    failure = _error(error)
                    row["error"] = failure
            row["streamed_memory"] = streamed_memory.result
            row["expert_scaling"] = _runtime_delta(before, fixture.runtime.stats(), len(ids))
            row["routing"] = observer.evidence(len(ids), completed=failure is None)
            row["loss"] = loss_metrics(reference_losses, head_losses)
            row["store"] = (store_evidence(state.store, budget) if state.store is not None
                            else {"status": "UNAVAILABLE", "reason": "store construction failed"})
            if failure is not None:
                refusal = (failure["type"] == "RuntimeError"
                           and "routing guard:" in failure["message"]
                           and "routed differently" in failure["message"])
                row["status"] = "REFUSED" if refusal and codec != "none" else "FAIL"
                row["gradient"] = {
                    "status": "BLOCKED", "reason": "failed step; partial gradients discarded",
                }
                result["routing_refusals"] += int(refusal)
                fixture.streamed.zero_grad(set_to_none=True)
                fixture.resident.zero_grad(set_to_none=True)
                del reference_grads
                # Do not retry a selected input or change routing. Continue the fixed schedule.
                continue
            candidate_grads = snapshot(fixture.streamed, gradients=True)
            row["gradient"] = gradient_metrics(reference_grads, candidate_grads)
            gradient = row["gradient"]
            numerical_ok = bool(gradient.get("finite")) and bool(row["loss"]["finite"])
            if codec == "none":
                gate = (numerical_ok and gradient.get("exact", False)
                        and row["loss"].get("exact", False))
            else:
                gate = (
                    numerical_ok
                    and gradient.get("global_rel_l2", math.inf)
                    <= bounds["gradient_global_rel_l2_max"]
                    and gradient.get("min_cosine", -math.inf)
                    >= bounds["gradient_min_cosine_min"]
                )
            row["g4_gradient_gate"] = "PASS" if gate else "FAIL"
            row["status"] = "PASS" if gate and row["routing"]["complete"] else "FAIL"
            if gate and row["routing"]["complete"]:
                before_updates = [snapshot(model)
                                  for model in (fixture.resident, fixture.streamed)]
                for optimizer in optimizers:
                    optimizer.step()
                row["optimizer_applied"] = True
                result["optimizer_steps"] += 1
                after_updates = [snapshot(model)
                                 for model in (fixture.resident, fixture.streamed)]
                deltas = [
                    {name: after[name] - before[name] for name in before}
                    for before, after in zip(before_updates, after_updates)
                ]
                update = gradient_metrics(deltas[0], deltas[1])
                row["adapter_update_agreement"] = {
                    "kind": "observed per-window optimizer delta, after minus before",
                    **update,
                }
                if codec == "none":
                    update_ok = update.get("finite", False) and update.get("exact", False)
                else:
                    update_ok = (
                        update.get("finite", False)
                        and update.get("global_rel_l2", math.inf)
                        <= bounds["gradient_global_rel_l2_max"]
                        and update.get("min_cosine", -math.inf)
                        >= bounds["gradient_min_cosine_min"]
                    )
                row["optimizer_update_gate"] = "PASS" if update_ok else "FAIL"
                if update_ok:
                    result["accepted_optimizer_steps"] += 1
                else:
                    row["status"] = "FAIL"
                del before_updates, after_updates, deltas
            else:
                row["adapter_update_agreement"] = {
                    "status": "BLOCKED", "reason": "gradient or routing gate failed; no update",
                }
            del reference_grads, candidate_grads
        final_adapters = snapshot(fixture.streamed)
        result["adapter_parameters_changed"] = any(
            not torch.equal(initial_adapters[name], final_adapters[name])
            for name in initial_adapters
        )
        del initial_adapters, final_adapters
        result["store_after"] = store_evidence(state.store, budget)
        if codec == "none":
            observer.close()
            observer = None
            result["routing_negative_control"] = _negative_control(
                fixture, state, *windows[-1],
            )
        passes = all(row["status"] == "PASS" for row in result["windows"])
        passes = (passes and result["accepted_optimizer_steps"] >= 2
                  and result["adapter_parameters_changed"]
                  and result["restore_e1"]["status"] == "PASS"
                  and result["restore_e1"]["branch_e2"]["status"] == "PASS")
        if codec == "none":
            passes = passes and result["routing_negative_control"]["status"] == "PASS"
        result["status"] = "PASS" if passes else (
            "REFUSED" if result["routing_refusals"] else "FAIL"
        )
        result["failure_reasons"] = [
            f"window {row['index']}: {row['status']}"
            for row in result["windows"] if row["status"] != "PASS"
        ]
        if result["accepted_optimizer_steps"] < 2:
            result["failure_reasons"].append("fewer than two accepted optimizer steps")
        if not result["adapter_parameters_changed"]:
            result["failure_reasons"].append("no adapter parameter changed")
        if result["restore_e1"]["status"] != "PASS":
            result["failure_reasons"].append("G4 E1 restoration threshold failed")
        if result["restore_e1"]["branch_e2"]["status"] != "PASS":
            result["failure_reasons"].append("G4 E2 residual-branch threshold failed")
        if codec == "none" and result["routing_negative_control"]["status"] != "PASS":
            result["failure_reasons"].append("independent routing negative control failed")
        result["short_final_window_exercised"] = any(
            row["short_final_window"] for row in result["windows"]
        )
        result["short_final_window_note"] = (
            "k=1 cannot have a nonempty window shorter than k" if k == 1 else None
        )
    except Exception as error:
        result["error"] = _error(error)
        result["traceback"] = traceback.format_exc()
    finally:
        if observer is not None:
            observer.close()
        if state is not None:
            try:
                state.close()
            except Exception as error:
                result["close_error"] = _error(error)
                result["status"] = "FAIL"
        result["budget_used_after_close_bytes"] = budget.used_bytes
        if budget.used_bytes:
            result["status"] = "FAIL"
            result["budget_release_error"] = "activation reservation remains after close"
        if fixture is not None:
            try:
                fixture.close()
            except Exception as error:
                result["runtime_close_error"] = _error(error)
                result["status"] = "FAIL"
    return result


def inspect_model_input(root: Path | None, *, device: str = "cpu") -> dict[str, Any]:
    """Read bounded local evidence, never AutoModel/from_pretrained for the giant model."""
    result: dict[str, Any] = {
        "status": "NOT_REQUESTED" if root is None else "FAIL",
        "training_671b": {
            "status": "BLOCKED", "optimizer_steps": 0,
            "reason": (
                "This command trains only the synthetic tiny fixture. Config/index inspection "
                "and a bounded weight slice do not exercise a DeepSeek-R1 671B training step. "
                "No full giant model is loaded, downloaded or instantiated."
            ),
        },
    }
    if root is None:
        return result
    try:
        root = Path(os.path.realpath(root))
        if not root.is_dir():
            raise ValueError(f"model input is not a local directory: {root}")
        available, configs = [], []
        truncated = False
        for directory, dirs, names in os.walk(root, followlinks=False):
            dirs.sort()
            for name in sorted(names):
                path = Path(directory) / name
                relative = path.relative_to(root).as_posix()
                available.append({"path": relative, "size_bytes": path.stat().st_size})
                if name == "config.json":
                    configs.append(path)
                if len(available) >= 10000:
                    truncated = True
                    break
            if truncated:
                break
        result.update({"root": str(root), "available_files": available,
                       "available_files_truncated_at_10000": truncated})
        selected = None
        for path in configs:
            if path.stat().st_size > 4 * 1024 * 1024:
                continue
            config = json.loads(path.read_text(encoding="utf-8"))
            if config.get("model_type") == "deepseek_v3":
                selected = (path, config)
                break
        if selected is None:
            raise ValueError("no local DeepSeek-V3/R1 config.json found")
        config_path, config = selected
        fields = (
            "_name_or_path", "model_type", "architectures", "transformers_version",
            "torch_dtype", "dtype", "num_hidden_layers", "hidden_size", "vocab_size",
            "n_routed_experts", "num_experts_per_tok", "moe_intermediate_size",
            "first_k_dense_replace", "quantization_config",
        )
        result["model_identity"] = {
            "kind": "observed local metadata; name alone is not a weight authenticity proof",
            "config_path": str(config_path), "config_sha256": _sha256(config_path),
            **{field: config.get(field) for field in fields},
        }
        shape_matches = (
            config.get("num_hidden_layers") == 61 and config.get("hidden_size") == 7168
            and config.get("n_routed_experts") == 256
        )
        result["matches_r1_671b_architecture_shape"] = shape_matches
        if not shape_matches:
            raise ValueError(
                "attached config does not match the 61-layer/7168-hidden/256-expert shape"
            )
        try:
            from .p3_fp8_memory import embedding_router_probe, full_shape_store_probes
        except ImportError:
            from p3_fp8_memory import embedding_router_probe, full_shape_store_probes

        result["full_shape_activation_storage"] = full_shape_store_probes(config, device=device)
        index_path = config_path.with_name("model.safetensors.index.json")
        if not index_path.is_file() or index_path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("local model.safetensors.index.json missing or exceeds 32 MiB bound")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("safetensors index has no tensor map")
        expert_keys = sorted(name for name in weight_map
                             if ".experts." in name and name.endswith("gate_proj.weight"))
        if not expert_keys:
            raise ValueError("index has no routed expert gate_proj.weight tensor")
        name = expert_keys[0]
        shard = Path(os.path.realpath(config_path.parent / weight_map[name]))
        if os.path.commonpath((str(root), str(shard))) != str(root):
            raise ValueError("index tensor shard escapes supplied model input directory")
        result["index"] = {
            "path": str(index_path), "sha256": _sha256(index_path),
            "tensor_count": len(weight_map), "shard_count": len(set(weight_map.values())),
            "metadata": index.get("metadata"),
            "available_referenced_shards": sum(
                (config_path.parent / filename).is_file() for filename in set(weight_map.values())
            ),
        }
        result["embedding_router_probe"] = embedding_router_probe(
            root=root, model_dir=config_path.parent, config_dict=config,
            weight_map=weight_map, device=device,
        )
        import torch
        from safetensors import safe_open

        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            view = handle.get_slice(name)
            shape = view.get_shape()
            if len(shape) != 2 or not all(type(dim) is int and dim > 0 for dim in shape):
                raise ValueError(f"expected a 2D nonempty expert tensor, got {shape}")
            tensor = view[:min(2, shape[0]), :min(16, shape[1])]
        if tensor.numel() > 256:
            raise AssertionError("bounded weight slice exceeded its 256-element limit")
        values = tensor.float()
        raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        result["weight_slice"] = {
            "kind": "observed safetensors slice, not a full tensor/model load",
            "shard": str(shard), "tensor": name, "full_shape": shape,
            "dtype": str(tensor.dtype), "slice_shape": list(tensor.shape),
            "slice_elements": tensor.numel(), "slice_bytes": len(raw),
            "slice_sha256": hashlib.sha256(raw).hexdigest(),
            "finite": bool(torch.isfinite(values).all()),
            "minimum": float(values.min()), "maximum": float(values.max()),
            "expected_gate_shape": [config.get("moe_intermediate_size"), config["hidden_size"]],
            "shape_matches_config": shape == [config.get("moe_intermediate_size"),
                                              config["hidden_size"]],
        }
        valid = result["weight_slice"]["finite"] and result["weight_slice"]["shape_matches_config"]
        valid = (
            valid and result["embedding_router_probe"]["status"] == "PASS"
            and result["full_shape_activation_storage"]["status"] == "PASS"
        )
        result["status"] = "PASS" if valid else "FAIL"
    except Exception as error:
        result["error"] = _error(error)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-input", type=Path)
    args = parser.parse_args(argv)
    console = Console()
    result: dict[str, Any] = {
        "schema_version": 1, "experiment": "Soup P3 phase 0 FP8 boundary activations",
        "verdict": "FAIL", "device": args.device, "compute_dtype": "float32",
        "requested_k": list(KS), "cases": [],
        "dtype_coverage": {
            "tiny_resident_eager_ga_and_product_training": "float32 only",
            "full_dimension_activation_store_roundtrip": (
                "bfloat16 compute/storage conversion, no model training; requires --model-input"
            ),
            "attached_embedding_router_probe": (
                "attached embedding dtype; native router float32 linear arithmetic, no training"
            ),
            "uncompressed_bfloat16_ga_parity": {
                "status": "NOT_RUN", "reason": (
                    "The existing native fixture constructs float32 training twins. BF16 "
                    "codec roundtrip is not BF16 gradient-accumulation parity evidence."
                ),
            },
        },
        "protocol": {
            "trainer_execution_scope": (
                "Production L2LState/run_window/_store_for and L2LStep with standalone SGD. "
                "Does not invoke HF Trainer.training_step, gradient scaler, scheduler, "
                "checkpoint/resume or the complete HF optimizer orchestration."
            ),
            "windows": "two full windows then max(1, k//2); unpadded variable sequence lengths",
            "optimizer": "torch.optim.SGD, lr=0.05, no momentum; at least two valid steps required",
            "strict_routing_guard": True, "routing_refusal_is_training_failure": True,
            "g4_thresholds_unchanged": True, "full_g4_quality_verdict_claimed": False,
            "fp8_exactness_claimed": False,
            "network": "none; local fixture and optional attached files only; HF offline enabled",
            "memory": "observed RSS/CUDA and storage/budget accounting labelled separately",
            "refusal_policy": (
                "Discard partial gradients, do not step, do not change router/input/seeds; "
                "continue every remaining predeclared window and every k."
            ),
        },
        "environment": {"python": sys.version, "platform": platform.platform()},
    }
    try:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8") != ":4096:8":
            raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        result["source"] = source_identity()
        result["thresholds"] = thresholds()
        result["environment"].update({
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name() if args.device == "cuda" else None,
            "deterministic_algorithms": True, "tf32": False, "torch_threads": 1,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            **{name: importlib.metadata.version(name)
               for name in ("transformers", "peft", "accelerate", "safetensors")},
        })
        result["attached_model"] = inspect_model_input(args.model_input, device=args.device)
        with tempfile.TemporaryDirectory(prefix="p3-fp8-smoke-") as temporary:
            for k in KS:
                case: dict[str, Any] = {"k": k, "arms": []}
                result["cases"].append(case)
                for codec in ("none", "fp8_e4m3"):
                    arm = run_arm(
                        Path(temporary) / f"k{k}-{codec}", k=k, device=args.device, codec=codec,
                    )
                    case["arms"].append(arm)
                    console.print(
                        f"k={k} codec={codec}: {arm['status']}; "
                        f"applied steps={arm['optimizer_steps']}; "
                        f"accepted steps={arm['accepted_optimizer_steps']}; "
                        f"routing refusals={arm['routing_refusals']}",
                    )
                    gc.collect()
                case["status"] = "PASS" if all(
                    arm["status"] == "PASS" for arm in case["arms"]
                ) else "FAIL"
        result["verdict"] = "PASS" if (
            len(result["cases"]) == len(KS)
            and all(case["status"] == "PASS" for case in result["cases"])
            and result["attached_model"]["status"] in ("PASS", "NOT_REQUESTED")
        ) else "FAIL"
    except Exception as error:
        result["error"] = _error(error)
        result["traceback"] = traceback.format_exc()
        console.print_exception()
    finally:
        nonfinite: list[str] = []
        serializable = _json_safe(result, nonfinite)
        serializable["nonfinite_metric_paths"] = nonfinite
        if nonfinite:
            serializable["verdict"] = "FAIL"
            result["verdict"] = "FAIL"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(serializable, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
    console.print(f"P3 FP8 smoke: {result['verdict']} ({args.output})")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
