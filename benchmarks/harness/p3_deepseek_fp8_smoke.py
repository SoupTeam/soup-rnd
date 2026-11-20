#!/usr/bin/env python3
"""Tiny synthetic DeepSeek NF4/BF16 training with block64 E4M3 and routing Oracle.

This is one candidate's execution smoke, not a baseline comparison, quality study,
FP16 claim, full P3 acceptance, 671B memory-fit result, or speed benchmark. Run on
Kaggle from the checkout with src on PYTHONPATH. No model download is needed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import random
import tempfile
import threading
import time
import traceback
import weakref
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Sequence

from rich.console import Console

SEED = 23
KS = (1, 8)
UPDATES = 2
SHAPE = (1, 64, 128)
HOST_BUDGET_BYTES = 128 * 1024**2


def _modules() -> tuple[Any, Any, Any]:
    prefix = f"{__package__}." if __package__ else ""
    return tuple(importlib.import_module(prefix + name) for name in (
        "b13_moe_fixture", "p3_realdata_codec", "p3_deepseek_oracle",
    ))


def _child(root: Path, name: str) -> Path:
    parent = os.path.realpath(root)
    child = os.path.realpath(root / name)
    if os.path.commonpath((parent, child)) != parent:
        raise ValueError("output child escapes the exclusive run directory")
    return Path(child)


def _json_value(value: Any) -> Any:
    """Refuse tensors and nonfinite numbers rather than serializing misleading data."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite number in smoke evidence")
        return value
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise TypeError(f"non-JSON evidence type: {type(value).__module__}.{type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_value(value), indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )


def _error(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_manifest() -> dict[str, Any]:
    fixture, codec, oracle = _modules()
    names = (
        "soup_cli.utils.l2l_step", "soup_cli.utils.layer_shard",
        "soup_cli.utils.layer_stream_runtime", "soup_cli.utils.expert_stream",
        "soup_cli.utils.expert_shard", "soup_cli.utils.host_memory_budget",
        "soup_cli.utils.peft_wiring", "soup_cli.utils.kimi_k2",
        "transformers.models.deepseek_v3.modeling_deepseek_v3",
    )
    names = (*names, "soup_cli.utils.layer_stream", "soup_cli.utils.l2l_rng",
             "soup_cli.utils.safetensors_reader")
    modules = [fixture, codec, oracle, *(importlib.import_module(name) for name in names)]
    files = {module.__name__: Path(module.__file__) for module in modules}
    files["driver"] = Path(__file__)
    native = importlib.import_module("transformers.models.deepseek_v3.modeling_deepseek_v3")
    return {
        "files": {name: {"path": os.path.realpath(path), "sha256": _sha256(path)}
                  for name, path in files.items()},
        "native_router_forward_sha256": hashlib.sha256(
            inspect.getsource(native.DeepseekV3TopkRouter.forward).encode("utf-8")
        ).hexdigest(),
        "launcher_revision_assertion": os.environ.get("SOUP_SOURCE_REVISION"),
        "identity_policy": "file hashes observed at runtime; launcher revision is not a git check",
    }


@contextmanager
def _preserve_process_state(device: str) -> Iterator[dict[str, Any]]:
    import torch

    previous_random = random.getstate()
    flags = {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }
    evidence: dict[str, Any] = {"restored": False, "previous_backend_flags": flags}
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(SEED)
            torch.manual_seed(SEED)
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            yield evidence
    finally:
        random.setstate(previous_random)
        torch.use_deterministic_algorithms(flags["deterministic"], warn_only=flags["warn_only"])
        torch.backends.cuda.matmul.allow_tf32 = flags["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = flags["cudnn_tf32"]
        torch.backends.cudnn.benchmark = flags["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = flags["cudnn_deterministic"]
        evidence["restored"] = True
        evidence["rng_scope"] = "Python random and torch CPU/all available CUDA devices"


class _MemoryObservation:
    """Sample process RSS separately from CUDA allocator counters, including teardown."""

    def __init__(self, device: str) -> None:
        import psutil

        self.device = device
        self.process = psutil.Process()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.peak_rss = 0
        self.samples = 0
        self.errors: list[dict[str, str]] = []
        self.worker: Any = None
        self.before: dict[str, Any] = {}

    def _rss(self) -> int:
        current = int(self.process.memory_info().rss)
        with self.lock:
            self.peak_rss = max(self.peak_rss, current)
            self.samples += 1
        return current

    def _observe(self) -> None:
        while not self.stop.wait(0.01):
            try:
                self._rss()
            except BaseException as error:
                self.errors.append(_error(error))
                return

    def snapshot(self) -> dict[str, Any]:
        import torch

        result: dict[str, Any] = {"rss_bytes": self._rss()}
        if self.device == "cuda":
            torch.cuda.synchronize()
            result.update(
                cuda_allocated_bytes=int(torch.cuda.memory_allocated()),
                cuda_reserved_bytes=int(torch.cuda.memory_reserved()),
                cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
            )
        return result

    def __enter__(self) -> _MemoryObservation:
        import torch

        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.before = self.snapshot()
        self.worker = threading.Thread(target=self._observe, name="deepseek-smoke-rss", daemon=True)
        self.worker.start()
        return self

    def finish(self) -> dict[str, Any]:
        self.stop.set()
        if self.worker is not None:
            self.worker.join()
        after = self.snapshot()
        return {
            "before": self.before, "after_teardown": after,
            "sampled_peak_rss_bytes": self.peak_rss, "rss_samples": self.samples,
            "sampling_interval_seconds": 0.01, "sampling_errors": self.errors,
            "scope": "fixture creation, training, evidence copies, adapter save and teardown",
            "rss_claim": "sampled process RSS, not exact peak or proof of OS RSS release",
            "cuda_claim": "allocated and reserved are separate allocator observations, not RSS",
        }


@contextmanager
def _window_resources(
    model: Any, shapes: Sequence[Sequence[int]], budget: Any, device: str,
    record: dict[str, Any],
) -> Iterator[tuple[Any, Any]]:
    import torch

    from soup_cli.utils.l2l_step import model_parts

    _, codec, oracle = _modules()
    store = probe = None
    failure = None
    cleanup_errors = []
    try:
        store = codec.make_store(
            codec.CodecPolicy("block64_e4m3", "e4m3", group_size=64), host_budget=budget,
            n_layers=len(model_parts(model)[1].layers), mb_shapes=shapes,
            dtype=torch.bfloat16, device=device, pin=device == "cuda",
        )
        # The benchmark codec declares mb_shapes; production L2LStep checks .k.
        # This is capacity metadata only. Codec storage/arithmetic stays unchanged.
        store.k = len(shapes)
        probe = oracle.DeepseekOracleProbe(model, shapes, budget)
        yield store, probe
    except BaseException as error:
        failure = error
        if device == "cuda":
            try:
                torch.cuda.synchronize()
            except BaseException as drain_error:
                cleanup_errors.append({"operation": "failure.cuda_drain", **_error(drain_error)})
        traceback.clear_frames(error.__traceback__)
        raise
    finally:
        for name, resource in (("oracle", probe), ("store", store)):
            if resource is None:
                continue
            try:
                record[name + "_before_close"] = resource.report()
            except BaseException as error:
                cleanup_errors.append({"operation": name + ".report", **_error(error)})
            try:
                resource.close()
            except BaseException as error:
                cleanup_errors.append({"operation": name + ".close", **_error(error)})
            try:
                record[name + "_after_close"] = resource.report()
                closed = record[name + "_after_close"]
                if name == "oracle" and (
                    not closed["closed"] or closed["records_remaining"]
                    or closed["pending_replays"] or closed["host_reserved_bytes"]
                ):
                    raise RuntimeError("Oracle retained live replay state after close")
                if name == "store" and (
                    not closed["cleanup"]["closed"] or closed["owned_host_storage_bytes"]
                    or closed["owned_device_storage_bytes"] or closed["charged_host_bytes"]
                ):
                    raise RuntimeError("FP8 store retained owned storage after close")
            except BaseException as error:
                cleanup_errors.append({"operation": name + ".report_after_close", **_error(error)})
        record["host_budget_after_close_bytes"] = budget.used_bytes
        record["cleanup_errors"] = cleanup_errors
        store = probe = resource = None
        if cleanup_errors or budget.used_bytes:
            message = f"window cleanup failed: {cleanup_errors}; budget={budget.used_bytes}"
            if failure is not None:
                if hasattr(failure, "add_note"):
                    failure.add_note(message)
            else:
                raise RuntimeError(message)
        failure = model = None


def _stepper(model: Any, runtime: Any, calls: Sequence[Any], probe: Any) -> Any:
    import torch

    from soup_cli.utils.l2l_step import L2LStep

    if not calls or len({id(call) for call in calls}) != len(calls):
        raise ValueError("each micro-batch needs a distinct captured LayerCall identity")

    class OracleL2LStep(L2LStep):
        def _layer(self, module: Any, hidden: Any, state: Any, call: Any) -> Any:
            layer = layer_ids[id(module)]
            microbatch = call_ids[id(call)]
            phase = "recompute" if torch.is_grad_enabled() else "forward"
            with probe.visit(layer, microbatch, phase):
                return super()._layer(module, hidden, state, call)

    step = OracleL2LStep(model, runtime, calls[0])
    layer_ids = {id(layer): index for index, layer in enumerate(step.layers)}
    call_ids = {id(call): index for index, call in enumerate(calls)}
    return step


def _adapter_digests(parameters: dict[str, Any]) -> tuple[str, dict[str, str]]:
    import torch

    combined = hashlib.sha256()
    tensors = {}
    for name, parameter in sorted(parameters.items()):
        value = parameter.detach().cpu().contiguous()
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        tensors[name] = digest
        combined.update(json.dumps([name, list(value.shape), str(value.dtype), digest]).encode())
    return combined.hexdigest(), tensors


def _apply_update(
    parameters: dict[str, Any], optimizer: Any, probe: Any, losses: Sequence[Any],
    record: dict[str, Any],
) -> None:
    """No optimizer write is allowed before losses, gradients and Oracle pass."""
    import torch

    record["optimizer_applied"] = False
    values = [float(loss.detach()) for loss in losses]
    record["losses"] = [value if math.isfinite(value) else None for value in values]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("missing or nonfinite training loss")
    if not parameters:
        raise ValueError("no trainable attention adapters")
    missing, nonfinite, zero = [], [], []
    squared_norm = 0.0
    for name, parameter in parameters.items():
        gradient = parameter.grad
        if gradient is None:
            missing.append(name)
        elif not bool(torch.isfinite(gradient).all()):
            nonfinite.append(name)
        elif not bool(gradient.ne(0).any()):
            zero.append(name)
        else:
            squared_norm += float(gradient.detach().double().square().sum())
    record["gradients"] = {
        "trainable_tensors": len(parameters), "missing": missing, "nonfinite": nonfinite,
        "zero": zero, "nonzero_tensors": len(parameters) - len(missing + nonfinite + zero),
        "global_l2": math.sqrt(squared_norm) if math.isfinite(squared_norm) else None,
    }
    if missing or nonfinite or zero or not math.isfinite(squared_norm):
        raise ValueError("every attention adapter needs a finite nonzero gradient")
    if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters.values()):
        raise ValueError("nonfinite adapter before optimizer update")
    record["oracle_before_update"] = probe.report()
    probe.require_complete()
    before, before_tensors = _adapter_digests(parameters)
    record["before_sha256"] = before
    optimizer.step()
    record["optimizer_applied"] = True
    if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters.values()):
        raise ValueError("nonfinite adapter after optimizer update")
    after, after_tensors = _adapter_digests(parameters)
    record["after_sha256"] = after
    record["changed_adapter_tensors"] = sum(
        before_tensors[name] != after_tensors[name] for name in parameters
    )
    record["adapters_finite_after_update"] = True
    if before == after or not record["changed_adapter_tensors"]:
        raise ValueError("positive-LR optimizer did not change any adapter")


class _ExecutionProbe:
    """Count native decoder/MoE/attention and production streamed-expert calls."""

    def __init__(self, model: Any) -> None:
        import torch
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
            DeepseekV3Attention,
            DeepseekV3DecoderLayer,
            DeepseekV3MoE,
        )

        from soup_cli.utils.expert_stream import streamed_experts_class

        self.counts: Counter[str] = Counter()
        self.handles: list[Any] = []
        self.input_dtypes: dict[str, set[str]] = {}
        classes = (
            (DeepseekV3DecoderLayer, "native_decoder"), (DeepseekV3Attention, "native_attention"),
            (DeepseekV3MoE, "native_moe"), (streamed_experts_class(), "streamed_experts"),
        )

        def hook_for(kind: str) -> Any:
            def observe(_module: Any, _args: Any, kwargs: Any) -> None:
                phase = "recompute" if torch.is_grad_enabled() else "forward"
                self.counts[f"{kind}_{phase}"] += 1
                hidden = _args[0] if _args else kwargs["hidden_states"]
                self.input_dtypes.setdefault(kind, set()).add(str(hidden.dtype))
                if hidden.dtype != torch.bfloat16:
                    raise RuntimeError(f"{kind} did not execute with BF16 hidden states")
            return observe

        try:
            for module in model.modules():
                for cls, kind in classes:
                    if isinstance(module, cls):
                        self.handles.append(module.register_forward_pre_hook(
                            hook_for(kind), with_kwargs=True,
                        ))
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _execution_required(counts: dict[str, int], k: int) -> None:
    for phase in ("forward", "recompute"):
        for kind, layers in (
            ("native_decoder", 3), ("native_attention", 3),
            ("native_moe", 2), ("streamed_experts", 2),
        ):
            if counts.get(f"{kind}_{phase}", 0) != layers * k:
                raise RuntimeError(f"missing {kind} {phase} execution for k={k}")


def _build_candidate(root: Path, device: str, record: dict[str, Any]) -> tuple[Any, Any, Any]:
    import torch

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream import stream_arch_of
    from soup_cli.utils.layer_stream_runtime import (
        build_meta_skeleton,
        build_streamed_model,
        quantised_layer_suffixes,
    )

    fixture, _, _ = _modules()
    config = fixture.moe_config(
        hidden_size=SHAPE[-1], intermediate_size=256, moe_intermediate_size=64,
        q_lora_rank=32, kv_lora_rank=32, qk_rope_head_dim=16,
        qk_nope_head_dim=16, v_head_dim=16, max_position_embeddings=128,
        attention_dropout=0.0, norm_topk_prob=True, routed_scaling_factor=1.25,
    )
    # HF config dictionaries contain integer label keys; use its JSON representation.
    record["model_config"] = json.loads(config.to_json_string(use_diff=False))
    weights = fixture.moe_weights_dir(root, config, seed=SEED)
    skeleton = build_meta_skeleton(
        str(weights), dtype="bfloat16", quant="nf4", double_quant=False,
    )
    suffixes = quantised_layer_suffixes(skeleton)
    del skeleton
    shards = root / "shards"
    index = shard_checkpoint(
        str(weights), str(shards), dtype="bfloat16", arch=stream_arch_of(config),
        expert_layout="per_expert", quant="nf4", quant_suffixes=suffixes,
        quant_device=device, double_quant=False,
    )
    specs = (*index.quant_specs.values(), *index.expert_quant_specs.values())
    if (
        index.expert_layout != "per_expert" or index.dtype != "bfloat16" or index.double_quant
        or not index.quant_specs or not index.expert_quant_specs
        or any(spec.quant_type != "nf4" or spec.blocksize != 64 or spec.nested for spec in specs)
    ):
        raise RuntimeError("candidate requires per-expert NF4 block64 without double quantization")
    record["quantization"] = {
        "weight_format": "nf4", "double_quant": index.double_quant,
        "expert_layout": index.expert_layout, "shard_dtype": index.dtype,
        "quant_device": index.quant_device,
        "projection_specs": {name: spec.to_json() for name, spec in index.quant_specs.items()},
        "expert_specs": {name: spec.to_json() for name, spec in index.expert_quant_specs.items()},
        "index_sha256": _sha256(shards / "index.json"),
        "synthetic_checkpoint_sha256": _sha256(weights / "model.safetensors"),
        "dense_fallback": False,
    }
    model, runtime = build_streamed_model(
        model_id=str(weights), shard_dir=str(shards), index=index,
        lora_config=fixture.moe_lora_config(targets=fixture.TARGETS),
        device=device, dtype="bfloat16", buffers=2, pin=False, seed=SEED,
        quant="nf4", double_quant=False,
    )
    try:
        parameters = fixture.trainables(model)
        if not parameters or any(
            "lora_" not in name or ".self_attn." not in name or parameter.is_meta
            for name, parameter in parameters.items()
        ):
            raise RuntimeError("only the five P3 attention LoRA targets may train")
        observed_targets = {
            target for target in fixture.TARGETS
            if any(f".{target}.lora_" in name for name in parameters)
        }
        if observed_targets != set(fixture.TARGETS) or len(parameters) != 3 * 5 * 2:
            raise RuntimeError("missing attention LoRA targets or unexpected adapter parameters")
        for parameter in parameters.values():
            parameter.data = parameter.data.to(dtype=torch.float32)
        fixture.randomize_adapters(model, seed=SEED)
        if runtime.expert_pool is None or runtime.expert_pool.dtype != "bfloat16":
            raise RuntimeError("no BF16 per-expert streaming pool installed")
        model.train()
        record["precision"] = {
            "compute_dtype": "bfloat16", "autocast": False,
            "base_parameter_dtype_counts": dict(Counter(
                str(parameter.dtype) for parameter in model.parameters()
                if not parameter.requires_grad
            )),
            "trainable_dtype_counts": dict(Counter(
                str(value.dtype) for value in parameters.values()
            )),
            "trainable_tensors": len(parameters),
            "trainable_elements": sum(value.numel() for value in parameters.values()),
            "targets": list(fixture.TARGETS),
            "expert_pool_dtype": runtime.expert_pool.dtype,
            "pool_tensor_dtype_counts": dict(Counter(
                str(value.dtype) for slot in runtime.pool.buffers for value in slot.values()
            )),
            "expert_pool_tensor_dtype_counts": dict(Counter(
                str(value.dtype) for value in runtime.expert_pool._buffers.values()
            )),
            "attention_implementation": getattr(model.config, "_attn_implementation", None),
            "base_parameter_device_counts": dict(Counter(
                str(value.device) for value in model.parameters() if not value.requires_grad
            )),
            "router_weight_and_bias_dtypes": {
                name: str(value.dtype) for name, value in (
                    *model.named_parameters(), *model.named_buffers(),
                ) if ".mlp.gate." in name
            },
        }
        return model, runtime, config
    except BaseException:
        runtime.close()
        raise


def _run_candidate(
    output: Path, device: str, learning_rate: float, report: dict[str, Any],
) -> None:
    import torch

    from soup_cli.utils.host_memory_budget import HostMemoryBudget
    from soup_cli.utils.l2l_step import capture_layer_call
    from soup_cli.utils.layer_stream_runtime import assert_streamed_adapter_saved

    fixture, _, _ = _modules()
    budget = HostMemoryBudget(HOST_BUDGET_BYTES)
    model = runtime = execution = optimizer = step = store = probe = None
    parameters: dict[str, Any] = {}
    initial: dict[str, Any] = {}
    ids: list[Any] = []
    calls: list[Any] = []
    losses: Any = None
    model_ref = runtime_ref = None
    failure = None
    cleanup_errors = []
    temporary = tempfile.TemporaryDirectory(prefix="p3-deepseek-smoke-")
    try:
        try:
            start = time.perf_counter()
            model, runtime, config = _build_candidate(Path(temporary.name), device, report)
            report["fixture_wall_seconds"] = time.perf_counter() - start
            model_ref, runtime_ref = weakref.ref(model), weakref.ref(runtime)
            _write_json(_child(output, "model_config.json"), report["model_config"])
            _write_json(_child(output, "quantization_manifest.json"), report["quantization"])
            parameters = fixture.trainables(model)
            initial = {name: value.detach().cpu().clone() for name, value in parameters.items()}
            execution = _ExecutionProbe(model)
            report["runtime_before_training"] = runtime.stats()
            for k in KS:
                with torch.no_grad():
                    for name, parameter in parameters.items():
                        parameter.copy_(initial[name])
                parameter = None
                optimizer = torch.optim.SGD(parameters.values(), lr=learning_rate)
                generator = torch.Generator().manual_seed(SEED)
                point: dict[str, Any] = {
                    "k": k, "status": "SMOKE_FAILED", "steps": [],
                    "initial_adapter_sha256": _adapter_digests(parameters)[0],
                    "optimizer": {"class": "torch.optim.SGD", "lr": learning_rate,
                                  "momentum": 0.0, "fresh_for_k": True},
                }
                report["points"].append(point)
                for update in range(UPDATES):
                    record: dict[str, Any] = {
                        "update": update + 1, "status": "SMOKE_FAILED", "optimizer_applied": False,
                    }
                    point["steps"].append(record)
                    started = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    ids = [torch.randint(0, config.vocab_size, SHAPE[:2], generator=generator).to(
                        device,
                    ) for _ in range(k)]
                    record["input_sha256"] = [hashlib.sha256(
                        value.cpu().numpy().tobytes()
                    ).hexdigest() for value in ids]
                    calls = [capture_layer_call(model, value, draws=False) for value in ids]
                    before_counts = dict(execution.counts)
                    with _window_resources(model, [SHAPE] * k, budget, device, record) as resources:
                        store, probe = resources
                        step = _stepper(model, runtime, calls, probe)
                        losses = step.run(ids, store, calls=calls)
                        record["runtime_load_deltas"] = asdict(step.counters)
                        for counter in (
                            "layer_loads", "large_loads",
                            "expert_fetches", "expert_bytes_read",
                        ):
                            if record["runtime_load_deltas"][counter] <= 0:
                                raise RuntimeError(f"no production {counter} in training window")
                        record["nf4_restores_total_observed"] = runtime.stats()["layer_restores"]
                        if record["nf4_restores_total_observed"] <= 0:
                            raise RuntimeError(
                                "no NF4 projection restore observed since construction"
                            )
                        record["execution_counts"] = {
                            name: count - before_counts.get(name, 0)
                            for name, count in execution.counts.items()
                        }
                        _execution_required(record["execution_counts"], k)
                        record["execution_input_dtypes"] = {
                            name: sorted(values) for name, values in execution.input_dtypes.items()
                        }
                        storage = store.report()
                        record["store_execution"] = storage
                        if storage["put_calls"] != 3 * k or storage["get_calls"] != 3 * k:
                            raise RuntimeError("not every decoder boundary used the FP8 store")
                        _apply_update(parameters, optimizer, probe, losses, record)
                    resources = step = store = probe = losses = None
                    ids.clear()
                    calls.clear()
                    optimizer.zero_grad(set_to_none=True)
                    record["wall_seconds"] = time.perf_counter() - started
                    record["status"] = "SMOKE_COMPLETE"
                    point["completed_optimizer_updates"] = sum(
                        item["optimizer_applied"] for item in point["steps"]
                    )
                    _write_json(_child(output, "result.json"), report)
                point["status"] = "SMOKE_COMPLETE"
                optimizer.state.clear()
                optimizer = None
            adapter_dir = _child(output, "final_adapter")
            model.save_pretrained(adapter_dir, safe_serialization=True, save_embedding_layers=False)
            assert_streamed_adapter_saved(model, str(adapter_dir))
            report["final_adapter"] = {
                "point_k": KS[-1], "completed_updates_at_point": UPDATES,
                "sha256": _adapter_digests(parameters)[0],
                "files": {path.name: _sha256(path) for path in sorted(adapter_dir.iterdir())
                          if path.is_file()},
                "save_guard": "assert_streamed_adapter_saved passed",
            }
            report["runtime_after_training"] = runtime.stats()
            if report["runtime_after_training"]["layer_restores"] <= 0:
                raise RuntimeError("candidate never restored NF4 projection weights")
        except BaseException as error:
            failure = error
            if device == "cuda":
                try:
                    torch.cuda.synchronize()
                except BaseException as drain_error:
                    cleanup_errors.append({
                        "operation": "failure.cuda_drain", **_error(drain_error),
                    })
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            # Finish CUDA work before closing pools or releasing live references.
            if device == "cuda":
                try:
                    torch.cuda.synchronize()
                except BaseException as error:
                    cleanup_errors.append({"operation": "cuda.synchronize", **_error(error)})
            if execution is not None:
                try:
                    execution.close()
                    report["execution_hooks_remaining"] = len(execution.handles)
                except BaseException as error:
                    cleanup_errors.append({"operation": "execution.close", **_error(error)})
            if optimizer is not None:
                try:
                    optimizer.zero_grad(set_to_none=True)
                    optimizer.state.clear()
                except BaseException as error:
                    cleanup_errors.append({"operation": "optimizer.clear", **_error(error)})
            if runtime is not None:
                try:
                    runtime.close()
                    report["runtime_cleanup"] = {
                        "prime_hook_removed": runtime.hook is None,
                        "expert_pool_closed": runtime.expert_pool.stats()["closed"],
                        "expert_reader_handles_closed": all(
                            reader._handle is None for reader in runtime.expert_readers
                        ),
                        "layer_buffers_released_by": (
                            "dropping model/runtime references after close"
                        ),
                    }
                    if not all(report["runtime_cleanup"][key] for key in (
                        "prime_hook_removed", "expert_pool_closed", "expert_reader_handles_closed",
                    )):
                        raise RuntimeError("runtime close left a hook, expert pool or reader open")
                except BaseException as error:
                    cleanup_errors.append({"operation": "runtime.close", **_error(error)})
            parameters.clear()
            initial.clear()
            ids.clear()
            calls.clear()
            model = runtime = execution = optimizer = step = store = probe = losses = None
            resources = parameter = None
            gc.collect()
            report["teardown"] = {
                "host_budget_used_bytes": budget.used_bytes,
                "model_collected": model_ref() is None if model_ref is not None else None,
                "runtime_collected": runtime_ref() is None if runtime_ref is not None else None,
                "cleanup_errors": cleanup_errors,
            }
            if runtime_ref is not None and not report["teardown"]["runtime_collected"]:
                cleanup_errors.append({"operation": "reference_release",
                                       "type": "RuntimeError", "message": "runtime still live"})
            if model_ref is not None and not report["teardown"]["model_collected"]:
                cleanup_errors.append({"operation": "reference_release",
                                       "type": "RuntimeError", "message": "model still live"})
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except BaseException as error:
                    cleanup_errors.append({"operation": "cuda.empty_cache", **_error(error)})
            if cleanup_errors or budget.used_bytes:
                message = f"candidate cleanup failed: {cleanup_errors}; budget={budget.used_bytes}"
                if failure is not None:
                    if hasattr(failure, "add_note"):
                        failure.add_note(message)
                else:
                    raise RuntimeError(message)
            failure = None
    finally:
        try:
            temporary.cleanup()
            report.setdefault("teardown", {})["temporary_checkpoint_and_shards_removed"] = (
                not Path(temporary.name).exists()
            )
            if not report["teardown"]["temporary_checkpoint_and_shards_removed"]:
                raise RuntimeError("temporary checkpoint/shard directory remains after cleanup")
        except BaseException as error:
            report.setdefault("teardown", {})["temporary_cleanup_error"] = _error(error)
            raise


def run_smoke(
    output: Path, *, device: str = "cuda", learning_rate: float = 0.05,
) -> dict[str, Any]:
    """Create exclusive artifacts; preserve partial evidence and propagate failures."""
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if device not in ("cuda", "cpu"):
        raise ValueError("device must be cuda or cpu; no precision fallback is supported")
    output = Path(os.path.realpath(output))
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema": "p3_deepseek_single_candidate_smoke_v1", "status": "SMOKE_FAILED",
        "scope": "synthetic tiny native DeepSeek single-candidate code execution only",
        "not_claimed": ["quality", "noninferiority", "production acceptance", "FP16 support",
                        "671B fit", "speedup", "natural routing stability"],
        "candidate": "NF4 noDQ / BF16 compute / FP32 attention LoRA / block64 E4M3 + Oracle",
        "seed": SEED, "batch_size": SHAPE[0], "sequence_length": SHAPE[1],
        "hidden_size": SHAPE[2], "ks": list(KS), "updates_per_k": UPDATES,
        "training_arms": 1, "resident_twin": False,
        "forced_routing": True, "natural_routing_measured": False,
        "synthetic": {"weights": True, "tokens": True, "seed": SEED,
                      "checkpoint_builder": "b13_moe_fixture.moe_weights_dir",
                      "source_model": "temporary CPU checkpoint generation, never a training arm"},
        "input_policy": "synthetic causal-LM token IDs; labels equal IDs, all shifted tokens count",
        "seed_policy": {
            "checkpoint": SEED, "adapter_materialization": SEED, "adapter_randomize": SEED,
            "token_generator_each_k": SEED,
            "fixture_note": "unchanged helper derives checkpoint perturbation generator as seed+1",
        },
        "device": device, "points": [],
        "capacity_seam": "store.k = len(mb_shapes), metadata only on unchanged codec instance",
        "routing_guard": "unchanged production L2LStep RoutingGuard remains enabled",
    }
    started = time.perf_counter()
    memory = None
    prior_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import psutil
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no CPU/dense fallback")
        host_available = int(psutil.virtual_memory().available)
        if host_available < 1024**3:
            raise MemoryError("tiny synthetic fixture requires 1 GiB available host admission")
        device_available = int(torch.cuda.mem_get_info()[0]) if device == "cuda" else None
        if device_available is not None and device_available < 512 * 1024**2:
            raise MemoryError("tiny synthetic fixture requires 512 MiB free CUDA admission")
        report["admission"] = {
            "host_available_bytes": host_available, "host_required_bytes": 1024**3,
            "cuda_available_bytes": device_available, "cuda_required_bytes": 512 * 1024**2,
            "store_oracle_host_budget_bytes": HOST_BUDGET_BYTES,
            "meaning": "fixed tiny-fixture admission, not a whole-process memory bound",
        }
        report["environment"] = {
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in (
                "torch", "transformers", "peft", "bitsandbytes", "accelerate", "safetensors",
                "psutil", "rich",
            )},
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name() if device == "cuda" else None,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        }
        memory = _MemoryObservation(device).__enter__()
        with _preserve_process_state(device) as state:
            report["process_state"] = state
            report["source"] = _source_manifest()
            _write_json(_child(output, "source_manifest.json"), report["source"])
            _run_candidate(output, device, learning_rate, report)
        teardown = report["teardown"]
        if teardown["model_collected"] is not True or teardown["runtime_collected"] is not True:
            raise RuntimeError("candidate model/runtime references remain after teardown")
        if len(report["points"]) != 2 or any(
            point.get("completed_optimizer_updates") != UPDATES
            or point["status"] != "SMOKE_COMPLETE" for point in report["points"]
        ):
            raise RuntimeError("not all four requested optimizer updates completed")
        report["status"] = "SMOKE_COMPLETE"
    except BaseException as error:
        report["status"] = "SMOKE_FAILED"
        report["error"] = _error(error)
        report["error"]["traceback"] = "".join(traceback.format_exception(
            type(error), error, error.__traceback__,
        ))
        traceback.clear_frames(error.__traceback__)
        raise
    finally:
        if prior_workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = prior_workspace
        report["cublas_environment_restored"] = os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG",
        ) == prior_workspace
        if memory is not None:
            try:
                report["memory"] = memory.finish()
                if report["memory"]["sampling_errors"]:
                    raise RuntimeError("RSS sampling failed; inspect result.json")
            except BaseException as error:
                report["status"] = "SMOKE_FAILED"
                report["memory_error"] = _error(error)
                if "error" not in report:
                    report["wall_seconds"] = time.perf_counter() - started
                    _write_json(_child(output, "result.json"), report)
                    raise
        report["wall_seconds"] = time.perf_counter() - started
        _write_json(_child(output, "result.json"), report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--output", type=Path, required=True, help="new, exclusive output directory",
    )
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args(argv)
    console = Console(width=100)
    try:
        result = run_smoke(args.output, device=args.device, learning_rate=args.learning_rate)
    except Exception as error:
        console.print(f"SMOKE_FAILED: {type(error).__name__}: {error}")
        console.print_exception(show_locals=False)
        return 1
    console.print(f"{result['status']}: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
