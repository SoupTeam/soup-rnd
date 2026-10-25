"""One isolated CUDA arm of the two-read/expert-cache qualification.

Use the controller, ``two_read_cache.py``, rather than invoking timing workers
directly. No cloud API, download or machine-management operation lives here.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import math
import os
import platform
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any


def _common() -> Any:
    # Direct script execution and package imports both use the same module.
    from benchmarks.harness import two_read_cache_common

    return two_read_cache_common


def _cpu_copy(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {str(key): _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_cpu_copy(item) for item in value]
    return value


def _adapters(model: Any) -> dict[str, Any]:
    from soup_cli.utils.layer_stream_runtime import canonical_named_parameters

    trainable = {
        name: parameter for name, parameter in canonical_named_parameters(model)
        if parameter.requires_grad
    }
    if not trainable or any("lora_" not in name for name in trainable):
        raise ValueError("the qualification requires nonempty, adapter-only trainable parameters")
    return trainable


def _adapter_values(model: Any) -> dict[str, Any]:
    return _cpu_copy(_adapters(model))


def _numeric_delta(before: Any, after: Any) -> Any:
    """Counter deltas only; current occupancy remains in the original snapshots."""
    if isinstance(before, dict) and isinstance(after, dict):
        return {
            key: _numeric_delta(before[key], after[key])
            for key in before.keys() & after.keys()
            if isinstance(before[key], (int, float, dict))
            and not isinstance(before[key], bool)
        }
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return after - before
    return None


def _proc_fields(path: Path) -> dict[str, int]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    result = {}
    for line in lines:
        name, separator, value = line.partition(":")
        if separator and value.strip().split():
            pieces = value.strip().split()
            try:
                result[name] = int(pieces[0]) * (1024 if pieces[-1] == "kB" else 1)
            except ValueError:
                continue
    return result


def _cgroup_memory() -> dict[str, Any]:
    """Use native mount-root-aware paths, including every visible ancestor."""
    from soup_cli.utils.stream_ram_cache import _cgroup_paths

    result: dict[str, Any] = {"constraints": []}
    try:
        paths = _cgroup_paths(Path("/proc/self/cgroup"), Path("/proc/self/mountinfo"))
        for version, directory in paths:
            item: dict[str, Any] = {"version": version, "path": str(directory)}
            names = (
                ("memory.max", "memory.current", "memory.peak", "memory.swap.max",
                 "memory.swap.current", "memory.events") if version == 2
                else ("memory.limit_in_bytes", "memory.usage_in_bytes",
                      "memory.max_usage_in_bytes", "memory.memsw.limit_in_bytes",
                      "memory.memsw.usage_in_bytes", "memory.failcnt")
            )
            for name in names:
                try:
                    text = (directory / name).read_text(encoding="utf-8").strip()
                    item[name] = int(text) if text.isdecimal() else text
                except OSError:
                    item[name] = None
            result["constraints"].append(item)
    except (OSError, ValueError, IndexError) as exc:
        result["error"] = repr(exc)
    return result


def _resources() -> dict[str, Any]:
    status = _proc_fields(Path("/proc/self/status"))
    result: dict[str, Any] = {
        "rss_bytes": status.get("VmRSS"),
        "process_lifetime_peak_rss_bytes": status.get("VmHWM"),
        "process_swap_bytes": status.get("VmSwap"),
        "process_io": _proc_fields(Path("/proc/self/io")),
        "host_memory": _proc_fields(Path("/proc/meminfo")),
        "cgroup": _cgroup_memory(),
    }
    if result["rss_bytes"] is None:
        try:
            import psutil

            result["rss_bytes"] = psutil.Process().memory_info().rss
        except (ImportError, OSError):
            pass
    return result


class ResourceSampler:
    """Stage-local sampled RSS/swap; allocator peaks are read on the main thread."""

    def __init__(self) -> None:
        self.before = _resources()
        self.max_rss_bytes: int | None = None
        self.max_swap_bytes: int | None = None
        self.max_cgroup_swap_bytes: int | None = None
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        snapshot = _resources()
        for source, destination in (("rss_bytes", "max_rss_bytes"),
                                    ("process_swap_bytes", "max_swap_bytes")):
            value = snapshot[source]
            if value is not None:
                previous = getattr(self, destination)
                setattr(self, destination, max(value, previous or 0))
        group_swaps = [
            item["memory.swap.current"] for item in snapshot.get("cgroup", {}).get(
                "constraints", [],
            ) if type(item.get("memory.swap.current")) is int
        ]
        if group_swaps:
            self.max_cgroup_swap_bytes = max(
                self.max_cgroup_swap_bytes or 0, *group_swaps,
            )
        self.samples += 1

    def _run(self) -> None:
        while not self._stop.wait(.05):
            self._sample()

    def start(self) -> ResourceSampler:
        self._sample()
        self._thread.start()
        return self

    def finish(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join()
        self._sample()
        after = _resources()
        return {
            "before": self.before, "after": after,
            "sampled_peak_rss_bytes": self.max_rss_bytes,
            "sampled_peak_process_swap_bytes": self.max_swap_bytes,
            "sampled_peak_cgroup_swap_bytes": self.max_cgroup_swap_bytes,
            "samples": self.samples, "sample_interval_seconds": .05,
            "process_io_delta": _numeric_delta(self.before["process_io"], after["process_io"]),
        }


def _runtime_snapshot(runtime: Any) -> dict[str, Any]:
    result = runtime.stats()
    result["expert_reader_direct_io"] = {
        str(reader.path): getattr(reader, "direct", None) for reader in runtime.expert_readers
    }
    return result


def _cuda_peaks(torch: Any) -> dict[str, int]:
    torch.cuda.synchronize()
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }


def _diagnostic_microbatch(
    model: Any, ids: Any, accumulation: int,
) -> dict[str, Any]:
    """Only supported decoder pre-hooks and router hooks; none on experts/MoE."""
    import torch

    captured: dict[str, Any] = {
        "input_ids": _cpu_copy(ids), "layer_inputs": {},
        "router_logits": {}, "routing_weights": {},
    }
    handles = []
    counts: dict[str, int] = {}

    def keep(group: str, key: str) -> Any:
        def hook(gradient: Any) -> None:
            captured[group][key] = _cpu_copy(gradient)

        return hook

    def on_input(key: str) -> Any:
        def hook(_module: Any, args: Any) -> None:
            if args and args[0].requires_grad:
                args[0].register_hook(keep("layer_inputs", key))

        return hook

    def on_router(key: str) -> Any:
        def hook(_module: Any, _args: Any, output: Any) -> None:
            # Checkpoint replay visits a router again; record both calls rather
            # than overwrite one and accidentally hide an absent gradient.
            visit = counts.get(key, 0)
            counts[key] = visit + 1
            for group, tensor in (("router_logits", output[0]),
                                  ("routing_weights", output[1])):
                if not torch.is_tensor(tensor) or not tensor.requires_grad:
                    raise RuntimeError(f"{group} {key} lacks differentiable output")
                label = f"{key}/visit-{visit}"
                captured[group][label] = {"values": _cpu_copy(tensor)}
                if group == "routing_weights":
                    captured[group][label]["indices"] = _cpu_copy(output[2])

                def save(gradient: Any, group: str = group, label: str = label) -> None:
                    captured[group][label]["gradients"] = _cpu_copy(gradient)

                tensor.register_hook(save)

        return hook

    layers = model.get_submodule("base_model.model.model.layers")
    for index, layer in enumerate(layers):
        key = str(index)
        handles.append(layer.register_forward_pre_hook(on_input(key)))
        inner = getattr(layer, "inner", layer)
        router = getattr(getattr(inner, "mlp", None), "gate", None)
        if router is not None:
            handles.append(router.register_forward_hook(on_router(key)))
    try:
        with torch.autocast(
            ids.device.type, dtype=torch.bfloat16, enabled=ids.device.type == "cuda",
        ):
            output = model(input_ids=ids, labels=ids, use_cache=False)
        captured["loss"] = _cpu_copy(output.loss)
        captured["logits"] = _cpu_copy(output.logits)
        (output.loss / accumulation).backward()
    finally:
        for handle in handles:
            handle.remove()
    captured["gradients"] = {}
    for name, parameter in _adapters(model).items():
        if parameter.grad is None:
            raise RuntimeError(f"adapter {name} has no gradient")
        captured["gradients"][name] = _cpu_copy(parameter.grad)
    for group in ("layer_inputs", "router_logits", "routing_weights"):
        if not captured[group]:
            raise RuntimeError(f"empty {group} diagnostic capture")
    # Non-reentrant checkpointing can abandon the replay before some router
    # outputs are consumed. Keep only the original, differentiable visit whose
    # gradient exists, identically for every arm; never synthesize zero grads.
    for group in ("router_logits", "routing_weights"):
        by_layer = {}
        for key, item in captured[group].items():
            if "gradients" in item:
                layer = key.partition("/visit-")[0]
                if layer in by_layer:
                    raise RuntimeError(f"multiple differentiated router visits for {layer}")
                by_layer[layer] = item
        if not by_layer or set(by_layer) != set(counts):
            raise RuntimeError(f"{group} lacks a differentiated capture for every router")
        captured[group] = by_layer
    return captured


def _optimizer_snapshot(optimizer: Any, adapters: dict[str, Any]) -> dict[str, Any]:
    return {
        name: _cpu_copy(optimizer.state[parameter]) for name, parameter in adapters.items()
    }


def _run_accuracy(model: Any, runtime: Any, batches: Any, spec: dict[str, Any]) -> Any:
    import torch

    adapters = _adapters(model)
    optimizer = _optimizer(model, spec)
    captured: dict[str, Any] = {
        "initial_adapters": _adapter_values(model), "microbatches": [], "updates": [],
    }
    accumulation = spec["accumulation"]
    for update in range(spec["accuracy_updates"]):
        model.zero_grad(set_to_none=True)
        for micro in range(accumulation):
            device = next(iter(adapters.values())).device
            ids = batches[update * accumulation + micro].to(device)
            captured["microbatches"].append(_diagnostic_microbatch(model, ids, accumulation))
        optimizer.step()
        captured["updates"].append({
            "adapters": _adapter_values(model),
            "optimizer": _optimizer_snapshot(optimizer, adapters),
        })
    changed = [
        name for name, parameter in adapters.items()
        if not torch.equal(parameter.detach().cpu(), captured["initial_adapters"][name])
    ]
    if len(changed) != len(adapters):
        unchanged = sorted(set(adapters) - set(changed))
        raise RuntimeError(f"adapters did not actually change: {unchanged}")
    return captured


def _optimizer(model: Any, spec: dict[str, Any]) -> Any:
    import torch

    return torch.optim.AdamW(
        _adapters(model).values(), lr=spec.get("learning_rate", .002),
        weight_decay=spec.get("weight_decay", .01), foreach=False, fused=False,
    )


def _finite_training_state(model: Any, optimizer: Any, losses: list[Any]) -> None:
    """One CUDA-to-host boolean transfer, outside every measured update interval."""
    import torch

    adapters = _adapters(model)
    device = next(iter(adapters.values())).device
    checks = [loss.detach().isfinite().all().to(device) for loss in losses]
    for name, parameter in adapters.items():
        if parameter.grad is None:
            raise RuntimeError(f"adapter {name} has no gradient during timing")
        checks.extend((parameter.detach().isfinite().all(), parameter.grad.isfinite().all()))
        state = optimizer.state[parameter]
        if not state or not {"step", "exp_avg", "exp_avg_sq"}.issubset(state):
            raise RuntimeError(f"adapter {name} lacks complete optimizer state")
        checks.extend(
            value.detach().isfinite().all().to(device) for value in state.values()
            if torch.is_tensor(value)
        )
    if not checks or not bool(torch.stack(checks).all().item()):
        raise RuntimeError("non-finite microbatch loss, gradient, adapter or optimizer moment")


def _enforce_stage_limits(spec: dict[str, Any], metrics: dict[str, Any]) -> None:
    resources = metrics["resources"]
    for allocation in (metrics["cuda"], metrics.get("cuda_including_diagnostics", {})):
        for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
            if allocation.get(key, 0) > spec["device_cap_bytes"]:
                raise RuntimeError(f"{key} exceeded the common VRAM cap")
    rss = resources["sampled_peak_rss_bytes"]
    swap = resources["sampled_peak_process_swap_bytes"]
    if rss is None or swap is None:
        raise RuntimeError("process RSS/swap instrumentation is unavailable")
    if rss > spec.get("max_host_bytes", 32 * 2**30):
        raise RuntimeError("sampled RSS exceeded the declared RAM cap")
    if swap != 0:
        raise RuntimeError("process swap was observed during this model stage")
    if resources.get("sampled_peak_cgroup_swap_bytes", 0):
        raise RuntimeError("cgroup swap was observed during this model stage")


def _timed_updates(
    model: Any, runtime: Any, optimizer: Any, batches: Any, spec: dict[str, Any], updates: int,
    progress: Any = None,
) -> dict[str, Any]:
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = _runtime_snapshot(runtime)
    sampler = ResourceSampler().start()
    seconds = []
    losses = []
    actual_step_peaks = []
    diagnostic_peaks = []
    started = time.perf_counter()
    try:
        for update in range(updates):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            step_started = time.perf_counter()
            model.zero_grad(set_to_none=True)
            micro_losses = []
            for micro in range(spec["accumulation"]):
                ids = batches[(update * spec["accumulation"] + micro) % len(batches)].to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(input_ids=ids, labels=ids, use_cache=False)
                    loss = output.loss / spec["accumulation"]
                    micro_losses.append(output.loss.detach())
                loss.backward()
                del output, loss
            optimizer.step()
            torch.cuda.synchronize()
            seconds.append(time.perf_counter() - step_started)
            actual_step_peaks.append(_cuda_peaks(torch))
            # Diagnostics and their temporary allocations are outside timing.
            _finite_training_state(model, optimizer, micro_losses)
            diagnostic_peaks.append(_cuda_peaks(torch))
            value = float(micro_losses[-1].cpu())
            if not math.isfinite(value):
                raise RuntimeError("non-finite loss during timing")
            losses.append(value)
            if progress is not None:
                progress({
                    "update_seconds": list(seconds), "losses": list(losses),
                    "updates_complete": len(seconds), "complete": False,
                })
    finally:
        resources = sampler.finish()
    after = _runtime_snapshot(runtime)
    peaks = _cuda_peaks(torch)
    for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
        peaks[key] = max(item[key] for item in actual_step_peaks)
    including_diagnostics = dict(peaks)
    for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
        including_diagnostics[key] = max(item[key] for item in diagnostic_peaks)
    result = {
        "update_seconds": seconds, "losses": losses,
        "updates_complete": len(seconds), "complete": True,
        "finite_state_checks_per_update": True,
        "total_seconds": time.perf_counter() - started,
        "tokens_per_update": spec["batch_size"] * spec["seq_len"] * spec["accumulation"],
        "runtime_before": before, "runtime_after": after,
        "runtime_delta": _numeric_delta(before, after),
        "resources": resources, "cuda": peaks,
        "cuda_including_diagnostics": including_diagnostics,
    }
    _enforce_stage_limits(spec, result)
    return result


def _build(spec: dict[str, Any], arm: str) -> tuple[Any, Any, Any]:
    import torch
    from peft import LoraConfig, TaskType

    from soup_cli.utils.layer_shard import read_shard_index
    from soup_cli.utils.layer_stream_runtime import (
        build_streamed_model,
        clear_stream_substitution_cache,
        set_stream_recompute,
    )
    from soup_cli.utils.stream_expert_cache import configure_stream_expert_cache

    index_path = Path(spec["index_path"])
    if os.path.realpath(index_path) != os.path.realpath(
        os.path.join(spec["shard_dir"], "index.json")
    ):
        raise ValueError("index_path must be the fingerprinted shard_dir/index.json")
    index = read_shard_index(spec["shard_dir"], index_name=index_path.name)
    if index.quant != "nf4" or index.expert_layout != "per_expert" or not index.expert_layers:
        raise ValueError("qualification requires actual NF4 per-expert shards")
    lora = LoraConfig(
        r=spec["lora_r"], lora_alpha=spec["lora_alpha"], lora_dropout=0., bias="none",
        target_modules=spec["lora_targets"], task_type=TaskType.CAUSAL_LM,
    )
    model, runtime = build_streamed_model(
        model_id=spec["weights"], shard_dir=spec["shard_dir"], index=index,
        lora_config=lora, device="cuda:0", dtype=spec["dtype"], quant=spec["quant"],
        buffers=2, pin=True, seed=spec["seed"], trust_remote_code=False,
        tier="disk", ram_cache_bytes=0, two_read_experts=arm in {"two_read", "two_read_cache"},
    )
    try:
        if not runtime.expert_readers:
            raise ValueError("the checkpoint has no routed expert readers")
        set_stream_recompute(model, True)
        model.config.use_cache = False
        model.eval()  # Same training graph with dropout disabled in every arm.
        adapter_names = _adapters(model)
        absent_targets = [
            target for target in spec["lora_targets"]
            if not any(f".{target}.lora_" in name for name in adapter_names)
        ]
        if absent_targets:
            raise ValueError(f"requested LoRA targets produced no adapters: {absent_targets}")
        generator = torch.Generator(device="cpu").manual_seed(spec["seed"] + 1000)
        with torch.no_grad():
            for name, parameter in _adapters(model).items():
                if "lora_B" in name:
                    parameter.copy_(
                        torch.randn(parameter.shape, generator=generator, dtype=torch.float32)
                        .mul_(.02).to(device=parameter.device, dtype=parameter.dtype)
                    )
        plan = None
        if arm in {"cache", "two_read_cache"}:
            plan = configure_stream_expert_cache(
                model, runtime, host_cache_bytes=spec["host_cache_bytes"],
                device_cache_bytes=spec["device_cache_bytes"], rows=spec["batch_size"],
                seq_len=spec["seq_len"], vocab_size=model.config.vocab_size,
                override_bytes=spec["device_cap_bytes"],
            )
            if not plan.enabled:
                raise ValueError("cache arm unexpectedly admitted no cache")
        runtime.expert_fetcher.clear_cache()
        clear_stream_substitution_cache(model)
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return model, runtime, dataclasses.asdict(plan) if plan is not None else None
    except BaseException:
        runtime.close()
        raise


def _environment(torch: Any) -> dict[str, Any]:
    versions = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes", "trl"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    properties = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": versions, "torch_cuda": torch.version.cuda,
        "gpu_name": properties.name, "gpu_total_bytes": properties.total_memory,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "adapter_b_initialization": "cpu-fp32-normal(seed+1000)*0.02 cast to adapter dtype",
        "training_mode": "eval with differentiable LoRA updates; dropout disabled",
    }


def run_worker(spec_path: Path, arm: str, phase: str, out: Path) -> int:
    """Write result.json even on failure; captures are separate, uncompressed tensors."""
    import torch

    common = _common()
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.json"
    if result_path.exists() or (out / "capture.pt").exists():
        raise FileExistsError(f"refusing to overwrite existing worker results: {out}")
    result: dict[str, Any] = {
        "status": "running", "phase": phase, "arm": arm,
        "spec_sha256": common.sha256_file(spec_path), "pid": os.getpid(),
    }
    common.atomic_json(result_path, result)
    runtime = None
    sampler = None
    started = time.perf_counter()
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if arm not in common.ARMS or phase not in {"accuracy", "timing"}:
            raise ValueError("unknown arm or phase")
        if spec.get("dtype") != "bfloat16" or spec.get("quant") != "nf4":
            raise ValueError("this qualification is fixed to NF4 storage/BF16 compute")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(
            including_emulation=False,
        ):
            raise RuntimeError("a real BF16-capable CUDA GPU is required")
        torch.cuda.set_device(0)
        torch.manual_seed(spec["seed"])
        torch.cuda.manual_seed_all(spec["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        result["environment"] = _environment(torch)
        result["resources_before"] = _resources()
        batches = torch.load(spec["training_data_path"], map_location="cpu", weights_only=True)
        expected = (
            spec["accuracy_updates"] * spec["accumulation"], spec["batch_size"], spec["seq_len"],
        )
        if not torch.is_tensor(batches) or batches.dtype != torch.long:
            raise ValueError("training_data_path must hold a torch.long token tensor")
        if tuple(batches.shape) != expected:
            raise ValueError(f"frozen input shape {tuple(batches.shape)} differs from {expected}")
        sampler = ResourceSampler().start()
        setup_started = time.perf_counter()
        model, runtime, plan = _build(spec, arm)
        result["setup"] = {
            "seconds": time.perf_counter() - setup_started, "cache_plan": plan,
            "resources": sampler.finish(), "cuda": _cuda_peaks(torch),
            "runtime_after": _runtime_snapshot(runtime),
        }
        _enforce_stage_limits(spec, result["setup"])
        sampler = None
        common.atomic_json(result_path, result)
        if phase == "accuracy":
            before = _runtime_snapshot(runtime)
            sampler = ResourceSampler().start()
            torch.cuda.reset_peak_memory_stats()
            accuracy_started = time.perf_counter()
            capture = _run_accuracy(model, runtime, batches, spec)
            capture_path = out / "capture.pt"
            torch.save(capture, capture_path)
            result["accuracy"] = {
                "seconds": time.perf_counter() - accuracy_started,
                "updates": len(capture["updates"]),
                "microbatches": len(capture["microbatches"]),
                "adapters": len(capture["initial_adapters"]),
                "capture_sha256": common.sha256_file(capture_path),
                "resources": sampler.finish(), "cuda": _cuda_peaks(torch),
                "runtime_before": before, "runtime_after": _runtime_snapshot(runtime),
            }
            result["accuracy"]["runtime_delta"] = _numeric_delta(
                before, result["accuracy"]["runtime_after"],
            )
            _enforce_stage_limits(spec, result["accuracy"])
            sampler = None
        else:
            optimizer = _optimizer(model, spec)

            def progress(stage: str) -> Any:
                def save(partial: dict[str, Any]) -> None:
                    result[stage] = partial
                    common.atomic_json(result_path, result)

                return save

            result["warmup"] = _timed_updates(
                model, runtime, optimizer, batches, spec, spec["warmup_updates"],
                progress=progress("warmup"),
            )
            common.atomic_json(result_path, result)
            result["timing"] = _timed_updates(
                model, runtime, optimizer, batches, spec, spec["timed_updates"],
                progress=progress("timing"),
            )
        result["status"] = "passed"
    except BaseException as exc:
        result.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
    finally:
        if sampler is not None:
            result["interrupted_stage_resources"] = sampler.finish()
        if runtime is not None:
            try:
                runtime.close()
            except BaseException as exc:
                result["cleanup_error"] = repr(exc)
                result["status"] = "failed"
        result["total_worker_seconds"] = time.perf_counter() - started
        result["resources_after"] = _resources()
        common.atomic_json(result_path, result)
    return 0 if result["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=(
        "ordinary", "two_read", "cache", "two_read_cache",
    ))
    parser.add_argument("--phase", required=True, choices=("accuracy", "timing"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    return run_worker(args.spec, args.arm, args.phase, args.out)


if __name__ == "__main__":
    # The repo root must precede site-packages, which may also expose a
    # ``benchmarks`` package. Production installation remains the controller's job.
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from rich.console import Console

    status = main()
    Console().print("Worker results saved; status:", "passed" if status == 0 else "failed")
    raise SystemExit(status)
