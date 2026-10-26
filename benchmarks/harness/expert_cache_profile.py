"""Bounded diagnostic CPU/CUDA profile of an already qualified expert-streaming arm.

This never rents hardware or downloads weights. It records only the first
microbatch of one accumulated update; its instrumented durations are not an
unprofiled throughput comparison. Large traces are retained losslessly as gzip.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import gzip
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAX_TRACE_BYTES = 128 * 2**20


def _worker() -> Any:
    from benchmarks.harness import two_read_cache_worker

    return two_read_cache_worker


def _hash_tensors(values: dict[str, Any]) -> dict[str, Any]:
    """Canonical tensor hashes without serializing a second model-sized capture."""
    import torch

    rows = {}
    for name, value in sorted(values.items()):
        if not torch.is_tensor(value):
            raise ValueError(f"expected a tensor for {name}")
        tensor = value.detach().to("cpu", copy=True).contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        rows[name] = {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
                      "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {"tensors": rows, "canonical_index_sha256": hashlib.sha256(encoded).hexdigest()}


def _training_hashes(model: Any, optimizer: Any) -> dict[str, Any]:
    worker = _worker()
    adapters = worker._adapters(model)
    values = {"adapter/" + name: value for name, value in adapters.items()}
    for name, value in adapters.items():
        if value.grad is not None:
            values["gradient/" + name] = value.grad
        for key, state in optimizer.state.get(value, {}).items():
            values[f"adamw/{name}/{key}"] = state
    return _hash_tensors(values)


@contextlib.contextmanager
def operation_markers(
    record_function: Any, targets: tuple[tuple[Any, str, str], ...] | None = None,
) -> Iterator[None]:
    """Transparent reversible wrappers; no expert/MoE/decoder forward hooks."""
    if targets is None:
        import bitsandbytes.functional as bnb_functional

        from soup_cli.utils import expert_stream, layer_stream_runtime

        targets = (
            (expert_stream.ExpertFetcher, "fetch", "soup.expert.fetch"),
            (expert_stream.ExpertShardReader, "read_host", "soup.expert.storage_read"),
            (expert_stream.ExpertShardReader, "validate_identity", "soup.expert.file_identity"),
            (layer_stream_runtime, "rebuild_quant_state", "soup.nf4.quant_state"),
            (bnb_functional, "dequantize_4bit", "soup.nf4.dequantize_4bit"),
            (bnb_functional, "dequantize_blockwise", "soup.nf4.dequantize_blockwise"),
        )
    with contextlib.ExitStack() as stack:
        for owner, name, label in targets:
            original = getattr(owner, name)

            def wrapped(*args: Any, _call: Any = original, _label: str = label,
                        **kwargs: Any) -> Any:
                with record_function(_label):
                    return _call(*args, **kwargs)

            stack.enter_context(patch.object(owner, name, functools.wraps(original)(wrapped)))
        yield


def profiler_summary(profiler: Any) -> dict[str, Any]:
    """Keep every aggregate, including scalar syncs and nested NF4 operations."""
    rows = []
    fields = ("cpu_time_total", "self_cpu_time_total", "device_time_total",
              "self_device_time_total")
    for event in profiler.key_averages():
        row = {"name": event.key, "count": event.count,
               "device_type": str(event.device_type)}
        for name in fields:
            value = float(getattr(event, name, 0.))
            if not math.isfinite(value):
                raise ValueError(f"non-finite profiler metric: {event.key}/{name}")
            row[name + "_us"] = value
        rows.append(row)
    events = profiler.events()
    cuda = [event for event in events if str(event.device_type).lower().endswith("cuda")]
    copies = [event for event in cuda if "memcpy" in event.name.lower()]
    h2d = [event for event in copies if any(
        tag in event.name.lower() for tag in ("htod", "h2d", "host to device")
    )]
    kernels = [event for event in cuda if not any(
        tag in event.name.lower() for tag in ("memcpy", "memset")
    )]
    return {"key_averages": sorted(rows, key=lambda row: row["name"]),
            "event_count": len(events), "cuda_event_count": len(cuda),
            "cuda_kernel_event_count": len(kernels), "cuda_copy_event_count": len(copies),
            "cuda_h2d_event_count": len(h2d),
            "time_units": "microseconds; inclusive/nested and concurrent totals are not additive",
            "timing_scope": "diagnostic profiler overhead; not an unprofiled speed verdict"}


def require_cuda_profile(summary: dict[str, Any]) -> None:
    if (summary["event_count"] <= summary["cuda_event_count"] or
            summary["cuda_kernel_event_count"] < 1 or summary["cuda_h2d_event_count"] < 1):
        raise RuntimeError("CPU events, actual CUDA kernels and CUDA H2D events are required")


def retain_trace(profiler: Any, out: Path, maximum_bytes: int) -> dict[str, Any]:
    """Lossless gzip plus decoded SHA proof; refuse, never silently truncate."""
    from benchmarks.harness.two_read_cache_common import atomic_json, sha256_file

    raw, encoded = out / "trace.json", out / "trace.json.gz"
    if raw.exists() or encoded.exists():
        raise FileExistsError("refusing to overwrite profiler trace")
    started = time.perf_counter()
    profiler.export_chrome_trace(str(raw))
    raw_size, raw_sha = raw.stat().st_size, sha256_file(raw)
    with raw.open("rb") as source, encoded.open("xb") as destination:
        with gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0) as compressor:
            for block in iter(lambda: source.read(2**20), b""):
                compressor.write(block)
    decoded, decoded_size = hashlib.sha256(), 0
    with gzip.open(encoded, "rb") as source:
        for block in iter(lambda: source.read(2**20), b""):
            decoded.update(block)
            decoded_size += len(block)
    if decoded.hexdigest() != raw_sha or decoded_size != raw_size:
        raise RuntimeError("lossless profiler trace verification failed")
    proof = {"path": encoded.name, "bytes": encoded.stat().st_size,
             "sha256": sha256_file(encoded), "decoded_bytes": raw_size,
             "decoded_sha256": raw_sha, "decoded_verified": True,
             "maximum_encoded_bytes": maximum_bytes,
             "within_bound": encoded.stat().st_size <= maximum_bytes,
             "export_compression_verification_seconds": time.perf_counter() - started}
    atomic_json(out / "trace-retention.json", proof)
    if not proof["within_bound"]:
        raise RuntimeError("compressed profiler trace exceeds its fixed size bound; raw retained")
    raw.unlink()  # All original bytes remain in the verified gzip member.
    return proof


def accumulated_profile_update(
    model: Any, optimizer: Any, batches: Any, spec: dict[str, Any], *, device: str,
    profiler: Any, record_function: Any,
) -> dict[str, Any]:
    """One real update; only its first complete forward/backward is profiled."""
    import torch

    def synchronize() -> None:
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    synchronize()
    started = time.perf_counter()
    model.zero_grad(set_to_none=True)
    losses = []
    for micro in range(spec["accumulation"]):
        context = profiler if micro == 0 else contextlib.nullcontext()
        markers = operation_markers(record_function) if micro == 0 else contextlib.nullcontext()
        with context, markers:
            with record_function("soup.profile.microbatch.input"):
                ids = batches[micro % len(batches)].to(device)
            autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                        if device.startswith("cuda") else contextlib.nullcontext())
            with record_function("soup.profile.microbatch.forward"), autocast:
                output = model(input_ids=ids, labels=ids, use_cache=False)
                loss = output.loss / spec["accumulation"]
                losses.append(output.loss.detach())
            with record_function("soup.profile.microbatch.backward"):
                loss.backward()
            del output, loss
            synchronize()
    optimizer.step()
    synchronize()
    elapsed = time.perf_counter() - started
    _worker()._finite_training_state(model, optimizer, losses)
    return {"losses": [float(loss.cpu()) for loss in losses],
            "accumulation": spec["accumulation"], "optimizer_updates": 1,
            "profiled_microbatches": [0], "unprofiled_microbatches": list(
                range(1, spec["accumulation"])),
            "whole_update_including_profiler_processing_seconds": elapsed,
            "finite_training_state": True}


def run_profile(spec_path: Path, arm: str, out: Path, *, warmup_updates: int = 2,
                max_trace_bytes: int = MAX_TRACE_BYTES) -> int:
    from benchmarks.harness.two_read_cache_common import atomic_json, sha256_file

    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("profile output must be a new or empty directory")
    result_path = out / "profile-result.json"
    result: dict[str, Any] = {"status": "running", "phase": "diagnostic_profile", "arm": arm,
                              "spec_sha256": sha256_file(spec_path), "pid": os.getpid(),
                              "unprofiled_speed_verdict": "NO VERDICT"}
    atomic_json(result_path, result)
    runtime, sampler = None, None
    started = time.perf_counter()
    worker = _worker()
    try:
        if arm not in ("two_read", "two_read_cache"):
            raise ValueError("diagnostics accept only two_read and two_read_cache")
        if warmup_updates != 2 or not 1 <= max_trace_bytes <= MAX_TRACE_BYTES:
            raise ValueError("two warmups and a trace limit at most 128 MiB are required")
        import torch

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(
            including_emulation=False,
        ):
            raise RuntimeError("native BF16 CUDA is required for profiling")
        from benchmarks.harness.two_read_cache import host_boundary, verify_spec

        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        verify_spec(spec, content=False)
        result["native_boundary_before"] = host_boundary(spec["max_host_bytes"])
        torch.cuda.set_device(0)
        torch.manual_seed(spec["seed"])
        torch.cuda.manual_seed_all(spec["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        result["environment"] = worker._environment(torch)
        result["resources_before"] = worker._resources()
        batches = torch.load(spec["training_data_path"], map_location="cpu", weights_only=True)
        expected = (spec["accuracy_updates"] * spec["accumulation"], spec["batch_size"],
                    spec["seq_len"])
        if not torch.is_tensor(batches) or batches.dtype != torch.long or tuple(
            batches.shape,
        ) != expected:
            raise ValueError("frozen profile token input shape/dtype differs from the spec")
        result["inputs"] = _hash_tensors({"input_ids": batches})
        sampler = worker.ResourceSampler().start()
        setup_started = time.perf_counter()
        model, runtime, plan = worker._build(spec, arm)
        result["setup"] = {"seconds": time.perf_counter() - setup_started, "cache_plan": plan,
                           "resources": sampler.finish(), "cuda": worker._cuda_peaks(torch),
                           "runtime_after": worker._runtime_snapshot(runtime)}
        sampler = None
        worker._enforce_stage_limits(spec, result["setup"])
        optimizer = worker._optimizer(model, spec)
        result["initial_adapters"] = _hash_tensors(worker._adapters(model))
        result["warmup"] = worker._timed_updates(
            model, runtime, optimizer, batches, spec, warmup_updates,
        )
        result["unprofiled_context"] = worker._timed_updates(
            model, runtime, optimizer, batches, spec, 1,
        )
        result["state_before_profile"] = _training_hashes(model, optimizer)
        result["scope"] = {
            "profiled_optimizer_updates": 1, "profiled_microbatches": [0],
            "microbatch_tokens": spec["batch_size"] * spec["seq_len"],
            "record_shapes": False, "profile_memory": False, "with_stack": False,
            "with_flops": False, "activities": ["CPU", "CUDA"],
            "included": "first microbatch all-layer forward/backward, input H2D and final sync",
            "excluded": "setup, warmup, second microbatch, AdamW step, finite checks and export",
            "representativeness": "one frozen-input microbatch; no full-step time attribution",
        }
        atomic_json(result_path, result)
        torch.cuda.reset_peak_memory_stats()
        before = worker._runtime_snapshot(runtime)
        sampler = worker.ResourceSampler().start()
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=False, with_stack=False, with_flops=False,
        )
        update = accumulated_profile_update(
            model, optimizer, batches, spec, device="cuda", profiler=profiler,
            record_function=torch.profiler.record_function,
        )
        after = worker._runtime_snapshot(runtime)
        result["profile_update"] = {**update, "resources": sampler.finish(),
                                    "cuda": worker._cuda_peaks(torch),
                                    "runtime_before": before, "runtime_after": after,
                                    "runtime_delta": worker._numeric_delta(before, after)}
        sampler = None
        worker._enforce_stage_limits(spec, result["profile_update"])
        sampler = worker.ResourceSampler().start()
        postprocess_started = time.perf_counter()
        summary = profiler_summary(profiler)
        atomic_json(out / "key-averages.json", summary)
        result["profile_event_counts"] = {
            key: value for key, value in summary.items() if key.endswith("count")
        }
        require_cuda_profile(summary)
        atomic_json(result_path, result)
        result["trace"] = retain_trace(profiler, out, max_trace_bytes)
        result["state_after_profile"] = _training_hashes(model, optimizer)
        result["native_boundary_after"] = host_boundary(spec["max_host_bytes"])
        verify_spec(spec, content=False)
        result["postprocess"] = {"seconds": time.perf_counter() - postprocess_started,
                                 "resources": sampler.finish(),
                                 "cuda": worker._cuda_peaks(torch)}
        sampler = None
        worker._enforce_stage_limits(spec, result["postprocess"])
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
                result.update(status="failed", cleanup_error=repr(exc))
        result["total_worker_seconds"] = time.perf_counter() - started
        result["resources_after"] = worker._resources()
        atomic_json(result_path, result)
    return 0 if result["status"] == "passed" else 1


def main(argv: list[str] | None = None) -> int:
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=("two_read", "two_read_cache"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--warmup-updates", default=2, type=int)
    parser.add_argument("--max-trace-bytes", default=MAX_TRACE_BYTES, type=int)
    args = parser.parse_args(argv)
    code = run_profile(args.spec, args.arm, args.out, warmup_updates=args.warmup_updates,
                       max_trace_bytes=args.max_trace_bytes)
    Console().print({"profile_result": str(args.out / "profile-result.json"), "exit_code": code})
    return code


if __name__ == "__main__":
    raise SystemExit(main())
