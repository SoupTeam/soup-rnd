"""CPU cache-allocation experiment on frozen routed-expert visits.

The buffers have Moonlight's packed NF4 shapes but synthetic contents. This is
neither model-quality evidence nor a GPU/SSD/end-to-end training benchmark.
Run each policy/mode in a fresh, sequential process; extraction is separate so
the large accuracy capture does not contribute to the measured cache RSS.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).parents[2]))

from benchmarks.harness.two_read_cache_common import atomic_json, sha256_file  # noqa: E402
from benchmarks.harness.two_read_cache_worker import ResourceSampler, _resources  # noqa: E402


def reconstruct_visits(capture: dict[str, Any], layout: str) -> list[tuple[int, int]]:
    """Reconstruct source-order visits; native tiny-model tests qualify this order.

    One full forward precedes reverse-layer backward. Ordinary checkpoint
    replay visits that layer's experts ascending; expert autograd backward
    visits them descending. Two-read suppresses replay fetches entirely.
    """
    import torch

    if layout not in {"ordinary", "two_read"}:
        raise ValueError("layout must be ordinary or two_read")
    batches = capture.get("microbatches")
    if not isinstance(batches, list) or not batches:
        raise ValueError("capture requires nonempty microbatches")
    visits = []
    for batch in batches:
        routes = batch.get("routing_weights")
        if not isinstance(routes, dict) or not routes:
            raise ValueError("microbatch lacks routed-expert indices")
        layers = {}
        for name, route in routes.items():
            layer = int(name)
            indices = route.get("indices")
            if (not isinstance(indices, torch.Tensor) or indices.dtype != torch.int64
                    or indices.ndim != 2 or not indices.numel()
                    or indices.device.type != "cpu" or indices.min().item() < 0):
                raise ValueError("indices must be a nonempty CPU int64 token-by-topk matrix")
            layers[layer] = sorted(int(value) for value in torch.unique(indices).tolist())
        for layer in sorted(layers):
            visits.extend((layer, expert) for expert in layers[layer])
        for layer in sorted(layers, reverse=True):
            if layout == "ordinary":
                visits.extend((layer, expert) for expert in layers[layer])
            visits.extend((layer, expert) for expert in reversed(layers[layer]))
    return visits


def extract_route(capture_path: Path, layout: str, output: Path) -> dict[str, Any]:
    import torch

    identity = sha256_file(capture_path)
    capture = torch.load(capture_path, map_location="cpu", weights_only=True)
    visits = reconstruct_visits(capture, layout)
    if sha256_file(capture_path) != identity:
        raise ValueError("capture changed during route extraction")
    result = {
        "schema_version": 1, "layout": layout, "source_capture_sha256": identity,
        "microbatches": len(capture["microbatches"]), "visits": visits,
        "unique_experts": len(set(visits)),
        "semantics": "reconstructed source order, qualified on tiny native autograd control",
    }
    atomic_json(output, result)
    return result


def staging_entry(hidden: int, intermediate: int) -> tuple[Any, dict[str, Any]]:
    """One reusable CPU allocation, with real double-quantized NF4 tensor shapes."""
    import torch

    if min(hidden, intermediate) < 1:
        raise ValueError("expert dimensions must be positive")
    descriptors = []
    for name, weights in (("gate_up", 2 * intermediate * hidden),
                          ("down", hidden * intermediate)):
        blocks = (weights + 63) // 64
        descriptors.extend((
            (name, ((weights + 1) // 2, 1), torch.uint8),
            (name + "::absmax", (blocks,), torch.uint8),
            (name + "::nested_absmax", ((blocks + 255) // 256,), torch.float32),
            (name + "::nested_offset", (), torch.float32),
        ))
    size = 0
    positions = []
    for name, shape, dtype in descriptors:
        element = torch.empty((), dtype=dtype).element_size()
        count = 1
        for dimension in shape:
            count *= dimension
        size = (size + element - 1) // element * element
        positions.append((name, shape, dtype, size, count * element))
        size += count * element
    owner = torch.empty(size, dtype=torch.uint8, device="cpu")
    tensors = {
        name: owner[offset:offset + length].view(dtype).view(shape)
        for name, shape, dtype, offset, length in positions
    }
    return owner, tensors


def _fill(tensors: dict[str, Any], key: tuple[int, int], *, sentinel: bool = False) -> None:
    for index, tensor in enumerate(tensors.values()):
        value = (key[0] * 131 + key[1] * 17 + index) % 251
        tensor.fill_(255 if sentinel else value)


def payload_sha256(tensors: Any) -> str:
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().contiguous().view(-1).view(torch.uint8)
                      .numpy().tobytes())
    return digest.hexdigest()


def _store(policy: str, budget: int) -> Any:
    if policy == "lru":
        from soup_cli.utils.expert_cache import TensorLRU

        return TensorLRU(budget, copy_on_insert=True)
    if policy == "static":
        from soup_cli.utils.expert_cache import TensorHotset

        return TensorHotset(budget, copy_on_insert=True)
    raise ValueError("policy must be lru or static")


def _native_peak_rss() -> int | None:
    if os.name == "nt":
        import psutil

        return int(psutil.Process().memory_info().peak_wset)
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def run_experiment(
    route_path: Path, output: Path, *, policy: str, budget_bytes: int,
    hidden: int = 2048, intermediate: int = 1408, mode: str = "timing",
    max_visits: int | None = None, device_hotset_bytes: int = 0,
    repeats: int = 1,
) -> dict[str, Any]:
    """Profile actual CPU stores, without constructing full expert weights or CUDA state."""
    import torch

    if (mode not in {"timing", "profile"} or min(budget_bytes, device_hotset_bytes) < 0
            or type(repeats) is not int or repeats < 1):
        raise ValueError("invalid CPU experiment mode or byte budget")
    output.mkdir(parents=True, exist_ok=False)
    route_hash = sha256_file(route_path)
    route = json.loads(route_path.read_text(encoding="utf-8"))
    visits = [tuple(item) for item in route["visits"]]
    if max_visits is not None:
        if max_visits < 1:
            raise ValueError("max_visits must be positive")
        visits = visits[:max_visits]
    if not visits or any(len(key) != 2 or any(type(item) is not int or item < 0 for item in key)
                         for key in visits):
        raise ValueError("visits must contain nonnegative integer layer/expert pairs")
    base_visits = len(visits)
    visits *= repeats
    torch.set_num_threads(1)
    owner, tensors = staging_entry(hidden, intermediate)
    payload = sum(value.numel() * value.element_size() for value in tensors.values())
    dense_pair_bytes = 3 * hidden * intermediate * 2
    store = _store(policy, budget_bytes)
    store_source = Path(sys.modules[type(store).__module__].__file__)
    source_hashes = {str(path): sha256_file(path) for path in (
        Path(__file__), store_source,
    )}
    watched = None
    watch_hash = None
    counts = {"hits": 0, "misses": 0, "admissions": 0, "rejections": 0,
              "device_hotset_visits": 0, "clone_calls_observed": 0, "copy_bytes_observed": 0}
    gpu_keys: set[tuple[int, int]] = set()
    cache_seconds = fill_seconds = 0.
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True,
    ) if mode == "profile" else contextlib.nullcontext()
    original_clone = torch.Tensor.clone

    def tracked_clone(tensor: Any, *args: Any, **kwargs: Any) -> Any:
        if tensor.device.type != "cpu":
            raise RuntimeError("CPU experiment refuses non-CPU tensors")
        result = original_clone(tensor, *args, **kwargs)
        counts["clone_calls_observed"] += 1
        counts["copy_bytes_observed"] += tensor.numel() * tensor.element_size()
        return result

    instrument = patch.object(torch.Tensor, "clone", tracked_clone) if mode == "profile" else (
        contextlib.nullcontext()
    )
    sampler = ResourceSampler().start()
    started, cpu_started = time.perf_counter(), time.process_time()
    with profiler as events, instrument:
        for key in visits:
            if key in gpu_keys:
                counts["device_hotset_visits"] += 1
                continue
            before = time.perf_counter()
            cached = store.get(key)
            cache_seconds += time.perf_counter() - before
            if cached is not None:
                counts["hits"] += 1
                continue
            counts["misses"] += 1
            before = time.perf_counter()
            _fill(tensors, key)
            fill_seconds += time.perf_counter() - before
            if (len(gpu_keys) + 1) * dense_pair_bytes <= device_hotset_bytes:
                # Model the native dense first-fit filter; allocate no GPU memory.
                gpu_keys.add(key)
                continue
            before = time.perf_counter()
            inserted = store.put(key, tensors)
            cache_seconds += time.perf_counter() - before
            counts["admissions" if inserted else "rejections"] += 1
            if inserted and watched is None:
                watched = store.get(key)
                watch_hash = payload_sha256(watched)
    wall, cpu = time.perf_counter() - started, time.process_time() - cpu_started
    resources = sampler.finish()
    stats = store.stats()
    _fill(tensors, (0, 0), sentinel=True)
    immutable = watched is None or payload_sha256(watched) == watch_hash
    if not immutable:
        raise RuntimeError("admitted cache entry aliases mutable reusable staging")
    expected_calls = counts["admissions"] * len(tensors)
    expected_bytes = counts["admissions"] * payload
    if mode == "profile" and (counts["clone_calls_observed"] != expected_calls
                              or counts["copy_bytes_observed"] != expected_bytes):
        raise RuntimeError("observed clone allocations differ from admitted complete payloads")
    if mode == "timing":
        counts["clone_calls_observed"] = counts["copy_bytes_observed"] = None
    operators = []
    if mode == "profile":
        for event in events.key_averages():
            if event.key in {"aten::clone", "aten::copy_", "aten::empty_strided", "aten::empty"}:
                operators.append({"operator": event.key, "calls": event.count,
                                  "self_cpu_microseconds": event.self_cpu_time_total,
                                  "cpu_memory_bytes": event.cpu_memory_usage,
                                  "self_cpu_memory_bytes": event.self_cpu_memory_usage})
        events.export_chrome_trace(str(output / "cpu-trace.json"))
    before_clear = _resources()
    store.clear()
    cached = watched = None
    after_clear = _resources()
    gc.collect()
    after_gc = _resources()
    if sha256_file(route_path) != route_hash or any(
        sha256_file(path) != identity for path, identity in source_hashes.items()
    ):
        raise ValueError("route or executed source changed during the experiment")
    result = {
        "status": "passed", "pid": os.getpid(), "policy": policy, "mode": mode,
        "route_sha256": route_hash, "source_fingerprint": source_hashes,
        "source_capture_sha256": route.get(
            "source_capture_sha256"), "route_layout": route.get("layout"),
        "visits": len(visits), "unique_experts": len(set(visits)), "counts": counts,
        "base_trace_visits": base_visits, "fixed_trace_repeats": repeats,
        "budget_bytes": budget_bytes, "payload_bytes_per_expert": payload,
        "staging_allocation_bytes": owner.numel(), "tensors_per_expert": len(tensors),
        "shapes": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                   for key, value in tensors.items()},
        "cloned_payload_bytes_derived": expected_bytes, "clone_calls_derived": expected_calls,
        "modeled_device_hotset_bytes": device_hotset_bytes,
        "modeled_device_entries": len(gpu_keys), "cache_stats": stats,
        "modeled_storage_reads": counts["misses"],
        "modeled_storage_payload_bytes": counts["misses"] * payload,
        "cache_operation_wall_seconds": cache_seconds, "synthetic_fill_seconds": fill_seconds,
        "whole_loop_wall_seconds": wall, "whole_loop_cpu_seconds": cpu,
        "resources": resources, "native_process_peak_rss_bytes": _native_peak_rss(),
        "retained_watch_payload_bytes": payload if watch_hash is not None else 0,
        "verification_lookup_hits": int(watch_hash is not None),
        "watch_payload_sha256": watch_hash, "immutable_after_staging_mutation": immutable,
        "cleanup_resources": {"before_clear": before_clear, "after_clear": after_clear,
                              "after_gc": after_gc,
                              "allocator_trim_performed": False,
                              "note": "Diagnostic only, after timing/profile; native allocator "
                              "may retain freed storage. No allocator knobs changed."},
        "operators": operators, "torch_version": torch.__version__, "threads": 1,
        "limitations": "Synthetic NF4-shaped bytes, reconstructed visits, CPU store only; "
        "no dequantization, H2D, SSD, model quality or end-to-end speed verdict. Profiler mode "
        "and synthetic fill timings cannot be used as uninstrumented training throughput.",
    }
    atomic_json(output / "result.json", result)
    return result


def main() -> int:
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract-route")
    extract.add_argument("--capture", required=True, type=Path)
    extract.add_argument("--layout", choices=("ordinary", "two_read"), default="two_read")
    extract.add_argument("--out", required=True, type=Path)
    run = sub.add_parser("run")
    run.add_argument("--route", required=True, type=Path)
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--policy", choices=("lru", "static"), required=True)
    run.add_argument("--mode", choices=("timing", "profile"), default="timing")
    run.add_argument("--budget-bytes", type=int, default=2 * 2**30)
    run.add_argument("--device-hotset-bytes", type=int, default=0)
    run.add_argument("--hidden", type=int, default=2048)
    run.add_argument("--intermediate", type=int, default=1408)
    run.add_argument("--max-visits", type=int)
    run.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.command == "extract-route":
        result = extract_route(args.capture, args.layout, args.out)
        Console().print({"visits": len(result["visits"]), "route": str(args.out)})
    else:
        result = run_experiment(
            args.route, args.out, policy=args.policy, mode=args.mode,
            budget_bytes=args.budget_bytes, hidden=args.hidden, intermediate=args.intermediate,
            max_visits=args.max_visits, device_hotset_bytes=args.device_hotset_bytes,
            repeats=args.repeats,
        )
        Console().print({key: result[key] for key in (
            "status", "policy", "mode", "counts", "cache_operation_wall_seconds",
        )})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
