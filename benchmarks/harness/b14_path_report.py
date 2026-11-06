"""B1.4 ticket 09: memory, disk reads and operation launches of the per_expert path.

``OpCounter`` counts what runs inside a ``with`` block, two ways:

  dispatch   every ATen operation the dispatcher sees (``TorchDispatchMode``). Works on CPU,
             so the counter is tested without a GPU. It counts operations, not GPU kernels:
             one operation can launch zero, one or several kernels.
  profiler   with ``profile_cuda=True``, ``torch.profiler`` with CUPTI records the CUDA
             kernels that ran (``cat == "kernel"`` in the Chrome trace), the device copies and
             the ``cudaLaunchKernel`` runtime calls. This is the count the card asks for
             ("number of GPU operation launches"). It needs a CUDA device.

``measure_path`` runs the path on a checkpoint directory and returns one dictionary: VRAM and
RAM (idle and peak), disk reads, layer loads, expert fetches and launches. It records no wall
time: the report is about capacity, not speed.

Run from the checkout root with its src on PYTHONPATH:
  python benchmarks/harness/b14_path_report.py --weights DIR --work DIR --output out.json
"""

from __future__ import annotations

import collections
import gc
import json
import resource
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from . import b13_moe_fixture as fixture
    from . import b14_real_moe_correctness as correctness
except ImportError:  # run as a script
    import b13_moe_fixture as fixture  # type: ignore[no-redef]
    import b14_real_moe_correctness as correctness  # type: ignore[no-redef]

_LAUNCH_CALLS = ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchCooperativeKernel")


def _is_cuda_launch_call(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in _LAUNCH_CALLS)


class OpCounter:
    """Count ATen operations, and optionally CUDA kernel launches, inside a ``with`` block."""

    def __init__(self, *, profile_cuda: bool = False) -> None:
        self._profile_cuda = profile_cuda
        self._mode: Any = None
        self._profiler: Any = None
        self.by_op: Dict[str, int] = {}
        self.cuda_kernel_launches: Optional[int] = None
        self.cuda_memcpy_events: Optional[int] = None
        self.cuda_runtime_launch_calls: Optional[int] = None
        self.cuda_kernels_by_name: Dict[str, int] = {}

    @property
    def aten_ops(self) -> int:
        return sum(self.by_op.values())

    def __enter__(self) -> "OpCounter":
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode

        counts: collections.Counter = collections.Counter()

        class _Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func: Any, types: Any, args: Any = (),
                                   kwargs: Any = None) -> Any:
                counts[str(func)] += 1
                return func(*args, **(kwargs or {}))

        self._counts = counts
        if self._profile_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("profile_cuda needs a CUDA device and none is available")
            from torch.profiler import ProfilerActivity, profile

            torch.cuda.synchronize()
            self._profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            )
            self._profiler.__enter__()
        self._mode = _Mode()
        self._mode.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        import torch

        self._mode.__exit__(*exc)
        self.by_op = dict(self._counts)
        if self._profiler is not None:
            torch.cuda.synchronize()
            self._profiler.__exit__(*exc)
            self._read_trace()
            self._profiler = None

    def _read_trace(self) -> None:
        with tempfile.TemporaryDirectory(prefix="b14-trace-") as folder:
            path = Path(folder) / "trace.json"
            self._profiler.export_chrome_trace(str(path))
            events = json.loads(path.read_text()).get("traceEvents", [])
        kernels: collections.Counter = collections.Counter()
        memcpy = launches = 0
        for event in events:
            category = event.get("cat")
            if category == "kernel":
                kernels[event.get("name", "")] += 1
            elif category == "gpu_memcpy":
                memcpy += 1
            elif category == "cuda_runtime" and _is_cuda_launch_call(event.get("name", "")):
                launches += 1
        self.cuda_kernel_launches = sum(kernels.values())
        self.cuda_memcpy_events = memcpy
        self.cuda_runtime_launch_calls = launches
        self.cuda_kernels_by_name = dict(kernels.most_common(25))


# --------------------------------------------------------------------------
# Memory and disk probes
# --------------------------------------------------------------------------
def _rss_bytes() -> int:
    import psutil

    return int(psutil.Process().memory_info().rss)


def _lifetime_peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _process_io() -> Optional[Dict[str, int]]:
    """Cumulative read counters of this process: ``read_bytes`` is what reached storage."""
    import psutil

    try:
        counters = psutil.Process().io_counters()
    except (AttributeError, NotImplementedError, psutil.Error):
        return None  # macOS has no per-process I/O counters
    return {"read_bytes": int(counters.read_bytes), "read_chars": int(counters.read_chars)}


class RssSampler:
    """Poll the process RSS from a thread; ``peak`` is the largest sample, a lower bound."""

    def __init__(self, interval: float = 0.02) -> None:
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.peak = 0

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self.peak = max(self.peak, _rss_bytes())

    def __enter__(self) -> "RssSampler":
        self.peak = _rss_bytes()
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, _rss_bytes())


def _cuda_memory(device: str) -> Dict[str, Optional[int]]:
    if device != "cuda":
        return {"allocated": None, "reserved": None}
    import torch

    return {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
    }


def _diff(after: Optional[Dict[str, int]], before: Optional[Dict[str, int]]) -> Any:
    if after is None or before is None:
        return None
    return {key: after[key] - before[key] for key in after}


def _lm_batch(config: Any, *, rows: int, seq: int) -> Dict[str, Any]:
    return fixture.moe_batch(config, seed=seq, rows=rows, seq=seq)


def _count_inside_experts(model: Any) -> tuple[List[int], List[Any]]:
    """Hooks that add the ATen operations run inside each ``mlp.experts`` forward to ``total[0]``.

    Forward only: the backward of an expert runs inside an autograd node, which has no module
    boundary to hook. Hook handles are returned so the caller removes them.
    """
    total = [0]
    handles: List[Any] = []
    for name, module in model.named_modules():
        if not name.endswith("mlp.experts"):
            continue
        open_counters: List[OpCounter] = []

        def before(
            _module: Any, _args: Any, open_counters: List[OpCounter] = open_counters,
        ) -> None:
            counter = OpCounter()
            counter.__enter__()
            open_counters.append(counter)

        def after(_module: Any, _args: Any, _out: Any,
                  open_counters: List[OpCounter] = open_counters) -> None:
            counter = open_counters.pop()
            counter.__exit__(None, None, None)
            total[0] += counter.aten_ops

        handles.append(module.register_forward_pre_hook(before))
        handles.append(module.register_forward_hook(after))
    return total, handles


def _one_pass(
    model: Any, runtime: Any, batch: Dict[str, Any], *, device: str, profile: bool,
) -> Dict[str, Any]:
    """One forward and backward pass with counters; returns the measured fields."""
    import torch

    model.train()
    model.zero_grad(set_to_none=True)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    stats_before = runtime.stats()
    io_before = _process_io()
    hits, handles = correctness._track_routing(model)
    experts_ops, expert_handles = _count_inside_experts(model)
    try:
        with RssSampler() as sampler:
            with OpCounter(profile_cuda=profile) as forward:
                out = model(**batch)
            with OpCounter(profile_cuda=profile) as backward:
                out.loss.backward()
            if device == "cuda":
                torch.cuda.synchronize()
    finally:
        for handle in handles + expert_handles:
            handle.remove()
    stats_after = runtime.stats()
    grads = [p.grad for p in fixture.trainables(model).values() if p.grad is not None]
    visited = sum(len(seen) for seen in hits.values())
    total = forward.aten_ops + backward.aten_ops
    cuda = None
    if profile:
        cuda = {
            "kernel_launches_forward": forward.cuda_kernel_launches,
            "kernel_launches_backward": backward.cuda_kernel_launches,
            "kernel_launches_total": forward.cuda_kernel_launches
            + backward.cuda_kernel_launches,
            "runtime_launch_calls_total": forward.cuda_runtime_launch_calls
            + backward.cuda_runtime_launch_calls,
            "memcpy_events_total": forward.cuda_memcpy_events + backward.cuda_memcpy_events,
            "kernels_by_name_forward": forward.cuda_kernels_by_name,
            "kernels_by_name_backward": backward.cuda_kernels_by_name,
        }
        cuda["kernel_launches_per_expert_visit"] = (
            cuda["kernel_launches_total"] / visited if visited else None
        )
    peak = (
        {
            "allocated": int(torch.cuda.max_memory_allocated()),
            "reserved": int(torch.cuda.max_memory_reserved()),
        }
        if device == "cuda"
        else {"allocated": None, "reserved": None}
    )
    return {
        "loss": float(out.loss.detach().float()),
        "grads_nonzero": sum(1 for g in grads if bool(g.abs().max() > 0)),
        "grads_missing": sum(
            1 for p in fixture.trainables(model).values() if p.grad is None
        ),
        "rss_peak_sampled_bytes": sampler.peak,
        "cuda_peak_allocated_bytes": peak["allocated"],
        "cuda_peak_reserved_bytes": peak["reserved"],
        "layer_loads": stats_after["layer_loads"] - stats_before["layer_loads"],
        "layer_restores": stats_after["layer_restores"] - stats_before["layer_restores"],
        "large_loads": stats_after["large_loads"] - stats_before["large_loads"],
        "expert_fetches": stats_after["expert_fetches"] - stats_before["expert_fetches"],
        "expert_bytes_read": (
            stats_after["expert_bytes_read"] - stats_before["expert_bytes_read"]
        ),
        "stream_disk_bytes": stats_after["disk_bytes"] - stats_before["disk_bytes"],
        "process_io": _diff(_process_io(), io_before),
        "experts_visited": visited,
        "experts_hit_by_layer": {layer: len(seen) for layer, seen in sorted(hits.items())},
        "aten_ops_forward": forward.aten_ops,
        "aten_ops_backward": backward.aten_ops,
        "aten_ops_total": total,
        "aten_ops_per_expert_visit": total / visited if visited else None,
        "aten_ops_in_experts_forward": experts_ops[0],
        "aten_ops_per_expert_visit_forward": experts_ops[0] / visited if visited else None,
        "aten_top_ops_forward": dict(
            collections.Counter(forward.by_op).most_common(15)
        ),
        "cuda": cuda,
    }


def _same_counts(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    keys = ("aten_ops_forward", "aten_ops_backward", "expert_fetches", "expert_bytes_read",
            "layer_loads", "experts_visited")
    same = all(left[key] == right[key] for key in keys)
    if left["cuda"] is not None and right["cuda"] is not None:
        same = same and (
            left["cuda"]["kernel_launches_total"] == right["cuda"]["kernel_launches_total"]
        )
    return same


def measure_path(
    weights: Path,
    work_dir: Path,
    *,
    device: str = "cpu",
    dtype: str = "bfloat16",
    tier: str = "ram",
    seqs: Sequence[int] = (16, 512, 2048),
    rows: int = 1,
    seed: int = 71,
    profile: Optional[bool] = None,
    keep_cache: bool = False,
) -> Dict[str, Any]:
    """Shard ``weights`` per expert, stream it on ``device`` and measure one pass per ``seqs``.

    Sequence lengths run in order; the first is also run twice to check the counts repeat.
    A CUDA out-of-memory error is recorded as ``status: "oom"`` and stops the sweep: the
    lengths after it are not run. ``profile`` defaults to ``device == "cuda"``.
    """
    import torch

    from soup_cli.utils.kimi_k2 import load_stream_config
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream import stream_arch_of
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    if device not in ("cpu", "cuda"):
        raise ValueError(f"device must be cpu or cuda, got {device!r}")
    profile = (device == "cuda") if profile is None else profile
    weights, work_dir = Path(weights), Path(work_dir)
    config = load_stream_config(str(weights), trust_remote_code=False)
    config.use_cache = False
    shards = work_dir / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    lora = fixture.moe_lora_config(targets=correctness.lora_targets(config))

    index = shard_checkpoint(
        str(weights), str(shards), dtype=dtype, arch=stream_arch_of(config),
        expert_layout="per_expert",
    )
    cache_bytes = sum(f.stat().st_size for f in shards.rglob("*") if f.is_file())
    peak_after_shard = _lifetime_peak_rss_bytes()
    gc.collect()
    result: Dict[str, Any] = {
        "device": device, "dtype": dtype, "tier": tier, "rows": rows, "profile": profile,
        "device_name": torch.cuda.get_device_name() if device == "cuda" else None,
        "device_total_bytes": (
            int(torch.cuda.get_device_properties(0).total_memory) if device == "cuda" else None
        ),
        "setup": {"cache_bytes": cache_bytes, "rss_peak_after_shard_bytes": peak_after_shard},
        "passes": [], "repeat": None,
    }
    streamed, runtime = build_streamed_model(
        model_id=str(weights), shard_dir=str(shards), index=index, lora_config=lora,
        device=device, dtype=dtype, buffers=2, pin=device == "cuda", seed=seed, quant="none",
        tier=tier,
    )
    try:
        fixture.randomize_adapters(streamed)
        gc.collect()
        idle_cuda = _cuda_memory(device)
        stats = runtime.stats()
        result["idle"] = {
            "rss_bytes": _rss_bytes(),
            "cuda_allocated_bytes": idle_cuda["allocated"],
            "cuda_reserved_bytes": idle_cuda["reserved"],
        }
        result["stream_stats"] = {
            key: stats[key] for key in (
                "n_layers", "buffers", "buffer_bytes", "large_buffer_bytes",
                "expert_buffer_bytes", "store_bytes", "staging_bytes", "resident_bytes",
                "pinned", "pinned_bytes", "tier", "read_ahead", "total_params",
            )
        }
        first_batch = _lm_batch(config, rows=rows, seq=seqs[0])
        # Warm-up: CUDA context, cuBLAS handles and allocator growth belong to no measured pass.
        _one_pass(streamed, runtime, first_batch, device=device, profile=False)
        for index_of, seq in enumerate(seqs):
            batch = first_batch if index_of == 0 else _lm_batch(config, rows=rows, seq=seq)
            try:
                measured = _one_pass(streamed, runtime, batch, device=device, profile=profile)
            except torch.cuda.OutOfMemoryError as error:
                result["passes"].append(
                    {"seq": seq, "rows": rows, "status": "oom", "message": str(error)[:300]}
                )
                break
            measured.update(seq=seq, rows=rows, status="ok")
            result["passes"].append(measured)
            if index_of == 0:
                again = _one_pass(streamed, runtime, batch, device=device, profile=profile)
                result["repeat"] = {"seq": seq, "identical": _same_counts(measured, again)}
    finally:
        runtime.close()
        del streamed
        gc.collect()
        if not keep_cache:
            shutil.rmtree(shards, ignore_errors=True)
    result["rss_peak_process_bytes"] = _lifetime_peak_rss_bytes()
    return result
