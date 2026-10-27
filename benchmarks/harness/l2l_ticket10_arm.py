#!/usr/bin/env python3
"""A5 ticket 10: the ticket-09 arm with the step split into parts and the host sensed.

Added to ``l2l_ticket09_arm.main`` (behaviour unchanged):

* a sampler thread reads ``/proc/stat``, ``/proc/loadavg`` and ``nvidia-smi`` once a second;
  the record gets ``sensors`` (``judge_run``) and the samples;
* before the run, a copy-bandwidth probe of the host (pageable and pinned, both ways);
* with ``--profile``, each part of a step is timed with the CUDA stream synchronised around
  it, and ``steps[i]["parts"]`` holds seconds per part. ``--profile`` slows the run, so the
  speed of a profiled run is not a speed result. Without it only the sensors run.

Parts: ``layer_wait`` (``pool.wait``, the compute stream waiting for a layer), ``activation_put``
and ``activation_get`` (L2L's copies to host RAM and back), ``optimizer`` (the accelerate
optimizer step), ``compute`` (the rest of the step). Benchmark harness code, not shipped.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import l2l_ticket10_profile as prof  # noqa: E402

GPU_QUERY = "clocks.sm,clocks.max.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active"


def read_gpu() -> Optional[Dict[str, Any]]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = out.stdout.strip().splitlines()[:1]
    return prof.parse_gpu_query(line[0]) if out.returncode == 0 and line else None


def read_stat() -> Optional[Dict[str, int]]:
    try:
        return prof.parse_proc_stat(Path("/proc/stat").read_text())
    except OSError:
        return None


class Sampler(threading.Thread):
    """One sample a second until ``stop``; each holds the CPU shares since the last one."""

    def __init__(self, period: float = 1.0):
        super().__init__(daemon=True)
        self.period, self.samples = period, []
        self._stop_event = threading.Event()
        self.cpu_start = time.process_time()
        self.wall_start = time.perf_counter()

    def run(self) -> None:
        before = read_stat()
        while not self._stop_event.wait(self.period):
            after = read_stat()
            sample: Dict[str, Any] = {"t": time.perf_counter() - self.wall_start,
                                      "gpu": read_gpu(), "cpu": None}
            if before and after:
                sample["cpu"] = prof.cpu_fractions(before, after)
            try:
                sample["loadavg1"] = float(Path("/proc/loadavg").read_text().split()[0])
            except (OSError, ValueError):
                sample["loadavg1"] = None
            self.samples.append(sample)
            before = after

    def stop(self) -> List[Dict[str, Any]]:
        self._stop_event.set()
        self.join(timeout=5)
        return self.samples

    def own_cpu_share(self) -> Optional[float]:
        """This process's CPU seconds over wall seconds over the visible cores."""
        cores = os.cpu_count() or 1
        wall = time.perf_counter() - self.wall_start
        return (time.process_time() - self.cpu_start) / wall / cores if wall > 0 else None


def host_bandwidth(nbytes: int = 1 << 30) -> Dict[str, Any]:
    """GB/s of a pageable host copy and, with CUDA, pinned host to device and back."""
    import torch

    out: Dict[str, Any] = {"bytes": nbytes}
    src = torch.empty(nbytes, dtype=torch.uint8)
    dst = torch.empty(nbytes, dtype=torch.uint8)
    src.fill_(1)
    dst.copy_(src)
    start = time.perf_counter()
    for _ in range(3):
        dst.copy_(src)
    out["pageable_copy_gb_s"] = 3 * nbytes / (time.perf_counter() - start) / 1e9
    if torch.cuda.is_available():
        pinned = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        device = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        pairs = (("h2d_pinned_gb_s", device, pinned), ("d2h_pinned_gb_s", pinned, device))
        for name, a, b in pairs:
            a.copy_(b, non_blocking=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(3):
                a.copy_(b, non_blocking=True)
            torch.cuda.synchronize()
            out[name] = 3 * nbytes / (time.perf_counter() - start) / 1e9
        del pinned, device
    return out


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    profile = "--profile" in argv
    argv = [a for a in argv if a != "--profile"]
    out = argv[argv.index("--out") + 1]

    import torch

    import soup_cli.utils.l2l_activations as store_module
    import soup_cli.utils.layer_stream_runtime as runtime_module

    cuda = torch.cuda.is_available()
    buckets = prof.Buckets(sync=torch.cuda.synchronize if cuda and profile else None)

    def wrap(cls: Any, name: str, bucket: str) -> None:
        real = getattr(cls, name)

        def timed(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not profile:
                return real(self, *args, **kwargs)
            with buckets.timed(bucket):
                return real(self, *args, **kwargs)

        setattr(cls, name, timed)

    wrap(runtime_module.LayerBufferPool, "wait", "layer_wait")
    wrap(runtime_module.LargeLayerBufferPool, "wait", "layer_wait")
    wrap(store_module.ActivationStore, "put", "activation_put")
    wrap(store_module.ActivationStore, "get", "activation_get")

    import accelerate.optimizer as accel_optimizer

    wrap(accel_optimizer.AcceleratedOptimizer, "step", "optimizer")

    import transformers

    parts_per_step: List[Dict[str, float]] = []

    class Callback(transformers.TrainerCallback):
        def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            buckets.take()

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            parts_per_step.append(buckets.take())

    real_init = transformers.Trainer.__init__

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        real_init(self, *args, **kwargs)
        self.add_callback(Callback())

    transformers.Trainer.__init__ = init

    bandwidth = host_bandwidth()
    sampler = Sampler()
    sampler.start()
    import l2l_ticket09_arm

    status = l2l_ticket09_arm.main(argv)
    samples = sampler.stop()
    path = Path(out)
    record = json.loads(path.read_text())
    record["profile"] = profile
    record["host_bandwidth"] = bandwidth
    record["sensor_samples"] = samples
    record["sensors"] = prof.judge_run(samples, own_cpu_share=sampler.own_cpu_share())
    if profile:
        rows = []
        for step, parts in zip(record["steps"], parts_per_step):
            step["parts"] = prof.decompose(step["step_s"], parts)
            rows.append(step["parts"])
        warm = rows[1:]
        record["parts_mean"] = prof.mean_parts(warm)
        loads = [s["layer_loads"] for s in record["steps"][1:len(rows)]]
        if warm and loads and sum(loads):
            record["layer_wait_per_load_s"] = (
                sum(r["layer_wait"] for r in warm) / sum(loads)
            )
    path.write_text(json.dumps(record, indent=1, default=str))
    return status


if __name__ == "__main__":
    sys.exit(main())
