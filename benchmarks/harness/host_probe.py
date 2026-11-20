"""What a rental host can do, in numbers (harness only): the CPU cost of the operations a step
issues, the copy bandwidths, the disk and the GPU settings.

Run on each host the same way and compare the JSON. The CPU figures are what the step is
limited by on a host that feeds the GPU slowly: the time the CPU needs to issue one operation,
not the arithmetic it does.

    python host_probe.py OUT.json [FILE_FOR_DISK_READ]
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Optional

from rich.console import Console


def cpu_us(fn: Callable[[], Any], reps: int = 2000, sync_every: int = 200) -> float:
    """Microseconds of CPU time per call of ``fn`` (the GPU is synchronised only every
    ``sync_every`` calls, so the figure is the cost of issuing the work)."""
    import torch

    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for index in range(reps):
        fn()
        if index % sync_every == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / reps * 1e6


def run(command: str) -> str:
    out = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
    return (out.stdout + out.stderr).strip()


def bandwidth_gbs(copy: Callable[[], Any], nbytes: int, reps: int = 8) -> float:
    import torch

    copy()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(reps):
        copy()
    torch.cuda.synchronize()
    return nbytes * reps / (time.perf_counter() - started) / 1e9


def probe(disk_file: Optional[str]) -> Dict[str, Any]:
    import torch

    dev = "cuda"
    x8 = torch.randn(8, 2048, device=dev, dtype=torch.bfloat16)
    w = torch.randn(2816, 2048, device=dev, dtype=torch.bfloat16)
    small = torch.zeros(1024, device=dev)
    pieces = [torch.randn(3, 2048, device=dev, dtype=torch.bfloat16) for _ in range(3)]
    out: Dict[str, Any] = {
        "platform": platform.platform(),
        "cpu_model": run("lscpu | grep 'Model name' | head -1 | cut -d: -f2").strip(),
        "cpu_threads": run("nproc"),
        "cpu_mhz_max": run("lscpu | grep -i 'max mhz' | head -1 | cut -d: -f2").strip(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_query": run("nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current,"
                         "clocks.max.sm,clocks.sm,power.limit,memory.total --format=csv,noheader"),
        "torch": torch.__version__,
    }
    t = time.perf_counter()
    total = 0
    for i in range(3_000_000):
        total += i
    out["python_loop_ns_per_iter"] = (time.perf_counter() - t) / 3_000_000 * 1e9
    out["cpu_us"] = {
        "tiny_cuda_kernel_add_": cpu_us(lambda: small.add_(1)),
        "torch_mm_M8_2048x2816": cpu_us(lambda: torch.mm(x8, w.t())),
        "torch_empty_cuda": cpu_us(lambda: torch.empty(64, 2048, device=dev, dtype=torch.bfloat16)),
        "torch_cat_3": cpu_us(lambda: torch.cat(pieces)),
        "torch_silu": cpu_us(lambda: torch.nn.functional.silu(x8)),
        "tensor_narrow_view": cpu_us(lambda: x8.narrow(0, 1, 4), reps=20000),
    }
    host = torch.empty(256 * 1024 * 1024, dtype=torch.uint8).pin_memory()
    gpu = torch.empty_like(host, device=dev)
    nbytes = host.numel()
    out["copy_gbs"] = {
        "h2d_pinned": bandwidth_gbs(lambda: gpu.copy_(host, non_blocking=True), nbytes),
        "d2h_pinned": bandwidth_gbs(lambda: host.copy_(gpu, non_blocking=True), nbytes),
        "d2d": bandwidth_gbs(lambda: gpu.clone(), nbytes),
    }
    if disk_file:
        out["disk_read_gbs_direct"] = run(
            f"dd if={disk_file} of=/dev/null bs=16M count=96 iflag=direct 2>&1 | tail -1"
        )
    out["ram"] = run("free -g | sed -n 2p")
    return out


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "host_probe.json"
    disk_file = sys.argv[2] if len(sys.argv) > 2 else None
    result = probe(disk_file)
    with open(target, "w") as handle:
        json.dump(result, handle, indent=1)
    Console().out(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
