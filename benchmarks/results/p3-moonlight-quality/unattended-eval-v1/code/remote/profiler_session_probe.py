"""Probe: does per-call torch.profiler start/stop lose CUDA kernel attribution after ~3500 cycles?

Mirrors p3_deepseek_quality_attention._start_profile/_stop_profile around one efficient-SDPA
call per cycle. No model, no project code. Writes one JSON result; never retries.
"""
from __future__ import annotations

import json
import sys
import time

import torch
import torch.nn.functional as F
from rich.console import Console
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile

CYCLES = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
device = torch.device("cuda")
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
q = torch.randn(2, 16, 106, 192, device=device, dtype=dtype)
k, v = torch.randn_like(q), torch.randn_like(q)
first_empty_kernels = first_empty_operators = None
empty_kernel_cycles = 0
started = time.monotonic()
for cycle in range(1, CYCLES + 1):
    torch.cuda.synchronize(device)
    profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                       record_shapes=False, profile_memory=False, with_stack=False)
    profiler.start()
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize(device)
    profiler.stop()
    operators, kernels = set(), set()
    for event in profiler.events():
        if event.name.startswith("aten::"):
            operators.add(event.name)
            pending = [event]
            while pending:
                child = pending.pop()
                kernels.update(kernel.name for kernel in child.kernels)
                pending.extend(child.cpu_children)
    if not operators and first_empty_operators is None:
        first_empty_operators = cycle
    if not kernels:
        empty_kernel_cycles += 1
        if first_empty_kernels is None:
            first_empty_kernels = cycle
result = {
    "cycles": CYCLES, "dtype": str(dtype), "device": torch.cuda.get_device_name(device),
    "torch": torch.__version__, "cuda": torch.version.cuda,
    "first_empty_kernels_cycle": first_empty_kernels,
    "first_empty_operators_cycle": first_empty_operators,
    "empty_kernel_cycles": empty_kernel_cycles, "wall_seconds": time.monotonic() - started,
}
with open(sys.argv[1], "x", encoding="utf-8") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
Console().print(result)
