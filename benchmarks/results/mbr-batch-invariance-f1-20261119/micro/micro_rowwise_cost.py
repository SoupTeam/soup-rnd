"""Diagnosis 1 micro loop: cost per row of one expert call, native against row-wise (invariant).

Same body as expert_stream.expert_mlp (the harness swap checks the caller's function name), one
Moonlight expert (hidden 2048, moe intermediate 1408, bf16, autocast off for the micro), forward
then the VJP the way WeightedFrozenExpert.backward does it. Prints microseconds per row by M.
Red means row-wise is slower than 1.10x native (the F1 threshold).
"""
import sys
import time

import torch
import torch.nn.functional as functional

sys.path.insert(0, "/workspace/p3-baseline/repo/benchmarks/harness")
from p72_invariant_expert_gemm import rowwise_expert_linear  # noqa: E402

HID, INTER = 2048, 1408
dev = "cuda"
gate_up = torch.randn(2 * INTER, HID, device=dev, dtype=torch.bfloat16) * 0.02
down = torch.randn(HID, INTER, device=dev, dtype=torch.bfloat16) * 0.02
act = torch.nn.functional.silu


def expert_mlp(hidden, gate_up, down, act_fn):
    gate, up = functional.linear(hidden, gate_up).chunk(2, dim=-1)
    return functional.linear(act_fn(gate) * up, down)


def step(m, with_mul_rowwise):
    hidden = torch.randn(m, HID, device=dev, dtype=torch.bfloat16, requires_grad=True)
    weights = torch.rand(m, 1, device=dev, dtype=torch.float32, requires_grad=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = expert_mlp(hidden, gate_up, down, act)
    if with_mul_rowwise:
        out = torch.cat([out[i:i + 1] * weights[i:i + 1] for i in range(m)])
    else:
        out = out * weights
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    grad = torch.ones_like(out)
    torch.autograd.grad(out, [hidden, weights], grad)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1


def median_us(m, rowwise, mul_rowwise, n=7):
    runs = []
    for _ in range(n + 2):
        if rowwise:
            with rowwise_expert_linear():
                runs.append(step(m, mul_rowwise))
        else:
            runs.append(step(m, mul_rowwise))
    runs = runs[2:]
    f = sorted(r[0] for r in runs)[len(runs) // 2]
    b = sorted(r[1] for r in runs)[len(runs) // 2]
    return f / m * 1e6, b / m * 1e6


print(f"torch {torch.__version__} gpu {torch.cuda.get_device_name(0)}")
print("%5s | %-24s | %-24s | %-24s"
      % ("M", "native fwd/bwd us/row", "rowwise linear fwd/bwd", "rowwise lin+mul fwd/bwd"))
for m in (4, 20, 58, 240):
    nat = median_us(m, False, False)
    lin = median_us(m, True, False)
    both = median_us(m, True, True)
    print("%5d | %9.1f %9.1f    | %9.1f %9.1f    | %9.1f %9.1f   ratio total %.1fx"
          % (m, *nat, *lin, *both, sum(both) / sum(nat)))
