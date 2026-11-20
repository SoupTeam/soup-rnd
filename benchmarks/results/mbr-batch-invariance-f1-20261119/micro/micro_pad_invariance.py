"""Diagnosis 1, remedy check (D1): pad every expert call to one fixed M with zero rows.

Checks (a) a row's forward output and its gradients are bit-equal whatever the real M and the
other rows are, and (b) the cost per real row. Same body as expert_stream.expert_mlp.
"""
import time

import torch
import torch.nn.functional as functional

HID, INTER, PAD = 2048, 1408, 512
dev = "cuda"
torch.manual_seed(0)
gate_up = torch.randn(2 * INTER, HID, device=dev, dtype=torch.bfloat16) * 0.02
down = torch.randn(HID, INTER, device=dev, dtype=torch.bfloat16) * 0.02


def expert_mlp(hidden, gate_up, down, act_fn):
    gate, up = functional.linear(hidden, gate_up).chunk(2, dim=-1)
    return functional.linear(act_fn(gate) * up, down)


def padded_step(x, w):
    """x [m, HID] bf16, w [m, 1] fp32 -> out [m, HID], grad_hidden [m, HID], grad_w [m, 1]."""
    m = x.shape[0]
    xp = torch.zeros(PAD, HID, device=dev, dtype=x.dtype)
    xp[:m] = x
    wp = torch.ones(PAD, 1, device=dev, dtype=w.dtype)
    wp[:m] = w
    xp.requires_grad_(True)
    wp.requires_grad_(True)
    out = expert_mlp(xp, gate_up, down, functional.silu) * wp
    g = torch.zeros_like(out)
    g[:m] = 1
    gh, gw = torch.autograd.grad(out, [xp, wp], g)
    return out[:m], gh[:m], gw[:m]


row = torch.randn(1, HID, device=dev, dtype=torch.bfloat16)
wrow = torch.rand(1, 1, device=dev, dtype=torch.float32)
ref = padded_step(row, wrow)
print("bit-equality of row 0 against real M=1 (padded to %d), random neighbours:" % PAD)
for m in (2, 4, 20, 58, 240, 512):
    x = torch.randn(m, HID, device=dev, dtype=torch.bfloat16)
    w = torch.rand(m, 1, device=dev, dtype=torch.float32)
    x[0], w[0] = row[0], wrow[0]
    got = padded_step(x, w)
    same = [bool(torch.equal(a[0:1], b)) for a, b in zip(got, ref)]
    print("  M=%3d out %s grad_hidden %s grad_weight %s" % (m, *same))

print("cost per real row (us), padded to %d:" % PAD)
for m in (4, 20, 58, 240):
    x = torch.randn(m, HID, device=dev, dtype=torch.bfloat16)
    w = torch.rand(m, 1, device=dev, dtype=torch.float32)
    for _ in range(3):
        padded_step(x, w)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        padded_step(x, w)
    torch.cuda.synchronize()
    total = (time.perf_counter() - t0) / 10
    print("  M=%3d  call %.2f ms  = %.1f us per real row (fwd+bwd)"
          % (m, total * 1e3, total / m * 1e6))
