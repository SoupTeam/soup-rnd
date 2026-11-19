"""Diagnosis 4: does a CUDA graph of one segmented expert call (forward + VJP) keep the eager bits,
and how much of the launch cost does a replay remove?

One Moonlight expert (hidden 2048, moe intermediate 1408, bf16). The call is the segmented one
of ``--expert-gemm segmented``: one F.linear per micro-batch segment, route-weight multiply per
segment, then the VJP for hidden and weights (what WeightedFrozenExpert.backward does).
Prints, per segment-count tuple: eager time, replay time, capture time, and bit-equality of
out / grad_hidden / grad_weight between eager and replay (same data, then fresh data).
"""
import time

import torch
import torch.nn.functional as functional

HID, INTER = 2048, 1408
dev = "cuda"
torch.manual_seed(0)
gate_up = torch.randn(2 * INTER, HID, device=dev, dtype=torch.bfloat16) * 0.02
down = torch.randn(HID, INTER, device=dev, dtype=torch.bfloat16) * 0.02


def expert(x, w, counts):
    hidden = torch.cat([functional.linear(p, gate_up) for p in x.split(counts)])
    gate, up = hidden.chunk(2, dim=-1)
    act = functional.silu(gate) * up
    y = torch.cat([functional.linear(p, down) for p in act.split(counts)])
    return torch.cat([a * b for a, b in zip(y.split(counts), w.split(counts))])


def eager(x, w, g, counts):
    out = expert(x, w, counts)
    gx, gw = torch.autograd.grad(out, [x, w], g)
    return out.detach(), gx, gw


def fresh(m):
    x = torch.randn(m, HID, device=dev, dtype=torch.bfloat16)
    w = torch.rand(m, 1, device=dev, dtype=torch.float32)
    g = torch.randn(m, HID, device=dev, dtype=torch.bfloat16)
    return x, w, g


def same(a, b):
    return [bool(torch.equal(p, q)) for p, q in zip(a, b)]


def timed(fn, n=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


print(f"torch {torch.__version__} gpu {torch.cuda.get_device_name(0)}")
print("counts -> eager ms | replay ms | capture ms | eager==replay (out, gx, gw)",
      "same data / fresh data")
for counts in ([4, 20, 58, 7, 33, 12, 1, 90], [1] * 8, [20] * 8, [240, 8, 8, 1], [3, 5]):
    counts = list(counts)
    m = sum(counts)
    x = torch.zeros(m, HID, device=dev, dtype=torch.bfloat16, requires_grad=True)
    w = torch.ones(m, 1, device=dev, dtype=torch.float32, requires_grad=True)
    g = torch.ones(m, HID, device=dev, dtype=torch.bfloat16)
    sx, sw, sg = fresh(m)
    with torch.no_grad():
        x.copy_(sx)
        w.copy_(sw)
        g.copy_(sg)
    # warmup on a side stream, as the CUDA graph notes require
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            eager(x, w, g, counts)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    t0 = time.perf_counter()
    with torch.cuda.graph(graph):
        s_out, s_gx, s_gw = eager(x, w, g, counts)
    torch.cuda.synchronize()
    capture_ms = (time.perf_counter() - t0) * 1e3
    graph.replay()
    torch.cuda.synchronize()
    ref = eager(x, w, g, counts)
    same_data = same((s_out, s_gx, s_gw), ref)
    nx, nw, ng = fresh(m)
    with torch.no_grad():
        x.copy_(nx)
        w.copy_(nw)
        g.copy_(ng)
    graph.replay()
    torch.cuda.synchronize()
    ref2 = eager(x, w, g, counts)
    fresh_data = same((s_out, s_gx, s_gw), ref2)
    eager_ms = timed(lambda: eager(x, w, g, counts))
    replay_ms = timed(graph.replay)
    print("%-26s -> %6.2f | %6.2f | %7.1f | %s / %s"
          % (counts, eager_ms, replay_ms, capture_ms, same_data, fresh_data))
