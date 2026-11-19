"""Diagnosis 2 loop: one native expert call over a joined group against one call per segment.

B0 joins the rows of 8 micro-batches into one expert call; A0 calls the expert once per
micro-batch. Same body as expert_stream.expert_mlp, forward and the VJP of WeightedFrozenExpert
(hidden and route weight). Prints, per case, whether every segment is bit-equal between the joined
call and the per-segment calls. Red = some segment differs.
"""
import torch
import torch.nn.functional as functional

HID, INTER = 2048, 1408
dev = "cuda"
torch.manual_seed(0)
gate_up = torch.randn(2 * INTER, HID, device=dev, dtype=torch.bfloat16) * 0.02
down = torch.randn(HID, INTER, device=dev, dtype=torch.bfloat16) * 0.02


def expert(x, w):
    x = x.clone().requires_grad_(True)
    w = w.clone().requires_grad_(True)
    gate, up = functional.linear(x, gate_up).chunk(2, dim=-1)
    out = functional.linear(functional.silu(gate) * up, down) * w
    gx, gw = torch.autograd.grad(out, [x, w], torch.ones_like(out))
    return out.detach(), gx, gw


def case(sizes):
    xs = [torch.randn(m, HID, device=dev, dtype=torch.bfloat16) for m in sizes]
    ws = [torch.rand(m, 1, device=dev, dtype=torch.float32) for m in sizes]
    joined = expert(torch.cat(xs), torch.cat(ws))
    off, flags = 0, []
    for x, w in zip(xs, ws):
        m = x.shape[0]
        part = expert(x, w)
        flags.append([bool(torch.equal(j[off:off + m], p)) for j, p in zip(joined, part)])
        off += m
    return flags


print(f"torch {torch.__version__} gpu {torch.cuda.get_device_name(0)}")
for sizes in ([20] * 8, [4, 20, 58, 240, 7, 33, 90, 12], [64] * 8, [1] * 8, [240] * 2):
    flags = case(sizes)
    ok = all(all(f) for f in flags)
    bad = [(m, f) for m, f in zip(sizes, flags) if not all(f)]
    print("sizes %-34s all segments bit-equal: %s" % (sizes, ok))
    for m, f in bad:
        print("   M=%3d differs: out=%s grad_hidden=%s grad_weight=%s" % (m, *[not v for v in f]))
