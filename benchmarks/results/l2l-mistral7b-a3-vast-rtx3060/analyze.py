# ruff: noqa
# One-off ticket A3-05 diagnostic, run on the Vast instance; kept as evidence, not linted.
import torch, re, collections
ga = torch.load("/root/runs/grads_ga.pt"); l2 = torch.load("/root/runs/grads_l2l.pt")
for step in range(len(ga)):
    by = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    tot_d = tot_g = 0.0
    nz = 0
    for n, g in ga[step].items():
        h = l2[step][n]
        d = (g.float() - h.float())
        L = int(re.search(r"layers\.(\d+)\.", n).group(1))
        e = by[L]; e[0] += d.pow(2).sum().item(); e[1] += g.float().pow(2).sum().item()
        e[2] = max(e[2], d.abs().max().item()); e[3] = max(e[3], g.abs().max().item())
        nz += int((g != h).sum())
    print(f"step {step}: differing elements {nz}")
    for L in sorted(by):
        if L % 4 == 0 or L in (1, 31):
            dd, gg, mx, gmx = by[L]
            print(f"  layer {L:2d}  rel_l2 {dd**.5/max(gg**.5,1e-30):.3e}  max|d| {mx:.3e}  max|g| {gmx:.3e}")
print("dtype", next(iter(ga[0].values())).dtype)
