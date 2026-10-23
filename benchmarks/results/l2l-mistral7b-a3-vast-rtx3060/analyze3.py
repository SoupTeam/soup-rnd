# ruff: noqa
# One-off ticket A3-05 diagnostic, run on the Vast instance; kept as evidence, not linted.
import torch
def rel(a, b, s):
    num = den = 0.0; mx = 0.0; ne = 0
    for n, g in a[s].items():
        d = g.double() - b[s][n].double()
        num += d.pow(2).sum().item(); den += g.double().pow(2).sum().item(); mx = max(mx, d.abs().max().item()); ne += int((g != b[s][n]).sum())
    return (num / den) ** .5, mx, ne
L = lambda f: torch.load(f"/root/runs/{f}.pt")
ga, l2, gaf, l2f = L("grads_ga"), L("grads_l2l"), L("grads_ga_fp32"), L("grads_l2l_fp32")
for s in range(3):
    print(f"step {s}:")
    for name, a, b in (("GA(fp32) vs L2L(fp32)", gaf, l2f), ("GA(bf16) vs GA(fp32)", ga, gaf), ("L2L(bf16) vs L2L(fp32)", l2, l2f)):
        r, mx, ne = rel(a, b, s)
        print(f"  {name:<24} rel_l2 {r:.3e}  max|d| {mx:.3e}  differing elements {ne}")
