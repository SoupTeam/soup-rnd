# ruff: noqa
# One-off ticket A3-05 diagnostic, run on the Vast instance; kept as evidence, not linted.
import torch
def rel(a, b):
    num = den = 0.0; mx = 0.0
    for s in range(len(a)):
        for n, g in a[s].items():
            d = g.float() - b[s][n].float()
            num += d.pow(2).sum().item(); den += g.float().pow(2).sum().item(); mx = max(mx, d.abs().max().item())
    return (num / den) ** .5, mx
def step_rel(a, b, s):
    num = den = 0.0
    for n, g in a[s].items():
        d = g.float() - b[s][n].float(); num += d.pow(2).sum().item(); den += g.float().pow(2).sum().item()
    return (num / den) ** .5
L = lambda f: torch.load(f"/root/runs/{f}.pt")
ga, l2, gae, l2e = L("grads_ga"), L("grads_l2l"), L("grads_ga_eff"), L("grads_l2l_eff")
for s in range(3):
    print(f"step {s}: GA(math) vs L2L(math) {step_rel(ga,l2,s):.3e} | GA(math) vs GA(eff) {step_rel(ga,gae,s):.3e} | L2L(math) vs L2L(eff) {step_rel(l2,l2e,s):.3e} | GA(eff) vs L2L(eff) {step_rel(gae,l2e,s):.3e}")
