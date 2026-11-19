import collections
import re
import sys

import torch

D = sys.argv[1]
a = torch.load(f"{D}/A1d.grads.pt")
for other in ("B1g2d", "B1d"):
    b = torch.load(f"{D}/{other}.grads.pt")
    bad = []
    for n in a:
        x, y = a[n], b[n]
        eq = x.shape == y.shape and torch.equal(
            x.view(torch.int16 if x.element_size() == 2 else torch.int32),
            y.view(torch.int16 if y.element_size() == 2 else torch.int32),
        )
        if not eq:
            xf, yf = x.double(), y.double()
            bad.append((n, ((xf - yf).norm() / xf.norm()).item(), (x != y).float().mean().item()))
    print(other, "tensors", len(a), "different", len(bad), "dtype", next(iter(a.values())).dtype)
    byk = collections.Counter(
        re.sub(r"layers\.\d+\.", "layers.N.", n).replace("base_model.model.model.", "")
        for n, _, _ in bad
    )
    for k, v in sorted(byk.items()):
        print("  ", v, k)
    bl = collections.Counter(
        int(re.search(r"layers\.(\d+)", n).group(1)) for n, _, _ in bad if "layers." in n
    )
    print("  by layer", sorted(bl.items()))
    for n, r, f in sorted(bad, key=lambda t: -t[1])[:3]:
        print("  worst", n[-60:], f"rel={r:.2e} frac_diff={f:.3f}")
    print("  same:", [n[-50:] for n in a if n not in {b_[0] for b_ in bad}][:6])
