"""Compare saved bits tensors pairwise: torch.equal on the loss and on every LoRA gradient.

Usage (from any folder; the tensors are outside git):
    python compare_repeats.py TENSOR.pt [TENSOR.pt ...]

Prints, for every pair, whether all tensors are equal, how many gradients differ, and the largest
absolute difference against the largest absolute gradient. Only the arguments are read, so pass
the files explicitly. Loads on CPU.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path


def main(paths: list[str]) -> int:
    import torch

    if len(paths) < 2:
        print(__doc__)
        return 2
    runs = {Path(p).parent.parent.name + "/" + Path(p).name: torch.load(p, weights_only=True)
            for p in paths}
    for (name_a, a), (name_b, b) in itertools.combinations(runs.items(), 2):
        names = sorted(a["grads"])
        same_names = names == sorted(b["grads"])
        loss_equal = bool(torch.equal(a["loss"], b["loss"]))
        differing = [n for n in names if same_names and
                     not torch.equal(a["grads"][n], b["grads"][n])]
        worst = max((a["grads"][n].double() - b["grads"][n].double()).abs().max().item()
                    for n in differing) if differing else 0.0
        scale = max(a["grads"][n].double().abs().max().item() for n in names)
        print(f"{name_a} vs {name_b}: loss equal={loss_equal} "
              f"grads differing={len(differing)} of {len(names)} "
              f"max abs diff={worst:.6g} max abs grad={scale:.6g} "
              f"all equal={loss_equal and not differing and same_names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
