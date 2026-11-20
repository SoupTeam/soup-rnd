"""Rows of a fp32 [M, H] -> [M, 1] sum differ between M=1 and larger M on CUDA (ticket 06 cause)."""

import torch

torch.manual_seed(0)
H = 2048
for dtype in (torch.bfloat16, torch.float32):
    a = torch.randn(64, H, device="cuda", dtype=dtype)
    b = torch.randn(64, H, device="cuda", dtype=dtype)
    alone = torch.cat([(a[i : i + 1] * b[i : i + 1]).sum(-1, keepdim=True) for i in range(64)])
    differing = {
        m: int(((a[:m] * b[:m]).sum(-1, keepdim=True) != alone[:m]).sum()) for m in (2, 8, 32, 64)
    }
    print(dtype, "rows differing from M=1, by M:", differing)
