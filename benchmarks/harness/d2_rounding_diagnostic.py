"""CPU arithmetic experiments for rounding boundaries; not vendor GPU kernel tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console


def relative(left, right) -> float:
    return (left.double() - right.double()).norm().item() / max(left.double().norm().item(), 1e-12)


def diagnose() -> dict:
    import torch
    import torch.nn.functional as functional

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(792)
    rows = []
    # Round every declared low-precision boundary, while keeping CPU matmuls fp32.
    for dtype in (torch.float16, torch.bfloat16):
        scale = 128 if dtype == torch.float16 else 1
        hidden = torch.randn(512, 32, generator=generator).to(dtype).float().requires_grad_()
        weight = (torch.randn(32000, 32, generator=generator) * 0.03).to(dtype).float()
        labels = torch.randint(0, 32000, (512,), generator=generator)
        logits = (hidden @ weight.T).to(dtype).float()
        mean_loss = functional.cross_entropy(logits, labels)
        scaled = torch.autograd.grad(mean_loss * scale, hidden, retain_graph=True)[0]
        reference = scaled.to(dtype).float() / scale
        raw = torch.autograd.grad(mean_loss, hidden, retain_graph=True)[0].to(dtype)
        late_scaled = (raw.float() * scale).to(dtype).float() / scale
        summed = torch.autograd.grad(mean_loss * 512, hidden)[0].to(dtype)
        sum_then_mean = (summed.float() * (scale / 512)).to(dtype).float() / scale
        rows.append({
            "dtype": str(dtype), "experiment": "head-gradient-scale-order",
            "scale": scale, "loss": mean_loss.item(),
            "late_scale_relative_l2": relative(reference, late_scaled),
            "sum_outside_relative_l2": relative(reference, sum_then_mean),
        })
        a = torch.randn(128, 256, generator=generator).to(dtype).float().requires_grad_()
        b = torch.randn(128, 256, generator=generator).to(dtype).float().requires_grad_()
        upstream = torch.randn(128, 256, generator=generator).to(dtype).float()
        split = (functional.silu(a).to(dtype).float() * b).to(dtype).float()
        fused = (functional.silu(a) * b).to(dtype).float()
        split_grads = torch.autograd.grad((split * upstream).sum(), (a, b), retain_graph=True)
        fused_grads = torch.autograd.grad((fused * upstream).sum(), (a, b))
        rows.append({
            "dtype": str(dtype), "experiment": "swiglu-intermediate-rounding",
            "output_relative_l2": relative(split, fused),
            "gate_gradient_relative_l2": relative(split_grads[0], fused_grads[0]),
            "up_gradient_relative_l2": relative(split_grads[1], fused_grads[1]),
        })
        x = torch.randn(128, 64, generator=generator).to(dtype).float()
        a = (torch.randn(4, 64, generator=generator) * 0.02).requires_grad_()
        b = (torch.randn(96, 4, generator=generator) * 0.02).requires_grad_()
        fp32 = functional.linear(functional.linear(x, a), b)
        low = functional.linear(functional.linear(x, a.to(dtype).float()).to(dtype).float(),
                                b.to(dtype).float()).to(dtype).float()
        direction = torch.randn(fp32.shape, generator=generator)
        high_grad = torch.autograd.grad((fp32 * direction).sum(), (a, b), retain_graph=True)
        low_grad = torch.autograd.grad((low * direction).sum(), (a, b))
        rows.append({
            "dtype": str(dtype), "experiment": "fp32-adapter-storage-vs-compute",
            "output_relative_l2": relative(fp32, low),
            "adapter_a_gradient_relative_l2": relative(high_grad[0], low_grad[0]),
            "adapter_b_gradient_relative_l2": relative(high_grad[1], low_grad[1]),
        })
    return {
        "scope": "CPU rounding-boundary emulation; does not execute vendor GPU kernels",
        "seed": 792, "head_shape": [512, 32, 32000], "swiglu_shape": [128, 256],
        "rows": rows,
        "limit": "Shows mechanisms; does not attribute all previous full-model differences.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose()
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    Console().print(result)


if __name__ == "__main__":
    main()
