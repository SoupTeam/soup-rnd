"""Actual FP32 vendor operator forward/backward probes, outside performance timers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

from rich.console import Console

EXPECTED = {
    "liger": {"rmsnorm", "swiglu", "rope", "linear-ce"},
    "unsloth": {"rmsnorm", "linear-ce", "lora-w", "lora-qkv", "lora-mlp"},
}
RELATIVE_LIMIT = 0.001


def retained_pair(a, b):
    """Preserve values and layout-independent shape/dtype for safetensors."""
    return a.detach().cpu().contiguous(), b.detach().cpu().contiguous()


def run(backend: str, output: Path) -> dict:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if backend == "unsloth":
        import unsloth  # noqa: F401
    import torch
    import torch.nn.functional as functional
    from safetensors.torch import save_file

    torch.manual_seed(792)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    tensors, cases = {}, {}

    def sample(*shape):
        return (torch.randn(*shape, device="cuda", dtype=torch.float32) * 0.1).detach()

    def check(name, inputs, reference, candidate):
        left = [item.detach().clone().requires_grad_() for item in inputs]
        right = [item.detach().clone().requires_grad_() for item in inputs]
        expected = reference(*left)
        actual = candidate(*right)
        expected = expected if isinstance(expected, tuple) else (expected,)
        actual = actual if isinstance(actual, tuple) else (actual,)
        upstream = [torch.randn_like(item) for item in expected]
        gradients = [
            torch.autograd.grad(sum((a * b).sum() for a, b in zip(values, upstream)), params)
            for values, params in ((expected, left), (actual, right))
        ]
        rows = []
        for kind, values in (("output", (expected, actual)), ("gradient", gradients)):
            for index, (a, b) in enumerate(zip(*values)):
                a, b = retained_pair(a, b)
                key = f"{name}/{kind}/{index}"
                tensors[key + "/reference"] = a
                tensors[key + "/candidate"] = b
                relative = (a.double() - b.double()).norm().item() / max(
                    a.double().norm().item(), 1e-12
                )
                finite = torch.isfinite(a).all().item() and torch.isfinite(b).all().item()
                rows.append({"quantity": key, "relative_l2": relative,
                             "passed": finite and relative <= RELATIVE_LIMIT})
        cases[name] = {"passed": all(item["passed"] for item in rows), "checks": rows}
        Console().print(backend, name, cases[name]["passed"])

    weight = sample(64).add_(1)
    def norm_ref(x):
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)) * weight
    if backend == "liger":
        from liger_kernel.ops.rope import LigerRopeFunction
        from liger_kernel.ops.swiglu import LigerSiLUMulFunction
        from liger_kernel.transformers.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyLoss,
        )
        from liger_kernel.transformers.rms_norm import LigerRMSNorm

        norm = LigerRMSNorm(64, in_place=False).cuda()
        norm.weight.data.copy_(weight)
        norm.weight.requires_grad_(False)
        check("rmsnorm", [sample(2, 8, 64)], norm_ref, norm)
        check("swiglu", [sample(8, 64), sample(8, 64)],
              lambda a, b: functional.silu(a) * b,
              lambda a, b: LigerSiLUMulFunction.apply(a, b, 1.0, 1.0))
        head, labels = sample(257, 32), torch.randint(0, 257, (17,), device="cuda")
        loss = LigerFusedLinearCrossEntropyLoss(reduction="sum")
        check("linear-ce", [sample(17, 32)],
              lambda x: functional.cross_entropy(x @ head.T, labels),
              lambda x: loss(head, x, labels) / 17)
        angle = sample(1, 8, 8).repeat(1, 1, 2)
        cosine, sine = angle.cos(), angle.sin()

        def rope_reference(q, k):
            def rotate(x):
                half = x.shape[-1] // 2
                return torch.cat((-x[..., half:], x[..., :half]), dim=-1)
            return tuple(x * cosine[:, None] + rotate(x) * sine[:, None] for x in (q, k))

        check("rope", [sample(1, 4, 8, 16), sample(1, 2, 8, 16)], rope_reference,
              lambda q, k: LigerRopeFunction.apply(q, k, cosine, sine, None, 1))
    else:
        from unsloth.kernels.fast_lora import LoRA_MLP, LoRA_QKV, LoRA_W
        from unsloth.kernels.rms_layernorm import Fast_RMS_Layernorm
        from unsloth.kernels.swiglu import swiglu_DWf_DW_dfg_kernel, swiglu_fg_kernel
        from unsloth_zoo.fused_losses import unsloth_fused_ce_loss

        check("rmsnorm", [sample(2, 8, 64)], norm_ref,
              lambda x: Fast_RMS_Layernorm.apply(x, weight, 1e-6, False))
        head, labels = sample(257, 32), torch.randint(0, 257, (1, 17), device="cuda")
        check("linear-ce", [sample(1, 17, 32)],
              lambda x: functional.cross_entropy((x @ head.T).view(-1, 257), labels.view(-1)),
              lambda x: unsloth_fused_ce_loss(
                  trainer=None, hidden_states=x, lm_head_weight=head, lm_head_bias=None,
                  labels=labels, scaling=128.0, torch_compile=False, shift_labels=False))

        def linear(x, base, a, b):
            return functional.linear(x, base) + functional.linear(functional.linear(x, a), b) * 2

        base = sample(32, 32)
        check("lora-w", [sample(1, 8, 32), sample(4, 32), sample(32, 4)],
              lambda x, a, b: linear(x, base, a, b),
              lambda x, a, b: LoRA_W.apply(x, base, None, a, b, 2))
        bases = [sample(32, 32) for _ in range(3)]

        def qkv_ref(x, *adapters):
            return tuple(linear(x, w, adapters[i * 2], adapters[i * 2 + 1])
                         for i, w in enumerate(bases))

        def qkv_fast(x, *adapters):
            args = []
            for i, w in enumerate(bases):
                args.extend((w, None, adapters[i * 2], adapters[i * 2 + 1], 2))
            return LoRA_QKV.apply(x, *args, True)

        adapters = [item for _ in range(3) for item in (sample(4, 32), sample(32, 4))]
        check("lora-qkv", [sample(1, 8, 32), *adapters], qkv_ref, qkv_fast)
        gate, up, down = sample(64, 32), sample(64, 32), sample(32, 64)

        def mlp_ref(x, ga, gb, ua, ub, da, db):
            return linear(functional.silu(linear(x, gate, ga, gb)) * linear(x, up, ua, ub),
                          down, da, db)

        def mlp_fast(x, ga, gb, ua, ub, da, db):
            return LoRA_MLP.apply(
                x, gate, None, ga, gb, 2, up, None, ua, ub, 2, down, None, da, db, 2,
                swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel, True)

        check("lora-mlp", [sample(1, 8, 32), sample(4, 32), sample(64, 4),
                           sample(4, 32), sample(64, 4), sample(4, 64), sample(32, 4)],
              mlp_ref, mlp_fast)
    save_file(tensors, str(output / "operator-pairs.safetensors"))
    result = {
        "backend": backend, "dtype": "float32", "relative_limit": RELATIVE_LIMIT,
        "cases": cases, "passed": set(cases) == EXPECTED[backend]
        and all(item["passed"] for item in cases.values()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "packages": {name: version(name) for name in ("torch", "triton", "liger-kernel",
                                                      "unsloth", "unsloth_zoo")},
        "scope": "specified FP32 operators only; not full-model precision equivalence",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def verify(root: Path, backend: str) -> dict:
    import torch
    from safetensors.torch import load_file

    result = json.loads((root / "result.json").read_text())
    if result["backend"] != backend or set(result["cases"]) != EXPECTED[backend]:
        raise ValueError("Missing/different operator cases")
    if result["relative_limit"] != RELATIVE_LIMIT or result["dtype"] != "float32":
        raise ValueError("Changed operator criterion")
    counts = {"rmsnorm": (1, 1), "swiglu": (1, 2), "rope": (2, 2),
              "linear-ce": (1, 1), "lora-w": (1, 3), "lora-qkv": (3, 7), "lora-mlp": (1, 7)}
    required = {
        f"{case}/{kind}/{index}/{arm}"
        for case in EXPECTED[backend]
        for kind, count in zip(("output", "gradient"), counts[case])
        for index in range(count)
        for arm in ("reference", "candidate")
    }
    tensors = load_file(str(root / "operator-pairs.safetensors"))
    if set(tensors) != required:
        raise ValueError("Incomplete operator tensors")
    for key in sorted(required):
        if not key.endswith("/reference"):
            continue
        a, b = tensors[key], tensors[key.removesuffix("reference") + "candidate"]
        if a.dtype != torch.float32 or b.dtype != a.dtype or a.shape != b.shape:
            raise ValueError("Wrong operator shape/dtype")
        relative = (a.double() - b.double()).norm().item() / max(a.double().norm().item(), 1e-12)
        if not torch.isfinite(a).all() or not torch.isfinite(b).all() or relative > RELATIVE_LIMIT:
            raise ValueError("Operator replay failure: " + key)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=EXPECTED, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return 0 if run(args.backend, args.output)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
