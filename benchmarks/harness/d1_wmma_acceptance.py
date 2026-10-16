"""Short checkpoint/large-buffer/streamed-decoder gates for BF16 WMMA only."""

import argparse
import copy
import os
import sys
from pathlib import Path

from d1_rules import compare_tensors, digest, write_new_json
from rich.console import Console


def large_alias(output):
    import bitsandbytes as bnb
    import bitsandbytes.functional as functional
    import torch
    from d1_fused_nf4 import linear
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(99991)
    n, k, m = 32768, 12288, 128
    pairs = []
    for _ in range(2):
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
        pairs.append(functional.quantize_4bit(weight, quant_type="nf4", compress_statistics=True))
        del weight
    (packed, state), (other, other_state) = pairs
    value = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    reference_x = value.clone().requires_grad_(True)
    reference = bnb.matmul_4bit(reference_x, packed.t(), quant_state=state)
    reference.backward(dy)
    pool, pooled_state = packed.clone(), copy.deepcopy(state)
    calls = []

    def refill(source, quant):
        pool.copy_(source)
        pooled_state.absmax.copy_(quant.absmax)
        pooled_state.offset.copy_(quant.offset)
        pooled_state.state2.absmax.copy_(quant.state2.absmax)
        pooled_state.state2.code.copy_(quant.state2.code)

    def body(x):
        calls.append(True)
        refill(packed, state)
        return linear(x, pool, pooled_state, backend="wmma")

    x = value.clone().requires_grad_(True)
    result = checkpoint(body, x, use_reentrant=False)
    refill(other, other_state)
    result.backward(dy)
    comparison = compare_tensors(
        {"forward": result, "dx": x.grad}, {"forward": reference, "dx": reference_x.grad}
    )

    class Broken(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.pool = pool
            return linear(x, pool, pooled_state, backend="wmma")

        @staticmethod
        def backward(ctx, gradient):
            from d1_wmma_nf4 import matmul

            saved = (
                ctx.pool,
                pooled_state.absmax,
                pooled_state.code,
                pooled_state.state2.absmax,
                pooled_state.state2.code,
                pooled_state.offset,
            )
            meta = (n, k, state.blocksize, True, state.state2.blocksize, state.dtype)
            return matmul(gradient, saved, meta, True)

    refill(packed, state)
    broken_x = value.clone().requires_grad_(True)
    broken = Broken.apply(broken_x)
    refill(other, other_state)
    broken.backward(dy)
    detected = bool(torch.isfinite(broken_x.grad).all()) and not torch.equal(
        broken_x.grad, reference_x.grad
    )
    witnesses = {
        "native_forward": reference,
        "native_dx": reference_x.grad,
        "forward": result,
        "dx": x.grad,
        "broken_dx": broken_x.grad,
    }
    path = output / "large-alias.pt"
    torch.save({name: tensor.detach().cpu() for name, tensor in witnesses.items()}, path)
    return {
        "passed": comparison["passed"] and len(calls) == 2 and detected,
        "comparison": comparison,
        "recompute_calls": len(calls),
        "negative_control_detected": detected,
        "shape": [m, n, k],
        "packed_bytes": packed.numel(),
        "above_165_mib": packed.numel() > 165 * 1024**2,
        "witness_sha256": digest(path),
        "differing": {
            "forward": int(torch.count_nonzero(result != reference)),
            "dx": int(torch.count_nonzero(x.grad != reference_x.grad)),
        },
    }


def run(output, full_fp32_reduction=False):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import d1_fused_nf4 as candidate
    import torch
    from d1_wmma_nf4 import matmul

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if full_fp32_reduction:
        # Separate diagnosis, never overwrite the default-math failure or
        # silently change Soup's production numerical settings.
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    original = candidate._matmul
    candidate._matmul = matmul
    try:
        alias = large_alias(output)
        write_new_json(output / "large-alias.json", alias)
        Console().print(alias)
        streams = []
        if alias["passed"]:
            from d1_nf4_study import streaming_control

            from soup_cli.utils import layer_stream_runtime as runtime

            original_linear = runtime.checkpoint_visible_nf4_linear
            original_gate = runtime._can_use_checkpoint_visible_nf4_gemm
            runtime.checkpoint_visible_nf4_linear = lambda *args, **kwargs: candidate.linear(
                *args, **kwargs, backend="wmma"
            )
            runtime._can_use_checkpoint_visible_nf4_gemm = lambda x, state: (
                x.dtype == torch.bfloat16
                and x.numel() // x.shape[-1] >= 128
                and state.shape[0] % 64 == 0
                and state.shape[1] % 32 == 0
            )
            try:
                for seq in (128, 2048):
                    row = streaming_control(output, "cuda", "bfloat16", seq, expected_fused=True)
                    streams.append(row)
                    write_new_json(output / f"stream-{seq}.json", row)
                    Console().print({"seq": seq, "passed": row["passed"]})
                    if not row["passed"]:
                        break
            finally:
                runtime.checkpoint_visible_nf4_linear = original_linear
                runtime._can_use_checkpoint_visible_nf4_gemm = original_gate
    finally:
        candidate._matmul = original
    result = {
        "source_sha": os.environ.get("D1_SOURCE_SHA"),
        "large_alias": alias,
        "streaming": streams,
        "passed": alias["passed"] and len(streams) == 2 and all(r["passed"] for r in streams),
        "d1_complete": False,
        "scope": "BF16 aligned only; native resident model unpatched",
        "allow_bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "default_math_acceptance": not full_fp32_reduction,
    }
    write_new_json(output / "acceptance.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnose-full-fp32-reduction", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a fresh output directory")
    args.output.mkdir(parents=True)
    result = run(args.output, args.diagnose_full_fp32_reduction)
    if not result["passed"]:
        sys.exit(2)
