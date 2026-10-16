"""Small D1 candidate screen: correctness before any timing/model download."""

from __future__ import annotations

import argparse
import os
import sys
from importlib.metadata import version
from pathlib import Path

from d1_rules import compare_tensors, digest, write_new_json
from rich.console import Console


def numerical_case(
    device: str, dtype: str, nested: bool, shape: tuple, witness_root: Path | None = None
) -> dict:
    import bitsandbytes as bnb
    import bitsandbytes.functional as functional
    import torch
    from d1_fused_nf4 import linear

    torch.manual_seed(842)
    precision = getattr(torch, dtype)
    m, n, k = shape
    weight = torch.randn(n, k, device=device, dtype=precision)
    packed, state = functional.quantize_4bit(weight, quant_type="nf4", compress_statistics=nested)
    value = torch.randn(m, k, device=device, dtype=precision, requires_grad=True)
    dy = torch.randn(m, n, device=device, dtype=precision)
    reference_x = value.detach().clone().requires_grad_(True)
    if device == "cuda":
        reference = bnb.matmul_4bit(reference_x, packed.t(), quant_state=state)
    else:
        reference = torch.nn.functional.linear(
            reference_x, functional.dequantize_4bit(packed, state)
        )
    reference.backward(dy)
    result = linear(value, packed, state)
    result.backward(dy)
    comparison = compare_tensors(
        {"forward": result.detach(), "dx": value.grad},
        {"forward": reference.detach(), "dx": reference_x.grad},
    )
    # Identity inputs isolate decode/rounding from reduction order. The full
    # dense result exists only in this untimed oracle, never as a kernel weight.
    decoded = linear(torch.eye(k, dtype=precision, device=device), packed, state).t()
    native_decoded = functional.dequantize_4bit(packed, state)
    decode = compare_tensors({"weight": decoded}, {"weight": native_decoded})
    row = {
        "dtype": dtype,
        "nested": nested,
        "shape": list(shape),
        "comparison": comparison,
        "decode": decode,
    }
    if witness_root is not None:
        witness_root.mkdir(parents=True, exist_ok=True)
        path = witness_root / f"{dtype}-nested{int(nested)}-{m}x{n}x{k}.pt"
        tensors = {
            "candidate_forward": result,
            "native_forward": reference,
            "candidate_dx": value.grad,
            "native_dx": reference_x.grad,
            "candidate_decoded": decoded,
            "native_decoded": native_decoded,
            "input": value,
            "dy": dy,
            "packed": packed,
            **{"state_" + name: tensor for name, tensor in state.as_dict(packed=True).items()},
        }
        torch.save({name: tensor.detach().cpu() for name, tensor in tensors.items()}, path)
        row["witness"] = {"file": path.name, "sha256": digest(path)}
    return row


def alias_case(device: str) -> dict:
    import bitsandbytes.functional as functional
    import torch
    from d1_fused_nf4 import linear
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(842)
    weights = [torch.randn(32, 128, dtype=torch.float16, device=device) for _index in range(2)]
    pairs = [
        functional.quantize_4bit(w, quant_type="nf4", compress_statistics=True) for w in weights
    ]
    packed, state = pairs[0]
    other_packed, _other_state = pairs[1]
    source = torch.randn(3, 128, dtype=torch.float16, device=device)
    dy = torch.randn(3, 32, dtype=torch.float16, device=device)
    reference = source.clone().requires_grad_(True)
    reference_out = linear(reference, packed.clone(), state)
    reference_out.backward(dy)
    pool = packed.clone()
    value = source.clone().requires_grad_(True)
    calls = []

    def forward(current):
        calls.append(True)
        pool.copy_(packed)
        return linear(current, pool, state)

    output = checkpoint(forward, value, use_reentrant=False)
    pool.copy_(other_packed)
    output.backward(dy)
    comparison = compare_tensors(
        {"forward": output.detach(), "dx": value.grad},
        {"forward": reference_out.detach(), "dx": reference.grad},
    )

    class Broken(torch.autograd.Function):
        @staticmethod
        def forward(ctx, current, mutable):
            # Deliberate control: pool stored outside save_for_backward.
            ctx.mutable = mutable
            return linear(current, mutable, state)

        @staticmethod
        def backward(ctx, gradient):
            dense = functional.dequantize_4bit(ctx.mutable, state)
            return gradient @ dense, None

    broken_x = source.clone().requires_grad_(True)
    broken_pool = packed.clone()
    broken = Broken.apply(broken_x, broken_pool)
    broken_pool.copy_(other_packed)
    broken.backward(dy)
    control_detected = bool(torch.isfinite(broken_x.grad).all()) and not torch.equal(
        broken_x.grad, reference.grad
    )
    # This fixture is small by design. It does NOT replace the existing >165 MiB
    # physical-GPU alias gate or the full streamed decoder/LoRA-gradient gate.
    return {
        "comparison": comparison,
        "recompute_calls": len(calls),
        "negative_control_detected": control_detected,
        "historical_large_alias_gate": "NOT_RUN",
    }


def run(device: str, witness_root: Path | None = None) -> dict:
    if device == "cpu":
        os.environ["TRITON_INTERPRET"] = "1"
    elif os.environ.get("TRITON_INTERPRET") == "1":
        raise ValueError("unset TRITON_INTERPRET for a real CUDA gate")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import triton

    torch.set_num_threads(2)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not label CPU execution a GPU gate")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    rows = []
    dtypes = ("float16", "bfloat16") if device == "cuda" else ("float16",)
    for dtype in dtypes:
        for nested in (False, True):
            for shape in ((3, 32, 128), (17, 67, 96)):
                row = numerical_case(device, dtype, nested, shape, witness_root)
                rows.append(row)
                Console().print(
                    {
                        "shape": shape,
                        "dtype": dtype,
                        "nested": nested,
                        "exact": row["comparison"]["passed"],
                        "decode_exact": row["decode"]["passed"],
                    }
                )
                # A failed physical-GPU gate stops before larger work/timing.
                if device == "cuda" and not (
                    row["comparison"]["passed"] and row["decode"]["passed"]
                ):
                    break
            else:
                continue
            break
        else:
            continue
        break
    exact = all(row["comparison"]["passed"] and row["decode"]["passed"] for row in rows)
    alias = alias_case(device) if device == "cpu" or exact else {"status": "NOT_RUN"}
    passed = exact and alias["comparison"]["passed"] and alias["recompute_calls"] >= 2
    passed = passed and alias["negative_control_detected"]
    return {
        "status": ("SMALL_CUDA_GATE_PASS" if passed else "SMALL_CUDA_GATE_FAIL")
        if device == "cuda"
        else "CPU_LOGIC_ONLY",
        "small_gate_exact": passed,
        "d1_complete": False,
        "device": device,
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid) if device == "cuda" else None,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "source_sha": os.environ.get("D1_SOURCE_SHA"),
        "reference": "unpatched native bitsandbytes" if device == "cuda" else "CPU dense oracle",
        "packages": {name: version(name) for name in ("torch", "bitsandbytes")},
        "triton_version": triton.__version__,
        "candidate_sha256": digest(Path(__file__).with_name("d1_fused_nf4.py")),
        "screen_sha256": digest(Path(__file__)),
        "cases": rows,
        "small_alias": alias,
        "performance_measured": False,
        "large_alias_and_streamed_decoder_gate": "NOT_RUN",
        "bf16_cpu_interpreter": "UNSUPPORTED; raw uint16 dot is not BF16 arithmetic"
        if device == "cpu"
        else "NOT_USED",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("retain earlier gate results; select a fresh output path")
    witness_root = args.output.with_suffix(".witnesses")
    if witness_root.exists():
        raise FileExistsError("retain earlier witnesses; select a fresh output path")
    result = run(args.device, witness_root)
    write_new_json(args.output, result)
    Console().print({"status": result["status"], "small_gate_exact": result["small_gate_exact"]})
    if not result["small_gate_exact"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
