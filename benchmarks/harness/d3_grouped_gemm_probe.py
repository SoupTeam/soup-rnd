"""Exploratory D3 BF16 arithmetic check, before NF4/kernel integration or timing.

Original small masked grouped-GEMM implementation; the Triton grouped-GEMM
tutorial informed scheduling only. This is not the D3 candidate or a formal gate.
It checks native eager arithmetic before investing in dequantization/group leases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SHAPES = (
    (32, 32, (0, 1, 7, 17)),
    (33, 47, (1, 17, 31, 33)),
    (2048, 2048, (1, 17, 63, 129)),
    (1024, 2048, (1, 17, 64, 128)),
)


def _build_kernel() -> Any:
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(
        input_ptrs, weight_ptrs, output_ptrs, row_counts,
        inner: tl.constexpr, columns: tl.constexpr,
        block_rows: tl.constexpr, block_columns: tl.constexpr, block_inner: tl.constexpr,
    ):
        group = tl.program_id(2)
        rows = tl.load(row_counts + group)
        row = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
        column = tl.program_id(1) * block_columns + tl.arange(0, block_columns)
        reduction = tl.arange(0, block_inner)
        inputs = tl.load(input_ptrs + group).to(tl.pointer_type(tl.bfloat16))
        weights = tl.load(weight_ptrs + group).to(tl.pointer_type(tl.bfloat16))
        outputs = tl.load(output_ptrs + group).to(tl.pointer_type(tl.bfloat16))
        accum = tl.zeros((block_rows, block_columns), dtype=tl.float32)
        for tile in range(tl.cdiv(inner, block_inner)):
            offset = tile * block_inner + reduction
            left = tl.load(
                inputs + row[:, None] * inner + offset[None, :],
                (row[:, None] < rows) & (offset[None, :] < inner), other=0,
            )
            right = tl.load(
                weights + column[None, :] * inner + offset[:, None],
                (column[None, :] < columns) & (offset[:, None] < inner), other=0,
            )
            accum = tl.dot(left, right, accum)
        tl.store(
            outputs + row[:, None] * columns + column[None, :],
            accum.to(tl.bfloat16), (row[:, None] < rows) & (column[None, :] < columns),
        )

    return kernel


def grouped_linear(inputs: list, weights: list, kernel: Any) -> list:
    import torch
    import triton

    if not inputs or len(inputs) != len(weights):
        raise ValueError("one input and weight tensor per group required")
    inner, columns = weights[0].shape[1], weights[0].shape[0]
    for hidden, weight in zip(inputs, weights):
        if (hidden.ndim != 2 or weight.shape != (columns, inner)
                or hidden.shape[1] != inner or not hidden.is_contiguous()
                or not weight.is_contiguous() or hidden.dtype != torch.bfloat16
                or weight.dtype != hidden.dtype or hidden.device.type != "cuda"
                or weight.device != hidden.device or hidden.device != inputs[0].device):
            raise ValueError("grouped probe requires contiguous compatible CUDA BF16 matrices")
    outputs = [torch.empty((hidden.shape[0], columns), device=hidden.device, dtype=hidden.dtype)
               for hidden in inputs]
    max_rows = max(hidden.shape[0] for hidden in inputs)
    if max_rows:
        pointers = [torch.tensor([tensor.data_ptr() for tensor in values],
                                 dtype=torch.int64, device=inputs[0].device)
                    for values in (inputs, weights, outputs)]
        rows = torch.tensor([hidden.shape[0] for hidden in inputs],
                            dtype=torch.int32, device=inputs[0].device)
        kernel[(triton.cdiv(max_rows, 32), triton.cdiv(columns, 64), len(inputs))](
            *pointers, rows, inner, columns, 32, 64, 32,
            num_warps=4, enable_fp_fusion=False,
        )
    return outputs


def _hash(tensor: Any) -> str:
    import torch

    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy()).hexdigest()


def torch_grouped_linear(inputs: list, weights: list) -> list:
    """Probe the existing public backend with jagged rows and only active groups."""
    import torch
    import torch.nn.functional as functional

    selected = [(hidden, weight) for hidden, weight in zip(inputs, weights)
                if hidden.shape[0]]
    if not selected:
        return [hidden.new_empty((0, weight.shape[0])) for hidden, weight in zip(inputs, weights)]
    # The public grouped_mm contract ignores the tail beyond the final offset.
    # Retain a sentinel row so the final offset is strictly inside the input.
    sentinel = inputs[0].new_zeros((1, inputs[0].shape[1]))
    hidden = torch.cat([pair[0] for pair in selected] + [sentinel])
    packed = torch.stack([pair[1] for pair in selected]).transpose(-1, -2)
    counts = [pair[0].shape[0] for pair in selected]
    offsets = torch.tensor(counts, device=hidden.device, dtype=torch.int32).cumsum(0).int()
    output = functional.grouped_mm(hidden, packed, offs=offsets)
    parts = iter(output[:sum(counts)].split(counts))
    return [next(parts) if value.shape[0] else value.new_empty((0, weight.shape[0]))
            for value, weight in zip(inputs, weights)]


def _run_checks(backend: str = "triton", progress: Any = None) -> dict:
    import torch
    import torch.nn.functional as functional
    import triton

    if (sys.platform != "linux" or not torch.cuda.is_available()
            or not torch.cuda.is_bf16_supported(including_emulation=False)
            or torch.cuda.get_device_capability(0)[0] < 8):
        raise RuntimeError("Linux sm_80+ native BF16 GPU required")
    kernel = _build_kernel() if backend == "triton" else None
    operation = (
        (lambda hidden, weight: grouped_linear(hidden, weight, kernel))
        if backend == "triton" else torch_grouped_linear
    )
    records = []
    unsupported = []
    for seed in (11, 29):
        for inner, columns, row_counts in SHAPES:
            if backend == "torch-grouped" and (inner * 2 % 16 or columns * 2 % 16):
                unsupported.append({
                    "seed": seed, "inner": inner, "columns": columns,
                    "reason": "public backend requires 16-byte-aligned leading strides",
                })
                continue
            generator = torch.Generator().manual_seed(seed)
            inputs = [(torch.randn(rows, inner, generator=generator) * 0.2).to(
                device="cuda", dtype=torch.bfloat16
            ) for rows in row_counts]
            weights = [(torch.randn(columns, inner, generator=generator) * 0.2).to(
                device="cuda", dtype=torch.bfloat16
            ) for _ in row_counts]
            with torch.no_grad():
                expected = [functional.linear(hidden, weight)
                            for hidden, weight in zip(inputs, weights)]
                repeat = [functional.linear(hidden, weight)
                          for hidden, weight in zip(inputs, weights)]
                grouped = operation(inputs, weights)
                single = [operation([hidden], [weight])[0]
                          for hidden, weight in zip(inputs, weights)]
                negative = [functional.linear(hidden, weight + 0.125)
                            for hidden, weight in zip(inputs, weights)]
            torch.cuda.synchronize()
            for group, (reference, again, observed, alone, changed) in enumerate(zip(
                expected, repeat, grouped, single, negative
            )):
                empty = reference.numel() == 0
                records.append({
                    "seed": seed, "inner": inner, "columns": columns,
                    "rows": row_counts[group], "group": group,
                    "native_repeat_equal": torch.equal(reference, again),
                    "native_equal": torch.equal(reference, observed),
                    "group_width_equal": torch.equal(alone, observed),
                    "finite": bool(torch.isfinite(observed).all()),
                    "negative_differs": None if empty else not torch.equal(reference, changed),
                    "max_abs_diff": 0.0 if empty else float(
                        (reference.float() - observed.float()).abs().max()
                    ),
                    "unequal_elements": int(torch.count_nonzero(reference != observed)),
                    "native_sha256": _hash(reference), "grouped_sha256": _hash(observed),
                })
                if progress is not None:
                    progress(records)
    exact = all(record["native_equal"] and record["native_repeat_equal"]
                and record["group_width_equal"] and record["finite"]
                and record["negative_differs"] is not False for record in records)
    return {
        "status": ("arithmetic_incomplete" if unsupported else "arithmetic_exact")
        if exact else "arithmetic_mismatch",
        "d3_accepted": False, "formal_measurement": False,
        "kind": "development BF16 GEMM probe; no NF4, expert routing, backward or timing",
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "backend": backend,
        "triton": triton.__version__, "records": records, "unsupported": unsupported,
    }


def run_probe(reduction: str = "current", backend: str = "triton", progress: Any = None) -> dict:
    import torch

    if reduction not in ("current", "fp32-reduction", "fp32-no-splitk"):
        raise ValueError("unknown reduction experiment")
    if backend not in ("triton", "torch-grouped"):
        raise ValueError("unknown grouped backend")
    settings_backend = torch.backends.cuda.matmul
    original = (settings_backend.allow_bf16_reduced_precision_reduction,
                settings_backend.allow_bf16_reduced_precision_reduction_split_k)
    try:
        if reduction == "fp32-reduction":
            settings_backend.allow_bf16_reduced_precision_reduction = False
        elif reduction == "fp32-no-splitk":
            settings_backend.allow_bf16_reduced_precision_reduction = (False, False)
        settings = {
            "experiment": reduction,
            "allow_bf16_reduced_precision_reduction": (
                settings_backend.allow_bf16_reduced_precision_reduction
            ),
            "allow_split_k": settings_backend.allow_bf16_reduced_precision_reduction_split_k,
            "allow_tf32": settings_backend.allow_tf32,
            "scope": "same setting for native reference and grouped probe; process-local",
        }
        result = _run_checks(backend, progress)
        result["reference_settings"] = settings
        return result
    finally:
        settings_backend.allow_bf16_reduced_precision_reduction = original


def main(argv: list[str] | None = None) -> int:
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reduction", default="current", choices=(
        "current", "fp32-reduction", "fp32-no-splitk",
    ))
    parser.add_argument("--backend", default="triton", choices=("triton", "torch-grouped"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "no_verdict", "formal_measurement": False, "d3_accepted": False,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    def progress(records: list) -> None:
        report["records"] = records
        (args.out / "progress.json").write_text(
            json.dumps(records, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )

    try:
        report.update(run_probe(args.reduction, args.backend, progress))
    except (Exception, KeyboardInterrupt) as exc:
        report["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        report["traceback"] = traceback.format_exc()
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    path = args.out / "probe.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    Console().print(f"{report['status']}: {path}")
    return 0 if report["status"] == "arithmetic_exact" else 1


if __name__ == "__main__":
    raise SystemExit(main())
