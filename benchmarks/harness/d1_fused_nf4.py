"""Experimental D1 tiled NF4 forward/dX; never selected by production dispatch.

packed_ptr weights and nested scales are decoded inside the dot-product tile.
There is no full dense-weight allocation or atomic/split-k reduction. This is
a candidate, not a validated CUDA implementation. Use the quick gate first.
"""

from __future__ import annotations

import os
from typing import Any

_KERNEL = None
_FUNCTION = None


def _kernel() -> Any:
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def tiled_nf4(
        input_ptr,
        packed_ptr,
        absmax_ptr,
        code_ptr,
        state2_absmax_ptr,
        state2_code_ptr,
        offset_ptr,
        output_ptr,
        m: tl.constexpr,
        n: tl.constexpr,
        k: tl.constexpr,
        blocksize: tl.constexpr,
        nested: tl.constexpr,
        state2_block: tl.constexpr,
        transpose: tl.constexpr,
        weight_dtype: tl.constexpr,
        bm: tl.constexpr,
        bn: tl.constexpr,
        bk: tl.constexpr,
    ):
        rows = (tl.program_id(0) * bm + tl.arange(0, bm)).to(tl.int64)
        columns = (tl.program_id(1) * bn + tl.arange(0, bn)).to(tl.int64)
        reduction = tl.arange(0, bk)
        reduce_size: tl.constexpr = n if transpose else k
        output_size: tl.constexpr = k if transpose else n
        accumulator = tl.full((bm, bn), 0, tl.float32)
        for start in range(tl.cdiv(reduce_size, bk)):
            inner = (start * bk + reduction).to(tl.int64)
            inputs = tl.load(
                input_ptr + rows[:, None] * reduce_size + inner[None, :],
                (rows[:, None] < m) & (inner[None, :] < reduce_size),
                other=0,
            )
            if transpose:
                positions = inner[:, None] * k + columns[None, :]
            else:
                positions = columns[None, :] * k + inner[:, None]
            valid = (inner[:, None] < reduce_size) & (columns[None, :] < output_size)
            packed = tl.load(packed_ptr + positions // 2, valid, other=0).to(tl.int32)
            indices = tl.where(positions % 2 == 0, packed >> 4, packed & 15)
            values = tl.load(code_ptr + indices)
            scale_indices = positions // blocksize
            scales = tl.load(absmax_ptr + scale_indices, valid, other=0)
            if nested:
                decoded = tl.load(state2_code_ptr + scales.to(tl.int32))
                nested_scale = tl.load(
                    state2_absmax_ptr + scale_indices // state2_block,
                    valid,
                    other=0,
                )
                scales = decoded * nested_scale + tl.load(offset_ptr)
            # Match dequantize_4bit's weight rounding BEFORE multiplying by x.
            weights = (values * scales).to(weight_dtype).to(inputs.dtype)
            weights = tl.where(valid, weights, 0)
            accumulator = tl.dot(inputs, weights, accumulator, input_precision="ieee")
        tl.store(
            output_ptr + rows[:, None] * output_size + columns[None, :],
            accumulator,
            (rows[:, None] < m) & (columns[None, :] < output_size),
        )

    _KERNEL = tiled_nf4
    return _KERNEL


def _matmul(value: Any, saved: tuple, metadata: tuple, transpose: bool) -> Any:
    import torch

    n, k, blocksize, nested, state2_block, weight_dtype = metadata
    reduce_size, output_size = (n, k) if transpose else (k, n)
    if value.shape[-1] != reduce_size:
        raise ValueError("NF4 input/weight dimensions differ")
    if value.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("NF4 candidate requires floating activations")
    if value.device.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        raise RuntimeError("CUDA required; CPU interpreter is logic preparation only")
    if value.device.type == "cpu" and value.dtype == torch.bfloat16:
        raise RuntimeError("BF16 is not supported reliably by the Triton CPU interpreter")
    import triton
    import triton.language as tl
    value = value.contiguous()
    m = value.numel() // reduce_size
    result = torch.empty((*value.shape[:-1], output_size), dtype=value.dtype, device=value.device)
    types = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}
    _kernel()[(triton.cdiv(m, 16), triton.cdiv(output_size, 32))](
        value,
        *saved,
        result,
        m,
        n,
        k,
        blocksize,
        nested,
        state2_block,
        transpose,
        types[weight_dtype],
        16,
        32,
        64,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return result


def _function() -> Any:
    global _FUNCTION
    if _FUNCTION is not None:
        return _FUNCTION
    import torch

    class FusedNF4(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, packed, absmax, code, state2_absmax, state2_code, offset, metadata):
            saved = (packed, absmax, code, state2_absmax, state2_code, offset)
            ctx.save_for_backward(*saved)
            ctx.nf4_meta = metadata
            ctx.owner_check = getattr(packed, "_soup_stream_owner_check", None)
            return _matmul(value, saved, metadata, False)

        @staticmethod
        def backward(ctx, grad_output):
            if ctx.owner_check is not None:
                ctx.owner_check()
            # Access through saved_tensors triggers checkpoint recomputation.
            grad_value = _matmul(grad_output, ctx.saved_tensors, ctx.nf4_meta, True)
            return (grad_value,) + (None,) * 7

    _FUNCTION = FusedNF4
    return _FUNCTION


def linear(value: Any, packed: Any, state: Any, bias: Any = None) -> Any:
    """Candidate only: all frozen tensor state goes through save_for_backward."""
    import torch

    if state.quant_type != "nf4" or len(state.shape) != 2:
        raise ValueError("candidate supports canonical 2D NF4 weights only")
    if packed.dtype != torch.uint8 or not packed.is_contiguous():
        raise ValueError("candidate requires contiguous uint8 packed weights")
    n, k = map(int, state.shape)
    if packed.numel() != (n * k + 1) // 2 or min(n, k) <= 0:
        raise ValueError("invalid packed NF4 extent")
    blocksize = int(state.blocksize)
    if blocksize <= 0 or state.code.numel() != 16 or state.code.dtype != torch.float32:
        raise ValueError("invalid NF4 blocksize/code table")
    blocks = (n * k + blocksize - 1) // blocksize
    if state.absmax.numel() != blocks or state.absmax.dtype != (
        torch.uint8 if state.nested else torch.float32
    ):
        raise ValueError("invalid NF4 scale extent/dtype")
    saved = [packed, state.absmax, state.code]
    empty = state.code.new_empty((0,))
    if state.nested:
        saved.extend((state.state2.absmax, state.state2.code, state.offset))
        state2_block = int(state.state2.blocksize)
        if (
            state2_block <= 0
            or state.state2.absmax.numel() != (blocks + state2_block - 1) // state2_block
            or state.state2.code.numel() != 256
            or state.offset.numel() != 1
            or any(t.dtype != torch.float32 for t in saved[3:])
        ):
            raise ValueError("invalid nested NF4 state")
    else:
        saved.extend((empty, empty, empty))
        state2_block = 0
    if any(tensor.device != value.device for tensor in saved):
        raise ValueError("all NF4 state must be on the input device")
    metadata = (n, k, blocksize, bool(state.nested), state2_block, state.dtype)
    output = _function().apply(value, *saved, metadata)
    return output if bias is None else output + bias
