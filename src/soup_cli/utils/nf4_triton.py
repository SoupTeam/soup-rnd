"""One-launch NF4 dequantization in Triton, as a drop-in for ``bitsandbytes.functional.dequantize_4bit``.

bitsandbytes dequantizes a double-quantized NF4 weight in two launches: the 8-bit absmax first
(``dequantize_blockwise``, then ``+ offset``), then the 4-bit codes. Every 4-bit linear does this
once in the forward and again in the backward, and on Qwen3.5-9B (stage 2 of ZERO-2, RTX 4090)
``kDequantizeBlockwise`` was the largest CUDA item after the GEMMs: 368 ms over two steps, about
11 % of a step. Here one launch does both levels: each lane reads one packed byte, its two 4-bit
codes, the block's 8-bit absmax and that block's second-level scale, and writes two adjacent
weights ``code * (code2[absmax8] * absmax2 + offset)``. The arithmetic is bitsandbytes' own in
fp32 (the absmax product rounded on its own, never contracted into an FMA, then the offset), so
the weight is meant to be bit-identical; ``tests/test_nf4_triton.py`` checks that against
bitsandbytes, and the ZERO-2 stage-2b runner checks it on the GPU for every weight of a layer.

Opt-in: ``SOUP_NF4_TRITON=1`` (``enable_from_env``) replaces ``dequantize_4bit`` in
``bitsandbytes.functional`` for the process. Anything the kernel does not cover (fp4, non-CUDA
tensors, an odd element count, an output dtype other than fp16 / bf16 / fp32) goes to the
original function unchanged.

Adapted from the Soup Kernel Challenge entry of team AG TEAM (Decentrathon 6.0,
github.com/BAITC-Hacks/hack-2227fe74-ag-team, ``solution/nf4.py``, Apache-2.0), which measured
it bit-identical to bitsandbytes on a T4 for Qwen2.5-1.5B. Changes: any output dtype (bf16 for
Qwen3.5), any second-level block size, the drop-in wrapper; their rounded-backward option and
q/k/v grouping are not taken.
"""

from __future__ import annotations

import os
from typing import Any

ENV = "SOUP_NF4_TRITON"
BLOCK = 2048  # packed bytes per program: 4096 weights

_original: Any = None
_kernel: Any = None


def _build_kernel() -> Any:
    import triton
    import triton.language as tl

    @triton.jit
    def nf4_dequant_kernel(packed_ptr, absmax_ptr, absmax2_ptr, code2_ptr, offset_ptr, code_ptr,
                           out_ptr, n_bytes, NESTED: tl.constexpr, BLOCKSIZE: tl.constexpr,
                           BLOCKSIZE2: tl.constexpr, BLOCK: tl.constexpr,
                           EXACT_MUL: tl.constexpr):
        pid = tl.program_id(0).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_bytes
        byte = tl.load(packed_ptr + offs, mask=mask, other=0).to(tl.int32)
        high = tl.load(code_ptr + (byte >> 4), mask=mask, other=0.0)
        low = tl.load(code_ptr + (byte & 15), mask=mask, other=0.0)
        block = offs // (BLOCKSIZE // 2)
        if NESTED:
            absmax8 = tl.load(absmax_ptr + block, mask=mask, other=0).to(tl.int32)
            c2 = tl.load(code2_ptr + absmax8, mask=mask, other=0.0)
            a2 = tl.load(absmax2_ptr + block // BLOCKSIZE2, mask=mask, other=0.0)
            if EXACT_MUL:  # mul.rn is never contracted into an FMA with the add below
                prod = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=r,r,r", [c2, a2],
                                                 dtype=tl.float32, is_pure=True, pack=1)
            else:  # the Triton interpreter (CPU tests) has no FMA contraction to avoid
                prod = c2 * a2
            scale = prod + tl.load(offset_ptr)
        else:
            scale = tl.load(absmax_ptr + block, mask=mask, other=0.0)
        pairs = tl.join(high * scale, low * scale)  # [BLOCK, 2], fp32
        if out_ptr.dtype.element_ty == tl.bfloat16:
            # Round to nearest even on the bit pattern, as torch's fp32 -> bf16 cast does (the
            # Triton interpreter's own cast truncates). These weights are finite.
            bits = pairs.to(tl.int32, bitcast=True)
            bits = (bits + (((bits >> 16) & 1) + 0x7FFF)) >> 16
            pairs = bits.to(tl.int16).to(tl.bfloat16, bitcast=True)
        else:
            pairs = pairs.to(out_ptr.dtype.element_ty)
        out = pid * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
        tl.store(out_ptr + out, tl.reshape(pairs, [2 * BLOCK]), mask=out < 2 * n_bytes)

    return nf4_dequant_kernel


def _interpreted() -> bool:
    return os.environ.get("TRITON_INTERPRET", "") == "1"


def covered(A: Any, quant_state: Any, out: Any) -> bool:
    """Whether the Triton path handles this call (else bitsandbytes does)."""
    import torch

    if quant_state is None or quant_state.quant_type != "nf4" or len(quant_state.shape) != 2:
        return False
    if quant_state.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    n = quant_state.shape[0] * quant_state.shape[1]
    if n % 2 or A.numel() * A.element_size() != n // 2 or A.dtype != torch.uint8:
        return False
    if not (A.is_cuda or _interpreted()):
        return False
    if out is not None and (out.dtype != quant_state.dtype or not out.is_contiguous()
                            or out.numel() != n):
        return False
    return True


def dequantize_nf4(A: Any, quant_state: Any, out: Any = None) -> Any:
    """The [N, K] weight of a bitsandbytes NF4 tensor in ``quant_state.dtype``, one launch."""
    import torch
    import triton

    global _kernel
    if _kernel is None:
        _kernel = _build_kernel()
    if out is None:
        out = torch.empty(tuple(quant_state.shape), dtype=quant_state.dtype, device=A.device)
    n_bytes = out.numel() // 2
    if quant_state.nested:
        s2 = quant_state.state2
        absmax2 = s2.absmax.float() if s2.absmax.dtype != torch.float32 else s2.absmax
        offset = torch.as_tensor(quant_state.offset, dtype=torch.float32, device=A.device).reshape(1)
        args = (quant_state.absmax, absmax2, s2.code, offset)
        bs2 = s2.blocksize
    else:
        absmax = quant_state.absmax.float()
        args = (absmax, absmax, absmax, absmax)
        bs2 = 1
    _kernel[(triton.cdiv(n_bytes, BLOCK),)](
        A.reshape(-1), *args, quant_state.code, out, n_bytes,
        NESTED=bool(quant_state.nested), BLOCKSIZE=int(quant_state.blocksize),
        BLOCKSIZE2=int(bs2), BLOCK=BLOCK, EXACT_MUL=not _interpreted(), num_warps=4)
    return out


def dequantize_4bit(A, quant_state=None, absmax=None, out=None, blocksize=None, quant_type="fp4"):
    """Drop-in for ``bitsandbytes.functional.dequantize_4bit``."""
    if not covered(A, quant_state, out):
        return _original(A, quant_state=quant_state, absmax=absmax, out=out,
                         blocksize=blocksize, quant_type=quant_type)
    result = dequantize_nf4(A, quant_state, out)
    if A.shape[0] == 1:  # bitsandbytes' shim for the transposed packed layout
        return result.t()
    return result


def enable() -> bool:
    """Route ``bitsandbytes.functional.dequantize_4bit`` through the Triton kernel. Idempotent."""
    global _original
    import bitsandbytes.functional as bnb_functional

    if _original is not None:
        return True
    _original = bnb_functional.dequantize_4bit
    bnb_functional.dequantize_4bit = dequantize_4bit
    return True


def disable() -> None:
    global _original
    if _original is None:
        return
    import bitsandbytes.functional as bnb_functional

    bnb_functional.dequantize_4bit = _original
    _original = None


def enabled() -> bool:
    return _original is not None


def enable_from_env() -> bool:
    """``SOUP_NF4_TRITON=1`` turns the kernel on; anything else leaves bitsandbytes alone."""
    if os.environ.get(ENV, "") != "1":
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return enable()
