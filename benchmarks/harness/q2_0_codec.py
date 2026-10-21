"""C5 ticket 01: Q2_0 codec for MoE expert weights, with NF4 behind the same interface.

Spec: .scratch/c5-q2-0-experts/issues/01-q2-0-codec.md; format research:
.scratch/c5-q2-0-experts/research-q2-0-format.md.

    python benchmarks/harness/q2_0_codec.py     print the SHA-256 of this file

Byte layout is llama.cpp's ``GGML_TYPE_Q2_0`` (MIT, read as a reference only):
64 weights per block, one fp16 scale ``d``, 2-bit codes 0..3 meaning
-1, 0, +1, +2, weight j of a block in byte j/4 at bits (j%4)*2, low bits first.
Restore is ``(q - 1) * d`` in float32. The tensor is flattened row-major and the
tail padded with zeros to 64.

The scale is our own choice, not llama.cpp's ``d = absmax`` (which never picks
the +2 level): a signed MSE search over fp16-rounded candidates, see
``_search_scale``. ``q2_0_absmax`` reproduces the llama.cpp reference quantizer
for comparison rows only.

Codecs: ``q2_0``, ``q2_0_absmax`` and ``nf4`` (bitsandbytes, block 64, double
quant on, which is Soup's shipped 4-bit default). torch and bitsandbytes are
imported inside functions.
"""

from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BLOCK = 64
FP16_MAX = 65504.0
CODECS = ("q2_0", "q2_0_absmax", "nf4")


@dataclass(frozen=True)
class Packed:
    """A quantized tensor. ``tensors`` holds exactly what is stored, so its bytes are the size."""

    codec: str
    shape: tuple[int, ...]
    tensors: dict[str, Any]
    state: Any = None


def _roundf(x):
    """C ``roundf``: halves away from zero. ``torch.round`` rounds halves to even.

    ``floor(|x| + 0.5)`` in float32 turns 0.49999997 into 1; adding the doubled
    fraction does not (the same arithmetic as gguf-py's ``np_roundf``).
    """
    import torch

    a = x.abs()
    floored = torch.floor(a)
    return torch.sign(x) * (floored + torch.floor(2 * (a - floored)))


def _codes(blocks, d):
    """2-bit codes 0..3 of ``blocks`` [B, 64] for float32 scales ``d`` [B, 1].

    A zero scale gives code 1 (value 0) for every weight, as llama.cpp does.
    """
    import torch

    safe = torch.where(d == 0, torch.ones_like(d), d)
    x = torch.where(d == 0, torch.zeros_like(blocks), blocks / safe)
    return (_roundf(x) + 1).clamp(0, 3)


def _shifts(device):
    """Bit offsets of the four codes in a ``qs`` byte, weight 0 in the low bits."""
    import torch

    return torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=device)


def _search_scale(blocks):
    """Per-block fp16 scale with the least squared error, sign allowed.

    Candidates are ``c / t`` for c in {max, min, absmax} of the block and t from
    1.0 to 2.5 in steps of 0.025, plus the exact ``max/2``, ``min/2`` and
    ``absmax``. Each is rounded to fp16 before its error is measured. The first
    candidate with the least error wins.
    """
    import torch

    hi = blocks.max(dim=1).values
    lo = blocks.min(dim=1).values
    amax = blocks.abs().max(dim=1).values
    candidates = [hi / 2, lo / 2, amax]
    for step in range(61):
        divisor = 1.0 + 0.025 * step
        candidates += [hi / divisor, lo / divisor, amax / divisor]

    best_err = torch.full_like(hi, float("inf"))
    best_d = torch.zeros_like(hi)
    for cand in candidates:
        d = cand.to(torch.float16).to(torch.float32)
        err = ((_codes(blocks, d[:, None]) - 1) * d[:, None] - blocks).pow(2).sum(dim=1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_d = torch.where(better, d, best_d)
    return best_d


def _absmax_reference(blocks):
    """llama.cpp ``quantize_row_q2_0_ref``: d = absmax, codes from the fp32 ``1 / d``.

    Returns (codes, d) with d already rounded to fp16, as the block stores it.
    """
    import torch

    amax = blocks.abs().max(dim=1, keepdim=True).values
    inv = torch.where(amax > 0, 1.0 / amax, torch.zeros_like(amax))
    codes = (_roundf(blocks * inv) + 1).clamp(0, 3)
    return codes, amax[:, 0].to(torch.float16).to(torch.float32)


def _quantize_q2_0(flat, shape, codec):
    """Both Q2_0 codecs. A block whose absmax exceeds fp16 max is refused: absmax
    is itself a candidate d, and the search does not fall back to a smaller one."""
    import torch

    n = flat.numel()
    n_blocks = -(-n // BLOCK)
    padded = torch.zeros(n_blocks * BLOCK, dtype=torch.float32, device=flat.device)
    padded[:n] = flat
    blocks = padded.view(n_blocks, BLOCK)
    amax = blocks.abs().max().item()
    if amax > FP16_MAX:
        raise ValueError(f"{codec}: a block has |d| = {amax:g} > {FP16_MAX:g}, beyond fp16")
    if codec == "q2_0_absmax":
        codes, d = _absmax_reference(blocks)
    else:
        d = _search_scale(blocks)
        codes = _codes(blocks, d[:, None])
    codes = codes.to(torch.uint8).view(n_blocks, BLOCK // 4, 4)
    qs = (codes << _shifts(flat.device)).sum(dim=2, dtype=torch.uint8)
    return Packed(codec, shape, {"d": d.to(torch.float16), "qs": qs})


def _quantize_nf4(weight):
    """bitsandbytes NF4, block 64, double quant on (Soup's 4-bit default, #321).

    ``tensors`` are the per-tensor buffers: packed codes, uint8 absmax, its fp32
    offset and the fp32 absmax of the absmax. The NF4 and 8-bit lookup tables
    in the state are constants of the format and are not counted.
    """
    import bitsandbytes.functional as bnb

    data, state = bnb.quantize_4bit(
        weight.detach().contiguous(), blocksize=BLOCK, quant_type="nf4",
        compress_statistics=True,
    )
    tensors = {
        "data": data,
        "absmax": state.absmax,
        "offset": state.offset,
        "absmax2": state.state2.absmax,
    }
    return Packed("nf4", tuple(weight.shape), tensors, state)


def _dequantize_q2_0(packed):
    import torch

    qs = packed.tensors["qs"]
    codes = ((qs[:, :, None] >> _shifts(qs.device)) & 3).view(qs.shape[0], BLOCK)
    values = (codes.to(torch.float32) - 1) * packed.tensors["d"].to(torch.float32)[:, None]
    return values.view(-1)[:math.prod(packed.shape)].view(packed.shape)


def quantize(weight, codec: str) -> Packed:
    """Quantize a float tensor with ``codec`` (one of ``CODECS``).

    Raises ValueError on NaN or Inf in ``weight`` and when a scale does not fit
    fp16; nothing is returned then.
    """
    import torch

    if codec not in CODECS:
        raise ValueError(f"unknown codec {codec!r}; expected one of {CODECS}")
    flat = weight.detach().reshape(-1).to(torch.float32)
    if not torch.isfinite(flat).all():
        raise ValueError(f"{codec}: weight has NaN or Inf")
    if codec == "nf4":
        return _quantize_nf4(weight)
    return _quantize_q2_0(flat, tuple(weight.shape), codec)


def dequantize(packed: Packed):
    """Restore a ``Packed`` tensor to float32 in its original shape."""
    if packed.codec == "nf4":
        import bitsandbytes.functional as bnb

        restored = bnb.dequantize_4bit(packed.tensors["data"], packed.state)
        return restored.float().view(packed.shape)
    return _dequantize_q2_0(packed)


def reconstruction_error(weight, packed: Packed) -> dict[str, float]:
    """Restore error of ``packed`` against ``weight``, both in float32.

    ``mse`` is the mean squared error per weight; ``rel_mse`` divides the summed
    squared error by the summed squared weight (0 for an all-zero weight).
    """
    import torch

    original = weight.detach().to(torch.float32)
    sq_err = (dequantize(packed).to(original.device) - original).pow(2).sum().item()
    sq_ref = original.pow(2).sum().item()
    return {
        "mse": sq_err / original.numel(),
        "rel_mse": sq_err / sq_ref if sq_ref > 0 else 0.0,
    }


def to_block_q2_0(packed: Packed):
    """The blocks as llama.cpp ``block_q2_0``: uint8 [n_blocks, 18], fp16 ``d`` then ``qs``."""
    import torch

    if not packed.codec.startswith("q2_0"):
        raise ValueError(f"{packed.codec} has no block_q2_0 layout")
    if sys.byteorder != "little":
        raise RuntimeError("block_q2_0 stores d little-endian; this host is big-endian")
    d = packed.tensors["d"].contiguous().view(torch.uint8).view(-1, 2)
    return torch.cat([d, packed.tensors["qs"]], dim=1)


def codec_sha256() -> str:
    """SHA-256 of this file's bytes, for the run record."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


if __name__ == "__main__":
    print(codec_sha256())
