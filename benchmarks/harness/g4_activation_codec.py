#!/usr/bin/env python3
"""G4: per-row compression of the L2L boundary activations.

The L2L store (``l2l_activations.ActivationStore``) parks the input of every decoder
layer for every micro-batch in host RAM between the forward walk and the backward
walk. This module stores that input compressed:

* ``bf16``: identity. The activation is stored in its own dtype (bf16 on CUDA, float32
  on the CPU test model), byte for byte. It is the null control: the same wrapper,
  no transform.
* ``int8``: symmetric per-row absmax, 127 levels a side, step ``absmax / 127``.
* ``fp8_e4m3``: per-row absmax scaled to 448 (the largest finite
  ``torch.float8_e4m3fn``), then a round-to-nearest cast.
* ``int4``: symmetric per-row absmax, 7 levels a side, two values per byte. It is the
  deliberately coarse positive control and never a candidate.

One row is one token's hidden vector, the last dimension of the chunk.

Scale dtype: float32. The stored scale is the row absmax itself; the level count (127,
448, 7) is a constant of the codec. float32 holds every bf16 value exactly (same
exponent range, longer mantissa), so the absmax round-trips exactly and the element of
largest magnitude in every row is restored exactly, a massive activation included.
Quantising as ``x / absmax * levels`` keeps the intermediate in [-1, 1], so neither a
bf16 extreme (3.39e38) nor a bf16 denormal absmax (9.2e-41) overflows or underflows
the step, as ``x * (127 / absmax)`` would. The cost is 4 bytes per row: 0.1% of a
4096-wide int8 row.

Zero row: absmax 0 is stored as 0, every payload value is 0, and the row restores to
exact zeros. A row holding a NaN or an Inf has no absmax to scale by, so a lossy codec
refuses it: ``encode`` raises ``NonFiniteActivationError``, and ``CodecActivationStore``
raises it from ``end_step`` (one host read per step, at its end), so ``L2LStep.run``
fails before the caller can apply the step. The identity codec stores whatever the
forward produced, byte for byte, as the plain store does.

Compression runs on the compute device before the device-to-host copy, and the
restore runs on the compute device after the host-to-device copy. Both are timed with
CUDA events on CUDA and ``time.perf_counter`` on CPU.

``CodecActivationStore`` has the ``ActivationStore`` interface (``put``, ``get``,
``begin_backward``, ``end_step``, ``close``, ``stats``, ``k``, ``n_layers``,
``chunk_bytes``), so ``l2l_schedule.L2LStep.run`` drives it unchanged. The payload
goes through an unmodified inner ``ActivationStore`` of the payload dtype and shape,
spill included. The scales live in one float32 slab of ``(n_layers, k, rows)``, pinned
when the store pins, and never spill. A payload chunk whose size is not a multiple of
the 4096-byte sector is refused for spill (no padding), so the spill file holds
exactly ``payload_chunk_bytes`` per spilled chunk and ``chunk_bytes`` is payload plus
scales exactly. The A4 planner takes ``payload_chunk_bytes`` as its ``chunk_bytes`` and
``scale_chunk_bytes`` as its ``pinned_floor_bytes``, both unrounded.

Benchmark harness code, not shipped. Top level imports the standard library and the
sibling harness module only; torch is imported inside functions.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import l2l_activations  # noqa: E402

IDENTITY = "bf16"
INT8 = "int8"
FP8 = "fp8_e4m3"
COARSE = "int4"
#: The codecs a training arm may use. ``int4`` is a control and never trains.
CANDIDATES = (INT8, FP8)
SCALE_BYTES = 4
SCALE_DTYPE = "float32"
SECTOR_BYTES = l2l_activations.SECTOR_BYTES
#: bf16 bytes per element: the ``source_element_bytes`` every arithmetic table assumes.
BF16_BYTES = 2


@dataclass(frozen=True)
class Codec:
    """One codec. ``levels`` is the largest stored magnitude; 0 means identity."""

    name: str
    levels: int
    payload_dtype: Optional[str]
    packed: bool = False

    @property
    def identity(self) -> bool:
        return self.levels == 0


CODECS: Dict[str, Codec] = {
    IDENTITY: Codec(IDENTITY, 0, None),
    INT8: Codec(INT8, 127, "int8"),
    FP8: Codec(FP8, 448, "float8_e4m3fn"),
    COARSE: Codec(COARSE, 7, "uint8", packed=True),
}


class NonFiniteActivationError(ValueError):
    """A boundary activation holds a NaN or an Inf, which a per-row absmax codec cannot
    encode: the scale would be non-finite and every restored value garbage."""


def codec_of(name: str) -> Codec:
    try:
        return CODECS[name]
    except KeyError:
        raise ValueError(f"unknown codec {name!r}; choose from {sorted(CODECS)}") from None


# ==========================================================================
# byte accounting (pure, no torch)
# ==========================================================================
def rows_of(shape: Tuple[int, ...]) -> int:
    """Tokens in a chunk: every dimension but the hidden one."""
    return math.prod(int(dim) for dim in shape[:-1])


def payload_shape(shape: Tuple[int, ...], codec: str) -> Tuple[int, ...]:
    spec = codec_of(codec)
    hidden = int(shape[-1])
    last = (hidden + 1) // 2 if spec.packed else hidden
    return tuple(int(dim) for dim in shape[:-1]) + (last,)


def payload_chunk_bytes(
    shape: Tuple[int, ...], codec: str, *, source_element_bytes: int = BF16_BYTES
) -> int:
    """Bytes of one chunk's payload. The identity stores ``source_element_bytes`` per value."""
    spec = codec_of(codec)
    element = source_element_bytes if spec.identity else 1
    return math.prod(payload_shape(shape, codec)) * element


def scale_chunk_bytes(shape: Tuple[int, ...], codec: str) -> int:
    """Bytes of one chunk's scales: one float32 per row, none for the identity."""
    return 0 if codec_of(codec).identity else rows_of(shape) * SCALE_BYTES


def compressed_chunk_bytes(
    shape: Tuple[int, ...], codec: str, *, source_element_bytes: int = BF16_BYTES
) -> int:
    """The host bytes one stored chunk costs: payload + scales. Sector padding is zero
    by construction: the store refuses to spill a payload that is not a sector multiple."""
    return payload_chunk_bytes(
        shape, codec, source_element_bytes=source_element_bytes
    ) + scale_chunk_bytes(shape, codec)


def bytes_per_token_per_layer(
    hidden: int, codec: str, *, source_element_bytes: int = BF16_BYTES
) -> int:
    return compressed_chunk_bytes(
        (1, int(hidden)), codec, source_element_bytes=source_element_bytes
    )


def tokens_at_budget(budget_bytes: int, *, n_layers: int, hidden: int, codec: str) -> int:
    """Tokens whose boundary activations fit ``budget_bytes``, before allocator rounding.
    Arithmetic, not a measurement, and never a speed."""
    return int(budget_bytes) // (int(n_layers) * bytes_per_token_per_layer(hidden, codec))


#: Labelled estimates for the giants and the 7B fixture: (decoder layers, hidden).
#: Kimi K2 and DeepSeek-V3 come from their configs (61 layers, hidden 7168);
#: bf16 at 61 x 7168 x 2 B is the "874 KB per token" of the program PDF.
SHAPES: Dict[str, Tuple[int, int]] = {
    "mistral-7b-v0.3": (32, 4096),
    "kimi-k2": (61, 7168),
    "deepseek-v3": (61, 7168),
}


def per_token_table(codecs: Tuple[str, ...] = (IDENTITY, INT8, FP8)) -> List[Dict[str, Any]]:
    """Host bytes per token over all layers, and the tokens-per-step factor at a fixed
    RAM budget, for each shape and codec. Labelled arithmetic."""
    rows = []
    for name, (layers, hidden) in SHAPES.items():
        base = layers * bytes_per_token_per_layer(hidden, IDENTITY)
        for codec in codecs:
            per_token = layers * bytes_per_token_per_layer(hidden, codec)
            rows.append({
                "shape": name, "layers": layers, "hidden": hidden, "codec": codec,
                "bytes_per_token": per_token,
                "tokens_per_step_factor_at_fixed_ram": base / per_token,
                "kind": "arithmetic estimate, not measured",
            })
    return rows


# ==========================================================================
# the codecs (torch)
# ==========================================================================
def encode(x: Any, codec: str, *, check_finite: bool = True) -> Tuple[Any, Optional[Any]]:
    """``(payload, scale)`` on ``x``'s device. ``scale`` is float32 of shape ``x.shape[:-1]``,
    None for the identity, whose payload is ``x`` itself.

    A lossy codec raises ``NonFiniteActivationError`` on a NaN or Inf anywhere in ``x``
    (a host sync). ``check_finite=False`` skips the check for a caller that tests the
    returned scale itself, as ``CodecActivationStore`` does once per step: a row with a
    NaN or Inf is exactly a row whose scale is not finite."""
    import torch

    spec = codec_of(codec)
    if spec.identity:
        return x, None
    with torch.no_grad():
        values = x.float()
        absmax = values.abs().amax(dim=-1, keepdim=True)
        if check_finite and not bool(torch.isfinite(absmax).all()):
            bad = int((~torch.isfinite(absmax)).sum())
            raise NonFiniteActivationError(
                f"{codec}: {bad} of {absmax.numel()} rows hold a NaN or an Inf; a per-row "
                "absmax codec cannot encode them")
        safe = torch.where(absmax > 0, absmax, torch.ones_like(absmax))
        unit = values / safe
        if spec.name == FP8:
            payload = (unit * spec.levels).clamp_(-spec.levels, spec.levels).to(
                torch.float8_e4m3fn
            )
        else:
            levels = torch.round(unit * spec.levels).clamp_(-spec.levels, spec.levels)
            if spec.packed:
                payload = _pack_nibbles(levels)
            else:
                payload = levels.to(torch.int8)
        return payload, absmax.squeeze(-1)


def decode(payload: Any, scale: Optional[Any], codec: str, dtype: Any, hidden: int) -> Any:
    """The restored activation, in ``dtype``, on ``payload``'s device."""
    import torch

    spec = codec_of(codec)
    if spec.identity:
        return payload
    with torch.no_grad():
        if spec.packed:
            levels = _unpack_nibbles(payload, hidden)
        else:
            levels = payload.float()
        return (levels / spec.levels * scale.unsqueeze(-1)).to(dtype)


def _pack_nibbles(levels: Any) -> Any:
    """Signed levels in [-7, 7] as unsigned nibbles (level + 8), two per byte, the even
    index in the low nibble. An odd hidden size gets one zero-level pad nibble."""
    import torch

    biased = (levels + 8).to(torch.uint8)
    if biased.shape[-1] % 2:
        pad = torch.full(biased.shape[:-1] + (1,), 8, dtype=torch.uint8, device=biased.device)
        biased = torch.cat([biased, pad], dim=-1)
    return biased[..., 0::2] | (biased[..., 1::2] << 4)


def _unpack_nibbles(packed: Any, hidden: int) -> Any:
    import torch

    low = (packed & 0x0F).float() - 8
    high = (packed >> 4).float() - 8
    out = torch.stack([low, high], dim=-1).flatten(-2)
    return out[..., :hidden]


def round_trip(x: Any, codec: str) -> Any:
    payload, scale = encode(x, codec)
    return decode(payload, scale, codec, x.dtype, int(x.shape[-1]))


def store_compress(x: Any, codec: str) -> Tuple[Any, Any, Any]:
    """What ``CodecActivationStore.put`` runs per chunk on the device, without a host
    sync: ``(payload, scale, all scales finite)``. A lossy codec only."""
    import torch

    payload, scale = encode(x, codec, check_finite=False)
    return payload, scale, torch.isfinite(scale).all()


# ==========================================================================
# error metrics (torch in, plain floats out)
# ==========================================================================
def restore_error(x: Any, x_hat: Any) -> Dict[str, float]:
    """How far ``x_hat`` is from ``x``, row by row (one row = one token). Sums run in
    float64, so a row at the bf16 extremes neither overflows nor underflows.

    ``rel_l2`` is ||x_hat - x|| / ||x|| over the whole tensor; ``err_sq`` and ``ref_sq``
    are its two squared norms, so chunks pool exactly (``pooled_rel_l2``). A few rows
    with massive activations dominate that norm, so ``rms_row_rel`` weights every token
    equally: the root mean square of the per-row relative L2 (pooled through ``rows``
    and ``row_rel_sq_sum``). ``max_abs_over_absmax`` is the largest element error over
    its row's absmax, and ``worst_row_rel`` the largest per-row relative L2. A zero row
    that comes back non-zero is infinitely wrong; one that comes back zero has no error.
    """
    import torch

    hidden = int(x.shape[-1])
    xf = x.detach().double().reshape(-1, hidden)
    diff = x_hat.detach().double().reshape(-1, hidden) - xf
    err_row = diff.pow(2).sum(dim=-1)
    ref_row = xf.pow(2).sum(dim=-1)
    absmax = xf.abs().amax(dim=-1)
    dmax = diff.abs().amax(dim=-1)
    inf = torch.full_like(err_row, math.inf)
    zero = torch.zeros_like(err_row)
    row_rel = torch.where(
        ref_row > 0, (err_row / ref_row.clamp_min(1e-300)).sqrt(),
        torch.where(err_row > 0, inf, zero),
    )
    ratio = torch.where(
        absmax > 0, dmax / absmax.clamp_min(1e-300), torch.where(dmax > 0, inf, zero)
    )
    err_sq, ref_sq = float(err_row.sum()), float(ref_row.sum())
    rows = int(row_rel.numel())
    row_rel_sq_sum = float(row_rel.pow(2).sum())
    return {
        "rel_l2": pooled_rel_l2([{"err_sq": err_sq, "ref_sq": ref_sq}]),
        "rms_row_rel": math.sqrt(row_rel_sq_sum / rows) if rows else 0.0,
        "max_abs": float(dmax.max()),
        "max_abs_over_absmax": float(ratio.max()),
        "worst_row_rel": float(row_rel.max()),
        "err_sq": err_sq,
        "ref_sq": ref_sq,
        "rows": rows,
        "row_rel_sq_sum": row_rel_sq_sum,
    }


def pooled_rel_l2(parts: List[Dict[str, float]]) -> float:
    """sqrt(sum err_sq / sum ref_sq): the relative L2 of the concatenated tensors."""
    err = sum(float(part["err_sq"]) for part in parts)
    ref = sum(float(part["ref_sq"]) for part in parts)
    if ref > 0:
        return math.sqrt(err / ref)
    return math.inf if err > 0 else 0.0


def merge_errors(parts: List[Dict[str, float]]) -> Dict[str, float]:
    """``restore_error`` results of several chunks as one: pooled norms, worst maxima."""
    rows = sum(int(part["rows"]) for part in parts)
    row_rel_sq_sum = sum(float(part["row_rel_sq_sum"]) for part in parts)
    return {
        "rel_l2": pooled_rel_l2(parts),
        "rms_row_rel": math.sqrt(row_rel_sq_sum / rows) if rows else 0.0,
        "max_abs": max(part["max_abs"] for part in parts),
        "max_abs_over_absmax": max(part["max_abs_over_absmax"] for part in parts),
        "worst_row_rel": max(part["worst_row_rel"] for part in parts),
        "err_sq": sum(part["err_sq"] for part in parts),
        "ref_sq": sum(part["ref_sq"] for part in parts),
        "rows": rows,
        "row_rel_sq_sum": row_rel_sq_sum,
    }


def squared_error(reference: Any, candidate: Any) -> Dict[str, float]:
    """The two squared norms behind a relative L2, in float64."""
    ref = reference.detach().double()
    return {
        "err_sq": float((candidate.detach().double() - ref).pow(2).sum()),
        "ref_sq": float(ref.pow(2).sum()),
    }


def grad_agreement(reference: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Per-tensor cosine and relative L2 of ``candidate`` against ``reference``, and the
    global relative L2 over all tensors together. A reference tensor of zeros has
    cosine 1 and relative L2 0 against zeros, cosine 0 and relative L2 inf otherwise."""
    if set(reference) != set(candidate):
        missing = sorted(set(reference) ^ set(candidate))[:4]
        raise ValueError(f"the two gradient sets name different tensors, e.g. {missing}")
    per: Dict[str, Dict[str, float]] = {}
    err_total = ref_total = 0.0
    for name in sorted(reference):
        ref = reference[name].detach().double().flatten()
        cand = candidate[name].detach().double().flatten()
        ref_sq = float(ref.dot(ref))
        cand_sq = float(cand.dot(cand))
        err_sq = float((cand - ref).pow(2).sum())
        err_total += err_sq
        ref_total += ref_sq
        if ref_sq > 0 and cand_sq > 0:
            cosine = float(ref.dot(cand)) / math.sqrt(ref_sq * cand_sq)
        else:
            cosine = 1.0 if ref_sq == cand_sq == 0 else 0.0
        if ref_sq > 0:
            rel = math.sqrt(err_sq / ref_sq)
        else:
            rel = 0.0 if err_sq == 0 else math.inf
        per[name] = {"cosine": cosine, "rel_l2": rel}
    worst = sorted(per.items(), key=lambda item: item[1]["rel_l2"], reverse=True)[:5]
    return {
        "tensors": len(per),
        "global_rel_l2": pooled_rel_l2([{"err_sq": err_total, "ref_sq": ref_total}]),
        "min_cosine": min((v["cosine"] for v in per.values()), default=1.0),
        "max_rel_l2": max((v["rel_l2"] for v in per.values()), default=0.0),
        "worst": [{"name": name, **values} for name, values in worst],
        "per_tensor": per,
    }


# ==========================================================================
# the store
# ==========================================================================
@dataclass
class CodecStats:
    """What the codec adds to ``StoreStats``: its bytes and its time."""

    codec: str
    source_chunk_bytes: int
    payload_chunk_bytes: int
    scale_chunk_bytes: int
    chunk_bytes: int
    timer: str
    compress_calls: int = 0
    restore_calls: int = 0
    compress_s: float = 0.0
    restore_s: float = 0.0
    compress_chunk_s: List[float] = field(default_factory=list)
    restore_chunk_s: List[float] = field(default_factory=list)


class CodecActivationStore:
    """``n_layers x k`` boundary activations, compressed per row on the compute device.

    ``chunk_bytes`` is ``payload_chunk_bytes + scale_chunk_bytes``: the
    ``compressed_chunk_bytes`` of the chunk shape. Only the payload goes through the
    inner ``ActivationStore`` and may spill; the scales are one resident slab. A spilled
    payload must be a multiple of the 4096 B sector: anything else is refused here,
    never padded. ``stats`` is the inner store's live ``StoreStats`` (its spill bytes and
    waits); ``codec_stats`` holds the codec's bytes and timings. CUDA timings are
    resolved by ``flush_timing()``, which the caller runs after its own end-of-step
    synchronize. The only host read the store adds is the per-step finite check in
    ``end_step``.
    """

    def __init__(
        self,
        *,
        codec: str,
        n_layers: int,
        k: int,
        chunk_shape: Tuple[int, ...],
        dtype: Any,
        device: Any,
        pin: bool,
        spill_layers: int = 0,
        spill_factory: Optional[Callable[[int], Any]] = None,
        ring: int = l2l_activations.DEFAULT_RING,
        keep_chunk_times: bool = False,
    ):
        import torch

        self.codec = codec_of(codec)
        self.shape = tuple(int(dim) for dim in chunk_shape)
        self.dtype = dtype
        self.device = torch.device(device)
        self.is_cuda = self.device.type == "cuda"
        self.hidden = int(self.shape[-1])
        source_element = torch.empty((), dtype=dtype).element_size()
        self.payload_chunk_bytes = payload_chunk_bytes(
            self.shape, codec, source_element_bytes=source_element
        )
        self.scale_chunk_bytes = scale_chunk_bytes(self.shape, codec)
        self.chunk_bytes = self.payload_chunk_bytes + self.scale_chunk_bytes
        if spill_layers and self.payload_chunk_bytes % SECTOR_BYTES:
            raise ValueError(
                f"spill needs a payload chunk that is a multiple of {SECTOR_BYTES} bytes; "
                f"{codec} on {self.shape} is {self.payload_chunk_bytes} bytes, "
                f"{self.payload_chunk_bytes % SECTOR_BYTES} past a multiple (no padding: "
                "keep every layer resident or change seq)"
            )
        payload_dtype = dtype if self.codec.identity else getattr(torch, self.codec.payload_dtype)
        self._inner = l2l_activations.ActivationStore(
            n_layers=n_layers,
            k=k,
            chunk_shape=payload_shape(self.shape, codec),
            dtype=payload_dtype,
            device=device,
            pin=pin,
            spill_layers=spill_layers,
            spill_factory=spill_factory,
            ring=ring,
        )
        try:
            self.n_layers = self._inner.n_layers
            self.k = self._inner.k
            self.spill_layers = self._inner.spill_layers
            self.stats = self._inner.stats
            # Measured now, while every staging buffer is still in its ring's free queue.
            self._staging_bytes = sum(
                int(buf.untyped_storage().nbytes())
                for ring_ in (getattr(self._inner, "_write_ring", None),
                              getattr(self._inner, "_read_ring", None))
                if ring_ is not None
                for buf, _event in list(ring_._free.queue)  # noqa: SLF001 — measured
            )
            rows = rows_of(self.shape)
            # One slab holds every chunk's scales: one allocation instead of
            # n_layers x k small ones. Scales never spill. Zeroed, so an unwritten
            # slot restores to zeros and is never named by the finite check.
            self._scale_slab: Optional[Any] = None
            if not self.codec.identity:
                self._scale_slab = torch.zeros(
                    (self.n_layers, self.k, rows), dtype=torch.float32, pin_memory=pin
                )
        except BaseException:
            # The inner store may already run spill threads over an open file.
            self._inner.close()
            raise
        self._keep = bool(keep_chunk_times)
        #: AND of ``isfinite(scale)`` over this step's puts, kept on the device so the
        #: walk never syncs; read once in ``end_step``.
        self._finite: Optional[Any] = None
        #: Off: no events and no counters, for the uninstrumented timing steps.
        self.timing_on = True
        self._pending: List[Tuple[str, Any, Any]] = []
        self.codec_stats = CodecStats(
            codec=self.codec.name,
            source_chunk_bytes=math.prod(self.shape) * source_element,
            payload_chunk_bytes=self.payload_chunk_bytes,
            scale_chunk_bytes=self.scale_chunk_bytes,
            chunk_bytes=self.chunk_bytes,
            timer="cuda-event" if self.is_cuda else "perf_counter",
        )

    # -- timing ----------------------------------------------------------------
    def _timed(self, kind: str, body: Callable[[], Any]) -> Any:
        if not self.timing_on:
            return body()
        if self.is_cuda:
            import torch

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = body()
            end.record()
            self._pending.append((kind, start, end))
            return out
        started = time.perf_counter()
        out = body()
        self._add(kind, time.perf_counter() - started)
        return out

    def _add(self, kind: str, seconds: float) -> None:
        stats = self.codec_stats
        if kind == "compress":
            stats.compress_calls += 1
            stats.compress_s += seconds
            if self._keep:
                stats.compress_chunk_s.append(seconds)
        else:
            stats.restore_calls += 1
            stats.restore_s += seconds
            if self._keep:
                stats.restore_chunk_s.append(seconds)

    def flush_timing(self) -> None:
        """Resolve the CUDA events recorded so far. Call after a device synchronize."""
        pending, self._pending = self._pending, []
        for kind, start, end in pending:
            end.synchronize()
            self._add(kind, start.elapsed_time(end) / 1000.0)

    # -- the ActivationStore interface -------------------------------------------
    def put(self, layer: int, mb: int, tensor: Any) -> None:
        if self.codec.identity:
            self._inner.put(layer, mb, tensor)
            return

        payload, scale, finite = self._timed(
            "compress", lambda: store_compress(tensor, self.codec.name))
        self._finite = finite if self._finite is None else self._finite & finite
        self._inner.put(layer, mb, payload)
        self._scale_slab[layer, mb].copy_(scale.reshape(-1), non_blocking=True)

    def get(self, layer: int, mb: int) -> Any:
        """The restored activation as a NEW tensor on the store's device."""
        payload = self._inner.get(layer, mb)
        if self.codec.identity:
            return payload
        scale = self._scale_slab[layer, mb].to(self.device, non_blocking=True, copy=True)
        scale = scale.view(self.shape[:-1])
        return self._timed(
            "restore",
            lambda: decode(payload, scale, self.codec.name, self.dtype, self.hidden),
        )

    def begin_backward(self) -> None:
        self._inner.begin_backward()

    def end_step(self) -> None:
        """The inner store's own check, then the codec's: refuse the step if any
        boundary held a NaN or an Inf. ``L2LStep.run`` calls this before it returns, so
        no optimizer step ever applies gradients recomputed from a garbage restore. One
        host read, at the end of the step, where the caller synchronizes anyway."""
        finite, self._finite = self._finite, None
        self._inner.end_step()
        if finite is not None and not bool(finite):
            import torch

            # bool() above waited for the step's work, the slab copies included.
            bad = (~torch.isfinite(self._scale_slab)).any(dim=-1).nonzero().tolist()
            raise NonFiniteActivationError(
                f"{self.codec.name}: {len(bad)} boundary chunk(s) held a NaN or an Inf, "
                f"first (layer, micro-batch) {[tuple(key) for key in bad[:4]]}; a per-row "
                "absmax codec cannot encode them, so the step is refused")

    def close(self) -> None:
        """Join the spill threads and remove the spill file (inner store), drop the
        scale slab. Idempotent."""
        self._inner.close()
        self._scale_slab = None
        self._finite = None

    # -- accounting --------------------------------------------------------------
    def host_bytes(self) -> Dict[str, Any]:
        """The bytes this store holds, read from its real tensor storages and its spill
        file, and the per-token-per-layer figure they imply. These are the requested
        bytes: the pinned allocator's power-of-two rounding (#901) is A4's to add.
        Spill staging rings (``2 x ring`` buffers of payload + one sector, measured at
        construction) are reported apart and excluded from the per-token figure."""
        import os

        inner = self._inner._resident  # noqa: SLF001 — measured, not assumed
        resident = sum(int(t.untyped_storage().nbytes()) for t in inner.values())
        scales = (
            int(self._scale_slab.untyped_storage().nbytes())
            if self._scale_slab is not None else 0
        )
        chunks = len(inner)
        rows = rows_of(self.shape)
        scale_per_chunk = scales / (self.n_layers * self.k)
        per_chunk = (resident / chunks if chunks else 0.0) + scale_per_chunk
        spill = self._inner._spill  # noqa: SLF001 — measured, not assumed
        spill_file = os.path.getsize(spill.path) if spill is not None else 0
        return {
            "codec": self.codec.name,
            "resident_payload_bytes": resident,
            "scale_bytes": scales,
            "spilled_payload_bytes": self.spill_layers * self.k * self.payload_chunk_bytes,
            "spill_file_bytes": spill_file,
            "spill_staging_bytes": self._staging_bytes,
            "host_bytes": resident + scales + self._staging_bytes,
            "resident_chunks": chunks,
            "measured_bytes_per_token_per_layer": per_chunk / rows if rows else None,
            "arithmetic_bytes_per_token_per_layer": self.chunk_bytes / rows if rows else None,
        }
