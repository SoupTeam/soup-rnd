"""Batch-invariant expert GEMM for the PR #72 MBR bridge (plan A2, harness only).

The routed experts' ``F.linear`` is swapped for one call per row (M=1), forward and backward
(the expert weights are frozen, so autograd only needs the input gradient and differentiates
the per-row calls). The result of a row then no longer depends on how many rows share a call,
which is what grouping micro-batches changes. Only calls made from ``expert_mlp`` are touched:
the router, attention and LoRA layers keep their native GEMMs.

The expert node also multiplies the expert output by the routing weight (``[M, 1]``), and the
gradient of that weight is a sum over the hidden axis whose order CUDA picks from M (found by the
ticket 06 diagnosis: fp32 rows differ between M=1 and M=2). Inside the expert node that multiply is
done one row per call as well.

This alone does not make grouped MBR bit-equal to L2L: LoRA gradients must also add up in the
same order (``L2LStep.run(mbr_split_backward=True)``), and in bf16 autocast the weight cache
must not share a cast across micro-batches. Plan section 6, row "Факт".
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, ContextManager, Iterator, List

from soup_cli.utils.expert_gemm import (  # noqa: F401
    _EXPERT_NODE_FUNCTIONS,
    _EXPERT_NODE_MODULE,
    _FRAME_DEPTH,
    LtMergedState,
    SegmentLinear,
    SegmentState,
    _called_from_expert_mlp,
    _called_from_expert_node,
    _expert_mlp_frame,
    _is_route_weight_mul,
    _node_key,
    lt_fused_expert_gemm,
    planned_grouped_routes,
    segment_rows,
    segmented_expert_gemm,
)

EXPERT_GEMM_CHOICES = ("native", "invariant", "segmented", "padded", "lt-merged", "lt-fused")


@contextmanager
def rowwise_expert_linear() -> Iterator[List[int]]:
    """Swap the experts' ``F.linear`` and route-weight multiply for one call per row.

    Yields ``[linear rows, multiply rows]``, the rows served by each swap."""
    import torch
    import torch.nn.functional as functional

    original = functional.linear
    original_mul = torch.Tensor.__mul__
    rows: List[int] = [0, 0]

    def linear(input: Any, weight: Any, bias: Any = None) -> Any:  # noqa: A002
        if not _called_from_expert_mlp():
            return original(input, weight, bias)
        flat = input.reshape(-1, input.shape[-1])
        rows[0] += int(flat.shape[0])
        out = [original(flat[i : i + 1], weight, bias) for i in range(flat.shape[0])]
        return torch.cat(out).reshape(*input.shape[:-1], -1)

    def mul(self: Any, other: Any) -> Any:
        if _is_route_weight_mul(self, other) and _called_from_expert_node():
            rows[1] += int(self.shape[0])
            return torch.cat([original_mul(self[i : i + 1], other[i : i + 1])
                              for i in range(self.shape[0])])
        return original_mul(self, other)

    functional.linear = linear  # type: ignore[assignment]
    torch.Tensor.__mul__ = mul  # type: ignore[method-assign]
    try:
        yield rows
    finally:
        functional.linear = original  # type: ignore[assignment]
        torch.Tensor.__mul__ = original_mul  # type: ignore[method-assign]


@dataclass
class PadState:
    """What the padded swaps served."""

    pad_rows: int
    linear_rows: int = 0
    mul_rows: int = 0


@contextmanager
def padded_expert_gemm(pad_rows: int = 512) -> Iterator[PadState]:
    """Pad every expert GEMM call and route-weight multiply to ``pad_rows`` rows with zeros.

    Plan D1. Every call then has one shape whatever the micro-batch or the group, so the kernel
    cuBLAS picks, and the order of the reductions, do not depend on how many rows share a call
    (checked in the micro tests: one row gives the same bits at real M = 1 .. 512 with 512 pad).
    Different pad sizes give different bits, so one size serves both L2L and MBR arms. A call with
    more rows than ``pad_rows`` is refused. Padded rows are zeros; their outputs are cut off.
    Same two-read expert node as ``rowwise_expert_linear``."""
    import torch
    import torch.nn.functional as functional

    if pad_rows < 1:
        raise ValueError(f"pad rows must be positive, got {pad_rows}")
    state = PadState(pad_rows=pad_rows)
    original = functional.linear
    original_mul = torch.Tensor.__mul__

    def padded(tensor: Any, fill: float) -> Any:
        rows = int(tensor.shape[0])
        if rows > pad_rows:
            raise RuntimeError(f"expert call has {rows} rows, more than the pad of {pad_rows}")
        if rows == pad_rows:
            return tensor
        extra = tensor.new_full((pad_rows - rows, *tensor.shape[1:]), fill)
        return torch.cat([tensor, extra])

    def linear(input: Any, weight: Any, bias: Any = None) -> Any:  # noqa: A002
        if not _called_from_expert_mlp():
            return original(input, weight, bias)
        flat = input.reshape(-1, input.shape[-1])
        rows = int(flat.shape[0])
        state.linear_rows += rows
        out = original(padded(flat, 0.0), weight, bias)[:rows]
        return out.reshape(*input.shape[:-1], -1)

    def mul(self: Any, other: Any) -> Any:
        if _is_route_weight_mul(self, other) and _called_from_expert_node():
            rows = int(self.shape[0])
            state.mul_rows += rows
            return original_mul(padded(self, 0.0), padded(other, 1.0))[:rows]
        return original_mul(self, other)

    functional.linear = linear  # type: ignore[assignment]
    torch.Tensor.__mul__ = mul  # type: ignore[method-assign]
    try:
        yield state
    finally:
        functional.linear = original  # type: ignore[assignment]
        torch.Tensor.__mul__ = original_mul  # type: ignore[method-assign]


def _sibling(name: str) -> Any:
    """A harness module: the bridge runs with benchmarks/harness on sys.path, tests import the
    package."""
    import importlib

    try:
        return importlib.import_module(name)
    except ImportError:
        return importlib.import_module(f"benchmarks.harness.{name}")


@contextmanager
def lt_merged_expert_gemm(
    backend: Any = None, native: Any = None, direct_backward: bool = False
) -> Iterator[LtMergedState]:
    """The segmented plan, with segments that A0 serves with one cuBLAS kernel merged into one
    cuBLASLt call forced onto that kernel (``a0_exact_lt``), forward and input gradient. Rows keep
    their A0 bits; segments without an A0-exact Lt algo keep the native per-segment call.
    ``backend`` defaults to the compiled extension (CUDA only); ``native`` (the reference the
    table matches, default ``a0_exact_lt.native_matmul``) is replaceable for CPU tests."""
    import torch

    lt = _sibling("a0_exact_lt")
    table = lt.AlgoTable(
        backend if backend is not None else lt.load_extension(),
        native=native if native is not None else lt.native_matmul,
    )
    state = LtMergedState(table=table)

    def segment_linear(flat: Any, weight: Any, bias: Any, counts: List[int]) -> Any:
        if bias is not None:
            raise RuntimeError("lt-merged expert GEMM expects bias-free expert projections")
        device = flat.device.type
        if torch.is_autocast_enabled(device):  # what F.linear's autocast wrapper would cast
            dtype = torch.get_autocast_dtype(device)
            flat, weight = flat.to(dtype), weight.to(dtype)
        with torch.autocast(device, enabled=False):
            return lt.MergedLinear.apply(flat, weight, counts, table)

    def merged_mm(x: Any, w: Any, transposed: bool, counts: List[int]) -> Any:
        return lt.merged_matmul(x.contiguous(), w, transposed, counts, table)

    with segmented_expert_gemm(
        segment_linear=segment_linear, state=state,
        direct_mm=merged_mm if direct_backward else None,
    ) as active:
        yield active  # type: ignore[misc]


def alternating_switch(step_index: int, enabled: bool) -> bool:
    """Whether an option that is alternated step by step is on for step ``step_index``: with
    ``enabled`` it is off for even steps and on for odd ones, so every pair of neighbouring steps
    differs only in the option (the host's drift cancels in the paired comparison)."""
    return bool(enabled and step_index % 2 == 1)


@contextmanager
def fast_row_gather_context(enabled: bool = True) -> Iterator[None]:
    """Streamed experts built while this is active (or already built) use the light row gathers of
    ``expert_stream.fast_gather_functions``: the class default of ``fast_row_gather`` is switched
    on and restored on exit."""
    if not enabled:
        yield
        return
    from soup_cli.utils.expert_stream import streamed_experts_class

    cls = streamed_experts_class()
    previous = cls.fast_row_gather
    cls.fast_row_gather = True
    try:
        yield
    finally:
        cls.fast_row_gather = previous


@contextmanager
def async_lease_context(enabled: bool = True) -> Iterator[None]:
    """``ExpertBufferPool.async_lease`` on for the pools used while this is active (the class
    default is switched on and restored on exit): a lease ends without a device-wide sync."""
    if not enabled:
        yield
        return
    from soup_cli.utils.expert_stream import ExpertBufferPool

    previous = ExpertBufferPool.async_lease
    ExpertBufferPool.async_lease = True
    try:
        yield
    finally:
        ExpertBufferPool.async_lease = previous


def expert_gemm_context(
    mode: str, pad_rows: int = 512, lt_backend: Any = None, direct_backward: bool = False,
    streams: int = 1,
) -> ContextManager[Any]:
    """``native``: no change. ``invariant``: the row-wise expert linear layer. ``segmented``:
    native GEMM called per micro-batch segment (the L2L shapes). ``padded``: every call padded
    to ``pad_rows`` rows (plan D1). ``lt-merged``: segmented, with same-kernel segments merged
    into one forced-algo cuBLASLt call (A0 bits kept). ``direct_backward`` (segmented and
    lt-merged only) computes the expert node's backward without autograd, same bits."""
    if mode == "lt-fused":
        return lt_fused_expert_gemm(lt_backend, streams=streams)
    if direct_backward and mode not in ("segmented", "lt-merged"):
        raise ValueError("the direct expert backward needs --expert-gemm segmented or lt-merged")
    if mode == "native":
        return nullcontext([0, 0])
    if mode == "invariant":
        return rowwise_expert_linear()
    if mode == "segmented":
        if direct_backward:
            return segmented_expert_gemm(
                direct_mm=_sibling("direct_expert_backward").segment_mm
            )
        return segmented_expert_gemm()
    if mode == "padded":
        return padded_expert_gemm(pad_rows)
    if mode == "lt-merged":
        return lt_merged_expert_gemm(lt_backend, direct_backward=direct_backward)
    raise ValueError(f"expert gemm must be one of {EXPERT_GEMM_CHOICES}, got {mode!r}")
