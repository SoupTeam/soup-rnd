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

import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, ContextManager, Dict, Iterator, List, Optional, Tuple

EXPERT_GEMM_CHOICES = ("native", "invariant", "segmented", "padded", "lt-merged", "lt-fused")
_FRAME_DEPTH = 4  # other patches of F.linear (the phase clock) may sit between us and the caller


def _called_from_expert_mlp() -> bool:
    frame = sys._getframe(2)
    for _ in range(_FRAME_DEPTH):
        if frame is None:
            return False
        if frame.f_code.co_name == "expert_mlp":
            return True
        frame = frame.f_back
    return False


def _is_route_weight_mul(values: Any, weights: Any) -> bool:
    """True for ``values [M, N] * weights [M, 1]``, the shape of the routing-weight multiply."""
    import torch

    return (
        isinstance(weights, torch.Tensor)
        and values.dim() == 2
        and weights.dim() == 2
        and weights.shape[1] == 1
        and values.shape[0] == weights.shape[0]
    )


_EXPERT_NODE_MODULE = "soup_cli.utils.expert_stream"
_EXPERT_NODE_FUNCTIONS = ("forward", "backward")


def _called_from_expert_node() -> bool:
    """True when the multiply is written in ``forward``/``backward`` of ``expert_stream``."""
    frame = sys._getframe(2)  # the patched __mul__ is frame 1, its caller frame 2
    return (
        frame.f_globals.get("__name__") == _EXPERT_NODE_MODULE
        and frame.f_code.co_name in _EXPERT_NODE_FUNCTIONS
    )


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


def segment_rows(
    token_idx: List[Any], boundaries: Any
) -> List[Tuple[Optional[Any], List[int]]]:
    """For each expert, how to order its rows so every micro-batch's rows are adjacent.

    ``token_idx`` lists, per expert, the joined-group token of each row (the expert module orders
    rows by top-k slot, then token, so one micro-batch's rows are not adjacent). ``boundaries``
    holds the first token index of every micro-batch but the first. Returns one
    ``(perm, counts)`` per expert: ``perm`` gathers the rows micro-batch by micro-batch with their
    original relative order (None when they already sit in one micro-batch), ``counts`` is the
    row count of each non-empty micro-batch, which is the M of that micro-batch's L2L call.
    One stable sort and one device-to-host read serve all experts."""
    import torch

    if not token_idx:
        return []
    segments = int(boundaries.numel()) + 1
    lengths = [int(tokens.numel()) for tokens in token_idx]
    tokens = torch.cat(token_idx)
    expert_of = torch.repeat_interleave(
        torch.arange(len(lengths), device=tokens.device),
        torch.tensor(lengths, device=tokens.device),
        output_size=sum(lengths),
    )
    key = expert_of * segments + torch.bucketize(tokens, boundaries, right=True)
    order = torch.argsort(key, stable=True)
    counts = torch.bincount(key, minlength=len(lengths) * segments).tolist()
    layout: List[Tuple[Optional[Any], List[int]]] = []
    offset = 0
    for index, length in enumerate(lengths):
        per_segment = [c for c in counts[index * segments : (index + 1) * segments] if c]
        perm = None if len(per_segment) <= 1 else order[offset : offset + length] - offset
        layout.append((perm, per_segment))
        offset += length
    return layout


def planned_grouped_routes(
    expert_mask: Any, boundaries: Any
) -> Iterator[Tuple[int, Any, Any, List[int]]]:
    """Hit experts with their rows ordered micro-batch by micro-batch and the segment counts, from
    one nonzero, one stable sort and one host read.

    The same rows, in the same order, as ``expert_stream._grouped_expert_routes`` followed by
    ``segment_rows`` and a gather per expert, without the second host read and the per-expert
    gathers: the coordinates are sorted once by (expert, micro-batch), the experts are views.
    Yields ``(expert, top_k_slot, token, counts)``; ``counts`` lists the non-empty segments."""
    import torch

    coordinates = torch.nonzero(expert_mask, as_tuple=False)
    experts = int(expert_mask.shape[0])
    segments = int(boundaries.numel()) + 1
    key = coordinates[:, 0] * segments + torch.bucketize(coordinates[:, 2], boundaries, right=True)
    order = torch.argsort(key, stable=True)
    counts = torch.bincount(key, minlength=experts * segments).tolist()
    ordered = coordinates.index_select(0, order)
    offset = 0
    for expert in range(experts):
        per_segment = counts[expert * segments : (expert + 1) * segments]
        total = sum(per_segment)
        if total == 0:
            continue
        group = ordered.narrow(0, offset, total)
        offset += total
        yield expert, group[:, 1], group[:, 2], [c for c in per_segment if c]
    if offset != coordinates.shape[0]:
        raise ValueError("grouped expert routes omitted nonzero coordinates")


@dataclass
class SegmentState:
    """Segment row counts by ``(layer, expert)`` and what the swaps served."""

    plans: Dict[Tuple[int, int], List[int]] = field(default_factory=dict)
    direct_stats: Dict[str, int] = field(default_factory=dict)  # --expert-direct-backward
    linear_rows: int = 0
    mul_rows: int = 0
    passthrough_calls: int = 0  # expert linear calls that found no plan


def _expert_mlp_frame() -> Optional[Any]:
    """The ``expert_mlp`` frame that called the patched ``F.linear`` (None if not from there)."""
    frame = sys._getframe(2)
    for _ in range(_FRAME_DEPTH):
        if frame is None:
            return None
        if frame.f_code.co_name == "expert_mlp":
            return frame
        frame = frame.f_back
    return None


def _node_key(frame: Any) -> Optional[Tuple[int, int]]:
    """``(layer, expert)`` of the expert node (``forward``/``backward`` of ``expert_stream``)."""
    for _ in range(_FRAME_DEPTH + 2):
        if frame is None:
            return None
        if (
            frame.f_globals.get("__name__") == _EXPERT_NODE_MODULE
            and frame.f_code.co_name in _EXPERT_NODE_FUNCTIONS
        ):
            local = frame.f_locals
            if "ctx" in local:
                return (local["ctx"].owner.layer_idx, local["ctx"].expert)
            return (local["owner"].layer_idx, local["expert"])
        frame = frame.f_back
    return None


SegmentLinear = Callable[[Any, Any, Any, List[int]], Any]


@contextmanager
def segmented_expert_gemm(
    segment_linear: Optional[SegmentLinear] = None,
    state: Optional[SegmentState] = None,
    direct_mm: Optional[Callable[..., Any]] = None,
    fused: Any = None,
) -> Iterator[SegmentState]:
    """Keep the native expert GEMM, but call it once per micro-batch segment of a joined group.

    The joined MBR group makes one expert call of M = all rows of the group; an L2L step makes one
    per micro-batch. Native GEMM bits depend on M, so the joined call differs from A0. Here the
    route walker orders each expert's rows micro-batch by micro-batch (stable, so each segment keeps
    its L2L row order), the rows of each expert call are cut into those segments (``split``) and
    each segment goes through the original ``F.linear`` and route-weight multiply, so every call
    has the L2L shape. No row is gathered per call: the order is set once, in the routes.
    Works with the two-read expert node
    (``WeightedFrozenExpert``): its forward and backward are keyed by ``(layer, expert)``.

    Hooks: the MBR capture (micro-batch sizes), ``StreamedExperts.forward`` (layer, boundaries)
    and the route walkers (per-expert token indices, giving the plan).

    ``segment_linear(flat, weight, bias, counts)`` replaces the per-segment ``F.linear`` calls
    (``lt-merged`` passes one that merges same-kernel segments); it must return the rows of every
    segment exactly as the per-segment calls would. ``direct_mm(x, w, transposed, counts)``
    turns on the direct expert backward (``direct_expert_backward``) with that segment matmul."""
    import torch
    import torch.nn.functional as functional

    from soup_cli.utils import expert_stream, l2l_step

    state = state if state is not None else SegmentState()
    pending: List[int] = []
    active: Dict[str, Any] = {"layer": None, "boundaries": None}

    original_linear = functional.linear
    original_mul = torch.Tensor.__mul__
    original_capture = l2l_step._route_preserving_deepseek_forward
    original_grouped = expert_stream._grouped_expert_routes
    original_default = expert_stream._default_expert_routes
    experts_class = expert_stream.streamed_experts_class()
    original_forward = experts_class.forward

    def capture(module: Any, captured: Any, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_capture(module, captured, hidden_states, *args, **kwargs)
        pending.append(int(captured[-1]["expert_input"].shape[0]))
        return result

    def experts_forward(self: Any, hidden_states: Any, top_k_index: Any, top_k_weights: Any) -> Any:
        # the micro-batch sizes in micro-batch order: set by the step, or the call order seen
        sizes = list(getattr(self, "_mbr_group_sizes", None) or pending)
        pending.clear()
        if sizes and sum(sizes) == hidden_states.shape[0]:
            active["layer"] = self.layer_idx
            active["boundaries"] = torch.tensor(sizes, device=hidden_states.device).cumsum(0)[:-1]
        try:
            return original_forward(self, hidden_states, top_k_index, top_k_weights)
        finally:
            active["layer"] = active["boundaries"] = None

    def walker(original: Any) -> Any:
        def routes(*args: Any, **kwargs: Any) -> Any:
            items = list(original(*args, **kwargs))
            if active["boundaries"] is None or not items:
                yield from items
                return
            layout = segment_rows([token_idx for _, _, token_idx in items], active["boundaries"])
            for (expert, top_k_pos, token_idx), (perm, counts) in zip(items, layout):
                state.plans[(active["layer"], expert)] = counts
                if perm is not None:
                    top_k_pos, token_idx = top_k_pos[perm], token_idx[perm]
                yield expert, top_k_pos, token_idx

        return routes

    def grouped_routes(expert_mask: Any) -> Any:
        if active["boundaries"] is None:
            yield from original_grouped(expert_mask)
            return
        for expert, top_k_pos, token_idx, counts in planned_grouped_routes(
            expert_mask, active["boundaries"]
        ):
            state.plans[(active["layer"], expert)] = counts
            yield expert, top_k_pos, token_idx

    def plan_for(key: Optional[Tuple[int, int]], rows: int) -> Optional[List[int]]:
        counts = state.plans.get(key) if key is not None else None
        if counts is not None and sum(counts) != rows:
            raise RuntimeError(
                f"segment plan covers {sum(counts)} rows, the expert call of {key} has {rows}"
            )
        return counts

    def linear(input: Any, weight: Any, bias: Any = None) -> Any:  # noqa: A002
        frame = _expert_mlp_frame()
        if frame is None:
            return original_linear(input, weight, bias)
        flat = input.reshape(-1, input.shape[-1])
        counts = plan_for(_node_key(frame.f_back), flat.shape[0])
        if counts is None:
            state.passthrough_calls += 1
            return original_linear(input, weight, bias)
        state.linear_rows += int(flat.shape[0])
        if segment_linear is not None:
            out = segment_linear(flat, weight, bias, counts)
        else:
            out = torch.cat([original_linear(part, weight, bias) for part in flat.split(counts)])
        return out.reshape(*input.shape[:-1], -1)

    def mul(self: Any, other: Any) -> Any:
        if _is_route_weight_mul(self, other):
            node = sys._getframe(1)  # the caller of the patched __mul__
            if (
                node.f_globals.get("__name__") == _EXPERT_NODE_MODULE
                and node.f_code.co_name in _EXPERT_NODE_FUNCTIONS
            ):
                counts = plan_for(_node_key(node), int(self.shape[0]))
                if counts is not None:
                    state.mul_rows += int(self.shape[0])
                    pairs = zip(self.split(counts), other.split(counts))
                    return torch.cat([original_mul(a, b) for a, b in pairs])
        return original_mul(self, other)

    if fused is None:  # the fused node never calls F.linear: no wrapper on every nn.Linear
        functional.linear = linear  # type: ignore[assignment]
    if direct_mm is None and fused is None:
        # With the direct backward the multiply runs only in the node's forward: elementwise,
        # so its bits do not depend on the segments, and the per-segment route-weight reduction
        # is done by expert_vjp. Only the autograd backward needs it cut.
        torch.Tensor.__mul__ = mul  # type: ignore[method-assign]
    l2l_step._route_preserving_deepseek_forward = capture
    expert_stream._grouped_expert_routes = grouped_routes
    expert_stream._default_expert_routes = walker(original_default)
    experts_class.forward = experts_forward
    try:
        if fused is not None:
            direct = _sibling("direct_expert_backward")
            function = direct.weighted_expert_function(original_forward)
            with direct.fused_node(plan_for, fused, state.direct_stats, function):
                yield state
        elif direct_mm is None:
            yield state
        else:
            direct = _sibling("direct_expert_backward")
            function = direct.weighted_expert_function(original_forward)
            with direct.direct_backward(plan_for, direct_mm, state.direct_stats, function):
                yield state
    finally:
        functional.linear = original_linear  # type: ignore[assignment]
        torch.Tensor.__mul__ = original_mul  # type: ignore[method-assign]
        l2l_step._route_preserving_deepseek_forward = original_capture
        expert_stream._grouped_expert_routes = original_grouped
        expert_stream._default_expert_routes = original_default
        experts_class.forward = original_forward


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


@dataclass
class LtMergedState(SegmentState):
    """Segment state plus the cuBLASLt algo table of ``lt-merged``."""

    table: Any = None

    @property
    def merged_calls(self) -> int:
        return int(self.table.stats.get("merged_calls", 0)) if self.table is not None else 0


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


@contextmanager
def lt_fused_expert_gemm(
    backend: Any = None, require_cuda: bool = True, streams: int = 1
) -> Iterator[LtMergedState]:
    """The segmented plan with each expert visit as one extension call in forward and one in
    the direct backward (``a0_exact_lt.FusedExpert``): merged A0-exact Lt GEMMs and the node's
    elementwise ops in C++, so the Python and dispatcher cost per GEMM is gone."""
    lt = _sibling("a0_exact_lt")
    fused = lt.FusedExpert(backend if backend is not None else lt.load_extension(), require_cuda)
    if streams != 1:
        fused.backend.set_streams(streams)
    state = LtMergedState(table=fused.table)
    with segmented_expert_gemm(state=state, fused=fused) as active:
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
