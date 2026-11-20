"""Expert GEMM plans for the route-preserving MBR step (``--expert-gemm``).

``native`` (the default) changes nothing. ``lt-fused`` keeps the A0 bits of grouped MBR: each
expert call is cut into the micro-batch segments L2L would have run, and every segment's GEMM goes
through a cuBLASLt call forced onto the kernel native cuBLAS picks for that segment (see
``a0_exact_lt``). ``lt-fused`` raises when it cannot run exactly; it never falls back to native
(ADR 0002).

The segmented plan hooks the MBR capture, ``StreamedExperts.forward``, and the route walkers, so
the harness modes ``segmented``, ``lt-merged`` and ``padded`` build on the same plan.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from soup_cli.utils import a0_exact_lt, direct_expert_backward

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
            direct = direct_expert_backward
            function = direct.weighted_expert_function(original_forward)
            with direct.fused_node(plan_for, fused, state.direct_stats, function):
                yield state
        elif direct_mm is None:
            yield state
        else:
            direct = direct_expert_backward
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
class LtMergedState(SegmentState):
    """Segment state plus the cuBLASLt algo table of ``lt-merged``."""

    table: Any = None

    @property
    def merged_calls(self) -> int:
        return int(self.table.stats.get("merged_calls", 0)) if self.table is not None else 0


@contextmanager
def lt_fused_expert_gemm(
    backend: Any = None, require_cuda: bool = True, streams: int = 1
) -> Iterator[LtMergedState]:
    """The segmented plan with each expert visit as one extension call in forward and one in
    the direct backward (``a0_exact_lt.FusedExpert``): merged A0-exact Lt GEMMs and the node's
    elementwise ops in C++, so the Python and dispatcher cost per GEMM is gone."""
    fused = a0_exact_lt.FusedExpert(
        backend if backend is not None else a0_exact_lt.load_extension(), require_cuda)
    if streams != 1:
        fused.backend.set_streams(streams)
    state = LtMergedState(table=fused.table)
    with segmented_expert_gemm(state=state, fused=fused) as active:
        yield active  # type: ignore[misc]
