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
from typing import Any, ContextManager, Dict, Iterator, List, Optional, Tuple

EXPERT_GEMM_CHOICES = ("native", "invariant", "segmented")
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
        if (
            isinstance(other, torch.Tensor)
            and self.dim() == 2
            and other.dim() == 2
            and other.shape[1] == 1
            and self.shape[0] == other.shape[0]
            and _called_from_expert_node()
        ):
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


def segment_plan(token_idx: Any, boundaries: Any) -> Tuple[Any, Any, List[int]]:
    """How to cut one expert's gathered rows into the calls an L2L step would make.

    ``token_idx`` is the joined-group token of each row (the expert module orders rows by top-k
    slot, then token, so one micro-batch's rows are not adjacent). ``boundaries`` holds the first
    token index of every micro-batch but the first. Returns ``(perm, inv, counts)``: ``perm``
    gathers the rows micro-batch by micro-batch with their original relative order, ``inv`` puts
    results back, ``counts`` is the row count of each non-empty micro-batch, which is the M the
    L2L call of that micro-batch has."""
    import torch

    segment = torch.bucketize(token_idx, boundaries, right=True)
    perm = torch.argsort(segment, stable=True)
    inv = torch.argsort(perm)
    counts = torch.bincount(segment, minlength=int(boundaries.numel()) + 1).tolist()
    return perm, inv, [count for count in counts if count]


@dataclass
class SegmentState:
    """Segment plans by ``(layer, expert)`` and what the swaps served."""

    plans: Dict[Tuple[int, int], Tuple[Any, Any, List[int]]] = field(default_factory=dict)
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


@contextmanager
def segmented_expert_gemm() -> Iterator[SegmentState]:
    """Keep the native expert GEMM, but call it once per micro-batch segment of a joined group.

    The joined MBR group makes one expert call of M = all rows of the group; an L2L step makes one
    per micro-batch. Native GEMM bits depend on M, so the joined call differs from A0. Here the
    rows of each expert call are cut back into the micro-batch segments (plan from the group's
    token counts) and each segment goes through the original ``F.linear`` and route-weight
    multiply, so every call has the L2L shape. Works with the two-read expert node
    (``WeightedFrozenExpert``): its forward and backward are keyed by ``(layer, expert)``.

    Hooks: the MBR capture (micro-batch sizes), ``StreamedExperts.forward`` (layer, boundaries)
    and the route walkers (per-expert token indices, giving the plan)."""
    import torch
    import torch.nn.functional as functional

    from soup_cli.utils import expert_stream, l2l_step

    state = SegmentState()
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
        sizes = list(pending)
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
            for expert, top_k_pos, token_idx in original(*args, **kwargs):
                if active["boundaries"] is not None:
                    state.plans[(active["layer"], expert)] = segment_plan(
                        token_idx, active["boundaries"]
                    )
                yield expert, top_k_pos, token_idx

        return routes

    def plan_for(key: Optional[Tuple[int, int]], rows: int) -> Optional[Tuple[Any, Any, List[int]]]:
        plan = state.plans.get(key) if key is not None else None
        if plan is not None and plan[0].numel() != rows:
            raise RuntimeError(
                f"segment plan covers {plan[0].numel()} rows, the expert call of {key} has {rows}"
            )
        return plan

    def split_apply(plan: Tuple[Any, Any, List[int]], function: Any, *tensors: Any) -> Any:
        perm, inv, counts = plan
        gathered = [tensor.index_select(0, perm) for tensor in tensors]
        parts, start = [], 0
        for count in counts:
            parts.append(function(*[g[start : start + count] for g in gathered]))
            start += count
        return torch.cat(parts).index_select(0, inv)

    def linear(input: Any, weight: Any, bias: Any = None) -> Any:  # noqa: A002
        frame = _expert_mlp_frame()
        if frame is None:
            return original_linear(input, weight, bias)
        flat = input.reshape(-1, input.shape[-1])
        plan = plan_for(_node_key(frame.f_back), flat.shape[0])
        if plan is None:
            state.passthrough_calls += 1
            return original_linear(input, weight, bias)
        state.linear_rows += int(flat.shape[0])
        out = split_apply(plan, lambda part: original_linear(part, weight, bias), flat)
        return out.reshape(*input.shape[:-1], -1)

    def mul(self: Any, other: Any) -> Any:
        if (
            isinstance(other, torch.Tensor)
            and self.dim() == 2
            and other.dim() == 2
            and other.shape[1] == 1
            and self.shape[0] == other.shape[0]
        ):
            node = sys._getframe(1)  # the caller of the patched __mul__
            if (
                node.f_globals.get("__name__") == _EXPERT_NODE_MODULE
                and node.f_code.co_name in _EXPERT_NODE_FUNCTIONS
            ):
                plan = plan_for(_node_key(node), int(self.shape[0]))
                if plan is not None:
                    state.mul_rows += int(self.shape[0])
                    return split_apply(plan, original_mul, self, other)
        return original_mul(self, other)

    functional.linear = linear  # type: ignore[assignment]
    torch.Tensor.__mul__ = mul  # type: ignore[method-assign]
    l2l_step._route_preserving_deepseek_forward = capture
    expert_stream._grouped_expert_routes = walker(original_grouped)
    expert_stream._default_expert_routes = walker(original_default)
    experts_class.forward = experts_forward
    try:
        yield state
    finally:
        functional.linear = original_linear  # type: ignore[assignment]
        torch.Tensor.__mul__ = original_mul  # type: ignore[method-assign]
        l2l_step._route_preserving_deepseek_forward = original_capture
        expert_stream._grouped_expert_routes = original_grouped
        expert_stream._default_expert_routes = original_default
        experts_class.forward = original_forward


def expert_gemm_context(mode: str) -> ContextManager[Any]:
    """``native``: no change. ``invariant``: the row-wise expert linear layer. ``segmented``:
    native GEMM called per micro-batch segment (the L2L shapes)."""
    if mode == "native":
        return nullcontext([0, 0])
    if mode == "invariant":
        return rowwise_expert_linear()
    if mode == "segmented":
        return segmented_expert_gemm()
    raise ValueError(f"expert gemm must be one of {EXPERT_GEMM_CHOICES}, got {mode!r}")
