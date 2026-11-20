"""Direct backward of the two-read expert node (used by the ``lt-fused`` expert GEMM).

Idea H3 of the DeepSeek P3 ``separate_fast`` report: the backward of ``WeightedFrozenExpert``
recomputes ``expert_mlp(hidden) * weights`` under autograd and differentiates it, which builds a
graph of many small ops per expert and segment. ``expert_vjp`` runs the same ops in the same
order without a graph:

- forward recompute, per segment: ``h = x @ W_gu.T``, ``gate, up = h.chunk(2)``,
  ``a = silu(gate) * up``, ``y = a @ W_d.T`` (``y`` feeds the route-weight gradient);
- route-weight gradient, per segment: ``sum_to((g * y).to(w.dtype), [M, 1])``, the formula and
  reduction of autograd's ``mul`` backward (per segment, as the segmented multiply does);
- input gradient, per segment: ``grad_y = (g * w).to(y.dtype)``, ``grad_a = grad_y @ W_d``,
  ``silu_backward``, ``grad_x = cat(grad_gate, grad_up) @ W_gu``, cast back to the hidden dtype.

The GEMMs go through ``mm(x, w, transposed, counts)`` (native per segment by default; the
``lt-merged`` mode passes its merged matmul). Only the stateless SiLU the node already requires
is supported.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

SegmentMatmul = Callable[[Any, Any, bool, Sequence[int]], Any]


def segment_mm(x: Any, w: Any, transposed: bool, counts: Sequence[int]) -> Any:
    """The A0 calls: ``F.linear`` per segment (forward) and its input gradient ``g @ W``."""
    import torch

    op = (lambda p: torch.mm(p, w.t())) if transposed else (lambda p: torch.mm(p, w))
    return torch.cat([op(p) for p in x.split(list(counts))])


def expert_vjp(
    hidden: Any,
    weights: Any,
    gate_up: Any,
    down: Any,
    grad_out: Any,
    counts: Sequence[int],
    *,
    need_hidden: bool,
    need_weights: bool,
    autocast_dtype: Optional[Any] = None,
    mm: SegmentMatmul = segment_mm,
) -> Tuple[Optional[Any], Optional[Any]]:
    """Gradients of ``expert_mlp(hidden) * weights`` for hidden and routing weights."""
    import torch
    import torch.nn.functional as functional

    counts = [int(c) for c in counts]
    x, w_gu, w_d = hidden, gate_up, down
    if autocast_dtype is not None:  # what F.linear's autocast wrapper casts
        x, w_gu, w_d = (t.to(autocast_dtype) for t in (x, w_gu, w_d))
    h = mm(x.contiguous(), w_gu, True, counts)
    gate, up = h.chunk(2, dim=-1)
    s = functional.silu(gate)
    a = s * up
    y = None
    if need_weights:
        y = mm(a.contiguous(), w_d, True, counts)
    # autograd casts the incoming gradient to the recomputed result's dtype (y * weights) before
    # any formula; a two-read recompute hands a bf16 grad_out for an fp32 result
    out_dtype = a.dtype if y is None else y.dtype
    grad_out = grad_out.to(torch.promote_types(out_dtype, weights.dtype))
    grad_weights = None
    if need_weights:
        parts = [
            (g * p).to(q.dtype).sum(1, keepdim=True)
            for g, p, q in zip(grad_out.split(counts), y.split(counts), weights.split(counts))
        ]
        grad_weights = torch.cat(parts)
    grad_hidden = None
    if need_hidden:
        y_dtype = a.dtype if y is None else y.dtype
        grad_y = (grad_out * weights).to(y_dtype)
        grad_a = mm(grad_y.contiguous(), w_d, False, counts)
        grad_gate = torch.ops.aten.silu_backward(grad_a * up, gate)
        grad_h = torch.cat([grad_gate, grad_a * s], dim=-1)
        grad_hidden = mm(grad_h, w_gu, False, counts).to(hidden.dtype)
    return grad_hidden, grad_weights


def weighted_expert_function(forward: Any, depth: int = 4) -> Any:
    """The ``WeightedFrozenExpert`` Function that ``StreamedExperts.forward`` closes over, found
    through wrappers (the phase clock and the segmented hook wrap ``forward``)."""
    found = _find_in_closure(forward, depth)
    if found is None:
        raise RuntimeError("StreamedExperts.forward does not reference WeightedFrozenExpert")
    return found


def _find_in_closure(function: Any, depth: int) -> Any:
    function = getattr(function, "__func__", function)
    code = getattr(function, "__code__", None)
    if code is None or depth < 0:
        return None
    cells = list(zip(code.co_freevars, function.__closure__ or ()))
    for name, cell in cells:
        if name == "WeightedFrozenExpert":
            return cell.cell_contents
    inner = [getattr(function, "__wrapped__", None)]
    for _, cell in cells:
        try:
            inner.append(cell.cell_contents)
        except ValueError:  # empty cell
            continue
    for candidate in inner:
        if callable(candidate) and not isinstance(candidate, type):
            found = _find_in_closure(candidate, depth - 1)
            if found is not None:
                return found
    return None


@contextmanager
def direct_backward(
    plan_for: Callable[[Tuple[int, int], int], Optional[List[int]]],
    mm: SegmentMatmul = segment_mm,
    stats: Optional[dict] = None,
    function: Any = None,
) -> Iterator[dict]:
    """Swap ``WeightedFrozenExpert.backward`` for ``expert_vjp`` on experts with a segment plan.

    ``plan_for((layer, expert), rows)`` gives the segment row counts (None: keep the autograd
    backward). Lease, prefetch hint and checks are those of the original backward.
    ``function`` defaults to the one the current ``StreamedExperts.forward`` uses; pass it when
    that forward is already wrapped."""
    import torch

    from soup_cli.utils import expert_stream

    stats = stats if stats is not None else {}
    stats.setdefault("direct", 0)
    stats.setdefault("autograd", 0)
    if function is None:
        function = weighted_expert_function(expert_stream.streamed_experts_class().forward)
    original = function.backward

    def backward(ctx: Any, grad_out: Any) -> Tuple[Any, ...]:
        hidden, weights = ctx.saved_tensors
        counts = plan_for((ctx.owner.layer_idx, ctx.expert), int(hidden.shape[0]))
        if counts is None or torch.is_grad_enabled():
            stats["autograd"] += 1
            return original(ctx, grad_out)
        need_hidden, need_weights = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        if not (need_hidden or need_weights):
            return None, None, None, None, None, None, None
        expert_stream._validate_training_activation(ctx.act_fn)
        autocast_dtype = ctx.autocast["dtype"] if ctx.autocast["enabled"] else None
        with ctx.owner.pool.lease(
            ctx.owner.reader, ctx.expert, phase="backward_dX", layer_idx=ctx.owner.layer_idx
        ) as (gate_up, down):
            ctx.owner.pool.prefetch(ctx.owner.reader, ctx.hint_prev)
            with torch.autocast(ctx.device_type, enabled=False):
                grad_hidden, grad_weights = expert_vjp(
                    hidden.detach(), weights.detach(), gate_up, down, grad_out, counts,
                    need_hidden=need_hidden, need_weights=need_weights,
                    autocast_dtype=autocast_dtype, mm=mm,
                )
        stats["direct"] += 1
        return grad_hidden, grad_weights, None, None, None, None, None

    function.backward = staticmethod(backward)
    try:
        yield stats
    finally:
        function.backward = staticmethod(original)


_SILU_TYPES = ("SiLU", "SiLUActivation")


def _python_forward(hidden: Any, weights: Any, gate_up: Any, down: Any,
                    counts: Sequence[int]) -> Any:
    """``expert_mlp(hidden) * weights`` with A0 calls per segment (fallback of the fused node)."""
    import torch.nn.functional as functional

    h = segment_mm(hidden.contiguous(), gate_up, True, counts)
    gate, up = h.chunk(2, dim=-1)
    return segment_mm(functional.silu(gate) * up, down, True, counts) * weights


@contextmanager
def fused_node(
    plan_for: Callable[[Tuple[int, int], int], Optional[List[int]]],
    fused: Any,
    stats: dict,
    function: Any,
) -> Iterator[dict]:
    """Swap ``WeightedFrozenExpert.forward`` and ``.backward`` for one extension call each
    (``fused.backend.expert_forward`` / ``expert_backward``) on experts with a segment plan,
    outside autocast, with the stateless SiLU. Lease, prefetch hints, saved tensors and the
    replay branch are those of the original node; anything else keeps the original code."""
    import torch

    from soup_cli.utils import expert_stream

    for key in ("fused_forward", "python_forward", "fused_backward", "python_backward",
                "autograd"):
        stats.setdefault(key, 0)
    original_forward, original_backward = function.forward, function.backward
    # Per call work that does not change between calls is done once: the activation type of an
    # owner, and the (autocast disabled) state the backward reads back from the node.
    silu_owners: Dict[int, bool] = {}
    autocast_off = {
        "enabled": False,
        "dtype": torch.get_autocast_dtype("cuda"),
        "cache_enabled": torch.is_autocast_cache_enabled(),
    }

    def forward(ctx: Any, hidden: Any, weights: Any, owner: Any, expert: int, replay: bool,
                nxt: Any = None, prv: Any = None) -> Any:
        device_type = hidden.device.type
        counts = plan_for((owner.layer_idx, expert), int(hidden.shape[0]))
        silu = silu_owners.get(id(owner))
        if silu is None:
            silu = silu_owners[id(owner)] = type(owner.act_fn).__name__ in _SILU_TYPES
        if counts is None or not silu or torch.is_autocast_enabled(device_type):
            return original_forward(ctx, hidden, weights, owner, expert, replay, nxt, prv)
        ctx.owner = owner
        ctx.act_fn = owner.act_fn
        ctx.expert = expert
        ctx.hint_prev = prv
        ctx.device_type = device_type
        ctx.autocast = autocast_off
        ctx.save_for_backward(hidden, weights)
        if replay:
            return hidden.new_zeros(()).expand_as(hidden)
        scope = expert_stream._active_scope()
        phase = "recompute_forward" if scope is not None and scope.replay else "init_forward"
        with owner.pool.lease(
            owner.reader, expert, phase=phase, layer_idx=owner.layer_idx
        ) as (gate_up, down):
            owner.pool.prefetch(owner.reader, nxt)
            if fused.usable(hidden, gate_up, down) and fused.ensure(counts, gate_up, down):
                stats["fused_forward"] += 1
                return fused.backend.expert_forward(hidden, weights, gate_up, down, counts)
            stats["python_forward"] += 1
            return _python_forward(hidden, weights, gate_up, down, counts)

    def backward(ctx: Any, grad_out: Any) -> Tuple[Any, ...]:
        hidden, weights = ctx.saved_tensors
        counts = plan_for((ctx.owner.layer_idx, ctx.expert), int(hidden.shape[0]))
        if counts is None or torch.is_grad_enabled() or ctx.autocast["enabled"]:
            stats["autograd"] += 1
            return original_backward(ctx, grad_out)
        need_hidden, need_weights = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        if not (need_hidden or need_weights):
            return None, None, None, None, None, None, None
        expert_stream._validate_training_activation(ctx.act_fn)
        with ctx.owner.pool.lease(
            ctx.owner.reader, ctx.expert, phase="backward_dX", layer_idx=ctx.owner.layer_idx
        ) as (gate_up, down):
            ctx.owner.pool.prefetch(ctx.owner.reader, ctx.hint_prev)
            if fused.usable(hidden, gate_up, down) and fused.ensure(counts, gate_up, down):
                grad_hidden, grad_weights = fused.backend.expert_backward(
                    hidden.detach(), weights.detach(), gate_up, down, grad_out, counts,
                    need_hidden, need_weights,
                )
                stats["fused_backward"] += 1
            else:
                grad_hidden, grad_weights = expert_vjp(
                    hidden.detach(), weights.detach(), gate_up, down, grad_out, counts,
                    need_hidden=need_hidden, need_weights=need_weights,
                )
                stats["python_backward"] += 1
        return (grad_hidden if need_hidden else None, grad_weights if need_weights else None,
                None, None, None, None, None)

    function.forward = staticmethod(forward)
    function.backward = staticmethod(backward)
    try:
        yield stats
    finally:
        function.forward = staticmethod(original_forward)
        function.backward = staticmethod(original_backward)
