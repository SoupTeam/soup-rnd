"""Zero-materialization streaming M2+ drop-in for ``separate_fast`` (+ optional H-C1 L2 split).

Eliminates three full-window ``top_k``-fold BF16 allocations present in
``p3-nf4-k64-4060ti-20261119/source/separate_fast.py``:
- ``h_sorted = h_all[sorted_tokens]`` (448 MiB on DeepSeek-V3 671B at k=64, S=64)
- ``dy_sorted = dy_all[sorted_tokens]`` (448 MiB)
- ``dh_chunks`` / ``dw_chunks`` dicts across all 256 experts (448 MiB)

Instead gathers ``h_all[tok_slice]`` and ``dy_all[tok_slice]`` per expert inside the loop
(~1.75 MiB transient per expert) and streams ``dh_all.index_add_`` directly in reverse
expert order — preserving 100% bit-for-bit identical gradients to ``separate_fast.py``
while saving 1.31 GiB at k=64 (2.63 GiB at k=128) on 671B.

Also supports opt-in ``split_gate_up_for_l2=True`` (H-C1): splits the 56 MiB ``gate_up``
matrix of DeepSeek-V3 671B into two 28 MiB halves (``gate_w`` and ``up_w``) so each
projection stays resident inside the RTX 4060 Ti's 32 MiB L2 cache across micro-batch splits.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


def build_window_route_plan(
    indices: list[Any], num_experts: int
) -> tuple[Any, Any, list[int], dict[int, tuple[int, int, list[int]]]]:
    """Group all (mb, slot, local_token) routes by (expert, mb, slot, local_token) in 1 sort."""
    import torch

    device = indices[0].device
    num_mbs = len(indices)
    top_k = int(indices[0].shape[1])
    lengths = [int(idx.shape[0]) for idx in indices]
    max_seq = max(lengths) if lengths else 1
    idx_cat = torch.cat(indices, dim=0)
    total_tokens = int(idx_cat.shape[0])
    if total_tokens == 0:
        return (
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            lengths,
            {},
        )

    if len(set(lengths)) == 1:
        seq = lengths[0]
        tok_ids = torch.arange(total_tokens, device=device, dtype=torch.int64)
        mb_ids = torch.div(tok_ids, seq, rounding_mode="floor")
        local_tok = tok_ids - mb_ids * seq
    else:
        mb_parts = []
        loc_parts = []
        for mb, length in enumerate(lengths):
            mb_parts.append(torch.full((length,), mb, device=device, dtype=torch.int64))
            loc_parts.append(torch.arange(length, device=device, dtype=torch.int64))
        mb_ids = torch.cat(mb_parts, dim=0)
        local_tok = torch.cat(loc_parts, dim=0)
        tok_ids = torch.arange(total_tokens, device=device, dtype=torch.int64)

    slot_ids = torch.arange(top_k, device=device, dtype=torch.int64).unsqueeze(0)
    flat_expert = idx_cat.reshape(-1)
    flat_mb = mb_ids.unsqueeze(1).expand(-1, top_k).reshape(-1)
    flat_tok = tok_ids.unsqueeze(1).expand(-1, top_k).reshape(-1)
    flat_loc = local_tok.unsqueeze(1).expand(-1, top_k).reshape(-1)
    flat_slot = slot_ids.expand(total_tokens, -1).reshape(-1)

    stride_slot = max_seq
    stride_mb = top_k * stride_slot
    stride_exp = num_mbs * stride_mb
    sort_key = (
        flat_expert * stride_exp + flat_mb * stride_mb + flat_slot * stride_slot + flat_loc
    )
    perm = torch.argsort(sort_key, stable=True)
    sorted_tokens = flat_tok[perm]
    sorted_slots = flat_slot[perm]

    pair_ids = flat_expert * num_mbs + flat_mb
    counts_2d = (
        torch.bincount(pair_ids, minlength=num_experts * num_mbs)
        .reshape(num_experts, num_mbs)
        .cpu()
        .tolist()
    )

    plan: dict[int, tuple[int, int, list[int]]] = {}
    cursor = 0
    for expert_id, mb_row in enumerate(counts_2d):
        nz = [cnt for cnt in mb_row if cnt > 0]
        if not nz:
            continue
        span = sum(nz)
        plan[expert_id] = (cursor, cursor + span, nz)
        cursor += span
    return sorted_tokens, sorted_slots, lengths, plan


def _closed_form_swiglu_forward(
    h_slice: Any,
    mb_sizes: list[int],
    gate_up: Any,
    down: Any,
    act_fn: Any,
    *,
    split_gate_up_for_l2: bool = False,
) -> Any:
    import torch
    import torch.nn.functional as functional

    if len(mb_sizes) == 1:
        gu = functional.linear(h_slice, gate_up)
        gate, up = gu.chunk(2, dim=-1)
        mid = act_fn(gate) * up
        return functional.linear(mid, down)

    parts = h_slice.split(mb_sizes, dim=0)
    if split_gate_up_for_l2:
        half = gate_up.shape[0] // 2
        gate_w, up_w = gate_up[:half], gate_up[half:]
        gate = torch.cat([functional.linear(part, gate_w) for part in parts], dim=0)
        up = torch.cat([functional.linear(part, up_w) for part in parts], dim=0)
    else:
        gu = torch.cat([functional.linear(part, gate_up) for part in parts], dim=0)
        gate, up = gu.chunk(2, dim=-1)
    mid = act_fn(gate) * up
    return torch.cat(
        [functional.linear(part, down) for part in mid.split(mb_sizes, dim=0)], dim=0
    )


def _closed_form_swiglu_backward(
    h_slice: Any,
    w_slice: Any,
    dy_slice: Any,
    mb_sizes: list[int],
    gate_up: Any,
    down: Any,
    autocast_cfg: dict[str, Any],
    need_h: bool,
    need_w: bool,
    *,
    route_grad_per_microbatch: bool = True,
    split_gate_up_for_l2: bool = False,
) -> tuple[Any | None, Any | None]:
    import torch
    import torch.nn.functional as functional

    compute_dtype = autocast_cfg["dtype"] if autocast_cfg["enabled"] else h_slice.dtype
    h_comp = h_slice.to(compute_dtype) if h_slice.dtype != compute_dtype else h_slice
    gu_w = gate_up.to(compute_dtype) if gate_up.dtype != compute_dtype else gate_up
    dn_w = down.to(compute_dtype) if down.dtype != compute_dtype else down

    if len(mb_sizes) == 1:
        gu = functional.linear(h_comp, gu_w)
        gate, up = gu.chunk(2, dim=-1)
    elif split_gate_up_for_l2:
        half = gu_w.shape[0] // 2
        gate_w, up_w = gu_w[:half], gu_w[half:]
        parts = h_comp.split(mb_sizes, dim=0)
        gate = torch.cat([functional.linear(part, gate_w) for part in parts], dim=0)
        up = torch.cat([functional.linear(part, up_w) for part in parts], dim=0)
    else:
        gu = torch.cat(
            [functional.linear(part, gu_w) for part in h_comp.split(mb_sizes, dim=0)], dim=0
        )
        gate, up = gu.chunk(2, dim=-1)

    act = functional.silu(gate)
    mid = act * up
    if len(mb_sizes) == 1:
        unweighted = functional.linear(mid, dn_w)
    else:
        unweighted = torch.cat(
            [functional.linear(part, dn_w) for part in mid.split(mb_sizes, dim=0)], dim=0
        )

    prod_dtype = torch.promote_types(unweighted.dtype, w_slice.dtype)
    dy_prod = dy_slice.to(prod_dtype)

    dw_out = None
    if need_w:
        product = dy_prod * unweighted.to(prod_dtype)
        if route_grad_per_microbatch and len(mb_sizes) > 1:
            grad_w_prod = torch.cat(
                [part.sum(dim=-1, keepdim=True) for part in product.split(mb_sizes, dim=0)],
                dim=0,
            )
        else:
            grad_w_prod = product.sum(dim=-1, keepdim=True)
        dw_out = grad_w_prod.to(w_slice.dtype)

    dh_out = None
    if need_h:
        d_unweighted = (dy_prod * w_slice.to(prod_dtype)).to(dn_w.dtype)
        if len(mb_sizes) == 1:
            d_mid = torch.matmul(d_unweighted, dn_w)
        else:
            d_mid = torch.cat(
                [torch.matmul(part, dn_w) for part in d_unweighted.split(mb_sizes, dim=0)],
                dim=0,
            )
        d_act = d_mid * up
        d_up = d_mid * act
        d_gate = torch.ops.aten.silu_backward.default(d_act, gate)
        if len(mb_sizes) == 1:
            d_gu = torch.cat((d_gate, d_up), dim=-1)
            dh_comp = torch.matmul(d_gu, gu_w)
        elif split_gate_up_for_l2:
            half = gu_w.shape[0] // 2
            gate_w, up_w = gu_w[:half], gu_w[half:]
            dh_gate = torch.cat(
                [torch.matmul(part, gate_w) for part in d_gate.split(mb_sizes, dim=0)], dim=0
            )
            dh_up = torch.cat(
                [torch.matmul(part, up_w) for part in d_up.split(mb_sizes, dim=0)], dim=0
            )
            dh_comp = dh_gate + dh_up
        else:
            d_gu = torch.cat((d_gate, d_up), dim=-1)
            dh_comp = torch.cat(
                [torch.matmul(part, gu_w) for part in d_gu.split(mb_sizes, dim=0)], dim=0
            )
        dh_out = dh_comp.to(h_slice.dtype)

    return dh_out, dw_out


@lru_cache(maxsize=4)
def _get_streaming_separate_fn(
    route_grad_per_microbatch: bool, split_gate_up_for_l2: bool
) -> Any:
    import torch

    from soup_cli.utils.expert_stream import _validate_training_activation

    class StreamingWindowSeparateExpertsFn(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: Any,
            owner: Any,
            controller: Any,
            replay: bool,
            num_mbs: int,
            *tensors: Any,
        ) -> Any:
            ctx.set_materialize_grads(False)
            hiddens = list(tensors[:num_mbs])
            indices = list(tensors[num_mbs : 2 * num_mbs])
            weights = list(tensors[2 * num_mbs : 3 * num_mbs])

            ctx.owner = owner
            ctx.controller = controller
            ctx.act_fn = owner.act_fn
            ctx.num_mbs = num_mbs
            ctx.device_type = hiddens[0].device.type
            ctx.autocast = {
                "enabled": torch.is_autocast_enabled(ctx.device_type),
                "dtype": torch.get_autocast_dtype(ctx.device_type),
                "cache_enabled": torch.is_autocast_cache_enabled(),
            }
            ctx.save_for_backward(*hiddens, *indices, *weights)

            if replay:
                return tuple(h.new_zeros(()).expand_as(h) for h in hiddens)

            sorted_tokens, sorted_slots, lengths, plan = build_window_route_plan(
                indices, owner.num_experts
            )
            h_all = torch.cat(hiddens, dim=0)
            w_all = torch.cat(weights, dim=0)
            out_all = torch.zeros_like(h_all)
            if not plan:
                return tuple(out_all.split(lengths, dim=0))

            active_experts = sorted(plan.keys())

            def _run_one(eid: int, gate_up: Any, down: Any) -> None:
                lo, hi, mb_sizes = plan[eid]
                tok_slice = sorted_tokens[lo:hi]
                slot_slice = sorted_slots[lo:hi]
                h_e = h_all[tok_slice]
                w_e = w_all[tok_slice, slot_slice].unsqueeze(-1)
                y_e = (
                    _closed_form_swiglu_forward(
                        h_e,
                        mb_sizes,
                        gate_up,
                        down,
                        ctx.act_fn,
                        split_gate_up_for_l2=split_gate_up_for_l2,
                    )
                    * w_e
                )
                out_all.index_add_(0, tok_slice, y_e.to(out_all.dtype))

            if controller is not None:
                for eid, gate_up, down in controller.iter_experts(owner.reader, active_experts):
                    _run_one(eid, gate_up, down)
            else:
                for eid in active_experts:
                    with owner.pool.lease(owner.reader, eid) as (gate_up, down):
                        _run_one(eid, gate_up, down)

            return tuple(out_all.split(lengths, dim=0))

        @staticmethod
        def backward(ctx: Any, *upstreams: Any) -> Any:
            if torch.is_grad_enabled():
                raise RuntimeError("the research expert is first-order only")
            _validate_training_activation(ctx.act_fn)

            num_mbs = ctx.num_mbs
            saved = ctx.saved_tensors
            hiddens = saved[:num_mbs]
            indices = saved[num_mbs : 2 * num_mbs]
            weights = saved[2 * num_mbs : 3 * num_mbs]

            need_h_flags = list(ctx.needs_input_grad[4 : 4 + num_mbs])
            need_w_flags = list(ctx.needs_input_grad[4 + 2 * num_mbs : 4 + 3 * num_mbs])
            active_mb = [
                upstreams[mb] is not None and (need_h_flags[mb] or need_w_flags[mb])
                for mb in range(num_mbs)
            ]
            if not any(active_mb):
                return (None, None, None, None, *([None] * (3 * num_mbs)))

            sub_map = [mb for mb, flag in enumerate(active_mb) if flag]
            sorted_tokens, sorted_slots, lengths, plan = build_window_route_plan(
                [indices[mb] for mb in sub_map], ctx.owner.num_experts
            )
            h_all = torch.cat([hiddens[mb] for mb in sub_map], dim=0)
            w_all = torch.cat([weights[mb] for mb in sub_map], dim=0)
            dy_all = torch.cat([upstreams[mb] for mb in sub_map], dim=0)
            any_h = any(need_h_flags[mb] for mb in sub_map)
            any_w = any(need_w_flags[mb] for mb in sub_map)
            dh_all = torch.zeros_like(h_all) if any_h else None
            dw_all = torch.zeros_like(w_all) if any_w else None

            if plan:
                rev_experts = sorted(plan.keys(), reverse=True)

                def _bwd_one(eid: int, gate_up: Any, down: Any) -> None:
                    lo, hi, mb_sizes = plan[eid]
                    tok_slice = sorted_tokens[lo:hi]
                    slot_slice = sorted_slots[lo:hi]
                    dh_e, dw_e = _closed_form_swiglu_backward(
                        h_all[tok_slice],
                        w_all[tok_slice, slot_slice].unsqueeze(-1),
                        dy_all[tok_slice],
                        mb_sizes,
                        gate_up,
                        down,
                        ctx.autocast,
                        any_h,
                        any_w,
                        route_grad_per_microbatch=route_grad_per_microbatch,
                        split_gate_up_for_l2=split_gate_up_for_l2,
                    )
                    if dh_all is not None and dh_e is not None:
                        dh_all.index_add_(0, tok_slice, dh_e)
                    if dw_all is not None and dw_e is not None:
                        dw_all[tok_slice, slot_slice] = dw_e.squeeze(-1)

                if ctx.controller is not None:
                    for eid, gate_up, down in ctx.controller.iter_experts(
                        ctx.owner.reader, rev_experts
                    ):
                        _bwd_one(eid, gate_up, down)
                else:
                    for eid in rev_experts:
                        with ctx.owner.pool.lease(ctx.owner.reader, eid) as (gate_up, down):
                            _bwd_one(eid, gate_up, down)

            dh_sub = dh_all.split(lengths, dim=0) if dh_all is not None else [None] * len(sub_map)
            dw_sub = dw_all.split(lengths, dim=0) if dw_all is not None else [None] * len(sub_map)
            grad_h = [None] * num_mbs
            grad_w = [None] * num_mbs
            for pos, mb in enumerate(sub_map):
                if need_h_flags[mb]:
                    grad_h[mb] = dh_sub[pos]
                if need_w_flags[mb]:
                    grad_w[mb] = dw_sub[pos]
            return (None, None, None, None, *grad_h, *([None] * num_mbs), *grad_w)

    return StreamingWindowSeparateExpertsFn


def separate_experts_fast_streaming_m2(
    owner: Any,
    hiddens: list[Any],
    indices: list[Any],
    weights: list[Any],
    controller: Any = None,
    *,
    route_grad_per_microbatch: bool = True,
    split_gate_up_for_l2: bool = False,
) -> list[Any]:
    """Drop-in zero-materialization M2+ version of ``separate_experts_fast``."""
    import torch

    from soup_cli.utils.expert_stream import (
        _active_scope,
        _validate_routing,
        _validate_training_activation,
    )

    if not owner._forward_lock.acquire(blocking=False):
        raise RuntimeError("expert forward is already active")
    try:
        owner.pool.validate_reader(owner.reader)
        for hidden, index, weight in zip(hiddens, indices, weights, strict=True):
            _validate_routing(hidden, index, weight, owner.pool)
            if index.shape[1] > owner.num_experts or (
                index.numel()
                and (index.min().item() < 0 or index.max().item() >= owner.num_experts)
            ):
                raise ValueError("routing contains an expert outside the reader")
            if torch.is_grad_enabled() and (hidden.requires_grad or weight.requires_grad):
                _validate_training_activation(owner.act_fn)
        scope = _active_scope()
        replay = scope is not None and scope.replay
        fn = _get_streaming_separate_fn(route_grad_per_microbatch, split_gate_up_for_l2)
        return list(fn.apply(owner, controller, replay, len(hiddens), *hiddens, *indices, *weights))
    finally:
        owner._forward_lock.release()
