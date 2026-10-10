"""Fast-LoRA shared-X Q/K/V autograd path for issue #838.

The q projection computes Q/K/V together and caches K/V only until the next
two projection calls with unchanged input, projection state and execution
context. Changed grad/autocast modes, adapter state or tensor versions discard
the speculative K/V and delegate to PEFT, as do cross-attention calls with a
different key/value tensor. This avoids copying architecture-specific attention
forwards while sharing one X@A_cat.T GEMM in the legacy same-context path.

An exact, pristine installed HF LlamaAttention running its canonical forward
may instead select correctness-first separate reference arithmetic for fp16/bf16
X with present fp32 adapters and autocast disabled. Its call-local scope/cache
cannot be activated by standalone q/k/v calls. ``grad_fn.reference_order`` reports
the selected arithmetic. Read-only projection output observers are supported;
mutating hooks and rewritten inner projections are not in this reference scope.
This relinquishes grouped GEMM fusion in that path, not a performance promise or
a guarantee of bit-exactness on other graphs, schedulers or devices.

A lone q_proj call intentionally retains one {x, k, v} graph until backward,
the next q call or unpatch. That is bounded to one attention module rather than a leak,
but callers doing projection-only probes should unpatch or issue the normal
k/v calls so the short-lived cache is drained.
"""

from __future__ import annotations

import types
import weakref
from contextvars import ContextVar
from typing import Any

from soup_cli.utils.fast_lora import (
    _FORWARD_OWNER_MARKER,
    _GROUP_PATCH_OWNER_MARKER,
    _as_dtype,
    _dense_weight,
    _flatten,
    _is_supported_lora_projection,
    _projection_state,
    _reference_parent,
    _reference_projections,
    _scaled_lora_add,
)

_PATCH_MARKER = "_soup_fast_lora_qkv"
_SINGLE_PROJECTION_PATCH_MARKER = "_soup_fast_lora_single_projection"
_CACHE_MARKER = "_soup_fast_lora_qkv_cache"
_CACHE_HIT_MARKER = "_soup_fast_lora_qkv_last_cache_hits"
_RESTORE_MARKER = "_soup_fast_lora_qkv_restore"
_INSTALLED_FORWARDS_MARKER = "_soup_fast_lora_qkv_installed_forwards"
_OWNER = "qkv"
_FUNCTION: Any = None
_REFERENCE_FUNCTION: Any = None
_PARENT_RESTORE_MARKER = "_soup_fast_lora_qkv_parent_restore"
_CALL: ContextVar[Any] = ContextVar("fast_lora_qkv_call", default=None)


class _AttentionCall:
    """Call-local cache: no scope leaks across threads, nesting or exceptions."""

    def __init__(self, owner: Any, reference: bool) -> None:
        self.owner = weakref.ref(owner)
        self.reference = reference
        self.generation = object()
        self.step = 0
        self.cache: Any = None

    def advance(self, name: str) -> None:
        if self.step >= 3 or name != ("q_proj", "k_proj", "v_proj")[self.step]:
            self.reference = False
        self.step += 1

__all__ = ["patch_fast_lora_qkv", "unpatch_fast_lora_qkv"]


def _qkv_function(*, reference_order: bool = False) -> Any:
    global _FUNCTION, _REFERENCE_FUNCTION
    cached = _REFERENCE_FUNCTION if reference_order else _FUNCTION
    if cached is not None:
        return cached

    import torch
    import torch.nn.functional as functional

    class _FastLoraQKV(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx, x, wq, bq, wk, bk, wv, bv,
            aq, bql, ak, bkl, av, bvl,
            sq, sk, sv, qq_meta, qk_meta, qv_meta, *qparts,
        ):
            # An unused projection must leave its adapter grads as None, not
            # zeros (optimizers apply weight decay to a materialized zero).
            ctx.set_materialize_grads(False)
            ctx.reference_order = reference_order
            metas = (qq_meta, qk_meta, qv_meta)
            counts = [0 if meta is None else int(meta["_count"]) for meta in metas]
            starts = (0, counts[0], counts[0] + counts[1])
            qlists = [
                list(qparts[starts[i] : starts[i] + counts[i]]) for i in range(3)
            ]
            weights = (wq, wk, wv)
            biases = (bq, bk, bv)
            outs = []
            for weight, bias, meta, parts in zip(weights, biases, metas, qlists):
                dense = _dense_weight(weight, meta, parts, x.dtype)
                outs.append(
                    functional.linear(
                        x, dense, _as_dtype(bias, x.dtype) if bias is not None else None
                    )
                )
                del dense

            adapters = ((aq, bql, sq), (ak, bkl, sk), (av, bvl, sv))
            ranks = [a.shape[0] if a.numel() else 0 for a, _b, _s in adapters]
            present = [i for i, rank in enumerate(ranks) if rank]
            if present:
                if reference_order:
                    h = torch.cat([
                        functional.linear(_as_dtype(x, adapters[i][0].dtype), adapters[i][0])
                        for i in present
                    ], dim=-1)
                else:
                    a_cat = torch.cat([adapters[i][0] for i in present], dim=0)
                    h = functional.linear(_as_dtype(x, a_cat.dtype), a_cat)
                cursor = 0
                for i in present:
                    a, b, scaling = adapters[i]
                    rank = a.shape[0]
                    hi = h[..., cursor : cursor + rank]
                    outs[i] = _scaled_lora_add(
                        outs[i], hi, b, scaling, quantized=metas[i] is not None
                    )
                    cursor += rank
            else:
                h = x.new_empty((*x.shape[:-1], 0))

            ctx.ranks = ranks
            ctx.present = present
            ctx.scalings = (float(sq), float(sk), float(sv))
            ctx.qmetas = metas
            ctx.qcounts = counts
            ctx.qparts_len = len(qparts)

            ctx.save_for_backward(
                x, wq, wk, wv, aq, bql, ak, bkl, av, bvl, h, *qparts
            )
            return tuple(outs)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, grad_q, grad_k, grad_v):
            saved = ctx.saved_tensors
            x, wq, wk, wv, aq, bql, ak, bkl, av, bvl, h = saved[:11]
            qparts = list(saved[11:])
            grads_out = (grad_q, grad_k, grad_v)
            weights = (wq, wk, wv)
            adapters = ((aq, bql), (ak, bkl), (av, bvl))
            sq, sk, sv = ctx.scalings
            scalings = (sq, sk, sv)
            cq, ck, cv = ctx.qcounts
            qlists = (
                qparts[:cq],
                qparts[cq : cq + ck],
                qparts[cq + ck : cq + ck + cv],
            )

            if ctx.reference_order:
                grad_x = None
                grad_as = [None, None, None]
                grad_bs = [None, None, None]
                adapter_dx = [None, None, None]
                cursor = 0
                for i in ctx.present:
                    a, b = adapters[i]
                    rank = ctx.ranks[i]
                    hi = h[..., cursor : cursor + rank].contiguous()
                    cursor += rank
                    grad = grads_out[i]
                    if grad is None:
                        continue
                    scaled = _as_dtype(grad, b.dtype) * scalings[i]
                    grad_bs[i] = _flatten(scaled).t() @ _flatten(hi)
                    dhi = torch.matmul(scaled, b)
                    grad_as[i] = _flatten(dhi).t() @ _as_dtype(_flatten(x), dhi.dtype)
                    if ctx.needs_input_grad[0]:
                        adapter_dx[i] = _as_dtype(torch.matmul(dhi, a), x.dtype)
                if ctx.needs_input_grad[0]:
                    # Standard Llama evaluates q, k, v. Keep each reverse edge
                    # (adapter then base) and its low-precision addition separate.
                    for i in reversed(range(3)):
                        grad = grads_out[i]
                        if grad is None:
                            continue
                        if adapter_dx[i] is not None:
                            term = adapter_dx[i]
                            grad_x = term if grad_x is None else grad_x + term
                        dense = _dense_weight(weights[i], ctx.qmetas[i], qlists[i], grad.dtype)
                        term = torch.matmul(grad, dense)
                        grad_x = term if grad_x is None else grad_x + term
                        del dense
                result = [
                    grad_x, None, None, None, None, None, None,
                    grad_as[0], grad_bs[0], grad_as[1], grad_bs[1], grad_as[2], grad_bs[2],
                    None, None, None, None, None, None,
                ]
                result.extend([None] * ctx.qparts_len)
                return tuple(result)

            grad_x = None
            # A frozen prefix can feed a trainable QKV adapter without needing
            # dX. The adapter gradients still need X, but none of the three
            # frozen bases: skip their GEMMs and NF4 dequantization entirely.
            if ctx.needs_input_grad[0]:
                for grad, weight, meta, parts in zip(
                    grads_out, weights, ctx.qmetas, qlists
                ):
                    if grad is None:
                        continue
                    dense = _dense_weight(weight, meta, parts, grad.dtype)
                    term = torch.matmul(grad, dense)

                    grad_x = term if grad_x is None else torch.add(grad_x, term)
                    del dense

            grad_as = [None, None, None]
            grad_bs = [None, None, None]
            used = [i for i in ctx.present if grads_out[i] is not None]
            if used:
                dh_parts = []
                cursor = 0
                for i in ctx.present:
                    a, b = adapters[i]
                    grad = grads_out[i]
                    rank = ctx.ranks[i]
                    hi = h[..., cursor : cursor + rank]
                    cursor += rank
                    if grad is None:
                        continue
                    grad_bs[i] = (
                        _flatten(_as_dtype(grad, hi.dtype) * scalings[i]).t() @ _flatten(hi)
                    )
                    dhi = torch.matmul(_as_dtype(grad, b.dtype) * scalings[i], b)
                    dh_parts.append(dhi)

                dh = torch.cat(dh_parts, dim=-1)
                grad_a_cat = _flatten(dh).t() @ _as_dtype(_flatten(x), dh.dtype)
                if grad_x is not None:
                    a_cat = torch.cat([adapters[i][0] for i in used], dim=0)
                    grad_x = torch.add(
                        grad_x,
                        _as_dtype(
                            torch.matmul(dh, _as_dtype(a_cat, dh.dtype)), grad_x.dtype
                        ),
                    )
                cursor = 0

                for i in used:
                    rank = ctx.ranks[i]
                    grad_as[i] = grad_a_cat[cursor : cursor + rank]
                    cursor += rank

            result = [
                grad_x,
                None, None, None, None, None, None,
                grad_as[0], grad_bs[0],
                grad_as[1], grad_bs[1],
                grad_as[2], grad_bs[2],
                None, None, None, None, None, None,
            ]
            result.extend([None] * ctx.qparts_len)
            return tuple(result)

    if reference_order:
        _REFERENCE_FUNCTION = _FastLoraQKV
    else:
        _FUNCTION = _FastLoraQKV
    return _FastLoraQKV


def _tensor_cache_key(tensor: Any) -> Any:
    """Track replacements, in-place writes, storage swaps and autograd flags."""
    if tensor is None or not tensor.numel():
        return None
    if tensor.is_inference():
        return None  # No version counter: reuse cannot be validated safely.
    return (
        id(tensor), tensor._version, tensor.data_ptr(), tensor.dtype, tensor.device,
        tuple(tensor.shape), tuple(tensor.stride()), tensor.requires_grad,
        # detach_().requires_grad_(True) changes lineage without a version bump.
        # Keep the node wrapper itself: its id can be recycled between reads.
        tensor.is_leaf, tensor.grad_fn,
    )


def _projection_cache_key(proj: Any, state: Any) -> Any:
    if state is None:
        return None
    tensors = (state.weight, state.bias, state.lora_a, state.lora_b, *state.qparts)
    if any(t is not None and t.numel() and t.is_inference() for t in tensors):
        return None
    return (
        id(proj), proj.training, tuple(getattr(proj, "active_adapters", ())),
        state.scaling, state.compute_dtype,
        None if state.qmeta is None else tuple(sorted(state.qmeta.items())),
        tuple(_tensor_cache_key(tensor) for tensor in tensors),
    )


def _execution_cache_key(x: Any) -> tuple[Any, ...]:
    import torch

    device_type = x.device.type
    autocast = torch.is_autocast_enabled(device_type)
    return (
        torch.is_grad_enabled(), torch.is_inference_mode_enabled(), autocast,
        torch.get_autocast_dtype(device_type) if autocast else None,
    )


def _install_attention_patch(attn: Any) -> None:
    fast = _qkv_function()
    attn_ref = weakref.ref(attn)
    projections = {name: getattr(attn, name) for name in ("q_proj", "k_proj", "v_proj")}
    originals = {name: proj.forward for name, proj in projections.items()}
    restore = {
        name: ("forward" in proj.__dict__, proj.__dict__.get("forward"))
        for name, proj in projections.items()
    }
    original_parent = attn.forward
    forwards: dict[str, Any] = {}

    def scope() -> Any:
        call = _CALL.get()
        return call if call is not None and call.owner() is attn else None

    def get_cache() -> Any:
        call = scope()
        return call.cache if call is not None else getattr(attn, _CACHE_MARKER, None)

    def set_cache(cache: Any) -> None:
        call = scope()
        if call is not None:
            call.cache = cache
        else:
            setattr(attn, _CACHE_MARKER, cache)

    def mode_key(x: Any, states: Any = None) -> Any:
        call = scope()
        reference = False
        if call is not None and call.reference:
            current = [getattr(attn, name) for name in projections]
            if states is None:
                states = [_projection_state(proj, x, allow_unadapted=True) for proj in current]
            reference = (
                getattr(attn.forward, "__func__", None) is parent_forward
                and _reference_parent(attn, "qkv", original_parent)
                and all(getattr(proj.forward, "__func__", None) is forwards[name]
                        for name, proj in zip(projections, current))
                and _reference_projections(
                    current, list(originals.values()), states, x, output_observers=True
                )
            )
        return (None if call is None else call.generation, reference)

    def parent_forward(self, *args, **kwargs):
        # Standalone q/k/v calls do not prove a canonical graph. The verified
        # original HF method must actually be running in this call context.
        setattr(self, _CACHE_MARKER, None)
        call = _AttentionCall(self, _reference_parent(self, "qkv", original_parent))
        token = _CALL.set(call)
        try:
            return original_parent(*args, **kwargs)
        finally:
            call.cache = None
            _CALL.reset(token)

    def projection_keys(x: Any) -> tuple[Any, ...]:
        return tuple(
            _projection_cache_key(
                getattr(attn, name),
                _projection_state(getattr(attn, name), x, allow_unadapted=True),
            )
            for name in projections
        )

    def cache_matches(cache: Any, x: Any, name: str) -> bool:
        return (
            cache is not None
            and cache.get("x") is x
            and name in cache
            and cache["execution_key"] == _execution_cache_key(x)
            and cache["input_key"] == _tensor_cache_key(x)
            and cache["projection_keys"] == projection_keys(x)
            and cache["mode_key"] == mode_key(x)
        )

    def q_forward(_proj, x, *args, **kwargs):
        call = scope()
        if call is not None:
            call.advance("q_proj")
        set_cache(None)
        setattr(attn, _CACHE_HIT_MARKER, (0, ()))
        if args or kwargs:
            return originals["q_proj"](x, *args, **kwargs)
        states = [
            _projection_state(getattr(attn, name), x, allow_unadapted=True)
            for name in ("q_proj", "k_proj", "v_proj")
        ]
        if any(state is None for state in states):
            return originals["q_proj"](x)
        input_key = _tensor_cache_key(x)
        state_keys = tuple(
            _projection_cache_key(getattr(attn, name), state)
            for name, state in zip(projections, states)
        )
        if input_key is None or any(key is None for key in state_keys):
            return originals["q_proj"](x)
        q, k, v = states
        # Concatenating A promotes sibling dtypes, but each B and PEFT's input
        # cast still belong to its own pair. Delegate instead of changing math.
        adapter_dtypes = {
            tensor.dtype for state in states
            for tensor in (state.lora_a, state.lora_b) if tensor.numel()
        }
        if len(adapter_dtypes) > 1:
            return originals["q_proj"](x)
        compute_dtypes = {
            state.compute_dtype for state in states if state.compute_dtype is not None
        }
        if len(compute_dtypes) > 1:
            return originals["q_proj"](x)

        input_dtype = x.dtype
        work_x = x
        if compute_dtypes:
            compute_dtype = next(iter(compute_dtypes))
            if work_x.dtype != compute_dtype:
                work_x = work_x.to(compute_dtype)

        qparts = [*q.qparts, *k.qparts, *v.qparts]
        arithmetic_key = mode_key(x, states)
        selected = _qkv_function(reference_order=True) if arithmetic_key[1] else fast
        q_out, k_out, v_out = selected.apply(
            work_x,
            q.weight, q.bias, k.weight, k.bias, v.weight, v.bias,
            q.lora_a, q.lora_b, k.lora_a, k.lora_b, v.lora_a, v.lora_b,
            q.scaling, k.scaling, v.scaling,
            q.qmeta, k.qmeta, v.qmeta,
            *qparts,
        )
        generation = object()
        call_ref = None if call is None else weakref.ref(call)

        def clear_unused_cache(_grads: Any) -> None:
            owner = attn_ref()
            active_call = None if call_ref is None else call_ref()
            cache = (
                active_call.cache if active_call is not None else
                None if owner is None or call_ref is not None else
                getattr(owner, _CACHE_MARKER, None)
            )
            if cache is not None and cache["generation"] is generation:
                if active_call is not None:
                    active_call.cache = None
                else:
                    setattr(owner, _CACHE_MARKER, None)

        # Any sibling can consume the fused saved state. The hook must not own
        # tensors/the module, or let an older graph discard a newer q's cache.
        # Register on the fused node before optional output dtype conversions.
        if q_out.grad_fn is not None:
            q_out.grad_fn.register_prehook(clear_unused_cache)
        if work_x is not x:
            q_out = q_out.to(input_dtype)
            k_out = k_out.to(input_dtype)
            v_out = v_out.to(input_dtype)

        set_cache(
            {
                "x": x,
                "generation": generation,
                "execution_key": _execution_cache_key(x),
                "input_key": input_key,
                "projection_keys": state_keys,
                "mode_key": arithmetic_key,
                "k": k_out,
                "v": v_out,
                "hits": 0,
                "grad_fns": [type(q_out.grad_fn).__name__],
            },
        )
        return q_out

    def k_forward(_proj, x, *args, **kwargs):
        call = scope()
        if call is not None:
            call.advance("k_proj")
        cache = get_cache()
        if not args and not kwargs and cache_matches(cache, x, "k"):
            out = cache.pop("k")
            cache["hits"] += 1
            cache["grad_fns"].append(type(out.grad_fn).__name__)
            return out
        set_cache(None)
        setattr(attn, _CACHE_HIT_MARKER, (0, ()))
        return originals["k_proj"](x, *args, **kwargs)

    def v_forward(_proj, x, *args, **kwargs):
        call = scope()
        if call is not None:
            call.advance("v_proj")
        cache = get_cache()
        if not args and not kwargs and cache_matches(cache, x, "v"):
            out = cache.pop("v")
            cache["hits"] += 1
            cache["grad_fns"].append(type(out.grad_fn).__name__)
            setattr(attn, _CACHE_HIT_MARKER, (cache["hits"], tuple(cache["grad_fns"])))
            set_cache(None)
            return out
        set_cache(None)
        setattr(attn, _CACHE_HIT_MARKER, (0, ()))
        return originals["v_proj"](x, *args, **kwargs)

    for forward in (q_forward, k_forward, v_forward):
        setattr(forward, _FORWARD_OWNER_MARKER, _OWNER)
    for name, forward in (
        ("q_proj", q_forward),
        ("k_proj", k_forward),
        ("v_proj", v_forward),
    ):
        projection = projections[name]
        forwards[name] = forward
        setattr(projection, _GROUP_PATCH_OWNER_MARKER, _OWNER)
        projection.forward = types.MethodType(forward, projection)
    if _reference_parent(attn, "qkv", original_parent):
        setattr(attn, _PARENT_RESTORE_MARKER,
                ("forward" in vars(attn), vars(attn).get("forward"), parent_forward))
        attn.forward = types.MethodType(parent_forward, attn)
    setattr(attn, "_soup_fast_lora_qkv_originals", originals)
    setattr(attn, _INSTALLED_FORWARDS_MARKER, forwards)
    setattr(attn, _RESTORE_MARKER, restore)
    setattr(attn, _PATCH_MARKER, True)


def patch_fast_lora_qkv(model: Any) -> int:
    """Patch structural q_proj/k_proj/v_proj attention modules."""
    count = 0
    for module in model.modules():
        if getattr(module, _PATCH_MARKER, False):
            continue
        if not all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj")):
            continue
        if not any(
            hasattr(getattr(module, name), "lora_A")
            for name in ("q_proj", "k_proj", "v_proj")
        ):
            continue
        projections = [
            getattr(module, name) for name in ("q_proj", "k_proj", "v_proj")
        ]
        # Shared-X fusion is only valid when Q, K and V consume the same-width
        # tensor.  Cross-attention modules such as TrOCR use a narrower memory
        # input for K/V; patching them would make q_proj speculatively apply the
        # K/V weights to the query and fail before delegation can occur.
        widths = {getattr(proj, "in_features", None) for proj in projections}
        if len(widths) != 1 or None in widths:
            continue
        if any(hasattr(proj, "modules_to_save") for proj in projections):
            continue
        if any(
            hasattr(proj, "lora_A") and not _is_supported_lora_projection(proj)
            for proj in projections
        ):
            continue
        if any(
            getattr(proj, _GROUP_PATCH_OWNER_MARKER, None) is not None
            for proj in projections
        ):
            continue
        if any(
            getattr(proj, _SINGLE_PROJECTION_PATCH_MARKER, False)
            for proj in projections
        ):
            continue
        _install_attention_patch(module)
        count += 1
    return count


def unpatch_fast_lora_qkv(model: Any) -> int:
    restored = 0
    for module in model.modules():
        originals = getattr(module, "_soup_fast_lora_qkv_originals", None)
        restore = getattr(module, _RESTORE_MARKER, None)
        if originals is None or restore is None or not getattr(module, _PATCH_MARKER, False):
            continue
        parent_restore = getattr(module, _PARENT_RESTORE_MARKER, None)
        if parent_restore is not None:
            had_forward, previous, installed = parent_restore
            if getattr(module.forward, "__func__", None) is installed:
                if had_forward:
                    module.forward = previous
                else:
                    delattr(module, "forward")
            delattr(module, _PARENT_RESTORE_MARKER)
        for name, (had_instance_forward, instance_forward) in restore.items():
            proj = getattr(module, name)
            installed = getattr(module, _INSTALLED_FORWARDS_MARKER)[name]
            if getattr(proj.forward, "__func__", None) is installed:
                if had_instance_forward:
                    proj.forward = instance_forward
                elif "forward" in proj.__dict__:
                    delattr(proj, "forward")
            if getattr(proj, _GROUP_PATCH_OWNER_MARKER, None) == _OWNER:
                delattr(proj, _GROUP_PATCH_OWNER_MARKER)
        for attr in (
            _CACHE_MARKER,
            _CACHE_HIT_MARKER,
            "_soup_fast_lora_qkv_originals",
            _RESTORE_MARKER,
            _INSTALLED_FORWARDS_MARKER,
            _PATCH_MARKER,
        ):
            if hasattr(module, attr):
                delattr(module, attr)
        restored += 1
    return restored
