"""Fast-LoRA fused SwiGLU MLP autograd path for issue #837.

Targets gate_proj/up_proj/down_proj blocks with SiLU/swish. This is the
correctness slice from tracker #792; it makes no throughput claim.

Pristine standard installed HF LlamaMLP with supported vanilla PEFT projections
can select separate reference arithmetic for low-precision X/fp32 adapters.
``grad_fn.reference_order`` distinguishes it from genuine approximate legacy
fusion. Only the parent's read-only output observers are supported in this scope;
inner hooks, replaced forwards/activations and autocast opt out. The separate
GEMMs are a correctness trade-off, not universal bit-exactness or a speedup claim.
"""

from __future__ import annotations

import logging
import types
from typing import Any

from soup_cli.utils.fast_lora import (
    _FORWARD_OWNER_MARKER,
    _as_dtype,
    _dense_weight,
    _flatten,
    _is_supported_lora_projection,
    _projection_forward_matches,
    _projection_state,
    _reference_hooks,
    _reference_method,
    _reference_parent,
    _reference_projections,
    _scaled_lora_add,
)

logger = logging.getLogger(__name__)
_PATCH_MARKER = "_soup_fast_lora_mlp"
_ORIGINAL_FORWARD_MARKER = "_soup_fast_lora_mlp_original_forward"
_HAD_INSTANCE_FORWARD_MARKER = "_soup_fast_lora_mlp_had_instance_forward"
_INSTALLED_FORWARD_MARKER = "_soup_fast_lora_mlp_installed_forward"
_FUNCTION: Any = None
_REFERENCE_FUNCTION: Any = None

__all__ = ["patch_fast_lora_mlp", "unpatch_fast_lora_mlp"]


def _mlp_function(*, reference_order: bool = False) -> Any:
    global _FUNCTION, _REFERENCE_FUNCTION
    cached = _REFERENCE_FUNCTION if reference_order else _FUNCTION
    if cached is not None:
        return cached

    import torch
    import torch.nn.functional as functional

    class _FastLoraSwiGLU(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx, x, wg, bg, wu, bu, wd, bd,
            ag, bgl, au, bul, ad, bdl,
            sg, su, sd, qg_meta, qu_meta, qd_meta, *qparts,
        ):
            ctx.reference_order = reference_order
            metas = (qg_meta, qu_meta, qd_meta)
            counts = [0 if meta is None else int(meta["_count"]) for meta in metas]
            starts = (0, counts[0], counts[0] + counts[1])
            qlists = [
                list(qparts[starts[i] : starts[i] + counts[i]]) for i in range(3)
            ]

            dense_g = _dense_weight(wg, qg_meta, qlists[0], x.dtype)
            dense_u = _dense_weight(wu, qu_meta, qlists[1], x.dtype)
            g = functional.linear(x, dense_g, _as_dtype(bg, x.dtype) if bg is not None else None)
            u = functional.linear(x, dense_u, _as_dtype(bu, x.dtype) if bu is not None else None)
            del dense_g, dense_u

            has_g, has_u, has_d = ag.numel() != 0, au.numel() != 0, ad.numel() != 0
            gu_parts = []
            if has_g:
                gu_parts.append(ag)
            if has_u:
                gu_parts.append(au)
            if gu_parts:
                # #837's advertised fusion: gate/up LoRA A projections share X,
                # so concatenate A and perform one GEMM, then split by rank.
                if reference_order:
                    h_gu = torch.cat([
                        functional.linear(_as_dtype(x, a.dtype), a) for a in gu_parts
                    ], dim=-1)
                else:
                    a_gu = torch.cat(gu_parts, dim=0)
                    h_gu = functional.linear(_as_dtype(x, a_gu.dtype), a_gu)
                cursor = 0
                if has_g:
                    rank = ag.shape[0]
                    hg = h_gu[..., cursor : cursor + rank]
                    g = _scaled_lora_add(g, hg, bgl, sg, quantized=qg_meta is not None)
                    cursor += rank
                if has_u:
                    rank = au.shape[0]
                    hu = h_gu[..., cursor : cursor + rank]
                    u = _scaled_lora_add(u, hu, bul, su, quantized=qu_meta is not None)
            else:
                h_gu = x.new_empty((*x.shape[:-1], 0))

            m = functional.silu(g) * u
            dense_d = _dense_weight(wd, qd_meta, qlists[2], m.dtype)
            y = functional.linear(m, dense_d, _as_dtype(bd, m.dtype) if bd is not None else None)
            del dense_d
            if has_d:
                hd = functional.linear(_as_dtype(m, ad.dtype), ad)
                y = _scaled_lora_add(y, hd, bdl, sd, quantized=qd_meta is not None)

            ctx.has_adapters = (has_g, has_u, has_d)
            ctx.gu_ranks = (ag.shape[0] if has_g else 0, au.shape[0] if has_u else 0)
            ctx.scalings = (float(sg), float(su), float(sd))
            ctx.qmetas = metas
            ctx.qcounts = counts
            ctx.qparts_len = len(qparts)
            ctx.save_for_backward(
                x, wg, wu, wd, g, u, ag, bgl, au, bul, ad, bdl, h_gu, *qparts
            )
            return y

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, grad_y):
            saved = ctx.saved_tensors
            x, wg, wu, wd, g, u, ag, bgl, au, bul, ad, bdl, h_gu = saved[:13]
            qparts = list(saved[13:])
            qg_meta, qu_meta, qd_meta = ctx.qmetas
            cg, cu, cd = ctx.qcounts
            qg = qparts[:cg]
            qu = qparts[cg : cg + cu]
            qd = qparts[cg + cu : cg + cu + cd]
            has_g, has_u, has_d = ctx.has_adapters
            sg, su, sd = ctx.scalings

            silu_g = torch.nn.functional.silu(g)
            m = silu_g * u
            grad_ad = grad_bdl = None

            dense_d = _dense_weight(wd, qd_meta, qd, grad_y.dtype)
            grad_m = torch.matmul(grad_y, dense_d)
            del dense_d
            if has_d:
                hd = torch.matmul(_as_dtype(m, ad.dtype), ad.t())
                grad_bdl = _flatten(_as_dtype(grad_y, hd.dtype) * sd).t() @ _flatten(hd)
                grad_hd = torch.matmul(_as_dtype(grad_y, bdl.dtype) * sd, bdl)
                grad_ad = _flatten(grad_hd).t() @ _as_dtype(_flatten(m), grad_hd.dtype)
                grad_m = torch.add(
                    grad_m,
                    _as_dtype(
                        torch.matmul(grad_hd, _as_dtype(ad, grad_hd.dtype)), grad_m.dtype
                    ),
                )

            grad_u = grad_m * silu_g
            # MulBackward rounds its result in the activation dtype before
            # SiLUBackward applies fp32 opmath. Keep that separate boundary;
            # evaluating the derivative itself in bf16 magnifies cancellation
            # near its zero and can dominate the NF4 error budget.
            grad_silu = grad_m * u
            grad_g = torch.ops.aten.silu_backward.default(grad_silu, g)

            if ctx.reference_order:
                grad_x = None
                grad_ag = grad_bgl = grad_au = grad_bul = None
                cursor = 0
                adapter_dx = {}
                for key, has, a, b, grad, scale, rank in (
                    ("gate", has_g, ag, bgl, grad_g, sg, ctx.gu_ranks[0]),
                    ("up", has_u, au, bul, grad_u, su, ctx.gu_ranks[1]),
                ):
                    if not has:
                        continue
                    hi = h_gu[..., cursor : cursor + rank].contiguous()
                    cursor += rank
                    scaled = _as_dtype(grad, b.dtype) * scale
                    gb = _flatten(scaled).t() @ _flatten(hi)
                    dh = torch.matmul(scaled, b)
                    ga = _flatten(dh).t() @ _as_dtype(_flatten(x), dh.dtype)
                    if key == "gate":
                        grad_ag, grad_bgl = ga, gb
                    else:
                        grad_au, grad_bul = ga, gb
                    if ctx.needs_input_grad[0]:
                        adapter_dx[key] = _as_dtype(torch.matmul(dh, a), x.dtype)
                if ctx.needs_input_grad[0]:
                    # Verified standard gate-before-up forward; reverse edges
                    # remain individually rounded, rather than a fused dX GEMM.
                    for key, grad, weight, meta, parts in (
                        ("up", grad_u, wu, qu_meta, qu),
                        ("gate", grad_g, wg, qg_meta, qg),
                    ):
                        if key in adapter_dx:
                            term = adapter_dx[key]
                            grad_x = term if grad_x is None else grad_x + term
                        dense = _dense_weight(weight, meta, parts, grad.dtype)
                        term = torch.matmul(grad, dense)
                        grad_x = term if grad_x is None else grad_x + term
                        del dense
                result = [
                    grad_x, None, None, None, None, None, None,
                    grad_ag, grad_bgl, grad_au, grad_bul, grad_ad, grad_bdl,
                    None, None, None, None, None, None,
                ]
                result.extend([None] * ctx.qparts_len)
                return tuple(result)

            grad_x = None
            if ctx.needs_input_grad[0]:
                dense_g = _dense_weight(wg, qg_meta, qg, grad_g.dtype)
                grad_x = torch.matmul(grad_g, dense_g)
                del dense_g
                dense_u = _dense_weight(wu, qu_meta, qu, grad_u.dtype)
                grad_x = torch.add(grad_x, torch.matmul(grad_u, dense_u))
                del dense_u

            grad_ag = grad_bgl = grad_au = grad_bul = None
            dh_parts = []
            a_parts = []
            cursor = 0
            if has_g:
                rank = ctx.gu_ranks[0]
                hg = h_gu[..., cursor : cursor + rank]
                grad_bgl = _flatten(_as_dtype(grad_g, hg.dtype) * sg).t() @ _flatten(hg)
                grad_hg = torch.matmul(_as_dtype(grad_g, bgl.dtype) * sg, bgl)
                dh_parts.append(grad_hg)
                a_parts.append(ag)
                cursor += rank
            if has_u:
                rank = ctx.gu_ranks[1]
                hu = h_gu[..., cursor : cursor + rank]
                grad_bul = _flatten(_as_dtype(grad_u, hu.dtype) * su).t() @ _flatten(hu)
                grad_hu = torch.matmul(_as_dtype(grad_u, bul.dtype) * su, bul)
                dh_parts.append(grad_hu)
                a_parts.append(au)

            if dh_parts:
                # One dA GEMM for the gate/up pair, mirroring the one forward
                # X @ A_gu^T GEMM above; split the concatenated result by rank.
                dh_gu = dh_parts[0] if len(dh_parts) == 1 else torch.cat(dh_parts, dim=-1)
                a_gu = a_parts[0] if len(a_parts) == 1 else torch.cat(a_parts, dim=0)
                grad_a_gu = _flatten(dh_gu).t() @ _as_dtype(_flatten(x), dh_gu.dtype)
                if grad_x is not None:
                    grad_x = torch.add(
                        grad_x,
                        _as_dtype(
                            torch.matmul(dh_gu, _as_dtype(a_gu, dh_gu.dtype)), grad_x.dtype
                        ),
                    )
                cursor = 0
                if has_g:
                    rank = ctx.gu_ranks[0]
                    grad_ag = grad_a_gu[cursor : cursor + rank]
                    cursor += rank
                if has_u:
                    rank = ctx.gu_ranks[1]
                    grad_au = grad_a_gu[cursor : cursor + rank]

            result = [
                grad_x,
                None, None, None, None, None, None,
                grad_ag, grad_bgl, grad_au, grad_bul, grad_ad, grad_bdl,
                None, None, None, None, None, None,
            ]
            result.extend([None] * ctx.qparts_len)
            return tuple(result)

    if reference_order:
        _REFERENCE_FUNCTION = _FastLoraSwiGLU
    else:
        _FUNCTION = _FastLoraSwiGLU
    return _FastLoraSwiGLU


def _make_mlp_forward(original_forward: Any) -> Any:
    fast = _mlp_function()
    module = getattr(original_forward, "__self__", None)
    names = ("gate_proj", "up_proj", "down_proj")
    originals = [] if module is None else [getattr(module, name).forward for name in names]

    def _forward(self, x, *args, **kwargs):
        if args or kwargs:
            return original_forward(x, *args, **kwargs)
        states = [
            _projection_state(getattr(self, name), x, allow_unadapted=True)
            for name in ("gate_proj", "up_proj", "down_proj")
        ]
        if any(state is None for state in states):
            return original_forward(x)
        # Concatenating A matrices promotes their dtype. Separate PEFT
        # projections cast each input independently; preserve that behavior
        # rather than feeding a promoted hidden tensor to a lower-dtype B.
        adapter_dtypes = {
            tensor.dtype
            for state in states
            for tensor in (state.lora_a, state.lora_b)
            if tensor.numel()
        }
        if len(adapter_dtypes) > 1:
            return original_forward(x)
        gate, up, down = states
        compute_dtypes = {
            state.compute_dtype for state in states if state.compute_dtype is not None
        }
        if len(compute_dtypes) > 1:
            return original_forward(x)
        input_dtype = x.dtype
        work_x = x
        if compute_dtypes:
            compute_dtype = next(iter(compute_dtypes))
            if work_x.dtype != compute_dtype:
                work_x = work_x.to(compute_dtype)

        qparts = [*gate.qparts, *up.qparts, *down.qparts]
        projections = [getattr(self, name) for name in names]
        reference_order = (
            _reference_parent(self, "mlp", original_forward)
            and _reference_method(self.act_fn, self.act_fn.forward)
            and _reference_hooks(self.act_fn)
            and all(_projection_forward_matches(proj, original)
                    for proj, original in zip(projections, originals))
            and _reference_projections(projections, originals, states, x)
        )
        selected = _mlp_function(reference_order=True) if reference_order else fast
        out = selected.apply(
            work_x,
            gate.weight, gate.bias, up.weight, up.bias, down.weight, down.bias,
            gate.lora_a, gate.lora_b, up.lora_a, up.lora_b, down.lora_a, down.lora_b,
            gate.scaling, up.scaling, down.scaling,
            gate.qmeta, up.qmeta, down.qmeta,
            *qparts,
        )
        return out if work_x is x else out.to(input_dtype)

    setattr(_forward, _FORWARD_OWNER_MARKER, "mlp")
    return _forward


def _module_uses_silu(module: Any) -> bool:
    act_fn = getattr(module, "act_fn", None)
    if act_fn is None:
        return False
    name = (getattr(act_fn, "__name__", "") or type(act_fn).__name__).lower()
    return "silu" in name or "swish" in name


def _looks_like_moe_expert(path: str, module: Any) -> bool:
    parts = path.lower().split(".")
    cls = type(module).__name__.lower()
    return (
        any(part == "experts" or part.startswith("expert") for part in parts)
        or "moe" in cls
        or "expert" in cls
        or "blocksparse" in cls
    )


def patch_fast_lora_mlp(model: Any) -> int:
    """Patch supported dense SiLU MLP blocks and return the number patched."""
    hidden_act = str(getattr(getattr(model, "config", None), "hidden_act", "")).lower()
    if hidden_act not in {"silu", "swish"}:
        logger.info(
            "Fast-LoRA MLP skipped: hidden_act=%r is not SiLU/swish", hidden_act or None
        )
        return 0

    patched = 0
    for path, module in model.named_modules():
        if getattr(module, _PATCH_MARKER, False):
            continue
        if _looks_like_moe_expert(path, module):
            continue
        if not all(
            hasattr(module, name) for name in ("gate_proj", "up_proj", "down_proj")
        ):
            continue
        if not _module_uses_silu(module):
            continue
        if not any(
            hasattr(getattr(module, name), "lora_A")
            for name in ("gate_proj", "up_proj", "down_proj")
        ):
            continue
        projections = [
            getattr(module, name) for name in ("gate_proj", "up_proj", "down_proj")
        ]
        if any(hasattr(proj, "modules_to_save") for proj in projections):
            continue
        if any(
            hasattr(proj, "lora_A") and not _is_supported_lora_projection(proj)
            for proj in projections
        ):
            continue
        setattr(module, _ORIGINAL_FORWARD_MARKER, module.forward)
        setattr(module, _HAD_INSTANCE_FORWARD_MARKER, "forward" in vars(module))
        setattr(module, _PATCH_MARKER, True)
        installed = _make_mlp_forward(module.forward)
        setattr(module, _INSTALLED_FORWARD_MARKER, installed)
        module.forward = types.MethodType(installed, module)
        patched += 1
    return patched


def unpatch_fast_lora_mlp(model: Any) -> int:
    """Restore original MLP forwards. Returns the number restored."""
    restored = 0
    for module in model.modules():
        original = getattr(module, _ORIGINAL_FORWARD_MARKER, None)
        if original is None or not getattr(module, _PATCH_MARKER, False):
            continue
        if getattr(module.forward, "__func__", None) is getattr(module, _INSTALLED_FORWARD_MARKER):
            if getattr(module, _HAD_INSTANCE_FORWARD_MARKER, False):
                module.forward = original
            else:
                del module.forward
        delattr(module, _ORIGINAL_FORWARD_MARKER)
        delattr(module, _HAD_INSTANCE_FORWARD_MARKER)
        delattr(module, _PATCH_MARKER)
        delattr(module, _INSTALLED_FORWARD_MARKER)
        restored += 1
    return restored
