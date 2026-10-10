"""Fast-LoRA single-projection autograd (correctness half of tracker #792).

Scope: the single-projection path from #839, which is ``o_proj`` plus any
adapted ``nn.Linear`` / ``bnb.nn.Linear4bit`` the MLP (#837) and QKV (#838)
paths do not cover. The kernel computes the same math as peft's generic
autograd with fewer intermediates; **no speedup is claimed here** -- the
tracker measures before any number is published.

What this module establishes for the sibling paths:

- instance-level forward patching (``types.MethodType``, as the streamed-layer
  dequant patch already does in ``layer_stream_runtime``);
- ``save_for_backward`` for every tensor the backward reads (#331: streamed
  pools rewrite their slots, so a saved reference is only safe when it went
  through the pool-aware save);
- NF4: the packed weight and its quant-state tensors are saved and the weight
  is re-dequantised in the backward; the dequantised matrix is never saved;
- delegation back to the unpatched peft forward whenever ``disable_adapters``,
  ``merged`` or ``adapter_names`` is set, the adapter is a fused variant
  (DoRA/VeRA and friends produce tuple results the hand-written backward does
  not model), dropout is non-zero, or the call shape is not a plain ``(x)``.

The patch covers only the single-projection path. When the QKV and MLP
patchers land they have to run first (or mark their modules) so those shapes
do not fall into this path.

The patched forward reads the base weight at call time on purpose: streamed
layers substitute weights via ``functional_call`` only for the duration of a
call, and a reference captured at patch time under streaming is a meta
placeholder.
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

_PATCH_MARKER = "_soup_fast_lora_single_projection"
_ORIGINAL_FORWARD_MARKER = "_soup_fast_lora_original_forward"
_INSTALLED_FORWARD_MARKER = "_soup_fast_lora_single_installed_forward"
_HAD_INSTANCE_FORWARD_MARKER = "_soup_fast_lora_single_had_instance_forward"
_GROUP_PATCH_OWNER_MARKER = "_soup_fast_lora_group_owner"
_FORWARD_OWNER_MARKER = "_soup_fast_lora_forward_owner"

__all__ = [
    "patch_fast_lora_single_projection",
    "unpatch_fast_lora_single_projection",
]

_FUNCTION: Any = None


def _as_dtype(tensor: Any, dtype: Any) -> Any:
    """Cast only when the dtype differs, so the uniform case stays op-for-op."""
    return tensor if tensor.dtype == dtype else tensor.to(dtype)


def _scaled_lora_add(
    out: Any, hidden: Any, lora_b: Any, scaling: float, *, quantized: bool = False
) -> Any:
    """Preserve dense versus Linear4bit's distinct scaled-update cast seams."""
    import torch

    lora_term = torch.matmul(hidden, lora_b.t()) * float(scaling)
    if quantized:
        # PEFT Linear4bit (autocast disabled) narrows the scaled update before
        # addition. Dense Linear promotes the addition, then narrows its result.
        return out + lora_term.to(out.dtype)
    promoted = torch.promote_types(out.dtype, lora_term.dtype)
    return (out.to(promoted) + lora_term.to(promoted)).to(out.dtype)


def _flatten(tensor: Any) -> Any:
    """Collapse the leading dimensions into one, for weight-shaped gradients.

    ``dB`` and ``dA`` sum over every leading dimension regardless of the input
    rank (``[N, in]`` for the unit tests, ``[B, S, in]`` from transformers).
    ``reshape`` is a view for the contiguous activation layouts that reach a
    projection; a non-contiguous grad pays a copy, which is the correct trade
    against a silent shape assumption.
    """
    return tensor.reshape(-1, tensor.shape[-1])


def _quant_state_parts(quant_state: Any) -> tuple[list[Any], dict[str, Any]]:
    """Split a ``QuantState`` into ``save_for_backward`` tensors and metadata.

    Roles are recorded in save order so the backward can rebuild the state
    without guessing which optional tensor is which (``offset`` and the nested
    ``state2`` are only present for some checkpoints).
    """
    roles = ["absmax", "code"]
    tensors = [quant_state.absmax, quant_state.code]
    meta: dict[str, Any] = {
        "shape": tuple(quant_state.shape),
        "dtype": quant_state.dtype,
        "blocksize": quant_state.blocksize,
        "quant_type": quant_state.quant_type,
    }
    if quant_state.offset is not None:
        roles.append("offset")
        tensors.append(quant_state.offset)
    if quant_state.state2 is not None:
        roles += ["state2_absmax", "state2_code"]
        tensors += [quant_state.state2.absmax, quant_state.state2.code]
        meta["state2_blocksize"] = quant_state.state2.blocksize
        if quant_state.state2.offset is not None:
            roles.append("state2_offset")
            tensors.append(quant_state.state2.offset)
    meta["roles"] = roles
    return tensors, meta


def _rebuild_quant_state(meta: dict[str, Any], tensors: list[Any]) -> Any:
    """Reassemble a ``QuantState`` from saved tensors (``layer_stream_runtime``
    precedent, including the nested ``state2`` dtype)."""
    import torch
    from bitsandbytes.functional import QuantState

    parts = dict(zip(meta["roles"], tensors))
    state2 = None
    if "state2_absmax" in parts:
        state2 = QuantState(
            absmax=parts["state2_absmax"],
            code=parts["state2_code"],
            blocksize=meta["state2_blocksize"],
            dtype=torch.float32,
            offset=parts.get("state2_offset"),
        )
    return QuantState(
        absmax=parts["absmax"],
        shape=torch.Size(meta["shape"]),
        dtype=meta["dtype"],
        blocksize=meta["blocksize"],
        code=parts["code"],
        quant_type=meta["quant_type"],
        offset=parts.get("offset"),
        state2=state2,
    )


class _ProjectionState(NamedTuple):
    weight: Any
    bias: Any
    lora_a: Any
    lora_b: Any
    scaling: float
    qmeta: dict[str, Any] | None
    qparts: list[Any]
    compute_dtype: Any


def _projection_state(
    proj: Any, x: Any, *, allow_unadapted: bool = False
) -> _ProjectionState | None:
    """Return one PEFT/base projection state or None when PEFT must run.

    This is shared by the single, MLP and QKV kernels so the seven delegation
    guards, streamed QuantState repair and bitsandbytes compute-dtype handling
    cannot drift between patchers.
    """
    lora_a_map = getattr(proj, "lora_A", None)
    if lora_a_map is not None and not _is_supported_lora_projection(proj):
        return None
    base = proj.get_base_layer() if hasattr(proj, "get_base_layer") else proj
    if not hasattr(base, "weight"):
        return None

    adapter = None
    if lora_a_map is None:
        if not allow_unadapted:
            return None
    else:
        if getattr(proj, "disable_adapters", False) or getattr(proj, "merged", False):
            return None
        active = getattr(proj, "active_adapters", None) or []
        if len(active) != 1:
            return None
        adapter = active[0]
        if adapter not in proj.lora_A:
            if not allow_unadapted:
                return None
            adapter = None
        else:
            if adapter in getattr(proj, "lora_variant", {}):
                return None
            # Dropout changes the adapter computation and mask lifetime; the
            # hand-written kernels model only the deterministic LoRA branch.
            if float(getattr(proj.lora_dropout[adapter], "p", 0.0)) != 0.0:
                return None
            # Conv1D-style/transposed storage needs the PEFT fan-in/fan-out
            # path rather than the Linear weight orientation used below.
            if getattr(proj, "fan_in_fan_out", False):
                return None
            # ``lora_bias`` adds another trained term that these Functions do
            # not accept or differentiate.
            if getattr(proj.lora_B[adapter], "bias", None) is not None:
                return None

    weight = base.weight
    qstate = getattr(weight, "quant_state", None)
    if qstate is None and getattr(base, "quant_state", None) is not None:
        # Streamed Params4bit views can temporarily carry the state on the
        # module rather than on the view itself. Repair the same way as the
        # single-projection path before any sibling reads it.
        try:
            weight.quant_state = base.quant_state
        except AttributeError:
            pass
        qstate = getattr(weight, "quant_state", None) or base.quant_state

    # An unadapted sibling may still be an 8-bit/custom projection. Without a
    # 4-bit QuantState it is not a dense floating weight and cannot use F.linear.
    if qstate is None and not weight.is_floating_point():
        return None

    compute_dtype = None
    qmeta = None
    qparts: list[Any] = []
    if qstate is not None:
        if not getattr(base, "compute_type_is_set", True) and hasattr(base, "set_compute_type"):
            base.set_compute_type(x)
            base.compute_type_is_set = True
        compute_dtype = getattr(base, "compute_dtype", None)
        qparts, qmeta = _quant_state_parts(qstate)
        qmeta = dict(qmeta)
        qmeta["_count"] = len(qparts)

    if adapter is None:
        lora_a = x.new_empty(0)
        lora_b = x.new_empty(0)
        scaling = 0.0
    else:
        lora_a = proj.lora_A[adapter].weight
        lora_b = proj.lora_B[adapter].weight
        scaling = float(proj.scaling[adapter])

    return _ProjectionState(
        weight,
        getattr(base, "bias", None),
        lora_a,
        lora_b,
        scaling,
        qmeta,
        qparts,
        compute_dtype,
    )


def _supported_lora_projection_types() -> tuple[type, ...]:
    """Return PEFT projection types whose weight contracts the kernels model."""
    from peft.tuners.lora import Linear as LoraLinear

    supported: tuple[type, ...] = (LoraLinear,)
    try:
        from peft.tuners.lora import bnb as lora_bnb

        supported = (LoraLinear, lora_bnb.Linear4bit)
    except Exception:  # noqa: BLE001 - bitsandbytes absence is a normal install
        pass
    return supported


def _is_supported_lora_projection(proj: Any) -> bool:
    """Exclude PEFT 8-bit/custom layers whose storage math differs."""
    return isinstance(proj, _supported_lora_projection_types())


_REFERENCE_DEFINITIONS: Any = None
_REFERENCE_CAST_METHOD: Any = None


def _verified_class_forward(cls: type, method_name: str = "forward") -> Any:
    """Anchor a pristine method to installed source, not a first-use monkeypatch."""
    import ast
    import inspect
    import sys
    import textwrap
    import types

    def signature(code: Any) -> Any:
        return (
            code.co_code, code.co_names, code.co_varnames, code.co_freevars,
            code.co_cellvars, code.co_argcount, code.co_posonlyargcount,
            code.co_kwonlyargcount,
            tuple(signature(c) if isinstance(c, types.CodeType) else c for c in code.co_consts),
        )

    current = getattr(cls, method_name)
    if not isinstance(current, types.FunctionType):
        return None
    # Code/qualname metadata can be copied into a function with a different
    # global namespace. It is not the installed definition's execution context.
    owner = sys.modules.get(cls.__module__)
    if owner is None or current.__globals__ is not vars(owner):
        return None
    if (current.__module__ != cls.__module__
            or current.__qualname__ != f"{cls.__name__}.{method_name}"):
        return None
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
        definition = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        # Runtime uses the last definition; preceding @overload declarations
        # are typing stubs, not the method that PEFT executes.
        method = next(node for node in reversed(definition.body)
                      if isinstance(node, ast.FunctionDef) and node.name == method_name)
        if method.decorator_list:
            return None
        # Compile, never execute, the complete installed module. On Python 3.12,
        # compiling a class alone can emit different call bytecode when a global
        # was imported by the surrounding module. Keep that compiler context.
        source_file = inspect.getsourcefile(owner)
        if source_file is None:
            return None
        compiled = compile(inspect.getsource(owner), source_file, "exec", dont_inherit=True)
        class_codes = [c for c in compiled.co_consts
                       if isinstance(c, types.CodeType) and c.co_name == cls.__name__]
        if len(class_codes) != 1:
            return None
        class_code = class_codes[0]
        expected = next(c for c in reversed(class_code.co_consts)
                        if isinstance(c, types.CodeType) and c.co_name == method_name)
        return current if signature(current.__code__) == signature(expected) else None
    except (OSError, TypeError, SyntaxError, StopIteration):
        return None


def _reference_definitions() -> Any:
    """Lazy, exact identities for the installed vanilla Llama/PEFT graph."""
    global _REFERENCE_DEFINITIONS, _REFERENCE_CAST_METHOD
    if _REFERENCE_DEFINITIONS is None:
        from peft.tuners.tuners_utils import BaseTunerLayer
        from torch import nn
        from transformers.activations import SiLUActivation
        from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP

        types = (LlamaAttention, LlamaMLP, nn.Linear, nn.Identity, nn.Dropout,
                 SiLUActivation, *_supported_lora_projection_types())
        try:
            from bitsandbytes.nn import Linear4bit

            types += (Linear4bit,)
        except ImportError:
            pass
        definitions = (
            {"qkv": LlamaAttention, "mlp": LlamaMLP},
            {cls: (method, None if method is None else method.__code__)
             for cls in types for method in (_verified_class_forward(cls),)},
        )
        cast = _verified_class_forward(BaseTunerLayer, "_cast_input_dtype")
        _REFERENCE_CAST_METHOD = (cast, None if cast is None else cast.__code__)
        # Publish the readiness sentinel last: peers must never observe class
        # definitions before the corresponding cast-method metadata exists.
        _REFERENCE_DEFINITIONS = definitions
    return _REFERENCE_DEFINITIONS


def _reference_method(module: Any, forward: Any) -> bool:
    _parents, definitions = _reference_definitions()
    trusted, code = definitions.get(type(module), (None, None))
    return (
        trusted is not None and type(module).forward is trusted and trusted.__code__ is code
        and getattr(forward, "__func__", None) is trusted
        and getattr(forward, "__self__", None) is module
    )


def _reference_hooks(module: Any, *, output_observers: bool = False) -> bool:
    """Only explicitly supported read-only output observers may remain.

    QKV child/parent output hooks execute normally. Mutating hooks are outside
    this scope; there is no hook-purity detector. MLP-bypassed inner hooks are
    unsupported even if read-only.
    """
    from torch.nn.modules import module as module_hooks

    return not (
        module_hooks._global_forward_pre_hooks or module_hooks._global_forward_hooks
        or module_hooks._global_backward_pre_hooks or module_hooks._global_backward_hooks
        or
        getattr(module, "_forward_pre_hooks", {}) or getattr(module, "_backward_hooks", {})
        or getattr(module, "_backward_pre_hooks", {})
        or (not output_observers and getattr(module, "_forward_hooks", {}))
    )


def _reference_parent(module: Any, kind: str, original_forward: Any) -> bool:
    parents, _definitions = _reference_definitions()
    return (
        type(module) is parents[kind] and _reference_method(module, original_forward)
        and _reference_hooks(module, output_observers=True)
    )


def _reference_projections(
    projections: Any, originals: Any, states: Any, x: Any, *, output_observers: bool = False
) -> bool:
    """Non-vacuous low-X/FP32-master scope; unsupported state remains legacy."""
    import torch
    from torch import nn

    if x.dtype not in (torch.float16, torch.bfloat16) or torch.is_autocast_enabled(x.device.type):
        return False
    present = [state for state in states if state is not None and state.lora_a.numel()]
    if not present or any(state is None for state in states):
        return False
    for proj, original, state in zip(projections, originals, states):
        if not _reference_method(proj, original):
            return False
        if not _reference_hooks(proj, output_observers=output_observers):
            return False
        base = proj.get_base_layer() if hasattr(proj, "get_base_layer") else proj
        base_forward = original if base is proj else base.forward
        if not _reference_method(base, base_forward):
            return False
        if base is not proj and not _reference_hooks(base):
            return False
        if state.weight.requires_grad or (state.bias is not None and state.bias.requires_grad):
            return False
        if state.compute_dtype not in (None, x.dtype):
            return False
        if state.qmeta is None and state.weight.dtype != x.dtype:
            return False
        if not state.lora_a.numel():
            continue
        if state.lora_a.dtype != torch.float32 or state.lora_b.dtype != torch.float32:
            return False
        if not getattr(proj, "cast_input_dtype_enabled", True):
            return False
        cast, cast_code = _REFERENCE_CAST_METHOD
        current_cast = proj._cast_input_dtype
        if (cast is None or getattr(current_cast, "__func__", None) is not cast
                or cast.__code__ is not cast_code
                or getattr(current_cast, "__self__", None) is not proj):
            return False
        adapter = proj.active_adapters[0]
        layers = (proj.lora_A[adapter], proj.lora_B[adapter], proj.lora_dropout[adapter])
        if type(layers[0]) is not nn.Linear or type(layers[1]) is not nn.Linear:
            return False
        if type(layers[2]) not in (nn.Identity, nn.Dropout):
            return False
        if any(not _reference_method(layer, layer.forward) or not _reference_hooks(layer)
               for layer in layers):
            return False
        if layers[0].bias is not None or layers[1].bias is not None:
            return False
    return True


def _projection_forward_matches(proj: Any, original: Any) -> bool:
    """Permit only our installed single wrapper around the same vanilla method.

    The normal group-then-single install patches MLP children too. They are not
    executed by the MLP Function; an arbitrary replacement is still ineligible.
    """
    return proj.forward == original or (
        getattr(proj, _PATCH_MARKER, False)
        and getattr(proj.forward, "__func__", None) is
        getattr(proj, _INSTALLED_FORWARD_MARKER, None)
        and getattr(proj, _ORIGINAL_FORWARD_MARKER, None) == original
    )


def _dense_weight(
    weight: Any,
    qmeta: dict[str, Any] | None,
    qparts: list[Any],
    dtype: Any,
) -> Any:
    """Return a dense view for one base projection without retaining it."""
    if qmeta is None:
        return _as_dtype(weight, dtype)
    from bitsandbytes.functional import dequantize_4bit

    state = _rebuild_quant_state(qmeta, qparts)
    return dequantize_4bit(weight, state).to(dtype)


def _single_projection_function() -> Any:
    """Build (once per process) the ``autograd.Function`` class.

    The class has to be built inside a function because its base class is
    ``torch.autograd.Function`` and torch must stay out of the light CLI import
    path (tests/test_cli_startup_is_light.py).
    """
    global _FUNCTION
    if _FUNCTION is not None:
        return _FUNCTION

    import torch

    class _FastLoraSingleProjection(torch.autograd.Function):
        """``Y = X @ W^T + b + s * (X @ A^T) @ B^T``, hand-written backward.

        ``W`` and ``b`` are frozen during LoRA training, so the backward returns
        ``None`` for both, and the input gradient still carries the base term
        ``dY @ W``. ``H`` (``[N, r]``) is saved rather than recomputed: the
        backward needs it twice (``dA`` and ``dB``) and it is rank-sized; the
        saved-bytes test reports the actual delta against peft's graph.

        NF4: ``weight`` is the packed tensor and ``quant_state`` carries the
        per-block tensors. The forward dequantises for the base matmul only;
        the backward rebuilds the ``QuantState`` from the saved tensors and
        dequantises again for ``dX``.
        """

        @staticmethod
        def forward(ctx, x, weight, bias, lora_a, lora_b, scaling, quant_state):
            import torch.nn.functional as functional

            h = functional.linear(_as_dtype(x, lora_a.dtype), lora_a)  # [N, r]
            if quant_state is None:
                out = functional.linear(x, weight, bias)
                ctx.save_for_backward(x, weight, lora_a, lora_b, h)
                ctx.qmeta = None
            else:
                from bitsandbytes.functional import dequantize_4bit

                # Transient on purpose: under streaming this dense matrix would
                # alias a pool slot, so it must not outlive the call.
                dense = dequantize_4bit(weight, quant_state).to(x.dtype)
                out = functional.linear(
                    x, dense, _as_dtype(bias, x.dtype) if bias is not None else None
                )
                del dense
                qs_tensors, qs_meta = _quant_state_parts(quant_state)
                ctx.save_for_backward(x, weight, lora_a, lora_b, h, *qs_tensors)
                ctx.qmeta = qs_meta

            ctx.scaling = float(scaling)
            # PEFT multiplies the completed update before addition; alpha=s
            # would reassociate the multiply and change non-power-of-two scales.
            out = _scaled_lora_add(
                out, h, lora_b, ctx.scaling, quantized=quant_state is not None
            )
            return out

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, grad_out):
            scaling = ctx.scaling
            # Non-reentrant checkpoint hooks allow each saved tensor to be
            # unpacked once. Reuse this tuple for the NF4 state as well.
            saved = ctx.saved_tensors
            x, weight, lora_a, lora_b, h = saved[:5]

            # dB sums over every leading dimension, so flatten it; dB and dA
            # are weight-shaped regardless of the input's rank.
            grad_b = (_as_dtype(_flatten(grad_out), h.dtype) * scaling).t() @ _flatten(h)
            # dH keeps the input's leading dimensions; dA = dH^T @ X.
            grad_h = torch.matmul(_as_dtype(grad_out, lora_b.dtype) * scaling, lora_b)
            grad_a = _flatten(grad_h).t() @ _as_dtype(_flatten(x), grad_h.dtype)

            grad_x = None
            if ctx.needs_input_grad[0]:
                if ctx.qmeta is None:
                    dense = weight
                else:
                    from bitsandbytes.functional import dequantize_4bit

                    quant_state = _rebuild_quant_state(ctx.qmeta, list(saved[5:]))
                    dense = dequantize_4bit(weight, quant_state).to(x.dtype)

                # dX = dY @ W, plus the LoRA term accumulated into the same
                # result: one add instead of separate dH @ A and sum
                # allocations. Skipped entirely when X needs no grad, which
                # recovers the 2x the unskipped dX GEMM costs on a layer-0
                # projection.
                grad_x = torch.matmul(grad_out, _as_dtype(dense, grad_out.dtype))
                grad_x = torch.add(
                    grad_x,
                    _as_dtype(
                        torch.matmul(grad_h, _as_dtype(lora_a, grad_h.dtype)), grad_x.dtype
                    ),
                )
            return grad_x, None, None, grad_a, grad_b, None, None

    _FUNCTION = _FastLoraSingleProjection
    return _FUNCTION


def _make_patched_forward(original_forward: Any) -> Any:
    """Wrap a bound peft forward; delegate whenever peft's own path must run."""
    fast = _single_projection_function()

    def _fast_lora_single_forward(self, x, *args, **kwargs):
        if args or kwargs:
            return original_forward(x, *args, **kwargs)
        state = _projection_state(self, x)
        if state is None:
            return original_forward(x)

        input_dtype = x.dtype
        work_x = x
        if state.compute_dtype is not None and work_x.dtype != state.compute_dtype:
            work_x = work_x.to(state.compute_dtype)

        out = fast.apply(
            work_x,
            state.weight,
            state.bias,
            state.lora_a,
            state.lora_b,
            state.scaling,
            None if state.qmeta is None else _rebuild_quant_state(state.qmeta, state.qparts),
        )
        return out if work_x is x else out.to(input_dtype)

    return _fast_lora_single_forward


def patch_fast_lora_single_projection(model: Any) -> int:
    """Replace the forward of every single-projection LoRA linear in ``model``.

    Returns the number of patched modules so a caller can assert it patched
    something (``install_dequant_forward`` precedent). Already-patched modules
    are skipped, which makes the call idempotent.
    """
    import types

    types_to_match = _supported_lora_projection_types()

    targets = []
    for child in model.modules():
        if getattr(child, _PATCH_MARKER, False):
            continue
        if getattr(child, _GROUP_PATCH_OWNER_MARKER, None) is not None:
            continue
        if isinstance(child, types_to_match):
            targets.append(child)

    for child in targets:
        setattr(child, _ORIGINAL_FORWARD_MARKER, child.forward)
        setattr(child, _HAD_INSTANCE_FORWARD_MARKER, "forward" in vars(child))
        setattr(child, _PATCH_MARKER, True)
        installed = _make_patched_forward(child.forward)
        setattr(child, _INSTALLED_FORWARD_MARKER, installed)
        child.forward = types.MethodType(installed, child)
    return len(targets)


def unpatch_fast_lora_single_projection(model: Any) -> int:
    """Restore the original peft forwards. Returns the number restored."""
    restored = 0
    for child in model.modules():
        original = getattr(child, _ORIGINAL_FORWARD_MARKER, None)
        if original is None or not getattr(child, _PATCH_MARKER, False):
            continue
        if getattr(child.forward, "__func__", None) is getattr(child, _INSTALLED_FORWARD_MARKER):
            if getattr(child, _HAD_INSTANCE_FORWARD_MARKER):
                child.forward = original
            else:
                delattr(child, "forward")
        delattr(child, _ORIGINAL_FORWARD_MARKER)
        delattr(child, _PATCH_MARKER)
        delattr(child, _INSTALLED_FORWARD_MARKER)
        delattr(child, _HAD_INSTANCE_FORWARD_MARKER)
        restored += 1
    return restored
