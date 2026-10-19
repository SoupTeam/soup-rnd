"""Instance-local Liger operations inside Soup's checkpointed streaming regions.

Never apply the upstream global model/CE patches here: they bypass streamed heads
or change other models in the process. No parameters or module paths are replaced.
CPU substitutes in tests establish integration only, not Liger kernel correctness.
"""

from __future__ import annotations

from types import MethodType
from typing import Any


def validate_streaming_liger_environment(device: str) -> None:
    """Refuse before weight resolution, sharding or runtime allocation."""
    if not str(device).startswith("cuda"):
        raise ValueError("streamed training.use_liger requires CUDA; no CPU/MPS fallback")
    from importlib.metadata import PackageNotFoundError, version

    from packaging.version import Version

    try:
        installed = version("liger-kernel")
    except PackageNotFoundError as exc:
        raise ValueError(
            'streamed Liger requires liger-kernel==0.8.1; '
            'pip install "soup-cli[liger]"'
        ) from exc
    if Version(installed) != Version("0.8.1"):
        raise ValueError(
            "streamed Liger requires liger-kernel==0.8.1; "
            f"found {installed}; install liger-kernel==0.8.1"
        )
    _assert_no_global_liger()
    _load_liger_ops()


def _assert_no_global_liger() -> None:
    """A prior resident patch must not silently enable unselected operations."""
    import torch.nn.functional as functional
    from transformers.models.llama import modeling_llama

    candidates = (
        modeling_llama.LlamaRMSNorm,
        modeling_llama.LlamaMLP,
        modeling_llama.apply_rotary_pos_emb,
        modeling_llama.LlamaForCausalLM.forward,
        functional.cross_entropy,
    )
    if any(getattr(value, "__module__", "").startswith("liger_kernel") for value in candidates):
        raise ValueError("streamed Liger cannot share a process with a prior global Liger patch")


def _load_liger_ops() -> tuple[Any, Any]:
    try:
        import math

        import torch
        from liger_kernel.ops.rms_norm import LigerRMSNormFunction, _rms_norm_backward_kernel
        from liger_kernel.ops.swiglu import LigerSiLUMulFunction
        from liger_kernel.ops.utils import torch_to_triton_dtype
    except ImportError as exc:
        raise ValueError("streamed Liger requires working Liger/Triton CUDA imports") from exc
    class LigerRMSNormExactFunction(LigerRMSNormFunction):
        @staticmethod
        def backward(ctx: Any, gradient: Any) -> tuple[Any, ...]:
            # The FP32 core is unweighted; Torch owns the low-precision affine
            # boundary. Preserve HF's separate multiply/add rounding in the
            # actual upstream Triton backward, rather than allowing CUDA FMA.
            if ctx.elementwise_affine or ctx.in_place:
                raise ValueError("streamed RMSNorm requires an unweighted out-of-place core")
            hidden, rstd = ctx.saved_tensors
            shape = gradient.shape
            columns = shape[-1]
            incoming = gradient.contiguous().view(-1, columns)
            outgoing = torch.empty_like(incoming)
            rows = incoming.shape[0]
            processors = torch.cuda.get_device_properties(hidden.device).multi_processor_count
            _rms_norm_backward_kernel[(processors,)](
                incoming, incoming.stride(0), outgoing, outgoing.stride(0),
                hidden, hidden.stride(0), torch_to_triton_dtype[hidden.dtype],
                None, 0, rstd, rstd.stride(0), None, 0, rows, columns,
                ctx.offset, math.ceil(rows / processors), ctx.casting_mode,
                elementwise_affine=False, BLOCK_SIZE=ctx.BLOCK_SIZE,
                num_warps=ctx.num_warps, enable_fp_fusion=False,
            )
            return outgoing.view(shape), None, None, None, None, None, None

    return LigerRMSNormExactFunction, LigerSiLUMulFunction


def _rms_norm(hidden: Any, weight: Any, eps: float, operation: Any) -> Any:
    """Keep HF's low-precision affine boundary outside the FP32 normalization."""
    normalized = operation.apply(hidden.float(), None, eps, 0.0, "llama", False)
    return weight * normalized.to(hidden.dtype)


def _silu_mul(gate: Any, up: Any, operation: Any) -> Any:
    """Preserve the rounded SiLU value and its low-precision incoming gradient."""
    import torch

    activated = operation.apply(gate.float(), torch.ones_like(gate, dtype=torch.float32))
    return activated.to(gate.dtype) * up


def validate_streaming_liger_model_config(
    config: Any, components: list[str] | None = None
) -> tuple[str, ...]:
    """Validate model metadata before tokenizer/weight downloads or sharding."""
    selected = tuple(sorted(components if components is not None else ["rms_norm"]))
    if (
        not selected
        or len(set(selected)) != len(selected)
        or set(selected) - {"rms_norm", "swiglu"}
    ):
        raise ValueError("streamed Liger components must be unique rms_norm and/or swiglu")
    if config.model_type != "llama":
        raise ValueError("streamed Liger initially supports Llama models only")
    if getattr(config, "pretraining_tp", 1) != 1:
        raise ValueError("streamed Liger requires pretraining_tp=1")
    if "swiglu" in selected and config.hidden_act not in {"silu", "swish"}:
        raise ValueError("streamed Liger requires a SiLU Llama MLP")
    return selected


def apply_streaming_liger(model: Any, components: list[str] | None = None) -> dict[str, int]:
    """Patch supported forwards after wrapping, before the measured VRAM probe.

    RMSNorm reads ``self.weight`` during functional_call, never a captured meta
    parameter or a recycled slot view. The affine multiply saves it for autograd, so
    the decoder's non-reentrant checkpoint restores it before backward consumes
    it. Out-of-place RMSNorm backward preserves shared residual gradients.
    SwiGLU retains the existing PEFT/bitsandbytes projection modules unchanged.
    """
    from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm

    from soup_cli.utils.layer_stream_runtime import StreamedDecoderLayer

    _assert_no_global_liger()
    base = model.get_base_model()
    selected = validate_streaming_liger_model_config(base.config, components)
    previous = getattr(model, "_soup_streaming_liger", None)
    if previous is not None:
        if previous[0] != selected:
            raise ValueError("cannot change Liger components on an already patched streamed model")
        return dict(previous[1])
    layers = list(base.model.layers)
    if not layers or any(
        not isinstance(layer, StreamedDecoderLayer) or not layer.use_checkpoint for layer in layers
    ):
        raise ValueError("streamed Liger requires Soup's checkpointed decoder wrappers")
    norms = [base.model.norm]
    mlps = []
    for layer in layers:
        norms.extend([layer.inner.input_layernorm, layer.inner.post_attention_layernorm])
        mlps.append(layer.inner.mlp)
    # Validate the whole structure before changing any forward. A previous global
    # Liger patch is unsupported, rather than a second patch with unknown semantics.
    if any(type(norm) is not LlamaRMSNorm for norm in norms):
        raise ValueError("streamed Liger requires unpatched LlamaRMSNorm instances")
    if any(type(mlp) is not LlamaMLP for mlp in mlps):
        raise ValueError("streamed Liger requires unpatched LlamaMLP instances")
    if any(parameter.requires_grad for norm in norms for parameter in norm.parameters()):
        raise ValueError("streamed Liger requires frozen base norms")
    rms_op, swiglu_op = _load_liger_ops()

    def rms_forward(self: Any, hidden_states: Any) -> Any:
        return _rms_norm(hidden_states, self.weight, self.variance_epsilon, rms_op)

    def mlp_forward(self: Any, hidden_states: Any) -> Any:
        activated = _silu_mul(self.gate_proj(hidden_states), self.up_proj(hidden_states), swiglu_op)
        return self.down_proj(activated)

    counts = {}
    if "rms_norm" in selected:
        for norm in norms:
            norm.forward = MethodType(rms_forward, norm)
        counts["rms_norm"] = len(norms)
    if "swiglu" in selected:
        for mlp in mlps:
            mlp.forward = MethodType(mlp_forward, mlp)
        counts["swiglu"] = len(mlps)
    model._soup_streaming_liger = (selected, dict(counts))
    return counts
