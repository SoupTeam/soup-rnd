"""Install the existing Fast-LoRA operations in the production SFT load path."""

from typing import Any


def apply_fast_lora_training(model: Any, *, streaming: bool = False) -> dict[str, int]:
    """Patch one ordinary Llama LoRA model, preserving parameters and adapter format.

    Installation counts are not evidence of executed kernels or faster training.
    Unsupported dynamic adapter states retain the low-level patchers' PEFT fallback.
    """
    from peft.tuners.lora.layer import Linear
    from torch import nn

    from soup_cli.utils.fast_lora import (
        patch_fast_lora_single_projection,
        unpatch_fast_lora_single_projection,
    )
    from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp, unpatch_fast_lora_mlp
    from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv, unpatch_fast_lora_qkv

    if getattr(getattr(model, "config", None), "model_type", None) != "llama":
        raise ValueError("training.fast_lora initially supports Llama models only")
    if getattr(model, "_soup_fast_lora_training_counts", None) is not None:
        return dict(model._soup_fast_lora_training_counts)
    projection_types = (Linear,)
    base_types = (nn.Linear,)
    if streaming:
        # NF4 is optional for dense streaming. Do not import bitsandbytes until
        # the model actually carries a quantized projection.
        if getattr(model, "is_loaded_in_4bit", False):
            from bitsandbytes.nn import Linear4bit
            from peft.tuners.lora.bnb import Linear4bit as LoraLinear4bit

            projection_types += (LoraLinear4bit,)
            base_types += (Linear4bit,)
    projections = [module for module in model.modules() if hasattr(module, "lora_A")]
    if not projections:
        raise ValueError("training.fast_lora found no ordinary LoRA Linear targets")
    for projection in projections:
        active = projection.active_adapters
        if type(projection) not in projection_types or len(active) != 1:
            raise ValueError("training.fast_lora requires one active ordinary Linear LoRA adapter")
        adapter = active[0]
        base = projection.get_base_layer()
        if type(base) is not nn.Linear and getattr(base.weight, "quant_type", None) != "nf4":
            raise ValueError("training.fast_lora supports streamed NF4 weights only")
        if (
            type(base) not in base_types
            or base.weight.requires_grad
            or projection.merged
            or projection.disable_adapters
            or adapter in getattr(projection, "lora_variant", {})
            or float(getattr(projection.lora_dropout[adapter], "p", 0)) != 0
            or projection.fan_in_fan_out
            or projection.lora_A[adapter].bias is not None
            or projection.lora_B[adapter].bias is not None
            or projection.lora_A[adapter].weight.is_meta
            or projection.lora_B[adapter].weight.is_meta
        ):
            raise ValueError(
                "training.fast_lora requires frozen dense/streamed NF4 weights "
                "and materialized plain dropout-free LoRA"
            )
    if any(
        parameter.requires_grad and "lora_" not in name
        for name, parameter in model.named_parameters()
    ):
        raise ValueError("training.fast_lora supports training LoRA parameters only")
    try:
        counts = {
            "mlp": patch_fast_lora_mlp(model),
            "qkv": patch_fast_lora_qkv(model),
            "single": patch_fast_lora_single_projection(model),
        }
        if not any(counts.values()):
            raise ValueError("training.fast_lora did not install any supported operations")
    except Exception:
        unpatch_fast_lora_single_projection(model)
        unpatch_fast_lora_qkv(model)
        unpatch_fast_lora_mlp(model)
        raise
    model._soup_fast_lora_training_counts = dict(counts)
    return counts
