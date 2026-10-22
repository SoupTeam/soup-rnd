#!/usr/bin/env python3
"""A0 harness view of the L2L step: the core lives in ``soup_cli.utils.l2l_step``.

What stays here is the gradient-accumulation reference and the adapter/gradient
comparison helpers the probe and the CPU tests use. Benchmark harness code, not
shipped. torch is imported inside functions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from soup_cli.utils.l2l_step import (  # noqa: F401
    SUPPORTED_ARCHS,
    L2LStep,
    LayerCall,
    capture_layer_call,
    label_tokens,
    model_parts,
)


def ga_step(
    model: Any, micro_batches: Sequence[Any], *, n_items: Optional[int] = None
) -> List[Any]:
    """Gradient accumulation through the shipped forward: k forward/backward calls,
    each micro-batch's summed loss divided by the label tokens of ALL k."""
    total = label_tokens(micro_batches) if n_items is None else int(n_items)
    losses = []
    for ids in micro_batches:
        out = model(input_ids=ids, labels=ids, num_items_in_batch=total)
        out.loss.backward()
        losses.append(out.loss.detach())
    return losses


def seeded_ga_step(
    model: Any, micro_batches: Sequence[Any], *, seed: int, step: int
) -> List[Any]:
    """``ga_step`` with the L2L state function installed as a pre-hook on every layer.

    The hook sets the generators from ``layer_seed(seed, step, mb, layer)`` before each
    layer call, which is what ``L2LStep`` does around its own calls. This is the reference
    for ``lora.dropout`` > 0; a stock accumulation draws every mask from one stream.
    """
    from soup_cli.utils.l2l_rng import set_layer_state
    from soup_cli.utils.layer_stream_runtime import decoder_owner

    current = {"mb": 0}
    handles = [
        layer.register_forward_pre_hook(
            lambda _module, _args, idx=idx: set_layer_state(seed, step, current["mb"], idx)
        )
        for idx, layer in enumerate(decoder_owner(model).layers)
    ]
    total = label_tokens(micro_batches)
    try:
        losses = []
        for mb, ids in enumerate(micro_batches):
            current["mb"] = mb
            out = model(input_ids=ids, labels=ids, num_items_in_batch=total)
            out.loss.backward()
            losses.append(out.loss.detach())
        return losses
    finally:
        for handle in handles:
            handle.remove()


# --------------------------------------------------------------------------
# adapters and gradients
# --------------------------------------------------------------------------
def lora_parameters(model: Any) -> Dict[str, Any]:
    from soup_cli.utils.layer_stream_runtime import canonical_named_parameters

    params = {
        name: param
        for name, param in canonical_named_parameters(model)
        if "lora_" in name and param.requires_grad
    }
    if not params:
        raise RuntimeError("the model has no trainable LoRA parameters")
    return params


def make_non_vacuous(model: Any, seed: int = 23, scale: float = 0.02) -> None:
    """PEFT initialises lora_B to zero, which makes every lora_A gradient zero."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in sorted(lora_parameters(model).items()):
            if "lora_B" in name:
                values = torch.randn(param.shape, generator=generator) * scale
                param.copy_(values.to(param.device, param.dtype))


def snapshot_adapters(model: Any) -> Dict[str, Any]:
    return {name: param.detach().clone() for name, param in lora_parameters(model).items()}


def zero_grads(model: Any) -> None:
    for param in lora_parameters(model).values():
        param.grad = None


def restore_adapters(model: Any, snapshot: Dict[str, Any]) -> None:
    import torch

    params = lora_parameters(model)
    if set(params) != set(snapshot):
        raise RuntimeError("the adapter snapshot does not match the model's LoRA parameters")
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(snapshot[name])
    zero_grads(model)


def collect_grads(model: Any) -> Dict[str, Any]:
    out = {}
    for name, param in lora_parameters(model).items():
        if param.grad is None:
            raise RuntimeError(f"{name} received no gradient")
        out[name] = param.grad.detach().to("cpu", copy=True)
    return out


def compare(reference: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    if set(reference) != set(candidate):
        missing = sorted(set(reference) ^ set(candidate))[:4]
        raise ValueError(f"the two gradient sets name different tensors, e.g. {missing}")
    unequal, zero, diffs = [], [], []
    for name in sorted(reference):
        left, right = reference[name], candidate[name]
        if not int(torch.count_nonzero(left)):
            zero.append(name)
        if not torch.equal(left, right):
            unequal.append(name)
            diffs.append((float((left.float() - right.float()).abs().max()), name))
    diffs.sort(reverse=True)
    return {
        "tensors": len(reference),
        "unequal": unequal,
        "max_abs": diffs[0][0] if diffs else 0.0,
        "worst": [{"name": name, "max_abs": value} for value, name in diffs[:5]],
        "all_zero": zero,
        "equal": not unequal,
    }


def losses_equal(left: Sequence[Any], right: Sequence[Any]) -> bool:
    import torch

    return len(left) == len(right) and all(
        torch.equal(a.detach().cpu(), b.detach().cpu()) for a, b in zip(left, right)
    )
