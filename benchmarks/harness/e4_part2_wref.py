"""E4 part 2, calibration rule section 5: the expert wrapper references (W).

The peft ParamWrapper is *not* run here. This module reproduces by hand, in plain torch,
the operation peft intends for a fused expert parameter:

    bf16 reference:  cast A and B to bf16, then per expert
                     W_eff = baddbmm(W_bf16, B_bf16, A_bf16, alpha=s), autocast off,
                     then the expert MLP with the pinned experts implementation.
    fp32 reference:  the same with fp32 factors, fp32 W_dq and an fp32 MLP, fp32 settings.

The factor layout is peft's (``tuners/lora/layer.py`` ``ParamWrapper.get_delta_factors``,
swapped-features branch) and is pinned against peft itself in the tests.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

R = 16
ALPHA = 32
SCALING = ALPHA / R
B_STD = 0.02


def peft_factor_layout(weight_a, weight_b, num_experts: int):
    """(lhs, rhs) with delta = scaling * lhs @ rhs, per expert, as peft lays the factors out.

    ``weight_a`` is lora_A.weight of shape (r * E, in); ``weight_b`` is lora_B.weight of shape
    (out, r * E). The fused expert parameter is (E, out, in), the swapped branch of peft.
    """
    a = weight_a.reshape(num_experts, -1, weight_a.shape[-1])  # (E, r, in)
    b = weight_b.reshape(weight_b.shape[0], -1, num_experts).permute(2, 0, 1)  # (E, out, r)
    return b, a


def init_factors(in_features: int, out_features: int, num_experts: int, generator,
                 r: int = R):
    """A in peft's default scheme (kaiming uniform, a=sqrt(5), on the (r*E, in) weight), then
    B ~ N(0, 0.02) from the same generator. Both fp32; neither requires grad yet."""
    import torch

    a = torch.empty(r * num_experts, in_features, dtype=torch.float32)
    torch.nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)
    b = torch.randn(out_features, r * num_experts, generator=generator,
                    dtype=torch.float32) * B_STD
    return a, b


def make_adapter_state(experts_module, generator) -> Dict[str, Any]:
    """One adapter on ``gate_up_proj`` (spanning gate and up) and one on ``down_proj``."""
    gate_up, down = experts_module.gate_up_proj, experts_module.down_proj
    n_exp = gate_up.shape[0]
    # peft swaps in/out for a (E, out, in) parameter: in = shape[2], out = shape[1]
    state: Dict[str, Any] = {"num_experts": n_exp}
    for name, param in (("gate_up_proj", gate_up), ("down_proj", down)):
        a, b = init_factors(param.shape[2], param.shape[1], n_exp, generator)
        state[name] = {"A": a, "B": b}
    return state


def effective_weight(w, a, b, num_experts: int, bf16_factors: bool):
    """Fold the adapter into ``w``. ``bf16_factors`` True casts A and B to bf16 first."""
    import torch

    lhs, rhs = peft_factor_layout(a, b, num_experts)
    if bf16_factors:
        lhs, rhs = lhs.to(torch.bfloat16), rhs.to(torch.bfloat16)
    with torch.autocast(device_type=w.device.type, enabled=False):
        return torch.baddbmm(w, lhs, rhs, alpha=SCALING)


@contextmanager
def experts_impl(experts_module, name: Optional[str]):
    """Temporarily set the experts implementation on the module's config."""
    config = experts_module.config
    previous = config._experts_implementation
    if name is not None:
        config._experts_implementation = name
    try:
        yield config._experts_implementation
    finally:
        config._experts_implementation = previous


def run_reference(experts_module, base: Dict[str, Any], routed: Dict[str, Any],
                  state: Dict[str, Any], g, precision: str, optimizer_kwargs: Dict[str, Any],
                  impl: Optional[str] = None) -> Dict[str, Any]:
    """One call of one reference: forward, backward, one AdamW step from fresh state.

    ``base`` holds the frozen written-back weights (bf16); ``routed`` holds ``hidden``
    (T, H) bf16, ``top_k_index`` and ``top_k_weights``; ``g`` is the upstream gradient (fp32,
    shape of the output). ``precision`` is ``bf16`` (the intended operation) or ``fp32``.
    Returns output, input gradient, A and B gradients per fused parameter, factors after the
    step, and which experts implementation ran.
    """
    import e4_part2_common as common
    import torch
    from torch.func import functional_call

    bf16 = precision == "bf16"
    n_exp = state["num_experts"]
    dtype = torch.bfloat16 if bf16 else torch.float32
    leaves = {}
    for name in ("gate_up_proj", "down_proj"):
        leaves[name] = {k: state[name][k].detach().clone().requires_grad_(True) for k in "AB"}
    hidden = routed["hidden"].to(dtype).detach().clone().requires_grad_(True)
    weights = {name: base[name].to(dtype) for name in ("gate_up_proj", "down_proj")}
    folded = {
        name: effective_weight(weights[name], leaves[name]["A"], leaves[name]["B"], n_exp, bf16)
        for name in weights
    }
    top_w = routed["top_k_weights"].to(dtype)
    with experts_impl(experts_module, impl) as used, \
            torch.autocast(device_type=hidden.device.type, enabled=False):
        out = functional_call(experts_module, folded, (hidden, routed["top_k_index"], top_w))
        if not bf16:
            common.assert_fp32_reference(out, where="W fp32 output")
        loss = (out.float() * g).sum()
        loss.backward()
    result: Dict[str, Any] = {
        "output": out.detach(), "input_grad": hidden.grad.detach(), "loss": float(loss.detach()),
        "experts_implementation": used,
    }
    optimizer = torch.optim.AdamW([t for p in leaves.values() for t in p.values()],
                                  **optimizer_kwargs)
    for name, pair in leaves.items():
        result[f"{name}.A_grad"] = pair["A"].grad.detach().clone()
        result[f"{name}.B_grad"] = pair["B"].grad.detach().clone()
    optimizer.step()
    for name, pair in leaves.items():
        result[f"{name}.A_after"] = pair["A"].detach().clone()
        result[f"{name}.B_after"] = pair["B"].detach().clone()
    return result


W_QUANTITIES = ("output", "input_grad", "A_grad", "B_grad", "update")


def reference_terms(kind: str) -> Tuple[str, ...]:
    """The raw reference tensors that feed one tol_W quantity."""
    if kind in ("output", "input_grad"):
        return (kind,)
    if kind == "update":
        return tuple(f"{n}.{k}_after" for n in ("gate_up_proj", "down_proj") for k in "AB")
    return tuple(f"{n}.{kind}" for n in ("gate_up_proj", "down_proj"))


def nonfinite_terms(bf16: Dict[str, Any], fp32: Dict[str, Any]) -> list:
    """Names of every reference tensor (either arm of the pair) holding a non-finite value."""
    import torch

    bad = []
    for q in W_QUANTITIES:
        for term in reference_terms(q):
            for side, ref in (("bf16", bf16), ("fp32", fp32)):
                if not bool(torch.isfinite(ref[term].float()).all()):
                    bad.append(f"{side} {term}")
    return bad


def compare_references(bf16: Dict[str, Any], fp32: Dict[str, Any]) -> Dict[str, float]:
    """max |bf16 reference - fp32 reference| per quantity (the arms of tol_W).

    Every raw reference tensor and every per-tensor difference is checked for finiteness before
    any maximum is taken: a quantity with a non-finite term is NaN, never the maximum of the
    finite ones (Python's ``max`` lets argument order decide what happens to a NaN).
    """
    import e4_part2_common as common
    import torch

    def diff(a, b) -> float:
        return float((a.float() - b.float()).abs().max())

    out = {}
    for q in W_QUANTITIES:
        terms = reference_terms(q)
        raw_ok = all(bool(torch.isfinite(r[t].float()).all()) for t in terms
                     for r in (bf16, fp32))
        out[q] = common.max_or_nan([diff(bf16[t], fp32[t]) for t in terms]) if raw_ok \
            else float("nan")
    return out


def tol_w(per_call: list) -> Dict[str, float]:
    """tol_W(quantity) = 2 x max over calls, layers and arms of the reference difference.

    A non-finite difference raises: it is never dropped from the maximum (rule section 8).
    """
    out = {}
    for q in W_QUANTITIES:
        values = [call[q] for call in per_call]
        bad = [v for v in values if not math.isfinite(v)]
        if bad:
            raise ValueError(f"tol_W {q}: {len(bad)} value(s) are not finite")
        out[q] = 2.0 * max(values)
    return out


def capture_routed(model, layer_indices: Tuple[int, ...], input_ids_list) -> Dict[int, list]:
    """Run ``model`` (bf16, no grad) over each row and capture, per layer, what its experts see.

    The experts module receives (hidden_states, top_k_index, top_k_weights); the hook keeps
    those, which is exactly the MLP-block input after the router, one entry per row.
    """
    import torch

    captured: Dict[int, list] = {i: [] for i in layer_indices}
    handles = []
    for i in layer_indices:
        experts = model.model.layers[i].mlp.experts

        def pre_hook(_module, args, _i=i):
            hidden, index, weights = args[:3]
            captured[_i].append({"hidden": hidden.detach().clone(),
                                 "top_k_index": index.detach().clone(),
                                 "top_k_weights": weights.detach().clone()})

        handles.append(experts.register_forward_pre_hook(pre_hook))
    try:
        with torch.no_grad():
            for ids in input_ids_list:
                model(input_ids=ids.unsqueeze(0), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return captured
