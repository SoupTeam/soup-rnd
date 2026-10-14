"""B2 tickets 06 to 08: DeepSeek-V3 streams, parity controls prove it on a tiny model
and its attention and router variants, and a DeepSeek-V3 class from Hub code is refused."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional

import pytest
import torch

from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR

# ==========================================================================
# the architecture gate
# ==========================================================================


def test_deepseek_v3_is_accepted_under_its_own_name() -> None:
    from soup_cli.utils.layer_stream import _STREAM_ARCH_ALIASES, stream_arch_of

    assert stream_arch_of(SimpleNamespace(model_type="deepseek_v3")) == "deepseek_v3"
    assert "deepseek_v3" not in _STREAM_ARCH_ALIASES
    assert "deepseek_v3" not in _STREAM_ARCH_ALIASES.values()


def test_kimi_k2_is_refused_naming_what_is_missing() -> None:
    from soup_cli.utils.layer_stream import stream_arch_of

    with pytest.raises(ValueError) as excinfo:
        stream_arch_of(SimpleNamespace(model_type="kimi_k2"))

    message = str(excinfo.value)
    assert "model_type='kimi_k2'" in message
    assert "does not know the kimi_k2 config type" in message
    assert "tokenizer is Hub code" in message
    assert "no conversion of Kimi K2's per-expert weights into fused experts" in message
    assert "checkpoint is FP8" in message
    assert "too large to stream as a single layer" in message


@pytest.mark.parametrize("model_type", ["deepseek_v32", "deepseek_v2", "kimi_k25"])
def test_other_deepseek_and_kimi_types_get_the_ordinary_refusal(model_type: str) -> None:
    from soup_cli.utils.layer_stream import SUPPORTED_STREAM_ARCHS, stream_arch_of

    with pytest.raises(ValueError) as excinfo:
        stream_arch_of(SimpleNamespace(model_type=model_type))

    assert str(excinfo.value) == (
        f"layer streaming does not support model_type={model_type!r}. "
        f"Supported: {', '.join(SUPPORTED_STREAM_ARCHS)}."
    )


# ==========================================================================
# the parity harness: resident against streamed on a tiny DeepSeek-V3
# ==========================================================================
#
# Tickets 07 to 11 add arms to this harness. An arm is a fixture config plus,
# optionally, an edit of the saved checkpoint; `_run_parity` does the rest.

_BIAS = "mlp.gate.e_score_correction_bias"
_N_EXPERTS = 16
_EXPERT_IMPLEMENTATIONS = ("grouped_mm", "eager")
_DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.gpu(reason="the CUDA fp32 parity control")),
]


def _tiny_deepseek_v3_config(**overrides: Any) -> Any:
    from transformers import DeepseekV3Config

    fields: Dict[str, Any] = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=3,
        first_k_dense_replace=1,
        n_routed_experts=_N_EXPERTS,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=4,
        topk_group=2,
        q_lora_rank=24,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    fields.update(overrides)
    return DeepseekV3Config(**fields)


def _moe_layers(config: Any) -> range:
    return range(config.first_k_dense_replace, config.num_hidden_layers)


def _save_fixture(weights: Path, config: Any) -> None:
    """Save a fp32 checkpoint with a spread router bias and per-expert keys on disk."""
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    torch.manual_seed(61)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    with torch.no_grad():
        for idx in _moe_layers(config):
            model.model.layers[idx].mlp.gate.e_score_correction_bias.normal_(0.0, 0.01)
    model.save_pretrained(weights, safe_serialization=True)
    last = config.num_hidden_layers - 1
    with safe_open(str(weights / "model.safetensors"), framework="pt") as handle:
        keys = set(handle.keys())
    assert f"model.layers.{last}.mlp.experts.{config.n_routed_experts - 1}.gate_proj.weight" in keys
    assert f"model.layers.{last}.mlp.experts.gate_up_proj" not in keys


def _lora(config: Any = None) -> Any:
    """The adapter Soup builds for ``target_modules: auto`` on DeepSeek-V3.

    Those are the MLA projections: ``q_a_proj`` and ``q_b_proj``, or the single
    ``q_proj`` when ``config.q_lora_rank`` is None (ticket 13). Naming
    ``gate_proj``, ``up_proj`` or ``down_proj`` here would not reach the shared
    experts: peft 0.21 rewrites those names on ``deepseek_v3`` into
    ``target_parameters`` on the fused routed experts, which is ``moe_lora``
    (ticket 10).
    """
    from peft import LoraConfig, TaskType

    from soup_cli.utils.peft_wiring import MOE_TEXT_LORA_TARGETS, resolve_lora_target_modules

    targets = (
        resolve_lora_target_modules(SimpleNamespace(config=config), "auto")
        if config is not None
        else list(MOE_TEXT_LORA_TARGETS["deepseek_v3"])
    )
    return LoraConfig(
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
        target_modules=targets,
        task_type=TaskType.CAUSAL_LM,
    )


def _strip_stream_wrapper(name: str) -> str:
    """A streamed layer's name without the ``.inner.`` its wrapper adds."""
    return name.replace(".inner.", ".")


def _base_weights(layer: Any) -> Dict[str, Any]:
    """A decoder layer's base weights by name, wrappers stripped.

    After a forward pass, peft's ``ParamWrapper`` leaves the fused experts it adapted
    filed under ``_buffers`` as ``Parameter`` objects, where ``named_parameters`` no
    longer lists them. Read them from there too.
    """
    from soup_cli.utils.layer_stream_runtime import strip_peft_wrappers

    found = {
        strip_peft_wrappers(name): param
        for name, param in layer.named_parameters()
        if "lora_" not in name
    }
    found.update(
        {
            strip_peft_wrappers(name): buf
            for name, buf in layer.named_buffers()
            if isinstance(buf, torch.nn.Parameter)
        }
    )
    return found


def _lora_params(model: Any) -> Dict[str, Any]:
    return {
        _strip_stream_wrapper(name): param
        for name, param in model.named_parameters()
        if "lora_" in name
    }


def _randomise_lora_b(model: Any) -> None:
    """PEFT starts ``lora_B`` at zero, which makes a detached adapter look equal."""
    gen = torch.Generator().manual_seed(67)
    with torch.no_grad():
        for name, param in sorted(_lora_params(model).items()):
            if "lora_B" in name:
                param.copy_(torch.randn(param.shape, generator=gen).to(param) * 0.05)


def _copy_lora(src: Any, dst: Any) -> None:
    source, target = _lora_params(src), _lora_params(dst)
    assert source and set(source) == set(target)
    with torch.no_grad():
        for name, param in target.items():
            param.copy_(source[name])


def _record_expert_hits(model: Any, hits: Dict[str, set]) -> List[Any]:
    """Hook every router; ``hits`` maps a router's name to the experts it chose."""
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3TopkRouter

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, DeepseekV3TopkRouter):
            seen = hits.setdefault(_strip_stream_wrapper(name), set())

            def hook(_module: Any, _args: Any, output: Any, seen: set = seen) -> None:
                seen.update(output[2].flatten().tolist())

            handles.append(module.register_forward_hook(hook))
    return handles


def _record_attention_masks(model: Any, masks: List[bool]) -> List[Any]:
    """Hook every attention module. ``masks`` gets one entry per call, true when the
    call received a mask tensor. A mask of ``None`` makes sdpa run ``is_causal``."""
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention

    def hook(_module: Any, _args: Any, kwargs: Dict[str, Any]) -> None:
        masks.append(isinstance(kwargs.get("attention_mask"), torch.Tensor))

    return [
        module.register_forward_pre_hook(hook, with_kwargs=True)
        for module in model.modules()
        if isinstance(module, DeepseekV3Attention)
    ]


@dataclass
class _Pass:
    logits: Any
    #: Every LoRA parameter's gradient by name; ``None`` where none reached it.
    grads: Dict[str, Any]
    expert_hits: Dict[str, set]
    #: Per attention call, whether it got an explicit mask tensor.
    attention_masks: List[bool]


def _forward_backward(model: Any, batch: Dict[str, Any]) -> _Pass:
    hits: Dict[str, set] = {}
    masks: List[bool] = []
    handles = _record_expert_hits(model, hits) + _record_attention_masks(model, masks)
    try:
        model.train()
        model.zero_grad(set_to_none=True)
        out = model(**batch)
        out.loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    grads = {
        name: None if param.grad is None else param.grad.detach().clone()
        for name, param in _lora_params(model).items()
    }
    return _Pass(out.logits.detach().clone(), grads, hits, masks)


#: Real tokens per row of the padded batch; the rest of each 64-token row is
#: right padding.
_SEQ_LEN = 64
_PADDED_LENGTHS = (64, 57, 40, 23, 64, 11, 48, 33)


def _batch(vocab_size: int, device: str, *, padded: bool) -> Dict[str, Any]:
    """A batch of 8 rows of ``_SEQ_LEN`` tokens. It has 8 rows because with 4 rows,
    layer 1 of arm 2 routed no token to expert 12. Padded, each row ends in padding
    that the mask hides and the labels ignore."""
    gen = torch.Generator().manual_seed(73)
    input_ids = torch.randint(0, vocab_size, (len(_PADDED_LENGTHS), _SEQ_LEN), generator=gen)
    if not padded:
        return {"input_ids": input_ids.to(device), "labels": input_ids.to(device)}
    attention_mask = torch.zeros_like(input_ids)
    for row, length in enumerate(_PADDED_LENGTHS):
        attention_mask[row, :length] = 1
    input_ids = input_ids.masked_fill(attention_mask == 0, 0)
    labels = input_ids.masked_fill(attention_mask == 0, -100)
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
    }


@dataclass
class _Parity:
    config: Any
    resident: _Pass
    streamed: _Pass
    meta_violations: List[str]
    streamed_weights: Dict[int, set]
    lora_names: List[str]
    #: Per decoder layer, the base weights stored as packed ``Params4bit``; empty without NF4.
    quantized: Dict[int, set] = field(default_factory=dict)
    #: The same, read from the resident model; empty without NF4.
    resident_quantized: Dict[int, set] = field(default_factory=dict)
    #: The runtime's tier and the class of the weight source that fed the streamed layers.
    tier: str = ""
    source: str = ""

    def forward_equal(self) -> bool:
        return torch.equal(self.streamed.logits, self.resident.logits)

    def backward_mismatches(self) -> List[str]:
        names = set(self.streamed.grads) | set(self.resident.grads)
        return sorted(
            name
            for name in names
            if self.streamed.grads.get(name) is None
            or self.resident.grads.get(name) is None
            or not torch.equal(self.streamed.grads[name], self.resident.grads[name])
        )

    def report(self) -> str:
        if self.forward_equal():
            forward = "forward: logits equal"
        else:
            diff = (self.streamed.logits - self.resident.logits).abs().max().item()
            forward = f"forward: logits differ, max |diff| {diff:.3e}"
        bad = self.backward_mismatches()
        backward = (
            f"backward: {len(bad)} of {len(self.lora_names)} LoRA gradients differ: {bad[:4]}"
            if bad
            else f"backward: all {len(self.lora_names)} LoRA gradients equal"
        )
        return f"{forward}; {backward}"


def _run_parity(
    tmp_path: Path,
    *,
    device: str,
    experts: str,
    edit_checkpoint: Optional[Callable[[Path], None]] = None,
    edit_streamed: Optional[Callable[[Any], None]] = None,
    config: Any = None,
    notices: Optional[List[str]] = None,
    padded: bool = False,
    quant: str = "none",
    tier: str = "ram",
    lora_config: Optional[Callable[[Path], Any]] = None,
) -> _Parity:
    """Shard and stream the fixture the way the training setup does, then run
    the resident model and the streamed model on one batch."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream import stream_arch_of
    from soup_cli.utils.layer_stream_runtime import build_streamed_model, decoder_owner
    from soup_cli.utils.peft_wiring import attach_adapter

    config = config if config is not None else _tiny_deepseek_v3_config()
    weights = tmp_path / "weights"
    _save_fixture(weights, config)
    if edit_checkpoint is not None:
        edit_checkpoint(weights)

    lora = lora_config(weights) if lora_config is not None else _lora(config)
    arch = stream_arch_of(AutoConfig.from_pretrained(str(weights)))
    shards = str(tmp_path / "shards")
    nf4 = quant == "nf4"
    quant_args: Dict[str, Any] = {}
    if nf4:
        from soup_cli.utils.layer_stream_runtime import (
            build_meta_skeleton,
            quantised_layer_suffixes,
        )

        quant_args = dict(
            quant=quant,
            quant_suffixes=quantised_layer_suffixes(
                build_meta_skeleton(str(weights), dtype="float32", quant=quant)
            ),
            # Production quantizes on the training device and keys its cache by it. The
            # CPU quantizer packs different bytes than the CUDA one the resident load uses.
            quant_device=device,
        )
    index = shard_checkpoint(
        str(weights),
        shards,
        dtype="float32",
        arch=arch,
        notify=notices.append if notices is not None else None,
        **quant_args,
    )
    streamed, runtime = build_streamed_model(
        model_id=str(weights),
        shard_dir=shards,
        index=index,
        lora_config=lora,
        device=device,
        dtype="float32",
        buffers=2,
        pin=False,
        seed=71,
        quant=quant,
        tier=tier,
    )
    try:
        streamed.get_base_model().set_experts_implementation(experts)
        assert streamed.get_base_model().config._experts_implementation == experts
        _randomise_lora_b(streamed)

        resident_kwargs: Dict[str, Any] = {}
        if nf4:
            from soup_cli.utils.layer_stream_runtime import build_nf4_config

            resident_kwargs = dict(
                quantization_config=build_nf4_config("float32"), device_map={"": device}
            )
        resident_base: Any = AutoModelForCausalLM.from_pretrained(
            str(weights), dtype=torch.float32, experts_implementation=experts, **resident_kwargs
        )
        if not nf4:
            resident_base.to(device)
        assert resident_base.config._experts_implementation == experts
        resident = attach_adapter(resident_base, lora)
        _copy_lora(streamed, resident)
        if edit_streamed is not None:
            edit_streamed(streamed)

        batch = _batch(config.vocab_size, device, padded=padded)
        streamed_pass = _forward_backward(streamed, batch)
        resident_pass = _forward_backward(resident, batch)
        layers = decoder_owner(streamed).layers
        base_weights = {idx: _base_weights(layer.inner) for idx, layer in enumerate(layers)}
        meta_violations = [
            f"layer {idx}: {name}"
            for idx, weights_of in base_weights.items()
            for name, param in weights_of.items()
            if not param.is_meta
        ]
        streamed_weights = {idx: set(weights_of) for idx, weights_of in base_weights.items()}
        lora_names = sorted(_lora_params(resident))
        quantized: Dict[int, set] = {}
        resident_quantized: Dict[int, set] = {}
        if nf4:
            # Read from the built model, not from quantised_layer_suffixes, so the
            # test does not trust the code that chose what to quantize.
            quantized = {
                idx: {
                    name.replace(".base_layer.", ".")
                    for name, param in layer.inner.named_parameters()
                    if type(param).__name__ == "Params4bit"
                }
                for idx, layer in enumerate(layers)
            }
            resident_quantized = {idx: set() for idx in quantized}
            for name, param in resident.named_parameters():
                found = re.search(r"\.layers\.(\d+)\.(.*)", name)
                if type(param).__name__ == "Params4bit" and found:
                    resident_quantized[int(found.group(1))].add(
                        found.group(2).replace(".base_layer.", ".")
                    )
        tier_used, source_used = runtime.tier, type(runtime.source).__name__
    finally:
        runtime.close()
    return _Parity(
        config,
        resident_pass,
        streamed_pass,
        meta_violations,
        streamed_weights,
        lora_names,
        quantized=quantized,
        resident_quantized=resident_quantized,
        tier=tier_used,
        source=source_used,
    )


def _assert_parity(parity: _Parity) -> None:
    """Every check an arm makes: coverage, meta, a live adapter, then equality,
    with forward and backward asserted separately."""
    config = parity.config
    for side in (parity.resident, parity.streamed):
        assert len(side.expert_hits) == len(_moe_layers(config))
        for router, seen in side.expert_hits.items():
            unused = set(range(config.n_routed_experts)) - seen
            assert not unused, f"{router} never chose experts {sorted(unused)}"
    for idx in _moe_layers(config):
        assert {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"} <= parity.streamed_weights[idx]
    assert parity.meta_violations == [], f"streamed weights left meta: {parity.meta_violations}"
    layer0 = [name for name in parity.lora_names if ".layers.0." in name]
    assert layer0
    for name in layer0:
        grad = parity.streamed.grads[name]
        assert grad is not None and bool(grad.ne(0).any()), f"{name}: no or zero gradient"
    assert parity.forward_equal(), parity.report()
    assert not parity.backward_mismatches(), parity.report()


@pytest.fixture
def deterministic(request: Any) -> Iterator[None]:
    """On CUDA, the parity runs under deterministic algorithms.

    cuBLAS reads ``CUBLAS_WORKSPACE_CONFIG`` when it creates its handle, so the
    variable must be set before the process first touches CUDA and must stay set
    for later tests. It is left in the environment for that reason. A process that
    touched CUDA without it skips, instead of running with no effect.
    """
    import os

    if request.node.callspec.params.get("device") != "cuda":
        yield
        return
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        if torch.cuda.is_initialized():
            pytest.skip(
                "CUDA was initialised before CUBLAS_WORKSPACE_CONFIG was set; run with "
                "CUBLAS_WORKSPACE_CONFIG=:4096:8 exported"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(False)


@_REQUIRES_TRAINING_FLOOR
@pytest.mark.parametrize("experts", _EXPERT_IMPLEMENTATIONS)
@pytest.mark.parametrize("device", _DEVICES)
def test_streamed_deepseek_v3_matches_resident(
    tmp_path: Path, device: str, experts: str, deterministic: None
) -> None:
    """Parity arm 1: the base fixture."""
    _assert_parity(_run_parity(tmp_path, device=device, experts=experts))


def _zero_router_bias(streamed: Any) -> None:
    from soup_cli.utils.layer_stream_runtime import decoder_owner

    zeroed = 0
    with torch.no_grad():
        for layer in decoder_owner(streamed).layers:
            for name, buf in layer.inner.named_buffers():
                if name == _BIAS:
                    assert bool(buf.ne(0).any())
                    buf.zero_()
                    zeroed += 1
    assert zeroed == len(_moe_layers(streamed.get_base_model().config))


@_REQUIRES_TRAINING_FLOOR
@pytest.mark.parametrize("device", _DEVICES)
def test_control_streamed_model_with_a_zero_router_bias_differs(
    tmp_path: Path, device: str, deterministic: None
) -> None:
    """The parity check can see a routing error: the bias the runtime fills matters."""
    parity = _run_parity(
        tmp_path, device=device, experts="grouped_mm", edit_streamed=_zero_router_bias
    )

    assert not parity.forward_equal(), parity.report()
    assert parity.backward_mismatches(), parity.report()


# --------------------------------------------------------------------------
# ticket 08: arms for the attention and router variants real checkpoints use
# --------------------------------------------------------------------------


def _yarn(
    factor: float,
    *,
    rope_theta: float,
    original_max_positions: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> Dict[str, Any]:
    """Yarn rope. The default betas are DeepSeek-V3's; Kimi K2 sets both to 1.0."""
    return {
        "rope_type": "yarn",
        "rope_theta": rope_theta,
        "factor": factor,
        "original_max_position_embeddings": original_max_positions,
        "beta_fast": beta_fast,
        "beta_slow": beta_slow,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    }


@dataclass(frozen=True)
class _Arm:
    overrides: Dict[str, Any]
    padded: bool = False


#: Arms 2 to 6. Each changes the base fixture's config or its batch.
_VARIANT_ARMS = {
    "arm2-q-lora-rank-none": _Arm({"q_lora_rank": None}),
    "arm3-rope-not-interleaved": _Arm({"rope_interleave": False}),
    "arm4-yarn-factor-4": _Arm(
        {"rope_parameters": _yarn(4.0, rope_theta=10000.0, original_max_positions=32)}
    ),
    "arm5-padded-batch": _Arm({}, padded=True),
    # Kimi K2's routing and rope under model_type=deepseek_v3: one expert group,
    # one dense layer, and K2's yarn (factor 32 over 4096 positions, theta 50000).
    "arm6-kimi-k2-shape": _Arm(
        {
            "n_group": 1,
            "topk_group": 1,
            "first_k_dense_replace": 1,
            "max_position_embeddings": 131072,
            "rope_parameters": _yarn(
                32.0,
                rope_theta=50000.0,
                original_max_positions=4096,
                beta_fast=1.0,
                beta_slow=1.0,
            ),
        }
    ),
}
_VARIANT_ARM_PARAMS = [pytest.param(arm, id=name) for name, arm in _VARIANT_ARMS.items()]


def _assert_arm_took_effect(parity: _Parity, arm: _Arm) -> None:
    """The variant is in the model that ran, not only in the dict above."""
    config = parity.config
    assert config.model_type == "deepseek_v3"
    for key, value in arm.overrides.items():
        if key == "rope_parameters":
            for field, expected in value.items():
                assert config.rope_parameters[field] == expected, field
        else:
            assert getattr(config, key) == value, key
    if "q_lora_rank" in arm.overrides:
        assert all(".q_a_proj." not in name for name in parity.lora_names)
        # The query projection is adapted and its gradient compared (ticket 13).
        assert any(".self_attn.q_proj.lora_" in name for name in parity.lora_names)
        for idx in range(config.num_hidden_layers):
            assert "self_attn.q_proj.weight" in parity.streamed_weights[idx]
    for side in (parity.resident, parity.streamed):
        assert side.attention_masks
        if arm.padded:
            assert all(side.attention_masks)
        else:
            assert not any(side.attention_masks)


@_REQUIRES_TRAINING_FLOOR
@pytest.mark.parametrize("arm", _VARIANT_ARM_PARAMS)
@pytest.mark.parametrize("experts", _EXPERT_IMPLEMENTATIONS)
@pytest.mark.parametrize("device", _DEVICES)
def test_streamed_deepseek_v3_variant_matches_resident(
    tmp_path: Path, device: str, experts: str, arm: _Arm, deterministic: None
) -> None:
    """Parity arms 2 to 6."""
    parity = _run_parity(
        tmp_path,
        device=device,
        experts=experts,
        config=_tiny_deepseek_v3_config(**arm.overrides),
        padded=arm.padded,
    )

    _assert_arm_took_effect(parity, arm)
    _assert_parity(parity)


@_REQUIRES_TRAINING_FLOOR
@pytest.mark.parametrize("arm", _VARIANT_ARM_PARAMS)
@pytest.mark.parametrize("device", _DEVICES)
def test_control_variant_with_a_zero_router_bias_differs(
    tmp_path: Path, device: str, arm: _Arm, deterministic: None
) -> None:
    """Each arm's parity check can see a routing error."""
    parity = _run_parity(
        tmp_path,
        device=device,
        experts="grouped_mm",
        config=_tiny_deepseek_v3_config(**arm.overrides),
        padded=arm.padded,
        edit_streamed=_zero_router_bias,
    )

    _assert_arm_took_effect(parity, arm)
    assert not parity.forward_equal(), parity.report()
    assert parity.backward_mismatches(), parity.report()


# --------------------------------------------------------------------------
# ticket 09: NF4 and the disk tier
# --------------------------------------------------------------------------

_ATTENTION_LINEARS = {
    "self_attn.q_a_proj.weight",
    "self_attn.q_b_proj.weight",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.kv_b_proj.weight",
    "self_attn.o_proj.weight",
}
_SHARED_EXPERT_LINEARS = {
    f"mlp.shared_experts.{name}_proj.weight" for name in ("gate", "up", "down")
}
#: The dense MLP of layer 0 is three ordinary linears, so NF4 quantizes it too.
_DENSE_MLP_LINEARS = {f"mlp.{name}_proj.weight" for name in ("gate", "up", "down")}


@_REQUIRES_TRAINING_FLOOR
@pytest.mark.parametrize("experts", _EXPERT_IMPLEMENTATIONS)
@pytest.mark.parametrize("device", _DEVICES)
def test_streamed_deepseek_v3_from_the_disk_tier_matches_resident(
    tmp_path: Path, device: str, experts: str, deterministic: None
) -> None:
    """Parity arm 11: the base fixture, streamed from the disk tier."""
    parity = _run_parity(tmp_path, device=device, experts=experts, tier="disk")

    assert (parity.tier, parity.source) == ("disk", "AsyncDiskSource")
    _assert_parity(parity)


@_REQUIRES_TRAINING_FLOOR
@pytest.mark.parametrize("experts", _EXPERT_IMPLEMENTATIONS)
@pytest.mark.parametrize("tier", ["ram", "disk"])
@pytest.mark.parametrize(
    "device", [pytest.param("cuda", marks=pytest.mark.gpu(reason="the 4-bit kernels are GPU-only"))]
)
def test_streamed_deepseek_v3_nf4_matches_resident_nf4(
    tmp_path: Path, device: str, tier: str, experts: str, deterministic: None
) -> None:
    """Parity arm 8: resident NF4 against streamed NF4 on the base fixture.

    NF4 quantizes the attention and shared-expert linears, and layer 0's dense MLP.
    The routed experts, the router weight, the router bias and the norms keep the
    base dtype.
    """
    pytest.importorskip("bitsandbytes")

    parity = _run_parity(tmp_path, device=device, experts=experts, quant="nf4", tier=tier)

    expected = {
        0: _ATTENTION_LINEARS | _DENSE_MLP_LINEARS,
        1: _ATTENTION_LINEARS | _SHARED_EXPERT_LINEARS,
        2: _ATTENTION_LINEARS | _SHARED_EXPERT_LINEARS,
    }
    assert parity.quantized == expected
    assert parity.resident_quantized == expected
    assert parity.tier == tier
    _assert_parity(parity)


def _add_mtp_layer(weights: Path) -> int:
    """Write an MTP layer into the checkpoint after the last decoder layer, as a real
    DeepSeek-V3 checkpoint carries one: a copy of the last decoder layer plus the
    MTP projections and norms. Returns the number of keys added."""
    from safetensors.torch import load_file, save_file

    path = str(weights / "model.safetensors")
    state = load_file(path)
    mtp = {
        key.replace("model.layers.2.", "model.layers.3.", 1): value.clone()
        for key, value in state.items()
        if key.startswith("model.layers.2.")
    }
    hidden = state["model.norm.weight"].shape[0]
    mtp.update(
        {
            "model.layers.3.enorm.weight": torch.ones(hidden),
            "model.layers.3.hnorm.weight": torch.ones(hidden),
            "model.layers.3.eh_proj.weight": torch.randn(hidden, 2 * hidden),
            "model.layers.3.shared_head.norm.weight": torch.ones(hidden),
            "model.layers.3.shared_head.head.weight": state["lm_head.weight"].clone(),
        }
    )
    save_file({**state, **mtp}, path, metadata={"format": "pt"})
    return len(mtp)


@_REQUIRES_TRAINING_FLOOR
@pytest.mark.parametrize("experts", _EXPERT_IMPLEMENTATIONS)
@pytest.mark.parametrize("device", _DEVICES)
def test_an_mtp_layer_is_dropped_and_the_streamed_model_still_matches(
    tmp_path: Path, device: str, experts: str, deterministic: None
) -> None:
    """Parity arm 7: the sharder drops the MTP layer's keys and says how many."""
    added: List[int] = []
    notices: List[str] = []

    parity = _run_parity(
        tmp_path,
        device=device,
        experts=experts,
        edit_checkpoint=lambda weights: added.append(_add_mtp_layer(weights)),
        notices=notices,
    )

    assert added and added[0] > 3 * _N_EXPERTS
    assert [notice for notice in notices if "Dropped" in notice] == [
        f"[yellow]Dropped {added[0]} checkpoint keys of decoder layers at or beyond "
        "the model's 3 layers (an MTP layer, for example).[/]"
    ]
    _assert_parity(parity)


# ==========================================================================
# ticket 07: a DeepSeek-V3 skeleton must be the transformers class
# ==========================================================================

_HUB_MODELING = '''\
from transformers import DeepseekV3ForCausalLM


class HubDeepseekV3ForCausalLM(DeepseekV3ForCausalLM):
    """Stands in for the modeling file a DeepSeek-V3 checkpoint ships on the Hub."""
'''

# DeepSeek's own modeling_deepseek.py imports this name, which transformers 5 removed,
# so the real Hub class fails to load before it can be built.
_HUB_MODELING_STALE_IMPORT = (
    "from transformers.utils.import_utils import is_torch_fx_available\n" + _HUB_MODELING
)


@pytest.fixture
def hub_modules_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Load Hub code into a module cache under ``tmp_path`` instead of ``~/.cache``,
    and forget the loaded modules afterwards."""
    import transformers.dynamic_module_utils as dynamic_module_utils

    def loaded() -> List[str]:
        return [name for name in sys.modules if name.split(".")[0] == "transformers_modules"]

    saved = {name: sys.modules.pop(name) for name in loaded()}
    monkeypatch.setattr(dynamic_module_utils, "HF_MODULES_CACHE", str(tmp_path / "hf_modules"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    try:
        yield
    finally:
        for name in loaded():
            del sys.modules[name]
        sys.modules.update(saved)


def _save_hub_code_fixture(
    weights: Path, *, auto_map: bool = True, modeling: str = _HUB_MODELING
) -> None:
    """The tiny DeepSeek-V3 with a local Hub-style modeling file that ``auto_map`` names."""
    _save_fixture(weights, _tiny_deepseek_v3_config())
    (weights / "modeling_hub_deepseek.py").write_text(modeling, encoding="utf-8")
    if auto_map:
        config_path = weights / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["auto_map"] = {
            "AutoModelForCausalLM": "modeling_hub_deepseek.HubDeepseekV3ForCausalLM"
        }
        config_path.write_text(json.dumps(config), encoding="utf-8")


@_REQUIRES_TRAINING_FLOOR
@pytest.mark.parametrize(
    "modeling",
    [_HUB_MODELING, _HUB_MODELING_STALE_IMPORT],
    ids=["hub-class", "hub-class-that-fails-to-import"],
)
def test_a_deepseek_v3_class_from_hub_code_is_refused(
    tmp_path: Path, hub_modules_cache: None, modeling: str
) -> None:
    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton

    weights = tmp_path / "weights"
    _save_hub_code_fixture(weights, modeling=modeling)

    with pytest.raises(ValueError) as excinfo:
        build_meta_skeleton(str(weights), dtype="float32", trust_remote_code=True)

    message = str(excinfo.value)
    assert "modeling_hub_deepseek.HubDeepseekV3ForCausalLM" in message
    assert "transformers DeepSeek-V3 class" in message
    assert "without --trust-remote-code" in message


@_REQUIRES_TRAINING_FLOOR
@pytest.mark.parametrize(
    ("auto_map", "trust_remote_code"),
    [(True, False), (False, True)],
    ids=["auto_map-without-remote-code", "remote-code-without-auto_map"],
)
def test_the_same_fixture_without_the_hub_class_builds(
    tmp_path: Path, hub_modules_cache: None, auto_map: bool, trust_remote_code: bool
) -> None:
    from transformers import DeepseekV3ForCausalLM

    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton

    weights = tmp_path / "weights"
    _save_hub_code_fixture(weights, auto_map=auto_map)

    model = build_meta_skeleton(str(weights), dtype="float32", trust_remote_code=trust_remote_code)

    assert type(model) is DeepseekV3ForCausalLM


@_REQUIRES_TRAINING_FLOOR
def test_a_deepseek_v3_folder_with_auto_map_passes_the_gate_without_the_flag(
    tmp_path: Path, hub_modules_cache: None
) -> None:
    from soup_cli.utils.trust_remote import (
        model_requires_trust_remote_code,
        resolve_trust_remote_code,
    )

    weights = tmp_path / "weights"
    _save_hub_code_fixture(weights)

    requires = model_requires_trust_remote_code(str(weights))

    assert requires is False
    assert resolve_trust_remote_code(
        str(weights), requested=False, requires_remote_code=bool(requires)
    ) is False


class _UnreadableIndex:
    """A shard index that fails the test if anything reads it."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the shard index was read: {name}")


@_REQUIRES_TRAINING_FLOOR
def test_the_refusal_fires_before_any_shard_is_read(
    tmp_path: Path, hub_modules_cache: None
) -> None:
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    weights = tmp_path / "weights"
    _save_hub_code_fixture(weights)

    with pytest.raises(ValueError, match="without --trust-remote-code"):
        build_streamed_model(
            model_id=str(weights),
            shard_dir=str(tmp_path / "no-shards"),
            index=_UnreadableIndex(),
            lora_config=_lora(),
            device="cpu",
            dtype="float32",
            pin=False,
            trust_remote_code=True,
        )


# ==========================================================================
# ticket 10: moe_lora on the fused routed experts
# ==========================================================================


def _moe_lora_config(weights: Path) -> Any:
    """The adapter the training setup builds for ``moe_lora: true`` on this fixture."""
    from peft import TaskType

    from soup_cli.config.schema import LoraConfig
    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton
    from soup_cli.utils.moe import resolve_moe_lora_targets
    from soup_cli.utils.peft_wiring import build_lora_config

    lora = LoraConfig(r=2, alpha=4, dropout=0.0)
    tcfg = SimpleNamespace(moe_lora=True, lora=lora)
    targets = resolve_moe_lora_targets(
        build_meta_skeleton(str(weights), dtype="float32"), tcfg, None
    )
    assert targets, "moe_lora resolved no targets on deepseek_v3"
    return build_lora_config(lora, target_modules=targets, task_type=TaskType.CAUSAL_LM)


@_REQUIRES_TRAINING_FLOOR
@pytest.mark.parametrize("experts", _EXPERT_IMPLEMENTATIONS)
@pytest.mark.parametrize("device", _DEVICES)
def test_streamed_deepseek_v3_moe_lora_matches_resident(
    tmp_path: Path, device: str, experts: str, deterministic: None
) -> None:
    """Parity arm 10: the adapters Soup builds for ``moe_lora: true``, on the routed experts."""
    parity = _run_parity(
        tmp_path, device=device, experts=experts, lora_config=_moe_lora_config
    )

    expert_adapters = [name for name in parity.lora_names if ".mlp.experts." in name]
    # peft wraps `experts` once per fused tensor (gate_up_proj, down_proj), so each
    # MoE layer has an adapter at both wrapper levels.
    for idx in _moe_layers(parity.config):
        prefix = f"base_model.model.model.layers.{idx}.mlp.experts."
        for level in ("", "base_layer."):
            for factor in ("lora_A", "lora_B"):
                assert f"{prefix}{level}{factor}.default.weight" in expert_adapters
    for name in expert_adapters:
        grad = parity.streamed.grads[name]
        assert grad is not None and bool(grad.ne(0).any()), f"{name}: no or zero gradient"
    _assert_parity(parity)


def _skeleton_and_layout(weights: Path) -> Any:
    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton, shard_layout_of

    skeleton = build_meta_skeleton(str(weights), dtype="float32")
    return skeleton, shard_layout_of(skeleton)


def _explicit_expert_targets_config(**fields: Any) -> Any:
    """Adapters on ``gate_proj``, ``up_proj`` and ``down_proj`` by name, no ``moe_lora``."""
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        target_modules=["gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        **fields,
    )


@pytest.fixture
def v3_weights(tmp_path: Path) -> Path:
    weights = tmp_path / "weights"
    _save_fixture(weights, _tiny_deepseek_v3_config())
    return weights


@_REQUIRES_TRAINING_FLOOR
@pytest.mark.parametrize(
    "make_config",
    [lambda weights: _explicit_expert_targets_config(), _moe_lora_config],
    ids=["explicit-targets", "moe_lora"],
)
def test_fused_expert_targets_the_shards_carry_are_accepted(
    v3_weights: Path, make_config: Callable[[Path], Any]
) -> None:
    from soup_cli.utils.layer_stream_runtime import refuse_unservable_target_parameters

    skeleton, layout = _skeleton_and_layout(v3_weights)

    refuse_unservable_target_parameters(make_config(v3_weights), skeleton, layout)


@_REQUIRES_TRAINING_FLOOR
def test_explicit_expert_targets_are_refused_when_the_shards_do_not_carry_fused_experts(
    v3_weights: Path,
) -> None:
    """No ``moe_lora`` here: the check reads the config peft converts, so naming
    ``gate_proj``/``up_proj``/``down_proj`` is enough to trip it."""
    from dataclasses import replace

    from soup_cli.utils.layer_stream_runtime import refuse_unservable_target_parameters

    skeleton, layout = _skeleton_and_layout(v3_weights)
    config = _explicit_expert_targets_config()

    with pytest.raises(ValueError) as excinfo:
        refuse_unservable_target_parameters(config, skeleton, replace(layout, stack_experts=False))

    message = str(excinfo.value)
    assert "layer streaming cannot serve" in message
    assert "gate_up_proj: the shards do not carry fused experts" in message
    assert "down_proj: the shards do not carry fused experts" in message
    assert config.target_modules == {"gate_proj", "up_proj", "down_proj"}, "the caller's config"


@_REQUIRES_TRAINING_FLOOR
def test_a_parameter_target_outside_the_decoder_layers_is_refused(v3_weights: Path) -> None:
    from soup_cli.utils.layer_stream_runtime import refuse_unservable_target_parameters

    skeleton, layout = _skeleton_and_layout(v3_weights)
    config = _explicit_expert_targets_config(target_parameters=["lm_head.weight"])
    config.target_modules = ["q_a_proj"]

    with pytest.raises(ValueError, match="lm_head.weight"):
        refuse_unservable_target_parameters(config, skeleton, layout)


def test_a_skeleton_without_a_model_type_is_left_to_peft() -> None:
    from soup_cli.utils.layer_stream_runtime import refuse_unservable_target_parameters

    refuse_unservable_target_parameters(
        SimpleNamespace(target_modules=["gate_proj"]), SimpleNamespace(), None
    )


def test_the_training_setup_checks_the_lora_config_it_built(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The adapter plan passes the config, the skeleton and the layout to the refusal."""
    from tests.test_qwen35_streaming import TestQwen35StreamingSetup

    calls: List[Any] = []
    monkeypatch.setattr(
        "soup_cli.utils.layer_stream_runtime.refuse_unservable_target_parameters",
        lambda lora_config, model, layout: calls.append((lora_config, model, layout)),
    )

    TestQwen35StreamingSetup().test_stream_setup_threads_moe_targets_and_union_budget(
        monkeypatch, tmp_path
    )

    assert len(calls) == 1
    lora_config, model, _layout = calls[0]
    assert {"gate_proj", "up_proj", "down_proj"} <= set(lora_config.target_modules)
    assert hasattr(model, "config")
