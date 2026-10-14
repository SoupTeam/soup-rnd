"""B2 ticket 15: LoRA on the shared experts of DeepSeek-V3 attaches to the shared experts.

peft 0.21.2 rewrites ``gate_proj`` / ``up_proj`` / ``down_proj`` targets on a
``deepseek_v3`` model into ``target_parameters`` on the fused routed experts, so a
config that names the shared experts trains the routed ones without an error.
"""

from __future__ import annotations

import re
from typing import Any, Set

import pytest
import torch

from tests.test_b2_deepseek_v3_stream import (
    _DEVICES,
    _EXPERT_IMPLEMENTATIONS,
    _moe_layers,
    _run_parity,
    _tiny_deepseek_v3_config,
    deterministic,  # noqa: F401  (pytest fixture)
)
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR

_SHARED_TARGETS = r".*mlp\.shared_experts\.(gate|up|down)_proj"


def _tiny_model() -> Any:
    from transformers import AutoModelForCausalLM

    torch.manual_seed(15)
    return AutoModelForCausalLM.from_config(_tiny_deepseek_v3_config(), dtype=torch.float32)


def _adapter_modules(model: Any) -> Set[str]:
    """Names of the modules (or parameter owners) that carry a LoRA weight."""
    names = set()
    for name, _ in model.named_parameters():
        if "lora_" in name:
            owner = name.rsplit(".lora_", 1)[0]
            names.add(re.sub(r"\.base_layer$", "", owner).removeprefix("base_model.model."))
    return names


def _expected_shared_modules(config: Any) -> Set[str]:
    return {
        f"model.layers.{idx}.mlp.shared_experts.{proj}_proj"
        for idx in _moe_layers(config)
        for proj in ("gate", "up", "down")
    }


def _config(targets: Any, **kwargs: Any) -> Any:
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=4, lora_alpha=8, target_modules=targets, task_type=TaskType.CAUSAL_LM, **kwargs
    )


def test_plain_peft_moves_shared_expert_targets_onto_the_routed_experts() -> None:
    """The defect this ticket works around. If peft fixes it, this test says so."""
    from peft import get_peft_model

    model = _tiny_model()

    peft_model = get_peft_model(model, _config(_SHARED_TARGETS))

    assert _adapter_modules(peft_model) == {
        f"model.layers.{idx}.mlp.experts" for idx in _moe_layers(model.config)
    }


def test_shared_expert_targets_attach_to_exactly_the_shared_experts() -> None:
    from soup_cli.utils.peft_wiring import attach_adapter

    model = _tiny_model()

    peft_model = attach_adapter(model, _config(_SHARED_TARGETS))

    assert _adapter_modules(peft_model) == _expected_shared_modules(model.config)


def test_a_list_of_shared_expert_names_attaches_the_same_way() -> None:
    from soup_cli.utils.peft_wiring import attach_adapter

    model = _tiny_model()
    names = ["shared_experts.gate_proj", "shared_experts.up_proj", "shared_experts.down_proj"]

    peft_model = attach_adapter(model, _config(names))

    assert _adapter_modules(peft_model) == _expected_shared_modules(model.config)


def test_attention_and_shared_experts_attach_together_and_nothing_else() -> None:
    from soup_cli.utils.peft_wiring import attach_adapter

    model = _tiny_model()
    config = model.config
    targets = ["q_a_proj", "q_b_proj", "shared_experts.gate_proj"]

    peft_model = attach_adapter(model, _config(targets))

    attention = {
        f"model.layers.{idx}.self_attn.{name}"
        for idx in range(config.num_hidden_layers)
        for name in ("q_a_proj", "q_b_proj")
    }
    shared = {f"model.layers.{idx}.mlp.shared_experts.gate_proj" for idx in _moe_layers(config)}
    assert _adapter_modules(peft_model) == attention | shared


def test_a_bare_projection_name_beside_shared_experts_is_refused() -> None:
    """``down_proj`` beside a shared-expert target would still become routed-expert LoRA."""
    from soup_cli.utils.peft_wiring import attach_adapter

    model = _tiny_model()

    with pytest.raises(ValueError, match="routed experts"):
        attach_adapter(model, _config(["shared_experts.gate_proj", "down_proj"]))


def test_the_peft_conversion_is_restored_after_the_attach() -> None:
    from peft.utils import transformers_weight_conversion as conversion

    from soup_cli.utils.peft_wiring import attach_adapter

    before = conversion.convert_peft_config_for_transformers

    attach_adapter(_tiny_model(), _config(_SHARED_TARGETS))

    assert conversion.convert_peft_config_for_transformers is before


def test_a_config_with_no_shared_expert_target_goes_through_plain_peft() -> None:
    """``moe_lora`` still reaches the routed experts the way it does today."""
    from soup_cli.utils.peft_wiring import attach_adapter

    model = _tiny_model()

    peft_model = attach_adapter(model, _config(["q_a_proj", "q_b_proj"]))

    assert _adapter_modules(peft_model) == {
        f"model.layers.{idx}.self_attn.{name}"
        for idx in range(model.config.num_hidden_layers)
        for name in ("q_a_proj", "q_b_proj")
    }


# ==========================================================================
# the streaming attach: build_streamed_model
# ==========================================================================


def _streamed_adapter_modules(tmp_path: Any, lora_config: Any) -> Set[str]:
    """Shard the fixture, build the streamed model, return where its LoRA weights sit."""
    from transformers import AutoConfig

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream import stream_arch_of
    from soup_cli.utils.layer_stream_runtime import build_streamed_model
    from tests.test_b2_deepseek_v3_stream import _save_fixture

    weights = tmp_path / "weights"
    _save_fixture(weights, _tiny_deepseek_v3_config())
    index = shard_checkpoint(
        str(weights),
        str(tmp_path / "shards"),
        dtype="float32",
        arch=stream_arch_of(AutoConfig.from_pretrained(str(weights))),
    )
    streamed, runtime = build_streamed_model(
        model_id=str(weights),
        shard_dir=str(tmp_path / "shards"),
        index=index,
        lora_config=lora_config,
        device="cpu",
        dtype="float32",
        buffers=2,
        pin=False,
        seed=71,
    )
    try:
        return {name.replace(".inner.", ".") for name in _adapter_modules(streamed)}
    finally:
        runtime.close()


def test_streamed_model_attaches_shared_expert_lora_to_the_shared_experts(tmp_path: Any) -> None:
    modules = _streamed_adapter_modules(tmp_path, _config(_SHARED_TARGETS))

    assert modules == _expected_shared_modules(_tiny_deepseek_v3_config())


# ==========================================================================
# parity: the streamed model against the resident one
# ==========================================================================


def _assert_shared_parity(parity: Any) -> None:
    """Both sides adapted the shared experts only, every gradient is live and equal."""
    config = parity.config
    expected = {
        f"base_model.model.model.layers.{idx}.mlp.shared_experts.{proj}_proj.lora_{ab}.default.weight"
        for idx in _moe_layers(config)
        for proj in ("gate", "up", "down")
        for ab in ("A", "B")
    }
    assert set(parity.lora_names) == expected
    for name in parity.lora_names:
        for side in (parity.streamed, parity.resident):
            grad = side.grads[name]
            assert grad is not None and bool(grad.ne(0).any()), f"{name}: no or zero gradient"
    assert parity.forward_equal(), parity.report()
    assert not parity.backward_mismatches(), parity.report()


@_REQUIRES_TRAINING_FLOOR
@pytest.mark.usefixtures("deterministic")
@pytest.mark.parametrize("experts", _EXPERT_IMPLEMENTATIONS)
@pytest.mark.parametrize("device", _DEVICES)
def test_streamed_shared_expert_lora_matches_resident(
    tmp_path: Any, device: str, experts: str
) -> None:
    parity = _run_parity(
        tmp_path,
        device=device,
        experts=experts,
        lora_config=lambda weights: _config(_SHARED_TARGETS),
    )

    _assert_shared_parity(parity)


@_REQUIRES_TRAINING_FLOOR
@pytest.mark.usefixtures("deterministic")
@pytest.mark.parametrize("experts", _EXPERT_IMPLEMENTATIONS)
@pytest.mark.parametrize("device", _DEVICES)
def test_streamed_shared_expert_lora_from_the_disk_tier_matches_resident(
    tmp_path: Any, device: str, experts: str
) -> None:
    parity = _run_parity(
        tmp_path,
        device=device,
        experts=experts,
        tier="disk",
        lora_config=lambda weights: _config(_SHARED_TARGETS),
    )

    assert (parity.tier, parity.source) == ("disk", "AsyncDiskSource")
    _assert_shared_parity(parity)


@_REQUIRES_TRAINING_FLOOR
@pytest.mark.usefixtures("deterministic")
@pytest.mark.parametrize("experts", _EXPERT_IMPLEMENTATIONS)
@pytest.mark.parametrize("tier", ["ram", "disk"])
@pytest.mark.parametrize(
    "device", [pytest.param("cuda", marks=pytest.mark.gpu(reason="the 4-bit kernels are GPU-only"))]
)
def test_streamed_nf4_shared_expert_lora_matches_resident_nf4(
    tmp_path: Any, device: str, tier: str, experts: str
) -> None:
    """The shared experts are NF4 here, so the adapters sit on ``Linear4bit`` layers."""
    pytest.importorskip("bitsandbytes")

    parity = _run_parity(
        tmp_path,
        device=device,
        experts=experts,
        quant="nf4",
        tier=tier,
        lora_config=lambda weights: _config(_SHARED_TARGETS),
    )

    shared = {f"mlp.shared_experts.{name}_proj.weight" for name in ("gate", "up", "down")}
    for idx in _moe_layers(parity.config):
        assert shared <= parity.quantized[idx]
        assert shared <= parity.resident_quantized[idx]
    _assert_shared_parity(parity)
