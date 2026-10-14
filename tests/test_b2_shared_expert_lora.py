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


# ==========================================================================
# differing controls: the checks above can fail
# ==========================================================================


def _nudge_one_shared_expert_adapter(streamed: Any) -> None:
    from tests.test_b2_deepseek_v3_stream import _lora_params

    named = [n for n in sorted(_lora_params(streamed)) if "shared_experts.up_proj.lora_B" in n]
    assert named
    with torch.no_grad():
        _lora_params(streamed)[named[0]].add_(0.1)


@_REQUIRES_TRAINING_FLOOR
@pytest.mark.usefixtures("deterministic")
@pytest.mark.parametrize("device", _DEVICES)
def test_control_a_perturbed_shared_expert_adapter_changes_logits_and_gradients(
    tmp_path: Any, device: str
) -> None:
    parity = _run_parity(
        tmp_path,
        device=device,
        experts="eager",
        lora_config=lambda weights: _config(_SHARED_TARGETS),
        edit_streamed=_nudge_one_shared_expert_adapter,
    )

    assert not parity.forward_equal(), parity.report()
    assert parity.backward_mismatches(), parity.report()


def test_control_plain_peft_gives_different_logits_for_the_same_config() -> None:
    """peft's default path adapts the routed experts, so the model it builds is another model."""
    from peft import get_peft_model

    from soup_cli.utils.peft_wiring import attach_adapter

    torch.manual_seed(15)
    ids = torch.randint(0, 64, (2, 16))
    logits = []
    for attach in (get_peft_model, attach_adapter):
        model = _tiny_model()
        peft_model = attach(model, _config(_SHARED_TARGETS))
        gen = torch.Generator().manual_seed(5)
        with torch.no_grad():
            for name, param in peft_model.named_parameters():
                if "lora_B" in name:
                    param.copy_(torch.randn(param.shape, generator=gen) * 0.2)
        peft_model.eval()
        with torch.no_grad():
            logits.append(peft_model(input_ids=ids).logits)

    assert not torch.allclose(logits[0], logits[1])


# ==========================================================================
# save and reload
# ==========================================================================


def _randomise_adapter(peft_model: Any) -> None:
    gen = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_" in name:
                param.copy_(torch.randn(param.shape, generator=gen) * 0.2)


def _logits(model: Any) -> Any:
    model.eval()
    with torch.no_grad():
        ids = torch.randint(0, 64, (2, 16), generator=torch.Generator().manual_seed(3))
        return model(input_ids=ids).logits


def test_a_saved_shared_expert_adapter_reloads_onto_the_shared_experts(tmp_path: Any) -> None:
    from soup_cli.utils.peft_wiring import attach_adapter, load_adapter

    model = _tiny_model()
    trained = attach_adapter(model, _config(_SHARED_TARGETS))
    _randomise_adapter(trained)
    expected = _logits(trained)
    trained.save_pretrained(tmp_path / "adapter")

    reloaded = load_adapter(_tiny_model(), str(tmp_path / "adapter"))

    assert _adapter_modules(reloaded) == _expected_shared_modules(model.config)
    assert torch.equal(_logits(reloaded), expected)


def test_an_existing_routed_expert_adapter_loads_the_way_peft_loads_it(tmp_path: Any) -> None:
    """A v4-style ``gate_proj`` / ``up_proj`` / ``down_proj`` adapter is still peft's to convert."""
    from peft import PeftModel, get_peft_model

    from soup_cli.utils.peft_wiring import load_adapter

    trained = get_peft_model(_tiny_model(), _config(["gate_proj", "up_proj", "down_proj"]))
    _randomise_adapter(trained)
    trained.save_pretrained(tmp_path / "adapter")

    ours = load_adapter(_tiny_model(), str(tmp_path / "adapter"))
    theirs = PeftModel.from_pretrained(_tiny_model(), str(tmp_path / "adapter"))

    assert _adapter_modules(ours) == _adapter_modules(theirs)
    assert torch.equal(_logits(ours), _logits(theirs))


def test_the_peft_conversion_is_restored_after_a_reload(tmp_path: Any) -> None:
    from peft.utils import transformers_weight_conversion as conversion

    from soup_cli.utils.peft_wiring import attach_adapter, load_adapter

    attach_adapter(_tiny_model(), _config(_SHARED_TARGETS)).save_pretrained(tmp_path / "adapter")
    before = conversion.convert_peft_config_for_transformers

    load_adapter(_tiny_model(), str(tmp_path / "adapter"))

    assert conversion.convert_peft_config_for_transformers is before


# ==========================================================================
# the adapter preflight sizes the adapter the build will attach
# ==========================================================================


def test_the_adapter_plan_counts_the_shared_expert_adapters_not_the_routed_ones(
    tmp_path: Any,
) -> None:
    from types import SimpleNamespace

    from soup_cli.utils.stream_adapter_preflight import build_stream_adapter_plan

    config = _tiny_deepseek_v3_config()
    config.save_pretrained(tmp_path)
    tcfg = SimpleNamespace(
        optimizer="adamw_torch",
        moe_lora=False,
        use_lorafa=False,
        lora=SimpleNamespace(
            r=4,
            alpha=8,
            dropout=0,
            target_modules=["shared_experts.gate_proj", "shared_experts.up_proj"],
            use_dora=False,
            use_rslora=False,
            rank_pattern=None,
            alpha_pattern=None,
            init_strategy="random",
        ),
        quantization="none",
        double_quant_on=True,
        stream_vram_override=None,
    )

    plan = build_stream_adapter_plan(
        str(tmp_path), config, tcfg, dtype="float32", quant="none",
        double_quant=True, trust_remote_code=False, on_cuda=False,
    )

    owners = {t.name.rsplit(".lora_", 1)[0].removeprefix("base_model.model.") for t in plan.tensors}
    assert owners == {
        f"model.layers.{idx}.mlp.shared_experts.{proj}_proj"
        for idx in _moe_layers(config)
        for proj in ("gate", "up")
    }


def test_a_regex_naming_attention_and_shared_experts_keeps_both() -> None:
    """The adapter-budget recommendation builds this form; it must not lose the attention half."""
    from soup_cli.utils.peft_wiring import attach_adapter

    model = _tiny_model()
    regex = r".*\.(?:self_attn\.(?:q_a_proj|q_b_proj)|mlp\.shared_experts\.(?:gate|up|down)_proj)"

    peft_model = attach_adapter(model, _config(regex))

    attention = {
        f"model.layers.{idx}.self_attn.{name}"
        for idx in range(model.config.num_hidden_layers)
        for name in ("q_a_proj", "q_b_proj")
    }
    assert _adapter_modules(peft_model) == attention | _expected_shared_modules(model.config)
