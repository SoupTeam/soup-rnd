
import re

import pytest
from peft import get_peft_model
from transformers import MistralConfig, MistralForCausalLM

from soup_cli.config.schema import LoraConfig
from soup_cli.utils.peft_wiring import (
    build_lora_config,
    resolve_top_k_layers,
)


def make_model():
    """Create a tiny Mistral model without downloading weights."""
    config = MistralConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
    )
    return MistralForCausalLM(config)


@pytest.mark.parametrize(
    "k,expected",
    [
        (1, [3]),
        (2, [2, 3]),
        (3, [1, 2, 3]),
        (4, [0, 1, 2, 3]),
    ],
)
def test_top_k_lora_attachment(k, expected):
    model = make_model()

    cfg = LoraConfig(
        r=4,
        alpha=8,
        dropout=0.0,
        top_k_layers=k,
        target_modules=["q_proj", "v_proj"],
    )

    selected = resolve_top_k_layers(model, k)
    assert selected == expected

    peft_cfg = build_lora_config(
        cfg,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
        layers_to_transform=selected,
    )

    model = get_peft_model(model, peft_cfg)

    trainable = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad
    ]

    assert trainable, "No trainable parameters found"

    active_layers = set()

    for name in trainable:
        match = re.search(r"\.layers\.(\d+)\.", name)
        assert match is not None, name

        layer = int(match.group(1))
        assert layer in selected, name
        assert "lora_" in name, name

        active_layers.add(layer)

    assert active_layers == set(expected)


@pytest.mark.parametrize("k", [0, -1, 5])
def test_invalid_top_k(k):
    model = make_model()

    with pytest.raises(ValueError):
        resolve_top_k_layers(model, k)


def test_default_lora_config():
    cfg = LoraConfig()

    assert cfg.top_k_layers is None

