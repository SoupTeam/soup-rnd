"""A5 ticket 01: ordinary recompute and profiled activation retention with L2L."""

from __future__ import annotations

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string


def _yaml(**training):
    fields = {
        "batch_size": 1,
        "quantization": "none",
        "stream_layers": True,
        "gradient_accumulation_steps": 2,
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0},
    }
    fields.update(training)
    return yaml.safe_dump(
        {
            "base": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "task": "sft",
            "data": {"train": "data.jsonl"},
            "training": fields,
        }
    )


@pytest.mark.parametrize("l2l", [True, False])
def test_ordinary_recompute_remains_available(l2l):
    config = load_config_from_string(_yaml(l2l=l2l, stream_recompute="on"))
    assert config.training.stream_recompute == "on"
    assert config.training.l2l is l2l


@pytest.mark.parametrize("l2l", [True, False])
def test_unconditional_retention_is_refused(l2l):
    with pytest.raises(ValueError, match="stream_recompute: Input should be 'on' or 'auto'"):
        load_config_from_string(_yaml(l2l=l2l, stream_recompute="off"))


@pytest.mark.parametrize("l2l", [True, False])
def test_automatic_retention_requires_a_profile(l2l):
    with pytest.raises(ValueError, match=r"require training\.stream_cache_profile"):
        load_config_from_string(_yaml(l2l=l2l, stream_recompute="auto"))


def test_profiled_retention_is_refused_with_l2l():
    with pytest.raises(ValueError, match="no L2L"):
        load_config_from_string(
            _yaml(
                l2l=True,
                stream_source="disk",
                stream_recompute="auto",
                stream_cache_profile="cache.json",
            )
        )
