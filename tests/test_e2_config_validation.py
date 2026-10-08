
import pytest

from soup_cli.config.schema import SoupConfig


def make_config(**training_overrides):
    training = {
        "frozen_prefix_cache": True,
        "gradient_checkpointing": False,
        "stream_layers": False,
        "packing": False,
        "multipack": False,
        "lora": {
            "r": 8,
            "dropout": 0.0,
            "top_k_layers": 2,
        },
    }
    training.update(training_overrides)

    return {
        "task": "sft",
        "backend": "transformers",
        "base": "mistralai/Mistral-7B-v0.1",
        "training": training,
        "data": {"train": "dummy.jsonl"},
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"stream_layers": True},
        {"packing": True},
        {"multipack": True},
        {"gradient_checkpointing": True},
        {"lora": {"r": 8, "dropout": 0.1, "top_k_layers": 2}},
        {"lora": {"r": 8, "dropout": 0.0}},
    ],
)
def test_invalid_e2_config(overrides):
    with pytest.raises(ValueError):
        SoupConfig.model_validate(make_config(**overrides))


def test_e2_disabled_by_default():
    config = SoupConfig.model_validate(
        make_config(frozen_prefix_cache=False)
    )
    assert config.training.frozen_prefix_cache is False

def test_valid_e2_config():
    config = SoupConfig.model_validate(make_config())

    assert config.task == "sft"
    assert config.backend == "transformers"
    assert config.training.frozen_prefix_cache is True
    assert config.training.lora.top_k_layers == 2
    assert config.training.lora.dropout == 0.0
