"""B1.4 in the P3 sense: public launch of deepseek_v3 with training.l2l.

Two seams: the config schema (``SoupConfig.model_validate``) and the launch guard
(``refuse_unsupported_l2l``), which sees the model the schema cannot.
"""

import pytest


def _config(**training):
    from soup_cli.config.schema import SoupConfig

    settings = {
        "stream_layers": True,
        "stream_source": "disk",
        "stream_expert_layout": "per_expert",
        "l2l": True,
        "quantization": "none",
        "batch_size": 1,
        "lora": {"r": 4, "dropout": 0, "target_modules": ["q_proj"]},
    }
    settings.update(training)
    return SoupConfig.model_validate({
        "base": "example/model", "task": "sft",
        "data": {"train": "train.jsonl", "chat_template": "chatml"},
        "training": settings,
    })


class _ModelConfig:
    def __init__(self, model_type):
        self.model_type = model_type


def _guard(model_type, **kwargs):
    from soup_cli.trainer.l2l_trainer import refuse_unsupported_l2l

    kwargs.setdefault("n_gpu", 1)
    refuse_unsupported_l2l(_ModelConfig(model_type), **kwargs)


# --------------------------------------------------------------------------
# schema seam
# --------------------------------------------------------------------------
def test_l2l_with_per_expert_streaming_validates():
    cfg = _config()
    assert cfg.training.l2l is True
    assert cfg.training.stream_expert_layout == "per_expert"
    assert cfg.training.stream_source == "disk"


def test_l2l_with_per_expert_streaming_validates_with_4bit_quantization():
    cfg = _config(quantization="4bit")
    assert (cfg.training.l2l, cfg.training.quantization) == (True, "4bit")


def test_l2l_defaults_do_not_change():
    from soup_cli.config.schema import TrainingConfig

    tcfg = TrainingConfig()
    assert tcfg.l2l is False
    assert tcfg.stream_expert_layout == "fused"


@pytest.mark.parametrize("change, message", [
    ({"stream_layers": False}, "l2l requires training.stream_layers=true"),
    ({"stream_source": "ram"}, "requires stream_source='disk'"),
    ({"moe_lora": True}, "cannot train routed-expert adapters"),
])
def test_l2l_with_per_expert_keeps_the_other_refusals(change, message):
    with pytest.raises(ValueError, match=message):
        _config(**change)


def test_l2l_with_per_expert_still_requires_sft():
    from soup_cli.config.schema import SoupConfig

    with pytest.raises(ValueError, match="sft"):
        SoupConfig.model_validate({
            "base": "example/model", "task": "dpo",
            "data": {"train": "train.jsonl", "format": "dpo"},
            "training": {
                "stream_layers": True, "stream_source": "disk",
                "stream_expert_layout": "per_expert", "l2l": True,
                "quantization": "none", "batch_size": 1,
                "lora": {"r": 4, "dropout": 0},
            },
        })


# --------------------------------------------------------------------------
# launch guard seam
# --------------------------------------------------------------------------
def test_guard_admits_deepseek_v3_with_per_expert_layout():
    _guard("deepseek_v3", expert_layout="per_expert")


@pytest.mark.parametrize("layout", ["fused", None])
def test_guard_refuses_deepseek_v3_without_per_expert_layout(layout):
    kwargs = {} if layout is None else {"expert_layout": layout}
    with pytest.raises(ValueError, match="deepseek_v3.*stream_expert_layout='per_expert'"):
        _guard("deepseek_v3", **kwargs)


def test_guard_still_refuses_other_architectures_with_per_expert_layout():
    with pytest.raises(ValueError, match=r"qwen2.*llama.*mistral.*deepseek_v3"):
        _guard("qwen2", expert_layout="per_expert")


@pytest.mark.parametrize("arch", ["llama", "mistral"])
@pytest.mark.parametrize("layout", ["fused", "per_expert"])
def test_guard_admits_llama_and_mistral_regardless_of_layout(arch, layout):
    _guard(arch, expert_layout=layout)


def test_guard_keeps_gpu_and_fp16_refusals_for_deepseek_v3():
    with pytest.raises(ValueError, match="2 GPU"):
        _guard("deepseek_v3", expert_layout="per_expert", n_gpu=2)
    with pytest.raises(ValueError, match="fp16"):
        _guard("deepseek_v3", expert_layout="per_expert", fp16=True)
