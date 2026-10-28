"""Configuration and memory accounting for the experimental B6 setup."""

import pytest


def _config(**training):
    from soup_cli.config.schema import SoupConfig

    settings = {
        "stream_layers": True,
        "stream_source": "disk",
        "stream_expert_layout": "per_expert",
        "quantization": "none",
        "batch_size": 1,
        "lora": {"r": 4, "dropout": 0, "target_modules": ["q_proj"]},
    }
    settings.update(training)
    return SoupConfig.model_validate({
        "base": "example/model", "data": {"train": "train.jsonl", "chat_template": "chatml"},
        "training": settings,
    })


def test_expert_layout_is_opt_in_and_roundtrips():
    from soup_cli.config.schema import TrainingConfig

    assert TrainingConfig().stream_expert_layout == "fused"
    cfg = _config()
    assert cfg.training.stream_expert_layout == "per_expert"
    assert type(cfg).model_validate(cfg.model_dump()).training.stream_expert_layout == "per_expert"


@pytest.mark.parametrize("change, message", [
    ({"stream_layers": False}, "stream_layers is false"),
    ({"stream_source": "ram"}, "requires stream_source='disk'"),
    ({"moe_lora": True}, "cannot train routed-expert adapters"),
    ({"lora": {"r": 4, "dropout": 0, "target_parameters": ["experts.down_proj"]}},
     "target_parameters is not yet supported"),
    ({"l2l": True}, "does not support training.l2l"),
    ({"stream_expert_layout": "unknown"}, "per_expert"),
])
def test_unsupported_configurations_refused(change, message):
    with pytest.raises(ValueError, match=message):
        _config(**change)


def test_expert_memory_preflight_reads_real_cache_headers(tmp_path):
    from soup_cli.trainer.stream_setup import _expert_buffer_requirements
    from soup_cli.utils.expert_stream import plan_expert_buffer
    from tests.test_b1_expert_shard import _checkpoint, _reader, _shard

    source, cache = tmp_path / "source", tmp_path / "cache"
    _checkpoint(source)
    index = _shard(source, cache, expert_layout="per_expert")
    reader = _reader(cache, index)
    try:
        plan = plan_expert_buffer(reader, dtype=index.dtype)
        assert _expert_buffer_requirements(str(cache), index) == (
            plan.weight_bytes, plan.host_staging_bytes,
        )
    finally:
        reader.close()


def test_fused_cache_does_not_require_expert_arena():
    from types import SimpleNamespace

    from soup_cli.trainer.stream_setup import _expert_buffer_requirements

    assert _expert_buffer_requirements("unused", SimpleNamespace(expert_layout="")) == (0, 0)


def test_public_setup_refuses_before_tokenizer_or_weight_download(monkeypatch):
    from soup_cli.trainer.stream_setup import StreamingSetupMixin
    from soup_cli.utils import kimi_k2

    def unexpected(*args, **kwargs):
        raise AssertionError("setup attempted network or tokenization before refusing")

    monkeypatch.setattr(kimi_k2, "load_stream_tokenizer", unexpected)
    wrapper = StreamingSetupMixin()
    wrapper.device = "cpu"
    wrapper._trust_remote_code = False
    cfg = _config()
    with pytest.raises(ValueError, match="requires quantization='4bit'"):
        wrapper._setup_streaming_transformers(cfg, cfg.training)


def test_expert_arena_is_included_in_vram_refusal(monkeypatch):
    from types import SimpleNamespace

    import torch

    from soup_cli.trainer.stream_setup import StreamingSetupMixin
    from soup_cli.utils import layer_stream

    monkeypatch.setattr(layer_stream, "estimate_stream_peak_vram", lambda **kwargs: 80)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (100, 100))
    wrapper = StreamingSetupMixin()
    cfg = SimpleNamespace(data=SimpleNamespace(max_length=8))
    tcfg = SimpleNamespace(
        batch_size=1, stream_buffers=2, stream_vram_override=None, stream_vram_probe=False,
    )
    shape = SimpleNamespace(vocab_size=8, hidden_size=4, intermediate_size=8)
    with pytest.raises(ValueError, match="predicted to need"):
        wrapper._stream_budget_lines(
            cfg, tcfg, model_config=shape, layer_bytes=1, embed_bytes=1,
            index=SimpleNamespace(n_layers=1), on_cuda=True, expert_weight_bytes=21,
            adapter_budget=SimpleNamespace(stored_parameters=0, total=None),
        )
