"""Compatibility must fail before downloads, and loaded adapters must execute."""

from types import SimpleNamespace

import pytest

from tests.test_liger_streaming import _config, _install_cpu_substitutes, _nonuniform_norm_base


@pytest.mark.parametrize(
    "updates,components,match",
    [
        ({"model_type": "qwen2"}, ["rms_norm"], "Llama"),
        ({"pretraining_tp": 2}, ["rms_norm"], "pretraining_tp"),
        ({"hidden_act": "relu"}, ["swiglu"], "SiLU"),
    ],
)
def test_unsupported_model_is_refused_before_tokenizer_or_weights(
    tmp_path, monkeypatch, updates, components, match
):
    from transformers import AutoConfig, AutoTokenizer

    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils import spectrum_scan

    model_config = SimpleNamespace(model_type="llama", pretraining_tp=1, hidden_act="silu")
    for name, value in updates.items():
        setattr(model_config, name, value)
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *_args, **_kwargs: model_config)

    def unexpected_download(*_args, **_kwargs):
        pytest.fail("unsupported streamed Liger model reached a tokenizer/weight download")

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", unexpected_download)
    monkeypatch.setattr(spectrum_scan, "resolve_model_weights", unexpected_download)
    wrapper = SFTTrainerWrapper(
        _config("local-base", tmp_path / "output", enabled=True, components=components),
        device="cpu",
    )
    with pytest.raises(ValueError, match=match):
        wrapper.setup({"train": []})


def test_rmsnorm_selection_does_not_require_swiglu_activation():
    from soup_cli.utils.liger_streaming import validate_streaming_liger_model_config

    config = SimpleNamespace(model_type="llama", pretraining_tp=1, hidden_act="relu")
    assert validate_streaming_liger_model_config(config, ["rms_norm"]) == ("rms_norm",)
    with pytest.raises(ValueError, match="SiLU"):
        validate_streaming_liger_model_config(config, ["swiglu"])


def test_checkpointing_false_and_peft_reload_execute_identical_predictions(tmp_path, monkeypatch):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils import layer_shard

    _install_cpu_substitutes(monkeypatch)
    torch.set_num_threads(1)
    base = _nonuniform_norm_base(tmp_path)
    monkeypatch.setattr(layer_shard, "resolve_shard_dir", lambda _base: str(tmp_path / "shards"))
    output = tmp_path / "adapter"
    payload = _config(base, output, enabled=True, components=["rms_norm", "swiglu"]).model_dump()
    payload["training"]["gradient_checkpointing"] = False
    config = SoupConfig.model_validate(payload)
    rows = [
        {
            "messages": [
                {"role": "user", "content": "hello world"},
                {"role": "assistant", "content": "the cat sat"},
            ]
        }
    ] * 4
    wrapper = SFTTrainerWrapper(config, device="cpu")
    try:
        wrapper.setup({"train": rows})
        assert all(layer.use_checkpoint for layer in wrapper.model.get_base_model().model.layers)
        assert not wrapper.trainer.args.gradient_checkpointing
        wrapper.trainer.args.disable_tqdm = True
        # Keep the runtime open for a final streamed prediction; the other
        # integration matrix exercises Soup's public train-and-close method.
        wrapper.trainer.train()
        wrapper.trainer.save_model(str(output))
        wrapper._assert_streamed_adapter_saved(str(output))
        wrapper.model.eval()
        tokens = torch.tensor([[4, 5, 6, 7, 8]])
        with torch.no_grad():
            streamed_logits = wrapper.model(tokens, use_cache=False).logits
        loaded = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base), output)
        loaded.eval()
        with torch.no_grad():
            reloaded_logits = loaded(tokens, use_cache=False).logits
        assert torch.isfinite(streamed_logits).all() and torch.isfinite(reloaded_logits).all()
        torch.testing.assert_close(streamed_logits, reloaded_logits, rtol=1e-5, atol=1e-6)
    finally:
        wrapper._close_stream_runtime()
