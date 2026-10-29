"""Tiny-model per-expert training and adapter reload."""

from pathlib import Path

import pytest

from tests.test_b2_deepseek_v3_stream import (
    _copy_lora,
    _lora,
    _lora_params,
    _randomise_lora_b,
    _save_fixture,
    _tiny_deepseek_v3_config,
)
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR


@_REQUIRES_TRAINING_FLOOR
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu(reason="CUDA parity"))]
)
def test_internal_per_expert_training_and_adapter_reload(
    tmp_path: Path, monkeypatch, device
) -> None:
    """Exercise validated per-expert installation without bypassing runtime checks."""
    import torch
    from peft import load_peft_weights, set_peft_model_state_dict
    from transformers import AutoModelForCausalLM

    from soup_cli.utils import layer_stream_runtime
    from soup_cli.utils.kimi_k2 import load_stream_config
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.peft_wiring import attach_adapter, load_adapter

    config = _tiny_deepseek_v3_config(n_routed_experts=4, n_group=2, topk_group=1)
    weights, shards = tmp_path / "weights", tmp_path / "shards"
    _save_fixture(weights, config)
    index = shard_checkpoint(
        str(weights), str(shards), dtype="float32", arch="deepseek_v3",
        expert_layout="per_expert",
    )
    assert index.expert_layers == (1, 2)
    lora = _lora(config)
    streamed, runtime = layer_stream_runtime.build_streamed_model(
        model_id=str(weights), shard_dir=str(shards), index=index, lora_config=lora,
        device=device, dtype="float32", buffers=2, pin=False, seed=71,
    )
    try:
        streamed.get_base_model().set_experts_implementation("eager")
        _randomise_lora_b(streamed)
        stream_config = load_stream_config(str(weights), trust_remote_code=False)
        resident_base = AutoModelForCausalLM.from_pretrained(
            str(weights), config=stream_config, dtype=torch.float32,
            experts_implementation="eager",
        ).to(device)
        resident = attach_adapter(resident_base, lora)
        _copy_lora(streamed, resident)
        tokens = torch.randint(0, config.vocab_size, (2, 16),
                               generator=torch.Generator().manual_seed(73)).to(device)
        batch = {"input_ids": tokens, "labels": tokens}
        outputs = []
        grads = []
        for model in (resident, streamed):
            model.train()
            model.zero_grad(set_to_none=True)
            result = model(**batch)
            fetches_after_forward = runtime.expert_pool.fetches if model is streamed else 0
            result.loss.backward()
            if model is streamed:
                assert fetches_after_forward > 0
                assert runtime.expert_pool.fetches > fetches_after_forward
            outputs.append(result.logits.detach().clone())
            grads.append({
                name: None if param.grad is None else param.grad.detach().clone()
                for name, param in _lora_params(model).items()
            })
        torch.testing.assert_close(outputs[1], outputs[0], rtol=1e-5, atol=1e-6)
        assert grads[0] and grads[0].keys() == grads[1].keys()
        for name in grads[0]:
            assert grads[0][name] is not None and grads[1][name] is not None, name
            assert torch.isfinite(grads[1][name]).all(), name
            assert grads[1][name].ne(0).any(), name
            torch.testing.assert_close(grads[1][name], grads[0][name], rtol=1e-5, atol=1e-6)
        before = {name: param.detach().clone() for name, param in _lora_params(streamed).items()}
        for model in (resident, streamed):
            torch.optim.SGD(_lora_params(model).values(), lr=0.05).step()
        updated = _lora_params(streamed)
        assert any(not torch.equal(before[name], param) for name, param in updated.items())
        for name, param in updated.items():
            torch.testing.assert_close(param, _lora_params(resident)[name], rtol=1e-5, atol=1e-6)

        adapter_dir = tmp_path / "adapter"
        streamed.save_pretrained(adapter_dir)
        layer_stream_runtime.assert_streamed_adapter_saved(streamed, str(adapter_dir))
        reloaded_base = AutoModelForCausalLM.from_pretrained(
            str(weights), dtype=torch.float32, experts_implementation="eager",
        ).to(device)
        reloaded = load_adapter(reloaded_base, str(adapter_dir))
        assert set(_lora_params(reloaded)) == set(updated)
        for name, param in _lora_params(reloaded).items():
            torch.testing.assert_close(param, updated[name], rtol=0, atol=0)
        streamed.eval()
        reloaded.eval()
        with torch.no_grad():
            torch.testing.assert_close(
                streamed(**batch).logits, reloaded(**batch).logits, rtol=1e-5, atol=1e-6
            )

        fresh_stream, fresh_runtime = layer_stream_runtime.build_streamed_model(
            model_id=str(weights), shard_dir=str(shards), index=index, lora_config=lora,
            device=device, dtype="float32", buffers=2, pin=False, seed=11,
        )
        try:
            fresh_stream.get_base_model().set_experts_implementation("eager")
            fresh_before = {
                name: param.detach().clone() for name, param in _lora_params(fresh_stream).items()
            }
            set_peft_model_state_dict(fresh_stream, load_peft_weights(str(adapter_dir)))
            fresh_params = _lora_params(fresh_stream)
            assert fresh_params.keys() == updated.keys()
            assert any(
                not torch.equal(fresh_before[name], param) for name, param in fresh_params.items()
            )
            for name, param in fresh_params.items():
                torch.testing.assert_close(param, updated[name], rtol=0, atol=0)
            fresh_stream.eval()
            with torch.no_grad():
                torch.testing.assert_close(
                    fresh_stream(**batch).logits, reloaded(**batch).logits,
                    rtol=1e-5, atol=1e-6,
                )
            assert fresh_runtime.expert_pool.fetches > 0
        finally:
            fresh_runtime.close()
    finally:
        runtime.close()
