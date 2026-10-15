
from unittest.mock import patch

import torch
from transformers import MistralConfig, MistralForCausalLM

from soup_cli.utils.frozen_prefix_cache import (
    FrozenPrefixCache,
    build_cache_metadata,
)
from soup_cli.utils.frozen_prefix_forward import FrozenPrefixRunner


def test_runner_inside_causal_lm(tmp_path):
    torch.manual_seed(42)

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

    model = MistralForCausalLM(config)
    model.eval()
    model.requires_grad_(False)

    # Make upper layers trainable for this integration smoke test.
    for layer in model.model.layers[2:]:
        layer.requires_grad_(True)

    def metadata_factory(*, input_ids, attention_mask, position_ids):
        return build_cache_metadata(
            model_revision="tiny-mistral-v1",
            frozen_prefix_fingerprint="fixed-test-weights",
            config_fingerprint="test-config-v1",
            cutoff=2,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    runner = FrozenPrefixRunner(
        decoder=model.model,
        cache=FrozenPrefixCache(tmp_path / "cache"),
        cutoff=2,
        metadata_factory=metadata_factory,
    )

    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    labels = input_ids.clone()

    with patch.object(model.model, "forward", side_effect=runner.forward):
        first = model(
            input_ids=input_ids,
            labels=labels,
            use_cache=False,
        )
        first.loss.backward()

        model.zero_grad(set_to_none=True)

        second = model(
            input_ids=input_ids,
            labels=labels,
            use_cache=False,
        )
        second.loss.backward()

    assert runner.misses == 1
    assert runner.hits == 1

    torch.testing.assert_close(
        first.loss,
        second.loss,
        rtol=1e-5,
        atol=1e-6,
    )

    assert any(
        param.grad is not None
        for layer in model.model.layers[2:]
        for param in layer.parameters()
    )

