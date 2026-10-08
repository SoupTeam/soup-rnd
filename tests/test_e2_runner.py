
import torch
from transformers import MistralConfig, MistralModel

from soup_cli.utils.frozen_prefix_cache import (
    FrozenPrefixCache,
    build_cache_metadata,
)
from soup_cli.utils.frozen_prefix_forward import FrozenPrefixRunner


def test_runner_cache_hit_and_equivalence(tmp_path):
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

    model = MistralModel(config)
    model.eval()
    model.requires_grad_(False)

    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    def metadata_factory(
        *,
        input_ids,
        attention_mask,
        position_ids,
    ):
        return build_cache_metadata(
            model_revision="tiny-mistral-v1",
            frozen_prefix_fingerprint="test-fixed-weights",
            config_fingerprint="test-config-v1",
            cutoff=2,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    runner = FrozenPrefixRunner(
        decoder=model,
        cache=FrozenPrefixCache(tmp_path / "cache"),
        cutoff=2,
        metadata_factory=metadata_factory,
    )

    frozen_calls = [0]

    def count_calls(module, inputs):
        frozen_calls[0] += 1

    handles = [
        layer.register_forward_pre_hook(count_calls)
        for layer in model.layers[:2]
    ]

    try:
        with torch.no_grad():
            expected = model(
                input_ids=input_ids,
                use_cache=False,
            ).last_hidden_state

        # Exclude the reference forward from call counting.
        frozen_calls[0] = 0

        with torch.no_grad():
            first = runner.forward(input_ids=input_ids)
            second = runner.forward(input_ids=input_ids)
    finally:
        for handle in handles:
            handle.remove()

    assert runner.misses == 1
    assert runner.hits == 1
    assert frozen_calls[0] == 2

    torch.testing.assert_close(
        expected,
        first.last_hidden_state,
        rtol=0,
        atol=0,
    )

    torch.testing.assert_close(
        expected,
        second.last_hidden_state,
        rtol=0,
        atol=0,
    )

