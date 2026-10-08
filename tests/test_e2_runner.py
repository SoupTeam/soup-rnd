import copy

import pytest
import torch
from transformers import MistralConfig, MistralModel

from soup_cli.utils.frozen_prefix_cache import (
    FrozenPrefixCache,
    build_cache_metadata,
)
from soup_cli.utils.frozen_prefix_forward import (
    FrozenPrefixRunner,
    install_frozen_prefix_cache,
)


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


def test_cache_invalidated_for_different_frozen_weights(tmp_path):
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

    def metadata_factory(*, input_ids, attention_mask, position_ids):
        return build_cache_metadata(
            model_revision="tiny-mistral-v1",
            frozen_prefix_fingerprint="placeholder",
            config_fingerprint="test-config-v1",
            cutoff=2,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    cache = FrozenPrefixCache(tmp_path / "cache")
    inputs = torch.tensor([[1, 2, 3, 4, 5]])

    first_runner = FrozenPrefixRunner(
        decoder=model,
        cache=cache,
        cutoff=2,
        metadata_factory=metadata_factory,
    )

    with torch.no_grad():
        first_runner.forward(input_ids=inputs)

    assert first_runner.misses == 1

    modified_model = copy.deepcopy(model)

    with torch.no_grad():
        next(modified_model.layers[0].parameters()).add_(0.01)

    second_runner = FrozenPrefixRunner(
        decoder=modified_model,
        cache=cache,
        cutoff=2,
        metadata_factory=metadata_factory,
    )

    with torch.no_grad():
        second_runner.forward(input_ids=inputs)

    assert second_runner.misses == 1
    assert second_runner.hits == 0

    assert (
        first_runner.frozen_prefix_fingerprint
        != second_runner.frozen_prefix_fingerprint
    )


def test_install_frozen_prefix_cache_restores_forward(tmp_path):
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

    # Freeze the complete model for this forward-only test.
    model.requires_grad_(False)

    original_forward = model.forward.__func__

    def metadata_factory(*, input_ids, attention_mask, position_ids):
        return build_cache_metadata(
            model_revision="tiny-mistral-v1",
            frozen_prefix_fingerprint="placeholder",
            config_fingerprint="test-config-v1",
            cutoff=2,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    cache = FrozenPrefixCache(tmp_path / "cache")
    inputs = torch.tensor([[1, 2, 3, 4, 5]])

    with install_frozen_prefix_cache(
        model=model,
        cache=cache,
        cutoff=2,
        metadata_factory=metadata_factory,
    ) as runner:
        assert model.forward.__func__ is not original_forward

        with torch.no_grad():
            model(input_ids=inputs, use_cache=False)
            model(input_ids=inputs, use_cache=False)

        assert runner.misses == 1
        assert runner.hits == 1

    # Original forward must be restored.
    assert model.forward.__func__ is original_forward

    # Also restore after an exception.
    with pytest.raises(RuntimeError, match="test failure"):
        with install_frozen_prefix_cache(
            model=model,
            cache=cache,
            cutoff=2,
            metadata_factory=metadata_factory,
        ):
            raise RuntimeError("test failure")

    assert model.forward.__func__ is original_forward
