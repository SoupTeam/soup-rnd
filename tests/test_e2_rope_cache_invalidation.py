"""Persistent E2 cache must not survive a change in effective RoPE settings."""

import copy

import torch
from transformers import MistralConfig, MistralModel

from soup_cli.utils.frozen_prefix_cache import (
    FrozenPrefixCache,
    build_cache_metadata,
    fingerprint_frozen_prefix,
    fingerprint_prefix_configuration,
)
from soup_cli.utils.frozen_prefix_forward import FrozenPrefixRunner


def test_changed_rope_theta_misses_persistent_cache_and_matches_uncached(tmp_path):
    torch.manual_seed(42)
    common = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        attention_dropout=0.0,
        sliding_window=None,
    )
    config_a = MistralConfig(**common, rope_theta=10000.0)
    config_b = MistralConfig(**common, rope_theta=1000.0)

    first = MistralModel(config_a).eval().requires_grad_(False)
    second = MistralModel(config_b).eval().requires_grad_(False)
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    # If a Transformers version persists inv_freq, loading the checkpoint
    # may copy the previous RoPE state. Rebuild it from the new config.
    second.rotary_emb = type(second.rotary_emb)(config_b)
    second.eval().requires_grad_(False)

    cutoff = 2
    # Every model parameter (embedding and decoder weights) is unchanged.
    second_parameters = dict(second.named_parameters())
    for name, value in first.named_parameters():
        if not name.startswith("rotary_emb."):
            assert torch.equal(value, second_parameters[name]), name
    assert fingerprint_prefix_configuration(first, cutoff) != fingerprint_prefix_configuration(
        second, cutoff
    )
    assert fingerprint_frozen_prefix(first, cutoff) != fingerprint_frozen_prefix(
        second, cutoff
    )

    inputs = torch.tensor([list(range(1, 25))], dtype=torch.long)
    positions = torch.arange(inputs.shape[1]).unsqueeze(0)
    mask = torch.ones_like(inputs)
    cache_directory = tmp_path / "shared-persistent-e2-cache"

    def run_model(model):
        config_fp = fingerprint_prefix_configuration(model, cutoff)

        def metadata_factory(*, input_ids, attention_mask, position_ids):
            return build_cache_metadata(
                model_revision="same-model-path-same-revision",
                frozen_prefix_fingerprint="runner-computed",
                config_fingerprint=config_fp,
                cutoff=cutoff,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        runner = FrozenPrefixRunner(
            decoder=model,
            cache=FrozenPrefixCache(cache_directory),
            cutoff=cutoff,
            metadata_factory=metadata_factory,
        )
        with torch.no_grad():
            reference = model(
                input_ids=inputs,
                attention_mask=mask,
                position_ids=positions,
                use_cache=False,
            ).last_hidden_state
            first_cached = runner.forward(
                input_ids=inputs,
                attention_mask=mask,
                position_ids=positions,
            ).last_hidden_state
            warm_cached = runner.forward(
                input_ids=inputs,
                attention_mask=mask,
                position_ids=positions,
            ).last_hidden_state
        assert runner.misses == 1
        assert runner.hits == 1
        torch.testing.assert_close(first_cached, reference, rtol=0, atol=0)
        torch.testing.assert_close(warm_cached, reference, rtol=0, atol=0)
        return reference

    old_reference = run_model(first)
    new_reference = run_model(second)

    assert not torch.equal(old_reference, new_reference), (
        "RoPE negative control failed: different theta did not change the output"
    )
    # The first model's activation cannot satisfy the second model's cold miss.
    assert len(list(cache_directory.glob("*.safetensors"))) == 2


def test_rope_scaling_configuration_changes_fingerprint():
    config = MistralConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
    )
    model = MistralModel(config).eval().requires_grad_(False)
    old = fingerprint_prefix_configuration(model, 2)
    # A new RoPE scaling setup must never hit the old persistent cache key.
    # This checks config fingerprinting only, not scaled forward semantics.
    model.config.rope_scaling = {"rope_type": "linear", "factor": 2.0}
    assert fingerprint_prefix_configuration(model, 2) != old
