
import pytest
import torch
from transformers import MistralConfig, MistralModel

from soup_cli.utils.frozen_prefix_cache import (
    FrozenPrefixCache,
    build_cache_metadata,
    cache_key,
    fingerprint_frozen_prefix,
)


def test_cache_save_load(tmp_path):
    cache = FrozenPrefixCache(tmp_path)

    activation = torch.randn(1, 4, 64)

    metadata = {
        "model": "tiny-mistral",
        "revision": "test-v1",
        "cutoff": 2,
        "sample_id": "sample-001",
    }

    cache.save(activation, metadata)
    loaded = cache.load(metadata)

    assert loaded is not None
    assert torch.equal(activation, loaded)


def test_cache_miss_on_changed_metadata(tmp_path):
    cache = FrozenPrefixCache(tmp_path)

    activation = torch.randn(1, 4, 64)

    original = {
        "model": "tiny-mistral",
        "revision": "test-v1",
        "cutoff": 2,
        "sample_id": "sample-001",
    }

    cache.save(activation, original)

    changed = dict(original)
    changed["revision"] = "test-v2"

    assert cache.load(changed) is None


def test_cache_key_is_deterministic():
    a = {"model": "mistral", "cutoff": 2}
    b = {"cutoff": 2, "model": "mistral"}

    assert cache_key(a) == cache_key(b)

def test_cache_invalidated_when_input_changes(tmp_path):
    cache = FrozenPrefixCache(tmp_path)

    activation = torch.randn(1, 4, 64)

    original = {
        "model": "tiny-mistral",
        "revision": "test-v1",
        "cutoff": 2,
        "sample_id": "sample-001",
        "input_ids": [1, 2, 3, 4],
    }

    cache.save(activation, original)

    changed = dict(original)
    changed["input_ids"] = [1, 2, 3, 5]

    assert cache.load(changed) is None




def test_automatic_cache_invalidation(tmp_path):
    cache = FrozenPrefixCache(tmp_path)

    inputs = torch.tensor([[1, 2, 3, 4]])
    changed_inputs = torch.tensor([[1, 2, 3, 5]])

    common = {
        "model_revision": "tiny-mistral-v1",
        "frozen_prefix_fingerprint": "weights-v1",
        "config_fingerprint": "config-v1",
        "cutoff": 2,
    }

    metadata = build_cache_metadata(
        **common,
        input_ids=inputs,
    )

    activation = torch.randn(1, 4, 64)
    cache.save(activation, metadata)

    assert torch.equal(cache.load(metadata), activation)

    changed_metadata = build_cache_metadata(
        **common,
        input_ids=changed_inputs,
    )

    assert cache.load(changed_metadata) is None

    changed_weights = build_cache_metadata(
        **{**common, "frozen_prefix_fingerprint": "weights-v2"},
        input_ids=inputs,
    )

    assert cache.load(changed_weights) is None

    changed_mask = build_cache_metadata(
        **common,
        input_ids=inputs,
        attention_mask=torch.tensor([[1, 1, 1, 0]]),
    )

    assert cache.load(changed_mask) is None


def test_atomic_cache_save(tmp_path):
    cache = FrozenPrefixCache(tmp_path)

    metadata = {
        "model": "tiny-mistral",
        "revision": "v1",
        "cutoff": 2,
    }

    activation = torch.randn(1, 4, 64)

    path = cache.save(activation, metadata)

    assert path.exists()
    assert torch.equal(cache.load(metadata), activation)

    temporary_files = list(tmp_path.glob(".e2-cache-*.tmp"))
    assert temporary_files == []

def test_corrupted_cache_is_rejected(tmp_path):
    cache = FrozenPrefixCache(tmp_path)

    metadata = {
        "model": "tiny-mistral",
        "revision": "v1",
        "cutoff": 2,
    }

    path = cache.save(torch.randn(1, 4, 64), metadata)

    # Simulate a corrupted cache file.
    path.write_bytes(b"corrupted-cache")

    with pytest.raises(Exception):
        cache.load(metadata)

def test_frozen_prefix_fingerprint_changes_with_weights():
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

    original = fingerprint_frozen_prefix(model, cutoff=2)

    # Repeated fingerprint must be stable.
    assert fingerprint_frozen_prefix(model, cutoff=2) == original

    # Upper trainable layers must not affect the fingerprint.
    with torch.no_grad():
        next(model.layers[3].parameters()).add_(0.01)

    assert fingerprint_frozen_prefix(model, cutoff=2) == original

    # Frozen-prefix changes must invalidate the fingerprint.
    with torch.no_grad():
        next(model.layers[0].parameters()).add_(0.01)

    assert fingerprint_frozen_prefix(model, cutoff=2) != original


def test_frozen_prefix_fingerprint_changes_with_embeddings():
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

    original = fingerprint_frozen_prefix(model, cutoff=2)

    with torch.no_grad():
        model.embed_tokens.weight[0, 0].add_(0.01)

    assert fingerprint_frozen_prefix(model, cutoff=2) != original


def test_cache_invalidated_when_attention_mask_changes(tmp_path):
    import torch

    from soup_cli.utils.frozen_prefix_cache import (
        FrozenPrefixCache,
        build_cache_metadata,
    )

    cache = FrozenPrefixCache(tmp_path)

    common = {
        "model_revision": "test-model-v1",
        "frozen_prefix_fingerprint": "frozen-weights-v1",
        "config_fingerprint": "config-v1",
        "cutoff": 2,
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "position_ids": torch.tensor([[0, 1, 2, 3]]),
    }

    original = build_cache_metadata(
        **common,
        attention_mask=torch.tensor([[1, 1, 1, 1]]),
    )

    changed = build_cache_metadata(
        **common,
        attention_mask=torch.tensor([[1, 1, 1, 0]]),
    )

    assert cache_key(original) != cache_key(changed)

    cache.save(torch.randn(1, 4, 8), original)

    assert cache.load(original) is not None
    assert cache.load(changed) is None


def test_cache_invalidated_when_position_ids_change(tmp_path):
    import torch

    from soup_cli.utils.frozen_prefix_cache import (
        FrozenPrefixCache,
        build_cache_metadata,
    )

    cache = FrozenPrefixCache(tmp_path)

    common = {
        "model_revision": "test-model-v1",
        "frozen_prefix_fingerprint": "frozen-weights-v1",
        "config_fingerprint": "config-v1",
        "cutoff": 2,
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
    }

    original = build_cache_metadata(
        **common,
        position_ids=torch.tensor([[0, 1, 2, 3]]),
    )

    changed = build_cache_metadata(
        **common,
        position_ids=torch.tensor([[4, 5, 6, 7]]),
    )

    assert cache_key(original) != cache_key(changed)

    cache.save(torch.randn(1, 4, 8), original)

    assert cache.load(original) is not None
    assert cache.load(changed) is None


def test_cache_persists_across_instances(tmp_path):
    import torch

    from soup_cli.utils.frozen_prefix_cache import (
        FrozenPrefixCache,
        build_cache_metadata,
    )

    metadata = build_cache_metadata(
        model_revision="test-model-v1",
        frozen_prefix_fingerprint="frozen-weights-v1",
        config_fingerprint="config-v1",
        cutoff=2,
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.tensor([[1, 1, 1]]),
        position_ids=torch.tensor([[0, 1, 2]]),
    )

    activation = torch.randn(1, 3, 8)

    first_cache = FrozenPrefixCache(tmp_path)
    first_cache.save(activation, metadata)

    second_cache = FrozenPrefixCache(tmp_path)
    restored = second_cache.load(metadata)

    assert restored is not None
    assert torch.equal(restored, activation)
