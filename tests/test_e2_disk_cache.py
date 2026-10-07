
import torch

from soup_cli.utils.frozen_prefix_cache import (
    FrozenPrefixCache,
    cache_key,
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
