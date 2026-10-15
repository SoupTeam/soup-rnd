"""Opt-in E2 cache timing: coverage and no effect on cache numerics."""

import torch
from transformers import MistralConfig, MistralModel

from soup_cli.bench.collector import BenchCollector
from soup_cli.utils.frozen_prefix_cache import (
    FrozenPrefixCache,
    build_cache_metadata,
    fingerprint_prefix_configuration,
)
from soup_cli.utils.frozen_prefix_forward import FrozenPrefixRunner


def _runner(tmp_path):
    torch.manual_seed(123)
    config = MistralConfig(
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
    decoder = MistralModel(config).eval().requires_grad_(False)
    cutoff = 2
    fingerprint = fingerprint_prefix_configuration(decoder, cutoff)

    def metadata_factory(*, input_ids, attention_mask, position_ids):
        return build_cache_metadata(
            model_revision="test-checkpoint",
            frozen_prefix_fingerprint="runner-computed",
            config_fingerprint=fingerprint,
            cutoff=cutoff,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    runner = FrozenPrefixRunner(
        decoder=decoder,
        cache=FrozenPrefixCache(tmp_path / "cache"),
        cutoff=cutoff,
        metadata_factory=metadata_factory,
    )
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return runner, input_ids, attention_mask


def test_cache_timing_is_opt_in(monkeypatch, tmp_path):
    monkeypatch.delenv("SOUP_E2_PROFILE_CACHE", raising=False)
    runner, input_ids, mask = _runner(tmp_path)
    assert runner.cache_timing_enabled is False
    assert runner.cache_timing_totals is None
    with torch.no_grad():
        first = runner.forward(input_ids=input_ids, attention_mask=mask)
        second = runner.forward(input_ids=input_ids, attention_mask=mask)
    assert runner.hits == 1
    assert runner.misses == 1
    assert torch.equal(first.last_hidden_state, second.last_hidden_state)


def test_cache_timing_tracks_cold_and_warm_epochs(monkeypatch, tmp_path):
    monkeypatch.setenv("SOUP_E2_PROFILE_CACHE", "1")
    runner, input_ids, mask = _runner(tmp_path)
    collector = BenchCollector(warmup_steps=0)
    collector.e2_runner_getter = lambda: runner

    assert runner.cache_timing_enabled is True
    collector.on_epoch_begin()
    with torch.no_grad():
        first = runner.forward(input_ids=input_ids, attention_mask=mask)
    collector.on_epoch_end()

    cold = collector.epoch_records[0]
    assert cold["cache_misses"] == 1
    assert cold["cache_hits"] == 0
    assert cold["cache_profile"]["prefix_compute_seconds"] > 0
    assert cold["cache_profile"]["disk_write_seconds"] > 0
    assert cold["cache_profile"]["cold_build_seconds"] > 0

    collector.on_epoch_begin()
    with torch.no_grad():
        second = runner.forward(input_ids=input_ids, attention_mask=mask)
    collector.on_epoch_end()
    warm = collector.epoch_records[1]
    assert warm["cache_misses"] == 0
    assert warm["cache_hits"] == 1
    assert warm["cache_profile"]["prefix_compute_seconds"] == 0
    assert warm["cache_profile"]["disk_write_seconds"] == 0
    assert warm["cache_profile"]["lookup_seconds"] > 0
    assert warm["cache_profile"]["host_to_device_seconds"] >= 0

    assert torch.equal(first.last_hidden_state, second.last_hidden_state)


def test_profiling_does_not_change_cached_output(monkeypatch, tmp_path):
    monkeypatch.setenv("SOUP_E2_PROFILE_CACHE", "1")
    profiled, profiled_ids, profiled_mask = _runner(tmp_path / "profiled")
    monkeypatch.delenv("SOUP_E2_PROFILE_CACHE")
    normal, normal_ids, normal_mask = _runner(tmp_path / "normal")

    with torch.no_grad():
        p_miss = profiled.forward(input_ids=profiled_ids, attention_mask=profiled_mask)
        n_miss = normal.forward(input_ids=normal_ids, attention_mask=normal_mask)
        p_hit = profiled.forward(input_ids=profiled_ids, attention_mask=profiled_mask)
        n_hit = normal.forward(input_ids=normal_ids, attention_mask=normal_mask)

    assert torch.equal(p_miss.last_hidden_state, n_miss.last_hidden_state)
    assert torch.equal(p_hit.last_hidden_state, n_hit.last_hidden_state)
