"""B1.4: on the per_expert path the layer read-ahead does not change the result.

There is no expert prefetch in the code: ``ExpertBufferPool.lease`` reads one expert
synchronously (see ``docs/expert-forward.md``). The only read-ahead is the layer one, the
disk tier's ``AsyncDiskSource`` driven by ``StreamPrefetcher``. It stages the non-expert
layer weights. This file checks that with it active the per_expert forward and backward
equal the run without it (the RAM tier reads nothing ahead), and that the experts are
still read by the pool, once per routed expert in a forward pass.

Seam: ``build_streamed_model`` on a per_expert cache, observed through ``run_moe_pass``.
Correctness only. No timing claim: the test does not show that a read finished early.
It runs on CPU with a 3-layer model, so it cannot see the staging-slot reuse race that
``AsyncDiskSource.release`` guards against on a pinned CUDA copy.
"""

from __future__ import annotations

import threading
from typing import Any, List

import pytest

from benchmarks.harness import b13_moe_fixture as fixture
from tests.test_b13_moe_runtime import (
    build_moe_streamed,
    grads_differ,
    record_expert_reads,
    routed_expert_hits,
    run_moe_pass,
    shard_moe,
)
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR

pytestmark = _REQUIRES_TRAINING_FLOOR


def _record_layer_reader_threads(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    """Log the thread that reads each layer file for the async disk reader."""
    from soup_cli.utils.async_disk_source import AsyncDiskSource

    log: List[Any] = []
    original = AsyncDiskSource._read_layer

    def observe(self: Any, idx: int, region: Any) -> Any:
        log.append((idx, threading.current_thread()))
        return original(self, idx, region)

    monkeypatch.setattr(AsyncDiskSource, "_read_layer", observe)
    return log


@pytest.mark.parametrize("read_ahead", [1, 2])
def test_per_expert_forward_and_backward_match_with_and_without_layer_read_ahead(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, read_ahead: int
) -> None:
    import torch

    config = fixture.moe_config()
    weights = fixture.moe_weights_dir(tmp_path, config)
    index = shard_moe(weights, tmp_path / "shards")
    assert index.expert_layout == "per_expert"
    reads = _record_layer_reader_threads(monkeypatch)

    plain = build_moe_streamed(weights, tmp_path / "shards", index=index, tier="ram")
    ahead = build_moe_streamed(
        weights, tmp_path / "shards", index=index, tier="disk", read_ahead=read_ahead
    )
    try:
        plain_stats, ahead_stats = plain.runtime.stats(), ahead.runtime.stats()
        assert (plain_stats["tier"], plain_stats["read_ahead"]) == ("ram", None)
        assert (ahead_stats["tier"], ahead_stats["read_ahead"]) == ("disk", read_ahead)
        fixture.copy_adapter_weights(plain.model, ahead.model)
        batch = fixture.moe_batch(config)

        want = run_moe_pass(plain.model, batch)
        got = run_moe_pass(ahead.model, batch)

        assert bool(want.logits.abs().sum() > 0)
        assert torch.equal(got.logits, want.logits)
        assert torch.equal(got.loss, want.loss)
        assert want.grads and all(g is not None for g in want.grads.values())
        assert grads_differ(got.grads, want.grads) == []
        # The read-ahead really ran: the layer files were read by the reader thread.
        assert reads, "the disk tier never read a layer file"
        assert threading.main_thread() not in {thread for _, thread in reads}
    finally:
        ahead.close()
        plain.close()


def test_layer_read_ahead_leaves_the_expert_reads_to_the_pool(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Experts are read by the pool in both runs, the same experts in the same count."""
    config = fixture.moe_config()
    weights = fixture.moe_weights_dir(tmp_path, config)
    index = shard_moe(weights, tmp_path / "shards")
    plain = build_moe_streamed(weights, tmp_path / "shards", index=index, tier="ram")
    ahead = build_moe_streamed(
        weights, tmp_path / "shards", index=index, tier="disk", read_ahead=2
    )
    try:
        fixture.copy_adapter_weights(plain.model, ahead.model)
        batch = fixture.moe_batch(config)
        hits = routed_expert_hits(plain.model, batch)
        assert hits and all(hits.values())
        plain_before = plain.runtime.stats()["expert_fetches"]
        plain_reads = record_expert_reads(plain.model)
        ahead_reads = record_expert_reads(ahead.model)
        run_moe_pass(plain.model, batch, backward=False)
        run_moe_pass(ahead.model, batch, backward=False)

        assert ahead_reads == plain_reads
        for layer, experts in hits.items():
            assert sorted(plain_reads[layer]) == sorted(experts)  # each routed expert once
        plain_fetches = plain.runtime.stats()["expert_fetches"] - plain_before
        assert plain_fetches == ahead.runtime.stats()["expert_fetches"] > 0
    finally:
        ahead.close()
        plain.close()
