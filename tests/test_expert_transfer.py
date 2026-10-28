"""Packed-byte ownership and actual CUDA consumer/producer lifetime controls."""

from __future__ import annotations

import pytest


def test_packing_preserves_scalar_sidecars_alignment_and_exact_bytes():
    import torch

    from soup_cli.utils.expert_transfer import packing_layout

    original = {"packed": torch.tensor([1, 2, 255], dtype=torch.uint8),
                "offset": torch.tensor(-.125, dtype=torch.float32),
                "absmax": torch.arange(6, dtype=torch.float32).view(2, 3),
                "other": torch.arange(4, dtype=torch.bfloat16)}
    layout, size = packing_layout(original)
    storage = torch.empty(size, dtype=torch.uint8)
    for name, offset, count, dtype, shape in layout:
        assert offset % original[name].element_size() == 0
        storage[offset:offset + count].copy_(original[name].reshape(-1).view(torch.uint8))
        restored = storage[offset:offset + count].view(dtype).view(shape)
        assert restored.shape == original[name].shape
        assert torch.equal(restored.reshape(-1).view(torch.uint8),
                           original[name].reshape(-1).view(torch.uint8))


@pytest.mark.parametrize("kind", ("empty", "zero", "strided", "meta"))
def test_packing_rejects_invalid_host_payload(kind):
    import torch

    from soup_cli.utils.expert_transfer import packing_layout

    values = {"empty": {}, "zero": {"x": torch.empty(0)},
              "strided": {"x": torch.arange(8).view(2, 4).t()},
              "meta": {"x": torch.empty(4, device="meta")}}
    with pytest.raises(ValueError):
        packing_layout(values[kind])


@pytest.mark.parametrize("mode", (True, None, "async", 1))
def test_transfer_mode_refusal_before_any_cuda_allocation(mode):
    from soup_cli.utils.expert_stream import ExpertFetcher

    with pytest.raises(ValueError, match="transfer mode"):
        ExpertFetcher(device="cpu", dtype="float32", transfer_mode=mode)


def test_peek_changes_no_cache_counters_or_recency():
    import torch

    from soup_cli.utils.expert_cache import TensorLRU

    cache = TensorLRU(8, copy_on_insert=True)
    cache.put("one", {"x": torch.tensor([1.])})
    cache.put("two", {"x": torch.tensor([2.])})
    before = cache.stats()
    assert cache.peek("one")["x"].item() == 1
    assert cache.peek("missing") is None
    assert cache.stats() == before
    cache.put("three", {"x": torch.tensor([3.])})
    assert cache.peek("one") is None  # peek did not protect the oldest entry.


@pytest.mark.parametrize("mode", ("packed", "prefetch"))
def test_packed_modes_refuse_cpu_without_fallback(mode):
    from soup_cli.utils.expert_stream import ExpertFetcher

    with pytest.raises(ValueError, match="requires CUDA"):
        ExpertFetcher(device="cpu", dtype="float32", transfer_mode=mode)


@pytest.mark.gpu(reason="Pinned DMA and CUDA event lifetime require a real GPU")
def test_real_cuda_packed_ring_preserves_delayed_consumers_and_host_ownership():
    import torch

    from soup_cli.utils.expert_transfer import PackedExpertTransfer

    transfer = PackedExpertTransfer("cuda:0", maximum_bytes=4096)
    consumer = torch.cuda.Stream()
    outputs = []
    try:
        for number in range(9):
            source = torch.full((1024,), number + 1, dtype=torch.uint8)
            ticket = transfer.stage({"packed": source, "offset": torch.tensor(float(number))})
            source.zero_()  # the transfer must own its pinned snapshot already.
            with torch.cuda.stream(consumer):
                moved = transfer.consume(ticket)
                torch.cuda._sleep(20_000_000)
                outputs.append((moved["packed"].clone(), moved["offset"].clone()))
                transfer.release(ticket, consumed=True)
        consumer.synchronize()
        for number, (packed, offset) in enumerate(outputs):
            assert torch.equal(packed.cpu(), torch.full((1024,), number + 1, dtype=torch.uint8))
            assert offset.item() == float(number)
        assert transfer.stats()["h2d_copies"] == 9
        assert transfer.stats()["pinned_host_bytes"] <= 2 * 4096
        assert transfer.stats()["device_staging_bytes"] <= 2 * 4096
    finally:
        transfer.close()
    with pytest.raises(RuntimeError):
        transfer.consume(ticket)


@pytest.mark.gpu(reason="Prove the delayed-consumer test detects a missing CUDA reuse fence")
def test_real_cuda_lifetime_control_rejects_missing_consumer_fence(monkeypatch):
    from soup_cli.utils.expert_transfer import PackedExpertTransfer

    release = PackedExpertTransfer.release

    def unsafe_release(self, ticket, *, consumed):
        release(self, ticket, consumed=False)

    monkeypatch.setattr(PackedExpertTransfer, "release", unsafe_release)
    with pytest.raises(AssertionError):
        test_real_cuda_packed_ring_preserves_delayed_consumers_and_host_ownership()


@pytest.mark.gpu(reason="Outstanding ticket refusal requires actual CUDA slots")
def test_real_cuda_unreleased_slot_refuses_overwrite_and_close_drains():
    import torch

    from soup_cli.utils.expert_transfer import PackedExpertTransfer

    transfer = PackedExpertTransfer("cuda:0", maximum_bytes=1024)
    tickets = []
    try:
        for _ in range(2):
            tickets.append(transfer.stage({"x": torch.ones(1024, dtype=torch.uint8)}))
        with pytest.raises(RuntimeError, match="outstanding consumer"):
            transfer.stage({"x": torch.zeros(1024, dtype=torch.uint8)})
        for ticket in tickets:
            transfer.release(ticket, consumed=False)
        with pytest.raises(ValueError, match="bound"):
            transfer.stage({"x": torch.ones(1025, dtype=torch.uint8)})
    finally:
        transfer.close()
    assert all(ticket.slot["ready"].query() for ticket in tickets)


@pytest.mark.gpu(reason="NF4 cancellation and held GPU weights require CUDA")
def test_real_cuda_prefetch_mismatch_reset_identity_and_dense_ownership(tmp_path):
    import os

    from soup_cli.utils.expert_stream import ExpertFetcher
    from tests.test_stream_expert_cache_fetcher import _reader, _same

    reader, _pairs, codes = _reader(tmp_path, nf4=True, device="cuda")
    reference = ExpertFetcher(device="cuda", dtype="float32", codes=codes)
    candidate = ExpertFetcher(device="cuda", dtype="float32", codes=codes,
                              transfer_mode="prefetch")
    try:
        expected = [reference.fetch(reader, number) for number in range(3)]
        candidate.prefetch(reader, 0)
        held = candidate.fetch(reader, 2)  # discard a mismatched pending request.
        _same(expected[2], held)
        candidate.prefetch(reader, 1)
        candidate.configure_cache(host_cache_policy="static")  # drain unused lookahead.
        _same(expected[0], candidate.fetch(reader, 0))
        _same(expected[2], held)  # later slots must not overwrite dense decoded weights.
        candidate.prefetch(reader, 1)
        status = os.stat(reader.path)
        os.utime(reader.path, ns=(status.st_atime_ns, status.st_mtime_ns + 2_000_000_000))
        with pytest.raises(ValueError, match="changed"):
            candidate.fetch(reader, 1)  # validate even when the bytes are prefetched.
    finally:
        candidate.close()
        reference.close()
        reader.close()


@pytest.mark.gpu(reason="Actual NF4/BF16 optimizer parity requires CUDA")
@pytest.mark.parametrize("mode", ("packed", "prefetch"))
def test_real_cuda_transfer_preserves_kanana_full_optimizer_trajectories(
    tmp_path, monkeypatch, mode, record_property,
):
    import json

    import torch

    from benchmarks.harness import two_read_cache_worker as worker
    from tests.test_two_read_cache_full_query import _full_query_trajectories
    from tests.test_two_read_cache_kanana import _kanana_overrides

    original = worker._run_accuracy
    calls = []
    original_flags = (torch.are_deterministic_algorithms_enabled(),
                      torch.is_deterministic_algorithms_warn_only_enabled(),
                      torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)

    def capture(model, runtime, batches, spec):
        runtime.expert_fetcher.configure_transfer(mode if calls else "sync")
        result = original(model, runtime, batches, spec)
        calls.append(runtime.expert_fetcher.cache_stats())
        return result

    monkeypatch.setattr(worker, "_run_accuracy", capture)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        reports = _full_query_trajectories(
            tmp_path, dtype="bfloat16", quant="nf4", device="cuda", bitwise=False,
            evidence_dir=tmp_path / "transfer-evidence", host_cache_policy="static",
            config_overrides=_kanana_overrides(),
        )
        assert len(calls) == 4 and "transfer" not in calls[0]
        assert all(row["transfer"]["h2d_copies"] > 0 for row in calls[1:])
        assert all(report["comparison"]["passed"] for report in reports.values())
        record_property("transfer_comparisons", json.dumps(reports, sort_keys=True))
    finally:
        torch.use_deterministic_algorithms(original_flags[0], warn_only=original_flags[1])
        torch.backends.cuda.matmul.allow_tf32 = original_flags[2]
        torch.backends.cudnn.allow_tf32 = original_flags[3]
