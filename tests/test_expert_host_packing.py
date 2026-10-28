"""Prepacked host-cache byte ownership, admitted RAM and actual CUDA trajectories."""

from __future__ import annotations

from typing import Any

import pytest


def _components(number: int = 0) -> dict[str, Any]:
    import torch

    return {"packed": torch.tensor([1 + number, 2 + number, 255], dtype=torch.uint8),
            "offset": torch.tensor(-.125 - number, dtype=torch.float32),
            "absmax": torch.arange(6, dtype=torch.float32).view(2, 3) + number,
            "nested": torch.arange(4, dtype=torch.bfloat16) + number}


def test_host_packing_owns_exact_bytes_scalar_views_and_zero_aligned_padding() -> None:
    import gc

    import torch

    from soup_cli.utils.expert_transfer import PackedHostEntry, PackedHostHotset, packing_layout

    source = _components()
    expected = {name: value.reshape(-1).view(torch.uint8).clone() for name, value in source.items()}
    layout, size = packing_layout(source)
    cache = PackedHostHotset(size)
    assert cache.put("expert", source)
    entry = cache.peek("expert")
    assert isinstance(entry, PackedHostEntry)
    assert entry.arena.device.type == "cpu" and not entry.arena.is_pinned()
    assert entry.arena.numel() == entry.arena.untyped_storage().nbytes() == size
    assert entry.arena._base is None
    assert tuple(entry) == tuple(source) and len(entry) == len(source)
    assert entry["offset"].shape == ()
    covered = torch.zeros(size, dtype=torch.bool)
    for name, offset, count, dtype, shape in layout:
        assert offset % 256 == 0
        assert entry[name].dtype == dtype and tuple(entry[name].shape) == shape
        assert entry[name].untyped_storage().data_ptr() == entry.arena.untyped_storage().data_ptr()
        assert entry[name].data_ptr() == entry.arena.data_ptr() + offset
        assert entry[name].untyped_storage().data_ptr() != source[name].untyped_storage().data_ptr()
        assert torch.equal(entry[name].reshape(-1).view(torch.uint8), expected[name])
        covered[offset:offset + count] = True
    assert (~covered).any() and torch.count_nonzero(entry.arena[~covered]).item() == 0
    for tensor in source.values():
        tensor.zero_()
    del source
    gc.collect()
    assert all(torch.equal(entry[name].reshape(-1).view(torch.uint8), raw)
               for name, raw in expected.items())
    stats = cache.stats()
    assert stats["payload_bytes"] == size > sum(raw.numel() for raw in expected.values())
    assert stats["copied_bytes"] == sum(raw.numel() for raw in expected.values())
    assert stats["entries"] == stats["insertions"] == 1


@pytest.mark.parametrize("case", ("zero", "padding", "duplicate"))
def test_packed_host_admission_refuses_before_arena_allocation(
    monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    from soup_cli.utils import expert_transfer as module

    source = _components()
    _layout, size = module.packing_layout(source)
    cache = module.PackedHostHotset({"zero": 0, "padding": size - 1, "duplicate": 2 * size}[case])
    if case == "duplicate":
        assert cache.put("expert", source)
    before = cache.stats()

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("rejected admission allocated a packed arena")

    monkeypatch.setattr(module, "PackedHostEntry", forbidden)
    assert not cache.put("expert", source)
    after = cache.stats()
    assert after["rejected_admissions"] == before["rejected_admissions"] + 1
    assert all(after[key] == before[key] for key in
               ("payload_bytes", "entries", "insertions", "copied_bytes"))


def test_packed_host_cache_peek_has_no_counter_effect_and_discard_reclaims_padding() -> None:
    from soup_cli.utils.expert_transfer import PackedHostHotset, packing_layout

    source = _components()
    _layout, size = packing_layout(source)
    cache = PackedHostHotset(2 * size)
    assert cache.put("one", source) and cache.put("two", _components(1))
    before = cache.stats()
    assert cache.peek("one") is not None and cache.peek("missing") is None
    assert cache.stats() == before
    assert cache.get("one") is not None and cache.get("missing") is None
    assert cache.stats()["hits"] == before["hits"] + 1
    assert cache.stats()["misses"] == before["misses"] + 1
    assert not cache.put("three", _components(2))
    assert cache.peek("one") is not None and cache.peek("two") is not None
    cache.discard("one")
    assert cache.stats()["payload_bytes"] == size
    assert cache.put("three", _components(2))
    assert cache.stats()["payload_bytes"] == 2 * size
    cache.clear()
    assert cache.stats()["payload_bytes"] == cache.stats()["entries"] == 0


@pytest.mark.parametrize("kind", ("empty_name", "sparse", "quantized", "meta", "zero", "strided"))
def test_packed_host_invalid_inputs_do_not_allocate_an_arena(
    monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    import torch

    from soup_cli.utils import expert_transfer as module

    values = {"empty_name": {"": torch.ones(1)},
              "sparse": {"x": torch.eye(2).to_sparse()},
              "quantized": {"x": torch.quantize_per_tensor(torch.ones(4), .1, 0, torch.qint8)},
              "meta": {"x": torch.empty(4, device="meta")},
              "zero": {"x": torch.empty(0)},
              "strided": {"x": torch.ones(2, 4).t()}}
    cache = module.PackedHostHotset(4096)
    before = cache.stats()

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("invalid input allocated a packed arena")

    monkeypatch.setattr(module, "PackedHostEntry", forbidden)
    with pytest.raises((TypeError, ValueError)):
        cache.put("expert", values[kind])
    assert cache.stats() == before


def test_packed_host_failed_copy_does_not_admit_a_partial_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from soup_cli.utils.expert_transfer import PackedHostHotset, packing_layout

    source = _components()
    _layout, size = packing_layout(source)
    cache = PackedHostHotset(size)
    original = torch.Tensor.copy_
    calls = []

    def interrupted(self: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(self.numel())
        if len(calls) == 2:
            raise RuntimeError("injected host-copy interruption")
        return original(self, value, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "copy_", interrupted)
        with pytest.raises(RuntimeError, match="injected host-copy interruption"):
            cache.put("expert", source)
    assert cache.peek("expert") is None
    assert cache.stats()["entries"] == cache.stats()["payload_bytes"] == 0
    assert cache.stats()["insertions"] == 0
    assert cache.put("expert", source)


@pytest.mark.gpu(reason="Actual pinned-copy ring lifetime requires CUDA")
def test_real_cuda_prepacked_cache_survives_two_tickets_ring_reuse_and_cache_clear() -> None:
    import gc

    import torch

    from soup_cli.utils.expert_transfer import (
        PackedExpertTransfer,
        PackedHostHotset,
        packing_layout,
    )

    sources = [_components(number) for number in range(3)]
    expected = [{name: value.clone() for name, value in source.items()} for source in sources]
    _layout, size = packing_layout(sources[0])
    cache = PackedHostHotset(3 * size)
    assert all(cache.put(number, source) for number, source in enumerate(sources))
    for source in sources:
        for value in source.values():
            value.zero_()
    transfer = PackedExpertTransfer("cuda:0", maximum_bytes=4096)
    consumer = torch.cuda.Stream()
    outputs = []
    try:
        for wave in range(4):
            numbers = [(wave * 2 + position) % 3 for position in range(2)]
            tickets = [transfer.stage(cache.peek(number)) for number in numbers]
            with pytest.raises(RuntimeError, match="outstanding consumer"):
                transfer.stage(cache.peek(numbers[0]))
            if wave == 3:
                cache.clear()  # no source arena may be loaned to the pending DMA.
                gc.collect()
            with torch.cuda.stream(consumer):
                for number, ticket in zip(numbers, tickets):
                    moved = transfer.consume(ticket)
                    torch.cuda._sleep(20_000_000)
                    outputs.append((number, {name: value.clone() for name, value in moved.items()}))
                    transfer.release(ticket, consumed=True)
        consumer.synchronize()
        assert all(torch.equal(moved[name].cpu(), expected[number][name])
                   for number, moved in outputs for name in moved)
        stats = transfer.stats()
        assert stats["h2d_copies"] == stats["host_copy_calls"] == len(outputs) == 8
        assert stats["layout_builds"] == 0
        assert stats["pinned_host_bytes"] == stats["device_staging_bytes"] == 2 * size
        assert stats["h2d_bytes"] == 8 * size
    finally:
        transfer.close()


@pytest.mark.gpu(reason="Actual prepacked host-cache NF4/BF16 AdamW parity requires CUDA")
def test_real_cuda_prepacked_host_cache_preserves_kanana_full_optimizer_trajectories(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, record_property: Any,
) -> None:
    import json

    import torch

    from benchmarks.harness import two_read_cache_worker as worker
    from tests.test_two_read_cache_full_query import _full_query_trajectories
    from tests.test_two_read_cache_kanana import _kanana_overrides

    original = worker._run_accuracy
    calls = []
    flags = (torch.are_deterministic_algorithms_enabled(),
             torch.is_deterministic_algorithms_warn_only_enabled(),
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)

    def capture(model: Any, runtime: Any, batches: Any, spec: Any) -> Any:
        runtime.expert_fetcher.configure_transfer("packed" if calls else "sync")
        if calls:
            runtime.expert_fetcher.configure_host_packing(True)
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
            evidence_dir=tmp_path / "host-packed-evidence", host_cache_policy="static",
            config_overrides=_kanana_overrides(),
        )
        assert len(calls) == 4 and "transfer" not in calls[0]
        assert all(report["comparison"]["passed"] for report in reports.values())
        assert all(row["transfer"]["h2d_copies"] > 0 for row in calls[1:])
        for row in calls[2:]:
            assert row["host"]["hits"] > 0 and row["host_prepacked"] is True
            assert row["host"]["payload_bytes"] <= row["host"]["budget_bytes"]
            assert row["transfer"]["layout_builds"] < row["transfer"]["h2d_copies"]
        record_property("prepacked_host_comparisons", json.dumps(reports, sort_keys=True))
    finally:
        torch.use_deterministic_algorithms(flags[0], warn_only=flags[1])
        torch.backends.cuda.matmul.allow_tf32 = flags[2]
        torch.backends.cudnn.allow_tf32 = flags[3]


@pytest.mark.gpu(reason="Combined late lookahead, prepacked cache and placement need real CUDA")
def test_real_cuda_prepacked_late_prefetch_placement_preserves_kanana_full_optimizer_trajectories(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, record_property: Any,
) -> None:
    import json
    import os

    import torch

    from benchmarks.harness import two_read_cache_worker as worker
    from soup_cli.utils.expert_stream import expert_bytes
    from soup_cli.utils.layer_stream_runtime import _torch_dtype
    from tests.test_two_read_cache_full_query import _full_query_trajectories
    from tests.test_two_read_cache_kanana import _kanana_overrides

    original = worker._run_accuracy
    calls, frozen_keys = [], []
    flags = (torch.are_deterministic_algorithms_enabled(),
             torch.is_deterministic_algorithms_warn_only_enabled(),
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)

    def capture(model: Any, runtime: Any, batches: Any, spec: Any) -> Any:
        fetcher = runtime.expert_fetcher
        fetcher.configure_transfer("prefetch_after" if calls else "sync")
        if calls:
            fetcher.configure_host_packing(True)
            budget = fetcher.cache_stats()["device"]["budget_bytes"]
            if budget:
                # A manual smoke-test plan, not traffic measured in the 30B study.
                # Each runtime owns different shard paths; never reuse reference keys.
                readers = runtime.expert_readers
                keys = tuple((os.path.realpath(reader.path), 0) for reader in readers)
                dense_bytes = sum(expert_bytes(
                    reader.shapes, dtype_bytes=_torch_dtype(fetcher.dtype).itemsize,
                    quantised=False,
                )[1] for reader in readers)
                assert keys and 0 < dense_bytes <= budget
                fetcher.configure_placement(keys)
                frozen_keys.append(keys)
            else:
                fetcher.configure_placement(None)
        result = original(model, runtime, batches, spec)
        calls.append(fetcher.cache_stats())
        return result

    monkeypatch.setattr(worker, "_run_accuracy", capture)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        reports = _full_query_trajectories(
            tmp_path, dtype="bfloat16", quant="nf4", device="cuda", bitwise=False,
            evidence_dir=tmp_path / "combined-host-packed-evidence", host_cache_policy="static",
            config_overrides=_kanana_overrides(),
        )
        assert len(calls) == 4 and "transfer" not in calls[0]
        assert all(report["comparison"]["passed"] for report in reports.values())
        assert all(row["transfer"]["mode"] == "prefetch_after" for row in calls[1:])
        assert len(frozen_keys) == 2
        for row, keys in zip(calls[2:], frozen_keys):
            assert row["host_prepacked"] is True and row["host"]["hits"] > 0
            assert row["device"]["hits"] > 0
            assert 0 < row["device"]["payload_bytes"] <= row["device"]["budget_bytes"]
            assert row["host"]["payload_bytes"] <= row["host"]["budget_bytes"]
            assert row["device_placement"] == {"policy": "profile_allowlist", "keys": len(keys)}
            assert row["transfer"]["layout_builds"] < row["transfer"]["h2d_copies"]
        record_property("combined_prepacked_late_placement_comparisons",
                        json.dumps(reports, sort_keys=True))
        record_property("combined_placement_scope",
                        "Manual expert-zero-per-current-reader smoke fixture, not measured traffic")
    finally:
        torch.use_deterministic_algorithms(flags[0], warn_only=flags[1])
        torch.backends.cuda.matmul.allow_tf32 = flags[2]
        torch.backends.cudnn.allow_tf32 = flags[3]
