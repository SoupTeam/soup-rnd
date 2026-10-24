"""GPU-resident layers must not occupy the RAM hotset or trigger SSD reads."""

from collections import Counter

import pytest

from tests.test_stream_ram_cache import source


@pytest.mark.parametrize("depth", [1, 2, 4])
@pytest.mark.parametrize("ram_bytes", [0, 16384])
def test_source_skips_gpu_layers_and_preserves_rotating_pool_bytes(
    tmp_path, monkeypatch, depth, ram_bytes,
):
    import torch

    from soup_cli.utils.async_disk_source import AsyncDiskSource
    from soup_cli.utils.layer_stream_runtime import DiskSource, LayerBufferPool
    from tests.test_issue971_async_disk_source import _raw_bytes

    reads = Counter()
    original = AsyncDiskSource._range_works

    def record(self, idx, region):
        reads[idx] += 1
        return original(self, idx, region)

    monkeypatch.setattr(AsyncDiskSource, "_range_works", record)
    folder, specs, ours = source(tmp_path, monkeypatch, depth=depth, budget=ram_bytes)
    shipped = DiskSource(folder, 8, specs)
    pool = LayerBufferPool(specs[0], n_buffers=2, device="cpu")
    try:
        pool.reserve_layers([3], ours)
        ours.set_resident_layers([3])
        stats = ours.ram_cache_stats()
        assert stats["resident_indices"] == [3]
        assert 3 not in stats["indices"]
        if ram_bytes:
            assert len(stats["indices"]) == 2
            assert stats["allocated_bytes"] <= ram_bytes
        walk = list(range(8)) + list(reversed(range(8)))
        for _ in range(2):
            for idx in walk:
                pool.load_async(idx, ours)
                pool.wait(idx)
        before = reads.copy()
        for _ in range(3):
            for idx in walk:
                pool.load_async(idx, ours)
                for name, actual in pool.wait(idx).items():
                    assert torch.equal(_raw_bytes(actual), _raw_bytes(shipped.get(idx, name)))
        for idx in stats["indices"] + [3]:
            assert reads[idx] == before[idx], "hot layers must not be read speculatively"
        with pytest.raises(RuntimeError, match="resident"):
            ours.get(3, next(iter(specs[3])))
        ours.set_resident_layers([])
        pool.clear_resident_layers()
        for idx in walk:
            pool.load_async(idx, ours)
            pool.wait(idx)
    finally:
        ours.close()
        shipped.close()


def test_ram_cache_can_be_enabled_after_residency_without_duplicate_allocation(
    tmp_path, monkeypatch,
):
    from soup_cli.utils import async_disk_source

    _folder, _specs, ours = source(tmp_path, monkeypatch, budget=0)
    original = ours._allocate_staging
    allocations = []

    def allocate(sizes):
        assert not ours._cache_arenas
        allocations.append(list(sizes))
        return original(sizes)

    monkeypatch.setattr(ours, "_allocate_staging", allocate)
    try:
        ours.set_resident_layers([3])
        ours.configure_ram_cache(16384)
        assert ours.ram_cache_stats()["indices"] == [2, 4]
        assert len(allocations) == 1
        ours.set_resident_layers([2, 3])
        assert not ({2, 3} & set(ours.ram_cache_stats()["indices"]))
        assert len(allocations) == 2
        assert async_disk_source.AsyncDiskSource is type(ours)
    finally:
        ours.close()


def test_cache_replanning_drains_events_before_releasing_memory(tmp_path, monkeypatch):
    _folder, specs, ours = source(tmp_path, monkeypatch)
    idx = ours.ram_cache_stats()["indices"][0]
    called = []

    class Event:
        def synchronize(self):
            assert ours._cache_arenas
            called.append(True)

    try:
        ours.get(idx, next(iter(specs[idx])))
        with pytest.raises(RuntimeError, match="loan|borrow"):
            ours.set_resident_layers([idx])
        ours.release(idx, Event())
        ours.set_resident_layers([idx])
        assert called == [True]
        assert idx not in ours.ram_cache_stats()["indices"]
    finally:
        ours.close()


@pytest.mark.parametrize("indices", [[True], [-1], [8], [1, 1], [3, 2], [1.5]])
def test_invalid_resident_indices_do_not_change_live_cache(tmp_path, monkeypatch, indices):
    _folder, _specs, ours = source(tmp_path, monkeypatch)
    try:
        before = ours.ram_cache_stats()
        with pytest.raises(ValueError, match="indices"):
            ours.set_resident_layers(indices)
        assert ours.ram_cache_stats() == before
    finally:
        ours.close()


@pytest.mark.parametrize("refuse", [False, True])
def test_disk_planner_publishes_exclusions_and_restores_them_on_fallback(refuse):
    from tests.test_stream_residency_planner import configure, fixture

    model, runtime, memory, _calls, good = fixture()
    runtime.tier = "disk"
    exclusions = []

    class Source:
        def set_resident_layers(self, indices):
            exclusions.append(tuple(indices))

    runtime.source = Source()

    def measure(*args, **kwargs):
        assert (
            not runtime.pool.resident_slots or exclusions[-1] == tuple(runtime.pool.resident_slots)
        )
        peak = good(*args, **kwargs)
        if refuse and runtime.pool.resident_slots:
            from dataclasses import replace

            peak = replace(peak, reserved_bytes=9999)
        return peak

    plan = configure(model, runtime, memory, measure)
    assert exclusions[0] == (1, 2, 3)
    if refuse:
        assert not plan.enabled
        assert exclusions[-1] == ()
        assert runtime.pool.resident_slots == {}
    else:
        assert plan.indices == (1, 2, 3)


@pytest.mark.gpu
def test_pinned_joint_cache_uses_real_cuda_pool_and_retains_exact_bytes(tmp_path, monkeypatch):
    import torch

    from soup_cli.utils.async_disk_source import AsyncDiskSource
    from soup_cli.utils.layer_stream_runtime import DiskSource, LayerBufferPool, RamSource
    from tests.test_issue971_async_disk_source import _raw_bytes

    folder, specs, seed_source = source(tmp_path, monkeypatch, budget=0)
    seed_source.close()
    ours = AsyncDiskSource(folder, 8, specs, pin=True, ram_cache_bytes=0)
    reference = DiskSource(folder, 8, specs)
    pool = LayerBufferPool(specs[0], device="cuda")
    stream = torch.cuda.Stream()
    retained = []
    try:
        pool.reserve_layers([3], ours, stream)
        ours.set_resident_layers([3])
        ours.configure_ram_cache(16384)
        assert set(ours.ram_cache_stats()["indices"]).isdisjoint(pool.resident_slots)
        for idx in (list(range(8)) + list(reversed(range(8)))) * 3:
            pool.load_async(idx, ours, stream)
            retained.append((idx, {key: value.clone() for key, value in pool.wait(idx).items()}))
        ours.close()
        torch.cuda.synchronize()
        for idx, values in retained:
            for name, actual in values.items():
                assert torch.equal(_raw_bytes(actual.cpu()), _raw_bytes(reference.get(idx, name)))
        assert RamSource.layer_specs_from_shards(folder, 8) == specs
    finally:
        ours.close()
        reference.close()


@pytest.mark.parametrize("fail", [False, True])
def test_real_trainer_allocates_ram_only_after_gpu_planning(tmp_path, monkeypatch, fail):
    from types import SimpleNamespace

    import soup_cli.utils.stream_residency as residency
    from tests.test_issue623_stream_pin_allocation import _drive_wiring

    seen = []

    def configure(_model, runtime, *_args, **_kwargs):
        seen.append(runtime)
        runtime.source.configure_ram_cache.assert_not_called()
        if fail:
            runtime.source.configure_ram_cache.side_effect = OSError("cache setup sentinel")
        return SimpleNamespace(indices=(1,))

    monkeypatch.setattr(residency, "configure_stream_residency", configure)

    def run():
        return _drive_wiring(
            tmp_path, monkeypatch, stream_pin_yaml="", stream_source="disk",
            extra_training_yaml="  stream_ram_cache_mb: 1024\n  stream_resident_layers: auto\n",
        )

    if fail:
        with pytest.raises(OSError, match="sentinel"):
            run()
        seen[0].close.assert_called_once()
    else:
        kwargs, _text = run()
        assert "ram_cache_bytes" not in kwargs
        seen[0].source.configure_ram_cache.assert_called_once_with(1024 * 2**20)
