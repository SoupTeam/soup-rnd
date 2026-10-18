"""Experimental C2 adapter: same source contracts at both physical-read depths."""
import importlib.util
import threading
from pathlib import Path

import pytest

from tests import test_issue971_async_disk_source as reference


@pytest.fixture
def factory():
    path = Path(__file__).parents[1] / "benchmarks/harness/c2_source_depth.py"
    spec = importlib.util.spec_from_file_location("c2_source_depth", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.depth_source


@pytest.mark.parametrize("depth", [1, 2])
def test_idle_range_worker_crosses_layer_boundary_only_at_depth_two(
    tmp_path, monkeypatch, factory, depth
):
    source_type = factory(depth)
    real = source_type._range_works
    slow = threading.Event()
    release = threading.Event()
    next_layer = threading.Event()

    def works(self, idx, region):
        jobs = real(self, idx, region)

        def wrap(job, part):
            def run():
                if idx == 1 and part == 0:
                    slow.set()
                    assert release.wait(5)
                if idx == 2:
                    next_layer.set()
                job()
            return run

        return [wrap(job, part) for part, job in enumerate(jobs)]

    monkeypatch.setattr(source_type, "_range_works", works)
    folder = reference._uniform_shards(tmp_path, 8, 2**20)
    spec = reference.RamSource.layer_specs_from_shards(folder, 8)
    source = source_type(folder, 8, spec, read_ahead=3, pin=False, read_ranges=4)
    try:
        source.get(0, "w")
        assert slow.wait(5)
        assert next_layer.wait(0.2) == (depth == 2)
        with source._ready:
            reads = list(source._pending.values())
            assert len(reads) <= depth
            assert len({read.slot for read in reads}) == len(reads)
            assert len(source._readers._threads) == 4
    finally:
        release.set()
        source.close()


@pytest.mark.parametrize("depth", [1, 2])
def test_all_tensor_bytes_against_independent_disk_source(tmp_path, monkeypatch, factory, depth):
    monkeypatch.setattr(reference, "AsyncDiskSource", factory(depth))
    reference.TestByteIdentityAgainstTheShippedSource().test_every_tensor_matches_disk_source(
        tmp_path, read_ahead=3
    )


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("method", [
    "test_a_read_error_surfaces_at_the_get_that_wanted_it",
    "test_every_later_get_raises_the_same_stored_error",
    "test_close_is_idempotent_and_stops_the_thread",
    "test_close_is_a_barrier_and_drops_the_pools_events",
])
def test_existing_error_and_close_contracts(tmp_path, monkeypatch, factory, depth, method):
    monkeypatch.setattr(reference, "AsyncDiskSource", factory(depth))
    getattr(reference.TestFailuresAreLoudAndNeverHang(), method)(tmp_path)


@pytest.mark.parametrize("depth", [1, 2])
def test_turnaround_keeps_wanted_slots(tmp_path, monkeypatch, factory, depth):
    monkeypatch.setattr(reference, "AsyncDiskSource", factory(depth))
    reference.TestDepthSurvivesTheTurnaround().test_a_layer_inside_the_window_is_not_the_eviction_victim(
        tmp_path
    )


@pytest.mark.parametrize("depth", [1, 2])
def test_unreleased_pinned_borrow_is_refused_on_cpu(tmp_path, monkeypatch, factory, depth):
    monkeypatch.setattr(reference, "AsyncDiskSource", factory(depth))
    reference.TestPinnedStagingRefusesAnUnreleasedBorrowWithoutAGpu().test_a_second_get_without_release_is_refused(
        tmp_path
    )


@pytest.mark.gpu(reason="real delayed CUDA copies must retain their host staging bytes")
@pytest.mark.parametrize("depth", [1, 2])
def test_real_pool_cuda_copies_keep_the_requested_layer(tmp_path, monkeypatch, factory, depth):
    monkeypatch.setattr(reference, "AsyncDiskSource", factory(depth))
    reference.TestTheDeviceGetsTheLayerItAskedFor().test_no_layer_reaches_the_device_holding_another_layers_weights(
        tmp_path, read_ahead=3
    )
