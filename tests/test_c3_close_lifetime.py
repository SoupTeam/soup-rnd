"""C3 close(): the weight lease outlives every user of the staging memory it covers.

Real CPU shards (one cached layer, one disk layer). A range read is held inside a
worker, and bounded joins are shortened to simulate their expiry; disk-copy events
are fakes that record when they are waited on.
"""

import threading

import pytest

from soup_cli.utils import partial_ram_source
from soup_cli.utils.partial_ram_source import PartialRamBudget, PartialRamSource
from tests.test_c3_partial_ram import shards

BUDGET = PartialRamBudget(1_000_000, 200, 68)


def _source(tmp_path):
    paths, specs = shards(tmp_path, count=2)
    source = PartialRamSource(
        str(tmp_path), 2, specs, pin=False, shard_paths=paths, budget=BUDGET, read_ahead=1
    )
    assert source.resident_indices == (0,)
    assert source.disk_indices == (1,)
    return source


def _charged(source):
    return source.host_budget.used_bytes - BUDGET.reserved_bytes


def test_blocked_range_worker_keeps_lease_until_retry(tmp_path, monkeypatch):
    from soup_cli.utils import async_disk_source
    from soup_cli.utils.async_disk_source import AsyncDiskSource

    entered, gate = threading.Event(), threading.Event()
    original_works = AsyncDiskSource._range_works

    def held_works(self, idx, region):
        def hold(work):
            def run():
                entered.set()
                gate.wait(timeout=60)
                work()

            return run

        return [hold(work) for work in original_works(self, idx, region)]

    original_close = async_disk_source._RangeReaders.close
    monkeypatch.setattr(AsyncDiskSource, "_range_works", held_works)
    monkeypatch.setattr(
        async_disk_source._RangeReaders, "close", lambda self: original_close(self, timeout=0.05)
    )
    monkeypatch.setattr(partial_ram_source, "CLOSE_JOIN_SECONDS", 0.05)

    source = _source(tmp_path)
    charged = _charged(source)
    assert charged > 0
    errors = []

    def consumer():
        try:
            source.get(1, "weight")
        except BaseException as error:  # noqa: BLE001 - recorded for the assertion
            errors.append(error)

    reader = threading.Thread(target=consumer, daemon=True)
    reader.start()
    try:
        assert entered.wait(timeout=10)
        with pytest.raises(RuntimeError, match="lease is retained"):
            source.close()
        # Staging is still being written: its bytes stay charged and unavailable.
        assert _charged(source) == charged
        available = BUDGET.limit_bytes - BUDGET.reserved_bytes - charged
        assert source.host_budget.available_bytes == available
        with pytest.raises(MemoryError):
            source.host_budget.reserve("other", available + 1)
        with pytest.raises(RuntimeError, match="closing"):
            source.get(0, "weight")
        with pytest.raises(RuntimeError, match="closing"):
            source.release(0)
    finally:
        gate.set()
    reader.join(timeout=10)
    assert not reader.is_alive()
    monkeypatch.setattr(partial_ram_source, "CLOSE_JOIN_SECONDS", 10.0)
    source.close()
    assert source.host_budget.used_bytes == BUDGET.reserved_bytes
    assert source.host_allocated_bytes == 0
    source.close()


class Event:
    def __init__(self, source, failures=0):
        self.source = source
        self.failures = failures
        self.charged_when_waited = []

    def query(self):
        return False

    def synchronize(self):
        self.charged_when_waited.append(_charged(self.source))
        if self.failures:
            self.failures -= 1
            raise RuntimeError("injected device error")


def test_released_disk_layer_event_is_drained_before_lease(tmp_path):
    source = _source(tmp_path)
    charged = _charged(source)
    event = Event(source)
    source.get(1, "weight")
    source.release(1, event)
    source.close()
    assert event.charged_when_waited == [charged]
    assert source.host_budget.used_bytes == BUDGET.reserved_bytes


@pytest.mark.parametrize("layer", [0, 1])
def test_failed_event_drain_retains_lease_and_retry_releases(tmp_path, layer):
    source = _source(tmp_path)
    charged = _charged(source)
    event = Event(source, failures=1)
    source.get(layer, "weight")
    source.release(layer, event)
    with pytest.raises(RuntimeError, match="injected"):
        source.close()
    assert _charged(source) == charged
    with pytest.raises(RuntimeError, match="closing"):
        source.get(layer, "weight")
    source.close()
    assert event.charged_when_waited == [charged, charged]
    assert source.host_budget.used_bytes == BUDGET.reserved_bytes
