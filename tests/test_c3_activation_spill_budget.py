"""Shared allocation/lifetime controls for C3 with uncompressed A4 spill."""

import json
import threading
from pathlib import Path

import pytest

from soup_cli.utils.host_memory_budget import HostMemoryBudget


def test_published_a4_allocation_contract():
    from soup_cli.utils.budgeted_activations import activation_allocation_bytes

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/c3_a4_allocation_contract.json").read_text()
    )
    assert fixture["source_commit"] == "d6f7f733e09599327c61e67a3cf5434bfaa78164"
    for row in fixture["cases"]:
        args = {key: value for key, value in row.items() if key != "expected_bytes"}
        assert activation_allocation_bytes(**args) == row["expected_bytes"], row


def make_store(tmp_path, budget, **overrides):
    import torch

    from soup_cli.utils.budgeted_activations import BudgetedActivationStore
    from soup_cli.utils.l2l_activations import BufferedSpillFile

    args = dict(
        host_budget=budget,
        n_layers=5,
        k=2,
        chunk_shape=(1, 16, 64),
        dtype=torch.float32,
        device="cpu",
        pin=False,
        spill_layers=3,
        ring=2,
        spill_factory=lambda size: BufferedSpillFile(str(tmp_path / "spill.bin"), size),
    )
    args.update(overrides)
    return BudgetedActivationStore(**args)


def test_spill_and_weights_preserve_activation_priority_and_exact_values(tmp_path):
    import torch

    from soup_cli.utils.partial_ram_source import PartialRamBudget, PartialRamSource
    from tests.test_c3_partial_ram import shards

    # 4 resident chunks plus four 8192-byte aligned staging allocations.
    activation_bytes = 4 * 4096 + 4 * 8192
    budget = HostMemoryBudget(1_000_000, reserved_bytes=200)
    store = make_store(tmp_path, budget)
    assert budget.used_bytes == 200 + activation_bytes
    paths, specs = shards(tmp_path)
    source = PartialRamSource(
        str(tmp_path),
        5,
        specs,
        pin=False,
        shard_paths=paths,
        budget=PartialRamBudget(1_000_000, 200, 136),
        host_budget=budget,
    )
    try:
        assert budget.used_bytes == 200 + activation_bytes + source.host_allocated_bytes
        assert 0 < len(source.resident_indices) < 5
        for step in range(2):
            for layer in range(5):
                for mb in range(2):
                    store.put(layer, mb, torch.full((1, 16, 64), float(step + layer + mb)))
            store.begin_backward()
            for layer in reversed(range(5)):
                for mb in range(2):
                    value = store.get(layer, mb)
                    assert torch.equal(value, torch.full_like(value, float(step + layer + mb)))
            store.end_step()
        assert store.stats.bytes_written == 2 * 3 * 2 * 4096
        assert store.stats.bytes_read == store.stats.bytes_written
        source.close()
        assert budget.used_bytes == 200 + activation_bytes
    finally:
        source.close()
        store.close()
    assert budget.used_bytes == 200
    assert not (tmp_path / "spill.bin").exists()
    assert not store._resident
    assert store._write_ring is None and store._read_ring is None
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        store.begin_backward()


def test_budget_refuses_before_spill_file_creation(tmp_path):
    budget = HostMemoryBudget(4 * 4096 + 4 * 8192 - 1)
    with pytest.raises(MemoryError):
        make_store(tmp_path, budget)
    assert budget.used_bytes == 0
    assert not (tmp_path / "spill.bin").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("ring", 0),
        ("ring", True),
        ("spill_layers", -1),
        ("spill_layers", 6),
        ("spill_layers", True),
    ],
)
def test_invalid_layout_leaves_no_lease(tmp_path, field, value):
    budget = HostMemoryBudget(1_000_000)
    with pytest.raises(ValueError):
        make_store(tmp_path, budget, **{field: value})
    assert budget.used_bytes == 0


def test_partial_staging_allocation_failure_closes_backend(tmp_path, monkeypatch):
    from soup_cli.utils import l2l_activations

    budget = HostMemoryBudget(1_000_000)
    original = l2l_activations.aligned_buffer
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise MemoryError("injected staging allocation failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(l2l_activations, "aligned_buffer", fail_second)
    with pytest.raises(MemoryError, match="injected"):
        make_store(tmp_path, budget)
    assert budget.used_bytes == 0
    assert not (tmp_path / "spill.bin").exists()


def test_close_retains_lease_until_workers_finish(tmp_path, monkeypatch):
    import torch

    from soup_cli.utils.l2l_activations import BufferedSpillFile

    entered, release = threading.Event(), threading.Event()

    class DelayedWriter(BufferedSpillFile):
        def write(self, offset, buf):
            entered.set()
            assert release.wait(5)
            super().write(offset, buf)

    budget = HostMemoryBudget(1_000_000)
    store = make_store(
        tmp_path,
        budget,
        spill_factory=lambda size: DelayedWriter(str(tmp_path / "spill.bin"), size),
    )
    store.put(0, 0, torch.zeros((1, 16, 64)))
    assert entered.wait(5)
    held = budget.used_bytes
    writer = store._threads[0]
    original_join = writer.join
    monkeypatch.setattr(writer, "join", lambda timeout=None: None)
    try:
        with pytest.raises(RuntimeError, match="worker"):
            store.close()
        assert budget.used_bytes == held
        assert (tmp_path / "spill.bin").exists()
        with pytest.raises(RuntimeError, match="clos"):
            store.put(0, 1, torch.zeros((1, 16, 64)))
    finally:
        release.set()
        monkeypatch.setattr(writer, "join", original_join)
        store.close()
    assert budget.used_bytes == 0
    assert not (tmp_path / "spill.bin").exists()


def test_backend_close_failure_retains_lease_until_retry(tmp_path):
    from soup_cli.utils.l2l_activations import BufferedSpillFile

    class FailedClose(BufferedSpillFile):
        attempts = 0

        def close(self, remove=True):
            self.attempts += 1
            if self.attempts == 1:
                raise OSError("injected close failure")
            return super().close(remove)

    budget = HostMemoryBudget(1_000_000)
    store = make_store(
        tmp_path,
        budget,
        spill_factory=lambda size: FailedClose(str(tmp_path / "spill.bin"), size),
    )
    with pytest.raises(OSError, match="injected"):
        store.close()
    assert budget.used_bytes > 0
    store.close()
    assert budget.used_bytes == 0


def test_device_sync_failure_retains_buffers_and_lease(tmp_path, monkeypatch):
    import torch

    budget = HostMemoryBudget(1_000_000)
    store = make_store(tmp_path, budget, spill_layers=0)
    store.is_cuda = True  # Inject synchronization failure without requiring a GPU.
    held = budget.used_bytes

    def fail_sync(device):
        raise RuntimeError("injected synchronization failure")

    monkeypatch.setattr(torch.cuda, "synchronize", fail_sync)
    with pytest.raises(RuntimeError, match="injected synchronization"):
        store.close()
    assert budget.used_bytes == held
    assert store._resident
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    store.close()
    assert not store._resident
    assert budget.used_bytes == 0


def test_second_worker_start_failure_drains_first_worker(tmp_path, monkeypatch):
    original = threading.Thread.start
    started = []

    def fail_reader(worker):
        if worker.name == "l2l-spill-reader":
            raise RuntimeError("injected worker start failure")
        original(worker)
        started.append(worker)

    monkeypatch.setattr(threading.Thread, "start", fail_reader)
    budget = HostMemoryBudget(1_000_000)
    with pytest.raises(RuntimeError, match="injected worker start"):
        make_store(tmp_path, budget)
    assert started and all(not worker.is_alive() for worker in started)
    assert budget.used_bytes == 0
    assert not (tmp_path / "spill.bin").exists()


def test_worker_io_failure_surfaces_and_releases_resources(tmp_path):
    import torch

    from soup_cli.utils.l2l_activations import BufferedSpillFile

    failed = threading.Event()

    class FailedWrite(BufferedSpillFile):
        def write(self, offset, buf):
            raise OSError("injected spill write failure")

    budget = HostMemoryBudget(1_000_000)
    store = make_store(
        tmp_path,
        budget,
        spill_factory=lambda size: FailedWrite(str(tmp_path / "spill.bin"), size),
    )
    original = store._fail

    def recorded_failure(error):
        original(error)
        failed.set()

    store._fail = recorded_failure
    try:
        store.put(0, 0, torch.zeros((1, 16, 64)))
        assert failed.wait(5)
        with pytest.raises(RuntimeError, match="injected spill write"):
            store.get(0, 0)
    finally:
        store.close()
    assert budget.used_bytes == 0
    assert store._error is None
    assert store._threads == []
    assert not (tmp_path / "spill.bin").exists()
