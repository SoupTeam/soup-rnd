"""Shared C3/A4 contract proposal: atomic leases, actual consumers, no GPU claim."""

import pytest

from soup_cli.utils.host_memory_budget import HostMemoryBudget


def test_reservations_are_exclusive_and_released():
    budget = HostMemoryBudget(100, reserved_bytes=20)
    first = budget.reserve("weights", 50)
    with pytest.raises(MemoryError):
        budget.reserve("activations", 31)
    with pytest.raises(ValueError):
        budget.reserve("weights", 1)
    assert budget.used_bytes == 70
    first.resize(30)
    second = budget.reserve("activations", 50)
    assert budget.used_bytes == 100
    first.close()
    first.close()
    assert budget.used_bytes == 70
    second.close()
    assert budget.used_bytes == 20
    with pytest.raises(RuntimeError):
        first.resize(1)


def test_failed_growth_does_not_change_lease():
    budget = HostMemoryBudget(100)
    lease = budget.reserve("weights", 50)
    with pytest.raises(MemoryError):
        lease.resize(101)
    assert budget.used_bytes == 50
    lease.close()


@pytest.mark.parametrize("bad", [True, -1, 1.5])
def test_invalid_byte_counts_fail(bad):
    with pytest.raises(ValueError):
        HostMemoryBudget(bad)


def test_two_concurrent_consumers_cannot_overbook():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    budget = HostMemoryBudget(100)
    ready = Barrier(2)

    def claim(name):
        ready.wait()
        try:
            return budget.reserve(name, 70)
        except MemoryError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        leases = list(executor.map(claim, ["weights", "activations"]))
    assert sum(lease is not None for lease in leases) == 1
    assert budget.used_bytes == 70
    for lease in leases:
        if lease:
            lease.close()
    assert budget.used_bytes == 0


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)])
def test_real_weights_and_activation_store_share_one_limit(tmp_path, device):
    import torch

    from soup_cli.utils.budgeted_activations import BudgetedActivationStore
    from soup_cli.utils.partial_ram_source import PartialRamBudget, PartialRamSource
    from tests.test_c3_partial_ram import shards

    paths, specs = shards(tmp_path)
    budget = HostMemoryBudget(1_000_000, reserved_bytes=200)
    source = PartialRamSource(
        str(tmp_path),
        5,
        specs,
        pin=False,
        shard_paths=paths,
        budget=PartialRamBudget(1_000_000, 200, 136),
        host_budget=budget,
    )
    weights_used = budget.used_bytes
    store = BudgetedActivationStore(
        host_budget=budget,
        n_layers=5,
        k=4,
        chunk_shape=(1, 8, 32),
        dtype=torch.float32,
        device=device,
        pin=False,
    )
    assert budget.used_bytes == weights_used + 5 * 4 * 1 * 8 * 32 * 4
    for layer in range(5):
        for mb in range(4):
            value = torch.full((1, 8, 32), float(layer + mb), device=device)
            store.put(layer, mb, value)
            assert torch.equal(store.get(layer, mb), value)
    store.close()
    store.close()
    assert budget.used_bytes == weights_used
    with pytest.raises(RuntimeError, match="closed"):
        store.get(0, 0)
    source.close()
    assert budget.used_bytes == 200


def test_activation_budget_failure_preserves_existing_weight_reservation(monkeypatch):
    import torch

    from soup_cli.utils.budgeted_activations import BudgetedActivationStore

    budget = HostMemoryBudget(100)
    lease = budget.reserve("c3.weights", 90)
    allocated = []
    real_empty = torch.empty

    def observe(*args, **kwargs):
        # A scalar dtype-size lookup is allowed; no actual activation allocation.
        if args and args[0] != ():
            allocated.append(args[0])
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", observe)
    with pytest.raises(MemoryError):
        BudgetedActivationStore(
            host_budget=budget,
            n_layers=2,
            k=4,
            chunk_shape=(1, 8, 32),
            dtype=torch.float32,
            device="cpu",
            pin=False,
        )
    assert not allocated
    assert budget.used_bytes == 90
    lease.close()
