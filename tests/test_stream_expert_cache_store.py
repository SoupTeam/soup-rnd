"""The expert cache owns complete entries without recycling live tensor storage."""

import gc
import weakref

import pytest

from soup_cli.utils.expert_cache import TensorLRU


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


@pytest.mark.parametrize("budget", [True, False, -1, 1.5, "16", None])
def test_invalid_payload_budgets_are_refused(budget):
    with pytest.raises(ValueError, match="non-negative integer"):
        TensorLRU(budget, copy_on_insert=True)


def test_copy_policy_must_be_explicitly_boolean():
    with pytest.raises(ValueError, match="must be a bool"):
        TensorLRU(16, copy_on_insert=1)


def test_zero_budget_never_clones_or_admits_and_counts_misses(torch, monkeypatch):
    cache = TensorLRU(0, copy_on_insert=True)

    def forbidden(*_args, **_kwargs):
        pytest.fail("a disabled cache must not clone")

    monkeypatch.setattr(torch.Tensor, "clone", forbidden)
    assert cache.put("expert", {"weight": torch.arange(4)}) is False
    assert cache.get("expert") is None
    assert cache.stats() == {
        "budget_bytes": 0, "payload_bytes": 0, "entries": 0,
        "hits": 0, "misses": 1, "evictions": 0,
    }


def test_complete_mixed_dtype_nf4_sidecars_and_scalars_own_their_storage(torch):
    cache = TensorLRU(64, copy_on_insert=True)
    staging = torch.arange(64, dtype=torch.uint8)
    tensors = {
        "0.gate_up": staging[:8].view(4, 2),
        "0.gate_up::absmax": staging[8:12],
        "0.gate_up::nested_absmax": staging[12:20].view(torch.float32),
        "0.gate_up::nested_offset": staging[20:24].view(torch.float32).view(()),
        "0.down": staging[24:40].view(torch.bfloat16).view(2, 4),
    }
    expected = {name: value.clone() for name, value in tensors.items()}
    assert cache.put(("reader", 0), tensors)
    got = cache.get(("reader", 0))
    assert set(got) == set(tensors)
    assert cache.stats()["payload_bytes"] == 40
    for name, value in got.items():
        assert value.data_ptr() != tensors[name].data_ptr()
        assert (value.dtype, value.device, value.shape) == (
            tensors[name].dtype, tensors[name].device, tensors[name].shape,
        )
    staging.zero_()
    for name, value in got.items():
        assert torch.equal(value, expected[name]), name
    with pytest.raises(TypeError):
        got["another"] = staging


@pytest.mark.parametrize("copy_on_insert", [True, False])
def test_exact_byte_ceiling_and_access_order_choose_eviction(torch, copy_on_insert):
    cache = TensorLRU(12, copy_on_insert=copy_on_insert)
    assert cache.put("a", {"gate_up": torch.tensor([1], dtype=torch.float32)})
    assert cache.put("b", {"gate_up": torch.tensor([2, 3], dtype=torch.float32)})
    assert cache.stats()["payload_bytes"] == 12
    assert cache.get("a")["gate_up"].item() == 1
    assert cache.put("c", {"gate_up": torch.tensor([4], dtype=torch.float32)})
    assert cache.get("b") is None
    assert cache.get("a")["gate_up"].item() == 1
    assert cache.get("c")["gate_up"].item() == 4
    assert cache.stats() == {
        "budget_bytes": 12, "payload_bytes": 8, "entries": 2,
        "hits": 3, "misses": 1, "evictions": 1,
    }


def test_oversized_complete_entry_bypasses_without_cloning_or_evicting(torch, monkeypatch):
    cache = TensorLRU(8, copy_on_insert=True)
    assert cache.put("kept", {"weight": torch.tensor([1.0])})

    def forbidden(*_args, **_kwargs):
        pytest.fail("an oversized entry must not clone")

    monkeypatch.setattr(torch.Tensor, "clone", forbidden)
    assert cache.put("large", {"gate_up": torch.zeros(2), "down": torch.zeros(1)}) is False
    assert cache.get("kept")["weight"].item() == 1
    assert cache.get("large") is None
    assert cache.stats()["payload_bytes"] == 4
    assert cache.stats()["evictions"] == 0


def test_clone_failure_does_not_publish_partial_entry_and_evicts_before_allocation(
    torch, monkeypatch,
):
    cache = TensorLRU(8, copy_on_insert=True)
    assert cache.put("old", {"weight": torch.ones(2)})
    real_clone = torch.Tensor.clone
    cloned = []
    calls = 0

    def failing_clone(tensor, *args, **kwargs):
        nonlocal calls
        calls += 1
        assert cache.stats()["payload_bytes"] == 0
        assert cache.stats()["entries"] == 0
        if calls == 2:
            raise RuntimeError("injected allocation failure")
        value = real_clone(tensor, *args, **kwargs)
        cloned.append(weakref.ref(value))
        return value

    monkeypatch.setattr(torch.Tensor, "clone", failing_clone)
    with pytest.raises(RuntimeError, match="injected allocation failure"):
        cache.put("new", {"gate_up": torch.ones(1), "down": torch.ones(1)})
    gc.collect()
    assert calls == 2 and cloned and all(ref() is None for ref in cloned)
    assert cache.get("new") is None
    assert cache.stats()["payload_bytes"] == 0
    assert cache.stats()["entries"] == 0
    assert cache.stats()["evictions"] == 1
    monkeypatch.setattr(torch.Tensor, "clone", real_clone)
    assert cache.put("new", {"weight": torch.tensor([7.0, 8.0])})
    assert torch.equal(cache.get("new")["weight"], torch.tensor([7.0, 8.0]))


def test_reference_mode_keeps_owned_tensors_valid_across_eviction_and_clear(torch):
    cache = TensorLRU(8, copy_on_insert=False)
    original = torch.tensor([1.0, 2.0])
    entry = {"weight": original}
    assert cache.put("first", entry)
    held = cache.get("first")
    assert held["weight"] is original
    entry.clear()
    assert cache.put("second", {"weight": torch.tensor([3.0, 4.0])})
    assert torch.equal(held["weight"], torch.tensor([1.0, 2.0]))
    assert cache.stats()["evictions"] == 1
    cache.clear()
    assert cache.stats()["payload_bytes"] == 0
    assert cache.stats()["entries"] == 0
    assert torch.equal(held["weight"], torch.tensor([1.0, 2.0]))


def test_clear_releases_cache_references_and_preserves_lifetime_counters(torch):
    cache = TensorLRU(4, copy_on_insert=True)
    cache.put("first", {"weight": torch.tensor([1.0])})
    held = cache.get("first")
    ref = weakref.ref(held["weight"])
    del held
    assert ref() is not None
    cache.clear()
    cache.clear()
    gc.collect()
    assert ref() is None
    assert cache.get("first") is None
    assert cache.stats() == {
        "budget_bytes": 4, "payload_bytes": 0, "entries": 0,
        "hits": 1, "misses": 1, "evictions": 0,
    }


def test_replacement_updates_complete_payload_and_lru_without_stale_parts(torch):
    cache = TensorLRU(12, copy_on_insert=True)
    cache.put("first", {"gate_up": torch.ones(1), "down": torch.ones(1)})
    cache.put("other", {"weight": torch.ones(1)})
    assert cache.put("first", {"replacement": torch.tensor([5.0])})
    assert set(cache.get("first")) == {"replacement"}
    assert cache.stats()["payload_bytes"] == 8
    assert cache.put("new", {"weight": torch.ones(2)})
    assert cache.get("other") is None
    assert cache.get("first")["replacement"].item() == 5
    assert cache.stats()["payload_bytes"] == 12
    assert cache.stats()["evictions"] == 1


def test_invalid_entries_fail_before_evicting_existing_tensor_sets(torch):
    cache = TensorLRU(4, copy_on_insert=True)
    assert cache.put("kept", {"weight": torch.ones(1)})
    for bad in ([torch.ones(1)], {"weight": object()}, {3: torch.ones(1)}):
        with pytest.raises(TypeError, match="entry must"):
            cache.put("invalid", bad)
    assert cache.stats()["evictions"] == 0
    assert cache.stats()["payload_bytes"] == 4
    assert cache.get("kept") is not None


def test_discard_releases_one_entry_without_changing_lookup_or_eviction_counters(torch):
    cache = TensorLRU(12, copy_on_insert=True)
    cache.put("first", {"weight": torch.tensor([1.0])})
    cache.put("second", {"weight": torch.tensor([2.0, 3.0])})
    held = cache.get("first")
    ref = weakref.ref(held["weight"])
    del held
    before = cache.stats()
    cache.discard("first")
    gc.collect()
    assert ref() is None
    assert cache.stats() == before | {"payload_bytes": 8, "entries": 1}
    cache.discard("absent")
    assert cache.stats() == before | {"payload_bytes": 8, "entries": 1}
    assert torch.equal(cache.get("second")["weight"], torch.tensor([2.0, 3.0]))
