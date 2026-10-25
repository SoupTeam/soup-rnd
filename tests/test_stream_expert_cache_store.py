"""The expert cache owns complete entries without recycling live tensor storage."""

import gc
import weakref

import pytest

from soup_cli.utils.expert_cache import TensorHotset, TensorLRU


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


@pytest.mark.parametrize("budget", [True, False, -1, 1.5, "16", None])
def test_static_hotset_validates_budget_and_copy_policy(budget):
    with pytest.raises(ValueError, match="non-negative integer"):
        TensorHotset(budget, copy_on_insert=True)
    with pytest.raises(ValueError, match="must be a bool"):
        TensorHotset(16, copy_on_insert=1)


def test_static_hotset_stops_cloning_after_full_and_never_evicts(torch, monkeypatch):
    cache = TensorHotset(8, copy_on_insert=True)
    staging = torch.tensor([1.0, 2.0])
    assert cache.put("first", {"weight": staging})
    held = cache.get("first")
    staging.zero_()
    before = cache.stats()

    def forbidden(*_args, **_kwargs):
        pytest.fail("full static hotset must never clone another miss")

    monkeypatch.setattr(torch.Tensor, "clone", forbidden)
    for index in range(32):
        assert cache.get(index) is None
        assert cache.put(index, {"weight": staging}) is False
        assert cache.stats()["payload_bytes"] <= cache.stats()["budget_bytes"]
    assert torch.equal(held["weight"], torch.tensor([1.0, 2.0]))
    assert cache.get("first") is held
    assert cache.stats() == before | {
        "hits": before["hits"] + 1, "misses": 32, "rejected_admissions": 32,
    }
    assert cache.stats()["insertions"] == 1
    assert cache.stats()["copied_bytes"] == 8
    assert cache.stats()["evictions"] == 0


def test_static_hotset_bypasses_large_entry_but_can_admit_a_later_smaller_one(torch, monkeypatch):
    cache = TensorHotset(12, copy_on_insert=True)
    assert cache.put("first", {"weight": torch.ones(2)})
    real_clone = torch.Tensor.clone
    calls = []

    def observed_clone(tensor, *args, **kwargs):
        calls.append(tensor.numel())
        return real_clone(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", observed_clone)
    assert cache.put("too-big", {"weight": torch.ones(4)}) is False
    assert cache.put("no-remaining-space", {"weight": torch.ones(2)}) is False
    assert calls == []
    assert cache.put("small", {"weight": torch.ones(1)})
    assert calls == [1]
    assert cache.stats()["payload_bytes"] == 12
    assert cache.stats()["entries"] == 2
    assert cache.stats()["rejected_admissions"] == 2
    assert cache.stats()["copied_bytes"] == 12
    assert cache.get("first") is not None


def test_static_hotset_never_overwrites_an_existing_identity(torch, monkeypatch):
    cache = TensorHotset(16, copy_on_insert=True)
    assert cache.put(("reader", 2, "nf4"), {"old": torch.tensor([7.0])})
    held = cache.get(("reader", 2, "nf4"))

    def forbidden(*_args, **_kwargs):
        pytest.fail("a duplicate identity must not allocate a replacement")

    monkeypatch.setattr(torch.Tensor, "clone", forbidden)
    assert cache.put(("reader", 2, "nf4"), {"new": torch.zeros(1, dtype=torch.float64)}) is False
    assert cache.get(("reader", 2, "nf4")) is held
    assert set(held) == {"old"} and held["old"].item() == 7
    assert cache.stats()["payload_bytes"] == 4
    assert cache.stats()["insertions"] == 1
    assert cache.stats()["copied_bytes"] == 4


def test_static_hotset_preserves_complete_eight_tensor_nf4_entry_from_reused_staging(torch):
    staging = torch.arange(128, dtype=torch.uint8)
    entry = {
        "gate_up": staging[:16],
        "gate_up::absmax": staging[16:20].view(torch.float32),
        "gate_up::nested_absmax": staging[20:24].view(torch.float32),
        "gate_up::nested_offset": staging[24:28].view(torch.float32).view(()),
        "down": staging[28:44],
        "down::absmax": staging[44:48].view(torch.float32),
        "down::nested_absmax": staging[48:52].view(torch.float32),
        "down::nested_offset": staging[52:56].view(torch.float32).view(()),
    }
    expected = {name: tensor.clone() for name, tensor in entry.items()}
    cache = TensorHotset(56, copy_on_insert=True)
    assert cache.put("expert", entry)
    held = cache.get("expert")
    assert len(held) == 8 and cache.stats()["payload_bytes"] == 56
    staging.zero_()
    entry.clear()
    for name, tensor in held.items():
        assert torch.equal(tensor, expected[name]), name
        assert tensor.dtype == expected[name].dtype and tensor.shape == expected[name].shape
        assert tensor.untyped_storage().data_ptr() != staging.untyped_storage().data_ptr()
    with pytest.raises(TypeError):
        held["new"] = staging
    assert cache.stats()["copied_bytes"] == 56


def test_static_hotset_failed_partial_clone_keeps_existing_entries_and_releases_work(
    torch, monkeypatch,
):
    cache = TensorHotset(12, copy_on_insert=True)
    assert cache.put("kept", {"weight": torch.tensor([9.0])})
    held = cache.get("kept")
    real_clone = torch.Tensor.clone
    cloned = []
    calls = 0

    def fail_second(tensor, *args, **kwargs):
        nonlocal calls
        calls += 1
        assert cache.stats()["entries"] == 1
        assert cache.stats()["payload_bytes"] == 4
        if calls == 2:
            raise RuntimeError("injected second-sidecar clone failure")
        result = real_clone(tensor, *args, **kwargs)
        cloned.append(weakref.ref(result))
        return result

    monkeypatch.setattr(torch.Tensor, "clone", fail_second)
    with pytest.raises(RuntimeError, match="second-sidecar"):
        cache.put("failed", {"gate_up": torch.ones(1), "down": torch.ones(1)})
    gc.collect()
    assert calls == 2 and all(reference() is None for reference in cloned)
    assert cache.get("failed") is None and cache.get("kept") is held
    assert held["weight"].item() == 9
    assert cache.stats()["payload_bytes"] == 4
    assert cache.stats()["insertions"] == 1
    assert cache.stats()["copied_bytes"] == 8
    assert cache.stats()["evictions"] == cache.stats()["rejected_admissions"] == 0
    monkeypatch.setattr(torch.Tensor, "clone", real_clone)
    assert cache.put("retry", {"gate_up": torch.ones(1), "down": torch.ones(1)})
    assert cache.stats()["payload_bytes"] == 12


@pytest.mark.parametrize("wrong_clone", ["dtype", "shape", "alias"])
def test_static_hotset_refuses_changed_clone_metadata_or_staging_alias(
    torch, monkeypatch, wrong_clone,
):
    cache = TensorHotset(16, copy_on_insert=True)
    assert cache.put("kept", {"weight": torch.ones(1)})
    real_clone = torch.Tensor.clone

    def bad_clone(tensor, *args, **kwargs):
        if wrong_clone == "dtype":
            return tensor.to(dtype=torch.float64)
        if wrong_clone == "shape":
            return real_clone(tensor, *args, **kwargs).reshape(1, -1)
        return tensor

    monkeypatch.setattr(torch.Tensor, "clone", bad_clone)
    with pytest.raises(RuntimeError, match="metadata or aliases"):
        cache.put("invalid", {"weight": torch.ones(1)})
    assert cache.get("invalid") is None
    assert cache.stats()["payload_bytes"] == 4
    assert cache.stats()["entries"] == cache.stats()["insertions"] == 1


def test_static_hotset_clear_discard_preserve_held_entries_and_lifetime_counters(torch):
    cache = TensorHotset(8, copy_on_insert=True)
    assert cache.put("first", {"weight": torch.tensor([1.0])})
    assert cache.put("second", {"weight": torch.tensor([2.0])})
    held = cache.get("first")
    before = cache.stats()
    cache.discard("first")
    assert cache.stats() == before | {"payload_bytes": 4, "entries": 1}
    assert cache.put("third", {"weight": torch.tensor([3.0])})
    before_clear = cache.stats()
    cache.clear()
    cache.clear()
    cache.discard("absent")
    assert cache.stats() == before_clear | {"payload_bytes": 0, "entries": 0}
    assert held["weight"].item() == 1
    assert cache.put("fourth", {"weight": torch.tensor([4.0, 5.0])})
    assert cache.stats()["payload_bytes"] == 8


def test_static_hotset_reference_mode_retains_immutable_owned_tensors_without_cloning(
    torch, monkeypatch,
):
    cache = TensorHotset(4, copy_on_insert=False)
    original = torch.tensor([8.0])

    def forbidden(*_args, **_kwargs):
        pytest.fail("immutable reference mode must not clone")

    monkeypatch.setattr(torch.Tensor, "clone", forbidden)
    assert cache.put("first", {"weight": original})
    held = cache.get("first")
    assert held["weight"] is original
    cache.clear()
    assert held["weight"] is original and original.item() == 8
    assert cache.stats()["copied_bytes"] == 0


def test_static_hotset_invalid_metadata_is_refused_before_any_clone_or_publication(
    torch, monkeypatch,
):
    cache = TensorHotset(8, copy_on_insert=True)
    assert cache.put("kept", {"weight": torch.ones(1)})

    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid metadata must be checked before cloning")

    monkeypatch.setattr(torch.Tensor, "clone", forbidden)
    for invalid in ({}, {"": torch.ones(1)}, {3: torch.ones(1)}, {"x": None},
                    {"x": torch.empty(0)}, {"x": torch.empty(1, device="meta")},
                    {"x": torch.ones(2).to_sparse()}):
        with pytest.raises((TypeError, ValueError)):
            cache.put("invalid", invalid)
    with pytest.raises(TypeError):
        cache.put([], {"weight": torch.ones(1)})
    assert cache.stats()["entries"] == 1
    assert cache.stats()["payload_bytes"] == cache.stats()["copied_bytes"] == 4
    assert cache.stats()["rejected_admissions"] == 0


def test_static_hotset_disabled_budget_bypasses_without_cloning(torch, monkeypatch):
    cache = TensorHotset(0, copy_on_insert=True)

    def forbidden(*_args, **_kwargs):
        pytest.fail("disabled static hotset must not clone")

    monkeypatch.setattr(torch.Tensor, "clone", forbidden)
    assert cache.put("expert", {"weight": torch.ones(1)}) is False
    assert cache.get("expert") is None
    assert cache.stats() == {
        "budget_bytes": 0, "payload_bytes": 0, "entries": 0, "hits": 0,
        "misses": 1, "evictions": 0, "insertions": 0, "copied_bytes": 0,
        "rejected_admissions": 1,
    }
