"""CPU-only controls for the traffic-proxy expert placement policy."""

from dataclasses import FrozenInstanceError

import pytest

from soup_cli.utils.expert_placement import ExpertUsageProfile


def test_placement_uses_frequency_cost_and_dense_memory_instead_of_first_fit():
    profile = ExpertUsageProfile()
    profile.record_request("cold", transfer_bytes=2, dense_bytes=8)
    profile.record_request("frequent", transfer_bytes=2, dense_bytes=4, count=8)
    profile.record_request("expensive", transfer_bytes=20, dense_bytes=4)
    plan = profile.plan(8)
    assert plan.keys == ("expensive", "frequent")
    assert plan.payload_bytes == plan.budget_bytes == 8
    assert plan.candidate_count == 3
    assert plan.total_requests == 10
    assert plan.selected_requests == 9
    assert plan.saved_transfer_bytes_proxy == 36
    assert plan.contains("expensive") and not plan.contains("cold")
    assert plan.policy == "frequency_transfer_bytes_per_dense_byte"


def test_equal_scores_keep_first_observed_order_not_later_update_order():
    profile = ExpertUsageProfile()
    profile.record_request(("layer-z", 1), transfer_bytes=4, dense_bytes=4)
    profile.record_request(("layer-a", 1), transfer_bytes=8, dense_bytes=4)
    profile.record_request(("layer-z", 1), transfer_bytes=4, dense_bytes=4)
    before = profile.snapshot()
    for _repeat in range(3):
        assert profile.plan(4).keys == (("layer-z", 1),)
        assert profile.snapshot() == before


def test_planner_skips_oversize_and_nonfitting_entries_without_eviction():
    profile = ExpertUsageProfile()
    profile.record_request("oversize", transfer_bytes=1000, dense_bytes=11)
    profile.record_request("first", transfer_bytes=18, dense_bytes=6)
    profile.record_request("cannot_fit_remaining", transfer_bytes=10, dense_bytes=5)
    profile.record_request("fills_remaining", transfer_bytes=4, dense_bytes=4)
    assert profile.plan(10).keys == ("first", "fills_remaining")
    assert profile.plan(10).payload_bytes == 10
    assert profile.plan(3).keys == ()


def test_empty_and_zero_budget_plans_preserve_all_observations():
    profile = ExpertUsageProfile()
    assert profile.plan(7).keys == ()
    assert profile.plan(7).payload_bytes == 0
    profile.record_request("expert", transfer_bytes=2, dense_bytes=4, count=3)
    before = profile.snapshot(), profile.stats()
    plan = profile.plan(0)
    assert plan.keys == () and plan.payload_bytes == 0
    assert plan.total_requests == 3 and plan.selected_requests == 0
    assert (profile.snapshot(), profile.stats()) == before


def test_exact_integer_ranking_does_not_collapse_large_nearby_counts():
    profile = ExpertUsageProfile()
    profile.record_request("slightly_less", transfer_bytes=1, dense_bytes=1, count=2**60)
    profile.record_request("slightly_more", transfer_bytes=1, dense_bytes=1, count=2**60 + 1)
    assert profile.plan(1).keys == ("slightly_more",)
    assert profile.plan(1).saved_transfer_bytes_proxy == 2**60 + 1


def test_exact_large_byte_budget_never_rounds_up_an_expert():
    profile = ExpertUsageProfile()
    profile.record_request("too_large", transfer_bytes=2**60, dense_bytes=2**60 + 1)
    assert profile.plan(2**60).keys == ()
    assert profile.plan(2**60 + 1).payload_bytes == 2**60 + 1


def test_snapshot_is_immutable_and_can_reconstruct_the_same_plan():
    profile = ExpertUsageProfile()
    profile.record_request(("path", 4), transfer_bytes=3, dense_bytes=6, count=2)
    profile.record_request(("path", 8), transfer_bytes=9, dense_bytes=10, count=3)
    snapshot = profile.snapshot()
    with pytest.raises(FrozenInstanceError):
        snapshot[0].requests = 999
    clone = ExpertUsageProfile()
    for record in snapshot:
        clone.record_request(record.key, count=record.requests,
                             transfer_bytes=record.transfer_bytes, dense_bytes=record.dense_bytes)
    assert clone.snapshot() == snapshot
    assert clone.plan(16) == profile.plan(16)
    profile.record_request(("path", 4), transfer_bytes=3, dense_bytes=6)
    assert snapshot[0].requests == 2


def test_plan_remains_immutable_after_more_profile_observations():
    profile = ExpertUsageProfile()
    profile.record_request("old", transfer_bytes=2, dense_bytes=4)
    plan = profile.plan(4)
    profile.record_request("new", transfer_bytes=2, dense_bytes=4, count=100)
    assert plan.keys == ("old",) and plan.contains("old") and not plan.contains("new")
    with pytest.raises(FrozenInstanceError):
        plan.payload_bytes = 100
    assert profile.plan(4).keys == ("new",)


@pytest.mark.parametrize("field", ["count", "transfer_bytes", "dense_bytes"])
@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.5, "1", None])
def test_invalid_request_metadata_is_atomic(field, bad):
    profile = ExpertUsageProfile()
    profile.record_request("existing", transfer_bytes=2, dense_bytes=4)
    before = profile.snapshot(), profile.stats()
    arguments = {"transfer_bytes": 2, "dense_bytes": 4, "count": 1, field: bad}
    with pytest.raises(ValueError):
        profile.record_request("new", **arguments)
    assert (profile.snapshot(), profile.stats()) == before


@pytest.mark.parametrize("bad", [True, False, -1, 1.5, "1", None])
def test_invalid_device_budget_refuses_without_changing_profile(bad):
    profile = ExpertUsageProfile()
    profile.record_request("expert", transfer_bytes=2, dense_bytes=4)
    before = profile.snapshot()
    with pytest.raises(ValueError):
        profile.plan(bad)
    assert profile.snapshot() == before


@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.5, "1", None])
def test_invalid_profile_entry_ceiling_is_refused(bad):
    with pytest.raises(ValueError):
        ExpertUsageProfile(max_entries=bad)


def test_profile_metadata_bound_refuses_new_key_but_allows_existing_demand():
    profile = ExpertUsageProfile(max_entries=1)
    profile.record_request("first", transfer_bytes=2, dense_bytes=4)
    before = profile.snapshot(), profile.stats()
    with pytest.raises(OverflowError):
        profile.record_request("second", transfer_bytes=2, dense_bytes=4)
    assert (profile.snapshot(), profile.stats()) == before
    profile.record_request("first", transfer_bytes=2, dense_bytes=4)
    assert profile.stats() == {"entries": 1, "max_entries": 1, "total_requests": 2}


@pytest.mark.parametrize("transfer,dense", [(3, 4), (2, 5)])
def test_same_key_cannot_change_materialization_costs(transfer, dense):
    profile = ExpertUsageProfile()
    profile.record_request("expert", transfer_bytes=2, dense_bytes=4)
    before = profile.snapshot(), profile.stats()
    with pytest.raises(ValueError, match="metadata changed"):
        profile.record_request("expert", transfer_bytes=transfer, dense_bytes=dense)
    assert (profile.snapshot(), profile.stats()) == before


def test_unhashable_key_refuses_without_partial_request_count():
    profile = ExpertUsageProfile()
    with pytest.raises(TypeError):
        profile.record_request([], transfer_bytes=2, dense_bytes=4)
    assert profile.stats()["total_requests"] == profile.stats()["entries"] == 0


def test_observation_and_planning_do_not_mutate_existing_cache_metadata():
    import torch

    from soup_cli.utils.expert_cache import TensorHotset

    cache = TensorHotset(16, copy_on_insert=True)
    keys = (("path", 8), ("path", 4))
    original = torch.tensor([1., 2.])
    for key in keys:
        assert cache.put(key, {"weight": original})
    assert cache.get(keys[0]) is not None
    before = cache.stats()
    identities = tuple(cache.peek(key) for key in keys)
    profile = ExpertUsageProfile()
    for key in keys:
        profile.record_request(key, transfer_bytes=2, dense_bytes=4)
    profile.plan(4)
    assert cache.stats() == before
    assert all(cache.peek(key) is entry for key, entry in zip(keys, identities))
    assert all(torch.equal(cache.peek(key)["weight"], original) for key in keys)
