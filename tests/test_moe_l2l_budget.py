"""CPU checks for the MoE L2L host-memory plan."""

import pytest

from soup_cli.utils.host_memory_budget import HostMemoryBudget
from soup_cli.utils.moe_l2l_budget import plan_moe_l2l

PLAN_ARGS = {
    "n_layers": 3,
    "n_moe_layers": 2,
    "max_shape": (2, 5, 7),
    "experts_per_token": 3,
    "requested_k": 4,
    "dtype": "bfloat16",
    "pin": False,
}


@pytest.mark.parametrize(
    ("dtype", "pin", "expected_activations"),
    [
        ("bfloat16", False, 1680),
        ("float32", False, 3360),
        ("bfloat16", True, 3072),
        ("float32", True, 6144),
    ],
)
def test_exact_fit_keeps_requested_window(dtype, pin, expected_activations):
    # Each routed layer retains 10 tokens x 3 int64 expert indices per micro-batch.
    # All eight records can be live while one more is copied for the first comparison.
    args = {**PLAN_ARGS, "dtype": dtype, "pin": pin}
    budget = HostMemoryBudget(expected_activations + 2160)

    plan = plan_moe_l2l(budget, **args)

    assert plan.effective_k == plan.requested_k == 4
    assert plan.activation_bytes == expected_activations
    assert plan.routing_record_bytes == 1920
    assert plan.routing_comparison_bytes == 240
    assert plan.host_bytes == plan.available_host_bytes == budget.limit_bytes
    assert budget.used_bytes == 0


def test_selection_uses_remaining_budget_and_all_integer_window_sizes():
    budget = HostMemoryBudget(40_000, reserved_bytes=961)
    staging = budget.reserve("reader staging", 4600)
    args = {**PLAN_ARGS, "requested_k": 64}

    plan = plan_moe_l2l(budget, **args)

    # 38 micro-batches need 34,440 bytes. The existing consumers leave one byte less.
    assert plan.requested_k == 64
    assert plan.effective_k == 37
    assert plan.available_host_bytes == 34_439
    assert plan.host_bytes == 33_540
    assert budget.used_bytes == 5561
    staging.close()
    assert plan_moe_l2l(budget, **args).effective_k == 43
    assert plan_moe_l2l(budget, **PLAN_ARGS).effective_k == 4


@pytest.mark.parametrize("available", [0, 1139])
def test_rejects_budget_too_small_for_one_microbatch(available):
    budget = HostMemoryBudget(1140, reserved_bytes=1140 - available)

    with pytest.raises(MemoryError, match=rf"k=1 needs 1140 host bytes.*{available} available"):
        plan_moe_l2l(budget, **PLAN_ARGS)

    assert budget.available_bytes == available


@pytest.mark.parametrize(
    "invalid",
    [
        {"requested_k": True},
        {"max_shape": (2, 5)},
        {"max_shape": (2, 0, 7)},
        {"n_moe_layers": 4},
        {"experts_per_token": 0},
        {"dtype": "float16"},
    ],
)
def test_rejects_unsupported_dimensions_and_dtype(invalid):
    with pytest.raises(ValueError):
        plan_moe_l2l(HostMemoryBudget(10_000), **{**PLAN_ARGS, **invalid})


def test_plan_covers_cpu_store_and_route_tensors_under_one_budget():
    import torch

    from soup_cli.utils.budgeted_activations import BudgetedActivationStore

    budget = HostMemoryBudget(3840)
    plan = plan_moe_l2l(budget, **PLAN_ARGS)
    routing = budget.reserve(
        "routing", plan.routing_record_bytes + plan.routing_comparison_bytes
    )
    records = comparison = None
    try:
        records = torch.empty((plan.effective_k, 2, 10, 3), dtype=torch.int64)
        comparison = torch.empty((10, 3), dtype=torch.int64)
        assert records.untyped_storage().nbytes() == plan.routing_record_bytes
        assert comparison.untyped_storage().nbytes() == plan.routing_comparison_bytes
        store = BudgetedActivationStore(
            host_budget=budget,
            n_layers=3,
            k=plan.effective_k,
            chunk_shape=(2, 5, 7),
            dtype=torch.bfloat16,
            device="cpu",
            pin=False,
        )
        try:
            activation = torch.ones((2, 5, 7), dtype=torch.bfloat16)
            store.put(2, 3, activation)
            assert torch.equal(store.get(2, 3), activation)
            assert budget.used_bytes == plan.host_bytes
            assert budget.available_bytes == 0
        finally:
            store.close()
    finally:
        del records, comparison
        routing.close()
    assert budget.used_bytes == 0
