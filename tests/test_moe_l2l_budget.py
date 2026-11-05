"""CPU checks for the MoE L2L host-memory plan."""

import pytest

from soup_cli.utils.host_memory_budget import HostMemoryBudget
from soup_cli.utils.moe_l2l_budget import plan_moe_l2l
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR

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


@_REQUIRES_TRAINING_FLOOR
def test_plan_covers_native_deepseek_l2l_buffers(tmp_path, monkeypatch):
    import torch

    from benchmarks.harness import b13_moe_fixture
    from soup_cli.utils import l2l_step
    from soup_cli.utils.budgeted_activations import BudgetedActivationStore

    rig = b13_moe_fixture.build_fixture(tmp_path)
    peak_records = peak_comparison = peak_routes = 0

    class ObservedGuard(l2l_step.RoutingGuard):
        def _watch(self, module, args, kwargs):
            nonlocal peak_records, peak_comparison, peak_routes
            if self._visit is not None and not self._visit[2]:
                # The first comparison still has all saved routes plus the new route.
                indices = kwargs["top_k_index"] if "top_k_index" in kwargs else args[1]
                comparison = indices.untyped_storage().nbytes()
                peak_comparison = max(peak_comparison, comparison)
                peak_routes = max(peak_routes, self.record_bytes() + comparison)
            super()._watch(module, args, kwargs)
            peak_records = max(peak_records, self.record_bytes())

        def record_bytes(self):
            return sum(indices.untyped_storage().nbytes() for indices in self._records.values())

    monkeypatch.setattr(l2l_step, "RoutingGuard", ObservedGuard)
    try:
        config = rig.config
        shape = (2, 6, config.hidden_size)
        budget = HostMemoryBudget(20_160)  # Fits four micro-batches and one comparison route.
        plan = plan_moe_l2l(
            budget,
            n_layers=config.num_hidden_layers,
            n_moe_layers=config.num_hidden_layers - config.first_k_dense_replace,
            max_shape=shape,
            experts_per_token=config.num_experts_per_tok,
            requested_k=64,
            dtype="float32",
            pin=False,
        )
        assert plan.effective_k == 4
        batches, labels = b13_moe_fixture.micro_batches(
            config, plan.effective_k, first=0, device="cpu"
        )
        step = l2l_step.L2LStep(
            rig.streamed, rig.runtime, l2l_step.capture_layer_call(rig.streamed, batches[0])
        )
        store = BudgetedActivationStore(
            host_budget=budget,
            n_layers=config.num_hidden_layers,
            k=plan.effective_k,
            chunk_shape=shape,
            dtype=torch.float32,
            device="cpu",
            pin=False,
        )
        try:
            n_items = sum(int((label[:, 1:] != -100).sum()) for label in labels)
            losses = step.run(batches, store, labels=labels, n_items=n_items)
            assert all(loss.isfinite() for loss in losses)
            activation_bytes = sum(
                tensor.untyped_storage().nbytes() for tensor in store._resident.values()
            )
            assert activation_bytes == plan.activation_bytes == budget.used_bytes
            assert peak_records == plan.routing_record_bytes
            assert peak_comparison == plan.routing_comparison_bytes
            assert activation_bytes + peak_routes == plan.host_bytes == budget.limit_bytes
        finally:
            store.close()
        assert budget.used_bytes == 0
    finally:
        rig.close()
