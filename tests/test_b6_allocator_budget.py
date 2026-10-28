"""CUDA block rounding must be budgeted, with an observed hard check retained."""

import pytest


@pytest.mark.parametrize("reserve,overhead,accepted", [
    (False, 1024, False), (True, 1024, True), (True, 33 * 2**20, False),
])
def test_observed_rounding_is_checked_against_explicit_budget(
    tmp_path, monkeypatch, reserve, overhead, accepted
):
    import torch

    from soup_cli.utils.expert_stream import ExpertBufferPool, expert_allocation_budget
    from tests.test_b1_expert_forward import _experts, _store

    module, _ = _experts()
    reader, initial = _store(tmp_path, module)
    plan = initial.plan
    initial.close()
    original_empty = torch.empty

    def cpu_storage(*args, **kwargs):
        if str(kwargs.get("device", "")).startswith("cuda"):
            kwargs["device"] = "cpu"
        return original_empty(*args, **kwargs)

    samples = iter([0, plan.weight_bytes + overhead])
    monkeypatch.setattr(torch, "empty", cpu_storage)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *args: next(samples))
    monkeypatch.setattr(ExpertBufferPool, "_synchronize", lambda self: None)
    budget = (expert_allocation_budget(plan, device="cuda:0") if reserve
              else plan.weight_bytes)
    try:
        if not accepted:
            with pytest.raises(ValueError, match="rounded the expert slot.*observed"):
                ExpertBufferPool(reader, device="cuda:0", dtype="float32",
                                 weight_budget_bytes=budget)
        else:
            pool = ExpertBufferPool(reader, device="cuda:0", dtype="float32",
                                    weight_budget_bytes=budget)
            assert pool.cuda_allocation_delta <= budget
            assert pool._arena.untyped_storage().nbytes() == plan.weight_bytes
            pool.close()
    finally:
        reader.close()
