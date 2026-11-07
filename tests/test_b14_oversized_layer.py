"""B1.4: a layer larger than the weight budget streams through one bounded slot.

Seam: ``plan_expert_buffer`` + ``ExpertBufferPool`` + ``streamed_experts_class``.
The VRAM limit is simulated by a ``weight_budget_bytes`` chosen from the layer size, not
from the plan. Real-GPU numbers belong to the B1.4 report, not to this file.
"""

from __future__ import annotations

import pytest

from tests.test_b1_expert_forward import _experts, _store, _streamed

EXPERTS = 64
HIDDEN_SIZE = 32  # fixed by ``_experts``


def _routes_hitting_every_expert():
    import torch

    gen = torch.Generator().manual_seed(11)
    hidden = torch.randn(EXPERTS, HIDDEN_SIZE, generator=gen)
    ids = torch.arange(EXPERTS)
    index = torch.stack([ids, (ids + 1) % EXPERTS], dim=1)
    weights = torch.rand(EXPERTS, 2, generator=gen)
    return hidden, index, weights


def _largest_tensor_recorder():
    """Record the byte size of the largest tensor any op returns while it is active."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves

    class Recorder(TorchDispatchMode):
        largest = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for leaf in tree_leaves(out):
                if isinstance(leaf, torch.Tensor):
                    self.largest = max(self.largest, leaf.numel() * leaf.element_size())
            return out

    return Recorder()


def test_layer_larger_than_limit_matches_resident_without_a_layer_sized_tensor(tmp_path):
    import torch

    from soup_cli.utils.expert_stream import ExpertBufferPool

    module, _ = _experts(count=EXPERTS)
    reader, first = _store(tmp_path, module)
    pool = None
    try:
        layer_bytes = first.plan.dense_bytes * EXPERTS
        slot_bytes = first.plan.weight_bytes
        first.close()
        limit = layer_bytes // 4
        assert slot_bytes * 8 < limit, "one slot must sit well inside a limit below the layer"
        pool = ExpertBufferPool(
            reader, device="cpu", dtype="float32", weight_budget_bytes=limit,
        )
        streamed = _streamed(module, reader, pool)
        args = _routes_hitting_every_expert()
        recorder = _largest_tensor_recorder()
        with torch.no_grad():
            want = module(*args)
            with recorder:
                got = streamed(*args)
        assert torch.equal(got, want)
        assert pool.fetches == reader.fetches == EXPERTS
        assert pool.plan.weight_bytes == slot_bytes <= limit < layer_bytes
        # The largest tensor is the router's one-hot mask (64, 2, 64), 65536 bytes: routing
        # bookkeeping, not weights. No tensor reaches the limit. Many small copies that
        # add up to a layer would not be caught here.
        assert recorder.largest < limit
    finally:
        reader.close()
        if pool is not None:
            pool.close()


def test_recorder_sees_a_layer_sized_tensor(tmp_path):
    """Negative control: the recorder is not vacuous, it reports a full-layer copy."""
    module, _ = _experts(count=EXPERTS)
    layer_bytes = module.gate_up_proj.numel() * module.gate_up_proj.element_size()
    recorder = _largest_tensor_recorder()
    with recorder:
        module.gate_up_proj.clone()
    assert recorder.largest >= layer_bytes


def test_budget_below_one_slot_is_refused_for_the_oversized_layer(tmp_path):
    from soup_cli.utils.expert_stream import ExpertBufferPool

    module, _ = _experts(count=EXPERTS)
    reader, first = _store(tmp_path, module)
    try:
        slot_bytes = first.plan.weight_bytes
        first.close()
        with pytest.raises(ValueError, match="slot needs"):
            ExpertBufferPool(
                reader, device="cpu", dtype="float32", weight_budget_bytes=slot_bytes - 1,
            )
    finally:
        reader.close()
