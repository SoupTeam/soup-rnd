"""``mbr_ordered_backward``: one engine call per group, LoRA gradients summed in micro-batch order.

The two-phase backward (``mbr_split_backward``) gets L2L's gradient order by running the rest of
each micro-batch's backward in its own engine call, 1 + group-size calls per layer. Autograd
runs ready nodes with the highest sequence number first, so a group recomputed from the last
micro-batch to the first makes the engine meet micro-batch 0's nodes first and add the
contributions of every LoRA weight in micro-batch order, in one call. These tests check, on CPU
with the segmented expert GEMM, that the gradients equal L2L's and that the engine is entered
once per group.
"""

from __future__ import annotations

from typing import Any, Callable, List

import pytest
import torch

from benchmarks.harness import b13_moe_fixture as fixture
from benchmarks.harness.p72_invariant_expert_gemm import segmented_expert_gemm
from soup_cli.utils import l2l_step
from soup_cli.utils.l2l_step import L2LStep
from tests.test_b13_l2l_moe import Rig
from tests.test_b13_l2l_moe import make_rig as make_rig  # noqa: F401  (pytest fixture)
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR
from tests.test_p72_mbr_bitexact import LENGTHS, differing, run_window


def window(make_rig: Callable[..., Rig], k: int = 3) -> Any:
    rig = make_rig(
        k=k, rows=1, seq=6,
        build_options={"two_read_experts": True, "grouped_expert_routes": True},
    )
    ids = [
        fixture.moe_batch(rig.config, seed=seed, rows=1, seq=LENGTHS[seed % len(LENGTHS)])[
            "input_ids"
        ]
        for seed in range(k)
    ]
    calls = [l2l_step.capture_layer_call(rig.model, item) for item in ids]
    return rig, ids, calls


@_REQUIRES_TRAINING_FLOOR
@pytest.mark.parametrize("k,dtype", [(3, "fp32"), (5, "fp32"), (3, "bf16_nocache"),
                                     (8, "bf16_nocache")])
def test_ordered_backward_equals_l2l(make_rig: Callable[..., Rig], k: int, dtype: str) -> None:
    rig, ids, calls = window(make_rig, k=k)
    l2l = run_window(rig, ids, calls, mbr=False, group_size=None, dtype=dtype)
    with segmented_expert_gemm():
        ordered = run_window(rig, ids, calls, mbr=True, group_size=k, dtype=dtype,
                             ordered_backward=True)
    assert differing(l2l, ordered) == []


@_REQUIRES_TRAINING_FLOOR
def test_without_ordering_or_splitting_the_gradient_order_differs(
    make_rig: Callable[..., Rig],
) -> None:
    """Control: the check can go red when neither the split nor the ordered recompute is on
    (CPU fp32: 20 LoRA tensors differ in the unordered single-call backward)."""
    rig, ids, calls = window(make_rig)
    l2l = run_window(rig, ids, calls, mbr=False, group_size=None, dtype="fp32")
    with segmented_expert_gemm():
        plain = run_window(rig, ids, calls, mbr=True, group_size=3, dtype="fp32")
    assert differing(l2l, plain) != []


@_REQUIRES_TRAINING_FLOOR
def test_ordered_backward_enters_the_engine_once_per_group_and_layer(
    make_rig: Callable[..., Rig],
) -> None:
    rig, ids, calls = window(make_rig)
    calls_made: List[int] = []
    original = torch.autograd.backward

    def counting(tensors: Any, *args: Any, **kwargs: Any) -> Any:
        calls_made.append(len(tensors) if isinstance(tensors, (list, tuple)) else 1)
        return original(tensors, *args, **kwargs)

    torch.autograd.backward = counting
    try:
        with segmented_expert_gemm():
            run_window(rig, ids, calls, mbr=True, group_size=3, dtype="fp32",
                       ordered_backward=True)
        ordered = len(calls_made)
        calls_made.clear()
        with segmented_expert_gemm():
            run_window(rig, ids, calls, mbr=True, group_size=3, dtype="fp32",
                       split_backward=True)
        split = len(calls_made)
    finally:
        torch.autograd.backward = original
    moe_layers = sum(1 for layer in rig.stepper.layers
                     if l2l_step._streamed_moe_experts(layer) is not None)
    assert split - ordered == 3 * moe_layers, (ordered, split, moe_layers)


def test_ordered_and_split_backward_exclude_each_other() -> None:
    step = L2LStep.__new__(L2LStep)
    with pytest.raises(ValueError, match="mbr_ordered_backward"):
        L2LStep.run(step, [], None, route_preserving_mbr=True, mbr_group_size=2,
                    mbr_split_backward=True, mbr_ordered_backward=True)
    with pytest.raises(ValueError, match="mbr_ordered_backward"):
        L2LStep.run(step, [], None, mbr_ordered_backward=True)
