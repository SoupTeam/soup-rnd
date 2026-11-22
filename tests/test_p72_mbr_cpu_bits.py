"""The tiny-MoE CPU bit check that keeps MBR regression coverage off the GPU.

The seam is the whole step through ``L2LStep.run`` on a fixed tiny MoE (spec: Testing
Decisions). ``A0`` is plain L2L. ``C`` is route-preserving MBR with the micro-batch
segments of the exact expert plan (the extension's Python stand-in on CPU) and the
two-phase backward. Losses and every trainable gradient are compared with
``torch.equal`` at k = 1, 2, 4, so a change in either path shows up in CI without a
card.

CPU native GEMM is what the stand-in runs: the fused backend computes the same
per-segment native calls the extension makes, which is why this checks the plan and
the node plumbing rather than cuBLASLt itself.
"""

from __future__ import annotations

from typing import Any, Callable, List

import pytest

from benchmarks.harness import b13_moe_fixture as fixture
from soup_cli.utils import l2l_step
from soup_cli.utils.expert_gemm import expert_gemm_context
from tests.test_b13_l2l_moe import Rig
from tests.test_b13_l2l_moe import make_rig as make_rig  # noqa: F401  (pytest fixture)
from tests.test_direct_expert_backward_mode import FusedCpuBackend
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR
from tests.test_p72_mbr_bitexact import LENGTHS, run_window

pytestmark = _REQUIRES_TRAINING_FLOOR

GROUPS = (1, 2, 4)


def _micro_batches(rig: Rig, count: int) -> List[Any]:
    """``count`` different micro-batches; the ragged lengths repeat after three."""
    return [
        fixture.moe_batch(rig.config, seed=seed, rows=1, seq=LENGTHS[seed % len(LENGTHS)])[
            "input_ids"
        ]
        for seed in range(count)
    ]


def _different(left: Any, right: Any) -> List[str]:
    import torch

    bad = [
        f"loss[{index}]"
        for index, (a, b) in enumerate(zip(left["losses"], right["losses"]))
        if not torch.equal(a, b)
    ]
    bad += [
        name for name in left["grads"]
        if not torch.equal(left["grads"][name], right["grads"][name])
    ]
    return bad


@pytest.mark.parametrize("group", GROUPS)
def test_a0_equals_c_on_cpu_at_group_1_2_and_4(
    make_rig: Callable[..., Rig], group: int
) -> None:
    rig = make_rig(
        k=group,
        rows=1,
        seq=6,
        build_options={"two_read_experts": True, "grouped_expert_routes": True},
    )
    ids = _micro_batches(rig, group)
    calls = [l2l_step.capture_layer_call(rig.model, item) for item in ids]

    a0 = run_window(rig, ids, calls, mbr=False, group_size=None, dtype="fp32")
    backend = FusedCpuBackend()
    with expert_gemm_context("lt-fused", backend=backend, require_cuda=False):
        c = run_window(rig, ids, calls, mbr=True, group_size=group, dtype="fp32",
                       split_backward=True)

    assert backend.forward_calls > 0 and backend.backward_calls > 0
    assert backend.registered, "the exact plan registered no algo for a segment"
    assert _different(a0, c) == []
