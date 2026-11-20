"""P72 MBR batch-invariance, ticket 06 diagnosis: the S1 pair on a small CUDA rig.

The Moonlight series (ticket 06) found L2L and MBR with the row-wise expert GEMM plus the two-phase
backward equal at group 1 but not at groups 2, 4 and 8, while the CPU rig of ticket 01 is equal.
This module runs the CPU scenario of ``test_p72_mbr_bitexact`` on a CUDA device, in seconds, so the
operator that depends on the group size can be found. Same criterion (plan D4): no tolerances,
integer views. Run in a fresh CUDA process:
``pytest -m gpu --no-cov tests/test_p72_mbr_bitexact_cuda.py``.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

import pytest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from benchmarks.harness.p72_invariant_expert_gemm import rowwise_expert_linear  # noqa: E402
from soup_cli.utils import l2l_step  # noqa: E402
from tests.test_b13_l2l_moe import (  # noqa: E402
    Rig,
    _collect,
    _zero_grads,
    micro_batches,
)
from tests.test_b13_l2l_moe import make_rig as make_rig  # noqa: E402,F401  (pytest fixture)
from tests.test_p72_mbr_bitexact import (  # noqa: E402
    Result,
    differing,
    observe,
)

pytestmark = pytest.mark.gpu

WINDOW = 4  # micro-batches; groups 1, 2, 4 divide it


def run_cuda_window(
    rig: Rig,
    ids: List[Any],
    calls: List[Any],
    *,
    mbr: bool,
    group_size: Optional[int],
    bf16: bool,
    split_backward: bool,
) -> Result:
    import torch

    _zero_grads(rig.model)
    kwargs: Dict[str, Any] = {"route_preserving_mbr": mbr}
    if mbr:
        kwargs.update(mbr_group_size=group_size, collect_phase_stats=True)
    if split_backward:
        kwargs["mbr_split_backward"] = True
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        losses = rig.stepper.run(ids, rig.store, calls=calls, **kwargs)
    collected = _collect(rig.model)
    assert collected and not [n for n, g in collected.items() if g is None]
    return {
        "losses": [loss.detach().clone() for loss in losses],
        "grads": {name: grad.clone() for name, grad in collected.items()},
    }


@pytest.fixture()
def cuda_window(make_rig: Callable[..., Rig]) -> Any:
    rig = make_rig(
        k=WINDOW,
        rows=1,
        seq=8,
        device="cuda",
        build_options={"two_read_experts": True, "grouped_expert_routes": True},
    )
    ids = micro_batches(rig, WINDOW)
    calls = [l2l_step.capture_layer_call(rig.model, item) for item in ids]
    return rig, ids, calls


@pytest.mark.parametrize("bf16", (False, True), ids=("fp32", "bf16"))
def test_cuda_control_l2l_equals_mbr_group_1_with_rowwise_experts(
    cuda_window: Any, bf16: bool
) -> None:
    rig, ids, calls = cuda_window
    with rowwise_expert_linear() as rows:
        l2l = run_cuda_window(rig, ids, calls, mbr=False, group_size=None, bf16=bf16,
                              split_backward=False)
        group1 = run_cuda_window(rig, ids, calls, mbr=True, group_size=1, bf16=bf16,
                                 split_backward=True)
    assert rows[0] > 0
    assert observe(f"cuda L2L vs MBR split g1 row-wise [bf16={bf16}]", l2l, group1)


@pytest.mark.parametrize("group", (2, 4))
@pytest.mark.parametrize("bf16", (False, True), ids=("fp32", "bf16"))
def test_cuda_l2l_equals_split_backward_grouped_with_rowwise_experts(
    cuda_window: Any, bf16: bool, group: int
) -> None:
    """The S1 pair on a small CUDA rig (red on Moonlight at groups 2, 4 and 8)."""
    rig, ids, calls = cuda_window
    with rowwise_expert_linear() as rows:
        l2l = run_cuda_window(rig, ids, calls, mbr=False, group_size=None, bf16=bf16,
                              split_backward=False)
        grouped = run_cuda_window(rig, ids, calls, mbr=True, group_size=group, bf16=bf16,
                                  split_backward=True)
    assert rows[0] > 0
    assert observe(f"cuda L2L vs MBR split g{group} row-wise [bf16={bf16}]", l2l, grouped), (
        differing(l2l, grouped)[:5]
    )


def test_cuda_route_weight_gradient_depends_on_rows_natively_not_rowwise() -> None:
    """Root cause of the Moonlight S1 failure (ticket 06): the gradient of the routing weight.

    ``expert_mlp(...) * weights`` backs into a fp32 sum over the hidden axis whose order CUDA
    picks from M, so one row differs between M=1 and M=2. The row-wise swap makes it M=1 per row.
    """
    import torch

    from tests.test_p72_mbr_bitexact import _expert_node_function

    torch.manual_seed(0)
    hidden, go = (torch.randn(64, 2048, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    node = _expert_node_function("backward")

    def per_row_reference() -> Any:
        rows = []
        for i in range(64):
            weight = torch.rand(1, 1, device="cuda", dtype=torch.float32, requires_grad=True)
            (grad,) = torch.autograd.grad(
                hidden[i : i + 1] * weight, weight, go[i : i + 1]
            )
            rows.append(grad)
        return torch.cat(rows)

    reference = per_row_reference()

    def grouped_grad() -> Any:
        weight = torch.rand(64, 1, device="cuda", dtype=torch.float32, requires_grad=True)
        (grad,) = torch.autograd.grad(node(hidden, weight), weight, go)
        return grad

    assert not torch.equal(grouped_grad(), reference), "native sum is already M-invariant here"
    with rowwise_expert_linear():
        assert torch.equal(grouped_grad(), reference)
