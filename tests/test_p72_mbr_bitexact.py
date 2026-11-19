"""P72 MBR batch-invariance, ticket 01: is route-preserving MBR bit-equal to L2L on CPU?

Seam: ``make_rig`` and the helpers of ``tests/test_b13_l2l_moe.py``. The scenario is the ragged
three-micro-batch window of ``test_mbr_ragged_streamed_window_matches_microbatch_dispatch``
(lengths 3, 6, 4), run once per variant.

Criterion (plan D4): no tolerances. The loss and every LoRA gradient are compared through their
integer view, so ``-0.0`` and ``0.0`` differ. A red outcome is an observation, not a bug of the
test: each ``observe`` call records it (prints it, and appends it to ``$P72_BITEXACT_LOG`` when
set) and the test asserts only what is known in advance:

* the control (MBR group 1 against plain L2L) must be equal, otherwise the harness is wrong;
* if group 3 is not equal to group 1, the row-wise swap must actually run (the outcome of the
  repaired comparison is recorded, not asserted: on CPU it stays red, see the ticket comments).
"""

from __future__ import annotations

import json
import os
import re
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional

import pytest
from rich.console import Console

from benchmarks.harness import b13_moe_fixture as fixture
from benchmarks.harness.p72_invariant_expert_gemm import rowwise_expert_linear
from soup_cli.utils import l2l_step
from tests.test_b13_l2l_moe import Rig, _collect, _zero_grads
from tests.test_b13_l2l_moe import make_rig as make_rig  # noqa: F401  (pytest fixture)
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR

pytestmark = _REQUIRES_TRAINING_FLOOR

LENGTHS = (3, 6, 4)
DTYPES = ("fp32", "bf16_autocast")

_LAYER = re.compile(r"layers\.\d+")
_INT_VIEW = {2: "int16", 4: "int32", 8: "int64"}


def bit_view(tensor: Any) -> Any:
    """The tensor's bits as integers, so -0.0 != 0.0 and NaN payloads count."""
    import torch

    flat = tensor.detach().contiguous()
    return flat.view(getattr(torch, _INT_VIEW[flat.element_size()]))


def bits_equal(left: Any, right: Any) -> bool:
    import torch

    return left.shape == right.shape and left.dtype == right.dtype and bool(
        torch.equal(bit_view(left), bit_view(right))
    )


Result = Dict[str, Any]  # "losses": list of tensors, "grads": name -> tensor


def differing(left: Result, right: Result) -> List[str]:
    """Names of the loss entries and LoRA gradients that are not bit-equal."""
    assert len(left["losses"]) == len(right["losses"])
    assert left["grads"].keys() == right["grads"].keys() and left["grads"]
    bad = [
        f"loss[{i}]"
        for i, (a, b) in enumerate(zip(left["losses"], right["losses"]))
        if not bits_equal(a, b)
    ]
    bad += [
        name for name in left["grads"]
        if not bits_equal(left["grads"][name], right["grads"][name])
    ]
    return bad


def observe(row: str, left: Result, right: Result) -> bool:
    """Record one outcome-table row; return whether the two results are bit-equal."""
    bad = differing(left, right)
    record = {
        "row": row,
        "bit_equal": not bad,
        "tensors_compared": len(left["losses"]) + len(left["grads"]),
        "tensors_different": len(bad),
        "first_different": bad[:3],
        "different_layers": sorted({m.group(0) for n in bad if (m := _LAYER.search(n))}),
    }
    line = "P72_BITEXACT " + json.dumps(record, sort_keys=True)
    Console(stderr=True).print(line, markup=False, highlight=False, soft_wrap=True)
    path = os.environ.get("P72_BITEXACT_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return not bad


def _autocast(dtype: str) -> Any:
    import torch

    if dtype == "fp32":
        return nullcontext()
    # "bf16_nocache": the autocast weight cache shares one bf16 cast of each LoRA weight across
    # the micro-batches of a window, so their gradients are summed in bf16 before the fp32 add.
    return torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=dtype != "bf16_nocache")


def run_window(
    rig: Rig,
    ids: List[Any],
    calls: List[Any],
    *,
    mbr: bool,
    group_size: Optional[int],
    dtype: str,
    split_backward: bool = False,
) -> Result:
    _zero_grads(rig.model)
    kwargs: Dict[str, Any] = {"route_preserving_mbr": mbr}
    if mbr:
        kwargs.update(mbr_group_size=group_size, collect_phase_stats=True)
    if split_backward:
        kwargs["mbr_split_backward"] = True
    with _autocast(dtype):
        losses = rig.stepper.run(ids, rig.store, calls=calls, **kwargs)
    collected = _collect(rig.model)
    missing = [name for name, grad in collected.items() if grad is None]
    assert collected and not missing, f"LoRA parameters without a gradient: {missing[:3]}"
    grads = {name: grad.clone() for name, grad in collected.items()}
    return {"losses": [loss.detach().clone() for loss in losses], "grads": grads}


@pytest.fixture()
def window(make_rig: Callable[..., Rig]) -> Any:
    rig = make_rig(
        k=3,
        rows=1,
        seq=6,
        build_options={"two_read_experts": True, "grouped_expert_routes": True},
    )
    ids = [
        fixture.moe_batch(rig.config, seed=seed, rows=1, seq=seq)["input_ids"]
        for seed, seq in enumerate(LENGTHS)
    ]
    calls = [l2l_step.capture_layer_call(rig.model, item) for item in ids]
    return rig, ids, calls


def test_helper_distinguishes_signed_zero_and_ulp() -> None:
    import torch

    assert bits_equal(torch.tensor([1.0]), torch.tensor([1.0]))
    assert torch.equal(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert not bits_equal(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert not bits_equal(torch.tensor([1.0]), torch.tensor([1.0]).nextafter(torch.tensor([2.0])))
    assert not bits_equal(torch.zeros(1), torch.zeros(1, dtype=torch.bfloat16))


def test_rowwise_linear_is_scoped_to_expert_mlp() -> None:
    import torch
    import torch.nn.functional as functional

    x, w = torch.randn(5, 8), torch.randn(4, 8)
    native = functional.linear(x, w)
    with rowwise_expert_linear() as rows:
        assert bits_equal(functional.linear(x, w), native) and rows[0] == 0  # not from expert_mlp

        def expert_mlp(hidden: Any) -> Any:  # same function name the swap keys on
            return functional.linear(hidden, w)

        out = expert_mlp(x)
    assert rows[0] == 5 and out.shape == native.shape
    torch.testing.assert_close(out, native)
    assert functional.linear(x, w).shape == native.shape and bits_equal(
        functional.linear(x, w), native
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_control_l2l_equals_mbr_group_1_bitwise(window: Any, dtype: str) -> None:
    rig, ids, calls = window
    l2l = run_window(rig, ids, calls, mbr=False, group_size=None, dtype=dtype)
    group1 = run_window(rig, ids, calls, mbr=True, group_size=1, dtype=dtype)
    assert observe(f"control L2L vs MBR g1 [{dtype}]", l2l, group1)


@pytest.mark.parametrize("dtype", DTYPES)
def test_group_1_vs_3_bitwise_outcome_and_rowwise_fix(window: Any, dtype: str) -> None:
    rig, ids, calls = window
    group1 = run_window(rig, ids, calls, mbr=True, group_size=1, dtype=dtype)
    group3 = run_window(rig, ids, calls, mbr=True, group_size=3, dtype=dtype)
    if observe(f"MBR g1 vs g3 native [{dtype}]", group1, group3):
        return  # green: nothing to repair
    # Red: record whether a row-wise expert linear layer repairs it (ticket 01, third box).
    with rowwise_expert_linear() as rows:
        fixed1 = run_window(rig, ids, calls, mbr=True, group_size=1, dtype=dtype)
        fixed3 = run_window(rig, ids, calls, mbr=True, group_size=3, dtype=dtype)
    assert rows[0] > 0, "the row-wise swap never ran; the observation would be vacuous"
    observe(f"MBR g1 vs g3 row-wise [{dtype}]", fixed1, fixed3)


@pytest.mark.parametrize(
    ("dtype", "n_micro_batches"),
    [("fp32", 2), ("bf16_nocache", 2), ("bf16_nocache", 3)],
)
def test_rowwise_experts_group_is_bitwise_equal_when_accumulation_is_order_free(
    make_rig: Callable[..., Rig], dtype: str, n_micro_batches: int
) -> None:
    """Diagnosis (ticket 01): what is left red after the row-wise experts, and why.

    Every expert dX/dW and the expert block dIn/dOut were bit-equal between group 1 and 3 in
    the red runs, so the difference sits in how per-micro-batch LoRA gradients are summed:

    * fp32: a sum of two gradients is order-free, a sum of three is not (the 3-micro-batch red
      row of ``test_group_1_vs_3_...``).
    * bf16 autocast: the weight cache sums the bf16 gradients of the micro-batches before the
      fp32 add, which differs from group 1 even for two. With the cache off, the per-use
      gradients are bf16-rounded, so even three of them add exactly in fp32.
    """
    rig = make_rig(
        k=n_micro_batches, rows=1, seq=6,
        build_options={"two_read_experts": True, "grouped_expert_routes": True},
    )
    ids = [
        fixture.moe_batch(rig.config, seed=seed, rows=1, seq=seq)["input_ids"]
        for seed, seq in enumerate(LENGTHS[:n_micro_batches])
    ]
    calls = [l2l_step.capture_layer_call(rig.model, item) for item in ids]
    with rowwise_expert_linear() as rows:
        group1 = run_window(rig, ids, calls, mbr=True, group_size=1, dtype=dtype)
        grouped = run_window(rig, ids, calls, mbr=True, group_size=n_micro_batches, dtype=dtype)
    assert rows[0] > 0
    assert observe(
        f"MBR g1 vs g{n_micro_batches} row-wise [{dtype}]", group1, grouped
    )


# --------------------------------------------------------------------------------------
# ticket 03: two-phase MBR backward (opt-in) makes the LoRA accumulation order the same
# --------------------------------------------------------------------------------------
SPLIT_DTYPES = ("fp32", "bf16_nocache", "bf16_autocast")


@pytest.mark.parametrize("dtype", SPLIT_DTYPES)
def test_split_backward_group_1_equals_group_3_bitwise_with_rowwise_experts(
    window: Any, dtype: str
) -> None:
    rig, ids, calls = window
    with rowwise_expert_linear() as rows:
        group1 = run_window(rig, ids, calls, mbr=True, group_size=1, dtype=dtype,
                            split_backward=True)
        group3 = run_window(rig, ids, calls, mbr=True, group_size=3, dtype=dtype,
                            split_backward=True)
    assert rows[0] > 0
    assert observe(f"MBR split-backward g1 vs g3 row-wise [{dtype}]", group1, group3)


@pytest.mark.parametrize("dtype", SPLIT_DTYPES)
def test_l2l_equals_split_backward_group_3_bitwise_with_rowwise_experts(
    window: Any, dtype: str
) -> None:
    """The S1 pair on CPU: plain L2L against grouped MBR with the two-phase backward."""
    rig, ids, calls = window
    with rowwise_expert_linear() as rows:
        l2l = run_window(rig, ids, calls, mbr=False, group_size=None, dtype=dtype)
        group3 = run_window(rig, ids, calls, mbr=True, group_size=3, dtype=dtype,
                            split_backward=True)
    assert rows[0] > 0
    assert observe(f"L2L vs MBR split-backward g3 row-wise [{dtype}]", l2l, group3)


def test_split_backward_flag_is_validated(window: Any) -> None:
    rig, ids, calls = window
    with pytest.raises(ValueError, match="only valid with route_preserving_mbr"):
        rig.stepper.run(ids, rig.store, calls=calls, mbr_split_backward=True)
    with pytest.raises(ValueError, match="must be a bool"):
        rig.stepper.run(ids, rig.store, calls=calls, route_preserving_mbr=True,
                        mbr_split_backward=1)


def test_rowwise_swap_serves_exactly_the_expert_rows(window: Any) -> None:
    """Ticket 03: the counter equals the expert rows, so no other linear layer was swapped.

    Expected from the route record, not from the swap: each MoE layer sends ``route_id_slots``
    (tokens x top-k) rows through two expert linears (gate_up, down) once in the forward and
    once in the backward (two-read replay recomputes nothing).
    """
    rig, ids, calls = window
    with rowwise_expert_linear() as rows:
        run_window(rig, ids, calls, mbr=True, group_size=3, dtype="fp32")
    stats = rig.stepper.route_preserving_mbr_stats
    slots = sum(stats["layers"][layer]["route_id_slots"] for layer in rig.moe_layers)
    assert slots == 2 * 13 * rig.config.num_experts_per_tok
    assert rows[0] == slots * 2 * 2


def test_split_backward_refuses_deferred_decoder_hooks(window: Any) -> None:
    """A deferred hook may read LoRA-bearing tensors, which phase one would sum group-wide."""
    rig, ids, calls = window
    layer = rig.stepper.layers[next(iter(rig.moe_layers))]

    @l2l_step.mbr_deferred_forward_hook
    def hook(_module: Any, _args: Any, output: Any) -> Any:
        return output + 0.01

    handle = layer.register_forward_hook(hook)
    try:
        with pytest.raises(RuntimeError, match="mbr_split_backward does not support"):
            run_window(rig, ids, calls, mbr=True, group_size=3, dtype="fp32",
                       split_backward=True)
    finally:
        handle.remove()


# --------------------------------------------------------------------------------------
# ticket 06 diagnosis: the routing-weight multiply of the expert node is a batch-dependent
# reduction on CUDA (its gradient sums over the hidden axis in fp32 per row group)
# --------------------------------------------------------------------------------------
def _expert_node_function(name: str) -> Any:
    """A function named ``name`` that lives in ``soup_cli.utils.expert_stream``'s namespace."""
    scope: Dict[str, Any] = {"__name__": "soup_cli.utils.expert_stream"}
    exec(f"def {name}(a, b):\n    return a * b\n", scope)  # noqa: S102
    return scope[name]


@pytest.mark.parametrize("name", ("forward", "backward"))
def test_rowwise_route_weight_multiply_is_scoped_to_the_expert_node(name: str) -> None:
    import torch

    x, weight = torch.randn(5, 8), torch.rand(5, 1)
    native = x * weight
    with rowwise_expert_linear() as rows:
        assert bits_equal(x * weight, native) and rows[1] == 0  # not from the expert node
        out = _expert_node_function(name)(x, weight)
        assert rows[1] == 5 and bits_equal(out, native)
        other = _expert_node_function("helper")(x, weight)  # same module, other function
        assert rows[1] == 5 and bits_equal(other, native)
    assert bits_equal(x * weight, native)


def test_rowwise_route_weight_multiply_leaves_other_shapes_alone() -> None:
    import torch

    scalar_like = torch.randn(5, 8)
    with rowwise_expert_linear() as rows:
        out = _expert_node_function("backward")(scalar_like, torch.randn(8))
        assert rows[1] == 0 and out.shape == scalar_like.shape
    # forward of a tracked weight: the gradient reaches the weight through the per-row multiply
    x, weight = torch.randn(5, 8), torch.rand(5, 1, requires_grad=True)
    with rowwise_expert_linear():
        (grad,) = torch.autograd.grad(_expert_node_function("backward")(x, weight).sum(), weight)
    assert bits_equal(grad, x.sum(-1, keepdim=True))
