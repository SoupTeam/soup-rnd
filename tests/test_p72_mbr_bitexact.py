"""P72 MBR batch-invariance, ticket 01: is route-preserving MBR bit-equal to L2L on CPU?

Seam: ``make_rig`` and the helpers of ``tests/test_b13_l2l_moe.py``. The scenario is the ragged
three-micro-batch window of ``test_mbr_ragged_streamed_window_matches_microbatch_dispatch``
(lengths 3, 6, 4), run once per variant.

Criterion (plan D4): no tolerances. The loss and every LoRA gradient are compared through their
integer view, so ``-0.0`` and ``0.0`` differ. A red outcome is an observation, not a bug of the
test: each ``observe`` call records it (prints it, and appends it to ``$P72_BITEXACT_LOG`` when
set) and the test asserts only what is known in advance:

* the control (MBR group 1 against plain L2L) must be equal, otherwise the harness is wrong;
* if group 3 is not equal to group 1, a row-wise expert linear layer must make it equal.
"""

from __future__ import annotations

import json
import os
import re
import sys
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Dict, Iterator, List, Optional

import pytest

from benchmarks.harness import b13_moe_fixture as fixture
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
    print(line, file=sys.stderr)
    path = os.environ.get("P72_BITEXACT_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return not bad


def _autocast(dtype: str) -> Any:
    import torch

    if dtype == "fp32":
        return nullcontext()
    return torch.autocast("cpu", dtype=torch.bfloat16)


@contextmanager
def rowwise_expert_linear(scope: str = "experts") -> Iterator[List[int]]:
    """Swap the experts' ``F.linear`` for one call per row (M=1), forward and backward.

    With ``scope="experts"`` only calls made from ``expert_mlp`` are touched, so the router,
    attention and the LoRA layers keep their native GEMMs; ``scope="all"`` is a diagnostic that
    swaps every ``F.linear`` call. Yields a counter of the rows that went through it, so a
    test can tell that the swap really happened.
    """
    import torch
    import torch.nn.functional as functional

    original = functional.linear
    rows: List[int] = [0]

    def linear(input: Any, weight: Any, bias: Any = None) -> Any:  # noqa: A002
        if scope == "experts" and sys._getframe(1).f_code.co_name != "expert_mlp":
            return original(input, weight, bias)
        flat = input.reshape(-1, input.shape[-1])
        rows[0] += int(flat.shape[0])
        out = [original(flat[i : i + 1], weight, bias) for i in range(flat.shape[0])]
        return torch.cat(out).reshape(*input.shape[:-1], -1)

    functional.linear = linear  # type: ignore[assignment]
    try:
        yield rows
    finally:
        functional.linear = original  # type: ignore[assignment]


def run_window(
    rig: Rig,
    ids: List[Any],
    calls: List[Any],
    *,
    mbr: bool,
    group_size: Optional[int],
    dtype: str,
) -> Result:
    _zero_grads(rig.model)
    kwargs: Dict[str, Any] = {"route_preserving_mbr": mbr}
    if mbr:
        kwargs.update(mbr_group_size=group_size, collect_phase_stats=True)
    with _autocast(dtype):
        losses = rig.stepper.run(ids, rig.store, calls=calls, **kwargs)
    grads = {n: g.clone() for n, g in _collect(rig.model).items() if g is not None}
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


def test_rowwise_linear_is_row_by_row_and_scoped() -> None:
    import torch
    import torch.nn.functional as functional

    x, w = torch.randn(5, 8), torch.randn(4, 8)
    native = functional.linear(x, w)
    with rowwise_expert_linear("experts") as rows:
        assert bits_equal(functional.linear(x, w), native) and rows[0] == 0  # not from expert_mlp
    with rowwise_expert_linear("all") as rows:
        out = functional.linear(x, w)
    assert rows[0] == 5 and out.shape == native.shape
    torch.testing.assert_close(out, native)
    assert functional.linear is not None and bits_equal(functional.linear(x, w), native)


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
    for scope in ("experts", "all"):
        with rowwise_expert_linear(scope) as rows:
            fixed1 = run_window(rig, ids, calls, mbr=True, group_size=1, dtype=dtype)
            fixed3 = run_window(rig, ids, calls, mbr=True, group_size=3, dtype=dtype)
        assert rows[0] > 0, "the row-wise swap never ran; the observation would be vacuous"
        observe(f"MBR g1 vs g3 row-wise {scope} [{dtype}]", fixed1, fixed3)
