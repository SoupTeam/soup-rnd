"""D1: ``--expert-gemm padded`` pads every expert GEMM call to one fixed M with zero rows.

CPU only. The shapes of the calls are asserted (arithmetic-independent): every expert linear call
has exactly ``pad_rows`` rows, whatever the micro-batch or the group.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, List

import pytest

from benchmarks.harness import b13_moe_fixture as fixture
from benchmarks.harness import p3_route_preserving_mbr_soup_bridge as bridge
from benchmarks.harness.p72_invariant_expert_gemm import (
    EXPERT_GEMM_CHOICES,
    expert_gemm_context,
    padded_expert_gemm,
)
from soup_cli.utils import l2l_step
from tests.test_b13_l2l_moe import Rig
from tests.test_b13_l2l_moe import make_rig as make_rig  # noqa: F401  (pytest fixture)
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR
from tests.test_p72_mbr_bitexact import LENGTHS, differing, run_window
from tests.test_p72_segmented_expert_gemm import _expert_node_module


def _recording_linear() -> Any:
    import torch.nn.functional as functional

    seen: List[int] = []
    original = functional.linear

    def recorder(input: Any, weight: Any, bias: Any = None) -> Any:  # noqa: A002
        seen.append(int(input.shape[0]))
        return original(input, weight, bias)

    return seen, original, recorder


def test_padded_linear_calls_with_the_fixed_rows_only_inside_expert_mlp() -> None:
    import torch
    import torch.nn.functional as functional

    seen, original, recorder = _recording_linear()
    x, w = torch.randn(5, 8), torch.randn(4, 8)
    functional.linear = recorder  # type: ignore[assignment]
    try:
        with padded_expert_gemm(pad_rows=16) as state:
            outside = functional.linear(x, w)  # not from expert_mlp: untouched
            assert seen == [5]
            seen.clear()
            out = _expert_node_module().forward(None, 0, x, w)
        assert seen == [16]
        assert state.linear_rows == 5
    finally:
        functional.linear = original  # type: ignore[assignment]
    assert out.shape == (5, 4)
    torch.testing.assert_close(out, original(x, w))
    torch.testing.assert_close(outside, original(x, w))


def test_padded_multiply_runs_at_the_fixed_rows_and_keeps_the_result() -> None:
    import torch

    x, w = torch.randn(5, 8), torch.randn(4, 8)
    with padded_expert_gemm(pad_rows=16) as state:
        out = _expert_node_module().forward(None, 0, x, w)
    assert state.mul_rows == 5
    torch.testing.assert_close(out, (x @ w.t()) * torch.ones(5, 1))


def test_more_rows_than_the_pad_fails_loudly() -> None:
    import torch

    with padded_expert_gemm(pad_rows=4):
        with pytest.raises(RuntimeError, match="pad"):
            _expert_node_module().forward(None, 0, torch.randn(5, 8), torch.randn(4, 8))


def test_pad_rows_must_be_positive() -> None:
    with pytest.raises(ValueError, match="pad"):
        with padded_expert_gemm(pad_rows=0):
            pass


def test_context_restores_the_patched_functions() -> None:
    import torch
    import torch.nn.functional as functional

    before = (functional.linear, torch.Tensor.__mul__)
    with padded_expert_gemm(pad_rows=8):
        assert functional.linear is not before[0]
    assert (functional.linear, torch.Tensor.__mul__) == before


def test_bridge_flags_for_padded() -> None:
    assert "padded" in EXPERT_GEMM_CHOICES
    parser = bridge.build_parser()
    args = parser.parse_args(["--expert-gemm", "padded", "--expert-pad-rows", "256"])
    assert (args.expert_gemm, args.expert_pad_rows) == ("padded", 256)
    assert parser.parse_args([]).expert_pad_rows == 512
    mbr = bridge.step_run_options("mbr", group_size=8, expert_gemm="padded", diagnostics=False)
    assert mbr["mbr_split_backward"] is True
    l2l = bridge.step_run_options("l2l", group_size=8, expert_gemm="padded", diagnostics=False)
    assert "mbr_split_backward" not in l2l


def test_padded_context_is_built_by_the_mode_name() -> None:
    with expert_gemm_context("padded", pad_rows=32) as state:
        assert state.pad_rows == 32


@_REQUIRES_TRAINING_FLOOR
@pytest.mark.parametrize("dtype", ["fp32", "bf16_nocache"])
def test_every_expert_call_has_the_pad_rows_and_l2l_equals_mbr(
    make_rig: Callable[..., Rig], dtype: str
) -> None:
    import torch.nn.functional as functional

    pad = 32
    rig = make_rig(
        k=3, rows=1, seq=6,
        build_options={"two_read_experts": True, "grouped_expert_routes": True},
    )
    ids = [
        fixture.moe_batch(rig.config, seed=seed, rows=1, seq=seq)["input_ids"]
        for seed, seq in enumerate(LENGTHS)
    ]
    calls = [l2l_step.capture_layer_call(rig.model, item) for item in ids]
    original = functional.linear

    def run(mbr: bool) -> Any:
        rows: List[int] = []

        def recorder(input: Any, weight: Any, bias: Any = None) -> Any:  # noqa: A002
            if any(sys._getframe(d).f_code.co_name == "expert_mlp" for d in range(1, 5)):
                rows.append(int(input.reshape(-1, input.shape[-1]).shape[0]))
            return original(input, weight, bias)

        functional.linear = recorder  # type: ignore[assignment]
        try:
            with padded_expert_gemm(pad_rows=pad) as state:
                result = run_window(
                    rig, ids, calls, mbr=mbr, group_size=3 if mbr else None, dtype=dtype,
                    split_backward=mbr,
                )
            assert state.linear_rows > 0, "the padded swap never ran"
        finally:
            functional.linear = original  # type: ignore[assignment]
        return rows, result

    l2l_rows, l2l = run(False)
    mbr_rows, mbr = run(True)
    assert l2l_rows and mbr_rows
    assert set(l2l_rows) == {pad} and set(mbr_rows) == {pad}
    assert differing(l2l, mbr) == []
