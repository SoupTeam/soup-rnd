"""Diagnosis 2: ``--expert-gemm segmented`` keeps the native expert GEMM but calls it once per
micro-batch segment of a joined MBR group, so every call has the shape the L2L (A0) call has.

All CPU. The shapes of the expert calls are asserted, which is arithmetic-independent: on CPU
native GEMM rows do not depend on M, so equal bits alone would not show the swap ran.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Callable, List

import pytest

from benchmarks.harness import b13_moe_fixture as fixture
from benchmarks.harness import p3_route_preserving_mbr_soup_bridge as bridge
from benchmarks.harness.p72_invariant_expert_gemm import (
    EXPERT_GEMM_CHOICES,
    alternating_switch,
    expert_gemm_context,
    segment_rows,
    segmented_expert_gemm,
)
from soup_cli.utils import l2l_step
from tests.test_b13_l2l_moe import Rig
from tests.test_b13_l2l_moe import make_rig as make_rig  # noqa: F401  (pytest fixture)
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR
from tests.test_p72_mbr_bitexact import LENGTHS, differing, run_window


def test_segment_rows_gathers_rows_by_micro_batch_in_their_original_order() -> None:
    import torch

    # Joined group of 3 micro-batches with 4, 3 and 2 tokens. Rows come in (top-k slot, token)
    # order, so one micro-batch's rows are not adjacent until they are gathered.
    token_idx = torch.tensor([0, 5, 1, 8, 4, 2, 6, 3, 7])
    ((perm, counts),) = segment_rows([token_idx], torch.tensor([4, 7]))
    assert counts == [4, 3, 2]
    assert token_idx[perm].tolist() == [0, 1, 2, 3, 5, 4, 6, 8, 7]


def test_segment_rows_skips_empty_micro_batches_and_keeps_single_segments_as_they_are() -> None:
    import torch

    layout = segment_rows([torch.tensor([6, 7, 0]), torch.tensor([1, 2])], torch.tensor([3, 6]))
    (perm, counts), (single_perm, single_counts) = layout
    assert counts == [1, 2] and perm.tolist() == [2, 0, 1]
    assert single_perm is None and single_counts == [2]


def test_segment_rows_of_several_experts_match_one_expert_at_a_time() -> None:
    import torch

    experts = [torch.tensor([0, 5, 1, 8, 4]), torch.tensor([7, 2, 6]), torch.tensor([3, 3 + 6])]
    boundaries = torch.tensor([4, 7])
    together = segment_rows(experts, boundaries)
    for tokens, (perm, counts) in zip(experts, together):
        ((alone_perm, alone_counts),) = segment_rows([tokens], boundaries)
        assert counts == alone_counts
        if alone_perm is None:
            assert perm is None
        else:
            assert torch.equal(perm, alone_perm)


def _expert_node_module() -> types.ModuleType:
    """A stand-in for ``soup_cli.utils.expert_stream`` with the two names the swap keys on."""
    namespace: dict = {"__name__": "soup_cli.utils.expert_stream"}
    exec(
        "import torch.nn.functional as functional\n"
        "def expert_mlp(hidden, weight):\n"
        "    return functional.linear(hidden, weight)\n"
        "def forward(owner, expert, hidden, weight):\n"
        "    return expert_mlp(hidden, weight) * weight.new_ones(hidden.shape[0], 1)\n",
        namespace,
    )
    module = types.ModuleType("fake_node")
    module.__dict__.update(namespace)
    return module


def test_segmented_linear_runs_one_call_per_segment_and_only_inside_expert_mlp() -> None:
    import torch
    import torch.nn.functional as functional

    seen: List[int] = []
    original = functional.linear

    def recorder(input: Any, weight: Any, bias: Any = None) -> Any:  # noqa: A002
        seen.append(int(input.shape[0]))
        return original(input, weight, bias)

    x, w = torch.randn(6, 8), torch.randn(4, 8)
    owner = types.SimpleNamespace(layer_idx=2)
    functional.linear = recorder  # type: ignore[assignment]
    try:
        with segmented_expert_gemm() as state:
            state.plans[(2, 5)] = [3, 3]
            outside = functional.linear(x, w)  # not from expert_mlp: untouched
            assert seen == [6]
            seen.clear()
            node = _expert_node_module()
            out = node.forward(owner, 5, x, w)
        assert seen == [3, 3]
        assert state.linear_rows == 6
    finally:
        functional.linear = original  # type: ignore[assignment]
    torch.testing.assert_close(out, original(x, w))
    torch.testing.assert_close(outside, original(x, w))


def test_expert_without_a_plan_passes_through_and_is_counted() -> None:
    import torch

    x, w = torch.randn(4, 8), torch.randn(3, 8)
    with segmented_expert_gemm() as state:
        node = _expert_node_module()
        node.forward(types.SimpleNamespace(layer_idx=0), 1, x, w)
    assert state.linear_rows == 0 and state.passthrough_calls >= 1


def test_a_plan_that_does_not_match_the_rows_fails_loudly() -> None:
    import torch

    x, w = torch.randn(4, 8), torch.randn(3, 8)
    with segmented_expert_gemm() as state:
        state.plans[(0, 1)] = [2, 1]
        node = _expert_node_module()
        with pytest.raises(RuntimeError, match="segment plan"):
            node.forward(types.SimpleNamespace(layer_idx=0), 1, x, w)


def test_context_restores_the_patched_functions() -> None:
    import torch
    import torch.nn.functional as functional

    from soup_cli.utils import expert_stream

    before = (
        functional.linear,
        torch.Tensor.__mul__,
        l2l_step._route_preserving_deepseek_forward,
        expert_stream._grouped_expert_routes,
        expert_stream._default_expert_routes,
        expert_stream.streamed_experts_class().forward,
    )
    with segmented_expert_gemm():
        assert functional.linear is not before[0]
    after = (
        functional.linear,
        torch.Tensor.__mul__,
        l2l_step._route_preserving_deepseek_forward,
        expert_stream._grouped_expert_routes,
        expert_stream._default_expert_routes,
        expert_stream.streamed_experts_class().forward,
    )
    assert before == after


# --- bridge flag ----------------------------------------------------------------------------


def test_bridge_flag_accepts_segmented_and_matches_the_module() -> None:
    assert "segmented" in EXPERT_GEMM_CHOICES
    parser = bridge.build_parser()
    assert parser.parse_args(["--expert-gemm", "segmented"]).expert_gemm == "segmented"
    action = next(a for a in parser._actions if a.dest == "expert_gemm")
    assert tuple(action.choices) == EXPERT_GEMM_CHOICES


def test_segmented_asks_mbr_for_the_two_phase_backward_only() -> None:
    mbr = bridge.step_run_options("mbr", group_size=8, expert_gemm="segmented", diagnostics=False)
    assert mbr["mbr_split_backward"] is True
    l2l = bridge.step_run_options("l2l", group_size=8, expert_gemm="segmented", diagnostics=False)
    assert "mbr_split_backward" not in l2l


def test_segmented_context_is_a_segmented_state_object() -> None:
    with expert_gemm_context("segmented") as state:
        assert hasattr(state, "plans") and hasattr(state, "linear_rows")


# --- the real module on CPU -----------------------------------------------------------------

@_REQUIRES_TRAINING_FLOOR
@pytest.mark.parametrize("dtype", ["fp32", "bf16_nocache"])
def test_segmented_group_calls_the_experts_with_the_l2l_shapes(
    make_rig: Callable[..., Rig], dtype: str
) -> None:
    import torch.nn.functional as functional

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

    def shapes_of(run: Callable[[], Any]) -> Any:
        rows: List[int] = []

        def recorder(input: Any, weight: Any, bias: Any = None) -> Any:  # noqa: A002
            # the segmented swap sits between this recorder and expert_mlp: look up to 5 frames
            if any(sys._getframe(depth).f_code.co_name == "expert_mlp" for depth in range(1, 6)):
                rows.append(int(input.reshape(-1, input.shape[-1]).shape[0]))
            return original(input, weight, bias)

        functional.linear = recorder  # type: ignore[assignment]
        try:
            result = run()
        finally:
            functional.linear = original  # type: ignore[assignment]
        return sorted(rows), result

    l2l_rows, l2l = shapes_of(
        lambda: run_window(rig, ids, calls, mbr=False, group_size=None, dtype=dtype)
    )

    def segmented_run() -> Any:
        with segmented_expert_gemm() as state:
            result = run_window(
                rig, ids, calls, mbr=True, group_size=3, dtype=dtype, split_backward=True
            )
        assert state.linear_rows > 0, "the segmented swap never ran"
        assert state.passthrough_calls == 0, "an expert call had no segment plan"
        return result

    seg_rows, seg = shapes_of(segmented_run)
    assert seg_rows == l2l_rows, "segmented MBR must call the experts with the L2L shapes"
    assert differing(l2l, seg) == []

    # control: the joined native MBR group makes fewer, larger calls, so the check can go red
    native_rows, _ = shapes_of(
        lambda: run_window(rig, ids, calls, mbr=True, group_size=3, dtype=dtype)
    )
    assert native_rows != l2l_rows


# --- one-pass planned routes --------------------------------------------------------------


def _reference_planned_routes(expert_mask, boundaries):
    """The old two-step walk: grouped routes, then segment_rows and one gather per expert."""
    from soup_cli.utils import expert_stream

    items = list(expert_stream._grouped_expert_routes(expert_mask))
    layout = segment_rows([tokens for _, _, tokens in items], boundaries)
    out = []
    for (expert, slots, tokens), (perm, counts) in zip(items, layout):
        if perm is not None:
            slots, tokens = slots[perm], tokens[perm]
        out.append((expert, slots.tolist(), tokens.tolist(), counts))
    return out


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_planned_routes_equal_the_grouped_walk_plus_segment_rows(seed: int) -> None:
    import torch
    import torch.nn.functional as functional

    from benchmarks.harness.p72_invariant_expert_gemm import planned_grouped_routes

    torch.manual_seed(seed)
    experts, top_k, tokens = 9, 3, 23
    top_k_index = torch.stack([torch.randperm(experts)[:top_k] for _ in range(tokens)])
    mask = functional.one_hot(top_k_index, num_classes=experts).permute(2, 1, 0)
    boundaries = torch.tensor([5, 6, 14, 20])  # micro-batches of 5, 1, 8, 6, 3 tokens
    want = _reference_planned_routes(mask, boundaries)
    got = [
        (expert, slots.tolist(), toks.tolist(), counts)
        for expert, slots, toks, counts in planned_grouped_routes(mask, boundaries)
    ]
    assert got == want
    assert any(len(counts) > 1 for *_, counts in want), "the case must cut some expert"


def test_planned_routes_read_the_host_once() -> None:
    import torch
    import torch.nn.functional as functional

    from benchmarks.harness.p72_invariant_expert_gemm import planned_grouped_routes

    mask = functional.one_hot(torch.tensor([[0, 1], [1, 2], [2, 0]]), num_classes=3)
    mask = mask.permute(2, 1, 0)
    reads = []
    original = torch.Tensor.tolist

    def counting(self):
        reads.append(tuple(self.shape))
        return original(self)

    torch.Tensor.tolist = counting
    try:
        list(planned_grouped_routes(mask, torch.tensor([1, 2])))
    finally:
        torch.Tensor.tolist = original
    assert len(reads) == 1, f"expected one host read, got {reads}"


def test_alternating_switch_turns_an_option_on_for_every_second_step_only_when_asked() -> None:
    on = [alternating_switch(i, True) for i in range(6)]
    assert on == [False, True, False, True, False, True]
    assert [alternating_switch(i, False) for i in range(4)] == [False] * 4
