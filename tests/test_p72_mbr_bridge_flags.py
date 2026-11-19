"""Ticket 03: the bridge flag ``--expert-gemm {native,invariant}``, checked without a GPU."""

from __future__ import annotations

import pytest

from benchmarks.harness import p3_route_preserving_mbr_soup_bridge as bridge
from benchmarks.harness.p72_invariant_expert_gemm import (
    EXPERT_GEMM_CHOICES,
    expert_gemm_context,
)


def test_expert_gemm_defaults_to_native() -> None:
    assert bridge.build_parser().parse_args([]).expert_gemm == "native"


def test_expert_gemm_accepts_invariant_and_nothing_else() -> None:
    parser = bridge.build_parser()
    assert parser.parse_args(["--expert-gemm", "invariant"]).expert_gemm == "invariant"
    with pytest.raises(SystemExit):
        parser.parse_args(["--expert-gemm", "triton"])


def test_bridge_choices_match_the_module() -> None:
    action = next(a for a in bridge.build_parser()._actions if a.dest == "expert_gemm")
    assert tuple(action.choices) == ("native", "batch_invariant", *EXPERT_GEMM_CHOICES[1:])


def test_per_call_flags_stay_per_call() -> None:
    """A and B arms run as separate bridge calls (plan section 4): these are per-invocation."""
    args = bridge.build_parser().parse_args(
        ["--expert-transfer", "packed", "--expert-host-cache-mib", "64", "--expert-prefetch"]
    )
    assert (args.expert_transfer, args.expert_host_cache_mib, args.expert_prefetch) == (
        "packed", 64, True,
    )


def test_native_changes_nothing_in_the_step_options() -> None:
    mbr = bridge.step_run_options("mbr", group_size=8, expert_gemm="native", diagnostics=False)
    assert mbr == {
        "route_preserving_mbr": True, "mbr_group_size": 8, "collect_phase_stats": False,
    }
    l2l = bridge.step_run_options("l2l", group_size=8, expert_gemm="native", diagnostics=True)
    assert l2l == {
        "route_preserving_mbr": False, "mbr_group_size": None, "collect_phase_stats": True,
    }


def test_invariant_asks_mbr_for_the_two_phase_backward_only() -> None:
    mbr = bridge.step_run_options("mbr", group_size=8, expert_gemm="invariant", diagnostics=False)
    assert mbr["mbr_split_backward"] is True
    l2l = bridge.step_run_options("l2l", group_size=8, expert_gemm="invariant", diagnostics=False)
    assert "mbr_split_backward" not in l2l


def test_unknown_expert_gemm_is_rejected() -> None:
    with pytest.raises(ValueError, match="expert gemm"):
        expert_gemm_context("triton")
    with pytest.raises(ValueError, match="expert gemm"):
        bridge.step_run_options("mbr", group_size=8, expert_gemm="triton", diagnostics=False)


def test_dump_grads_defaults_to_off_and_takes_a_path() -> None:
    parser = bridge.build_parser()
    assert parser.parse_args([]).dump_grads is None
    assert parser.parse_args(["--dump-grads", "g.pt"]).dump_grads == "g.pt"


def test_save_gradients_round_trips_bit_for_bit(tmp_path) -> None:
    import torch

    grads = {"b": torch.tensor([0.0, -0.0], dtype=torch.bfloat16), "a": torch.randn(3, 2)}
    path = tmp_path / "g.pt"
    bridge.save_gradients(path, grads)
    loaded = torch.load(path)
    assert sorted(loaded) == ["a", "b"]
    assert torch.equal(loaded["a"], grads["a"])
    assert loaded["b"].view(torch.int16).tolist() == grads["b"].view(torch.int16).tolist()
