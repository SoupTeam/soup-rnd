"""``--expert-direct-backward`` on the real module (CPU): the segmented plan with the expert
node's backward computed directly must equal the autograd backward bit for bit, and the direct
backward must really run."""

from __future__ import annotations

from typing import Any, Callable, List

from benchmarks.harness import b13_moe_fixture as fixture
from benchmarks.harness import p3_route_preserving_mbr_soup_bridge as bridge
from benchmarks.harness.p72_invariant_expert_gemm import (
    expert_gemm_context,
    lt_merged_expert_gemm,
    segmented_expert_gemm,
)
from soup_cli.utils import l2l_step
from tests.test_a0_exact_lt_mode import CpuBackend, rowwise
from tests.test_b13_l2l_moe import Rig
from tests.test_b13_l2l_moe import make_rig as make_rig  # noqa: F401  (pytest fixture)
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR
from tests.test_p72_mbr_bitexact import LENGTHS, differing, run_window


def window(make_rig: Callable[..., Rig]) -> tuple:
    rig = make_rig(
        k=3, rows=1, seq=6,
        build_options={"two_read_experts": True, "grouped_expert_routes": True},
    )
    ids = [
        fixture.moe_batch(rig.config, seed=seed, rows=1, seq=seq)["input_ids"]
        for seed, seq in enumerate(LENGTHS)
    ]
    calls = [l2l_step.capture_layer_call(rig.model, item) for item in ids]
    return rig, ids, calls


def mbr(rig: Rig, ids: Any, calls: Any, dtype: str = "fp32") -> Any:
    return run_window(rig, ids, calls, mbr=True, group_size=3, dtype=dtype, split_backward=True)


def test_bridge_flag_turns_on_the_direct_backward() -> None:
    parser = bridge.build_parser()
    assert parser.parse_args([]).expert_direct_backward is False
    args = parser.parse_args(["--expert-gemm", "segmented", "--expert-direct-backward"])
    assert args.expert_direct_backward is True


@_REQUIRES_TRAINING_FLOOR
def test_segmented_direct_backward_equals_the_autograd_backward(
    make_rig: Callable[..., Rig],
) -> None:
    rig, ids, calls = window(make_rig)
    with segmented_expert_gemm():
        reference = mbr(rig, ids, calls)
    with expert_gemm_context("segmented", direct_backward=True) as state:
        direct = mbr(rig, ids, calls)
    assert state.direct_stats["direct"] > 0 and state.direct_stats["autograd"] == 0
    # the forward route-weight multiply is elementwise (no M-dependent reduction), and the
    # direct backward makes its own per-segment reduction, so the forward one is not cut
    assert state.mul_rows == 0
    assert differing(reference, direct) == []


@_REQUIRES_TRAINING_FLOOR
def test_lt_merged_direct_backward_merges_and_equals_the_rowwise_reference(
    make_rig: Callable[..., Rig],
) -> None:
    import torch

    rig, ids, calls = window(make_rig)

    def per_segment_rowwise(flat: Any, weight: Any, bias: Any, counts: List[int]) -> Any:
        return torch.cat([rowwise(part, weight, True) for part in flat.split(counts)])

    with segmented_expert_gemm(segment_linear=per_segment_rowwise):
        reference = mbr(rig, ids, calls)
    backend = CpuBackend()
    with lt_merged_expert_gemm(backend, native=rowwise, direct_backward=True) as state:
        merged = mbr(rig, ids, calls)
    assert state.direct_stats["direct"] > 0 and state.merged_calls > 0
    assert any(not transposed for _, transposed in backend.calls)
    assert differing(reference, merged) == []


class FusedCpuBackend(CpuBackend):
    """``expert_forward``/``expert_backward`` in Python (what the C++ extension computes), so the
    lt-fused node patch can be checked on CPU against the segmented autograd run."""

    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0
        self.backward_calls = 0
        self.registered: List[tuple] = []

    def matmul(self, x: Any, w: Any, transposed: bool, algo: Any) -> Any:
        from soup_cli.utils.a0_exact_lt import native_matmul

        return native_matmul(x, w, transposed)  # discovery runs one segment: exact on CPU

    def register_algo(self, out_cols: int, k: int, transposed: bool, rows: int,
                      algo: List[int]) -> None:
        self.registered.append((out_cols, k, transposed, rows))

    def expert_forward(self, hidden: Any, weights: Any, gate_up: Any, down: Any,
                       counts: List[int]) -> Any:
        import torch
        import torch.nn.functional as functional

        from benchmarks.harness.direct_expert_backward import segment_mm

        self.forward_calls += 1
        h = segment_mm(hidden, gate_up, True, counts)
        gate, up = h.chunk(2, dim=-1)
        y = segment_mm(functional.silu(gate) * up, down, True, counts)
        return torch.mul(y, weights)

    def expert_backward(self, hidden: Any, weights: Any, gate_up: Any, down: Any, grad_out: Any,
                        counts: List[int], need_hidden: bool, need_weights: bool) -> tuple:
        from benchmarks.harness.direct_expert_backward import expert_vjp

        self.backward_calls += 1
        return expert_vjp(hidden, weights, gate_up, down, grad_out, counts,
                          need_hidden=need_hidden, need_weights=need_weights)


@_REQUIRES_TRAINING_FLOOR
def test_lt_fused_node_equals_the_segmented_autograd_run(make_rig: Callable[..., Rig]) -> None:
    from benchmarks.harness.p72_invariant_expert_gemm import lt_fused_expert_gemm

    rig, ids, calls = window(make_rig)
    with segmented_expert_gemm():
        reference = mbr(rig, ids, calls)
    backend = FusedCpuBackend()
    with lt_fused_expert_gemm(backend, require_cuda=False) as state:
        fused = mbr(rig, ids, calls)
    assert backend.forward_calls > 0 and backend.backward_calls > 0
    assert backend.registered, "algos must be registered before the fused calls"
    assert state.direct_stats.get("fused_forward", 0) == backend.forward_calls
    assert differing(reference, fused) == []


def test_bridge_split_flag_forces_the_two_phase_backward_for_mbr_only() -> None:
    args = bridge.build_parser().parse_args(["--mbr-split-backward"])
    assert args.mbr_split_backward is True
    mbr = bridge.step_run_options("mbr", group_size=8, expert_gemm="native", diagnostics=False,
                                  split_backward=True)
    assert mbr["mbr_split_backward"] is True
    l2l = bridge.step_run_options("l2l", group_size=8, expert_gemm="native", diagnostics=False,
                                  split_backward=True)
    assert "mbr_split_backward" not in l2l


def test_ordered_flag_replaces_the_split_for_mbr_only() -> None:
    assert bridge.build_parser().parse_args(["--mbr-ordered-backward"]).mbr_ordered_backward
    mbr = bridge.step_run_options("mbr", group_size=8, expert_gemm="lt-fused", diagnostics=False,
                                  ordered_backward=True)
    assert mbr["mbr_ordered_backward"] is True and "mbr_split_backward" not in mbr
    l2l = bridge.step_run_options("l2l", group_size=8, expert_gemm="lt-fused", diagnostics=False,
                                  ordered_backward=True)
    assert "mbr_ordered_backward" not in l2l and "mbr_split_backward" not in l2l


def test_stream_flag_defaults_to_one_and_reaches_the_extension() -> None:
    from benchmarks.harness.p72_invariant_expert_gemm import lt_fused_expert_gemm

    assert bridge.build_parser().parse_args([]).expert_streams == 1
    assert bridge.build_parser().parse_args(["--expert-streams", "2"]).expert_streams == 2

    class Backend(FusedCpuBackend):
        streams = None

        def set_streams(self, count: int) -> None:
            self.streams = count

    backend = Backend()
    with lt_fused_expert_gemm(backend, require_cuda=False, streams=2):
        pass
    assert backend.streams == 2
    other = Backend()
    with lt_fused_expert_gemm(other, require_cuda=False):
        pass
    assert other.streams is None  # one stream (the default) leaves the extension alone
