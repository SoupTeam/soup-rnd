"""``--expert-gemm lt-merged``: the segmented plan, with same-algo segments merged into one
cuBLASLt call. All CPU: a fake backend stands in for the extension and computes row by row, and
the test checks that merged forward and backward calls ran and changed no bit against the
segmented plan computed the same way. Equality with A0 itself needs the GPU (micro script)."""

from __future__ import annotations

from typing import Any, Callable, List, Tuple

from benchmarks.harness import b13_moe_fixture as fixture
from benchmarks.harness import p3_route_preserving_mbr_soup_bridge as bridge
from benchmarks.harness.a0_exact_lt import native_matmul
from benchmarks.harness.p72_invariant_expert_gemm import (
    EXPERT_GEMM_CHOICES,
    expert_gemm_context,
    lt_merged_expert_gemm,
    segmented_expert_gemm,
)
from soup_cli.utils import l2l_step
from tests.test_b13_l2l_moe import Rig
from tests.test_b13_l2l_moe import make_rig as make_rig  # noqa: F401  (pytest fixture)
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR
from tests.test_p72_mbr_bitexact import LENGTHS, differing, run_window


def rowwise(x: Any, w: Any, transposed: bool) -> Any:
    """Every row computed alone (M = 1). On CPU, BLAS rows depend on M, so a merged ``mm``
    cannot stand in for the per-segment one; rows computed one at a time can."""
    import torch

    return torch.cat([native_matmul(x[i : i + 1], w, transposed) for i in range(x.shape[0])])


class CpuBackend:
    """One algo for every shape; the merged product is computed row by row."""

    def __init__(self) -> None:
        self.calls: List[Tuple[int, bool]] = []

    @staticmethod
    def supports(x: Any, w: Any) -> bool:
        return True

    def heuristics(self, x: Any, w: Any, transposed: bool, limit: int) -> List[Any]:
        import torch

        return [torch.zeros(8, dtype=torch.long)]

    def matmul(self, x: Any, w: Any, transposed: bool, algo: Any) -> Any:
        self.calls.append((int(x.shape[0]), transposed))
        return rowwise(x, w, transposed)


def test_bridge_flag_accepts_lt_merged_and_matches_the_module() -> None:
    assert "lt-merged" in EXPERT_GEMM_CHOICES
    parser = bridge.build_parser()
    assert parser.parse_args(["--expert-gemm", "lt-merged"]).expert_gemm == "lt-merged"
    action = next(a for a in parser._actions if a.dest == "expert_gemm")
    assert tuple(action.choices) == EXPERT_GEMM_CHOICES


def test_lt_merged_asks_mbr_for_the_two_phase_backward_only() -> None:
    mbr = bridge.step_run_options("mbr", group_size=8, expert_gemm="lt-merged", diagnostics=False)
    assert mbr["mbr_split_backward"] is True
    l2l = bridge.step_run_options("l2l", group_size=8, expert_gemm="lt-merged", diagnostics=False)
    assert "mbr_split_backward" not in l2l


@_REQUIRES_TRAINING_FLOOR
def test_lt_merged_group_merges_segments_like_the_segmented_plan(
    make_rig: Callable[..., Rig],
) -> None:
    """Reference: the segmented plan with each segment computed row by row. lt-merged with the
    same row-by-row products must give the same bits after merging, reordering and the merged
    input-gradient calls, so this checks the plumbing; the A0 bits are a GPU check."""
    import torch

    rig = make_rig(
        k=3, rows=1, seq=6,
        build_options={"two_read_experts": True, "grouped_expert_routes": True},
    )
    ids = [
        fixture.moe_batch(rig.config, seed=seed, rows=1, seq=seq)["input_ids"]
        for seed, seq in enumerate(LENGTHS)
    ]
    calls = [l2l_step.capture_layer_call(rig.model, item) for item in ids]

    def per_segment_rowwise(flat: Any, weight: Any, bias: Any, counts: List[int]) -> Any:
        return torch.cat([rowwise(part, weight, True) for part in flat.split(counts)])

    with segmented_expert_gemm(segment_linear=per_segment_rowwise):
        reference = run_window(rig, ids, calls, mbr=True, group_size=3, dtype="fp32",
                               split_backward=True)

    backend = CpuBackend()
    with lt_merged_expert_gemm(backend, native=rowwise) as state:
        merged = run_window(rig, ids, calls, mbr=True, group_size=3, dtype="fp32",
                            split_backward=True)
    assert state.linear_rows > 0 and state.passthrough_calls == 0
    assert any(not transposed for _, transposed in backend.calls), "no merged backward call"
    assert state.merged_calls > 0, "no call merged two segments"
    assert differing(reference, merged) == []


def test_lt_merged_context_is_reachable_through_expert_gemm_context() -> None:
    with expert_gemm_context("lt-merged", lt_backend=CpuBackend()) as state:
        assert state.table is not None and state.merged_calls == 0
