"""PR #72 F1 block driver: the pilot, void replacement and the paired summary, all with fakes.

No test starts the bridge or reads a sensor; the arm watch is a fake whose outcome is chosen
per run, and run values are synthetic seconds per step.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

import p72_f1_blocks as f1  # noqa: E402
from arm_validity import PILOT_BLOCKS, Run  # noqa: E402


class FakeWatch:
    """An arm watch whose outcome is read from ``outcomes[run]`` (default ok)."""

    outcomes: Dict[int, str] = {}
    seen: List[Dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        FakeWatch.seen.append(kwargs)
        self.record = None

    def __enter__(self) -> "FakeWatch":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.record = {"outcome": FakeWatch.outcomes.get(self.kwargs["run"], "ok")}
        return False


@pytest.fixture(autouse=True)
def reset_watch():
    FakeWatch.outcomes, FakeWatch.seen = {}, []


def runner(arm: str, run: int) -> Dict[str, Any]:
    # B is 60% faster than A with a little run-to-run noise.
    base = 19.6 if arm == "A" else 7.4
    return {"value": base + 0.05 * (run % 3), "extra": {"grad_sha256": ["same"]}}


def drive(**overrides: Any) -> Dict[str, Any]:
    options = dict(label="x", required=("suspend", "power"), model_path="/cache", max_blocks=10,
                   watch_factory=FakeWatch)
    options.update(overrides)
    return f1.run_blocks(runner, **options)


def test_the_pilot_is_five_blocks_of_a_b_b_a_with_the_same_required_checks():
    outcome = drive()
    assert len(outcome["blocks"]) == PILOT_BLOCKS
    assert [r["arm"] for r in outcome["records"][:4]] == ["A", "B", "B", "A"]
    assert [r["run"] for r in outcome["records"]] == list(range(1, 4 * PILOT_BLOCKS + 1))
    assert {tuple(w["required"]) for w in FakeWatch.seen} == {("suspend", "power")}
    assert {w["model_path"] for w in FakeWatch.seen} == {"/cache"}


def test_a_void_run_makes_its_block_not_count_and_a_replacement_runs():
    FakeWatch.outcomes = {6: "void"}  # the second run of block 2
    outcome = drive()
    assert len(outcome["blocks"]) == PILOT_BLOCKS + 1
    summary = f1.summarize(outcome["blocks"])
    assert summary["valid_blocks"] == PILOT_BLOCKS and summary["void_blocks"] == 1
    assert any(r["arm_record"]["outcome"] == "void" for r in outcome["records"]), "kept on record"


def test_it_stops_instead_of_adding_blocks_forever():
    FakeWatch.outcomes = {run: "void" for run in range(1, 400)}
    with pytest.raises(RuntimeError, match="still wants more"):
        drive(max_blocks=6)


def test_summary_reports_the_paired_difference_and_the_blocks_needed():
    outcome = drive()
    summary = f1.summarize(outcome["blocks"])
    assert summary["mean_diff_percent"] == pytest.approx(-62.3, abs=1.0)
    assert summary["significant_5pct"] is True and summary["df"] == PILOT_BLOCKS - 1
    assert summary["repeats"]["blocks_needed"] >= 2


def test_an_effect_buried_in_noise_is_not_called_significant():
    # B minus A per block: +0.3, -0.3, +0.2, -0.2
    blocks = [[Run(10.0, "ok"), Run(10.4, "ok"), Run(10.2, "ok"), Run(10.0, "ok")],
              [Run(10.0, "ok"), Run(9.8, "ok"), Run(9.6, "ok"), Run(10.0, "ok")],
              [Run(10.0, "ok"), Run(10.3, "ok"), Run(10.1, "ok"), Run(10.0, "ok")],
              [Run(10.0, "ok"), Run(9.9, "ok"), Run(9.7, "ok"), Run(10.0, "ok")]]
    summary = f1.summarize(blocks)
    assert summary["significant_5pct"] is False


def test_identical_differences_of_zero_are_not_an_effect():
    blocks = [[Run(10.0, "ok"), Run(10.0, "ok"), Run(10.0, "ok"), Run(10.0, "ok")]] * 3
    summary = f1.summarize(blocks)
    assert summary["t"] == 0.0 and summary["significant_5pct"] is False


def test_t_critical_uses_the_next_lower_tabulated_df():
    assert f1.t_critical(4) == 2.776 and f1.t_critical(11) == 2.228 and f1.t_critical(99) == 2.042


def test_b0_against_b1_differ_only_in_the_expert_gemm_flag():
    # Ticket 07: complete + MBR group 8 + prefetch on both arms; A is native GEMM, B invariant.
    flags = f1.arm_flags("b0-b1", host_cache_mib=12288)
    assert flags["A"] == ["--expert-transfer", "packed", "--expert-prefetch",
                          "--expert-host-cache-mib", "12288"]
    assert flags["B"] == flags["A"] + ["--expert-gemm", "invariant"]


def test_the_pr72_mode_keeps_sync_against_packed_prefetch():
    flags = f1.arm_flags("pr72", host_cache_mib=12288)
    assert flags["A"] == ["--expert-transfer", "sync"]
    assert "--expert-gemm" not in flags["B"]


def test_an_unknown_arm_mode_is_refused():
    with pytest.raises(ValueError, match="unknown arm mode"):
        f1.arm_flags("b1-b2", host_cache_mib=1)


def test_a0_against_b0s_overrides_the_kind_and_adds_the_segmented_gemm():
    flags = f1.arm_flags("a0-b0s", host_cache_mib=12288)
    # A0 is the L2L step: complete (packed + host cache), no prefetch, native GEMM
    assert flags["A"] == ["--profile-kinds", "l2l", "--expert-transfer", "packed",
                          "--expert-host-cache-mib", "12288"]
    # B0s is the MBR step (the driver's default kind): complete + prefetch + segmented GEMM
    assert flags["B"] == ["--expert-transfer", "packed", "--expert-prefetch",
                          "--expert-host-cache-mib", "12288", "--expert-gemm", "segmented"]


def test_b0_against_b0s_differ_only_in_the_expert_gemm_flag():
    flags = f1.arm_flags("b0-b0s", host_cache_mib=12288)
    assert flags["B"] == flags["A"] + ["--expert-gemm", "segmented"]
    assert "--profile-kinds" not in flags["A"]


def test_every_run_record_is_handed_out_as_soon_as_the_run_ends():
    seen = []
    drive(on_record=lambda record: seen.append((record["run"], record["arm_record"]["outcome"])))
    assert [run for run, _ in seen] == list(range(1, 4 * PILOT_BLOCKS + 1))
    assert {outcome for _, outcome in seen} == {"ok"}


def test_b0_against_the_direct_backward_arms_add_the_flag_to_b_only():
    complete = ["--expert-transfer", "packed", "--expert-prefetch",
                "--expert-host-cache-mib", "12288"]
    flags = f1.arm_flags("b0-b0sd", host_cache_mib=12288)
    assert flags["A"] == complete
    assert flags["B"] == complete + ["--expert-gemm", "segmented", "--expert-direct-backward"]
    flags = f1.arm_flags("b0-ltd", host_cache_mib=12288)
    assert flags["A"] == complete
    assert flags["B"] == complete + ["--expert-gemm", "lt-merged", "--expert-direct-backward"]



def test_b0_against_b0_with_the_two_phase_backward_only_adds_the_split_flag():
    complete = ["--expert-transfer", "packed", "--expert-prefetch",
                "--expert-host-cache-mib", "12288"]
    flags = f1.arm_flags("b0-split", host_cache_mib=12288)
    assert flags["A"] == complete
    assert flags["B"] == complete + ["--mbr-split-backward"]


def test_ordered_backward_arm_is_lt_fused_with_the_ordered_flag_instead_of_the_split():
    complete = ["--expert-transfer", "packed", "--expert-prefetch",
                "--expert-host-cache-mib", "12288"]
    flags = f1.arm_flags("b0-ltfo", host_cache_mib=12288)
    assert flags["A"] == complete
    assert flags["B"] == complete + ["--expert-gemm", "lt-fused", "--mbr-ordered-backward"]


def test_best_exact_arm_adds_the_fast_row_gather_to_the_ordered_lt_fused_arm_only_on_b():
    complete = ["--expert-transfer", "packed", "--expert-prefetch",
                "--expert-host-cache-mib", "12288"]
    flags = f1.arm_flags("b0-ltfog", host_cache_mib=12288)
    assert flags["A"] == complete                      # stock B0 stays the reference
    assert flags["B"] == complete + ["--expert-gemm", "lt-fused", "--mbr-ordered-backward",
                                     "--expert-fast-gather"]


def test_final_exact_arm_adds_fast_gather_and_async_lease_only_on_b():
    complete = ["--expert-transfer", "packed", "--expert-prefetch",
                "--expert-host-cache-mib", "12288"]
    flags = f1.arm_flags("b0-ltfoga", host_cache_mib=12288)
    assert flags["A"] == complete                      # stock B0 stays frozen
    assert flags["B"] == complete + ["--expert-gemm", "lt-fused", "--mbr-ordered-backward",
                                     "--expert-fast-gather", "--expert-async-lease"]


def test_a0_against_the_ordered_lt_fused_arm_pairs_l2l_with_the_exact_mbr():
    flags = f1.arm_flags("a0-ltfo", host_cache_mib=12288)
    assert flags["A"] == ["--profile-kinds", "l2l", "--expert-transfer", "packed",
                          "--expert-host-cache-mib", "12288"]
    assert flags["B"] == ["--expert-transfer", "packed", "--expert-prefetch",
                          "--expert-host-cache-mib", "12288", "--expert-gemm", "lt-fused",
                          "--mbr-ordered-backward"]


def test_stream_arms_add_two_streams_to_the_exact_arm_only():
    complete = ["--expert-transfer", "packed", "--expert-prefetch",
                "--expert-host-cache-mib", "12288"]
    exact = complete + ["--expert-gemm", "lt-fused", "--mbr-ordered-backward"]
    flags = f1.arm_flags("ltfo-ltfs", host_cache_mib=12288)
    assert flags["A"] == exact and flags["B"] == exact + ["--expert-streams", "2"]
    assert f1.arm_flags("b0-ltfs", host_cache_mib=12288)["B"] == exact + ["--expert-streams", "2"]


def test_three_stream_arms_pair_the_exact_mbr_with_and_without_streams_and_with_b0():
    complete = ["--expert-transfer", "packed", "--expert-prefetch",
                "--expert-host-cache-mib", "12288"]
    exact = complete + ["--expert-gemm", "lt-fused", "--mbr-ordered-backward"]
    three = exact + ["--expert-streams", "3"]
    flags = f1.arm_flags("ltfo-ltfs3", host_cache_mib=12288)
    assert flags["A"] == exact and flags["B"] == three
    flags = f1.arm_flags("b0-ltfs3", host_cache_mib=12288)
    assert flags["A"] == complete and flags["B"] == three
