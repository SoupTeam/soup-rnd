"""A-B-C-C-B-A blocks: native L2L A0, B0 (Erlan's MBR) and the exact MBR, drift balanced.

Every arm sits at mean position 3.5 of the block (A at 1 and 6, B at 2 and 5, C at 3 and 4), so a
steady drift of the host cancels in any pair. Pure arithmetic on CPU; the driver itself needs the
CUDA host.
"""

from __future__ import annotations

import math

import pytest

from benchmarks.harness import p72_abc_blocks as abc


def test_the_block_order_puts_every_arm_at_the_same_mean_position() -> None:
    assert abc.ORDER == ("A", "B", "C", "C", "B", "A")
    mean = {arm: sum(i for i, a in enumerate(abc.ORDER) if a == arm) / 2 for arm in "ABC"}
    assert mean == {"A": 2.5, "B": 2.5, "C": 2.5}


def test_arm_flags_are_native_l2l_stock_b0_and_the_final_exact_mbr() -> None:
    flags = abc.arm_flags(host_cache_mib=12288)
    complete = ["--expert-transfer", "packed", "--expert-prefetch",
                "--expert-host-cache-mib", "12288"]
    assert flags["A"] == ["--profile-kinds", "l2l", "--expert-transfer", "packed",
                          "--expert-host-cache-mib", "12288"]
    assert flags["B"] == complete
    assert flags["C"] == complete + ["--expert-gemm", "lt-fused", "--mbr-ordered-backward",
                                     "--expert-fast-gather", "--expert-async-lease"]


def stats(fetches: int, step_seconds: float, packed: int = 4_866_048) -> dict:
    return {"model_step_seconds": step_seconds,
            "expert_pool_stats_after_run": {"fetches": fetches,
                                            "plan": {"packed_region_bytes": packed}}}


def test_step_metrics_take_fetches_per_step_from_the_pool_counter_after_the_warm_run() -> None:
    runs = [stats(3144, 10.4), stats(6288, 6.6), stats(9432, 6.4), stats(12576, 6.5)]
    out = abc.step_metrics(runs)
    assert out["step_seconds"] == 6.5                # median of the three measured steps
    assert out["fetches"] == 3144                    # per step, not cumulative
    assert out["logical_bytes"] == 3144 * 4_866_048
    assert out["timed_steps"] == [6.6, 6.4, 6.5]


def run(arm: str, value: float, fetches: int = 100, outcome: str = "ok") -> dict:
    return {"arm": arm, "value": value, "fetches": fetches,
            "logical_bytes": fetches * 1000, "outcome": outcome}


def block(a: float, b: float, c: float) -> list[dict]:
    return [run("A", a, 400), run("B", b, 100), run("C", c, 100),
            run("C", c, 100), run("B", b, 100), run("A", a, 400)]


def test_summary_gives_arm_means_pair_differences_and_a_paired_t_over_blocks() -> None:
    blocks = [block(20.0, 6.0, 5.4), block(21.0, 6.2, 5.5), block(20.5, 6.1, 5.3)]
    out = abc.summarize(blocks)
    assert out["valid_blocks"] == 3 and out["void_blocks"] == 0
    assert out["arms"]["A"]["fetches"] == 400 and out["arms"]["B"]["fetches"] == 100
    assert out["arms"]["A"]["step_seconds"] == pytest.approx(20.5)
    cb = out["comparisons"]["C-B"]
    assert cb["mean_diff_seconds"] == pytest.approx(-0.7, abs=1e-6)
    assert cb["mean_diff_percent"] < 0
    assert cb["df"] == 2 and cb["significant_5pct"] is True
    ca = out["comparisons"]["C-A"]
    assert ca["mean_diff_percent"] == pytest.approx(-73.7, abs=0.2)


def test_a_block_with_a_void_run_is_left_out_and_counted() -> None:
    bad = block(20.0, 6.0, 5.0)
    bad[2] = run("C", 5.0, outcome="void")
    out = abc.summarize([block(20.0, 6.0, 5.4), bad, block(20.5, 6.1, 5.3)])
    assert out["valid_blocks"] == 2 and out["void_blocks"] == 1


def test_identical_block_differences_give_an_infinite_t_not_a_division_by_zero() -> None:
    out = abc.summarize([block(20.0, 6.0, 5.0)] * 3)
    assert math.isinf(out["comparisons"]["C-B"]["t"])
