"""Ticket 14 tail: the price of the segmented expert GEMM (arm S against C and A0).

The campaign measures three hands in A0-C-S-S-C-A0 blocks: A0 is the context, C the ordered arm,
S is C with ``expert_gemm="segmented"``. The analysis reports what segmentation costs against C
and how much of C's speed over A0 is left, with a paired t over the five blocks and the memory
peaks of every hand.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict

import pytest

from benchmarks.harness import mbr_measure as mm

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "benchmarks" / "mbr-moonlight-seg-speed-20261124"
ANALYZE = CAMPAIGN / "analyze.py"
REPLACE = CAMPAIGN / "replace_void.py"


def _module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(arm: str, block: int, step: float, *, k: int = 8, seq: int = 64,
         **changes: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "arm": arm, "k": k, "seq": seq, "kind": "speed", "mode": "normal", "block": block,
        "status": "ok", "outcome": "ok", "step_seconds": step,
        "peak_allocated": 2**30, "peak_reserved": 2 * 2**30, "peak_nvml": 3 * 2**30,
        "peak_ram": 4 * 2**30, "environment": {"versions": {"torch": "2.14.1+cu130"}},
    }
    row.update(changes)
    return row


def test_a_speed_line_names_model_card_window_arm_mode_torch_and_seconds() -> None:
    line = _module(ANALYZE, "mbr_seg_analyze")._result_lines([_row("S", 0, 12.0)])[0]
    for field in ("Moonlight-16B-A3B", "RTX 3050 Laptop", "window (8,64)", "arm S",
                  "C + segmented", "mode speed (normal)", "torch 2.14.1+cu130", "step 12.00 s"):
        assert field in line, line


def test_a_speed_line_carries_the_three_memory_peaks() -> None:
    line = _module(ANALYZE, "mbr_seg_analyze")._result_lines([_row("C", 0, 12.0)])[0]
    assert "allocated 1.00 GiB" in line and "reserved 2.00 GiB" in line, line
    assert "NVML 3.00 GiB" in line and "RAM 4.00 GiB" in line, line


def _blocks(k: int, seq: int, a0: float, c: float, s: list) -> list:
    rows = []
    for block, s_step in enumerate(s):
        rows.append(_row("C", block, c, k=k, seq=seq))
        rows.append(_row("S", block, s_step, k=k, seq=seq))
        if k <= mm.A0_MAX_K:
            rows.append(_row("A0", block, a0, k=k, seq=seq))
    return rows


def test_the_window_reports_both_deltas_and_the_paired_t() -> None:
    """Worked example: C 100 s, S 110..114 s over five blocks, A0 500 s.

    S against C averages -12%; S against A0 saves 78% (A0/S = 4.42x, C/A0 = 5.0x). The per-block
    gain spread is 1 point, so t = -0.12 / (0.01581 / sqrt(5)) = -16.97.
    """
    module = _module(ANALYZE, "mbr_seg_analyze")
    stats = module.window_stats(_blocks(8, 64, 500.0, 100.0, [110.0, 111.0, 112.0, 113.0,
                                                               114.0]), k=8, seq=64)
    assert stats["A0_step_s"] == pytest.approx(500.0)
    assert stats["C_step_s"] == pytest.approx(100.0)
    assert stats["S_step_s"] == pytest.approx(112.0)
    assert stats["S_vs_C"]["mean_gain_pct"] == pytest.approx(-12.0)
    assert stats["S_vs_C"]["t"] == pytest.approx(-16.97, abs=0.01)
    assert stats["S_vs_A0"]["mean_gain_pct"] == pytest.approx(77.6)
    assert stats["speedup_C_over_A0"] == pytest.approx(5.0)
    assert stats["speedup_S_over_A0"] == pytest.approx(4.46)  # 500 / 112, rounded
    assert stats["per_block"]["0"] == {"A0": 500.0, "C": 100.0, "S": 110.0}


def test_above_k8_there_is_no_a0_delta_but_the_ordered_delta_stays() -> None:
    module = _module(ANALYZE, "mbr_seg_analyze")
    stats = module.window_stats(_blocks(64, 64, 0.0, 100.0, [120.0] * 5), k=64, seq=64)
    assert stats["A0_step_s"] is None
    assert stats["S_vs_A0"] is None
    assert stats["S_vs_C"]["mean_gain_pct"] == pytest.approx(-20.0)
    assert stats["per_block"]["0"] == {"C": 100.0, "S": 120.0}


def test_the_cost_reading_splits_at_three_percent_and_two_times() -> None:
    module = _module(ANALYZE, "mbr_seg_analyze")
    assert module.cost_reading(-0.02)["class"] == "almost_free"
    assert module.cost_reading(-0.03)["class"] == "almost_free"
    assert module.cost_reading(-0.09)["class"] == "visible"
    assert module.cost_reading(-0.50)["class"] == "multiples"
    assert module.cost_reading(-1.20)["class"] == "multiples"


def test_the_replacement_plan_names_every_segmented_cell_whose_run_was_void() -> None:
    module = _module(REPLACE, "mbr_seg_replace")
    plan = mm.segmented_speed_plan(blocks=1)
    settled = [dict(item, outcome="ok") for item in plan if item["arm"] in ("A0", "C")]
    pending = module.pending_cells(plan, settled)
    assert [p["arm"] for p in pending] == ["S", "S"] * len(mm.SEGMENTED_WINDOWS)


def test_the_replacement_plan_re_runs_a_void_run_of_a_two_run_cell() -> None:
    """Each arm runs twice per block; one ok run does not settle the cell for both."""
    module = _module(REPLACE, "mbr_seg_replace")
    plan = mm.segmented_speed_plan(blocks=1)
    rows = [dict(item, outcome="ok") for item in plan]
    first_a0 = next(i for i, item in enumerate(rows) if item["arm"] == "A0")
    rows[first_a0]["outcome"] = "void"
    pending = module.pending_cells(plan, rows)
    assert [(p["arm"], p["k"], p["seq"], p["block"]) for p in pending] == [("A0", 8, 64, 0)]


def test_the_replacement_namespace_carries_the_deadline_the_sweep_reads(
        tmp_path: Path) -> None:
    """``sweep`` reads ``deadline_hours`` on every plan; a missing field crashed round 1."""
    import argparse

    module = _module(REPLACE, "mbr_seg_replace")
    args = argparse.Namespace(source="s", cache="c", steps=4, device="auto", lora_r=16,
                              lora_targets=("q_proj",), tokenizer=None, label="x")
    run_args = module.sweep_args(args, tmp_path)
    assert run_args.deadline_hours == 0 and run_args.out == str(tmp_path)


def test_the_seg_speed_command_takes_the_block_count() -> None:
    import subprocess
    import sys

    harness = ROOT / "benchmarks" / "harness" / "mbr_measure.py"
    proc = subprocess.run([sys.executable, str(harness), "seg-speed", "--help"],
                          capture_output=True, text=True, check=True)
    assert "--blocks" in proc.stdout, proc.stdout
