"""Ticket 16: the C-S-E-E-S-C speed campaign and the E-against-S CPU bit cross.

The campaign measures three hands on the three windows that fit 4 GiB: C (ordered backward),
S (C + segmented) and E (C + lt-fused). The analysis reports the paired deltas S-C, E-C and E-S
with a t over the five blocks, the memory peaks of every hand, and the three candidate
combinations. The bit cross reads the stored E tensors of ticket 03 and the stored S tensors of
ticket 14 and compares them with ``torch.equal`` on the loss and all 216 LoRA gradients.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from benchmarks.harness import mbr_measure as mm

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "benchmarks" / "mbr-moonlight-esc-speed-20261125"
ANALYZE = CAMPAIGN / "analyze.py"
REPLACE = CAMPAIGN / "replace_void.py"
BITS_CROSS = CAMPAIGN / "bits_cross.py"


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
    line = _module(ANALYZE, "mbr_esc_analyze")._result_lines([_row("E", 0, 12.0)])[0]
    for field in ("Moonlight-16B-A3B", "RTX 3050 Laptop", "window (8,64)", "arm E",
                  "C + lt-fused", "mode speed (normal)", "torch 2.14.1+cu130", "step 12.00 s"):
        assert field in line, line


def test_a_speed_line_carries_the_three_memory_peaks() -> None:
    line = _module(ANALYZE, "mbr_esc_analyze")._result_lines([_row("S", 0, 12.0)])[0]
    assert "allocated 1.00 GiB" in line and "reserved 2.00 GiB" in line, line
    assert "NVML 3.00 GiB" in line and "RAM 4.00 GiB" in line, line


def _blocks(k: int, seq: int, **arms_steps: List[float]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for arm, steps in arms_steps.items():
        for block, step in enumerate(steps):
            rows.append(_row(arm, block, step, k=k, seq=seq))
    return rows


def test_the_window_reports_all_three_paired_deltas_with_their_t() -> None:
    """Worked example: C 100, S 110..118, E 104..112 over five blocks.

    S-C averages -14% (per-block -10,-12,-14,-16,-18, spread 3.162, t = -14/1.414 = -9.90);
    E-C averages -8% (t = -8/1.414 = -5.66); E-S averages +5.27%.
    """
    module = _module(ANALYZE, "mbr_esc_analyze")
    stats = module.window_stats(_blocks(8, 64, C=[100.0] * 5, S=[110.0, 112.0, 114.0, 116.0,
                                                                118.0],
                                       E=[104.0, 106.0, 108.0, 110.0, 112.0]), k=8, seq=64)
    assert stats["C_step_s"] == pytest.approx(100.0)
    assert stats["S_step_s"] == pytest.approx(114.0)
    assert stats["E_step_s"] == pytest.approx(108.0)
    assert stats["deltas"]["S_vs_C"]["mean_gain_pct"] == pytest.approx(-14.0)
    assert stats["deltas"]["S_vs_C"]["t"] == pytest.approx(-9.90, abs=0.01)
    assert stats["deltas"]["E_vs_C"]["mean_gain_pct"] == pytest.approx(-8.0)
    assert stats["deltas"]["E_vs_C"]["t"] == pytest.approx(-5.66, abs=0.01)
    assert stats["deltas"]["E_vs_S"]["mean_gain_pct"] == pytest.approx(5.27, abs=0.01)
    assert stats["deltas"]["E_vs_S"]["t"] > 50
    assert stats["per_block"]["0"] == {"C": 100.0, "S": 110.0, "E": 104.0}


def test_every_window_has_all_three_hands_so_no_delta_is_missing_at_k64() -> None:
    module = _module(ANALYZE, "mbr_esc_analyze")
    stats = module.window_stats(_blocks(64, 64, C=[100.0] * 5, S=[120.0] * 5, E=[110.0] * 5),
                                k=64, seq=64)
    assert all(stats["deltas"][name] is not None
               for name in ("S_vs_C", "E_vs_C", "E_vs_S"))


def test_the_memory_block_names_every_hand_and_peak() -> None:
    module = _module(ANALYZE, "mbr_esc_analyze")
    stats = module.window_stats(_blocks(8, 512, C=[100.0], S=[110.0], E=[104.0]), k=8, seq=512)
    assert set(stats["memory"]) == {"C", "S", "E"}
    assert stats["memory"]["E"]["peak_nvml"] == 3 * 2**30


def test_the_three_combinations_read_the_arm_of_each_window_and_sum_the_steps() -> None:
    """S everywhere / S on k<=8 + E on k=64 / E everywhere, on the same three windows."""
    module = _module(ANALYZE, "mbr_esc_analyze")
    windows = {
        "(8,64)": {"C_step_s": 100.0, "S_step_s": 110.0, "E_step_s": 104.0},
        "(8,512)": {"C_step_s": 200.0, "S_step_s": 220.0, "E_step_s": 208.0},
        "(64,64)": {"C_step_s": 300.0, "S_step_s": 500.0, "E_step_s": 360.0},
    }
    table = module.combination_table(windows)
    assert table["C everywhere"]["step_sum_s"] == pytest.approx(600.0)
    assert table["S everywhere"]["step_sum_s"] == pytest.approx(830.0)
    assert table["S on k<=8 + E on k=64"]["step_sum_s"] == pytest.approx(690.0)
    assert table["E everywhere"]["step_sum_s"] == pytest.approx(672.0)
    assert table["S on k<=8 + E on k=64"]["per_window"]["(64,64)"]["arm"] == "E"
    assert table["S everywhere"]["per_window"]["(64,64)"]["arm"] == "S"
    assert table["E everywhere"]["steps_rel_C"]["(8,512)"] == pytest.approx(1.04)


def test_the_repeats_rule_reads_a_pair_of_hands_as_its_abba_block() -> None:
    """N of METHOD.md from the pilot: flat hands need the 2-block floor, noisy ones far more."""
    module = _module(ANALYZE, "mbr_esc_analyze")
    flat = _blocks(8, 64, C=[100.0] * 5, S=[110.0] * 5)
    assert module.blocks_needed_for(flat, k=8, seq=64, now="S", before="C") == 2
    noisy = _blocks(8, 64, C=[100.0, 120.0, 80.0, 110.0, 90.0],
                    S=[150.0, 90.0, 200.0, 60.0, 160.0])
    assert module.blocks_needed_for(noisy, k=8, seq=64, now="S", before="C") > 5
    short = _blocks(8, 64, C=[100.0], S=[110.0])
    assert module.blocks_needed_for(short, k=8, seq=64, now="S", before="C") is None


def test_the_combination_table_marks_the_bits_of_a_window_it_has_no_reference_for() -> None:
    module = _module(ANALYZE, "mbr_esc_analyze")
    windows = {"(8,64)": {"C_step_s": 100.0, "S_step_s": 110.0, "E_step_s": 104.0},
               "(64,64)": {"C_step_s": 300.0, "S_step_s": 500.0, "E_step_s": 360.0}}
    table = module.combination_table(windows, bits={"8,64": True, "64,64": None})
    assert table["S everywhere"]["bits"]["(8,64)"] == "E == S == A0 (direct)"
    assert table["S everywhere"]["bits"]["(64,64)"] == "no A0 reference above k=8"


def test_a_window_above_k8_reads_as_no_reference_even_without_a_bit_cell() -> None:
    """The cross has no k=64 cell: A0 does not run there, so there is nothing to compare."""
    module = _module(ANALYZE, "mbr_esc_analyze")
    windows = {"(8,64)": {"C_step_s": 100.0, "S_step_s": 110.0, "E_step_s": 104.0},
               "(64,64)": {"C_step_s": 300.0, "S_step_s": 500.0, "E_step_s": 360.0}}
    table = module.combination_table(windows, bits={"8,64": True})
    assert table["E everywhere"]["bits"]["(64,64)"] == "no A0 reference above k=8"


def test_the_replacement_plan_names_every_esc_cell_whose_run_was_void() -> None:
    module = _module(REPLACE, "mbr_esc_replace")
    plan = mm.esc_speed_plan(blocks=1)
    settled = [dict(item, outcome="ok") for item in plan if item["arm"] == "C"]
    pending = module.pending_cells(plan, settled)
    assert sorted({p["arm"] for p in pending}) == ["E", "S"]
    assert len(pending) == 4 * len(mm.ESC_WINDOWS)  # S, E, E, S per window


def test_the_replacement_namespace_carries_the_deadline_the_sweep_reads(
        tmp_path: Path) -> None:
    import argparse

    module = _module(REPLACE, "mbr_esc_replace")
    args = argparse.Namespace(source="s", cache="c", steps=4, device="auto", lora_r=16,
                              lora_targets=("q_proj",), tokenizer=None, label="x")
    run_args = module.sweep_args(args, tmp_path)
    assert run_args.deadline_hours == 0 and run_args.out == str(tmp_path)


def test_the_esc_speed_command_takes_the_block_count_and_the_start_block() -> None:
    import subprocess
    import sys

    harness = ROOT / "benchmarks" / "harness" / "mbr_measure.py"
    proc = subprocess.run([sys.executable, str(harness), "esc-speed", "--help"],
                          capture_output=True, text=True, check=True)
    assert "--blocks" in proc.stdout and "--start-block" in proc.stdout and "--gate" in \
        proc.stdout, proc.stdout


# ---- the CPU cross of the stored bits, E against S ----
def _save(path: Path, *, loss: List[float], grads: Dict[str, Any]) -> None:
    import torch

    torch.save({"loss": torch.tensor(loss, dtype=torch.float64),
                "grads": {name: torch.tensor(value, dtype=torch.float32)
                          for name, value in grads.items()}}, path)


def test_the_bit_cross_counts_every_equal_and_every_differing_gradient(tmp_path: Path) -> None:
    module = _module(BITS_CROSS, "mbr_esc_bits")
    e, s = tmp_path / "E.pt", tmp_path / "S.pt"
    _save(e, loss=[1.0, 2.0], grads={"a": [1.0, 2.0], "b": [3.0]})
    _save(s, loss=[1.0, 2.0], grads={"a": [1.0, 2.0], "b": [3.0]})
    same = module.cross_cell(e, s)
    assert same["loss_equal"] is True and same["grad_total"] == 2
    assert same["grads_differing"] == 0 and same["max_abs_diff"] == 0.0

    _save(s, loss=[1.0, 2.0], grads={"a": [1.0, 2.0], "b": [3.5]})
    other = module.cross_cell(e, s)
    assert other["loss_equal"] is True and other["grads_differing"] == 1
    assert other["max_abs_diff"] == pytest.approx(0.5)


def test_a_different_loss_or_a_different_gradient_name_is_never_equal(tmp_path: Path) -> None:
    module = _module(BITS_CROSS, "mbr_esc_bits")
    e, s = tmp_path / "E.pt", tmp_path / "S.pt"
    _save(e, loss=[1.0], grads={"a": [1.0]})
    _save(s, loss=[1.5], grads={"a": [1.0]})
    assert module.cross_cell(e, s)["loss_equal"] is False
    _save(s, loss=[1.0], grads={"b": [1.0]})
    # A name on one side only cannot be compared, so both names count as differing.
    assert module.cross_cell(e, s)["grads_differing"] == 2


def test_the_dataset_hashes_come_from_the_rows_of_each_side(tmp_path: Path) -> None:
    module = _module(BITS_CROSS, "mbr_esc_bits")
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"seq": 64, "dataset_sha256": "abc", "arm": "E"}) + "\n"
                    + json.dumps({"seq": 512, "dataset_sha256": "def", "arm": "E"}) + "\n",
                    encoding="utf-8")
    assert module.sha_by_seq([path]) == {64: {"abc"}, 512: {"def"}}
