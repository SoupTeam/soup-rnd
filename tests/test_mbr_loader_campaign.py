"""Ticket 05, part 2: the result lines of the loader campaign (``analyze.py``).

The campaign writes one line per measured cell, and the map's reporting rule says each line
names the model, the card, the window, the arm, the mode and the torch version. Memory is
printed in GiB (the peaks are bytes), so the unit on the line must be the unit it divides by.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
ANALYZE = ROOT / "benchmarks" / "mbr-moonlight-loader-20261123" / "analyze.py"


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("mbr_loader_analyze", ANALYZE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(**changes: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "arm": "H", "k": 8, "seq": 64, "kind": "bits", "block": 0, "status": "ok",
        "loss": 1.5, "tensor_file": "H-k8-s64.pt",
        "levers": {"expert_loader": "cpp_async_2slot"},
        "environment": {"versions": {"torch": "2.14.1+cu130"}},
        "peak_allocated": 2**30, "peak_reserved": 3 * 2**30, "peak_nvml": 4 * 2**30,
        "peak_ram": 5 * 2**30,
    }
    row.update(changes)
    return row


def test_a_bits_line_names_model_card_window_arm_mode_and_torch() -> None:
    line = _module()._result_lines([_row()], kind="bits")[0]
    for field in ("Moonlight-16B-A3B", "RTX 3050 Laptop", "window (8,64)", "arm H",
                  "cpp_async_2slot loader", "mode bits (deterministic)",
                  "torch 2.14.1+cu130"):
        assert field in line, line


def test_memory_on_a_line_is_in_the_unit_the_bytes_are_divided_by() -> None:
    line = _module()._result_lines([_row()], kind="bits")[0]
    assert "allocated 1.00 GiB" in line and "reserved 3.00 GiB" in line, line
    assert "NVML 4.00 GiB" in line and "RAM 5.00 GiB" in line, line
    assert "MiB" not in line, line


def test_a_speed_line_says_normal_mode_and_keeps_the_loader() -> None:
    row = _row(kind="speed", mode="normal", tensor_file=None, step_seconds=12.0,
               environment={"versions": {"torch": "2.14.1+cu130"}})
    line = _module()._result_lines([row], kind="speed")[0]
    assert "mode speed (normal)" in line and "arm H" in line, line


def _speed_row(arm: str, step: float, *, block: int, outcome: str = "ok", **extra: Any):
    row = {"arm": arm, "k": 8, "seq": 64, "kind": "speed", "status": "ok", "block": block,
           "step_seconds": step, "outcome": outcome, "tensor_file": None,
           "environment": {"versions": {"torch": "2.14.1+cu130"}},
           "levers": {"expert_loader": "cpp_async_2slot" if arm == "H" else "async_lease"}}
    row.update(extra)
    return row


def _write_speed_folder(folder: Path, rows: list) -> None:
    import json

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    record = {"outcome": "ok", "checks": {}, "gpu_clock": {"sm_clock_mhz": {"max": 1900}}}
    (folder / "arm_records.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_a_void_run_is_dropped_in_favour_of_its_replacement(tmp_path: Path) -> None:
    rows = []
    for block in range(5):
        rows += [_speed_row("G", 100.0, block=block), _speed_row("H", 90.0, block=block)]
    rows.insert(0, _speed_row("G", 999.0, block=0, outcome="void", peak_nvml=2**40))
    rows[1]["peak_nvml"] = 2**30
    folder = tmp_path / "speed"
    _write_speed_folder(folder, rows)
    speed = _module()._speed(folder)
    window = speed["windows"]["(8,64)"]
    assert window["mean_gain_pct"] == 10.0, window
    assert all(gain == 10.0 for gain in window["gains_pct"]), window
    assert window["mem_G"]["peak_nvml"] == 2**30, window


def test_the_replacement_plan_names_every_cell_whose_run_was_void() -> None:
    import importlib.util

    path = ROOT / "benchmarks" / "mbr-moonlight-loader-20261123" / "replace_void.py"
    spec = importlib.util.spec_from_file_location("mbr_loader_replace", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = [
        {"arm": "G", "k": 8, "seq": 64, "block": 0},
        {"arm": "H", "k": 8, "seq": 64, "block": 0},
        {"arm": "G", "k": 8, "seq": 512, "block": 0},
    ]
    rows = [_speed_row("G", 100.0, block=0),
            _speed_row("H", 90.0, block=0, outcome="void"),
            _speed_row("H", 91.0, block=0)]
    pending = module.pending_cells(plan, rows)
    assert pending == [{"arm": "G", "k": 8, "seq": 512, "block": 0}], pending


def test_a_block_with_a_void_run_is_not_counted_and_its_replacement_is(tmp_path: Path) -> None:
    """``speed-rule.md``: a block with a void run is not counted, and the block runs again under
    its own number. The excluded block's surviving ``ok`` arm must not slip into the pair."""
    import json

    module = _module()
    rows = []
    for block in range(1, 6):
        rows += [_speed_row("G", 100.0, block=block), _speed_row("H", 90.0, block=block)]
    rows += [_speed_row("G", 100.0, block=0), _speed_row("H", 120.0, block=0)]
    folder = tmp_path / "speed"
    _write_speed_folder(folder, rows)
    (folder / "excluded_blocks.json").write_text(
        json.dumps([{"k": 8, "seq": 64, "block": 0, "reason": "void power run"}]) + "\n",
        encoding="utf-8")
    window = module._speed(folder)["windows"]["(8,64)"]
    assert window["gains_pct"] == [10.0] * 5, window
    assert window["mean_gain_pct"] == 10.0, window


def test_the_gate_written_into_analysis_json_is_json() -> None:
    """``analysis.json`` holds the harness gate. ``lever_gate`` keys its windows by ``(k, seq)``
    tuples, and a tuple key is not JSON: the writer must name each window the way the rest of the
    analysis does, ``"(8,64)"``."""
    import json

    module = _module()
    rows = []
    for block in range(5):
        rows += [_speed_row("G", 100.0, block=block), _speed_row("H", 90.0, block=block)]
    gate = module.mm.lever_gate(rows, lever="H", previous="G",
                                windows=module.mm.LOADER_WINDOWS, min_windows=2)
    text = json.dumps(module._gate_json(gate))
    assert '"(8,64)"' in text, text
    assert "(8, 64)" not in text, text


def test_the_loader_speed_command_takes_the_block_count() -> None:
    """``run_speed_h.sh`` passes ``--blocks 5`` to ``loader-speed``; the parser must take it."""
    import subprocess
    import sys

    harness = ROOT / "benchmarks" / "harness" / "mbr_measure.py"
    proc = subprocess.run([sys.executable, str(harness), "loader-speed", "--help"],
                          capture_output=True, text=True, check=True)
    assert "--blocks" in proc.stdout, proc.stdout
