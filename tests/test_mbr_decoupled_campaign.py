"""Ticket 06, part 2: the result lines and the large-window event of the decoupled campaign.

The campaign writes one line per measured cell, and the map's reporting rule says each line
names the model, the card, the window, the arm, the mode and the torch version. The
(64,512) window is the ticket's local ace: C OOMs there, so the VRAM event and the switch
condition (C fits -> decoupled is not enabled) are part of the analysis, not a footnote.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict

from benchmarks.harness import mbr_measure as mm

ROOT = Path(__file__).resolve().parents[1]
ANALYZE = ROOT / "benchmarks" / "mbr-moonlight-decoupled-20261124" / "analyze.py"


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("mbr_decoupled_analyze", ANALYZE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(**changes: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "arm": "D", "k": 8, "seq": 64, "kind": "bits", "block": 0, "status": "ok",
        "loss": 1.5, "tensor_file": "D-k8-s64.pt",
        "levers": {"expert_gemm": "native"},
        "environment": {"versions": {"torch": "2.14.1+cu130"}},
        "peak_allocated": 2**30, "peak_reserved": 3 * 2**30, "peak_nvml": 4 * 2**30,
        "peak_ram": 5 * 2**30,
    }
    row.update(changes)
    return row


def test_a_bits_line_names_model_card_window_arm_mode_and_torch() -> None:
    line = _module()._result_lines([_row()], kind="bits")[0]
    for field in ("Moonlight-16B-A3B", "RTX 3050 Laptop", "window (8,64)", "arm D",
                  "decoupled MBR", "mode bits (deterministic)", "torch 2.14.1+cu130"):
        assert field in line, line


def test_memory_on_a_line_is_in_the_unit_the_bytes_are_divided_by() -> None:
    line = _module()._result_lines([_row()], kind="bits")[0]
    assert "allocated 1.00 GiB" in line and "reserved 3.00 GiB" in line, line
    assert "NVML 4.00 GiB" in line and "RAM 5.00 GiB" in line, line
    assert "MiB" not in line, line


def test_a_speed_line_says_normal_mode_and_names_the_hands() -> None:
    row = _row(arm="C", kind="speed", mode="normal", tensor_file=None, step_seconds=12.0,
               environment={"versions": {"torch": "2.14.1+cu130"}})
    line = _module()._result_lines([row], kind="speed")[0]
    assert "mode speed (normal)" in line and "arm C" in line, line
    assert "ordered backward" in line, line


def _bit(arm: str, status: str) -> Dict[str, Any]:
    return _row(arm=arm, k=64, seq=512, status=status, tensor_file=f"{arm}-k64-s512.pt")


def test_c_oom_at_64_512_with_d_ok_is_the_vram_event_and_opens_the_switch() -> None:
    event = _module()._large_window([_bit("C", "OOM"), _bit("D", "ok")])
    assert event["c_status"] == "OOM" and event["d_status"] == "ok"
    assert event["c_oom_relieved"] is True
    assert event["large_window_enabled"] is True


def test_c_fitting_at_64_512_closes_the_large_window_switch() -> None:
    event = _module()._large_window([_bit("C", "ok"), _bit("D", "ok")])
    assert event["c_oom_relieved"] is False
    assert event["large_window_enabled"] is False


def test_an_unmeasured_window_says_so_instead_of_inventing_a_switch_state() -> None:
    event = _module()._large_window([_bit("D", "ok")])
    assert event["status"] == "not measured"
    assert "large_window_enabled" not in event


def test_the_gate_written_into_analysis_json_is_json() -> None:
    import json

    module = _module()
    rows = []
    for block in range(5):
        rows += [{"arm": "C", "k": 8, "seq": 64, "kind": "speed", "status": "ok",
                  "block": block, "step_seconds": 100.0, "outcome": "ok",
                  "peak_allocated": 1, "peak_reserved": 1, "peak_nvml": 1},
                 {"arm": "D", "k": 8, "seq": 64, "kind": "speed", "status": "ok",
                  "block": block, "step_seconds": 90.0, "outcome": "ok",
                  "peak_allocated": 1, "peak_reserved": 1, "peak_nvml": 1}]
    gate = module.mm.lever_gate(rows, lever="D", previous="C",
                                windows=module.mm.DECOUPLED_WINDOWS,
                                min_windows=module.mm.DECOUPLED_MIN_WINDOWS)
    text = json.dumps(module._gate_json(gate))
    assert '"(8,64)"' in text, text
    assert "(8, 64)" not in text, text


def test_the_replacement_plan_names_every_cell_whose_run_was_void() -> None:
    import importlib.util

    path = ROOT / "benchmarks" / "mbr-moonlight-decoupled-20261124" / "replace_void.py"
    spec = importlib.util.spec_from_file_location("mbr_decoupled_replace", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = mm.decoupled_speed_plan(blocks=1)
    assert len(plan) == len(mm.DECOUPLED_WINDOWS) * 4
    settled = [dict(item, outcome="ok") for item in plan if item["arm"] == "C"]
    pending = module.pending_cells(plan, settled)
    assert [p["arm"] for p in pending] == ["D", "D"] * len(mm.DECOUPLED_WINDOWS), pending


def test_the_decoupled_speed_command_takes_the_block_count() -> None:
    """``run_speed.sh`` passes ``--blocks 5`` to ``decoupled-speed``; the parser must take it."""
    import subprocess
    import sys

    harness = ROOT / "benchmarks" / "harness" / "mbr_measure.py"
    proc = subprocess.run([sys.executable, str(harness), "decoupled-speed", "--help"],
                          capture_output=True, text=True, check=True)
    assert "--blocks" in proc.stdout, proc.stdout
