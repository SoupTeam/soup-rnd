"""Ticket 05, part 2: the result lines of the loader campaign (``analyze.py``).

The campaign writes one line per measured cell, and the map's reporting rule says each line
names the model, the card, the window, the arm, the mode and the torch version. Memory is
printed in GiB (the peaks are bytes), so the unit on the line must be the unit it divides by.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict

import pytest

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


def _archived_bits_folder(folder: Path, cells: list) -> None:
    """An archived bits folder: ``rows.jsonl`` without ``grad_shapes`` and each cell's saved
    payload under ``tensors/``, bound to its row by ``grad_sha256`` and the loss total."""
    import json

    import torch

    from benchmarks.harness import mbr_measure as mm

    (folder / "tensors").mkdir(parents=True)
    rows = []
    for arm, k, grad in cells:
        name = mm.tensor_file_name(arm, k, 64)
        payload = {"loss": torch.tensor([2.0] + [0.0] * (k - 1), dtype=torch.float64),
                   "grads": {"w": torch.tensor(grad)}}
        torch.save(payload, folder / "tensors" / name)
        flat = {"w": payload["grads"]["w"].float().flatten().tolist()}
        rows.append({"arm": arm, "k": k, "seq": 64, "kind": "bits", "block": 0,
                     "status": "ok", "outcome": "ok", "loss": 2.0, "tensor_file": name,
                     "grad_sha256": mm.grad_digest(flat),
                     "environment": {"versions": {"torch": "2.14.1+cu130"}}})
    (folder / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows),
                                       encoding="utf-8")


def _bits_row(folder: Path, arm: str, k: int, grad: list, **changes: Any) -> Dict[str, Any]:
    """One bits row of ``folder``, its payload saved under ``tensors/`` and bound by digest."""
    import torch

    from benchmarks.harness import mbr_measure as mm

    (folder / "tensors").mkdir(parents=True, exist_ok=True)
    name = mm.tensor_file_name(arm, k, 64)
    payload = {"loss": torch.tensor([2.0] + [0.0] * (k - 1), dtype=torch.float64),
               "grads": {"w": torch.tensor(grad)}}
    torch.save(payload, folder / "tensors" / name)
    flat = {"w": payload["grads"]["w"].float().flatten().tolist()}
    row = {"arm": arm, "k": k, "seq": 64, "kind": "bits", "block": 0, "status": "ok",
           "outcome": "ok", "loss": 2.0, "tensor_file": name, "grad_sha256": mm.grad_digest(flat),
           "environment": {"versions": {"torch": "2.14.1+cu130"}}}
    row.update(changes)
    return row


def _write_bits_rows(folder: Path, rows: list) -> None:
    import json

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows),
                                       encoding="utf-8")


def test_a_void_bits_row_gets_no_loader_verdict(tmp_path: Path) -> None:
    rows = [_bits_row(tmp_path, "A0", 1, [1.0, 2.0]),
            _bits_row(tmp_path, "H", 1, [1.0, 2.0], outcome="void")]
    _write_bits_rows(tmp_path, rows)
    assert _module()._bits(tmp_path, {"w": [2]})["verdicts"] == {"k1-H": "void"}


def test_a_bits_row_must_bind_to_its_arm_record_when_the_folder_has_records(tmp_path: Path) -> None:
    import json

    rows = [_bits_row(tmp_path, "A0", 1, [1.0, 2.0]), _bits_row(tmp_path, "H", 1, [1.0, 2.0])]
    _write_bits_rows(tmp_path, rows)
    records = [{"arm": "A0", "outcome": "ok"}, {"arm": "C", "outcome": "ok"}]
    (tmp_path / "arm_records.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    assert _module()._bits(tmp_path, {"w": [2]})["verdicts"] == {"k1-H": "void"}


def test_a_loader_diff_reads_the_row_the_verdict_reads_and_never_an_escape(tmp_path: Path) -> None:
    """The verdict reads the last ok A0 row, so the diff must read it too. An A0 row whose
    ``tensor_file`` leaves the folder is never read; its payload would move ``max_abs_diff``."""
    import torch

    folder = tmp_path / "bits"
    folder.mkdir()
    torch.save({"loss": torch.tensor([2.0], dtype=torch.float64),
                "grads": {"w": torch.tensor([9.0, 9.0])}}, folder / "escape.pt")
    escaped = _bits_row(folder, "A0", 1, [1.0, 2.0], tensor_file="../escape.pt")
    good = _bits_row(folder, "A0", 1, [1.0, 2.0])
    other = _bits_row(folder, "D", 1, [1.0, 3.0])
    _write_bits_rows(folder, [escaped, good, other])
    report = _module()._bits(folder, {"w": [2]})
    assert report["verdicts"] == {"k1-D": "differs"}, report["verdicts"]
    assert report["diff_detail"]["k1-D"]["max_abs_diff"] == pytest.approx(1.0), report


def test_archived_bits_rows_take_their_verdict_from_an_explicit_manifest(tmp_path: Path) -> None:
    _archived_bits_folder(tmp_path, [("A0", 1, [1.0, 2.0]), ("H", 1, [1.0, 2.0]),
                                     ("A0", 2, [1.0, 2.0]), ("H", 2, [1.0, 3.0])])
    verdicts = _module()._bits(tmp_path, {"w": [2]})["verdicts"]
    assert verdicts == {"k1-H": "equal", "k2-H": "differs"}, verdicts


def test_a_mismatched_manifest_gives_archived_bits_rows_no_verdict(tmp_path: Path) -> None:
    import json

    from benchmarks.harness import mbr_measure as mm

    _archived_bits_folder(tmp_path, [("A0", 1, [1.0, 2.0]), ("H", 1, [1.0, 2.0])])
    module = _module()
    root = tmp_path / "campaign"
    root.mkdir()
    tampered = dict(mm.MOONLIGHT_GRAD_SHAPES)
    tampered[next(iter(tampered))] = [1, 1]
    (root / "grad_manifest.json").write_text(json.dumps(tampered), encoding="utf-8")
    verdicts = module._bits(tmp_path, module._grad_manifest(root))["verdicts"]
    assert verdicts["k1-H"].startswith("no verdict: base:no_expected_grads"), verdicts


def test_the_checked_in_grad_manifest_is_the_authoritative_216_names_and_shapes() -> None:
    from benchmarks.harness import mbr_measure as mm

    path = ANALYZE.parent / "grad_manifest.json"
    assert mm.load_grad_manifest(path) == mm.MOONLIGHT_GRAD_SHAPES
    assert len(mm.MOONLIGHT_GRAD_SHAPES) == 216
