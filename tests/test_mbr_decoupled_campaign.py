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

import pytest

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


def test_the_diff_detail_carries_grad_rel_l2_and_min_cos(tmp_path: Path) -> None:
    """D1's opt-in row: a lever that differs needs ``grad_rel_l2`` and ``min_cos``.

    Worked example: grad ``a`` goes from (1,0) to (0,1), so its difference has squared norm 2
    and the total reference squared norm is 1 + 1 = 2, giving ``grad_rel_l2`` = 1; grad ``b``
    is unchanged with cosine 1, so ``min_cos`` is 0 (``a``'s cosine).
    """
    import torch

    base = {"arm": "A0", "k": 2, "seq": 64, "status": "ok", "tensor_file": "a0.pt"}
    other = {"arm": "D", "k": 2, "seq": 64, "status": "ok", "tensor_file": "d.pt"}
    torch.save({"loss": torch.tensor([1.0], dtype=torch.float64),
                "grads": {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([0.0, 1.0])}},
               tmp_path / "a0.pt")
    torch.save({"loss": torch.tensor([1.0], dtype=torch.float64),
                "grads": {"a": torch.tensor([0.0, 1.0]), "b": torch.tensor([0.0, 1.0])}},
               tmp_path / "d.pt")
    detail = _module()._diff_detail(tmp_path, base, other)
    assert detail["grad_rel_l2"] == pytest.approx(1.0)
    assert detail["min_cos"] == pytest.approx(0.0)
    assert detail["loss_equal"] is True


def test_a_repeat_cell_is_compared_across_folders(tmp_path: Path) -> None:
    """The repeat pass lives in its own folder, so the comparison loads both files itself."""
    import importlib.util

    import torch

    path = ROOT / "benchmarks" / "mbr-moonlight-decoupled-20261124" / "compare_repeats.py"
    spec = importlib.util.spec_from_file_location("mbr_decoupled_compare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    first, repeat = tmp_path / "first" / "tensors", tmp_path / "repeat" / "tensors"
    first.mkdir(parents=True)
    repeat.mkdir(parents=True)
    cell = {"loss": torch.tensor([2.0], dtype=torch.float64),
            "grads": {"w": torch.tensor([1.0, 2.0])}}
    torch.save(cell, first / "D-k2-s64.pt")
    torch.save(cell, repeat / "D-k2-s64.pt")
    same = module.compare_cell(first, repeat, "D", 2, 64)
    assert same["loss_equal"] is True and same["grads_differing"] == 0
    torch.save({"loss": cell["loss"], "grads": {"w": torch.tensor([1.0, 3.0])}},
               repeat / "D-k2-s64.pt")
    differs = module.compare_cell(first, repeat, "D", 2, 64)
    assert differs["grads_differing"] == 1 and differs["grad_rel_l2"] > 0


def test_the_decoupled_speed_command_takes_the_block_count() -> None:
    """``run_speed.sh`` passes ``--blocks 5`` to ``decoupled-speed``; the parser must take it."""
    import subprocess
    import sys

    harness = ROOT / "benchmarks" / "harness" / "mbr_measure.py"
    proc = subprocess.run([sys.executable, str(harness), "decoupled-speed", "--help"],
                          capture_output=True, text=True, check=True)
    assert "--blocks" in proc.stdout, proc.stdout


def _archived_bits_folder(folder: Path, cells: list) -> None:
    """An archived bits folder: ``rows.jsonl`` without ``grad_shapes`` and each cell's saved
    payload under ``tensors/``, bound to its row by ``grad_sha256`` and the loss total."""
    import json

    import torch

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


def test_archived_bits_rows_take_their_verdict_from_an_explicit_manifest(tmp_path: Path) -> None:
    _archived_bits_folder(tmp_path, [("A0", 1, [1.0, 2.0]), ("D", 1, [1.0, 2.0]),
                                     ("A0", 2, [1.0, 2.0]), ("D", 2, [1.0, 3.0])])
    verdicts = _module()._bits(tmp_path, {"w": [2]})["verdicts"]
    assert verdicts == {"k1-D": "equal", "k2-D": "differs"}, verdicts


def test_a_mismatched_manifest_gives_archived_bits_rows_no_verdict(tmp_path: Path) -> None:
    _archived_bits_folder(tmp_path, [("A0", 1, [1.0, 2.0]), ("D", 1, [1.0, 2.0])])
    import json

    module = _module()
    root = tmp_path / "campaign"
    root.mkdir()
    tampered = dict(mm.MOONLIGHT_GRAD_SHAPES)
    tampered[next(iter(tampered))] = [1, 1]
    (root / "grad_manifest.json").write_text(json.dumps(tampered), encoding="utf-8")
    verdicts = module._bits(tmp_path, module._grad_manifest(root))["verdicts"]
    assert verdicts["k1-D"].startswith("no verdict: base:no_expected_grads"), verdicts


def test_the_checked_in_grad_manifest_is_the_authoritative_216_names_and_shapes() -> None:
    path = ANALYZE.parent / "grad_manifest.json"
    assert mm.load_grad_manifest(path) == mm.MOONLIGHT_GRAD_SHAPES
    assert len(mm.MOONLIGHT_GRAD_SHAPES) == 216


def test_a_duplicate_row_gives_the_verdict_and_the_diff_the_same_row(tmp_path: Path) -> None:
    """The verdict keeps the last row of each arm; the diff detail must load that same row."""
    import json

    import torch

    (tmp_path / "tensors").mkdir()
    rows = []
    for arm, grad, name in (("A0", [1.0, 2.0], "A0-k1-s64.pt"),
                            ("D", [1.0, 2.0], "D-first-k1-s64.pt"),
                            ("D", [1.0, 3.0], "D-k1-s64.pt")):
        payload = {"loss": torch.tensor([2.0], dtype=torch.float64),
                   "grads": {"w": torch.tensor(grad)}}
        torch.save(payload, tmp_path / "tensors" / name)
        flat = {"w": payload["grads"]["w"].float().flatten().tolist()}
        rows.append({"arm": arm, "k": 1, "seq": 64, "kind": "bits", "block": 0, "status": "ok",
                     "outcome": "ok", "loss": 2.0, "tensor_file": name,
                     "grad_sha256": mm.grad_digest(flat),
                     "environment": {"versions": {"torch": "2.14.1+cu130"}}})
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows),
                                         encoding="utf-8")
    report = _module()._bits(tmp_path, {"w": [2]})
    assert report["verdicts"] == {"k1-D": "differs"}, report["verdicts"]
    assert report["diff_detail"]["k1-D"]["grads_differing"] == 1, report["diff_detail"]


def _rewrite_rows(folder: Path, edit) -> None:
    import json

    path = folder / "rows.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    edit(rows)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_a_void_row_with_correct_tensors_gets_no_verdict(tmp_path: Path) -> None:
    _archived_bits_folder(tmp_path, [("A0", 1, [1.0, 2.0]), ("D", 1, [1.0, 2.0])])
    _rewrite_rows(tmp_path, lambda rows: rows[1].update(outcome="void"))
    verdicts = _module()._bits(tmp_path, {"w": [2]})["verdicts"]
    assert verdicts == {"k1-D": "void"}, verdicts


def test_a_row_must_bind_to_its_arm_record_when_the_folder_has_records(tmp_path: Path) -> None:
    import json

    _archived_bits_folder(tmp_path, [("A0", 1, [1.0, 2.0]), ("D", 1, [1.0, 2.0])])
    records = [{"arm": "A0", "outcome": "ok"}, {"arm": "C", "outcome": "ok"}]
    path = tmp_path / "arm_records.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    assert _module()._bits(tmp_path, {"w": [2]})["verdicts"] == {"k1-D": "void"}
    records[1]["arm"] = "D"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    assert _module()._bits(tmp_path, {"w": [2]})["verdicts"] == {"k1-D": "equal"}


def test_a_row_without_an_outcome_gets_no_verdict(tmp_path: Path) -> None:
    _archived_bits_folder(tmp_path, [("A0", 1, [1.0, 2.0]), ("D", 1, [1.0, 2.0])])
    _rewrite_rows(tmp_path, lambda rows: rows[1].pop("outcome"))
    verdicts = _module()._bits(tmp_path, {"w": [2]})["verdicts"]
    assert verdicts == {"k1-D": "void"}, verdicts


def test_a_row_must_bind_to_its_arm_record_by_run_not_by_position(tmp_path: Path) -> None:
    import json

    _archived_bits_folder(tmp_path, [("A0", 1, [1.0, 2.0]), ("D", 1, [1.0, 2.0])])
    _rewrite_rows(tmp_path, lambda rows: [row.update(run=n) for n, row in enumerate(rows, 1)])
    path = tmp_path / "arm_records.jsonl"
    records = [{"arm": "A0", "run": 1, "outcome": "ok"}, {"arm": "D", "run": 3, "outcome": "ok"}]
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    assert _module()._bits(tmp_path, {"w": [2]})["verdicts"] == {"k1-D": "void"}
    records[1]["run"] = 2
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    assert _module()._bits(tmp_path, {"w": [2]})["verdicts"] == {"k1-D": "equal"}


def test_a_diff_path_that_leaves_the_tensor_folder_is_refused(tmp_path: Path) -> None:
    import torch

    folder = tmp_path / "tensors"
    folder.mkdir()
    payload = {"loss": torch.tensor([1.0], dtype=torch.float64),
               "grads": {"a": torch.tensor([1.0])}}
    torch.save(payload, folder / "a0.pt")
    torch.save(payload, tmp_path / "escape.pt")
    base = {"arm": "A0", "tensor_file": "a0.pt"}
    other = {"arm": "D", "tensor_file": "../escape.pt"}
    with pytest.raises(ValueError, match="outside"):
        _module()._diff_detail(folder, base, other)
