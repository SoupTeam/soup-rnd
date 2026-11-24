"""Ticket 15: the segmented profile campaign (py-spy leaf classes, attribution, layout).

The campaign profiles arms C (native expert GEMM) and S (``expert_gemm="segmented"``) on (8,64)
and (64,64): one traced step per cell, one py-spy snapshot per cell, then the layout and the
attribution of S's slower step. Everything here runs on synthetic inputs; no model, no CUDA.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "benchmarks" / "mbr-moonlight-segprof-20261124"
ANALYZE = CAMPAIGN / "analyze.py"
PYSPY = CAMPAIGN / "analyze_pyspy.py"
PROFILE_STEP = CAMPAIGN / "profile_step.py"


def _module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _speedscope(frames: List[Dict[str, str]], samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A ``py-spy --format speedscope`` document: shared frames, samples and their weights."""
    return {"shared": {"frames": frames},
            "profiles": [{"samples": [s["stack"] for s in samples],
                          "weights": [s["weight"] for s in samples]}]}


FRAMES = [
    {"name": "run", "file": "/repo/src/soup_cli/utils/l2l_step.py"},
    {"name": "expert_mlp", "file": "/repo/src/soup_cli/utils/expert_gemm.py"},
    {"name": "cuLaunchKernel", "file": "/usr/local/cuda/lib64/libcudart.so"},
    {"name": "read", "file": "[kernel.kptr_restrict]"},
    {"name": "mm", "file": "/venv/lib/python3.12/site-packages/torch/_ops.py"},
    {"name": "collect", "file": "/repo/src/soup_cli/utils/expert_stream.py"},
    {"name": "PyEval_EvalFrameDefault", "file": "python3.12"},
]


def _document() -> Dict[str, Any]:
    """10 s inside the step: 3 s expert GEMM, 2 s CUDA runtime, 1 s disk, 1 s torch Python,
    2 s other Python, 1 s native code; 9 s outside the step is dropped."""
    return _speedscope(FRAMES, [
        {"stack": [0, 1], "weight": 3.0},
        {"stack": [0, 2], "weight": 2.0},
        {"stack": [0, 3], "weight": 1.0},
        {"stack": [0, 4], "weight": 1.0},
        {"stack": [0, 5], "weight": 2.0},
        {"stack": [0, 6], "weight": 1.0},
        {"stack": [1], "weight": 9.0},
    ])


def test_the_leaf_classes_are_shares_of_the_sampled_step() -> None:
    result = _module(PYSPY, "segprof_pyspy").classify(_document())
    assert result["step_seconds"] == 10.0
    assert result["expert_gemm_share"] == 0.3
    assert result["cuda_runtime_share"] == 0.2
    assert result["disk_share"] == 0.1
    assert result["torch_python_share"] == 0.1
    assert result["other_python_share"] == 0.2
    assert result["native_other_share"] == 0.1


def test_the_top_leaf_frames_of_the_expert_gemm_class_are_named() -> None:
    result = _module(PYSPY, "segprof_pyspy").classify(_document())
    assert result["expert_gemm_leaf_frames"] == {"expert_mlp": 3.0}


def test_a_profile_without_a_step_sample_has_no_shares() -> None:
    document = _speedscope(FRAMES, [{"stack": [1], "weight": 4.0}])
    result = _module(PYSPY, "segprof_pyspy").classify(document)
    assert result["step_seconds"] == 0.0
    assert result["expert_gemm_share"] is None
    assert result["expert_gemm_leaf_frames"] == {}


def test_the_expert_gemm_file_wins_before_the_cuda_name_rule() -> None:
    frames = [{"name": "run", "file": "/repo/src/soup_cli/utils/l2l_step.py"},
              {"name": "cudaLaunchKernel", "file": "/repo/src/soup_cli/utils/expert_gemm.py"}]
    document = _speedscope(frames, [{"stack": [0, 1], "weight": 2.0}])
    result = _module(PYSPY, "segprof_pyspy").classify(document)
    assert result["expert_gemm_share"] == 1.0
    assert result["cuda_runtime_share"] == 0.0


def _cell(step: float, kernel: float, expert: float, **changes: Any) -> Dict[str, Any]:
    cell = {"step_seconds": step, "kernel_seconds": kernel, "expert_gemm_seconds": expert}
    cell.update(changes)
    return cell


def test_an_even_split_of_the_slower_step_is_mixed() -> None:
    """C 20 s / 18 s kernels / 15 s expert, S 23 s / 20 s kernels / 16 s expert.

    The 3 s gap is 1 s expert GEMM, 1 s other kernels, 1 s outside the kernels: no component
    reaches 0.6.
    """
    result = _module(ANALYZE, "segprof_analyze").attribute(
        _cell(20.0, 18.0, 15.0), _cell(23.0, 20.0, 16.0))
    assert result["verdict"] == "mixed"
    assert result["shares"]["expert_gemm"] == 0.3333
    assert result["shares"]["other_kernel"] == 0.3333
    assert result["shares"]["non_kernel"] == 0.3333


def test_expert_gemm_kernels_naming_the_cost_at_the_sixty_percent_line() -> None:
    """2.4 s of a 3 s gap is expert GEMM (0.8), 0.3 s other kernels, 0.3 s outside."""
    module = _module(ANALYZE, "segprof_analyze")
    result = module.attribute(_cell(20.0, 18.0, 15.0), _cell(23.0, 20.7, 17.4))
    assert result["verdict"] == "kernels"
    assert result["shares"]["expert_gemm"] == 0.8
    # exactly 0.6 is still a named verdict
    result = module.attribute(_cell(20.0, 18.0, 15.0), _cell(23.0, 20.4, 16.8))
    assert result["shares"]["expert_gemm"] == 0.6
    assert result["verdict"] == "kernels"


def test_a_gap_outside_the_kernels_is_dispatch() -> None:
    """C 20 s, S 21 s: kernels contribute 0.1 s, the rest of the second is Python dispatch."""
    result = _module(ANALYZE, "segprof_analyze").attribute(
        _cell(20.0, 18.0, 15.0), _cell(21.0, 18.1, 15.1))
    assert result["verdict"] == "dispatch"
    assert result["shares"]["non_kernel"] == 0.9


def test_an_unchanged_step_has_no_attribution() -> None:
    result = _module(ANALYZE, "segprof_analyze").attribute(
        _cell(20.0, 18.0, 15.0), _cell(20.0, 18.0, 15.0))
    assert result["verdict"] == "mixed"
    assert result["shares"] == {}


def _layout_cell(step: float = 20.51, kernel: float = 18.0, expert: float = 3.2,
                 rows: int = 512, **changes: Any) -> Dict[str, Any]:
    cell: Dict[str, Any] = {
        "step_seconds": step, "kernel_seconds": kernel, "busy_share": 0.9,
        "exposed_seconds": 0.5, "launches": 120, "expert_calls": 64,
        "expert_gemm_seconds": expert, "expert_rows": rows, "us_per_row": 6250.0,
        "kernel_table": {"k_a": {"calls": 64, "seconds": 3.2, "rows_min": 8, "rows_max": 16}},
        "by_rows": {8: {"calls": 32, "seconds": 1.6, "us_per_row": 200.0},
                    16: {"calls": 32, "seconds": 1.6, "us_per_row": 100.0}},
        "pyspy": {"expert_gemm_share": 0.12, "cuda_runtime_share": 0.4, "disk_share": 0.01},
    }
    cell.update(changes)
    return cell


def test_the_layout_table_puts_c_and_s_side_by_side_per_window() -> None:
    module = _module(ANALYZE, "segprof_analyze")
    text = module.layout_table({"(8,64)": {"C": _layout_cell(),
                                           "S": _layout_cell(23.10, 20.0, 4.1)}})
    assert ("| arm | step s | kernel s | busy share | exposed s | launches | expert calls "
            "| expert s | expert rows | µs/row | py-spy expert | py-spy cuda | py-spy disk |"
            in text), text
    assert ("| C | 20.51 | 18.00 | 0.900 | 0.50 | 120 | 64 | 3.20 | 512 | 6250.0 "
            "| 0.1200 | 0.4000 | 0.0100 |" in text), text
    assert "| S | 23.10 | 20.00 |" in text


def test_the_layout_carries_the_m_histogram_kernel_table_and_per_row_sections() -> None:
    text = _module(ANALYZE, "segprof_analyze").layout_table(
        {"(8,64)": {"S": _layout_cell()}})
    assert "### M histogram, arm S, window (8,64)" in text
    assert "| rows | calls |" in text and "| 8 | 32 |" in text
    assert "### Kernel table, arm S, window (8,64)" in text
    assert "| k_a | 64 | 3.20 | 8–16 |" in text
    assert "### Per row, arm S, window (8,64)" in text
    assert "| 8 | 32 | 1.60 | 200.0 |" in text


def test_a_layout_line_names_model_card_window_arm_mode_and_torch() -> None:
    rows = [_row_like("S", 23.10)]
    line = _module(ANALYZE, "segprof_analyze").result_lines(rows)[0]
    for field in ("Moonlight-16B-A3B", "RTX 3050 Laptop", "window (8,64)", "arm S",
                  "C + segmented", "mode speed (normal)", "torch 2.14.1+cu130", "step 23.10 s"):
        assert field in line, line


def _row_like(arm: str, step: float, *, k: int = 8, seq: int = 64,
              **changes: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "arm": arm, "k": k, "seq": seq, "kind": "speed", "mode": "normal",
        "status": "ok", "outcome": "ok", "step_seconds": step,
        "environment": {"versions": {"torch": "2.14.1+cu130"}},
    }
    row.update(changes)
    return row


def test_analyze_writes_three_files_even_without_any_measured_piece(
        tmp_path: Path) -> None:
    module = _module(ANALYZE, "segprof_analyze")
    analysis = module.analyze(tmp_path)
    assert (tmp_path / "analysis.json").exists()
    assert (tmp_path / "layout.md").exists() and (tmp_path / "rows.md").exists()
    assert analysis["windows"] == ["(8,64)", "(64,64)"]
    assert analysis["rows"] == 0 and analysis["traces"] == 0
    assert json.loads((tmp_path / "analysis.json").read_text())["traces"] == 0


def test_analyze_reads_the_trace_the_rows_and_the_pyspy_profile(tmp_path: Path) -> None:
    (tmp_path / "trace").mkdir()
    (tmp_path / "runs").mkdir()
    (tmp_path / "pyspy").mkdir()
    entry = {"row": _row_like("C", 20.51),
             "trace": {"wall_seconds": 40.0, "cpu_self_seconds": 1.0,
                       "expert_gemm": {"calls": 64, "rows_total": 512, "seconds": 3.2,
                                       "rows": {16: 64},
                                       "kernels": {"k_a": {"calls": 64, "seconds": 3.2,
                                                           "rows_min": 16, "rows_max": 16}},
                                       "by_rows": {16: {"calls": 64, "seconds": 3.2,
                                                        "us_per_row": 6250.0}}},
                       "timeline": {"kernels": 120, "copies": 2, "kernel_seconds": 18.0,
                                    "busy_seconds": 18.0, "span_seconds": 20.0,
                                    "gap_seconds": 2.0, "exposed_seconds": 0.5, "top": []}}}
    (tmp_path / "trace" / "C-k8-s64.json").write_text(json.dumps(entry), encoding="utf-8")
    (tmp_path / "runs" / "rows.jsonl").write_text(
        json.dumps(_row_like("C", 20.51)) + "\n", encoding="utf-8")
    (tmp_path / "pyspy" / "C-k8-s64.speedscope.json").write_text(
        json.dumps(_document()), encoding="utf-8")

    analysis = _module(ANALYZE, "segprof_analyze").analyze(tmp_path)
    cell = analysis["cells"]["(8,64)"]["C"]
    assert cell["step_seconds"] == 20.51
    assert cell["kernel_seconds"] == 18.0
    assert cell["expert_calls"] == 64
    assert cell["expert_rows"] == 512
    assert cell["us_per_row"] == 6250.0
    assert cell["busy_share"] == 0.9
    assert cell["pyspy"]["expert_gemm_share"] == 0.3
    assert analysis["traces"] == 1 and analysis["rows"] == 1
    assert "window (8,64)" in (tmp_path / "layout.md").read_text()


def test_the_profile_step_help_builds() -> None:
    proc = subprocess.run([sys.executable, str(PROFILE_STEP), "--help"],
                          capture_output=True, text=True, check=True)
    for flag in ("--arm", "--k", "--seq", "--steps", "--source", "--cache", "--out",
                 "--lora-targets", "--trace", "--pyspy"):
        assert flag in proc.stdout, proc.stdout
