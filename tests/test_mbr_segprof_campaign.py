"""Ticket 15: the segmented profile campaign (py-spy leaf classes, attribution, layout).

The campaign counts the expert GEMMs of C (native expert GEMM) and S (``expert_gemm="segmented"``)
on (8,64) and (64,64), traces the whole step at (8,64) only, prices the M histogram with the
isolated per-M sweep and takes one py-spy snapshot per cell. Everything here runs on synthetic
inputs; no model, no CUDA.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "benchmarks" / "mbr-moonlight-segprof-20261124"
ANALYZE = CAMPAIGN / "analyze.py"
PYSPY = CAMPAIGN / "analyze_pyspy.py"
PROFILE_STEP = CAMPAIGN / "profile_step.py"
SWEEP_ROWS = CAMPAIGN / "sweep_rows.py"


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


def _cell(step: float, model: float, *, busy: Optional[float] = None) -> Dict[str, Any]:
    cell = {"step_seconds": step, "expert_seconds_model": model, "busy_seconds": busy}
    return cell


def test_the_sweep_model_naming_the_cost_at_the_sixty_percent_line() -> None:
    """2.4 s of a 3 s gap is the sweep's expert model (0.8), 0.6 s is outside it."""
    module = _module(ANALYZE, "segprof_analyze")
    result = module.attribute(_cell(20.0, 15.0), _cell(23.0, 17.4))
    assert result["verdict"] == "kernels"
    assert result["shares"]["expert_model_seconds"] == 0.8
    # exactly 0.6 is still a named verdict
    result = module.attribute(_cell(20.0, 15.0), _cell(23.0, 16.8))
    assert result["shares"]["expert_model_seconds"] == 0.6
    assert result["verdict"] == "kernels"


def test_a_gap_outside_the_kernels_is_dispatch() -> None:
    """C 20 s, S 21 s: the model adds 0.1 s, the rest of the second is outside the kernels."""
    result = _module(ANALYZE, "segprof_analyze").attribute(
        _cell(20.0, 15.0), _cell(21.0, 15.1))
    assert result["verdict"] == "dispatch"
    assert result["shares"]["non_kernel_seconds"] == 0.9


def test_an_even_split_of_the_slower_step_is_mixed() -> None:
    """C 20 s / 18 s busy, S 23 s / 19.2 s busy: 1.5 s model, 1.5 s outside it, 1.2 s busy."""
    result = _module(ANALYZE, "segprof_analyze").attribute(
        _cell(20.0, 15.0, busy=18.0), _cell(23.0, 16.5, busy=19.2))
    assert result["verdict"] == "mixed"
    assert result["shares"]["expert_model_seconds"] == 0.5
    assert result["shares"]["non_kernel_seconds"] == 0.5
    assert result["shares"]["measured_busy_seconds"] == 0.4
    assert result["measured"] is True


def test_an_untraced_cell_rests_on_the_model_and_says_so() -> None:
    result = _module(ANALYZE, "segprof_analyze").attribute(
        _cell(20.0, 15.0), _cell(23.0, 17.4))
    assert result["measured"] is False
    assert "measured_busy_seconds" not in result["shares"]


def test_a_cell_without_a_sweep_model_has_no_attribution() -> None:
    """Without the per-M price there is no expert model, so no share is guessed."""
    result = _module(ANALYZE, "segprof_analyze").attribute(
        {"step_seconds": 20.0, "expert_seconds_model": None, "busy_seconds": 18.0},
        {"step_seconds": 23.0, "expert_seconds_model": None, "busy_seconds": 18.0})
    assert result == {"verdict": "mixed", "shares": {}, "measured": False}


def test_an_unchanged_step_has_no_attribution() -> None:
    result = _module(ANALYZE, "segprof_analyze").attribute(
        _cell(20.0, 15.0), _cell(20.0, 15.0))
    assert result["verdict"] == "mixed"
    assert result["shares"] == {}


def _sweep(gate_up: float = 120.0, down: float = 60.0,
           rows: tuple = (8, 16)) -> Dict[str, Any]:
    """A ``sweep.json``-shaped document: one per-M price per projection."""
    return {"model": "Moonlight-16B-A3B", "card": "RTX 3050 Laptop sm_86 4 GiB",
            "torch": "2.14.1+cu130", "reps": 20, "warmup": 5,
            "projections": {
                "gate_up": {"weight": [2816, 2048],
                            "rows": {str(m): {"calls": 20, "us_per_call": gate_up,
                                              "us_per_row": gate_up / m} for m in rows}},
                "down": {"weight": [2048, 1408],
                         "rows": {str(m): {"calls": 20, "us_per_call": down,
                                           "us_per_row": down / m} for m in rows}}}}


def test_the_per_m_price_is_the_mean_of_the_two_projections() -> None:
    prices = _module(ANALYZE, "segprof_analyze").per_call_prices(_sweep(120.0, 60.0, (8, 16)))
    assert prices == {8: 90.0, 16: 90.0}
    assert list(prices) == [8, 16]
    assert _module(ANALYZE, "segprof_analyze").per_call_prices(None) == {}


def _row_like(arm: str, step: float, *, k: int = 8, seq: int = 64,
              **changes: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "arm": arm, "k": k, "seq": seq, "kind": "speed", "mode": "normal",
        "status": "ok", "outcome": "ok", "step_seconds": step,
        "environment": {"versions": {"torch": "2.14.1+cu130"}},
    }
    row.update(changes)
    return row


def _count_row(arm: str, step: float, *, k: int = 8, seq: int = 64, calls: int = 64,
               rows: Optional[Dict[int, int]] = None, **changes: Any) -> Dict[str, Any]:
    row = _row_like(arm, step, k=k, seq=seq, **changes)
    row["expert_gemm_calls"] = {
        "per_step": [calls], "median": calls, "rows": rows if rows is not None else {16: calls},
        "phases": {"forward": calls // 2, "backward": calls - calls // 2}, "steps": 1}
    return row


def test_the_step_seconds_are_the_median_of_the_count_runs() -> None:
    """A traced row's step does not enter: only the ``--count`` runs set the step time."""
    module = _module(ANALYZE, "segprof_analyze")
    rows = [_count_row("C", 20.0), _count_row("C", 22.0), _row_like("C", 99.0)]
    cell = module.cell(rows, prices={16: 90.0})
    assert cell["step_seconds"] == 21.0
    assert cell["expert_calls"] == 64
    assert cell["expert_rows"] == {16: 128}


def test_the_model_prices_the_merged_histogram_and_gives_microseconds_per_call() -> None:
    """64 calls of 16 rows at 90 µs: 0.00576 s of expert GEMM, 90 µs per call."""
    module = _module(ANALYZE, "segprof_analyze")
    cell = module.cell([_count_row("C", 20.0)], prices={16: 90.0})
    assert cell["expert_seconds_model"] == 0.00576
    assert cell["us_per_call"] == 90.0
    assert cell["measured"] is False
    assert cell["kernel_seconds"] is None


def test_a_row_count_without_a_sweep_price_has_no_model_and_names_the_rows() -> None:
    module = _module(ANALYZE, "segprof_analyze")
    cell = module.cell([_count_row("S", 23.0, rows={16: 32, 24: 4})], prices={16: 90.0})
    assert cell["expert_seconds_model"] is None
    assert cell["us_per_call"] is None
    assert cell["missing_rows"] == [24]


def _layout_cell(step: float = 20.51, *, calls: int = 64, model: float = 3.2,
                 busy: Optional[float] = 18.0, **changes: Any) -> Dict[str, Any]:
    cell: Dict[str, Any] = {
        "step_seconds": step, "expert_calls": calls, "expert_rows": {8: 32, 16: 32},
        "expert_seconds_model": model,
        "us_per_call": model * 1e6 / calls if model is not None and calls else None,
        "missing_rows": [], "measured": busy is not None,
        "kernel_seconds": 18.0 if busy is not None else None, "busy_seconds": busy,
        "exposed_seconds": 0.5 if busy is not None else None,
        "launches": 120 if busy is not None else None,
        "pyspy": {"expert_gemm_share": 0.12, "cuda_runtime_share": 0.4, "disk_share": 0.01,
                  "torch_python_share": 0.2, "other_python_share": 0.2,
                  "native_other_share": 0.07},
    }
    cell.update(changes)
    return cell


def test_the_layout_table_puts_c_and_s_side_by_side_per_window() -> None:
    module = _module(ANALYZE, "segprof_analyze")
    text = module.layout_table({"(8,64)": {"C": _layout_cell(),
                                           "S": _layout_cell(23.10, model=4.1)}})
    assert ("| arm | step s | expert calls | expert s (model) | µs/call | kernel s | busy s "
            "| exposed s | launches | py-spy expert | py-spy cuda | py-spy disk | measured |"
            in text), text
    assert ("| C | 20.51 | 64 | 3.20 | 50000.0 | 18.00 | 18.00 | 0.50 | 120 "
            "| 0.1200 | 0.4000 | 0.0100 | yes |" in text), text
    assert "| S | 23.10 | 64 | 4.10 |" in text


def test_the_layout_carries_the_m_histogram_of_each_arm() -> None:
    text = _module(ANALYZE, "segprof_analyze").layout_table(
        {"(8,64)": {"S": _layout_cell()}})
    assert "### M histogram, arm S, window (8,64)" in text
    assert "| rows | calls |" in text and "| 8 | 32 |" in text


def test_the_layout_carries_the_sweep_per_m_table_with_both_arms_calls() -> None:
    module = _module(ANALYZE, "segprof_analyze")
    text = module.layout_table({"(8,64)": {"C": _layout_cell(), "S": _layout_cell()}},
                               sweep=_sweep())
    assert "### Per-M price (sweep, mean of gate_up and down), window (8,64)" in text
    assert "| rows | gate_up µs | down µs | mean µs | C calls | S calls |" in text
    assert "| 8 | 120.0 | 60.0 | 90.0 | 32 | 32 |" in text


def test_the_layout_carries_the_pyspy_shares_and_the_attribution() -> None:
    module = _module(ANALYZE, "segprof_analyze")
    attribution = {"(8,64)": {"verdict": "kernels", "measured": True,
                              "shares": {"expert_model_seconds": 0.8,
                                         "non_kernel_seconds": 0.2}}}
    text = module.layout_table({"(8,64)": {"C": _layout_cell(), "S": _layout_cell()}},
                               attribution=attribution)
    assert "### py-spy leaf shares, window (8,64)" in text
    assert "| arm | expert | cuda | disk | torch py | other py | native other |" in text
    assert "| C | 0.1200 | 0.4000 | 0.0100 | 0.2000 | 0.2000 | 0.0700 |" in text
    assert "verdict: kernels (measured: true)" in text
    assert "| expert_model_seconds | 0.8000 |" in text


def test_the_layout_says_a_window_without_a_whole_step_trace() -> None:
    module = _module(ANALYZE, "segprof_analyze")
    text = module.layout_table({"(64,64)": {"C": _layout_cell(busy=None),
                                            "S": _layout_cell(23.10, busy=None)}})
    assert "OOM-killed" in text
    assert "| no |" in text


def test_a_layout_line_names_model_card_window_arm_mode_and_torch() -> None:
    rows = [_count_row("S", 23.10)]
    line = _module(ANALYZE, "segprof_analyze").result_lines(rows)[0]
    for field in ("Moonlight-16B-A3B", "RTX 3050 Laptop", "window (8,64)", "arm S",
                  "C + segmented", "mode speed (normal)", "torch 2.14.1+cu130", "step 23.10 s",
                  "expert calls 64"):
        assert field in line, line


def test_analyze_writes_three_files_even_without_any_measured_piece(
        tmp_path: Path) -> None:
    module = _module(ANALYZE, "segprof_analyze")
    analysis = module.analyze(tmp_path)
    assert (tmp_path / "analysis.json").exists()
    assert (tmp_path / "layout.md").exists() and (tmp_path / "rows.md").exists()
    assert analysis["windows"] == ["(8,64)", "(64,64)"]
    assert analysis["rows"] == 0 and analysis["traces"] == 0
    assert analysis["traced_windows"] == []
    assert json.loads((tmp_path / "analysis.json").read_text())["traces"] == 0


def test_analyze_reads_the_counters_the_trace_the_pyspy_and_the_sweep(tmp_path: Path) -> None:
    (tmp_path / "trace").mkdir()
    (tmp_path / "runs").mkdir()
    (tmp_path / "pyspy").mkdir()
    entry = {"row": _row_like("C", 20.51),
             "trace": {"wall_seconds": 40.0, "cpu_self_seconds": 0.0,
                       "expert_gemm": {"calls": 0, "rows_total": 0, "seconds": 0.0, "rows": {},
                                       "kernels": {}, "by_rows": {}},
                       "timeline": {"kernels": 120, "copies": 2, "kernel_seconds": 18.0,
                                    "busy_seconds": 18.0, "span_seconds": 20.0,
                                    "gap_seconds": 2.0, "exposed_seconds": 0.5, "top": []}}}
    (tmp_path / "trace" / "C-k8-s64.json").write_text(json.dumps(entry), encoding="utf-8")
    (tmp_path / "runs" / "rows.jsonl").write_text(
        json.dumps(_count_row("C", 20.51)) + "\n" + json.dumps(_count_row("S", 23.10)) + "\n",
        encoding="utf-8")
    (tmp_path / "pyspy" / "C-k8-s64.speedscope.json").write_text(
        json.dumps(_document()), encoding="utf-8")
    (tmp_path / "sweep.json").write_text(json.dumps(_sweep()), encoding="utf-8")

    analysis = _module(ANALYZE, "segprof_analyze").analyze(tmp_path)
    cell = analysis["cells"]["(8,64)"]["C"]
    assert cell["step_seconds"] == 20.51
    assert cell["expert_calls"] == 64
    assert cell["expert_rows"] == {16: 64}
    assert cell["expert_seconds_model"] == 0.00576
    assert cell["us_per_call"] == 90.0
    assert cell["measured"] is True
    assert cell["kernel_seconds"] == 18.0 and cell["busy_seconds"] == 18.0
    assert cell["pyspy"]["expert_gemm_share"] == 0.3
    assert analysis["cells"]["(8,64)"]["S"]["measured"] is False
    assert analysis["traced_windows"] == ["(8,64)"]
    assert analysis["traces"] == 1 and analysis["rows"] == 2
    assert analysis["attribution"]["(8,64)"]["measured"] is False
    text = (tmp_path / "layout.md").read_text()
    assert "window (8,64)" in text and "window (64,64)" in text


def test_the_sweep_report_has_one_table_per_projection_naming_model_card_and_torch() -> None:
    module = _module(SWEEP_ROWS, "segprof_sweep")
    text = "\n".join(module.report_lines(_sweep()))
    assert "Moonlight-16B-A3B" in text
    assert "RTX 3050 Laptop" in text and "2.14.1+cu130" in text
    assert text.count("| M | kernel | calls/reps | µs/call | µs/row |") == 2
    assert "## gate_up: weight (2816, 2048)" in text
    assert "## down: weight (2048, 1408)" in text
    assert "| 8 | total | 1.0 | 120.000 | 15.000 |" in text
    assert "| 8 | total | 1.0 | 60.000 | 7.500 |" in text


def test_the_sweep_refuses_a_host_off_ac_power(tmp_path: Path) -> None:
    module = _module(SWEEP_ROWS, "segprof_sweep")
    online = tmp_path / "online"
    online.write_text("1\n", encoding="utf-8")
    assert module.ac_online(online) is True
    online.write_text("0\n", encoding="utf-8")
    assert module.ac_online(online) is False
    assert module.ac_online(tmp_path / "missing") is False


def test_the_profile_step_help_builds() -> None:
    proc = subprocess.run([sys.executable, str(PROFILE_STEP), "--help"],
                          capture_output=True, text=True, check=True)
    for flag in ("--arm", "--k", "--seq", "--steps", "--source", "--cache", "--out",
                 "--lora-targets", "--trace", "--pyspy", "--count", "--gpu-util"):
        assert flag in proc.stdout, proc.stdout
