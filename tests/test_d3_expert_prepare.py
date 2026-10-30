"""D3 preparation and adverse controls; no claim of a grouped CUDA implementation."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "benchmarks/harness/d3_expert_prepare.py"
    spec = importlib.util.spec_from_file_location("d3_prepare_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("width", [1, 2, 4])
def test_group_plan_retains_tail_and_does_not_expand_full_expert_stack(width):
    plan = _module().group_plan([0, 2, 3, 7, 9], width=width, slot_bytes=4096, budget=16384)
    assert [expert for group in plan["groups"] for expert in group] == [0, 2, 3, 7, 9]
    assert max(map(len, plan["groups"])) <= width
    assert plan["weight_bytes_upper_bound"] == width * 4096
    assert plan["slots"] < 5


def test_group_plan_checks_one_byte_over_budget_and_empty_is_zero_slots():
    module = _module()
    assert module.group_plan([1, 3], width=2, slot_bytes=512, budget=1024)["slots"] == 2
    with pytest.raises(ValueError, match="above weight budget"):
        module.group_plan([1, 3], width=2, slot_bytes=512, budget=1023)
    assert module.group_plan([], width=4, slot_bytes=512, budget=1)["weight_bytes_upper_bound"] == 0


@pytest.mark.parametrize("active,width,slot,budget", [
    ([1, 1], 2, 512, 1024), ([2, 1], 2, 512, 1024), ([-1], 2, 512, 1024),
    ([True], 2, 512, 1024), ([1], True, 512, 1024), ([1], 0, 512, 1024),
    ([1], 2, 0, 1024), ([1], 2, 512, -1), ([1], 2.0, 512, 1024),
])
def test_group_plan_refuses_invalid_inputs(active, width, slot, budget):
    with pytest.raises(ValueError):
        _module().group_plan(active, width=width, slot_bytes=slot, budget=budget)


def test_plan_has_no_executable_formal_protocol():
    plan = _module().preparation_plan()
    assert plan["formal_execution_enabled"] is False
    assert plan["candidate_implemented"] is True
    assert plan["candidate_acceptance_complete"] is False
    assert plan["hardware"]["two_physical_ssds_required"] is False
    assert plan["remaining_before_formal_run"]


def test_draft_cannot_be_relabelled_an_executable_protocol(tmp_path, monkeypatch):
    module = _module()
    rule = json.loads(module.DRAFT.read_text())
    rule["formal_execution_enabled"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(rule))
    monkeypatch.setattr(module, "DRAFT", path)
    with pytest.raises(ValueError, match="disabled draft"):
        module.preparation_plan()


@pytest.mark.parametrize("case", ["balanced", "skewed-unused", "all-experts", "empty"])
def test_small_baseline_uses_weighted_native_loop_and_single_loads(tmp_path, case):
    result = _module().baseline_control(tmp_path / case, device="cpu", codec="dense", case=case)
    assert result["passed"] and result["loads_ok"]
    assert result["expected_sha256"] == result["observed_sha256"]
    assert "unchanged-B1.2" in result["kind"]
    assert result["physical_io_claim"] is False
    progress = json.loads((tmp_path / case / "progress.json").read_text())
    if case == "empty":
        assert result["negative_differs"] is None
        assert progress["fetches"] == 0 and result["trace"] == []
    else:
        assert result["negative_differs"] and result["non_vacuous"]
        assert progress["fetches"] == len(result["traffic"])
    if case == "skewed-unused":
        assert set(result["traffic"]) == {0, 1, 3}
        assert len(set(result["traffic"].values())) > 1


def test_routing_is_reproducible_nonunit_and_has_tail_tokens():
    import torch

    module = _module()
    first = module.routing_fixture("skewed-unused", device="cpu", dtype=torch.float32)
    second = module.routing_fixture("skewed-unused", device="cpu", dtype=torch.float32)
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    assert first[0].shape[0] == 17
    assert not torch.equal(first[2], torch.ones_like(first[2]))


@pytest.mark.parametrize("defect", ["numeric", "repeat", "negative", "nan", "zero"])
def test_exact_control_rejects_each_acceptance_defect(defect):
    import torch

    expected = torch.ones(2, 4)
    repeat, observed, negative = expected.clone(), expected.clone(), expected + 1
    if defect == "numeric":
        # Tiny but representable: allclose would accept this; exactness must fail.
        observed[0, 0] = torch.nextafter(observed[0, 0], torch.tensor(2.0))
        assert torch.allclose(expected, observed)
    elif defect == "repeat":
        repeat[0, 0] = 9
    elif defect == "negative":
        negative = expected.clone()
    elif defect == "nan":
        observed[0, 0] = float("nan")
    else:
        expected.zero_()
        repeat.zero_()
        observed.zero_()
    assert not _module().exact_control(expected, repeat, observed, negative)["passed"]


def test_control_rejects_shape_dtype_and_baseline_route_corruption(tmp_path, monkeypatch):
    import torch

    module = _module()
    with pytest.raises(ValueError, match="shape or dtype"):
        module.exact_control(torch.ones(2), torch.ones(2), torch.ones(2).double(), torch.zeros(2))
    original = module.routing_fixture

    def zero_routes(*args, **kwargs):
        hidden, index, weights = original(*args, **kwargs)
        return hidden, index, weights.zero_()

    monkeypatch.setattr(module, "routing_fixture", zero_routes)
    result = module.baseline_control(
        tmp_path / "zero", device="cpu", codec="dense", case="balanced",
    )
    assert result["loads_ok"] and not result["passed"]


def test_trace_counts_kernels_not_cpu_calls_or_memcopies():
    trace = {"traceEvents": [
        {"cat": "cpu_op", "ph": "X", "name": "aten::mm", "dur": 10},
        {"cat": "cuda_runtime", "ph": "X", "name": "cudaLaunchKernel", "dur": 2},
        {"cat": "gpu_memcpy", "ph": "X", "name": "Memcpy HtoD", "dur": 4},
        {"cat": "kernel", "ph": "X", "name": "gemm", "dur": 3},
        {"cat": "kernel", "ph": "X", "name": "gemm", "dur": 3},
    ]}
    result = _module().kernel_trace_summary(trace)
    assert result == {"cuda_kernel_events": 2, "kernels": {"gemm": 2}, "usable": True}
    assert not _module().kernel_trace_summary({"traceEvents": trace["traceEvents"][:3]})["usable"]


def test_trace_rejects_nonfinite_kernel_duration():
    with pytest.raises(ValueError, match="duration"):
        _module().kernel_trace_summary({"traceEvents": [
            {"cat": "kernel", "ph": "X", "dur": float("nan")},
        ]})


def test_cpu_preparation_persists_success_without_claiming_d3(tmp_path):
    out = tmp_path / "control"
    assert _module().main(["cpu-control", "--out", str(out)]) == 0
    report = json.loads((out / "preparation.json").read_text())
    assert report["status"] == "instrument_controls_passed"
    assert report["d3_accepted"] is False and report["formal_measurement"] is False
    assert len(report["controls"]) == 4
    assert len(report["source"]["working_file_sha256"]) == 8


def test_preparation_keeps_partial_failures_and_refuses_overwrite(tmp_path, monkeypatch):
    module = _module()
    out = tmp_path / "failed"
    original = module.baseline_control

    def broken(folder, **kwargs):
        if kwargs["case"] == "skewed-unused":
            raise OSError("injected read failure")
        return original(folder, **kwargs)

    monkeypatch.setattr(module, "baseline_control", broken)
    assert module.main(["cpu-control", "--out", str(out)]) == 1
    path = out / "preparation.json"
    report = json.loads(path.read_text())
    assert report["status"] == "not_ready" and len(report["controls"]) == 1
    assert report["exception"]["type"] == "OSError"
    saved = path.read_bytes()
    with pytest.raises(FileExistsError):
        module.main(["cpu-control", "--out", str(out)])
    assert path.read_bytes() == saved


def test_cuda_preflight_refuses_non_linux_without_a_gpu_attempt(tmp_path, monkeypatch):
    module = _module()
    monkeypatch.setattr(module.sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="Linux"):
        module.cuda_preflight(tmp_path)


def test_plan_and_help_do_not_import_heavy_dependencies():
    path = _module().ROOT / "benchmarks/harness/d3_expert_prepare.py"
    code = (
        "import importlib.util, sys; "
        f"s=importlib.util.spec_from_file_location('d3', {str(path)!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "m.preparation_plan(); "
        "assert not any(x in sys.modules for x in "
        "('torch','transformers','bitsandbytes','triton','peft'))"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
