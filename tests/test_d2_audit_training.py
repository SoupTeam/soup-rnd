"""Independent CPU audit must reject incomplete and altered full-step evidence."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def auditor():
    path = Path(__file__).parents[1] / "benchmarks/harness/d2_audit_training.py"
    spec = importlib.util.spec_from_file_location("training_auditor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_evidence(folder, auditor):
    import torch
    from safetensors.torch import save_file

    folder.mkdir()
    ready = {
        "adapter_keys": sorted(auditor.adapter_schema()),
        "fixture_sha256": "frozen",
        "harness_sha256": "frozen-source",
        "execution": {"deterministic_algorithms": True, "cublas_workspace_config": ":4096:8"},
        "environment": {
            "packages": {
                name: "same" for name in ("torch", "transformers", "peft", "triton", "safetensors")
            }
        },
    }
    result = {"steps": 50, "rows": [{"step": index, "loss": 1.0} for index in range(1, 51)]}
    for name, data in (
        ("ready", ready),
        ("result", result),
        ("validation", {"base_unchanged": True}),
    ):
        (folder / f"{name}.json").write_text(json.dumps(data))
    save_file(
        {
            f"step-{step}/{kind}/{name}": torch.ones(1)
            for step in (1, 50)
            for kind in ("weight", "gradient")
            for name in ready["adapter_keys"]
        },
        str(folder / "snapshots.safetensors"),
    )
    return folder


@pytest.mark.parametrize("defect", ["none", "weight", "nan", "loss", "both_missing"])
def test_replay_requires_full_loss_curve_and_all_tensors(auditor, tmp_path, defect):
    import torch
    from safetensors.torch import load_file, save_file

    left, right = [run_evidence(tmp_path / name, auditor) for name in ("peft", "candidate")]
    if defect == "loss":
        path = right / "result.json"
        data = json.loads(path.read_text())
        data["rows"][48]["loss"] = 1.01
        path.write_text(json.dumps(data))
    elif defect != "none":
        for folder in [left, right] if defect == "both_missing" else [right]:
            path = folder / "snapshots.safetensors"
            tensors = load_file(str(path))
            key = next(key for key in tensors if "/weight/" in key)
            if defect == "both_missing":
                tensors.pop(key)
            else:
                tensors[key] = torch.full((1,), float("nan") if defect == "nan" else 2.0)
            save_file(tensors, str(path))
    assert auditor.replay(left, right)["passed"] == (defect == "none")


def test_absent_matrix_is_incomplete(auditor, tmp_path):
    result = auditor.audit(tmp_path)
    assert not result["complete"]
    assert len(result["gates"]) == 8 and len(result["arms"]) == 72


@pytest.mark.parametrize("defect", ["missing", "mixed_mode", "mixed_cublas"])
def test_replay_rejects_missing_or_mixed_execution_settings(auditor, tmp_path, defect):
    left, right = [run_evidence(tmp_path / name, auditor) for name in ("peft", "candidate")]
    path = right / "ready.json"
    ready = json.loads(path.read_text())
    if defect == "missing":
        ready.pop("execution")
    elif defect == "mixed_mode":
        ready["execution"]["deterministic_algorithms"] = False
    else:
        ready["execution"]["cublas_workspace_config"] = None
    path.write_text(json.dumps(ready))
    assert not auditor.replay(left, right)["passed"]


@pytest.mark.parametrize("defect", ["none", "peer", "nan_power", "clock", "uuid"])
def test_gpu_context_and_telemetry_fail_closed(auditor, defect):
    row = {
        "available": True,
        "processes": [1234],
        "graphics": [],
        "clock_mhz": 1700,
        "power_mw": 200000,
        "pstate": 0,
        "throttle": 0,
        "uuid": "GPU-A",
    }
    telemetry = {
        "before": [{**row, "processes": []}],
        "context_birth": [row],
        "live": [dict(row) for _ in range(3)],
        "after": [{**row, "processes": []}],
    }
    ready = {"pid": 99, "gpu_uuid": "a"}  # container and NVML host PIDs may differ
    if defect == "peer":
        telemetry["live"][1]["processes"] = [1234, 5678]
    elif defect == "nan_power":
        telemetry["live"][1]["power_mw"] = float("nan")
    elif defect == "clock":
        telemetry["live"][1]["clock_mhz"] = 1900
    elif defect == "uuid":
        ready["gpu_uuid"] = "other"
    assert auditor.lifecycle(telemetry, ready, 0)["passed"] == (defect == "none")
