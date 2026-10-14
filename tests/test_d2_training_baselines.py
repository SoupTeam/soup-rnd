"""Incomplete or incompatible baseline evidence cannot establish speedup."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def harness():
    folder = Path(__file__).parents[1] / "benchmarks/harness"
    sys.path.insert(0, str(folder))
    try:
        spec = importlib.util.spec_from_file_location(
            "d2_training", folder / "d2_training_baselines.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(folder))


def evidence(root, harness):
    import torch
    from safetensors.torch import save_file

    root.mkdir()
    ready = {
        "fixture_sha256": "same frozen fixture",
        "harness_sha256": "same frozen harness",
        "environment": {"packages": dict.fromkeys(harness.COMMON_PACKAGES, "pinned")},
        "adapter_keys": ["lora_A", "lora_B"],
    }
    result = {"steps": 2, "rows": [{"step": 1, "loss": 1.0}, {"step": 2, "loss": 0.9}]}
    for name, value in (
        ("ready", ready),
        ("result", result),
        ("validation", {"base_unchanged": True}),
    ):
        (root / f"{name}.json").write_text(json.dumps(value))
    tensors = {
        f"step-{step}/{kind}/{name}": torch.ones(2, 2)
        for step in (1, 2)
        for kind in ("weight", "gradient")
        for name in ready["adapter_keys"]
    }
    save_file(tensors, str(root / "snapshots.safetensors"))
    return root


def modify_json(path, edit):
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


@pytest.mark.parametrize("defect", ["fixture", "source", "versions", "steps", "adapter", "base"])
def test_reject_incompatible_or_incomplete_runs(harness, tmp_path, defect):
    left, right = [evidence(tmp_path / name, harness) for name in ("reference", "candidate")]
    if defect == "fixture":
        modify_json(right / "ready.json", lambda data: data.update(fixture_sha256="different"))
    elif defect == "source":
        modify_json(right / "ready.json", lambda data: data.update(harness_sha256="different"))
    elif defect == "versions":
        modify_json(
            right / "ready.json",
            lambda data: data["environment"]["packages"].update(torch="different"),
        )
    elif defect == "steps":
        modify_json(right / "result.json", lambda data: data["rows"].pop())
    elif defect == "adapter":
        modify_json(right / "ready.json", lambda data: data.update(adapter_keys=["lora_A"]))
    else:
        modify_json(right / "validation.json", lambda data: data.update(base_unchanged=False))
    with pytest.raises(AssertionError):
        harness.compare(left, right)


@pytest.mark.parametrize("defect", ["changed", "nan", "missing", "both_missing"])
def test_reject_bad_gradient_evidence(harness, tmp_path, defect):
    import torch
    from safetensors.torch import load_file, save_file

    left, right = [evidence(tmp_path / name, harness) for name in ("reference", "candidate")]
    key = "step-2/gradient/lora_A"
    for folder in [left, right] if defect == "both_missing" else [right]:
        tensors = load_file(str(folder / "snapshots.safetensors"))
        if defect in ("missing", "both_missing"):
            tensors.pop(key)
        else:
            tensors[key] = torch.full((2, 2), float("nan") if defect == "nan" else 2.0)
        save_file(tensors, str(folder / "snapshots.safetensors"))
    if defect in ("missing", "both_missing"):
        with pytest.raises(AssertionError):
            harness.compare(left, right)
    else:
        assert not harness.compare(left, right)["passed"]


def test_reject_loss_drift_despite_equal_gradients(harness, tmp_path):
    left, right = [evidence(tmp_path / name, harness) for name in ("reference", "candidate")]
    modify_json(right / "result.json", lambda data: data["rows"][1].update(loss=0.91))
    assert not harness.compare(left, right)["passed"]


def test_accept_complete_equal_evidence(harness, tmp_path):
    left, right = [evidence(tmp_path / name, harness) for name in ("reference", "candidate")]
    assert harness.compare(left, right)["passed"]


def test_reject_modified_fixture_and_path_escape(harness, tmp_path):
    payload = tmp_path / "payload"
    payload.write_bytes(b"original")
    manifest = {
        "profile": "smoke",
        "dtype": "float32",
        "files": {"payload": harness.digest_file(payload)},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert harness.verify_fixture(tmp_path) == manifest
    payload.write_bytes(b"changed")
    with pytest.raises(AssertionError, match="fixture changed"):
        harness.verify_fixture(tmp_path)
    manifest["files"] = {"../escape": "irrelevant"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AssertionError, match="fixture escape"):
        harness.verify_fixture(tmp_path)


def test_complete_cpu_step_soup_matches_peft(harness, tmp_path, monkeypatch):
    import torch

    # Text-only fixture: do not import unused sklearn/audio/vision native extensions.
    import transformers.utils.import_utils as imports

    for name in ("is_sklearn_available", "is_librosa_available", "is_torchvision_available"):
        monkeypatch.setattr(imports, name, lambda: False)
    real_profile = torch.autograd.profiler.profile

    def current_profile_api(*, use_device=None):
        # PyTorch 2.14 removed use_cuda. Keep actual event collection and training.
        return real_profile(use_device=use_device)

    monkeypatch.setattr(torch.autograd.profiler, "profile", current_profile_api)
    fixture = tmp_path / "fixture"
    harness.prepare(fixture, "smoke", "float32")
    for backend in ("peft", "soup"):
        harness.arm(
            argparse.Namespace(
                fixture=str(fixture),
                output=str(tmp_path / backend),
                backend=backend,
                device="cpu",
                phase="gate",
                steps=2,
                warmup=20,
                managed=False,
            )
        )
    assert harness.compare(tmp_path / "peft", tmp_path / "soup")["passed"]
    ready = json.loads((tmp_path / "soup/ready.json").read_text())
    assert all(ready["execution"]["patch_counts"].values())
    result = json.loads((tmp_path / "soup/result.json").read_text())
    assert result["performance_status"] == "CPU_SMOKE_ONLY"


def test_restore_loader_mutation_from_exact_checkpoint(harness, tmp_path):
    from types import SimpleNamespace

    import torch
    from safetensors.torch import save_file

    source = torch.arange(8, dtype=torch.float16).reshape(2, 4)
    name = "base_model.model.model.embed_tokens.weight"
    parameter = torch.nn.Parameter(torch.zeros_like(source), requires_grad=False)
    model = SimpleNamespace(named_parameters=lambda: [(name, parameter)])
    (tmp_path / "model").mkdir()
    save_file({"model.embed_tokens.weight": source}, str(tmp_path / "model/model.safetensors"))
    expected = {name: harness.tensor_digest(source)}
    restored = harness.restore_frozen_bases(model, tmp_path, expected, torch.float16)
    assert len(restored) == 1 and torch.equal(parameter, source)
    assert restored[0]["restored_sha256"] == expected[name]
    assert not harness.restore_frozen_bases(model, tmp_path, expected, torch.float16)
    parameter.data.zero_()
    with pytest.raises(AssertionError, match="canonical source differs"):
        harness.restore_frozen_bases(model, tmp_path, {name: "wrong hash"}, torch.float16)


def test_deterministic_control_enforces_runtime_flag(harness, tmp_path, monkeypatch):
    import torch
    import transformers.utils.import_utils as imports

    for name in ("is_sklearn_available", "is_librosa_available", "is_torchvision_available"):
        monkeypatch.setattr(imports, name, lambda: False)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "prior-test-setting")
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        fixture = tmp_path / "fixture"
        harness.prepare(fixture, "smoke", "float32")
        _, evidence = harness.load_backend(
            fixture, harness.verify_fixture(fixture), "peft", "cpu", deterministic=True
        )
        assert torch.are_deterministic_algorithms_enabled()
        assert evidence["deterministic_algorithms"] is True
        assert evidence["cublas_workspace_config"] == ":4096:8"
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)
