"""Stack timing must keep Soup correctness and external structural checks hard."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def load(name):
    folder = Path(__file__).parents[1] / "benchmarks/harness"
    sys.path.insert(0, str(folder))
    try:
        spec = importlib.util.spec_from_file_location(name, folder / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(folder))


@pytest.mark.parametrize("defect", ["none", "soup", "nonfinite", "shape", "missing_dtype"])
def test_only_finite_external_drift_is_admitted(defect):
    harness = load("d2_training_baselines")
    checks = {
        dtype: {backend: {"passed": backend in ("peft", "soup"), "finite_losses": True,
                          "tensors": [{"finite": True, "shape_dtype": True}]}
                for backend in harness.BACKENDS}
        for dtype in ("float16", "bfloat16")
    }
    if defect == "soup":
        checks["float16"]["soup"]["passed"] = False
    elif defect == "nonfinite":
        checks["float16"]["liger"]["tensors"][0]["finite"] = False
    elif defect == "shape":
        checks["bfloat16"]["unsloth"]["tensors"][0]["shape_dtype"] = False
    elif defect == "missing_dtype":
        checks.pop("bfloat16")
    assert harness.admit_stack_gates(checks) == (defect == "none")
    if defect == "none":
        assert not checks["float16"]["liger"]["passed"]  # no relabelled equivalence PASS


def test_scaler_hook_receives_current_scaler_after_reset(monkeypatch):
    harness = load("d2_training_baselines")
    helper = ModuleType("unsloth.models._utils")
    calls = []

    def hook(trainer, model):
        calls.append(trainer.accelerator.scaler)
        model.accelerator_scaler = trainer.accelerator.scaler

    helper._unsloth_attach_accelerator_scaler = hook
    monkeypatch.setitem(sys.modules, "unsloth.models._utils", helper)
    model, first, reset = SimpleNamespace(), object(), object()
    harness.attach_backend_scaler(model, "unsloth", first)
    harness.attach_backend_scaler(model, "unsloth", reset)
    assert calls == [first, reset] and model.accelerator_scaler is reset


@pytest.mark.parametrize("stack,external", [(False, False), (True, False), (True, True)])
def test_external_scope_keeps_both_requested_baselines(stack, external):
    harness = load("d2_training_baselines")
    selected = harness.timing_candidates(SimpleNamespace(
        stack_comparison=stack, external_only=external))
    assert selected == (("liger", "unsloth") if external else
                        ("peft", "liger", "unsloth") if stack else harness.BACKENDS[1:])


def test_external_scope_cannot_change_strict_protocol():
    harness = load("d2_training_baselines")
    with pytest.raises(AssertionError, match="external-only requires"):
        harness.timing_candidates(SimpleNamespace(stack_comparison=False, external_only=True))


def test_boost_requires_separate_stack_protocol():
    harness = load("d2_training_baselines")
    with pytest.raises(AssertionError, match="default-boost requires"):
        harness.timing_candidates(SimpleNamespace(stack_comparison=False, default_boost=True))


@pytest.mark.parametrize("values", [(1200, 1400, 1300), (0, 1400, 1300),
                                    (float("nan"), 1400, 1300), (1200,)])
def test_boost_still_requires_complete_clock_observations(values):
    harness = load("d2_training_baselines")
    rows = [{"clock_mhz": value} for value in values]
    assert not harness.admit_clocks(rows)
    expected = len(values) == 3 and values[0] == 1200
    assert harness.admit_clocks(rows, default_boost=True) == expected


def test_operator_evidence_preserves_noncontiguous_outputs(tmp_path):
    import torch
    from safetensors.torch import load_file, save_file

    operators = load("d2_operator_probes")
    raw = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).transpose(1, 2)
    assert not raw.is_contiguous()
    a, b = operators.retained_pair(raw, raw.clone())
    path = tmp_path / "pairs.safetensors"
    save_file({"reference": a, "candidate": b}, str(path))
    actual = load_file(str(path))
    for value in actual.values():
        assert torch.equal(value, raw) and value.shape == raw.shape and value.dtype == raw.dtype


@pytest.mark.parametrize("backend", ["liger", "unsloth"])
@pytest.mark.parametrize("defect", ["none", "false_pass", "missing", "criterion"])
def test_operator_replay_uses_raw_complete_pairs(tmp_path, backend, defect):
    import torch
    from safetensors.torch import save_file

    operators = load("d2_operator_probes")
    counts = {"rmsnorm": (1, 1), "swiglu": (1, 2), "rope": (2, 2),
              "linear-ce": (1, 1), "lora-w": (1, 3), "lora-qkv": (3, 7), "lora-mlp": (1, 7)}
    result = {"backend": backend, "dtype": "float32", "relative_limit": 0.001,
              "cases": {case: {"passed": True} for case in operators.EXPECTED[backend]}}
    tensors = {
        f"{case}/{kind}/{index}/{arm}": torch.ones(2)
        for case in operators.EXPECTED[backend]
        for kind, count in zip(("output", "gradient"), counts[case])
        for index in range(count)
        for arm in ("reference", "candidate")
    }
    if defect == "false_pass":
        key = next(key for key in tensors if key.endswith("/candidate"))
        tensors[key] = torch.full((2,), 1.01)
    elif defect == "missing":
        tensors.pop(next(iter(tensors)))
    elif defect == "criterion":
        result["relative_limit"] = 1.0
    save_file(tensors, str(tmp_path / "operator-pairs.safetensors"))
    (tmp_path / "result.json").write_text(json.dumps(result))
    if defect == "none":
        operators.verify(tmp_path, backend)
    else:
        with pytest.raises(ValueError):
            operators.verify(tmp_path, backend)
