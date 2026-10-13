"""Regressions for the opt-in E2 ABBA FP16 GradScaler wrapper.

No GPU, full model, or Accelerate installation is needed: the wrapper calls are
faked so we can test its fail-closed instrumentation contract.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

WRAPPER = Path(__file__).resolve().parents[1] / "benchmarks/e2/bench_train_amp.py"


def _load_wrapper():
    spec = importlib.util.spec_from_file_location("test_e2_bench_train_amp", WRAPPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_accelerate(monkeypatch, *, actual_scale=None):
    class Scaler:
        def __init__(self, scale):
            self.scale = scale

        def get_scale(self):
            return self.scale

    calls = []

    def original_get_grad_scaler(*args, **kwargs):
        calls.append((args, kwargs.copy()))
        value = kwargs["init_scale"] if actual_scale is None else actual_scale
        return Scaler(value)

    accelerator_module = SimpleNamespace(get_grad_scaler=original_get_grad_scaler)
    fake_accelerate = ModuleType("accelerate")
    fake_accelerate.accelerator = accelerator_module
    monkeypatch.setitem(sys.modules, "accelerate", fake_accelerate)
    return accelerator_module, original_get_grad_scaler, calls


def test_amp_wrapper_sets_scale_and_restores_original(monkeypatch, capsys):
    module = _load_wrapper()
    accelerator, original, calls = _install_fake_accelerate(monkeypatch)
    argv = ["wrapper", "--amp-init-scale", "1024", "bench", "train", "--steps", "2"]
    monkeypatch.setattr(sys, "argv", argv)

    def fake_run_module(name, *, run_name, alter_sys):
        assert (name, run_name, alter_sys) == ("soup_cli", "__main__", True)
        assert sys.argv[:3] == ["soup_cli", "bench", "train"]
        accelerator.get_grad_scaler()
        return {}

    monkeypatch.setattr(module.runpy, "run_module", fake_run_module)
    module.main()
    stdout = capsys.readouterr().out
    assert "E2_AMP_SCALER_CREATED init_scale=1024.0" in stdout
    assert "E2_AMP_SCALERS_CREATED: 1" in stdout
    assert "E2_AMP_FINAL_SCALE: 1024.0" in stdout
    assert calls[0][1]["init_scale"] == 1024.0
    assert accelerator.get_grad_scaler is original


def test_amp_wrapper_fails_if_no_scaler(monkeypatch, capsys):
    module = _load_wrapper()
    accelerator, original, _ = _install_fake_accelerate(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["wrapper", "--amp-init-scale", "1024", "bench", "train"])
    monkeypatch.setattr(module.runpy, "run_module", lambda *args, **kwargs: {})
    with pytest.raises(RuntimeError, match="Expected exactly one AMP GradScaler"):
        module.main()
    assert "E2_AMP_SCALERS_CREATED: 0" in capsys.readouterr().out
    assert accelerator.get_grad_scaler is original


def test_amp_wrapper_fails_if_multiple_scalers(monkeypatch):
    module = _load_wrapper()
    accelerator, original, _ = _install_fake_accelerate(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["wrapper", "--amp-init-scale", "1024", "bench", "train"])

    def fake_run_module(*args, **kwargs):
        accelerator.get_grad_scaler()
        accelerator.get_grad_scaler()

    monkeypatch.setattr(module.runpy, "run_module", fake_run_module)
    with pytest.raises(RuntimeError, match="More than one GradScaler"):
        module.main()
    assert accelerator.get_grad_scaler is original


def test_amp_wrapper_rejects_mismatched_initial_scale(monkeypatch):
    module = _load_wrapper()
    accelerator, original, _ = _install_fake_accelerate(monkeypatch, actual_scale=512.0)
    monkeypatch.setattr(sys, "argv", ["wrapper", "--amp-init-scale", "1024", "bench", "train"])
    monkeypatch.setattr(
        module.runpy,
        "run_module",
        lambda *args, **kwargs: accelerator.get_grad_scaler(),
    )
    with pytest.raises(RuntimeError, match="GradScaler initial scale mismatch"):
        module.main()
    assert accelerator.get_grad_scaler is original


@pytest.mark.parametrize("bad_scale", ["0", "-1", "nan", "inf"])
def test_amp_wrapper_rejects_invalid_scale(monkeypatch, bad_scale):
    module = _load_wrapper()
    monkeypatch.setattr(sys, "argv", ["wrapper", "--amp-init-scale", bad_scale, "bench", "train"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2


def test_amp_wrapper_rejects_other_commands(monkeypatch):
    module = _load_wrapper()
    monkeypatch.setattr(sys, "argv", ["wrapper", "--amp-init-scale", "1024", "train"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
