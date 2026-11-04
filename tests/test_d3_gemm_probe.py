"""Probe dispatch, restoration and adverse-report controls; GPU arithmetic is separate."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "benchmarks/harness/d3_grouped_gemm_probe.py"
    spec = importlib.util.spec_from_file_location("d3_gemm_probe_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _PrecisionSettings:
    def __init__(self):
        self._reduced = True
        self.allow_bf16_reduced_precision_reduction_split_k = True
        self.allow_tf32 = False

    @property
    def allow_bf16_reduced_precision_reduction(self):
        return self._reduced

    @allow_bf16_reduced_precision_reduction.setter
    def allow_bf16_reduced_precision_reduction(self, value):
        pair = value if isinstance(value, tuple) else (value, True)
        self._reduced, self.allow_bf16_reduced_precision_reduction_split_k = pair


@pytest.mark.parametrize("backend", ["triton", "torch-grouped"])
@pytest.mark.parametrize("mode,want", [
    ("current", (True, True)), ("fp32-reduction", (False, True)),
    ("fp32-no-splitk", (False, False)),
])
def test_backend_dispatch_and_precision_are_independent(monkeypatch, backend, mode, want):
    import torch

    module = _module()
    settings = _PrecisionSettings()
    monkeypatch.setattr(torch.backends.cuda, "matmul", settings)
    calls = []

    def check(selected, progress):
        calls.append(selected)
        assert (settings.allow_bf16_reduced_precision_reduction,
                settings.allow_bf16_reduced_precision_reduction_split_k) == want
        return {"status": "unit fixture"}

    monkeypatch.setattr(module, "_run_checks", check)
    result = module.run_probe(mode, backend)
    assert calls == [backend]
    assert result["reference_settings"]["experiment"] == mode
    assert settings.allow_bf16_reduced_precision_reduction is True
    assert settings.allow_bf16_reduced_precision_reduction_split_k is True


def test_error_still_restores_precision(monkeypatch):
    import torch

    module = _module()
    settings = _PrecisionSettings()
    monkeypatch.setattr(torch.backends.cuda, "matmul", settings)

    def broken(*args):
        raise RuntimeError("injected kernel failure")

    monkeypatch.setattr(module, "_run_checks", broken)
    with pytest.raises(RuntimeError, match="kernel failure"):
        module.run_probe("fp32-no-splitk", "triton")
    assert settings.allow_bf16_reduced_precision_reduction is True
    assert settings.allow_bf16_reduced_precision_reduction_split_k is True


def test_public_wrapper_keeps_jagged_order_weights_and_empty_groups(monkeypatch):
    import torch
    import torch.nn.functional as functional

    seen = []

    def grouped(hidden, packed, *, offs):
        assert offs.dtype == torch.int32
        assert offs[-1] < hidden.shape[0]
        start = 0
        pieces = []
        for expert, end in enumerate(offs.tolist()):
            pieces.append(hidden[start:end] @ packed[expert])
            start = end
        seen.append(offs.tolist())
        return torch.cat(pieces)

    monkeypatch.setattr(functional, "grouped_mm", grouped)
    generator = torch.Generator().manual_seed(9)
    inputs = [torch.randn(rows, 32, generator=generator) for rows in (0, 1, 17, 3)]
    weights = [torch.randn(48, 32, generator=generator) for _ in inputs]
    result = _module().torch_grouped_linear(inputs, weights)
    assert seen == [[1, 18, 21]]
    for hidden, weight, actual in zip(inputs, weights, result):
        assert torch.equal(actual, functional.linear(hidden, weight))


def test_all_empty_does_not_call_backend(monkeypatch):
    import torch
    import torch.nn.functional as functional

    def unexpected(*args, **kwargs):
        raise AssertionError("empty groups should launch no GEMM")

    monkeypatch.setattr(functional, "grouped_mm", unexpected)
    outputs = _module().torch_grouped_linear([torch.empty(0, 32)], [torch.empty(48, 32)])
    assert outputs[0].shape == (0, 48)


def test_exception_keeps_partial_records_and_does_not_pass(tmp_path, monkeypatch):
    module = _module()

    def broken(reduction, backend, progress):
        assert backend == "torch-grouped"
        progress([{"fixture_record": 1}])
        raise RuntimeError("alignment failure")

    monkeypatch.setattr(module, "run_probe", broken)
    out = tmp_path / "attempt"
    assert module.main(["--backend", "torch-grouped", "--out", str(out)]) == 1
    report = json.loads((out / "probe.json").read_text())
    assert report["status"] == "no_verdict" and report["d3_accepted"] is False
    assert report["records"] == [{"fixture_record": 1}]
    assert json.loads((out / "progress.json").read_text()) == report["records"]
    original = (out / "probe.json").read_bytes()
    with pytest.raises(FileExistsError):
        module.main(["--out", str(out)])
    assert (out / "probe.json").read_bytes() == original


def test_import_does_not_require_gpu_or_triton():
    path = Path(__file__).resolve().parents[1] / "benchmarks/harness/d3_grouped_gemm_probe.py"
    code = (
        "import importlib.util,sys; "
        f"s=importlib.util.spec_from_file_location('d3_probe', {str(path)!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "assert 'torch' not in sys.modules and 'triton' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
