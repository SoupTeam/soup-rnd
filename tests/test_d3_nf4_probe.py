"""Decode metadata/budget and failure controls runnable without CUDA or Triton."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "benchmarks/harness/d3_grouped_nf4_probe.py"
    spec = importlib.util.spec_from_file_location("d3_nf4_probe_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(nested=False, count=2):
    import torch

    packed = [torch.zeros(33, dtype=torch.uint8) for _ in range(count)]
    states = []
    for _ in range(count):
        state = SimpleNamespace(
            shape=torch.Size([5, 13]), dtype=torch.bfloat16, quant_type="nf4", blocksize=64,
            nested=nested, code=torch.arange(16, dtype=torch.float32),
            absmax=torch.ones(2, dtype=torch.uint8 if nested else torch.float32),
        )
        if nested:
            state.state2 = SimpleNamespace(
                blocksize=256, code=torch.arange(256, dtype=torch.float32),
                absmax=torch.ones(1),
            )
            state.offset = torch.tensor(0.125)
        states.append(state)
    return packed, states


@pytest.mark.parametrize("nested", [False, True])
def test_plan_counts_odd_nibbles_and_all_nested_storage(nested, monkeypatch):
    import torch

    module = _module()
    packed, states = _inputs(nested)

    def unexpected(*args, **kwargs):
        raise AssertionError("planner must not allocate output tensors")

    monkeypatch.setattr(torch, "empty", unexpected)
    plan = module.decode_plan(packed, states, weight_budget_bytes=100000)
    assert plan["output_bytes"] == 2 * 65 * 2
    assert plan["elements_per_expert"] == 65
    assert plan["existing_input_storage_bytes"] >= 2 * (33 + 64 + (2 if nested else 8))
    required = plan["logical_weight_bytes"]
    assert module.decode_plan(packed, states, weight_budget_bytes=required)
    with pytest.raises(ValueError, match="above budget"):
        module.prepare_decode(packed, states, weight_budget_bytes=required - 1)


@pytest.mark.parametrize("defect", [
    "packed_length", "packed_dtype", "stats_size", "code_length", "wrong_code",
    "compute_dtype", "shape", "blocksize", "nested_code", "nested_scale", "offset",
])
def test_metadata_refused_before_any_pointer_kernel(defect):
    import torch

    packed, states = _inputs(nested=True)
    if defect == "packed_length":
        packed[1] = packed[1][:-1]
    elif defect == "packed_dtype":
        packed[1] = packed[1].float()
    elif defect == "stats_size":
        states[1].absmax = torch.zeros(1, dtype=torch.uint8)
    elif defect == "code_length":
        states[1].code = torch.zeros(15)
    elif defect == "wrong_code":
        states[1].code[0] = -1
    elif defect == "compute_dtype":
        states[1].dtype = torch.float16
    elif defect == "shape":
        states[0].shape = (0, 13)
    elif defect == "blocksize":
        states[0].blocksize = 63
    elif defect == "nested_code":
        states[1].state2.code = torch.zeros(255)
    elif defect == "nested_scale":
        states[1].state2.absmax = torch.zeros(2)
    else:
        states[1].offset = torch.zeros(2)
    with pytest.raises(ValueError):
        _module().prepare_decode(packed, states)


@pytest.mark.parametrize("budget", [0, -1, True, 1.0])
def test_invalid_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="positive integer"):
        _module().decode_plan(*_inputs(), weight_budget_bytes=budget)


def test_empty_or_more_than_four_experts_are_refused():
    with pytest.raises(ValueError, match="one to four"):
        _module().decode_plan([], [], weight_budget_bytes=100000)
    with pytest.raises(ValueError, match="one to four"):
        _module().decode_plan(*_inputs(count=5), weight_budget_bytes=100000)


def test_failed_probe_preserves_output_and_cannot_overwrite(tmp_path, monkeypatch):
    module = _module()

    def broken(folder):
        raise RuntimeError("injected compilation failure")

    monkeypatch.setattr(module, "run_probe", broken)
    out = tmp_path / "failed"
    assert module.main(["--out", str(out)]) == 1
    report = json.loads((out / "probe.json").read_text())
    assert report["status"] == "no_verdict" and report["d3_accepted"] is False
    saved = (out / "probe.json").read_bytes()
    with pytest.raises(FileExistsError):
        module.main(["--out", str(out)])
    assert (out / "probe.json").read_bytes() == saved


def test_lazy_import_does_not_need_triton():
    path = Path(__file__).resolve().parents[1] / "benchmarks/harness/d3_grouped_nf4_probe.py"
    code = (
        "import importlib.util,sys; "
        f"s=importlib.util.spec_from_file_location('d3_nf4', {str(path)!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "assert 'torch' not in sys.modules and 'triton' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
