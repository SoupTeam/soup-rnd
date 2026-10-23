"""Controls for the trained-layer development runner, using synthetic CPU weights."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _module(name):
    path = Path(__file__).resolve().parents[1] / f"benchmarks/harness/{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("codec", ["dense", "nf4", "nf4-dq"])
def test_runner_exactness_negative_controls_and_partial_groups(tmp_path, codec):
    import torch

    config, tensors = _module("d3_expert_prepare")._fixture()
    tensors["mlp.gate.weight"] = torch.randn(8, 32, generator=torch.Generator().manual_seed(1))
    tensors["post_attention_layernorm.weight"] = torch.ones(32)
    result = _module("d3_grouped_restore_check").restore_control(
        tmp_path / codec, config=config, tensors=tensors, device="cpu", codec=codec,
    )
    assert result["passed"] and result["negative_differs"] and result["b12_equal"]
    assert len(result["groups"]) == 3 and all(row["passed"] for row in result["groups"])
    assert all(row["observed_sha256"] == result["reference_sha256"] for row in result["groups"])
    assert json.loads((tmp_path / codec / "progress.json").read_text()) == result


def test_runner_rejects_invalid_widths_without_starting_a_cell(tmp_path):
    module = _module("d3_grouped_restore_check")
    for widths in ((), (True,), (1, 1), (3,)):
        with pytest.raises(ValueError, match="widths"):
            module.restore_control(tmp_path / "invalid", config=None, tensors={},
                                   device="cpu", codec="dense", widths=widths)
    assert not (tmp_path / "invalid").exists()
