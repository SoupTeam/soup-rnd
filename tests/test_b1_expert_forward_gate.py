"""Small instrument fixtures; these are not trained-checkpoint gate results."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def _harness():
    path = Path(__file__).resolve().parents[1] / "benchmarks/harness/b1_expert_forward_gate.py"
    spec = importlib.util.spec_from_file_location("b1_forward_gate_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rule():
    return {
        "tokens": 12, "input_seed": 1234, "weight_budget_bytes": 65536,
        "negative_control_down_delta": 0.125,
    }


def _fixture():
    import torch
    from transformers import OlmoeConfig

    from tests.test_b1_expert_forward import _experts

    module, _ = _experts()
    config = OlmoeConfig(hidden_size=32, intermediate_size=16, num_experts=8, num_experts_per_tok=2)
    tensors = {
        "post_attention_layernorm.weight": torch.ones(32),
        "mlp.gate.weight": torch.randn(8, 32) * 0.1,
    }
    for expert in range(8):
        gate, up = module.gate_up_proj[expert].chunk(2, dim=0)
        tensors[f"mlp.experts.{expert}.gate_proj.weight"] = gate
        tensors[f"mlp.experts.{expert}.up_proj.weight"] = up
        tensors[f"mlp.experts.{expert}.down_proj.weight"] = module.down_proj[expert]
    return config, tensors


def test_gate_instrument_on_small_synthetic_fixture(tmp_path):
    config, tensors = _fixture()
    result = _harness()._run_cell("cpu-fp32", 0, config, tensors, tmp_path, _rule())
    assert result["verdict"] == "PASS"
    assert result["negative_differs"] and result["loads_ok"] and result["budget_ok"]
    assert result["expected_sha256"] == result["observed_sha256"]
    assert result["negative_sha256"] != result["expected_sha256"]
    assert len(result["routing_indices"]) == 12
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["reader_fetches"] == len(progress["trace"]) > 0


def test_gate_preserves_failed_instrument_progress(tmp_path):
    config, tensors = _fixture()
    rule = _rule()
    rule["weight_budget_bytes"] = 1
    with pytest.raises(ValueError, match="slot needs"):
        _harness()._run_cell("cpu-fp32", 0, config, tensors, tmp_path, rule)
    assert (tmp_path / "experts.safetensors").is_file()
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["reader_fetches"] == 0 and progress["pool"] is None


def test_checkpoint_identity_checks_full_bytes_and_extra_shards(tmp_path):
    module = _harness()
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"instrument bytes, not a real model")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    rule = {"checkpoint_files": {path.name: expected}}
    assert module._checkpoint_identity(tmp_path, rule) == {path.name: expected}
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module._checkpoint_identity(tmp_path, rule)
    (tmp_path / "extra.safetensors").touch()
    with pytest.raises(ValueError, match="shard list"):
        module._checkpoint_identity(tmp_path, rule)


def test_formal_gate_refuses_uncommitted_instrument_files():
    # Mock only the Git interface, not the gate's refusal check.
    from unittest.mock import patch

    module = _harness()
    with patch.object(module.subprocess, "check_output", return_value=b"?? new_source.py\n"):
        with pytest.raises(ValueError, match="commit the source/protocol"):
            module._frozen_source(module.ROOT)


def test_formal_gate_keeps_failure_and_refuses_to_overwrite(tmp_path, monkeypatch):
    module = _harness()
    rule = {"layers": [0], "cells": ["cpu-fp32"], "limitations": [], "package_versions": {}}
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(rule))
    monkeypatch.setattr(module, "PROTOCOL", protocol)
    monkeypatch.setattr(module, "_frozen_source", lambda root: {"unit_fixture": True})
    monkeypatch.setattr(module, "_box", lambda: {})

    def broken(*args):
        raise OSError("deliberate missing checkpoint")

    monkeypatch.setattr(module, "_checkpoint_identity", broken)
    session = tmp_path / "session"
    args = ["--model-dir", str(tmp_path), "--session", str(session), "--layer", "0",
            "--cell", "cpu-fp32", "--power-source", "server"]
    assert module.main(args) == 1
    out = session / "layer-00-cpu-fp32/result.json"
    result = json.loads(out.read_text())
    assert result["verdict"] == "NO_VERDICT" and result["exception"]["type"] == "OSError"
    preserved = out.read_bytes()
    with pytest.raises(FileExistsError):
        module.main(args)
    assert out.read_bytes() == preserved
