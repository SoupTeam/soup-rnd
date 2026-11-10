"""Resource timing cannot bypass full-model gate or live-input admission."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest


def _binding():
    policy = {"sha256": "a" * 64}
    metadata = {"source": {"revision": "pinned"}, "cache": {"sha256": "b" * 64},
                "precision": {"compute_dtype": "bfloat16"}}
    environment = {"torch": "pinned", "cuda": "pinned", "packages": {"peft": "pinned"},
                   "gpu": {"name": "approved-gpu"}, "cublas_workspace_config": ":4096:8"}
    gate = {"model": copy.deepcopy(metadata), "initial_adapter_sha256": "c" * 64,
            "attention_policy": copy.deepcopy(policy), "environment": copy.deepcopy(environment)}
    execution = {"attention_policy": policy, "gate_environment": environment}
    return gate, metadata, execution


@pytest.mark.parametrize("changed", ["source", "cache", "precision", "initial", "policy",
                                     "packages", "gpu"])
def test_timing_and_training_refuse_different_gate_inputs(changed):
    from benchmarks.harness.p3_deepseek_quality_train import require_gate_binding

    gate, metadata, execution = _binding()
    if changed in {"source", "cache", "precision"}:
        gate["model"][changed] = {"different": True}
    elif changed == "initial":
        gate["initial_adapter_sha256"] = "d" * 64
    elif changed == "policy":
        gate["attention_policy"] = {"sha256": "e" * 64}
    else:
        gate["environment"][changed] = {"different": True}
    with pytest.raises(ValueError):
        require_gate_binding(gate, model_metadata=metadata, initial_sha256="c" * 64,
                             execution_identity=execution)


@pytest.mark.parametrize("verdict,fixture", [("FAIL", False), ("NO_VERDICT", False),
                                            ("PASS", True), ("PASS", False)])
def test_timing_refuses_failed_fixture_or_partial_gate_before_native_execution(
    tmp_path, monkeypatch, verdict, fixture,
):
    from benchmarks.harness import p3_deepseek_quality_timing as timing

    gate = tmp_path / "gate" / "manifest.json"
    gate.parent.mkdir()
    gate.write_text(json.dumps({"schema": "p3-deepseek-quality-gate-v1", "status": "COMPLETE",
                               "verdict": verdict, "qualification_only": fixture}),
                    encoding="utf-8")

    def hardware_must_not_run(*args, **kwargs):
        raise AssertionError("invalid gate reached CUDA admission")

    monkeypatch.setattr(timing.scoring, "_admit_native_execution", hardware_must_not_run)
    output = tmp_path / "timing"
    result = timing.run_timing(
        source=tmp_path / "source", cache=tmp_path / "cache", initial=tmp_path / "initial",
        initial_sha256="c" * 64, data_root=tmp_path / "data", output=output,
        max_tokens_per_batch=8192, deadline_seconds=5400, gate=gate,
        gate_sha256=hashlib.sha256(gate.read_bytes()).hexdigest(),
    )
    assert result["status"] == "FAILED"
    assert result["error"]["type"] == "ValueError"
    assert result["deadline_seconds"] == 5400
    assert result["quality_outcomes_computed"] is False
    assert result["endpoints"] == {}
    assert "hardware" not in result
    assert (output / "manifest.json").is_file()


@pytest.mark.parametrize("deadline", [0, 5401, True])
def test_timing_invalid_bound_precedes_gate_file_access(tmp_path, deadline):
    from benchmarks.harness import p3_deepseek_quality_timing as timing

    result = timing.run_timing(
        source=tmp_path / "source", cache=tmp_path / "cache", initial=tmp_path / "initial",
        initial_sha256="c" * 64, data_root=tmp_path / "data", output=tmp_path / "timing",
        max_tokens_per_batch=8192, deadline_seconds=deadline,
        gate=tmp_path / "missing" / "manifest.json", gate_sha256="d" * 64,
    )
    assert result["status"] == "FAILED"
    assert result["error"]["type"] == "ValueError"
    assert "deadline_seconds" not in result
    assert result["endpoints"] == {}
