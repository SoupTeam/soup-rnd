"""Control-flow tests, not native model or numerical qualification receipts."""

from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from benchmarks.harness import p3_deepseek_quality_gate as gate
from tests.test_p3_deepseek_quality_gate import _metric_input, _window


def _run_observations(tmp_path, monkeypatch, failure):
    import torch

    from benchmarks.harness import p3_deepseek_quality_model as model

    layers, _ = _metric_input()
    names = ["attention.lora_A", "attention.lora_B"]
    parameters = {name: torch.tensor([1.0, 2.0]) for name in names}
    native = SimpleNamespace(model=SimpleNamespace(layers=[object()] * 3))
    session = SimpleNamespace(
        model=SimpleNamespace(get_base_model=lambda: native), metadata={},
        config=SimpleNamespace(num_hidden_layers=3, vocab_size=64),
    )
    closed = tmp_path / "session-closed"

    @contextmanager
    def open_session(*args, **kwargs):
        try:
            yield session
        finally:
            closed.touch()

    def observed_pass(_session, *, name, directory, **kwargs):
        directory.mkdir()
        (directory / "entered.txt").write_text(name, encoding="utf-8")
        row = {
            "status": "COMPLETE", "losses": [1.0] * 8,
            "cleanup_errors": [], "host_budget_after_close_bytes": 0,
            "adapter_unchanged": True, "layers": copy.deepcopy(layers),
        }
        factor = 1.125 if name == "ga_floor" else 2.0 if name == "int4_oracle" else 1.0
        if name == "bf16_l2l" and failure == "a3":
            factor = 2.0
        if name == "ga_floor" and failure == "floor_oom":
            row.update(status="FAILED", error={"type": "OutOfMemoryError", "message": "OOM"})
            return row, {}, None
        if name == "ga_repeat" and failure == "repeat":
            row["losses"][0] = 1.5
        if name == "fp8_oracle":
            if failure == "E3":
                factor = 1.1
            elif failure in {"E1", "E2"}:
                kind = "restore" if failure == "E1" else "branch"
                metric = row["layers"][0][kind]
                metric.update(rms_row_rel=0.5, worst_row_rel=0.5, rel_l2=0.5,
                              err_sq=0.25, row_rel_sq_sum=128.0)
            elif failure == "loss":
                row["losses"][0] = 1.5
            elif failure == "coverage":
                row["layers"].pop()
            elif failure == "candidate_oom":
                row.update(status="FAILED",
                           error={"type": "OutOfMemoryError", "message": "OOM"})
                return row, {}, None
        if name == "int4_oracle":
            for layer in row["layers"]:
                for kind in ("restore", "branch"):
                    layer[kind].update(rms_row_rel=0.5, worst_row_rel=0.5, rel_l2=0.5,
                                       err_sq=0.25, row_rel_sq_sum=128.0)
        return row, {name: value * factor for name, value in parameters.items()}, None

    monkeypatch.setattr(model, "open_session", open_session)
    monkeypatch.setattr(model, "trainable_parameters", lambda _model: parameters)
    monkeypatch.setattr(gate, "_run_pass", observed_pass)
    # Only native execution is substituted. Verdict calculations and publication are real.
    monkeypatch.setattr(gate, "require_ampere_cuda", lambda _device: {
        "name": "control-flow-test", "capability": [8, 0],
        "bf16_supported": True, "native_bf16": True,
    })
    monkeypatch.setattr(gate, "_source_hashes", lambda: {})
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    initial = tmp_path / "initial.safetensors"
    initial.write_bytes(b"control-flow-only")
    output = tmp_path / "evidence"
    result = gate.run_gate(
        source=tmp_path / "source", cache=tmp_path / "cache", initial_adapter=initial,
        output=output, window=_window(), device="cpu", dtype="bfloat16", fixture=True,
    )
    assert closed.is_file()
    assert result["qualification_only"] is True
    persisted = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert persisted == result
    assert (output / "manifest.json").is_file()
    return result, output


@pytest.mark.parametrize("failure,verdict", [("a3", "FAIL"), ("floor_oom", "NO_VERDICT")])
def test_a3_refusal_never_executes_candidate_or_control(tmp_path, monkeypatch, failure, verdict):
    result, output = _run_observations(tmp_path, monkeypatch, failure)
    assert result["verdict"] == verdict
    assert not (output / "fp8_oracle").exists()
    assert not (output / "int4_oracle").exists()
    assert result["passes"]["fp8_oracle"]["status"] == "NOT_APPLICABLE"
    if failure == "floor_oom":
        assert result["status"] == "FAILED"
        assert (output / "failure.json").is_file()


@pytest.mark.parametrize("failure", ["E1", "E2", "E3", "loss", "coverage", "candidate_oom"])
def test_candidate_refusal_never_executes_discrimination_control(tmp_path, monkeypatch, failure):
    result, output = _run_observations(tmp_path, monkeypatch, failure)
    assert result["a3"]["status"] == "PASS"
    assert result["verdict"] == ("FAIL" if failure in gate.GATES else "NO_VERDICT")
    assert not (output / "int4_oracle").exists()
    assert result["passes"]["int4_oracle"]["status"] == "NOT_APPLICABLE"
    assert (output / "fp8_oracle" / "pass.json").is_file()


def test_ga_repeat_failure_stops_before_l2l(tmp_path, monkeypatch):
    result, output = _run_observations(tmp_path, monkeypatch, "repeat")
    assert result["verdict"] == "NO_VERDICT"
    assert result["status"] == "FAILED"
    assert not (output / "bf16_l2l").exists()
    assert (output / "ga_repeat" / "pass.json").is_file()


def test_passing_candidate_still_requires_discriminating_control(tmp_path, monkeypatch):
    result, output = _run_observations(tmp_path, monkeypatch, None)
    assert result["a3"]["status"] == "PASS"
    assert all(value["status"] == "PASS" for value in result["candidate_gates"].values())
    assert result["control"]["discrimination"]["status"] == "PASS"
    assert (output / "int4_oracle" / "pass.json").is_file()
    assert result["verdict"] == "NO_VERDICT"  # A control-flow fixture cannot admit production.


def test_completed_observations_survive_final_publication_failure(tmp_path, monkeypatch):
    def failed_publish(*args):
        raise OSError("injected final publisher failure")

    monkeypatch.setattr(gate, "_publish", failed_publish)
    with pytest.raises(OSError, match="final publisher"):
        _run_observations(tmp_path, monkeypatch, "a3")
    output = tmp_path / "evidence"
    assert not (output / "manifest.json").exists()
    start = json.loads((output / "execution-start.json").read_text())
    assert start["admission_sufficient"] is False
    assert start["partial_execution"]["initial_adapter_sha256"]
    for name in ("ga", "ga_repeat", "bf16_l2l", "ga_floor"):
        record = json.loads((output / name / "pass-observation.json").read_text())
        assert record["admission_sufficient"] is False
        assert record["pass"]["losses"] == [1.0] * 8
    assert not (output / "fp8_oracle").exists()


def test_gate_refuses_imported_implementation_outside_checkout(tmp_path, monkeypatch):
    from soup_cli.utils import l2l_step

    monkeypatch.setattr(l2l_step, "__file__", str(tmp_path / "stale_install" / "l2l_step.py"))
    with pytest.raises(ValueError, match="outside the admitted checkout"):
        gate._source_hashes()
