"""Synthetic evidence fixtures test reporting refusal, not distributed execution."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_HARNESS = Path(__file__).parents[1] / "benchmarks/harness"
_SPEC = importlib.util.spec_from_file_location("h3_campaign", _HARNESS / "h3_fsdp_campaign.py")
campaign = importlib.util.module_from_spec(_SPEC)
sys.path.insert(0, str(_HARNESS))
try:
    _SPEC.loader.exec_module(campaign)
finally:
    sys.path.pop(0)


def test_failed_command_retains_log_and_exit_status(tmp_path):
    output = tmp_path / "train.log"
    with pytest.raises(RuntimeError, match="command failed"):
        campaign.run_logged([sys.executable, "-c", "raise RuntimeError('sentinel')"],
                            output, dict(os.environ), timeout=10)
    assert "sentinel" in output.read_text()
    assert json.loads(output.with_suffix(".exit.json").read_text())["returncode"] != 0
    assert json.loads(output.with_suffix(".command.json").read_text())[0] == sys.executable


def test_pilot_fsdp2_keeps_original_parameter_and_loss_dtypes():
    from types import SimpleNamespace

    import torch
    from h3_fsdp_run import configure_fsdp2_pilot_precision

    plugin = SimpleNamespace(fsdp_version=2)
    state = SimpleNamespace(fsdp_plugin=plugin)
    trainer = SimpleNamespace(accelerator=SimpleNamespace(state=state))
    configure_fsdp2_pilot_precision(trainer)
    policy = plugin.mixed_precision_policy
    assert policy.param_dtype is None  # PEFT's FP32 adapters must not become BF16
    assert policy.output_dtype is None  # retain FP32 loss alongside BF16 logits
    assert policy.reduce_dtype is torch.float32


def test_timeout_terminates_the_entire_torchrun_group(monkeypatch, tmp_path):
    calls = []

    class Process:
        pid = 12345

        def wait(self, timeout=None):
            if timeout == 1:
                raise subprocess.TimeoutExpired("torchrun", timeout)
            calls.append(("wait", timeout))
            return -15

    def popen(command, **kwargs):
        assert kwargs["start_new_session"] is True
        return Process()

    monkeypatch.setattr(campaign.platform, "system", lambda: "Linux")
    monkeypatch.setattr(campaign.subprocess, "Popen", popen)
    monkeypatch.setattr(campaign.os, "killpg", lambda pid, sig: calls.append((pid, sig)),
                        raising=False)
    with pytest.raises(subprocess.TimeoutExpired):
        campaign.run_logged(["torchrun"], tmp_path / "timeout.log", {}, 1)
    assert calls[0] == (12345, campaign.signal.SIGTERM)
    assert json.loads((tmp_path / "timeout.exit.json").read_text())["status"] == "timeout"


@pytest.fixture
def evidence(tmp_path):
    import torch

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    campaign.write_json(prepared / "prepared.json", {"cpu_fixture": False})
    arms = ["one", "two"]
    campaign.write_json(tmp_path / "campaign.json", {"arms": arms})
    for name, world in zip(arms, (1, 2)):
        folder = tmp_path / name
        folder.mkdir()
        for rank in range(world):
            campaign.write_json(folder / f"rank-{rank}.json", {
                "status": "training_passed_reload_pending", "world_size": world,
                "global_batch": 2, "steps": 5, "correctness_steps": 1,
                "warmup_steps": 1, "measured_steps": 2, "training_memory": {},
                "tokens_per_second": 10 * world, "seconds_per_optimizer_step": 1,
                "data_windows": [["a", "b"]], "fsdp_version": 2 if world == 2 else None})
            campaign.write_json(folder / f"environment-rank-{rank}.json", {
                "git_status": "", "versions": {}, "device": "SYNTHETIC",
                "host": "fixture", "commit": "abc"})
        campaign.write_json(folder / "config-rank-0.json", {
            "output": name, "training": {"gradient_accumulation_steps": 2 // world}})
        for filename in ("reload.json", "reload-checkpoint.json"):
            campaign.write_json(folder / filename, {"status": "passed"})
        for filename in ("first-gradients.pt", "first-adapter.pt"):
            torch.save({"lora_A": torch.ones(2)}, folder / filename)
    return tmp_path, prepared


def test_summary_uses_successful_complete_arms_only(evidence):
    root, prepared = evidence
    assert campaign.summarize(root, prepared)["speedup_2"] == 2
    record = root / "two/rank-1.json"
    value = campaign.read_json(record)
    value["status"] = "failed"
    campaign.write_json(record, value)
    with pytest.raises(ValueError, match="invalid arm"):
        campaign.summarize(root, prepared)
    assert campaign.read_json(root / "summary.json")["status"] != "passed"


def test_summary_detects_different_actual_data_windows(evidence):
    root, prepared = evidence
    record = root / "two/rank-0.json"
    value = campaign.read_json(record)
    value["data_windows"] = [["a", "c"]]
    campaign.write_json(record, value)
    with pytest.raises(ValueError, match="data windows"):
        campaign.summarize(root, prepared)


def test_summary_rejects_a_false_pass_with_wrong_first_step(evidence):
    import torch

    root, prepared = evidence
    torch.save({"lora_A": torch.zeros(2)}, root / "two/first-gradients.pt")
    with pytest.raises(AssertionError):
        campaign.summarize(root, prepared)


def test_cpu_fixture_cannot_enter_scaling(evidence):
    root, prepared = evidence
    campaign.write_json(prepared / "prepared.json", {"cpu_fixture": True})
    with pytest.raises(ValueError, match="CPU fixture"):
        campaign.summarize(root, prepared)


def test_ranks_with_different_precision_policies_are_rejected(evidence):
    root, prepared = evidence
    path = root / "two/rank-1.json"
    rank = campaign.read_json(path)
    rank["fsdp_policy"] = {"reduce_dtype": "bf16"}
    campaign.write_json(path, rank)
    with pytest.raises(ValueError, match="precision policies"):
        campaign.summarize(root, prepared)
