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


def test_conservative_bf16_sets_backend_controls_and_restores_attention(monkeypatch):
    import torch
    from h3_fsdp_run import (
        configure_conservative_bf16,
        observed_backend_controls,
        pilot_attention,
    )

    before = observed_backend_controls()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    try:
        configure_conservative_bf16()
        with pilot_attention(True):
            actual = observed_backend_controls()
            assert actual == {
                "deterministic": True, "workspace": ":4096:8",
                "float32_matmul_precision": "highest",
                "bf16_reduced_precision_reduction": False,
                "math_sdp_reduced_precision_reduction": False,
                "math_sdp": True, "flash_sdp": False,
                "efficient_sdp": False, "cudnn_sdp": False,
            }
        restored = observed_backend_controls()
        for name in ("math_sdp", "flash_sdp", "efficient_sdp", "cudnn_sdp"):
            assert restored[name] == before[name]
    finally:
        torch.set_float32_matmul_precision(before["float32_matmul_precision"])
        torch.use_deterministic_algorithms(before["deterministic"], warn_only=warn_only)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            before["bf16_reduced_precision_reduction"])
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(
            before["math_sdp_reduced_precision_reduction"])


def test_conservative_controls_cannot_disagree_across_ranks(evidence):
    root, prepared = evidence
    for arm, world in (("one", 1), ("two", 2)):
        for rank in range(world):
            path = root / arm / f"rank-{rank}.json"
            value = campaign.read_json(path)
            value.update(bf16_conservative=True, forward_backend_controls={"math_sdp": True})
            campaign.write_json(path, value)
    assert campaign.summarize(root, prepared)["speedup_2"] == 2
    path = root / "two/rank-1.json"
    value = campaign.read_json(path)
    value["forward_backend_controls"]["math_sdp"] = False
    campaign.write_json(path, value)
    with pytest.raises(ValueError, match="conservative precision controls"):
        campaign.summarize(root, prepared)


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


def test_cross_precision_arms_cannot_be_pooled(evidence):
    root, prepared = evidence
    path = root / "two/rank-0.json"
    value = campaign.read_json(path)
    value["precision"] = "fp32"
    campaign.write_json(path, value)
    with pytest.raises(ValueError, match="identities differ"):
        campaign.summarize(root, prepared)


def test_ranks_with_different_precision_policies_are_rejected(evidence):
    root, prepared = evidence
    path = root / "two/rank-1.json"
    rank = campaign.read_json(path)
    rank["fsdp_policy"] = {"reduce_dtype": "bf16"}
    campaign.write_json(path, rank)
    with pytest.raises(ValueError, match="precision policies"):
        campaign.summarize(root, prepared)


def test_rank_zero_cannot_hide_another_ranks_conservative_mode(evidence):
    root, prepared = evidence
    path = root / "two/rank-1.json"
    value = campaign.read_json(path)
    value["bf16_conservative"] = True
    campaign.write_json(path, value)
    with pytest.raises(ValueError, match="conservative precision controls"):
        campaign.summarize(root, prepared)


def complete_archive_fixture(evidence):
    root, prepared = evidence
    arm = root / "one"
    for name in ("reload-reference.pt", "adapter_model.safetensors",
                 "checkpoint-5/adapter_model.safetensors", "tokenizer.json",
                 "checkpoint-5/optimizer.bin"):
        path = arm / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"synthetic archive fixture, not GPU evidence")
    return root, prepared, arm


def test_arm_archive_is_hash_verified_and_keeps_numeric_references(evidence):
    import hashlib
    import tarfile

    root, prepared, arm = complete_archive_fixture(evidence)
    result = campaign.archive_completed_arm(root, prepared, "one")
    archive = root.parent / result["archive"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == result["sha256"]
    assert campaign.read_json(archive.with_suffix(".json")) == result
    metadata = prepared / "prepared.json"
    assert result["files"]["prepared/prepared.json"] == {
        "bytes": metadata.stat().st_size,
        "sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
    }
    with tarfile.open(archive) as source:
        names = source.getnames()
        assert "one/reload-reference.pt" in names
        assert "one/checkpoint-5/adapter_model.safetensors" in names
        assert "prepared/prepared.json" in names
        assert "one/tokenizer.json" not in names
        assert "one/checkpoint-5/optimizer.bin" not in names
    with pytest.raises(ValueError, match="overwrite"):
        campaign.archive_completed_arm(root, prepared, "one")


def test_missing_numeric_reference_cannot_produce_a_sealed_archive(evidence):
    root, prepared, arm = complete_archive_fixture(evidence)
    (arm / "reload-reference.pt").unlink()
    with pytest.raises(ValueError, match="missing numeric references"):
        campaign.archive_completed_arm(root, prepared, "one")
    assert not (root.parent / f"{root.name}-one.tar.json").exists()


def test_failed_reload_cannot_produce_a_sealed_archive(evidence):
    root, prepared, arm = complete_archive_fixture(evidence)
    campaign.write_json(arm / "reload.json", {"status": "failed"})
    with pytest.raises(ValueError, match="failed adapter reload"):
        campaign.archive_completed_arm(root, prepared, "one")


def test_collector_must_acknowledge_exact_archive_bytes(tmp_path):
    archive = tmp_path / "arm.tar.gz"
    expected = {"bytes": 100, "sha256": "expected"}
    receipt = archive.with_suffix(".receipt.json")
    campaign.write_json(receipt, expected)
    campaign.wait_for_archive_receipt(archive, expected, 1)
    campaign.write_json(receipt, {"bytes": 99, "sha256": "expected"})
    with pytest.raises(ValueError, match="does not match"):
        campaign.wait_for_archive_receipt(archive, expected, 1)


def test_missing_collection_receipt_stops_before_the_next_arm(monkeypatch, tmp_path):
    times = iter([0, 2])
    monkeypatch.setattr(campaign.time, "monotonic", lambda: next(times))
    with pytest.raises(TimeoutError, match="not acknowledged"):
        campaign.wait_for_archive_receipt(tmp_path / "arm.tar.gz", {"bytes": 1}, 1)
