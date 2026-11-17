"""Real native updates, failed windows and infrastructure-only checkpoint resume.

GPU tests execute the study's exact L1024/k4 and L4096/k1 layouts on native-BF16
Ampere-or-newer CUDA with strict efficient attention. CPU tests cover contracts,
serialization and data rules only. Fault injection never replaces successful execution.
"""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from benchmarks.harness import p3_deepseek_quality_attention as attention
from benchmarks.harness import p3_deepseek_quality_model as model_api
from benchmarks.harness import p3_deepseek_quality_train as train
from tests.test_p3_deepseek_quality_model import _checkpoint, _training_stack


def _evidence_root(temporary: Path, name: str) -> Path:
    configured = os.environ.get("P3_QUALITY_EVIDENCE")
    if not configured:
        return temporary
    root = Path(configured) / name
    root.mkdir(parents=True, exist_ok=False)
    return root


def _gpu_model_paths(mutable: Path) -> tuple[Path, Path, Path]:
    published = _evidence_root(mutable, "shared-model")
    return published, mutable / "cache", published / "initial.safetensors"


def test_shared_gpu_cache_stays_under_pytest_root_when_evidence_is_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    published = tmp_path / "published"
    monkeypatch.setenv("P3_QUALITY_EVIDENCE", str(published))
    mutable = tmp_path / "pytest-allowed"
    mutable.mkdir()
    source_root, cache, initial = _gpu_model_paths(mutable)
    assert cache == mutable / "cache"
    assert source_root == published / "shared-model"
    assert initial == source_root / "initial.safetensors"
    assert os.path.commonpath((os.path.realpath(cache), os.path.realpath(mutable))) == (
        os.path.realpath(mutable))
    assert os.path.commonpath((os.path.realpath(cache), os.path.realpath(source_root))) != (
        os.path.realpath(source_root))


def _window(k: int, *, length: int = 8, index: int = 0) -> dict:
    batches = []
    for microbatch in range(k):
        ids = [[1 + (token + microbatch + index) % 60 for token in range(length)]]
        labels = [[-100 if token < 2 else value for token, value in enumerate(ids[0])]]
        batches.append({"input_ids": ids, "labels": labels})
    return {"window_id": f"fixture/{index}", "micro_batches": batches,
            "valid_target_count": k * (length - 2), "input_token_count": k * length}


@pytest.mark.gpu
@pytest.mark.parametrize("arm", train.ARMS)
@pytest.mark.parametrize("k", [1, 4])
def test_ampere_native_three_updates_preserve_targets_warmup_and_real_oracle(
    tmp_path: Path, arm: str, k: int,
) -> None:
    torch = _training_stack("cuda")
    source = _checkpoint(tmp_path)
    with train._deterministic_run(), model_api.open_session(
        source, tmp_path / "cache", device="cuda", prepare=True,
    ) as session:
        initial = tmp_path / "initial.safetensors"
        model_api.write_initial_adapter(session.model, initial)
        params = model_api.trainable_parameters(session.model)
        optimizer, scheduler = train.new_optimizer(params, steps=3)
        records = []
        for index in range(3):
            record, gradients = train.run_window(
                session, _window(k, index=index), arm=arm, optimizer=optimizer,
                scheduler=scheduler, capture_gradients=index == 0,
            )
            records.append(record)
            assert record["input_token_count"] == k * 8
            assert record["valid_target_count"] == k * 6
            assert record["status"] == "COMPLETE"
            assert record["attention_backend"] == "EFFICIENT_ATTENTION"
            attention.require_execution_evidence(
                record["attention"], backend="EFFICIENT_ATTENTION", dtype="bfloat16",
                min_calls=2 * k,
            )
            assert record["autocast"] is False
            assert record["routing_guard"]["outcome"] == "PASSED"
            assert record["gradient_norm_before_clip"] > 0
            assert record["runtime_load_deltas"]["expert_fetches"] > 0
            assert record["runtime_load_deltas"]["expert_bytes_read"] > 0
            assert record["resources"]["sampled_rss_peak_bytes"] > 0
            assert record["resources"]["boundary_store_host_bytes"] > 0
            if index == 0:
                assert set(gradients) == set(params)
                assert all(value.dtype == torch.float32 for value in gradients.values())
            if arm == "oracle_block64_e4m3":
                oracle = record["oracle_before_close"]
                assert oracle["complete"] is True
                assert oracle["counts"] == {
                    "native_forward": k, "oracle_recompute": k, "oracle_vjp": k,
                }
                assert oracle["natural_routing_measured"] is False
        assert [row["learning_rate_used"] for row in records] == [0.0, 0.0001, 0.0002]
        assert records[0]["adapter_before_sha256"] == records[0]["adapter_after_sha256"]
        assert all(row["adapter_before_sha256"] != row["adapter_after_sha256"]
                   for row in records[1:])


@pytest.mark.gpu
def test_candidate_boundary_payload_is_half_plus_fp32_scales(tmp_path: Path) -> None:
    _training_stack("cuda")
    source = _checkpoint(tmp_path)
    with train._deterministic_run(), model_api.open_session(
        source, tmp_path / "cache", device="cuda", prepare=True,
    ) as session:
        initial = tmp_path / "initial.safetensors"
        model_api.write_initial_adapter(session.model, initial)
        baseline, _ = train.run_window(session, _window(4), arm="bf16_l2l")
        model_api.load_initial_adapter(session.model, initial)
        candidate, _ = train.run_window(session, _window(4), arm="oracle_block64_e4m3")
        left, right = baseline["store_before_close"], candidate["store_before_close"]
        assert left["payload_bytes"] == 2 * 4 * 8 * 128 * 2
        assert right["payload_bytes"] == left["payload_bytes"] // 2
        assert right["scale_bytes"] == 2 * 4 * 8 * 2 * 4
        assert right["padding_bytes"] == 0
        assert right["owned_host_storage_bytes"] < left["owned_host_storage_bytes"]
        measured_bf16 = baseline["activation_storage"]
        measured_fp8 = candidate["activation_storage"]
        assert measured_bf16["components"]["payload_including_padding"]["host_bytes"] == 16384
        assert measured_fp8["components"]["payload_including_padding"]["host_bytes"] == 8192
        assert measured_fp8["components"]["scales"]["host_bytes"] == 512
        assert measured_fp8["host_tensor_storage_bytes"] == 8704
        assert measured_fp8["oracle_owned_storage_after_backward"]["host_bytes"] == 896
        assert measured_bf16["oracle_owned_storage_after_backward"] is None


def _inject_failure(model: Any, monkeypatch: pytest.MonkeyPatch, fault: str) -> list[Any]:
    import torch

    from benchmarks.harness.p3_deepseek_oracle import DeepseekOracleProbe
    from soup_cli.utils.l2l_step import model_parts

    if fault == "routing":
        gate = model_parts(model)[1].layers[1].inner.mlp.gate

        def change_routes(_module: Any, _args: Any, output: Any) -> Any:
            if not torch.is_grad_enabled():
                return output
            logits, weights, indices = output
            return logits, weights, (indices + 1) % 8

        return [gate.register_forward_hook(change_routes)]
    if fault == "missing_vjp":
        original = DeepseekOracleProbe._complete_replay

        def lose_vjp(probe: Any, state: Any) -> None:
            original(probe, state)
            probe._seen_vjp.clear()
            probe._counts["oracle_vjp"] = 0

        monkeypatch.setattr(DeepseekOracleProbe, "_complete_replay", lose_vjp)
        return []
    head = model_parts(model)[0].lm_head

    def corrupt_logits(_module: Any, _args: Any, output: Any) -> Any:
        return output * float("nan")

    return [head.register_forward_hook(corrupt_logits)]


@pytest.mark.gpu
@pytest.mark.parametrize("fault", ["routing", "missing_vjp", "nonfinite_loss"])
def test_native_failure_refuses_optimizer_and_keeps_failed_window_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    torch = _training_stack("cuda")

    source = _checkpoint(tmp_path)
    with train._deterministic_run(), model_api.open_session(
        source, tmp_path / "cache", device="cuda", prepare=True,
    ) as session:
        model_api.write_initial_adapter(session.model, tmp_path / "initial.safetensors")
        params = model_api.trainable_parameters(session.model)
        before = model_api.adapter_digest(params)[0]
        optimizer = torch.optim.AdamW(list(params.values()), lr=2e-4)
        handles = _inject_failure(session.model, monkeypatch, fault)
        arm = "bf16_l2l" if fault == "routing" else "oracle_block64_e4m3"
        try:
            with pytest.raises(train.WindowFailureError) as caught:
                train.run_window(session, _window(1), arm=arm, optimizer=optimizer)
        finally:
            for handle in handles:
                handle.remove()
        record = caught.value.record
        assert record["status"] == "FAILED"
        assert record["optimizer_applied"] is False
        assert optimizer.state == {}
        assert model_api.adapter_digest(params)[0] == before
        json.dumps(record, allow_nan=False)


@pytest.mark.gpu
@pytest.mark.parametrize("damage", ["missing_kernel", "extra_cross_attention"])
def test_invalid_attention_evidence_refuses_optimizer_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str,
) -> None:
    torch = _training_stack("cuda")
    source = _checkpoint(tmp_path)
    original = attention._AttentionObservation.report

    def invalid_evidence(observed: Any) -> dict:
        receipt = original(observed)
        if damage == "missing_kernel":
            for sample in receipt["samples"]:
                sample["backward"]["cuda_kernel_names"] = []
        else:
            call = copy.deepcopy(receipt["unique_calls"][0])
            call["key_shape"][2] //= 2
            call["value_shape"][2] //= 2
            sample = copy.deepcopy(receipt["samples"][0])
            sample.update(call)
            receipt["unique_calls"].append(call)
            receipt["samples"].append(sample)
        return receipt

    monkeypatch.setattr(attention._AttentionObservation, "report", invalid_evidence)
    with train._deterministic_run(), model_api.open_session(
        source, tmp_path / "cache", device="cuda", prepare=True,
    ) as session:
        model_api.write_initial_adapter(session.model, tmp_path / "initial.safetensors")
        parameters = model_api.trainable_parameters(session.model)
        before = model_api.adapter_digest(parameters)[0]
        optimizer = torch.optim.AdamW(list(parameters.values()), lr=2e-4)
        with pytest.raises(train.WindowFailureError) as caught:
            train.run_window(session, _window(1), arm="bf16_l2l", optimizer=optimizer)
        assert caught.value.record["optimizer_applied"] is False
        assert caught.value.record["attention"]["samples"][0]["forward"]["cuda_kernel_names"]
        assert optimizer.state == {}
        assert model_api.adapter_digest(parameters)[0] == before


def test_cpu_training_cannot_use_fixture_admission_to_reach_model_execution(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="FAILED"):
        train.run_training(
            source=tmp_path / "absent-source", cache=tmp_path / "cache",
            pack=tmp_path / "absent-pack", pack_sha256="a" * 64,
            initial_adapter=tmp_path / "initial", arm="bf16_l2l", length=1024,
            steps=3, output=tmp_path / "refused", device="cpu", test_fixture=True,
        )
    receipt = json.loads((tmp_path / "refused/manifest.json").read_text(encoding="utf-8"))
    assert "CUDA" in receipt["error"]["message"]
    assert not (tmp_path / "cache").exists()


@pytest.fixture(scope="module")
def study_pack(tmp_path_factory: Any) -> tuple[Path, str]:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    from benchmarks.harness import p3_deepseek_quality_data as data
    from tests.test_p3_deepseek_quality_data import _sources

    root = tmp_path_factory.mktemp("quality-train-pack")
    tokenizer_dir = root / "train-fixture"
    backend = Tokenizer(models.WordLevel(
        {"<pad>": 0, "<eos>": 1, "<unk>": 2, "p": 3, "a": 4, "b": 5},
        unk_token="<unk>",
    ))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<eos>", pad_token="<pad>", unk_token="<unk>",
    )
    tokenizer.save_pretrained(tokenizer_dir)
    spec = {"repo_id": "test-only/train-fixture", "revision": tokenizer_dir.name,
            "test_only": True, "files": {
                path.name: {"bytes": path.stat().st_size, "sha256": model_api.sha256(path)}
                for path in tokenizer_dir.iterdir() if path.is_file()
            }}
    sources = _sources(root / "sources.jsonl")
    pack = root / "pack"
    data.build_pack(sources=sources, sources_sha256=model_api.sha256(sources),
                    tokenizer_dir=tokenizer_dir, output=pack, _test_only_tokenizer=spec)
    return pack, model_api.sha256(pack / "manifest.json")


def _run_arguments(root: Path, source: Path, cache: Path, initial: Path,
                   pack: tuple[Path, str], **overrides: Any) -> dict[str, Any]:
    return {"source": source, "cache": cache, "initial_adapter": initial,
            "pack": pack[0], "pack_sha256": pack[1], "arm": "oracle_block64_e4m3",
            "length": 1024, "steps": 3, "output": root, "device": "cuda",
            "test_fixture": True, **overrides}


@pytest.fixture(scope="module")
def gpu_shared_model(tmp_path_factory: Any) -> tuple[Path, Path, Path]:
    _training_stack("cuda")
    mutable = tmp_path_factory.mktemp("quality-gpu-model")
    root, cache, initial = _gpu_model_paths(mutable)
    source = _checkpoint(root, exact_mla=True)
    with model_api.open_session(source, cache, prepare=True) as session:
        model_api.write_initial_adapter(session.model, initial)
    return source, cache, initial


@pytest.mark.gpu
@pytest.mark.parametrize("length", [1024, 4096])
@pytest.mark.parametrize("arm", train.ARMS)
def test_gpu_three_step_study_layout_saves_every_checkpoint_and_guarded_adapter(
    tmp_path: Path, study_pack: tuple[Path, str], gpu_shared_model: tuple[Path, Path, Path],
    length: int, arm: str,
) -> None:
    _training_stack("cuda")
    tmp_path = _evidence_root(tmp_path, f"{arm}-L{length}")
    source, cache, initial = gpu_shared_model
    output = tmp_path / "run"
    manifest = train.run_training(**_run_arguments(
        output, source, cache, initial, study_pack, length=length, arm=arm,
    ))
    assert manifest["status"] == "COMPLETE"
    assert manifest["completed_steps"] == 3
    assert manifest["initial_adapter_sha256"] == model_api.sha256(initial)
    rows = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2, 3]
    assert all(row["input_token_count"] == 4096 for row in rows)
    assert all(row["k"] == 4096 // length for row in rows)
    for row in rows:
        assert row["attention_backend"] == "EFFICIENT_ATTENTION"
        attention.require_execution_evidence(
            row["attention"], backend="EFFICIENT_ATTENTION", dtype="bfloat16",
            min_calls=2 * (4096 // length),
        )
        for call in row["attention"]["unique_calls"]:
            assert call["query_shape"] == call["key_shape"] == [1, 128, length, 192]
            assert call["value_shape"] == [1, 128, length, 128]
    assert manifest["identity"]["attention_policy"] == attention.policy_identity()
    assert manifest["identity"]["execution"]["attention_policy"] == attention.policy_identity()
    assert manifest["identity"]["execution"]["hardware"]["native_bf16"] is True
    for step in (1, 2, 3):
        checkpoint = output / "checkpoints" / f"step-{step:06d}"
        complete = json.loads((checkpoint / "manifest.json").read_text())
        assert complete["completed_steps"] == complete["pack_cursor"] == step
        assert set(complete["files"]) >= {"adapter.safetensors", "training-state.pt"}
    adapter = output / "final_adapter"
    saved = json.loads((adapter / "manifest.json").read_text())
    assert saved["schema"] == "p3-deepseek-quality-adapter-v1"
    assert saved["identity"]["completed_steps"] == 3
    assert saved["identity"]["attention_policy"] == attention.policy_identity()
    assert saved["save_guard"] == "assert_streamed_adapter_saved passed"
    assert all(model_api.sha256(adapter / name) == digest
               for name, digest in saved["files"].items())
    assert (output / "step-1-gradients.safetensors").is_file()
    delta = manifest["adapter_delta"]
    assert delta["delta_l2"] > 0
    assert delta["initial_l2"] > 0
    assert delta["relative_l2"] == delta["delta_l2"] / delta["initial_l2"]
    assert delta["final_parameter_sha256"] == manifest["final_parameter_sha256"]
    assert delta["final_adapter_manifest_sha256"] == model_api.sha256(adapter / "manifest.json")


@pytest.mark.gpu
@pytest.mark.parametrize("length", [1024, 4096])
@pytest.mark.parametrize("arm", train.ARMS)
def test_instance_loss_resume_matches_uninterrupted_adapter_and_digest_chain(
    tmp_path: Path, study_pack: tuple[Path, str], length: int, arm: str,
) -> None:
    torch = _training_stack("cuda")
    cache = tmp_path / "cache"
    tmp_path = _evidence_root(tmp_path, f"instance-loss-resume-{arm}-L{length}")
    source = _checkpoint(tmp_path, exact_mla=True)
    initial = tmp_path / "initial.safetensors"
    with model_api.open_session(source, cache, prepare=True) as session:
        model_api.write_initial_adapter(session.model, initial)
    full = train.run_training(
        **_run_arguments(tmp_path / "full", source, cache, initial, study_pack,
                         length=length, arm=arm))
    interrupted = tmp_path / "interrupted"
    arguments = _run_arguments(interrupted, source, cache, initial, study_pack,
                               length=length, arm=arm)
    script = "\n".join([
        "import json, os, sys", "from pathlib import Path",
        "from benchmarks.harness import p3_deepseek_quality_train as train",
        "arguments = json.loads(sys.argv[1])",
        "for name in ('source','cache','initial_adapter','pack','output'):",
        "    arguments[name] = Path(arguments[name])",
        "import torch", "torch.set_num_threads(int(sys.argv[2]))",
        "original = train._save_checkpoint", "def terminate(*args, **kwargs):",
        "    value = original(*args, **kwargs)",
        "    if kwargs['completed_steps'] == 2: os._exit(87)", "    return value",
        "train._save_checkpoint = terminate", "train.run_training(**arguments)",
    ])
    encoded = json.dumps({key: str(value) if isinstance(value, Path) else value
                          for key, value in arguments.items()})
    result = subprocess.run(
        [sys.executable, "-c", script, encoded, str(torch.get_num_threads())],
        check=False, timeout=3600,
    )
    assert result.returncode == 87
    assert not (interrupted / "manifest.json").exists()
    running = json.loads((interrupted / "RUNNING.json").read_text())
    event = tmp_path / "instance-loss.json"
    event.write_text(json.dumps({
        "event": "instance_loss", "run_identity_sha256": running["run_identity_sha256"],
        "provider": "test-subprocess", "instance_id": str(os.getpid()),
        "detected_at": "2026-11-10T00:00:00Z", "description": "worker abruptly exited87",
    }), encoding="utf-8")
    resumed = train.run_training(**_run_arguments(
        tmp_path / "resumed", source, cache, initial, study_pack,
        resume=interrupted, infrastructure_event=event, length=length, arm=arm,
    ))
    assert resumed["final_parameter_sha256"] == full["final_parameter_sha256"]
    assert resumed["digest_chain"] == full["digest_chain"]
    assert resumed["resumed_from_step"] == 2
    rows = (tmp_path / "resumed" / "steps.jsonl").read_text().splitlines()
    assert [json.loads(line)["step"] for line in rows] == [1, 2, 3]
    full_rows = [json.loads(line) for line in (tmp_path / "full" / "steps.jsonl").read_text(
        encoding="utf-8").splitlines()]
    resumed_rows = [json.loads(line) for line in rows]
    for key in ("adapter_before_sha256", "adapter_after_sha256", "loss", "digest_chain"):
        assert [row[key] for row in resumed_rows] == [row[key] for row in full_rows]
    assert resumed["identity"] == full["identity"] == running["identity"]
    full_state = torch.load(tmp_path / "full/checkpoints/step-000003/training-state.pt",
                            map_location="cpu", weights_only=True)
    resumed_state = torch.load(tmp_path / "resumed/checkpoints/step-000003/training-state.pt",
                               map_location="cpu", weights_only=True)
    assert full_state["scheduler"] == resumed_state["scheduler"]
    assert full_state["optimizer"]["param_groups"] == resumed_state["optimizer"]["param_groups"]
    for parameter, saved in full_state["optimizer"]["state"].items():
        for key, value in saved.items():
            assert torch.equal(value, resumed_state["optimizer"]["state"][parameter][key])
    assert torch.equal(full_state["rng"]["torch_cpu"], resumed_state["rng"]["torch_cpu"])
    assert all(torch.equal(left, right) for left, right in zip(
        full_state["rng"]["torch_cuda"], resumed_state["rng"]["torch_cuda"], strict=True))
    (tmp_path / "resume-identity-receipt.json").write_text(json.dumps({
        "status": "PASS", "resumed_from_step": 2,
        "final_parameter_sha256": resumed["final_parameter_sha256"],
        "digest_chain": resumed["digest_chain"], "steps_compared": 3,
        "full_manifest_sha256": model_api.sha256(tmp_path / "full" / "manifest.json"),
        "resumed_manifest_sha256": model_api.sha256(tmp_path / "resumed" / "manifest.json"),
    }, indent=2) + "\n", encoding="utf-8")


@pytest.mark.gpu
def test_seeded_run_enforces_strict_determinism_and_restores_process_state() -> None:
    import random

    torch = _training_stack("cuda")
    python_state = random.getstate()
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all()
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        with train._deterministic_run():
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            random.random()
            torch.rand(2)
            torch.rand(2, device="cuda")
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert random.getstate() == python_state
        assert torch.equal(torch.random.get_rng_state(), cpu_state)
        assert all(torch.equal(left, right) for left, right in zip(
            torch.cuda.get_rng_state_all(), cuda_state, strict=True))
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.mark.parametrize("changed", [
    "pack_sha256", "arm", "length", "initial_adapter_sha256", "attention_policy", "execution",
])
def test_resume_identity_rejects_changed_scientific_inputs(changed: str) -> None:
    identity = {"pack_sha256": "a" * 64, "arm": "bf16_l2l", "length": 1024,
                "initial_adapter_sha256": "b" * 64,
                "attention_policy": attention.policy_identity(),
                "execution": {"attention_backend": "EFFICIENT_ATTENTION", "warn_only": False}}
    proposed = {**identity, changed: "different"}
    with pytest.raises(ValueError, match="identity"):
        train.require_resume_identity(identity, proposed)


@pytest.mark.parametrize("terminal", ["COMPLETE", "FAILED"])
def test_resume_never_reopens_a_terminal_attempt(tmp_path: Path, terminal: str) -> None:
    old = tmp_path / "old"
    old.mkdir()
    (old / "manifest.json").write_text(json.dumps({"status": terminal}), encoding="utf-8")
    with pytest.raises(ValueError, match="terminal"):
        train.load_resume(old, identity={}, infrastructure_event=tmp_path / "event.json")


def test_resume_requires_explicit_infrastructure_loss_event(tmp_path: Path) -> None:
    old = tmp_path / "old"
    old.mkdir()
    (old / "RUNNING.json").write_text(json.dumps({"run_identity_sha256": "a" * 64}),
                                     encoding="utf-8")
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"event": "out_of_memory"}), encoding="utf-8")
    with pytest.raises(ValueError, match="instance_loss"):
        train.load_resume(old, identity={}, infrastructure_event=event)


def test_full_training_refuses_test_pack_without_python_fixture_admission(
    tmp_path: Path, study_pack: tuple[Path, str],
) -> None:
    with pytest.raises(RuntimeError, match="FAILED"):
        train.run_training(source=tmp_path / "missing", cache=tmp_path / "cache",
                           initial_adapter=tmp_path / "initial", pack=study_pack[0],
                           pack_sha256=study_pack[1], arm="bf16_l2l", length=1024, steps=3,
                           output=tmp_path / "refused", device="cuda")
    failed = json.loads((tmp_path / "refused" / "manifest.json").read_text())
    assert failed["status"] == "FAILED"
    assert not (tmp_path / "refused" / "final_adapter").exists()


def test_journal_recovers_only_a_torn_last_line_from_committed_checkpoint(tmp_path: Path) -> None:
    committed = [{"step": 1}, {"step": 2}]
    path = tmp_path / "steps.jsonl"
    path.write_bytes(b'{"step":1}\n{"ste')
    evidence = train._journal_rows(path, committed)
    assert evidence["trailing_partial_line_ignored"] is True
    assert evidence["last_committed_row_recovered_from_checkpoint"] is True
    path.write_text('{"step":1}\n{"step":9}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting"):
        train._journal_rows(path, committed)


def test_journal_never_discards_a_complete_uncommitted_update(tmp_path: Path) -> None:
    path = tmp_path / "steps.jsonl"
    path.write_text('{"step":1}\n{"step":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="uncommitted"):
        train._journal_rows(path, [{"step": 1}])


@pytest.mark.gpu
def test_optimizer_checkpoint_roundtrip_restores_rng_schedule_and_parameter_updates(
    tmp_path: Path,
) -> None:
    torch = _training_stack("cuda")
    from safetensors.torch import load_file

    source = _checkpoint(tmp_path)
    with train._deterministic_run(), model_api.open_session(
        source, tmp_path / "cache", device="cuda", prepare=True,
    ) as session:
        model_api.write_initial_adapter(session.model, tmp_path / "initial.safetensors")
        params = model_api.trainable_parameters(session.model)
        chain = model_api.adapter_digest(params)[0]
        identity = {"initial_parameter_sha256": chain, "arm": "oracle_block64_e4m3",
                    "length": 8, "pack_sha256": "a" * 64,
                    "initial_adapter_sha256": model_api.sha256(tmp_path / "initial.safetensors"),
                    "attention_policy": attention.policy_identity(),
                    "execution": train._execution_identity("cuda")}
        optimizer, scheduler = train.new_optimizer(params, steps=4)
        rows = []
        for index in range(2):
            record, gradients = train.run_window(
                session, _window(1, index=index), arm="oracle_block64_e4m3",
                optimizer=optimizer, scheduler=scheduler, capture_gradients=index == 0,
            )
            record.update(step=index + 1, previous_digest_chain=chain)
            chain = train._chain(chain, record)
            record["digest_chain"] = chain
            rows.append(record)
            if index == 0:
                receipt = train._save_tensors(tmp_path / "step-1-gradients.safetensors",
                                              gradients, schema="test-preclip-gradients")
                train._write_json(tmp_path / "step-1-gradients.json", receipt)
            train._append_row(tmp_path / "steps.jsonl", record)
        saved = train._save_checkpoint(
            tmp_path, parameters=params, optimizer=optimizer, scheduler=scheduler,
            identity=identity, completed_steps=2, rows=rows, digest_chain=chain,
        )
        train._write_json(tmp_path / "RUNNING.json", {
            "identity": identity, "run_identity_sha256": train._object_hash(identity),
        })
        event = tmp_path / "infrastructure-event.json"
        train._write_json(event, {
            "event": "instance_loss", "run_identity_sha256": train._object_hash(identity),
            "provider": "test", "instance_id": "ampere-native", "detected_at": "2026-11-10",
            "description": "explicit checkpoint admission qualification",
        })
        torn = tmp_path / "checkpoints" / "step-000003"
        torn.mkdir()
        (torn / ".manifest.pending.json").write_bytes(b'{"status":')
        folder, admitted, _ = train.load_resume(tmp_path, identity=identity,
                                               infrastructure_event=event)
        assert admitted == saved
        for key in ("pack_sha256", "arm", "length", "initial_adapter_sha256",
                    "attention_policy", "execution"):
            with pytest.raises(ValueError, match="identity"):
                train.load_resume(tmp_path, identity={**identity, key: "different"},
                                  infrastructure_event=event)
        expected_rng = torch.rand(4)
        expected_cuda_rng = torch.rand(4, device="cuda")
        expected_lr = scheduler.get_last_lr()
        expected_params = load_file(str(folder / "adapter.safetensors"))
        train.run_window(session, _window(1, index=2), arm="oracle_block64_e4m3",
                         optimizer=optimizer, scheduler=scheduler)
        uninterrupted_digest = model_api.adapter_digest(params)[0]
        torch.manual_seed(999)
        restored_optimizer, restored_scheduler = train.new_optimizer(params, steps=4)
        train._restore_checkpoint(session, folder, saved, restored_optimizer, restored_scheduler)
        assert torch.equal(torch.rand(4), expected_rng)
        assert restored_scheduler.get_last_lr() == expected_lr
        assert torch.equal(torch.rand(4, device="cuda"), expected_cuda_rng)
        assert all(torch.equal(value.cpu(), expected_params[name])
                   for name, value in params.items())
        assert len(restored_optimizer.state) == len(params)
        train.run_window(session, _window(1, index=2), arm="oracle_block64_e4m3",
                         optimizer=restored_optimizer, scheduler=restored_scheduler)
        assert model_api.adapter_digest(params)[0] == uninterrupted_digest
        checkpoint_manifest = folder / "manifest.json"
        committed_bytes = checkpoint_manifest.read_bytes()
        damaged = json.loads(committed_bytes)
        damaged["rows"][0]["attention"]["samples"][0]["backward"]["cuda_kernel_names"] = []
        checkpoint_manifest.write_text(json.dumps(damaged), encoding="utf-8")
        with pytest.raises(ValueError, match="attention"):
            train.load_resume(tmp_path, identity=identity, infrastructure_event=event)
        checkpoint_manifest.write_bytes(committed_bytes)
        with (folder / "training-state.pt").open("ab") as stream:
            stream.write(b"corrupt")
        with pytest.raises(ValueError, match="hash"):
            train.load_resume(tmp_path, identity=identity, infrastructure_event=event)


@pytest.mark.parametrize("field", [
    "pack_sha256", "initial_adapter_sha256", "training", "model", "attention_policy",
    "schema", "status", "include_base_model_evaluation",
])
def test_protocol_refuses_changed_training_identity(tmp_path: Path, field: str) -> None:
    protocol = {
        "schema": "p3-deepseek-quality-protocol-v1", "status": "FROZEN",
        "include_base_model_evaluation": False,
        "model": {"model_id": model_api.MODEL_ID, "revision": model_api.REVISION},
        "pack_sha256": "a" * 64, "initial_adapter_sha256": "b" * 64,
        "attention_policy": attention.policy_identity(),
        "training": {"seed": 23, "steps": 3, "lengths": [1024, 4096],
                     "input_tokens_per_step": 4096, "micro_batch_size": 1,
                     "optimizer": train.OPTIMIZER,
                     "scheduler": {"name": "cosine", "warmup_steps": 2, "horizon": 3}},
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    admitted, _ = train._protocol(path, model_api.sha256(path), steps=3, length=1024,
                                pack_sha256="a" * 64, initial_sha256="b" * 64,
                                pilot=False, test_fixture=False)
    assert admitted == protocol
    protocol[field] = "changed"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError):
        train._protocol(path, model_api.sha256(path), steps=3, length=1024,
                        pack_sha256="a" * 64, initial_sha256="b" * 64,
                        pilot=False, test_fixture=False)


def test_old_math_protocol_cannot_admit_full_training(tmp_path: Path) -> None:
    protocol = {
        "model": {"model_id": model_api.MODEL_ID, "revision": model_api.REVISION},
        "pack_sha256": "a" * 64, "initial_adapter_sha256": "b" * 64,
        "training": {"seed": 23, "steps": 3, "lengths": [1024, 4096],
                     "input_tokens_per_step": 4096, "micro_batch_size": 1,
                     "optimizer": train.OPTIMIZER,
                     "scheduler": {"name": "cosine", "warmup_steps": 2, "horizon": 3}},
    }
    path = tmp_path / "math-main-protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="attention"):
        train._protocol(path, model_api.sha256(path), steps=3, length=1024,
                        pack_sha256="a" * 64, initial_sha256="b" * 64,
                        pilot=False, test_fixture=False)


def test_prefreeze_pilot_never_admits_more_than_one_step() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        train._protocol(None, None, steps=2, length=1024, pack_sha256="a" * 64,
                        initial_sha256="b" * 64, pilot=True, test_fixture=False)


@pytest.mark.gpu
@pytest.mark.parametrize("fault", ["routing", "missing_vjp", "nonfinite_loss"])
def test_run_failure_writes_terminal_failed_manifest_and_forbids_resume(
    tmp_path: Path, study_pack: tuple[Path, str], monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    _training_stack("cuda")
    root = _evidence_root(tmp_path, f"failure-{fault}")
    source = _checkpoint(root)
    cache, initial = tmp_path / "cache", root / "initial.safetensors"
    with model_api.open_session(source, cache, device="cuda", prepare=True) as session:
        model_api.write_initial_adapter(session.model, initial)
    original = model_api.open_session
    handles = []

    def open_faulty(*args: Any, **kwargs: Any) -> Any:
        session = original(*args, **kwargs)
        handles.extend(_inject_failure(session.model, monkeypatch, fault))
        return session

    monkeypatch.setattr(model_api, "open_session", open_faulty)
    output = root / "failed"
    arm = "bf16_l2l" if fault == "routing" else "oracle_block64_e4m3"
    try:
        with pytest.raises(RuntimeError, match="FAILED"):
            train.run_training(**_run_arguments(
                output, source, cache, initial, study_pack, arm=arm,
            ))
    finally:
        for handle in handles:
            handle.remove()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    failed = json.loads((output / "failed-window.json").read_text(encoding="utf-8"))
    assert manifest["status"] == failed["status"] == "FAILED"
    assert manifest["resumable"] is False
    assert failed["optimizer_applied"] is False
    assert not (output / "final_adapter").exists()
    with pytest.raises(ValueError, match="terminal"):
        train.load_resume(output, identity={}, infrastructure_event=root / "anything.json")


def test_terminal_signal_restores_handler_and_refuses_resume() -> None:
    import signal

    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(RuntimeError, match="cannot resume"):
        with train._terminal_signals():
            handler = signal.getsignal(signal.SIGTERM)
            handler(signal.SIGTERM, None)
    assert signal.getsignal(signal.SIGTERM) == previous


def test_existing_run_directory_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "manifest.json"
    sentinel.write_text('{"status":"COMPLETE"}\n', encoding="utf-8")
    with pytest.raises(FileExistsError):
        train.run_training(source=tmp_path / "source", cache=tmp_path / "cache",
                           pack=tmp_path / "pack", pack_sha256="a" * 64,
                           initial_adapter=tmp_path / "initial", arm="bf16_l2l",
                           length=1024, steps=3, output=output)
    assert sentinel.read_text(encoding="utf-8") == '{"status":"COMPLETE"}\n'


@pytest.mark.gpu
def test_initial_adapter_cli_refuses_tiny_checkpoint(tmp_path: Path) -> None:
    _training_stack("cuda")
    source = _checkpoint(tmp_path)
    output = tmp_path / "initial.safetensors"
    result = train.main([
        "initial-adapter", "--source", str(source), "--cache", str(tmp_path / "cache"),
        "--output", str(output), "--device", "cuda", "--prepare",
    ])
    assert result == 1
    assert not output.exists()


def test_cli_has_no_fixture_bypass(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as caught:
        train.main([
            "--source", str(tmp_path), "--cache", str(tmp_path / "cache"),
            "--pack", str(tmp_path / "pack"), "--pack-sha256", "a" * 64,
            "--initial-adapter", str(tmp_path / "initial"), "--arm", "bf16_l2l",
            "--length", "1024", "--steps", "3", "--output", str(tmp_path / "output"),
            "--test-fixture",
        ])
    assert caught.value.code == 2


def test_adapter_delta_has_independent_global_frobenius_norm_and_hash_binding(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    model = torch.nn.Linear(2, 2, bias=True)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3.0, 4.0], [0.0, 0.0]]))
        model.bias.copy_(torch.tensor([0.0, 12.0]))
    initial = tmp_path / "initial.safetensors"
    save_file({"weight": model.weight.detach().clone(), "bias": model.bias.detach().clone()},
              str(initial))
    with torch.no_grad():
        model.weight.add_(torch.tensor([[0.0, 0.0], [3.0, 4.0]]))
        model.bias.add_(torch.tensor([12.0, 0.0]))
    measured = train.adapter_delta(model, initial)
    assert measured["delta_l2"] == 13.0
    assert measured["initial_l2"] == 13.0
    assert measured["relative_l2"] == 1.0
    assert measured["tensor_count"] == 2
    assert measured["element_count"] == 6
    assert measured["initial_adapter_sha256"] == model_api.sha256(initial)
    assert measured["final_parameter_sha256"] == model_api.adapter_digest(
        model_api.trainable_parameters(model))[0]


def test_adapter_delta_zero_initial_norm_is_not_divided_by_zero(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    model = torch.nn.Linear(2, 1, bias=False)
    initial = tmp_path / "zero.safetensors"
    save_file({"weight": torch.zeros((1, 2))}, str(initial))
    with torch.no_grad():
        model.weight.fill_(1.0)
    measured = train.adapter_delta(model, initial)
    assert measured["initial_l2"] == 0
    assert measured["delta_l2"] == math.sqrt(2)
    assert measured["relative_l2"] is None


@pytest.mark.parametrize("changed", [
    "torch", "cuda", "packages", "gpu", "missing", "cublas_workspace_config",
])
def test_gate_environment_cannot_admit_changed_software_or_gpu(changed: str) -> None:
    environment = {
        "torch": "2.8.0+cu128", "cuda": "12.8",
        "cublas_workspace_config": ":4096:8",
        "packages": {"transformers": "5.19.0", "peft": "0.21.0",
                     "bitsandbytes": "0.48.1", "accelerate": "1.10.1",
                     "safetensors": "0.6.2"},
        "gpu": {"name": "NVIDIA A100", "capability": [8, 0], "bf16_supported": True,
                "native_bf16": True},
    }
    train.require_gate_environment(environment, json.loads(json.dumps(environment)))
    changed_environment = json.loads(json.dumps(environment))
    if changed == "missing":
        changed_environment.pop("gpu")
    elif changed == "packages":
        changed_environment["packages"]["transformers"] = "5.20.0"
    elif changed == "gpu":
        changed_environment["gpu"]["name"] = "NVIDIA RTX 3060"
    else:
        changed_environment[changed] = "different"
    with pytest.raises(ValueError, match="environment"):
        train.require_gate_environment(changed_environment, environment)


@pytest.mark.parametrize("corruption", ["names", "shape", "missing_pass", "prefix", "live_dtype"])
def test_gate_workload_is_bound_to_live_adapter_and_training_prefix(corruption: str) -> None:
    import torch

    from benchmarks.harness.p3_deepseek_quality_gate import gate_prefix

    parameters = {"attention.lora_A": torch.nn.Parameter(torch.zeros(2, 3))}
    window = _window(4, length=1024)
    window["common_boundary_mask_interval"] = 1024
    _, _, prefix = gate_prefix(window)
    receipt = {
        "trainable_names": sorted(parameters), "input": prefix,
        "passes": {name: {"gradient_checks": {"attention.lora_A": {"shape": [2, 3]}}}
                   for name in ("ga", "ga_repeat", "bf16_l2l", "ga_floor",
                                "fp8_oracle", "int4_oracle")},
    }
    train._require_gate_workload(receipt, parameters, window)
    if corruption == "names":
        receipt["trainable_names"] = ["different.lora_A"]
    elif corruption == "shape":
        receipt["passes"]["ga"]["gradient_checks"]["attention.lora_A"]["shape"] = [3, 2]
    elif corruption == "missing_pass":
        del receipt["passes"]["ga_repeat"]
    elif corruption == "live_dtype":
        parameters["attention.lora_A"] = torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.bfloat16))
    else:
        changed = json.loads(json.dumps(window))
        changed["micro_batches"][0]["input_ids"][0][5] += 1
        receipt["input"] = gate_prefix(changed)[2]
    with pytest.raises(ValueError, match="gradient gate"):
        train._require_gate_workload(receipt, parameters, window)
