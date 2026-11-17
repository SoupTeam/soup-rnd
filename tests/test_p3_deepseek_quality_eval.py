"""Scorer qualification on approved remote hardware, never the editing workstation."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_p3_stage2_generation import tokenizer as byte_tokenizer_fixture


@pytest.fixture(name="tokenizer")
def _tokenizer():
    return byte_tokenizer_fixture.__wrapped__()


@pytest.fixture(name="evaluator")
def _evaluator():
    from benchmarks.harness import p3_deepseek_quality_eval

    return p3_deepseek_quality_eval


def _attention_policy():
    from benchmarks.harness.p3_deepseek_quality_attention import policy_identity

    return policy_identity()


def _compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path, value):
    path.write_text(_compact(value) + "\n", encoding="utf-8")


def _payload(path, value):
    value = copy.deepcopy(value)
    value["manifest_payload_sha256"] = hashlib.sha256(_compact(value).encode()).hexdigest()
    _json(path, value)
    return value


@pytest.fixture(name="twins")
def _twins(tmp_path, tokenizer):
    """Independent native and per-expert streamed V3, with identical nonzero adapters."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from benchmarks.harness import b13_moe_fixture as fixture
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream import stream_arch_of
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    config = fixture.moe_config(vocab_size=len(tokenizer), max_position_embeddings=2048)
    weights = fixture.moe_weights_dir(tmp_path, config)
    cache = tmp_path / "cache"
    device = os.environ.get("P3_QUALITY_DEVICE", "cpu")
    index = shard_checkpoint(
        str(weights), str(cache), dtype="float32", arch=stream_arch_of(config),
        expert_layout="per_expert",
    )
    streamed, runtime = build_streamed_model(
        model_id=str(weights), shard_dir=str(cache), index=index,
        lora_config=fixture.moe_lora_config(), device=device, dtype="float32",
        buffers=2, pin=False, seed=71, quant="none", tier="disk",
    )
    try:
        fixture.randomize_adapters(streamed)
        resident = fixture.resident_adapter(
            fixture.resident_base(weights), fixture.moe_lora_config(), source=streamed,
        ).to(device)
        yield SimpleNamespace(
            model=streamed, runtime=runtime, resident=resident, weights=weights, cache=cache,
            metadata={"source": {"fixture": "native-v3"}, "cache": {"sha256": "a" * 64}},
        )
    finally:
        runtime.close()


def test_streamed_tiny_v3_matches_resident_frozen_scores(evaluator, twins, tokenizer):
    from benchmarks.harness import p3_stage2v2_loglik as frozen

    requests = [("a ", "b"), ("longer context", " answer"), ("q", " xy")]
    expected = frozen.score_requests(
        twins.resident, tokenizer, requests, max_tokens_per_batch=128,
    )
    observed = evaluator.score_requests(
        twins, tokenizer, requests, max_tokens_per_batch=128, test_fixture=True,
    )
    for left, right in zip(expected, observed, strict=True):
        assert right["logprob"] == pytest.approx(left["logprob"], abs=2e-4)
        assert right["greedy"] == left["greedy"]
        assert right["n_tokens"] == left["n_tokens"]
    task = {"endpoint": "arc", "metric": "acc_norm", "choices": ["x", "long"], "gold": 1}
    assert frozen.score_task(task, observed[:2]) == frozen.score_task(task, expected[:2])
    task = {"endpoint": "lambada", "metric": "acc"}
    assert frozen.score_task(task, observed[2:]) == frozen.score_task(task, expected[2:])
    if os.environ.get("P3_QUALITY_EVIDENCE"):
        root = Path(os.environ["P3_QUALITY_EVIDENCE"]) / "scorer"
        root.mkdir(parents=True, exist_ok=True)
        _json(root / "resident-streamed-likelihood.json", {
            "device": twins.runtime.device, "requests": requests,
            "resident": expected, "streamed": observed,
        })
    # Explicit admission must not weaken the frozen resident-only public scorer.
    with pytest.raises((ValueError, RuntimeError)):
        frozen.score_requests(twins.model, tokenizer, requests, max_tokens_per_batch=128)


def test_streamed_nll_matches_frozen_response_eos_policy(evaluator, twins, tokenizer):
    from benchmarks.harness import p3_stage2v2_loglik as frozen

    expected = frozen.teacher_forced_nll(twins.resident, tokenizer, "Q:", " ok")
    observed = evaluator.teacher_forced_nll(
        twins, tokenizer, "Q:", " ok", test_fixture=True,
    )
    assert observed["n_targets"] == 4  # space, o, k, appended EOS; no prompt labels.
    assert observed["nll_sum"] == pytest.approx(expected["nll_sum"], abs=2e-4)
    assert observed["nll_mean"] == pytest.approx(expected["nll_mean"], abs=1e-4)
    if os.environ.get("P3_QUALITY_EVIDENCE"):
        root = Path(os.environ["P3_QUALITY_EVIDENCE"]) / "scorer"
        root.mkdir(parents=True, exist_ok=True)
        _json(root / "resident-streamed-nll.json", {
            "device": twins.runtime.device, "resident": expected, "streamed": observed,
        })


def test_base_mode_removes_nonzero_lora_effect_without_opening_adapter(
    evaluator, twins, tokenizer, monkeypatch,
):
    import torch

    from benchmarks.harness import b13_moe_fixture as fixture
    from benchmarks.harness import p3_stage2v2_loglik as frozen

    def forbidden(*_args, **_kwargs):
        raise AssertionError("base mode opened an adapter file")

    monkeypatch.setattr(evaluator, "admit_adapter", forbidden)
    identity = evaluator.configure_adapter(twins, None, None)
    assert identity == {"mode": "base", "adapter_manifest_sha256": None, "zero_lora_B": True}
    factors = fixture.trainables(twins.model)
    assert all(torch.count_nonzero(value).item() == 0 for name, value in factors.items()
               if ".lora_B." in name)
    reference = fixture.resident_base(twins.weights).to(twins.runtime.device)
    requests = [("context", " answer")]
    expected = frozen.score_requests(reference, tokenizer, requests, max_tokens_per_batch=128)
    observed = evaluator.score_requests(
        twins, tokenizer, requests, max_tokens_per_batch=128, test_fixture=True,
    )
    assert observed[0]["logprob"] == pytest.approx(expected[0]["logprob"], abs=2e-4)


def test_runtime_refuses_extra_hook_and_oracle_forward(evaluator, twins, tokenizer):
    hook = twins.model.lm_head.register_forward_hook(lambda *_args: None)
    try:
        with pytest.raises((ValueError, RuntimeError), match="hook"):
            evaluator.score_requests(
                twins, tokenizer, [("x", " y")], max_tokens_per_batch=16, test_fixture=True,
            )
    finally:
        hook.remove()
    gate = next(module for name, module in twins.model.named_modules() if name.endswith("mlp.gate"))
    original = gate.forward
    gate.forward = lambda *args, **kwargs: original(*args, **kwargs)
    try:
        with pytest.raises((ValueError, RuntimeError), match="forward"):
            evaluator.score_requests(
                twins, tokenizer, [("x", " y")], max_tokens_per_batch=16, test_fixture=True,
            )
    finally:
        del gate.forward


def test_scoring_preserves_rng_modes_and_padding_on_failure(evaluator, twins, tokenizer):
    import torch

    twins.model.train()
    tokenizer.padding_side = "left"
    tokenizer.backend_tokenizer.enable_truncation(max_length=8)
    state = torch.random.get_rng_state().clone()
    modes = [module.training for module in twins.model.modules()]
    backend = copy.deepcopy(tokenizer.backend_tokenizer.truncation)
    with pytest.raises(ValueError, match="context limit"):
        evaluator.score_requests(
            twins, tokenizer, [("x" * 2050, " y")], max_tokens_per_batch=8192,
            test_fixture=True,
        )
    assert torch.equal(torch.random.get_rng_state(), state)
    assert [module.training for module in twins.model.modules()] == modes
    assert tokenizer.padding_side == "left"
    assert tokenizer.backend_tokenizer.truncation == backend


def _saved_adapter(twins, root, *, protocol="b" * 64):
    root.mkdir()
    twins.model.save_pretrained(root, safe_serialization=True)
    files = {path.name: _hash(path) for path in root.iterdir() if path.is_file()}
    value = {
        "schema": "p3-deepseek-quality-adapter-v1", "files": files,
        "identity": {
            **twins.metadata, "length": 1024, "seed": 23, "arm": "bf16_l2l",
            "initial_adapter_sha256": "c" * 64, "pack_sha256": "d" * 64,
            "protocol_sha256": protocol, "completed_steps": 64,
            "attention_policy": _attention_policy(),
        },
    }
    _json(root / "manifest.json", value)
    return root, _hash(root / "manifest.json")


def test_wrong_adapter_manifest_hash_rejected_before_tensor_load(evaluator, tmp_path):
    with pytest.raises(ValueError, match="adapter manifest.*SHA256"):
        evaluator.admit_adapter(tmp_path / "missing", "0" * 64)


def test_adapter_file_tampering_unlisted_weights_and_exact_tensor_load(
    evaluator, twins, tmp_path,
):
    import torch

    from benchmarks.harness import b13_moe_fixture as fixture

    root, digest = _saved_adapter(twins, tmp_path / "adapter")
    expected = {name: value.detach().clone()
                for name, value in fixture.trainables(twins.model).items()}
    evaluator.configure_adapter(twins, None, None)
    evaluator.configure_adapter(twins, root, digest)
    for name, value in fixture.trainables(twins.model).items():
        assert torch.equal(value, expected[name])
    (root / "unlisted.safetensors").write_bytes(b"not admitted")
    with pytest.raises(ValueError, match="unlisted"):
        evaluator.admit_adapter(root, digest)
    (root / "unlisted.safetensors").unlink()
    with (root / "adapter_model.safetensors").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="file.*SHA256"):
        evaluator.admit_adapter(root, digest)


def test_adapter_wrong_source_and_missing_tensor_fail_without_partial_copy(
    evaluator, twins, tmp_path,
):
    import torch
    from safetensors.torch import load_file, save_file

    from benchmarks.harness import b13_moe_fixture as fixture

    root, _ = _saved_adapter(twins, tmp_path / "adapter")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["identity"]["source"] = {"fixture": "wrong"}
    _json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="source/cache"):
        evaluator.configure_adapter(twins, root, _hash(root / "manifest.json"))
    manifest["identity"]["source"] = twins.metadata["source"]
    tensors = load_file(str(root / "adapter_model.safetensors"))
    tensors.pop(next(iter(tensors)))
    save_file(tensors, str(root / "adapter_model.safetensors"))
    manifest["files"]["adapter_model.safetensors"] = _hash(root / "adapter_model.safetensors")
    _json(root / "manifest.json", manifest)
    before = {name: value.clone() for name, value in fixture.trainables(twins.model).items()}
    with pytest.raises(ValueError, match="tensor"):
        evaluator.configure_adapter(twins, root, _hash(root / "manifest.json"))
    assert all(torch.equal(value, before[name])
               for name, value in fixture.trainables(twins.model).items())


@pytest.fixture(name="dataset")
def _dataset(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    files, endpoints = {}, {}
    metrics = {"hellaswag": "acc_norm", "arc": "acc_norm", "boolq": "acc",
               "lambada": "acc", "gsm8k": "nll", "mbpp": "nll", "boolq_dev": "acc"}
    for endpoint, metric in metrics.items():
        development = endpoint.endswith("_dev")
        rows = []
        for index in range(4):
            task_id = f"{endpoint}@revision/test.parquet#row={index}"
            row = {
                "endpoint": endpoint, "metric": metric, "task_id": task_id,
                "source_ref": task_id,
                "source_row_sha256": hashlib.sha256(task_id.encode()).hexdigest(),
                "source_row_ordinal": index, "source_split": "train" if development else "test",
                "role": "development" if development else "evaluation", "context": "Q:",
            }
            if endpoint in ("lambada", "gsm8k", "mbpp"):
                row.update(target=" answer", gold=" answer")
            else:
                row.update(choices=["no", "yes"], gold=1)
            rows.append(row)
        folder = root / endpoint
        folder.mkdir()
        path = folder / "tasks.jsonl"
        path.write_text("".join(_compact(row) + "\n" for row in rows), encoding="utf-8")
        descriptor = {"path": f"{endpoint}/tasks.jsonl", "rows": 4,
                      "bytes": path.stat().st_size, "sha256": _hash(path)}
        ids = [row["task_id"] for row in rows]
        files[f"{endpoint}_tasks"] = descriptor
        endpoints[endpoint] = {
            "role": rows[0]["role"], "metric": metric, "retained_row_count": 4,
            "retained_task_ids": ids, "files": {"tasks": descriptor},
            "ordered_task_ids_sha256": hashlib.sha256(_compact(ids).encode()).hexdigest(),
            "source_split": "train" if development else "test",
        }
    _payload(root / "manifest.json", {
        "schema": "p3-stage2-v2-data-v1", "files": files, "endpoints": endpoints,
    })
    return root


def test_exact_shards_reject_gap_overlap_reorder_and_hash_changes(evaluator, dataset, tmp_path):
    manifest_sha = _hash(dataset / "manifest.json")
    selected, binding = evaluator.admit_tasks(
        dataset, "arc", 1, 3, manifest_sha256=manifest_sha,
    )
    assert [row["task_id"] for row in selected] == [
        "arc@revision/test.parquet#row=1", "arc@revision/test.parquet#row=2",
    ]
    assert binding["complete_endpoint_row_count"] == 4
    with (dataset / "arc/tasks.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError):
        evaluator.admit_tasks(dataset, "arc", 1, 3, manifest_sha256=manifest_sha)


def _shard(evaluator, root, *, endpoint, arm, start, stop, ids, protocol="b" * 64):
    root.mkdir()
    nll = endpoint in ("gsm8k", "mbpp")
    identity = {
        "mode": "final", "arm": arm, "length": 1024, "seed": 23,
        "adapter_manifest_sha256": ("e" if arm == "bf16_l2l" else "f") * 64,
        "initial_adapter_sha256": "c" * 64, "pack_sha256": "d" * 64,
        "protocol_sha256": protocol, "completed_steps": 64,
        "attention_policy": _attention_policy(),
        "source": {"fixture": "same"}, "cache": {"fixture": "same"},
    }
    rows = []
    for index in range(start, stop):
        result = {"pass": index % 2 == 0, "results": [
            {"logprob": -1.0, "greedy": True, "n_tokens": 1},
            {"logprob": -2.0, "greedy": False, "n_tokens": 1},
        ], "pred": 0, "pred_norm": 0}
        task_scoring = {"endpoint": endpoint, "metric": "acc_norm", "choices": ["no", "yes"],
                        "gold": 0 if index % 2 == 0 else 1}
        if endpoint == "boolq":
            task_scoring["metric"] = "acc"
        if endpoint == "lambada":
            task_scoring = {"endpoint": endpoint, "metric": "acc"}
            result["results"] = [result["results"][0]]
            result["results"][0]["greedy"] = index % 2 == 0
        if nll:
            result = {"pass": None, "nll_sum": float(index + 1), "n_targets": 2,
                      "nll_mean": (index + 1) / 2,
                      "results": [{"logprob": -float(index + 1), "n_tokens": 2, "greedy": False}]}
        rows.append({
            "endpoint": endpoint, "task_id": ids[index], "task_index": index,
            "length": 1024, "seed": 23, "adapter_identity": identity, **result,
            "task_scoring": task_scoring,
        })
    path = root / "rows.jsonl"
    path.write_text("".join(_compact(row) + "\n" for row in rows), encoding="utf-8")
    from tests.test_p3_deepseek_quality_attention import _serialized_evidence

    batches = root / "batches.jsonl"
    batches.write_text(_compact({
        "event": "batch_complete", "forward_calls": 1, "attention": [_serialized_evidence()],
    }) + "\n", encoding="utf-8")
    _payload(root / "manifest.json", {
        "schema": "p3-deepseek-quality-eval-v1", "status": "COMPLETE",
        "selected_shard_complete": True, "endpoint": endpoint, "arm": arm,
        "length": 1024, "seed": 23, "protocol_sha256": protocol,
        "adapter_identity": identity,
        "attention_policy": _attention_policy(),
        "source_files_sha256": evaluator._source_files(),
        "model": {"source": identity["source"], "cache": identity["cache"]},
        "binding": {"start": start, "stop": stop, "complete_endpoint_row_count": len(ids),
                    "ordered_task_ids": ids[start:stop], "data_manifest_sha256": "a" * 64,
                    "endpoint_ordered_task_ids_sha256": hashlib.sha256(
                        _compact(ids).encode()).hexdigest()},
        "files": {
            "rows.jsonl": {"sha256": _hash(path), "bytes": path.stat().st_size},
            "batches.jsonl": {"sha256": _hash(batches), "bytes": batches.stat().st_size},
        },
    })
    return root


@pytest.mark.parametrize("damage", [
    "policy", "missing_batches", "empty_attention", "math", "missing_forward", "failed_batch",
    "executor_source", "policy_source",
])
def test_pairing_refuses_rehashed_attention_contract_corruption(evaluator, tmp_path, damage):
    root = _shard(evaluator, tmp_path / "shard", endpoint="arc", arm="bf16_l2l",
                  start=0, stop=1, ids=["arc/0"])
    manifest = json.loads((root / "manifest.json").read_text())
    manifest.pop("manifest_payload_sha256")
    if damage == "policy":
        manifest["adapter_identity"].pop("attention_policy")
        rows = root / "rows.jsonl"
        values = [json.loads(line) for line in rows.read_text().splitlines()]
        for row in values:
            row["adapter_identity"] = manifest["adapter_identity"]
        rows.write_text("".join(_compact(row) + "\n" for row in values), encoding="utf-8")
        manifest["files"]["rows.jsonl"] = {
            "sha256": _hash(rows), "bytes": rows.stat().st_size,
        }
    elif damage == "missing_batches":
        manifest["files"].pop("batches.jsonl")
    elif damage in ("executor_source", "policy_source"):
        key = (evaluator.attention_policy.__name__ if damage == "executor_source"
               else _attention_policy()["path"])
        manifest["source_files_sha256"].pop(key)
    else:
        batches = root / "batches.jsonl"
        event = json.loads(batches.read_text())
        if damage == "empty_attention":
            event["attention"] = []
        elif damage == "missing_forward":
            event["forward_calls"] = 2
        elif damage == "failed_batch":
            event["event"] = "batch_failed"
        else:
            from tests.test_p3_deepseek_quality_attention import _serialized_evidence

            event["attention"] = [_serialized_evidence("MATH")]
        batches.write_text(_compact(event) + "\n", encoding="utf-8")
        manifest["files"]["batches.jsonl"] = {
            "sha256": _hash(batches), "bytes": batches.stat().st_size,
        }
    _payload(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="attention|backend"):
        evaluator._collect([root], "bf16_l2l")


@pytest.mark.parametrize("workspace", [None, ":16:8"])
def test_native_admission_refuses_workspace_before_cuda_probe(evaluator, monkeypatch, workspace):
    if workspace is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", workspace)
    monkeypatch.setattr(evaluator.attention_policy, "require_ampere_cuda",
                        lambda device: pytest.fail("CUDA probed before workspace admission"))
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        evaluator._admit_native_execution("cuda")


def test_pairing_requires_complete_once_only_coverage_and_exports_statistics_rows(
    evaluator, tmp_path,
):
    ids = [f"arc/{index}" for index in range(4)]
    baseline = [_shard(evaluator, tmp_path / "b0", endpoint="arc", arm="bf16_l2l",
                       start=0, stop=2, ids=ids),
                _shard(evaluator, tmp_path / "b1", endpoint="arc", arm="bf16_l2l",
                       start=2, stop=4, ids=ids)]
    candidate = [_shard(evaluator, tmp_path / "c", endpoint="arc", arm="oracle_block64_e4m3",
                        start=0, stop=4, ids=ids)]
    paired, secondary = evaluator.pair_shards(baseline, candidate)
    assert secondary == []
    assert paired[0] == {"benchmark": "arc", "length": 1024, "seed": 23,
                         "task_id": "arc/0", "baseline_pass": True, "candidate_pass": True}
    assert len(paired) == 4
    with pytest.raises(ValueError, match="coverage"):
        evaluator.pair_shards(baseline[:1], candidate)
    with pytest.raises(ValueError, match="coverage"):
        evaluator.pair_shards(baseline + baseline[:1], candidate)
    with pytest.raises(ValueError, match="coverage|identity"):
        evaluator.pair_shards(candidate, baseline)


def test_secondary_nll_pairs_preserve_targets_and_candidate_minus_baseline(evaluator, tmp_path):
    ids = ["mbpp/0", "mbpp/1"]
    baseline = [_shard(evaluator, tmp_path / "b", endpoint="mbpp", arm="bf16_l2l",
                       start=0, stop=2, ids=ids)]
    candidate = [_shard(evaluator, tmp_path / "c", endpoint="mbpp", arm="oracle_block64_e4m3",
                        start=0, stop=2, ids=ids)]
    paired, secondary = evaluator.pair_shards(baseline, candidate)
    assert paired == []
    assert len(secondary) == 2
    assert secondary[0]["nll_mean_difference"] == 0.0
    assert secondary[0]["n_targets"] == 2


@pytest.mark.parametrize("trained", [False, True])
def test_unfrozen_trained_or_base_test_evaluation_fails_before_loading(
    evaluator, tmp_path, trained,
):
    output = tmp_path / "failed"
    result = evaluator.run_endpoint(
        data_root=tmp_path / "missing", endpoint="arc", start=0, stop=1,
        source=tmp_path / "missing", cache=tmp_path / "missing", output=output,
        adapter=tmp_path / "trained" if trained else None,
        adapter_sha256="0" * 64 if trained else None,
    )
    assert result["status"] == "FAILED"
    assert result["selected_shard_complete"] is False
    reason = "freeze" if trained else "untrained base"
    assert reason in result["failure"]["message"].lower()
    saved = json.loads((output / "manifest.json").read_text())
    assert saved["status"] == "FAILED"


def test_existing_output_is_preserved(evaluator, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep").write_text("unchanged")
    with pytest.raises(FileExistsError):
        evaluator.run_endpoint(
            data_root=tmp_path, endpoint="arc", start=0, stop=1,
            source=tmp_path, cache=tmp_path, output=output,
        )
    assert (output / "keep").read_text() == "unchanged"


def test_streamed_native_padded_vocabulary_keeps_all_probability_mass(
    evaluator, tmp_path, tokenizer,
):
    import torch

    from benchmarks.harness import b13_moe_fixture as fixture
    from benchmarks.harness import p3_stage2v2_loglik as frozen
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    config = fixture.moe_config(vocab_size=len(tokenizer) + 8, max_position_embeddings=2048)
    weights = fixture.moe_weights_dir(tmp_path, config)
    # Derive a padded head row from the native hidden state of the fixed context.
    # Its logit is positive; every tokenizable row is zero. This tests a padded
    # greedy winner rather than relying on random argmax luck.
    from safetensors.torch import load_file, save_file

    checkpoint = weights / "model.safetensors"
    tensors = load_file(str(checkpoint))
    probe = fixture.resident_base(weights).eval()
    probe_ids = tokenizer.encode("q", add_special_tokens=False)
    with torch.inference_mode():
        hidden = probe.model(torch.tensor([probe_ids]), use_cache=False).last_hidden_state[0, -1]
        padded = hidden.float() / hidden.float().square().sum() * 8
    tensors["lm_head.weight"][:-1].zero_()
    tensors["lm_head.weight"][-1].copy_(padded)
    save_file(tensors, str(checkpoint), metadata={"format": "pt"})
    cache = tmp_path / "cache"
    index = shard_checkpoint(
        str(weights), str(cache), dtype="float32", arch="deepseek_v3",
        expert_layout="per_expert",
    )
    model, runtime = build_streamed_model(
        model_id=str(weights), shard_dir=str(cache), index=index,
        lora_config=fixture.moe_lora_config(), device="cpu", dtype="float32",
        buffers=2, pin=False, quant="none", tier="disk",
    )
    try:
        session = SimpleNamespace(model=model, runtime=runtime)
        context = tokenizer.encode("q", add_special_tokens=False)
        target = tokenizer.encode(" a", add_special_tokens=False)
        resident = fixture.resident_base(weights).eval()
        inputs = torch.tensor([context + target[:-1]])
        with torch.inference_mode():
            logits = resident(inputs, use_cache=False).logits[0, len(context) - 1:].float()
            labels = torch.tensor(target)[:, None]
            expected = logits.log_softmax(-1).gather(1, labels).sum().item()
            cropped = logits[:, :len(tokenizer)].log_softmax(-1).gather(1, labels).sum().item()
            native_greedy = logits.argmax(-1).eq(labels.squeeze(1)).all().item()
            assert logits[0].argmax().item() == len(tokenizer) + 7
            assert native_greedy is False
        observed = evaluator.score_requests(
            session, tokenizer, [("q", " a")], max_tokens_per_batch=64, test_fixture=True,
        )[0]
        assert observed["logprob"] == pytest.approx(expected, abs=2e-4)
        assert abs(observed["logprob"] - cropped) > 0.01
        assert observed["greedy"] == native_greedy
        assert frozen.score_task({"endpoint": "lambada", "metric": "acc"}, [observed]) == {
            "pred": 0, "pred_norm": 0, "pass": bool(native_greedy),
        }
    finally:
        runtime.close()


def test_pairing_rejects_changed_rows_reordered_ids_and_adapter_swap(evaluator, tmp_path):
    ids = ["arc/0", "arc/1"]
    baseline = _shard(evaluator, tmp_path / "b", endpoint="arc", arm="bf16_l2l",
                      start=0, stop=2, ids=ids)
    candidate = _shard(evaluator, tmp_path / "c", endpoint="arc", arm="oracle_block64_e4m3",
                       start=0, stop=2, ids=ids)
    path = candidate / "rows.jsonl"
    original = path.read_text()
    path.write_text(original + "{}\n")
    with pytest.raises(ValueError, match="SHA256"):
        evaluator.pair_shards([baseline], [candidate])
    path.write_text(original)
    manifest = json.loads((candidate / "manifest.json").read_text())
    manifest.pop("manifest_payload_sha256")
    original_manifest = copy.deepcopy(manifest)
    path.write_text("\n".join(reversed(original.splitlines())) + "\n")
    manifest["files"]["rows.jsonl"] = {"sha256": _hash(path), "bytes": path.stat().st_size}
    _payload(candidate / "manifest.json", manifest)
    with pytest.raises(ValueError, match="coverage"):
        evaluator.pair_shards([baseline], [candidate])
    manifest = original_manifest
    manifest["adapter_identity"]["initial_adapter_sha256"] = "0" * 64
    changed_rows = [json.loads(line) for line in original.splitlines()]
    for row in changed_rows:
        row["adapter_identity"] = manifest["adapter_identity"]
    path.write_text("".join(_compact(row) + "\n" for row in changed_rows))
    manifest["files"]["rows.jsonl"] = {"sha256": _hash(path), "bytes": path.stat().st_size}
    _payload(candidate / "manifest.json", manifest)
    with pytest.raises(ValueError, match="identity"):
        evaluator.pair_shards([baseline], [candidate])


def test_protocol_hash_freezes_metric_and_training_scope(evaluator, tmp_path):
    reference = {"loglik": {"prepend_bos": False}, "scoring": {"truncation": False}}
    value = {
        "schema": "p3-deepseek-quality-protocol-v1", **reference,
        "status": "FROZEN",
        "include_base_model_evaluation": False,
        "attention_policy": _attention_policy(),
        "data": {"manifest_sha256": evaluator.DATA_MANIFEST_SHA256},
        "training": {"seed": 23, "lengths": [1024], "steps": 32},
        "initial_adapter_sha256": "c" * 64, "pack_sha256": "d" * 64,
    }
    path = tmp_path / "protocol.json"
    _json(path, value)
    assert evaluator.admit_protocol(path, _hash(path), reference)["training"]["steps"] == 32
    for status in (None, "DRAFT"):
        if status is None:
            value.pop("status")
        else:
            value["status"] = status
        _json(path, value)
        with pytest.raises(ValueError, match="freeze"):
            evaluator.admit_protocol(path, _hash(path), reference)
    value["status"] = "FROZEN"
    approved = value["attention_policy"]
    changed_backend = copy.deepcopy(approved)
    changed_backend["contract"]["main_backend"] = "MATH"
    for invalid in (None, {**approved, "sha256": "0" * 64}, changed_backend):
        value["attention_policy"] = invalid
        _json(path, value)
        with pytest.raises(ValueError, match="attention policy"):
            evaluator.admit_protocol(path, _hash(path), reference)
    value["attention_policy"] = approved
    _json(path, value)
    with pytest.raises(ValueError, match="SHA256"):
        evaluator.admit_protocol(path, "0" * 64, reference)
    value["include_base_model_evaluation"] = True
    _json(path, value)
    with pytest.raises(ValueError, match="forbid untrained base"):
        evaluator.admit_protocol(path, _hash(path), reference)
    value["include_base_model_evaluation"] = False
    value["scoring"] = {"truncation": True}
    _json(path, value)
    with pytest.raises(ValueError, match="scoring rules"):
        evaluator.admit_protocol(path, _hash(path), reference)


def test_cli_pair_requires_all_frozen_endpoints_and_writes_statistics_rows(
    evaluator, tmp_path, dataset, monkeypatch,
):
    reference = {"loglik": {"hellaswag_n": 4}, "scoring": {"truncation": False},
                 "data": {"full_retained_evaluation_tasks": {"hellaswag": 4}}}
    monkeypatch.setattr(evaluator, "_scoring_protocol", lambda _path: (
        reference, {"sha256": "b" * 64},
    ))
    study = {
        "schema": "p3-deepseek-quality-protocol-v1", "loglik": reference["loglik"],
        "status": "FROZEN",
        "include_base_model_evaluation": False,
        "attention_policy": _attention_policy(),
        "scoring": reference["scoring"],
        "data": {"manifest_sha256": _hash(dataset / "manifest.json")},
        "training": {"seed": 23, "lengths": [1024], "steps": 64},
        "initial_adapter_sha256": "c" * 64, "pack_sha256": "d" * 64,
    }
    monkeypatch.setattr(evaluator, "DATA_MANIFEST_SHA256", _hash(dataset / "manifest.json"))
    # The function default is a production pin; pass the fixture digest explicitly.
    admit_tasks = evaluator.admit_tasks
    monkeypatch.setattr(evaluator, "admit_tasks", lambda *args, **kwargs: admit_tasks(
        *args, manifest_sha256=_hash(dataset / "manifest.json"), **kwargs,
    ))
    protocol = tmp_path / "protocol.json"
    _json(protocol, study)
    digest = _hash(protocol)
    arms = {"bf16_l2l": [], "oracle_block64_e4m3": []}
    for endpoint in ("hellaswag", "arc", "boolq", "lambada", "gsm8k", "mbpp"):
        tasks, binding = admit_tasks(dataset, endpoint, 0, 4,
                                     manifest_sha256=_hash(dataset / "manifest.json"))
        ids = [row["task_id"] for row in tasks]
        for arm, paths in arms.items():
            root = _shard(evaluator, tmp_path / f"{endpoint}-{arm}", endpoint=endpoint,
                          arm=arm, start=0, stop=4, ids=ids, protocol=digest)
            manifest = json.loads((root / "manifest.json").read_text())
            manifest.pop("manifest_payload_sha256")
            manifest.update(binding=binding, loglik=reference["loglik"],
                            scoring=reference["scoring"])
            rows = [json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines()]
            for task, scored in zip(tasks, rows, strict=True):
                scored["task_sha256"] = hashlib.sha256(_compact(task).encode()).hexdigest()
                scored["task_scoring"] = {name: task[name] for name in (
                    "endpoint", "metric", "choices", "gold", "target") if name in task}
                if endpoint in ("hellaswag", "arc", "boolq"):
                    scored["pass"] = False  # gold1, fixed native-like results choose0.
                elif endpoint == "lambada":
                    scored["pass"] = scored["results"][0]["greedy"]
            path = root / "rows.jsonl"
            path.write_text("".join(_compact(row) + "\n" for row in rows))
            manifest["files"]["rows.jsonl"] = {"sha256": _hash(path), "bytes": path.stat().st_size}
            _payload(root / "manifest.json", manifest)
            paths.append(root)
    output = tmp_path / "paired"
    arguments = ["pair", "--baseline", *(str(path) for path in arms["bf16_l2l"]),
                 "--candidate", *(str(path) for path in arms["oracle_block64_e4m3"]),
                 "--data-root", str(dataset), "--protocol", str(protocol),
                 "--protocol-sha256", digest, "--output", str(output)]
    assert evaluator.main(arguments) == 0
    rows = [json.loads(line) for line in (output / "pairs.jsonl").read_text().splitlines()]
    assert len(rows) == 16
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["files"]["pairs.jsonl"]["sha256"] == _hash(output / "pairs.jsonl")
    assert manifest["status"] == "COMPLETE" and manifest["secondary_rows"] == 8
    assert set(manifest["expected_tasks"]) == {"hellaswag", "arc", "boolq", "lambada"}
    assert set(manifest["expected_secondary_tasks"]) == {"gsm8k", "mbpp"}
    result = evaluator.pair(
        baseline=arms["bf16_l2l"][:-1], candidate=arms["oracle_block64_e4m3"],
        output=tmp_path / "incomplete", data_root=dataset, protocol=protocol,
        protocol_sha256=digest,
    )
    assert result["status"] == "FAILED"
    assert "coverage" in result["failure"]["message"]


@pytest.mark.gpu
def test_endpoint_cli_executes_native_fixture_and_preserves_complete_evidence(
    evaluator, tokenizer, dataset, tmp_path, monkeypatch, request,
):
    from benchmarks.harness import b13_moe_fixture as fixture
    from benchmarks.harness import p3_deepseek_quality_model as models

    config = fixture.moe_config(vocab_size=len(tokenizer), max_position_embeddings=2048)
    weights = fixture.moe_weights_dir(tmp_path, config)
    cache = tmp_path / "native-cache"
    twins = models.open_session(weights, cache, device="cuda", prepare=True)
    request.addfinalizer(twins.close)

    source_metadata = copy.deepcopy(twins.metadata)
    monkeypatch.setattr(models, "require_production_source", lambda _value: None)
    monkeypatch.setattr(models, "open_session", lambda *_args, **_kwargs: twins)
    # Fixture seams replace only source qualification/lookup. Scoring, tokenization,
    # runtime checks, metrics, data admission and evidence publication stay real.
    token_dir = tmp_path / "tokens"
    token_dir.mkdir()
    token_file = token_dir / "tokenizer.json"
    token_file.write_text("fixture identity")
    monkeypatch.setattr(evaluator, "_load_tokenizer", lambda _source: (
        tokenizer, {"files": {"tokenizer.json": {"sha256": _hash(token_file)}}},
    ))
    original_tasks = evaluator.admit_tasks
    monkeypatch.setattr(evaluator, "admit_tasks", lambda root, endpoint, start, stop, **kwargs: (
        original_tasks(root, endpoint, start, stop, manifest_sha256=_hash(root / "manifest.json"),
                       **kwargs)
    ))
    reference = {
        "loglik": {"max_tokens_per_batch": 128, "deadline_seconds": 2700,
                   "logits_memory_limit_bytes": 536870912, "reduction_chunk_tokens": 128},
        "scoring": {"truncation": False},
    }
    monkeypatch.setattr(evaluator, "_scoring_protocol", lambda _path: (
        reference, {"sha256": "b" * 64},
    ))
    protocol = tmp_path / "frozen-protocol.json"
    _json(protocol, {
        "schema": "p3-deepseek-quality-protocol-v1", **reference,
        "status": "FROZEN",
        "include_base_model_evaluation": False,
        "attention_policy": _attention_policy(),
        "data": {"manifest_sha256": evaluator.DATA_MANIFEST_SHA256},
        "training": {"seed": 23, "lengths": [1024], "steps": 64},
        "initial_adapter_sha256": "c" * 64, "pack_sha256": "d" * 64,
    })
    protocol_sha256 = _hash(protocol)
    adapter, adapter_sha256 = _saved_adapter(
        twins, tmp_path / "trained-fixture-adapter", protocol=protocol_sha256,
    )
    output = tmp_path / "endpoint"
    assert evaluator.main([
        "run-endpoint", "--data-root", str(dataset), "--endpoint", "boolq",
        "--start", "0", "--stop", "4", "--source", str(token_dir),
        "--cache", str(cache), "--device", "cuda", "--output", str(output),
        "--protocol", str(protocol), "--protocol-sha256", protocol_sha256,
        "--adapter", str(adapter), "--adapter-sha256", adapter_sha256,
    ]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    rows = [json.loads(line) for line in (output / "rows.jsonl").read_text().splitlines()]
    assert manifest["status"] == "COMPLETE" and manifest["row_count"] == 4
    assert manifest["model"] == source_metadata
    assert all(row["adapter_identity"]["mode"] == "final" for row in rows)
    assert all(type(row["pass"]) is bool and len(row["results"]) == 2 for row in rows)
    assert manifest["files"]["rows.jsonl"]["sha256"] == _hash(output / "rows.jsonl")
    assert twins.closed
    batches = [json.loads(line) for line in (output / "batches.jsonl").read_text().splitlines()]
    for batch in batches:
        assert batch["event"] == "batch_complete"
        assert len(batch["attention"]) == batch["forward_calls"]
        for report in batch["attention"]:
            evaluator.attention_policy.require_execution_evidence(
                report, backend="EFFICIENT_ATTENTION", dtype="bfloat16", backward=False,
            )
    evaluator._collect([output], "bf16_l2l")


@pytest.mark.gpu
def test_ledger_nf4_session_executes_frozen_reducer_on_native_tiny_v3(
    evaluator, tmp_path, tokenizer,
):
    import torch

    from benchmarks.harness import b13_moe_fixture as fixture
    from benchmarks.harness.p3_deepseek_quality_model import open_session

    config = fixture.moe_config(vocab_size=len(tokenizer), max_position_embeddings=2048)
    weights = fixture.moe_weights_dir(tmp_path, config)
    with open_session(weights, tmp_path / "cache", prepare=True) as session:
        identity = evaluator.configure_adapter(session, None, None)
        events = []
        requests = [("Q" * length, " a") for length in range(20, 38)]
        observed = evaluator._score(
            session, tokenizer,
            lambda: evaluator.frozen._request_tokens(tokenizer, requests),
            max_tokens_per_batch=42, observer=events.append,
        )
        from benchmarks.harness.p3_deepseek_quality_attention import (
            MAIN_BACKEND,
            require_execution_evidence,
        )

        assert identity["zero_lora_B"] is True
        assert len(observed) == len(requests)
        # More shapes than one observer permits must remain valid across batches.
        assert len(events) > 16
        for event in events:
            assert event["attention"]
            for report in event["attention"]:
                require_execution_evidence(
                    report, backend=MAIN_BACKEND, dtype="bfloat16", backward=False,
                )
        assert session.runtime.stats()["expert_fetches"] > 0
        assert session.runtime.expert_pool.dtype == "bfloat16"
        assert all(parameter.dtype == torch.float32 for parameter in session.model.parameters()
                   if parameter.requires_grad)


def test_streamed_choice_char_normalization_and_multitoken_lambada(evaluator, twins, tokenizer):
    from benchmarks.harness import p3_stage2v2_loglik as frozen

    row = {"endpoint": "arc", "metric": "acc_norm", "choices": ["x", "very long"], "gold": 1}
    likelihoods = [{"logprob": -1.1, "n_tokens": 1, "greedy": False},
                   {"logprob": -2.0, "n_tokens": 2, "greedy": False}]
    assert frozen.score_task(row, likelihoods) == {"pred": 0, "pred_norm": 1, "pass": True}
    likelihoods = evaluator.score_requests(
        twins, tokenizer, [("a ", "b"), ("Q:", " multiple tokens")],
        max_tokens_per_batch=128, test_fixture=True,
    )
    assert likelihoods[0]["n_tokens"] == 2  # trailing context space moved into target.
    assert likelihoods[1]["n_tokens"] == 16
    assert frozen.score_task({"endpoint": "lambada", "metric": "acc"}, [
        {"logprob": -1.0, "n_tokens": 3, "greedy": False},
    ])["pass"] is False


def test_frozen_measured_schedule_changes_only_approved_fields(evaluator, tmp_path):
    reference = {
        "loglik": {"hellaswag_n": 10042, "max_tokens_per_batch": 8192,
                   "deadline_seconds": 2700, "logits_memory_limit_bytes": 536870912,
                   "prepend_bos": False},
        "scoring": {"truncation": False},
        "data": {"full_retained_evaluation_tasks": {"hellaswag": 10042}},
    }
    value = {
        "schema": "p3-deepseek-quality-protocol-v1", "loglik": dict(reference["loglik"]),
        "status": "FROZEN",
        "include_base_model_evaluation": False,
        "attention_policy": _attention_policy(),
        "scoring": reference["scoring"],
        "data": {"manifest_sha256": evaluator.DATA_MANIFEST_SHA256},
        "training": {"seed": 23, "lengths": [1024], "steps": 32},
        "initial_adapter_sha256": "c" * 64, "pack_sha256": "d" * 64,
    }
    value["loglik"].update(hellaswag_n=3000, max_tokens_per_batch=4096, deadline_seconds=1800)
    path = tmp_path / "protocol.json"
    _json(path, value)
    assert evaluator.admit_protocol(path, _hash(path), reference)["loglik"]["hellaswag_n"] == 3000
    value["loglik"]["logits_memory_limit_bytes"] = 1073741824
    _json(path, value)
    with pytest.raises(ValueError, match="scoring rules"):
        evaluator.admit_protocol(path, _hash(path), reference)
    value["loglik"]["logits_memory_limit_bytes"] = 536870912
    value["loglik"]["hellaswag_n"] = 10043
    _json(path, value)
    with pytest.raises(ValueError, match="retained manifest count"):
        evaluator.admit_protocol(path, _hash(path), reference)


@pytest.mark.parametrize("start,stop", [(True, 2), (0, False), (-1, 2), (2, 2), (0, 5)])
def test_shard_bounds_never_drop_or_expand_tasks(evaluator, dataset, start, stop):
    with pytest.raises(ValueError, match="shard interval"):
        evaluator.admit_tasks(dataset, "arc", start, stop,
                              manifest_sha256=_hash(dataset / "manifest.json"))


def test_frozen_hellaswag_subsample_uses_hash_rank_not_first_source_rows(evaluator, dataset):
    path = dataset / "hellaswag/tasks.jsonl"
    source = [json.loads(line) for line in path.read_text().splitlines()]
    expected = sorted(source, key=lambda row: (
        hashlib.sha256(("p3-stage2-v2-hellaswag\0" + row["task_id"]).encode()).hexdigest(),
        row["task_id"],
    ))[:2]
    selected, binding = evaluator.admit_tasks(
        dataset, "hellaswag", 0, 2, manifest_sha256=_hash(dataset / "manifest.json"),
        hellaswag_n=2,
    )
    assert [row["task_id"] for row in selected] == [row["task_id"] for row in expected]
    assert binding["complete_endpoint_row_count"] == 2


@pytest.mark.parametrize("endpoint", ["boolq_dev", "arc", "gsm8k"])
@pytest.mark.parametrize("frozen_protocol", [False, True])
def test_untrained_base_quality_refused_before_data_or_model_loading(
    evaluator, tmp_path, endpoint, frozen_protocol, monkeypatch,
):
    def forbid_protocol_access(_path):
        raise AssertionError("resource admission must not run for a forbidden base evaluation")

    monkeypatch.setattr(evaluator, "_scoring_protocol", forbid_protocol_access)
    output = tmp_path / "refused"
    result = evaluator.run_endpoint(
        data_root=tmp_path / "must-not-open-data", endpoint=endpoint, start=0, stop=1,
        source=tmp_path / "must-not-open-model", cache=tmp_path / "must-not-open-cache",
        output=output, protocol=tmp_path / "must-not-open-protocol" if frozen_protocol else None,
        protocol_sha256="a" * 64 if frozen_protocol else None,
    )
    assert result["status"] == "FAILED"
    assert result["failure"]["type"] == "ValueError"
    assert result["include_base_model_evaluation"] is False
    assert not (output / "rows.jsonl").exists()
    assert not (output / "batches.jsonl").exists()
