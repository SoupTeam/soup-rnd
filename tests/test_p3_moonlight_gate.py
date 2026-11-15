"""Serialized gate admission contracts, not native GPU or full-model execution evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import pytest

from benchmarks.harness import p3_deepseek_quality_gate as gate
from benchmarks.harness import p3_deepseek_quality_smoke as smoke
from tests.test_p3_deepseek_quality_gate import _attention_receipt, _window

MOONLIGHT = "moonlight-16b-a3b"
DEEPSEEK = "deepseek-v3"
PASSES = ("ga", "ga_repeat", "bf16_l2l", "ga_floor", "fp8_oracle", "int4_oracle")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _commit(root: Path, result: dict[str, Any]) -> Path:
    """Rehash mutations so their refusal must come from evidence, not an obsolete wrapper."""
    for name, row in result["passes"].items():
        artifact = root / name / "gradients.pt"
        if artifact.is_file():
            row["gradient_artifact"] = {
                "path": "gradients.pt", "bytes": artifact.stat().st_size,
                "sha256": _digest(artifact),
            }
        (root / name / "pass.json").write_text(json.dumps(row), encoding="utf-8")
    (root / "result.json").write_text(json.dumps(result), encoding="utf-8")
    files = {
        path.relative_to(root).as_posix(): {"bytes": path.stat().st_size, "sha256": _digest(path)}
        for path in root.rglob("*") if path.is_file() and path.name != "manifest.json"
    }
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": gate.SCHEMA, "profile_id": result["profile_id"],
        "status": result["status"], "verdict": result["verdict"],
        "qualification_only": result["qualification_only"], "files": files,
    }), encoding="utf-8")
    return manifest


def _errors(relative: float, maximum: float, reference: float) -> dict[str, Any]:
    return {
        "rows": 512, "rms_row_rel": relative, "worst_row_rel": relative,
        "rel_l2": relative, "max_abs": maximum, "max_abs_over_absmax": relative,
        "err_sq": reference * relative**2, "ref_sq": reference,
        "row_rel_sq_sum": 512 * relative**2,
    }


def _gates(*, coarse: bool, layers: int, tensors: int) -> dict[str, Any]:
    return {
        "E1": {"status": "FAIL" if coarse else "PASS",
               "layers_over": list(range(layers)) if coarse else [],
               "max_rms_row_rel": 0.5 if coarse else 0.0,
               "max_worst_row_rel": 0.5 if coarse else 0.0,
               "rms_bound": 0.05, "worst_row_bound": 0.25},
        "E2": {"status": "FAIL" if coarse else "PASS",
               "layers_over": list(range(layers)) if coarse else [],
               "max_branch_rms_row_rel": 1.5 if coarse else 0.0, "bound": 0.1,
               "semantics": "one executed Oracle recompute residual branch from restored input"},
        "E3": {"status": "FAIL" if coarse else "PASS",
               "global_rel_l2": 1.0 if coarse else 0.0, "min_cosine": 1.0,
               "tensors": tensors, "global_rel_l2_bound": 0.05, "min_cosine_bound": 0.99},
    }


def _receipt(root: Path, *, profile_id: str = MOONLIGHT, verdict: str = "FAIL",
             production: bool = False) -> tuple[Path, dict[str, Any]]:
    """Hand-derived constant tensors plus pinned metadata; no claimed CUDA execution.

    Production-shaped variants exercise the public loader without replacing any validator.
    Expanded tensor storage keeps this serialized contract fixture small on disk.
    """
    import torch

    from benchmarks.harness import p3_deepseek_quality_model as model_api
    from benchmarks.harness.p3_deepseek_quality_profiles import get_profile

    profile = get_profile(profile_id)
    policy = gate.policy_identity(profile_id=profile_id)
    layer_count = int(profile.shape["num_hidden_layers"]) if production else 1
    hidden = int(profile.shape["hidden_size"]) if production else 2
    dense = int(profile.shape["first_k_dense_replace"]) if production else 0
    names = sorted(f"model.layers.{layer}.self_attn.{target}.lora_{factor}.default.weight"
                   for layer in range(layer_count) for target in profile.lora_targets
                   for factor in ("A", "B"))
    passes = {}
    for arm in PASSES:
        coarse = arm == "fp8_oracle" and verdict == "FAIL" or (
            arm == "int4_oracle" and verdict != "NO_VERDICT")
        factor = 1.125 if arm == "ga_floor" else 2.0 if coarse else 1.0
        folder = root / arm
        folder.mkdir(parents=True)
        torch.save({name: torch.tensor([[factor, 2 * factor], [factor, 2 * factor]])
                    for name in names}, folder / "gradients.pt")
        backend = "MATH" if arm == "ga_floor" else "EFFICIENT_ATTENTION"
        attention = _attention_receipt(backend)
        attention["profile_id"] = profile_id
        attention["executed_sdpa_calls"] = layer_count * 8
        for call in attention["unique_calls"] + attention["samples"]:
            call["query_shape"] = call["key_shape"] = [1, profile.attention_heads, 64, 192]
            call["value_shape"] = [1, profile.attention_heads, 64, 128]
        row = {
            "name": arm, "profile_id": profile_id, "status": "COMPLETE",
            "optimizer_applied": False, "adapter_unchanged": True,
            "adapter_before_sha256": "a" * 64, "adapter_after_sha256": "a" * 64,
            "initial_adapter_sha256": "b" * 64, "valid_target_count": 504,
            "host_budget_after_close_bytes": 0, "cleanup_errors": [],
            "diagnostic_errors": [], "losses": [1.0] * 8,
            "runtime_load_deltas": {key: 1 for key in (
                "layer_loads", "large_loads", "expert_fetches", "expert_bytes_read")},
            "gradient_checks": {name: {"present": True, "finite": True, "nonzero": True,
                                        "shape": [2, 2]} for name in names},
            "routing_guard": "production_L2LStep_enabled" if arm not in (
                "ga", "ga_repeat", "ga_floor") else "ordinary_streamed_GA",
            "backend_requested": backend, "attention": attention,
            "layer_forward_counts": [8] * layer_count,
        }
        if arm in ("bf16_l2l", "fp8_oracle", "int4_oracle"):
            row["layers"] = []
            for layer in range(layer_count):
                row["layers"].append({
                    "layer": layer,
                    "pairs": [{"microbatch": mb, "rows": 64, "finite": True,
                               "forward_equal": True} for mb in range(8)],
                    "restore": _errors(0.5 if coarse else 0.0,
                                       0.5 if coarse else 0.0, 512.0 * hidden),
                    "branch": _errors(1.5 if coarse else 0.0,
                                      1.5 if coarse else 0.0, 512.0 * hidden),
                    "output": _errors(1.0 if coarse else 0.0,
                                      2.0 if coarse else 0.0, 2048.0 * hidden),
                })
                phases = ("forward", "recompute") if arm == "bf16_l2l" else ("recompute",)
                for phase in phases:
                    (folder / phase).mkdir(exist_ok=True)
                    for mb in range(8):
                        torch.save({
                            "input": torch.tensor(1.5 if coarse else 1.0,
                                                  dtype=torch.bfloat16).expand(1, 64, hidden),
                            "output": torch.tensor(4.0 if coarse else 2.0,
                                                   dtype=torch.bfloat16).expand(1, 64, hidden),
                        }, folder / phase / f"layer-{layer:03d}-mb-{mb:03d}.pt")
        if "oracle" in arm:
            error = 1.0 if coarse else 0.0
            row["forward_losses_equal"] = True
            row["gradient_agreement"] = {
                "tensors": len(names), "global_rel_l2": error, "min_cosine": 1.0,
                "max_rel_l2": error,
                "worst": [{"name": name, "cosine": 1.0, "rel_l2": error}
                          for name in names[:5]],
                "per_tensor": {name: {"cosine": 1.0, "rel_l2": error,
                                       "reference_finite": True, "candidate_finite": True,
                                       "shape_equal": True} for name in names},
            }
            pairs = (layer_count - dense) * 8
            row["oracle"] = {
                "complete": True, "expected_pairs": pairs,
                "counts": {key: pairs for key in ("native_forward", "oracle_recompute",
                                                  "oracle_vjp")},
                "forced_routing": True, "natural_routing_measured": False,
                "records_remaining": 0, "pending_replays": 0, "failure": None,
                "cleanup_errors": [],
            }
            row["oracle_after_close"] = {
                "closed": True, "host_reserved_bytes": 0, "records_remaining": 0,
                "pending_replays": 0, "owned_host_storage_bytes": 0,
                "owned_device_storage_bytes": 0,
                "cleanup": {"original_forwards_restored": True, "hooks_removed": True},
                "cleanup_errors": [],
            }
        if arm == "fp8_oracle":
            row["store_after_close"] = {
                "cleanup": {"closed": True}, "owned_host_storage_bytes": 0,
                "owned_device_storage_bytes": 0, "charged_host_bytes": 0,
            }
        passes[arm] = row
    comparison = {"grads_equal": True, "losses_equal": True, "rates_equal": True,
                  "steps_equal": True, "steps": [1, 1], "max_abs": 0.0, "worst": [],
                  "equal": True}
    _, _, identity = gate.gate_prefix(_window())
    inputs = [[[index % 64 for index in range(64)]] for _ in range(8)]
    labels = [[[-100, *batch[0][1:]]] for batch in inputs]
    identity.update(valid_target_count=504, per_microbatch_targets=[63] * 8,
                    input_sha256=hashlib.sha256(struct.pack(
                        "<512q", *(value for batch in inputs for value in batch[0]))).hexdigest(),
                    labels_sha256=hashlib.sha256(struct.pack(
                        "<512q", *(value for batch in labels for value in batch[0]))).hexdigest())
    (root / "gate-input.json").write_text(json.dumps({
        "profile_id": profile_id, "identity": identity, "input_ids": inputs, "labels": labels,
    }), encoding="utf-8")
    source = {"profile_id": profile_id,
              "shape": {"first_k_dense_replace": dense, "hidden_size": hidden}}
    cache = {"profile_id": profile_id, "n_layers": layer_count}
    control = _gates(coarse=verdict != "NO_VERDICT", layers=layer_count, tensors=len(names))
    result = {
        "schema": gate.SCHEMA, "profile_id": profile_id, "status": "COMPLETE",
        "verdict": verdict, "qualification_only": not production,
        "dtype": "bfloat16", "device": "cuda", "seed": 23, "optimizer_updates": 0,
        "session_closed": True, "nonfinite_evidence_fields": [],
        "attention_policy": policy, "cublas_environment_restored": True,
        "determinism": {"restored": True, "execution": {
            "algorithms_enabled": True, "warn_only": False, "allow_tf32": False,
            "cudnn_allow_tf32": False, "cudnn_benchmark": False, "cudnn_deterministic": True}},
        "environment": {"torch": "serialized-test-stack", "cuda": "12.8",
                        "packages": {name: "serialized-test-stack" for name in (
                            "transformers", "peft", "bitsandbytes", "accelerate", "safetensors")},
                        "cublas_workspace_config": ":4096:8",
                        "gpu": {key: value for key, value in passes["ga"]["attention"][
                            "hardware"].items() if key in (
                                "name", "capability", "native_bf16", "bf16_supported")}},
        "initial_adapter_sha256": "b" * 64, "input": identity, "trainable_names": names,
        "layers": list(range(layer_count)), "passes": passes,
        "model": {"profile_id": profile_id, "source": source, "cache": cache},
        "floor": {"observed": True, "backend": "MATH",
                  "main_backend": "EFFICIENT_ATTENTION", "factor": 1.5},
        "a3": {"status": "PASS", "independent_floor_observed": True,
               "optimizer_updates": 0, "comparison_observations": 1,
               "rule": {"a_vs_a": comparison, "l2l_vs_ga": comparison, "pass": True,
                        "note": "", "status": "within_tolerance", "tolerance": {
                            "mode": "bf16", "factor": 1.5, "limit_fp32": 1e-4,
                            "grad_rel": [0.0], "loss_rel": [0.0], "floor_grad": [0.125],
                            "floor_loss": [0.0], "judged": True, "failing": [], "pass": True,
                            "floor_problem": None, "grad_limit": [0.1875], "loss_limit": [0.0]}}},
        "candidate_gates": _gates(coarse=verdict == "FAIL", layers=layer_count,
                                  tensors=len(names)),
        "control": {"gates": control, "discrimination": {
            "status": "NO_VERDICT" if verdict == "NO_VERDICT" else "PASS",
            "not_failed": ["E1", "E2", "E3"] if verdict == "NO_VERDICT" else [],
            "detail": "Frozen G4 V5 requires the int4 control to fail E1, E2 and E3; "
            "a nondiscriminating fixture never gets a fabricated control failure"}},
    }
    if production:
        inventory = model_api.source_inventory(profile_id=profile_id)
        declared = {name: {**item, "sha256": item["hub_sha256"] or "c" * 64,
                           "hub_commit": profile.revision,
                           "sha256_origin": "hub_lfs_content_identity_not_locally_hashed"
                           if name in profile.weight_files else "locally_measured_metadata"}
                    for name, item in inventory.items()}
        source.update(
            shape=dict(profile.shape), model_id=profile.model_id, revision=profile.revision,
            production_qualified=True, files={name: {"bytes": inventory[name]["bytes"],
                                                    "mtime_ns": 1}
                                              for name in profile.weight_files},
            weight_files=len(profile.weight_files), weight_bytes=profile.weight_bytes,
            source_verification="hub_commit_etag_and_size_plus_cache_source_fingerprint",
            receipt_sha256="d" * 64, expected_source_files=declared,
            inventory_sha256=profile.source_inventory_sha256,
            config_sha256=policy["contract"]["model"]["config_sha256"],
            checkpoint_index_sha256="e" * 64, cache_source_fingerprint="f" * 64,
            weight_content_sha256_measured_this_session=False)
        if profile_id == MOONLIGHT:
            source["aux_admission"] = model_api.native_aux_admission(profile_id=profile_id)
            source["runtime_config_normalization"] = [
                {"field": field, "source_value": value, "runtime": "absent",
                 "reason": "no numerical consumer in pinned native deepseek_v3; "
                 "generic config validation read only"}
                for field, value in (("aux_loss_alpha", 0.001), ("seq_aux", True))
            ]
        cache.update(dtype="bfloat16", quant="nf4", double_quant=False,
                     expert_layout="per_expert", tier="disk", index_sha256="1" * 64,
                     source_fingerprint="f" * 64, source_parameters=profile.parameters)
        result["model"]["precision"] = {
            "compute_dtype": "bfloat16", "trainable_dtype": "float32", "autocast": False,
            "attention_implementation": "sdpa",
            "lora": {"rank": 2, "alpha": 4, "dropout": 0.0,
                     "targets": list(profile.lora_targets)},
        }
        pin = {"repo_id": profile.model_id, "revision": profile.revision, "test_only": False,
               "files": {name: dict(spec) for name, spec in profile.tokenizer.files.items()}}
        tokenizer = {
            "profile_id": profile_id, "repo_id": profile.model_id, "revision": profile.revision,
            "test_only": False, "pin": pin, "is_fast": True, "add_special_tokens": False,
            "prepend_bos": False, "weights_loaded": False, "append_native_eos_per_document": True,
            "vocabulary_size": profile.tokenizer.vocabulary_size,
            "eos_id": profile.tokenizer.eos_id,
            "bos_id": profile.tokenizer.bos_id, "pad_id": profile.tokenizer.pad_id,
            "split_special_tokens": True,
            "files": {name: {"path": name, **spec} for name, spec in pin["files"].items()},
        }
        result["pack"] = {
            "manifest_sha256": policy["contract"]["training_pack_manifest_sha256"],
            "manifest": {"profile_id": profile_id, "test_only": False,
                         "training_source_row_count": 7801, "tokenizer": tokenizer,
                         "files": {f"tokenizer/{name}": {"path": f"tokenizer/{name}", **spec}
                                   for name, spec in pin["files"].items()}},
        }
        result["source_files_sha256"] = gate._source_hashes(profile_id=profile_id)
    return _commit(root, result), result


def _verify(manifest: Path, *, profile_id: str = MOONLIGHT) -> dict[str, Any]:
    wrapper = json.loads(manifest.read_text(encoding="utf-8"))
    result = json.loads((manifest.parent / "result.json").read_text(encoding="utf-8"))
    return gate._verify_numerical_evidence(manifest.parent, wrapper["files"], result,
                                           profile_id=profile_id)


@pytest.mark.usefixtures("moonlight_aux_admission")
@pytest.mark.parametrize("verdict", ["FAIL", "NO_VERDICT"])
def test_moonlight_public_loader_preserves_complete_numerical_diagnostic(
    tmp_path: Path, verdict: str,
) -> None:
    manifest, _ = _receipt(tmp_path, production=True, verdict=verdict)
    result = gate.load_gate(manifest, _digest(manifest), profile_id=MOONLIGHT)
    assert result["verdict"] == verdict
    assert result["profile_id"] == MOONLIGHT
    assert set(result["passes"]) == set(PASSES)
    assert all(row["status"] == "COMPLETE" for row in result["passes"].values())
    if verdict == "FAIL":
        assert result["candidate_gates"]["E3"]["global_rel_l2"] == 1.0
    else:
        assert result["control"]["discrimination"]["not_failed"] == ["E1", "E2", "E3"]


def test_finite_zero_tensor_remains_diagnostic_in_all_six_passes(tmp_path):
    import torch

    manifest, result = _receipt(tmp_path, verdict="FAIL")
    zero_name = result["trainable_names"][0]
    gradients = {}
    for name in PASSES:
        path = tmp_path / name / "gradients.pt"
        gradients[name] = torch.load(path, map_location="cpu", weights_only=True)
        gradients[name][zero_name].zero_()
        torch.save(gradients[name], path)
        result["passes"][name]["gradient_checks"][zero_name]["nonzero"] = False
    for name in ("fp8_oracle", "int4_oracle"):
        result["passes"][name]["gradient_agreement"] = gate.gradient_agreement(
            gradients["bf16_l2l"], gradients[name])
    _commit(tmp_path, result)
    loaded = _verify(manifest)
    assert loaded["verdict"] == "FAIL"
    assert set(loaded["passes"]) == set(PASSES)
    assert all(row["status"] == "COMPLETE"
               and row["gradient_checks"][zero_name]["nonzero"] is False
               for row in loaded["passes"].values())


def test_deepseek_numeric_failure_still_cannot_admit_training(tmp_path: Path) -> None:
    manifest, _ = _receipt(tmp_path, profile_id=DEEPSEEK)
    with pytest.raises(ValueError, match="PASS|admit|verdict"):
        _verify(manifest, profile_id=DEEPSEEK)
    with pytest.raises(ValueError, match="PASS|production"):
        gate.load_gate(manifest, _digest(manifest))


@pytest.mark.parametrize("requested,recorded", [(DEEPSEEK, MOONLIGHT), (MOONLIGHT, DEEPSEEK)])
def test_public_loader_refuses_cross_profile_before_numerical_admission(
    tmp_path: Path, requested: str, recorded: str,
) -> None:
    manifest, _ = _receipt(tmp_path, profile_id=recorded)
    with pytest.raises(ValueError, match="profile"):
        gate.load_gate(manifest, _digest(manifest), profile_id=requested)


@pytest.mark.parametrize("mutation", [
    "missing_pass", "failed_pass", "nan_gradient", "infinite_loss", "missing_boundary",
    "bf16_boundary", "missing_vjp", "unclean_oracle", "routing_guard", "wrong_heads",
    "tensor_inventory", "layer_coverage", "pair_coverage", "shape_mismatch", "pass_profile",
    "missing_attention_profile", "warn_only", "math_main", "fake_pass", "fake_no_verdict",
    "claimed_complete_error", "missing_ga_layer", "missing_bf16_layer",
])
def test_moonlight_diagnostic_never_relaxes_technical_or_verdict_evidence(
    tmp_path: Path, mutation: str,
) -> None:
    import torch

    manifest, result = _receipt(tmp_path)
    row = result["passes"]["fp8_oracle"]
    if mutation == "missing_pass":
        del result["passes"]["ga_floor"]
    elif mutation == "failed_pass":
        row["status"] = "FAILED"
        row["error"] = {"type": "OutOfMemoryError", "message": "serialized OOM fixture"}
    elif mutation == "claimed_complete_error":
        row["error"] = {"type": "RoutingReplayError", "message": "refused route fixture"}
    elif mutation == "missing_ga_layer":
        result["passes"]["ga"]["layer_forward_counts"] = [7]
    elif mutation == "missing_bf16_layer":
        result["passes"]["bf16_l2l"]["layers"] = []
    elif mutation == "nan_gradient":
        path = tmp_path / "fp8_oracle" / "gradients.pt"
        values = torch.load(path, map_location="cpu", weights_only=True)
        values[result["trainable_names"][0]][0, 0] = float("nan")
        torch.save(values, path)
    elif mutation == "infinite_loss":
        row["losses"][0] = float("inf")
    elif mutation == "missing_boundary":
        (tmp_path / "fp8_oracle/recompute/layer-000-mb-003.pt").unlink()
    elif mutation == "bf16_boundary":
        path = tmp_path / "bf16_l2l/recompute/layer-000-mb-003.pt"
        values = torch.load(path, map_location="cpu", weights_only=True)
        values["output"] = torch.full((1, 64, 2), float("nan"), dtype=torch.bfloat16)
        torch.save(values, path)
    elif mutation == "missing_vjp":
        row["oracle"]["counts"]["oracle_vjp"] -= 1
    elif mutation == "unclean_oracle":
        row["oracle_after_close"]["owned_device_storage_bytes"] = 1
    elif mutation == "routing_guard":
        row["routing_guard"] = "disabled"
    elif mutation == "wrong_heads":
        result["qualification_only"] = False
        for call in row["attention"]["unique_calls"] + row["attention"]["samples"]:
            for field in ("query_shape", "key_shape", "value_shape"):
                call[field][1] = 128
    elif mutation == "tensor_inventory":
        index = next(index for index, name in enumerate(result["trainable_names"])
                     if ".q_proj." in name)
        result["trainable_names"][index] = result["trainable_names"][index].replace(
            ".q_proj.", ".q_b_proj.")
    elif mutation == "layer_coverage":
        row["layers"].append(copy.deepcopy(row["layers"][0]))
    elif mutation == "pair_coverage":
        row["layers"][0]["pairs"].pop()
    elif mutation == "shape_mismatch":
        path = tmp_path / "fp8_oracle/gradients.pt"
        values = torch.load(path, map_location="cpu", weights_only=True)
        name = result["trainable_names"][0]
        values[name] = torch.ones(2, 3)
        row["gradient_checks"][name]["shape"] = [2, 3]
        torch.save(values, path)
    elif mutation == "pass_profile":
        row["profile_id"] = DEEPSEEK
    elif mutation == "missing_attention_profile":
        del row["attention"]["profile_id"]
    elif mutation == "warn_only":
        result["determinism"]["execution"]["warn_only"] = True
    elif mutation == "math_main":
        row["backend_requested"] = "MATH"
    else:
        result["verdict"] = "PASS" if mutation == "fake_pass" else "NO_VERDICT"
    _commit(tmp_path, result)
    with pytest.raises(ValueError):
        _verify(manifest)


@pytest.mark.parametrize("heads", [2, 128])
def test_production_moonlight_requires_sixteen_executed_heads(heads: int) -> None:
    observation = _attention_receipt("EFFICIENT_ATTENTION")
    observation["profile_id"] = MOONLIGHT
    row = {"backend_requested": "EFFICIENT_ATTENTION", "attention": observation}
    for call in observation["unique_calls"] + observation["samples"]:
        call["query_shape"] = call["key_shape"] = [1, heads, 64, 192]
        call["value_shape"] = [1, heads, 64, 128]
    with pytest.raises(ValueError, match="MLA"):
        gate._require_pass_attention("ga", row, min_calls=8, production=True,
                                     profile_id=MOONLIGHT)
    for call in observation["unique_calls"] + observation["samples"]:
        for field in ("query_shape", "key_shape", "value_shape"):
            call[field][1] = 16
    gate._require_pass_attention("ga", row, min_calls=8, production=True, profile_id=MOONLIGHT)


def test_moonlight_fixture_numerics_never_become_production_admission(tmp_path: Path) -> None:
    manifest, _ = _receipt(tmp_path)
    assert _verify(manifest)["verdict"] == "FAIL"
    with pytest.raises(ValueError, match="production|qualification"):
        gate.load_gate(manifest, _digest(manifest), profile_id=MOONLIGHT)


def test_moonlight_smoke_cpu_refusal_keeps_profile_and_fixture_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "smoke"
    with pytest.raises(RuntimeError, match="smoke FAILED"):
        smoke.run_smoke(output=output, profile_id=MOONLIGHT)
    report = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    failure = json.loads((output / "FAILED.json").read_text(encoding="utf-8"))
    assert report["profile_id"] == failure["profile_id"] == MOONLIGHT
    assert report["status"] == "FAILED"
    assert report["qualification_only"] is report["test_fixture"] is True
    assert report["scientific_admission"] is False
    assert report["attempts"] == []


@pytest.mark.parametrize("a3_status", ["FAIL", "NO_VERDICT"])
def test_moonlight_a3_numerical_failure_preserves_other_complete_diagnostics(
    tmp_path: Path, a3_status: str,
) -> None:
    import torch

    from benchmarks.harness.l2l_mistral_stand import RunRecord, decide

    manifest, result = _receipt(tmp_path, verdict="PASS")
    changed = "bf16_l2l" if a3_status == "FAIL" else "ga_repeat"
    path = tmp_path / changed / "gradients.pt"
    values = torch.load(path, map_location="cpu", weights_only=True)
    torch.save({name: 2 * value for name, value in values.items()}, path)
    if changed == "bf16_l2l":
        # Keep E3 exact against the changed BF16 reference; A3 alone must fail.
        torch.save({name: 2 * value for name, value in values.items()},
                   tmp_path / "fp8_oracle/gradients.pt")
        for metric in result["passes"]["int4_oracle"]["gradient_agreement"]["per_tensor"].values():
            metric["rel_l2"] = 0.0
        agreement = result["passes"]["int4_oracle"]["gradient_agreement"]
        agreement.update(global_rel_l2=0.0, max_rel_l2=0.0)
        for part in agreement["worst"]:
            part["rel_l2"] = 0.0
        result["control"]["gates"]["E3"].update(status="PASS", global_rel_l2=0.0)
        result["control"]["discrimination"].update(status="NO_VERDICT", not_failed=["E3"])

    def record(name: str) -> RunRecord:
        gradients = torch.load(tmp_path / name / "gradients.pt", map_location="cpu",
                               weights_only=True)
        return RunRecord(logs=[{"loss": 8.0, "learning_rate": 0.0}], rates=[0.0],
                         grads=[gradients], steps=1, l2l=name == "bf16_l2l")

    # The frozen A3 rule is the independent reference, not the gate adapter around it.
    rule = decide(record("ga"), record("ga_repeat"), record("bf16_l2l"), mode="bf16",
                  floor=record("ga_floor"))
    result["a3"] = json.loads(json.dumps({
        "status": a3_status, "rule": rule, "independent_floor_observed": True,
        "optimizer_updates": 0, "comparison_observations": 1,
    }))
    result["verdict"] = a3_status
    _commit(tmp_path, result)
    verified = _verify(manifest)
    assert verified["verdict"] == verified["a3"]["status"] == a3_status
    assert all(row["status"] == "COMPLETE" for row in verified["passes"].values())


@pytest.mark.usefixtures("moonlight_aux_admission")
def test_public_moonlight_loader_rechecks_raw_vjp_and_every_live_binding(tmp_path: Path) -> None:
    manifest, result = _receipt(tmp_path, production=True)
    assert gate.load_gate(manifest, _digest(manifest), profile_id=MOONLIGHT)["verdict"] == "FAIL"
    pristine = copy.deepcopy(result)
    for mutation in ("vjp", "pack", "cache", "config", "adapter", "environment",
                     "source", "profile", "verdict"):
        result = copy.deepcopy(pristine)
        if mutation == "vjp":
            result["passes"]["fp8_oracle"]["oracle"]["counts"]["oracle_vjp"] -= 1
        elif mutation == "pack":
            result["pack"]["manifest_sha256"] = "0" * 64
        elif mutation == "cache":
            result["model"]["cache"]["source_fingerprint"] = "0" * 64
        elif mutation == "config":
            result["model"]["source"]["config_sha256"] = "0" * 64
        elif mutation == "adapter":
            result["passes"]["ga_floor"]["initial_adapter_sha256"] = "0" * 64
        elif mutation == "environment":
            del result["environment"]["packages"]["transformers"]
        elif mutation == "source":
            result["model"]["source"]["revision"] = "0" * 40
        elif mutation == "profile":
            result["model"]["cache"]["profile_id"] = DEEPSEEK
        else:
            result["verdict"] = "PASS"
        _commit(tmp_path, result)
        with pytest.raises(ValueError):
            gate.load_gate(manifest, _digest(manifest), profile_id=MOONLIGHT)


@pytest.mark.gpu
def test_native_moonlight_smoke_allocates_profile_heads_without_claiming_model_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from benchmarks.harness.p3_deepseek_quality_attention import require_ampere_cuda

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        require_ampere_cuda("cuda")
    except ValueError as error:
        pytest.skip(str(error))
    (tmp_path / "receipts").mkdir()
    report = {"profile_id": MOONLIGHT, "attempts": []}
    with gate._deterministic("cuda"):
        smoke._attention_cases(tmp_path, report)
    assert len(report["attempts"]) == 6
    for row in report["attempts"]:
        assert row["status"] == "COMPLETE"
        assert row["profile_id"] == MOONLIGHT
        assert row["heads"] == 16 and row["qk_dim"] == 192 and row["v_dim"] == 128
        assert row["qualification_only"] is row["test_fixture"] is True
        assert row["scientific_admission"] is False
        for call in row["attention"]["unique_calls"]:
            assert call["query_shape"] == call["key_shape"] == [1, 16, row["length"], 192]
            assert call["value_shape"] == [1, 16, row["length"], 128]
    torch.cuda.empty_cache()


@pytest.mark.usefixtures("moonlight_aux_admission")
def test_moonlight_cli_accepts_only_raw_verified_diagnostic_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, result = _receipt(tmp_path, production=True)
    monkeypatch.setattr(gate, "run_gate", lambda **_arguments: result)
    arguments = ["--source", "unused", "--cache", "unused", "--pack", "unused",
                 "--pack-sha256", "0" * 64, "--initial-adapter", "unused",
                 "--output", str(tmp_path), "--profile", MOONLIGHT]
    # Only native execution is replaced. CLI admission still hashes and loads every raw artifact.
    assert gate.main(arguments) == 0
    assert result["verdict"] == "FAIL"
    result["passes"]["fp8_oracle"]["oracle"]["counts"]["oracle_vjp"] -= 1
    _commit(tmp_path, result)
    with pytest.raises(ValueError, match="VJP"):
        gate.main(arguments)
    result["status"] = "FAILED"
    assert gate.main(arguments) == 1
    assert manifest.is_file()


@pytest.mark.gpu
@pytest.mark.parametrize("perturbed", ["ga_repeat", "bf16_l2l", "fp8_oracle"])
def test_native_moonlight_fixture_completes_all_six_after_finite_numerical_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, perturbed: str,
) -> None:
    import torch

    from benchmarks.harness import p3_deepseek_quality_model as model_api
    from benchmarks.harness.b13_moe_fixture import moe_config, moe_weights_dir
    from benchmarks.harness.p3_deepseek_quality_attention import require_ampere_cuda

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        require_ampere_cuda("cuda")
    except ValueError as error:
        pytest.skip(str(error))
    config = moe_config(
        num_hidden_layers=2, first_k_dense_replace=1, q_lora_rank=None,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=1024,
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    config._experts_implementation = "eager"
    source = moe_weights_dir(tmp_path, config)
    cache = tmp_path / "cache"
    initial = tmp_path / "initial.safetensors"
    with model_api.open_session(source, cache, device="cuda", prepare=True,
                                profile_id=MOONLIGHT) as session:
        model_api.write_initial_adapter(session.model, initial, profile_id=MOONLIGHT)
    native_pass = gate._run_pass

    def measured_pass(*arguments: Any, **options: Any) -> Any:
        row, gradients, archive = native_pass(*arguments, **options)
        if options["name"] == perturbed and row["status"] == "COMPLETE":
            # Finite snapshot fault injection after native backward/cleanup, not a mocked pass.
            gradients = {name: value * 1e10 for name, value in gradients.items()}
            artifact = options["directory"] / "gradients.pt"
            torch.save(gradients, artifact)
            row["gradient_artifact"].update(bytes=artifact.stat().st_size,
                                             sha256=_digest(artifact))
        return row, gradients, archive

    monkeypatch.setattr(gate, "_run_pass", measured_pass)
    report = gate.run_gate(
        source=source, cache=cache, initial_adapter=initial, output=tmp_path / "evidence",
        window=_window(), fixture=True, profile_id=MOONLIGHT,
    )
    assert report["status"] == "COMPLETE", report.get("error")
    assert set(report["passes"]) == set(PASSES)
    assert all(row["status"] == "COMPLETE" for row in report["passes"].values())
    assert "stopped_after" not in report
    assert report["qualification_only"] is True
    assert report["verdict"] != "PASS"
    if perturbed == "ga_repeat":
        assert report["a3"]["status"] == "NO_VERDICT"
    elif perturbed == "bf16_l2l":
        assert report["a3"]["status"] == "FAIL"
    else:
        assert report["candidate_gates"]["E3"]["status"] == "FAIL"
    assert report["passes"]["int4_oracle"]["oracle"]["complete"] is True
    assert report["session_closed"] is True
    manifest = tmp_path / "evidence/manifest.json"
    with pytest.raises(ValueError, match="production|qualification"):
        gate.load_gate(manifest, _digest(manifest), profile_id=MOONLIGHT)
