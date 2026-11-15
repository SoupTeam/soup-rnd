"""Profile-bound study admission, with native execution reserved for remote verification."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.harness import p3_deepseek_quality_attention as attention
from benchmarks.harness import p3_deepseek_quality_eval as scoring
from benchmarks.harness import p3_deepseek_quality_train as train
from benchmarks.harness.p3_deepseek_quality_profiles import get_profile

MOONLIGHT = "moonlight-16b-a3b"


def _json(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol(profile_id: str, *, steps: int = 64) -> dict:
    profile = get_profile(profile_id)
    return {
        "schema": "p3-deepseek-quality-protocol-v1", "status": "FROZEN",
        "profile_id": profile_id, "include_base_model_evaluation": False,
        "numerical_gate_blocks_quality": False, "arms": list(train.ARMS),
        "evaluation_storage": "resident_nf4" if profile_id == MOONLIGHT else "disk",
        "model": {"model_id": profile.model_id, "revision": profile.revision},
        "attention_policy": attention.policy_identity(profile_id),
        "initial_adapter_sha256": "a" * 64, "pack_sha256": "b" * 64,
        "training": {
            "seed": 23, "steps": steps, "lengths": [1024, 4096],
            "input_tokens_per_step": 4096, "micro_batch_size": 1,
            "optimizer": copy.deepcopy(train.OPTIMIZER),
            "scheduler": {"name": "cosine", "warmup_steps": 2, "horizon": steps},
        },
        "budget": {"limit_usd": 15, "compute_stop_usd": 14, "closeout_reserve_usd": 1,
                   "upper_bound_total_usd": 14.5, "adjustments": []},
        "data": {"manifest_sha256": scoring.DATA_MANIFEST_SHA256},
        "loglik": {"prepend_bos": False, "hellaswag_n": 10042},
        "scoring": {"truncation": False},
    }


def test_training_rejects_cross_profile_protocol_before_aux_admission(tmp_path: Path) -> None:
    path = tmp_path / "protocol.json"
    digest = _json(path, _protocol("deepseek-v3", steps=3))
    with pytest.raises(ValueError, match="profile"):
        train._protocol(path, digest, steps=64, length=1024, pack_sha256="b" * 64,
                        initial_sha256="a" * 64, pilot=False, test_fixture=False,
                        profile_id=MOONLIGHT)


@pytest.mark.parametrize("steps", [1, 3, 63, 65])
def test_moonlight_full_training_cannot_shorten_fixed_horizon(steps: int) -> None:
    with pytest.raises(ValueError, match="64"):
        train._protocol(None, None, steps=steps, length=1024, pack_sha256="b" * 64,
                        initial_sha256="a" * 64, pilot=False, test_fixture=True,
                        profile_id=MOONLIGHT)


@pytest.mark.parametrize("damage", ["missing_profile", "blocking_gate", "one_arm", "steps"])
def test_eval_protocol_rejects_moonlight_scope_conflicts(tmp_path: Path, damage: str) -> None:
    value = _protocol(MOONLIGHT)
    if damage == "missing_profile":
        value.pop("profile_id")
    elif damage == "blocking_gate":
        value["numerical_gate_blocks_quality"] = True
    elif damage == "one_arm":
        value["arms"] = ["bf16_l2l"]
    else:
        value["training"]["steps"] = 3
    path = tmp_path / "protocol.json"
    digest = _json(path, value)
    with pytest.raises(ValueError, match="profile|Moonlight"):
        scoring.admit_protocol(path, digest, {
            "loglik": {"prepend_bos": False, "hellaswag_n": 10042},
            "scoring": {"truncation": False},
            "data": {"full_retained_evaluation_tasks": {"hellaswag": 10042}},
        }, profile_id=MOONLIGHT)


def test_moonlight_evaluation_rejects_unqualified_storage(tmp_path: Path, monkeypatch) -> None:
    admission = {"contract": {"sha256": "d" * 64}, "evidence": {"sha256": "e" * 64}}
    monkeypatch.setattr(train.model_api, "native_aux_admission", lambda _profile: admission)
    value = _protocol(MOONLIGHT)
    value.update(
        aux_admission=admission,
        source={"profile_id": MOONLIGHT, "aux_admission": admission},
        cache={"profile_id": MOONLIGHT}, tokenizer={"profile_id": MOONLIGHT},
    )
    reference = {
        "loglik": value["loglik"], "scoring": value["scoring"],
        "data": {"full_retained_evaluation_tasks": {"hellaswag": 10042}},
    }
    path = tmp_path / "protocol.json"
    scoring.admit_protocol(path, _json(path, value), reference, profile_id=MOONLIGHT)
    for storage in ("disk", "ram", None):
        value["evaluation_storage"] = storage
        with pytest.raises(ValueError):
            scoring.admit_protocol(path, _json(path, value), reference, profile_id=MOONLIGHT)


def test_gate_binding_cannot_reinterpret_missing_or_foreign_profile() -> None:
    metadata = {"profile_id": MOONLIGHT, "source": {"profile_id": MOONLIGHT},
                "cache": {"profile_id": MOONLIGHT}}
    execution = {"profile_id": MOONLIGHT, "attention_policy": {"sha256": "a" * 64},
                 "gate_environment": {"gpu": "pinned"}}
    result = {"profile_id": MOONLIGHT, "model": copy.deepcopy(metadata),
              "initial_adapter_sha256": "b" * 64,
              "attention_policy": execution["attention_policy"],
              "environment": execution["gate_environment"]}
    train.require_gate_binding(result, model_metadata=metadata, initial_sha256="b" * 64,
                               execution_identity=execution, profile_id=MOONLIGHT)
    for profile in (None, "deepseek-v3"):
        changed = copy.deepcopy(result)
        if profile is None:
            changed.pop("profile_id")
        else:
            changed["profile_id"] = profile
        with pytest.raises(ValueError, match="profile"):
            train.require_gate_binding(changed, model_metadata=metadata,
                                       initial_sha256="b" * 64, execution_identity=execution,
                                       profile_id=MOONLIGHT)


def test_moonlight_base_quality_is_refused_before_resource_or_model_admission(
    tmp_path: Path,
) -> None:
    result = scoring.run_endpoint(
        data_root=tmp_path / "absent-data", endpoint="arc", start=0, stop=1,
        source=tmp_path / "absent-source", cache=tmp_path / "absent-cache",
        output=tmp_path / "evaluation", profile_id=MOONLIGHT,
    )
    assert result["status"] == "FAILED"
    assert "untrained base" in result["failure"]["message"]
    assert result["profile_id"] == MOONLIGHT
    assert "model" not in result


def _adapter(root: Path, *, profile_id: str = MOONLIGHT, steps: int = 64) -> str:
    import torch
    from safetensors.torch import save_file

    root.mkdir()
    _json(root / "adapter_config.json", {
        "peft_type": "LORA", "r": 2, "lora_alpha": 4, "lora_dropout": 0,
        "bias": "none", "target_modules": ["q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"],
    })
    save_file({"admission_fixture": torch.ones(1, dtype=torch.float32)},
              str(root / "adapter_model.safetensors"))
    identity = {
        "profile_id": profile_id, "arm": "bf16_l2l", "length": 1024, "seed": 23,
        "completed_steps": steps, "initial_adapter_sha256": "a" * 64,
        "pack_sha256": "b" * 64, "protocol_sha256": "c" * 64,
        "source": {"profile_id": profile_id, "receipt_sha256": "d" * 64},
        "cache": {"profile_id": profile_id, "sha256": "e" * 64},
        "attention_policy": attention.policy_identity(profile_id),
        "tokenizer": {"profile_id": profile_id},
    }
    return _json(root / "manifest.json", {
        "schema": "p3-deepseek-quality-adapter-v1", "profile_id": profile_id,
        "identity": identity,
        "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in root.iterdir()},
    })


def test_moonlight_final_adapter_uses_native_targets_and_refuses_cross_profile(
    tmp_path: Path,
) -> None:
    root = tmp_path / "final"
    digest = _adapter(root)
    admitted = scoring.admit_adapter(root, digest, profile_id=MOONLIGHT)
    assert admitted["identity"]["completed_steps"] == 64
    with pytest.raises(ValueError, match="profile"):
        scoring.admit_adapter(root, digest)
    value = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    value["identity"]["completed_steps"] = 3
    digest = _json(root / "manifest.json", value)
    with pytest.raises(ValueError, match="64"):
        scoring.admit_adapter(root, digest, profile_id=MOONLIGHT)
    value["identity"]["completed_steps"] = 64
    value["identity"]["tokenizer"]["profile_id"] = "deepseek-v3"
    digest = _json(root / "manifest.json", value)
    with pytest.raises(ValueError, match="profile"):
        scoring.admit_adapter(root, digest, profile_id=MOONLIGHT)
    value["identity"]["tokenizer"]["profile_id"] = MOONLIGHT
    config_path = root / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["target_modules"] = ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]
    value["files"]["adapter_config.json"] = _json(config_path, config)
    digest = _json(root / "manifest.json", value)
    with pytest.raises(ValueError, match="adapter configuration"):
        scoring.admit_adapter(root, digest, profile_id=MOONLIGHT)


@pytest.mark.parametrize("verdict", ["FAIL", "NO_VERDICT"])
def test_numeric_diagnostic_policy_never_admits_incomplete_gate(
    tmp_path: Path, verdict: str,
) -> None:
    from benchmarks.harness import p3_deepseek_quality_timing as timing

    gate = tmp_path / "gate.json"
    digest = _json(gate, {
        "schema": "p3-deepseek-quality-gate-v1", "profile_id": MOONLIGHT,
        "status": "COMPLETE", "verdict": verdict, "qualification_only": False,
    })
    result = timing.run_timing(
        source=tmp_path / "absent-source", cache=tmp_path / "absent-cache",
        initial=tmp_path / "absent-initial", initial_sha256="a" * 64,
        data_root=tmp_path / "absent-data", output=tmp_path / "timing",
        max_tokens_per_batch=8192, deadline_seconds=900,
        gate=gate, gate_sha256=digest, profile_id=MOONLIGHT,
    )
    assert result["status"] == "FAILED"
    assert result["error"]["type"] == "ValueError"
    assert "hardware" not in result
    assert result["endpoints"] == {}
    assert result["quality_outcomes_computed"] is False


@pytest.mark.parametrize("limit", [9, 31])
def test_moonlight_budget_omission_requires_prior_prefix_cut(limit: int) -> None:
    value = _protocol(MOONLIGHT)
    value["budget"].update(
        limit_usd=limit, compute_stop_usd=limit - 1, upper_bound_total_usd=limit - 0.5,
    )
    train._require_moonlight_budget(value)
    value["training"]["lengths"] = [1024]
    with pytest.raises(ValueError, match="budget cuts"):
        train._require_moonlight_budget(value)
    value["loglik"]["hellaswag_n"] = 3000
    value["budget"]["adjustments"] = [
        {"action": "hellaswag_prefix", "before_n": 10042, "after_n": 3000,
         "upper_bound_before_usd": limit + 5, "upper_bound_after_usd": limit + 2},
        {"action": "omit_L4096_pair", "upper_bound_before_usd": limit + 2,
         "upper_bound_after_usd": limit - 0.5},
    ]
    train._require_moonlight_budget(value)
    value["budget"]["adjustments"].reverse()
    with pytest.raises(ValueError, match="budget cuts"):
        train._require_moonlight_budget(value)


@pytest.mark.parametrize("damage", ["nonfinite", "fit_before_omission", "count", "reserve"])
def test_moonlight_budget_cannot_claim_unmeasured_or_unneeded_cuts(damage: str) -> None:
    value = _protocol(MOONLIGHT)
    value["training"]["lengths"] = [1024]
    value["loglik"]["hellaswag_n"] = 3000
    value["budget"]["adjustments"] = [
        {"action": "hellaswag_prefix", "before_n": 10042, "after_n": 3000,
         "upper_bound_before_usd": 20, "upper_bound_after_usd": 17},
        {"action": "omit_L4096_pair", "upper_bound_before_usd": 17,
         "upper_bound_after_usd": 14.5},
    ]
    if damage == "nonfinite":
        value["budget"]["upper_bound_total_usd"] = float("nan")
    elif damage == "fit_before_omission":
        value["budget"]["adjustments"][0]["upper_bound_after_usd"] = 14
        value["budget"]["adjustments"][1]["upper_bound_before_usd"] = 14
    elif damage == "count":
        value["loglik"]["hellaswag_n"] = 3001
    else:
        value["budget"]["closeout_reserve_usd"] = 0
    with pytest.raises(ValueError):
        train._require_moonlight_budget(value)


@pytest.mark.parametrize("verdict", ["FAIL", "NO_VERDICT"])
def test_raw_verified_numeric_diagnostics_keep_their_training_receipt_verdict(
    tmp_path: Path, verdict: str,
) -> None:
    from tests.test_p3_moonlight_gate import _receipt, _verify

    manifest, _ = _receipt(tmp_path, profile_id=MOONLIGHT, verdict=verdict, production=False)
    verified = _verify(manifest, profile_id=MOONLIGHT)
    receipt = train._gate_admission(
        verified, hashlib.sha256(manifest.read_bytes()).hexdigest(), profile_id=MOONLIGHT,
    )
    assert receipt["overall"] == verdict
    assert verified["qualification_only"] is True


def test_missing_oracle_vjp_is_not_a_nonblocking_numeric_diagnostic(tmp_path: Path) -> None:
    from tests.test_p3_moonlight_gate import _commit, _receipt, _verify

    manifest, result = _receipt(tmp_path, profile_id=MOONLIGHT, verdict="FAIL", production=False)
    result["passes"]["fp8_oracle"]["oracle"]["counts"]["oracle_vjp"] = 0
    _commit(tmp_path, result)
    with pytest.raises(ValueError):
        _verify(manifest, profile_id=MOONLIGHT)


def test_retained_arms_own_independent_optimizer_state_and_lr0_first_update() -> None:
    import torch

    left = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    right = torch.nn.Parameter(left.detach().clone())
    baseline, baseline_schedule = train.new_optimizer({"adapter": left}, steps=64)
    candidate, candidate_schedule = train.new_optimizer({"adapter": right}, steps=64)
    for parameter, optimizer, scheduler in (
        (left, baseline, baseline_schedule), (right, candidate, candidate_schedule),
    ):
        parameter.grad = torch.tensor([0.5, -0.25])
        assert optimizer.param_groups[0]["lr"] == 0
        before = parameter.detach().clone()
        optimizer.step()
        scheduler.step()
        assert torch.equal(parameter, before)
        assert optimizer.param_groups[0]["lr"] == 0.0001
    baseline.step()
    baseline_schedule.step()
    assert not torch.equal(left, right)
    assert int(baseline.state[left]["step"]) == 2
    assert int(candidate.state[right]["step"]) == 1
    assert (baseline.state[left]["exp_avg"].data_ptr()
            != candidate.state[right]["exp_avg"].data_ptr())
    assert candidate_schedule.last_epoch == 1


def test_resume_refuses_cross_profile_before_opening_checkpoint_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="profile"):
        train.load_resume(tmp_path / "absent", identity={"profile_id": "deepseek-v3"},
                          infrastructure_event=tmp_path / "absent-event", profile_id=MOONLIGHT)


@pytest.mark.usefixtures("moonlight_aux_admission")
@pytest.mark.parametrize("verdict", ["FAIL", "NO_VERDICT"])
def test_timing_retains_verified_numeric_diagnostic_before_native_cpu_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verdict: str,
) -> None:
    from benchmarks.harness import p3_deepseek_quality_timing as timing
    from tests.test_p3_moonlight_gate import _receipt

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    manifest, _ = _receipt(tmp_path / "gate", profile_id=MOONLIGHT,
                           verdict=verdict, production=True)
    result = timing.run_timing(
        source=tmp_path / "absent-source", cache=tmp_path / "absent-cache",
        initial=tmp_path / "absent-initial", initial_sha256="a" * 64,
        data_root=tmp_path / "absent-data", output=tmp_path / "timing",
        max_tokens_per_batch=8192, deadline_seconds=900, device="cpu",
        gate=manifest, gate_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        profile_id=MOONLIGHT,
    )
    assert result["status"] == "FAILED"
    assert result["gate"]["overall"] == verdict
    assert result["error"]["type"] == "ValueError"
    assert "CUDA" in result["error"]["message"]
    assert result["quality_outcomes_computed"] is False
    assert result["evaluation_rows_opened"] == 0
    assert result["endpoints"] == {}


def test_nonfinite_raw_gradient_cannot_be_relaxed_as_numeric_gate_failure(tmp_path: Path) -> None:
    import torch

    from tests.test_p3_moonlight_gate import _commit, _receipt, _verify

    manifest, result = _receipt(tmp_path, profile_id=MOONLIGHT, verdict="FAIL", production=False)
    path = tmp_path / "fp8_oracle" / "gradients.pt"
    tensors = torch.load(path, weights_only=True)
    next(iter(tensors.values()))[0, 0] = float("nan")
    torch.save(tensors, path)
    _commit(tmp_path, result)
    with pytest.raises(ValueError):
        _verify(manifest, profile_id=MOONLIGHT)


@pytest.mark.usefixtures("moonlight_aux_admission")
def test_frozen_moonlight_protocol_binds_approved_auxiliary_evidence(tmp_path: Path) -> None:
    from benchmarks.harness import p3_deepseek_quality_model as model_api

    admission = model_api.native_aux_admission(MOONLIGHT)
    assert admission is not None
    value = _protocol(MOONLIGHT)
    value.update(
        aux_admission=copy.deepcopy(admission),
        source={"profile_id": MOONLIGHT, "aux_admission": copy.deepcopy(admission)},
        cache={"profile_id": MOONLIGHT}, tokenizer={"profile_id": MOONLIGHT},
    )
    path = tmp_path / "frozen.json"
    digest = _json(path, value)
    admitted, _ = train._protocol(
        path, digest, steps=64, length=1024, pack_sha256="b" * 64,
        initial_sha256="a" * 64, pilot=False, test_fixture=False, profile_id=MOONLIGHT,
    )
    assert admitted["training"]["steps"] == 64
    value["aux_admission"]["evidence"]["sha256"] = "0" * 64
    digest = _json(path, value)
    with pytest.raises(ValueError, match="auxiliary admission"):
        train._protocol(path, digest, steps=64, length=1024, pack_sha256="b" * 64,
                        initial_sha256="a" * 64, pilot=False, test_fixture=False,
                        profile_id=MOONLIGHT)


def test_short_moonlight_arm_writes_terminal_failure_before_pack_or_model(tmp_path: Path) -> None:
    output = tmp_path / "short-arm"
    with pytest.raises(RuntimeError, match="training FAILED"):
        train.run_training(
            source=tmp_path / "absent-source", cache=tmp_path / "absent-cache",
            pack=tmp_path / "absent-pack", pack_sha256="a" * 64,
            initial_adapter=tmp_path / "absent-initial", arm="bf16_l2l", length=1024,
            steps=3, output=output, profile_id=MOONLIGHT,
        )
    result = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    terminal = json.loads((output / "FAILED.json").read_text(encoding="utf-8"))
    assert result["profile_id"] == terminal["profile_id"] == MOONLIGHT
    assert result["completed_steps"] == 0
    assert result["resumable"] is False
    assert "64" in result["error"]["message"]
    assert not (tmp_path / "absent-cache").exists()


@pytest.mark.parametrize("all_zero", [False, True])
def test_finite_zero_training_gradients_are_diagnostic_only_for_moonlight(all_zero):
    import torch

    parameters = {name: torch.nn.Parameter(torch.ones(2)) for name in ("zero", "other")}
    parameters["zero"].grad = torch.zeros(2)
    parameters["other"].grad = torch.zeros(2) if all_zero else torch.ones(2)
    evidence, captured, norm = train._gradients(parameters, True, profile_id=MOONLIGHT)
    assert evidence["zero"]["nonzero"] == 0
    assert evidence["zero"]["l2"] == 0.0 and evidence["zero"]["finite"] is True
    assert torch.equal(captured["zero"], torch.zeros(2))
    assert norm == pytest.approx(0.0 if all_zero else 2**0.5)
    with pytest.raises(ValueError, match="zero"):
        train._gradients(parameters, True)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_moonlight_nonfinite_training_gradient_still_stops(value):
    import torch

    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.grad = torch.tensor([value, 0.0])
    with pytest.raises(ValueError, match="nonfinite"):
        train._gradients({"adapter": parameter}, False, profile_id=MOONLIGHT)
