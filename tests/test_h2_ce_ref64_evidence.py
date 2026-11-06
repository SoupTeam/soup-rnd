"""Synthetic evidence tests: exercise seals/authority without claiming a CUDA run."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks/harness"))
sys.path.insert(0, str(Path(__file__).parent))
import h2_ce_confirmation as runner  # noqa: E402
import h2_ce_ref64_evidence as gates  # noqa: E402
import h2_ce_ref64_review as review  # noqa: E402
from test_h2_ce_confirmation import authority  # noqa: E402
from test_h2_ce_controls import good_record  # noqa: E402
from test_h2_ce_fp64_reference import build_tree, make_spec  # noqa: E402


def write(path, value):
    path.write_text(json.dumps(value))


def reseal(path):
    (path / "files_sha256.json").unlink(missing_ok=True)
    runner.tree_seal(path)


def sealed_tree(tmp_path, monkeypatch):
    # Only the large model tensor reader is synthetic. All authority, trace, seal,
    # hook, process-control and contract checks execute their production code.
    root = build_tree(tmp_path, monkeypatch, make_spec())
    monkeypatch.setattr(review, "validate_series_authority", gates.validate_series_authority)
    contract = authority(tmp_path)
    contract.update(series="ref64", evidence_root=str(root))
    write(root / "contract.json", contract)
    initial = json.loads((runner.SPEC / "operator_study.json").read_text())["reference_identity"]
    for phase in runner.REF64_PHASES:
        child = root / phase / "runner"
        data = json.loads((child / "evidence.json").read_text())
        arm, kind = runner.PHASE_ARMS[phase]
        data.update(
            phase=phase,
            initial_identity=initial["before_sha256"],
            **{key: contract[key] for key in ("source_commit", "gpu_uuid", "boot_id")},
        )
        data["ref64"]["config"] = "real"
        calls = int(arm == "ce_only")
        data["patch_after"] = {
            "status": "PASS_EXPLICIT_PATCH_BINDINGS",
            "arm": arm,
            "patch_calls": {"class": calls, "instance": calls},
        }
        write(child / "evidence.json", data)
        write(child / "contract.json", contract)
        controls = good_record()
        controls["phase"] = phase
        write(child / "controls.json", controls)
        trace = child / ("pass_0_0" if kind == "perturbed" else "probe_00") / "kernel_trace.json"
        write(
            trace,
            {
                "traceEvents": [
                    {
                        "cat": "kernel",
                        "name": "liger_cross_entropy_kernel"
                        if arm == "ce_only"
                        else "ordinary_gemm",
                    }
                ]
            },
        )
        reseal(child)
    return root


def test_sealed_series_uses_real_authority_gate(tmp_path, monkeypatch):
    result = review.review_ref64(sealed_tree(tmp_path, monkeypatch))
    assert result["run_status"] == "COMPLETE", result
    assert result["label"].startswith("LIGER_CLOSER_BY_FLIPS")


@pytest.mark.parametrize(
    "fault",
    [
        "gpu_uuid",
        "boot_id",
        "source_commit",
        "initial_identity",
        "phase",
        "bindings",
        "foreign_reads",
        "unknown_owner",
        "extra_kernel",
        "contract",
        "source_pin",
        "missing_receipt",
        "cpu_config",
        "missing_hashes",
        "bad_hash",
        "unsealed_change",
        "wrong_series",
        "missing_flips",
        "wrong_seed",
        "wrong_count",
    ],
)
def test_invalid_evidence_never_gets_a_scientific_label(tmp_path, monkeypatch, fault):
    root = sealed_tree(tmp_path, monkeypatch)
    phase = (
        "ref64_perturbed_probe02"
        if fault in {"cpu_config", "missing_flips", "wrong_seed", "wrong_count"}
        else "baseline_probes"
    )
    child = root / phase / "runner"
    data = json.loads((child / "evidence.json").read_text())
    if fault in {"gpu_uuid", "boot_id", "source_commit", "initial_identity", "phase"}:
        data[fault] = "wrong"
    elif fault == "bindings":
        data["patch_after"]["status"] = "FAILED"
    elif fault == "missing_receipt":
        data.pop("determinism_receipt")
    elif fault == "cpu_config":
        data["ref64"]["config"] = "test"
    elif fault == "missing_flips":
        data["ref64"]["flips"] = []
    elif fault in {"wrong_seed", "wrong_count"}:
        data["ref64"]["flips"][0]["seed" if fault == "wrong_seed" else "flipped"] += 1
    elif fault in {"foreign_reads", "unknown_owner"}:
        p = child / "controls.json"
        controls = json.loads(p.read_text())
        if fault == "foreign_reads":
            controls["f1"]["checks"]["foreign_reads"]["outcome"] = "void"
        else:
            controls["ownership"]["before"] = None
        write(p, controls)
    elif fault == "extra_kernel":
        write(
            child / "probe_00/kernel_trace.json",
            {"traceEvents": [{"cat": "kernel", "name": "liger_cross_entropy_kernel"}]},
        )
    elif fault in {"contract", "source_pin", "wrong_series"}:
        path = child / "contract.json" if fault == "contract" else root / "contract.json"
        contract = json.loads(path.read_text())
        if fault == "source_pin":
            contract["artifacts_sha256"]["benchmarks/harness/h2_ce_ref64_evidence.py"] = "0" * 64
        elif fault == "wrong_series":
            contract["series"] = "noise_floor"
        else:
            contract["gpu_uuid"] = "wrong"
        write(path, contract)
    elif fault in {"missing_hashes", "bad_hash"}:
        # Missing values used to all compare equal as None.
        for name in runner.REF64_PHASES:
            target = root / name / "runner"
            path = target / "hooks/lm_head_sha256.json"
            hashes = {} if fault == "missing_hashes" else json.loads(path.read_text())
            if hashes:
                hashes[next(iter(hashes))] = "not-a-hash"
            write(path, hashes)
            reseal(target)
    write(child / "evidence.json", data)
    if fault != "unsealed_change":
        reseal(child)
    else:
        (child / "hooks/lm_head_sha256.json").write_text("{}")
    result = review.review_ref64(root)
    assert result["label"] == result["run_status"] == "VOID", result


@pytest.mark.parametrize(
    "error,status",
    [
        ("ValueError: Unexpected global patch binding", "VOID"),
        ("ValueError: GPU identity mismatch", "VOID"),
        ("TimeoutError: deadline", "INCOMPLETE"),
    ],
)
def test_offline_failure_classification_agrees_with_runner(tmp_path, monkeypatch, error, status):
    root = sealed_tree(tmp_path, monkeypatch)
    child = root / "ref64_probes/runner"
    write(child / "evidence.json", {"status": "FAILED", "error": error})
    result = review.review_ref64(root)
    assert result["run_status"] == result["label"] == runner.ref64_failure_status(child) == status


@pytest.mark.parametrize("fault", [None, "input", "initial", "cpu", "missing_gradient"])
def test_actual_probe_binding_gate(tmp_path, monkeypatch, fault):
    import torch

    spec = tmp_path / "spec"
    spec.mkdir()
    monkeypatch.setattr(runner, "SPEC", spec)
    before = {"trainable": {"x.lora_A": {}, "x.lora_B": {}}}
    write(
        spec / "operator_study.json",
        {"reference_identity": {"before_sha256": gates.identity(before)}},
    )
    batch = {"input_ids": [[1, 2, 3]], "labels": [[-100, 2, 3]], "attention_mask": [[1, 1, 1]]}
    write(spec / "ce_confirmation_workset.json", {"probes": [batch] * 4})
    values = {"input." + k: torch.tensor(v) for k, v in batch.items()}
    values.update({"gradient." + k: torch.zeros(1) for k in before["trainable"]})
    data = {"compute_policy": "bf16-autocast-cuda", "before": before}
    if fault == "input":
        values["input.input_ids"][0, 1] = 99
    elif fault == "initial":
        data["before"] = {"trainable": {}}
    elif fault == "cpu":
        data["compute_policy"] = "fp32-cpu-test"
    elif fault == "missing_gradient":
        del values["gradient.x.lora_A"]
    for phase, key in [("baseline_probes", "probe_00"), ("ref64_perturbed_probe02", "pass_0_0")]:
        path = tmp_path / phase / "runner" / key / "model_probe"
        if fault is None:
            gates.validate_probe(path, data, values)
        else:
            with pytest.raises(ValueError):
                gates.validate_probe(path, data, values)


@pytest.mark.parametrize("fault", [None, "dtype", "shape", "nan", "ignored"])
def test_dlogits_contract(tmp_path, monkeypatch, fault):
    import torch

    monkeypatch.setattr(gates, "REAL_VOCAB", 5)  # reduced CPU test only
    labels = torch.tensor([[-100, 2, -100]])
    grad = torch.zeros(1, 3, 5, dtype=torch.bfloat16)
    grad[:, 0] = 0.25
    if fault == "dtype":
        grad = grad.float()
    elif fault == "shape":
        grad = grad[..., :4]
    elif fault == "nan":
        grad[0, 0, 0] = float("nan")
    elif fault == "ignored":
        grad[0, 2, 0] = 1
    if fault is None:
        gates.validate_dlogits(grad, labels)
    else:
        with pytest.raises(ValueError):
            gates.validate_dlogits(grad, labels)


def test_probe_reader_always_runs_binding_gate(tmp_path, monkeypatch):
    import h2_numerics

    monkeypatch.setattr(h2_numerics, "read_probe", lambda path: ({}, {}))

    def refuse(*args):
        raise ValueError("frozen input mismatch")

    monkeypatch.setattr(review, "validate_probe", refuse)
    with pytest.raises(ValueError, match="frozen input"):
        review.read_probe_tree(tmp_path)


def test_all_missing_forward_hashes_do_not_pass_control():
    from test_h2_ce_fp64_reference import good_controls

    data = good_controls()
    for hashes in data["forward_hashes"].values():
        for key in hashes:
            hashes[key] = None
    assert any("control 3" in r for r in review.evaluate_controls(data)["failed"])


@pytest.mark.parametrize("status", ["VOID", "INCOMPLETE", "COMPLETE"])
def test_series_status_does_not_mask_invalid_review(tmp_path, monkeypatch, status):
    contract = authority(tmp_path)
    contract["series"] = "ref64"
    monkeypatch.setattr(runner, "validate_contract", lambda value: {})

    def child(argv, out, **kwargs):
        target = Path(argv[argv.index("--out") + 1])
        target.mkdir(parents=True)
        write(target / "evidence.json", {"status": "COMPLETE_CE_CAPTURE"})
        return {
            "status": "CHILD_EXITED",
            "group_cleanup_confirmed": True,
            "child_status": {"returncode": 0},
        }

    monkeypatch.setattr(runner, "bounded_process", child)
    monkeypatch.setattr(runner, "read_child", lambda *args: {})
    monkeypatch.setattr(
        review, "review_ref64", lambda *args: {"run_status": status, "label": status}
    )
    result = runner.run_series(contract, Path(contract["evidence_root"]))
    assert result["status"] == ("COMPLETE_CE_REF64_SERIES" if status == "COMPLETE" else status)


def test_series_deadline_before_next_child_is_incomplete(tmp_path, monkeypatch):
    contract = authority(tmp_path)
    contract.update(series="ref64", deadline_unix_s=0)
    monkeypatch.setattr(runner, "validate_contract", lambda value: {})
    result = runner.run_series(contract, Path(contract["evidence_root"]))
    assert result["status"] == "INCOMPLETE"
    assert result["phases"] == {}


def test_bad_child_control_stops_series_before_next_phase(tmp_path, monkeypatch):
    contract = authority(tmp_path)
    contract["series"] = "ref64"
    monkeypatch.setattr(runner, "validate_contract", lambda value: {})
    launched = []

    def child(argv, out, **kwargs):
        target = Path(argv[argv.index("--out") + 1])
        target.mkdir(parents=True)
        launched.append(target)
        write(target / "evidence.json", {"status": "COMPLETE_CE_CAPTURE"})
        return {
            "status": "CHILD_EXITED",
            "group_cleanup_confirmed": True,
            "child_status": {"returncode": 0},
        }

    def invalid(*args):
        raise ValueError("F1 control failed")

    monkeypatch.setattr(runner, "bounded_process", child)
    monkeypatch.setattr(runner, "read_child", invalid)
    result = runner.run_series(contract, Path(contract["evidence_root"]))
    assert result["status"] == "VOID"
    assert len(launched) == 1
