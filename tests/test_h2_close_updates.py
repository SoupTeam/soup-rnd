"""H2-close stage U: the `updates` series and its per-arm review (CPU fixtures)."""

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "harness"))
sys.path.insert(0, str(ROOT / "tests"))

import h2_ce_confirmation  # noqa: E402
import h2_close_updates as updates  # noqa: E402
from test_h2_ce_training import fixture_run  # noqa: E402

IDENTITY = {"source_commit": "c" * 40, "gpu_uuid": "GPU-x", "boot_id": "boot"}


def test_updates_series_is_the_two_training_children_and_pins_the_new_files():
    assert h2_ce_confirmation.SERIES["updates"] == ("baseline_training", "ce_only_training")
    assert h2_ce_confirmation.series_phases({"series": "updates"}) == tuple(updates.ARMS.values())
    assert "benchmarks/harness/h2_close_updates.py" in h2_ce_confirmation.ARTIFACTS
    assert "benchmarks/h2/H2_CLOSE_PROTOCOL.txt" in h2_ce_confirmation.ARTIFACTS
    assert tuple(updates.ARMS.values()) == h2_ce_confirmation.SERIES["updates"]


def build(tmp_path, *, initial="init", tamper=None):
    root = tmp_path / "series"
    contract = {"series": "updates", **IDENTITY}
    root.mkdir()
    (root / "contract.json").write_text(json.dumps(contract))
    work = None
    for phase in updates.ARMS.values():
        runner = root / phase / "runner"
        runner.mkdir(parents=True)
        work = fixture_run(runner / "training")
        record = json.loads((runner / "training" / "training.json").read_text())
        record["compute_policy"] = "bf16-autocast-cuda"
        (runner / "training" / "training.json").write_text(json.dumps(record))
        pins = json.loads((runner / "training" / "files_sha256.json").read_text())
        pins["training.json"] = h2_ce_confirmation.sha_file(runner / "training" / "training.json")
        (runner / "training" / "files_sha256.json").write_text(json.dumps(pins))
        evidence = {
            **IDENTITY,
            "status": "COMPLETE_CE_CAPTURE",
            "phase": phase,
            "initial_identity": initial,
            "kernels": {"family_events": {}},
            "patch_after": {"patch_calls": {"class": 0, "instance": 0}},
        }
        (runner / "evidence.json").write_text(json.dumps(evidence))
        (runner / "contract.json").write_text(json.dumps(contract))
    if tamper:
        tamper(root)
    return root, work


@pytest.fixture
def captured(monkeypatch):
    monkeypatch.setattr(updates, "classify_child", lambda root, phase: {"outcome": "CAPTURED"})


def review(root, work):
    return updates.review_updates(root, work=work, expected_initial="init", check_sources=False)


def test_two_clean_arms_pass_and_differences_are_only_measured(tmp_path, captured):
    root, work = build(tmp_path)
    result = review(root, work)
    assert {k: v["outcome"] for k, v in result["arms"].items()} == {
        "A": "UPDATES_PASS",
        "L": "UPDATES_PASS",
    }
    assert result["arms"]["A"]["optimizer_updates"] == 4
    assert result["arms"]["A"]["microbatches"] == 16
    pair = result["pair_measurements"]
    assert pair["gated"] is False and pair["identical_final_adapters"] is True
    assert pair["max_abs_loss_difference"] == 0
    assert result["performance_verdict"] == "NO_VERDICT"


def test_wrong_initial_state_fails_its_arm(tmp_path, captured):
    root, work = build(tmp_path, initial="other")
    result = review(root, work)
    assert result["arms"]["A"]["outcome"] == "UPDATES_FAIL"
    assert "pair_measurements" not in result


def test_tampered_training_evidence_fails_that_arm_only(tmp_path, captured):
    def tamper(root):
        path = root / "ce_only_training/runner/training/adapter_02.safetensors"
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(bytes(data))

    root, work = build(tmp_path, tamper=tamper)
    result = review(root, work)
    assert result["arms"]["A"]["outcome"] == "UPDATES_PASS"
    assert result["arms"]["L"]["outcome"] == "UPDATES_FAIL"
    assert result["arms"]["L"]["row"] in ("U1", "U2", "U3", "U4", "U?")


def test_child_identity_or_contract_mismatch_fails(tmp_path, captured):
    def tamper(root):
        path = root / "baseline_training/runner/evidence.json"
        data = json.loads(path.read_text())
        data["gpu_uuid"] = "GPU-other"
        path.write_text(json.dumps(data))

    root, work = build(tmp_path, tamper=tamper)
    assert review(root, work)["arms"]["A"]["outcome"] == "UPDATES_FAIL"


def test_child_contract_must_equal_the_series_contract(tmp_path, captured):
    def tamper(root):
        (root / "ce_only_training/runner/contract.json").write_text(json.dumps({"other": 1}))

    root, work = build(tmp_path, tamper=tamper)
    result = review(root, work)
    assert result["arms"]["L"]["outcome"] == "UPDATES_FAIL"
    assert result["arms"]["A"]["outcome"] == "UPDATES_PASS"


def test_pair_with_different_optimizer_is_a_u4_failure(tmp_path, captured):
    def tamper(root):
        runner = root / "ce_only_training/runner/training"
        record = json.loads((runner / "training.json").read_text())
        record["optimizer"]["groups"][0]["names"] = list(record["optimizer"]["groups"][0]["names"])
        record["optimizer"]["max_grad_norm_note"] = "differs"
        (runner / "training.json").write_text(json.dumps(record))
        pins = json.loads((runner / "files_sha256.json").read_text())
        pins["training.json"] = h2_ce_confirmation.sha_file(runner / "training.json")
        (runner / "files_sha256.json").write_text(json.dumps(pins))

    root, work = build(tmp_path, tamper=tamper)
    result = review(root, work)
    assert {v["outcome"] for v in result["arms"].values()} == {"UPDATES_FAIL"}
    assert {v["row"] for v in result["arms"].values()} == {"U4"}


def test_sources_are_checked_against_the_contract_by_default(tmp_path):
    root, work = build(tmp_path)
    with pytest.raises(ValueError, match="source differs"):
        updates.review_updates(root, work=work, expected_initial="init")


# --------------------------------------------------------------------------- child classification


def child(tmp_path, evidence=None, controls=None):
    runner = tmp_path / "s" / "baseline_training" / "runner"
    runner.mkdir(parents=True)
    if evidence is not None:
        (runner / "evidence.json").write_text(json.dumps(evidence))
    if controls is not None:
        (runner / "controls.json").write_text(json.dumps(controls))
    return tmp_path / "s"


def test_missing_evidence_is_incomplete(tmp_path):
    root = child(tmp_path)
    assert updates.classify_child(root, "baseline_training")["outcome"] == "INCOMPLETE"


def test_bad_controls_are_void_before_anything_else(tmp_path):
    root = child(tmp_path, {"error": "ValueError: No adapter changed"}, {"phase": "x"})
    state = updates.classify_child(root, "baseline_training")
    assert state["outcome"] == "VOID"


def test_errors_map_to_fail_or_incomplete(tmp_path, monkeypatch):
    import h2_ce_controls

    monkeypatch.setattr(h2_ce_controls, "validate_controls", lambda data, phase: None)
    root = child(tmp_path, {"error": "TimeoutError: deadline"}, {})
    assert updates.classify_child(root, "baseline_training")["outcome"] == "INCOMPLETE"
    shutil.rmtree(tmp_path / "s")
    root = child(tmp_path, {"error": "ValueError: No adapter changed"}, {})
    state = updates.classify_child(root, "baseline_training")
    assert state == {
        "outcome": "UPDATES_FAIL",
        "row": "U2",
        "reason": "ValueError: No adapter changed",
    }


def test_row_mapping_covers_the_reader_messages():
    assert updates._row("Invalid accumulation normalization denominator") == "U1"
    assert updates._row("Frozen state changed") == "U2"
    assert updates._row("Nonfinite optimizer moments") == "U2"
    assert updates._row("Executed kernel attribution mismatch: rope") == "U5"
    assert updates._row("something else") == "U?"


@pytest.mark.parametrize("failure", [None, "child", "review_raises"])
def test_updates_series_runs_two_fresh_children_then_reviews(tmp_path, monkeypatch, failure):
    import copy

    from test_h2_ce_confirmation import authority

    data = authority(tmp_path)
    data["series"] = "updates"
    data["deadline_unix_s"] = data["series_started_unix_s"] + 2400
    monkeypatch.setattr(h2_ce_confirmation, "validate_contract", lambda value: {})
    launched = []

    def bounded(argv, out, **kwargs):
        phase = argv[argv.index("--phase") + 1]
        launched.append(phase)
        target = Path(argv[argv.index("--out") + 1])
        target.mkdir(parents=True)
        (target / "evidence.json").write_text(json.dumps({"status": "COMPLETE_CE_CAPTURE"}))
        code = 2 if failure == "child" and len(launched) == 1 else 0
        return {
            "status": "CHILD_EXITED",
            "group_cleanup_confirmed": True,
            "child_status": {"returncode": code},
        }

    def fake_review(root, **kwargs):
        if failure == "review_raises":
            raise ValueError("Offline reviewer/source differs from captured authority")
        stopped = failure == "child"
        return {
            "arms": {
                "A": {"outcome": "UPDATES_FAIL" if stopped else "UPDATES_PASS"},
                "L": {"outcome": "INCOMPLETE" if stopped else "UPDATES_PASS"},
            }
        }

    monkeypatch.setattr(h2_ce_confirmation, "bounded_process", bounded)
    monkeypatch.setattr(updates, "review_updates", fake_review)
    result = h2_ce_confirmation.run_series(copy.deepcopy(data), Path(data["evidence_root"]))
    root = Path(data["evidence_root"])
    if failure is None:
        assert launched == ["baseline_training", "ce_only_training"]
        assert result["status"] == "COMPLETE_H2_UPDATES_SERIES"
        assert result["arms"] == {"A": "UPDATES_PASS", "L": "UPDATES_PASS"}
        assert (root / "updates_review.json").exists()
    elif failure == "child":
        assert launched == ["baseline_training"]
        assert result["status"] == "STOPPED_H2_UPDATES_SERIES"
        assert result["arms"] == {"A": "UPDATES_FAIL", "L": "INCOMPLETE"}
    else:
        assert result["status"] == "VOID" and "source differs" in result["error"]
    assert json.loads((root / "series.json").read_text())["status"] == result["status"]
