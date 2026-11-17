"""The committed C3 evidence packets replay to their recorded outcomes, and tampering fails."""

import json
import shutil
from pathlib import Path

import pytest

from benchmarks.c3.replay_evidence import replay

ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "c3" / "evidence"


def test_attempt017_replays_numerical_pass_without_speed_verdict():
    report = replay(ROOT / "20261031_attempt017")
    assert report["failed_checks"] == []
    assert report["status"] == "NUMERICAL_PASS_GATES_PASS"
    assert report["performance_verdict"] == "NO_VERDICT_INSUFFICIENT_REPETITIONS"
    assert report["completed_arms"] == 20 and report["cuda_controls_passed"] == 5
    assert report["repetitions"]["blocks_to_add"] == 15
    assert report["descriptive_process_read_reduction_pct"] == pytest.approx(25.0)


def test_attempt019_replays_recorded_foreign_read_stop():
    report = replay(ROOT / "20261031_attempt019")
    assert report["failed_checks"] == []
    assert report["status"] == "BLOCKED_FOREIGN_READS"
    assert report["completed_arms"] == 22 and report["cuda_controls_passed"] == 6
    assert [row["index"] for row in report["f1_failures"]] == [14, 19]
    assert report["partial_arms"] == [{"index": 22, "arm": "B", "status": "RUNNING", "records": 0}]
    assert report["numerical_completed_arms"]["status"] == "NUMERICAL_PASS"


def test_modified_member_is_detected(tmp_path):
    packet = tmp_path / "packet"
    shutil.copytree(ROOT / "20261031_attempt017", packet)
    arm = packet / "evidence/comparison/arm_01_B/result.json"
    data = json.loads(arm.read_text())
    data["records"][0]["updates"][0]["loss"] += 1e-6
    arm.write_text(json.dumps(data))
    report = replay(packet)
    assert "member:comparison/arm_01_B/result.json" in report["failed_checks"]
    assert report["status"] == "REPLAY_CHECKS_FAILED"


def test_replay_uses_measured_gates_when_live_code_changes(monkeypatch):
    from benchmarks.c3 import foreign_read_control, io_calibration
    from benchmarks.harness import arm_validity

    def changed(*args, **kwargs):
        pytest.fail("historical replay used the live gate")

    monkeypatch.setattr(io_calibration, "assess", changed)
    monkeypatch.setattr(foreign_read_control, "assess", changed)
    monkeypatch.setattr(arm_validity, "arm_outcome", changed)
    monkeypatch.setattr(arm_validity, "blocks_to_add", changed)
    report = replay(ROOT / "20261031_attempt017")
    assert report["failed_checks"] == []
    assert report["repetitions"]["blocks_to_add"] == 15
    assert all(value == "frozen measured copy from the packet"
               for value in report["gate_code_provenance"].values())


@pytest.mark.parametrize("filename", [
    "io_calibration.py.txt", "foreign_read_control.py.txt", "arm_validity.py.txt",
    "isolated_compare.py",
])
def test_changed_frozen_gate_is_rejected_before_execution(tmp_path, filename):
    packet = tmp_path / "packet"
    shutil.copytree(ROOT / "20261031_attempt017", packet)
    path = packet / "frozen_gate" / filename
    path.write_bytes(path.read_bytes() + b'\nraise AssertionError("unverified gate executed")\n')
    with pytest.raises(ValueError, match="frozen gate source hash or size mismatch"):
        replay(packet)
