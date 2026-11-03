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
