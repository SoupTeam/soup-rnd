"""CPU tests for the readiness decision of A5 ticket 06 (benchmarks/harness)."""

from __future__ import annotations

import json
import subprocess
import sys

from tests.test_l2l_probe_harness import HARNESS, _load


def _evidence(**over):
    ev = {
        "r1": [{"source": "a", "pass": True}, {"source": "b", "pass": True}],
        "r2": [{"source": "c", "pass": True}],
        "r3": [{"source": "a", "l2l": 100, "ga": 110}, {"source": "b", "l2l": 100, "ga": 100}],
    }
    ev.update(over)
    return ev


def test_all_criteria_met_with_open_limits_is_ready_with_limits():
    dec = _load("l2l_ticket06_decision")
    out = dec.decide(_evidence())
    assert out["verdict"] == "готов с ограничениями"
    assert out["criteria"] == {"R1": True, "R2": True, "R3": True}
    assert out["limits"]


def test_all_criteria_met_and_no_limits_is_ready():
    dec = _load("l2l_ticket06_decision")
    assert dec.decide(_evidence(), limits=[])["verdict"] == "готов"


def test_one_failed_criterion_is_not_ready_and_names_it():
    dec = _load("l2l_ticket06_decision")
    out = dec.decide(_evidence(r1=[{"source": "a", "pass": True}, {"source": "b", "pass": False}]))
    assert out["verdict"] == "не готов" and out["criteria"]["R1"] is False
    out = dec.decide(_evidence(r3=[{"source": "a", "l2l": 120, "ga": 110}]))
    assert out["verdict"] == "не готов" and out["criteria"]["R3"] is False


def test_a_criterion_without_records_fails_instead_of_passing():
    dec = _load("l2l_ticket06_decision")
    out = dec.decide(_evidence(r2=[]))
    assert out["verdict"] == "не готов" and out["criteria"]["R2"] is False


def test_command_reads_the_committed_raw_records():
    done = subprocess.run(
        [sys.executable, str(HARNESS / "l2l_ticket06_decision.py")],
        capture_output=True, text=True, check=False,
    )
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout)
    assert out["verdict"] == "готов с ограничениями"
    assert out["criteria"] == {"R1": True, "R2": True, "R3": True}
    assert len(out["evidence"]["r1"]) == 3 and len(out["evidence"]["r2"]) == 4
