"""CPU tests for the readiness decision of A5 ticket 06 (benchmarks/harness)."""

from __future__ import annotations

import json
import subprocess
import sys

from tests.test_l2l_probe_harness import HARNESS, _load


def _arm(**checks):
    outcomes = {"suspend": "ok", "power": "ok", "foreign_reads": "ok", **checks}
    return {
        "source": "a", "arm": "A", "round": 1, "run": 0, "required": list(outcomes),
        "outcome": "ok" if set(outcomes.values()) == {"ok"} else "unknown", "checks": outcomes,
    }


def _evidence(**over):
    ev = {
        "r1": [{"source": "a", "pass": True}, {"source": "b", "pass": True}],
        "r2": [{"source": "c", "pass": True}],
        "r3": [{"source": "a", "l2l": 100, "ga": 110}, {"source": "b", "l2l": 100, "ga": 100}],
        "arms": [_arm()],
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


def test_a_required_check_that_is_unknown_gives_no_verdict_even_when_criteria_pass():
    dec = _load("l2l_ticket06_decision")
    out = dec.decide(_evidence(arms=[_arm(), _arm(foreign_reads="unknown")]))
    assert out["verdict"] == "нет вердикта"
    assert out["criteria"] == {"R1": True, "R2": True, "R3": True}
    assert out["blocked_by"] == {"foreign_reads:unknown": 1}
    assert out["arm_outcomes"] == {"ok": 1, "unknown": 1}


def test_a_void_check_gives_no_verdict_and_a_failed_criterion_does_not_override_it():
    dec = _load("l2l_ticket06_decision")
    arms = [_arm(power="void")]
    out = dec.decide(_evidence(arms=arms, r1=[{"source": "a", "pass": False}]))
    assert out["verdict"] == "нет вердикта" and out["blocked_by"] == {"power:void": 1}


def test_a_required_check_missing_from_the_record_counts_as_unknown():
    dec = _load("l2l_ticket06_decision")
    arm = _arm()
    del arm["checks"]["suspend"]
    assert dec.decide(_evidence(arms=[arm]))["blocked_by"] == {"suspend:unknown": 1}


def test_no_arm_records_in_the_valid_set_gives_no_verdict():
    dec = _load("l2l_ticket06_decision")
    out = dec.decide(_evidence(arms=[]))
    assert out["verdict"] == "нет вердикта" and out["blocked_by"] == {"no arm records": 1}


def test_command_reads_only_the_valid_set_and_the_rerun_gate_records(tmp_path):
    done = subprocess.run(
        [sys.executable, str(HARNESS / "l2l_ticket06_decision.py")],
        capture_output=True, text=True, check=False,
    )
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout)
    # earlier sets have the required foreign_reads check unknown, so the gate does not read them
    assert out["verdict"] == "нет вердикта"
    assert all("rerun-sft" not in item["source"] for item in out["evidence"]["r1"])
    assert len(out["evidence"]["r2"]) == 4
