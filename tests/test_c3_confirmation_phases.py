"""The exact prospective phase plan must be executable without GPU or rental."""

import hashlib
import os
import time

import pytest

from benchmarks.c3.confirmation_phases import PHASES, supervise_phase, validate_plan


@pytest.mark.skipif(os.name != "posix", reason="requires real POSIX process groups")
def test_whole_phase_plan_runs_through_real_supervisor(tmp_path):
    # This would have caught attempt018's 5400s-versus-1800s rejection before payment.
    validate_plan(time.time() + 7200)
    script = tmp_path / "phase.sh"
    script.write_text("exit 0\n")
    sha = hashlib.sha256(script.read_bytes()).hexdigest()
    for name in PHASES:
        result = supervise_phase(name, script, sha, tmp_path / name, time.time() + 10)
        assert result["status"] == "PASS"
        assert result["group_cleanup_confirmed"]


def test_expired_plan_is_not_admitted():
    with pytest.raises(ValueError):
        validate_plan(time.time() - 1)
