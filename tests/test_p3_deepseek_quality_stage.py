"""Exercise real child processes; no models, provider credentials or rental calls."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name != "posix", reason="pilot process groups require Linux")


def test_stage_preserves_failure_output_and_cannot_overwrite_evidence(tmp_path):
    from benchmarks.harness.p3_deepseek_quality_stage import run_stage

    output = tmp_path / "stage"
    result = run_stage(
        [sys.executable, "-c", "import os; os.write(1,b'partial-evidence'); raise SystemExit(7)"],
        output=output, seconds=10,
    )
    assert result["status"] == "FAILED"
    assert result["returncode"] == 7
    assert (output / "stdout.log").read_bytes() == b"partial-evidence"
    assert json.loads((output / "result.json").read_text())["returncode"] == 7
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_stage([sys.executable, "-c", "raise SystemExit(0)"], output=output, seconds=10)
    assert (output / "manifest.json").read_bytes() == before


def test_timeout_kills_term_ignoring_child_and_keeps_output(tmp_path):
    from benchmarks.harness.p3_deepseek_quality_stage import run_stage

    code = (
        "import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "os.write(1,b'entered'); time.sleep(60)"
    )
    result = run_stage([sys.executable, "-c", code], output=tmp_path / "stage", seconds=2)
    assert result["status"] == "TIMED_OUT"
    assert result["returncode"] == -signal.SIGKILL
    assert result["wall_seconds"] < 10
    assert (tmp_path / "stage" / "stdout.log").read_bytes() == b"entered"
    assert result["provider_cleanup_performed"] is False


def test_success_is_not_a_gate_or_budget_admission(tmp_path):
    from benchmarks.harness.p3_deepseek_quality_stage import run_stage

    result = run_stage([sys.executable, "-c", "pass"], output=tmp_path / "stage", seconds=10)
    assert result["status"] == "COMPLETE"
    assert result["returncode"] == 0
    assert result["scientific_admission"] is False
    assert result["provider_cleanup_performed"] is False


def test_cli_timeout_returns_nonzero_without_a_live_descendant(tmp_path):
    pid_file = tmp_path / "descendant.pid"
    descendant = (
        "import os,signal,time; from pathlib import Path;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        f"Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    parent = (
        "import signal,subprocess,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        f"subprocess.Popen([sys.executable,'-c',{descendant!r}]); time.sleep(60)"
    )
    done = subprocess.run([
        sys.executable, "-m", "benchmarks.harness.p3_deepseek_quality_stage",
        "--output", str(tmp_path / "stage"), "--seconds", "3", "--",
        sys.executable, "-c", parent,
    ], capture_output=True, text=True, timeout=15, check=False)
    assert done.returncode == 124, done.stderr
    pid = int(pid_file.read_text())
    state = Path(f"/proc/{pid}/stat")
    assert not state.exists() or state.read_text().split(") ", 1)[1].split()[0] == "Z"
    assert (tmp_path / "stage" / "manifest.json").is_file()


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf"), True])
def test_invalid_deadline_never_creates_a_stage(tmp_path, seconds):
    from benchmarks.harness.p3_deepseek_quality_stage import run_stage

    with pytest.raises(ValueError):
        run_stage([sys.executable, "-c", "pass"], output=tmp_path / "stage", seconds=seconds)
    assert not (tmp_path / "stage").exists()
