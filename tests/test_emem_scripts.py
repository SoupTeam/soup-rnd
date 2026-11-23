"""CPU-only smoke checks for the Ticket 13 Bash driver."""

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BASH = shutil.which("bash")
DRIVER = (
    Path(__file__).resolve().parents[1] / "benchmarks/mbr-moonlight-emem-20261123/run_mem.sh"
)
pytestmark = pytest.mark.skipif(
    BASH is None or sys.platform == "win32", reason="requires Bash on a POSIX host"
)


def _bash(*args: str) -> subprocess.CompletedProcess[str]:
    assert BASH is not None
    return subprocess.run(
        [BASH, *args], capture_output=True, text=True, timeout=10, check=False
    )


@pytest.fixture
def python_call_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "existing results"
    out.mkdir()
    (out / "rows_canonical.jsonl").touch()

    venv = tmp_path / "fake venv"
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)
    marker = tmp_path / "python-called"
    for name in ("python", "python3"):
        executable = bin_dir / name
        executable.write_text('#!/bin/sh\n: > "$PYTHON_CALLED"\nexit 99\n', encoding="utf-8")
        executable.chmod(0o755)

    monkeypatch.delenv("BASH_ENV", raising=False)
    monkeypatch.setenv("OUT", str(out))
    monkeypatch.setenv("VENV", str(venv))
    monkeypatch.setenv("PATH", str(bin_dir), prepend=":")
    monkeypatch.setenv("PYTHON_CALLED", str(marker))
    return marker


def test_run_mem_bash_syntax() -> None:
    result = _bash("-n", str(DRIVER))
    assert result.returncode == 0, result.stderr


def test_run_mem_help(python_call_marker: Path) -> None:
    result = _bash(str(DRIVER), "--help")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        "Usage: [VENV=DIR] [MOONLIGHT_ROOT=DIR] [OUT=DIR] [REV=HEX] run_mem.sh"
    )
    assert not python_call_marker.exists()


def test_run_mem_rejects_unknown_argument(python_call_marker: Path) -> None:
    result = _bash(str(DRIVER), "--unknown")
    assert result.returncode == 2, result.stderr
    assert not python_call_marker.exists()


def test_run_mem_refuses_existing_rows_before_invoking_python(python_call_marker: Path) -> None:
    result = _bash(str(DRIVER))
    assert result.returncode != 0
    assert "rows_canonical.jsonl" in result.stderr
    assert "raw/" in result.stderr
    assert not python_call_marker.exists()


WINDOWS = ("8-s64", "8-s512", "64-s64")


@pytest.fixture
def script_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Fake venv, empty output directory, and a replaceable AC-online file."""
    out = tmp_path / "out"
    venv = tmp_path / "fake venv"
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)
    marker = tmp_path / "python-called"
    for name in ("python", "python3"):
        executable = bin_dir / name
        executable.write_text(
            '#!/bin/sh\n'
            'case "$*" in\n'
            '  *run_mem.py*) : > "$PYTHON_CALLED"; exit "${PYTHON_RUN_CODE:-0}" ;;\n'
            'esac\n'
            'exit 0\n',
            encoding="utf-8",
        )
        executable.chmod(0o755)
    ac_online = tmp_path / "ac-online"
    ac_online.write_text("1\n", encoding="utf-8")

    monkeypatch.delenv("BASH_ENV", raising=False)
    monkeypatch.setenv("OUT", str(out))
    monkeypatch.setenv("VENV", str(venv))
    monkeypatch.setenv("PATH", str(bin_dir), prepend=":")
    monkeypatch.setenv("PYTHON_CALLED", str(marker))
    monkeypatch.setenv("AC_ONLINE", str(ac_online))
    monkeypatch.setenv("INHIBIT_CMD", "")
    return SimpleNamespace(out=out, marker=marker, ac_online=ac_online)


def test_run_mem_refuses_every_cell_when_ac_power_is_offline(
    script_env: SimpleNamespace,
) -> None:
    script_env.ac_online.write_text("0\n", encoding="utf-8")
    result = _bash(str(DRIVER))
    assert result.returncode == 6, result.stderr
    assert result.stderr.count("AC power is not online") == 6
    assert "mem-C-k8-s64" in result.stderr
    assert "mem-E-k64-s64" in result.stderr
    assert not script_env.marker.exists()
    assert list((script_env.out / "raw" / "logs").iterdir()) == []


@pytest.mark.parametrize(("run_code", "expected_exit"), [(0, 0), (3, 6)])
def test_run_mem_runs_every_window_and_counts_failed_cells(
    script_env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
    run_code: int, expected_exit: int,
) -> None:
    monkeypatch.setenv("PYTHON_RUN_CODE", str(run_code))
    result = _bash(str(DRIVER))
    assert result.returncode == expected_exit, result.stderr
    assert result.stdout.count(f"exit={run_code}") == 6
    logs = sorted(path.name for path in (script_env.out / "raw" / "logs").iterdir())
    expected = sorted(f"mem-{arm}-k{window}.log" for window in WINDOWS for arm in ("C", "E"))
    assert logs == expected
    assert script_env.marker.exists()
