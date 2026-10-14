"""CPU-only regression tests for E2 ABBA precision CLI rules."""

import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import strip_ansi

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "benchmarks/e2/run_e2_abba_verified.py"


def run_abba(*args):
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("args", "expected_error"),
    [
        (
            ("--precision", "bf16", "--amp-init-scale", "1024"),
            "BF16 must not use --amp-init-scale",
        ),
        (
            ("--precision", "fp16", "--pilot", "--execute"),
            "FP16 execution requires --amp-init-scale",
        ),
        (
            ("--execute",),
            "Final execution requires --precision",
        ),
    ],
)
def test_invalid_precision_combinations(args, expected_error):
    result = run_abba(*args)

    assert result.returncode != 0
    assert expected_error in result.stderr


def test_bf16_final_dry_run():
    result = run_abba(
        "--baseline", "experiments/e2_mistral7b_abba_baseline.yaml",
        "--cached", "experiments/e2_mistral7b_abba_cached.yaml",
        "--cycles", "3",
        "--warmup", "8",
        "--precision", "bf16",
        "--decision-rule-file",
        "benchmarks/e2/performance_decision_rule.md",
    )

    assert result.returncode == 0, result.stderr
    stdout = strip_ansi(result.stdout)
    assert "4000 optimizer steps" in stdout
    assert "A-B-B-A-A-B-B-A-A-B-B-A" in result.stdout
    assert "DRY RUN ONLY" in result.stdout


def test_fp16_dry_run_remains_supported():
    result = run_abba(
        "--precision", "fp16",
        "--amp-init-scale", "1024",
    )

    assert result.returncode == 0, result.stderr
    assert "DRY RUN ONLY" in result.stdout


def test_original_pilot_dry_run_remains_supported():
    result = run_abba(
        "--cycles", "1",
        "--warmup", "2",
        "--pilot",
    )

    assert result.returncode == 0, result.stderr
    assert "A-B-B-A" in result.stdout
    assert "DRY RUN ONLY" in result.stdout
