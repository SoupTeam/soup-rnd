"""Clock evidence cannot accept missing, nonfinite or unstable samples."""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def auditor():
    folder = Path(__file__).parents[1] / "benchmarks/harness"
    sys.path.insert(0, str(folder))
    try:
        spec = importlib.util.spec_from_file_location("timing_audit", folder / "d2_audit_timing.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(folder))


@pytest.mark.parametrize(
    "clocks",
    [
        [],
        [1800],
        [1800, 1800],
        [1800, None, 1800],
        [1800, float("nan"), 1800],
        [1800, float("inf"), 1800],
        [0, 0, 0],
        [1000, 1060, 1000],
    ],
)
def test_reject_incomplete_or_unstable_clock_evidence(auditor, clocks):
    assert not auditor.clock_verdict([{"clock_mhz": value} for value in clocks])["passed"]


def test_accept_stable_observed_clocks(auditor):
    result = auditor.clock_verdict([{"clock_mhz": value} for value in (1800, 1815, 1830)])
    assert result["passed"] and result["samples"] == 3
    assert result["spread"] == pytest.approx(1830 / 1800 - 1)
