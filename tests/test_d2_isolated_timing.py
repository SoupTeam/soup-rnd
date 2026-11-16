"""Ownership evidence must reject interference and incomplete lifecycles."""

import copy
import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def timing():
    folder = Path(__file__).parents[1] / "benchmarks/harness"
    sys.path.insert(0, str(folder))
    try:
        spec = importlib.util.spec_from_file_location(
            "isolated_timing", folder / "d2_isolated_timing.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(folder))


def sample(processes):
    return {
        "available": True,
        "processes": processes,
        "graphics": [],
        "clock_mhz": 1800,
        "power_mw": 250000,
        "pstate": 2,
        "throttle": 0,
    }


def test_ownership_requires_observed_birth_and_death(timing):
    result = timing.lifecycle_verdict([sample([])], [sample([99999])], [sample([])], 0)
    assert result["status"] == "LIFECYCLE_VALID"
    assert result["nvml_pid"] == 99999


@pytest.mark.parametrize(
    "fault",
    [
        "before",
        "after",
        "two",
        "changed",
        "missing",
        "clock",
        "power",
        "graphics",
        "throttle",
        "exit",
        "no_live",
    ],
)
def test_ownership_fails_closed(timing, fault):
    before, live, after, code = [sample([])], [sample([99999])], [sample([])], 0
    if fault == "before":
        before = [sample([99999])]
    elif fault == "after":
        after = [sample([99999])]
    elif fault == "two":
        live = [sample([99999, 88888])]
    elif fault == "changed":
        live.append(sample([88888]))
    elif fault == "missing":
        live[0]["available"] = False
    elif fault in ("clock", "power"):
        live[0].pop("clock_mhz" if fault == "clock" else "power_mw")
    elif fault == "graphics":
        live[0]["graphics"] = [88888]
    elif fault == "throttle":
        live[0]["throttle"] = 0x40
    elif fault == "exit":
        code = 1
    else:
        live = []
    original = copy.deepcopy((before, live, after))
    assert timing.lifecycle_verdict(before, live, after, code)["status"] == "VOID"
    assert (before, live, after) == original
