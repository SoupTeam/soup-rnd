"""Timed-step disk counters must enclose each step without missing samples."""
import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def probe():
    folder = Path(__file__).parents[1] / "benchmarks/harness"
    spec = importlib.util.spec_from_file_location("c2_step_probe", folder / "c2_step_probe.py")
    module = importlib.util.module_from_spec(spec)
    old_path = sys.path[:]
    sys.path.insert(0, str(folder))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


def test_step_uses_enclosing_counter_samples(probe):
    samples = [(index / 4, {"read": index * 100, "write": 0}) for index in range(9)]
    delta, duration = probe.io_window(samples, 0.3, 0.9)
    assert delta == {"read": 300, "write": 0}
    assert duration == 0.75


@pytest.mark.parametrize("start,end", [(-0.1, 0.9), (0.3, 3.0)])
def test_missing_edge_sample_is_not_assumed_zero(probe, start, end):
    samples = [(index / 4, {"read": index * 100, "write": 0}) for index in range(9)]
    with pytest.raises(ValueError, match="enclose"):
        probe.io_window(samples, start, end)


def test_suspended_or_sparse_counter_window_is_invalid(probe):
    samples = [(stamp, {"read": 100, "write": 0}) for stamp in [0, 0.25, 2.75, 3.0]]
    with pytest.raises(ValueError, match="missing counter samples or suspend"):
        probe.io_window(samples, 0.1, 2.9)
