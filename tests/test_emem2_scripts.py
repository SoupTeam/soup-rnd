"""The phase-2 power/clock gate, without a GPU: the verdict is pure once the samples exist."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_GATE = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/mbr-moonlight-emem2-20261124/gate.py"
)


def load_gate() -> ModuleType:
    spec = importlib.util.spec_from_file_location("emem2_gate", _GATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate() -> ModuleType:
    return load_gate()


def test_healthy_card_passes(gate: ModuleType) -> None:
    assert gate.assess("1", [1740, 1950, 1912], 80.0) == []


def test_a_power_cap_at_a_healthy_clock_is_not_a_failure(gate: ModuleType) -> None:
    """The 80 W TGP cap can be active while the clock stays at 1912 MHz (the phase-2 pass saw it);
    the failure this gate guards against is the collapsed 15 W limit, not the normal cap."""
    assert gate.assess("1", [1912], 80.0) == []


def test_battery_fails(gate: ModuleType) -> None:
    problems = gate.assess("0", [1912], 80.0)
    assert any("AC not online" in problem for problem in problems)


def test_a_collapsed_clock_fails(gate: ModuleType) -> None:
    problems = gate.assess("1", [1740, 1330], 80.0)
    assert any("1500 MHz" in problem for problem in problems)


def test_a_collapsed_power_limit_fails(gate: ModuleType) -> None:
    problems = gate.assess("1", [1912], 15.0)
    assert any("15.0 W" in problem for problem in problems)


def test_no_samples_fails(gate: ModuleType) -> None:
    problems = gate.assess("1", [], 80.0)
    assert any("no clock samples" in problem for problem in problems)
