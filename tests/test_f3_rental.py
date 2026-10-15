"""Tests for the F3 rental entry point (benchmarks/harness/f3_rental.py, F3 ticket 08).

Fakes only: no network, no GPU, no vast.ai.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"


@pytest.fixture(scope="module")
def rental():
    sys.path.insert(0, str(HARNESS))
    name = "_f3_rental_test_module"
    spec = importlib.util.spec_from_file_location(name, HARNESS / "f3_rental.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
        sys.path.remove(str(HARNESS))
    return module


def test_arm_command_uses_batch_16_and_the_venv_python(rental):
    command = rental.arm_command("unsloth", Path("/tmp/out.json"), steps=20)
    assert command[0].endswith("f3-unsloth/bin/python")
    assert command[command.index("--batch") + 1] == "16"
    assert command[command.index("--model") + 1] == "Qwen/Qwen2.5-0.5B"
    assert command[command.index("--steps") + 1] == "20"
    assert "--control" not in command


def test_c1_specs_audit_unsloth_and_scope_out_chronicals(rental):
    specs = rental.c1_specs()
    assert [s.arm for s in specs] == ["A-unsloth", "B-chronicals"]
    assert specs[0].uses_trainer is True
    assert specs[1].uses_trainer is False
    assert "custom loop" in specs[1].scope_reason


def test_sleep_observation_records_which_checks_answer(rental):
    class Watched:
        record = {
            "outcome": "valid",
            "checks": {
                "suspend": {"outcome": "valid", "reason": "", "evidence": {}},
                "power": {"outcome": "unknown", "reason": "no battery", "evidence": {}},
                "foreign_reads": {"outcome": "unknown", "reason": "no device", "evidence": {}},
            },
        }

    class Watch:
        def __enter__(self):
            return Watched()

        def __exit__(self, *exc):
            return None

    slept = []
    result = rental.observe_sleep(lambda **kw: Watch(), slept.append, seconds=60)
    assert slept == [60]
    assert result["answering"] == ["suspend"]
    assert result["not_answering"] == ["power", "foreign_reads"]
    assert result["label"] == "A100-PCIE-80GB vast.ai, real"


def test_sleep_observation_without_a_record_answers_nothing(rental):
    class Watch:
        def __enter__(self):
            return type("W", (), {"record": None})()

        def __exit__(self, *exc):
            return None

    result = rental.observe_sleep(lambda **kw: Watch(), lambda s: None)
    assert result["answering"] == []
    assert result["outcome"] == "unknown"
