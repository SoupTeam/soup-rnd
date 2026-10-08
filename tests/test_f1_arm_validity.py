"""Tests for the arm validity watch (benchmarks/harness/arm_validity.py, F1 ticket 02).

The module is benchmark harness code, not shipped. Every sensor reading below is
synthetic: the fake sensors return made-up numbers, and no test writes into
benchmarks/results/.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"


@pytest.fixture(scope="module")
def validity():
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    name = "_arm_validity_test_module"
    spec = importlib.util.spec_from_file_location(name, HARNESS / "arm_validity.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _readings(*values):
    """A synthetic sensor that returns ``values`` in order, then repeats the last one."""
    queue = list(values)

    def sensor():
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return sensor


def _watch(validity, **kwargs):
    kwargs.setdefault("sleep_offset", _readings(25_588.0, 25_588.0))
    kwargs.setdefault("suspend_count", _readings(7, 7))
    with validity.ArmWatch(arm="A", round=2, run=5, label="synthetic", **kwargs) as watch:
        pass
    return watch.record


def test_clean_arm_is_ok_with_identity_and_suspend_check(validity):
    record = _watch(validity)

    assert record["arm"] == "A"
    assert record["round"] == 2
    assert record["run"] == 5
    assert record["label"] == "synthetic"
    suspend = record["checks"]["suspend"]
    assert suspend["outcome"] == "ok"
    assert suspend["reason"]
    assert suspend["evidence"]["sleep_offset_start_s"] == 25_588.0
    assert suspend["evidence"]["sleep_offset_end_s"] == 25_588.0
    assert record["outcome"] == "ok"
    json.dumps(record)


def test_boot_time_offset_growth_over_one_second_voids_the_arm(validity):
    record = _watch(validity, sleep_offset=_readings(25_588.0, 25_650.5))

    suspend = record["checks"]["suspend"]
    assert suspend["outcome"] == "void"
    assert "62.5" in suspend["reason"]
    assert suspend["evidence"]["sleep_offset_start_s"] == 25_588.0
    assert suspend["evidence"]["sleep_offset_end_s"] == 25_650.5
    assert record["outcome"] == "void"


def test_boot_time_offset_jitter_under_one_second_stays_ok(validity):
    record = _watch(validity, sleep_offset=_readings(25_588.0, 25_588.9))

    assert record["checks"]["suspend"]["outcome"] == "ok"


def test_suspend_counter_increase_voids_the_arm(validity):
    record = _watch(validity, suspend_count=_readings(7, 8))

    suspend = record["checks"]["suspend"]
    assert suspend["outcome"] == "void"
    assert "suspend count" in suspend["reason"]
    assert suspend["evidence"]["suspend_count_start"] == 7
    assert suspend["evidence"]["suspend_count_end"] == 8
    assert record["outcome"] == "void"


def _raises():
    raise PermissionError("synthetic: permission denied")


@pytest.mark.parametrize(
    ("sensors", "reason_part"),
    [
        ({"sleep_offset": _raises}, "PermissionError"),
        ({"sleep_offset": _readings(None)}, "returned None"),
        ({"suspend_count": _raises}, "PermissionError"),
        ({"suspend_count": _readings(7, None)}, "returned None"),
    ],
    ids=["offset-raises", "offset-none", "count-raises", "count-none-at-end"],
)
def test_silent_sensor_makes_suspend_unknown_with_reason(validity, sensors, reason_part):
    record = _watch(validity, **sensors)

    suspend = record["checks"]["suspend"]
    assert suspend["outcome"] == "unknown"
    assert reason_part in suspend["reason"]
    assert record["outcome"] == "unknown"


def test_sleep_seen_by_one_sensor_is_void_even_when_the_other_is_silent(validity):
    record = _watch(validity, sleep_offset=_readings(25_588.0, 25_650.5), suspend_count=_raises)

    assert record["checks"]["suspend"]["outcome"] == "void"
    assert record["outcome"] == "void"


def test_unknown_check_that_is_not_required_leaves_the_arm_ok(validity):
    record = _watch(validity, suspend_count=_raises, required=())

    assert record["checks"]["suspend"]["outcome"] == "unknown"
    assert record["required"] == []
    assert record["outcome"] == "ok"


def test_void_check_voids_the_arm_even_when_not_required(validity):
    record = _watch(validity, suspend_count=_readings(7, 8), required=())

    assert record["outcome"] == "void"


def test_every_check_is_required_by_default(validity):
    record = _watch(validity)

    assert record["required"] == ["suspend"]


def test_unknown_required_check_name_is_rejected(validity):
    with pytest.raises(ValueError, match="powr"):
        validity.ArmWatch(arm="A", round=0, run=0, required=("suspend", "powr"))


def test_wall_clock_jump_alone_does_not_void_on_linux(validity, monkeypatch):
    wall = _readings(1_000.0, 4_600.0)
    monkeypatch.setattr(validity.time, "time", wall)

    record = _watch(validity)

    assert record["checks"]["suspend"]["outcome"] == "ok"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux sensors only")
def test_real_linux_sensors_return_values(validity):
    if not Path(validity.SUSPEND_SUCCESS_PATH).exists():
        pytest.skip(f"this kernel has no {validity.SUSPEND_SUCCESS_PATH}")
    with validity.ArmWatch(arm="A", round=0, run=0, label="real sensors, test only") as watch:
        pass

    evidence = watch.record["checks"]["suspend"]["evidence"]
    assert isinstance(evidence["sleep_offset_start_s"], float)
    assert isinstance(evidence["sleep_offset_end_s"], float)
    assert isinstance(evidence["suspend_count_start"], int)
    assert isinstance(evidence["suspend_count_end"], int)
