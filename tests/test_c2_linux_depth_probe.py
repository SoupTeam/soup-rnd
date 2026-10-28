"""Reject ambiguous physical mappings and contaminated Linux C2 arms."""
import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def probe():
    harness = Path(__file__).parents[1] / "benchmarks" / "harness"
    spec = importlib.util.spec_from_file_location(
        "c2_linux_depth_probe", harness / "c2_linux_depth_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    old_path = sys.path[:]
    sys.path.insert(0, str(harness))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


def test_mapping_requires_unique_matching_physical_device(probe):
    before = {"nvme0n1": {"read": 100, "write": 0},
              "nvme1n1": {"read": 100, "write": 0}}
    after = {"nvme0n1": {"read": 1100, "write": 0},
             "nvme1n1": {"read": 100, "write": 0}}
    assert probe.identify_device(before, after, 1000) == "nvme0n1"
    for increments in [(500, 500), (1000, 1000), (1000, 150), (0, 0), (1200, 0)]:
        changed = {name: dict(after[name], read=100 + amount)
                   for name, amount in zip(before, increments)}
        with pytest.raises(ValueError, match="unique physical NVMe"):
            probe.identify_device(before, changed, 1000)


@pytest.mark.parametrize("delta,seconds,expected", [
    ({"read": 1000, "write": 0}, 2.0, []),
    ({"read": 0, "write": 0}, 2.0, ["physical read bytes outside 80-120%"]),
    ({"read": 1300, "write": 0}, 2.0, ["physical read bytes outside 80-120%"]),
    ({"read": 1000, "write": 2**23}, 2.0, ["foreign writes >1 MiB/s"]),
])
def test_counter_rule_cannot_turn_zero_or_contaminated_reads_into_pass(
    probe, delta, seconds, expected
):
    assert probe.arm_reasons(delta, seconds, 1000) == expected


def test_missing_or_negative_counters_refused(probe):
    for delta in ({}, {"read": -1, "write": 0}, {"read": 1000, "write": -1}):
        assert probe.arm_reasons(delta, 2.0, 1000)


def test_temperature_reads_hwmon_class_even_without_nvme_class_mount(tmp_path, probe):
    sensor = tmp_path / "hwmon1"
    sensor.mkdir()
    (sensor / "name").write_text("nvme\n")
    (sensor / "temp1_input").write_text("42000\n")
    assert probe.temperature_readings(tmp_path)[str(sensor)] == {
        "name": "nvme", "millidegrees_c": {"temp1_input": 42000}
    }
    (sensor / "temp1_input").write_text("unavailable")
    assert "error" in probe.temperature_readings(tmp_path)[str(sensor)]


def test_diagnostic_write_allowance_is_bounded_and_default_remains_strict(probe):
    amount = 10 * 2**30
    burst = {"read": amount, "write": 13 * 2**20}
    assert probe.arm_reasons(burst, 3, amount) == ["foreign writes >1 MiB/s"]
    assert probe.arm_reasons(burst, 3, amount, True) == []
    assert probe.arm_reasons(dict(burst, write=30 * 2**20), 3, amount, True)
    assert probe.arm_reasons(burst, 1, amount, True)  # rate limit also applies


@pytest.mark.parametrize("diagnostic,read_bytes,passes", [
    (False, 0, False), (True, 0, True), (True, 13 * 2**20, False),
])
def test_diagnostic_idle_still_rejects_foreign_reads(
    probe, monkeypatch, diagnostic, read_bytes, passes
):
    values = iter([{"nvme0n1": {"read": 0, "write": 0}},
                   {"nvme0n1": {"read": read_bytes, "write": 13 * 2**20}}])
    stamps = iter([0, 5])
    monkeypatch.setattr(probe, "counters", lambda: next(values))
    monkeypatch.setattr(probe.time, "sleep", lambda _: None)
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(stamps))
    if passes:
        assert probe.quiet(diagnostic)["nvme0n1"]["write"] > 2**20
    else:
        with pytest.raises(RuntimeError, match="not idle"):
            probe.quiet(diagnostic)
