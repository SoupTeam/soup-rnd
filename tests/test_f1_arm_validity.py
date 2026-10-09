"""Tests for the arm validity watch (benchmarks/harness/arm_validity.py, F1 tickets 02-05).

The module is benchmark harness code, not shipped. Every sensor reading below is
synthetic: the fake sensors return made-up numbers, and no test writes into
benchmarks/results/.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
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


ON_MAINS = {"battery": True, "online": {"ACAD": True, "ucsi-source-psy-USBC000:001": False}}
OFF_MAINS = {"battery": True, "online": {"ACAD": False, "ucsi-source-psy-USBC000:001": False}}
NO_BATTERY = {"battery": False, "online": {"ucsi-source-psy-USBC000:001": False}}


GB = 1_000_000_000


def _processes(own, others=None, uninspectable=0):
    """A synthetic process-reads snapshot: ``others`` maps pid to (name, read bytes)."""
    return {
        "own": own,
        "others": {
            pid: {"name": name, "read_bytes": read} for pid, (name, read) in (others or {}).items()
        },
        "uninspectable": uninspectable,
    }


QUIET_PROCESSES = _processes(own=10 * GB, others={4242: ("firefox", 300_000)}, uninspectable=292)


def _disk(*values):
    """A synthetic disk counter for nvme0n1 that returns ``values`` like ``_readings``."""
    read = _readings(*values)

    def sensor(device):
        assert device == "nvme0n1"
        return read()

    return sensor


def _foreign_sensors(kwargs):
    kwargs.setdefault("model_path", "/synthetic/models/qwen")
    kwargs.setdefault("disk_device", lambda path: "nvme0n1")
    kwargs.setdefault("disk_reads", _disk(50 * GB, 50 * GB))
    kwargs.setdefault("process_reads", _readings(QUIET_PROCESSES))


MIB = 1024 * 1024

PEERS = [{"pid": 3141, "type": "C", "name": "/usr/bin/python3", "used_mib": 412}]


def _box_sensors(kwargs):
    kwargs.setdefault("ram_available", _readings(7_800 * MIB, 6_100 * MIB))
    kwargs.setdefault("gpu_memory_used", _readings(12, 2_900))
    kwargs.setdefault("gpu_processes", _readings([], PEERS))
    kwargs.setdefault("gpu_clock", _readings({"sm_clock_mhz": 1_500, "reasons": []}))


def _watch(validity, **kwargs):
    kwargs.setdefault("sleep_offset", _readings(25_588.0, 25_588.0))
    kwargs.setdefault("suspend_count", _readings(7, 7))
    kwargs.setdefault("power_supplies", _readings(ON_MAINS))
    _foreign_sensors(kwargs)
    _box_sensors(kwargs)
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

    assert record["required"] == ["suspend", "power", "foreign_reads"]


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


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux sensors only")
def test_real_linux_power_sensor_returns_values(validity):
    if not Path(validity.POWER_SUPPLY_DIR).exists():
        pytest.skip(f"this kernel has no {validity.POWER_SUPPLY_DIR}")
    with validity.ArmWatch(arm="A", round=0, run=0, label="real sensors, test only") as watch:
        pass

    power = watch.record["checks"]["power"]["evidence"]
    assert power["samples"] >= 2
    assert power["unknown_samples"] == 0
    reading = power["last_reading"]
    assert isinstance(reading["battery"], bool)
    assert all(isinstance(online, bool) for online in reading["online"].values())


class _CountingSensor:
    """A synthetic sensor like ``_readings`` that also counts its calls."""

    def __init__(self, *values):
        self.calls = 0
        self._next = _readings(*values)

    def __call__(self):
        self.calls += 1
        return self._next()


def _watch_while_sampling(validity, sensor, samples, sampled="power_supplies", **kwargs):
    """Run an arm with ``sensor`` as the ``sampled`` sensor on a 10 ms interval
    until it has been called ``samples`` times."""
    interval = {"power_supplies": "power_interval_s", "gpu_clock": "gpu_clock_interval_s"}
    kwargs[sampled] = sensor
    kwargs[interval[sampled]] = 0.01
    kwargs.setdefault("sleep_offset", _readings(25_588.0))
    kwargs.setdefault("suspend_count", _readings(7))
    kwargs.setdefault("power_supplies", _readings(ON_MAINS))
    _foreign_sensors(kwargs)
    _box_sensors(kwargs)
    with validity.ArmWatch(arm="A", round=2, run=5, label="synthetic", **kwargs) as watch:
        deadline = time.monotonic() + 5.0
        while sensor.calls < samples and time.monotonic() < deadline:
            time.sleep(0.005)
    assert sensor.calls >= samples, f"the {sampled} sampler never ran"
    return watch.record


def test_one_sample_off_mains_voids_the_arm(validity):
    sensor = _CountingSensor(ON_MAINS, ON_MAINS, OFF_MAINS, ON_MAINS)
    record = _watch_while_sampling(validity, sensor, samples=4)

    power = record["checks"]["power"]
    assert power["outcome"] == "void"
    assert "off mains" in power["reason"]
    assert power["evidence"]["samples"] == sensor.calls
    off = power["evidence"]["off_mains_samples"]
    assert len(off) == 1
    assert off[0]["online"] == OFF_MAINS["online"]
    assert record["outcome"] == "void"


def test_all_samples_on_mains_keeps_power_ok(validity):
    sensor = _CountingSensor(ON_MAINS)
    record = _watch_while_sampling(validity, sensor, samples=3)

    power = record["checks"]["power"]
    assert power["outcome"] == "ok"
    assert power["reason"]
    assert power["evidence"]["samples"] == sensor.calls
    assert power["evidence"]["off_mains_samples"] == []
    assert record["outcome"] == "ok"
    json.dumps(record)


def test_no_battery_keeps_power_ok_with_that_reason(validity):
    record = _watch(validity, power_supplies=_readings(NO_BATTERY))

    power = record["checks"]["power"]
    assert power["outcome"] == "ok"
    assert "no battery" in power["reason"]
    assert power["evidence"]["off_mains_samples"] == []
    assert record["outcome"] == "ok"


@pytest.mark.parametrize(
    ("sensor", "reason_part"),
    [
        (_raises, "PermissionError"),
        (_readings(None), "returned None"),
        (_readings(ON_MAINS, None), "returned None"),
    ],
    ids=["raises", "none", "none-in-one-sample"],
)
def test_silent_power_sensor_makes_power_unknown_with_reason(validity, sensor, reason_part):
    record = _watch(validity, power_supplies=sensor)

    power = record["checks"]["power"]
    assert power["outcome"] == "unknown"
    assert reason_part in power["reason"]
    assert record["outcome"] == "unknown"


def test_off_mains_sample_voids_even_when_another_sample_is_silent(validity):
    record = _watch(validity, power_supplies=_readings(OFF_MAINS, None))

    assert record["checks"]["power"]["outcome"] == "void"
    assert record["outcome"] == "void"


def test_two_gb_of_foreign_reads_voids_the_arm(validity):
    record = _watch(
        validity,
        disk_reads=_disk(50 * GB, 53 * GB),
        process_reads=_readings(_processes(own=10 * GB), _processes(own=11 * GB)),
    )

    foreign = record["checks"]["foreign_reads"]
    assert foreign["outcome"] == "void"
    assert foreign["evidence"]["foreign_read_bytes"] == 2 * GB
    assert foreign["evidence"]["device"] == "nvme0n1"
    assert foreign["evidence"]["model_path"] == "/synthetic/models/qwen"
    assert "nvme0n1" in foreign["reason"]
    assert foreign["unattributed_bytes"] == 2 * GB
    assert foreign["readers"] == []
    assert record["outcome"] == "void"
    json.dumps(record)


def test_reads_only_by_the_benchmark_and_its_children_stay_ok(validity):
    record = _watch(
        validity,
        disk_reads=_disk(50 * GB, 58 * GB),
        process_reads=_readings(
            _processes(own=10 * GB, others={4242: ("firefox", 300_000)}),
            _processes(own=18 * GB, others={4242: ("firefox", 300_000)}),
        ),
    )

    foreign = record["checks"]["foreign_reads"]
    assert foreign["outcome"] == "ok"
    assert foreign["evidence"]["own_read_bytes"] == 8 * GB
    assert foreign["evidence"]["foreign_read_bytes"] == 0
    assert foreign["readers"] == []
    assert foreign["unattributed_bytes"] == 0
    assert record["outcome"] == "ok"


def test_same_user_reader_is_named_and_the_rest_is_unattributed(validity):
    record = _watch(
        validity,
        disk_reads=_disk(50 * GB, 54 * GB),
        process_reads=_readings(
            _processes(own=10 * GB, others={4242: ("firefox", 300_000)}, uninspectable=292),
            _processes(
                own=11 * GB,
                others={4242: ("firefox", 300_000), 5150: ("rg", 2 * GB)},
                uninspectable=293,
            ),
        ),
    )

    foreign = record["checks"]["foreign_reads"]
    assert foreign["outcome"] == "void"
    assert foreign["evidence"]["foreign_read_bytes"] == 3 * GB
    assert foreign["readers"] == [{"pid": 5150, "name": "rg", "read_bytes": 2 * GB}]
    assert foreign["unattributed_bytes"] == 1 * GB
    assert foreign["evidence"]["uninspectable_processes"] == 293


@pytest.mark.parametrize(
    ("sensors", "reason_part"),
    [
        ({"disk_reads": lambda device: _raises()}, "PermissionError"),
        ({"disk_reads": _disk(50 * GB, None)}, "returned None"),
        ({"process_reads": _raises}, "PermissionError"),
        ({"process_reads": _readings(None)}, "returned None"),
        ({"disk_device": lambda path: _raises()}, "PermissionError"),
        ({"disk_device": lambda path: None}, "maps to no block device"),
        ({"model_path": None}, "no model path"),
    ],
    ids=[
        "disk-raises", "disk-none-at-end", "processes-raise", "processes-none",
        "device-raises", "no-device", "no-model-path",
    ],
)
def test_silent_disk_sensor_makes_foreign_reads_unknown_with_reason(
    validity, sensors, reason_part
):
    record = _watch(validity, **sensors)

    foreign = record["checks"]["foreign_reads"]
    assert foreign["outcome"] == "unknown"
    assert reason_part in foreign["reason"]
    assert foreign["unattributed_bytes"] is None
    assert record["outcome"] == "unknown"
    json.dumps(record)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux sensors only")
def test_real_linux_disk_sensors_return_values(validity):
    model_path = str(HARNESS)
    try:
        device = validity.disk_device(model_path)
    except ValueError as exc:
        pytest.skip(str(exc))
    if device is None:
        pytest.skip(f"{model_path} is on a file system with no block device")
    with validity.ArmWatch(
        arm="A", round=0, run=0, label="real sensors, test only", model_path=model_path
    ) as watch:
        pass

    foreign = watch.record["checks"]["foreign_reads"]
    assert foreign["outcome"] != "unknown", foreign["reason"]
    evidence = foreign["evidence"]
    assert isinstance(evidence["device"], str)
    assert isinstance(evidence["disk_read_bytes_start"], int)
    assert evidence["disk_read_bytes_end"] >= evidence["disk_read_bytes_start"]
    assert isinstance(evidence["own_read_bytes"], int)
    assert isinstance(evidence["uninspectable_processes"], int)
    assert isinstance(foreign["unattributed_bytes"], int)
    json.dumps(watch.record)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux sensors only")
def test_unreaped_child_makes_foreign_reads_unknown_with_reason(validity):
    child = None
    with validity.ArmWatch(
        arm="A", round=0, run=0, label="real sensors, test only",
        model_path="/synthetic/models/qwen", disk_device=lambda path: "nvme0n1",
        disk_reads=_disk(50 * GB, 50 * GB),
    ) as watch:
        child = subprocess.Popen(["true"])
        deadline = time.monotonic() + 5.0
        status = Path(f"/proc/{child.pid}/status")
        while "State:\tZ" not in status.read_text() and time.monotonic() < deadline:
            time.sleep(0.01)
    child.wait()

    foreign = watch.record["checks"]["foreign_reads"]
    assert foreign["outcome"] == "unknown"
    assert str(child.pid) in foreign["reason"]
    assert "reap" in foreign["reason"]


def test_box_stamps_before_and_after_hold_the_machine_state(validity):
    record = _watch(validity, power_supplies=_readings(ON_MAINS, OFF_MAINS))

    before = record["box_stamp_before"]
    after = record["box_stamp_after"]
    assert isinstance(before["unix_s"], float)
    assert after["unix_s"] >= before["unix_s"]
    assert before["ram_available_bytes"] == 7_800 * MIB
    assert after["ram_available_bytes"] == 6_100 * MIB
    assert before["power_source"] == "mains"
    assert after["power_source"] == "battery"
    assert before["gpu_processes"] == []
    assert after["gpu_processes"] == PEERS
    assert before["gpu_memory_used_mib"] == 12
    assert after["gpu_memory_used_mib"] == 2_900
    assert before["unknown"] == {}
    json.dumps(record)


def test_failing_ram_sensor_is_unknown_in_the_box_stamp_and_leaves_the_arm_ok(validity):
    record = _watch(validity, ram_available=_raises)

    for stamp in (record["box_stamp_before"], record["box_stamp_after"]):
        assert stamp["ram_available_bytes"] is None
        assert "PermissionError" in stamp["unknown"]["ram_available_bytes"]
        assert stamp["gpu_memory_used_mib"] is not None
    assert record["outcome"] == "ok"


def _clock(mhz, *reasons):
    return {"sm_clock_mhz": mhz, "reasons": list(reasons)}


def test_clock_trace_is_stored_as_min_median_max(validity):
    sensor = _CountingSensor(_clock(1_800), _clock(1_200), _clock(1_500))
    record = _watch_while_sampling(validity, sensor, samples=3, sampled="gpu_clock")

    clock = record["gpu_clock"]
    assert clock["outcome"] == "ok"
    assert clock["reason"]
    assert clock["samples"] == sensor.calls
    assert clock["unknown_samples"] == 0
    assert clock["sm_clock_mhz"] == {"min": 1_200, "median": 1_500, "max": 1_800}
    assert clock["reasons_seen"] == {}
    assert record["outcome"] == "ok"
    json.dumps(record)


def test_throttle_reason_is_recorded_and_leaves_the_arm_ok(validity):
    sensor = _CountingSensor(
        _clock(1_800), _clock(900, "sw_thermal_slowdown", "sw_power_cap"), _clock(1_700)
    )
    record = _watch_while_sampling(validity, sensor, samples=3, sampled="gpu_clock")

    clock = record["gpu_clock"]
    assert clock["outcome"] == "ok"
    assert clock["reasons_seen"] == {"sw_thermal_slowdown": 1, "sw_power_cap": 1}
    assert "sw_thermal_slowdown" in clock["reason"]
    assert clock["sm_clock_mhz"]["min"] == 900
    assert record["outcome"] == "ok"


def _real_gpu_sensors(validity):
    return {
        "gpu_memory_used": validity.gpu_memory_used,
        "gpu_processes": validity.gpu_processes,
        "gpu_clock": validity.gpu_clock,
    }


def test_missing_nvidia_smi_gives_unknown_gpu_fields_with_the_same_shape(
    validity, monkeypatch, tmp_path
):
    normal = _watch(validity)
    monkeypatch.setenv("PATH", str(tmp_path))
    record = _watch(validity, **_real_gpu_sensors(validity))

    clock = record["gpu_clock"]
    assert clock["outcome"] == "unknown"
    assert "nvidia-smi" in clock["reason"]
    assert clock["sm_clock_mhz"] == {"min": None, "median": None, "max": None}
    assert clock["reasons_seen"] == {}
    assert clock.keys() == normal["gpu_clock"].keys()
    for side in ("box_stamp_before", "box_stamp_after"):
        stamp = record[side]
        assert stamp.keys() == normal[side].keys()
        assert stamp["gpu_processes"] is None
        assert stamp["gpu_memory_used_mib"] is None
        assert "nvidia-smi" in stamp["unknown"]["gpu_processes"]
        assert "nvidia-smi" in stamp["unknown"]["gpu_memory_used_mib"]
    assert record.keys() == normal.keys()
    assert record["outcome"] == "ok"
    json.dumps(record)


def _fake_nvidia_smi(directory, script):
    """A synthetic nvidia-smi on PATH: a shell script whose case branches answer each query."""
    tool = directory / "nvidia-smi"
    tool.write_text(f'#!/bin/sh\ncase "$*" in\n{script}\nesac\n')
    tool.chmod(0o755)


@pytest.mark.skipif(sys.platform == "win32", reason="the fake nvidia-smi is a shell script")
def test_nvidia_smi_output_is_parsed_into_the_record(validity, monkeypatch, tmp_path):
    _fake_nvidia_smi(tmp_path, """\
  *clocks.sm*) echo "1500, 0x0000000000000024" ;;
  *memory.used*) echo "2900" ;;
  *PIDS*) cat <<'END' ;;
==============NVSMI LOG==============

Attached GPUs                             : 1
GPU 00000000:01:00.0
    Processes
        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 5122
            Type                          : G
            Name                          : /usr/bin/gnome-shell
            Used GPU Memory               : 1 MiB
        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 3141
            Type                          : C
            Name                          : C:\\tools\\run: a, b.exe
            Used GPU Memory               : Not available
END""")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    record = _watch(validity, **_real_gpu_sensors(validity))

    clock = record["gpu_clock"]
    assert clock["outcome"] == "ok"
    assert clock["sm_clock_mhz"] == {"min": 1_500, "median": 1_500, "max": 1_500}
    assert clock["reasons_seen"] == {
        "sw_power_cap": clock["samples"], "sw_thermal_slowdown": clock["samples"]
    }
    stamp = record["box_stamp_after"]
    assert stamp["gpu_memory_used_mib"] == 2_900
    assert stamp["gpu_processes"] == [
        {"pid": 5122, "type": "G", "name": "/usr/bin/gnome-shell", "used_mib": 1},
        {"pid": 3141, "type": "C", "name": "C:\\tools\\run: a, b.exe", "used_mib": None},
    ]
    assert stamp["unknown"] == {}


@pytest.mark.skipif(sys.platform == "win32", reason="the fake nvidia-smi is a shell script")
def test_failing_nvidia_smi_gives_unknown_with_its_message(validity, monkeypatch, tmp_path):
    _fake_nvidia_smi(tmp_path, '  *) echo "No devices were found" >&2; exit 6 ;;')
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    record = _watch(validity, **_real_gpu_sensors(validity))

    clock = record["gpu_clock"]
    assert clock["outcome"] == "unknown"
    assert "No devices were found" in clock["reason"]
    assert "No devices were found" in record["box_stamp_before"]["unknown"]["gpu_processes"]
    assert record["outcome"] == "ok"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux sensors only")
def test_real_box_stamp_and_gpu_clock_sensors_return_values(validity):
    if shutil.which("nvidia-smi") is None:
        pytest.skip("nvidia-smi is not on PATH")
    with validity.ArmWatch(arm="A", round=0, run=0, label="real sensors, test only") as watch:
        pass

    for side in ("box_stamp_before", "box_stamp_after"):
        stamp = watch.record[side]
        assert stamp["unknown"] == {}, stamp["unknown"]
        assert stamp["ram_available_bytes"] > 0
        assert stamp["power_source"] in ("mains", "battery")
        assert isinstance(stamp["gpu_processes"], list)
        assert stamp["gpu_memory_used_mib"] >= 0
    clock = watch.record["gpu_clock"]
    assert clock["outcome"] == "ok", clock["reason"]
    assert clock["sm_clock_mhz"]["min"] > 0
    json.dumps(watch.record)
