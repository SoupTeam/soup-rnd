"""A reference ceiling must be measured on the same, frequency-matched GPU."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks/harness"))


def window(start=1_000_000_000):
    times = [start + offset for offset in (100_000_000, 500_000_000, 900_000_000)]
    return {
        "gpu_uuid": "GPU-test",
        "timed_start_ns": start,
        "timed_end_ns": start + 1_000_000_000,
        "actual_timed_seconds": 1.0,
        "actual_timed_steps": 200,
        "actual_warmup_steps": 20,
        "actual_warmup_seconds": 1.0,
        "step_cuda_ms": [0.2] * 200,
        "clock_sm_mhz": [1200] * 3,
        "power_limit_mw": [300000] * 3,
        "memory_util_pct": [0, 40, 100],
        "telemetry_gpu_matches_cuda": True,
        "telemetry_errors": [],
        "telemetry": [
            {
                "time_ns": time_ns,
                "uuid": "GPU-test",
                "available": True,
                "sm_mhz": 1200,
                "power_limit_mw": 300000,
                "memory_util_pct": utilization,
            }
            for time_ns, utilization in zip(times, (0, 40, 100))
        ],
    }


def test_same_device_and_frequency_reference_passes():
    from d1_reference import validate_reference

    assert validate_reference(window(), window(3_000_000_000)) == {
        "passed": True,
        "reasons": [],
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("gpu_uuid", "GPU-other"),
        ("telemetry_gpu_matches_cuda", False),
        ("telemetry_errors", ["NVML unavailable"]),
        ("telemetry", []),
        ("clock_sm_mhz", [1200]),
        ("memory_util_pct", []),
        ("power_limit_mw", [300000]),
        ("timed_start_ns", 0),
        ("timed_end_ns", float("nan")),
        ("actual_timed_seconds", 0.5),
        ("actual_timed_steps", 199),
        ("actual_warmup_steps", 19),
        ("actual_warmup_seconds", 0.99),
        ("step_cuda_ms", [float("nan")] * 200),
        ("step_cuda_ms", [0] * 200),
    ],
)
def test_reference_rejects_missing_forged_or_short_measurement(field, value):
    from d1_reference import validate_reference

    reference = window(3_000_000_000)
    reference[field] = value
    assert not validate_reference(window(), reference)["passed"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("time_ns", 999),
        ("uuid", "GPU-other"),
        ("available", False),
        ("sm_mhz", None),
        ("sm_mhz", 1261),
        ("power_limit_mw", 310000),
        ("memory_util_pct", -1),
        ("memory_util_pct", 101),
    ],
)
def test_raw_samples_override_claimed_success(field, value):
    from d1_reference import validate_reference

    reference = window(3_000_000_000)
    reference["telemetry"][0][field] = value
    assert not validate_reference(window(), reference)["passed"]


def test_matching_aggregates_do_not_hide_clock_or_power_change():
    from d1_reference import validate_reference

    for field, aggregate, changed in (
        ("sm_mhz", "clock_sm_mhz", 1261),
        ("power_limit_mw", "power_limit_mw", 310000),
    ):
        reference = window(3_000_000_000)
        for sample in reference["telemetry"]:
            sample[field] = changed
        reference[aggregate] = [changed] * 3
        assert not validate_reference(window(), reference)["passed"]


def test_no_overlapping_reference_window():
    from d1_reference import validate_reference

    assert not validate_reference(window(), window())["passed"]


def test_report_recomputes_reference_instead_of_trusting_passed_flag():
    from d1_rules import summarize

    rows = []
    for block in range(5):
        for run, arm in enumerate("ABBA"):
            primary = window()
            primary.update(
                block=block,
                run=run,
                arm=arm,
                case="projection",
                source_sha="source",
                rule_sha256="rule",
                settings_sha256="settings",
                median_step_ms=1,
                path="dense",
                peak_allocated_bytes=10,
                peak_reserved_bytes=20,
                validity={"outcome": "ok"},
                peers_clean=True,
                dense_gemm_reference=window(3_000_000_000),
                dense_gemm_reference_validation={"passed": True},
            )
            rows.append(primary)
    assert summarize(rows)["comparisons"][0]["accepted_blocks"] == 5
    tampered = copy.deepcopy(rows)
    tampered[0]["dense_gemm_reference"]["telemetry"] = []
    result = summarize(tampered)["comparisons"][0]
    assert result["accepted_blocks"] == 4
    assert result["status"] == "NO_VERDICT"


def test_reference_timer_extends_work_until_minimum_duration(monkeypatch):
    import d1_nf4_study as study
    import torch

    clock, steps = {"ns": 0}, []

    def monotonic_ns():
        clock["ns"] += 250_000_000
        return clock["ns"]

    class Event:
        def __init__(self, **_kwargs):
            pass

        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, _other):
            return 0.2

    class Telemetry:
        errors = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.samples = [
                {
                    "time_ns": clock["ns"] - 1,
                    "available": True,
                    "uuid": "GPU-00000000-0000-0000-0000-000000000001",
                    "sm_mhz": 1200,
                    "power_limit_mw": 300000,
                    "memory_util_pct": 0,
                }
            ]

    class Properties:
        uuid = "GPU-00000000-0000-0000-0000-000000000001"

    monkeypatch.setattr(study.time, "monotonic_ns", monotonic_ns)
    monkeypatch.setattr(study, "Telemetry", Telemetry)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 100)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 200)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: Properties())
    result = study.timed_steps(lambda: steps.append(True), count=2, warmup=0, timed_seconds=1)
    assert result["actual_timed_seconds"] >= 1
    assert result["actual_timed_steps"] == len(steps) == len(result["step_cuda_ms"])
    assert len(steps) > 2
    assert result["peak_allocated_bytes"] == 100
    assert result["telemetry_gpu_matches_cuda"]
