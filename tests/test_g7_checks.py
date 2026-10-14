"""Tests for the G7 extra arm checks (benchmarks/harness/g7_checks.py, G7 ticket 02).

``gpu_power_limit`` and ``swap`` run beside F1's arm watch and join its outcome
rule. Every reading below is synthetic: the readers are fakes that return
made-up power queries and swap counters, and no test runs nvidia-smi or writes
into benchmarks/results/.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HARNESS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


@pytest.fixture(scope="module")
def g7():
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    return _load("_g7_checks_test_module", "g7_checks.py")


def _power_query(limit: str = "80.00 W", module_limit: str = "N/A") -> str:
    """The shape of ``nvidia-smi -q -d POWER`` on the RTX 3050 laptop: the limit
    sits under "GPU Power Readings", and "Module Power Readings" repeats the
    field as N/A."""
    return f"""
==============NVSMI LOG==============

Attached GPUs                                          : 1
GPU 00000000:01:00.0
    GPU Power Readings
        Average Power Draw                             : 6.26 W
        Instantaneous Power Draw                       : 6.24 W
        Current Power Limit                            : {limit}
        Requested Power Limit                          : N/A
        Default Power Limit                            : 80.00 W
    Module Power Readings
        Average Power Draw                             : N/A
        Current Power Limit                            : {module_limit}
"""


POWER_QUERY_WITHOUT_LIMIT = """
GPU 00000000:01:00.0
    GPU Power Readings
        Average Power Draw                             : 6.26 W
        Default Power Limit                            : 80.00 W
"""


def _sequence(*values):
    """A fake reader that returns ``values`` in order, then repeats the last one.
    A value that is an exception is raised instead."""
    queue = list(values)

    def reader():
        value = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(value, Exception):
            raise value
        return value

    return reader


def _power_check(g7, *queries):
    checks = g7.ExtraChecks(
        power_query=_sequence(*queries), swap_counters=_sequence({"pswpin": 1, "pswpout": 2})
    )
    checks.start()
    checks.stop()
    return checks.results()["gpu_power_limit"]


def _swap_check(g7, *counters):
    checks = g7.ExtraChecks(
        power_query=_sequence(_power_query()), swap_counters=_sequence(*counters)
    )
    checks.start()
    checks.stop()
    return checks.results()["swap"]


# --- gpu_power_limit -------------------------------------------------------------------


def test_80_watts_before_and_after_is_ok(g7):
    check = _power_check(g7, _power_query("80.00 W"), _power_query("80.00 W"))

    assert check["outcome"] == "ok"
    assert check["reason"]
    assert check["evidence"]["power_limit_w_start"] == 80.0
    assert check["evidence"]["power_limit_w_end"] == 80.0


def test_15_watts_voids_the_check_and_names_the_value(g7):
    check = _power_check(g7, _power_query("15.00 W"), _power_query("15.00 W"))

    assert check["outcome"] == "void"
    assert "15.00" in check["reason"]
    assert check["evidence"]["power_limit_w_start"] == 15.0


def test_a_cap_that_appears_during_the_arm_voids_the_check(g7):
    check = _power_check(g7, _power_query("80.00 W"), _power_query("10.00 W"))

    assert check["outcome"] == "void"
    assert check["evidence"]["power_limit_w_end"] == 10.0


def test_a_limit_above_80_watts_is_void_too(g7):
    check = _power_check(g7, _power_query("95.00 W"), _power_query("95.00 W"))

    assert check["outcome"] == "void"


def test_not_available_limit_is_unknown_with_a_reason(g7):
    check = _power_check(g7, _power_query("N/A"), _power_query("N/A"))

    assert check["outcome"] == "unknown"
    assert "N/A" in check["reason"]


def test_missing_limit_field_is_unknown_with_a_reason(g7):
    check = _power_check(g7, POWER_QUERY_WITHOUT_LIMIT, POWER_QUERY_WITHOUT_LIMIT)

    assert check["outcome"] == "unknown"
    assert "Current Power Limit" in check["reason"]


def test_failing_power_query_is_unknown_with_its_message(g7):
    check = _power_check(g7, _power_query(), RuntimeError("nvidia-smi exited 9"))

    assert check["outcome"] == "unknown"
    assert "nvidia-smi exited 9" in check["reason"]


def test_a_cap_seen_at_one_end_is_void_even_when_the_other_end_is_silent(g7):
    check = _power_check(g7, _power_query("15.00 W"), _power_query("N/A"))

    assert check["outcome"] == "void"


def test_module_section_does_not_mask_the_gpu_limit(g7):
    check = _power_check(
        g7, _power_query("15.00 W", module_limit="80.00 W"), _power_query("15.00 W")
    )

    assert check["outcome"] == "void"


# --- swap ------------------------------------------------------------------------------


def test_unchanged_swap_counters_are_ok(g7):
    counters = {"pswpin": 523_137, "pswpout": 1_264_078}
    check = _swap_check(g7, counters, dict(counters))

    assert check["outcome"] == "ok"
    assert check["evidence"]["swapped_in_pages"] == 0
    assert check["evidence"]["swapped_out_pages"] == 0


def test_swap_out_increase_voids_the_check_with_the_page_counts(g7):
    check = _swap_check(
        g7, {"pswpin": 100, "pswpout": 200}, {"pswpin": 100, "pswpout": 401_000}
    )

    assert check["outcome"] == "void"
    assert check["evidence"]["swapped_out_pages"] == 400_800
    assert check["evidence"]["swapped_in_pages"] == 0
    assert "400800" in check["reason"]


def test_swap_in_increase_voids_the_check_with_the_page_counts(g7):
    check = _swap_check(g7, {"pswpin": 100, "pswpout": 200}, {"pswpin": 160, "pswpout": 200})

    assert check["outcome"] == "void"
    assert check["evidence"]["swapped_in_pages"] == 60


def test_unreadable_swap_counter_is_unknown_with_a_reason(g7):
    check = _swap_check(g7, {"pswpin": 1, "pswpout": 2}, OSError("no /proc/vmstat"))

    assert check["outcome"] == "unknown"
    assert "no /proc/vmstat" in check["reason"]


def test_swap_counters_missing_a_key_are_unknown(g7):
    check = _swap_check(g7, {"pswpin": 1}, {"pswpin": 1})

    assert check["outcome"] == "unknown"


def test_a_swap_increase_is_void_even_when_the_end_reading_is_missing_a_counter(g7):
    check = _swap_check(g7, {"pswpin": 1, "pswpout": 2}, {"pswpin": 9, "pswpout": None})

    assert check["outcome"] == "void"


def test_real_swap_counters_read_this_machine(g7):
    if not sys.platform.startswith("linux"):
        pytest.skip("reads /proc/vmstat")
    counters = g7.swap_counters()

    assert counters["pswpin"] >= 0
    assert counters["pswpout"] >= 0


# --- combined outcome ------------------------------------------------------------------

ON_MAINS = {"battery": True, "online": {"ACAD": True}}
GB = 1_000_000_000
QUIET = {"own": 10 * GB, "others": {}, "uninspectable": 0}


def _watch(g7, *, power_query=None, swap_counters=None, **overrides):
    """A G7 arm watch with synthetic F1 sensors. Every F1 check is ok unless overridden."""
    kwargs = dict(
        platform="linux",
        sleep_offset=_sequence(25_588.0),
        suspend_count=_sequence(7),
        power_supplies=_sequence(ON_MAINS),
        model_path="/synthetic/shards",
        disk_device=lambda path: "nvme0n1",
        disk_reads=lambda device: 50 * GB,
        other_disk_reads=lambda device: 100 * GB,
        process_reads=_sequence(QUIET),
        ram_available=_sequence(8 * GB),
        commit_charge=_sequence({"used_bytes": 9 * GB, "limit_bytes": 24 * GB}),
        gpu_memory_used=_sequence(12),
        gpu_processes=_sequence([]),
        gpu_clock=_sequence({"sm_clock_mhz": 1_500, "reasons": []}),
    )
    kwargs.update(overrides)
    watch = g7.G7ArmWatch(
        arm="bare-16",
        round=1,
        run=1,
        label="RTX 3050, synthetic",
        power_query=power_query or _sequence(_power_query()),
        swap_counters=swap_counters or _sequence({"pswpin": 5, "pswpout": 6}),
        **kwargs,
    )
    with watch:
        pass
    return watch.record


FIVE = ["suspend", "power", "foreign_reads", "gpu_power_limit", "swap"]


def test_clean_arm_holds_all_five_checks_and_is_ok(g7):
    record = _watch(g7)

    assert record["required"] == FIVE
    assert list(record["checks"]) == FIVE
    assert record["outcome"] == "ok"
    assert record["label"] == "RTX 3050, synthetic"
    json.dumps(record)


def test_a_power_cap_voids_the_arm_though_every_f1_check_is_ok(g7):
    record = _watch(g7, power_query=_sequence(_power_query("15.00 W")))

    assert record["checks"]["suspend"]["outcome"] == "ok"
    assert record["checks"]["gpu_power_limit"]["outcome"] == "void"
    assert record["outcome"] == "void"


def test_swapping_voids_the_arm(g7):
    record = _watch(
        g7, swap_counters=_sequence({"pswpin": 5, "pswpout": 6}, {"pswpin": 5, "pswpout": 99})
    )

    assert record["checks"]["swap"]["outcome"] == "void"
    assert record["outcome"] == "void"


def test_an_unreadable_power_limit_makes_the_arm_unknown(g7):
    record = _watch(g7, power_query=_sequence(_power_query("N/A")))

    assert record["checks"]["gpu_power_limit"]["outcome"] == "unknown"
    assert record["outcome"] == "unknown"


def test_an_unreadable_swap_counter_makes_the_arm_unknown(g7):
    record = _watch(g7, swap_counters=_sequence(OSError("gone")))

    assert record["outcome"] == "unknown"


def test_an_f1_check_that_voids_still_voids_the_arm(g7):
    record = _watch(g7, suspend_count=_sequence(7, 8))

    assert record["checks"]["suspend"]["outcome"] == "void"
    assert record["checks"]["gpu_power_limit"]["outcome"] == "ok"
    assert record["outcome"] == "void"


def test_void_beats_unknown_across_the_five_checks(g7):
    record = _watch(
        g7,
        power_query=_sequence(_power_query("N/A")),
        swap_counters=_sequence({"pswpin": 1, "pswpout": 1}, {"pswpin": 1, "pswpout": 2}),
    )

    assert record["outcome"] == "void"


def test_the_extra_checks_read_before_and_after_the_arm(g7):
    reads = []

    def power_query():
        reads.append("power")
        return _power_query()

    def swap_counters():
        reads.append("swap")
        return {"pswpin": 1, "pswpout": 1}

    watch = g7.G7ArmWatch(
        arm="bare-16",
        round=1,
        run=1,
        label="RTX 3050, synthetic",
        power_query=power_query,
        swap_counters=swap_counters,
        platform="darwin",
        gpu_memory_used=_sequence(12),
        gpu_processes=_sequence([]),
        gpu_clock=_sequence({"sm_clock_mhz": 1_500, "reasons": []}),
    )
    with watch:
        reads.append("arm")

    assert reads == ["power", "swap", "arm", "power", "swap"]
    assert watch.record["checks"]["suspend"]["outcome"] == "unknown"
