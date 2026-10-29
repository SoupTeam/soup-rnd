"""C2 harness contracts: scheduling and bytes, not hardware performance claims."""

import importlib.util
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

HARNESS = Path(__file__).parents[1] / "benchmarks" / "harness" / "c2_depth_probe.py"
spec = importlib.util.spec_from_file_location("c2_depth_probe", HARNESS)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize("depth", [1, 2])
def test_every_span_verified_before_slot_reuse_and_workers_bounded(depth):
    slots = [bytearray(64), bytearray(64)]
    active = 0
    peak = 0
    lock = threading.Lock()
    verified = []
    owners = {}

    def works(item, slot):
        if slot in owners:
            assert owners[slot] in verified
        owners[slot] = item

        def write(start):
            def run():
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                slots[slot][start : start + 16] = bytes([item]) * 16
                with lock:
                    active -= 1

            return run

        return [write(start) for start in range(0, 64, 16)]

    def verify(item, slot):
        assert slots[slot] == bytes([item]) * 64
        verified.append(item)

    result = probe.pipeline(list(range(19)), depth, 4, works, verify)
    assert verified == list(range(19))
    assert peak <= 4
    assert result["max_pending_spans"] == depth
    assert result["max_pending_ranges"] <= depth * 4
    assert result["completed"] == 19


@pytest.mark.parametrize("depth", [1, 2])
def test_next_layer_can_use_idle_worker_only_at_depth_two(depth):
    slow_started = threading.Event()
    release = threading.Event()
    next_started = threading.Event()

    def works(item, slot):
        def slow():
            slow_started.set()
            assert release.wait(5)

        return [slow, lambda: None] if item == 0 else [next_started.set, lambda: None]

    with ThreadPoolExecutor(max_workers=1) as outer:
        future = outer.submit(probe.pipeline, [0, 1], depth, 2, works)
        try:
            assert slow_started.wait(5)
            assert next_started.wait(0.2) == (depth == 2)
        finally:
            release.set()
        assert future.result(timeout=5)["completed"] == 2


@pytest.mark.parametrize("failure", ["read", "submit", "verify"])
def test_error_drains_all_writers_before_return(failure):
    writer_started = threading.Event()
    release = threading.Event()
    writer_done = threading.Event()

    def writer():
        writer_started.set()
        assert release.wait(5)
        writer_done.set()

    def fail():
        raise OSError("read failed")

    def works(item, slot):
        if failure == "submit" and item == 1:
            raise OSError("submit failed")
        if failure == "read" and item == 0:
            return [fail, writer]
        return [lambda: None, writer]

    def verify(item, slot):
        if failure == "verify":
            raise OSError("verify failed")

    with ThreadPoolExecutor(max_workers=1) as outer:
        future = outer.submit(probe.pipeline, [0, 1], 2, 2, works, verify)
        try:
            assert writer_started.wait(5)
            assert not future.done()
        finally:
            release.set()
        with pytest.raises(OSError, match=failure):
            future.result(timeout=5)
    assert writer_done.is_set()


def test_byte_mismatch_stops_and_drains():
    def works(item, slot):
        return [lambda: None]

    def verify(item, slot):
        raise probe.ByteMismatchError(f"bad span {item}")

    with pytest.raises(probe.ByteMismatchError, match="bad span 0"):
        probe.pipeline([0, 1, 2], 2, 1, works, verify)


@pytest.mark.parametrize("depth,workers", [(0, 4), (3, 4), (1, 0)])
def test_bad_limits_rejected(depth, workers):
    with pytest.raises(ValueError):
        probe.pipeline([0], depth, workers, lambda item, slot: [])


def record(ratios):
    return {
        "correctness_ok": True,
        "invalid_reasons": [],
        "complete": True,
        "rounds": [
            {"order": [1, 2] if index % 2 == 0 else [2, 1],
             "arms": {"1": {"gb_s": 4.0}, "2": {"gb_s": 4.0 * ratio}}}
            for index, ratio in enumerate(ratios)
        ],
    }


@pytest.mark.parametrize(
    "ratios,expected",
    [([1.11] * 6, "PASS primitive"), ([1.0] * 6, "FAIL benefit"),
     ([0.94] * 6, "FAIL regression"), ([1.106, 1.101, 0.963] * 2, "NO VERDICT")],
)
def test_frozen_decision_rule(ratios, expected):
    assert probe.verdict(record(ratios), reviewed=True)["verdict"] == expected


@pytest.mark.parametrize("reason", ["counter missing", "battery", "foreign writes"])
def test_invalid_sequence_cannot_pass(reason):
    data = record([1.2] * 6)
    data["invalid_reasons"] = [reason]
    assert probe.verdict(data, reviewed=True)["verdict"] == "NO VERDICT"


def test_missing_or_partial_evidence_and_unreviewed_host_cannot_pass():
    for data in ({}, record([1.2] * 5)):
        assert probe.verdict(data, reviewed=True)["verdict"] == "NO VERDICT"
    assert probe.verdict(record([1.2] * 6))["verdict"] == "NO VERDICT"
    data = record([1.2] * 6)
    data["correctness_ok"] = False
    assert probe.verdict(data, reviewed=True)["verdict"] == "FAIL correctness"


def test_counter_validity_detects_missing_contamination_and_gaps():
    samples = [(i * 0.25, {"read": 4e9, "write": 0, "ac": 1}) for i in range(12)]
    assert probe.counter_reasons(samples, 0, 3, 4.0) == []
    assert probe.counter_reasons([], 0, 3, 4.0)
    for key, value in (("write", 2**21), ("read", 5e9), ("ac", 0)):
        changed = [(stamp, dict(row, **{key: value})) for stamp, row in samples]
        assert probe.counter_reasons(changed, 0, 3, 4.0)
    assert probe.counter_reasons([(0.1, samples[0][1]), (2.5, samples[0][1])], 0, 3, 4)
