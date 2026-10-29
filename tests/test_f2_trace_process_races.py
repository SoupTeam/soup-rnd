"""A vanished telemetry child must not erase the still-live training process."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks" / "harness"))
import f2_reference as f2  # noqa: E402


def one_sample(tmp_path, monkeypatch, *, failed_sensor, error, failed_root=False):
    """The child exits after census but before one of its sensor reads."""
    def member(pid, rss, reads, writes, *, failure=None):
        def memory_info():
            if failure is not None and failed_sensor == "memory":
                raise failure(pid)
            return SimpleNamespace(rss=rss)

        def io_counters():
            if failure is not None and failed_sensor == "io":
                raise failure(pid)
            return SimpleNamespace(read_bytes=reads, write_bytes=writes)

        return SimpleNamespace(pid=pid, memory_info=memory_info, io_counters=io_counters)

    root = member(101, 100, 1000, 10, failure=error if failed_root else None)
    live = member(102, 40, 400, 4)
    transient = member(103, 20, 200, 2, failure=None if failed_root else error)
    root.children = lambda recursive: [live, transient]
    monkeypatch.setattr(psutil, "Process", lambda pid: root)
    monkeypatch.setattr(f2.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        stdout="70,1665,3000,100,200,P2,60", stderr=""))
    trace = f2.Trace(tmp_path / "trace.jsonl", "physical-gpu")
    trace.process = SimpleNamespace(pid=101)
    monkeypatch.setattr(trace.stop, "wait", lambda seconds: trace.stop.set())
    trace._sample()
    assert len(trace.rows) == 1
    assert json.loads(trace.path.read_text()) == trace.rows[0]
    return trace.rows[0]


@pytest.mark.parametrize("sensor", ["memory", "io"])
def test_vanished_child_keeps_root_and_other_live_children(tmp_path, monkeypatch, sensor):
    row = one_sample(tmp_path, monkeypatch, failed_sensor=sensor, error=psutil.NoSuchProcess)
    assert (row["rss_bytes"], row["read_bytes"], row["write_bytes"]) == (140, 1400, 14)


@pytest.mark.parametrize("error", [psutil.NoSuchProcess, psutil.AccessDenied])
def test_missing_root_sensor_stays_unknown(tmp_path, monkeypatch, error):
    row = one_sample(tmp_path, monkeypatch, failed_sensor="memory", error=error, failed_root=True)
    assert row["rss_bytes"] is row["read_bytes"] is row["write_bytes"] is None


def test_inaccessible_live_child_stays_unknown(tmp_path, monkeypatch):
    row = one_sample(tmp_path, monkeypatch, failed_sensor="io", error=psutil.AccessDenied)
    assert row["rss_bytes"] is row["read_bytes"] is row["write_bytes"] is None
