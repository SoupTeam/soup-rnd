"""Real CPU process isolation and rejection tests; no CUDA inference from CPU passes."""

import copy
import os
import time

import pytest

from benchmarks.c3.isolated_compare import compare_results, run


@pytest.mark.skipif(os.name != "posix", reason="POSIX supervisor required")
def test_fresh_processes_preserve_exact_training(tmp_path, monkeypatch):
    from benchmarks.c3 import isolated_compare

    original = isolated_compare.supervise
    deadlines = []

    def record_deadline(script, sha, output, bound, deadline):
        deadlines.append(deadline)
        return original(script, sha, output, bound, deadline)

    monkeypatch.setattr(isolated_compare, "supervise", record_deadline)
    outer_deadline = time.time() + 180
    result = run(tmp_path / "isolated", "cpu-smoke", outer_deadline)
    assert result["status"] == "NUMERICAL_PASS", result
    assert deadlines == [outer_deadline - 40] * 2
    assert result["outer_deadline_unix_s"] == outer_deadline
    assert result["work_deadline_unix_s"] == outer_deadline - 40
    assert result["performance_verdict"] == "NO_VERDICT"
    first, second = result["arms"]
    assert first["pid"] != second["pid"] != os.getpid()
    assert first["order"] == ["A"] and second["order"] == ["B"]
    assert second["records"][0]["resident_indices"]
    assert second["records"][0]["range_reads_total"]["bytes"] < (
        first["records"][0]["range_reads_total"]["bytes"]
    )
    bad = copy.deepcopy(result["arms"])
    bad[1]["records"][0]["updates"][0]["loss"] += 0.01
    assert compare_results(bad, ["A", "B"])["status"] == "BLOCKED"
    bad = copy.deepcopy(result["arms"])
    bad[1]["weight_sha256"] = {}
    assert compare_results(bad, ["A", "B"])["status"] == "BLOCKED"
    assert compare_results(result["arms"][:1], ["A", "B"])["status"] == "BLOCKED"
    assert compare_results(result["arms"], ["B", "A"])["status"] == "BLOCKED"


def test_empty_comparison_is_not_a_pass():
    assert compare_results([], ["A", "B"])["status"] == "BLOCKED"


@pytest.mark.skipif(os.name != "posix", reason="POSIX supervisor required")
def test_existing_evidence_is_not_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        run(tmp_path, "cpu-smoke")


def test_non_posix_supervisor_is_explicitly_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmarks.c3.isolated_compare.os.name", "nt")
    with pytest.raises(RuntimeError, match="POSIX"):
        run(tmp_path / "unused", "cpu-smoke")


@pytest.mark.skipif(os.name != "posix", reason="POSIX supervisor required")
@pytest.mark.parametrize("deadline", [None, True, "2000", float("nan"),
                                     float("inf"), -1, 1000, 1040])
def test_invalid_cuda_deadline_refused_before_output(tmp_path, monkeypatch, deadline):
    monkeypatch.setattr("benchmarks.c3.isolated_compare.time.time", lambda: 1000)
    output = tmp_path / "unused"
    with pytest.raises(ValueError, match="deadline"):
        run(output, "cuda-pilot", deadline)
    assert not output.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX supervisor required")
def test_calibration_cannot_extend_deadline_or_start_late_arm(tmp_path, monkeypatch):
    from benchmarks.c3 import foreign_read_control, io_calibration, isolated_compare

    clock = [1000]
    monkeypatch.setattr(isolated_compare.time, "time", lambda: clock[0])
    monkeypatch.setattr(io_calibration, "run", lambda _: {"status": "PASS_COUNTER_CALIBRATION"})

    def finish_control(*_):
        clock[0] = 1060
        return {"status": "PASS_FOREIGN_READER_DETECTED"}

    monkeypatch.setattr(foreign_read_control, "run", finish_control)
    monkeypatch.setattr(isolated_compare, "supervise",
                        lambda *_: pytest.fail("expired deadline must not spawn an arm"))
    output = tmp_path / "expired"
    result = run(output, "cuda-pilot", 1100)
    assert result["status"] == "BLOCKED"
    assert result["arms"] == []
    assert "deadline expired" in result["error"]
    assert result["work_deadline_unix_s"] == 1060
    assert not list(output.glob("arm_*.sh"))
