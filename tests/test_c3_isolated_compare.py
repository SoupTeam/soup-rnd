"""Real CPU process isolation and rejection tests; no CUDA inference from CPU passes."""

import copy
import os

import pytest

from benchmarks.c3.isolated_compare import compare_results, run


@pytest.mark.skipif(os.name != "posix", reason="POSIX supervisor required")
def test_fresh_processes_preserve_exact_training(tmp_path):
    result = run(tmp_path / "isolated", "cpu-smoke")
    assert result["status"] == "NUMERICAL_PASS", result
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
