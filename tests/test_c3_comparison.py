"""Local runner controls. CPU evidence cannot become a CUDA performance verdict."""

import json

import pytest

from benchmarks.c3.compare import run


def test_cpu_comparison_preserves_exactness_and_measures_reads(tmp_path):
    result = run(tmp_path / "run", "cpu-smoke")
    assert result["status"] == "NUMERICAL_PASS"
    assert result["exact_training_match"]
    assert result["performance_verdict"] == "NO_VERDICT"
    before, after = result["records"]
    assert 0 < after["range_reads_total"]["bytes"] < before["range_reads_total"]["bytes"]
    assert 0 < len(after["resident_indices"]) < 4
    assert len(before["updates"]) == len(after["updates"]) == 3


def test_existing_output_cannot_be_reused(tmp_path):
    (tmp_path / "sentinel").write_text("keep")
    with pytest.raises(FileExistsError):
        run(tmp_path, "cpu-smoke")
    assert (tmp_path / "sentinel").read_text() == "keep"


def test_setup_failure_is_preserved(tmp_path, monkeypatch):
    from soup_cli.utils import layer_shard

    def fail(*args, **kwargs):
        raise OSError("injected sharding failure")

    monkeypatch.setattr(layer_shard, "shard_checkpoint", fail)
    target = tmp_path / "failed"
    with pytest.raises(OSError, match="injected"):
        run(target, "cpu-smoke")
    result = json.loads((target / "result.json").read_text())
    assert result["status"] == "FAILED"
    assert "injected" in result["error"]
    assert result["performance_verdict"] == "NO_VERDICT"
