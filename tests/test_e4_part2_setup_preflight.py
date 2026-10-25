"""A data root with no pull record is a setup error: refused before any attempt (exit 6, no
journal write), so it does not use the section 8 rerun. CPU only."""

# ruff: noqa: I001 (the support import must run before the harness imports)
import json
import types

import pytest

import tests.e4_part2_support  # noqa: F401  (puts the harness on sys.path)
import e4_part2_calibration as cal  # noqa: E402
import e4_part2_common as common  # noqa: E402
import e4_part2_transfer as transfer  # noqa: E402


def pull_record(data_root, manifest, sha=None):
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / transfer.VERIFIED_NAME).write_text(json.dumps(
        {"manifest_sha256": sha or common.sha256_file(manifest), "files": {}}))


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path / "quant-manifest.json"
    path.write_text("{}")
    return path


def test_a_missing_pull_record_is_a_setup_problem_that_names_the_pull(tmp_path, manifest):
    problem = cal.setup_problem(tmp_path / "data", manifest)
    assert problem is not None
    assert transfer.VERIFIED_NAME in problem and "transfer pull" in problem


def test_a_record_for_another_manifest_is_a_setup_problem(tmp_path, manifest):
    pull_record(tmp_path / "data", manifest, sha="0" * 64)
    problem = cal.setup_problem(tmp_path / "data", manifest)
    assert problem is not None and "another manifest" in problem


def test_a_matching_pull_record_passes(tmp_path, manifest):
    pull_record(tmp_path / "data", manifest)
    assert cal.setup_problem(tmp_path / "data", manifest) is None


def test_preflight_refuses_a_missing_pull_record_before_any_output_is_written(
        tmp_path, manifest, monkeypatch):
    tests.e4_part2_support.stub_pynvml(monkeypatch)
    args = types.SimpleNamespace(instance_id="i-1", device="cuda", gpu_index=0, fake_card=None,
                                 dry_run=False, data_root=tmp_path / "data", manifest=manifest)
    with pytest.raises(common.CalibrationStopped, match="transfer pull"):
        cal.preflight(args, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_the_dry_run_does_not_need_a_pull_record(tmp_path, manifest):
    args = types.SimpleNamespace(instance_id="dry-1", device="cuda", gpu_index=0,
                                 fake_card="a100-40", dry_run=True, data_root=None,
                                 manifest=None)
    assert cal.preflight(args, tmp_path / "out") == "F"
