"""The calibration needs pynvml on a GPU box: refused before the first attempt when missing,
not voided twice. CPU only; nothing here touches a GPU."""

# ruff: noqa: I001 (the support import must run before the harness imports)
import sys
import types

import pytest

import tests.e4_part2_support  # noqa: F401  (puts the harness on sys.path)
import e4_part2_calibration as cal  # noqa: E402
import e4_part2_common as common  # noqa: E402
import e4_part2_dryrun as dryrun  # noqa: E402


def cuda_args(tmp_path, **over):
    base = dict(instance_id="i-1", device="cuda", gpu_index=0, fake_card=None, dry_run=False,
                data_root=None, manifest=None)
    return types.SimpleNamespace(**{**base, **over})


def test_preflight_refuses_before_any_attempt_when_pynvml_is_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)  # import pynvml raises ImportError
    monkeypatch.setattr(common, "require_single_gpu",
                        lambda _i: pytest.fail("GPU checks must not run"))
    with pytest.raises(common.CalibrationStopped, match="nvidia-ml-py"):
        cal.preflight(cuda_args(tmp_path), tmp_path / "out")
    assert not (tmp_path / "out").exists()  # no verdict, no journal, nothing written


def test_preflight_passes_when_pynvml_is_importable(tmp_path, monkeypatch):
    tests.e4_part2_support.stub_setup(monkeypatch)
    monkeypatch.setitem(sys.modules, "pynvml", types.ModuleType("pynvml"))
    monkeypatch.setattr(common, "require_single_gpu", lambda _i: None)
    monkeypatch.setattr(common, "box_stamp",
                        lambda _g, inst: dryrun.fake_stamp("a100-40", inst))
    assert cal.preflight(cuda_args(tmp_path), tmp_path / "out") == "F"


def test_dry_run_does_not_require_pynvml(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)
    args = cuda_args(tmp_path, instance_id="dry-1", fake_card="a100-40", dry_run=True)
    assert cal.preflight(args, tmp_path / "out") == "F"
