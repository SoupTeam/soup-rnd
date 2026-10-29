"""E4 quantization metadata must not prevent collection on Windows."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HARNESS = Path(__file__).parents[1] / "benchmarks" / "harness"
QUANT = HARNESS / "e4_part2_quant.py"


def test_quant_imports_without_resource_or_training_libraries():
    script = """
import builtins
import runpy
import sys

original_import = builtins.__import__

def without_resource(name, *args, **kwargs):
    if name == "resource":
        raise ModuleNotFoundError("resource is unavailable on Windows", name="resource")
    return original_import(name, *args, **kwargs)

builtins.__import__ = without_resource
sys.path.insert(0, sys.argv[1])
module = runpy.run_path(sys.argv[2], run_name="e4_quant_import_test")
assert module["_peak_rss_bytes"]() is None
assert not {"torch", "transformers", "bitsandbytes"}.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(HARNESS), str(QUANT)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def quant_module(monkeypatch):
    monkeypatch.syspath_prepend(str(HARNESS))
    return runpy.run_path(str(QUANT), run_name="e4_quant_metadata_test")


@pytest.mark.parametrize(("platform", "expected"), [("darwin", 17), ("linux", 17 * 1024)])
def test_peak_rss_is_process_only_and_in_bytes(quant_module, monkeypatch, platform, expected):
    calls = []

    def usage(who):
        calls.append(who)
        return SimpleNamespace(ru_maxrss=17)

    monkeypatch.setitem(sys.modules, "resource", SimpleNamespace(RUSAGE_SELF=0, getrusage=usage))
    monkeypatch.setattr(sys, "platform", platform)
    assert quant_module["_peak_rss_bytes"]() == expected
    assert calls == [0]


def test_peak_rss_does_not_hide_measurement_errors(quant_module, monkeypatch):
    def broken_usage(_who):
        raise OSError("measurement failed")

    monkeypatch.setitem(sys.modules, "resource", SimpleNamespace(
        RUSAGE_SELF=0, getrusage=broken_usage,
    ))
    with pytest.raises(OSError, match="measurement failed"):
        quant_module["_peak_rss_bytes"]()
