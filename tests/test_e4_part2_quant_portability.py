"""The E4 quantization harness imports without Unix-only resource telemetry."""

import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HARNESS = Path(__file__).parents[1] / "benchmarks" / "harness"


@pytest.fixture
def quant(monkeypatch):
    monkeypatch.syspath_prepend(str(HARNESS))
    return importlib.import_module("e4_part2_quant")


def test_harness_imports_without_resource():
    code = (
        "import sys; sys.modules['resource'] = None; "
        "sys.path.insert(0, sys.argv[1]); "
        "import e4_part2_quant as quant; "
        "assert len(quant.enumerate_units()) == 3136"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(HARNESS)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_missing_resource_reports_unknown_peak_rss(quant, monkeypatch):
    monkeypatch.setitem(sys.modules, "resource", None)
    assert quant._peak_rss_bytes() is None


@pytest.mark.parametrize("platform,expected", [("darwin", 123), ("linux", 123 * 1024)])
def test_peak_rss_is_in_bytes_and_excludes_children(quant, monkeypatch, platform, expected):
    calls = []

    def getrusage(who):
        calls.append(who)
        return SimpleNamespace(ru_maxrss=123)

    resource = SimpleNamespace(RUSAGE_SELF=0, getrusage=getrusage)
    monkeypatch.setitem(sys.modules, "resource", resource)
    monkeypatch.setattr(sys, "platform", platform)
    assert quant._peak_rss_bytes() == expected
    assert calls == [0]
