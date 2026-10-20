"""Read-only overlay association must refuse ambiguity rather than fake F1 OK."""

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "benchmarks/harness/d3_f1_mapping_probe.py"
    spec = importlib.util.spec_from_file_location("d3_mapping_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_exact_single_device_read_delta_is_accepted():
    module = _module()
    before = module.io_reads("8:0 rbytes=17 wbytes=20\n259:0 rbytes=100 rios=2\n")
    after = module.io_reads("8:0 rbytes=17 wbytes=50\n259:0 rbytes=1048676 rios=10\n")
    assert module.exact_read_device(before, after, 1048576) == "259:0"


@pytest.mark.parametrize("after", [
    {"8:0": 0, "259:0": 0}, {"8:0": 0, "259:0": 1048575},
    {"8:0": 1, "259:0": 1048576}, {"8:0": 1048576, "259:0": 1048576},
    {"8:0": -1, "259:0": 1048576},
])
def test_missing_ambiguous_or_reset_counters_are_refused(after):
    with pytest.raises(ValueError):
        _module().exact_read_device({"8:0": 0, "259:0": 0}, after, 1048576)


@pytest.mark.parametrize("text", ["", "nvme rbytes=12", "8:0 wbytes=2", "8:0 rbytes=-1",
                                     "8:0 rbytes=2\n8:0 rbytes=3\n"])
def test_bad_sensor_answers_cannot_become_a_disk_mapping(text):
    with pytest.raises((ValueError, KeyError)):
        _module().io_reads(text)


def test_ambiguous_discovery_retains_raw_counter_evidence(tmp_path, monkeypatch):
    module = _module()
    path = tmp_path / "weights.safetensors"
    path.write_bytes(bytes(module.READ_BYTES))
    replies = iter(("259:0 rbytes=0\n", f"259:0 rbytes={module.READ_BYTES + 4096}\n"))
    original = Path.read_text

    def read(target, *args, **kwargs):
        if target.name == "io.stat":
            return next(replies)
        if target.as_posix().endswith("/proc/self/cgroup"):
            return "0::/\n"
        return original(target, *args, **kwargs)

    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(module, "direct_read", lambda target: module.READ_BYTES)
    with pytest.raises(module.MappingFailureError) as error:
        module.discover([path])
    evidence = error.value.evidence
    assert evidence["status"] == "requires_review"
    assert "device" not in evidence["files"][0]
    raw = evidence["files"][0]["reads"][0]
    assert raw["before"]["259:0"] == 0
    assert raw["after"]["259:0"] == module.READ_BYTES + 4096
