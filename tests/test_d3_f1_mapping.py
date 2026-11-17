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


def test_overlay_loop_requires_explicit_container_scope():
    module = _module()
    assert module._is_overlay_loop("loop0", "overlay", True)
    assert not module._is_overlay_loop("loop0", "overlay", False)
    assert not module._is_overlay_loop("loop0", "ext4", True)
    assert not module._is_overlay_loop("nvme0n1", "overlay", True)


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


def test_guest_disk_exact_read_requires_unchanged_other_disk_counter():
    module = _module()
    module.guest_read_delta({"watched": 100, "other": 17},
                            {"watched": 100 + module.READ_BYTES, "other": 17},
                            module.READ_BYTES)


@pytest.mark.parametrize("after", [
    {"watched": 1048676, "other": 18},
    {"watched": 1048675, "other": 17},
    {"watched": 1048677, "other": 17},
    {"watched": 99, "other": 17},
    {"watched": None, "other": 17},
    {"watched": 1048676, "other": 16},
])
def test_guest_disk_noise_missing_or_reset_counter_cannot_pass(after):
    with pytest.raises(ValueError, match="isolated read"):
        _module().guest_read_delta({"watched": 100, "other": 17}, after, 1048576)


def test_counter_source_cannot_silently_fall_back():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/harness"))
    from arm_validity import cgroup_disk_reads, disk_reads
    from d3_f1_controls import verified_counters

    assert verified_counters({"counter_source": "guest-diskstats"})[0] is disk_reads
    assert verified_counters({})[0] is cgroup_disk_reads
    with pytest.raises(ValueError, match="unrecognized"):
        verified_counters({"counter_source": "disable_other_disk_check"})
