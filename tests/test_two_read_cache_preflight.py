"""Preflight must reject changed evidence or invalid resource boundaries before paid work."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.harness import two_read_cache as runner
from benchmarks.harness.two_read_cache_common import fingerprint_files, sha256_file
from tests.test_two_read_cache_runner import _spec


def _bound_spec(tmp_path: Path, monkeypatch) -> dict:
    spec = _spec()
    for key in ("weights", "shard_dir"):
        directory = tmp_path / key
        directory.mkdir()
        (directory / "config.json" if key == "weights" else directory / "index.json").write_text(
            "{}", encoding="utf-8",
        )
        (directory / "part.safetensors").write_bytes(b"original")
        spec[key] = str(directory)
    spec["index_path"] = str(Path(spec["shard_dir"]) / "index.json")
    for label, key, function in (("model", "weights", runner._model_files),
                                  ("store", "shard_dir", runner._store_files)):
        root = Path(spec[key])
        files = function(root)
        spec[f"{label}_fingerprint"] = fingerprint_files(files, root)
        spec[f"{label}_stat"] = {
            path.relative_to(root).as_posix(): [path.stat().st_size, path.stat().st_mtime_ns]
            for path in files
        }
    tokens = tmp_path / "inputs.pt"
    tokens.write_bytes(b"frozen")
    spec.update(training_data_path=str(tokens), training_data_sha256=sha256_file(tokens),
                source_fingerprint={"code.py": "abc"}, environment={"packages": "same"},
                hardware={"uuid": "GPU-one", "name": "card", "capability": [8, 9],
                          "physical_vram_bytes": 8 * runner.GIB})
    monkeypatch.setattr(runner, "source_fingerprint", lambda: {"code.py": "abc"})
    monkeypatch.setattr(runner, "environment", lambda: {"packages": "same"})
    monkeypatch.setattr(runner, "hardware", lambda: dict(spec["hardware"]))
    monkeypatch.setattr(runner, "host_boundary", lambda _limit: {})
    return spec


@pytest.mark.parametrize("content", [True, False])
def test_unchanged_spec_is_bound_to_actual_local_files(tmp_path, monkeypatch, content) -> None:
    runner.verify_spec(_bound_spec(tmp_path, monkeypatch), content=content)


@pytest.mark.parametrize("target", ["weights", "shard_dir"])
def test_final_content_hash_detects_same_size_restored_mtime_write(
    tmp_path, monkeypatch, target,
) -> None:
    import os

    spec = _bound_spec(tmp_path, monkeypatch)
    path = Path(spec[target]) / "part.safetensors"
    stat = path.stat()
    path.write_bytes(b"modified")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="identity changed"):
        runner.verify_spec(spec, content=True)


@pytest.mark.parametrize("mutation", ["source", "inputs", "environment", "gpu"])
def test_changed_run_identity_refuses_qualification(tmp_path, monkeypatch, mutation) -> None:
    spec = _bound_spec(tmp_path, monkeypatch)
    if mutation == "source":
        monkeypatch.setattr(runner, "source_fingerprint", lambda: {"code.py": "changed"})
    elif mutation == "inputs":
        Path(spec["training_data_path"]).write_bytes(b"other tokens")
    elif mutation == "environment":
        monkeypatch.setattr(runner, "environment", lambda: {"packages": "other"})
    else:
        monkeypatch.setattr(runner, "hardware", lambda: {**spec["hardware"], "uuid": "GPU-two"})
    with pytest.raises(ValueError):
        runner.verify_spec(spec, content=True)


def test_worker_cannot_select_an_unfingerprinted_index(tmp_path, monkeypatch) -> None:
    spec = _bound_spec(tmp_path, monkeypatch)
    spec["index_path"] = str(Path(spec["shard_dir"]) / "other.json")
    Path(spec["index_path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprinted"):
        runner.verify_spec(spec, content=True)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_invalid_prepare_arguments_fail_before_cuda_or_sharding(monkeypatch, bad) -> None:
    args = runner.parse_args([
        "prepare", "--weights", "not-loaded", "--shards", "not-created", "--out", "not-created",
        "--host-cache-gib", str(bad), "--device-cache-gib", "0.25",
    ])
    monkeypatch.setattr(runner, "hardware", lambda: pytest.fail("invalid args reached CUDA"))
    with pytest.raises(ValueError):
        runner.prepare(args)


@pytest.mark.parametrize("total,swap,cap,swapcap,passes", [
    (64, 0, 32, 0, True), (32, 0, None, None, True),
    (64, 0, None, 0, False), (64, 1, 32, None, False), (64, 1, 32, 0, True),
])
def test_host_boundary_requires_a_real_cap_and_disabled_swap(
    monkeypatch, total, swap, cap, swapcap, passes,
) -> None:
    real_read = Path.read_text

    def read(path: Path, *args, **kwargs):
        if path.as_posix() == "/proc/meminfo":
            return f"MemTotal: {total * 2**20} kB\nSwapTotal: {swap * 2**20} kB\n"
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(runner.platform, "system", lambda: "Linux")
    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(runner, "_cgroup_limits", lambda: {
        "memory.max": None if cap is None else cap * runner.GIB,
        "memory.swap.max": swapcap, "paths": {},
    })
    if passes:
        report = runner.host_boundary(32 * runner.GIB)
        assert report["effective_limit_bytes"] == 32 * runner.GIB
        assert report["swap_disabled"]
    else:
        with pytest.raises(ValueError):
            runner.host_boundary(32 * runner.GIB)


@pytest.mark.parametrize("memory,capability,uuid,passes", [
    (8, 8, "GPU-one", True), (24, 8, "GPU-one", False),
    (8, 7, "GPU-one", False), (8, 8, "", False),
])
def test_physical_gpu_and_native_bf16_gate(monkeypatch, memory, capability, uuid, passes) -> None:
    import torch

    cuda = torch.cuda
    monkeypatch.setattr(cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda, "current_device", lambda: 0)
    monkeypatch.setattr(cuda, "get_device_properties", lambda _device: SimpleNamespace(
        total_memory=memory * runner.GIB, name="card", uuid=uuid,
    ))
    monkeypatch.setattr(cuda, "get_device_capability", lambda _device: (capability, 0))
    monkeypatch.setattr(
        cuda, "is_bf16_supported", lambda including_emulation: not including_emulation,
    )
    monkeypatch.setattr(
        runner.subprocess, "check_output", lambda *args, **kwargs: "GPU-one,card,8,1",
    )
    if passes:
        assert runner._hardware_in_process()["uuid"] == uuid
    else:
        with pytest.raises(ValueError):
            runner._hardware_in_process()


def test_parent_hardware_probe_runs_in_a_child_without_initializing_its_cuda(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("parent initialized CUDA"))
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout=json.dumps({"uuid": "GPU-child"}), stderr="",
    ))
    assert runner.hardware() == {"uuid": "GPU-child"}
