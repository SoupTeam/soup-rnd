"""Preparation boundaries only. No live network, model execution or GPU is needed."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.harness import p3_deepseek_quality_model as model_api
from benchmarks.harness import p3_deepseek_quality_prepare as prepare


def _hub_info() -> SimpleNamespace:
    inventory = model_api.source_inventory()
    names = [*sorted(prepare.WEIGHTS), *sorted(set(inventory) - prepare.WEIGHTS)]
    rows = []
    for name in names:
        row = inventory[name]
        lfs = (None if row["hub_sha256"] is None else
               SimpleNamespace(sha256=row["hub_sha256"], size=row["bytes"]))
        rows.append(SimpleNamespace(
            rfilename=name, size=row["bytes"], lfs=lfs, blob_id=row["hub_etag"]))
    return SimpleNamespace(sha=model_api.REVISION, siblings=rows)


def test_pinned_inventory_has_complete_weight_coverage() -> None:
    records = prepare.hub_inventory(_hub_info())
    assert len([name for name in records if name.endswith(".safetensors")]) == 163
    assert sum(row["bytes"] for name, row in records.items()
               if name.endswith(".safetensors")) == 688_586_727_753
    assert records["model-00001-of-000163.safetensors"]["hub_sha256"] == (
        "b933b099f335d0cc95380200bbc3d1dc4b240f0c81e6de1a192b4921142403f8")


@pytest.mark.parametrize("mutation", ["revision", "missing", "extra", "total", "duplicate",
                                      "hashless", "lfs_size", "malformed", "tokenizer"])
def test_bad_hub_metadata_refuses_before_download(mutation: str) -> None:
    info = _hub_info()
    if mutation == "revision":
        info.sha = "0" * 40
    elif mutation == "missing":
        info.siblings.pop(0)
    elif mutation == "extra":
        info.siblings.append(SimpleNamespace(rfilename="extra.safetensors", size=1))
    elif mutation == "total":
        info.siblings[0].size += 1
        info.siblings[0].lfs.size += 1
    elif mutation == "duplicate":
        info.siblings.append(info.siblings[0])
    elif mutation == "hashless":
        info.siblings[0].lfs.sha256 = None
    elif mutation == "lfs_size":
        info.siblings[0].lfs.size += 1
    elif mutation == "malformed":
        info.siblings[0].size = True
    else:
        info.siblings = [row for row in info.siblings if row.rfilename != "tokenizer.json"]
    with pytest.raises(ValueError):
        prepare.hub_inventory(info)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/escape", "a\\b", "a/../b",
                                  "a//b", "./config.json"])
def test_untrusted_hub_paths_are_rejected(tmp_path: Path, name: str) -> None:
    info = _hub_info()
    info.siblings.append(SimpleNamespace(rfilename=name, size=10, lfs=None, blob_id="b" * 40))
    with pytest.raises(ValueError):
        prepare.hub_inventory(info)
    with pytest.raises(ValueError):
        prepare.safe_child(tmp_path, name)


@pytest.mark.parametrize("field,value", [("commit_hash", "0" * 40), ("etag", "0" * 64),
                                         ("size", 1), ("etag", None),
                                         ("location", "http://unsafe.invalid/file")])
def test_head_metadata_must_match_commit_etag_and_size(field: str, value: object) -> None:
    descriptor = prepare.hub_inventory(_hub_info())["model-00001-of-000163.safetensors"]
    metadata = SimpleNamespace(commit_hash=model_api.REVISION, etag=f"{1:064x}",
                               size=4_000_000_000, location="https://cdn.example/file")
    setattr(metadata, field, value)
    with pytest.raises(ValueError):
        prepare.verify_head(metadata, descriptor)


def _disk_fixture(root: Path, *, filesystem: str = "ext4", nvme: bool = True) -> tuple:
    data = root / "data"
    data.mkdir()
    proc = root / "proc"
    (proc / "self").mkdir(parents=True)
    sysfs = root / "sys"
    (sysfs / "dev/block").mkdir(parents=True)
    disk = sysfs / "devices/pci0000:00/0000:00:01.0/nvme/nvme0/nvme0n1"
    (disk / "queue").mkdir(parents=True)
    (disk / "queue/rotational").write_text("0")
    (disk / "dev").write_text("259:0")
    (disk.parent / "transport").write_text("pcie" if nvme else "tcp")
    (disk.parent / "subsystem").symlink_to(sysfs / "class/nvme", target_is_directory=True)
    (sysfs / "dev/block/259:0").symlink_to(disk, target_is_directory=True)
    mount = str(data).replace("\\", "/")
    (proc / "self/mountinfo").write_text(
        f"31 20 259:0 / {mount} rw - {filesystem} /dev/not-a-marketing-name rw\n")
    return data, proc, sysfs


@pytest.mark.skipif(os.name == "nt", reason="Linux sysfs and mount fixtures use symlinks")
@pytest.mark.parametrize("filesystem,nvme", [("overlay", True), ("nfs", True), ("ext4", False)])
def test_overlay_network_and_nonlocal_nvme_refuse(
    tmp_path: Path, filesystem: str, nvme: bool,
) -> None:
    data, proc, sysfs = _disk_fixture(tmp_path, filesystem=filesystem, nvme=nvme)
    with pytest.raises(ValueError):
        prepare.inspect_nvme(data, proc=proc, sysfs=sysfs, device_number="259:0")


@pytest.mark.skipif(os.name == "nt", reason="Linux sysfs and mount fixtures use symlinks")
def test_disk_provenance_is_bound_to_target_mount(tmp_path: Path) -> None:
    data, proc, sysfs = _disk_fixture(tmp_path)
    result = prepare.inspect_nvme(data, proc=proc, sysfs=sysfs, device_number="259:0")
    assert result["major_minor"] == "259:0"
    assert result["transport"] == "pcie"
    with pytest.raises(ValueError):
        prepare.inspect_nvme(data, proc=proc, sysfs=sysfs, device_number="8:0")


@pytest.mark.parametrize("free,ram", [(1_399_999_999_999, 128 * 2**30),
                                    (1_400_000_000_000, 64 * 2**30 - 1)])
def test_space_and_effective_ram_refuse(free: int, ram: int) -> None:
    with pytest.raises(ValueError):
        prepare.require_capacity(free, {"effective_bytes": ram})


def test_effective_ram_uses_tighter_cgroup_limit(tmp_path: Path, monkeypatch) -> None:
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "meminfo").write_text("MemTotal: 134217728 kB\n")
    (proc / "self/cgroup").write_text("0::/worker\n")
    group = tmp_path / "cgroup"
    (group / "worker").mkdir(parents=True)
    (group / "memory.max").write_text(str(32 * 2**30))
    (group / "memory.current").write_text("0")
    (group / "memory.stat").write_text("inactive_file 0\n")
    (group / "worker/memory.max").write_text("max")
    memory = prepare.inspect_ram(proc=proc, v2_root=group, v1_root=tmp_path / "v1")
    assert memory["host_bytes"] == 128 * 2**30
    assert memory["effective_bytes"] == 32 * 2**30
    with pytest.raises(ValueError):
        prepare.require_capacity(2_000_000_000_000, memory)


def test_hashless_small_metadata_is_rejected() -> None:
    info = _hub_info()
    info.siblings[-1].blob_id = None
    with pytest.raises(ValueError):
        prepare.hub_inventory(info)


def test_small_metadata_git_blob_and_local_sha_are_distinct(tmp_path: Path) -> None:
    content = b'{"model_type":"deepseek_v3"}'
    path = tmp_path / "config.json"
    path.write_bytes(content)
    git_hash = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
    result = prepare.local_descriptor(path, {"bytes": len(content), "hub_etag": git_hash,
                                             "hub_sha256": None})
    assert result["sha256"] == hashlib.sha256(content).hexdigest()
    assert result["hub_etag"] == git_hash
    assert result["sha256_origin"] == "locally_measured_metadata"
    path.write_bytes(b"x" * len(content))
    with pytest.raises(ValueError):
        prepare.local_descriptor(path, {"bytes": len(content), "hub_etag": git_hash,
                                        "hub_sha256": None})


def _sparse_source(root: Path) -> Path:
    source = root / model_api.REVISION
    source.mkdir()
    config = {"model_type": "deepseek_v3", "hidden_size": 7168, "num_hidden_layers": 61,
              "n_routed_experts": 256, "num_experts_per_tok": 8, "first_k_dense_replace": 3,
              "vocab_size": 129280, "q_lora_rank": 1536, "kv_lora_rank": 512}
    (source / "config.json").write_text(json.dumps(config))
    names = [f"model-{number:05d}-of-000163.safetensors" for number in range(1, 164)]
    (source / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {f"tensor.{number}": name for number, name in enumerate(names)}}))
    records = {}
    for number, name in enumerate(names, 1):
        size = 4_000_000_000 if number < 163 else 40_586_727_753
        with (source / name).open("wb") as stream:
            stream.truncate(size)
        records[name] = prepare.local_descriptor(source / name, {
            "bytes": size, "hub_etag": f"{number:064x}", "hub_sha256": f"{number:064x}"})
    for name in ("config.json", "model.safetensors.index.json"):
        payload = (source / name).read_bytes()
        etag = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        records[name] = prepare.local_descriptor(source / name, {
            "bytes": len(payload), "hub_etag": etag, "hub_sha256": None})
    prepare.write_json(source / "checkpoint-identity.json", prepare.source_receipt(records))
    return source


@pytest.mark.skipif(os.name == "nt", reason="688GB sparse logical files on Linux only")
def test_invented_sparse_receipt_cannot_claim_production(tmp_path: Path) -> None:
    source = _sparse_source(tmp_path)
    config = json.loads((source / "config.json").read_text())
    with pytest.raises(ValueError, match="inventory|provenance"):
        model_api._source_identity(source, config)


def test_duplicate_output_and_source_are_never_replaced(tmp_path: Path) -> None:
    output = tmp_path / "receipt"
    output.mkdir()
    sentinel = output / "keep"
    sentinel.write_text("immutable")
    args = SimpleNamespace(approved_pilot=True, output=str(output), source=str(tmp_path / "bad"))
    with pytest.raises(FileExistsError):
        prepare.download(args)
    assert sentinel.read_text() == "immutable"


def test_acknowledgement_is_required_without_side_effects(tmp_path: Path) -> None:
    for command in (prepare.download, prepare.convert):
        args = SimpleNamespace(approved_pilot=False, output=str(tmp_path / "receipt"))
        with pytest.raises(ValueError, match="approved-pilot"):
            command(args)
        assert not (tmp_path / "receipt").exists()


class _Observation:
    def __init__(self, *args, **kwargs):
        self.result = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.result = {"process_rss_sampled_peak_bytes": 123, "cuda_allocated_peak_bytes": 456,
                       "cuda_reserved_peak_bytes": 789, "sampling_errors": []}


@pytest.mark.parametrize("failure", [None, "adapter", "admission", "close", "open",
                                     "telemetry", "threads"])
def test_conversion_retains_failure_closes_session_and_restores_threads(
    tmp_path: Path, monkeypatch, failure: str | None,
) -> None:
    import sys

    source = tmp_path / model_api.REVISION
    source.mkdir()
    cache = tmp_path / "cache"
    output = tmp_path / "convert"
    state = {"threads": 3, "closed": False, "writes": 0}

    def set_threads(value):
        state["threads"] = value
        if failure == "threads" and value == 3:
            raise RuntimeError("thread restoration failure")

    torch = SimpleNamespace(get_num_threads=lambda: state["threads"],
                            set_num_threads=set_threads)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(prepare, "admit_machine", lambda *args, **kwargs: {"admitted": True})
    monkeypatch.setattr(prepare, "admit_source", lambda path: {"production_qualified": True})
    monkeypatch.setattr(prepare, "memory_observation", _Observation)

    def cgroup():
        if failure == "telemetry":
            raise OSError("unreadable cgroup peak")
        return {"peak_scope": "lifetime"}

    monkeypatch.setattr(prepare, "cgroup_observation", cgroup)
    monkeypatch.setattr(prepare, "require_ampere_cuda", lambda device: {"name": "test Ampere"})
    from soup_cli.utils import expert_cache_budget

    monkeypatch.setattr(expert_cache_budget, "new_expert_cache_budget", lambda paths: 1)

    def close():
        state["closed"] = True
        if failure == "close":
            raise RuntimeError("cleanup failure")

    def open_session(source_arg, cache_arg, **kwargs):
        assert kwargs == {"device": "cuda", "dtype": "bfloat16", "prepare": True}
        assert state["threads"] == 8
        cache_arg.mkdir()
        (cache_arg / "partial").write_text("retained")
        if failure == "open":
            raise RuntimeError("conversion failure")
        return SimpleNamespace(model=object(), close=close, closed=True,
                               metadata={"source": {}, "cache": {}})

    def write_adapter(model, path, *, seed):
        assert seed == 23
        state["writes"] += 1
        if failure == "adapter":
            raise RuntimeError("adapter failure")
        path.write_bytes(b"adapter")
        return {"sha256": hashlib.sha256(b"adapter").hexdigest(), "seed": 23}

    def require_source(metadata):
        if failure == "admission":
            raise ValueError("production admission failure")

    monkeypatch.setattr(model_api, "open_session", open_session)
    monkeypatch.setattr(model_api, "write_initial_adapter", write_adapter)
    monkeypatch.setattr(model_api, "require_production_source", require_source)
    args = SimpleNamespace(approved_pilot=True, source=str(source), cache=str(cache),
                           output=str(output))
    if failure:
        with pytest.raises((RuntimeError, ValueError)):
            prepare.convert(args)
    else:
        prepare.convert(args)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == ("FAILED" if failure else "PASS")
    assert manifest["operator_assertion"]["external_human_approval_verified"] is False
    assert manifest["resources"]["memory"]["process_rss_sampled_peak_bytes"] == 123
    assert state["threads"] == 3
    assert state["closed"] is (failure != "open")
    assert state["writes"] == (0 if failure in {"admission", "open"} else 1)
    assert (cache / "partial").read_text() == "retained"
    with pytest.raises(FileExistsError):
        prepare.convert(args)


def test_existing_cache_refuses_before_session(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / model_api.REVISION
    source.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "immutable").write_text("prior attempt")
    args = SimpleNamespace(approved_pilot=True, source=str(source), cache=str(cache),
                           output=str(tmp_path / "receipt"))
    with pytest.raises(FileExistsError):
        prepare.convert(args)
    assert (cache / "immutable").read_text() == "prior attempt"
    assert json.loads((tmp_path / "receipt/manifest.json").read_text())["status"] == "FAILED"


def test_parser_exposes_only_separate_download_and_convert() -> None:
    parser = prepare.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["train"])
    args = parser.parse_args(["download", "--source", "/data/" + model_api.REVISION,
                              "--output", "/data/download-receipt", "--approved-pilot"])
    assert args.command == "download"
    assert args.approved_pilot is True


@pytest.mark.parametrize("source_name", ["wrongrevision", "e815299b"])
def test_wrong_source_basename_records_failure(tmp_path: Path, source_name: str) -> None:
    output = tmp_path / "receipt"
    args = SimpleNamespace(approved_pilot=True, source=str(tmp_path / source_name),
                           output=str(output))
    with pytest.raises(ValueError, match="basename"):
        prepare.download(args)
    assert json.loads((output / "manifest.json").read_text())["status"] == "FAILED"


def test_existing_download_destination_is_preserved(tmp_path: Path) -> None:
    source = tmp_path / model_api.REVISION
    source.mkdir()
    (source / "immutable").write_text("prior attempt")
    args = SimpleNamespace(approved_pilot=True, source=str(source),
                           output=str(tmp_path / "receipt"))
    with pytest.raises(FileExistsError):
        prepare.download(args)
    assert (source / "immutable").read_text() == "prior attempt"
    assert json.loads((tmp_path / "receipt/manifest.json").read_text())["status"] == "FAILED"


def test_nested_receipt_cannot_modify_source(tmp_path: Path) -> None:
    source = tmp_path / model_api.REVISION
    source.mkdir()
    output = source / "nested-receipt"
    args = SimpleNamespace(approved_pilot=True, source=str(source), output=str(output))
    with pytest.raises(ValueError):
        prepare.download(args)
    assert not output.exists()


def test_unreadable_cgroup_limit_fails_closed(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "meminfo").write_text("MemTotal: 134217728 kB\n")
    (proc / "self/cgroup").write_text("0::/worker\n")
    group = tmp_path / "cgroup"
    (group / "worker").mkdir(parents=True)
    (group / "worker/memory.max").write_text("malformed")
    with pytest.raises(ValueError):
        prepare.inspect_ram(proc=proc, v2_root=group, v1_root=tmp_path / "v1")


@pytest.mark.parametrize("failure", [None, "payload", "head", "metadata"])
def test_download_writes_identity_only_after_all_payloads_and_retains_failure(
    tmp_path: Path, monkeypatch, failure: str | None,
) -> None:
    import sys

    info = _hub_info()
    contents = {
        "config.json": json.dumps({"model_type": "deepseek_v3", "hidden_size": 7168,
                                  "num_hidden_layers": 61, "n_routed_experts": 256,
                                  "num_experts_per_tok": 8, "first_k_dense_replace": 3,
                                  "vocab_size": 129280, "q_lora_rank": 1536,
                                  "kv_lora_rank": 512}).encode(),
        "generation_config.json": b'{"eos_token_id":1}',
        "tokenizer.json": b'{"model":{"type":"BPE"}}',
        "tokenizer_config.json": b'{"eos_token":"EOS"}',
        "model.safetensors.index.json": json.dumps({"weight_map": {
            f"tensor.{number}": f"model-{number:05d}-of-000163.safetensors"
            for number in range(1, 164)}}).encode(),
    }
    for row in info.siblings:
        if row.rfilename in contents:
            payload = contents[row.rfilename]
            row.size = len(payload)
            row.blob_id = hashlib.sha1(
                b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
    fixture_pin = {
        row.rfilename: {"bytes": row.size,
                        "hub_etag": row.lfs.sha256 if row.lfs is not None else row.blob_id,
                        "hub_sha256": row.lfs.sha256 if row.lfs is not None else None}
        for row in info.siblings
    }
    monkeypatch.setattr(model_api, "source_inventory", lambda: fixture_pin)
    inventory = prepare.hub_inventory(info)
    state = {"shards": 0, "heads": 0}

    def head(url, *, token):
        state["heads"] += 1
        name = url.rsplit("/", 1)[1]
        row = inventory[name]
        return SimpleNamespace(commit_hash=("0" * 40 if failure == "head" else model_api.REVISION),
                               etag=row["hub_etag"], size=row["bytes"],
                               location="https://cdn.example/" + name)

    def payload(url, path, descriptor):
        if path.name.endswith(".safetensors"):
            state["shards"] += 1
            if failure == "payload":
                path.write_bytes(b"partial payload")
                raise OSError("network interrupted")
            path.write_bytes(b"one bounded fixture payload")
        else:
            path.write_bytes(b"invalid" if failure == "metadata" else contents[path.name])

    real_local = prepare.local_descriptor

    def local(path, row):
        if path.name.endswith(".safetensors"):
            return {**row, "sha256": row["hub_sha256"],
                    "sha256_origin": "hub_lfs_content_identity_not_locally_hashed"}
        return real_local(path, row)

    fake_hub = SimpleNamespace(
        HfApi=lambda **kwargs: SimpleNamespace(model_info=lambda *args, **kwargs: info),
        get_hf_file_metadata=head,
        hf_hub_url=lambda model, name, **kwargs: "https://hub.example/" + name,
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setattr(prepare, "admit_machine", lambda *args, **kwargs: {"admitted": True})
    monkeypatch.setattr(prepare, "admit_source", lambda source: {"production_qualified": True})
    monkeypatch.setattr(prepare, "_stream_download", payload)
    monkeypatch.setattr(prepare, "local_descriptor", local)
    source, output = tmp_path / model_api.REVISION, tmp_path / "receipt"
    args = SimpleNamespace(approved_pilot=True, source=str(source), output=str(output))
    if failure:
        with pytest.raises((OSError, ValueError)):
            prepare.download(args)
    else:
        prepare.download(args)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == ("FAILED" if failure else "PASS")
    assert (source / "checkpoint-identity.json").exists() is (failure is None)
    assert state["shards"] == (1 if failure == "payload" else 0 if failure else 163)
    if failure == "payload":
        assert (source / "model-00001-of-000163.safetensors").read_bytes() == b"partial payload"


@pytest.mark.parametrize("length,payload", [("5", b"short"), ("4", b"too long"),
                                          (None, b"bad"), (None, b"excess")])
def test_streamed_download_rejects_unverified_response(
    tmp_path: Path, monkeypatch, length: str | None, payload: bytes,
) -> None:
    import io

    class Response(io.BytesIO):
        status = 200
        url = "https://cdn.example/file"
        headers = {"Content-Length": length}

    monkeypatch.setattr(prepare, "_open_https", lambda *args: Response(payload))
    with pytest.raises(ValueError):
        prepare._stream_download("https://cdn.example/file", tmp_path / "file",
                                 {"bytes": 4, "hub_etag": "e" * 64})


@pytest.mark.parametrize("reason", ["overlay", "nonNVMe", "lowspace", "low-effectiveRAM"])
def test_download_admission_precedes_every_hub_call(
    tmp_path: Path, monkeypatch, reason: str,
) -> None:
    import sys

    def refuse(*args, **kwargs):
        raise ValueError(reason)

    def forbidden(**kwargs):
        pytest.fail("network client constructed before machine admission")

    monkeypatch.setattr(prepare, "admit_machine", refuse)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=forbidden))
    source, output = tmp_path / model_api.REVISION, tmp_path / "receipt"
    args = SimpleNamespace(approved_pilot=True, source=str(source), output=str(output))
    with pytest.raises(ValueError, match=reason):
        prepare.download(args)
    assert not source.exists()
    assert json.loads((output / "manifest.json").read_text())["status"] == "FAILED"


def test_hidden_cgroup_controller_is_not_unlimited_ram(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "meminfo").write_text("MemTotal: 134217728 kB\n")
    (proc / "self/cgroup").write_text("0::/worker\n")
    with pytest.raises(ValueError, match="hidden|unavailable"):
        prepare.inspect_ram(proc=proc, v2_root=tmp_path / "missing", v1_root=tmp_path / "v1")


@pytest.mark.skipif(os.name == "nt", reason="Linux sysfs and mount fixtures use symlinks")
def test_sata_device_cannot_borrow_unrelated_nvme_provenance(tmp_path: Path) -> None:
    data, proc, sysfs = _disk_fixture(tmp_path)
    disk = sysfs / "devices/pci0000:00/0000:00:01.0/nvme/nvme0/nvme0n1"
    (disk.parent / "subsystem").unlink()
    (disk.parent / "subsystem").symlink_to(sysfs / "class/scsi_host", target_is_directory=True)
    with pytest.raises(ValueError, match="NVMe"):
        prepare.inspect_nvme(data, proc=proc, sysfs=sysfs, device_number="259:0")


def test_small_metadata_over_hashing_bound_refuses() -> None:
    info = _hub_info()
    info.siblings[-1].size = 64 * 2**20
    with pytest.raises(ValueError, match="bound"):
        prepare.hub_inventory(info)


def test_exclusive_json_receipt_cannot_be_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    prepare.write_json(path, {"status": "FAILED"})
    with pytest.raises(FileExistsError):
        prepare.write_json(path, {"status": "PASS"})
    assert json.loads(path.read_text()) == {"status": "FAILED"}


@pytest.mark.skipif(os.name == "nt", reason="Linux fixture symlink")
def test_safe_child_rejects_symlink_escape(tmp_path: Path) -> None:
    source, elsewhere = tmp_path / "source", tmp_path / "elsewhere"
    source.mkdir()
    elsewhere.mkdir()
    (source / "nested").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError):
        prepare.safe_child(source, "nested/config.json")


def test_effective_ram_honors_v1_limit(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "meminfo").write_text("MemTotal: 134217728 kB\n")
    (proc / "self/cgroup").write_text("4:memory:/worker\n")
    group = tmp_path / "memory"
    (group / "worker").mkdir(parents=True)
    (group / "worker/memory.limit_in_bytes").write_text(str(48 * 2**30))
    (group / "worker/memory.usage_in_bytes").write_text("0")
    (group / "worker/memory.stat").write_text("total_inactive_file 0\n")
    result = prepare.inspect_ram(proc=proc, v2_root=tmp_path / "v2", v1_root=group)
    assert result["effective_bytes"] == 48 * 2**30


def test_source_admission_rejects_tiny_checkpoint_before_model_load(tmp_path: Path) -> None:
    source = tmp_path / model_api.REVISION
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"deepseek_v3"}')
    (source / "model-00001-of-000163.safetensors").write_bytes(b"not production")
    with pytest.raises(ValueError, match="production"):
        prepare.admit_source(source)
