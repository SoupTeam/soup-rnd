"""Portable raw-archive replay verifies bytes and refuses unsafe/tampered extraction."""
import hashlib
import json
import sys
import zipfile

import pytest

from benchmarks.harness.d3_review_replay import file_hash, materialize, member_path
from benchmarks.harness.d3_trainer_analysis import cuda_kernel_names


def _archive(tmp_path, *, bare=False, tamper=False):
    path = tmp_path / "raw.zip"
    payload = b"raw"
    stamp = hashlib.sha256(payload).hexdigest()
    entry = stamp if bare else {"sha256": stamp, "bytes": len(payload)}
    manifest = {"session/record.json": entry}
    if not bare:
        manifest = {"payloads": manifest}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("session/record.json", b"bad" if tamper else payload)
        archive.writestr("MANIFEST.json", json.dumps(manifest))
    expected = {"bytes": path.stat().st_size, "sha256": file_hash(path), "payloads": 1}
    return path, expected


@pytest.mark.parametrize("bare", [False, True])
def test_both_original_manifest_formats_are_verified_and_reusable(tmp_path, bare):
    path, expected = _archive(tmp_path, bare=bare)
    output = tmp_path / "out"
    materialize(path, output, expected)
    assert (output / "session/record.json").read_bytes() == b"raw"
    materialize(path, output, expected)


def test_payload_hash_mismatch_never_becomes_an_extracted_record(tmp_path):
    path, expected = _archive(tmp_path, tamper=True)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="payload SHA-256 mismatch"):
        materialize(path, output, expected)
    assert not (output / "session/record.json").exists()


def test_changed_outer_archive_is_refused_before_creating_output(tmp_path):
    path, expected = _archive(tmp_path)
    expected["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="raw archive SHA-256 mismatch"):
        materialize(path, tmp_path / "out", expected)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("name", ["../outside", "/absolute", "C:/drive", "a\\escape"])
def test_unsafe_archive_names_cannot_escape_output(tmp_path, name):
    with pytest.raises(ValueError, match="unsafe archive member"):
        member_path(tmp_path, name)


def test_existing_replay_evidence_is_never_silently_overwritten(tmp_path):
    path, expected = _archive(tmp_path)
    output = tmp_path / "out"
    (output / "session").mkdir(parents=True)
    target = output / "session/record.json"
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="existing replay file differs"):
        materialize(path, output, expected)
    assert target.read_bytes() == b"changed"


@pytest.mark.parametrize("streamed", [False, True])
def test_profile_parser_preserves_every_ordered_duplicate_kernel(tmp_path, monkeypatch, streamed):
    if streamed:
        pytest.importorskip("ijson")
    else:
        monkeypatch.setitem(sys.modules, "ijson", None)
    path = tmp_path / "trace.json"
    events = [
        {"name": "ignored", "cat": "cpu_op", "ph": "X"},
        {"name": "first", "cat": "kernel", "ph": "X"},
        {"name": "fused", "cat": "kernel", "ph": "X"},
        {"name": "ignored", "cat": "kernel", "ph": "i"},
        {"name": "first", "cat": "kernel", "ph": "X"},
    ]
    path.write_text(json.dumps({"traceEvents": events}))
    assert cuda_kernel_names(path) == ["first", "fused", "first"]


def test_incremental_profile_parser_does_not_accept_truncated_json(tmp_path):
    pytest.importorskip("ijson")
    import ijson

    path = tmp_path / "trace.json"
    path.write_text('{"traceEvents":[{"name":"fused","cat":"kernel","ph":"X"}')
    with pytest.raises(ijson.JSONError):
        cuda_kernel_names(path)
