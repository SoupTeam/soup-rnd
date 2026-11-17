"""Archive corruption and escaping logical names cannot become a valid audit."""

import hashlib
import json
import zipfile

import pytest

from benchmarks.harness.d3_fused_trainer_publication import materialize


def archive_fixture(root):
    payload = b"retained raw bytes"
    digest = hashlib.sha256(payload).hexdigest()
    part = root / "evidence-01.zip"
    with zipfile.ZipFile(part, "w") as zipped:
        zipped.writestr("objects/" + digest, payload)
    manifest = {
        "parts": {
            part.name: {
                "bytes": part.stat().st_size,
                "sha256": hashlib.sha256(part.read_bytes()).hexdigest(),
            }
        },
        "objects": {
            digest: {"part": part.name, "member": "objects/" + digest, "bytes": len(payload)}
        },
        "logical_files": {
            name: {"sha256": digest, "bytes": len(payload)}
            for name in ("one/raw.bin", "two/raw.bin")
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return part, manifest, payload


def test_identical_logical_files_recover_original_bytes(tmp_path):
    _part, _manifest, payload = archive_fixture(tmp_path)
    out = tmp_path / "materialized"
    materialize(tmp_path, out)
    assert (out / "one/raw.bin").read_bytes() == payload
    assert (out / "two/raw.bin").read_bytes() == payload


def test_changed_archive_part_refused_before_materialization(tmp_path):
    part, _manifest, _payload = archive_fixture(tmp_path)
    part.write_bytes(part.read_bytes() + b"changed")
    out = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="part identity"):
        materialize(tmp_path, out)
    assert not out.exists()


def test_recomputed_part_hash_cannot_hide_changed_object(tmp_path):
    part, manifest, _payload = archive_fixture(tmp_path)
    digest = next(iter(manifest["objects"]))
    with zipfile.ZipFile(part, "w") as zipped:
        zipped.writestr("objects/" + digest, b"tampered raw bytes")
    manifest["parts"][part.name] = {
        "bytes": part.stat().st_size,
        "sha256": hashlib.sha256(part.read_bytes()).hexdigest(),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="payload hash"):
        materialize(tmp_path, tmp_path / "audit")


def test_logical_path_cannot_escape_the_materialization_root(tmp_path):
    _part, manifest, _payload = archive_fixture(tmp_path)
    manifest["logical_files"]["../escaped.bin"] = manifest["logical_files"].pop("one/raw.bin")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        materialize(tmp_path, tmp_path / "audit")
    assert not (tmp_path / "escaped.bin").exists()
