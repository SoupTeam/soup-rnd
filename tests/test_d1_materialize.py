"""Reviewer evidence must materialize losslessly without archive traversal."""

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks/harness"))


def package(root, *, bad_name=None, symlink=False):
    archive = root / "proof.tgz"
    content = b"saved tensor bytes"
    with tarfile.open(archive, "w:gz") as handle:
        first = tarfile.TarInfo(bad_name or "qa/first.pt")
        first.size = len(content)
        handle.addfile(first, io.BytesIO(content))
        second = tarfile.TarInfo("qa/second.pt")
        second.type = tarfile.SYMTYPE if symlink else tarfile.LNKTYPE
        second.linkname = "qa/first.pt"
        handle.addfile(second)
    data = archive.read_bytes()
    parts = []
    for index, chunk in enumerate((data[:20], data[20:])):
        name = f"proof.part{index:03d}"
        (root / name).write_bytes(chunk)
        parts.append(
            {"file": name, "bytes": len(chunk), "sha256": hashlib.sha256(chunk).hexdigest()}
        )
    files = [
        {
            "file": f"qa/{name}.pt",
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for name in ("first", "second")
    ]
    manifest = root / "proof.json"
    manifest.write_text(
        json.dumps(
            {"parts": parts, "archive_sha256": hashlib.sha256(data).hexdigest(), "files": files}
        )
    )
    return manifest


def test_hard_links_restore_exact_duplicate_bytes(tmp_path):
    from d1_materialize import materialize

    result = materialize(package(tmp_path), tmp_path / "out")
    assert result["status"] == "MATERIALIZED"
    assert (tmp_path / "out/qa/first.pt").read_bytes() == b"saved tensor bytes"
    assert (tmp_path / "out/qa/second.pt").read_bytes() == b"saved tensor bytes"


@pytest.mark.parametrize("attack", ["traversal", "symlink", "tampered"])
def test_rejects_unsafe_or_changed_archive(tmp_path, attack):
    from d1_materialize import materialize

    manifest = package(
        tmp_path,
        bad_name="../outside.pt" if attack == "traversal" else None,
        symlink=attack == "symlink",
    )
    if attack == "tampered":
        (tmp_path / "proof.part001").write_bytes(b"changed")
    with pytest.raises(ValueError):
        materialize(manifest, tmp_path / "out")
    assert not (tmp_path / "outside.pt").exists()
