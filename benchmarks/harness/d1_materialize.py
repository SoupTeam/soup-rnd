"""Restore a hash-pinned reviewer archive, including exact duplicate files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

from d1_rules import digest, evidence_path
from rich.console import Console


def materialize(manifest_path: Path, output: Path, hardlink_duplicates: bool = False) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {row["file"]: row for row in manifest["files"]}
    if len(expected) != len(manifest["files"]) or not expected:
        raise ValueError("empty or duplicate file manifest")
    output.mkdir(parents=True, exist_ok=False)
    for name in expected:
        evidence_path(output, name)
    with tempfile.TemporaryFile() as joined:
        archive_hash = hashlib.sha256()
        for part in manifest["parts"]:
            path = evidence_path(manifest_path.parent, part["file"])
            if path.stat().st_size != part["bytes"] or digest(path) != part["sha256"]:
                raise ValueError("evidence part size or SHA-256 mismatch")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    archive_hash.update(chunk)
                    joined.write(chunk)
        if archive_hash.hexdigest() != manifest["archive_sha256"]:
            raise ValueError("joined archive SHA-256 mismatch")
        joined.seek(0)
        restored = set()
        with tarfile.open(fileobj=joined, mode="r:gz") as archive:
            for member in archive:
                target = evidence_path(output, member.name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if member.name not in expected or member.name in restored:
                    raise ValueError("unlisted or duplicate archive member")
                target.parent.mkdir(parents=True, exist_ok=True)
                if member.isfile():
                    if member.size != expected[member.name]["bytes"]:
                        raise ValueError("archive member size differs from manifest")
                    with archive.extractfile(member) as source, target.open("xb") as handle:
                        shutil.copyfileobj(source, handle)
                elif member.islnk():
                    if member.linkname not in restored:
                        raise ValueError("hard link target is not a previously verified file")
                    source = evidence_path(output, member.linkname)
                    if hardlink_duplicates:
                        os.link(source, target)
                    else:
                        with source.open("rb") as handle, target.open("xb") as destination:
                            shutil.copyfileobj(handle, destination)
                else:
                    raise ValueError("symlinks and special files are not evidence files")
                row = expected[member.name]
                if target.stat().st_size != row["bytes"] or digest(target) != row["sha256"]:
                    raise ValueError("restored evidence size or SHA-256 mismatch")
                restored.add(member.name)
        if restored != expected.keys():
            raise ValueError("archive is missing expected files")
    return {
        "status": "MATERIALIZED",
        "files": len(restored),
        "archive_sha256": manifest["archive_sha256"],
        "duplicate_storage": "filesystem hard links" if hardlink_duplicates else "copies",
        "scope": "Exact stored bytes only; run numerical replay separately.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardlink-duplicates", action="store_true")
    args = parser.parse_args()
    Console().print(materialize(args.manifest, args.output, args.hardlink_duplicates))


if __name__ == "__main__":
    main()
