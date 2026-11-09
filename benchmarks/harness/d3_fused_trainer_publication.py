"""Verify deduplicated preparation archives and replay the synthetic launcher on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from benchmarks.harness import d3_fused_trainer_protocol as protocol
from benchmarks.harness import d3_trainer_protocol as historical
from benchmarks.harness.d3_fused_trainer_analysis import analyze
from benchmarks.harness.d3_trainer_gate import require_exact


def materialize(folder, destination):
    manifest = json.loads((folder / "manifest.json").read_text())
    if {row["sha256"] for row in manifest["logical_files"].values()} != set(manifest["objects"]):
        raise ValueError("unbound or missing archived object")
    for name, info in manifest["parts"].items():
        path = historical.contained(folder, name)
        if path.stat().st_size != info["bytes"] or historical.file_hash(path) != info["sha256"]:
            raise ValueError("archive part identity changed: " + name)
    destination.mkdir(parents=True, exist_ok=False)
    objects = destination / ".objects"
    objects.mkdir()
    seen = set()
    for name in manifest["parts"]:
        with zipfile.ZipFile(folder / name) as archive:
            expected = {
                info["member"] for info in manifest["objects"].values() if info["part"] == name
            }
            if set(archive.namelist()) != expected or len(archive.namelist()) != len(expected):
                raise ValueError("extra/duplicate/missing ZIP object")
            if archive.testzip() is not None:
                raise ValueError("ZIP CRC failure")
            for digest, info in manifest["objects"].items():
                if info["part"] != name:
                    continue
                if re.fullmatch("[0-9a-f]{64}", digest) is None:
                    raise ValueError("invalid object identity")
                payload = archive.read(info["member"])
                if len(payload) != info["bytes"] or hashlib.sha256(payload).hexdigest() != digest:
                    raise ValueError("archived payload hash mismatch")
                (objects / digest).write_bytes(payload)
                seen.add(digest)
    if seen != set(manifest["objects"]):
        raise ValueError("objects refer to an unknown archive part")
    for name, info in manifest["logical_files"].items():
        require_exact(info["bytes"], manifest["objects"][info["sha256"]]["bytes"], "payload size")
        target = historical.contained(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(objects / info["sha256"], target)
        except OSError:
            shutil.copyfile(objects / info["sha256"], target)
    return manifest


def qualify_sources(materialized, root=protocol.ROOT):
    receipt = json.loads((materialized / "source/D3-FUSED-SYNTHETIC-SOURCE.json").read_text())
    critical = [
        *[
            name
            for name in protocol.ENGINES
            if not name.endswith(("d3_fused_trainer_protocol.py", "d3_fused_trainer_analysis.py"))
        ],
        *[p.relative_to(root).as_posix() for p in (root / "src").rglob("*.py")],
    ]
    for name in critical:
        expected = receipt["source_files"][name]["lf_sha256"]
        actual = hashlib.sha256(
            historical.contained(root, name).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        require_exact(actual, expected, "tested numerical/launcher source: " + name)
    return receipt


def verify(folder, destination):
    if (folder / "SHA256SUMS.json").is_file():
        for name, expected in json.loads((folder / "SHA256SUMS.json").read_text()).items():
            require_exact(
                historical.file_hash(historical.contained(folder, name)),
                expected,
                "publication checksum",
            )
    manifest = materialize(folder, destination)
    session = destination / "data/synthetic-01/inputs/launcher-03"
    result = analyze(session)
    require_exact(
        result, json.loads((session / "analysis.json").read_text()), "analyzer byte values"
    )
    qualify_sources(destination)
    return {
        "kind": "offline preparation evidence audit; no new CUDA measurement",
        "logical_files": len(manifest["logical_files"]),
        "unique_objects": len(manifest["objects"]),
        "archive_parts": len(manifest["parts"]),
        "synthetic_processes": result["fresh_processes"],
        "native_tensor_payloads": result["native_tensor_payloads_verified"],
        "status": result["status"],
        "formal_pretrained_execution": False,
    }


if __name__ == "__main__":
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--materialize", type=Path)
    args = parser.parse_args()
    if args.materialize:
        result = verify(args.folder, args.materialize)
    else:
        with tempfile.TemporaryDirectory(prefix="d3-fused-evidence-") as temporary:
            result = verify(args.folder, Path(temporary) / "evidence")
    Console().print_json(data=result)
