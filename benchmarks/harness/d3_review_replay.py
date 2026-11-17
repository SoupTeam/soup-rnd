"""Portable CPU replay of the two complete pretrained D3 raw archives; no GPU run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import stat
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

from rich.console import Console

ROOT = Path(os.path.realpath(__file__)).parents[2]
STUDIES = {
    "staged": {
        "bytes": 5173245384,
        "sha256": "73383ef1cc1cdd9a2182dbdf9f7ed639eb7c0238074a4c46f2a68af9bf285b0a",
        "head": "f781e92726c9e6b524d75e7e7f3e1058350ba2c6",
        "payloads": 6119,
        "native_tensors": 4362,
        "publication": "d3-public-trainer-attempt-03-f781e927-20261030",
    },
    "fused": {
        "bytes": 2942937841,
        "sha256": "3221d8ec19dbb01cbb1cadce9927f5cb284250e6a8070dd98328ddb6152fe6c1",
        "head": "921d86a19d96e62ecc12c7c6aef49074f3ebb14d",
        "payloads": 4131,
        "native_tensors": 2908,
        "publication": "d3-fused-trained-921d86a1-20261101",
    },
}
console = Console()


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def member_path(root: Path, name: str) -> Path:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or "\\" in name or ":" in name or not name:
        raise ValueError(f"unsafe archive member: {name!r}")
    target = os.path.realpath(root / name)
    if os.path.commonpath((os.path.realpath(root), target)) != os.path.realpath(root):
        raise ValueError("archive member escapes replay directory")
    return Path(target)


def materialize(archive_path: Path, output: Path, expected: dict) -> dict:
    if archive_path.stat().st_size != expected["bytes"]:
        raise ValueError("wrong or incomplete archive length")
    if file_hash(archive_path) != expected["sha256"]:
        raise ValueError("raw archive SHA-256 mismatch")
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("MANIFEST.json"))
        # The earlier staged archive used a bare path -> SHA map; fused also
        # records each payload's length. Preserve both original manifest formats.
        payloads = manifest.get("payloads", manifest)
        if len(payloads) != expected["payloads"]:
            raise ValueError("unexpected payload count")
        names = archive.namelist()
        if (len(names) != len(set(name.casefold() for name in names))
                or set(names) != set(payloads) | {"MANIFEST.json"}):
            raise ValueError("duplicate or undeclared archive members")
        output.mkdir(parents=True, exist_ok=True)
        for index, (name, record) in enumerate(payloads.items(), 1):
            target = member_path(output, name)
            info = archive.getinfo(name)
            if isinstance(record, str):
                record = {"sha256": record, "bytes": info.file_size}
            if stat.S_ISLNK(info.external_attr >> 16) or info.file_size != record["bytes"]:
                raise ValueError("symlink or changed archive member size")
            if target.exists():
                if not target.is_file() or file_hash(target) != record["sha256"]:
                    raise ValueError(f"existing replay file differs: {target}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".replay-part")
            hasher = hashlib.sha256()
            with archive.open(info) as source, temporary.open("wb") as dest:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    hasher.update(block)
                    dest.write(block)
            if hasher.hexdigest() != record["sha256"]:
                raise ValueError(f"raw payload SHA-256 mismatch: {name}")
            temporary.replace(target)
            if index % 1000 == 0:
                console.print(f"Verified/extracted {index}/{len(payloads)} payloads")
        (output / "MANIFEST.json").write_bytes(archive.read("MANIFEST.json"))
    return manifest


def verify_git_source(source: dict, head: str) -> int:
    if source["git_head"] != head:
        raise ValueError("wrong measured source revision")
    blobs = source["git_blob_sha256"]
    request = "".join(f"{head}:{name}\n" for name in blobs).encode()
    result = subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "--batch"], input=request,
        capture_output=True, check=True,
    )
    stream = io.BytesIO(result.stdout)
    for name, expected in blobs.items():
        header = stream.readline().split()
        if len(header) != 3 or header[1] != b"blob":
            raise ValueError(f"missing measured Git blob: {head}:{name}")
        payload = stream.read(int(header[2]))
        if stream.read(1) != b"\n" or hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"measured Git source mismatch: {name}")
    return len(blobs)


def replay(kind: str, archive: Path, output: Path) -> dict:
    import torch

    from benchmarks.harness.d3_trainer_gate import save

    expected = STUDIES[kind]
    console.print(f"Checking complete {kind} raw archive")
    materialize(archive, output, expected)
    session = output / "session"
    info = json.loads((session / "session.json").read_text())
    source_count = verify_git_source(info["source"], expected["head"])
    torch.set_num_threads(1)
    if kind == "staged":
        from benchmarks.harness.d3_trainer_analysis import analyze
    else:
        from benchmarks.harness.d3_fused_trainer_analysis import analyze
    result = analyze(session)
    if result["native_tensor_payloads_verified"] != expected["native_tensors"]:
        raise ValueError("native tensor replay is incomplete")
    actual = output / "replayed-analysis.json"
    save(actual, result)
    published = ROOT / "benchmarks/results" / expected["publication"] / "analysis.json"
    if actual.read_bytes() != published.read_bytes():
        raise ValueError("replayed analysis differs from the published bytes")
    receipt = {
        "study": kind, "measured_commit": expected["head"],
        "archive_sha256": expected["sha256"], "archive_bytes": expected["bytes"],
        "payloads_verified": expected["payloads"], "measured_git_blobs_verified": source_count,
        "native_tensor_payloads_verified": expected["native_tensors"],
        "analysis_byte_equality": True, "analysis_sha256": file_hash(actual),
        "status": result["status"], "new_gpu_execution": False,
    }
    save(output / "replay-verification.json", receipt)
    console.print(json.dumps(receipt, indent=2))
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", type=Path)
    parser.add_argument("--fused", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.staged and not args.fused:
        parser.error("supply --staged and/or --fused")
    if importlib.util.find_spec("ijson") is None:
        parser.error("install ijson for bounded-memory CUDA trace replay")
    output = Path(os.path.realpath(args.out))
    try:
        inside_repo = os.path.commonpath((str(ROOT), str(output))) == str(ROOT)
    except ValueError:
        inside_repo = False
    if inside_repo:
        parser.error("--out must be outside the Git checkout (multi-GB evidence)")
    for kind in STUDIES:
        archive = getattr(args, kind)
        if archive is not None:
            replay(kind, archive, output / kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
