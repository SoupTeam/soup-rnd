"""CPU/archive replay of the source-pinned D3 trained-step study; no CUDA rerun."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

HEAD = "0611f46b4407a238ec549b0d8ce16ed4bd3b4c8c"
FULL_SHA = "101f57588dcac92f515225251250260fcbc5c2e22f24a4a9ae6d1efcbc24e7ba"


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify(full_archive, checkout, output):
    assert digest(full_archive) == FULL_SHA, "full archive SHA-256 differs"
    output.mkdir(exist_ok=False)
    with zipfile.ZipFile(full_archive) as archive:
        assert archive.testzip() is None, "ZIP CRC failure"
        manifest = json.loads(archive.read("SHA256SUMS.json"))
        assert set(archive.namelist()) == set(manifest) | {"SHA256SUMS.json"}
        for name, expected in manifest.items():
            target = os.path.realpath(output / name)
            root = os.path.realpath(output)
            assert os.path.commonpath((root, target)) == root
            archive.extract(name, output)
            path = output / name
            assert path.stat().st_size == expected["bytes"], name
            assert digest(path) == expected["sha256"], name
    source = output / "source"
    session = output / "session"
    receipt = json.loads((source / "FROZEN-D3-SOURCE.json").read_text())
    assert receipt["git_head"] == HEAD
    names = list(receipt["git_blob_sha256"])
    query = "".join(HEAD + ":" + name + "\n" for name in names).encode()
    result = subprocess.run(
        ["git", "-C", str(checkout), "cat-file", "--batch"],
        input=query, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    position = 0
    for name in names:
        end = result.stdout.index(b"\n", position)
        header = result.stdout[position:end].split()
        assert len(header) == 3 and header[1] == b"blob", name
        size = int(header[2])
        position = end + 1
        data = memoryview(result.stdout)[position:position + size]
        expected = receipt["git_blob_sha256"][name]
        assert hashlib.sha256(data).hexdigest() == expected, name
        assert digest(source / name) == expected, name
        position += size
        assert result.stdout[position:position + 1] == b"\n"
        position += 1
    del result
    sys.path.insert(0, str(source / "benchmarks/harness"))
    from d3_training_analysis import analyze

    replay = analyze(session)
    replay_path = output / "analysis-replay.json"
    encoder = json.JSONEncoder(indent=2, allow_nan=False)
    with replay_path.open("wb") as handle:
        for text in encoder.iterencode(replay):
            handle.write(text.encode())
        handle.write(b"\n")
    assert replay_path.stat().st_size == (session / "analysis.json").stat().st_size
    assert digest(replay_path) == digest(session / "analysis.json"), "analyzer replay differs"
    oracles = [name for name in manifest if name.startswith("session/correctness-")
               and "-native/step-" in name and name.endswith(".safetensors")]
    assert len(oracles) == 12
    return {
        "status": "verified_complete_study_pending_independent_review",
        "source_commit": HEAD,
        "full_archive_sha256": FULL_SHA,
        "payload_hashes_verified": len(manifest),
        "committed_source_blobs_verified": len(names),
        "native_oracle_files_hashed": len(oracles),
        "correctness_cells": replay["correctness_cells"],
        "timing_cells": replay["timing_cells"],
        "profile_cells": replay["profile_cells"],
        "analyzer_byte_identity_sha256": digest(replay_path),
        "analyzer_reproduces_archived_bytes": True,
        "fresh_cuda_execution": False,
        "d3_accepted": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--extract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.full, args.checkout, args.extract)
    args.out.write_bytes((json.dumps(report, indent=2) + "\n").encode())
