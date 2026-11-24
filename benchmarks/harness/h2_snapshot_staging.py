"""Bounded, non-resuming transport for an immutable, previously verified snapshot.

Transport success proves bytes only. The existing structural verifier must still run.
Each file uses one request, avoiding thousands of range-request round trips.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path


def validate_manifest(manifest: dict) -> None:
    """Reject ambiguous identities and paths before creating a destination."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", manifest["repo"]):
        raise ValueError("Invalid repository")
    if not re.fullmatch(r"[a-f0-9]{40}", manifest["revision"]):
        raise ValueError("Immutable commit required")
    sizes, hashes = manifest["files_bytes"], manifest["files_sha256"]
    if not sizes or sizes.keys() != hashes.keys():
        raise ValueError("Manifest file sets differ")
    for name, size in sizes.items():
        if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", name):
            raise ValueError("Only flat regular filenames supported")
        if type(size) is not int or size <= 0:
            raise ValueError("Invalid file size")
        if not re.fullmatch(r"[a-f0-9]{64}", hashes[name]):
            raise ValueError("Invalid SHA256")
    if sum(sizes.values()) != manifest["total_bytes"]:
        raise ValueError("Total differs")


def stage(
    manifest: dict,
    destination: Path,
    evidence: Path,
    deadline_seconds: float = 1800,
    *,
    opener=urllib.request.urlopen,
) -> dict:
    """Download once; fail closed, preserving partials and a receipt on failure."""
    validate_manifest(manifest)
    if not 0 < deadline_seconds <= 3600:
        raise ValueError("Deadline must be positive and at most one hour")
    evidence.mkdir(parents=True, exist_ok=False)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    stop, lock = threading.Event(), threading.Lock()
    progress = {}

    def check() -> None:
        if stop.is_set() or time.monotonic() - started >= deadline_seconds:
            raise RuntimeError("Staging stopped or deadline exceeded")

    def record(name: str, **values) -> None:
        with lock:
            progress[name] = values
            temp = evidence / "progress.tmp"
            temp.write_text(json.dumps({"files": progress, "elapsed_s":
                                       time.monotonic() - started}, indent=2) + "\n")
            os.replace(temp, evidence / "progress.json")

    def transfer(name: str) -> None:
        partial = destination / (name + ".partial")
        count = 0
        try:
            check()
            expected = manifest["files_bytes"][name]
            url = ("https://huggingface.co/" + manifest["repo"] + "/resolve/"
                   + manifest["revision"] + "/" + urllib.parse.quote(name))
            request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
            timeout = min(60, max(1, deadline_seconds - (time.monotonic() - started)))
            with opener(request, timeout=timeout) as response, partial.open("xb") as output:
                if response.status != 200:
                    raise ValueError("Whole-file HTTP 200 required")
                if int(response.headers.get("Content-Length", "-1")) != expected:
                    raise ValueError("Content-Length mismatch")
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("Unexpected content encoding")
                while count < expected:
                    check()
                    data = response.read(min(8 * 1024 * 1024, expected - count))
                    if not data:
                        raise ValueError("Truncated response")
                    output.write(data)
                    count += len(data)
                    record(name, status="DOWNLOADING", bytes=count)
                if response.read(1):
                    raise ValueError("Oversized response")
                output.flush()
                os.fsync(output.fileno())
            record(name, status="VERIFYING_READBACK", bytes=count)
            digest = hashlib.sha256()
            with partial.open("rb") as source:
                for data in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    check()
                    digest.update(data)
            if digest.hexdigest() != manifest["files_sha256"][name]:
                raise ValueError("Readback SHA256 mismatch")
            check()
            partial.rename(destination / name)
            record(name, status="VERIFIED", bytes=count, sha256=digest.hexdigest())
        except Exception as error:
            stop.set()
            record(name, status="FAILED", bytes=count, error=f"{type(error).__name__}: {error}")
            raise

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(transfer, manifest["files_bytes"]))
        check()
        result = {"status": "PASS_REMOTE_SNAPSHOT_SHA", "files": progress}
    except Exception as error:
        result = {"status": "FAILED_REMOTE_SNAPSHOT_STAGING", "files": progress,
                  "error": f"{type(error).__name__}: {error}"}
    result.update(repo=manifest["repo"], revision=manifest["revision"], retries=0,
                  total_bytes=sum(row["bytes"] for row in progress.values()),
                  elapsed_s=time.monotonic() - started, structural_audit="PENDING",
                  performance_verdict="NO_VERDICT")
    (evidence / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--deadline-seconds", type=float, default=1800)
    args = parser.parse_args()
    result = stage(json.loads(args.manifest.read_text()), args.destination, args.evidence,
                   args.deadline_seconds)
    return 0 if result["status"] == "PASS_REMOTE_SNAPSHOT_SHA" else 2


if __name__ == "__main__":
    raise SystemExit(main())
