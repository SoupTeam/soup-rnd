"""Replay G2's published verdicts from the frozen measured source (no GPU, no network).

The two measured commits (40aea9b9 session 1, 5027d2ad session 2) are not on the public
history. They differ from public ``fdf067fc`` in 7 harness files. Those 7 files per session
are stored under ``benchmarks/results/g2-gpu-20261031/frozen-source/``; every other file the
protocol hashed is read from ``fdf067fc`` in this repository's history.

For each session this script

1. rebuilds the measured ``src/`` + ``benchmarks/harness/`` tree and checks every one of the
   634 hashes in the session's ``protocol.json`` ``source_sha256``;
2. runs that tree's own ``g2_rule.py`` on the published session records and requires the
   result to equal the published ``verdict.json``;
3. session 1 only: recomputes each sustained-I/O summary from the per-second
   ``samples.jsonl.gz`` with that tree's ``g2_disk.summarize`` and requires equality with
   the published ``arm.json`` summary;
4. checks the SHA-256 of every file listed in ``raw-records/MANIFEST.json`` and
   ``frozen-source/MANIFEST.json``.

It does not rerun kernels, I/O or the F1 sensors; it shows the published numbers follow from
the published records under the measured rules. Needs ``git`` and this repository's history.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PUBLIC_BASE = "fdf067fc8bdfc056bc41269988335e8007be9389"
SESSIONS = {
    "session1-rtx4090": "40aea9b92b6611735a05bec2862fcca0fa31f497",
    "session2-rtx3090": "5027d2ada3392593d064bbebf4e442da68c4b1c7",
}
RESULTS = Path(__file__).resolve().parents[1] / "results" / "g2-gpu-20261031"

_SUMMARY_SNIPPET = r"""
import json, sys, g2_disk
payload = json.load(sys.stdin)
rows, want = payload["rows"], payload["summary"]
dirs = tuple(k for k in ("write", "read", "weight") if want.get(k))
got = g2_disk.summarize(rows, directions=dirs, warmup_s=want["warmup_s"],
                        window_s=want["window_s"], end_s=want["end_s"])
sys.exit(0 if json.dumps(got, sort_keys=True) == json.dumps(want, sort_keys=True) else 1)
"""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_tree(repo: Path, session: str, dest: Path) -> List[str]:
    """Write the measured tree under ``dest``; return every hash mismatch (empty = exact)."""
    wanted = json.loads((RESULTS / session / "protocol.json").read_text(encoding="utf-8"))[
        "source_sha256"]
    # Raw blobs (not `git archive`, which applies core.autocrlf / eol attributes).
    batch = subprocess.run(
        ["git", "cat-file", "--batch"], cwd=repo, check=True, capture_output=True,
        input="".join(f"{PUBLIC_BASE}:{rel}\n" for rel in sorted(wanted)).encode()).stdout
    cursor = 0
    for rel in sorted(wanted):
        end = batch.index(b"\n", cursor)
        header = batch[cursor:end].split()
        if len(header) != 3:  # "<name> missing": the file is frozen-source only
            cursor = end + 1
            continue
        size = int(header[2])
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(batch[end + 1:end + 1 + size])
        cursor = end + 1 + size + 1
    frozen = RESULTS / "frozen-source" / f"{session}-{SESSIONS[session][:8]}"
    for path in frozen.rglob("*.frozen"):
        target = dest / path.relative_to(frozen).as_posix()[: -len(".frozen")]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    bad = []
    for rel, digest in sorted(wanted.items()):
        file = dest / rel
        if not file.is_file() or sha256(file.read_bytes()) != digest:
            bad.append(rel)
    return bad


def run_rule(tree: Path, session: str, scratch: Path) -> Optional[str]:
    out = scratch / f"{session}-verdict.json"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(tree / "src"), str(tree / "benchmarks" / "harness")]), CUDA_VISIBLE_DEVICES="")
    done = subprocess.run(
        [sys.executable, str(tree / "benchmarks" / "harness" / "g2_rule.py"),
         str(RESULTS / session), "--out", str(out)],
        cwd=tree, env=env, capture_output=True, text=True, check=False)
    if done.returncode != 0 or not out.is_file():
        return f"g2_rule exited {done.returncode}: {done.stderr[-300:]}"
    got = json.loads(out.read_text(encoding="utf-8"))
    want = json.loads((RESULTS / session / "verdict.json").read_text(encoding="utf-8"))
    return None if got == want else "recomputed verdict differs from the published verdict.json"


def check_disk_samples(tree: Path) -> List[str]:
    problems = []
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(tree / "src"), str(tree / "benchmarks" / "harness")]), CUDA_VISIBLE_DEVICES="")
    for arm in sorted(p for p in (RESULTS / "disk-rtx4090").iterdir() if p.is_dir()):
        samples = RESULTS / "raw-records" / "disk-rtx4090" / arm.name / "samples.jsonl.gz"
        summary = json.loads((arm / "arm.json").read_text(encoding="utf-8"))["summary"]
        rows = [json.loads(line) for line in
                gzip.decompress(samples.read_bytes()).decode("utf-8").splitlines()]
        done = subprocess.run(
            [sys.executable, "-c", _SUMMARY_SNIPPET],
            input=json.dumps({"rows": rows, "summary": summary}), env=env,
            cwd=tree, capture_output=True, text=True, check=False)
        if done.returncode != 0:
            problems.append(f"{arm.name}: summary from samples differs ({done.stderr[-200:]})")
    return problems


def check_manifest(directory: Path) -> List[str]:
    entries: List[Dict[str, Any]] = json.loads(
        (directory / "MANIFEST.json").read_text(encoding="utf-8"))["files"]
    problems = []
    for entry in entries:
        file = directory / entry["file"]
        if not file.is_file() or sha256(file.read_bytes()) != entry["published_sha256"]:
            problems.append(f"{directory.name}/{entry['file']}: hash mismatch or missing")
    return problems


def has_public_base(repo: Path) -> bool:
    done = subprocess.run(["git", "cat-file", "-e", f"{PUBLIC_BASE}^{{commit}}"], cwd=repo,
                          capture_output=True, check=False)
    return done.returncode == 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]),
                        help="repository with the public history (default: this one)")
    args = parser.parse_args(argv)
    repo = Path(args.repo)
    if not has_public_base(repo):
        print(f"PROBLEM: public commit {PUBLIC_BASE[:8]} is not in {repo}; the replay needs "
              "the full history (git fetch --unshallow)")
        return 2
    problems: List[str] = []
    with tempfile.TemporaryDirectory(prefix="g2-replay-") as tmp:
        scratch = Path(tmp)
        for session in SESSIONS:
            tree = scratch / session
            bad = build_tree(repo, session, tree)
            print(f"{session}: measured tree rebuilt, {len(bad)} hash mismatches")
            problems += [f"{session}: {name} differs from protocol source_sha256" for name in bad]
            if bad:
                continue
            outcome = run_rule(tree, session, scratch)
            print(f"{session}: verdict {'reproduced' if outcome is None else 'NOT reproduced'}")
            problems += [f"{session}: {outcome}"] if outcome else []
            if session == "session1-rtx4090":
                found = check_disk_samples(tree)
                print(f"{session}: 10 sustained-I/O summaries, {len(found)} differ")
                problems += found
    for directory in (RESULTS / "raw-records", RESULTS / "frozen-source"):
        found = check_manifest(directory)
        print(f"{directory.name}: manifest checked, {len(found)} problems")
        problems += found
    for line in problems:
        print("PROBLEM:", line)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
