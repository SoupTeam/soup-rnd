"""G2 replay: the tamper checks the published README promises, on a tiny fake repository.

CPU only, tmp_path only. The real history is never read: CI clones shallowly, so the
public commit the replay rebuilds from may be absent there.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "benchmarks" / "harness"
SESSION = "session-x"
SESSION_COMMIT = "abcdef0123456789abcdef0123456789abcdef01"
PUBLIC_A = b"value = 1\nother = 2\n"  # stored with LF endings
PUBLIC_H = b"def h():\n    return 'public'\n"
MEASURED_H = b"def h():\n    return 'measured'\n"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
         *args], cwd=repo, capture_output=True, text=True, check=True)
    return done.stdout.strip()


@pytest.fixture(scope="module")
def replay():
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location("g2_replay", HARNESS / "g2_replay.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["g2_replay"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def world(tmp_path, replay, monkeypatch):
    """A public commit holding two files, a measured variant of one, and the results dir."""
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "benchmarks" / "harness").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "core.autocrlf", "false")
    (repo / "src" / "a.py").write_bytes(PUBLIC_A)
    (repo / "benchmarks" / "harness" / "h.py").write_bytes(PUBLIC_H)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "public")
    base = _git(repo, "rev-parse", "HEAD")
    # What broke `git archive`: a checkout that rewrites LF to CRLF changes the hashed bytes.
    _git(repo, "config", "core.autocrlf", "true")

    results = tmp_path / "results"
    session = results / SESSION
    session.mkdir(parents=True)
    protocol = {"source_sha256": {"src/a.py": _sha(PUBLIC_A),
                                  "benchmarks/harness/h.py": _sha(MEASURED_H)}}
    (session / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    frozen = results / "frozen-source" / f"{SESSION}-{SESSION_COMMIT[:8]}"
    (frozen / "benchmarks" / "harness").mkdir(parents=True)
    (frozen / "benchmarks" / "harness" / "h.py.frozen").write_bytes(MEASURED_H)

    monkeypatch.setattr(replay, "PUBLIC_BASE", base)
    monkeypatch.setattr(replay, "SESSIONS", {SESSION: SESSION_COMMIT})
    monkeypatch.setattr(replay, "RESULTS", results)
    return {"repo": repo, "results": results, "frozen": frozen / "benchmarks" / "harness" /
            "h.py.frozen", "dest": tmp_path / "tree"}


def _manifest(directory: Path, files: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, data in files.items():
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(data)
        entries.append({"file": name, "published_sha256": _sha(data)})
    (directory / "MANIFEST.json").write_text(json.dumps({"files": entries}), encoding="utf-8")


# ---------------------------------------------------------------------------
# build_tree
# ---------------------------------------------------------------------------
def test_the_measured_tree_is_rebuilt_byte_exact_under_autocrlf(replay, world):
    assert replay.build_tree(world["repo"], SESSION, world["dest"]) == []
    assert (world["dest"] / "src" / "a.py").read_bytes() == PUBLIC_A  # no CRLF rewrite
    assert (world["dest"] / "benchmarks" / "harness" / "h.py").read_bytes() == MEASURED_H


def test_an_altered_frozen_file_is_named(replay, world):
    world["frozen"].write_bytes(MEASURED_H + b"# edited\n")
    assert replay.build_tree(world["repo"], SESSION, world["dest"]) == [
        "benchmarks/harness/h.py"]


def test_a_file_that_is_neither_public_nor_frozen_is_named(replay, world):
    world["frozen"].unlink()
    protocol = world["results"] / SESSION / "protocol.json"
    data = json.loads(protocol.read_text(encoding="utf-8"))
    data["source_sha256"]["benchmarks/harness/new.py"] = _sha(b"x")
    protocol.write_text(json.dumps(data), encoding="utf-8")
    bad = replay.build_tree(world["repo"], SESSION, world["dest"])
    assert bad == ["benchmarks/harness/h.py", "benchmarks/harness/new.py"]


# ---------------------------------------------------------------------------
# manifests and main()
# ---------------------------------------------------------------------------
def test_a_manifest_names_altered_and_missing_files(replay, tmp_path):
    directory = tmp_path / "records"
    _manifest(directory, {"a.txt": b"one", "sub/b.txt": b"two", "c.txt": b"three"})
    assert replay.check_manifest(directory) == []
    (directory / "a.txt").write_bytes(b"ONE")
    (directory / "c.txt").unlink()
    assert replay.check_manifest(directory) == [
        "records/a.txt: hash mismatch or missing", "records/c.txt: hash mismatch or missing"]


def test_main_exit_codes_follow_the_manifests(replay, world, capsys):
    _manifest(world["results"] / "raw-records", {"x.log": b"log"})
    _manifest(world["results"] / "frozen-source", {"y.frozen": b"src"})
    replay.SESSIONS.clear()  # manifests only: no source tree, no GPU rule run
    assert replay.main(["--repo", str(world["repo"])]) == 0
    (world["results"] / "raw-records" / "x.log").write_bytes(b"tampered")
    assert replay.main(["--repo", str(world["repo"])]) == 1
    assert "raw-records/x.log" in capsys.readouterr().out


def test_main_refuses_a_history_without_the_public_commit(replay, world, monkeypatch, capsys):
    monkeypatch.setattr(replay, "PUBLIC_BASE", "0" * 40)
    assert replay.main(["--repo", str(world["repo"])]) == 2
    assert "full history" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# sustained-I/O summaries recomputed from per-second samples
# ---------------------------------------------------------------------------
def _write_arm(results: Path, rows: list, summary: dict) -> Path:
    arm = results / "disk-rtx4090" / "p0_write"
    arm.mkdir(parents=True)
    (arm / "arm.json").write_text(json.dumps({"summary": summary}), encoding="utf-8")
    samples = results / "raw-records" / "disk-rtx4090" / "p0_write"
    samples.mkdir(parents=True)
    text = "".join(json.dumps(row) + "\n" for row in rows).encode()
    (samples / "samples.jsonl.gz").write_bytes(gzip.compress(text, mtime=0))
    return samples / "samples.jsonl.gz"


def _rows():
    return [{"elapsed_s": float(t), "total_write_bytes": 1e6 * t, "write_busy_s": 0.5 * t,
             "total_read_bytes": 0, "read_busy_s": 0.0,
             "total_weight_bytes": 0, "weight_busy_s": 0.0} for t in range(1, 41)]


def test_the_disk_summary_is_recomputed_from_samples_and_a_changed_sample_is_caught(
        replay, world):
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    import g2_disk

    rows = _rows()
    summary = g2_disk.summarize(rows, warmup_s=5, window_s=1, directions=("write",))
    samples = _write_arm(world["results"], rows, summary)
    assert replay.check_disk_samples(ROOT) == []

    rows[20]["total_write_bytes"] += 5e8  # one cumulative counter, mid-run
    samples.write_bytes(gzip.compress(
        "".join(json.dumps(row) + "\n" for row in rows).encode(), mtime=0))
    problems = replay.check_disk_samples(ROOT)
    assert len(problems) == 1 and problems[0].startswith("p0_write: summary from samples")
