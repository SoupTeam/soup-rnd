#!/usr/bin/env python3
"""Compact, path-scrubbed copy of one A4 session directory for publication.

Copies every ``*.json`` record of a session (no logs), replaces the box's absolute paths
with placeholders, writes LF-only manifests and ``provenance.json`` with each file's
original and published SHA-256, then re-runs the rule on the published copy and checks
its gates and verdict equal the session's own ``verdict.json``::

    python benchmarks/harness/a4_publish.py <session dir> <out dir> \\
        --replace /root/a4/soup-rnd=<work> --replace /root/a4=<box> --replace /mnt=<mnt>

Order matters: the longest prefix first. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def scrub(text: str, replacements: Sequence[Tuple[str, str]]) -> str:
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def publish(session: Path, out: Path, replacements: Sequence[Tuple[str, str]],
            extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"{out} exists; publication writes a new directory")
    files: List[Dict[str, str]] = []
    for path in sorted(session.rglob("*.json")):
        rel = path.relative_to(session).as_posix()
        raw = path.read_bytes()
        published = scrub(raw.decode("utf-8"), replacements).replace("\r\n", "\n")
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(published, encoding="utf-8", newline="\n")
        files.append({"file": rel, "raw_sha256": sha256(raw),
                      "published_sha256": sha256(published.encode("utf-8"))})
    leftovers = [entry["file"] for entry in files
                 for old, _ in replacements
                 if old in (out / entry["file"]).read_text(encoding="utf-8")]
    if leftovers:
        raise RuntimeError(f"paths survived the scrub in {sorted(set(leftovers))}")
    provenance = {"session": session.name, "path_placeholders": dict(
        (new, old) for old, new in replacements), "files": files, **(extra or {})}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n",
                                         encoding="utf-8", newline="\n")
    lines = [f"{entry['published_sha256']}  {entry['file']}" for entry in files]
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return provenance


def replay_matches(session: Path, out: Path) -> Dict[str, Any]:
    """The rule on the published copy vs the session's own verdict: same verdict, same
    gate statuses and details (details may carry scrubbed paths, so compare statuses and
    the verdict exactly, details after the same scrub)."""
    import a4_rule

    mine = a4_rule.evaluate(str(out))
    theirs = json.loads((out / "verdict.json").read_text(encoding="utf-8"))
    same = {"verdict": mine["verdict"] == theirs["verdict"],
            "gates": {g: mine["gates"][g]["status"] == theirs["gates"][g]["status"]
                      for g in theirs["gates"]}}
    same["all"] = same["verdict"] and all(same["gates"].values())
    return same


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("session")
    parser.add_argument("out")
    parser.add_argument("--replace", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--note", action="append", default=[], metavar="KEY=VALUE",
                        help="extra provenance fields (measured commit, rule, ...)")
    parser.add_argument("--no-replay", action="store_true",
                        help="skip the rule replay (a session judged by an older rule)")
    args = parser.parse_args(argv)
    replacements = [tuple(item.split("=", 1)) for item in args.replace]
    extra = dict(item.split("=", 1) for item in args.note)
    session, out = Path(args.session), Path(args.out)
    provenance = publish(session, out, replacements, extra)
    print(f"{len(provenance['files'])} JSON records published to {out}")
    if not args.no_replay:
        same = replay_matches(session, out)
        print(f"rule replay on the published copy: {same}")
        if not same["all"]:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
