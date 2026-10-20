"""C5 ticket 03: the Dolly train split and held-out panel.

Spec: .scratch/c5-q2-0-experts/issues/03-rule-and-data.md. Rule:
benchmarks/gate-c5-q2-0-experts.md.

Row ``i`` is the ``i``-th JSON line (from 0) of ``databricks-dolly-15k.jsonl``
at the pinned revision. Rows are sorted by SHA-256 of ``f"{i}\\n{instruction}"``
in UTF-8 (hex digest, ties by ``i``); the first 512 are the held-out panel and
the next 4000 the train split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from rich.console import Console

console = Console()

DATASET = "databricks/databricks-dolly-15k"
REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
FILENAME = "databricks-dolly-15k.jsonl"
# The LFS oid of FILENAME at REVISION; any other file is refused.
SOURCE_SHA256 = "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"
HELDOUT_ROWS = 512
TRAIN_ROWS = 4000
FIELDS = ("instruction", "context", "response", "category")


def sort_key(row: int, instruction: str) -> str:
    return hashlib.sha256(f"{row}\n{instruction}".encode()).hexdigest()


def build_splits(
    rows: Sequence[Mapping[str, Any]], *, heldout_rows: int, train_rows: int,
) -> dict[str, list[dict[str, Any]]]:
    order = sorted(range(len(rows)), key=lambda i: (sort_key(i, rows[i]["instruction"]), i))
    ordered = [{"row": i, **{field: rows[i][field] for field in FIELDS}} for i in order]
    return {
        "heldout": ordered[:heldout_rows],
        "train": ordered[heldout_rows:heldout_rows + train_rows],
    }


def serialize(split: Sequence[Mapping[str, Any]]) -> bytes:
    """One JSON object per row, keys sorted, no spaces, UTF-8, each line ending in \\n."""
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        .encode()
        for row in split
    )


def split_digest(split: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(serialize(split)).hexdigest()


def parse_rows(payload: bytes) -> list[dict[str, Any]]:
    """Split on b"\\n" only: str.splitlines would also split on U+2028 inside a text."""
    return [json.loads(line) for line in payload.split(b"\n") if line.strip()]


def _read_source(source: str | None) -> bytes:
    if source is not None:
        return pathlib.Path(source).read_bytes()
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, FILENAME, repo_type="dataset", revision=REVISION)
    return pathlib.Path(path).read_bytes()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--source", help=f"local {FILENAME}; downloaded at the pinned revision if omitted",
    )
    parser.add_argument("--out-dir", help="writes heldout.jsonl, train.jsonl and manifest.json")
    parser.add_argument(
        "--train-rows", type=int, choices=(TRAIN_ROWS, 2000), default=TRAIN_ROWS,
        help="2000 only after a committed amendment to the rule (pilot over 2 hours)",
    )
    args = parser.parse_args(argv)

    payload = _read_source(args.source)
    source_sha256 = hashlib.sha256(payload).hexdigest()
    if source_sha256 != SOURCE_SHA256:
        console.print(f"source sha256 {source_sha256} is not the pinned {SOURCE_SHA256}")
        return 1
    rows = parse_rows(payload)
    needed = HELDOUT_ROWS + args.train_rows
    if len(rows) < needed:
        console.print(f"source has {len(rows)} rows, needs {needed}")
        return 1

    splits = build_splits(rows, heldout_rows=HELDOUT_ROWS, train_rows=args.train_rows)
    digests = {
        name: {"rows": len(split), "sha256": split_digest(split)} for name, split in splits.items()
    }
    manifest = {
        "note": "temporary set until the dataset protocol",
        "source": {
            "dataset": DATASET, "revision": REVISION, "file": FILENAME,
            "sha256": source_sha256, "rows": len(rows), "license": "CC BY-SA 3.0",
        },
        "order": "sha256(f'{row}\\n{instruction}'.encode('utf-8')).hexdigest(), ties by row",
        "harness_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "splits": digests,
    }
    if args.out_dir:
        out_dir = pathlib.Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, split in splits.items():
            (out_dir / f"{name}.jsonl").write_bytes(serialize(split))
        (out_dir / "manifest.json").write_bytes((json.dumps(manifest, indent=1) + "\n").encode())

    console.print(
        f"source {FILENAME} @ {REVISION}: {len(rows)} rows, sha256 {source_sha256}",
        soft_wrap=True,
    )
    for name, entry in digests.items():
        console.print(f"{name}: {entry['rows']} rows, sha256 {entry['sha256']}", soft_wrap=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
