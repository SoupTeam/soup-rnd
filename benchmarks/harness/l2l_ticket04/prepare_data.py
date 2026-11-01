#!/usr/bin/env python3
"""A5 ticket 04: the two data files of the run, from the first rows of tatsu-lab/alpaca.

``alpaca64.jsonl`` holds the first 64 rows that pass the filter (an output and an instruction,
at most 900 characters in all, so rows are mostly under the 256-token limit). It is the
training set of both arms. ``alpaca_large.jsonl`` holds the first ``--large`` rows that pass,
for the run that needs a window larger than the host RAM. The dataset revision and the sha256
of both files go into ``data.json`` next to them. Benchmark harness code, not shipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = "tatsu-lab/alpaca"
MAX_CHARS = 900


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--large", type=int, default=20000)
    parsed = parser.parse_args()

    from datasets import load_dataset
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(REPO).sha
    data = load_dataset(REPO, split="train", revision=revision)
    kept = []
    for row in data:
        size = len(row["instruction"]) + len(row["input"]) + len(row["output"])
        if row["output"].strip() and row["instruction"].strip() and size <= MAX_CHARS:
            kept.append({k: row[k] for k in ("instruction", "input", "output")})
        if len(kept) >= parsed.large:
            break
    out = Path(parsed.out)
    out.mkdir(parents=True, exist_ok=True)
    facts = {"dataset": REPO, "revision": revision, "max_chars": MAX_CHARS, "files": {}}
    for name, count in (("alpaca64.jsonl", parsed.rows), ("alpaca_large.jsonl", len(kept))):
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept[:count])
        (out / name).write_text(text, encoding="utf-8")
        facts["files"][name] = {"rows": count, "sha256": hashlib.sha256(text.encode()).hexdigest()}
    (out / "data.json").write_text(json.dumps(facts, indent=1))
    print(json.dumps(facts, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
