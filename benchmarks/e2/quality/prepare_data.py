"""Prepare fixed, deduplicated Alpaca data for E2 quality evaluation."""

import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "benchmarks" / "e2" / "quality" / "data"
OUTPUT.mkdir(parents=True, exist_ok=True)

SEED = 42
TRAIN_SIZE = 2000
VAL_SIZE = 200


def canonical(row):
    return {
        "instruction": row["instruction"].strip(),
        "input": row["input"].strip(),
        "output": row["output"].strip(),
    }


def digest(row):
    encoded = json.dumps(
        row,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    unique = {}
    for row in dataset:
        item = canonical(row)
        if all(item.values()) or (
            item["instruction"] and item["output"]
        ):
            unique.setdefault(digest(item), item)

    rows = list(unique.values())

    rng = random.Random(SEED)
    rng.shuffle(rows)

    required = TRAIN_SIZE + VAL_SIZE
    if len(rows) < required:
        raise RuntimeError("Not enough unique examples")

    train = rows[:TRAIN_SIZE]
    val = rows[TRAIN_SIZE:required]

    train_hashes = {digest(row) for row in train}
    val_hashes = {digest(row) for row in val}

    assert train_hashes.isdisjoint(val_hashes)

    train_path = OUTPUT / "train.jsonl"
    val_path = OUTPUT / "validation.jsonl"

    save_jsonl(train_path, train)
    save_jsonl(val_path, val)

    manifest = {
        "dataset": "tatsu-lab/alpaca",
        "seed": SEED,
        "train_examples": len(train),
        "validation_examples": len(val),
        "train_sha256": hashlib.sha256(
            train_path.read_bytes()
        ).hexdigest(),
        "validation_sha256": hashlib.sha256(
            val_path.read_bytes()
        ).hexdigest(),
    }

    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
