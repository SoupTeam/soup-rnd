
"""Prepare a reproducible Top-K LoRA quality dataset."""

import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset

SEED = 42
TRAIN_SIZE = 2000
VAL_SIZE = 200
OUTPUT = Path("experiments/e2_quality_data.jsonl")


def main():
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    unique = {}
    for row in dataset:
        record = {
            "instruction": str(row["instruction"]).strip(),
            "input": str(row["input"]).strip(),
            "output": str(row["output"]).strip(),
        }

        if not record["instruction"] or not record["output"]:
            continue

        key = json.dumps(
            record, sort_keys=True, ensure_ascii=False
        )
        unique[key] = record

    rows = list(unique.values())
    random.Random(SEED).shuffle(rows)

    required = TRAIN_SIZE + VAL_SIZE
    if len(rows) < required:
        raise RuntimeError(
            f"Only {len(rows)} unique rows; need {required}"
        )

    selected = rows[:required]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as file:
        for row in selected:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()

    print("Total:", len(selected))
    print("Train:", TRAIN_SIZE)
    print("Validation:", VAL_SIZE)
    print("SHA256:", digest)
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()

