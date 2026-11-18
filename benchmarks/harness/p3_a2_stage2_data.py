"""Reconstruct the public 160/64 development split, not the private 2026-11-18 rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from p3_prepare_fresh_metamath_eval import DATASET, PROMPT, REVISION
from rich.console import Console


def main() -> None:
    import torch
    from datasets import Dataset, DatasetDict, load_dataset
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    seed = 20261117
    rng = random.Random(seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False,
                                              local_files_only=True)
    data = load_dataset(DATASET, split="train", revision=REVISION)
    order = list(range(len(data)))
    rng.shuffle(order)
    seen = set()
    families: dict[str, list] = {"GSM": [], "MATH": []}
    for idx in order:
        row = data[idx]
        family = row["type"].split("_", 1)[0]
        if family not in families or len(families[family]) == 112:
            continue
        if not all(row.get(k) for k in ("original_question", "query", "response")):
            continue
        key = " ".join(row["original_question"].split()).casefold()
        if key in seen:
            continue
        prompt = tokenizer(PROMPT.format(query=row["query"].strip()), add_special_tokens=True)[
            "input_ids"
        ]
        answer = tokenizer(row["response"].strip(), add_special_tokens=False)["input_ids"]
        if not prompt or not answer:
            continue
        if tokenizer.eos_token_id is not None:
            answer = [*answer, tokenizer.eos_token_id]
        ids = prompt + answer
        if len(ids) > 256:
            continue
        seen.add(key)
        families[family].append({
            "source_row": idx, "group_key": key,
            "group_sha256": hashlib.sha256(key.encode()).hexdigest(),
            "type": row["type"], "input_ids": ids, "labels": [-100] * len(prompt) + answer,
        })
        if all(len(rows) == 112 for rows in families.values()):
            break
    if any(len(rows) != 112 for rows in families.values()):
        raise ValueError("not enough unique short GSM/MATH groups")
    train = families["GSM"][:80] + families["MATH"][:80]
    dev = families["GSM"][80:] + families["MATH"][80:]
    rng.shuffle(train)
    rng.shuffle(dev)
    splits = {"train": train, "validation": dev}
    manifest = {
        "scope": __doc__, "dataset": DATASET, "revision": REVISION, "seed": seed,
        "max_length": 256, "dataset_fingerprint": data._fingerprint,
        "splits": {k: [{key: r[key] for key in ("source_row", "group_sha256", "type")}
                       for r in rows] for k, rows in splits.items()},
    }
    (args.output / "split.json").write_text(json.dumps(manifest, indent=2) + "\n")
    torch.save({"train_steps": [train[i:i + 8] for i in range(0, len(train), 8)],
                "heldout_examples": dev, "manifest": manifest}, args.output / "reference.pt")
    # soup train preflights data.train before loading its actual Arrow pre_tokenized cache.
    columns = {k: [{"input_ids": r["input_ids"], "labels": r["labels"],
                    "attention_mask": [1] * len(r["input_ids"])} for r in rows]
               for k, rows in splits.items()}
    (args.output / "train.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in columns["train"])
    )
    DatasetDict({k: Dataset.from_list(rows) for k, rows in columns.items()}).save_to_disk(
        str(args.output / "arrow")
    )
    Console().print(f"Prepared {len(train)} training and {len(dev)} development groups")


if __name__ == "__main__":
    main()
