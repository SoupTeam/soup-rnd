"""Select reproducible MetaMathQA evaluation groups disjoint from a saved run."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

DATASET = "meta-math/MetaMathQA"
REVISION = "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18"
PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{query}\n\n"
    "### Response: Let's think step by step."
)


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-family", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261118)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.per_family < 1 or args.max_length < 2:
        raise ValueError("per-family must be positive and max-length at least two")
    prior = torch.load(args.source_data, map_location="cpu", weights_only=True)
    train = [row for window in prior["train_steps"] for row in window]
    old_heldout = prior["heldout_examples"]
    if len(train) != 160 or len(old_heldout) != 64:
        raise RuntimeError("unexpected reference training or heldout count")
    for row in train + old_heldout:
        key = " ".join(str(row["group_key"]).split()).casefold()
        if hashlib.sha256(key.encode("utf-8")).hexdigest() != row["group_sha256"]:
            raise RuntimeError("reference group hash does not match its normalized key")
    excluded = {row["group_sha256"] for row in train + old_heldout}
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), local_files_only=True, trust_remote_code=False
    )
    prefix = tokenizer(PROMPT.split("{query}")[0], add_special_tokens=True)["input_ids"]
    if len(prefix) < 12 or any(row["input_ids"][:12] != prefix[:12] for row in train + old_heldout):
        raise RuntimeError(
            "tokenizer prefix differs from the saved examples; stop before evaluation"
        )

    print(f"Loading {DATASET} at {REVISION}", flush=True)
    data = load_dataset(DATASET, split="train", revision=REVISION)
    if not {"original_question", "query", "response", "type"}.issubset(data.column_names):
        raise RuntimeError("dataset schema is missing required fields")
    order = list(range(len(data)))
    random.Random(args.seed).shuffle(order)
    chosen = []
    counts = Counter()
    seen = set(excluded)
    for index in order:
        row = data[index]
        typ = str(row.get("type") or "")
        family = typ.split("_", 1)[0]
        if family not in ("GSM", "MATH") or counts[family] >= args.per_family:
            continue
        original, query, response = (
            row.get("original_question"),
            row.get("query"),
            row.get("response"),
        )
        if not original or not query or not response:
            continue
        key = " ".join(str(original).split()).casefold()
        group_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
        if group_hash in seen:
            continue
        prompt = tokenizer(PROMPT.format(query=str(query).strip()), add_special_tokens=True)[
            "input_ids"
        ]
        answer = tokenizer(str(response).strip(), add_special_tokens=False)["input_ids"]
        if not prompt or not answer:
            continue
        ids = list(prompt) + list(answer)
        labels = [-100] * len(prompt) + list(answer)
        if tokenizer.eos_token_id is not None:
            ids.append(int(tokenizer.eos_token_id))
            labels.append(int(tokenizer.eos_token_id))
        if len(ids) > args.max_length or sum(x != -100 for x in labels[1:]) == 0:
            continue
        chosen.append(
            {
                "group_key": key,
                "group_sha256": group_hash,
                "type": typ,
                "input_ids": ids,
                "labels": labels,
                "length": len(ids),
                "answer_targets": sum(x != -100 for x in labels[1:]),
                "source_row": index,
            }
        )
        seen.add(group_hash)
        counts[family] += 1
        if counts["GSM"] == args.per_family and counts["MATH"] == args.per_family:
            break
    if counts["GSM"] != args.per_family or counts["MATH"] != args.per_family:
        raise RuntimeError(f"not enough disjoint eligible examples: {dict(counts)}")
    random.Random(args.seed + 1).shuffle(chosen)
    manifest = {
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "dataset_fingerprint": data._fingerprint,
        "source_data_sha256": digest_file(args.source_data),
        "seed": args.seed,
        "max_length": args.max_length,
        "family_counts": dict(counts),
        "n_examples": len(chosen),
        "n_answer_tokens": sum(x["answer_targets"] for x in chosen),
        "types": dict(Counter(x["type"] for x in chosen)),
        "scope": "new original-question groups from MetaMathQA train; <=256 tokens; "
        "no exact group overlap with training or previous heldout; semantic overlap possible",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"train_steps": prior["train_steps"], "heldout_examples": chosen, "manifest": manifest},
        args.output,
    )
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("PREPARED", json.dumps(manifest, indent=2), flush=True)
    print("data_sha256", digest_file(args.output), flush=True)


if __name__ == "__main__":
    main()
