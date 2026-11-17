"""Recheck raw train/eval exclusion independently of the preparation module."""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path

from rich.console import Console


def normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def keys(row: dict) -> dict:
    messages = row["messages"]
    content = "\n".join(message["role"] + ": " + message["content"] for message in messages)
    prompt = next(message["content"] for message in messages if message["role"] == "user")
    spans = set()
    for message in messages:
        if message["role"] != "system":
            words = normalized(message["content"]).split()
            spans.update(" ".join(words[start:start + 13]) for start in range(len(words) - 12))
    return {"id": {row["id"]}, "conversation": {normalized(content)},
            "prompt": {normalized(prompt)}, "span13": spans,
            "source_prompt_id": {str(row["source_prompt_id"])}
            if row.get("source_prompt_id") is not None else set()}


def audit(root: Path) -> dict:
    profile = json.loads((root / "profile.json").read_text())
    forbidden = {name: set() for name in keys({"id": "dummy", "messages": [
        {"role": "user", "content": "dummy"}]})}
    eval_rows = 0
    with (root / "eval.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            for name, values in keys(row).items():
                forbidden[name].update(values)
            eval_rows += 1
    expected_removed = set(profile["removed_overlap_ids"])
    observed_removed, reasons = set(), Counter()
    combinations = Counter()
    raw_rows = 0
    with (root / "train.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            matched = sorted(name for name, values in keys(row).items()
                             if values & forbidden[name])
            if matched:
                observed_removed.add(row["id"])
                reasons.update(matched)
                combinations["+".join(matched)] += 1
            raw_rows += 1
    assert expected_removed == observed_removed, "Exclusion set differs from raw-data recount"
    assert raw_rows - len(observed_removed) == profile["eligible_train_rows"]
    return {"status": "PASS_FROZEN_LEXICAL_EXCLUSION_RECHECK", "raw_train_rows": raw_rows,
            "eval_rows": eval_rows, "flagged_rows": len(observed_removed),
            "reason_counts_nonexclusive": dict(reasons),
            "reason_combinations": dict(combinations),
            "common_span_false_positives_possible": True,
            "semantic_contamination_or_pretraining_contamination_not_ruled_out": True,
            "no_model_evaluation_performed": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    with (args.root / "independent_lexical_recheck.json").open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    Console().print(result)
