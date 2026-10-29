"""Pinned SQuAD subset preparation and QA evaluation for the D1 operating-point study."""

from __future__ import annotations

import argparse
import collections
import json
import re
import string
import urllib.request
from collections.abc import Mapping
from pathlib import Path

from d1_rules import digest, evidence_path, write_new_json
from rich.console import Console

INSTRUCTION = (
    "Answer the question using only the context. Return a short exact answer from the "
    "context, or NO_ANSWER if the context does not contain the answer."
)


def normalize(text: str) -> str:
    lower = text.lower()
    without_punctuation = "".join(char for char in lower if char not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", without_punctuation).split())


def _scores(prediction: str, answers: list[str]) -> tuple[float, float]:
    prediction = "" if prediction.strip() == "NO_ANSWER" else prediction
    predicted = normalize(prediction).split()
    em, f1 = [], []
    for answer in answers or [""]:
        expected = normalize(answer).split()
        em.append(float(predicted == expected))
        if not predicted or not expected:
            f1.append(float(predicted == expected))
            continue
        overlap = sum((collections.Counter(predicted) & collections.Counter(expected)).values())
        f1.append(2 * overlap / (len(predicted) + len(expected)))
    return max(em), max(f1)


def score(rows: list[dict]) -> dict:
    groups = {"all": [], "answerable": [], "unanswerable": []}
    false_answers = 0
    for row in rows:
        prediction = row.get("prediction")
        if not isinstance(prediction, str):
            raise ValueError("missing prediction cannot be treated as an abstention")
        values = _scores(prediction, row["answers"])
        groups["all"].append(values)
        groups["unanswerable" if row["unanswerable"] else "answerable"].append(values)
        if row["unanswerable"] and prediction.strip() not in ("", "NO_ANSWER"):
            false_answers += 1
    result = {}
    for name, values in groups.items():
        result[name] = {
            "count": len(values),
            "exact_match": 100 * sum(value[0] for value in values) / len(values)
            if values
            else None,
            "f1": 100 * sum(value[1] for value in values) / len(values) if values else None,
        }
    negatives = len(groups["unanswerable"])
    result.update(
        exact_match=result["all"]["exact_match"],
        f1=result["all"]["f1"],
        false_answer_rate_unanswerable=false_answers / negatives if negatives else None,
    )
    return result


def prompt(tokenizer, row: dict) -> list[int]:
    result = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": f"Context: {row['context']}\nQuestion: {row['question']}"},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    return list(result["input_ids"] if isinstance(result, Mapping) else result)


def tokenize(tokenizer, row: dict, length: int = 1024) -> dict:
    inputs = prompt(tokenizer, row)
    answer = "NO_ANSWER" if row["unanswerable"] else row["answers"][0]["text"]
    target = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(inputs) + len(target) > length:
        raise ValueError("example requires truncation; prepare a different declared cohort")
    pad = length - len(inputs) - len(target)
    return {
        "input_ids": inputs + target + [tokenizer.pad_token_id] * pad,
        "labels": [-100] * len(inputs) + target + [-100] * pad,
        "attention_mask": [1] * (len(inputs) + len(target)) + [0] * pad,
    }


def prepare(manifest_path: Path, output: Path, counts: dict) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    pool = {}
    sources = manifest["sources"]
    for name, source in sources.items():
        url = source["url"]
        if not url.startswith("https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/"):
            raise ValueError("unexpected dataset source")
        if f"/{manifest['dataset_revision']}/" not in url:
            raise ValueError("source revision differs from manifest")
        raw = output / f"source-{name}.json"
        with urllib.request.urlopen(url, timeout=60) as response:
            with raw.open("xb") as handle:
                handle.write(response.read())
        if digest(raw) != source["sha256"]:
            raise ValueError("dataset source SHA-256 mismatch")
        for article in json.loads(raw.read_text(encoding="utf-8"))["data"]:
            for paragraph in article["paragraphs"]:
                for question in paragraph["qas"]:
                    pool[question["id"]] = {
                        "id": question["id"],
                        "title": article["title"],
                        "context": paragraph["context"],
                        "question": question["question"],
                        "answers": question["answers"],
                        "unanswerable": question["is_impossible"],
                    }
    selected, files = {}, {}
    for name, count in counts.items():
        ids = manifest["splits"][name]["ids"]
        if count <= 0 or count > len(ids):
            raise ValueError(f"invalid {name} count")
        selected[name] = [pool[key] for key in ids[:count]]
        path = output / f"{name}.jsonl"
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            for row in selected[name]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        files[name] = {"file": path.name, "sha256": digest(path), "count": count}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for field in ("id", "title", "context"):
            if {row[field] for row in selected[left]} & {row[field] for row in selected[right]}:
                raise ValueError(f"shared {field} in {left}/{right}")
    result = {
        "status": "PREPARED_NOT_TRAINED",
        "source_manifest_sha256": digest(manifest_path),
        "model": manifest["model"],
        "model_revision": manifest["model_revision"],
        "tokenizer_sha256": manifest["tokenizer_sha256"],
        "splits": files,
        "selection": "prefix of the prior fixed hash order; bounded pilot, not full D2",
    }
    write_new_json(output / "dataset.json", result)
    return result


def load_split(directory: Path, name: str) -> list[dict]:
    manifest = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
    record = manifest["splits"][name]
    path = evidence_path(directory, record["file"])
    if digest(path) != record["sha256"]:
        raise ValueError("prepared split SHA-256 mismatch")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != record["count"]:
        raise ValueError("prepared split count mismatch")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=int, default=256)
    parser.add_argument("--validation", type=int, default=128)
    parser.add_argument("--test", type=int, default=64)
    args = parser.parse_args()
    result = prepare(
        args.manifest,
        args.output,
        {name: getattr(args, name) for name in ("train", "validation", "test")},
    )
    Console().print(result)


if __name__ == "__main__":
    main()
