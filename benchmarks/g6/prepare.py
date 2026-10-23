"""CPU-only G6 data preparation. No model weights, training, or GPU allocation.

Input: JSONL conversations with string id and messages [{role, content}].
The exact render is role + ': ' + content, separated by newlines, with BOS/EOS.
This is an explicit all-token research recipe, not an assistant-only template.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

from rich.console import Console


def digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def validate(rows: list[dict]) -> None:
    seen = set()
    for row in rows:
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in seen:
            raise ValueError("Missing, empty, or duplicate row id")
        seen.add(row["id"])
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("Conversation must have messages")
        for message in messages:
            if message.get("role") not in {"system", "user", "assistant"}:
                raise ValueError("Unsupported role")
            if not isinstance(message.get("content"), str) or not message["content"].strip():
                raise ValueError("Empty or non-string message")
        if not any(message["role"] == "user" for message in messages):
            raise ValueError("Conversation must contain a user message")
        if not any(message["role"] == "assistant" for message in messages):
            raise ValueError("Conversation must contain an assistant message")


def render(row: dict) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in row["messages"])


def overlap_keys(row: dict) -> set[str]:
    """Exact normalized conversation/prompt plus normalized 13-word spans.

    A conservative lexical screen, not a semantic decontamination guarantee.
    """
    prompt = next(m["content"] for m in row["messages"] if m["role"] == "user")
    keys = {"row:" + digest(normalize(render(row))), "prompt:" + digest(normalize(prompt))}
    if row.get("source_prompt_id") is not None:
        keys.add("source_prompt_id:" + digest(str(row["source_prompt_id"])))
    for message in row["messages"]:
        if message["role"] == "system":
            continue
        words = normalize(message["content"]).split()
        keys.update("span:" + digest(words[i:i + 13]) for i in range(len(words) - 12))
    return keys


def clean_pool(train: list[dict], evaluation: list[dict]) -> tuple[list[dict], list[str]]:
    validate(train)
    validate(evaluation)
    if not evaluation:
        raise ValueError("A nonempty independent eval split is required")
    forbidden = set().union(*(overlap_keys(row) for row in evaluation))
    eval_ids = {row["id"] for row in evaluation}
    kept, removed = [], []
    for row in train:
        if row["id"] in eval_ids or overlap_keys(row) & forbidden:
            removed.append(row["id"])
        else:
            kept.append(row)
    return kept, removed


def token_rows(rows: list[dict], tokenizer: object, max_length: int) -> list[dict]:
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("BOS and EOS token ids are required")
    result = []
    for row in rows:
        body = tokenizer.encode(render(row), add_special_tokens=False)
        # No per-policy truncation: one shared max-length rule, EOS is retained.
        ids = [tokenizer.bos_token_id, *body[:max_length - 2], tokenizer.eos_token_id]
        result.append({"id": row["id"], "input_ids": ids,
                       "content_sha256": digest(normalize(render(row))),
                       "source_tokens": len(body) + 2, "truncated": len(body) + 2 > max_length})
    return result


def take_budget(rows: list[dict], budget: int) -> list[dict]:
    if budget <= 0:
        raise ValueError("Budget must be positive")
    result, remaining = [], budget
    for row in rows:
        ids = row["input_ids"]
        if len(ids) < 2:
            raise ValueError("A causal example requires at least two tokens")
        useful = min(len(ids) - 1, remaining)
        result.append({**row, "input_ids": ids[:useful + 1],
                       "budget_tail_truncated": useful < len(ids) - 1})
        remaining -= useful
        if remaining == 0:
            return result
    raise ValueError(f"Insufficient unique-pass tokens: missing {remaining}; no cycling allowed")


def make_arms(rows: list[dict], scores: dict[str, float], budget: int, seed: int) -> dict:
    rng = random.Random(seed)
    ordered = list(rows)
    rng.shuffle(ordered)
    baseline = take_budget(ordered, budget)
    seen = set()
    dedup = []
    for row in ordered:
        if row["content_sha256"] not in seen:
            seen.add(row["content_sha256"])
            dedup.append(row)
    filtered = [row for row in ordered if scores[row["id"]] >= 0.5]
    return {
        "baseline": baseline,
        "exact_dedup": take_budget(dedup, budget),
        "coherence_filter": take_budget(filtered, budget),
        # Sort AFTER budget selection, keeping baseline's exact token multiset.
        "length_curriculum": sorted(baseline, key=lambda row: len(row["input_ids"])),
    }


def make_order_study_arms(rows: list[dict], scores: dict[str, float],
                          budget: int, seed: int) -> dict:
    """Draft v2: avoid exact-dedup's identity arm on the profiled clean corpus."""
    ordered = list(rows)
    random.Random(seed).shuffle(ordered)
    baseline = take_budget(ordered, budget)
    curriculum = sorted(baseline, key=lambda row: len(row["input_ids"]))
    return {
        "baseline": baseline,
        "coherence_filter": take_budget(
            [row for row in ordered if scores[row["id"]] >= 0.7], budget),
        "length_curriculum": curriculum,
        "reverse_length_curriculum": sorted(
            baseline, key=lambda row: -len(row["input_ids"])),
    }


def token_accounting(input_ids: list[list[int]], labels: list[list[int]],
                     attention_mask: list[list[int]]) -> dict:
    """Count a collated batch after the causal shift; pads cannot be targets."""
    if not (len(input_ids) == len(labels) == len(attention_mask)):
        raise ValueError("Batch dimensions differ")
    useful = processed = nonpadding = 0
    for ids, target, mask in zip(input_ids, labels, attention_mask):
        if not (len(ids) == len(target) == len(mask)) or not ids:
            raise ValueError("Sequence dimensions differ or empty sequence")
        if any(value not in (0, 1) for value in mask):
            raise ValueError("Mask must be binary")
        if mask[0] != 1 or any(mask[i] > mask[i - 1] for i in range(1, len(mask))):
            raise ValueError("Only right-padded sequences are supported")
        if any(not keep and label != -100 for keep, label in zip(mask, target)):
            raise ValueError("Padding is incorrectly supervised")
        if any(label != -100 and label != token for label, token in zip(target, ids)):
            raise ValueError("Causal labels disagree with input ids")
        processed += len(ids)
        nonpadding += sum(mask)
        useful += sum(label != -100 and bool(keep)
                      for label, keep in zip(target[1:], mask[1:]))
    return {"useful_loss_tokens": useful, "processed_input_slots": processed,
            "nonpadding_input_tokens": nonpadding, "padded_input_slots": processed - nonpadding}


def token_multiset(rows: list[dict]) -> Counter:
    return Counter(tuple(row["input_ids"]) for row in rows)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=5_000_000)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=901)
    parser.add_argument("--policy-set", choices=["original", "order-study-v2"],
                        default="original")
    args = parser.parse_args()
    if args.max_length < 3 or args.budget <= 0 or args.out.exists():
        parser.error("Need length>=3, budget>0, and a new output directory")
    import transformers
    from transformers import AutoTokenizer

    from soup_cli.utils.quality import compute_coherence_score

    metadata_path = Path(__file__).with_name("source_metadata.json")
    metadata = json.loads(metadata_path.read_text())
    model = metadata["models"]
    tokenizer = AutoTokenizer.from_pretrained(model["id"], revision=model["revision"],
                                             trust_remote_code=False)
    evaluation = read_jsonl(args.eval)
    raw_train = read_jsonl(args.train)
    train, removed = clean_pool(raw_train, evaluation)
    scores = dict(zip((row["id"] for row in train),
                      compute_coherence_score([render(row) for row in train])))
    encoded_train = token_rows(train, tokenizer, args.max_length)
    constructor = make_arms if args.policy_set == "original" else make_order_study_arms
    arms = constructor(encoded_train, scores, args.budget, args.seed)
    assert token_multiset(arms["baseline"]) == token_multiset(arms["length_curriculum"])
    eval_tokens = token_rows(evaluation, tokenizer, args.max_length)
    # All checks above run before publication; existing artifacts are never overwritten.
    args.out.mkdir(parents=True, exist_ok=False)
    outputs = {**arms, "eval": eval_tokens}
    for name, rows in outputs.items():
        with (args.out / f"{name}.jsonl").open("x") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "status": "DATA_PREPARED_NOT_TRAINED", "metadata": metadata,
        "transformers": transformers.__version__, "seed": args.seed,
        "tokenizer_vocab_sha256": digest(tokenizer.get_vocab()),
        "tokenizer_special_tokens_sha256": digest(tokenizer.special_tokens_map),
        "budget": args.budget, "max_length": args.max_length,
        "useful_definition": "nonignored causal targets after one-position shift",
        "render": "newline-separated role: content, BOS/EOS, all-token loss",
        "removed_overlap_ids": removed,
        "policy_set": args.policy_set,
        "coherence_threshold": 0.5 if args.policy_set == "original" else 0.7,
        "pool_summary": {
            "source_rows": len(raw_train), "eligible_rows": len(train),
            "normalized_unique_conversations": len(
                {row["content_sha256"] for row in encoded_train}),
            "coherence_pass_rows": sum(value >= (
                0.5 if args.policy_set == "original" else 0.7) for value in scores.values()),
            "available_useful_tokens_baseline": sum(
                len(row["input_ids"]) - 1 for row in encoded_train),
            "truncated_conversations": sum(row["truncated"] for row in encoded_train),
        },
        "policy_sequence_hashes": {name: digest([row["input_ids"] for row in rows])
                                   for name, rows in arms.items()},
        "leakage_screen": "normalized full conversation, first user prompt, 13-word spans",
        "semantic_leakage_not_proven_absent": True,
        "inputs": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in (args.train, args.eval)},
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "quality_toolkit_sha256": hashlib.sha256(
            (Path(__file__).parents[2] / "src/soup_cli/utils/quality.py").read_bytes()
        ).hexdigest(),
        "outputs": {name: {"sha256": hashlib.sha256(
            (args.out / f"{name}.jsonl").read_bytes()).hexdigest(), "rows": len(rows),
            "useful_tokens": sum(len(row["input_ids"]) - 1 for row in rows)}
                    for name, rows in outputs.items()},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    Console().print(
        f"Prepared {len(arms)} arms; no training. Manifest: {args.out / 'manifest.json'}"
    )


if __name__ == "__main__":
    main()
