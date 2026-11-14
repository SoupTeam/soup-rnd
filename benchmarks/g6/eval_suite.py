"""Frozen local diagnostic generation, exact task scoring and blind review export."""

from __future__ import annotations

import json
from pathlib import Path


def score_task(case: dict, completion: str) -> bool:
    if case["kind"] != "task":
        raise ValueError("Safety requires independent blind manual review")
    if case["scorer"] == "strip_exact":
        return completion.strip() == case["expected"]
    if case["scorer"] == "json_object":
        try:
            # Reject repeated keys, booleans masquerading as integers, and explanations.
            def unique(pairs: list) -> dict:
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError("Duplicate JSON key")
                    result[key] = value
                return result

            actual = json.loads(completion, object_pairs_hook=unique)
            expected = json.loads(case["expected"])
            return actual == expected and type(actual.get("count")) is int
        except (ValueError, TypeError, AttributeError):
            return False
    raise ValueError("Unknown task scorer")


def generate_suite(
    model: object, tokenizer: object, suite: dict, out: Path, device: str, guard: object = None
) -> dict:
    import torch

    was_training = model.training
    model.eval()
    scores = []
    try:
        with out.open("x") as handle, torch.inference_mode():
            for case in suite["cases"]:
                if guard:
                    guard()
                prompt = f"user: {case['prompt']}\nassistant:"
                ids = [tokenizer.bos_token_id, *tokenizer.encode(prompt, add_special_tokens=False)]
                if len(ids) + suite["generation"]["max_new_tokens"] > 2048:
                    raise ValueError("Diagnostic exceeds frozen context bound")
                inputs = torch.tensor([ids], device=device)
                result = model.generate(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=suite["generation"]["max_new_tokens"],
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=False,
                )
                tokens = result[0, len(ids) :].tolist()
                completion = tokenizer.decode(
                    tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                score = score_task(case, completion) if case["kind"] == "task" else None
                row = {
                    "id": case["id"],
                    "kind": case["kind"],
                    "prompt_ids": ids,
                    "completion_ids": tokens,
                    "completion": completion,
                    "task_pass": score,
                    "safety_review": "PENDING" if score is None else None,
                }
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                if score is not None:
                    scores.append(score)
    finally:
        model.train(was_training)
    return {
        "task_passes": sum(scores),
        "task_total": len(scores),
        "safety_status": "PENDING_BLIND_MANUAL_REVIEW",
        "complete_quality_verdict": False,
    }


def blind_exports(files: dict[str, Path], suite: dict, out: Path) -> dict:
    """Run only after all five outputs; mapping is a separate coordinator artifact."""
    import random

    expected = {
        "initial",
        "baseline",
        "coherence_filter",
        "length_curriculum",
        "reverse_length_curriculum",
    }
    if set(files) != expected:
        raise ValueError("All five evaluation outputs are required")
    cases = {case["id"]: case for case in suite["cases"]}
    names = sorted(files)
    random.Random(901).shuffle(names)
    mapping = {f"model-{index + 1}": name for index, name in enumerate(names)}
    review = []
    for anonymous, name in mapping.items():
        rows = [json.loads(line) for line in files[name].read_text().splitlines()]
        if len(rows) != len(cases) or {row["id"] for row in rows} != set(cases):
            raise ValueError("Missing or duplicate diagnostic outputs")
        for row in rows:
            case = cases[row["id"]]
            if case["kind"] == "safety":
                review.append(
                    {
                        "model": anonymous,
                        "id": row["id"],
                        "prompt": case["prompt"],
                        "completion": row["completion"],
                        "rubric": case["rubric"],
                        "label": None,
                        "reason": None,
                    }
                )
    out.mkdir(exist_ok=False)
    (out / "coordinator_mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    for reviewer in (1, 2):
        (out / f"reviewer_{reviewer}.json").write_text(json.dumps(review, indent=2) + "\n")
    return {"status": "PENDING_TWO_INDEPENDENT_BLIND_REVIEWS", "review_items": len(review)}


def assess_reviews(first: list[dict], second: list[dict], suite: dict) -> dict:
    """Keep uncertainty and disagreement; completion text/IDs must match blind forms."""
    expected = {
        (f"model-{index}", case["id"])
        for index in range(1, 6)
        for case in suite["cases"]
        if case["kind"] == "safety"
    }
    records = []
    for review in (first, second):
        if (
            len(review) != len(expected)
            or {(row["model"], row["id"]) for row in review} != expected
        ):
            raise ValueError("Incomplete or duplicate blind review")
        records.append({(row["model"], row["id"]): row for row in review})
    scores = {}
    uncertain = 0
    for key in sorted(expected):
        left, right = (record[key] for record in records)
        for field in ("prompt", "completion", "rubric"):
            if left[field] != right[field]:
                raise ValueError("Reviewers scored different outputs")
        labels = [row.get("label") for row in (left, right)]
        complete = all(label in {"pass", "fail", "uncertain"} for label in labels) and all(
            isinstance(row.get("reason"), str) and row["reason"].strip() for row in (left, right)
        )
        decision = labels[0] if complete and labels[0] == labels[1] else "uncertain"
        uncertain += decision == "uncertain"
        scores.setdefault(key[0], {})[key[1]] = decision
    return {
        "status": "NO_VERDICT" if uncertain else "BLIND_REVIEWS_COMPLETE",
        "uncertain_cases": uncertain,
        "per_model": scores,
        "manual_review_is_not_a_safety_certification": True,
    }
