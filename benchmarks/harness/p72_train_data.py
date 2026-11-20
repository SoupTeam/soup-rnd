"""Training windows and held-out groups for the adapter-equality run (harness only, pure Python).

Records follow ``p3_prepare_fresh_metamath_eval.py``: ``input_ids`` is prompt, answer and an
end-of-text token, ``labels`` is ``-100`` over the prompt, ``group_sha256`` hashes the normalised
original question. One window is a list of records (one optimizer step); a run of N epochs
repeats the windows in order.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

DATASET = "meta-math/MetaMathQA"
REVISION = "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18"
PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{query}\n\n"
    "### Response: Let's think step by step."
)


def group_hash(original_question: str) -> Tuple[str, str]:
    key = " ".join(str(original_question).split()).casefold()
    return key, hashlib.sha256(key.encode("utf-8")).hexdigest()


def build_record(tokenizer: Any, row: Dict[str, Any], max_length: int) -> Optional[Dict[str, Any]]:
    """The record of one MetaMathQA row, or None when it has no answer or is too long."""
    original, query, response = row.get("original_question"), row.get("query"), row.get("response")
    if not original or not query or not response:
        return None
    key, digest = group_hash(original)
    prompt_text = PROMPT.format(query=str(query).strip())
    prompt = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
    answer = tokenizer(str(response).strip(), add_special_tokens=False)["input_ids"]
    if not prompt or not answer:
        return None
    ids = list(prompt) + list(answer)
    labels = [-100] * len(prompt) + list(answer)
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
        labels.append(int(tokenizer.eos_token_id))
    if len(ids) > max_length or sum(x != -100 for x in labels[1:]) == 0:
        return None
    return {"group_key": key, "group_sha256": digest, "type": str(row.get("type") or ""),
            "input_ids": ids, "labels": labels, "length": len(ids),
            "answer_targets": sum(x != -100 for x in labels[1:])}


def select_records(
    rows: Sequence[Dict[str, Any]], tokenizer: Any, *, train_count: int, train_max_length: int,
    eval_per_family: int, eval_max_length: int, seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """``train_count`` short records, then ``eval_per_family`` GSM and as many MATH records for
    evaluation, all with distinct original questions. Rows are taken in a seeded shuffle order."""
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    train: List[Dict[str, Any]] = []
    heldout: List[Dict[str, Any]] = []
    seen = set()
    counts: Counter = Counter()
    for index in order:
        row = rows[index]
        original = row.get("original_question")
        if not original:
            continue
        _, digest = group_hash(original)
        if digest in seen:
            continue
        if len(train) < train_count:
            record = build_record(tokenizer, row, train_max_length)
            if record is not None:
                record["source_row"] = index
                train.append(record)
                seen.add(digest)
            continue
        family = str(row.get("type") or "").split("_", 1)[0]
        if family not in ("GSM", "MATH") or counts[family] >= eval_per_family:
            continue
        record = build_record(tokenizer, row, eval_max_length)
        if record is not None:
            record["source_row"] = index
            heldout.append(record)
            seen.add(digest)
            counts[family] += 1
        if counts["GSM"] == eval_per_family and counts["MATH"] == eval_per_family:
            break
    if len(train) != train_count or counts["GSM"] != eval_per_family \
            or counts["MATH"] != eval_per_family:
        raise RuntimeError(
            f"not enough eligible examples: train {len(train)}, held-out {dict(counts)}")
    random.Random(seed + 1).shuffle(heldout)
    return {"train": train, "heldout": heldout}


def make_windows(records: Sequence[Dict[str, Any]], per_window: int) -> List[List[Dict[str, Any]]]:
    if per_window < 1 or len(records) % per_window:
        raise ValueError(f"{len(records)} records do not split into windows of {per_window}")
    return [list(records[i:i + per_window]) for i in range(0, len(records), per_window)]


def steps_for_epochs(windows: Sequence[Any], epochs: int) -> List[Any]:
    return list(windows) * epochs


def window_arrays(window: Sequence[Dict[str, Any]], seq: int, pad_id: int
                  ) -> Tuple[List[List[int]], List[List[int]]]:
    """Right-padded ``input_ids`` and ``labels`` (pads are ``-100``): in a causal model the pads
    sit after every real token, so they change no real token's logits or loss."""
    ids_out, labels_out = [], []
    for record in window:
        ids, labels = list(record["input_ids"]), list(record["labels"])
        if len(ids) > seq:
            raise ValueError(f"record of {len(ids)} tokens is longer than the sequence {seq}")
        pad = seq - len(ids)
        ids_out.append(ids + [pad_id] * pad)
        labels_out.append(labels + [-100] * pad)
    return ids_out, labels_out
