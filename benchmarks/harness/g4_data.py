#!/usr/bin/env python3
"""G4 data: the pinned Dolly file, C5's held-out panel and train split, packed L2L rows.

The split is C5's (``benchmarks/gate-c5-q2-0-experts.md``): row ``i`` is the ``i``-th
JSON line of ``databricks-dolly-15k.jsonl`` at the pinned revision; rows are sorted by
SHA-256 of ``f"{i}\\n{instruction}"`` (ties by ``i``); the first 512 are the held-out
panel and the next 4000 the train split. The digests of both splits are C5's, so a G4
run reads exactly the rows C5 read.

Every row becomes an alpaca conversation through Soup's ``format_to_messages``
(``instruction``, ``context`` as the input, ``response`` as the output) and is rendered
by the model's own chat template.

* Train: the rendered conversations, in split order, are tokenized and concatenated
  into one stream, which is cut into rows of exactly ``seq`` tokens; the remainder is
  dropped. ``L2LStep`` computes the loss with ``labels == input_ids``, so the training
  loss covers every token of a row, prompt included, and a row may span two
  conversations. No row is padded, so no attention mask is needed.
* Panel: one conversation per example, truncated to ``max_length`` tokens. The task
  loss covers the response: every token after the tokenized prompt (the user turn plus
  the generation prompt), the template's end-of-turn tokens included.

Benchmark harness code, not shipped. torch and transformers are imported inside
functions.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import struct
from typing import Any, Dict, List, Mapping, Optional, Sequence

DATASET = "databricks/databricks-dolly-15k"
REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
FILENAME = "databricks-dolly-15k.jsonl"
SOURCE_SHA256 = "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"
HELDOUT_ROWS = 512
TRAIN_ROWS = 4000
FIELDS = ("instruction", "context", "response", "category")
#: C5's published split digests (benchmarks/results/c5-data/manifest.json on C5's branch).
HELDOUT_SHA256 = "b4b2ab938ca72a52c660d810045e3f31f842e3107e00738f827abb3f932d2a37"
TRAIN_SHA256 = "a915bbbf18ebd1fae1dd869bdc8e2a0efbd2ab522d3a0c733cc05b212befa0b7"
PANEL_MAX_LENGTH = 1024


def sort_key(row: int, instruction: str) -> str:
    return hashlib.sha256(f"{row}\n{instruction}".encode()).hexdigest()


def parse_rows(payload: bytes) -> List[Dict[str, Any]]:
    """Split on b"\\n" only: str.splitlines would also split on U+2028 inside a text."""
    return [json.loads(line) for line in payload.split(b"\n") if line.strip()]


def build_splits(
    rows: Sequence[Mapping[str, Any]], *, heldout_rows: int = HELDOUT_ROWS,
    train_rows: int = TRAIN_ROWS,
) -> Dict[str, List[Dict[str, Any]]]:
    order = sorted(range(len(rows)), key=lambda i: (sort_key(i, rows[i]["instruction"]), i))
    ordered = [{"row": i, **{name: rows[i][name] for name in FIELDS}} for i in order]
    return {
        "heldout": ordered[:heldout_rows],
        "train": ordered[heldout_rows:heldout_rows + train_rows],
    }


def serialize(split: Sequence[Mapping[str, Any]]) -> bytes:
    """One JSON object per row, keys sorted, no spaces, UTF-8, each line ending in \\n."""
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        .encode()
        for row in split
    )


def split_digest(split: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(serialize(split)).hexdigest()


def source_path(source: Optional[str] = None) -> str:
    if source is not None:
        return source
    from huggingface_hub import hf_hub_download

    return hf_hub_download(DATASET, FILENAME, repo_type="dataset", revision=REVISION)


def load_splits(source: Optional[str] = None) -> Dict[str, Any]:
    """The two splits and their provenance. Refuses any file but the pinned one, and a
    split whose digest is not C5's."""
    payload = pathlib.Path(source_path(source)).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"{FILENAME} sha256 {digest} is not the pinned {SOURCE_SHA256}")
    splits = build_splits(parse_rows(payload))
    digests = {name: split_digest(rows) for name, rows in splits.items()}
    if digests != {"heldout": HELDOUT_SHA256, "train": TRAIN_SHA256}:
        raise ValueError(f"split digests {digests} differ from C5's")
    return {
        "splits": splits,
        "provenance": {
            "dataset": DATASET, "revision": REVISION, "file": FILENAME,
            "sha256": digest, "split_sha256": digests,
            "order": "sha256(f'{row}\\n{instruction}'), ties by row (C5)",
        },
    }


def messages(row: Mapping[str, Any]) -> List[Dict[str, str]]:
    from soup_cli.data.formats import format_to_messages

    alpaca = {"instruction": row["instruction"], "input": row["context"],
              "output": row["response"]}
    return format_to_messages(alpaca, "alpaca")["messages"]


def _ids(tokenizer: Any, text: str) -> List[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def render(tokenizer: Any, row: Mapping[str, Any]) -> Dict[str, Any]:
    """The conversation's ids and how many of them are the prompt. Refuses a template
    whose prompt does not tokenize as a prefix of the whole conversation."""
    convo = messages(row)
    prompt = tokenizer.apply_chat_template(convo[:1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(convo, tokenize=False)
    prompt_ids, ids = _ids(tokenizer, prompt), _ids(tokenizer, full)
    if ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(f"row {row['row']}: the prompt does not tokenize as a prefix")
    return {"row": row["row"], "ids": ids, "prompt_tokens": len(prompt_ids)}


def pack_train(tokenizer: Any, rows: Sequence[Mapping[str, Any]], *, seq: int,
               limit: Optional[int] = None) -> List[List[int]]:
    """Rows of exactly ``seq`` tokens from the rendered conversations in split order.
    ``limit`` stops once that many rows exist."""
    stream: List[int] = []
    out: List[List[int]] = []
    for row in rows:
        stream.extend(render(tokenizer, row)["ids"])
        while len(stream) >= seq:
            out.append(stream[:seq])
            del stream[:seq]
            if limit is not None and len(out) >= limit:
                return out
    return out


def encode_panel(tokenizer: Any, rows: Sequence[Mapping[str, Any]], *,
                 max_length: int = PANEL_MAX_LENGTH) -> List[Dict[str, Any]]:
    """Each example truncated to ``max_length`` tokens; ``response_from`` is the first
    response position. An example whose prompt fills the window scores no token."""
    out = []
    for row in rows:
        rendered = render(tokenizer, row)
        ids = rendered["ids"][:max_length]
        start = min(rendered["prompt_tokens"], len(ids))
        out.append({"row": rendered["row"], "ids": ids, "response_from": start,
                    "response_tokens": max(0, len(ids) - max(start, 1))})
    return out


def ids_digest(rows: Sequence[Sequence[int]]) -> str:
    """SHA-256 of the rows as little-endian uint32 ids, each row prefixed by its length."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(struct.pack("<I", len(row)))
        digest.update(struct.pack(f"<{len(row)}I", *row))
    return digest.hexdigest()
