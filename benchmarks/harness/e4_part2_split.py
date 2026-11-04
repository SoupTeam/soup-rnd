"""E4 part 2, calibration rule section 2 and 2a: the Dolly split, identity, prompts, labels.

The split is fixed here and used unchanged by the main rule. Everything below is a pure
function of (dataset rows, tokenizer) so the tests can pin it on synthetic rows.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import e4_part2_common as common

DATASET = "databricks/databricks-dolly-15k"
MAX_PROMPT_TOKENS = 384
MAX_RESPONSE_TOKENS = 128  # response plus EOS
DECISION = (0, 2048)
SMOKE = (2048, 2304)
TRAIN = (2304, 6400)
MIN_ELIGIBLE = TRAIN[1]
#: Pre-commit count from the review script (Dolly bdd27f4d); the harness recomputes it.
PRE_COMMIT_ELIGIBLE = 11952

WRAPPER_WITH_CONTEXT = (
    "### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n")
WRAPPER_NO_CONTEXT = "### Instruction:\n{instruction}\n\n### Response:\n"
IGNORE_INDEX = -100


class InsufficientRowsError(RuntimeError):
    """Fewer than 6,400 eligible rows: stop, report, no calibration run (rule section 2)."""


def normalize_field(text: str) -> str:
    """Model text: Unicode NFC, leading and trailing whitespace stripped, inside kept."""
    return unicodedata.normalize("NFC", text).strip()


def identity_key(field: str) -> str:
    """Identity key of one model-text field: internal whitespace runs collapsed to one space.

    ``str.split()`` splits on every character ``str.isspace`` accepts, so tabs, newlines and
    Unicode spaces all collapse. The rule does not name the whitespace set; this is the pin.
    """
    return " ".join(field.split())


def identity(instruction: str, context: str, response: str) -> str:
    """SHA-256 of the UTF-8 bytes of the JSON list of the three identity keys (rule step 2)."""
    keys = [identity_key(instruction), identity_key(context), identity_key(response)]
    blob = json.dumps(keys, ensure_ascii=False, separators=(",", ":"))
    return common.sha256_bytes(blob.encode("utf-8"))


def build_prompt(instruction: str, context: str) -> str:
    """Rule 2a: the wrapper with Context when it is non-empty, the plain one otherwise."""
    if context:
        return WRAPPER_WITH_CONTEXT.format(instruction=instruction, context=context)
    return WRAPPER_NO_CONTEXT.format(instruction=instruction)


def gate_wrapper(prompt: str) -> str:
    """The gate-suite wrapper: the no-Context form around a suite prompt (rule 2a, section 6)."""
    return WRAPPER_NO_CONTEXT.format(instruction=prompt)


def encode_example(tokenizer, instruction: str, context: str, response: str) -> Dict[str, Any]:
    """Token ids and labels for one normalized row.

    The prompt and the response are tokenized separately, without special tokens; the
    response gets no added leading space and EOS appended. Prompt tokens carry label -100,
    response and EOS are targets.
    """
    prompt_ids = tokenizer(build_prompt(instruction, context), add_special_tokens=False).input_ids
    response_ids = tokenizer(response, add_special_tokens=False).input_ids + [
        tokenizer.eos_token_id]
    return {
        "prompt_ids": prompt_ids,
        "response_ids": response_ids,
        "input_ids": prompt_ids + response_ids,
        "labels": [IGNORE_INDEX] * len(prompt_ids) + response_ids,
    }


def pad_batch(examples: Sequence[Dict[str, Any]], pad_id: int) -> Dict[str, List[List[int]]]:
    """Right-pad to the longest; padding carries label -100 and attention 0."""
    width = max(len(e["input_ids"]) for e in examples)
    ids, labels, mask = [], [], []
    for e in examples:
        fill = width - len(e["input_ids"])
        ids.append(e["input_ids"] + [pad_id] * fill)
        labels.append(e["labels"] + [IGNORE_INDEX] * fill)
        mask.append([1] * len(e["input_ids"]) + [0] * fill)
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


@dataclass
class SplitResult:
    funnel: Dict[str, int]
    eligible: List[Dict[str, Any]]  # sorted by identity; fields below
    manifests: Dict[str, List[Dict[str, Any]]]
    boundary: Dict[str, int]  # separate vs joint tokenization agreement, informational

    def rows(self, split: str) -> List[Dict[str, Any]]:
        return self.manifests[split]


def build_split(raw_rows: Iterable[Dict[str, str]], tokenizer,
                boundary_check: bool = True) -> SplitResult:
    """Steps 1-7 of rule section 2. ``raw_rows`` are Dolly rows in dataset order."""
    funnel = {"raw": 0, "empty_instruction_or_response": 0, "duplicate": 0,
              "prompt_over_384": 0, "response_over_128": 0, "both_over": 0}
    seen = set()
    survivors = []
    for row_id, raw in enumerate(raw_rows):
        funnel["raw"] += 1
        instruction = normalize_field(raw["instruction"])
        context = normalize_field(raw.get("context") or "")
        response = normalize_field(raw["response"])
        if not instruction or not response:
            funnel["empty_instruction_or_response"] += 1
            continue
        ident = identity(instruction, context, response)
        if ident in seen:
            funnel["duplicate"] += 1
            continue
        seen.add(ident)
        survivors.append((row_id, ident, instruction, context, response))
    eligible: List[Dict[str, Any]] = []
    boundary = {"checked": 0, "joint_tokenization_differs": 0}
    for row_id, ident, instruction, context, response in survivors:
        enc = encode_example(tokenizer, instruction, context, response)
        long_prompt = len(enc["prompt_ids"]) > MAX_PROMPT_TOKENS
        long_response = len(enc["response_ids"]) > MAX_RESPONSE_TOKENS
        if long_prompt or long_response:
            key = "both_over" if long_prompt and long_response else (
                "prompt_over_384" if long_prompt else "response_over_128")
            funnel[key] += 1
            continue
        if boundary_check:  # informational: does tokenizing the concatenation agree?
            joint = tokenizer(build_prompt(instruction, context) + response,
                              add_special_tokens=False).input_ids
            boundary["checked"] += 1
            if joint != enc["prompt_ids"] + enc["response_ids"][:-1]:
                boundary["joint_tokenization_differs"] += 1
        eligible.append({
            "row_id": row_id, "identity": ident,
            "prompt_tokens": len(enc["prompt_ids"]), "response_tokens": len(enc["response_ids"]),
            "instruction": instruction, "context": context, "response": response,
        })
    eligible.sort(key=lambda r: r["identity"])  # hex string, ascending
    funnel["eligible"] = len(eligible)
    if len(eligible) < MIN_ELIGIBLE:
        raise InsufficientRowsError(
            f"{len(eligible)} eligible rows, fewer than {MIN_ELIGIBLE}: stop and report")
    manifests = {
        "DECISION": eligible[DECISION[0]:DECISION[1]],
        "SMOKE": eligible[SMOKE[0]:SMOKE[1]],
        "TRAIN": eligible[TRAIN[0]:TRAIN[1]],
    }
    return SplitResult(funnel, eligible, manifests, boundary)


# --------------------------------------------------------------------------
# manifests and the recorded wrapper evidence
# --------------------------------------------------------------------------

MANIFEST_FIELDS = ("row_id", "identity", "prompt_tokens", "response_tokens")


def manifest_lines(rows: Sequence[Dict[str, Any]]) -> bytes:
    """Canonical manifest bytes: one JSON object per line, fixed key order, LF endings."""
    return "".join(
        json.dumps({k: row[k] for k in MANIFEST_FIELDS}, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def write_manifests(result: SplitResult, directory: Path) -> Dict[str, Dict[str, Any]]:
    """Write ``<split>.manifest.jsonl`` and return the name, count and SHA-256 of each."""
    directory.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, rows in result.manifests.items():
        data = manifest_lines(rows)
        path = directory / f"{name.lower()}.manifest.jsonl"
        path.write_bytes(data)
        written[name] = {"file": path.name, "rows": len(rows), "sha256": common.sha256_bytes(data)}
    return written


def wrapper_record(tokenizer, example_with: Dict[str, str], example_without: Dict[str, str]
                   ) -> Dict[str, Any]:
    """The rule-2a evidence: wrapper text and SHA-256, token ids of both forms, worked examples."""
    pieces = ["### Instruction:\n", "\n\n### Context:\n", "\n\n### Response:\n"]
    record: Dict[str, Any] = {
        "wrapper_with_context": WRAPPER_WITH_CONTEXT,
        "wrapper_with_context_sha256": common.sha256_bytes(WRAPPER_WITH_CONTEXT.encode()),
        "wrapper_no_context": WRAPPER_NO_CONTEXT,
        "wrapper_no_context_sha256": common.sha256_bytes(WRAPPER_NO_CONTEXT.encode()),
        "wrapper_piece_token_ids": {
            piece: tokenizer(piece, add_special_tokens=False).input_ids for piece in pieces},
        "eos_token_id": tokenizer.eos_token_id,
        "bos_added": False,
    }
    for name, example in (("with_context", example_with), ("without_context", example_without)):
        row = {k: normalize_field(example.get(k, "")) for k in ("instruction", "context",
                                                                "response")}
        enc = encode_example(tokenizer, row["instruction"], row["context"], row["response"])
        record[f"worked_example_{name}"] = {
            "instruction": row["instruction"], "context": row["context"],
            "response": row["response"], "prompt_text": build_prompt(
                row["instruction"], row["context"]),
            "input_ids": enc["input_ids"], "labels": enc["labels"],
            "n_prompt": len(enc["prompt_ids"]), "n_targets": len(enc["response_ids"]),
        }
    return record


def resolve_dataset_revision(prefix: Optional[str]) -> str:
    """Full commit hash of the dataset (the rule records the full revision, not a prefix)."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(DATASET, revision=prefix)
    return info.sha


def load_dolly(revision: str) -> List[Dict[str, str]]:
    from datasets import load_dataset

    data = load_dataset(DATASET, split="train", revision=revision)
    return [dict(r) for r in data]


def run_split(out: Path, tokenizer_dir: Path, dataset_revision: Optional[str],
              log=print) -> Dict[str, Any]:
    """Build the split from the real dataset and tokenizer; write manifests and the record."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    revision = resolve_dataset_revision(dataset_revision)
    rows = load_dolly(revision)
    result = build_split(rows, tokenizer)
    manifests = write_manifests(result, out)
    first_with = next(r for r in rows if (r.get("context") or "").strip())
    first_without = next(r for r in rows if not (r.get("context") or "").strip())
    record = {
        "dataset": DATASET, "dataset_revision": revision,
        "tokenizer_dir_files": {p.name: common.sha256_file(p) for p in sorted(
            Path(tokenizer_dir).glob("tokenizer*.json"))},
        "model": {"repo": common.MODEL_REPO, "revision": common.MODEL_REV},
        "funnel": result.funnel, "pre_commit_eligible": PRE_COMMIT_ELIGIBLE,
        "boundary_check": result.boundary, "manifests": manifests,
        "wrappers": wrapper_record(tokenizer, first_with, first_without),
        "provenance": common.provenance(),
    }
    (out / "split-record.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    log(f"eligible {result.funnel['eligible']} (pre-commit count {PRE_COMMIT_ELIGIBLE})")
    return record
