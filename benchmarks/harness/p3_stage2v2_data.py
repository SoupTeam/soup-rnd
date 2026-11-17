"""Build immutable, overlap-audited Stage2 v2 evaluation and development tasks.

This is an outcome-informed redesign after the v1 floors, not completion of v1.
No model, tokenizer, score, log-likelihood or NLL is computed here. Development
rows come only from the three explicitly pinned training splits. A successful
manifest is written last; a failed or existing directory is never reused.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

from benchmarks.harness.p3_realdata_corpus import (
    SOURCES as LEGACY_SOURCES,
)
from benchmarks.harness.p3_realdata_corpus import (
    _compact,
    _file_hash,
    _inside,
    _source_file,
    _text_hash,
    _verify_file,
)
from benchmarks.harness.p3_stage2_dataset import (
    _artifact,
    _json,
    _jsonl,
    _normalized,
    _relative,
    _tokens,
)

POLICY = "p3-stage2-v2-data-v1"
V1_PROTOCOL_SHA256 = "170c0b700c137c1f6e0c6788376927e4aaa4e99866b990bc83d0afc42a62bc3c"
V1_DATASET_MANIFEST_SHA256 = "c840fdbcd4330062d2f6a9df29c178da6e28326fec0241b982cfc7f4c499e712"
TRAINING_SOURCES_SHA256 = "8ab0bdfdefde056387835fcaf38914fe2c6879cde1cf89a8f6169629630675a7"
LEDGER_SHA256 = "d329c545a8d24067e64177396ea133acb88a1849063dfab56e21839d5f0d5b86"
TRAINING_ROWS = 7801
V1_GSM8K_ROWS = 1298
GSM8K_V2_ROWS = 1288
EXPECTED_GSM_INTERVALS = ((0, 8), (1296, 1298))
DEVELOPMENT_COUNTS = {"hellaswag_dev": 2000, "arc_dev": 1000, "boolq_dev": 1000}
_HELLASWAG_RANK_PREFIX = "p3-stage2-v2-hellaswag\0"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_EVALUATION_ENDPOINTS = ("hellaswag", "arc", "boolq", "lambada", "gsm8k", "mbpp")
_PRIMARY_ENDPOINTS = ("hellaswag", "arc", "boolq", "lambada")
_METRICS = {"hellaswag": "acc_norm", "arc": "acc_norm", "boolq": "acc",
            "lambada": "acc", "gsm8k": "nll", "mbpp": "nll"}

# Every source identity, byte count, digest and physical row count is fixed by B1.
_SOURCES = (
    {"endpoint": "hellaswag", "role": "evaluation", "repository": "Rowan/hellaswag",
     "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76", "split": "validation",
     "path": "data/validation-00000-of-00001.parquet", "bytes": 6315951,
     "sha256": "899813071e1e95efafec90f856e1987d2150fa4d020fc005df6962c259f660cd",
     "expected_rows": 10042,
     "fields": ["activity_label", "ctx_a", "ctx_b", "endings", "label"]},
    {"endpoint": "hellaswag_dev", "role": "development", "repository": "Rowan/hellaswag",
     "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76", "split": "train",
     "path": "data/train-00000-of-00001.parquet", "bytes": 24365524,
     "sha256": "cacb12587faa63d7f723a72d61d12bfa94b140446f5a6a0a2e1c6906ab88bf02",
     "expected_rows": 39905,
     "fields": ["activity_label", "ctx_a", "ctx_b", "endings", "label"]},
    {"endpoint": "arc", "role": "evaluation", "repository": "allenai/ai2_arc",
     "revision": "210d026faf9955653af8916fad021475a3f00453", "split": "test",
     "path": "ARC-Easy/test-00000-of-00001.parquet", "bytes": 346257,
     "sha256": "4160597d618ae851c7eb04e281574f3f654776216ac6b6641588d64527b47177",
     "expected_rows": 2376, "fields": ["question", "choices", "answerKey"]},
    {"endpoint": "arc", "role": "evaluation", "repository": "allenai/ai2_arc",
     "revision": "210d026faf9955653af8916fad021475a3f00453", "split": "test",
     "path": "ARC-Challenge/test-00000-of-00001.parquet", "bytes": 203808,
     "sha256": "62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9",
     "expected_rows": 1172, "fields": ["question", "choices", "answerKey"]},
    {"endpoint": "arc_dev", "role": "development", "repository": "allenai/ai2_arc",
     "revision": "210d026faf9955653af8916fad021475a3f00453", "split": "train",
     "path": "ARC-Easy/train-00000-of-00001.parquet", "bytes": 330598,
     "sha256": "b315db8a4be597dc7daa50a4e70d48dd7c990c32085629e6ccd8c926beaa80b5",
     "expected_rows": 2251, "fields": ["question", "choices", "answerKey"]},
    {"endpoint": "boolq", "role": "evaluation", "repository": "google/boolq",
     "revision": "35b264d03638db9f4ce671b711558bf7ff0f80d5", "split": "validation",
     "path": "data/validation-00000-of-00001.parquet", "bytes": 1257630,
     "sha256": "52355d11524b4b874a9b9dcc278feb10f672d52c4f4eff9872e695ede59820f8",
     "expected_rows": 3270, "fields": ["passage", "question", "answer"]},
    {"endpoint": "boolq_dev", "role": "development", "repository": "google/boolq",
     "revision": "35b264d03638db9f4ce671b711558bf7ff0f80d5", "split": "train",
     "path": "data/train-00000-of-00001.parquet", "bytes": 3685146,
     "sha256": "4f028e992c0bd4df30b9f056f4946b64f5c23028034ff0ed5ea467d8538cc623",
     "expected_rows": 9427, "fields": ["passage", "question", "answer"]},
    {"endpoint": "lambada", "role": "evaluation", "repository": "EleutherAI/lambada_openai",
     "revision": "900124bf3b8235c6daf21033af9948b3f07346c4", "split": "test",
     "path": "default/test/default.parquet", "bytes": 1157764,
     "sha256": "99d8a9e54cda761c6340ddd4fa6feda7f846ffb0e716afa52968dea75495eea7",
     "expected_rows": 5153, "fields": ["text"]},
    {"endpoint": "mbpp", "role": "evaluation",
     "repository": "google-research-datasets/mbpp",
     "revision": "4bb6404fdc6cacfda99d4ac4205087b89d32030c", "split": "test",
     "path": "full/test-00000-of-00001.parquet", "bytes": 115824,
     "sha256": "566fd53060ffba5766dace1d1e2f4c38906781526de222b0dfbdbc325b696c77",
     "expected_rows": 500, "fields": ["task_id", "text", "code", "test_list",
                                        "test_setup_code", "challenge_test_list"]},
)
SOURCES = tuple({**source, "url": (
    f"https://huggingface.co/datasets/{source['repository']}/resolve/"
    f"{source['revision']}/{source['path']}"
)} for source in _SOURCES)
_GSM8K_SOURCE = next(source for source in LEGACY_SOURCES
                     if source["domain"] == "gsm8k" and source["split"] == "test")


class AdmissionError(ValueError):
    """No success manifest was published; report_path identifies preserved failure evidence."""

    def __init__(self, reason: str, report_path: Path | None = None) -> None:
        super().__init__(reason)
        self.report_path = report_path


class _ContractError(ValueError):
    """A safe, source-text-free admission failure."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise _ContractError(reason)


def _string(value: Any, reason: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), reason)
    return value


def preprocess(text: str) -> str:
    """Exact lm-evaluation-harness v0.4.9 HellaSwag preprocessing, not whitespace collapse."""
    return re.sub(r"\[.*?\]", "", text.strip().replace(" [title]", ". ")).replace("  ", " ")


def _source_ref(source: dict, ordinal: int) -> str:
    return f"{source['repository']}@{source['revision']}/{source['path']}#row={ordinal}"


def _task(source: dict, ordinal: int, raw: dict) -> tuple[dict, str]:
    """Return the scoring task and its public problem, never a reference-based audit view."""
    endpoint = source["endpoint"]
    family = endpoint.removesuffix("_dev")
    _require(family in _METRICS and type(ordinal) is int and ordinal >= 0, "invalid_task_source")
    role = source["role"]
    _require(role in {"evaluation", "development"}, "invalid_source_role")
    _require((role == "development") == endpoint.endswith("_dev"), "source_role_mismatch")
    _require(role != "development" or source["split"] == "train", "development_not_train")
    _require(role != "evaluation" or source["split"] != "train", "evaluation_is_train")
    _require(isinstance(raw, dict), "source_row_not_object")
    reference = _source_ref(source, ordinal)
    row = {"endpoint": endpoint, "role": role, "source_split": source["split"],
           "source_row_ordinal": ordinal, "task_id": reference, "source_ref": reference,
           "source_row_sha256": _text_hash(_compact(raw)), "metric": _METRICS[family]}
    if family == "hellaswag":
        activity = _string(raw.get("activity_label"), "hellaswag_activity_label")
        context_a = _string(raw.get("ctx_a"), "hellaswag_ctx_a")
        _require(isinstance(raw.get("ctx_b"), str), "hellaswag_ctx_b")
        endings = raw.get("endings")
        _require(isinstance(endings, list) and len(endings) == 4
                 and all(isinstance(value, str) and value.strip() for value in endings),
                 "hellaswag_endings")
        label = raw.get("label")
        _require(type(label) is int or isinstance(label, str) and label in {"0", "1", "2", "3"},
                 "hellaswag_label")
        gold = int(label)
        _require(0 <= gold < 4, "hellaswag_label")
        context = preprocess(activity + ": " + context_a + " " + raw["ctx_b"].capitalize())
        choices = [preprocess(value) for value in endings]
        _require(all(value.strip() for value in choices), "hellaswag_empty_preprocessed_ending")
        row.update(context=context, choices=choices, gold=gold)
        problem = context
    elif family == "arc":
        question = _string(raw.get("question"), "arc_question")
        choices = raw.get("choices")
        _require(isinstance(choices, dict), "arc_choices")
        labels, texts = choices.get("label"), choices.get("text")
        _require(isinstance(labels, list) and isinstance(texts, list)
                 and 2 <= len(labels) == len(texts)
                 and all(isinstance(value, str) and value for value in labels)
                 and len(set(labels)) == len(labels)
                 and all(isinstance(value, str) and value.strip() for value in texts),
                 "arc_choices")
        answer = raw.get("answerKey")
        _require(isinstance(answer, str) and answer in labels, "arc_answer_key")
        row.update(context=f"Question: {question}\nAnswer:", choices=list(texts),
                   gold=labels.index(answer))
        problem = question
    elif family == "boolq":
        passage = _string(raw.get("passage"), "boolq_passage")
        question = _string(raw.get("question"), "boolq_question")
        _require(type(raw.get("answer")) is bool, "boolq_answer")
        row.update(context=f"{passage}\nQuestion: {question}?\nAnswer:",
                   choices=["no", "yes"], gold=int(raw["answer"]))
        problem = passage + "\n" + question
    elif family == "lambada":
        text = _string(raw.get("text"), "lambada_text")
        context, separator, word = text.rpartition(" ")
        _require(bool(separator) and bool(context.strip()) and bool(word.strip()),
                 "lambada_last_word")
        row.update(context=context, target=" " + word, gold=" " + word)
        problem = context
    else:
        _require(family == "mbpp", "unsupported_source_family")
        problem = _string(raw.get("text"), "mbpp_text")
        code = _string(raw.get("code"), "mbpp_code")
        _require(type(raw.get("task_id")) is int, "mbpp_task_id")
        _require(isinstance(raw.get("test_setup_code"), str), "mbpp_test_setup")
        for name in ("test_list", "challenge_test_list"):
            _require(isinstance(raw.get(name), list)
                     and all(isinstance(value, str) for value in raw[name]), "mbpp_test_list")
        row.update(context=f"Question:\n{problem}\nAnswer:\n", target=code, gold=code)
    return row, problem


def _read_source(source: dict, cache_dir: Path, root: Path) -> Iterator[tuple[int, dict]]:
    import pyarrow.parquet as pq

    path = _source_file(source, cache_dir, root)
    _verify_file(path, source)
    parquet = pq.ParquetFile(path)
    _require(parquet.metadata.num_rows == source["expected_rows"], "source_row_count_mismatch")
    names = parquet.schema_arrow.names
    _require(len(names) == len(set(names)) and set(source["fields"]).issubset(names),
             "source_schema_mismatch")
    count = 0
    for batch in parquet.iter_batches(batch_size=1024, use_threads=False):
        for raw in batch.to_pylist():
            yield count, raw
            count += 1
    _require(count == source["expected_rows"], "source_decoded_row_count_mismatch")


def _admit_v1(root: Path, ledger_path: Path) -> tuple[list[dict], list[dict], dict, dict]:
    manifest_path = _relative(root, "manifest.json")
    manifest_size, manifest_digest = _file_hash(manifest_path)
    _require(manifest_digest == V1_DATASET_MANIFEST_SHA256, "v1_manifest_file_hash_mismatch")
    manifest = _json(manifest_path.read_text(encoding="utf-8"))
    _require(isinstance(manifest, dict) and manifest.get("schema") == "p3-stage2-dataset-v1",
             "v1_manifest_schema_mismatch")
    payload = {key: value for key, value in manifest.items() if key != "manifest_payload_sha256"}
    _require(manifest.get("manifest_payload_sha256") == _text_hash(_compact(payload)),
             "v1_manifest_payload_hash_mismatch")
    files = manifest.get("files")
    _require(isinstance(files, dict), "v1_files_missing")
    paths = {}
    for name, count in (("training_sources", TRAINING_ROWS),
                        ("gsm8k_generation", V1_GSM8K_ROWS), ("gsm8k_scoring", V1_GSM8K_ROWS)):
        descriptor = files.get(name)
        _require(isinstance(descriptor, dict) and descriptor.get("rows") == count,
                 "v1_required_file_count_mismatch")
        paths[name] = _artifact(root, descriptor)
    _require(files["training_sources"]["sha256"] == TRAINING_SOURCES_SHA256,
             "training_sources_hash_mismatch")
    training = [{name: row.get(name) for name in (
        "task_id", "source_ref", "source_row_sha256", "problem_text", "prompt",
    )} for row in _jsonl(paths["training_sources"])]
    _require(len(training) == TRAINING_ROWS, "training_sources_row_count_mismatch")
    generation = list(_jsonl(paths["gsm8k_generation"]))
    scoring = list(_jsonl(paths["gsm8k_scoring"]))
    tasks, problems = _gsm_tasks(generation, scoring)
    ledger_size, ledger_digest = _file_hash(ledger_path)
    _require(ledger_digest == LEDGER_SHA256, "ledger_file_hash_mismatch")
    ledger = _json(ledger_path.read_text(encoding="utf-8"))
    exposed = _exposed_indices(ledger, len(tasks))
    inputs = {"v1_dataset_manifest": {"path": "manifest.json", "bytes": manifest_size,
                                          "sha256": manifest_digest},
              "training_sources": files["training_sources"],
              "gsm8k_generation": files["gsm8k_generation"],
              "gsm8k_scoring": files["gsm8k_scoring"],
              "official_generation_ledger": {"path": ledger_path.name, "bytes": ledger_size,
                                               "sha256": ledger_digest},
              "official_gsm8k_intervals": [list(pair) for pair in EXPECTED_GSM_INTERVALS],
              "official_gsm8k_exposed_indices": sorted(exposed)}
    return training, tasks, problems, inputs


def _gsm_tasks(generation: list[dict], scoring: list[dict]) -> tuple[list[dict], dict[str, str]]:
    _require(len(generation) == len(scoring), "gsm8k_join_count_mismatch")
    source = _GSM8K_SOURCE
    old_prefix = (f"{source['repository']}@{source['revision']}/{source['config']}/"
                  f"{source['split']}/{source['path']}#row=")
    prompt_prefix, prompt_suffix = "Question:\n", "\nAnswer:\n"
    tasks, problems = [], {}
    for index, (generated, scored) in enumerate(zip(generation, scoring)):
        keys = ("benchmark", "task_id", "source_ref", "source_id", "source_row_sha256")
        _require(all(generated.get(key) == scored.get(key) for key in keys)
                 and generated.get("benchmark") == "gsm8k", "gsm8k_join_identity_mismatch")
        original_ref = generated.get("source_ref")
        ordinal = generated.get("source_id")
        _require(type(ordinal) is int and 0 <= ordinal < source["expected_rows"]
                 and original_ref == old_prefix + str(ordinal)
                 and generated.get("task_id") == original_ref, "gsm8k_source_identity_mismatch")
        context = _string(generated.get("prompt"), "gsm8k_prompt_missing")
        _require(context.startswith(prompt_prefix) and context.endswith(prompt_suffix)
                 and generated.get("prompt_sha256") == _text_hash(context), "gsm8k_prompt_mismatch")
        question = _string(context[len(prompt_prefix):-len(prompt_suffix)],
                           "gsm8k_question_missing")
        reference = scored.get("scoring")
        _require(isinstance(reference, dict), "gsm8k_scoring_missing")
        answer = _string(reference.get("answer"), "gsm8k_reference_missing")
        row_digest = _text_hash(_compact({"question": question, "answer": answer}))
        _require(generated.get("source_row_sha256") == row_digest, "gsm8k_source_row_hash_mismatch")
        task_id = _source_ref(source, ordinal)
        _require(task_id not in problems, "gsm8k_duplicate_source_identity")
        tasks.append({"endpoint": "gsm8k", "role": "evaluation", "source_split": "test",
                      "source_row_ordinal": ordinal, "task_id": task_id, "source_ref": task_id,
                      "source_row_sha256": row_digest, "context": context, "target": answer,
                      "gold": answer, "metric": "nll", "v1_task_id": generated["task_id"],
                      "v1_index": index})
        problems[task_id] = question
    return tasks, problems


def _exposed_indices(ledger: dict, total: int) -> set[int]:
    _require(isinstance(ledger, dict)
             and ledger.get("schema") == "p3-stage2-official-generation-ledger-v1"
             and ledger.get("protocol_file_sha256") == V1_PROTOCOL_SHA256, "ledger_schema_mismatch")
    intervals = ledger.get("intervals")
    _require(isinstance(intervals, list), "ledger_intervals_missing")
    observed = set()
    for row in intervals:
        _require(isinstance(row, dict), "ledger_interval_not_object")
        if row.get("benchmark") != "gsm8k":
            continue
        start, stop = row.get("start_index"), row.get("stop_index")
        _require(type(start) is int and type(stop) is int and 0 <= start < stop <= total,
                 "ledger_interval_bounds")
        observed.add((start, stop))
    _require(observed == set(EXPECTED_GSM_INTERVALS), "ledger_gsm8k_intervals_mismatch")
    return {index for start, stop in observed for index in range(start, stop)}


def _grams(text: str, masked: bool) -> set[tuple[str, ...]]:
    tokens = _tokens(text)
    if masked:
        tokens = ["<NUM>" if token.isnumeric() else token for token in tokens]
    return {tuple(tokens[index:index + 5]) for index in range(len(tokens) - 4)}


class _TrainingOverlap:
    """Complete train/evaluation join with the v1 exact integer 4/5 Jaccard rule.

    Posting counts give exact set intersections. Length bounds discard impossible
    matches but never approximate or sample. Unlike a combined self-join, this
    avoids computing unrelated training/training or evaluation/evaluation edges.
    """

    def __init__(self, training: list[dict]) -> None:
        self.training_rows = len(training)
        self.entries: list[dict] = []
        self.exact: dict[str, list[int]] = defaultdict(list)
        self.indices: dict[bool, dict[tuple[str, ...], list[int]]] = {
            False: defaultdict(list), True: defaultdict(list),
        }
        self.sizes: dict[bool, list[int]] = {False: [], True: []}
        seen = set()
        for row in training:
            identity = {name: row.get(name) for name in (
                "task_id", "source_ref", "source_row_sha256",
            )}
            _require(all(isinstance(value, str) and value for value in identity.values())
                     and bool(_SHA256.fullmatch(identity["source_row_sha256"])),
                     "training_identity_missing")
            _require(identity["task_id"] not in seen, "duplicate_training_task_id")
            seen.add(identity["task_id"])
            problem = _string(row.get("problem_text"), "training_problem_text_missing")
            prompt = _string(row.get("prompt"), "training_prompt_missing")
            views = [("problem_text", problem)]
            if prompt != problem:
                views.append(("prompt", prompt))
            for view, text in views:
                entry = len(self.entries)
                self.entries.append({**identity, "training_view": view})
                self.exact[_normalized(text)].append(entry)
                for masked in (False, True):
                    values = _grams(text, masked)
                    self.sizes[masked].append(len(values))
                    for gram in values:
                        self.indices[masked][gram].append(entry)

    def audit(self, task: dict, problem: str) -> dict:
        context = _string(task.get("context"), "evaluation_context_missing")
        _string(problem, "evaluation_problem_missing")
        views = [("context", context)]
        if problem != context:
            views.append(("problem_text", problem))
        matches = []
        for view, text in views:
            for entry in self.exact.get(_normalized(text), ()):
                matches.append(self._match(entry, view, "normalized_exact"))
            for masked in (False, True):
                values = _grams(text, masked)
                size = len(values)
                if not size:
                    continue
                counts: Counter[int] = Counter()
                sizes = self.sizes[masked]
                for gram in values:
                    for entry in self.indices[masked].get(gram, ()):
                        if 5 * sizes[entry] >= 4 * size and 5 * size >= 4 * sizes[entry]:
                            counts[entry] += 1
                for entry in sorted(counts):
                    intersection = counts[entry]
                    union = sizes[entry] + size - intersection
                    if 5 * intersection >= 4 * union:
                        kind = "numeric_masked_token5_jaccard" if masked else "token5_jaccard"
                        matches.append({**self._match(entry, view, kind),
                                        "intersection": intersection, "union": union,
                                        "training_gram_count": sizes[entry],
                                        "evaluation_gram_count": size})
        reason = ("training_exact_overlap" if any(match["kind"] == "normalized_exact"
                                                for match in matches) else
                  "training_near_overlap" if matches else None)
        return {**{name: task[name] for name in (
            "endpoint", "task_id", "source_ref", "source_row_sha256",
        )}, "context_sha256": _text_hash(context), "problem_text_sha256": _text_hash(problem),
                "reason": reason, "training_rows_compared": self.training_rows,
                "masked_and_unmasked_passes_complete": True, "matches": matches}

    def _match(self, entry: int, view: str, kind: str) -> dict:
        identity = self.entries[entry]
        return {"training_task_id": identity["task_id"],
                "training_source_ref": identity["source_ref"],
                "training_source_row_sha256": identity["source_row_sha256"],
                "training_view": identity["training_view"], "evaluation_view": view, "kind": kind}


def select_hellaswag_tasks(tasks: list[dict], n: int) -> list[dict]:
    """Select an outcome-independent prefix only after B3 has frozen its size.

    The builder itself always emits the full admitted split in physical source
    order. This helper does not authorize sizing, score rows or mutate tasks.
    """
    _require(type(n) is int and 0 < n <= len(tasks), "invalid_hellaswag_selection_count")
    identities = []
    for row in tasks:
        _require(row.get("endpoint") == "hellaswag"
                 and isinstance(row.get("task_id"), str) and bool(row["task_id"]),
                 "invalid_hellaswag_selection_task")
        identities.append(row["task_id"])
    _require(len(set(identities)) == len(identities), "duplicate_hellaswag_selection_task")
    return sorted(tasks, key=lambda row: (
        _text_hash(_HELLASWAG_RANK_PREFIX + row["task_id"]), row["task_id"],
    ))[:n]


def _write_jsonl(root: Path, relative: str, rows: list[dict]) -> dict:
    path = _relative(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(_compact(row) + "\n")
    size, digest = _file_hash(path)
    return {"path": relative, "bytes": size, "sha256": digest, "rows": len(rows)}


def _write_json(root: Path, relative: str, value: dict) -> dict:
    path = _relative(root, relative)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_compact(value) + "\n")
    size, digest = _file_hash(path)
    return {"path": relative, "bytes": size, "sha256": digest}


def _publish_manifest(root: Path, manifest: dict) -> None:
    staged = _relative(root, "manifest.staging.json")
    destination = _relative(root, "manifest.json")
    payload = _compact(manifest) + "\n"
    expected_size, expected_digest = len(payload.encode("utf-8")), _text_hash(payload)
    with staged.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    size, digest = _file_hash(staged)
    _require(size == expected_size and digest == expected_digest,
             "manifest_staging_verification_mismatch")
    # Same-directory hard linking is no-overwrite publication. Keep the staged
    # evidence: nothing fallible may follow successful publication.
    os.link(staged, destination)


def _endpoint_spec(endpoint: str, initial: int, tasks: list[dict], exclusions: list[dict],
                   descriptor: dict, sources: list[dict]) -> dict:
    development = endpoint.endswith("_dev")
    official = sum(row["reason"] == "official_generation_exposure" for row in exclusions)
    overlap = sum(row["reason"] != "official_generation_exposure" for row in exclusions)
    result = {"endpoint": endpoint, "role": "development" if development else "evaluation",
              "metric": _METRICS[endpoint.removesuffix("_dev")],
              "source_split": "train" if development else sorted({row["source_split"]
                                                                    for row in tasks}),
              "initial_source_row_count": initial, "excluded_row_count": len(exclusions),
              "official_exposure_excluded_row_count": official,
              "training_overlap_excluded_row_count": overlap,
              "not_selected_row_count": initial - len(tasks) if development else 0,
              "retained_row_count": len(tasks),
              "retained_task_ids": [row["task_id"] for row in tasks],
              "ordered_task_ids_sha256": _text_hash(_compact([row["task_id"] for row in tasks])),
              "ordering_rule": "physical_source_order",
              "source_files": [source["path"] for source in sources
                               if source["endpoint"] == endpoint],
              "files": {"tasks": descriptor}}
    if development:
        result["selection_rule"] = "first_n_physical_training_rows_no_evaluation_rows"
        result["requested_development_rows"] = DEVELOPMENT_COUNTS[endpoint]
        result["overlap_audit"] = "not_an_evaluation_endpoint_development_only"
    if endpoint == "arc":
        result["ordering_rule"] = "ARC-Easy_then_ARC-Challenge_each_in_physical_source_order"
    if endpoint == "gsm8k":
        result["ordering_rule"] = "v1_generation_order_after_official_and_training_exclusions"
        result["expected_retained_row_count"] = GSM8K_V2_ROWS
        result["source_files"] = ["v1/gsm8k/generation.jsonl", "v1/gsm8k/scoring.jsonl"]
    return result


def build_v2_data(*, cache_dir: Path, v1_dataset_root: Path, ledger_path: Path,
                  output_dir: Path) -> dict:
    """Build all pinned tasks, audit every evaluation row and commit the manifest last.

    v1_dataset_root is the frozen v1 dataset root containing manifest.json,
    training-sources.jsonl and GSM8K generation/scoring artifacts. Other v1 files
    need not be mounted. A row-count or retained GSM8K mismatch fails closed and
    leaves whatever evidence was already written in the fresh output directory.
    """
    root = Path(os.path.realpath(output_dir))
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise AdmissionError("existing_output_preserved") from None
    try:
        v1_root = Path(os.path.realpath(v1_dataset_root))
        training, gsm_tasks, gsm_problems, inputs = _admit_v1(v1_root, Path(ledger_path))
        overlap = _TrainingOverlap(training)
        tasks: dict[str, list[dict]] = {name: [] for name in (
            *_EVALUATION_ENDPOINTS, *DEVELOPMENT_COUNTS,
        )}
        problems = dict(gsm_problems)
        initial: Counter[str] = Counter()
        source_records = []
        for source in SOURCES:
            endpoint = source["endpoint"]
            _require(endpoint in tasks, "unknown_source_endpoint")
            observed = 0
            for ordinal, raw in _read_source(source, Path(cache_dir), root):
                task, problem = _task(source, ordinal, raw)
                observed += 1
                if source["role"] == "development" and ordinal >= DEVELOPMENT_COUNTS[endpoint]:
                    continue
                tasks[endpoint].append(task)
                problems[task["task_id"]] = problem
            initial[endpoint] += observed
            source_records.append({**source, "observed_rows": observed,
                                   "verified_sha256": source["sha256"]})
        tasks["gsm8k"] = gsm_tasks
        initial["gsm8k"] = len(gsm_tasks)
        seen = set()
        for endpoint, rows in tasks.items():
            for task in rows:
                _require(task["task_id"] not in seen, "duplicate_task_identity")
                seen.add(task["task_id"])
            if endpoint in DEVELOPMENT_COUNTS:
                _require(len(rows) == DEVELOPMENT_COUNTS[endpoint], "development_count_mismatch")
        exposed = set(inputs["official_gsm8k_exposed_indices"])
        audit_rows, exclusion_rows = [], []
        per_endpoint_exclusions: dict[str, list[dict]] = defaultdict(list)
        for endpoint in _EVALUATION_ENDPOINTS:
            retained = []
            for task in tasks[endpoint]:
                audit = overlap.audit(task, problems[task["task_id"]])
                audit_rows.append(audit)
                official = endpoint == "gsm8k" and task["v1_index"] in exposed
                if official or audit["reason"] is not None:
                    reason = "official_generation_exposure" if official else audit["reason"]
                    reasons = (["official_generation_exposure"] if official else [])
                    if audit["reason"] is not None:
                        reasons.append(audit["reason"])
                    excluded = {**{key: task[key] for key in (
                        "endpoint", "task_id", "source_ref", "source_row_sha256",
                    )}, "reason": reason, "reasons": reasons, "matches": audit["matches"]}
                    if endpoint == "gsm8k":
                        excluded.update(v1_index=task["v1_index"], v1_task_id=task["v1_task_id"])
                    exclusion_rows.append(excluded)
                    per_endpoint_exclusions[endpoint].append(excluded)
                else:
                    retained.append(task)
            tasks[endpoint] = retained
        files, endpoints = {}, {}
        for endpoint, rows in tasks.items():
            descriptor = _write_jsonl(root, f"{endpoint}/tasks.jsonl", rows)
            files[f"{endpoint}_tasks"] = descriptor
            endpoints[endpoint] = _endpoint_spec(endpoint, initial[endpoint], rows,
                                                   per_endpoint_exclusions[endpoint], descriptor,
                                                   source_records)
        files["overlap_audit"] = _write_jsonl(root, "overlap-audit.jsonl", audit_rows)
        files["exclusions"] = _write_jsonl(root, "exclusions.jsonl", exclusion_rows)
        overlap_policy = {
            "training_rows": len(training), "evaluation_rows_audited": len(audit_rows),
            "training_views": ["problem_text", "prompt"],
            "evaluation_views": ["context", "problem_text"],
            "exact": "unicode_NFKC_then_unicode_whitespace_collapse_v1",
            "near": "set_token5_jaccard_at_least_4_over_5_exact_integer_comparison",
            "passes": ["token5_jaccard", "numeric_masked_token5_jaccard"],
            "tokenization": "p3_stage2_dataset._tokens",
            "numeric_mask": "token.isnumeric_to_literal_<NUM>",
            "empty_5gram_sets": "never_a_near_match",
            "reference_answers_or_code_compared": False,
            "exclusion_precedence": ["official_generation_exposure", "training_exact_overlap",
                                     "training_near_overlap"],
            "pretraining_contamination_proven_absent": False,
        }
        audit_document = {"schema": "p3-stage2-v2-data-audit-v1", "inputs": inputs,
                          "sources": source_records, "endpoints": endpoints,
                          "overlap_policy": overlap_policy}
        files["data_audit"] = _write_json(root, "data-audit.json", audit_document)
        _require(endpoints["gsm8k"]["retained_row_count"] == GSM8K_V2_ROWS,
                 "gsm8k_retained_count_mismatch")
        _require(all(tasks[name] for name in _EVALUATION_ENDPOINTS), "empty_evaluation_endpoint")
        boolq_counts = Counter(row["gold"] for row in tasks["boolq"])
        chance = {"hellaswag": 0.25,
                  "arc": sum(1 / len(row["choices"]) for row in tasks["arc"]) / len(tasks["arc"]),
                  "boolq": max(boolq_counts.values()) / len(tasks["boolq"]), "lambada": 0.0}
        manifest = {
            "schema": POLICY, "status": "DATA_ADMITTED_PENDING_PROTOCOL_FREEZE",
            "files": files, "inputs": inputs, "sources": source_records, "endpoints": endpoints,
            "primary_endpoints": list(_PRIMARY_ENDPOINTS), "secondary_endpoints": ["gsm8k", "mbpp"],
            "development_endpoints": list(DEVELOPMENT_COUNTS), "chance": chance,
            "boolq_retained_label_counts": {str(key): boolq_counts[key] for key in (0, 1)},
            "chance_policy": "retained_gold_only_no_model_outcomes",
            "source_row_hash_policy": "sha256_sorted_compact_utf8_json_all_original_fields",
            "task_identity_policy": "{repository}@{revision}/{path}#row={physical_ordinal}",
            "overlap_policy": overlap_policy,
            "hellaswag_selection": {"status": "FULL_SPLIT_NO_BUDGET_SUBSAMPLE_APPLIED",
                                    "rank_prefix": _HELLASWAG_RANK_PREFIX,
                                    "rank_algorithm": "sha256_utf8_prefix_plus_task_id",
                                    "tie_break": "task_id",
                                    "sizing_authority": "B3_after_dev_pilot"},
            "prompt_scoring_reference": "lm-evaluation-harness-v0.4.9-zero-shot",
            "multiple_choice_continuation": "one_leading_space_plus_choice",
            "nll_labels": "p3_stage2_dataset._encode_document_response_only_one_appended_EOS",
            "evaluation_outcomes_computed": False, "protocol_freeze_required_before_outcomes": True,
            "development_adapters": "initial_seed23_only_never_final_checkpoint",
            "endpoints_selected_after_v1_floors": True,
            "study_relationship": "separate_outcome_informed_redesign_not_completion_of_v1",
            "v1_verdict": "BLOCKED_INCOMPLETE_COVERAGE",
            "quality_non_degradation_admitted": False,
        }
        manifest["manifest_payload_sha256"] = _text_hash(_compact(manifest))
        _publish_manifest(root, manifest)
        return manifest
    except Exception as error:
        reason = str(error) if isinstance(error, _ContractError) else (
            f"source_or_artifact_admission:{type(error).__name__}")
        failure_path = _inside(root, root / "failure.json")
        with failure_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(_compact({"schema": POLICY, "status": "ADMISSION_FAILED",
                                   "reason": reason, "successful_denominator": None,
                                   "evaluation_outcomes_computed": False,
                                   "quality_non_degradation_admitted": False}) + "\n")
        raise AdmissionError(reason, failure_path) from None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--v1-dataset-root", type=Path, required=True)
    parser.add_argument("--ledger-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    from rich.console import Console

    console = Console()
    try:
        manifest = build_v2_data(cache_dir=arguments.cache_dir,
                                 v1_dataset_root=arguments.v1_dataset_root,
                                 ledger_path=arguments.ledger_path, output_dir=arguments.output)
    except AdmissionError as error:
        console.print(_compact({"status": "ADMISSION_FAILED", "reason": str(error),
                                "report_path": (str(error.report_path)
                                                if error.report_path else None)}))
        raise SystemExit(1) from None
    console.print(_compact({"status": manifest["status"],
                            "retained_counts": {key: value["retained_row_count"]
                                                for key, value in manifest["endpoints"].items()},
                            "manifest_payload_sha256": manifest["manifest_payload_sha256"]}))


if __name__ == "__main__":
    main()
