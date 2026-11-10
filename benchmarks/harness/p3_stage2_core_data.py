"""Data-only Stage2 GSM8K, HumanEval and training-source admission.

Reuses the historical corpus byte pins and exact row/field identity contract. No
model, tokenizer or source/reference/generated program is imported or executed.
Successful manifests admit source inventory only. Shared overlap/independence,
packing, inference and isolated HumanEval runtime proof remain pending.

``build_core_sources(*, cache_dir, output_dir, exposure_audit_path)`` returns the
root manifest. Its ``evaluations`` map contains the two endpoint manifests. Every
manifest is published after its referenced payloads, with the root commit marker
last. Exclusive hard links preserve prior evidence, including partial publications.
Failure reports contain safe reason codes and identities, never problem text.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import unicodedata
import uuid
from pathlib import Path
from typing import Any

from benchmarks.harness.p3_realdata_corpus import (
    SOURCES as LEGACY_SOURCES,
)
from benchmarks.harness.p3_realdata_corpus import (
    _compact,
    _file_hash,
    _inside,
    _read_rows,
    _source_file,
    _text_hash,
    _verify_file,
)

POLICY = "p3-stage2-core-data-v1"
PRIOR_MANIFEST_SHA256 = "41178adcf2216491bfdc198d4c42a58f0e7a3418b964df967fcbc8e62c06119e"
EXPECTED_REMAINDERS = {("gsm8k", "train"): 7433, ("gsm8k", "test"): 1299,
                       ("humaneval", "test"): 101}
SOURCES = tuple({**source, "url": (
    f"https://huggingface.co/datasets/{source['repository']}/resolve/"
    f"{source['revision']}/{source['path']}"
)} for source in LEGACY_SOURCES)
MBPP_SOURCE = {
    "domain": "mbpp", "repository": "google-research-datasets/mbpp",
    "revision": "4bb6404fdc6cacfda99d4ac4205087b89d32030c", "config": "full", "split": "train",
    "path": "full/train-00000-of-00001.parquet", "expected_rows": 374, "bytes": 87223,
    "sha256": "09d125ca31edacb7800be8c67c45abff618faf0214ff551291817d06bdb914ae",
    "fields": ["task_id", "text", "code", "test_list", "test_setup_code", "challenge_test_list"],
    "url": "https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/"
           "4bb6404fdc6cacfda99d4ac4205087b89d32030c/full/train-00000-of-00001.parquet",
}
GSM8K_GENERATION_POLICY = {
    "policy": POLICY, "prompt": "Question:\n{question}\nAnswer:\n", "format": "plain_completion",
    "chat_template": False, "fim_wrapper": False, "do_sample": False, "num_beams": 1,
    "num_return_sequences": 1, "max_new_tokens": 1024,
    "stop": ["literal_stop", "eos", "max_new_tokens"], "stop_strings": ["\nQuestion:"],
    "truncate_prompt": False,
    "continuation_extraction": "token_ids_after_full_input_width_before_generated_eos",
    "clean_up_tokenization_spaces": False, "postprocessing": "none",
}
HUMANEVAL_GENERATION_POLICY = {
    **GSM8K_GENERATION_POLICY, "prompt": "unchanged_source_prompt",
    "format": "native_body_continuation", "max_new_tokens": 512,
    "stop_strings": ["\nclass", "\ndef", "\n#", "\nif", "\nprint"],
}
GENERATION_POLICIES = {"gsm8k": GSM8K_GENERATION_POLICY, "humaneval": HUMANEVAL_GENERATION_POLICY}
# Optional minus, not a literal backslash. Intentionally preserves upstream regex quirks.
_GSM_ANSWER = re.compile(r"#### (\-?[0-9\.\,]+)")
_ENDPOINT_FILES = ("generation.jsonl", "scoring.jsonl", "audit.jsonl", "manifest.json")
_ARTIFACTS = ("training-sources.jsonl",) + tuple(
    f"{benchmark}/{name}" for benchmark in GENERATION_POLICIES for name in _ENDPOINT_FILES
) + ("manifest.json",)
_NORMALIZATION = "unicode_NFKC_then_unicode_whitespace_collapse_v1"


class AdmissionError(ValueError):
    """Complete source admission failed; report_path preserves non-sensitive evidence."""

    def __init__(self, report_path: Path, reason: str) -> None:
        self.report_path = report_path
        super().__init__(f"CORE_DATA_ADMISSION_FAILED:{reason}; report={report_path}")


class _ContractError(ValueError):
    """Safe, text-free admission reason."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise _ContractError(reason)


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise _ContractError("nonfinite_json_constant")


def _read_json(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                        parse_constant=_reject_constant)
    # Also rejects decimal exponents that overflow Python's finite float range.
    _compact(result)
    _require(isinstance(result, dict), "json_root_must_be_object")
    return result


def _extract_answer(text: Any) -> str | None:
    if not isinstance(text, str):
        return None
    match = _GSM_ANSWER.search(text)
    return match.group(1).strip().replace(",", "") if match else None


def score_gsm8k(record: dict, completion: str) -> dict:
    """Strict first-marker string scoring, never numeric equivalence or last-answer search.

    ``record`` is one scoring.jsonl row with ``scoring.answer``. The caller must
    supply the single generated continuation, already sliced at its token EOS/cap.
    Literal stops are applied defensively here too. Missing gold invalidates data
    admission; a non-text completion is infrastructure, not a model verdict.
    """
    result = {"status": "DATA_ADMISSION_ERROR", "score": None, "passed": None,
              "gold_answer": None, "extracted_answer": None,
              "reason": "invalid_gold_extraction", "stop_offset": None, "stop_reason": None}
    scoring = record.get("scoring") if isinstance(record, dict) else None
    gold = _extract_answer(scoring.get("answer")) if isinstance(scoring, dict) else None
    if gold is None:
        return result
    result["gold_answer"] = gold
    if not isinstance(completion, str):
        result.update(status="INFRASTRUCTURE_ERROR", reason="completion_must_be_text")
        return result
    stops = [(completion.find(stop), reason) for stop, reason in (
        ("\nQuestion:", "literal_stop"), ("<|endoftext|>", "eos"),
    ) if stop in completion]
    if stops:
        offset, reason = min(stops, key=lambda item: item[0])
        completion = completion[:offset]
        result.update(stop_offset=offset, stop_reason=reason)
    extracted = _extract_answer(completion)
    passed = extracted is not None and extracted == gold
    result.update(status="PASS" if passed else "FAIL", score=int(passed), passed=passed,
                  extracted_answer=extracted,
                  reason="exact_match" if passed else (
                      "missing_answer_marker" if extracted is None else "answer_mismatch"))
    return result


def _read_values(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return [row for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, use_threads=False)
            for row in batch.to_pylist()]


def _load_legacy(source: dict, cache_dir: Path, root: Path) -> list[tuple[dict, dict]]:
    path = _source_file(source, cache_dir, root)
    identities = _read_rows(path, source)
    values = _read_values(path)
    _require(len(values) == len(identities) == source["expected_rows"], "source_coverage_mismatch")
    result = []
    for ordinal, (identity, raw) in enumerate(zip(identities, values)):
        _require(identity["physical_row_ordinal"] == ordinal, "source_ordinal_mismatch")
        _require(identity["source_row_sha256"] == _text_hash(_compact(raw)),
                 "source_row_hash_mismatch")
        _require(identity["source_field_sha256"] == {
            name: _text_hash(value) for name, value in sorted(raw.items())
        }, "source_field_hash_mismatch")
        if source["domain"] == "gsm8k":
            _require(bool(raw["question"].strip()), "empty_gsm_question")
            _require(_extract_answer(raw["answer"]) is not None, "invalid_gsm_gold")
        else:
            _require(all(raw[name].strip() for name in source["fields"]),
                     "empty_humaneval_field")
            _require(raw["entry_point"].isidentifier(), "invalid_humaneval_entry_point")
        result.append((identity, raw))
    _verify_file(path, source)
    return result


def _load_mbpp(source: dict, cache_dir: Path, root: Path) -> list[tuple[dict, dict]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = _source_file(source, cache_dir, root)
    parquet = pq.ParquetFile(path)
    _require(parquet.metadata.num_rows == source["expected_rows"], "mbpp_physical_count_mismatch")
    schema = parquet.schema_arrow
    _require(len(schema.names) == len(source["fields"])
             and set(schema.names) == set(source["fields"]), "mbpp_schema_fields_mismatch")
    for field in schema:
        kind = field.type
        if field.name == "task_id":
            valid = pa.types.is_integer(kind)
        elif field.name in {"test_list", "challenge_test_list"}:
            valid = ((pa.types.is_list(kind) or pa.types.is_large_list(kind))
                     and (pa.types.is_string(kind.value_type)
                          or pa.types.is_large_string(kind.value_type)))
        else:
            valid = pa.types.is_string(kind) or pa.types.is_large_string(kind)
        _require(valid, f"mbpp_schema_type_mismatch:{field.name}")
    result = []
    seen = set()
    for raw in _read_values(path):
        ordinal = len(result)
        source_id = raw["task_id"]
        _require(type(source_id) is int and source_id > 0 and source_id not in seen,
                 "mbpp_invalid_or_duplicate_task_id")
        seen.add(source_id)
        for name in ("text", "code", "test_setup_code"):
            _require(isinstance(raw[name], str), f"mbpp_nontext_field:{name}")
        _require(bool(raw["text"].strip()) and bool(raw["code"].strip()), "mbpp_empty_text_or_code")
        for name in ("test_list", "challenge_test_list"):
            tests = raw[name]
            _require(isinstance(tests, list) and all(
                isinstance(test, str) and bool(test.strip()) for test in tests
            ), f"mbpp_invalid_tests:{name}")
        _require(bool(raw["test_list"]), "mbpp_empty_test_list")
        identity = {
            "row_key": _row_key(source, ordinal), "domain": "mbpp", "split": "train",
            "physical_row_ordinal": ordinal, "task_id_or_null": source_id,
            "source_row_sha256": _text_hash(_compact(raw)),
            "source_field_sha256": {
                name: _text_hash(value if isinstance(value, str) else _compact(value))
                for name, value in sorted(raw.items())
            },
        }
        result.append((identity, raw))
    _require(len(result) == source["expected_rows"], "mbpp_decoded_count_mismatch")
    _verify_file(path, source)
    return result


def _row_key(source: dict, ordinal: int) -> str:
    return (f"{source['repository']}@{source['revision']}/{source['config']}/"
            f"{source['split']}/{source['path']}#row={ordinal}")


def _source_key(source: dict) -> tuple[str, str]:
    return source["domain"], source["split"]


def _exposure_descriptor(identity: dict) -> dict:
    return {"physical_row_ordinal": identity["physical_row_ordinal"],
            "task_id": identity["task_id_or_null"],
            "source_row_sha256": identity["source_row_sha256"]}


def _check_exposure_list(value: Any, expected: list[dict], reason: str) -> None:
    _require(isinstance(value, list), reason)
    for row in value:
        _require(isinstance(row, dict) and set(row) == {
            "physical_row_ordinal", "task_id", "source_row_sha256",
        }, reason)
        _require(type(row["physical_row_ordinal"]) is int, reason)
    actual = [_compact(row) for row in value]
    _require(len(actual) == len(set(actual)) and sorted(actual) == sorted(
        _compact(row) for row in expected
    ), reason)


def _verify_exposure(audit_path: Path, loaded: dict) -> tuple[dict, dict[tuple, set[str]]]:
    audit = _read_json(audit_path)
    _require(audit.get("schema") == "p3-stage2-exposure-audit-v1", "exposure_schema_mismatch")
    _require(audit.get("prior_corpus_manifest_sha256") == PRIOR_MANIFEST_SHA256,
             "exposure_prior_manifest_identity_mismatch")
    reference = audit.get("prior_corpus")
    _require(isinstance(reference, str) and bool(reference), "missing_prior_corpus_reference")
    prior_path = Path(reference)
    if not prior_path.is_absolute():
        repository = Path(__file__).parents[2]
        prior_path = _inside(repository, repository / prior_path)
    prior = _read_json(prior_path)
    manifest = prior.get("manifest")
    _require(isinstance(manifest, dict), "missing_prior_manifest")
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    _require(manifest.get("manifest_sha256") == PRIOR_MANIFEST_SHA256
             and _text_hash(_compact(payload)) == PRIOR_MANIFEST_SHA256,
             "prior_manifest_hash_mismatch")
    _require(manifest.get("manifest_hash_exclusion") == "manifest_sha256",
             "prior_manifest_hash_policy_mismatch")
    expected_rows = {identity["row_key"]: identity for rows in loaded.values()
                     for identity, _ in rows}
    prior_rows = manifest.get("rows")
    _require(isinstance(prior_rows, list) and len(prior_rows) == len(expected_rows),
             "prior_source_row_coverage_mismatch")
    seen = set()
    for row in prior_rows:
        _require(isinstance(row, dict), "malformed_prior_source_row")
        key = row.get("row_key")
        _require(isinstance(key, str) and key in expected_rows and key not in seen,
                 "unknown_or_duplicate_prior_source_row")
        seen.add(key)
        expected = {k: v for k, v in expected_rows[key].items() if k != "text"}
        packing_fields = {
            "content_token_count", "document_token_count", "document_tokens_sha256",
        }
        _require({k: v for k, v in row.items() if k not in packing_fields} == expected,
                 "prior_source_row_or_field_identity_mismatch")
        annotations = packing_fields.intersection(row)
        _require(not annotations or annotations == packing_fields,
                 "partial_prior_token_annotations")
        if annotations:
            content_count = row["content_token_count"]
            document_count = row["document_token_count"]
            token_digest = row["document_tokens_sha256"]
            _require(type(content_count) is int and content_count >= 0
                     and type(document_count) is int and document_count == content_count + 1
                     and isinstance(token_digest, str)
                     and re.fullmatch(r"[0-9a-f]{64}", token_digest) is not None,
                     "malformed_prior_token_annotations")
    prior_sources = manifest.get("sources")
    _require(isinstance(prior_sources, list) and len(prior_sources) == len(SOURCES),
             "prior_source_pin_coverage_mismatch")
    for source, observed in zip(SOURCES, prior_sources):
        _require(isinstance(observed, dict) and all(observed.get(name) == value
                 for name, value in source.items()), "prior_source_pin_mismatch")
        _require(observed.get("observed_rows") == source["expected_rows"]
                 and observed.get("verified_sha256") == source["sha256"],
                 "prior_source_verification_mismatch")
    cases = prior.get("cases")
    _require(isinstance(cases, list) and all(isinstance(case, dict) for case in cases),
             "malformed_prior_cases")
    case_ids = [case.get("case_id") for case in cases]
    _require(all(isinstance(case_id, str) for case_id in case_ids)
             and len(case_ids) == len(set(case_ids))
             and case_ids == manifest.get("case_order") == audit.get("prior_case_ids"),
             "prior_case_coverage_mismatch")
    consumed = {key: set() for key in loaded}
    for case in cases:
        spans = case.get("source_row_spans")
        _require(isinstance(spans, list) and bool(spans), "missing_prior_source_spans")
        for span in spans:
            _require(isinstance(span, dict), "malformed_prior_source_span")
            key = span.get("row_key")
            _require(isinstance(key, str) and key in expected_rows, "unknown_prior_source_span")
            identity = expected_rows[key]
            _require(all(span.get(name) == identity[name] for name in (
                "source_row_sha256", "task_id_or_null", "text_sha256",
            )), "prior_span_identity_mismatch")
            _require(case.get("domain") == identity["domain"], "prior_span_domain_mismatch")
            consumed[(identity["domain"], identity["split"])].add(key)
    groups = audit.get("groups")
    _require(isinstance(groups, list) and len(groups) == len(SOURCES), "exposure_group_coverage")
    seen_groups = set()
    for group in groups:
        _require(isinstance(group, dict), "malformed_exposure_group")
        key = group.get("domain"), group.get("split")
        _require(all(isinstance(part, str) for part in key) and key in loaded
                 and key not in seen_groups, "unknown_or_duplicate_exposure_group")
        seen_groups.add(key)
        source = next(source for source in SOURCES if _source_key(source) == key)
        expected_source = {name: source[name] for name in (
            "repository", "revision", "config", "path", "sha256", "bytes",
        )}
        expected_source["observed_rows"] = source["expected_rows"]
        _require(group.get("source") == expected_source, "exposure_source_pin_mismatch")
        consumed_rows = [_exposure_descriptor(identity) for identity, _ in loaded[key]
                         if identity["row_key"] in consumed[key]]
        remaining = [_exposure_descriptor(identity) for identity, _ in loaded[key]
                     if identity["row_key"] not in consumed[key]]
        _check_exposure_list(group.get("previously_consumed_rows"), consumed_rows,
                             "exposure_consumed_hash_or_ordinal_mismatch")
        _check_exposure_list(group.get("evaluation_eligible_by_exposure_only"),
                             [] if key[1] == "train" else remaining,
                             "exposure_remainder_coverage_mismatch")
        _require(type(group.get("previously_consumed_count")) is int
                 and group["previously_consumed_count"] == len(consumed_rows),
                 "exposure_consumed_count_mismatch")
        _require(type(group.get("unconsumed_count")) is int
                 and group["unconsumed_count"] == len(remaining) == EXPECTED_REMAINDERS[key],
                 "initial_remainder_count_mismatch")
    return {
        "exposure_audit_sha256": _file_hash(audit_path)[1],
        "prior_corpus_manifest_sha256": PRIOR_MANIFEST_SHA256,
        "prior_corpus_file_sha256": _file_hash(prior_path)[1],
        "consumed_counts": {f"{key[0]}/{key[1]}": len(rows) for key, rows in consumed.items()},
        "rule": "union_of_all_prior_case_source_row_spans_partial_exposure_excludes_whole_row",
    }, consumed


def _identity(source: dict, identity: dict) -> dict:
    source_id = (identity["physical_row_ordinal"] if source["domain"] == "gsm8k"
                 else identity["task_id_or_null"])
    # GSM8K has no publisher task ID; its pinned physical ordinal is the original identity.
    task_id = (identity["row_key"] if source["domain"] == "gsm8k" else
               str(identity["task_id_or_null"]))
    return {"benchmark": source["domain"], "task_id": task_id, "source_id": source_id,
            "source_ref": identity["row_key"], "source_row_sha256": identity["source_row_sha256"]}


def _audit_record(source: dict, identity: dict, raw: dict) -> dict:
    problem = raw["question"] if source["domain"] == "gsm8k" else raw[
        "prompt" if source["domain"] == "humaneval" else "text"]
    audit = {
        **_identity(source, identity), "physical_row_ordinal": identity["physical_row_ordinal"],
        "task_id_or_null": identity["task_id_or_null"],
        "source_artifact_sha256": source["sha256"], "source_artifact_url": source["url"],
        "source_field_sha256": identity["source_field_sha256"], "raw_source": raw,
        "problem_text": problem,
        "question_raw_sha256": _text_hash(problem),
        "question_normalized_sha256": _text_hash(_normalized(problem)),
        "normalization_policy": _NORMALIZATION,
        "independence_status": "UNASSESSED_SHARED_CLUSTERING_REQUIRED",
        "code_normalization_limit": "text_fingerprint_not_semantic_code_equivalence",
    }
    if source["domain"] in {"humaneval", "mbpp"}:
        code = raw["canonical_solution" if source["domain"] == "humaneval" else "code"]
        tests = ([raw["test"]] if source["domain"] == "humaneval" else
                 raw["test_list"] + raw["challenge_test_list"])
        audit.update(code_raw_sha256=_text_hash(code),
                     code_normalized_sha256=_text_hash(_normalized(code)),
                     test_raw_sha256=[_text_hash(test) for test in tests],
                     test_normalized_sha256=[_text_hash(_normalized(test)) for test in tests],
                     code_execution="NEVER_EXECUTED_BY_DATA_BUILDER")
    return audit


def _evaluation_records(source: dict, identity: dict, raw: dict) -> tuple[dict, dict, dict]:
    benchmark = source["domain"]
    common = _identity(source, identity)
    prompt = ("Question:\n" + raw["question"] + "\nAnswer:\n" if benchmark == "gsm8k"
              else raw["prompt"])
    generation = {**common, "prompt": prompt, "prompt_sha256": _text_hash(prompt),
                  "generation_policy": GENERATION_POLICIES[benchmark]}
    scoring = ({"answer": raw["answer"]} if benchmark == "gsm8k" else {
        "prompt": raw["prompt"], "test": raw["test"], "entry_point": raw["entry_point"],
        "timeout_seconds": 3,
    })
    return generation, {**common, "scoring": scoring}, _audit_record(source, identity, raw)


def _training_record(source: dict, identity: dict, raw: dict) -> dict:
    math = source["domain"] == "gsm8k"
    problem = raw["question" if math else "text"]
    response = raw["answer" if math else "code"]
    prompt = "Question:\n" + problem + "\nAnswer:\n"
    return {
        **_identity(source, identity), "domain": "math" if math else "code",
        "source_domain": source["domain"], "split": "train", "row_key": identity["row_key"],
        "physical_row_ordinal": identity["physical_row_ordinal"],
        "source_artifact_sha256": source["sha256"],
        "source_field_sha256": identity["source_field_sha256"],
        "problem_text": problem,
        "prompt": prompt, "response": response, "prompt_sha256": _text_hash(prompt),
        "response_sha256": _text_hash(response), "problem_raw_sha256": _text_hash(problem),
        "problem_normalized_sha256": _text_hash(_normalized(problem)),
        "normalization_policy": _NORMALIZATION, "sft_input_fields": ["prompt", "response"],
        "audit": _audit_record(source, identity, raw),
    }


def _write_jsonl(path: Path, records: list[dict]) -> dict:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(_compact(record) + "\n")
    size, digest = _file_hash(path)
    return {"path": path.name, "bytes": size, "sha256": digest, "rows": len(records)}


def _write_manifest(path: Path, manifest: dict) -> None:
    manifest["manifest_payload_sha256"] = _text_hash(_compact(manifest))
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_compact(manifest) + "\n")


def _failure_report(root: Path, reason: str) -> AdmissionError:
    path = _inside(root, root / f"admission-failure-{uuid.uuid4().hex}.json")
    report = {"schema": "p3-stage2-core-data-admission-failure-v1",
              "status": "DATA_ADMISSION_FAILED", "reason": reason,
              "admitted_denominator": None, "quality_non_degradation_admitted": False}
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_compact(report) + "\n")
    return AdmissionError(path, reason)


def build_core_sources(*, cache_dir: Path, output_dir: Path, exposure_audit_path: Path) -> dict:
    """Admit full pinned sources, exclude proven exposure, and publish joined text artifacts.

    No extra deduplication or token allocation occurs here. Only a successful root
    manifest commits the complete build; payloads alone never establish admission.
    """
    root = Path(os.path.realpath(output_dir))
    root.mkdir(parents=True, exist_ok=True)
    try:
        for name in _ARTIFACTS:
            _require(not _inside(root, root / name).exists(), f"existing_artifact_preserved:{name}")
        loaded = {_source_key(source): _load_legacy(source, Path(cache_dir), root)
                  for source in SOURCES}
        exposure, consumed = _verify_exposure(Path(exposure_audit_path), loaded)
        mbpp = _load_mbpp(MBPP_SOURCE, Path(cache_dir), root)
        source_counts = {}
        training = []
        evaluations = {}
        with tempfile.TemporaryDirectory(prefix=".core-data-admission-", dir=root) as temporary:
            staging = _inside(root, Path(temporary))
            for source in SOURCES:
                key = _source_key(source)
                remaining = [(identity, raw) for identity, raw in loaded[key]
                             if identity["row_key"] not in consumed[key]]
                _require(len(remaining) == EXPECTED_REMAINDERS[key], "initial_remainder_mismatch")
                source_counts[f"{key[0]}/{key[1]}"] = len(remaining)
                if key[1] == "train":
                    training.extend(_training_record(source, identity, raw)
                                    for identity, raw in remaining)
                    continue
                folder = staging / key[0]
                folder.mkdir()
                records = [
                    _evaluation_records(source, identity, raw) for identity, raw in remaining
                ]
                files = {name.removesuffix(".jsonl"): _write_jsonl(
                    folder / name, [record[index] for record in records]
                ) for index, name in enumerate(_ENDPOINT_FILES[:3])}
                manifest = {
                    "schema": "p3-stage2-core-evaluation-v1", "policy": POLICY,
                    "benchmark": key[0], "status": "DATA_ADMITTED_PENDING_SHARED_DEDUP_AND_RUNTIME",
                    "source_row_count": len(remaining),
                    "source_counts": {f"{key[0]}/{key[1]}": len(remaining)},
                    "sources": [{**source, "observed_rows": len(loaded[key])}],
                    "generation_policy": GENERATION_POLICIES[key[0]], "files": files,
                    "manifest_path": "manifest.json", "exposure": exposure,
                    "source_row_hash_policy": "sha256_sorted_compact_utf8_json_all_original_fields",
                    "independent_problem_count": None, "overlap_admission": "PARENT_REQUIRED",
                    "runtime_admission": "PENDING_NATIVE_GENERATION_AND_QUALIFIED_CODE_EXECUTOR",
                    "quality_non_degradation_admitted": False,
                }
                _write_manifest(folder / "manifest.json", manifest)
                evaluations[key[0]] = manifest
            training.extend(_training_record(MBPP_SOURCE, identity, raw) for identity, raw in mbpp)
            source_counts["mbpp/train"] = len(mbpp)
            training_file = _write_jsonl(staging / "training-sources.jsonl", training)
            result = {
                "schema": POLICY, "policy": POLICY,
                "status": "DATA_ADMITTED_PENDING_SHARED_DEDUP_AND_RUNTIME",
                "source_counts": source_counts, "evaluations": evaluations,
                "sources": [*SOURCES, MBPP_SOURCE], "exposure": exposure,
                "files": {"training_sources": training_file}, "manifest_path": "manifest.json",
                "training_input_contract": "only_prompt_and_response_never_audit_fields",
                "packing_and_allocation": "PARENT_REQUIRED_NO_TOKENIZATION_PERFORMED",
                "initial_counts_before_parent_overlap_exclusions": True,
                "independent_problem_count": None, "quality_non_degradation_admitted": False,
            }
            _write_manifest(staging / "manifest.json", result)
            for name in _ARTIFACTS:
                destination = _inside(root, root / name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.link(staging / name, destination)
            return result
    except _ContractError as error:
        raise _failure_report(root, str(error)) from None
    except Exception as error:
        # Never echo arbitrary decoder/file content into a report or CLI error.
        reason = f"source_or_artifact_admission:{type(error).__name__}"
        raise _failure_report(root, reason) from None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exposure-audit", type=Path, required=True)
    arguments = parser.parse_args()
    result = build_core_sources(cache_dir=arguments.cache_dir, output_dir=arguments.output,
                                exposure_audit_path=arguments.exposure_audit)
    from rich.console import Console

    Console().print(_compact({"status": result["status"], "source_counts": result["source_counts"],
                             "manifest_payload_sha256": result["manifest_payload_sha256"]}))


if __name__ == "__main__":
    main()
