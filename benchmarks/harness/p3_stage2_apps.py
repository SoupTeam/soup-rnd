"""Pinned, data-only APPS admission and aggregation of trusted executor records.

This module never imports the official evaluator, executes candidate/reference code,
loads a model, or qualifies a sandbox. Shared overlap admission and every execution
remain the integration owner's responsibility. Source metadata is not independence
proof. No schema failure silently reduces the complete two-shard test denominator.

Public interfaces:
    build_evaluation(*, cache_dir: Path, output_dir: Path) -> dict
    make_prompt(record: dict) -> str
    aggregate_execution_result(execution: dict, *, expected_case_count: int) -> dict

``make_prompt`` consumes public question/starter_code and already admitted fn_name
metadata. It never reads input_output or solutions. Scoring stores original JSON as
an immutable string; the executor must decode it afresh for each candidate because
the official comparison mutates expected values. Starter code is prompt text, not
code to glue onto the continuation. All source-row hashes use canonical UTF-8 JSON,
not Parquet physical bytes; source artifacts have separate byte-level SHA-256 pins.

The output manifest is the admission commit marker and is published last. Its files
map has relative generation/scoring/audit paths with byte hashes and row counts;
manifest_path names the manifest itself. manifest_payload_sha256 hashes canonical
manifest JSON before that field is added, avoiding a recursive file-hash claim.
Previously published evidence is never overwritten. An interrupted publication can
leave uncommitted JSONL files but no successful manifest; never infer admission from
those files alone. Failure reports are uniquely named and contain IDs/reasons, not
hidden case examples. Shared clustering must consume every admitted source row.

Accepted trusted executor record, schema ``p3-stage2-apps-execution-v1``:
    {
      "schema": "p3-stage2-apps-execution-v1",
      "generation_count": 1,
      "executor_manifest_sha256": "<64 lowercase hexadecimal characters>",
      "status": "completed" | "compile_error" | "runtime_error" | "timeout"
                | "infrastructure_error",
      "case_results": [{"case_index": 0, "verdict": true}, ...],
      "timeout": {                         # required only for timeout
        "phase": "compile" | "case" | "whole_problem",
        "limit_seconds": 4 | 11,
        "elapsed_seconds": <finite number at least limit_seconds>,
        "termination_reason": "execution_deadline",
        "case_index": <next unobserved index for case; null otherwise>
      },
      "reason": "<optional nonempty infrastructure diagnostic>"
    }

No other fields are accepted. Case records are a zero-based ordered prefix, with
exactly one verdict per observed case. Verdicts are JSON booleans or the exact JSON
integers -2, -1, 0, 1, never floats, nested generations or truthy error objects.
``completed`` requires all expected cases, and only positive verdicts pass. Terminal
compile/runtime/qualified deadline statuses fail without invented test observations.
Compile failure has no case verdicts. Runtime failure and timeout may have a shorter
observed prefix, not complete coverage plus an unrelated terminal status. Compile
and case deadlines are 4 seconds; whole-problem deadline is 11 seconds. Deadline
records must identify an execution deadline, not controller/startup/dependency failure.
Missing, partial, extra, malformed, duplicate or reordered completed results are
infrastructure errors with score null. A complete negative case vector is model failure.

The executor manifest hash is a reference, NOT a qualification check performed here.
The caller MUST match it to independently verified, frozen sandbox/runtime/scorer/
timing evidence and exact task/source-row joins before using these results. A boolean
``sandbox_qualified`` field does not establish that evidence and is rejected. The
caller owns trusted record authenticity; this pure adapter cannot authenticate a hash
or trust a candidate-controlled verdict. No passing aggregate admits an endpoint by
itself. The result status is PASS, FAIL or INFRASTRUCTURE_ERROR, with score 1, 0 or null.

Data-only CLI, never model execution:
    python -m benchmarks.harness.p3_stage2_apps --cache-dir CACHE --output OUTPUT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata
import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterator, TextIO
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from benchmarks.harness.p3_realdata_corpus import _file_hash, _inside, _source_file

BENCHMARK = "apps"
POLICY = "p3-stage2-apps-v1"
REPOSITORY = "codeparrot/apps"
PARQUET_REVISION = "0f10e424e13e1c2a69f851e153097b71b6734a1f"
SOURCE_REVISION = "21e74ddf8de1a21436da12e3e653065c5213e9d1"
OFFICIAL_REVISION = "362aedc3c71cd7d9bd2fc96a6c80e11dbc38c7a5"
ORDERED_SOURCE_IDS_SHA256 = "1580fcfa77255bf7af43dd809450b9fced82475b9ba68bd20d41997b95243d79"
FIELDS = (
    "problem_id", "question", "solutions", "input_output", "difficulty", "url", "starter_code",
)
DIFFICULTIES = frozenset(("introductory", "interview", "competition"))
SOURCES = (
    {
        "repository": REPOSITORY, "revision": PARQUET_REVISION,
        "config": "all", "split": "test", "path": "all/test/0000.parquet",
        "url": f"https://huggingface.co/datasets/{REPOSITORY}/resolve/"
               f"{PARQUET_REVISION}/all/test/0000.parquet",
        "bytes": 424202850, "expected_rows": 2000, "first_problem_id": 0,
        "sha256": "f1c36415a9aabe114558a1b0a943f1dc50f2468f12c741585505f84ace577514",
        "row_groups": [1000, 1000],
        "created_by": "parquet-cpp-arrow version 11.0.0",
    },
    {
        "repository": REPOSITORY, "revision": PARQUET_REVISION,
        "config": "all", "split": "test", "path": "all/test/0001.parquet",
        "url": f"https://huggingface.co/datasets/{REPOSITORY}/resolve/"
               f"{PARQUET_REVISION}/all/test/0001.parquet",
        "bytes": 319566256, "expected_rows": 3000, "first_problem_id": 2000,
        "sha256": "3b02746c5f8ede88388a4635b8d38ab3a29b31f9f964d4758652bcec9dc85913",
        "row_groups": [1000, 1000, 1000],
        "created_by": "parquet-cpp-arrow version 11.0.0",
    },
)
GENERATION_POLICY = {
    "policy": POLICY,
    "prompt": "official_generate_gpt_codes_no_peeking",
    "official_repository": "hendrycks/apps",
    "official_revision": OFFICIAL_REVISION,
    "prompt_source_path": "eval/generate_gpt_codes.py",
    "format": "plain_full_program_completion",
    "chat_template": False,
    "fim_wrapper": False,
    "do_sample": False,
    "num_beams": 1,
    "num_return_sequences": 1,
    "max_new_tokens": 1024,
    "stop": ["eos", "max_new_tokens"],
    "stop_strings": [],
    "append_starter_code": False,
    "truncate_prompt": False,
    "continuation_extraction": "token_ids_after_full_input_width_before_generated_eos",
    "clean_up_tokenization_spaces": False,
    "postprocessing": "none",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ARTIFACTS = ("generation.jsonl", "scoring.jsonl", "audit.jsonl", "manifest.json")


class AdmissionError(ValueError):
    """The intended complete endpoint failed admission; report_path preserves evidence."""

    def __init__(self, report_path: Path, message: str) -> None:
        self.report_path = report_path
        super().__init__(f"APPS_DATA_ADMISSION_FAILED: {message}; report={report_path}")


def _compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False)


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _reject_constant(value: str) -> None:
    raise ValueError("nonfinite_json_number")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _strict_json(raw: str) -> Any:
    result = json.loads(raw, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    # JSON numbers such as 1e999 overflow the default float parser without using NaN tokens.
    pending = [result]
    while pending:
        value = pending.pop()
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite_json_number")
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return result


def _mode(fn_name: Any) -> str:
    if fn_name is None:
        return "standard_input"
    if not isinstance(fn_name, str) or fn_name == "":
        raise ValueError("fn_name_must_be_null_or_nonempty_string")
    return "call_based"


def make_prompt(record: dict) -> str:
    """Build the pinned public-only no-peeking prompt with source whitespace intact."""
    question = record.get("question")
    starter = record.get("starter_code")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question_must_be_nonempty_string")
    if not isinstance(starter, str):
        raise ValueError("starter_code_must_be_string")
    mode = _mode(record.get("fn_name"))
    prompt = "\nQUESTION:\n" + question
    if starter != "":
        prompt += "\n" + starter
    prompt += "\nUse Call-Based format" if mode == "call_based" else "\nUse Standard Input format"
    return prompt + "\nANSWER:\n"


def _validate_cases(raw: Any) -> tuple[dict, str]:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("input_output_must_be_nonempty_json_string")
    try:
        cases = _strict_json(raw)
    except (ValueError, RecursionError, OverflowError):
        raise ValueError("input_output_invalid_json") from None
    if not isinstance(cases, dict):
        raise ValueError("input_output_must_be_object")
    inputs, outputs = cases.get("inputs"), cases.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        raise ValueError("input_output_requires_input_and_output_lists")
    if not inputs or len(inputs) != len(outputs):
        raise ValueError("input_output_requires_nonzero_equal_case_counts")
    mode = _mode(cases.get("fn_name"))
    for index, (arguments, expected) in enumerate(zip(inputs, outputs)):
        if mode == "call_based":
            if not isinstance(arguments, list):
                raise ValueError(f"call_based_case_{index}_requires_positional_argument_list")
            # Every strict JSON output value supports the official Python equality branch.
            # Do not pre-apply its narrow, exception-caught integer dictionary-key repairs.
        else:
            for name, value in (("input", arguments), ("output", expected)):
                if not (isinstance(value, str) or (
                    isinstance(value, list) and all(isinstance(item, str) for item in value)
                )):
                    raise ValueError(f"standard_input_case_{index}_{name}_requires_string_or_lines")
    return cases, mode


def _iter_rows(path: Path, source: dict) -> Iterator[tuple[int, dict]]:
    """Decode one row at a time; never materialize a shard or both full shards."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    try:
        metadata = parquet.metadata
        if metadata.num_rows != source["expected_rows"]:
            raise ValueError("source_row_count_mismatch")
        schema = parquet.schema_arrow
        if schema.names != list(FIELDS):
            raise ValueError("source_schema_field_mismatch")
        for field in schema:
            expected = pa.int64() if field.name == "problem_id" else pa.string()
            if field.type != expected or not field.nullable:
                raise ValueError(f"source_schema_type_or_nullability_mismatch:{field.name}")
        groups = [metadata.row_group(index).num_rows for index in range(metadata.num_row_groups)]
        if "row_groups" in source and groups != source["row_groups"]:
            raise ValueError("source_row_group_count_mismatch")
        if "created_by" in source and metadata.created_by != source["created_by"]:
            raise ValueError("source_created_by_mismatch")
        ordinal = 0
        for batch in parquet.iter_batches(batch_size=1, use_threads=False):
            for row in batch.to_pylist():
                yield ordinal, row
                ordinal += 1
        if ordinal != source["expected_rows"]:
            raise ValueError("decoded_source_row_count_mismatch")
    finally:
        parquet.close()


def _source_ref(source: dict, ordinal: int) -> str:
    return (f"{source['repository']}@{source['revision']}/{source['config']}/"
            f"{source['split']}/{source['path']}#row={ordinal}")


def _task_id(source: dict, source_id: int) -> str:
    return (f"{BENCHMARK}:{source['repository']}@{source['revision']}/"
            f"{source['config']}/{source['split']}/{source_id}")


def _url_provenance(raw: Any) -> dict:
    if raw is None or raw == "":
        return {"status": "missing", "raw": raw,
                "raw_sha256": _text_hash(raw) if isinstance(raw, str) else None, "canonical": None}
    result = {"status": "ambiguous", "raw": raw, "raw_sha256": _text_hash(raw),
              "canonical": None, "canonicalization": "conservative_url_v1_no_mirror_inference"}
    try:
        parts = urlsplit(raw)
        if (parts.scheme.lower() not in {"http", "https"} or not parts.hostname
                or parts.username is not None or parts.password is not None
                or any(character.isspace() for character in raw)):
            return result
        # Remove only recognized tracking fields; retain all task-selecting query values.
        query = [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
                 if not key.lower().startswith("utm_") and key.lower() not in {"fbclid", "gclid"}]
        canonical = urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path,
                               urlencode(query), ""))
    except ValueError:
        return result
    result.update(status="present_unresolved_identity", canonical=canonical,
                  canonical_sha256=_text_hash(canonical))
    return result


def _solutions_provenance(raw: Any) -> dict:
    result = {"status": "missing", "raw_sha256": _text_hash(raw) if isinstance(raw, str) else None,
              "code_sha256": [], "code_normalized_sha256": []}
    if raw is None or not raw.strip():
        return result
    try:
        values = _strict_json(raw)
    except (ValueError, RecursionError, OverflowError):
        result["status"] = "ambiguous_invalid_reference_json_not_required_for_scoring"
        return result
    if values is None or values == []:
        return result
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        result["status"] = "ambiguous_reference_shape_not_required_for_scoring"
        return result
    result.update(status="present_unexecuted", code_sha256=[_text_hash(value) for value in values],
                  code_normalized_sha256=[_text_hash(_normalized(value)) for value in values])
    return result


def _records(row: dict, source: dict, ordinal: int) -> tuple[dict, dict, dict]:
    source_id = row.get("problem_id")
    expected_id = source["first_problem_id"] + ordinal
    if type(source_id) is not int or source_id != expected_id:
        raise ValueError(f"source_id_coverage_mismatch:expected={expected_id}")
    for field in ("question", "starter_code", "difficulty"):
        if not isinstance(row.get(field), str):
            raise ValueError(f"{field}_must_be_string")
    if row["difficulty"] not in DIFFICULTIES:
        raise ValueError("difficulty_not_recognized")
    for field in ("solutions", "url"):
        if row.get(field) is not None and not isinstance(row[field], str):
            raise ValueError(f"{field}_must_be_string_or_null")
    cases, mode = _validate_cases(row.get("input_output"))
    prompt = make_prompt({"question": row["question"], "starter_code": row["starter_code"],
                          "fn_name": cases.get("fn_name")})
    source_hash = _text_hash(_compact(row))
    identity = {"benchmark": BENCHMARK, "task_id": _task_id(source, source_id),
                "source_row_sha256": source_hash}
    reference = _source_ref(source, ordinal)
    generation = {
        **identity, "source_id": source_id, "source_ref": reference,
        "prompt": prompt, "prompt_sha256": _text_hash(prompt),
        "generation_policy": GENERATION_POLICY,
    }
    scoring = {**identity, "scoring": {
        "mode": mode, "fn_name": cases.get("fn_name"), "case_count": len(cases["inputs"]),
        "input_output_json": row["input_output"],
        "input_output_sha256": _text_hash(row["input_output"]),
        "decode_policy": "strict_json_no_duplicates_no_nonfinite_fresh_per_candidate",
        "official_repository": "hendrycks/apps", "official_revision": OFFICIAL_REVISION,
        "official_scorer_path": "eval/testing_util.py",
        "case_state": "reuse_module_or_solution_instance_within_problem_only",
        "candidate_program": "entire_raw_continuation_without_starter_glue_or_repairs",
        "step_limit_seconds": 4, "whole_problem_limit_seconds": 11,
        "executor_admission": "INDEPENDENT_QUALIFICATION_REQUIRED",
    }}
    audit = {
        **identity, "source_id": source_id, "source_ref": reference,
        "physical_row_ordinal": ordinal, "source_artifact_url": source["url"],
        "source_artifact_sha256": source["sha256"], "difficulty": row["difficulty"],
        "source_lineage": {
            "repository": source["repository"], "parquet_revision": source["revision"],
            "loader_revision": SOURCE_REVISION,
            "conversion_equivalence": "PUBLISHER_LINEAGE_NOT_INDEPENDENT_ROW_EQUIVALENCE_PROOF",
        },
        "source_url_provenance": _url_provenance(row.get("url")),
        "problem_text": row["question"],
        "question_raw_sha256": _text_hash(row["question"]),
        "question_normalized_sha256": _text_hash(_normalized(row["question"])),
        "starter_code_raw_sha256": _text_hash(row["starter_code"]),
        "starter_code_normalized_sha256": _text_hash(_normalized(row["starter_code"])),
        "normalization_policy": "unicode_NFKC_then_unicode_whitespace_collapse_v1",
        "code_normalization_limit": "text_fingerprint_not_semantic_code_equivalence",
        "input_output_raw_sha256": _text_hash(row["input_output"]),
        "test_cases_sha256": _text_hash(_compact({
            "fn_name": cases.get("fn_name"), "inputs": cases["inputs"], "outputs": cases["outputs"],
        })),
        "case_count": len(cases["inputs"]), "evaluation_mode": mode,
        "solutions_provenance": _solutions_provenance(row.get("solutions")),
        "independence_status": "UNASSESSED_SHARED_CLUSTERING_REQUIRED",
    }
    return generation, scoring, audit


def _write_jsonl(stream: TextIO, value: dict) -> None:
    stream.write(_compact(value) + "\n")


def _failure_report(root: Path, findings: list[dict]) -> AdmissionError:
    report_path = _inside(root, root / f"admission-failure-{uuid.uuid4().hex}.json")
    report = {
        "schema": "p3-stage2-apps-admission-failure-v1", "status": "DATA_ADMISSION_FAILED",
        "benchmark": BENCHMARK, "intended_source_rows": sum(s["expected_rows"] for s in SOURCES),
        "admitted_denominator": None, "apps_strict_accuracy": None,
        "quality_non_degradation_admitted": False, "findings": findings,
    }
    with report_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_compact(report) + "\n")
    return AdmissionError(report_path, "; ".join(item["reason"] for item in findings[:3]))


def build_evaluation(*, cache_dir: Path, output_dir: Path) -> dict:
    """Stream both pinned shards into joined artifacts, publishing only complete admission.

    Existing evidence files are never overwritten. A failure writes a uniquely named
    admission report and raises AdmissionError; staged partial output is discarded.
    A successful source count is inventory, not a frozen independent denominator.
    """
    root = Path(os.path.realpath(output_dir))
    root.mkdir(parents=True, exist_ok=True)
    findings: list[dict] = []
    for name in _ARTIFACTS:
        if _inside(root, root / name).exists():
            findings.append({"source_id": None, "reason": f"existing_artifact_preserved:{name}"})
    if findings:
        raise _failure_report(root, findings)
    ordered_ids = hashlib.sha256()
    observed_count = 0
    source_reports = []
    try:
        with tempfile.TemporaryDirectory(prefix=".apps-admission-", dir=root) as temporary:
            staging = _inside(root, Path(temporary))
            with ExitStack() as stack:
                streams = [stack.enter_context((staging / name).open(
                    "x", encoding="utf-8", newline="\n",
                )) for name in _ARTIFACTS[:3]]
                for source in SOURCES:
                    observed_source_rows = 0
                    try:
                        path = _source_file(source, Path(cache_dir), root)
                        for ordinal, row in _iter_rows(path, source):
                            observed_source_rows += 1
                            source_id = row.get("problem_id")
                            if type(source_id) is int:
                                ordered_ids.update(f"{source_id}\n".encode("ascii"))
                            try:
                                records = _records(row, source, ordinal)
                                for stream, record in zip(streams, records):
                                    _write_jsonl(stream, record)
                                observed_count += 1
                            except (ValueError, TypeError, RecursionError, OverflowError) as error:
                                findings.append({
                                    "source_id": source_id,
                                    "source_ref": _source_ref(source, ordinal),
                                    "physical_row_ordinal": ordinal, "reason": str(error),
                                })
                    except Exception as error:
                        # A corrupt/download/schema artifact is infrastructure, never a model score.
                        findings.append({
                            "source_id": None, "source_artifact_url": source["url"],
                            "reason": f"source_admission:{type(error).__name__}:{error}",
                        })
                    source_reports.append({**source, "observed_rows": observed_source_rows})
            if ordered_ids.hexdigest() != ORDERED_SOURCE_IDS_SHA256:
                findings.append({"source_id": None, "reason": "ordered_source_ids_sha256_mismatch",
                                 "observed_sha256": ordered_ids.hexdigest()})
            if observed_count != sum(source["expected_rows"] for source in SOURCES):
                findings.append({
                    "source_id": None, "reason": "complete_source_row_count_not_admitted",
                })
            if findings:
                raise _failure_report(root, findings)
            files = {}
            for name in _ARTIFACTS[:3]:
                size, digest = _file_hash(staging / name)
                files[name.removesuffix(".jsonl")] = {
                    "path": name, "bytes": size, "sha256": digest, "rows": observed_count,
                }
            manifest = {
                "schema": "p3-stage2-apps-evaluation-v1", "benchmark": BENCHMARK, "policy": POLICY,
                "status": "DATA_ADMITTED_PENDING_SHARED_DEDUP_AND_INFERENCE",
                "source_row_count": observed_count, "source_counts": {"all/test": observed_count},
                "ordered_source_ids_sha256": ordered_ids.hexdigest(), "sources": source_reports,
                "source_pins": {"loader_revision": SOURCE_REVISION,
                                "official_evaluator_revision": OFFICIAL_REVISION},
                "generation_policy": GENERATION_POLICY, "files": files,
                "manifest_path": "manifest.json",
                "source_row_hash_policy": (
                    "sha256_canonical_sorted_compact_utf8_json_all_source_fields"
                ),
                "json_parser_policy": "reject_duplicate_keys_nonfinite_numbers_and_float_overflow",
                "independent_problem_count": None, "overlap_admission": "PARENT_REQUIRED",
                "executor_admission": "INDEPENDENT_QUALIFICATION_REQUIRED",
                "apps_strict_accuracy": None, "quality_non_degradation_admitted": False,
            }
            # Hash the canonical payload before adding this nonrecursive identity field.
            manifest["manifest_payload_sha256"] = _text_hash(_compact(manifest))
            (staging / "manifest.json").write_text(_compact(manifest) + "\n", encoding="utf-8")
            # Exclusive same-filesystem links cannot overwrite evidence on any platform.
            # Publish the manifest commit marker last; staging cleanup removes its links.
            for name in _ARTIFACTS:
                destination = _inside(root, root / name)
                os.link(staging / name, destination)
            return manifest
    except AdmissionError:
        raise
    except Exception as error:
        raise _failure_report(root, [{
            "source_id": None, "reason": f"artifact_admission:{type(error).__name__}:{error}",
        }]) from error


def _valid_timeout(timeout: Any, observed: int, expected: int) -> bool:
    if not isinstance(timeout, dict) or set(timeout) != {
        "phase", "limit_seconds", "elapsed_seconds", "termination_reason", "case_index",
    }:
        return False
    phase = timeout["phase"]
    if not isinstance(phase, str) or phase not in {"compile", "case", "whole_problem"}:
        return False
    limit, elapsed = timeout["limit_seconds"], timeout["elapsed_seconds"]
    required_limit = 11 if phase == "whole_problem" else 4
    if type(limit) not in (int, float) or limit != required_limit:
        return False
    if (type(elapsed) not in (int, float) or elapsed < limit
            or (type(elapsed) is float and not math.isfinite(elapsed))):
        return False
    if timeout["termination_reason"] != "execution_deadline":
        return False
    index = timeout["case_index"]
    if phase == "case":
        return type(index) is int and index == observed and index < expected
    return index is None and (phase != "compile" or observed == 0)


def aggregate_execution_result(execution: dict, *, expected_case_count: int) -> dict:
    """Validate one trusted candidate's record and return strict problem-level success.

    This function never runs or repairs code and never upgrades a missing observation
    into a model verdict. The module docstring defines the entire accepted schema.
    """
    result = {
        "status": "INFRASTRUCTURE_ERROR", "score": None,
        "expected_case_count": expected_case_count, "observed_case_count": 0,
        "passed_case_count": 0, "complete_case_coverage": False, "terminal_status": None,
        "reason": "invalid_expected_case_count",
    }
    if type(expected_case_count) is not int or expected_case_count <= 0:
        return result
    if not isinstance(execution, dict):
        result["reason"] = "executor_record_must_be_object"
        return result
    required = {"schema", "generation_count", "executor_manifest_sha256", "status", "case_results"}
    status = execution.get("status")
    allowed = required | {"reason"} | ({"timeout"} if status == "timeout" else set())
    if not required <= set(execution) or not set(execution) <= allowed:
        result["reason"] = "executor_record_fields_mismatch"
        return result
    if execution["schema"] != "p3-stage2-apps-execution-v1":
        result["reason"] = "executor_schema_mismatch"
        return result
    if type(execution["generation_count"]) is not int or execution["generation_count"] != 1:
        result["reason"] = "exactly_one_generation_required"
        return result
    executor_hash = execution["executor_manifest_sha256"]
    if not isinstance(executor_hash, str) or not _SHA256.fullmatch(executor_hash):
        result["reason"] = "executor_manifest_reference_required_not_sandbox_qualification"
        return result
    if not isinstance(status, str) or status not in {
        "completed", "compile_error", "runtime_error", "timeout", "infrastructure_error",
    }:
        result["reason"] = "unrecognized_executor_status"
        return result
    if "reason" in execution and (
        not isinstance(execution["reason"], str) or not execution["reason"].strip()
    ):
        result["reason"] = "malformed_executor_diagnostic"
        return result
    cases = execution["case_results"]
    if not isinstance(cases, list) or len(cases) > expected_case_count:
        result["reason"] = "malformed_or_extra_case_results"
        return result
    passed = 0
    for index, case in enumerate(cases):
        if (not isinstance(case, dict) or set(case) != {"case_index", "verdict"}
                or type(case["case_index"]) is not int or case["case_index"] != index):
            result["reason"] = "missing_duplicate_or_reordered_case_identity"
            return result
        verdict = case["verdict"]
        if not (type(verdict) is bool or (type(verdict) is int and verdict in {-2, -1, 0, 1})):
            result["reason"] = "malformed_case_verdict"
            return result
        passed += int(verdict > 0)
    observed = len(cases)
    result.update(observed_case_count=observed, passed_case_count=passed,
                  complete_case_coverage=observed == expected_case_count)
    if status == "infrastructure_error":
        result["reason"] = execution.get("reason", "executor_reported_infrastructure_error")
        return result
    if status == "completed":
        if observed != expected_case_count:
            result["reason"] = "incomplete_case_coverage"
            return result
        success = passed == expected_case_count
        result.update(status="PASS" if success else "FAIL", score=int(success),
                      reason="all_cases_positive" if success else "nonpositive_case_verdict")
        return result
    if observed == expected_case_count or (status == "compile_error" and observed != 0):
        result["reason"] = "terminal_status_conflicts_with_case_coverage"
        return result
    if status == "timeout" and not _valid_timeout(
        execution.get("timeout"), observed, expected_case_count,
    ):
        result["reason"] = "unqualified_or_infrastructure_timeout"
        return result
    result.update(status="FAIL", score=0, terminal_status=status, reason=f"terminal_{status}")
    return result


def main(argv: list[str] | None = None) -> int:
    """Prepare data only, with no execution or model/scorer imports."""
    from rich.console import Console

    parser = argparse.ArgumentParser(description="Prepare pinned, data-only APPS test artifacts.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    console = Console()
    try:
        manifest = build_evaluation(cache_dir=args.cache_dir, output_dir=args.output)
    except AdmissionError as error:
        console.print(str(error), markup=False)
        return 1
    console.print(f"{manifest['status']}: {manifest['source_row_count']} source rows; "
                  f"manifest={args.output / 'manifest.json'}", markup=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
