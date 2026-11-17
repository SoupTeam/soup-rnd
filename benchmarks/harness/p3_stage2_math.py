"""Prepare the pinned Eleuther MATH test split and score string continuations.

Data-only preparation, on Kaggle::

    python -m benchmarks.harness.p3_stage2_math --cache-dir /kaggle/working/cache \
        --output /kaggle/working/math-evaluation

Metadata-only diagnosis checks full pinned source coverage and records auxiliary
content relationships without loading the scorer or publishing evaluation inputs.
Add ``--metadata-only`` and use a fresh output directory. Reports contain source
identities, raw auxiliary IDs, labels, counts, and hashes, never answer text.

``build_evaluation`` publishes only after all 5,000 pinned Eleuther config/test rows,
source identities, and unchanged gold answers pass admission. Split provenance uses
the published archive-loader lineage, not independently verified original filenames.
All 12,500 PRM rows remain auxiliary provenance and overlap data, never ID authority.
``score_completion`` accepts one scoring.jsonl row and the already EOS/cap-trimmed
continuation, never the prompt. It raises on infrastructure or malformed gold.
It does not tokenize, generate, execute answers, or use a symbolic evaluator.

The unchanged trusted normalizer and box-search module are fetched and imported
only on Kaggle, after verification against their original Git blob identities.
The upstream license accompanies those runtime assets. No reference normalizer
is copied into this module. See research-math.md for the frozen protocol and
provenance; pending shared overlap review is not evidence of independence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, TextIO
from urllib.request import urlopen

from benchmarks.harness.p3_realdata_corpus import (
    _compact,
    _file_hash,
    _inside,
    _source_file,
    _text_hash,
    _verify_file,
)

BENCHMARK = "math"
POLICY = "p3-stage2-math-original-test-string-v1"
DATASET_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
CROSSWALK_REVISION = "7ecc794703b2877f63226f2477a49b34f9b25163"
SCORER_REVISION = "985bdc1696e88e8643f081a0ff4719da39f2ae2a"
SOURCE_ADMISSION_POLICY = "p3-stage2-math-eleuther-test-source-v2"
SOURCE_LINEAGE = {
    "historical_loader_revision": "3730e0d9543219af79a2ddf93da274bb55d7fc27",
    "historical_loader_url": (
        "https://huggingface.co/datasets/EleutherAI/hendrycks_math/raw/"
        "3730e0d9543219af79a2ddf93da274bb55d7fc27/hendrycks_math.py"
    ),
    "parquet_conversion_revision": "7811dc4cf37bd3a0384adc84f33da0695629b4dc",
    "basis": "published_loader_maps_MATH_split_config_directories_to_preserved_Parquet_splits",
    "original_filename_verification": False,
    "auxiliary_PRM_ids_authoritative": False,
}
ORIGINAL_POOL_ROWS = 12500
TEST_COUNTS = {
    "algebra": 1187,
    "counting_and_probability": 474,
    "geometry": 479,
    "intermediate_algebra": 903,
    "number_theory": 540,
    "prealgebra": 871,
    "precalculus": 546,
}
SUBJECT_DIRECTORIES = {
    "Algebra": "algebra",
    "Counting & Probability": "counting_and_probability",
    "Geometry": "geometry",
    "Intermediate Algebra": "intermediate_algebra",
    "Number Theory": "number_theory",
    "Prealgebra": "prealgebra",
    "Precalculus": "precalculus",
}
_PARQUET_IDENTITIES = (
    ("algebra", 353114, "0e04a850dac9cfc758eac0c80afd2d0cc65e29ea72f916c8ac1045b12af937f5"),
    (
        "counting_and_probability", 175032,
        "c324818949c75a6227c0328a8499d58023cbbf6757cfc1736f5ab24cc8d87966",
    ),
    ("geometry", 263835, "98720f3a8134ad9784fd8951195245264495df03ffc5c9485a9a54ff5ae2431a"),
    (
        "intermediate_algebra", 394797,
        "e859f72486c951c2017da880541ca729940eaa986b0364eb23e238983df9bfeb",
    ),
    (
        "number_theory", 182003,
        "ba156c9c9fa2040836ba12135fd964e6e67d6bccd8c469ec4a3ad941f8f6fbe9",
    ),
    ("prealgebra", 267729, "301e5d8ade5c7b5038032532e498bdc9dca799e7f1061a63f2e7d980803240cc"),
    ("precalculus", 242241, "da37474ba3761ae5727af3844a7ba963a735bd6a14c65e6137593669a4fe3b9c"),
)
PARQUET_SOURCES = tuple(
    {
        "repository": "EleutherAI/hendrycks_math",
        "revision": DATASET_REVISION,
        "config": subject,
        "split": "test",
        "path": f"{subject}/test-00000-of-00001.parquet",
        "url": (
            "https://huggingface.co/datasets/EleutherAI/hendrycks_math/resolve/"
            f"{DATASET_REVISION}/{subject}/test-00000-of-00001.parquet"
        ),
        "fields": ["problem", "level", "type", "solution"],
        "expected_rows": TEST_COUNTS[subject],
        "bytes": size,
        "sha256": digest,
    }
    for subject, size, digest in _PARQUET_IDENTITIES
)
CROSSWALK_SOURCES = tuple(
    {
        "repository": "openai/prm800k",
        "revision": CROSSWALK_REVISION,
        "path": f"prm800k/math_splits/{name}.jsonl",
        "url": (
            "https://media.githubusercontent.com/media/openai/prm800k/"
            f"{CROSSWALK_REVISION}/prm800k/math_splits/{name}.jsonl"
        ),
        "expected_rows": rows,
        "bytes": size,
        "sha256": digest,
    }
    for name, rows, size, digest in (
        (
            "train", 12000, 10896985,
            "90d96daeac3fe343ebb1e22ce93dd99690f75983e957f88de42f87cffe1e8076",
        ),
        (
            "test", 500, 446564,
            "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132",
        ),
    )
)
# These are Git blob SHA-1 values, not file SHA-256 values. Admission also records
# SHA-256 of the actual verified bytes, without inventing a preverified SHA-256 pin.
SCORER_ASSETS = tuple(
    {
        "repository": "hendrycks/math",
        "revision": SCORER_REVISION,
        "path": path,
        "url": f"https://raw.githubusercontent.com/hendrycks/math/{SCORER_REVISION}/{path}",
        "bytes": size,
        "git_blob_sha1": blob,
    }
    for path, size, blob in (
        ("modeling/math_equivalence.py", 4471, "b5c066fb6e88d9dae845999d8776a3e8c9ea1c82"),
        ("modeling/dataset/util.py", 2522, "f0311c39b18ed66fd2ac0fd9b510899765a4c8ab"),
        ("LICENSE", 1070, "2884bd5e0ebe3af32f148e407ca9648153b6f2fa"),
    )
)
REMOVE_BOXED_PROVENANCE = {
    "repository": "hendrycks/math",
    "revision": SCORER_REVISION,
    "path": "modeling/eval_math_gpt.py",
    "git_blob_sha1": "3ba416a1f63c3bf3d0e705820e167a99e6879f1d",
    "function": "remove_boxed",
    "adaptation": "Equivalent strict wrapper check for string-or-None inputs only",
    "license_asset": "LICENSE",
}
GENERATION_POLICY = {
    "policy": POLICY,
    "model": "ibm-granite/granite-3.1-1b-a400m-base",
    "model_revision": "408b6e90baab8cf24f4aa9f8e19703ffa0a53b29",
    "prompt_mode": "plain_completion_zero_shot",
    "add_special_tokens": False,
    "chat_template": None,
    "truncation": False,
    "do_sample": False,
    "num_beams": 1,
    "num_return_sequences": 1,
    "max_new_tokens": 1024,
    "min_new_tokens": 0,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "extra_logits_processors": [],
    "forced_answer_tokens": [],
    "literal_text_stops": [],
    "termination": ["pinned_tokenizer_eos", "1024_new_token_cap"],
    "decode_continuation_only": True,
    "exclude_first_generated_eos_and_following_padding": True,
    "skip_special_tokens": False,
    "clean_up_tokenization_spaces": False,
    "retain_raw_token_ids_and_text": True,
    "context_overflow": "block_full_set_protocol_never_truncate_or_drop",
    "budget_hit": "score_fixed_prefix_without_extension_or_retry",
}
PROMPT_PREFIX = "Problem:\n"
PROMPT_SUFFIX = "\nSolve the problem and put the final answer in \\boxed{...}.\nSolution:\n"
NORMALIZED_PROBLEM_POLICY = "Unicode NFKC followed by split/join whitespace with one ASCII space"
_OUTPUT_NAMES = (
    "generation.jsonl", "scoring.jsonl", "audit.jsonl", "crosswalk_audit.jsonl", "manifest.json",
)
_METADATA_OUTPUT_NAMES = (
    "metadata-crosswalk.jsonl", "metadata-joins.jsonl", "metadata-report.json",
)
_SCORER: ReferenceScorer | None = None


class AdmissionError(ValueError):
    """The full prospective dataset could not be admitted; no subset is valid."""

    def __init__(self, message: str, *, metadata: dict | None = None):
        super().__init__(message)
        self.metadata = {} if metadata is None else metadata


class ScorerInfrastructureError(RuntimeError):
    """A trusted asset, runtime, or scoring input failed, rather than the model."""


class MalformedGoldError(AdmissionError):
    """The original gold cannot be scored under the frozen metric."""


@dataclass(frozen=True)
class ReferenceScorer:
    equivalence: ModuleType
    boxes: ModuleType
    assets: tuple[dict, ...]


def _on_kaggle() -> bool:
    return sys.platform.startswith("linux") and Path("/kaggle/working").is_dir()


def _normalized_problem(problem: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", problem).split())


def make_prompt(record: dict) -> str:
    """Use only the unchanged problem, with the prospectively frozen zero-shot text."""
    problem = record.get("problem")
    if not isinstance(problem, str):
        raise AdmissionError("MATH problem must be a string")
    return PROMPT_PREFIX + problem + PROMPT_SUFFIX


def _field_hashes(record: dict) -> dict[str, str]:
    return {
        field: _text_hash(value if isinstance(value, str) else _compact(value))
        for field, value in sorted(record.items())
    }


def _source_ref(source: dict, ordinal: int) -> str:
    return (
        f"{source['repository']}@{source['revision']}/{source['path']}"
        f"#row={ordinal}"
    )


def _verify_asset(path: Path, source: dict) -> tuple[int, str]:
    """Verify payload SHA-256, or the explicitly named Git blob SHA-1 identity."""
    if "sha256" in source:
        try:
            _verify_file(path, source)
        except ValueError as error:
            raise AdmissionError(str(error)) from error
        return source["bytes"], source["sha256"]
    if "git_blob_sha1" not in source:
        raise AdmissionError("MATH source has no immutable identity")
    blob_hash = hashlib.sha1(usedforsecurity=False)
    blob_hash.update(f"blob {source['bytes']}\0".encode("ascii"))
    content_hash = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            blob_hash.update(chunk)
            content_hash.update(chunk)
    if size != source["bytes"] or blob_hash.hexdigest() != source["git_blob_sha1"]:
        raise AdmissionError(f"MATH_SOURCE_MISMATCH: {source['repository']}/{source['path']}")
    return size, content_hash.hexdigest()


def _asset_file(source: dict, cache_dir: Path, root: Path) -> Path:
    """Download bounded immutable bytes, retaining verified assets under root/sources."""
    identity = source.get("sha256", source.get("git_blob_sha1"))
    filename = f"{identity}-{Path(source['path']).name}"
    directory = _inside(root, root / "sources")
    directory.mkdir(parents=True, exist_ok=True)
    destination = _inside(root, directory / filename)
    cached = cache_dir / "p3-stage2-math" / filename
    if destination.is_file():
        _verify_asset(destination, source)
        return destination
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".download", delete=False) as stream:
            temporary = Path(stream.name)
            if cached.is_file():
                _verify_asset(cached, source)
                with cached.open("rb") as incoming:
                    shutil.copyfileobj(incoming, stream)
            else:
                size = 0
                with urlopen(source["url"], timeout=120) as response:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        size += len(chunk)
                        if size > source["bytes"]:
                            raise AdmissionError(
                                f"MATH_SOURCE_MISMATCH: {source['path']} oversized"
                            )
                        stream.write(chunk)
        _verify_asset(temporary, source)
        os.replace(temporary, destination)
        return destination
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _dataset_source_file(source: dict, cache_dir: Path, root: Path) -> Path:
    if source["path"].endswith(".parquet"):
        return _source_file(source, cache_dir, root)
    return _asset_file(source, cache_dir, root)


def _import_verified_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ScorerInfrastructureError(f"Cannot load pinned trusted scorer asset: {path.name}")
    module = importlib.util.module_from_spec(spec)
    # Only immutable, verified upstream scorer source reaches this Python loader.
    # Dataset strings and model completions are never imported or executed.
    spec.loader.exec_module(module)
    return module


def _load_reference_scorer(cache_dir: Path, output_dir: Path | None = None) -> ReferenceScorer:
    if not _on_kaggle():
        raise ScorerInfrastructureError("Pinned trusted MATH scorer may be loaded only on Kaggle")
    if sys.flags.optimize:
        raise ScorerInfrastructureError("Original MATH scorer requires Python assertions enabled")
    root = Path(os.path.realpath(output_dir if output_dir is not None else cache_dir))
    root.mkdir(parents=True, exist_ok=True)
    try:
        paths = {}
        assets = []
        for source in SCORER_ASSETS:
            path = _asset_file(source, cache_dir, root)
            size, sha256 = _verify_asset(path, source)
            paths[source["path"]] = path
            assets.append({
                **source,
                "verified_file_sha256": sha256,
                "file": {
                    "path": path.relative_to(root).as_posix(), "bytes": size, "sha256": sha256,
                },
            })
        equivalence = _import_verified_module(
            paths["modeling/math_equivalence.py"], "_p3_math_original_equivalence"
        )
        boxes = _import_verified_module(
            paths["modeling/dataset/util.py"], "_p3_math_original_boxes"
        )
        for module, name in (
            (equivalence, "_strip_string"), (equivalence, "is_equiv"),
            (boxes, "last_boxed_only_string"),
        ):
            if not callable(getattr(module, name, None)):
                raise ScorerInfrastructureError(f"Pinned scorer is missing {name}")
        return ReferenceScorer(equivalence, boxes, tuple(assets))
    except ScorerInfrastructureError:
        raise
    except Exception as error:
        raise ScorerInfrastructureError("Cannot admit pinned trusted MATH scorer assets") from error


def _get_scorer() -> ReferenceScorer:
    global _SCORER
    if _SCORER is None:
        cache = Path(os.environ.get("P3_STAGE2_MATH_CACHE_DIR", "/kaggle/working/.cache/math"))
        _SCORER = _load_reference_scorer(cache)
    return _SCORER


def _remove_boxed(box: str | None) -> str | None:
    """Strict string-or-None equivalent of the original remove_boxed, not a repair.

    The tiny wrapper is adapted from the function identified by
    REMOVE_BOXED_PROVENANCE. Its MIT permission notice is retained in the verified
    upstream LICENSE runtime asset, alongside the unchanged string scorer.
    """
    left = "\\boxed{"
    if box is None or not box.startswith(left) or not box.endswith("}"):
        return None
    return box[len(left):-1]


def _extract_answer(text: str, scorer: ReferenceScorer) -> tuple[str | None, str]:
    try:
        box = scorer.boxes.last_boxed_only_string(text)
    except Exception as error:
        raise ScorerInfrastructureError("Trusted MATH box extractor failed") from error
    if box is None:
        status = "missing_box" if "\\boxed" not in text and "\\fbox" not in text else "unclosed_box"
        return None, status
    answer = _remove_boxed(box)
    if answer is None:
        return None, "rejected_box_wrapper"
    if not answer:
        return answer, "empty_answer"
    return answer, "ok"


def _validate_gold(solution: str, scorer: ReferenceScorer, task_id: str) -> tuple[str, str]:
    if not isinstance(solution, str):
        raise MalformedGoldError(f"MATH gold is not a string: {task_id}")
    gold, status = _extract_answer(solution, scorer)
    if status != "ok" or gold is None:
        raise MalformedGoldError(f"MATH malformed gold: {task_id}: {status}")
    try:
        normalized = scorer.equivalence._strip_string(gold)
    except (MemoryError, RecursionError) as error:
        raise ScorerInfrastructureError(
            "MATH gold normalization exhausted runtime resources"
        ) from error
    except Exception as error:
        raise MalformedGoldError(
            f"MATH gold normalization failed: {task_id}: {type(error).__name__}"
        ) from error
    if not isinstance(normalized, str):
        raise ScorerInfrastructureError("Trusted MATH normalizer returned non-string gold")
    if not normalized:
        raise MalformedGoldError(f"MATH normalized gold is empty: {task_id}")
    return gold, normalized


def score_completion(record: dict, completion: str) -> dict:
    """Score only the fixed continuation; malformed gold/infrastructure raise.

    ``record`` is the complete scoring.jsonl row. The caller may add a boolean
    ``budget_hit`` from the frozen token-generation metadata. Text alone cannot
    establish whether EOS or the cap terminated a continuation; absent metadata
    is reported as null rather than inferred from answer formatting.
    """
    if not isinstance(record, dict) or not isinstance(completion, str):
        raise ScorerInfrastructureError("MATH scoring requires a record and text continuation")
    payload = record.get("scoring")
    if (
        record.get("benchmark") != BENCHMARK
        or not isinstance(record.get("task_id"), str)
        or not isinstance(payload, dict)
        or "solution" not in payload
    ):
        raise ScorerInfrastructureError("MATH scoring row is missing its identity/gold payload")
    budget_hit = record.get("budget_hit")
    if budget_hit is not None and not isinstance(budget_hit, bool):
        raise ScorerInfrastructureError("MATH budget_hit must be a boolean when supplied")
    scorer = _get_scorer()
    gold, _ = _validate_gold(payload["solution"], scorer, record["task_id"])
    prediction, parse_status = _extract_answer(completion, scorer)
    result = {
        "benchmark": BENCHMARK,
        "task_id": record["task_id"],
        "source_row_sha256": record.get("source_row_sha256"),
        "metric": "MATH_original_test_normalized_string_answer_accuracy",
        "scoring_policy": POLICY,
        "status": "scored",
        "score": 0,
        "passed": False,
        "parse_status": parse_status,
        "prediction": prediction,
        "normalization_fallback": False,
        "normalization_error_type": None,
        "budget_hit": budget_hit,
    }
    if parse_status != "ok":
        return result
    try:
        normalized = scorer.equivalence._strip_string(prediction)
    except (MemoryError, RecursionError) as error:
        raise ScorerInfrastructureError(
            "MATH prediction normalization exhausted runtime resources"
        ) from error
    except Exception as error:
        # Preserve is_equiv's original raw-string equality exception fallback.
        result["normalization_fallback"] = True
        result["normalization_error_type"] = type(error).__name__
    else:
        if not isinstance(normalized, str):
            raise ScorerInfrastructureError(
                "Trusted MATH normalizer returned non-string prediction"
            )
        if not normalized:
            result["parse_status"] = "normalized_empty_answer"
            return result
    try:
        equivalent = scorer.equivalence.is_equiv(prediction, gold)
    except Exception as error:
        raise ScorerInfrastructureError("Trusted MATH equivalence scorer failed") from error
    if not isinstance(equivalent, bool):
        raise ScorerInfrastructureError("Trusted MATH equivalence scorer returned non-boolean")
    result["passed"] = equivalent
    result["score"] = int(equivalent)
    return result


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict:
    record = {}
    for key, value in pairs:
        if key in record:
            raise AdmissionError("MATH auxiliary JSON contains duplicate keys")
        record[key] = value
    return record


def _validate_auxiliary_row(record: dict) -> int:
    """Validate pinned fields, never treat raw unique_id contents as identity."""
    fields = {"problem", "solution", "answer", "subject", "unique_id", "level"}
    if not isinstance(record, dict) or set(record) != fields:
        raise AdmissionError(
            "MATH auxiliary schema must have exactly the pinned six fields",
            metadata={
                "record_type": type(record).__name__,
                "observed_fields": sorted(record) if isinstance(record, dict) else None,
                "expected_fields": sorted(fields),
            },
        )
    if any(not isinstance(record[name], str) for name in fields - {"level"}):
        raise AdmissionError(
            "MATH auxiliary contains a non-string source field",
            metadata={"field_types": {name: type(record[name]).__name__ for name in fields}},
        )
    metadata = {
        "raw_original_id": record["unique_id"],
        "subject": record["subject"],
        "level_type": type(record["level"]).__name__,
    }
    level = record["level"]
    if type(level) is not int or level not in range(1, 6):
        raise AdmissionError(
            "MATH auxiliary level must be an integer from 1 to 5", metadata=metadata
        )
    if record["subject"] not in SUBJECT_DIRECTORIES:
        raise AdmissionError("MATH unknown declared auxiliary subject", metadata=metadata)
    return level


def _evaluation_key(record: dict, config: str, source_ref: str) -> tuple[str, str, str, int]:
    metadata = {
        "source_ref": source_ref,
        "artifact_config": config,
        "declared_type": record.get("type"),
        "level": record.get("level"),
        "source_field_sha256": _field_hashes(record),
    }
    if config not in TEST_COUNTS:
        raise AdmissionError("MATH unknown Eleuther artifact config", metadata=metadata)
    if record.get("type") not in SUBJECT_DIRECTORIES:
        raise AdmissionError("MATH unknown declared Eleuther type", metadata=metadata)
    level = re.fullmatch(r"Level ([1-5])", record.get("level", ""))
    if level is None:
        raise AdmissionError("MATH Eleuther level must be exactly 'Level n'", metadata=metadata)
    return record["problem"], record["solution"], record["type"], int(level[1])


def _require_source_coverage(observed: Counter[str]) -> None:
    if dict(observed) != TEST_COUNTS:
        raise AdmissionError(
            "MATH full test source coverage disagrees with pinned config counts",
            metadata={
                "observed_rows_by_config": dict(observed),
                "expected_rows_by_config": TEST_COUNTS,
                "observed_rows": sum(observed.values()),
                "expected_rows": sum(TEST_COUNTS.values()),
            },
        )


def _write_row(stream: TextIO, row: dict) -> None:
    stream.write(_compact(row) + "\n")


class _AuxiliaryAudit:
    """Keep every physical PRM row and exact many-to-many content relationship."""

    def __init__(self):
        self.entries: list[tuple[tuple[str, str, str, int], dict]] = []
        self.by_key: dict[tuple[str, str, str, int], list[dict]] = {}
        self.raw_id_refs: dict[str, list[str]] = {}
        self.raw_directory_subject_counts: Counter[tuple[str, str, str]] = Counter()
        self.auxiliary_refs: set[str] = set()
        self.test_refs: set[str] = set()
        self.test_refs_by_key: dict[tuple[str, str, str, int], list[str]] = {}
        self.test_by_config: Counter[str] = Counter()

    @property
    def total_rows(self) -> int:
        return len(self.entries)

    def add(self, record: dict, source: dict, ordinal: int, line: bytes) -> None:
        source_ref = _source_ref(source, ordinal)
        if source_ref in self.auxiliary_refs:
            raise AdmissionError(
                "MATH duplicate auxiliary source_ref", metadata={"source_ref": source_ref}
            )
        self.auxiliary_refs.add(source_ref)
        raw_id = record["unique_id"]
        path = re.fullmatch(r"(train|test)/([a-z_]+)/([^/\\\x00-\x1f]+\.json)", raw_id)
        raw_prefix, raw_directory = (path[1], path[2]) if path else ("", "")
        self.raw_directory_subject_counts[(raw_prefix, raw_directory, record["subject"])] += 1
        self.raw_id_refs.setdefault(raw_id, []).append(source_ref)
        entry = {
            "benchmark": BENCHMARK,
            "source_ref": source_ref,
            "source_line": ordinal + 1,
            "source_line_sha256": hashlib.sha256(line).hexdigest(),
            "source_row_sha256": _text_hash(_compact(record)),
            "source_field_sha256": _field_hashes(record),
            "raw_original_id": raw_id,
            "raw_id_prefix": raw_prefix or None,
            "raw_id_directory": raw_directory or None,
            "subject": record["subject"],
            "level": record["level"],
            "source_url": source["url"],
            "artifact_sha256": source["sha256"],
            "raw_problem_sha256": _text_hash(record["problem"]),
            "normalized_problem_sha256": _text_hash(_normalized_problem(record["problem"])),
            "normalized_problem_policy": NORMALIZED_PROBLEM_POLICY,
            "source_use": "auxiliary_provenance_overlap_only_not_identity_membership_or_gold",
        }
        key = (record["problem"], record["solution"], record["subject"], record["level"])
        self.entries.append((key, entry))
        self.by_key.setdefault(key, []).append(entry)

    def observe(self, record: dict, config: str, *, source_ref: str) -> dict:
        key = _evaluation_key(record, config, source_ref)
        if source_ref in self.test_refs:
            raise AdmissionError(
                "MATH duplicate evaluation source_ref", metadata={"source_ref": source_ref}
            )
        self.test_refs.add(source_ref)
        self.test_refs_by_key.setdefault(key, []).append(source_ref)
        self.test_by_config[config] += 1
        candidates = self.by_key.get(key, [])
        return {
            "join_policy": "exact_problem_solution_subject_level_not_ID_or_directory",
            "identity_authority": False,
            "candidate_count": len(candidates),
            "candidates": [
                {
                    name: candidate[name]
                    for name in ("source_ref", "source_line", "raw_original_id", "subject", "level")
                }
                for candidate in candidates
            ],
        }

    def write_audit(self, stream: TextIO) -> None:
        for key, entry in self.entries:
            matches = self.test_refs_by_key.get(key, [])
            _write_row(stream, {
                **entry,
                "raw_id_count": len(self.raw_id_refs[entry["raw_original_id"]]),
                "exact_content_auxiliary_rows": len(self.by_key[key]),
                "exact_matching_test_rows": len(matches),
                "matching_test_source_refs": matches,
            })

    def summary(self) -> dict:
        unmatched_test = single_test = multiple_test = pairs = 0
        for key, refs in self.test_refs_by_key.items():
            candidates = len(self.by_key.get(key, []))
            if candidates == 0:
                unmatched_test += len(refs)
            elif candidates == 1:
                single_test += len(refs)
            else:
                multiple_test += len(refs)
            pairs += candidates * len(refs)
        matched_auxiliary = sum(
            len(rows) for key, rows in self.by_key.items() if key in self.test_refs_by_key
        )
        return {
            "role": "auxiliary_provenance_overlap_not_evaluation_identity_or_gold",
            "rows": self.total_rows,
            "raw_id_unique_values": len(self.raw_id_refs),
            "raw_id_duplicate_groups": [
                {"raw_original_id": raw_id, "rows": len(refs), "source_refs": refs}
                for raw_id, refs in sorted(self.raw_id_refs.items()) if len(refs) > 1
            ],
            "raw_id_directory_subject_counts": [
                {
                    "raw_id_prefix": prefix or None,
                    "raw_id_directory": directory or None,
                    "subject": subject,
                    "rows": count,
                }
                for (prefix, directory, subject), count
                in sorted(self.raw_directory_subject_counts.items())
            ],
            "exact_content": {
                "join_fields": ["problem", "solution", "subject/type", "integer_level"],
                "raw_ID_prefix_or_directory_used": False,
                "source_test_rows": len(self.test_refs),
                "source_test_distinct_content_groups": len(self.test_refs_by_key),
                "source_test_rows_without_auxiliary_match": unmatched_test,
                "source_test_rows_with_single_auxiliary_row": single_test,
                "source_test_rows_with_multiple_auxiliary_rows": multiple_test,
                "auxiliary_rows_with_test_match": matched_auxiliary,
                "auxiliary_rows_without_test_match": self.total_rows - matched_auxiliary,
                "source_test_duplicate_content_groups": sum(
                    len(refs) > 1 for refs in self.test_refs_by_key.values()
                ),
                "auxiliary_duplicate_content_groups": sum(
                    len(rows) > 1 for rows in self.by_key.values()
                ),
                "exact_relation_pairs": pairs,
                "unmatched_or_multiple_matches_block_source_admission": False,
            },
        }


def _load_auxiliary(sources: list[tuple[dict, Path]]) -> _AuxiliaryAudit:
    auxiliary = _AuxiliaryAudit()
    for source, path in sources:
        _verify_asset(path, source)
        source_count = 0
        with path.open("rb") as stream:
            for ordinal, line in enumerate(stream):
                location = {"source_ref": _source_ref(source, ordinal), "source_line": ordinal + 1}
                try:
                    record = json.loads(line, object_pairs_hook=_unique_json_object)
                    _validate_auxiliary_row(record)
                    auxiliary.add(record, source, ordinal, line)
                except AdmissionError as error:
                    error.metadata.update(location)
                    raise
                except (ValueError, UnicodeError) as error:
                    raise AdmissionError(
                        "MATH invalid auxiliary JSON", metadata=location
                    ) from error
                source_count += 1
        if source_count != source["expected_rows"]:
            raise AdmissionError(
                f"MATH wrong auxiliary source count: {source['path']}",
                metadata={"observed_rows": source_count, "expected_rows": source["expected_rows"]},
            )
    if auxiliary.total_rows != ORIGINAL_POOL_ROWS:
        raise AdmissionError(
            "MATH wrong auxiliary pool count",
            metadata={"observed_rows": auxiliary.total_rows, "expected_rows": ORIGINAL_POOL_ROWS},
        )
    return auxiliary


def _iter_parquet_rows(path: Path, source: dict) -> Iterator[dict]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    _verify_asset(path, source)
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != source["expected_rows"]:
        raise AdmissionError(f"MATH wrong Parquet row count: {source['path']}")
    schema = parquet.schema_arrow
    if len(schema.names) != len(source["fields"]) or set(schema.names) != set(source["fields"]):
        raise AdmissionError(f"MATH unexpected Parquet fields: {source['path']}")
    for field in schema:
        if not (pa.types.is_string(field.type) or pa.types.is_large_string(field.type)):
            raise AdmissionError(f"MATH non-string Parquet field: {source['path']}/{field.name}")
    count = 0
    for batch in parquet.iter_batches(batch_size=512, use_threads=False):
        for record in batch.to_pylist():
            if any(not isinstance(record[name], str) for name in source["fields"]):
                raise AdmissionError(f"MATH null/non-string value: {source['path']} row {count}")
            count += 1
            yield record
    if count != source["expected_rows"]:
        raise AdmissionError(f"MATH wrong decoded Parquet row count: {source['path']}")


def _stream_evaluation(
    source_paths: list[tuple[dict, Path]],
    auxiliary: _AuxiliaryAudit,
    scorer: ReferenceScorer,
    generation: TextIO,
    scoring: TextIO,
    audit: TextIO,
) -> tuple[int, dict[str, int]]:
    count = 0
    for source, path in source_paths:
        config = source["config"]
        for ordinal, record in enumerate(_iter_parquet_rows(path, source)):
            source_ref = _source_ref(source, ordinal)
            content_audit = auxiliary.observe(record, config, source_ref=source_ref)
            task_id = f"math/{source_ref}"
            _, normalized_gold = _validate_gold(record["solution"], scorer, task_id)
            row_hash = _text_hash(_compact(record))
            prompt = make_prompt(record)
            identity = {
                "benchmark": BENCHMARK, "task_id": task_id, "source_row_sha256": row_hash,
            }
            _write_row(generation, {
                **identity,
                "source_id": source_ref,
                "source_ref": source_ref,
                "prompt": prompt,
                "prompt_sha256": _text_hash(prompt),
                "generation_policy": GENERATION_POLICY,
            })
            _write_row(scoring, {
                **identity,
                "source_ref": source_ref,
                "scoring": {
                    "solution": record["solution"],
                    "source_ref": source_ref,
                    "subject": record["type"],
                    "level": record["level"],
                    "scoring_policy": POLICY,
                    "gold_source": "unmodified_Eleuther_original_solution_not_PRM_answer",
                    "normalized_gold_sha256": _text_hash(normalized_gold),
                },
            })
            _write_row(audit, {
                **identity,
                "source_id": source_ref,
                "source_ref": source_ref,
                "source_url": source["url"],
                "artifact_sha256": source["sha256"],
                "physical_row_ordinal": ordinal,
                "source_field_sha256": _field_hashes(record),
                "artifact_config": config,
                "config_matches_declared_type": (
                    config == SUBJECT_DIRECTORIES[record["type"]]
                ),
                "subject": record["type"],
                "level": record["level"],
                "problem_text": record["problem"],
                "raw_problem_sha256": _text_hash(record["problem"]),
                "normalized_problem_sha256": _text_hash(_normalized_problem(record["problem"])),
                "normalized_problem_policy": NORMALIZED_PROBLEM_POLICY,
                "prompt_sha256": _text_hash(prompt),
                "auxiliary_content_audit": content_audit,
                "lineage": source_ref,
                "lineage_basis": SOURCE_LINEAGE["basis"],
                "original_filename_verified": False,
                "source_admission_policy": SOURCE_ADMISSION_POLICY,
                "overlap_and_independent_group_decision": "pending_shared_audit",
            })
            count += 1
    _require_source_coverage(auxiliary.test_by_config)
    return count, dict(auxiliary.test_by_config)


def _publish_outputs(stage: Path, root: Path) -> None:
    """Create outputs exclusively, rolling back only this attempt if publishing fails."""
    created: list[Path] = []
    try:
        # The manifest is the final commit marker, never a partially copied file.
        for name in _OUTPUT_NAMES[:-1]:
            destination = _inside(root, root / name)
            with destination.open("xb") as outgoing:
                created.append(destination)
                with (stage / name).open("rb") as incoming:
                    shutil.copyfileobj(incoming, outgoing)
        os.link(stage / "manifest.json", _inside(root, root / "manifest.json"))
    except BaseException:
        for path in reversed(created):
            path.unlink()
        raise


def _write_failure_report(root: Path, error: Exception) -> None:
    report = {
        "schema": "p3-stage2-math-admission-report-v2",
        "benchmark": BENCHMARK,
        "status": "ADMISSION_FAILED",
        "policy": POLICY,
        "source_admission_policy": SOURCE_ADMISSION_POLICY,
        "source_lineage": SOURCE_LINEAGE,
        "error_type": type(error).__name__,
        "error": str(error),
        "error_metadata": getattr(error, "metadata", {}),
        "successful_denominator": None,
        "no_subset_admitted": True,
        "sources": list(PARQUET_SOURCES) + list(CROSSWALK_SOURCES),
        "trusted_scorer_sources": list(SCORER_ASSETS),
    }
    destination = _inside(root, root / "admission-report.json")
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            _write_row(stream, report)
    except FileExistsError:
        # Preserve previous reports and every historical output, even on a retry.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=root,
            prefix="admission-report-", suffix=".json", delete=False,
        ) as stream:
            _write_row(stream, report)


def diagnose_metadata(*, cache_dir: Path, output_dir: Path) -> dict:
    """Check full pinned source coverage and audit auxiliary content relationships.

    Only metadata is retained. Downloaded source payloads stay in a temporary
    directory. No scorer imports, gold validation, prompts or evaluation manifest
    occur. Missing or multiple auxiliary matches are audit facts, not subset gates.
    """
    if not _on_kaggle():
        raise AdmissionError("MATH metadata diagnosis may run only on Kaggle")
    root = Path(os.path.realpath(output_dir))
    root.mkdir(parents=True, exist_ok=True)
    for name in (*_OUTPUT_NAMES, *_METADATA_OUTPUT_NAMES):
        if _inside(root, root / name).exists():
            raise AdmissionError(
                f"MATH output evidence already exists: {name}; use a fresh output"
            )
    report = {
        "schema": "p3-stage2-math-metadata-diagnostic-v2",
        "benchmark": BENCHMARK,
        "status": "METADATA_DIAGNOSTIC_FAILED",
        "policy": POLICY,
        "source_admission_policy": SOURCE_ADMISSION_POLICY,
        "source_lineage": SOURCE_LINEAGE,
        "source_coverage_complete": False,
        "content_audit_complete": False,
        "gold_validation": "NOT_RUN",
        "trusted_scorer_loaded": False,
        "successful_denominator": None,
        "no_subset_admitted": True,
        "sources": list(PARQUET_SOURCES) + list(CROSSWALK_SOURCES),
    }
    auxiliary = None
    try:
        with tempfile.TemporaryDirectory(dir=root, prefix=".math-metadata-sources-") as temporary:
            source_root = _inside(root, Path(temporary))
            source_paths = [
                (source, _dataset_source_file(source, Path(cache_dir), source_root))
                for source in PARQUET_SOURCES
            ]
            auxiliary_paths = [
                (source, _dataset_source_file(source, Path(cache_dir), source_root))
                for source in CROSSWALK_SOURCES
            ]
            auxiliary = _load_auxiliary(auxiliary_paths)
            with (root / "metadata-joins.jsonl").open(
                "x", encoding="utf-8", newline="\n"
            ) as joined:
                for source, path in source_paths:
                    config = source["config"]
                    for ordinal, record in enumerate(_iter_parquet_rows(path, source)):
                        source_ref = _source_ref(source, ordinal)
                        content_audit = auxiliary.observe(record, config, source_ref=source_ref)
                        _write_row(joined, {
                            "source_ref": source_ref,
                            "artifact_config": config,
                            "physical_row_ordinal": ordinal,
                            "subject": record["type"],
                            "config_matches_declared_type": (
                                config == SUBJECT_DIRECTORIES[record["type"]]
                            ),
                            "level": record["level"],
                            "auxiliary_content_audit": content_audit,
                            "source_field_sha256": _field_hashes(record),
                        })
            _require_source_coverage(auxiliary.test_by_config)
            report["source_coverage_complete"] = True
            report["content_audit_complete"] = True
            report["status"] = "METADATA_COMPLETE_NOT_DATA_ADMISSION"
    except Exception as error:
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        report["error_metadata"] = getattr(error, "metadata", {})
        raise
    finally:
        report["observed_test_rows"] = 0 if auxiliary is None else len(auxiliary.test_refs)
        report["observed_test_by_config"] = (
            {} if auxiliary is None else dict(auxiliary.test_by_config)
        )
        report["auxiliary_provenance_audit"] = (
            None if auxiliary is None else auxiliary.summary()
        )
        if auxiliary is not None:
            with (root / "metadata-crosswalk.jsonl").open(
                "x", encoding="utf-8", newline="\n"
            ) as pool:
                auxiliary.write_audit(pool)
        report["metadata_files_complete"] = report["source_coverage_complete"]
        report["files"] = {}
        for name in _METADATA_OUTPUT_NAMES[:-1]:
            path = _inside(root, root / name)
            if path.is_file():
                size, digest = _file_hash(path)
                report["files"][name] = {"path": name, "bytes": size, "sha256": digest}
        with (root / "metadata-report.json").open("x", encoding="utf-8", newline="\n") as stream:
            _write_row(stream, report)
    return report


def build_evaluation(*, cache_dir: Path, output_dir: Path) -> dict:
    """Stream every pinned config/test source row, or report failure and raise.

    Outputs are immutable evidence. Use an output directory without existing final
    output files. Verified source downloads and earlier failure reports can remain.
    Shared project overlap, independent groups, model-context fit, inference-path
    qualification, and resolved EOS/tokenizer settings are the integration owner's
    later admission gates, not claims made by this data adapter.
    """
    global _SCORER
    root = Path(os.path.realpath(output_dir))
    root.mkdir(parents=True, exist_ok=True)
    try:
        for name in (*_OUTPUT_NAMES, *_METADATA_OUTPUT_NAMES):
            if _inside(root, root / name).exists():
                raise AdmissionError(
                    f"MATH output evidence already exists: {name}; use a fresh output"
                )
        scorer = _load_reference_scorer(Path(cache_dir), root)
        _SCORER = scorer
        source_paths = [
            (source, _dataset_source_file(source, Path(cache_dir), root))
            for source in PARQUET_SOURCES
        ]
        auxiliary_paths = [
            (source, _dataset_source_file(source, Path(cache_dir), root))
            for source in CROSSWALK_SOURCES
        ]
        with tempfile.TemporaryDirectory(dir=root, prefix=".math-admission-") as stage_name:
            stage = _inside(root, Path(stage_name))
            with (
                (stage / "crosswalk_audit.jsonl").open("w", encoding="utf-8", newline="\n") as pool,
                (stage / "generation.jsonl").open(
                    "w", encoding="utf-8", newline="\n"
                ) as generation,
                (stage / "scoring.jsonl").open("w", encoding="utf-8", newline="\n") as scoring,
                (stage / "audit.jsonl").open("w", encoding="utf-8", newline="\n") as audit,
            ):
                auxiliary = _load_auxiliary(auxiliary_paths)
                count, configs = _stream_evaluation(
                    source_paths, auxiliary, scorer, generation, scoring, audit
                )
                auxiliary.write_audit(pool)
            files = {}
            for name in _OUTPUT_NAMES[:-1]:
                size, sha256 = _file_hash(stage / name)
                files[name.removesuffix(".jsonl")] = {
                    "path": name, "bytes": size, "sha256": sha256,
                    "rows": ORIGINAL_POOL_ROWS if name == "crosswalk_audit.jsonl" else count,
                }
            manifest = {
                "schema": "p3-stage2-math-evaluation-v2",
                "benchmark": BENCHMARK,
                "policy": POLICY,
                "source_admission_policy": SOURCE_ADMISSION_POLICY,
                "source_lineage": SOURCE_LINEAGE,
                "status": "DATA_ADMITTED_PENDING_SHARED_DEDUP_AND_INFERENCE",
                "files": files,
                "source_counts": {
                    "test_rows": count,
                    "test_by_config": configs,
                    "auxiliary_pool_rows": auxiliary.total_rows,
                },
                "source_identity_contract": {
                    "membership": "all_pinned_Eleuther_config_test_artifact_rows",
                    "identity": "pinned_source_ref_artifact_config_split_physical_ordinal",
                    "task_id": "math_slash_source_ref",
                    "original_filename_verified": False,
                    "PRM_original_ID_prefix_directory_or_uniqueness_authoritative": False,
                    "auxiliary_content_join": "exact_audit_only_never_membership_or_gold",
                },
                "auxiliary_provenance_audit": auxiliary.summary(),
                "sources": list(PARQUET_SOURCES) + list(CROSSWALK_SOURCES),
                "trusted_scorer": {
                    "assets": list(scorer.assets),
                    "remove_boxed_provenance": REMOVE_BOXED_PROVENANCE,
                    "execution": "verified_trusted_string_source_only_on_Kaggle",
                    "assertions_required": True,
                    "generated_or_reference_solution_execution": False,
                    "symbolic_evaluation": False,
                },
                "generation_policy": GENERATION_POLICY,
                "prompt": {"prefix": PROMPT_PREFIX, "suffix": PROMPT_SUFFIX},
                "order": "research_subject_table_then_Parquet_physical_row_order",
                "source_row_hash_policy": "SHA256_UTF8_sorted_compact_JSON_original_fields",
                "source_field_hash_policy": "SHA256_exact_UTF8_strings_or_compact_JSON_integer",
                "crosswalk_source_line_hash_policy": "SHA256_exact_line_bytes_including_terminator",
                "normalized_problem_policy": NORMALIZED_PROBLEM_POLICY,
                "independent_problem_groups": None,
                "shared_overlap_status": "PENDING_PARENT_REVIEW_NO_ROWS_SILENTLY_DROPPED",
                "runtime_admission": [
                    "qualify_pinned_scorer_on_synthetic_fixtures",
                    "complete_project_wide_overlap_and_original_problem_group_audit",
                    "full_prompt_plus_1024_new_tokens_fits_context_without_truncation",
                    "qualified_identical_inference_path_and_resolved_tokenizer_EOS_policy",
                ],
                "maximum_nominal_test_rows": count,
                "statistical_pass_claimed": False,
            }
            manifest["manifest_sha256"] = _text_hash(_compact(manifest))
            with (stage / "manifest.json").open("w", encoding="utf-8", newline="\n") as stream:
                _write_row(stream, manifest)
            _publish_outputs(stage, root)
            return manifest
    except Exception as error:
        _write_failure_report(root, error)
        raise


def main(argv: list[str] | None = None) -> int:
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metadata-only", action="store_true",
        help="Audit pinned source coverage and auxiliary content without scorer or admission",
    )
    arguments = parser.parse_args(argv)
    console = Console()
    try:
        operation = diagnose_metadata if arguments.metadata_only else build_evaluation
        report = operation(cache_dir=arguments.cache_dir, output_dir=arguments.output)
    except Exception as error:
        operation_name = "metadata diagnostic" if arguments.metadata_only else "admission"
        console.print(
            f"[red]MATH {operation_name} failed: {type(error).__name__}[/red]", highlight=False
        )
        console.print(str(error), markup=False, highlight=False)
        return 1
    if arguments.metadata_only:
        console.print(
            f"MATH metadata: {report['observed_test_rows']} source rows; {report['status']}",
            highlight=False,
        )
    else:
        console.print(
            f"MATH: {report['source_counts']['test_rows']} pinned test source rows; "
            f"{report['status']}",
            highlight=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
