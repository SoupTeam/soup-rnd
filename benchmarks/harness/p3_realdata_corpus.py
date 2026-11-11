"""Pinned GSM8K/HumanEval text packs for the benchmark-only trained-MoE study.

Only ``build_corpus`` performs I/O. It never fetches a model, executes source code,
uses chat templates, or imports torch. Downloads and generated files stay below
``output_dir``; ``cache_dir/p3-realdata/<sha256>.parquet`` is an optional read-only
source cache. The caller owns the frozen tokenizer and its provenance specification.
Sequence lists are one-dimensional; the caller adds the singleton batch dimension.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path
from typing import Any
from urllib.request import urlopen

POLICY = "p3-realdata-v1"
SELECTION_SEED = "p3-realdata-v1/20261106"
LENGTHS = (256, 1024, 4096)
PARTITIONS = ("calibration", "discovery", "holdout")
DOMAINS = ("gsm8k", "humaneval")

SOURCES = (
    {
        "domain": "gsm8k", "repository": "openai/gsm8k",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "config": "main", "split": "train",
        "path": "main/train-00000-of-00001.parquet",
        "bytes": 2306545, "expected_rows": 7473,
        "sha256": "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d",
        "fields": ["question", "answer"],
    },
    {
        "domain": "gsm8k", "repository": "openai/gsm8k",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "config": "main", "split": "test",
        "path": "main/test-00000-of-00001.parquet",
        "bytes": 419088, "expected_rows": 1319,
        "sha256": "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59",
        "fields": ["question", "answer"],
    },
    {
        "domain": "humaneval", "repository": "openai/openai_humaneval",
        "revision": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
        "config": "openai_humaneval", "split": "test",
        "path": "openai_humaneval/test-00000-of-00001.parquet",
        "bytes": 83920, "expected_rows": 164,
        "sha256": "2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6",
        "fields": ["task_id", "prompt", "canonical_solution", "test", "entry_point"],
    },
)

# These notices describe redistributed dataset content, not this Python module.
_MIT_PERMISSION = (
    'Permission is hereby granted, free of charge, to any person obtaining a copy\n'
    'of this software and associated documentation files (the "Software"), to deal\n'
    'in the Software without restriction, including without limitation the rights\n'
    'to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n'
    'copies of the Software, and to permit persons to whom the Software is\n'
    'furnished to do so, subject to the following conditions:\n\n'
    'The above copyright notice and this permission notice shall be included in all\n'
    'copies or substantial portions of the Software.\n\n'
    'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n'
    'IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n'
    'FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n'
    'AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n'
    'LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n'
    'OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n'
    'SOFTWARE.'
)


def _compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sequence_hash(values: list[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(values)}q", *values)).hexdigest()


def _inside(root: Path, path: Path) -> Path:
    root_name = os.path.realpath(root)
    path_name = os.path.realpath(path)
    if os.path.commonpath((root_name, path_name)) != root_name:
        raise ValueError("Corpus output path escapes output_dir")
    return Path(path_name)


def _file_hash(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _verify_file(path: Path, source: dict[str, Any]) -> None:
    size, digest = _file_hash(path)
    if size != source["bytes"] or digest != source["sha256"]:
        raise ValueError(
            f"CORPUS_SOURCE_MISMATCH: {source['repository']}/{source['path']}: "
            f"expected {source['bytes']} bytes sha256={source['sha256']}; "
            f"observed {size} bytes sha256={digest}"
        )


def _source_file(source: dict[str, Any], cache_dir: Path, root: Path) -> Path:
    """Use verified cached bytes, or download only the explicit immutable URL."""
    cache_file = cache_dir / "p3-realdata" / f"{source['sha256']}.parquet"
    if cache_file.is_file():
        _verify_file(cache_file, source)
        return cache_file
    directory = _inside(root, root / "sources")
    directory.mkdir(parents=True, exist_ok=True)
    destination = _inside(root, directory / f"{source['sha256']}.parquet")
    if destination.is_file():
        _verify_file(destination, source)
        return destination
    url = source["url"]
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".download", delete=False) as stream:
            temporary = Path(stream.name)
            size = 0
            with urlopen(url, timeout=120) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    size += len(chunk)
                    if size > source["bytes"]:
                        raise ValueError(f"CORPUS_SOURCE_MISMATCH: oversized download: {url}")
                    stream.write(chunk)
        _verify_file(temporary, source)
        os.replace(temporary, destination)
        return destination
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _read_rows(path: Path, source: dict[str, Any]) -> list[dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != source["expected_rows"]:
        raise ValueError(f"CORPUS_SCHEMA_MISMATCH: unexpected row count in {source['path']}")
    schema = parquet.schema_arrow
    if len(schema.names) != len(source["fields"]) or set(schema.names) != set(source["fields"]):
        raise ValueError(f"CORPUS_SCHEMA_MISMATCH: unexpected fields in {source['path']}")
    for field in schema:
        if not (pa.types.is_string(field.type) or pa.types.is_large_string(field.type)):
            raise ValueError(f"CORPUS_SCHEMA_MISMATCH: {field.name} must be a string")
    rows = []
    task_ids: set[str] = set()
    for batch in parquet.iter_batches(batch_size=1024, use_threads=False):
        for values in batch.to_pylist():
            if any(not isinstance(values[name], str) for name in source["fields"]):
                raise ValueError(f"CORPUS_SCHEMA_MISMATCH: null/non-string in {source['path']}")
            ordinal = len(rows)
            key = (f"{source['repository']}@{source['revision']}/{source['config']}/"
                   f"{source['split']}/{source['path']}#row={ordinal}")
            if source["domain"] == "gsm8k":
                text = "Question:\n" + values["question"] + "\nAnswer:\n" + values["answer"]
                task_id = None
            else:
                text = values["prompt"] + values["canonical_solution"]
                task_id = values["task_id"]
                if not task_id or task_id in task_ids:
                    raise ValueError("CORPUS_SCHEMA_MISMATCH: empty or duplicate HumanEval task_id")
                task_ids.add(task_id)
            rows.append({
                "row_key": key, "domain": source["domain"], "split": source["split"],
                "physical_row_ordinal": ordinal, "task_id_or_null": task_id,
                "source_row_sha256": _text_hash(_compact(values)),
                "source_field_sha256": {
                    name: _text_hash(value) for name, value in sorted(values.items())
                },
                "text_sha256": _text_hash(text),
                "rank_sha256": _text_hash(SELECTION_SEED + "\n" + key), "text": text,
            })
            if source["domain"] == "humaneval":
                rows[-1]["entry_point"] = values["entry_point"]
    if len(rows) != source["expected_rows"]:
        raise ValueError(f"CORPUS_SCHEMA_MISMATCH: decoded row count in {source['path']}")
    return rows


def _select_rows(rows: list[dict[str, Any]]) -> tuple[dict, list, list, list, list]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["domain"], row["text"]), []).append(row)
    eligible: dict[tuple[str, str], list[dict[str, Any]]] = {}
    excluded = []
    for group in groups.values():
        test_rows = [row for row in group
                     if row["domain"] == "gsm8k" and row["split"] == "test"]
        retained = min(test_rows or group, key=lambda row: row["row_key"])
        eligible.setdefault((retained["domain"], retained["split"]), []).append(retained)
        for row in sorted(group, key=lambda item: item["row_key"]):
            if row is retained:
                continue
            excluded.append({
                "row_key": row["row_key"], "retained_row_key": retained["row_key"],
                "text_sha256": row["text_sha256"],
                "reason": ("gsm8k_test_reserved" if test_rows and row["split"] == "train"
                           else "exact_rendered_text_duplicate"),
            })
    for pool in eligible.values():
        pool.sort(key=lambda row: (row["rank_sha256"], row["row_key"]))
    requests = (
        ("calibration", "gsm8k", "train", 0, 64),
        ("discovery", "gsm8k", "train", 64, 64),
        ("holdout", "gsm8k", "test", 0, 128),
        ("calibration", "humaneval", "test", 0, 48),
        ("discovery", "humaneval", "test", 48, 48),
        ("holdout", "humaneval", "test", 96, 68),
    )
    selected = {}
    counts = []
    failures = []
    for partition, domain, split, start, count in requests:
        pool = eligible.get((domain, split), [])
        picked = pool[start:start + count]
        selected[(partition, domain)] = picked
        counts.append({
            "partition": partition, "domain": domain, "split": split,
            "eligible_split_rows": len(pool), "requested_rows": count,
            "selected_rows": len(picked), "row_keys": [row["row_key"] for row in picked],
        })
        if len(picked) != count:
            failures.append({
                "partition": partition, "domain": domain, "required_rows": count,
                "available_rows": len(picked), "code": "CORPUS_INSUFFICIENT_ROWS",
            })
    chosen = {row["row_key"] for pool in selected.values() for row in pool}
    reserve = [row["row_key"] for key in sorted(eligible) for row in eligible[key]
               if row["row_key"] not in chosen]
    return selected, sorted(excluded, key=lambda row: row["row_key"]), counts, reserve, failures


def _tokenizer_settings(tokenizer: Any, spec: dict) -> tuple[dict, int, int | None, int]:
    if not isinstance(spec, dict):
        raise ValueError("tokenizer_spec must be a JSON dictionary")
    frozen = json.loads(_compact(spec))
    required = {
        "repo_id", "revision", "vocabulary_size", "eos_id", "bos_id",
        "prepend_bos_per_pack", "tokenizer_class", "files_sha256", "library_versions",
    }
    missing = sorted(required - frozen.keys())
    if missing:
        raise ValueError(f"CORPUS_TOKENIZER_INCOMPATIBLE: missing provenance fields: {missing}")
    if (frozen["repo_id"] != "ibm-granite/granite-3.1-1b-a400m-base" or
            frozen["revision"] != "408b6e90baab8cf24f4aa9f8e19703ffa0a53b29"):
        raise ValueError("CORPUS_TOKENIZER_INCOMPATIBLE: expected pinned Granite tokenizer")
    if frozen["tokenizer_class"] != type(tokenizer).__name__:
        raise ValueError("CORPUS_TOKENIZER_INCOMPATIBLE: tokenizer class differs from spec")
    files = frozen["files_sha256"]
    if not isinstance(files, dict) or not files:
        raise ValueError("CORPUS_TOKENIZER_INCOMPATIBLE: tokenizer file hashes are required")
    for name, digest in files.items():
        if (not isinstance(name, str) or not name or name.startswith(("/", "\\")) or
                ":" in name or ".." in name.replace("\\", "/").split("/") or
                not isinstance(digest, str) or len(digest) != 64 or
                any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError("CORPUS_TOKENIZER_INCOMPATIBLE: invalid relative file/hash entry")
    versions = frozen["library_versions"]
    if (not isinstance(versions, dict) or not versions or
            any(not isinstance(value, str) or not value for value in versions.values())):
        raise ValueError("CORPUS_TOKENIZER_INCOMPATIBLE: library versions are required")
    eos_id = frozen["eos_id"]
    prepend_bos = frozen["prepend_bos_per_pack"]
    if type(eos_id) is not int or type(prepend_bos) is not bool:
        raise ValueError("tokenizer_spec requires explicit eos_id and prepend_bos_per_pack")
    vocabulary_size = len(tokenizer)
    if (type(frozen["vocabulary_size"]) is not int or
            frozen["vocabulary_size"] != vocabulary_size):
        raise ValueError("CORPUS_TOKENIZER_INCOMPATIBLE: vocabulary_size differs from tokenizer")
    native_eos = getattr(tokenizer, "eos_token_id", None)
    native_eos_ids = native_eos if isinstance(native_eos, (tuple, list)) else [native_eos]
    if eos_id not in native_eos_ids or not 0 <= eos_id < vocabulary_size:
        raise ValueError("CORPUS_TOKENIZER_INCOMPATIBLE: eos_id is not a native tokenizer EOS")
    bos_id = frozen["bos_id"]
    if bos_id != getattr(tokenizer, "bos_token_id", None) or (
        bos_id is not None and (type(bos_id) is not int or not 0 <= bos_id < vocabulary_size)
    ):
        raise ValueError("CORPUS_TOKENIZER_INCOMPATIBLE: bos_id is not a native tokenizer BOS")
    if prepend_bos and bos_id is None:
        raise ValueError("CORPUS_TOKENIZER_INCOMPATIBLE: requested BOS but tokenizer has none")
    return frozen, eos_id, bos_id if prepend_bos else None, vocabulary_size


def _encode_rows(
    tokenizer: Any, rows: list[dict[str, Any]], eos_id: int, vocabulary_size: int,
) -> None:
    for row in rows:
        tokens = tokenizer.encode(row["text"], add_special_tokens=False, truncation=False,
                                  padding=False)
        if not isinstance(tokens, list) or any(
            type(token) is not int or not 0 <= token < vocabulary_size for token in tokens
        ):
            raise ValueError("CORPUS_TOKENIZER_INCOMPATIBLE: encode must return native integer IDs")
        row["content_token_count"] = len(tokens)
        row["tokens"] = tokens + [eos_id]
        row["document_token_count"] = len(row["tokens"])
        row["document_tokens_sha256"] = _sequence_hash(row["tokens"])


def _pack_pool(
    rows: list[dict[str, Any]], partition: str, domain: str, blocks: int, bos_id: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    cursor = 0
    packed = []
    tails = []
    failures = []
    for block_index in range(blocks):
        tokens = [] if bos_id is None else [bos_id]
        spans = []
        while len(tokens) < LENGTHS[-1] and cursor < len(rows):
            row = rows[cursor]
            cursor += 1
            start = len(tokens)
            taken = min(len(row["tokens"]), LENGTHS[-1] - start)
            tokens.extend(row["tokens"][:taken])
            eos_position = start + row["content_token_count"]
            spans.append({
                "row_key": row["row_key"], "task_id_or_null": row["task_id_or_null"],
                "source_row_sha256": row["source_row_sha256"],
                "text_sha256": row["text_sha256"],
                "document_token_start": 0, "document_token_stop": taken,
                "document_token_count": row["document_token_count"],
                "document_content_token_count": row["content_token_count"],
                "packed_start": start, "packed_stop": start + taken,
                "eos_positions": [eos_position] if eos_position < start + taken else [],
            })
            if taken < len(row["tokens"]):
                tails.append({
                    "base_block_index": block_index, "row_key": row["row_key"],
                    "document_token_start": taken, "document_token_stop": len(row["tokens"]),
                    "discarded_token_count": len(row["tokens"]) - taken,
                })
        if len(tokens) != LENGTHS[-1]:
            failures.append({
                "code": "CORPUS_INSUFFICIENT_TOKENS", "partition": partition,
                "domain": domain, "base_block_index": block_index,
                "required_positions": LENGTHS[-1], "available_positions": len(tokens),
                "available_real_document_tokens": len(tokens) - int(bos_id is not None),
                "consumed_rows": [span["row_key"] for span in spans],
            })
            break
        packed.append({"input_ids": tokens, "source_row_spans": spans})
    unused = rows[cursor:]
    report = {
        "partition": partition, "domain": domain, "selected_document_count": len(rows),
        "selected_document_token_count": sum(row["document_token_count"] for row in rows),
        "completed_base_blocks": len(packed), "discarded_tails": tails,
        "unused_selected_rows": [row["row_key"] for row in unused],
        "unused_selected_token_count": sum(row["document_token_count"] for row in unused),
    }
    return packed, report, failures


def _case(
    packed: dict[str, Any], partition: str, domain: str, block_index: int, length: int,
    bos_id: int | None,
) -> dict[str, Any]:
    ids = packed["input_ids"][:length]
    labels = ids.copy()
    labels[0] = -100
    spans = []
    for original in packed["source_row_spans"]:
        if original["packed_start"] >= length:
            break
        span = dict(original)
        span["packed_stop"] = min(length, span["packed_stop"])
        count = span["packed_stop"] - span["packed_start"]
        span["document_token_stop"] = count
        span["eos_positions"] = [pos for pos in span["eos_positions"] if pos < length]
        span["content_token_count"] = min(count, span["document_content_token_count"])
        span["token_count"] = count
        span["token_share"] = count / length
        if span["content_token_count"]:
            labels[span["packed_start"]] = -100
        spans.append(span)
    case_id = f"{POLICY}/{partition}/{domain}/b{block_index}/L{length}"
    attention = [1] * length
    positions = list(range(length))
    arrays = {"input_ids": ids, "labels": labels,
              "attention_mask": attention, "position_ids": positions}
    hashes = {"input_sha256": _sequence_hash(ids), "labels_sha256": _sequence_hash(labels),
              "attention_sha256": _sequence_hash(attention),
              "positions_sha256": _sequence_hash(positions)}
    return {
        "case_id": case_id, "partition": partition, "domain": domain,
        "base_block_index": block_index, "sequence_length": length, **arrays, **hashes,
        "valid_target_count": sum(label != -100 for label in labels[1:]),
        "sequence_shape": [1, length], "sequence_dtype": "int64",
        "sequence_byte_order": "little", "source_row_spans": spans,
        "distinct_contributing_documents": len(spans),
        "bos_positions": [] if bos_id is None else [0],
        "paired_rng_seed": int.from_bytes(
            hashlib.sha256((SELECTION_SEED + "\n" + case_id).encode("utf-8")).digest()[:4],
            "big",
        ),
    }


def _write_corpus(root: Path, manifest: dict[str, Any], cases: list[dict[str, Any]]) -> dict:
    manifest["manifest_sha256"] = _text_hash(_compact(manifest))
    result = {"manifest": manifest, "cases": cases}
    destination = _inside(root, root / "corpus.json")
    destination.write_text(_compact(result), encoding="utf-8", newline="")
    return result


def build_corpus(
    tokenizer: Any, *, tokenizer_spec: dict, cache_dir: Path, output_dir: Path, blocks: int = 1,
) -> dict:
    """Write and return ``{'manifest': ..., 'cases': [...]}`` without tensor creation.

    All three pinned Parquet objects must match both byte size and SHA-256 before
    parsing. Rows are deduplicated before seeded ranking and fixed partitioning.
    Each independent block consumes whole, previously unused rows and discards the
    cut document's remainder. Prefix cases share that block intentionally.

    ``tokenizer_spec`` is JSON data with explicit ``eos_id`` and
    ``prepend_bos_per_pack`` plus ``bos_id`` when enabled. The caller supplies the
    tokenizer repository/revision, library/configuration and file-hash provenance;
    this function preserves it without fetching or changing tokenizer artifacts.
    ``cache_dir`` is read-only. Missing source bytes download below ``output_dir``.
    A capacity failure writes an explicit failed manifest with no cases and raises
    ValueError rather than changing row counts, repeating content, or padding.
    """
    if type(blocks) is not int or blocks < 1:
        raise ValueError("blocks must be a positive integer declared before corpus selection")
    spec, eos_id, bos_id, vocabulary_size = _tokenizer_settings(tokenizer, tokenizer_spec)
    root = Path(os.path.realpath(output_dir))
    root.mkdir(parents=True, exist_ok=True)
    source_records = []
    rows = []
    for pinned in SOURCES:
        source = dict(pinned)
        source["url"] = (f"https://huggingface.co/datasets/{source['repository']}/resolve/"
                         f"{source['revision']}/{source['path']}")
        path = _source_file(source, Path(cache_dir), root)
        loaded = _read_rows(path, source)
        source["verified_sha256"] = source["sha256"]
        source["observed_rows"] = len(loaded)
        source_records.append(source)
        rows.extend(loaded)
    selected, exclusions, counts, reserve, failures = _select_rows(rows)
    manifest = {
        "policy": POLICY, "selection_seed": SELECTION_SEED, "requested_blocks": blocks,
        "base_sequence_length": LENGTHS[-1], "prefix_lengths": list(LENGTHS),
        "sources": source_records, "tokenizer_spec": spec,
        "tokenizer_observed_vocabulary_size": vocabulary_size,
        "duplicate_exclusions": exclusions, "partitions": counts, "unused_reserve_rows": reserve,
        "selection_policy": "deduplicate exact rendered text; GSM8K test wins; seeded SHA256 rank",
        "rendering": {"gsm8k": "Question:\\n{question}\\nAnswer:\\n{answer}",
                      "humaneval": "prompt + canonical_solution; no separator; test excluded",
                      "normalization": "none; preserve Unicode and source whitespace"},
        "tokenization": {"add_special_tokens": False, "truncation": False, "padding": False,
                         "chat_template": False, "append_eos_per_document": eos_id,
                         "prepend_bos_per_pack": bos_id is not None, "bos_id": bos_id},
        "packing_policy": "rank order; crop 4096; discard final-document tail; next whole row",
        "label_policy": "full causal labels; mask position zero and each document first content "
                        "token; EOS supervised; native model shifts labels",
        "position_policy": "0..L-1; full native causal attention; no segment resets or padding",
        "sequence_hash_policy": (
            "SHA256 of contiguous little-endian signed int64 bytes, shape [1,L]"
        ),
        "json_hash_policy": (
            "SHA256 of sorted-key compact UTF-8 JSON, ensure_ascii=False, no newline"
        ),
        "manifest_hash_exclusion": "manifest_sha256",
        "paired_seed_policy": (
            "first four SHA256 bytes of selection_seed + LF + case_id, big-endian"
        ),
        "evaluation_scope": "teacher-forced storage/gradient study; no accuracy/pass@k or "
                            "pretraining-contamination claim; HumanEval text is never executed",
        "licenses": {
            "gsm8k": {"license": "MIT", "upstream":
                      "https://github.com/openai/grade-school-math/blob/master/LICENSE",
                      "notice": "MIT License\n\nCopyright (c) 2021 OpenAI\n\n" + _MIT_PERMISSION},
            "humaneval": {"license": "MIT", "upstream":
                         "https://github.com/openai/human-eval/blob/master/LICENSE",
                         "notice": "The MIT License\n\nCopyright (c) OpenAI (https://openai.com)\n\n"
                                   + _MIT_PERMISSION},
        },
    }
    cases = []
    pools = []
    if not failures:
        for partition in PARTITIONS:
            for domain in DOMAINS:
                pool = selected[(partition, domain)]
                _encode_rows(tokenizer, pool, eos_id, vocabulary_size)
                packed, report, pool_failures = _pack_pool(pool, partition, domain, blocks, bos_id)
                pools.append(report)
                failures.extend(pool_failures)
                for block_index, base in enumerate(packed):
                    for length in LENGTHS:
                        cases.append(_case(base, partition, domain, block_index, length, bos_id))
    manifest["rows"] = [
        {key: value for key, value in row.items() if key not in ("text", "tokens")}
        for row in sorted(rows, key=lambda row: row["row_key"])
    ]
    manifest["packing_pools"] = pools
    manifest["capacity_failures"] = failures
    manifest["status"] = failures[0]["code"] if failures else "ready"
    if failures:
        _write_corpus(root, manifest, [])
        raise ValueError(f"{manifest['status']}: " + _compact(failures))
    manifest["case_order"] = [case["case_id"] for case in cases]
    # One microbatch per window. Never manufacture a larger accumulation factor by reuse.
    manifest["accumulation_windows"] = [[case["case_id"]] for case in cases]
    manifest["case_checksums"] = [
        {key: value for key, value in case.items()
         if key.endswith("sha256") or key in ("case_id", "sequence_shape", "sequence_dtype",
                                              "sequence_byte_order", "valid_target_count")}
        for case in cases
    ]
    return _write_corpus(root, manifest, cases)
