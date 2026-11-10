"""Build and verify the DeepSeek-V3 paired response-only training pack.

CLI::

    python -m benchmarks.harness.p3_deepseek_quality_data \
        --sources training-sources.jsonl --sources-sha256 HEX \
        --tokenizer-dir snapshots/e815299b0bcbac849fa540c768ef21845365c9eb \
        --output NEW_DIR

Only tokenizer assets are loaded. Source bytes, physical JSONL row hashes and every
published file are bound to a manifest written last. Production admission accepts
only the frozen 7801 source rows and the pinned DeepSeek tokenizer. Each domain
cycles the frozen source order to 131072 tokens, recording every occurrence and
the final cutoff exactly as Granite does. The private Python fixture argument is
test-only, appears in evidence and has no CLI flag.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any, TextIO

from benchmarks.harness import p3_stage2_dataset as frozen

MODEL_ID = "deepseek-ai/DeepSeek-V3"
REVISION = "e815299b0bcbac849fa540c768ef21845365c9eb"
TRAINING_SOURCES_SHA256 = "8ab0bdfdefde056387835fcaf38914fe2c6879cde1cf89a8f6169629630675a7"
POLICY = "p3-deepseek-quality-training-pack-v1"
STATUS = "PAIRED_DATA_PACKED_RUNTIME_AND_PRECISION_PENDING"
_DOMAIN_TOKENS = 131072
_LENGTHS = (1024, 4096)
# SHA256 of bytes whose Git blob OIDs match the primary HF revision's tree:
# tokenizer.json: 51083c600ec00c7f57ae1af7fef7108c52ce7376
# tokenizer_config.json: 3ef38058ece03201b9af6baf99b28f16a020ad64
# https://huggingface.co/api/models/deepseek-ai/DeepSeek-V3/tree/
# e815299b0bcbac849fa540c768ef21845365c9eb
_TOKENIZER_FILES = {
    "tokenizer.json": {
        "bytes": 7847652,
        "sha256": "621ac2e32d0dba658404412318818aaa8ce8cda492e59830109d8da6b517fb41",
    },
    "tokenizer_config.json": {
        "bytes": 3128,
        "sha256": "637bcd1a08cf7c772ce6a383196b22930921c79c3c73223c5afc0c7f41545546",
    },
}
_TOKENIZER_ASSETS = {
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "merges.txt", "tokenizer.model", "spiece.model",
}
_REQUIRED_FILES = {
    "training_sources": "training-sources.jsonl",
    "row_usage": "row-usage.jsonl",
    "document_spans": "document-spans.jsonl",
    "windows_L1024": "windows-L1024.json",
    "windows_L4096": "windows-L4096.json",
}


class AdmissionError(ValueError):
    """An unsuccessful build, optionally with fresh failure evidence."""

    def __init__(self, reason: str, report_path: Path | None = None):
        self.reason = reason
        self.report_path = report_path
        super().__init__(f"DEEPSEEK_DATA_ADMISSION_FAILED:{reason}; report={report_path}")


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _digest(value: Any) -> bool:
    return isinstance(value, str) and frozen._SHA256.fullmatch(value) is not None


def _production_pin() -> dict:
    return {"repo_id": MODEL_ID, "revision": REVISION, "test_only": False,
            "files": {name: dict(descriptor) for name, descriptor in _TOKENIZER_FILES.items()}}


def _pin(fixture: dict | None) -> dict:
    if fixture is None:
        return _production_pin()
    _require(isinstance(fixture, dict) and fixture.get("test_only") is True,
             "explicit_test_only_tokenizer_identity_required")
    for key in ("repo_id", "revision"):
        _require(isinstance(fixture.get(key), str) and bool(fixture[key]),
                 "test_only_tokenizer_identity_missing")
    _require("/" not in fixture["revision"] and "\\" not in fixture["revision"]
             and fixture["repo_id"] != MODEL_ID, "test_only_tokenizer_identity_invalid")
    files = fixture.get("files")
    _require(isinstance(files, dict) and {"tokenizer.json", "tokenizer_config.json"} <= files.keys()
             and set(files) <= _TOKENIZER_ASSETS, "test_only_tokenizer_files_invalid")
    for descriptor in files.values():
        _require(isinstance(descriptor, dict) and type(descriptor.get("bytes")) is int
                 and descriptor["bytes"] > 0 and _digest(descriptor.get("sha256")),
                 "test_only_tokenizer_file_pin_invalid")
    return {"repo_id": fixture["repo_id"], "revision": fixture["revision"], "test_only": True,
            "files": {name: {"bytes": value["bytes"], "sha256": value["sha256"]}
                      for name, value in sorted(files.items())}}


def _check_tokenizer_files(root: Path, pin: dict) -> dict:
    files = {}
    for name, expected in sorted(pin["files"].items()):
        # HF snapshots may link to the cache's sibling blobs directory; bytes stay pinned.
        path = root / name
        _require(path.is_file(), "tokenizer_file_missing")
        descriptor = frozen._descriptor(path, name)
        _require(all(descriptor[key] == expected[key] for key in ("bytes", "sha256")),
                 "tokenizer_file_hash_mismatch")
        files[name] = descriptor
    _require(not any((root / name).exists() for name in _TOKENIZER_ASSETS - files.keys()),
             "unpinned_tokenizer_file_present")
    return files


def load_tokenizer(tokenizer_dir: Path, *, _test_only_tokenizer: dict | None = None
                   ) -> tuple[Any, dict]:
    """Load only locally pinned fast-tokenizer bytes, never Granite or model weights."""
    pin = _pin(_test_only_tokenizer)
    root = Path(tokenizer_dir)
    _require(root.name == pin["revision"], "tokenizer_revision_mismatch")
    files = _check_tokenizer_files(root, pin)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    _require(tokenizer.is_fast is True, "fast_tokenizer_offsets_required")
    _require(type(tokenizer.eos_token_id) is int and 0 <= tokenizer.eos_token_id < len(tokenizer),
             "native_single_eos_required")
    if not pin["test_only"]:
        _require(len(tokenizer) == 128815 and tokenizer.eos_token_id == 1
                 and tokenizer.bos_token_id == 0 and tokenizer.pad_token_id == 1,
                 "deepseek_tokenizer_vocabulary_or_special_ids_mismatch")
    metadata = {"repo_id": pin["repo_id"], "revision": pin["revision"],
                "test_only": pin["test_only"], "pin": pin, "files": files,
                "tokenizer_class": type(tokenizer).__name__, "is_fast": True,
                "vocabulary_size": len(tokenizer), "eos_id": tokenizer.eos_token_id,
                "bos_id": tokenizer.bos_token_id, "pad_id": tokenizer.pad_token_id,
                "library_versions": {
                    name: version(name) for name in ("transformers", "tokenizers")},
                "weights_loaded": False, "add_special_tokens": False, "prepend_bos": False,
                "append_native_eos_per_document": True}
    return tokenizer, metadata


def _read_sources(path: Path, expected: str, *, test_only: bool,
                  diagnostics: dict | None = None) -> tuple[list[dict], list[dict]]:
    _require(_digest(expected), "invalid_training_sources_sha256")
    if not test_only:
        _require(expected == TRAINING_SOURCES_SHA256, "frozen_training_sources_sha256_required")
    _require(frozen._file_hash(path)[1] == expected, "training_sources_hash_mismatch")
    rows, identities = [], []
    refs, source_hashes, texts = set(), set(), set()
    observed = hashlib.sha256()
    if diagnostics is not None:
        diagnostics["training_source_rows_by_domain"] = {"math": 0, "code": 0}
    with path.open("rb") as stream:
        for ordinal, raw in enumerate(stream):
            observed.update(raw)
            row = frozen._json(raw.decode("utf-8"))
            _require(isinstance(row, dict), "training_source_row_not_object")
            frozen._training_contract(row)
            ref, source_hash = row.get("source_ref"), row.get("source_row_sha256")
            _require(isinstance(ref, str) and bool(ref) and _digest(source_hash),
                     "training_source_identity_invalid")
            _require(row["benchmark"] == row["source_domain"], "training_benchmark_mismatch")
            text = (row["domain"], row["prompt"], row["response"])
            _require(ref not in refs and source_hash not in source_hashes and text not in texts,
                     "duplicate_training_source_row")
            refs.add(ref)
            source_hashes.add(source_hash)
            texts.add(text)
            rows.append(row)
            identities.append({"physical_row_ordinal": ordinal, "source_ref": ref,
                               "task_id": row["task_id"], "domain": row["domain"],
                               "source_row_sha256": source_hash, "raw_row_bytes": len(raw),
                               "raw_row_sha256": hashlib.sha256(raw).hexdigest()})
            if diagnostics is not None:
                diagnostics["training_source_rows_by_domain"][row["domain"]] += 1
    _require(observed.hexdigest() == expected, "training_sources_changed_during_read")
    counts = Counter(row["domain"] for row in rows)
    _require(set(counts) == {"math", "code"}, "training_domains_missing")
    if not test_only:
        _require(len(rows) == 7801 and counts == {"math": 7431, "code": 370},
                 "frozen_training_source_count_mismatch")
    return rows, identities


def _available_tokens(tokenizer: Any, rows: list[dict], identities: list[dict]) -> dict:
    available = {"math": 0, "code": 0}
    for row, identity in zip(rows, identities):
        count = frozen._encode_document(tokenizer, row)["document_token_count"]
        identity["document_token_count"] = count
        available[row["domain"]] += count
    return available


def _row_usage(identities: list[dict], spans: list[dict]) -> list[dict]:
    used = {(span["source_ref"], span["cycle"]): span for span in spans}
    _require(len(used) == len(spans), "duplicate_source_cycle_occurrence")
    by_ref = {identity["source_ref"]: identity for identity in identities}
    occurrences = [(identity["source_ref"], 0) for identity in identities]
    occurrences.extend((span["source_ref"], span["cycle"]) for span in spans if span["cycle"] > 0)
    usage = []
    for ref, cycle in occurrences:
        span = used.get((ref, cycle))
        usage.append({**by_ref[ref], "cycle": cycle,
                      "used_input_token_count": span["stream_end"] - span["stream_start"]
                      if span is not None else 0,
                      "appended_eos_included": span["appended_eos_included"]
                      if span is not None else False})
    return usage


def _usage_metadata(identities: list[dict], spans: list[dict]) -> dict:
    unique_counts, cycle_counts = {}, {}
    occurrences = {}
    for domain in ("math", "code"):
        available = sum(row["domain"] == domain for row in identities)
        domain_spans = [span for span in spans if span["domain"] == domain]
        unique_counts[domain] = {"available": available,
                                 "used": len({span["source_ref"] for span in domain_spans})}
        max_cycle = max(span["cycle"] for span in domain_spans)
        complete_documents = sum(not span["final_document_cutoff"] for span in domain_spans)
        cycle_counts[domain] = {"started": max_cycle + 1,
                                "completed": complete_documents // available,
                                "repeated": max_cycle}
        occurrences[domain] = len(domain_spans)
    return {"unique_source_counts_by_domain": unique_counts, "cycle_counts_by_domain": cycle_counts,
            "row_usage": {
                "available_by_domain": {domain: value["available"]
                                        for domain, value in unique_counts.items()},
                "used_by_domain": {
                    domain: value["used"] for domain, value in unique_counts.items()},
                "occurrences_by_domain": occurrences}}


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(frozen._compact(value) + "\n")


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)


def _failure(root: Path, error: Exception, diagnostics: dict) -> AdmissionError:
    reason = str(error) if type(error) is ValueError else f"prerequisite:{type(error).__name__}"
    if not reason or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_:" for char in reason):
        reason = f"admission:{type(error).__name__}"
    path = root / "FAILED.json"
    _write_json(path, {"schema": POLICY, "status": "FAILED", "reason": reason,
                       "quality_non_degradation_admitted": False, **diagnostics})
    return AdmissionError(reason, path)


def build_pack(*, sources: Path, sources_sha256: str, tokenizer_dir: Path, output: Path,
               _test_only_tokenizer: dict | None = None) -> dict:
    """Freeze 64 windows per length with Granite's fixed-order domain cycling."""
    try:
        root = frozen._fresh_root(Path(output))
    except frozen.AdmissionError:
        raise AdmissionError("existing_output_preserved") from None
    diagnostics: dict = {}
    try:
        pin = _pin(_test_only_tokenizer)
        diagnostics = {"training_sources_sha256": sources_sha256, "test_only": pin["test_only"]}
        with tempfile.TemporaryDirectory(prefix=".deepseek-pack-", dir=root) as temporary:
            stage = Path(temporary)
            source_copy = stage / "training-sources.jsonl"
            _copy(Path(sources), source_copy)
            rows, identities = _read_sources(source_copy, sources_sha256,
                                             test_only=pin["test_only"], diagnostics=diagnostics)
            tokenizer, tokenizer_spec = load_tokenizer(
                Path(tokenizer_dir), _test_only_tokenizer=_test_only_tokenizer)
            available = _available_tokens(tokenizer, rows, identities)
            diagnostics.update({
                "unique_available_tokens_by_domain": available,
                "domain_input_token_counts": {domain: _DOMAIN_TOKENS for domain in available},
                "tokenizer_repo_id": tokenizer_spec["repo_id"],
                "tokenizer_revision": tokenizer_spec["revision"],
            })
            windows, spans, metadata = frozen._pack(tokenizer, rows)
            usage = _row_usage(identities, spans)
            files = {"training_sources": {**frozen._descriptor(source_copy, source_copy.name),
                                          "rows": len(rows)},
                     "row_usage": frozen._write_jsonl(stage / "row-usage.jsonl", usage),
                     "document_spans": frozen._write_jsonl(stage / "document-spans.jsonl", spans)}
            for length, layout in windows.items():
                name = f"windows-L{length}.json"
                _write_json(stage / name, layout)
                files[f"windows_L{length}"] = {**frozen._descriptor(stage / name, name),
                                               "format": "json_array", "rows": 64,
                                               "length": length, "batch_size": 1,
                                               "micro_batches_per_window": 4096 // length}
            for name in pin["files"]:
                target = stage / "tokenizer" / name
                _copy(Path(tokenizer_dir) / name, target)
                descriptor = frozen._descriptor(target, f"tokenizer/{name}")
                _require(all(descriptor[key] == pin["files"][name][key]
                             for key in ("bytes", "sha256")),
                         "tokenizer_file_changed_during_copy")
                files[f"tokenizer/{name}"] = descriptor
            manifest = {"schema": POLICY, "status": STATUS, "policy": POLICY,
                        "test_only": pin["test_only"], "training_sources_sha256": sources_sha256,
                        "training_source_row_count": len(rows), "files": files,
                        "raw_row_hash_policy": "sha256_exact_physical_line_including_terminator",
                        "tokenizer": tokenizer_spec, **metadata,
                        "unique_available_tokens_by_domain": available,
                        **_usage_metadata(identities, spans),
                        "window_count_per_length": 64, "domain_windows": {"math": 32, "code": 32},
                        "supervised_target_counts_per_length": {
                            str(length): sum(row["valid_target_count"] for row in layout)
                            for length, layout in windows.items()},
                        "granite_reference_supervised_targets_per_arm": 179326,
                        "sft_input_fields": ["prompt", "response"],
                        "response_label_policy": (
                            "joint_character_offsets_start_at_or_after_response_plus_native_EOS"),
                        "common_boundary_mask_interval": 1024,
                        "source_order": "sha256_p3-stage2-data-order-v1_NUL_source_ref",
                        "exhaustion_policy": (
                            "cycle_fixed_order_record_identity_cycle_and_final_cutoff"),
                        "seeds": [23], "seed_changes_data": False,
                        "model_weights_loaded": False, "quality_non_degradation_admitted": False}
            frozen._write_manifest(stage / "manifest.json", manifest)
            frozen._publish(stage, root, [*(value["path"] for value in files.values()),
                                          "manifest.json"])
            return manifest
    except Exception as error:
        raise _failure(root, error, diagnostics) from None


def _tokenizer_contract(manifest: dict) -> dict:
    spec = manifest.get("tokenizer")
    _require(isinstance(spec, dict) and type(manifest.get("test_only")) is bool,
             "tokenizer_identity_missing")
    pin = _pin(spec.get("pin")) if manifest["test_only"] else _production_pin()
    _require(spec.get("pin") == pin and spec.get("repo_id") == pin["repo_id"]
             and spec.get("revision") == pin["revision"]
             and spec.get("test_only") is manifest["test_only"], "tokenizer_identity_mismatch")
    _require(spec.get("is_fast") is True and spec.get("add_special_tokens") is False
             and spec.get("prepend_bos") is False and spec.get("weights_loaded") is False
             and spec.get("append_native_eos_per_document") is True,
             "tokenizer_encoding_policy_mismatch")
    vocabulary, eos = spec.get("vocabulary_size"), spec.get("eos_id")
    _require(type(vocabulary) is int and vocabulary > 0 and type(eos) is int
             and 0 <= eos < vocabulary, "tokenizer_vocabulary_invalid")
    if not pin["test_only"]:
        _require((vocabulary, eos, spec.get("bos_id"), spec.get("pad_id")) == (128815, 1, 0, 1),
                 "deepseek_tokenizer_vocabulary_or_special_ids_mismatch")
    _require(isinstance(spec.get("files"), dict) and set(spec["files"]) == set(pin["files"]),
             "tokenizer_files_missing")
    for name, expected in pin["files"].items():
        _require(spec["files"][name] == {"path": name, **expected},
                 "tokenizer_file_identity_mismatch")
        _require(manifest["files"].get(f"tokenizer/{name}") == {
            "path": f"tokenizer/{name}", **expected}, "tokenizer_file_descriptor_mismatch")
    return pin


def _validate_usage(root: Path, manifest: dict) -> None:
    _, identities = _read_sources(root / "training-sources.jsonl",
                                   manifest.get("training_sources_sha256"),
                                   test_only=manifest["test_only"])
    usage = list(frozen._jsonl(root / "row-usage.jsonl"))
    spans = list(frozen._jsonl(root / "document-spans.jsonl"))
    _require(len(identities) == manifest.get("training_source_row_count")
             == manifest["files"]["training_sources"].get("rows")
             and len(usage) >= len(identities), "training_source_row_count_mismatch")
    available = {"math": 0, "code": 0}
    by_ref = {}
    for identity, row in zip(identities, usage):
        _require(all(row.get(key) == value for key, value in identity.items())
                 and type(row.get("cycle")) is int and row["cycle"] == 0,
                 "raw_training_row_identity_mismatch")
        count = row.get("document_token_count")
        _require(type(count) is int and count > 0, "source_row_token_count_invalid")
        identity["document_token_count"] = count
        available[row["domain"]] += count
        by_ref[row["source_ref"]] = identity
    _require(available == manifest.get("unique_available_tokens_by_domain"),
             "unique_available_token_count_mismatch")
    seen = set()
    ordered_spans = []
    for domain in ("math", "code"):
        cursor = 0
        domain_spans = [span for span in spans if span.get("domain") == domain]
        ordered_spans.extend(domain_spans)
        order = manifest.get("domain_source_order", {}).get(domain)
        expected_order = sorted(
            (ref for ref, row in by_ref.items() if row["domain"] == domain),
            key=lambda ref: (frozen._text_hash(frozen._ORDER_PREFIX + ref), ref))
        _require(order == expected_order and bool(order) and bool(domain_spans),
                 "domain_source_order_mismatch")
        for number, span in enumerate(domain_spans):
            cycle, position = divmod(number, len(order))
            ref = span.get("source_ref")
            row = by_ref.get(ref)
            _require(row is not None and row["domain"] == domain and ref == order[position]
                     and type(span.get("cycle")) is int and span["cycle"] == cycle
                     and type(span.get("source_order_position")) is int
                     and span["source_order_position"] == position
                     and type(span.get("domain_document_number")) is int
                     and span["domain_document_number"] == number,
                     "source_cycle_or_order_identity_mismatch")
            key = (ref, cycle)
            _require(key not in seen and cursor < _DOMAIN_TOKENS,
                     "duplicate_or_excess_source_cycle_occurrence")
            seen.add(key)
            count = row["document_token_count"]
            take = min(count, _DOMAIN_TOKENS - cursor)
            end = cursor + take
            _require(all(type(span.get(field)) is int for field in (
                "stream_start", "stream_end", "document_token_start", "document_token_end",
                "document_token_count"))
                and span["stream_start"] == cursor and span["stream_end"] == end
                and span["document_token_start"] == 0 and span["document_token_end"] == take
                and span["document_token_count"] == count
                and span.get("source_row_sha256") == row["source_row_sha256"]
                and span.get("task_id") == row["task_id"]
                and span.get("appended_eos_included") is (take == count)
                and span.get("final_document_cutoff") is (take < count),
                "document_span_usage_mismatch")
            cursor = end
        _require(cursor == _DOMAIN_TOKENS
                 and manifest.get("final_cutoffs", {}).get(domain) == domain_spans[-1],
                 "domain_span_coverage_mismatch")
    _require(ordered_spans == spans, "domain_span_order_mismatch")
    _require(usage == _row_usage(identities, spans), "source_cycle_usage_mismatch")
    for key, expected in _usage_metadata(identities, spans).items():
        _require(manifest.get(key) == expected, "source_cycle_or_unique_count_mismatch")


def _validate_windows(root: Path, manifest: dict) -> dict[int, list[dict]]:
    from benchmarks.harness import p3_stage2_training as training

    layouts = {}
    pairings = {}
    spans = list(frozen._jsonl(root / "document-spans.jsonl"))
    domain_ids = {"math": [], "code": []}
    domain_labels = {"math": [], "code": []}
    for length in _LENGTHS:
        descriptor = manifest["files"][f"windows_L{length}"]
        _require(all(descriptor.get(key) == value for key, value in {
            "format": "json_array", "rows": 64, "length": length, "batch_size": 1,
            "micro_batches_per_window": 4096 // length}.items()), "window_descriptor_mismatch")
        windows = frozen._json((root / descriptor["path"]).read_text(encoding="utf-8"))
        _require(isinstance(windows, list), "windows_must_be_array")
        infos = training._study_windows(windows, preflight=False)
        ids, labels = [], []
        for index, info in enumerate(infos):
            window = info["window"]
            domain = "math" if index % 2 == 0 else "code"
            start = (index // 2) * 4096
            _require(info["micro_batch_shapes"] == [[1, length]] * (4096 // length)
                     and window.get("window_id") == f"{frozen.PACKING_POLICY}/window-{index:03d}"
                     and window.get("domain") == domain
                     and window.get("domain_stream_start") == start
                     and window.get("domain_stream_end") == start + 4096
                     and window.get("common_boundary_mask_interval") == 1024,
                     "window_layout_or_order_mismatch")
            expected_spans = []
            for span in spans:
                if span["domain"] != domain:
                    continue
                left = max(start, span["stream_start"])
                right = min(start + 4096, span["stream_end"])
                if left < right:
                    expected_spans.append({
                        "source_ref": span["source_ref"], "cycle": span["cycle"],
                        "domain_document_number": span["domain_document_number"],
                        "window_token_start": left - start, "window_token_end": right - start,
                        "document_token_start": left - span["stream_start"],
                        "document_token_end": right - span["stream_start"]})
            _require(window.get("document_spans") == expected_spans,
                     "window_document_span_mismatch")
            for batch in window["micro_batches"]:
                _require(all(token < manifest["tokenizer"]["vocabulary_size"]
                             for token in batch["input_ids"][0]), "window_vocabulary_mismatch")
                ids.extend(batch["input_ids"][0])
                labels.extend(batch["labels"][0])
                if length == 1024:
                    domain_ids[domain].extend(batch["input_ids"][0])
                    domain_labels[domain].extend(batch["labels"][0])
        targets = sum(info["valid_target_count"] for info in infos)
        input_hash, label_hash = frozen._ids_hash(ids), frozen._ids_hash(labels)
        pairings[str(length)] = {
            "input_sha256": input_hash, "labels_sha256": label_hash, "valid_target_count": targets,
            "pairing_sha256": frozen._pairing(input_hash, label_hash, targets),
            "ordered_window_pairing_sha256": hashlib.sha256(b"".join(
                bytes.fromhex(info["pairing_sha256"]) for info in infos)).hexdigest()}
        layouts[length] = windows
    _require(pairings["1024"] == pairings["4096"] == manifest.get("flat_pairing", {}).get("1024")
             and manifest.get("flat_pairing") == pairings, "flat_layout_pairing_mismatch")
    _require(manifest.get("trained_target_count") == pairings["1024"]["valid_target_count"]
             and manifest.get("supervised_target_counts_per_length") == {
                 length: value["valid_target_count"] for length, value in pairings.items()},
             "supervised_target_count_mismatch")
    domains = {"math": 0, "code": 0}
    for short, long in zip(layouts[1024], layouts[4096]):
        _require(short["document_spans"] == long["document_spans"], "paired_document_span_mismatch")
        domains[short["domain"]] += short["valid_target_count"]
    _require(domains == manifest.get("domain_trained_target_counts"),
             "domain_supervised_target_count_mismatch")
    eos = manifest["tokenizer"]["eos_id"]
    for span in spans:
        domain = span["domain"]
        start, stop = span["stream_start"], span["stream_end"]
        ids = domain_ids[domain][start:stop]
        _require(frozen._ids_hash(ids) == span.get("input_sha256"),
                 "document_input_hash_mismatch")
        if span["appended_eos_included"]:
            _require(ids[-1] == eos, "document_native_eos_mismatch")
            if (stop - 1) % 1024:
                _require(domain_labels[domain][stop - 1] == eos,
                         "document_native_eos_supervision_mismatch")
    return layouts


def load_pack(pack: Path, pack_sha256: str, *, length: int, steps: int
              ) -> tuple[list[dict], dict]:
    """Verify the manifest and every pack file before returning a frozen-order prefix.

    ``pack_sha256`` is SHA256 of manifest.json bytes, not its internal payload hash.
    Fixture packs remain visibly test-only; full-model callers must reject them.
    No tokenizer, model weights or CUDA context is loaded by this function.
    """
    _require(type(length) is int and length in _LENGTHS, "unsupported_training_length")
    _require(type(steps) is int and 1 <= steps <= 64, "insufficient_windows_or_invalid_steps")
    _require(_digest(pack_sha256), "invalid_pack_manifest_hash")
    root = Path(pack)
    _require(frozen._file_hash(root / "manifest.json")[1] == pack_sha256,
             "pack_manifest_file_hash_mismatch")
    manifest = frozen._read_manifest(root)
    _require(manifest.get("schema") == POLICY and manifest.get("policy") == POLICY
             and manifest.get("status") == STATUS, "completed_deepseek_pack_required")
    files = manifest["files"]
    for key, name in _REQUIRED_FILES.items():
        _require(isinstance(files.get(key), dict) and files[key].get("path") == name,
                 "required_pack_file_missing_or_path_mismatch")
    pin = _tokenizer_contract(manifest)
    _require(set(files) == {*_REQUIRED_FILES, *(f"tokenizer/{name}" for name in pin["files"])},
             "unexpected_pack_file_descriptor")
    _require(manifest.get("input_token_count") == 262144
             and manifest.get("domain_input_token_counts") == {"math": 131072, "code": 131072}
             and manifest.get("domain_windows") == {"math": 32, "code": 32}
             and manifest.get("window_count_per_length") == 64
             and manifest.get("common_boundary_mask_interval") == 1024
             and manifest.get("exhaustion_policy") == (
                 "cycle_fixed_order_record_identity_cycle_and_final_cutoff")
             and manifest.get("sft_input_fields") == ["prompt", "response"]
             and manifest.get("raw_row_hash_policy") == (
                 "sha256_exact_physical_line_including_terminator")
             and manifest.get("response_label_policy") == (
                 "joint_character_offsets_start_at_or_after_response_plus_native_EOS")
             and manifest.get("source_order") == "sha256_p3-stage2-data-order-v1_NUL_source_ref"
             and manifest.get("seeds") == [23] and manifest.get("seed_changes_data") is False
             and manifest.get("model_weights_loaded") is False
             and manifest.get("quality_non_degradation_admitted") is False,
             "pack_policy_or_allocation_mismatch")
    _require(files["training_sources"]["sha256"] == manifest.get("training_sources_sha256"),
             "training_sources_descriptor_hash_mismatch")
    _validate_usage(root, manifest)
    layouts = _validate_windows(root, manifest)
    return layouts[length][:steps], manifest


class _Parser(argparse.ArgumentParser):
    def _print_message(self, message: str, file: TextIO | None = None) -> None:
        from rich.console import Console

        if message:
            Console(file=file or sys.stdout, highlight=False, markup=False).print(message, end="")


def main(argv: list[str] | None = None) -> int:
    """Build a production-only training pack, returning a shell exit status."""
    from rich.console import Console

    parser = _Parser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--sources-sha256", required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    console = Console(highlight=False, markup=False)
    try:
        manifest = build_pack(sources=args.sources, sources_sha256=args.sources_sha256,
                              tokenizer_dir=args.tokenizer_dir, output=args.output)
    except AdmissionError as error:
        console.print(str(error))
        return 1
    digest = frozen._file_hash(args.output / "manifest.json")[1]
    console.print(f"Pack manifest SHA256: {digest}")
    console.print({"supervised_targets_per_length": manifest["supervised_target_counts_per_length"],
                   "granite_reference_per_arm": 179326})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
