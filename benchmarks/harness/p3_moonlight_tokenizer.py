"""Deterministic Moonlight fast-tokenizer conversion and frozen-corpus parity.

Both CLIs are local-files-only and never execute the Hub tokenizer's Python code.
``convert --source-dir REVISION --output NEW_PARENT/REVISION --report REPORT``
converts the admitted source bytes twice and checks the retained v1 byte pins.
``parity --source-dir REVISION --tokenizer-dir REVISION --repo REPO --output NEW_DIR``
checks every frozen training/dev/eval request and records exact IDs and UTF-8 offsets.
The parity report does not supply the model-reference input: that comes from the pack.
"""

from __future__ import annotations

import argparse
import ast
import base64
import bisect
import copy
import hashlib
import importlib.metadata
import inspect
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TextIO

MODEL_ID = "moonshotai/Moonlight-16B-A3B"
REVISION = "476b36a473d4467f94469414bef6cee75c9c8172"
VOCAB_BOUND = 163840
MAX_FILE_BYTES = 64 * 1024**2
MAX_TEXT_CHARACTERS = 400000
PACKAGE_PINS = {
    "transformers": "5.19.0", "tokenizers": "0.23.3",
    "tiktoken": "0.14.0", "regex": "2026.11.9", "blobfile": "3.1.0",
}
PROTOCOL_PATH = "benchmarks/results/p3-stage2-v2/p3-stage2-v2-protocol.json"
PROTOCOL_SHA256 = "685fa5bee528969d9b2eef423c90280d179432ba017fbebed12b687f608a8249"
DATA_ROOT = "benchmarks/results/p3-stage2-v2/data-build-kaggle-v1/evidence/data"
DATA_MANIFEST_SHA256 = "07f3c65cce76d0b209ce20aff7b096857077ac906156ac4ecf2c7955141b2b81"
TRAINING_PATH = (
    "benchmarks/results/p3-stage2-quality/admission-kaggle-v3/"
    "evidence/dataset/training-sources.jsonl"
)
TRAINING_SHA256 = "8ab0bdfdefde056387835fcaf38914fe2c6879cde1cf89a8f6169629630675a7"
SOURCE_PINS = {
    "benchmarks/harness/p3_stage2_dataset.py":
        "48c0e5cba7adc6164cadf428c0ba34844ab631817f1b4942829688d824ef4bda",
    "benchmarks/harness/p3_stage2_generation.py":
        "ed24c24e6ad148e75f80c10fc34322d76e9c3c9bd775f46630a916f2753d56d1",
    "benchmarks/harness/p3_stage2v2_loglik.py":
        "08c37c95bdf25088454cdbfdbca74dc4283ed279685a4431f782efe47b1b579f",
}
HUB_FILES = {
    "tiktoken.model": {
        "identity_kind": "lfs_sha256",
        "identity": "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103",
        "limit_bytes": 2795286,
    },
    "tokenizer_config.json": {
        "identity_kind": "git_blob_sha1",
        "identity": "6eee45b6e1ff757799cf1da5d52d0c266d5894c3",
        "limit_bytes": 65536,
    },
    "tokenization_moonshot.py": {
        "identity_kind": "git_blob_sha1",
        "identity": "6c4903cc80d4e347ba5e25d57160268edd7ee1e4",
        "limit_bytes": 131072,
    },
}
ENDPOINT_COUNTS = {
    "hellaswag_dev": 2000, "arc_dev": 1000, "boolq_dev": 1000,
    "hellaswag": 10042, "arc": 3548, "boolq": 3270,
    "lambada": 5153, "gsm8k": 1288, "mbpp": 494,
}
SPECIAL_IDS = {"bos_token": 163584, "eos_token": 163585,
               "pad_token": 163838, "unk_token": 163839}

PROFILE_ID = "moonlight-16b-a3b"
GENERATED_FILES = {
    "tokenizer.json": {
        "bytes": 19592675,
        "sha256": "a5b137b65b40e0283a3c646472444078503dd3cb83c64b69379beca74288773d",
    },
    "tokenizer_config.json": {
        "bytes": 344,
        "sha256": "7e0f5761740d72e66d290afdce2cf9d10c598d018503554066ee55d5bf2a30a2",
    },
    "chat_template.jinja": {
        "bytes": 527,
        "sha256": "0af7f5845651f3ba85a82d70d8af968a1da0a60f4c7bb662a9c305cfc9cc224c",
    },
}


class AdmissionError(ValueError):
    def __init__(self, code: str, **details: Any):
        super().__init__(code)
        self.code = code
        self.details = details


def require(condition: bool, code: str, **details: Any) -> None:
    if not condition:
        raise AdmissionError(code, **details)


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def object_hash(value: Any) -> str:
    return sha256(compact(value).encode("utf-8"))


def descriptor(raw: bytes) -> dict:
    return {"bytes": len(raw), "sha256": sha256(raw)}


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(compact(value) + "\n")
    os.replace(temporary, path)


def child(root: Path, relative: str) -> Path:
    require(isinstance(relative, str) and bool(relative) and "\\" not in relative
            and ":" not in relative and not Path(relative).is_absolute()
            and all(part not in {"", ".", ".."} for part in relative.split("/")),
            "invalid_relative_path", path=relative)
    base = os.path.realpath(root)
    path = os.path.realpath(root / relative)
    require(path != base and os.path.commonpath((base, path)) == base,
            "path_escapes_root", path=relative)
    return Path(path)


def checkpoint(output: Path, report: dict, stage: str | None = None) -> None:
    if stage is not None:
        report["stage"] = stage
    write_json(output / "progress.json", report)


def bounded_read(path: Path, maximum: int = MAX_FILE_BYTES) -> bytes:
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    require(len(raw) <= maximum, "file_size_limit", path=path.name, limit=maximum)
    return raw


def read_json(raw: bytes) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict:
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate_json_key", key=key)
            value[key] = item
        return value

    return json.loads(raw.decode("utf-8"), object_pairs_hook=unique)


def verified_file(repo: Path, relative: str, expected: str, report: dict,
                  bindings: dict, expected_bytes: int | None = None) -> bytes:
    raw = bounded_read(child(repo, relative))
    observed = descriptor(raw)
    report["frozen_inputs"][relative] = {
        **observed, "expected_sha256": expected, "expected_bytes": expected_bytes,
        "verified": observed["sha256"] == expected
        and (expected_bytes is None or observed["bytes"] == expected_bytes),
    }
    require(report["frozen_inputs"][relative]["verified"], "frozen_file_mismatch",
            path=relative, observed=observed, expected_sha256=expected,
            expected_bytes=expected_bytes)
    bindings[relative] = observed
    return raw


def frozen_functions(raw: bytes, relative: str, names: tuple[str, ...],
                     namespace: dict, report: dict) -> SimpleNamespace:
    """Load only named functions from hash-verified LOCAL frozen source snapshots."""
    tree = ast.parse(raw.decode("utf-8"), filename=relative)
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    require(len(selected) == len(names) and {node.name for node in selected} == set(names),
            "frozen_function_inventory_mismatch", path=relative, names=list(names))
    for node in selected:
        require(not node.decorator_list, "unexpected_frozen_function_decorator", name=node.name)
        snippet = ast.get_source_segment(raw.decode("utf-8"), node)
        report["frozen_functions"][relative + ":" + node.name] = {
            "source_sha256": sha256(snippet.encode("utf-8")),
            "first_line": node.lineno, "last_line": node.end_lineno,
        }
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    # The Hub .py file never reaches this function. These three local source files
    # must match both SOURCE_PINS and the independently pinned frozen protocol.
    exec(compile(module, relative, "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def coverage_key(role: str, endpoint: str, split: str) -> str:
    return f"{role}/{endpoint}/{split}"


def bucket(report: dict, key: str) -> dict:
    return report["coverage"].setdefault(key, {
        "required_rows": 0, "completed_rows": 0,
        "required_requests": 0, "completed_requests": 0,
        "required_texts_by_part": {}, "attempted_texts_by_part": {},
        "passed_texts_by_part": {}, "reference_tokens": 0,
        "verified_utf8_bytes": 0, "verified_characters": 0,
    })


def increment(mapping: dict, key: str, amount: int = 1) -> None:
    mapping[key] = mapping.get(key, 0) + amount


def admit_corpus(repo: Path, output: Path, report: dict) -> tuple:
    bindings = {}
    protocol = read_json(verified_file(repo, PROTOCOL_PATH, PROTOCOL_SHA256, report, bindings))
    require(protocol.get("schema") == "p3-stage2-v2-protocol"
            and protocol.get("status") == "FROZEN", "frozen_protocol_schema")
    require(protocol["data"]["dataset_manifest_file_sha256"] == DATA_MANIFEST_SHA256,
            "protocol_data_manifest_binding")
    require(protocol["loglik"]["add_special_tokens"] is False
            and protocol["loglik"]["prepend_bos"] is False, "frozen_tokenization_policy")
    sources = {}
    for relative, digest in SOURCE_PINS.items():
        require(protocol["source_files_sha256"].get(relative) == digest,
                "protocol_source_binding", path=relative)
        sources[relative] = verified_file(repo, relative, digest, report, bindings)
    generation_path = "benchmarks/harness/p3_stage2_generation.py"
    generation = frozen_functions(sources[generation_path], generation_path,
                                  ("_backend_state", "_restore_backend"),
                                  {"copy": copy}, report)
    dataset_path = "benchmarks/harness/p3_stage2_dataset.py"
    dataset = frozen_functions(sources[dataset_path], dataset_path,
                               ("_training_contract", "_encode_document"),
                               {"_require": require, "_text_hash": lambda text:
                                sha256(text.encode("utf-8"))}, report)
    loglik_path = "benchmarks/harness/p3_stage2v2_loglik.py"
    loglik = frozen_functions(sources[loglik_path], loglik_path,
                              ("_requests", "_tokenize", "encode_pair"),
                              {"_require": require, "generation": generation}, report)
    manifest_relative = DATA_ROOT + "/manifest.json"
    manifest = read_json(verified_file(repo, manifest_relative, DATA_MANIFEST_SHA256,
                                       report, bindings))
    require(manifest.get("schema") == "p3-stage2-v2-data-v1", "data_manifest_schema")
    payload = {key: value for key, value in manifest.items() if key != "manifest_payload_sha256"}
    require(object_hash(payload) == manifest["manifest_payload_sha256"]
            == protocol["data"]["dataset_manifest_payload_sha256"], "data_manifest_payload")
    task_bytes = {}
    for name, spec in sorted(manifest["files"].items()):
        relative = DATA_ROOT + "/" + spec["path"]
        raw = verified_file(repo, relative, spec["sha256"], report, bindings, spec["bytes"])
        if "rows" in spec:
            require(len(raw.splitlines()) == spec["rows"], "data_file_row_count", path=relative)
        if name.endswith("_tasks"):
            task_bytes[name.removesuffix("_tasks")] = raw
    require(set(manifest["endpoints"]) == set(task_bytes) == set(ENDPOINT_COUNTS),
            "endpoint_inventory_mismatch")
    training_spec = manifest["inputs"]["training_sources"]
    require(training_spec["sha256"] == TRAINING_SHA256 and training_spec["rows"] == 7801,
            "frozen_training_manifest_binding")
    training_raw = verified_file(repo, TRAINING_PATH, TRAINING_SHA256, report, bindings,
                                 training_spec["bytes"])
    training = []
    endpoints = {}
    refs, source_hashes, texts = set(), set(), set()
    inventory = output / "corpus-inventory.jsonl"
    with inventory.open("x", encoding="utf-8", newline="\n") as ledger:
        for ordinal, raw in enumerate(training_raw.splitlines(keepends=True)):
            row = read_json(raw)
            dataset._training_contract(row)
            require(row["benchmark"] == row["source_domain"], "training_benchmark_mismatch")
            ref, source_hash = row["source_ref"], row["source_row_sha256"]
            require(isinstance(ref, str) and ref and isinstance(source_hash, str)
                    and len(source_hash) == 64
                    and all(c in "0123456789abcdef" for c in source_hash),
                    "training_source_identity")
            text_identity = (row["domain"], row["prompt"], row["response"])
            require(ref not in refs and source_hash not in source_hashes
                    and text_identity not in texts,
                    "duplicate_training_row")
            refs.add(ref)
            source_hashes.add(source_hash)
            texts.add(text_identity)
            identity = {"role": "training", "endpoint": row["benchmark"], "split": "train",
                        "physical_row_ordinal": ordinal, "task_id": row["task_id"],
                        "source_ref": ref, "source_row_sha256": source_hash,
                        "raw_row_sha256": sha256(raw)}
            row["_probe_identity"] = identity
            training.append(row)
            ledger.write(compact({**identity, "text_sha256": {
                part: sha256(text.encode("utf-8")) for part, text in (
                    ("prompt", row["prompt"]), ("response", row["response"]),
                    ("document", row["prompt"] + row["response"]))}}) + "\n")
            counts = bucket(report, coverage_key("training", row["benchmark"], "train"))
            counts["required_rows"] += 1
            counts["required_requests"] += 1
            for part in ("prompt", "response", "document"):
                increment(counts["required_texts_by_part"], part)
        require(len(training) == 7801
                and Counter(row["domain"] for row in training) == {"math": 7431, "code": 370},
                "frozen_training_row_count")
        for endpoint, expected_count in ENDPOINT_COUNTS.items():
            entry = manifest["endpoints"][endpoint]
            role = "development" if endpoint.endswith("_dev") else "evaluation"
            require(entry["role"] == role and entry["retained_row_count"] == expected_count,
                    "endpoint_role_or_count", endpoint=endpoint)
            require(entry["files"]["tasks"] == manifest["files"][endpoint + "_tasks"],
                    "endpoint_file_binding", endpoint=endpoint)
            rows = [read_json(raw) for raw in task_bytes[endpoint].splitlines()]
            ids = [row["task_id"] for row in rows]
            require(len(rows) == expected_count and len(set(ids)) == expected_count
                    and ids == entry["retained_task_ids"]
                    and object_hash(ids) == entry["ordered_task_ids_sha256"],
                    "complete_ordered_endpoint_coverage", endpoint=endpoint)
            if role == "evaluation":
                require(protocol["data"]["full_retained_evaluation_tasks"][endpoint] == len(rows)
                        and protocol["data"]["source_ordered_task_ids_sha256"][endpoint]
                        == object_hash(ids), "protocol_endpoint_binding", endpoint=endpoint)
            for ordinal, (row, raw) in enumerate(zip(
                    rows, task_bytes[endpoint].splitlines(keepends=True), strict=True)):
                split = row["source_split"]
                require(row["endpoint"] == endpoint and row["role"] == role
                        and row["source_ref"] == row["task_id"]
                        and row["metric"] == entry["metric"], "task_identity", endpoint=endpoint)
                require(split == "train" if role == "development" else
                        split in ("validation", "test"), "task_split", endpoint=endpoint)
                if role == "development":
                    require(row["source_row_ordinal"] == ordinal
                            and row["source_ref"].endswith(f"#row={ordinal}"),
                            "development_physical_order", endpoint=endpoint)
                requests, sizes = loglik._requests([row])
                require(sizes == [len(requests)] and requests, "request_cardinality")
                identity = {"role": role, "endpoint": endpoint, "split": split,
                            "physical_row_ordinal": ordinal, "task_id": row["task_id"],
                            "source_ref": row["source_ref"],
                            "source_row_sha256": row["source_row_sha256"],
                            "raw_row_sha256": sha256(raw)}
                row["_probe_identity"] = identity
                ledger.write(compact({**identity, "requests": [
                    {"request_index": index, "context_sha256": sha256(context.encode("utf-8")),
                     "continuation_sha256": sha256(continuation.encode("utf-8")),
                     "joint_text_sha256": sha256((context + continuation).encode("utf-8"))}
                    for index, (context, continuation) in enumerate(requests)]}) + "\n")
                counts = bucket(report, coverage_key(role, endpoint, split))
                counts["required_rows"] += 1
                counts["required_requests"] += len(requests)
                parts = (("prompt", "response", "document") if row["metric"] == "nll"
                         else ("request_context", "request_continuation",
                               "effective_context", "joint_document"))
                for part in parts:
                    increment(counts["required_texts_by_part"], part, len(requests))
            endpoints[endpoint] = rows
    report["corpus_inventory"] = {"path": inventory.name, **descriptor(inventory.read_bytes())}
    report["source_verification_scope"] = (
        "All three executed frozen tokenizer/request source files, pinned protocol, "
        "complete v2 data-manifest file inventory, and the original 7801 training rows. "
        "No unrelated model/training implementation is imported or admitted by this probe."
    )
    return training, endpoints, dataset, loglik, bindings


def package_provenance(report: dict) -> None:
    for name in (*PACKAGE_PINS, "rich", "huggingface-hub", "numpy", "packaging"):
        try:
            distribution = importlib.metadata.distribution(name)
            record = {"version": distribution.version, "native_files": {}}
            for file in distribution.files or ():
                if str(file).endswith((".so", ".pyd", ".dist-info/RECORD")):
                    raw = Path(distribution.locate_file(file)).read_bytes()
                    record["native_files"][str(file)] = descriptor(raw)
            report["packages"][name] = record
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = {"version": None}
    for name, expected in PACKAGE_PINS.items():
        require(report["packages"][name]["version"] == expected, "package_version_mismatch",
                package=name, expected=expected, observed=report["packages"][name]["version"])
    for name in ("rich", "blobfile"):
        require(report["packages"][name]["version"] is not None, "missing_dependency", package=name)


def reference_pattern(raw: bytes) -> tuple[str, int]:
    """Interpret the two literal class attributes, never compile remote source."""
    tree = ast.parse(raw.decode("utf-8"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "TikTokenTokenizer"]
    require(len(classes) == 1, "reference_class_missing")
    attributes = {}
    for node in classes[0].body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                attributes[target.id] = node.value
    pattern = attributes.get("pat_str")
    require(isinstance(pattern, ast.Call) and isinstance(pattern.func, ast.Attribute)
            and isinstance(pattern.func.value, ast.Constant) and pattern.func.value.value == "|"
            and pattern.func.attr == "join" and len(pattern.args) == 1 and not pattern.keywords,
            "reference_pattern_not_literal_join")
    pieces = ast.literal_eval(pattern.args[0])
    require(isinstance(pieces, list) and len(pieces) == 8
            and all(isinstance(piece, str) for piece in pieces), "reference_pattern_parts")
    reserved = ast.literal_eval(attributes["num_reserved_special_tokens"])
    require(reserved == 256, "reference_reserved_token_count")
    return "|".join(pieces), reserved


def mergeable_ranks(raw: bytes) -> dict[bytes, int]:
    result = {}
    for ordinal, line in enumerate(raw.splitlines()):
        fields = line.split()
        require(len(fields) == 2, "invalid_tiktoken_row", row=ordinal)
        token, rank = base64.b64decode(fields[0], validate=True), int(fields[1])
        require(token and token not in result and rank == ordinal,
                "noncontiguous_or_duplicate_tiktoken_rank", row=ordinal, rank=rank)
        result[token] = rank
    require(len(result) == 163584 and all(bytes([byte]) in result for byte in range(256)),
            "base_vocabulary_identity")
    return result


def file_inventory(root: Path) -> dict:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            name = os.path.relpath(path, root).replace(os.sep, "/")
            files[name] = descriptor(bounded_read(child(root, name)))
    return files


def convert_once(path: Path, config: dict, model_path: Path, pattern: str,
                 added: dict[int, dict]) -> None:
    from tokenizers import AddedToken
    from transformers import PreTrainedTokenizerFast
    from transformers.convert_slow_tokenizer import TikTokenConverter

    # TikTokenConverter consumes dict keys in insertion order. Supply the full
    # ID-ordered reference vocabulary below the model bound so sparse configured
    # IDs retain their positions without modifying the converter or its backend.
    special_mapping = {spec["content"]: token_id for token_id, spec in sorted(added.items())}
    backend = TikTokenConverter(vocab_file=str(model_path), pattern=pattern,
                                add_prefix_space=False,
                                extra_special_tokens=special_mapping).converted()
    options = {key: copy.deepcopy(value) for key, value in config.items()
               if key not in ("tokenizer_class", "auto_map", "added_tokens_decoder",
                              "additional_special_tokens", "extra_special_tokens")}
    require(config["extra_special_tokens"] == {}, "unexpected_named_extra_specials")
    by_content = {spec["content"]: spec for spec in added.values()}
    for name in SPECIAL_IDS:
        options[name] = AddedToken(**by_content[config[name]])
    # Transformers v5 renamed this list. Passing the legacy list beside the
    # source's empty extra_special_tokens dict would discard the list on save.
    options["extra_special_tokens"] = [
        AddedToken(**by_content[token]) for token in config["additional_special_tokens"]
    ]
    # The frozen data policy treats literal special-token strings as ordinary text.
    # Persist it globally so production AutoTokenizer loading has the same behavior.
    options["split_special_tokens"] = True
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, **options)
    path.mkdir(parents=True)
    tokenizer.save_pretrained(path)
    saved_path = path / "tokenizer_config.json"
    saved = read_json(saved_path.read_bytes())
    require(saved.get("split_special_tokens") is True,
            "ordinary_literal_policy_not_serialized")
    saved["tokenizer_class"] = "PreTrainedTokenizerFast"
    saved.pop("auto_map", None)
    write_json(saved_path, saved)
    allowed = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
               "added_tokens.json", "chat_template.jinja"}
    require({"tokenizer.json", "tokenizer_config.json"} <= set(file_inventory(path)) <= allowed,
            "unexpected_generated_files")


def tokenizer_admission(root: Path, config: dict, ranks: dict, added: dict,
                        pattern: str, report: dict, configured_ids: set[int]) -> Any:
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    vocab = tokenizer.get_vocab()
    identity = {"class": type(tokenizer).__name__, "is_fast": tokenizer.is_fast,
                "length": len(tokenizer), "vocab_size": tokenizer.vocab_size,
                "max_id": max(vocab.values()), "trust_remote_code": False,
                "local_files_only": True, "split_special_tokens": tokenizer.split_special_tokens,
                "added_tokens": {}, "configured_added_tokens": {}, "special_ids": {}}
    report["fast_tokenizer"] = identity
    require(isinstance(tokenizer, PreTrainedTokenizerFast) and tokenizer.is_fast,
            "fast_tokenizer_required")
    require(tokenizer.split_special_tokens is True, "ordinary_literal_policy_not_reloaded")
    require(len(tokenizer) == VOCAB_BOUND
            and all(type(value) is int and 0 <= value < VOCAB_BOUND for value in vocab.values()),
            "fast_vocabulary_out_of_bounds", identity=identity)
    byte_encoder = bytes_to_unicode()
    for token_bytes, token_id in ranks.items():
        token_string = "".join(byte_encoder[byte] for byte in token_bytes)
        require(vocab.get(token_string) == token_id, "base_token_id_mismatch",
                token_bytes_hex=token_bytes.hex(), expected_id=token_id,
                observed_id=vocab.get(token_string))
    for token_id, spec in sorted(added.items()):
        observed_id = tokenizer.convert_tokens_to_ids(spec["content"])
        observation = {"content": spec["content"], "observed_id": observed_id}
        identity["added_tokens"][str(token_id)] = observation
        if token_id in configured_ids:
            identity["configured_added_tokens"][str(token_id)] = observation
    for name, expected in SPECIAL_IDS.items():
        identity["special_ids"][name + "_id"] = {
            "expected": expected, "observed": getattr(tokenizer, name + "_id"),
        }
    for token_id, spec in sorted(added.items()):
        observed_id = identity["added_tokens"][str(token_id)]["observed_id"]
        if observed_id != token_id:
            details = {"kind": "source_defined_added_token_id", "content": spec["content"],
                       "expected_id": token_id, "observed_id": observed_id,
                       "configured_token": token_id in configured_ids}
            report["first_mismatch"] = details
            raise AdmissionError("source_defined_added_token_id_mismatch", **details)
        observed = tokenizer.added_tokens_decoder.get(token_id)
        require(observed is not None and all(getattr(observed, key) == value
                for key, value in spec.items()), "added_token_properties_mismatch", id=token_id)
    require(set(tokenizer.get_added_vocab()) == {spec["content"] for spec in added.values()},
            "non_source_added_token_present")
    for name, expected in SPECIAL_IDS.items():
        require(getattr(tokenizer, name + "_id") == expected, "special_id_mismatch", name=name)
        require(getattr(tokenizer, name) == config[name], "special_string_mismatch", name=name)
    require(tokenizer.extra_special_tokens == config["additional_special_tokens"],
            "configured_extra_special_tokens_changed")
    require(tokenizer.model_max_length == config["model_max_length"]
            and tokenizer.clean_up_tokenization_spaces == config["clean_up_tokenization_spaces"]
            and tokenizer.chat_template == config["chat_template"], "saved_configuration_changed")
    backend = read_json(tokenizer.backend_tokenizer.to_str().encode("utf-8"))
    require(backend.get("normalizer") is None, "unexpected_text_normalization")
    require(backend.get("post_processor", {}).get("trim_offsets") is False,
            "whitespace_trimming_not_allowed")
    pretokenizers = backend.get("pre_tokenizer", {}).get("pretokenizers", [])
    require(len(pretokenizers) == 2
            and pretokenizers[0].get("pattern") == {"Regex": pattern}
            and pretokenizers[1].get("add_prefix_space") is False
            and pretokenizers[1].get("use_regex") is False,
            "converted_pattern_or_bytelevel_changed")
    require(tokenizer.backend_tokenizer.padding is None
            and tokenizer.backend_tokenizer.truncation is None,
            "serialized_padding_or_truncation")
    return tokenizer

class CheckedTokenizer:
    """Observe unchanged frozen tokenizer calls and compare every character and ID."""

    def __init__(self, fast: Any, reference: Any, output: Path, report: dict, ledger: Any):
        self.fast = fast
        self.reference = reference
        self.output = output
        self.report = report
        self.ledger = ledger
        self.identity: dict = {}
        self.parts: list[str] = []
        self.checked = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.fast, name)

    def __len__(self) -> int:
        return len(self.fast)

    def bind(self, identity: dict, parts: tuple[str, ...]) -> None:
        require(not self.parts, "frozen_call_sequence_incomplete", remaining=self.parts)
        self.identity, self.parts = identity, list(parts)

    def mismatch(self, code: str, text: str, expected: list, encoded: dict, **details: Any) -> None:
        record = {"kind": code, **self.identity, "text_sha256": sha256(text.encode("utf-8")),
                  "text_characters": len(text), **details}
        self.report["first_mismatch"] = record
        write_json(self.output / "first-mismatch.json", {
            **record, "text": text, "reference_ids": expected,
            "fast_ids": encoded.get("input_ids"), "fast_offsets": encoded.get("offset_mapping"),
        })
        raise AdmissionError(code, **record)

    def __call__(self, text: str, **options: Any) -> dict:
        require(bool(self.parts), "unexpected_frozen_tokenizer_call")
        part = self.parts.pop(0)
        identity = {**self.identity, "part": part}
        self.identity = identity
        require(isinstance(text, str) and len(text) <= MAX_TEXT_CHARACTERS,
                "text_exceeds_unsplit_reference_bound", identity=identity)
        require("split_special_tokens" not in options
                and options.get("add_special_tokens") is False,
                "unexpected_parity_only_encode_policy")
        key = coverage_key(identity["role"], identity["endpoint"], identity["split"])
        counts = bucket(self.report, key)
        increment(counts["attempted_texts_by_part"], part)
        raw = text.encode("utf-8")
        record = {**identity, "text_sha256": sha256(raw), "utf8_bytes": len(raw),
                  "characters": len(text), "status": "STARTED"}
        expected, encoded = [], {}
        try:
            expected = self.reference.encode(text, disallowed_special=())
            encoded = self.fast(text, **{**options, "return_offsets_mapping": True})
            ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
            record.update(reference_token_count=len(expected), fast_token_count=len(ids),
                          reference_ids_sha256=object_hash(expected),
                          fast_ids_sha256=object_hash(ids),
                          offsets_sha256=object_hash(offsets))
            if any(type(token) is not int or not 0 <= token < VOCAB_BOUND
                   for tokens in (expected, ids) for token in tokens):
                self.mismatch("emitted_id_out_of_bounds", text, expected, encoded)
            if ids != expected:
                index = next((index for index, pair in enumerate(zip(expected, ids))
                              if pair[0] != pair[1]), min(len(expected), len(ids)))
                self.mismatch("token_id_parity", text, expected, encoded,
                              first_token_index=index,
                              reference_window=expected[max(0, index - 8):index + 9],
                              fast_window=ids[max(0, index - 8):index + 9])
            pieces = [self.reference.decode_single_token_bytes(token) for token in expected]
            concatenated = b"".join(pieces)
            if concatenated != raw or concatenated.decode("utf-8", errors="strict") != text:
                self.mismatch("reference_full_byte_reconstruction", text, expected, encoded,
                              concatenated_sha256=sha256(concatenated))
            decoded = self.fast.decode(ids, skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False)
            if decoded != text or decoded.encode("utf-8") != raw:
                self.mismatch("fast_full_text_reconstruction", text, expected, encoded,
                              decoded_sha256=sha256(decoded.encode("utf-8")))
            if len(offsets) != len(pieces):
                self.mismatch("offset_cardinality", text, expected, encoded)
            boundaries = [0]
            for character in text:
                boundaries.append(boundaries[-1] + len(character.encode("utf-8")))
            cursor, covered_end = 0, 0
            for index, (piece, offset) in enumerate(zip(pieces, offsets, strict=True)):
                end_byte = cursor + len(piece)
                require(bool(piece), "empty_reference_token_bytes", token_index=index)
                # A token may contain only part of a UTF-8 character. Its exact
                # character span is the smallest interval intersecting its bytes.
                start = bisect.bisect_right(boundaries, cursor) - 1
                end = bisect.bisect_left(boundaries, end_byte)
                exact = [start, end]
                if (not isinstance(offset, (list, tuple)) or len(offset) != 2
                        or any(type(value) is not int for value in offset)
                        or list(offset) != exact or raw[cursor:end_byte] != piece
                        or not 0 <= start < end <= len(text) or start > covered_end):
                    self.mismatch("exact_offset_parity", text, expected, encoded,
                                  token_index=index, reference_id=expected[index],
                                  token_bytes_hex=piece.hex(), byte_interval=[cursor, end_byte],
                                  expected_character_interval=exact, observed_offset=offset,
                                  source_excerpt=text[max(0, start - 24):min(len(text), end + 24)])
                covered_end = max(covered_end, end)
                cursor = end_byte
            if cursor != len(raw) or covered_end != len(text):
                self.mismatch("source_positions_not_fully_covered", text, expected, encoded,
                              covered_bytes=cursor, covered_characters=covered_end)
            record["status"] = "PASS"
            increment(counts["passed_texts_by_part"], part)
            counts["reference_tokens"] += len(expected)
            counts["verified_utf8_bytes"] += len(raw)
            counts["verified_characters"] += len(text)
            self.checked += 1
        except BaseException as error:
            record["status"] = "FAILED"
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            if self.report["first_mismatch"] is None:
                self.report["first_mismatch"] = {**record, "kind": "tokenization_exception"}
                write_json(self.output / "first-mismatch.json", {
                    **self.report["first_mismatch"], "text": text,
                    "reference_ids": expected, "fast_encoding": dict(encoded),
                })
            raise
        finally:
            self.ledger.write(compact(record) + "\n")
            if record["status"] != "PASS" or self.checked % 500 == 0:
                self.ledger.flush()
                checkpoint(self.output, self.report)
        return encoded


def _check_corpus(training: list, endpoints: dict, dataset: Any, loglik: Any,
                  fast: Any, reference: Any, output: Path, report: dict) -> None:
    with (output / "text-checks.jsonl").open("x", encoding="utf-8", newline="\n") as ledger:
        checked = CheckedTokenizer(fast, reference, output, report, ledger)
        for row in training:
            identity = row["_probe_identity"]
            checked.bind(identity, ("prompt", "response", "document"))
            checked(row["prompt"], add_special_tokens=False, truncation=False, padding=False)
            checked(row["response"], add_special_tokens=False, truncation=False, padding=False)
            encoded = dataset._encode_document(checked, row)
            require(encoded["input_ids"][-1] == SPECIAL_IDS["eos_token"], "native_document_eos")
            counts = bucket(report, coverage_key("training", row["benchmark"], "train"))
            counts["completed_rows"] += 1
            counts["completed_requests"] += 1
        for endpoint, rows in endpoints.items():
            checkpoint(output, report, "parity:" + endpoint)
            for row in rows:
                identity = row["_probe_identity"]
                counts = bucket(report, coverage_key(identity["role"], endpoint, identity["split"]))
                requests, sizes = loglik._requests([row])
                require(sizes == [len(requests)], "request_count_changed")
                for request_index, (context, continuation) in enumerate(requests):
                    request_identity = {**identity, "request_index": request_index}
                    if row["metric"] == "nll":
                        checked.bind(request_identity, ("prompt", "response", "document"))
                        checked(context, add_special_tokens=False, truncation=False, padding=False)
                        checked(continuation, add_special_tokens=False,
                                truncation=False, padding=False)
                        dataset._encode_document(
                            checked, {"prompt": context, "response": continuation})
                    else:
                        checked.bind(request_identity, ("request_context", "request_continuation",
                                                        "effective_context", "joint_document"))
                        checked(context, add_special_tokens=False, truncation=False, padding=False)
                        checked(continuation, add_special_tokens=False,
                                truncation=False, padding=False)
                        # This executes the frozen trailing-space transfer and joint
                        # slicing. No reconstruction of its request strings is used.
                        loglik.encode_pair(checked, context, continuation)
                    require(not checked.parts, "request_text_coverage_incomplete")
                    counts["completed_requests"] += 1
                counts["completed_rows"] += 1
        require(not checked.parts, "final_text_coverage_incomplete")
    for key, counts in report["coverage"].items():
        require(counts["required_rows"] == counts["completed_rows"]
                and counts["required_requests"] == counts["completed_requests"]
                and counts["required_texts_by_part"] == counts["attempted_texts_by_part"]
                == counts["passed_texts_by_part"], "full_corpus_coverage_required", group=key)


def _report(operation: str) -> dict:
    return {"schema": "p3-moonlight-tokenizer-v1", "operation": operation,
            "profile_id": PROFILE_ID, "model_id": MODEL_ID, "revision": REVISION,
            "status": "STARTED", "packages": {}, "installed_sources": {},
            "frozen_inputs": {}, "frozen_functions": {}, "coverage": {},
            "first_mismatch": None, "weights_loaded": False,
            "trust_remote_code": False, "remote_python_executed": False}


def _installed_provenance(report: dict) -> None:
    package_provenance(report)
    import tiktoken
    from tiktoken.load import load_tiktoken_bpe
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    from transformers.convert_slow_tokenizer import TikTokenConverter

    report["implementation"] = descriptor(Path(__file__).read_bytes())
    report["installed_sources"] = {
        obj.__module__ + "." + obj.__name__: descriptor(Path(inspect.getfile(obj)).read_bytes())
        for obj in (AutoTokenizer, PreTrainedTokenizerFast, TikTokenConverter,
                    tiktoken.Encoding, load_tiktoken_bpe)
    }
    report["converter_source"] = {
        "module": "transformers.convert_slow_tokenizer",
        **descriptor(Path(inspect.getfile(TikTokenConverter)).read_bytes()),
        "class_source_sha256": sha256(inspect.getsource(TikTokenConverter).encode("utf-8")),
        "patches_applied": False,
    }


def _admit_sources(source_dir: Path) -> dict:
    root = Path(source_dir)
    require(root.name == REVISION, "source_revision_mismatch")
    paths, files, content = {}, {}, {}
    for name, pin in HUB_FILES.items():
        path = child(root, name)
        raw = bounded_read(path, pin["limit_bytes"])
        if pin["identity_kind"] == "lfs_sha256":
            identity = sha256(raw)
        else:
            identity = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        require(identity == pin["identity"], "source_identity_mismatch", file=name)
        paths[name], content[name] = path, raw
        files[name] = {**descriptor(raw), **pin, "verified": True}
    config = read_json(content["tokenizer_config.json"])
    pattern, reserved = reference_pattern(content["tokenization_moonshot.py"])
    ranks = mergeable_ranks(content["tiktoken.model"])
    added = {int(key): spec for key, spec in config["added_tokens_decoder"].items()}
    require(len(added) == 9 and len({spec["content"] for spec in added.values()}) == len(added)
            and all(len(ranks) <= token_id < VOCAB_BOUND for token_id in added),
            "configured_added_token_inventory")
    require(all(spec == {"content": spec["content"], "lstrip": False, "normalized": False,
                         "rstrip": False, "single_word": False, "special": True}
                for spec in added.values()), "configured_added_token_properties")
    for name, token_id in SPECIAL_IDS.items():
        require(config[name] == added[token_id]["content"], "configured_special_binding", name=name)
    specials = {added.get(token_id, {"content": f"<|reserved_token_{token_id}|>"})["content"]:
                token_id for token_id in range(len(ranks), len(ranks) + reserved + 2)}
    model_added = {
        token_id: copy.deepcopy(added.get(token_id, {
            "content": f"<|reserved_token_{token_id}|>", "lstrip": False, "rstrip": False,
            "normalized": False, "single_word": False, "special": True,
        })) for token_id in range(len(ranks), VOCAB_BOUND)
    }
    require({spec["content"]: token_id for token_id, spec in model_added.items()}
            == {token: token_id for token, token_id in specials.items() if token_id < VOCAB_BOUND},
            "source_defined_special_mapping")
    return {"paths": paths, "files": files, "config": config, "pattern": pattern,
            "ranks": ranks, "added": model_added, "configured_ids": set(added),
            "specials": specials}


def _reference(source: dict) -> Any:
    import tiktoken

    return tiktoken.Encoding(name=f"moonlight-{REVISION}", pat_str=source["pattern"],
                             mergeable_ranks=source["ranks"], special_tokens=source["specials"])


def _source_provenance(source: dict, report: dict) -> None:
    report["source_files"] = source["files"]
    report["reference"] = {
        "pattern": source["pattern"], "pattern_sha256": sha256(source["pattern"].encode("utf-8")),
        "base_vocabulary_size": len(source["ranks"]), "n_vocab": VOCAB_BOUND + 2,
        "special_mapping": source["specials"],
        "special_mapping_sha256": object_hash(source["specials"]),
        "configured_added_tokens": source["config"]["added_tokens_decoder"],
        "encode_options": {"disallowed_special": [], "allowed_special": []},
        "reference_only_out_of_model_range": {
            token: token_id for token, token_id in source["specials"].items()
            if token_id >= VOCAB_BOUND},
    }
    model_specials = {spec["content"]: token_id for token_id, spec in source["added"].items()}
    report["conversion"] = {
        "special_mapping": model_specials, "special_mapping_sha256": object_hash(model_specials),
        "configured_special_count": len(source["configured_ids"]),
        "source_defined_implicit_count": len(model_specials) - len(source["configured_ids"]),
        "reference_only_count": len(source["specials"]) - len(model_specials),
        "split_special_tokens": True, "backend_patches": False,
    }


def _admit_converted_tokenizer(tokenizer_dir: Path) -> tuple[Any, dict]:
    """Admit the independent v1 byte pins before the final model profile exists.

    This private seam returns the same tokenizer metadata as data.load_tokenizer.
    It does not read an attention hash or publish a provisional model profile.
    """
    from transformers import AutoTokenizer

    root = Path(tokenizer_dir)
    require(root.name == REVISION, "tokenizer_revision_mismatch")
    require(file_inventory(root) == GENERATED_FILES, "generated_tokenizer_pin_mismatch")
    tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    require(tokenizer.is_fast is True and tokenizer.split_special_tokens is True,
            "ordinary_literal_fast_tokenizer_required")
    require(len(tokenizer) == VOCAB_BOUND and all(
        getattr(tokenizer, name + "_id") == token_id for name, token_id in SPECIAL_IDS.items()),
        "tokenizer_vocabulary_or_special_ids_mismatch")
    pin = {"repo_id": MODEL_ID, "revision": REVISION, "test_only": False,
           "files": copy.deepcopy(GENERATED_FILES)}
    metadata = {
        "profile_id": PROFILE_ID, "repo_id": MODEL_ID, "revision": REVISION,
        "test_only": False, "pin": pin,
        "files": {name: {"path": name, **spec} for name, spec in GENERATED_FILES.items()},
        "tokenizer_class": type(tokenizer).__name__, "is_fast": True,
        "vocabulary_size": len(tokenizer), "eos_id": tokenizer.eos_token_id,
        "bos_id": tokenizer.bos_token_id, "pad_id": tokenizer.pad_token_id,
        "library_versions": {name: importlib.metadata.version(name)
                             for name in ("transformers", "tokenizers")},
        "weights_loaded": False, "add_special_tokens": False, "prepend_bos": False,
        "append_native_eos_per_document": True, "split_special_tokens": True,
    }
    return tokenizer, metadata


def convert(*, source_dir: Path, output: Path, report_path: Path | None = None) -> dict:
    """Convert pinned local sources twice, admit exact v1 bytes and publish once.

    ``output`` must be a new directory named with the immutable model revision.
    A report, when requested, lives outside that tokenizer-only directory.
    """
    output = Path(output)
    require(output.name == REVISION, "output_revision_mismatch")
    require(not output.exists(), "existing_output_preserved")
    if report_path is not None:
        report_path = Path(report_path)
        require(not report_path.exists(), "existing_report_preserved")
        base, target = os.path.realpath(output), os.path.realpath(report_path)
        try:
            within_output = os.path.commonpath((base, target)) == base
        except ValueError:
            within_output = False
        require(not within_output, "report_inside_tokenizer_directory")
        report_path.parent.mkdir(parents=True, exist_ok=True)
    report = _report("convert")
    try:
        source = _admit_sources(Path(source_dir))
        _installed_provenance(report)
        _source_provenance(source, report)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".moonlight-tokenizer-", dir=output.parent) as work:
            roots = [Path(work) / f"generation-{index}" / REVISION for index in (1, 2)]
            for root in roots:
                convert_once(root, source["config"], source["paths"]["tiktoken.model"],
                             source["pattern"], source["added"])
            inventories = [file_inventory(root) for root in roots]
            require(inventories[0] == inventories[1], "nondeterministic_generated_file_digests")
            for name in inventories[0]:
                require((roots[0] / name).read_bytes() == (roots[1] / name).read_bytes(),
                        "nondeterministic_generated_file_bytes", file=name)
            require(inventories[0] == GENERATED_FILES, "generated_tokenizer_pin_mismatch")
            tokenizer_admission(roots[0], source["config"], source["ranks"], source["added"],
                                source["pattern"], report, source["configured_ids"])
            report.update(files=inventories[0], two_generations_byte_equal=True,
                          generated_files={f"generation-{index}/{REVISION}": inventory
                                           for index, inventory in enumerate(inventories, 1)})
            # mkdir is exclusive; a racing producer cannot have its output overwritten.
            output.mkdir()
            for name in inventories[0]:
                with (roots[0] / name).open("rb") as incoming:
                    with (output / name).open("xb") as outgoing:
                        shutil.copyfileobj(incoming, outgoing)
            report["status"] = "PASS"
    except Exception as error:
        report.update(status="FAILED", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        if report_path is not None:
            with report_path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(compact(report) + "\n")
    return report


def parity(*, source_dir: Path, tokenizer_dir: Path, repo: Path, output: Path) -> dict:
    """Run the admitted exact-ID/offset checker on all frozen training/dev/eval texts."""
    output = Path(output)
    require(not output.exists(), "existing_output_preserved")
    output.mkdir(parents=True)
    report = _report("parity")
    try:
        source = _admit_sources(Path(source_dir))
        _installed_provenance(report)
        _source_provenance(source, report)
        fast, metadata = _admit_converted_tokenizer(Path(tokenizer_dir))
        report["tokenizer"] = metadata
        tokenizer_admission(Path(tokenizer_dir), source["config"], source["ranks"],
                            source["added"], source["pattern"], report, source["configured_ids"])
        training, endpoints, dataset, loglik, bindings = admit_corpus(Path(repo), output, report)
        report["corpus_bindings_sha256"] = object_hash(bindings)
        _check_corpus(training, endpoints, dataset, loglik, fast, _reference(source),
                      output, report)
        texts = sum(sum(group["passed_texts_by_part"].values())
                    for group in report["coverage"].values())
        require(texts == 348941, "frozen_text_check_count_mismatch", observed=texts)
        report.update(status="PASS", full_admission=True, exact_id_and_offset_text_checks=texts,
                      text_checks={"path": "text-checks.jsonl",
                                   **_stream_descriptor(output / "text-checks.jsonl")})
    except Exception as error:
        report.update(status="FAILED", full_admission=False,
                      error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        checkpoint(output, report)
        write_json(output / "report.json", report)
    return report


def _stream_descriptor(path: Path) -> dict:
    digest, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for raw in iter(lambda: stream.read(1024 * 1024), b""):
            count += len(raw)
            digest.update(raw)
    return {"bytes": count, "sha256": digest.hexdigest()}


class _Parser(argparse.ArgumentParser):
    def _print_message(self, message: str, file: TextIO | None = None) -> None:
        from rich.console import Console

        if message:
            Console(file=file or sys.stdout, highlight=False, markup=False).print(message, end="")


def main(argv: list[str] | None = None) -> int:
    from rich.console import Console

    parser = _Parser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)
    conversion = commands.add_parser("convert", help="Convert admitted local source bytes twice")
    conversion.add_argument("--source-dir", required=True, type=Path)
    conversion.add_argument("--output", required=True, type=Path)
    conversion.add_argument("--report", required=True, type=Path)
    comparison = commands.add_parser("parity", help="Check all frozen text IDs and exact offsets")
    comparison.add_argument("--source-dir", required=True, type=Path)
    comparison.add_argument("--tokenizer-dir", required=True, type=Path)
    comparison.add_argument("--repo", required=True, type=Path)
    comparison.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    console = Console(highlight=False, markup=False)
    try:
        if args.command == "convert":
            result = convert(source_dir=args.source_dir, output=args.output,
                             report_path=args.report)
        else:
            result = parity(source_dir=args.source_dir, tokenizer_dir=args.tokenizer_dir,
                            repo=args.repo, output=args.output)
    except Exception as error:
        console.print(f"MOONLIGHT_TOKENIZER_ADMISSION_FAILED:{type(error).__name__}:{error}")
        return 1
    console.print({"status": result["status"], "profile_id": PROFILE_ID,
                   "revision": REVISION, "operation": args.command})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
