"""Frozen Stage2 source clustering and paired response-only training packs.

Data admission is not runtime, statistical, numerical or quality admission. Only
public problems enter lexical comparison. Source programs are parsed for AST
fingerprints, never executed. The pinned MATH adapter executes trusted string
scorer assets only on Kaggle. No model weights are loaded by either entry point.
"""

from __future__ import annotations

import ast
import errno
import hashlib
import json
import math
import os
import platform
import re
import shutil
import struct
import tempfile
import unicodedata
import uuid
from collections import Counter, defaultdict
from importlib.metadata import version
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterator

from benchmarks.harness import p3_realdata_model as native
from benchmarks.harness import p3_stage2_apps as apps
from benchmarks.harness import p3_stage2_core_data as core
from benchmarks.harness import p3_stage2_math as math_data
from benchmarks.harness.p3_realdata_corpus import _compact, _file_hash, _inside, _text_hash

POLICY = "p3-stage2-dataset-v1"
PACKING_POLICY = "p3-stage2-training-pack-v1"
_ORDER_PREFIX = "p3-stage2-data-order-v1\0"
_BENCHMARKS = ("gsm8k", "humaneval", "apps", "math")
_DOMAIN_TOKENS = 131072
_WINDOW_TOKENS = 4096
_HISTORICAL_COUNTS = {"gsm8k/train": 40, "gsm8k/test": 20, "humaneval/test": 63}
_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class AdmissionError(ValueError):
    """No successful dataset/pack was published; reports contain no source text."""

    def __init__(self, reason: str, report_path: Path | None = None):
        self.reason = reason
        self.report_path = report_path
        super().__init__(f"STAGE2_DATASET_ADMISSION_FAILED:{reason}; report={report_path}")


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _unique(pairs: list[tuple[str, Any]]) -> dict:
    value = {}
    for key, item in pairs:
        _require(key not in value, "duplicate_json_key")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError("nonfinite_json_number")


def _json(text: str) -> Any:
    value = json.loads(text, object_pairs_hook=_unique, parse_constant=_reject_constant)
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, float):
            _require(math.isfinite(item), "nonfinite_json_number")
        elif isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return value


def _jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = _json(line)
            _require(isinstance(value, dict), "jsonl_row_not_object")
            yield value


def _relative(root: Path, name: Any) -> Path:
    _require(isinstance(name, str) and bool(name) and "\\" not in name
             and ":" not in name and not Path(name).is_absolute()
             and all(part not in {"", ".", ".."} for part in name.split("/")),
             "invalid_artifact_relative_path")
    return _inside(root, root / name)


def _artifact(root: Path, descriptor: dict) -> Path:
    _require(isinstance(descriptor, dict), "artifact_descriptor_not_object")
    path = _relative(root, descriptor.get("path"))
    size, digest = _file_hash(path)
    _require(type(descriptor.get("bytes")) is int and descriptor["bytes"] == size
             and descriptor.get("sha256") == digest, "artifact_bytes_or_hash_mismatch")
    if path.suffix == ".jsonl":
        observed = sum(1 for _ in _jsonl(path))
        _require(type(descriptor.get("rows")) is int and descriptor["rows"] == observed,
                 "artifact_row_count_mismatch")
    elif descriptor.get("format") == "json_array":
        rows = _json(path.read_text(encoding="utf-8"))
        _require(isinstance(rows, list) and type(descriptor.get("rows")) is int
                 and len(rows) == descriptor["rows"], "artifact_array_count_mismatch")
    return path


def _read_manifest(root: Path) -> dict:
    manifest = _json((root / "manifest.json").read_text(encoding="utf-8"))
    _require(isinstance(manifest, dict), "manifest_not_object")
    keys = [key for key in ("manifest_payload_sha256", "manifest_sha256") if key in manifest]
    _require(len(keys) == 1, "manifest_hash_field_missing_or_ambiguous")
    key = keys[0]
    payload = {name: value for name, value in manifest.items() if name != key}
    _require(manifest[key] == _text_hash(_compact(payload)), "manifest_payload_hash_mismatch")
    files = manifest.get("files")
    _require(isinstance(files, dict), "manifest_files_missing")
    for descriptor in files.values():
        _artifact(root, descriptor)
    return manifest


def _compare_jsonl(original: Path, reconstructed: Path) -> None:
    absent = object()
    for left, right in zip_longest(_jsonl(original), _jsonl(reconstructed), fillvalue=absent):
        _require(left is not absent and right is not absent and left == right,
                 "source_reconstruction_row_or_coverage_mismatch")


def _joined(root: Path, manifest: dict) -> Iterator[tuple[dict, dict, dict]]:
    paths = [_artifact(root, manifest["files"][name])
             for name in ("generation", "scoring", "audit")]
    absent = object()
    seen = set()
    for rows in zip_longest(*(_jsonl(path) for path in paths), fillvalue=absent):
        _require(all(row is not absent for row in rows), "endpoint_identity_join_count_mismatch")
        identities = [(row.get("benchmark"), row.get("task_id"), row.get("source_row_sha256"))
                      for row in rows]
        identity = identities[0]
        _require(all(value == identity for value in identities) and identity not in seen
                 and all(isinstance(value, str) and value for value in identity),
                 "endpoint_identity_join_mismatch_or_duplicate")
        seen.add(identity)
        refs = [row["source_ref"] for row in rows if "source_ref" in row]
        _require(refs and all(ref == refs[0] for ref in refs), "endpoint_source_ref_join_mismatch")
        yield rows


def _write_jsonl(path: Path, records: Any) -> dict:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        count = 0
        for row in records:
            stream.write(_compact(row) + "\n")
            count += 1
    size, digest = _file_hash(path)
    return {"path": path.name, "bytes": size, "sha256": digest, "rows": count}


def _write_manifest(path: Path, manifest: dict) -> None:
    _require("manifest_payload_sha256" not in manifest, "manifest_already_hashed")
    manifest["manifest_payload_sha256"] = _text_hash(_compact(manifest))
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_compact(manifest) + "\n")


def _descriptor(path: Path, relative: str) -> dict:
    size, digest = _file_hash(path)
    return {"path": relative, "bytes": size, "sha256": digest}


def _fresh_root(output_dir: Path) -> Path:
    root = Path(os.path.realpath(output_dir))
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise AdmissionError("existing_output_preserved") from None
    return root


def _failure(root: Path, error: Exception) -> AdmissionError:
    # Third-party exceptions may include problem/test text. Preserve only their type.
    reason = str(error) if type(error) is ValueError else f"prerequisite:{type(error).__name__}"
    if not re.fullmatch(r"[a-z0-9_:]+", reason):
        reason = f"admission:{type(error).__name__}"
    path = root / f"admission-failure-{uuid.uuid4().hex}.json"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_compact({"schema": POLICY, "status": "ADMISSION_FAILED",
                               "reason": reason, "successful_denominator": None,
                               "quality_non_degradation_admitted": False}) + "\n")
    return AdmissionError(reason, path)


def _publish(stage: Path, root: Path, names: list[str]) -> None:
    _require(names and names[-1] == "manifest.json", "root_manifest_must_publish_last")
    for name in names:
        destination = _relative(root, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # A concurrent writer or interrupted prior publication cannot be overwritten.
        os.link(_relative(stage, name), destination)


def _stage_input(source: Path, destination: Path) -> None:
    """Reuse bytes without modifying evidence, including read-only dataset mounts."""
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        with source.open("rb") as incoming, destination.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)


def _seed_sources(original: Path, reconstructed: Path) -> None:
    reconstructed.mkdir(parents=True)
    directory = original / "sources"
    if not directory.is_dir():
        return
    target = reconstructed / "sources"
    target.mkdir()
    for source in sorted(directory.iterdir(), key=lambda item: item.name):
        if source.is_file():
            source = _inside(original, source)
            _stage_input(source, _relative(target, source.name))


def _compare_admission(original: Path, reconstructed: Path, expected: dict) -> dict:
    observed = _read_manifest(original)
    _require(observed == expected, "source_reconstruction_manifest_mismatch")
    for descriptor in observed["files"].values():
        name = descriptor["path"]
        if name.endswith(".jsonl"):
            _compare_jsonl(_relative(original, name), _relative(reconstructed, name))
    return observed


def _admit_inputs(*, core_root: Path, apps_root: Path, math_root: Path,
                  exposure_audit_path: Path, cache_dir: Path, stage: Path) -> dict:
    """Reapply the existing complete adapters, not a parallel admission convention."""
    expected = {}
    for name, original in (("core", core_root), ("apps", apps_root), ("math", math_root)):
        target = stage / f"reconstructed-{name}"
        _seed_sources(original, target)
        if name == "core":
            expected[name] = core.build_core_sources(
                cache_dir=cache_dir, output_dir=target, exposure_audit_path=exposure_audit_path)
        elif name == "apps":
            expected[name] = apps.build_evaluation(cache_dir=cache_dir, output_dir=target)
        else:
            expected[name] = math_data.build_evaluation(cache_dir=cache_dir, output_dir=target)
        _compare_admission(original, target, expected[name])
    endpoints = {}
    for benchmark in _BENCHMARKS:
        owner = "core" if benchmark in core.GENERATION_POLICIES else benchmark
        original = core_root / benchmark if owner == "core" else (
            apps_root if owner == "apps" else math_root)
        rebuilt = stage / f"reconstructed-{owner}"
        wanted = expected[owner]
        if owner == "core":
            rebuilt /= benchmark
            wanted = wanted["evaluations"][benchmark]
        endpoints[benchmark] = _compare_admission(original, rebuilt, wanted)
    # Recover exposed public records from the exact historical readers and row joins.
    rebuilt_core = stage / "reconstructed-core"
    loaded = {core._source_key(source): core._load_legacy(source, cache_dir, rebuilt_core)
              for source in core.SOURCES}
    exposure, consumed = core._verify_exposure(exposure_audit_path, loaded)
    _require(exposure == expected["core"]["exposure"], "exposure_reconstruction_mismatch")
    _require(exposure["consumed_counts"] == _HISTORICAL_COUNTS,
             "historical_exposure_count_mismatch")
    historical = []
    for source in core.SOURCES:
        key = core._source_key(source)
        for identity, raw in loaded[key]:
            if identity["row_key"] in consumed[key]:
                historical.append(_node(core._audit_record(source, identity, raw),
                                        role="historical"))
    _require(len(historical) == 123, "historical_exposure_coverage_mismatch")
    pool_nodes = _auxiliary_nodes(cache_dir, stage / "reconstructed-math")
    inputs = {}
    for name, root in (("core", core_root), ("apps", apps_root), ("math", math_root)):
        inputs[name] = {"manifest": _descriptor(root / "manifest.json", "manifest.json"),
                        "payload_sha256": expected[name].get("manifest_payload_sha256",
                                                               expected[name].get("manifest_sha256")),
                        "files": expected[name]["files"],
                        "source_admission_manifest": expected[name]}
    inputs["exposure_audit"] = _descriptor(exposure_audit_path, exposure_audit_path.name)
    return {"endpoints": endpoints, "core": expected["core"], "historical": historical,
            "pool_nodes": pool_nodes, "exposure": exposure, "inputs": inputs}


def _auxiliary_nodes(cache_dir: Path, rebuilt_math: Path) -> list[dict]:
    """PRM row references are audit identities; raw filename claims are not."""
    nodes = []
    for source in math_data.CROSSWALK_SOURCES:
        path = math_data._dataset_source_file(source, cache_dir, rebuilt_math)
        for ordinal, row in enumerate(_jsonl(path)):
            source_ref = math_data._source_ref(source, ordinal)
            node = _node({
                "benchmark": "math", "task_id": f"math-auxiliary/{source_ref}",
                "source_ref": source_ref, "source_row_sha256": _text_hash(_compact(row)),
                "problem_text": row["problem"],
            }, role="audit_only")
            node["auxiliary_provenance"] = {
                "raw_original_id": row["unique_id"], "raw_subject": row["subject"],
                "raw_level": row["level"], "source_artifact_sha256": source["sha256"],
                "identity_authority": "immutable_source_ref_not_raw_original_id_or_prefix",
                "source_use": "overlap_audit_only_never_SFT_or_evaluation_membership",
            }
            nodes.append(node)
    _require(len(nodes) == math_data.ORIGINAL_POOL_ROWS,
             "math_auxiliary_pool_coverage_mismatch")
    return nodes


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(unicodedata.normalize("NFKC", text))


class _WithoutDocstrings(ast.NodeTransformer):
    def _body(self, node: Any) -> Any:
        self.generic_visit(node)
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:]
        return node

    # Names are prescribed by ast.NodeTransformer's dispatch protocol.
    visit_Module = _body  # noqa: N815
    visit_FunctionDef = _body  # noqa: N815
    visit_AsyncFunctionDef = _body  # noqa: N815
    visit_ClassDef = _body  # noqa: N815


def _ast_fingerprint(program: str) -> dict:
    if not program.strip():
        return {"status": "empty_complete_reference_program", "sha256": None}
    try:
        tree = _WithoutDocstrings().visit(ast.parse(program))
        if not tree.body:
            return {"status": "empty_complete_reference_program", "sha256": None}
        representation = ast.dump(tree, annotate_fields=True, include_attributes=False)
    except (SyntaxError, ValueError, RecursionError):
        return {"status": "unparseable_complete_program", "sha256": None}
    return {"status": "present_complete_reference_program", "sha256": _text_hash(representation),
            "policy": "python_ast_dump_no_positions_no_docstrings_identifiers_unchanged"}


def _node(audit: dict, *, role: str) -> dict:
    common = {key: audit.get(key) for key in (
        "benchmark", "task_id", "source_ref", "source_row_sha256")}
    _require(all(isinstance(value, str) and value for value in common.values()),
             "overlap_node_identity_missing")
    _require(bool(_SHA256.fullmatch(common["source_row_sha256"])), "invalid_source_row_sha256")
    problem = audit.get("problem_text")
    _require(isinstance(problem, str), "public_problem_text_missing")
    fingerprints: dict[str, set[str]] = defaultdict(set)

    def add_text(family: str, text: Any) -> None:
        if isinstance(text, str) and text.strip():
            fingerprints[f"{family}_raw"].add(_text_hash(text))
            normalized = _normalized(text)
            if normalized:
                fingerprints[f"{family}_normalized"].add(_text_hash(normalized))

    def add_digest(family: str, digest: Any) -> None:
        if isinstance(digest, str) and _SHA256.fullmatch(digest):
            fingerprints[family].add(digest)

    add_text("public_problem", problem)
    raw = audit.get("raw_source", {})
    benchmark = common["benchmark"]
    ast_record = {"status": "not_a_reference_program_benchmark", "sha256": None}
    if benchmark in {"humaneval", "mbpp"}:
        program = (raw.get("prompt", "") + raw.get("canonical_solution", "")
                   if benchmark == "humaneval" else raw.get("code", ""))
        add_text("whole_reference", program)
        ast_record = _ast_fingerprint(program)
        add_digest("whole_reference_ast", ast_record["sha256"])
        if benchmark == "humaneval":
            suite = raw.get("test", "")
        else:
            pieces = [raw.get("test_setup_code", ""), *raw.get("test_list", []),
                      *raw.get("challenge_test_list", [])]
            suite = "\n".join(piece for piece in pieces if piece)
        add_text("whole_tests", suite)
    elif benchmark == "apps":
        ast_record = {"status": "unavailable_apps_complete_program_ast_not_supplied",
                      "sha256": None}
        provenance = audit.get("solutions_provenance", {})
        empty = _text_hash("")
        raw_hashes = provenance.get("code_sha256", [])
        normalized_hashes = provenance.get("code_normalized_sha256", [])
        _require(len(raw_hashes) == len(normalized_hashes), "reference_hash_pair_coverage_mismatch")
        for raw_digest, normalized_digest in zip(raw_hashes, normalized_hashes):
            if raw_digest != empty and normalized_digest != empty:
                add_digest("whole_reference_raw", raw_digest)
                add_digest("whole_reference_normalized", normalized_digest)
        url = audit.get("source_url_provenance", {}).get("canonical")
        if isinstance(url, str) and url:
            add_digest("canonical_source_url", _text_hash(url))
        add_digest("whole_tests_raw", audit.get("input_output_raw_sha256"))
        add_digest("whole_test_cases_json", audit.get("test_cases_sha256"))
    return {**common, "node_id": _text_hash(role + "\0" + common["source_ref"]),
            "role": role, "problem_text": problem, "ast": ast_record,
            "fingerprints": {key: sorted(value) for key, value in sorted(fingerprints.items())}}


def _near_edges(nodes: list[dict], masked: bool) -> Iterator[tuple[int, int, dict]]:
    """Complete set-similarity join using globally ordered prefixes and exact verification."""
    grams = []
    frequency: Counter = Counter()
    for node in nodes:
        tokens = _tokens(node["problem_text"])
        if masked:
            tokens = ["<NUM>" if token.isnumeric() else token for token in tokens]
        values = {tuple(tokens[index:index + 5]) for index in range(len(tokens) - 4)}
        grams.append(values)
        frequency.update(values)
    inverted: dict[tuple[str, ...], list[int]] = defaultdict(list)
    order = sorted(
        range(len(nodes)), key=lambda index: (len(grams[index]), nodes[index]["node_id"])
    )
    for right in order:
        values = grams[right]
        size = len(values)
        if not size:
            continue
        prefix_length = size - (4 * size + 4) // 5 + 1
        prefix = sorted(values, key=lambda gram: (frequency[gram], gram))[:prefix_length]
        candidates = set()
        for gram in prefix:
            for left in inverted[gram]:
                if 5 * len(grams[left]) >= 4 * size:
                    candidates.add(left)
        for left in sorted(candidates, key=lambda index: nodes[index]["node_id"]):
            intersection = len(grams[left] & values)
            union = len(grams[left]) + size - intersection
            if 5 * intersection >= 4 * union:
                yield min(left, right), max(left, right), {
                    "kind": "numeric_masked_token5_jaccard" if masked else "token5_jaccard",
                    "intersection": intersection, "union": union,
                    "left_gram_count": len(grams[min(left, right)]),
                    "right_gram_count": len(grams[max(left, right)]),
                }
        for gram in prefix:
            inverted[gram].append(right)


def _cluster(nodes: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    _require(len({node["node_id"] for node in nodes}) == len(nodes), "duplicate_overlap_node")
    parent = list(range(len(nodes)))
    edges = []

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def edge(left: int, right: int, evidence: dict) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)
        pair = sorted((nodes[left]["node_id"], nodes[right]["node_id"]))
        if nodes[left]["node_id"] != pair[0] and "left_gram_count" in evidence:
            evidence = {**evidence, "left_gram_count": evidence["right_gram_count"],
                        "right_gram_count": evidence["left_gram_count"]}
        edges.append({"left_node_id": pair[0], "right_node_id": pair[1], **evidence})

    first = {}
    for index in sorted(range(len(nodes)), key=lambda item: nodes[item]["node_id"]):
        for family, digests in nodes[index]["fingerprints"].items():
            for digest in digests:
                key = family, digest
                if key in first:
                    edge(first[key], index, {"kind": "exact_fingerprint", "family": family,
                                              "sha256": digest})
                else:
                    first[key] = index
    for masked in (False, True):
        for left, right, evidence in _near_edges(nodes, masked):
            edge(left, right, evidence)
    groups: dict[int, list[dict]] = defaultdict(list)
    for index, node in enumerate(nodes):
        groups[find(index)].append(node)
    clusters, decisions = [], []
    for members in groups.values():
        members.sort(key=lambda node: node["node_id"])
        cluster_id = _text_hash(_compact([node["node_id"] for node in members]))
        historical = any(node["role"] == "historical" for node in members)
        evaluation = [node for node in members if node["role"] == "evaluation"]
        benchmarks = sorted({node["benchmark"] for node in evaluation})
        winners = {}
        if not historical:
            for benchmark in benchmarks:
                candidates = [node for node in evaluation if node["benchmark"] == benchmark]
                winner = min(candidates, key=lambda node: (
                    _text_hash(benchmark + "\0" + node["task_id"]), node["source_ref"]))
                winners[benchmark] = winner["node_id"]
        training = [node for node in members if node["role"] == "training"]
        train_winner = (min(training, key=lambda node: node["source_ref"])["node_id"]
                        if training and not historical and not evaluation else None)
        summary = [{key: node[key] for key in (
            "node_id", "role", "benchmark", "task_id", "source_ref", "source_row_sha256")}
                   for node in members]
        clusters.append({"cluster_id": cluster_id, "members": summary,
                         "touches_historical_exposure": historical,
                         "touches_evaluation_source": bool(evaluation),
                         "evaluation_benchmarks": benchmarks,
                         "cross_benchmark_evaluation_correlation": len(benchmarks) > 1,
                         "retained_evaluation_node_ids": winners,
                         "retained_training_node_id": train_winner})
        for node in members:
            reasons = []
            role = node["role"]
            if historical:
                reasons.append("historical_exposure_cluster")
            if role == "training" and evaluation:
                reasons.append("evaluation_source_cluster")
            if role in {"historical", "audit_only"}:
                reasons.append(f"{role}_not_eligible")
            retained = (role == "evaluation" and winners.get(node["benchmark"]) == node["node_id"]
                        or role == "training" and train_winner == node["node_id"])
            if not retained and not reasons:
                reasons.append("within_benchmark_cluster_duplicate" if role == "evaluation"
                               else "training_cluster_duplicate")
            decisions.append({**{key: node[key] for key in (
                "node_id", "role", "benchmark", "task_id", "source_ref", "source_row_sha256")},
                "cluster_id": cluster_id, "retained": retained, "reasons": reasons})
    return (sorted(clusters, key=lambda row: row["cluster_id"]),
            sorted(decisions, key=lambda row: row["node_id"]),
            sorted(edges, key=_compact))


def _training_contract(row: dict) -> None:
    domain = row.get("domain")
    _require(domain in {"math", "code"} and row.get("source_domain") == (
        "gsm8k" if domain == "math" else "mbpp") and row.get("split") == "train",
        "training_source_domain_or_split_mismatch")
    _require(row.get("sft_input_fields") == ["prompt", "response"], "sft_input_fields_mismatch")
    for field in ("prompt", "response"):
        value = row.get(field)
        _require(isinstance(value, str) and bool(value.strip())
                 and row.get(f"{field}_sha256") == _text_hash(value),
                 "training_text_or_hash_mismatch")
    for field in ("benchmark", "task_id", "source_ref", "source_row_sha256"):
        _require(
            row.get(field) == row.get("audit", {}).get(field), "training_audit_identity_mismatch"
        )


def build_dataset(*, core_root: Path, apps_root: Path, math_root: Path,
                  exposure_audit_path: Path, cache_dir: Path, output_dir: Path) -> dict:
    """Verify complete adapters and sources, freeze overlap exclusions, publish last."""
    root = _fresh_root(output_dir)
    try:
        with tempfile.TemporaryDirectory(prefix=".stage2-data-", dir=root) as temporary:
            stage = Path(temporary)
            admitted = _admit_inputs(core_root=Path(core_root), apps_root=Path(apps_root),
                                     math_root=Path(math_root),
                                     exposure_audit_path=Path(exposure_audit_path),
                                     cache_dir=Path(cache_dir), stage=stage)
            nodes = [*admitted["historical"], *admitted["pool_nodes"]]
            endpoint_roots = {"gsm8k": Path(core_root) / "gsm8k",
                              "humaneval": Path(core_root) / "humaneval",
                              "apps": Path(apps_root), "math": Path(math_root)}
            initial = {}
            for benchmark, source in endpoint_roots.items():
                initial[benchmark] = 0
                for _, _, audit in _joined(source, admitted["endpoints"][benchmark]):
                    nodes.append(_node(audit, role="evaluation"))
                    initial[benchmark] += 1
            training_path = _artifact(
                Path(core_root), admitted["core"]["files"]["training_sources"]
            )
            training = list(_jsonl(training_path))
            for row in training:
                _training_contract(row)
                nodes.append(_node(row["audit"], role="training"))
            nodes.sort(key=lambda node: node["node_id"])
            clusters, decisions, edges = _cluster(nodes)
            by_node = {row["node_id"]: row for row in decisions}
            files = {}
            for name, records in (("clusters", clusters), ("exclusions", decisions),
                                  ("overlap-edges", edges), ("fingerprints", (
                                      {key: value for key, value in node.items()
                                       if key != "problem_text"}
                                      for node in nodes))):
                files[name] = _write_jsonl(stage / f"{name}.jsonl", records)
            retained_training = [row for row in training if by_node[
                _text_hash("training\0" + row["source_ref"])]["retained"]]
            _require(all(any(row["domain"] == domain for row in retained_training)
                         for domain in ("math", "code")), "empty_retained_training_domain")
            files["training_sources"] = _write_jsonl(stage / "training-sources.jsonl",
                                                      retained_training)
            endpoints = {}
            publish = []
            for benchmark, source in endpoint_roots.items():
                folder = stage / benchmark
                folder.mkdir()
                source_manifest = admitted["endpoints"][benchmark]
                kept = []
                streams = [(folder / f"{name}.jsonl").open("x", encoding="utf-8", newline="\n")
                           for name in ("generation", "scoring", "audit")]
                try:
                    for generation, scoring, audit in _joined(source, source_manifest):
                        decision = by_node[_text_hash("evaluation\0" + audit["source_ref"])]
                        if not decision["retained"]:
                            continue
                        kept.append(generation["task_id"])
                        for stream, record in zip(streams, (
                            generation, scoring, {**audit, "stage2_overlap": decision},
                        )):
                            stream.write(_compact(record) + "\n")
                finally:
                    for stream in streams:
                        stream.close()
                _require(bool(kept), "empty_retained_evaluation_endpoint")
                endpoint_files = {}
                for name in ("generation", "scoring", "audit"):
                    relative = f"{benchmark}/{name}.jsonl"
                    descriptor = {**_descriptor(folder / f"{name}.jsonl", f"{name}.jsonl"),
                                  "rows": len(kept)}
                    endpoint_files[name] = descriptor
                    files[f"{benchmark}_{name}"] = {**descriptor, "path": relative}
                    publish.append(relative)
                if benchmark == "math":
                    # Every auxiliary row and collision remains audit evidence, never SFT.
                    crosswalk = source_manifest["files"]["crosswalk_audit"]
                    original = _artifact(source, crosswalk)
                    _stage_input(original, folder / "crosswalk_audit.jsonl")
                    endpoint_files["crosswalk_audit"] = {
                        **crosswalk, "path": "crosswalk_audit.jsonl"}
                    files["math_crosswalk_audit"] = {
                        **crosswalk, "path": "math/crosswalk_audit.jsonl"}
                    publish.append("math/crosswalk_audit.jsonl")
                endpoint = {"schema": "p3-stage2-filtered-evaluation-v1", "policy": POLICY,
                            "benchmark": benchmark,
                            "status": "LEXICAL_AUDIT_COMPLETE_RUNTIME_AND_PRECISION_PENDING",
                            "generation_policy": source_manifest["generation_policy"],
                            "sources": source_manifest["sources"], "files": endpoint_files,
                            "source_admission_manifest": source_manifest,
                            "input_manifest_sha256": source_manifest.get("manifest_payload_sha256",
                                                                         source_manifest.get("manifest_sha256")),
                            "initial_source_row_count": initial[benchmark],
                            "retained_row_count": len(kept), "retained_task_ids": kept,
                            "excluded_row_count": initial[benchmark] - len(kept),
                            "denominator_meaning": (
                                "lexical_cluster_representatives_conditional_on_audit"),
                            "proven_iid": False, "pretraining_clean_proven": False,
                            "quality_non_degradation_admitted": False}
                _write_manifest(folder / "manifest.json", endpoint)
                endpoints[benchmark] = endpoint
                manifest_relative = f"{benchmark}/manifest.json"
                files[f"{benchmark}_manifest"] = _descriptor(folder / "manifest.json",
                                                              manifest_relative)
                publish.append(manifest_relative)
            exclusions = [row for row in decisions if row["role"] in {"training", "evaluation"}
                          and not row["retained"]]
            manifest = {"schema": POLICY, "policy": POLICY,
                        "status": "LEXICAL_AUDIT_COMPLETE_RUNTIME_AND_PRECISION_PENDING",
                        "files": files, "evaluations": endpoints, "inputs": admitted["inputs"],
                        "exposure": admitted["exposure"],
                        "counts": {"historical": len(admitted["historical"]),
                                   "math_auxiliary_pool_audit_only": len(admitted["pool_nodes"]),
                                   "evaluation_initial": initial,
                                   "evaluation_retained": {key: value["retained_row_count"]
                                                           for key, value in endpoints.items()},
                                   "evaluation_excluded": {
                                       key: initial[key] - value["retained_row_count"]
                                       for key, value in endpoints.items()},
                                   "training_initial": len(training),
                                   "training_initial_by_domain": dict(Counter(
                                       row["domain"] for row in training)),
                                   "training_retained": len(retained_training),
                                   "training_retained_by_domain": dict(Counter(
                                       row["domain"] for row in retained_training)),
                                   "clusters": len(clusters), "overlap_edges": len(edges),
                                   "excluded_by_reason": dict(sorted(Counter(
                                       reason for row in exclusions
                                       for reason in row["reasons"]).items())),
                                   "cross_benchmark_evaluation_clusters": sum(
                                       row["cross_benchmark_evaluation_correlation"]
                                       for row in clusters)},
                        "ast_coverage_by_benchmark": {
                            benchmark: dict(sorted(Counter(
                                node["ast"]["status"] for node in nodes
                                if node["benchmark"] == benchmark).items()))
                            for benchmark in sorted({node["benchmark"] for node in nodes})},
                        "runtime_provenance": {
                            "python_version": platform.python_version(),
                            "unicode_version": unicodedata.unidata_version},
                        "overlap_policy": {
                            "normalization": "Unicode_NFKC_then_whitespace_collapse",
                            "near_tokens": r"case_sensitive_Unicode_NFKC_regex_\w+|[^\w\s]",
                            "shingle_size": 5, "jaccard_threshold": "4/5_integer_exact",
                            "passes": ["original", "purely_isnumeric_tokens_to_<NUM>"],
                            "short_text": "fewer_than_5_tokens_exact_normalization_only",
                            "candidate_filter": (
                                "global_rarity_prefix_n_minus_ceil_4n_over_5_plus_1_length_ratio"),
                            "exact_edge_evidence": (
                                "deterministic_bucket_star_not_all_pair_enumeration"),
                            "evaluation_representative": "minimum_sha256_benchmark_NUL_task_id",
                            "training_representative": "lexicographic_minimum_source_ref",
                            "historical_rule": "any_touching_connected_cluster_excluded",
                            "training_rule": (
                                "exclude_any_evaluation_source_or_historical_touching_cluster"),
                            "numeric_masking": (
                                "conservative_template_grouping_not_independence_proof")},
                        "sft_input_fields": ["prompt", "response"],
                        "independence_claim": (
                            "lexical_clusters_only_not_proven_IID_or_pretraining_clean"),
                        "pool_endpoints": False, "quality_non_degradation_admitted": False}
            _write_manifest(stage / "manifest.json", manifest)
            root_files = [descriptor["path"] for descriptor in files.values()
                          if "/" not in descriptor["path"]]
            _publish(stage, root, [*root_files, *publish, "manifest.json"])
            return manifest
    except Exception as error:
        raise _failure(root, error) from None


def _load_tokenizer(cache_dir: Path) -> tuple[Any, dict]:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    _require(version("transformers") == native.TRANSFORMERS_VERSION,
             "pinned_transformers_version_required")
    snapshot = Path(snapshot_download(native.MODEL_ID, revision=native.REVISION,
                                      cache_dir=str(cache_dir),
                                      allow_patterns=sorted(native._TOKENIZER_FILES)))
    _require(snapshot.name == native.REVISION, "tokenizer_revision_mismatch")
    files = {name: _descriptor(snapshot / name, name) for name in sorted(native._TOKENIZER_FILES)
             if (snapshot / name).is_file()}
    _require("tokenizer.json" in files and "tokenizer_config.json" in files,
             "required_tokenizer_artifacts_missing")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    _require(tokenizer.is_fast and len(tokenizer) == native._EXPECTED_CONFIG["vocab_size"],
             "native_fast_tokenizer_vocabulary_required")
    _require(tokenizer.eos_token_id == 0 and tokenizer.bos_token_id == 0
             and tokenizer.pad_token_id == 0, "native_tokenizer_eos_bos_pad_mismatch")
    return tokenizer, {"repo_id": native.MODEL_ID, "revision": native.REVISION,
                       "tokenizer_class": type(tokenizer).__name__, "is_fast": True,
                       "vocabulary_size": len(tokenizer), "eos_id": tokenizer.eos_token_id,
                       "bos_id": tokenizer.bos_token_id, "pad_id": tokenizer.pad_token_id,
                       "files": files, "library_versions": {name: version(name) for name in (
                           "transformers", "tokenizers", "huggingface_hub")},
                       "weights_loaded": False, "add_special_tokens": False,
                       "prepend_bos": False, "append_native_eos_per_document": True}


def _encode_document(tokenizer: Any, row: dict) -> dict:
    _require(getattr(tokenizer, "is_fast", False) is True, "fast_tokenizer_offsets_required")
    eos = getattr(tokenizer, "eos_token_id", None)
    vocabulary = len(tokenizer)
    _require(type(eos) is int and 0 <= eos < vocabulary, "native_single_eos_required")
    prompt, response = row["prompt"], row["response"]
    _require(isinstance(prompt, str) and isinstance(response, str) and bool(response.strip()),
             "prompt_response_text_required")
    text = prompt + response
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded.get("input_ids"), encoded.get("offset_mapping")
    _require(isinstance(ids, list) and bool(ids) and isinstance(offsets, list)
             and len(ids) == len(offsets), "character_offsets_unavailable")
    boundary = len(prompt)
    labels, straddlers = [], []
    last_start, last_end = -1, -1
    for index, (token, offset) in enumerate(zip(ids, offsets)):
        _require(type(token) is int and 0 <= token < vocabulary, "invalid_tokenizer_input_id")
        _require(isinstance(offset, (list, tuple)) and len(offset) == 2
                 and all(type(item) is int for item in offset), "invalid_character_offset_shape")
        start, end = offset
        _require(0 <= start < end <= len(text) and start >= last_start and end >= last_end,
                 "ambiguous_or_nonmonotone_character_offsets")
        last_start, last_end = start, end
        labels.append(token if start >= boundary else -100)
        if start < boundary < end:
            straddlers.append(index)
    return {"input_ids": [*ids, eos], "labels": [*labels, eos],
            "response_boundary_char": boundary, "boundary_straddling_masked_tokens": straddlers,
            "appended_eos_token_index": len(ids), "document_token_count": len(ids) + 1}


def _domain_stream(
    tokenizer: Any, rows: list[dict], domain: str
) -> tuple[list[int], list[int], list[dict]]:
    ordered = sorted((row for row in rows if row["domain"] == domain), key=lambda row: (
        _text_hash(_ORDER_PREFIX + row["source_ref"]), row["source_ref"]))
    _require(bool(ordered), "empty_training_domain")
    encoded = {}
    ids, labels, spans = [], [], []
    document_number = 0
    while len(ids) < _DOMAIN_TOKENS:
        cycle, position = divmod(document_number, len(ordered))
        row = ordered[position]
        ref = row["source_ref"]
        if ref not in encoded:
            encoded[ref] = _encode_document(tokenizer, row)
        document = encoded[ref]
        start = len(ids)
        take = min(document["document_token_count"], _DOMAIN_TOKENS - start)
        ids.extend(document["input_ids"][:take])
        labels.extend(document["labels"][:take])
        spans.append({"domain": domain, "source_ref": ref,
                      "source_row_sha256": row["source_row_sha256"], "task_id": row["task_id"],
                      "cycle": cycle, "source_order_position": position,
                      "domain_document_number": document_number,
                      "stream_start": start, "stream_end": start + take,
                      "document_token_start": 0, "document_token_end": take,
                      "document_token_count": document["document_token_count"],
                      "final_document_cutoff": take < document["document_token_count"],
                      "response_boundary_char": document["response_boundary_char"],
                      "boundary_straddling_masked_tokens": [index for index in
                          document["boundary_straddling_masked_tokens"] if index < take],
                      "appended_eos_included": document["appended_eos_token_index"] < take,
                      "input_sha256": _ids_hash(document["input_ids"][:take]),
                      "labels_before_common_boundary_mask_sha256": _ids_hash(
                          document["labels"][:take])})
        document_number += 1
    for boundary in range(0, len(labels), 1024):
        labels[boundary] = -100
    return ids, labels, spans


def _ids_hash(values: list[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(values)}q", *values)).hexdigest()


def _pairing(inputs: str, labels: str, targets: int) -> str:
    return hashlib.sha256(b"p3-stage2-pair-v1\0" + bytes.fromhex(inputs)
                          + bytes.fromhex(labels) + struct.pack("<q", targets)).hexdigest()


def _window(ids: list[int], labels: list[int], *, length: int, index: int,
            domain: str, domain_start: int, spans: list[dict]) -> dict:
    micro_batches = []
    for offset in range(0, _WINDOW_TOKENS, length):
        batch_ids, batch_labels = ids[offset:offset + length], labels[offset:offset + length]
        micro_batches.append({"input_ids": [batch_ids], "labels": [batch_labels],
                              "input_sha256": _ids_hash(batch_ids),
                              "labels_sha256": _ids_hash(batch_labels)})
    input_hash, label_hash = _ids_hash(ids), _ids_hash(labels)
    targets = sum(label != -100 for label in labels)
    _require(targets > 0, "zero_target_window")
    return {"window_id": f"{PACKING_POLICY}/window-{index:03d}", "domain": domain,
            "domain_stream_start": domain_start, "domain_stream_end": domain_start + 4096,
            "micro_batches": micro_batches, "valid_target_count": targets,
            "input_token_count": len(ids), "input_sha256": input_hash,
            "labels_sha256": label_hash,
            "pairing_sha256": _pairing(input_hash, label_hash, targets),
            "document_spans": spans, "common_boundary_mask_interval": 1024}


def _pack(tokenizer: Any, rows: list[dict]) -> tuple[dict, list[dict], dict]:
    # Reuse the consumer's checks, including shifted-label denominators and checksums.
    from benchmarks.harness import p3_stage2_training as training

    streams = {domain: _domain_stream(tokenizer, rows, domain) for domain in ("math", "code")}
    windows = {1024: [], 4096: []}
    flat_ids, flat_labels = [], []
    for index in range(64):
        domain = "math" if index % 2 == 0 else "code"
        start = (index // 2) * _WINDOW_TOKENS
        ids, labels, spans = streams[domain]
        window_ids, window_labels = ids[start:start + 4096], labels[start:start + 4096]
        window_spans = []
        for span in spans:
            left, right = max(start, span["stream_start"]), min(start + 4096, span["stream_end"])
            if left < right:
                window_spans.append({"source_ref": span["source_ref"], "cycle": span["cycle"],
                                     "domain_document_number": span["domain_document_number"],
                                     "window_token_start": left - start,
                                     "window_token_end": right - start,
                                     "document_token_start": left - span["stream_start"],
                                     "document_token_end": right - span["stream_start"]})
        for length in windows:
            window = _window(window_ids, window_labels, length=length, index=index,
                             domain=domain, domain_start=start, spans=window_spans)
            training._inspect_window(window)
            windows[length].append(window)
        _require(all(windows[1024][-1][key] == windows[4096][-1][key] for key in (
            "input_sha256", "labels_sha256", "pairing_sha256", "valid_target_count")),
            "paired_window_checksum_or_target_mismatch")
        flat_ids.extend(window_ids)
        flat_labels.extend(window_labels)
    paired = {}
    for length, layout in windows.items():
        training._study_windows(layout, preflight=False)
        actual_ids = [token for window in layout for mb in window["micro_batches"]
                      for row in mb["input_ids"] for token in row]
        actual_labels = [token for window in layout for mb in window["micro_batches"]
                         for row in mb["labels"] for token in row]
        _require(actual_ids == flat_ids and actual_labels == flat_labels,
                 "flat_layout_pairing_mismatch")
        input_hash, label_hash = _ids_hash(actual_ids), _ids_hash(actual_labels)
        targets = sum(window["valid_target_count"] for window in layout)
        paired[str(length)] = {"input_sha256": input_hash, "labels_sha256": label_hash,
                               "valid_target_count": targets,
                               "pairing_sha256": _pairing(input_hash, label_hash, targets),
                               "ordered_window_pairing_sha256": hashlib.sha256(b"".join(
                                   bytes.fromhex(window["pairing_sha256"]) for window in layout
                               )).hexdigest()}
    _require(paired["1024"] == paired["4096"], "flat_checksum_or_target_mismatch")
    spans = [span for domain in ("math", "code") for span in streams[domain][2]]
    metadata = {"flat_pairing": paired, "input_token_count": len(flat_ids),
                "trained_target_count": paired["1024"]["valid_target_count"],
                "domain_input_token_counts": {domain: len(values[0])
                                               for domain, values in streams.items()},
                "domain_trained_target_counts": {domain: sum(token != -100 for token in values[1])
                                                  for domain, values in streams.items()},
                "final_cutoffs": {domain: values[2][-1] for domain, values in streams.items()},
                "boundary_straddling_masked_token_occurrences": sum(
                    len(span["boundary_straddling_masked_tokens"]) for span in spans),
                "domain_source_order": {domain: sorted(
                    [row["source_ref"] for row in rows if row["domain"] == domain],
                    key=lambda ref: (_text_hash(_ORDER_PREFIX + ref), ref))
                    for domain in ("math", "code")}}
    _require(metadata["input_token_count"] == 262144
             and metadata["domain_input_token_counts"] == {"math": 131072, "code": 131072},
             "exact_input_token_allocation_mismatch")
    return windows, spans, metadata


def pack_training(*, dataset_root: Path, cache_dir: Path, output_dir: Path) -> dict:
    """Pack one frozen source order for both lengths and all initialization seeds."""
    root = _fresh_root(output_dir)
    try:
        dataset_root = Path(dataset_root)
        dataset = _read_manifest(dataset_root)
        _require(
            dataset.get("schema") == POLICY
            and dataset.get("status") == "LEXICAL_AUDIT_COMPLETE_RUNTIME_AND_PRECISION_PENDING",
            "completed_dataset_required",
        )
        decisions = list(_jsonl(_artifact(dataset_root, dataset["files"]["exclusions"])))
        allowed = {row["source_ref"]: row for row in decisions
                   if row["role"] == "training" and row["retained"]}
        clusters = list(_jsonl(_artifact(dataset_root, dataset["files"]["clusters"])))
        by_cluster = {row["cluster_id"]: row for row in clusters}
        _require(len(by_cluster) == len(clusters), "duplicate_cluster_evidence")
        rows = list(_jsonl(_artifact(dataset_root, dataset["files"]["training_sources"])))
        _require(
            len(rows) == len(allowed) and len({row["source_ref"] for row in rows}) == len(rows),
            "retained_training_coverage_mismatch",
        )
        for row in rows:
            _training_contract(row)
            decision = allowed.get(row["source_ref"])
            _require(decision is not None and not decision["reasons"]
                     and all(decision[key] == row[key] for key in (
                         "benchmark", "task_id", "source_ref", "source_row_sha256")),
                     "retained_training_exclusion_join_mismatch")
            cluster = by_cluster.get(decision["cluster_id"])
            _require(cluster is not None and not cluster["touches_historical_exposure"]
                     and not cluster["touches_evaluation_source"]
                     and cluster["retained_training_node_id"] == decision["node_id"]
                     and any(member["node_id"] == decision["node_id"]
                             and member["source_ref"] == row["source_ref"]
                             and member["source_row_sha256"] == row["source_row_sha256"]
                             for member in cluster["members"]),
                     "retained_training_cluster_join_mismatch")
        tokenizer, tokenizer_spec = _load_tokenizer(Path(cache_dir))
        windows, spans, metadata = _pack(tokenizer, rows)
        with tempfile.TemporaryDirectory(prefix=".stage2-pack-", dir=root) as temporary:
            stage = Path(temporary)
            files = {"document_spans": _write_jsonl(stage / "document-spans.jsonl", spans)}
            for length, layout in windows.items():
                name = f"windows-L{length}.json"
                with (stage / name).open("x", encoding="utf-8", newline="\n") as stream:
                    stream.write(_compact(layout) + "\n")
                files[f"windows_L{length}"] = {**_descriptor(stage / name, name),
                                               "format": "json_array", "rows": 64,
                                               "length": length, "batch_size": 1,
                                               "micro_batches_per_window": 4096 // length}
            manifest = {"schema": PACKING_POLICY, "policy": PACKING_POLICY,
                        "status": "PAIRED_DATA_PACKED_RUNTIME_AND_PRECISION_PENDING",
                        "dataset_manifest_sha256": dataset["manifest_payload_sha256"],
                        "dataset_manifest_file": _descriptor(
                            dataset_root / "manifest.json", "manifest.json"),
                        "training_sources": dataset["files"]["training_sources"],
                        "files": files, "tokenizer": tokenizer_spec, **metadata,
                        "window_count_per_length": 64, "domain_windows": {"math": 32, "code": 32},
                        "sft_input_fields": ["prompt", "response"],
                        "response_label_policy": (
                            "joint_character_offsets_start_at_or_after_response_plus_native_EOS"),
                        "common_boundary_mask_interval": 1024,
                        "source_order": "sha256_p3-stage2-data-order-v1_NUL_source_ref",
                        "exhaustion_policy": (
                            "cycle_fixed_order_record_identity_cycle_and_final_cutoff"),
                        "seeds": [23, 41, 67, 89, 113], "seed_changes_data": False,
                        "evaluation_prompts_tokenized_or_truncated": False,
                        "runtime_provenance": {
                            "python_version": platform.python_version(),
                            "unicode_version": unicodedata.unidata_version},
                        "model_weights_loaded": False, "quality_non_degradation_admitted": False}
            _write_manifest(stage / "manifest.json", manifest)
            _publish(stage, root, [*(value["path"] for value in files.values()), "manifest.json"])
            return manifest
    except Exception as error:
        raise _failure(root, error) from None
