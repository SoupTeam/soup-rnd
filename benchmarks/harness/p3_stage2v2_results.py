"""Assemble immutable Stage 2 v2 controller evidence without running a model.

Example, from the checkout root on the integration owner's analysis worker::

    python -m benchmarks.harness.p3_stage2v2_results \
        --protocol-path /inputs/p3-stage2-v2-protocol.json \
        --protocol-sha256 <frozen-file-sha256> --data-root /inputs/v2-data \
        --controller-root /inputs/evaluation-01 --controller-root /inputs/evaluation-02 \
        --output /working/results-v2

Only complete, hash-bound two-arm controller pairs contribute observations.
Failed attempts remain in the report. Missing primary coverage blocks the primary
analysis instead of redefining its task set. This is an outcome-informed endpoint
redesign after v1 floors; v1 remains BLOCKED_INCOMPLETE_COVERAGE.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import statistics
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rich.console import Console

from benchmarks.harness.p3_realdata_corpus import _compact, _file_hash, _text_hash
from benchmarks.harness.p3_stage2_dataset import _json, _relative
from benchmarks.harness.p3_stage2_statistics import summarize_pairs
from benchmarks.harness.p3_stage2v2_decision import apply_informativeness_gate

ARMS = ("bf16_l2l", "oracle_block64_e4m3")
SEEDS = (23, 41, 67, 89, 113)
LENGTHS = (1024, 4096)
PRIMARY = ("hellaswag", "arc", "boolq", "lambada")
SECONDARY = ("gsm8k", "mbpp")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PAIR_FIELDS = ("seed", "length", "endpoint", "start", "stop")
_RESOURCE_FIELDS = (
    "cuda_allocated_peak_bytes",
    "cuda_reserved_peak_bytes",
    "process_rss_sampled_peak_bytes",
)
_SOURCE_ROOT = Path(__file__).parents[2]
_RUNNER_SOURCE = "benchmarks/results/p3-stage2-v2/orchestration/stage2v2_eval_runner.py"
_ANALYSIS_SOURCES = (
    "benchmarks/harness/p3_stage2v2_results.py",
    "benchmarks/harness/p3_stage2v2_decision.py",
    "benchmarks/harness/p3_stage2_statistics.py",
    "benchmarks/harness/p3_realdata_corpus.py",
    "benchmarks/harness/p3_stage2_dataset.py",
    "benchmarks/harness/p3_stage2_apps.py",
    "benchmarks/harness/p3_stage2_math.py",
    "benchmarks/harness/p3_stage2_core_data.py",
)


class AdmissionError(ValueError):
    """Assembly failed; report_path names the immutable failure manifest if created."""

    def __init__(self, reason: str, report_path: Path | None = None) -> None:
        super().__init__(reason)
        self.report_path = report_path


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"invalid {name}")
    return value


def _number(value: Any, name: str, minimum: float | None = None) -> float:
    _require(type(value) in {int, float}, f"invalid {name}")
    result = float(value)
    _require(math.isfinite(result) and (minimum is None or result >= minimum), f"invalid {name}")
    return result


def _digest(value: Any, name: str) -> str:
    _require(isinstance(value, str) and _SHA256.fullmatch(value) is not None, f"invalid {name}")
    return value


def _identifier(value: Any, name: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"invalid {name}")
    return value


def _real(path: Path) -> Path:
    return Path(os.path.realpath(path))


def _contained(root: Path, path: Path) -> bool:
    try:
        return os.path.commonpath((os.path.realpath(root), os.path.realpath(path))) == (
            os.path.realpath(root)
        )
    except ValueError:
        return False


def _file(root: Path, name: str, descriptor: dict) -> tuple[Path, dict]:
    path = _relative(root, name)
    _require(isinstance(descriptor, dict), "file descriptor must be an object")
    if "path" in descriptor:
        _require(descriptor["path"] == name, "file descriptor path mismatch")
    size, digest = _file_hash(path)
    _require(
        type(descriptor.get("bytes")) is int
        and descriptor["bytes"] == size
        and descriptor.get("sha256") == digest,
        f"file bytes or hash mismatch: {name}",
    )
    return path, {"path": str(path), "bytes": size, "sha256": digest}


def _read_json(path: Path, expected_hash: str, name: str) -> tuple[dict, dict]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    _require(digest == _digest(expected_hash, name + " hash"), f"{name} hash mismatch")
    value = _json(payload.decode("utf-8"))
    _require(isinstance(value, dict), f"{name} must be an object")
    return value, {"path": str(path), "bytes": len(payload), "sha256": digest}


def _rows(path: Path, descriptor: dict) -> list[dict]:
    # Bind the bytes being parsed as well as the preceding descriptor admission.
    payload = path.read_bytes()
    _require(
        len(payload) == descriptor["bytes"]
        and hashlib.sha256(payload).hexdigest() == descriptor["sha256"],
        f"file bytes or hash changed while reading: {path.name}",
    )
    rows = [_json(line.decode("utf-8")) for line in payload.splitlines()]
    _require(all(isinstance(row, dict) for row in rows), "JSONL rows must be objects")
    if "rows" in descriptor:
        _require(
            type(descriptor["rows"]) is int and descriptor["rows"] == len(rows),
            "file row count mismatch",
        )
    return rows


def _payload_hash(value: dict, field: str, name: str) -> None:
    expected = _digest(value.get(field), name + " payload hash")
    payload = {key: item for key, item in value.items() if key != field}
    _require(_text_hash(_compact(payload)) == expected, f"{name} payload hash mismatch")


def _pair_key(pair: dict) -> tuple:
    _require(isinstance(pair, dict), "frozen execution pair must be an object")
    _integer(pair.get("seed"), "pair seed")
    _integer(pair.get("length"), "pair length", 1)
    _identifier(pair.get("endpoint"), "pair endpoint")
    start = _integer(pair.get("start"), "pair start")
    stop = _integer(pair.get("stop"), "pair stop", 1)
    _require(start < stop, "invalid frozen execution pair interval")
    return tuple(pair[field] for field in _PAIR_FIELDS)


def _verify_sources(protocol: dict) -> dict[str, dict]:
    pins = protocol["source_files_sha256"]
    _require(
        (set(_ANALYSIS_SOURCES) | {_RUNNER_SOURCE}) <= set(pins),
        "analysis source inventory is incomplete",
    )
    verified = {}
    # B3 freezes transitive local imports too. Check every frozen entry.
    for name, expected in pins.items():
        path = _relative(_SOURCE_ROOT, name)
        _require(path.is_file(), f"analysis source is missing: {name}")
        size, digest = _file_hash(path)
        _require(digest == expected, f"analysis source hash mismatch: {name}")
        verified[name] = {"path": str(path), "bytes": size, "sha256": digest}
    return verified


def _validate_protocol(protocol: dict) -> None:
    _require(
        protocol.get("schema") == "p3-stage2-v2-protocol" and protocol.get("status") == "FROZEN",
        "protocol is not frozen v2",
    )
    _identifier(protocol.get("base_revision"), "protocol base revision")
    _digest(protocol.get("v1_protocol_sha256"), "original v1 protocol hash")
    _require(
        _number(protocol.get("margin"), "margin") == 0.01
        and _number(protocol.get("one_sided_alpha"), "alpha") == 0.025,
        "protocol margin or alpha differs from the approved design",
    )
    for name, expected in (
        ("arms", ARMS),
        ("seeds", SEEDS),
        ("lengths", LENGTHS),
        ("primary_endpoints", PRIMARY),
        ("secondary_endpoints", SECONDARY),
    ):
        _require(
            isinstance(protocol.get(name), list) and protocol[name] == list(expected),
            f"protocol {name} differs from the frozen design",
        )
        if name in {"seeds", "lengths"}:
            _require(all(type(value) is int for value in protocol[name]), f"invalid {name}")
    _require(
        protocol.get("outcome_informed_redesign_after_v1_floors") is True
        and protocol.get("v1_decision") == "BLOCKED_INCOMPLETE_COVERAGE",
        "protocol must preserve the outcome-informed redesign and blocked v1 verdict",
    )
    tasks = protocol.get("expected_tasks")
    _require(
        isinstance(tasks, dict) and set(tasks) == set(PRIMARY + SECONDARY),
        "protocol expected_tasks must contain all six endpoints",
    )
    for endpoint, identifiers in tasks.items():
        _require(isinstance(identifiers, list) and bool(identifiers), "empty frozen task set")
        for identifier in identifiers:
            _identifier(identifier, "frozen task ID")
        _require(len(identifiers) == len(set(identifiers)), f"duplicate frozen tasks: {endpoint}")
    chance = protocol.get("chance")
    _require(isinstance(chance, dict), "protocol chance must be an object")
    for endpoint in PRIMARY:
        _require(0 <= _number(chance.get(endpoint), "chance") <= 1, "invalid chance probability")
    if "floor_offset" in protocol:
        _require(_number(protocol["floor_offset"], "floor offset") == 0.05, "changed floor offset")
    required = protocol.get("required_primary_cells")
    _require(isinstance(required, list) and len(required) == 40, "all40 primary cells are required")
    required_keys = []
    for cell in required:
        _require(isinstance(cell, dict), "invalid required primary cell")
        required_keys.append(
            (
                cell.get("benchmark"),
                _integer(cell.get("length"), "cell length", 1),
                _integer(cell.get("seed"), "cell seed"),
            )
        )
    design = {
        (endpoint, length, seed) for endpoint in PRIMARY for length in LENGTHS for seed in SEEDS
    }
    _require(
        len(set(required_keys)) == 40 and set(required_keys) == design,
        "required primary cells differ from all40 design",
    )
    data = protocol.get("data")
    _require(isinstance(data, dict), "missing protocol data binding")
    _digest(data.get("dataset_manifest_file_sha256"), "data manifest hash")
    for field in (
        "retained_evaluation_tasks",
        "full_retained_evaluation_tasks",
        "ordered_task_ids_sha256",
        "source_ordered_task_ids_sha256",
    ):
        values = data.get(field)
        _require(
            isinstance(values, dict) and set(values) == set(PRIMARY + SECONDARY),
            f"missing six-endpoint data binding: {field}",
        )
        for endpoint, value in values.items():
            if field.endswith("sha256"):
                _digest(value, field)
            else:
                _integer(value, field, 1)
    for endpoint in PRIMARY + SECONDARY:
        _require(
            data["retained_evaluation_tasks"][endpoint] == len(tasks[endpoint])
            and data["full_retained_evaluation_tasks"][endpoint] >= len(tasks[endpoint]),
            "frozen evaluation counts disagree with expected tasks",
        )
    _digest(protocol.get("training_audit_sha256"), "training audit hash")
    sources = protocol.get("source_files_sha256")
    _require(isinstance(sources, dict) and bool(sources), "missing frozen source hashes")
    for name, digest in sources.items():
        _identifier(name, "source filename")
        _digest(digest, "source hash")
    checkpoints = protocol.get("checkpoints")
    _require(isinstance(checkpoints, dict), "missing checkpoint mapping")
    expected_keys = {
        f"L{length}-seed{seed}-{arm}" for length in LENGTHS for seed in SEEDS for arm in ARMS
    }
    _require(set(checkpoints) == expected_keys, "checkpoint mapping must retain all20 finals")
    for length in LENGTHS:
        for seed in SEEDS:
            for arm in ARMS:
                checkpoint = checkpoints[f"L{length}-seed{seed}-{arm}"]
                _require(
                    isinstance(checkpoint, dict)
                    and checkpoint.get("arm") == arm
                    and type(checkpoint.get("seed")) is int
                    and checkpoint["seed"] == seed
                    and type(checkpoint.get("length")) is int
                    and checkpoint["length"] == length,
                    "checkpoint identity mismatch",
                )
                _require(
                    isinstance(checkpoint.get("final_step64"), dict), "missing final checkpoint"
                )
                _digest(checkpoint["final_step64"].get("sha256"), "final checkpoint hash")
    worklist = protocol.get("execution_pairs")
    _require(isinstance(worklist, list) and bool(worklist), "empty frozen execution pairs")
    intervals: dict[tuple, list[tuple[int, int]]] = {}
    seen = set()
    for pair in worklist:
        key = _pair_key(pair)
        seed, length, endpoint, start, stop = key
        _require(
            seed in SEEDS
            and length in LENGTHS
            and endpoint in tasks
            and stop <= len(tasks[endpoint])
            and pair.get("arms") == list(ARMS),
            "invalid frozen execution pair design",
        )
        _require(key not in seen, "duplicate frozen execution pair")
        seen.add(key)
        intervals.setdefault((seed, length, endpoint), []).append((start, stop))
    for length in LENGTHS:
        for seed in SEEDS:
            for endpoint, identifiers in tasks.items():
                cursor = 0
                for start, stop in sorted(intervals.get((seed, length, endpoint), [])):
                    _require(
                        start == cursor, "overlapping or missing frozen execution pair interval"
                    )
                    cursor = stop
                _require(cursor == len(identifiers), "incomplete frozen execution pair worklist")


def _read_tasks(data_root: Path, protocol: dict, inputs: dict) -> tuple[dict, dict]:
    path = _relative(data_root, "manifest.json")
    manifest, binding = _read_json(
        path, protocol["data"]["dataset_manifest_file_sha256"], "data manifest"
    )
    inputs["data_manifest"] = binding
    _require(
        manifest.get("schema") == "p3-stage2-v2-data-v1"
        and manifest.get("status") == "DATA_ADMITTED_PENDING_PROTOCOL_FREEZE",
        "data manifest is not admitted v2 data",
    )
    _payload_hash(manifest, "manifest_payload_sha256", "data manifest")
    endpoint_specs = manifest.get("endpoints")
    _require(isinstance(endpoint_specs, dict), "missing endpoint task descriptors")
    tasks = {}
    task_files = {}
    inputs["task_files"] = {}
    for endpoint, expected_ids in protocol["expected_tasks"].items():
        spec = endpoint_specs.get(endpoint)
        _require(
            isinstance(spec, dict) and spec.get("role") == "evaluation",
            "endpoint is not admitted evaluation data",
        )
        descriptor = spec["files"]["tasks"]
        task_files[endpoint] = descriptor
        path, binding = _file(data_root, descriptor["path"], descriptor)
        rows = _rows(path, descriptor)
        inputs["task_files"][endpoint] = binding
        mapping = {}
        for row in rows:
            task_id = _identifier(row.get("task_id"), "data task ID")
            _require(task_id not in mapping, "duplicate data task ID")
            _require(
                row.get("endpoint") == endpoint and row.get("role") == "evaluation",
                "data task endpoint or role mismatch",
            )
            _identifier(row.get("source_ref"), "data source reference")
            _digest(row.get("source_row_sha256"), "data source row hash")
            _identifier(row.get("context"), "task context")
            expected_metric = (
                "acc_norm"
                if endpoint in {"hellaswag", "arc"}
                else ("nll" if endpoint in SECONDARY else "acc")
            )
            _require(row.get("metric") == expected_metric and "gold" in row, "data metric mismatch")
            if endpoint in {"hellaswag", "arc", "boolq"}:
                choices = row.get("choices")
                _require(
                    isinstance(choices, list)
                    and len(choices) >= 2
                    and all(isinstance(choice, str) and bool(choice) for choice in choices),
                    "invalid data choices",
                )
                _require(
                    type(row["gold"]) is int and 0 <= row["gold"] < len(choices),
                    "invalid data gold index",
                )
            else:
                _identifier(row.get("target"), "data target")
            mapping[task_id] = row
        _require(
            spec.get("retained_task_ids") == list(mapping)
            and type(spec.get("retained_row_count")) is int
            and spec["retained_row_count"] == len(rows),
            "retained data task identity mismatch",
        )
        source_order_hash = _text_hash(_compact(list(mapping)))
        _require(
            spec.get("ordered_task_ids_sha256") == source_order_hash
            and protocol["data"]["source_ordered_task_ids_sha256"][endpoint] == source_order_hash,
            "source ordered task hash mismatch",
        )
        _require(
            protocol["data"]["full_retained_evaluation_tasks"][endpoint] == len(rows),
            "full retained evaluation count mismatch",
        )
        _require(
            all(task_id in mapping for task_id in expected_ids), "frozen task absent from data"
        )
        _require(
            protocol["data"]["ordered_task_ids_sha256"][endpoint]
            == _text_hash(_compact(expected_ids)),
            "frozen ordered task hash mismatch",
        )
        tasks[endpoint] = {task_id: mapping[task_id] for task_id in expected_ids}
    return tasks, task_files


def _validate_score(row: dict, *, task: dict, task_index: int, identity: dict) -> None:
    _payload_hash(row, "row_sha256", "score row")
    _require(row.get("schema") == "p3-stage2-v2-loglik-row-v1", "unknown score row schema")
    for name, value in identity.items():
        _require(
            type(row.get(name)) is type(value) and row[name] == value,
            f"score row {name} identity mismatch",
        )
    _require(
        type(row.get("task_index")) is int and row["task_index"] == task_index,
        "score row task index or duplicate task mismatch",
    )
    for field in ("task_id", "source_ref", "source_row_sha256", "metric", "gold"):
        _require(
            type(row.get(field)) is type(task[field]) and row[field] == task[field],
            f"score row {field} does not bind to the frozen task",
        )
    score = _number(row.get("score"), "score")
    if task["metric"] == "nll":
        count = _integer(row.get("n_targets"), "NLL target count", 1)
        total = _number(row.get("nll_sum"), "NLL sum", 0)
        mean = _number(row.get("nll_mean"), "NLL mean", 0)
        _require(
            math.isclose(total / count, mean, rel_tol=1e-10, abs_tol=1e-10)
            and score == mean
            and "pass" not in row,
            "inconsistent NLL task mean or score",
        )
        return
    likelihood = row.get("results")
    request_count = 1 if task["endpoint"] == "lambada" else len(task["choices"])
    _require(
        isinstance(likelihood, list) and len(likelihood) == request_count,
        "likelihood result count mismatch",
    )
    for result in likelihood:
        _require(isinstance(result, dict), "likelihood result must be an object")
        _require(
            _number(result.get("logprob"), "likelihood logprob") <= 0, "positive likelihood logprob"
        )
        _require(type(result.get("greedy")) is bool, "invalid likelihood greedy verdict")
        _integer(result.get("n_tokens"), "likelihood target tokens", 1)
    _require(
        type(row.get("pass")) is bool and score == float(row["pass"]), "inconsistent pass score"
    )
    if task["endpoint"] == "lambada":
        _require(row["pass"] is likelihood[0]["greedy"], "inconsistent LAMBADA likelihood verdict")
        _require(
            _integer(row.get("pred"), "LAMBADA prediction") == 0
            and _integer(row.get("pred_norm"), "LAMBADA normalized prediction") == 0,
            "inconsistent LAMBADA prediction convention",
        )
    else:
        prediction = max(range(request_count), key=lambda index: likelihood[index]["logprob"])
        normalized = max(
            range(request_count),
            key=lambda index: likelihood[index]["logprob"] / len(task["choices"][index]),
        )
        _require(
            _integer(row.get("pred"), "prediction") == prediction
            and _integer(row.get("pred_norm"), "normalized prediction") == normalized,
            "recorded prediction differs from first likelihood argmax",
        )
        selected = normalized if task["metric"] == "acc_norm" else prediction
        _require(row["pass"] is (selected == task["gold"]), "inconsistent likelihood pass verdict")


def _admit_shard_inputs(
    path: Path,
    manifest: dict,
    identity: dict,
    pair: dict,
    protocol: dict,
    tasks: dict,
    task_files: dict,
    inputs: dict,
) -> None:
    descriptor = manifest["files"]["admission.json"]
    admission, _ = _read_json(path, descriptor["sha256"], "shard admission")
    _require(
        admission.get("schema") == "p3-stage2-v2-loglik-admission-v1",
        "unknown shard admission schema",
    )
    endpoint = pair["endpoint"]
    selected = [tasks[endpoint][task_id] for task_id in manifest["ordered_task_ids"]]
    selection = {
        "data_manifest_sha256": protocol["data"]["dataset_manifest_file_sha256"],
        "complete_endpoint_row_count": protocol["data"]["retained_evaluation_tasks"][endpoint],
        "start": pair["start"],
        "stop": pair["stop"],
        "ordered_task_ids": [task["task_id"] for task in selected],
        "selected_rows_sha256": _text_hash(_compact(selected)),
        "tasks_file": task_files[endpoint],
    }
    for name, expected in {**identity, **selection}.items():
        _require(
            type(admission.get(name)) is type(expected)
            and _compact(admission[name]) == _compact(expected),
            f"shard admission {name} binding mismatch",
        )
    for name, expected in selection.items():
        _require(
            type(manifest.get(name)) is type(expected)
            and _compact(manifest[name]) == _compact(expected),
            f"shard admission/manifest {name} binding mismatch",
        )
    run_key = f"L{pair['length']}-seed{pair['seed']}-{identity['arm']}"
    _require(
        admission.get("training_run_key") == run_key
        and _compact(admission.get("checkpoint"))
        == _compact(protocol["checkpoints"][run_key]["final_step64"]),
        "shard admission checkpoint binding mismatch",
    )
    _require(
        admission.get("v1_protocol_sha256") == protocol["v1_protocol_sha256"]
        and admission.get("source_files_sha256") == manifest["source_files_sha256"],
        "shard admission original protocol or source binding mismatch",
    )
    protocol_file, training_audit = admission.get("protocol_file"), admission.get("training_audit")
    _require(
        isinstance(protocol_file, dict) and isinstance(training_audit, dict),
        "shard admission protocol/audit descriptors are missing",
    )
    for name, value in (("protocol_file", protocol_file), ("training_audit", training_audit)):
        _identifier(value.get("path"), "shard admission " + name + " historical path")
        _integer(value.get("bytes"), "shard admission " + name + " bytes", 1)
    _require(
        protocol_file["bytes"] == inputs["protocol"]["bytes"]
        and protocol_file.get("sha256") == identity["protocol_sha256"]
        and training_audit.get("sha256") == protocol["training_audit_sha256"],
        "shard admission protocol/audit file binding mismatch",
    )


def _admit_batches(batches: list[dict], rows: list[dict], manifest: dict) -> None:
    targets = (
        [row["n_targets"] for row in rows]
        if rows[0]["metric"] == "nll"
        else [result["n_tokens"] for row in rows for result in row["results"]]
    )
    totals = {
        "batch_count": len(batches),
        "request_count": len(targets),
        "scored_tokens": sum(targets),
    }
    _require(bool(batches), "complete shard has no batch records")
    for field, expected in totals.items():
        _require(
            type(manifest.get(field)) is int and manifest[field] == expected,
            f"manifest batch/request/token total mismatch: {field}",
        )
    seen = set()
    for index, batch in enumerate(batches):
        _require(
            batch.get("schema") == "p3-stage2-v2-loglik-batch-v1"
            and batch.get("event") == "batch_complete"
            and batch.get("endpoint") == manifest["endpoint"]
            and type(batch.get("batch_index")) is int
            and batch["batch_index"] == index,
            "unbound batch schema, endpoint, event or batch identity",
        )
        indices = batch.get("request_indices")
        _require(isinstance(indices, list) and bool(indices), "batch request indices are missing")
        for request in indices:
            _require(
                type(request) is int and 0 <= request < len(targets) and request not in seen,
                "duplicate or out-of-range batch request index",
            )
            seen.add(request)
        _require(
            type(batch.get("scored_tokens")) is int
            and batch["scored_tokens"] == sum(targets[request] for request in indices),
            "batch scored-token total differs from request targets",
        )
        for field in (
            "input_tokens",
            "padded_input_tokens",
            "forward_calls",
            "logits_estimated_peak_bytes",
        ):
            _integer(batch.get(field), "batch " + field, 1)
        _require(
            batch["padded_input_tokens"] >= batch["input_tokens"],
            "batch padded token count is below input token count",
        )
        measurement = batch.get("measurement")
        _require(
            isinstance(measurement, dict) and set(_RESOURCE_FIELDS) <= set(measurement),
            "missing batch resource measurements",
        )
        for field in _RESOURCE_FIELDS:
            if measurement[field] is not None:
                _integer(measurement[field], "batch " + field)
        _number(measurement.get("wall_seconds"), "batch wall seconds", 0)
        _identifier(measurement.get("scope"), "batch measurement scope")
    _require(seen == set(range(len(targets))), "batch requests do not form the complete partition")


def _read_shard(
    root: Path,
    receipt: dict,
    pair: dict,
    protocol: dict,
    protocol_hash: str,
    tasks: dict,
    task_files: dict,
    inputs: dict,
    resources: list[dict],
    *,
    admitted_for_analysis: bool = True,
) -> list[dict]:
    path = _relative(root, receipt.get("manifest"))
    manifest, binding = _read_json(path, receipt.get("sha256"), "shard manifest")
    inputs["shards"].append(binding)
    _payload_hash(manifest, "manifest_payload_sha256", "shard manifest")
    _require(
        manifest.get("schema") == "p3-stage2-v2-loglik-run-v1"
        and manifest.get("status") == "COMPLETE"
        and manifest.get("selected_shard_complete") is True,
        "incomplete shard manifest",
    )
    cleanup = manifest.get("cleanup")
    _require(
        isinstance(cleanup, dict) and cleanup.get("complete") is True,
        "shard cleanup is incomplete or unrecorded",
    )
    arm = receipt["arm"]
    checkpoint = protocol["checkpoints"][f"L{pair['length']}-seed{pair['seed']}-{arm}"]
    identity = {
        "endpoint": pair["endpoint"],
        "arm": arm,
        "seed": pair["seed"],
        "length": pair["length"],
        "protocol_sha256": protocol_hash,
        "audit_sha256": protocol["training_audit_sha256"],
        "checkpoint_sha256": checkpoint["final_step64"]["sha256"],
    }
    manifest_identity = {
        **identity,
        "start": pair["start"],
        "stop": pair["stop"],
        "data_manifest_sha256": protocol["data"]["dataset_manifest_file_sha256"],
    }
    for name, value in manifest_identity.items():
        _require(
            type(manifest.get(name)) is type(value) and manifest[name] == value,
            f"shard manifest {name} identity mismatch",
        )
    from benchmarks.harness.p3_stage2v2_loglik import SOURCE_PATHS

    sources = manifest.get("source_files_sha256")
    _require(
        isinstance(sources, dict) and set(sources) == set(SOURCE_PATHS),
        "shard source hash inventory is incomplete or unexpected",
    )
    _require(
        all(
            name in protocol["source_files_sha256"]
            and protocol["source_files_sha256"][name] == digest
            for name, digest in sources.items()
        ),
        "shard source hash mismatch",
    )
    expected_ids = protocol["expected_tasks"][pair["endpoint"]][pair["start"] : pair["stop"]]
    _require(
        manifest.get("ordered_task_ids") == expected_ids, "shard ordered task identity mismatch"
    )
    for name in ("requested_rows", "completed_rows"):
        _require(
            type(manifest.get(name)) is int and manifest[name] == len(expected_ids),
            "shard task count mismatch",
        )
    files = manifest.get("files")
    _require(
        isinstance(files, dict)
        and {"scores.jsonl", "batches.jsonl", "admission.json"} <= set(files),
        "missing mandatory score, batch or admission file descriptor",
    )
    verified = {}
    for name, descriptor in files.items():
        file_path, file_binding = _file(path.parent, name, descriptor)
        inputs["shard_files"].append(file_binding)
        verified[name] = file_path
    _admit_shard_inputs(
        verified["admission.json"], manifest, identity, pair, protocol, tasks, task_files, inputs
    )
    rows = _rows(verified["scores.jsonl"], files["scores.jsonl"])
    _require(len(rows) == len(expected_ids), "score row task count mismatch")
    for offset, (task_id, row) in enumerate(zip(expected_ids, rows)):
        _validate_score(
            row,
            task=tasks[pair["endpoint"]][task_id],
            task_index=pair["start"] + offset,
            identity=identity,
        )
    batches = _rows(verified["batches.jsonl"], files["batches.jsonl"])
    _admit_batches(batches, rows, manifest)
    for index, batch in enumerate(batches):
        resources.append(
            {
                **batch,
                **identity,
                "controller_root": str(root),
                "manifest_sha256": binding["sha256"],
                "batch_index": index,
                "admitted_for_analysis": admitted_for_analysis,
            }
        )
    return rows


def _controller_artifacts(
    root: Path, run: dict, protocol: dict, protocol_hash: str, inputs: dict
) -> set[tuple]:
    runner = _relative(root, "stage2v2_eval_runner.py")
    size, digest = _file_hash(runner)
    _require(
        digest == protocol["source_files_sha256"][_RUNNER_SOURCE],
        "copied controller runner source hash mismatch",
    )
    evidence = {"runner": {"path": str(runner), "bytes": size, "sha256": digest}}
    copied, evidence["protocol"] = _read_json(
        _relative(root, "p3-stage2-v2-protocol.json"), protocol_hash, "copied controller protocol"
    )
    _require(copied == protocol, "copied controller protocol differs from frozen input")
    admission, evidence["execution_admission"] = _read_json(
        _relative(root, "execution-admission.json"),
        run.get("admission_sha256"),
        "controller execution admission",
    )
    _require(
        admission.get("schema") == "p3-stage2v2-execution-admission-v1"
        and admission.get("phase") == "evaluate"
        and admission.get("gpu") is True,
        "controller execution admission is not evaluation",
    )
    _require(
        run.get("base_revision") == protocol["base_revision"],
        "controller base revision differs from frozen protocol",
    )
    declared_inputs, bindings = admission.get("inputs"), run.get("input_bindings")
    _require(
        isinstance(declared_inputs, dict)
        and isinstance(bindings, dict)
        and set(declared_inputs) == set(bindings),
        "controller input binding inventory mismatch",
    )
    for name, specification in declared_inputs.items():
        _require(
            isinstance(specification, dict) and isinstance(bindings[name], dict),
            "invalid controller input binding",
        )
        _digest(specification.get("sha256"), "execution input hash")
        _identifier(specification.get("pattern"), "execution input pattern")
        _identifier(bindings[name].get("path"), "controller input historical path")
        _require(
            {key: value for key, value in bindings[name].items() if key != "path"} == specification,
            "controller input binding differs from execution admission",
        )
    _require(
        admission.get("protocol_sha256") == protocol_hash
        and declared_inputs.get("PROTOCOL", {}).get("sha256") == protocol_hash,
        "controller execution protocol input binding mismatch",
    )
    data_hash = protocol["data"]["dataset_manifest_file_sha256"]
    _require(
        declared_inputs.get("AUDIT", {}).get("sha256") == protocol["training_audit_sha256"]
        and any(spec["sha256"] == data_hash for spec in declared_inputs.values()),
        "controller audit/data input binding mismatch",
    )
    overlay_hash = _digest(run.get("source_overlay_sha256"), "controller overlay hash")
    _require(
        admission.get("source_overlay_sha256") == overlay_hash,
        "controller execution overlay hash mismatch",
    )
    overlay = _relative(root, "source-overlay.zip")
    overlay_size, observed = _file_hash(overlay)
    _require(observed == overlay_hash, "controller source overlay file hash mismatch")
    source_pins = admission.get("source_files_sha256")
    _require(
        isinstance(source_pins, dict) and bool(source_pins),
        "controller overlay source inventory is missing",
    )
    with zipfile.ZipFile(overlay) as archive:
        names = archive.namelist()
        _require(
            len(names) == len(set(names)) and set(names) == set(source_pins),
            "controller overlay member/source inventory mismatch",
        )
        for name in names:
            _relative(_SOURCE_ROOT, name)
            expected = _digest(source_pins[name], "controller overlay source hash")
            _require(not archive.getinfo(name).is_dir(), "controller overlay contains a directory")
            member_hash = hashlib.sha256()
            with archive.open(name) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    member_hash.update(chunk)
            _require(member_hash.hexdigest() == expected, "controller overlay member hash mismatch")
            if name in protocol["source_files_sha256"]:
                _require(
                    expected == protocol["source_files_sha256"][name],
                    "controller overlay frozen source hash mismatch",
                )
    _require(
        _file_hash(overlay) == (overlay_size, overlay_hash),
        "controller source overlay changed while reading",
    )
    evidence["source_overlay"] = {
        "path": str(overlay),
        "bytes": overlay_size,
        "sha256": overlay_hash,
    }
    inputs["controller_artifacts"].append(evidence)
    requested = admission.get("execution_pairs")
    _require(
        isinstance(requested, list) and bool(requested),
        "controller admission has no frozen execution pairs",
    )
    frozen = {_pair_key(pair): index for index, pair in enumerate(protocol["execution_pairs"])}
    positions = []
    keys = set()
    for pair in requested:
        key = _pair_key(pair)
        _require(
            key in frozen and pair.get("arms") == list(ARMS),
            "controller admission contains an unfrozen execution pair",
        )
        positions.append(frozen[key])
        keys.add(key)
    _require(
        positions == sorted(set(positions)),
        "controller admission has duplicate or unordered frozen execution pairs",
    )
    return keys


def _collect(
    roots: Sequence[Path],
    protocol: dict,
    protocol_hash: str,
    tasks: dict,
    task_files: dict,
    inputs: dict,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    observations = {}
    controllers, failures, batches = [], [], []
    seen_intervals: dict[tuple, list[tuple[int, int]]] = {}
    frozen = {_pair_key(pair) for pair in protocol["execution_pairs"]}
    inputs.update(controllers=[], controller_artifacts=[], shards=[], shard_files=[])
    seen_roots = set()
    for supplied in roots:
        root = _real(supplied)
        _require(str(root) not in seen_roots, "duplicate controller root")
        seen_roots.add(str(root))
        path = _relative(root, "run.json")
        payload = path.read_bytes()
        run = _json(payload.decode("utf-8"))
        binding = {
            "path": str(path),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        inputs["controllers"].append(binding)
        _require(
            isinstance(run, dict)
            and run.get("schema") == "p3-stage2v2-controller-v1"
            and run.get("phase") == "evaluate"
            and run.get("protocol_sha256") == protocol_hash,
            "controller is not bound to frozen v2 evaluation",
        )
        _require(run.get("cleanup_complete") is True, "controller cleanup is incomplete")
        requested = _controller_artifacts(root, run, protocol, protocol_hash, inputs)
        _identifier(run.get("status"), "controller status")
        _number(run.get("controller_wall_seconds"), "controller wall seconds", 0)
        commands, pairs = run.get("commands"), run.get("pairs")
        _require(
            isinstance(commands, list) and isinstance(pairs, list), "invalid controller records"
        )
        for command in commands:
            _require(isinstance(command, dict), "invalid command receipt")
            _identifier(command.get("name"), "command name")
            _number(command.get("seconds"), "command seconds", 0)
        controllers.append({**run, "root": str(root), "run_sha256": binding["sha256"]})
        partial = []
        for pair in pairs:
            key = _pair_key(pair)
            _require(
                key in frozen and key in requested,
                "controller pair is not a frozen execution pair from its admission",
            )
            _identifier(pair.get("status"), "pair status")
            receipts = pair.get("arms")
            _require(isinstance(receipts, list) and len(receipts) <= 2, "invalid pair arms")
            _require(all(isinstance(receipt, dict) for receipt in receipts), "invalid arm receipt")
            expected_arms = (
                list(ARMS) if pair["status"] == "COMPLETE" else list(ARMS[: len(receipts)])
            )
            _require(
                [receipt.get("arm") for receipt in receipts] == expected_arms,
                "pair arms must retain baseline then candidate",
            )
            if pair["status"] != "COMPLETE":
                partial.append(pair)
                for receipt in receipts:
                    _read_shard(
                        root,
                        receipt,
                        pair,
                        protocol,
                        protocol_hash,
                        tasks,
                        task_files,
                        inputs,
                        batches,
                        admitted_for_analysis=False,
                    )
                continue
            seed, length, endpoint, start, stop = key
            group = (seed, length, endpoint)
            intervals = seen_intervals.setdefault(group, [])
            _require(
                not any(
                    start < previous_stop and previous_start < stop
                    for previous_start, previous_stop in intervals
                ),
                "overlapping or duplicate successful pair intervals",
            )
            intervals.append((start, stop))
            for receipt in receipts:
                rows = _read_shard(
                    root, receipt, pair, protocol, protocol_hash, tasks, task_files, inputs, batches
                )
                for row in rows:
                    identity = (endpoint, length, seed, row["task_id"], row["arm"])
                    _require(
                        identity not in observations, "duplicate successful task/arm observation"
                    )
                    observations[identity] = row
        failed_commands = [
            command
            for command in commands
            if command.get("returncode") != 0 or command.get("interruption")
        ]
        if (
            partial
            or failed_commands
            or run["status"] != "PLANNED_PAIRS_COMPLETE_NOT_GLOBAL_ADMISSION"
        ):
            failures.append(
                {
                    "controller_root": str(root),
                    "run_sha256": binding["sha256"],
                    "status": run["status"],
                    "error_type": run.get("error_type"),
                    "error": run.get("error"),
                    "pairs": partial,
                    "failed_commands": failed_commands,
                }
            )
    return list(observations.values()), controllers, failures, batches


def _index_pairs(
    rows: Sequence[dict],
    expected_tasks: dict[str, list[str]],
    seeds: Sequence[int],
    lengths: Sequence[int],
) -> dict[tuple, dict]:
    index = {}
    task_sets = {benchmark: set(identifiers) for benchmark, identifiers in expected_tasks.items()}
    for row in rows:
        _require(isinstance(row, dict), "paired row must be an object")
        benchmark = row.get("benchmark")
        _require(
            benchmark in expected_tasks
            and row.get("length") in lengths
            and row.get("seed") in seeds
            and row.get("task_id") in task_sets[benchmark],
            "unexpected paired task identity",
        )
        _integer(row.get("length"), "paired length", 1)
        _integer(row.get("seed"), "paired seed")
        _require(
            type(row.get("baseline_pass")) is bool and type(row.get("candidate_pass")) is bool,
            "paired verdicts must be bools",
        )
        key = (benchmark, row["length"], row["seed"], row["task_id"])
        _require(key not in index, "duplicate paired task identity")
        index[key] = row
    return index


def summarize_primary(rows: Sequence[dict], protocol: dict) -> dict:
    """Apply unchanged statistics only to the full frozen primary design."""
    tasks = {
        endpoint: protocol["expected_tasks"][endpoint] for endpoint in protocol["primary_endpoints"]
    }
    seeds, lengths = protocol["seeds"], protocol["lengths"]
    index = _index_pairs(rows, tasks, seeds, lengths)
    missing = []
    cells = []
    for endpoint, identifiers in tasks.items():
        for length in lengths:
            for seed in seeds:
                absent = [
                    task_id
                    for task_id in identifiers
                    if (endpoint, length, seed, task_id) not in index
                ]
                missing.extend(
                    {
                        "benchmark": endpoint,
                        "length": length,
                        "seed": seed,
                        "task_id": task_id,
                        "missing_arms": list(ARMS),
                    }
                    for task_id in absent
                )
                cells.append(
                    {
                        "benchmark": endpoint,
                        "length": length,
                        "seed": seed,
                        "expected_n": len(identifiers),
                        "observed_n": len(identifiers) - len(absent),
                        "n": None,
                        "baseline_accuracy": None,
                        "candidate_accuracy": None,
                        "harms": None,
                        "gains": None,
                        "pointwise_bounds": None,
                        "simultaneous_bounds": None,
                        "informative": None,
                        "cell_result": "BLOCKED_INCOMPLETE_COVERAGE",
                    }
                )
    if missing:
        return {
            "schema": "p3-stage2-paired-quality-summary-v1",
            "decision": "BLOCKED_INCOMPLETE_COVERAGE",
            "statistical_decision": "BLOCKED_INCOMPLETE_COVERAGE",
            "margin": protocol["margin"],
            "one_sided_alpha": protocol["one_sided_alpha"],
            "comparison_count": len(cells),
            "cells": cells,
            "endpoints": [],
            "floor_cells": [],
            "coverage": {
                "complete": False,
                "expected_paired_rows": len(index) + len(missing),
                "observed_paired_rows": len(index),
                "missing_identities": missing,
                "benchmarks": list(tasks),
                "seeds": list(seeds),
                "lengths": list(lengths),
            },
            "analysis_status": "No partial-set primary statistics were computed",
        }
    raw = summarize_pairs(
        list(rows),
        expected_tasks=tasks,
        seeds=seeds,
        lengths=lengths,
        margin=protocol["margin"],
        one_sided_alpha=protocol["one_sided_alpha"],
    )
    result = apply_informativeness_gate(raw, chance=protocol["chance"])
    result["coverage"]["missing_identities"] = []
    for cell in result["cells"]:
        cell["cell_result"] = (
            "EXCESSIVE_DEGRADATION"
            if cell["familywise_excessive_degradation"]
            else "NONINFERIOR"
            if cell["conditional_noninferiority"]
            else "INCONCLUSIVE"
        )
    return result


def _t_summary(values: Sequence[float]) -> dict:
    count = len(values)
    _require(count > 0, "cannot summarize empty task differences")
    mean = math.fsum(values) / count
    result = {
        "mean_difference": mean,
        "degrees_of_freedom": count - 1,
        "standard_error": None,
        "confidence_level": 0.95,
        "confidence_interval": None,
        "status": "INSUFFICIENT_TASKS" if count == 1 else "COMPLETE",
    }
    if count > 1:
        from scipy.stats import t

        standard_error = statistics.stdev(values) / math.sqrt(count)
        critical = float(t.ppf(0.975, count - 1))
        _require(math.isfinite(critical), "invalid Student t quantile")
        radius = critical * standard_error
        result.update(
            standard_error=standard_error, confidence_interval=[mean - radius, mean + radius]
        )
    return result


def summarize_clustered_success(
    rows: Sequence[dict],
    *,
    expected_tasks: dict[str, list[str]],
    seeds: Sequence[int],
    lengths: Sequence[int],
) -> list[dict]:
    """Descriptive fixed-seed candidate-minus-BF16 means, clustered by task.

    Equal-size clusters first average all fixed seeds within each task. Task
    clusters are assumed independent and identically distributed; dependencies
    between the listed seeds within a task are unrestricted. These intervals do
    not change the primary decision and make no future-seed claim.
    """
    index = _index_pairs(rows, expected_tasks, seeds, lengths)
    output = []
    for benchmark, task_ids in expected_tasks.items():
        for length in lengths:
            missing = [
                {"task_id": task_id, "seed": seed}
                for task_id in task_ids
                for seed in seeds
                if (benchmark, length, seed, task_id) not in index
            ]
            cell = {
                "benchmark": benchmark,
                "length": length,
                "n_tasks": len(task_ids),
                "seed_count": len(seeds),
                "seeds": list(seeds),
                "descriptive_only": True,
                "units": "candidate minus BF16 success probability",
                "cluster_assumption": (
                    "Independent identically distributed task clusters; fixed seeds"
                ),
                "missing_identities": missing,
            }
            if missing:
                cell.update(
                    status="INCOMPLETE_PRIMARY_PAIRS",
                    mean_difference=None,
                    standard_error=None,
                    confidence_interval=None,
                )
            else:
                differences = [
                    math.fsum(
                        int(index[(benchmark, length, seed, task_id)]["candidate_pass"])
                        - int(index[(benchmark, length, seed, task_id)]["baseline_pass"])
                        for seed in seeds
                    )
                    / len(seeds)
                    for task_id in task_ids
                ]
                cell.update(_t_summary(differences))
            output.append(cell)
    return output


def summarize_nll(
    rows: Sequence[dict],
    *,
    expected_tasks: dict[str, list[str]],
    seeds: Sequence[int],
    lengths: Sequence[int],
) -> list[dict]:
    """Paired task-mean NLL differences with separate token-level totals."""
    index = {}
    task_sets = {endpoint: set(identifiers) for endpoint, identifiers in expected_tasks.items()}
    for row in rows:
        endpoint = row.get("endpoint")
        _require(
            endpoint in expected_tasks
            and row.get("seed") in seeds
            and row.get("length") in lengths
            and row.get("task_id") in task_sets[endpoint]
            and row.get("arm") in ARMS,
            "unexpected NLL identity",
        )
        _integer(row.get("seed"), "NLL seed")
        _integer(row.get("length"), "NLL length", 1)
        key = (endpoint, row["length"], row["seed"], row["task_id"], row["arm"])
        _require(key not in index, "duplicate NLL task/arm observation")
        count = _integer(row.get("n_targets"), "NLL target count", 1)
        mean = _number(row.get("nll_mean"), "NLL mean", 0)
        total = _number(row.get("nll_sum"), "NLL sum", 0)
        _require(
            math.isclose(total / count, mean, rel_tol=1e-10, abs_tol=1e-10),
            "inconsistent NLL task mean",
        )
        index[key] = row
    output = []
    for endpoint, task_ids in expected_tasks.items():
        for length in lengths:
            for seed in seeds:
                missing = []
                for task_id in task_ids:
                    arms = [
                        arm for arm in ARMS if (endpoint, length, seed, task_id, arm) not in index
                    ]
                    if arms:
                        missing.append({"task_id": task_id, "missing_arms": arms})
                cell = {
                    "benchmark": endpoint,
                    "length": length,
                    "seed": seed,
                    "expected_n": len(task_ids),
                    "descriptive_only": True,
                    "units": "candidate minus BF16 task-mean NLL, nats per supervised token",
                    "missing_identities": missing,
                }
                if missing:
                    cell.update(
                        status="INCOMPLETE_NLL",
                        mean_difference=None,
                        baseline_mean=None,
                        candidate_mean=None,
                        standard_error=None,
                        confidence_interval=None,
                        relative_mean_change=None,
                    )
                else:
                    paired = [
                        [index[(endpoint, length, seed, task_id, arm)] for arm in ARMS]
                        for task_id in task_ids
                    ]
                    cell.update(
                        _t_summary(
                            [
                                candidate["nll_mean"] - baseline["nll_mean"]
                                for baseline, candidate in paired
                            ]
                        )
                    )
                    cell["n"] = len(paired)
                    for arm_index, label in enumerate(("baseline", "candidate")):
                        mean = math.fsum(pair[arm_index]["nll_mean"] for pair in paired) / len(
                            paired
                        )
                        total = math.fsum(pair[arm_index]["nll_sum"] for pair in paired)
                        count = sum(pair[arm_index]["n_targets"] for pair in paired)
                        cell.update(
                            {
                                label + "_mean": mean,
                                label + "_nll_sum": total,
                                label + "_n_targets": count,
                                label + "_token_weighted_mean": total / count,
                            }
                        )
                    baseline, candidate = cell["baseline_mean"], cell["candidate_mean"]
                    cell["relative_mean_change"] = (
                        (candidate - baseline) / baseline if baseline else None
                    )
                    cell["relative_mean_change_status"] = (
                        "DEFINED" if baseline else "UNDEFINED_BASELINE_ZERO"
                    )
                output.append(cell)
    return output


def _paired_rows(observations: Sequence[dict], protocol: dict) -> list[dict]:
    index = {
        (row["endpoint"], row["length"], row["seed"], row["task_id"], row["arm"]): row
        for row in observations
    }
    rows = []
    for endpoint in protocol["primary_endpoints"]:
        for length in protocol["lengths"]:
            for seed in protocol["seeds"]:
                for task_id in protocol["expected_tasks"][endpoint]:
                    baseline = index.get((endpoint, length, seed, task_id, ARMS[0]))
                    candidate = index.get((endpoint, length, seed, task_id, ARMS[1]))
                    _require(
                        (baseline is None) == (candidate is None), "unbound single-arm observation"
                    )
                    if baseline is not None:
                        rows.append(
                            {
                                "benchmark": endpoint,
                                "length": length,
                                "seed": seed,
                                "task_id": task_id,
                                "baseline_pass": baseline["pass"],
                                "candidate_pass": candidate["pass"],
                            }
                        )
    return rows


def _likelihood_consistency(observations: Sequence[dict], tasks: dict) -> dict:
    groups = {}
    for row in observations:
        if row["endpoint"] not in PRIMARY:
            continue
        task = tasks[row["endpoint"]][row["task_id"]]
        prompt = {field: task[field] for field in ("context", "choices", "target") if field in task}
        key = (
            row["endpoint"],
            row["length"],
            row["task_id"],
            row["source_ref"],
            row["source_row_sha256"],
            _compact(prompt),
            _compact(row["results"]),
        )
        verdict = (row["metric"], row["score"], row["pred"], row["pred_norm"], row["pass"])
        groups.setdefault(key, []).append(verdict)
    matched = [values for values in groups.values() if len(values) > 1]
    consistent = all(all(verdict == values[0] for verdict in values[1:]) for values in matched)
    _require(consistent, "byte-identical likelihood payloads have inconsistent verdicts")
    return {
        "consistent": consistent,
        "matched_groups": len(matched),
        "matched_observations": sum(len(values) for values in matched),
        "comparisons": sum(len(values) - 1 for values in matched),
        "status": "CONSISTENT" if matched else "NO_MATCHING_PAYLOADS",
        "identity_policy": (
            "Equal endpoint, length, task, source bytes and prompt; compact result bytes"
        ),
        "generated_completion_consistency_claimed": False,
    }


def _resources(controllers: Sequence[dict], batches: list[dict]) -> dict:
    wall = math.fsum(controller["controller_wall_seconds"] for controller in controllers)
    explicit_gpu_hours = [controller.get("gpu_hours_used") for controller in controllers]
    result = {
        "controller_wall_seconds_total": wall,
        "controller_session_hours_total": wall / 3600,
        "controllers": [
            {
                "root": controller["root"],
                "status": controller["status"],
                "controller_wall_seconds": controller["controller_wall_seconds"],
                "finished_at": controller.get("finished_at"),
                "cleanup_complete": controller["cleanup_complete"],
            }
            for controller in controllers
        ],
        "commands": [
            {**command, "controller_root": controller["root"]}
            for controller in controllers
            for command in controller["commands"]
        ],
        "batches": batches,
        "gpu_hours_used": None,
        "natural_rmr": None,
        "scope": "Controller/session wall time is not billed GPU time; no RAM plus VRAM total",
        "rss_peak_scope": "Whole-process sampled lower bound; short-lived peaks may be missed",
    }
    if all(value is not None for value in explicit_gpu_hours):
        result["gpu_hours_used"] = math.fsum(
            _number(value, "recorded GPU hours", 0) for value in explicit_gpu_hours
        )
    for field in _RESOURCE_FIELDS:
        values = [
            batch["measurement"][field]
            for batch in batches
            if batch["measurement"].get(field) is not None
        ]
        result[field + "_max"] = max(values, default=None)
        result[field + "_missing_batches"] = len(batches) - len(values)
    return result


def _display(value: Any) -> str:
    if value is None:
        return "not computed"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: list[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(_display(value) for value in row) + " |" for row in rows],
        "",
    ]


def _markdown(summary: dict) -> str:
    missing_count = len(summary["coverage"]["missing_identities"])
    lines = [
        "# Stage 2 v2 results",
        "",
        "This is an outcome-informed endpoint redesign after v1 floors. "
        "It does not complete v1; v1 remains `BLOCKED_INCOMPLETE_COVERAGE`.",
        "",
        f"Frozen protocol file SHA-256: `{summary['protocol_sha256']}`.",
        "",
        f"Final decision: `{summary['decision']}`. "
        f"Unchanged statistical decision: `{summary['statistical_decision']}`.",
        "",
        "All 40 frozen primary cells remain required, including floor cells. "
        "The margin is 0.01 absolute and one-sided alpha is 0.025. "
        "Missing coverage prevents all partial-set primary statistics. "
        "Noninferiority requires every pointwise upper bound U strictly below the margin. "
        "Excessive degradation requires a familywise-adjusted lower bound strictly above it. "
        "The unchanged Clopper-Pearson marginal bounds and union bounds condition on the "
        "declared within-cell independent, identically distributed task-pair model, "
        "not independent seeds or independent seed-by-task observations. "
        "Frozen finite-set accuracies are exact; the intervals do not repair "
        "outcome-informed selection or contamination.",
        "",
        "A primary cell is informative exactly when BF16 accuracy is at least frozen "
        "chance plus 0.05. Any floor cell vetoes global noninferiority, but never vetoes "
        "excessive degradation. No floor cell is dropped.",
        "",
        "## Primary cells",
        "",
    ]
    lines += _table(
        [
            "Endpoint",
            "Length",
            "Seed",
            "Expected n",
            "Observed n",
            "BF16 accuracy",
            "Candidate accuracy",
            "Harms",
            "Gains",
            "Pointwise U",
            "Cell result",
            "Informative",
        ],
        [
            [
                cell["benchmark"],
                cell["length"],
                cell["seed"],
                cell.get("expected_n", cell.get("n")),
                cell.get("observed_n", cell.get("n")),
                cell["baseline_accuracy"],
                cell["candidate_accuracy"],
                cell["harms"],
                cell["gains"],
                cell["pointwise_bounds"]["upper"] if cell["pointwise_bounds"] else None,
                cell["cell_result"],
                cell["informative"],
            ]
            for cell in summary["cells"]
        ],
    )
    lines += [
        f"Missing primary task-pair identities: {missing_count}. "
        "The exact identities and missing arms are in summary.json.",
        "",
        "## Fixed-seed means",
        "",
        "Equal-weight means over the five frozen trained seed pairs. "
        "These are not future-seed estimates or pooled independent task counts.",
        "",
    ]
    lines += _table(
        [
            "Endpoint",
            "Length",
            "BF16 mean accuracy",
            "Candidate mean accuracy",
            "Mean degradation, BF16 minus candidate",
        ],
        [
            [
                cell["benchmark"],
                cell["length"],
                cell["mean"]["baseline_accuracy"],
                cell["mean"]["candidate_accuracy"],
                cell["mean"]["degradation"],
            ]
            for cell in summary["endpoints"]
        ],
    )
    lines += [
        "## Task-clustered success differences",
        "",
        "Descriptive only. Average the five fixed seeds within each task first, then use "
        "sample SD of task means divided by sqrt(number of tasks). The two-sided 95% "
        "Student t interval has df = tasks minus one. It assumes independent, identically "
        "distributed task clusters, allows within-task seed dependence and does not change "
        "the primary decision. Differences are candidate minus BF16.",
        "",
    ]
    lines += _table(
        [
            "Endpoint",
            "Length",
            "Tasks",
            "Seeds",
            "Mean difference",
            "Cluster SE",
            "95% t interval",
            "Status",
        ],
        [
            [
                cell["benchmark"],
                cell["length"],
                cell["n_tasks"],
                cell["seed_count"],
                cell["mean_difference"],
                cell["standard_error"],
                cell["confidence_interval"],
                cell["status"],
            ]
            for cell in summary["clustered_success"]
        ],
    )
    lines += [
        "## Held-out NLL",
        "",
        "Descriptive only, in nats per supervised token. NLL summaries average "
        "task means equally, with a paired two-sided 95% Student t interval for candidate "
        "minus BF16. All secondary task pairs must be present. Relative change is "
        "candidate task mean divided by BF16 task mean minus one and is undefined at a "
        "zero BF16 mean. Token sums, counts and token-weighted means remain separate in "
        "summary.json and never replace the task mean.",
        "",
    ]
    lines += _table(
        [
            "Endpoint",
            "Length",
            "Seed",
            "BF16 task mean",
            "Candidate task mean",
            "Mean difference",
            "95% t interval",
            "Relative mean change",
            "Status",
        ],
        [
            [
                cell["benchmark"],
                cell["length"],
                cell["seed"],
                cell["baseline_mean"],
                cell["candidate_mean"],
                cell["mean_difference"],
                cell["confidence_interval"],
                cell["relative_mean_change"],
                cell["status"],
            ]
            for cell in summary["nll"]
        ],
    )
    consistency = summary["likelihood_verdict_consistency"]
    lines += [
        "## Likelihood verdict consistency",
        "",
        f"S7 adaptation: {consistency['matched_groups']} groups, "
        f"{consistency['comparisons']} comparisons, status `{consistency['status']}`. "
        "Only equal source and prompt identities with byte-identical compact likelihood "
        "result payloads are compared, within the same length. No generated-completion "
        "consistency claim is made because this evaluation does not generate completions.",
        "",
        "## Resource facts",
        "",
        "Controller wall duration and its session-hour conversion are recorded duration, "
        "not billed GPU-h. CUDA allocated, CUDA reserved and sampled process RSS peaks are "
        "separate whole-process observations. Do not add RAM and VRAM. Forced Oracle index "
        "agreement is not natural RMR, which is not measured here.",
        "",
    ]
    resources = summary["resources"]
    lines += _table(
        ["Resource", "Recorded value", "Unit"],
        [
            ["Controller wall total", resources["controller_wall_seconds_total"], "seconds"],
            [
                "Controller session duration total",
                resources["controller_session_hours_total"],
                "hours",
            ],
            ["GPU-h used", resources["gpu_hours_used"], "explicit recorded GPU-h only"],
            [
                "Maximum per-batch CUDA allocated peak",
                resources["cuda_allocated_peak_bytes_max"],
                "bytes",
            ],
            [
                "Maximum per-batch CUDA reserved peak",
                resources["cuda_reserved_peak_bytes_max"],
                "bytes",
            ],
            [
                "Maximum per-batch sampled process RSS",
                resources["process_rss_sampled_peak_bytes_max"],
                "bytes",
            ],
        ],
    )
    lines += _table(
        ["Controller", "Status", "Wall seconds", "Cleanup complete"],
        [
            [row["root"], row["status"], row["controller_wall_seconds"], row["cleanup_complete"]]
            for row in resources["controllers"]
        ],
    )
    lines += _table(
        ["Controller", "Command", "Seconds", "Return code", "Interruption"],
        [
            [
                row["controller_root"],
                row["name"],
                row["seconds"],
                row.get("returncode"),
                row.get("interruption"),
            ]
            for row in resources["commands"]
        ],
    )
    lines += _table(
        [
            "Controller",
            "Endpoint",
            "Arm",
            "Length",
            "Seed",
            "Batch",
            "Analysis pair",
            "CUDA allocated peak bytes",
            "CUDA reserved peak bytes",
            "Sampled RSS peak bytes",
            "Batch wall seconds",
        ],
        [
            [
                row["controller_root"],
                row["endpoint"],
                row["arm"],
                row["length"],
                row["seed"],
                row["batch_index"],
                row["admitted_for_analysis"],
                row["measurement"].get("cuda_allocated_peak_bytes"),
                row["measurement"].get("cuda_reserved_peak_bytes"),
                row["measurement"].get("process_rss_sampled_peak_bytes"),
                row["measurement"].get("wall_seconds"),
            ]
            for row in resources["batches"]
        ],
    )
    lines += [
        "Every consumed batch measurement and its shard/protocol/checkpoint identity "
        "is retained in summary.json under resources.batches. Completed arms from "
        "incomplete pairs have analysis-pair=false and contribute no score observations. "
        "Unreceipted failed-shard files are not read or admitted. Failed command times "
        "and controller records remain reported. Sampled RSS is a lower bound, and "
        "short-lived peaks may be missed.",
        "",
        "## Retained failed attempts",
        "",
    ]
    lines += _table(
        ["Controller", "Status", "Error type", "Error", "Incomplete pairs", "Failed commands"],
        [
            [
                row["controller_root"],
                row["status"],
                row["error_type"],
                row["error"],
                len(row["pairs"]),
                len(row["failed_commands"]),
            ]
            for row in summary["failure_attempts"]
        ],
    )
    lines += [
        "No duplicate successful task/arm observation is selected over another. "
        "Only whole, complete admitted pairs contribute results. All supplied run records, "
        "including incomplete pairs and failed command receipts, are retained in summary.json.",
        "",
    ]
    return "\n".join(lines)


def _write(root: Path, name: str, text: str) -> dict:
    path = _relative(root, name)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    size, digest = _file_hash(path)
    return {"bytes": size, "sha256": digest}


def _stage_manifest(root: Path, manifest: dict, protocol: dict) -> tuple[Path, Path, int, str]:
    staged = _relative(root, "manifest.staging.json")
    destination = _relative(root, "manifest.json")
    payload = _compact(manifest) + "\n"
    expected_size, expected_hash = len(payload.encode("utf-8")), _text_hash(payload)
    with staged.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    _require(
        _file_hash(staged) == (expected_size, expected_hash),
        "terminal manifest staging verification mismatch",
    )
    _verify_sources(protocol)
    return staged, destination, expected_size, expected_hash


def assemble_results(
    *,
    protocol_path: Path,
    protocol_sha256: str,
    data_root: Path,
    controller_roots: Sequence[Path],
    output_dir: Path,
) -> dict:
    """Admit collected evidence and exclusively publish an immutable report directory.

    Successful assembly, even when coverage is blocked, writes manifest.json last.
    Malformed, changed or ambiguous evidence writes failure-manifest.json instead.
    An existing output directory is never touched.
    """
    protocol_path, data_root, output = _real(protocol_path), _real(data_root), _real(output_dir)
    roots = [_real(root) for root in controller_roots]
    if not roots:
        raise AdmissionError("at least one controller root is required")
    if _contained(data_root, output) or any(_contained(root, output) for root in roots):
        raise AdmissionError("fresh output must not be inside immutable input roots")
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise AdmissionError("fresh output directory required: " + str(error)) from None
    inputs: dict[str, Any] = {}
    files = {}
    commit: tuple[Path, Path, int, str] | None = None
    try:
        protocol, inputs["protocol"] = _read_json(protocol_path, protocol_sha256, "protocol")
        _validate_protocol(protocol)
        inputs["analysis_sources"] = _verify_sources(protocol)
        tasks, task_files = _read_tasks(data_root, protocol, inputs)
        observations, controllers, failures, batches = _collect(
            roots, protocol, protocol_sha256, tasks, task_files, inputs
        )
        paired = _paired_rows(observations, protocol)
        _verify_sources(protocol)
        primary = summarize_primary(paired, protocol)
        summary = {
            **primary,
            "schema": "p3-stage2-v2-results-v1",
            "assembly_status": "COMPLETE",
            "protocol_sha256": protocol_sha256,
            "inputs": inputs,
            "controllers": controllers,
            "failure_attempts": failures,
            "outcome_informed_redesign_after_v1_floors": True,
            "v1_decision": "BLOCKED_INCOMPLETE_COVERAGE",
            "clustered_success": summarize_clustered_success(
                paired,
                expected_tasks={name: protocol["expected_tasks"][name] for name in PRIMARY},
                seeds=protocol["seeds"],
                lengths=protocol["lengths"],
            ),
            "nll": summarize_nll(
                [row for row in observations if row["endpoint"] in SECONDARY],
                expected_tasks={name: protocol["expected_tasks"][name] for name in SECONDARY},
                seeds=protocol["seeds"],
                lengths=protocol["lengths"],
            ),
            "likelihood_verdict_consistency": _likelihood_consistency(observations, tasks),
            "resources": _resources(controllers, batches),
        }
        files["paired-rows.jsonl"] = {
            **_write(output, "paired-rows.jsonl", "".join(_compact(row) + "\n" for row in paired)),
            "rows": len(paired),
        }
        files["summary.json"] = _write(output, "summary.json", _compact(summary) + "\n")
        files["results-v2.md"] = _write(output, "results-v2.md", _markdown(summary))
        manifest = {
            "schema": "p3-stage2-v2-results-manifest-v1",
            "status": "COMPLETE",
            "protocol_sha256": protocol_sha256,
            "decision": summary["decision"],
            "statistical_decision": summary["statistical_decision"],
            "inputs": inputs,
            "files": files,
            "outcome_informed_redesign_after_v1_floors": True,
            "v1_decision": "BLOCKED_INCOMPLETE_COVERAGE",
        }
        manifest["manifest_payload_sha256"] = _text_hash(_compact(manifest))
        commit = _stage_manifest(output, manifest, protocol)
        # Keep staging evidence. This exclusive link is the final fallible
        # operation on the normal success path.
        os.link(commit[0], commit[1])
        return summary
    except BaseException as error:
        if commit is not None:
            try:
                committed = _file_hash(commit[1]) == (commit[2], commit[3])
            except FileNotFoundError:
                committed = False
            except Exception:
                # Publication may have succeeded before an interruption. If its
                # status cannot be inspected, never publish contradictory FAILED.
                if not isinstance(error, Exception):
                    raise error
                raise AdmissionError(
                    "terminal publication status is unknown after interruption"
                ) from None
            if committed:
                if not isinstance(error, Exception):
                    raise error
                return summary
        failure = {
            "schema": "p3-stage2-v2-results-manifest-v1",
            "status": "FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
            "inputs": inputs,
            "protocol_sha256": protocol_sha256,
            "controller_roots": [str(root) for root in roots],
            "files": files,
            "quality_admitted": False,
            "outcome_informed_redesign_after_v1_floors": True,
            "v1_decision": "BLOCKED_INCOMPLETE_COVERAGE",
        }
        failure["manifest_payload_sha256"] = _text_hash(_compact(failure))
        _write(output, "failure-manifest.json", _compact(failure) + "\n")
        if not isinstance(error, Exception):
            raise
        raise AdmissionError(str(error), output / "failure-manifest.json") from None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-path", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--controller-root",
        type=Path,
        action="append",
        required=True,
        help="Collected controller directory containing run.json; repeat for each job",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Fresh, nonexistent output directory"
    )
    arguments = parser.parse_args(argv)
    console = Console(width=100)
    try:
        summary = assemble_results(
            protocol_path=arguments.protocol_path,
            protocol_sha256=arguments.protocol_sha256,
            data_root=arguments.data_root,
            controller_roots=arguments.controller_root,
            output_dir=arguments.output,
        )
    except AdmissionError as error:
        console.print(
            _compact(
                {
                    "status": "FAILED",
                    "error": str(error),
                    "failure_manifest": str(error.report_path) if error.report_path else None,
                }
            ),
            markup=False,
            soft_wrap=True,
        )
        return 1
    console.print(
        _compact(
            {
                "status": summary["assembly_status"],
                "decision": summary["decision"],
                "statistical_decision": summary["statistical_decision"],
                "output": str(_real(arguments.output)),
            }
        ),
        markup=False,
        soft_wrap=True,
    )
    return 0 if summary["coverage"]["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
