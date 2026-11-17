"""Synthetic immutable controller evidence and descriptive v2 statistical boundaries.

No checkpoint, model, tokenizer, source download or live outcome is used here.
The integration owner executes these tests on Kaggle, not on the editing host.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path
from typing import Callable

import pytest

from benchmarks.harness import p3_stage2v2_results as results
from benchmarks.harness.p3_stage2v2_data import select_hellaswag_tasks
from benchmarks.harness.p3_stage2v2_loglik import SOURCE_PATHS

ARMS = ["bf16_l2l", "oracle_block64_e4m3"]
SEEDS = [23, 41, 67, 89, 113]
LENGTHS = [1024, 4096]
PRIMARY = ["hellaswag", "arc", "boolq", "lambada"]
SECONDARY = ["gsm8k", "mbpp"]
AUDIT = "a" * 64
V1_PROTOCOL = "170c0b700c137c1f6e0c6788376927e4aaa4e99866b990bc83d0afc42a62bc3c"
BASE_REVISION = "54be201337cb7ac8ba5e58ba9ff5b55ac7b0bc81"
RUNNER = "benchmarks/results/p3-stage2-v2/orchestration/stage2v2_eval_runner.py"
SOURCE_NAMES = set(SOURCE_PATHS) | {
    RUNNER,
    "benchmarks/harness/p3_stage2v2_results.py",
    "benchmarks/harness/p3_stage2v2_decision.py",
    "benchmarks/harness/p3_stage2_statistics.py",
    "benchmarks/harness/p3_stage2_apps.py",
    "benchmarks/harness/p3_stage2_math.py",
    "benchmarks/harness/p3_stage2_core_data.py",
}


@pytest.fixture(autouse=True)
def _isolated_analysis_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository = Path(results.__file__).parents[2]
    source_root = tmp_path / "analysis-source"
    for name in SOURCE_NAMES:
        destination = source_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((repository / name).read_bytes())
    monkeypatch.setattr(results, "_SOURCE_ROOT", source_root)


def _source_files() -> dict[str, str]:
    return {
        name: _digest((results._SOURCE_ROOT / name).read_bytes()) for name in sorted(SOURCE_NAMES)
    }


def _compact(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _save(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_compact(value) + "\n", encoding="utf-8")
    return _digest(path.read_bytes())


def _save_rows(path: Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(_compact(row) + "\n" for row in rows), encoding="utf-8")
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": _digest(payload)}


def _payload_hash(value: dict, field: str) -> dict:
    payload = {key: item for key, item in value.items() if key != field}
    return {**payload, field: _digest(_compact(payload).encode("utf-8"))}


def _task(endpoint: str, index: int) -> dict:
    task_id = f"{endpoint}-task-{index}"
    row = {
        "endpoint": endpoint,
        "role": "evaluation",
        "task_id": task_id,
        "source_ref": task_id,
        "source_row_sha256": _digest(task_id.encode("utf-8")),
        "context": f"A synthetic question with caf\u00e9 {task_id}",
        "metric": "acc_norm" if endpoint in {"hellaswag", "arc"} else "acc",
    }
    if endpoint in SECONDARY:
        row.update(metric="nll", target="reference", gold="reference")
    elif endpoint == "lambada":
        row.update(target=" word", gold=" word")
    else:
        choices = ["no", "yes"] if endpoint == "boolq" else ["aaaa", "bbbb", "cccc", "dddd"]
        row.update(choices=choices, gold=index % 2 if endpoint == "boolq" else 0)
    return row


def _protocol(tasks: dict[str, list[dict]], data_hash: str) -> dict:
    boolq_labels = [row["gold"] for row in tasks["boolq"]]
    boolq_chance = max(boolq_labels.count(0), boolq_labels.count(1)) / len(boolq_labels)
    return {
        "schema": "p3-stage2-v2-protocol",
        "status": "FROZEN",
        "margin": 0.01,
        "base_revision": BASE_REVISION,
        "v1_protocol_sha256": V1_PROTOCOL,
        "one_sided_alpha": 0.025,
        "arms": ARMS,
        "seeds": SEEDS,
        "lengths": LENGTHS,
        "primary_endpoints": PRIMARY,
        "secondary_endpoints": SECONDARY,
        "chance": {"hellaswag": 0.25, "arc": 0.25, "boolq": boolq_chance, "lambada": 0.0},
        "required_primary_cells": [
            {"benchmark": endpoint, "length": length, "seed": seed}
            for endpoint in PRIMARY
            for length in LENGTHS
            for seed in SEEDS
        ],
        "expected_tasks": {name: [row["task_id"] for row in rows] for name, rows in tasks.items()},
        "execution_pairs": [
            {
                "endpoint": endpoint,
                "length": length,
                "seed": seed,
                "start": 0,
                "stop": len(tasks[endpoint]),
                "arms": ARMS,
            }
            for length in LENGTHS
            for seed in SEEDS
            for endpoint in PRIMARY + SECONDARY
        ],
        "data": {
            "dataset_manifest_file_sha256": data_hash,
            "retained_evaluation_tasks": {name: len(rows) for name, rows in tasks.items()},
            "full_retained_evaluation_tasks": {name: len(rows) for name, rows in tasks.items()},
            "ordered_task_ids_sha256": {
                name: _digest(_compact([row["task_id"] for row in rows]).encode("utf-8"))
                for name, rows in tasks.items()
            },
            "source_ordered_task_ids_sha256": {
                name: _digest(_compact([row["task_id"] for row in rows]).encode("utf-8"))
                for name, rows in tasks.items()
            },
        },
        "training_audit_sha256": AUDIT,
        "source_files_sha256": _source_files(),
        "checkpoints": {
            f"L{length}-seed{seed}-{arm}": {
                "arm": arm,
                "length": length,
                "seed": seed,
                "directory": f"L{length}-seed{seed}-{arm}",
                "protocol_sha256": V1_PROTOCOL,
                "final_step64": {
                    "file": "step-064.adapters.safetensors",
                    "bytes": 123,
                    "path": f"L{length}-seed{seed}-{arm}/arm/step-064.adapters.safetensors",
                    "sha256": _digest(f"{length}-{seed}-{arm}".encode("utf-8")),
                    "tensor_manifest": {"adapter": {"shape": [2, 2], "dtype": "float32"}},
                    "metadata": {"seed": str(seed), "protocol_sha256": V1_PROTOCOL},
                },
            }
            for length in LENGTHS
            for seed in SEEDS
            for arm in ARMS
        },
        "outcome_informed_redesign_after_v1_floors": True,
        "v1_decision": "BLOCKED_INCOMPLETE_COVERAGE",
    }


class _Evidence:
    """Test-only writer, independent of production admission and hashing helpers."""

    def __init__(
        self,
        root: Path,
        primary_n: int = 4,
        secondary_n: int = 2,
        *,
        hellaswag_n: int | None = None,
    ) -> None:
        self.data = root / "data"
        self.controller = root / "controller"
        self.output = root / "assembled"
        self.task_files: dict[str, dict] = {}
        self.tasks = {
            endpoint: [
                _task(endpoint, index)
                for index in range(primary_n if endpoint in PRIMARY else secondary_n)
            ]
            for endpoint in PRIMARY + SECONDARY
        }
        self.selected_tasks = dict(self.tasks)
        if hellaswag_n is not None and hellaswag_n < len(self.tasks["hellaswag"]):
            self.selected_tasks["hellaswag"] = select_hellaswag_tasks(
                self.tasks["hellaswag"], hellaswag_n
            )
        endpoints = {}
        files = {}
        for endpoint, rows in self.tasks.items():
            relative = f"{endpoint}/tasks.jsonl"
            descriptor = {
                "path": relative,
                **_save_rows(self.data / relative, rows),
                "rows": len(rows),
            }
            files[f"{endpoint}_tasks"] = descriptor
            self.task_files[endpoint] = descriptor
            endpoints[endpoint] = {
                "endpoint": endpoint,
                "role": "evaluation",
                "metric": rows[0]["metric"],
                "retained_row_count": len(rows),
                "retained_task_ids": [row["task_id"] for row in rows],
                "ordered_task_ids_sha256": _digest(
                    _compact([row["task_id"] for row in rows]).encode("utf-8")
                ),
                "files": {"tasks": descriptor},
            }
        manifest = _payload_hash(
            {
                "schema": "p3-stage2-v2-data-v1",
                "status": "DATA_ADMITTED_PENDING_PROTOCOL_FREEZE",
                "endpoints": endpoints,
                "files": files,
            },
            "manifest_payload_sha256",
        )
        self.protocol = _protocol(self.selected_tasks, _save(self.data / "manifest.json", manifest))
        self.protocol["data"]["full_retained_evaluation_tasks"] = {
            name: len(rows) for name, rows in self.tasks.items()
        }
        self.protocol["data"]["source_ordered_task_ids_sha256"] = {
            name: _digest(_compact([row["task_id"] for row in rows]).encode("utf-8"))
            for name, rows in self.tasks.items()
        }
        self.protocol_path = root / "protocol.json"
        self.protocol_hash = _save(self.protocol_path, self.protocol)
        self.scorer_sources = {
            name: self.protocol["source_files_sha256"][name] for name in SOURCE_PATHS
        }
        self.run = {
            "schema": "p3-stage2v2-controller-v1",
            "phase": "evaluate",
            "status": "PLANNED_PAIRS_COMPLETE_NOT_GLOBAL_ADMISSION",
            "protocol_sha256": self.protocol_hash,
            "cleanup_complete": True,
            "base_revision": BASE_REVISION,
            "controller_wall_seconds": 3600.0,
            "finished_at": 2000000000.0,
            "commands": [{"name": "synthetic-command", "seconds": 5.5, "returncode": 0}],
            "pairs": [],
        }
        for number, pair in enumerate(self.protocol["execution_pairs"]):
            receipts = []
            for arm in ARMS:
                folder = self.controller / "shards" / f"{number:03d}-{arm}"
                run_key = f"L{pair['length']}-seed{pair['seed']}-{arm}"
                checkpoint_descriptor = self.protocol["checkpoints"][run_key]["final_step64"]
                checkpoint = checkpoint_descriptor["sha256"]
                rows = []
                for index, task in enumerate(self.selected_tasks[pair["endpoint"]]):
                    row = {
                        "schema": "p3-stage2-v2-loglik-row-v1",
                        "endpoint": pair["endpoint"],
                        "arm": arm,
                        "seed": pair["seed"],
                        "length": pair["length"],
                        "protocol_sha256": self.protocol_hash,
                        "checkpoint_sha256": checkpoint,
                        "audit_sha256": AUDIT,
                        "task_index": index,
                        "task_id": task["task_id"],
                        "source_ref": task["source_ref"],
                        "source_row_sha256": task["source_row_sha256"],
                        "metric": task["metric"],
                        "gold": task["gold"],
                    }
                    if pair["endpoint"] in SECONDARY:
                        count = 1 if index == 0 else 9
                        mean = float(index * 2 + 1) * (1 if arm == ARMS[0] else 2)
                        row.update(score=mean, nll_mean=mean, n_targets=count, nll_sum=mean * count)
                    else:
                        requests = 1 if pair["endpoint"] == "lambada" else len(task["choices"])
                        prediction = task["gold"] if type(task["gold"]) is int else 0
                        row.update(
                            score=1.0,
                            pred=prediction,
                            pred_norm=prediction,
                            **{"pass": True},
                            results=[
                                {
                                    "logprob": -1.0 if choice == prediction else -5.0,
                                    "greedy": choice == prediction,
                                    "n_tokens": 1,
                                }
                                for choice in range(requests)
                            ],
                        )
                    rows.append(_payload_hash(row, "row_sha256"))
                score_file = _save_rows(folder / "scores.jsonl", rows)
                binding = {
                    "data_manifest_sha256": self.protocol["data"]["dataset_manifest_file_sha256"],
                    "complete_endpoint_row_count": len(self.selected_tasks[pair["endpoint"]]),
                    "start": pair["start"],
                    "stop": pair["stop"],
                    "ordered_task_ids": [row["task_id"] for row in rows],
                    "selected_rows_sha256": _digest(
                        _compact(self.selected_tasks[pair["endpoint"]]).encode("utf-8")
                    ),
                    "tasks_file": self.task_files[pair["endpoint"]],
                }
                admission = {
                    "schema": "p3-stage2-v2-loglik-admission-v1",
                    "endpoint": pair["endpoint"],
                    "arm": arm,
                    "seed": pair["seed"],
                    "length": pair["length"],
                    "protocol_sha256": self.protocol_hash,
                    "audit_sha256": AUDIT,
                    "checkpoint_sha256": checkpoint,
                    **binding,
                    "checkpoint": checkpoint_descriptor,
                    "training_run_key": run_key,
                    "training_audit": {
                        "path": "/historical/training-audit.json",
                        "bytes": 1234,
                        "sha256": AUDIT,
                    },
                    "v1_protocol_sha256": V1_PROTOCOL,
                    "protocol_file": {
                        "path": "/historical/p3-stage2-v2-protocol.json",
                        "bytes": len(self.protocol_path.read_bytes()),
                        "sha256": self.protocol_hash,
                    },
                    "source_files_sha256": self.scorer_sources,
                }
                _save(folder / "admission.json", admission)
                admission_payload = (folder / "admission.json").read_bytes()
                admission_file = {
                    "bytes": len(admission_payload),
                    "sha256": _digest(admission_payload),
                }
                request_count = sum(len(row["results"]) if "results" in row else 1 for row in rows)
                scored_tokens = sum(
                    sum(result["n_tokens"] for result in row["results"])
                    if "results" in row
                    else row["n_targets"]
                    for row in rows
                )
                batch_file = _save_rows(
                    folder / "batches.jsonl",
                    [
                        {
                            "schema": "p3-stage2-v2-loglik-batch-v1",
                            "batch_index": 0,
                            "event": "batch_complete",
                            "endpoint": pair["endpoint"],
                            "request_indices": list(range(request_count)),
                            "input_tokens": request_count * 2,
                            "padded_input_tokens": request_count * 2,
                            "scored_tokens": scored_tokens,
                            "forward_calls": 1,
                            "logits_estimated_peak_bytes": 64,
                            "measurement": {
                                "cuda_allocated_peak_bytes": 100,
                                "cuda_reserved_peak_bytes": 200,
                                "process_rss_sampled_peak_bytes": 300,
                                "wall_seconds": 0.25,
                                "scope": "synthetic batch",
                            },
                        }
                    ],
                )
                measured = _payload_hash(
                    {
                        "schema": "p3-stage2-v2-loglik-run-v1",
                        "status": "COMPLETE",
                        "selected_shard_complete": True,
                        "endpoint": pair["endpoint"],
                        "arm": arm,
                        "seed": pair["seed"],
                        "length": pair["length"],
                        "protocol_sha256": self.protocol_hash,
                        "checkpoint_sha256": checkpoint,
                        "audit_sha256": AUDIT,
                        **binding,
                        "source_files_sha256": self.scorer_sources,
                        "start": pair["start"],
                        "stop": pair["stop"],
                        "requested_rows": len(rows),
                        "completed_rows": len(rows),
                        "batch_count": 1,
                        "request_count": request_count,
                        "scored_tokens": scored_tokens,
                        "cleanup": {"complete": True},
                        "ordered_task_ids": [row["task_id"] for row in rows],
                        "files": {
                            "scores.jsonl": score_file,
                            "batches.jsonl": batch_file,
                            "admission.json": admission_file,
                        },
                    },
                    "manifest_payload_sha256",
                )
                digest = _save(folder / "manifest.json", measured)
                receipts.append(
                    {
                        "arm": arm,
                        "manifest": f"shards/{number:03d}-{arm}/manifest.json",
                        "sha256": digest,
                    }
                )
            self.run["pairs"].append(
                {**pair, "arms": receipts, "status": "COMPLETE", "seconds": 1.0}
            )
        self.controller_artifacts(self.controller, self.run)
        self.save_run()

    def save_run(self) -> None:
        _save(self.controller / "run.json", self.run)

    def controller_artifacts(self, root: Path, run: dict, pairs: list[dict] | None = None) -> None:
        root.mkdir(parents=True, exist_ok=True)
        archive_payload = io.BytesIO()
        original_v1 = "benchmarks/results/p3-stage2-quality/frozen-study-protocol-v1.json"
        extras = {
            "tests/test_p3_stage2v2_results.py": Path(__file__).read_bytes(),
            original_v1: (Path(results.__file__).parents[2] / original_v1).read_bytes(),
        }
        overlay_pins = {
            **self.protocol["source_files_sha256"],
            **{name: _digest(payload) for name, payload in extras.items()},
        }
        with zipfile.ZipFile(archive_payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(overlay_pins):
                payload = (
                    extras[name] if name in extras else (results._SOURCE_ROOT / name).read_bytes()
                )
                archive.writestr(name, payload)
        overlay = archive_payload.getvalue()
        (root / "source-overlay.zip").write_bytes(overlay)
        inputs = {
            "PROTOCOL": {"pattern": "**/p3-stage2-v2-protocol.json", "sha256": self.protocol_hash},
            "AUDIT": {"pattern": "**/training-audit.json", "sha256": AUDIT},
            "DATA_MANIFEST": {
                "pattern": "**/manifest.json",
                "sha256": self.protocol["data"]["dataset_manifest_file_sha256"],
            },
            "MATRIX_MANIFEST": {"pattern": "**/run.json", "sha256": "e" * 64},
        }
        names = {
            "PROTOCOL": "p3-stage2-v2-protocol.json",
            "AUDIT": "training-audit.json",
            "DATA_MANIFEST": "manifest.json",
            "MATRIX_MANIFEST": "run.json",
        }
        admission = {
            "schema": "p3-stage2v2-execution-admission-v1",
            "phase": "evaluate",
            "gpu": True,
            "protocol_sha256": self.protocol_hash,
            "source_overlay_sha256": _digest(overlay),
            "source_files_sha256": overlay_pins,
            "inputs": inputs,
            "derived_paths": {
                "DATA": {"binding": "DATA_MANIFEST", "relative": "."},
                "MATRIX": {"binding": "MATRIX_MANIFEST", "relative": "."},
            },
            "execution_pairs": pairs if pairs is not None else self.protocol["execution_pairs"],
            "controller_limit_seconds": 7200,
            "per_shard_limit_seconds": 2700,
            "measured_pair_seconds": 1.0,
        }
        run.update(
            admission_sha256=_save(root / "execution-admission.json", admission),
            source_overlay_sha256=_digest(overlay),
            input_bindings={
                name: {"path": f"/kaggle/input/fixture/{name}/{names[name]}", **spec}
                for name, spec in inputs.items()
            },
        )
        (root / "stage2v2_eval_runner.py").write_bytes((results._SOURCE_ROOT / RUNNER).read_bytes())
        (root / "p3-stage2-v2-protocol.json").write_bytes(self.protocol_path.read_bytes())

    def mutate_admission(self, change: Callable[[dict], None], pair: int = 0, arm: int = 0) -> None:
        receipt = self.receipt(pair, arm)
        path = self.controller / receipt["manifest"]
        manifest = json.loads(path.read_text(encoding="utf-8"))
        admission_path = path.parent / "admission.json"
        admission = json.loads(admission_path.read_text(encoding="utf-8"))
        change(admission)
        _save(admission_path, admission)
        payload = admission_path.read_bytes()
        manifest["files"]["admission.json"] = {"bytes": len(payload), "sha256": _digest(payload)}
        receipt["sha256"] = _save(path, _payload_hash(manifest, "manifest_payload_sha256"))
        self.save_run()

    def mutate_batches(self, change: Callable[[dict, list[dict]], None]) -> None:
        receipt = self.receipt()
        path = self.controller / receipt["manifest"]
        manifest = json.loads(path.read_text(encoding="utf-8"))
        batches_path = path.parent / "batches.jsonl"
        batches = [
            json.loads(line) for line in batches_path.read_text(encoding="utf-8").splitlines()
        ]
        change(manifest, batches)
        manifest["files"]["batches.jsonl"] = _save_rows(batches_path, batches)
        receipt["sha256"] = _save(path, _payload_hash(manifest, "manifest_payload_sha256"))
        self.save_run()

    def mutate_controller_admission(self, change: Callable[[dict], None]) -> None:
        path = self.controller / "execution-admission.json"
        admission = json.loads(path.read_text(encoding="utf-8"))
        change(admission)
        self.run["admission_sha256"] = _save(path, admission)
        self.save_run()

    def mutate_overlay(self, name: str, *, rebind_member: bool) -> None:
        path = self.controller / "source-overlay.zip"
        with zipfile.ZipFile(path) as archive:
            payloads = {member: archive.read(member) for member in archive.namelist()}
        payloads[name] += b"\n# changed archived producer source\n"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for member, payload in payloads.items():
                archive.writestr(member, payload)
        path.write_bytes(buffer.getvalue())
        digest = _digest(path.read_bytes())
        self.run["source_overlay_sha256"] = digest

        def change(admission: dict) -> None:
            admission["source_overlay_sha256"] = digest
            if rebind_member:
                admission["source_files_sha256"][name] = _digest(payloads[name])

        self.mutate_controller_admission(change)

    def assemble(self, controllers: list[Path] | None = None) -> dict:
        return results.assemble_results(
            protocol_path=self.protocol_path,
            protocol_sha256=self.protocol_hash,
            data_root=self.data,
            controller_roots=controllers or [self.controller],
            output_dir=self.output,
        )

    def receipt(self, pair: int = 0, arm: int = 0) -> dict:
        return self.run["pairs"][pair]["arms"][arm]

    def mutate_shard(
        self,
        change: Callable[[dict, list[dict]], None],
        *,
        pair: int = 0,
        arm: int = 0,
        rehash_rows: bool = True,
    ) -> None:
        receipt = self.receipt(pair, arm)
        manifest_path = self.controller / receipt["manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scores_path = manifest_path.parent / "scores.jsonl"
        rows = [json.loads(line) for line in scores_path.read_text(encoding="utf-8").splitlines()]
        change(manifest, rows)
        if rehash_rows:
            rows = [_payload_hash(row, "row_sha256") for row in rows]
        manifest["files"]["scores.jsonl"] = _save_rows(scores_path, rows)
        receipt["sha256"] = _save(manifest_path, _payload_hash(manifest, "manifest_payload_sha256"))
        self.save_run()


def _paired(n: int = 437) -> tuple[dict, list[dict]]:
    tasks = {
        endpoint: [_task(endpoint, index) for index in range(n)] for endpoint in PRIMARY + SECONDARY
    }
    protocol = _protocol(tasks, "c" * 64)
    rows = [
        {
            "benchmark": endpoint,
            "length": length,
            "seed": seed,
            "task_id": task["task_id"],
            "baseline_pass": True,
            "candidate_pass": True,
        }
        for endpoint in PRIMARY
        for length in LENGTHS
        for seed in SEEDS
        for task in tasks[endpoint]
    ]
    return protocol, rows


def test_complete_admitted_records_publish_all_cells_and_hash_bound_outputs(
    tmp_path: Path,
) -> None:
    evidence = _Evidence(tmp_path)
    summary = evidence.assemble()
    assert summary["coverage"]["complete"] is True
    assert len(summary["cells"]) == 40
    assert summary["statistical_decision"] == summary["decision"] == "INCONCLUSIVE"
    assert len(summary["endpoints"]) == len(summary["clustered_success"]) == 8
    assert len(summary["nll"]) == 20
    assert all(cell["n"] == 4 for cell in summary["cells"])
    assert all(cell["informative"] for cell in summary["cells"])
    paired = [
        json.loads(line)
        for line in (evidence.output / "paired-rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(paired) == 160
    assert paired[0]["baseline_pass"] is True
    assert paired[0]["candidate_pass"] is True
    manifest = json.loads((evidence.output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETE"
    for name in ("summary.json", "paired-rows.jsonl", "results-v2.md"):
        payload = (evidence.output / name).read_bytes()
        assert manifest["files"][name]["sha256"] == _digest(payload)
        assert manifest["files"][name]["bytes"] == len(payload)
    report = (evidence.output / "results-v2.md").read_text(encoding="utf-8")
    for endpoint in PRIMARY:
        for length in LENGTHS:
            for seed in SEEDS:
                assert f"| {endpoint} | {length} | {seed} |" in report
    assert "outcome-informed endpoint redesign after v1 floors" in report
    assert "v1 remains `BLOCKED_INCOMPLETE_COVERAGE`" in report
    assert "generated-completion consistency" in report


def test_partial_failed_pair_is_not_admitted_or_summarized_as_complete(tmp_path: Path) -> None:
    evidence = _Evidence(tmp_path)
    evidence.run.update(status="FAILED", error_type="TimeoutError", error="declared deadline")
    evidence.run["pairs"][0].update(status="STARTED", arms=[evidence.receipt()])
    evidence.run["commands"].append({"name": "failed-candidate", "seconds": 9.0, "returncode": 1})
    evidence.save_run()
    summary = evidence.assemble()
    assert summary["decision"] == summary["statistical_decision"] == "BLOCKED_INCOMPLETE_COVERAGE"
    assert summary["coverage"]["complete"] is False
    missing = summary["coverage"]["missing_identities"]
    assert len(missing) == 4
    assert missing[0] == {
        "benchmark": "hellaswag",
        "length": 1024,
        "seed": 23,
        "task_id": "hellaswag-task-0",
        "missing_arms": ARMS,
    }
    assert len(summary["cells"]) == 40
    assert all(cell["pointwise_bounds"] is None for cell in summary["cells"])
    assert summary["failure_attempts"][0]["status"] == "FAILED"
    assert summary["failure_attempts"][0]["pairs"][0]["status"] == "STARTED"


def test_full_pair_retry_preserves_failed_attempt_without_using_its_one_arm(
    tmp_path: Path,
) -> None:
    evidence = _Evidence(tmp_path)
    retry = tmp_path / "retry"
    complete = copy.deepcopy(evidence.run["pairs"][0])
    for receipt in complete["arms"]:
        original = evidence.controller / receipt["manifest"]
        folder = retry / Path(receipt["manifest"]).parent
        folder.mkdir(parents=True)
        for name in ("manifest.json", "scores.jsonl", "batches.jsonl", "admission.json"):
            (folder / name).write_bytes((original.parent / name).read_bytes())
    retry_run = {**evidence.run, "pairs": [complete]}
    evidence.controller_artifacts(retry, retry_run, pairs=[evidence.protocol["execution_pairs"][0]])
    _save(retry / "run.json", retry_run)
    evidence.run.update(status="FAILED", error="candidate process stopped")
    evidence.run["pairs"][0].update(status="STARTED", arms=[evidence.receipt()])
    evidence.save_run()
    summary = evidence.assemble([evidence.controller, retry])
    assert summary["coverage"]["complete"] is True
    assert len(summary["failure_attempts"]) == 1
    assert len(summary["controllers"]) == 2


def test_duplicate_success_is_rejected_instead_of_selecting_one(tmp_path: Path) -> None:
    evidence = _Evidence(tmp_path)
    evidence.run["pairs"].append(copy.deepcopy(evidence.run["pairs"][0]))
    evidence.save_run()
    with pytest.raises(results.AdmissionError, match="overlapping|duplicate"):
        evidence.assemble()
    failure = json.loads((evidence.output / "failure-manifest.json").read_text(encoding="utf-8"))
    assert failure["status"] == "FAILED"
    assert not (evidence.output / "manifest.json").exists()


def test_repeated_controller_root_is_rejected(tmp_path: Path) -> None:
    evidence = _Evidence(tmp_path)
    with pytest.raises(results.AdmissionError, match="duplicate controller"):
        evidence.assemble([evidence.controller, evidence.controller / "."])


@pytest.mark.parametrize(
    "change",
    [
        lambda pair: pair.update(start=1),
        lambda pair: pair.update(stop=5),
        lambda pair: pair.update(seed=999),
        lambda pair: pair.update(endpoint="hellaswag_dev"),
    ],
)
def test_unfrozen_execution_pair_is_rejected(
    tmp_path: Path,
    change: Callable[[dict], None],
) -> None:
    evidence = _Evidence(tmp_path)
    change(evidence.run["pairs"][0])
    evidence.save_run()
    with pytest.raises(results.AdmissionError, match="frozen execution pair"):
        evidence.assemble()


def test_complete_pair_requires_both_arms_in_frozen_order(tmp_path: Path) -> None:
    evidence = _Evidence(tmp_path)
    evidence.run["pairs"][0]["arms"].reverse()
    evidence.save_run()
    with pytest.raises(results.AdmissionError, match="arms"):
        evidence.assemble()


def test_unrecorded_cleanup_cannot_admit_successes(tmp_path: Path) -> None:
    evidence = _Evidence(tmp_path)
    evidence.run["cleanup_complete"] = False
    evidence.save_run()
    with pytest.raises(results.AdmissionError, match="cleanup"):
        evidence.assemble()


def test_controller_manifest_path_cannot_escape_collected_root(tmp_path: Path) -> None:
    evidence = _Evidence(tmp_path)
    source = evidence.controller / evidence.receipt()["manifest"]
    outside = tmp_path / "outside.json"
    outside.write_bytes(source.read_bytes())
    evidence.receipt().update(manifest="../outside.json", sha256=_digest(outside.read_bytes()))
    evidence.save_run()
    with pytest.raises(results.AdmissionError, match="path|contain"):
        evidence.assemble()


@pytest.mark.requires_symlink
def test_hash_correct_symlink_escape_is_rejected(tmp_path: Path) -> None:
    evidence = _Evidence(tmp_path)
    score = (evidence.controller / evidence.receipt()["manifest"]).parent / "scores.jsonl"
    outside = tmp_path / "outside-scores.jsonl"
    outside.write_bytes(score.read_bytes())
    score.unlink()
    score.symlink_to(outside)
    with pytest.raises(results.AdmissionError, match="path|contain|escapes"):
        evidence.assemble()


def test_changed_score_file_is_rejected_before_statistical_analysis(tmp_path: Path) -> None:
    evidence = _Evidence(tmp_path)
    score = (evidence.controller / evidence.receipt()["manifest"]).parent / "scores.jsonl"
    with score.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(results.AdmissionError, match="file.*hash|bytes.*hash"):
        evidence.assemble()


def test_changed_row_with_rebound_file_hash_still_fails_row_hash(tmp_path: Path) -> None:
    evidence = _Evidence(tmp_path)
    evidence.mutate_shard(lambda _manifest, rows: rows[0].update(score=0.0), rehash_rows=False)
    with pytest.raises(results.AdmissionError, match="row.*hash"):
        evidence.assemble()


@pytest.mark.parametrize(
    "field,value",
    [
        ("protocol_sha256", "d" * 64),
        ("checkpoint_sha256", "d" * 64),
        ("audit_sha256", "d" * 64),
        ("source_ref", "unbound-source"),
        ("source_row_sha256", "d" * 64),
        ("task_index", 1),
        ("metric", "acc"),
    ],
)
def test_self_hashed_unbound_score_row_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    evidence = _Evidence(tmp_path)
    evidence.mutate_shard(lambda _manifest, rows: rows[0].update({field: value}))
    with pytest.raises(results.AdmissionError):
        evidence.assemble()


def test_self_hashed_inconsistent_likelihood_verdict_is_rejected(tmp_path: Path) -> None:
    evidence = _Evidence(tmp_path)
    evidence.mutate_shard(lambda _manifest, rows: rows[0].update({"pass": False, "score": 0.0}))
    with pytest.raises(results.AdmissionError, match="verdict|pass|likelihood"):
        evidence.assemble()


def test_self_hashed_inconsistent_nll_mean_is_rejected(tmp_path: Path) -> None:
    evidence = _Evidence(tmp_path)
    evidence.mutate_shard(
        lambda _manifest, rows: rows[0].update(nll_mean=100.0, score=100.0), pair=4
    )
    with pytest.raises(results.AdmissionError, match="NLL|nll"):
        evidence.assemble()


def test_duplicate_score_task_cannot_replace_missing_task(tmp_path: Path) -> None:
    evidence = _Evidence(tmp_path)
    evidence.mutate_shard(lambda _manifest, rows: rows.__setitem__(1, copy.deepcopy(rows[0])))
    with pytest.raises(results.AdmissionError, match="task|duplicate|index"):
        evidence.assemble()


def test_changed_data_or_protocol_hash_is_rejected_and_failure_is_preserved(
    tmp_path: Path,
) -> None:
    evidence = _Evidence(tmp_path)
    with (evidence.data / "manifest.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(results.AdmissionError, match="data.*hash|manifest.*hash"):
        evidence.assemble()
    assert (evidence.output / "failure-manifest.json").is_file()
    assert not (evidence.output / "summary.json").exists()


def test_protocol_file_hash_is_authoritative(tmp_path: Path) -> None:
    evidence = _Evidence(tmp_path)
    evidence.protocol["status"] = "DRAFT"
    _save(evidence.protocol_path, evidence.protocol)
    with pytest.raises(results.AdmissionError, match="protocol.*hash"):
        evidence.assemble()


def test_incomplete_secondary_coverage_is_explicit_and_does_not_block_primary(
    tmp_path: Path,
) -> None:
    evidence = _Evidence(tmp_path)
    evidence.run["pairs"][4].update(status="FAILED", arms=[])
    evidence.run.update(status="FAILED", error="NLL command failed")
    evidence.save_run()
    summary = evidence.assemble()
    assert summary["coverage"]["complete"] is True
    cell = next(
        row
        for row in summary["nll"]
        if row["benchmark"] == "gsm8k" and row["seed"] == 23 and row["length"] == 1024
    )
    assert cell["status"] == "INCOMPLETE_NLL"
    assert len(cell["missing_identities"]) == 2
    assert cell["mean_difference"] is None
    assert summary["decision"] != "BLOCKED_INCOMPLETE_COVERAGE"


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    evidence = _Evidence(tmp_path)
    evidence.output.mkdir()
    marker = evidence.output / "keep.txt"
    marker.write_text("immutable", encoding="utf-8")
    with pytest.raises(results.AdmissionError, match="fresh|exist"):
        evidence.assemble()
    assert marker.read_text(encoding="utf-8") == "immutable"
    assert list(evidence.output.iterdir()) == [marker]


def test_likelihood_s7_and_resource_records_keep_units_separate(tmp_path: Path) -> None:
    evidence = _Evidence(tmp_path)
    summary = evidence.assemble()
    consistency = summary["likelihood_verdict_consistency"]
    assert consistency["consistent"] is True
    assert consistency["matched_groups"] == 32
    assert consistency["comparisons"] == 288
    assert consistency["generated_completion_consistency_claimed"] is False
    resources = summary["resources"]
    assert resources["controller_wall_seconds_total"] == 3600.0
    assert resources["controller_session_hours_total"] == 1.0
    assert resources["commands"][0]["seconds"] == 5.5
    assert len(resources["batches"]) == 120
    assert resources["cuda_allocated_peak_bytes_max"] == 100
    assert resources["cuda_reserved_peak_bytes_max"] == 200
    assert resources["process_rss_sampled_peak_bytes_max"] == 300
    assert resources["gpu_hours_used"] is None
    assert resources["natural_rmr"] is None


def test_real_statistics_zero_discordance_boundary_and_floor_veto_keep_all40() -> None:
    protocol, rows = _paired(436)
    small = results.summarize_primary(rows, protocol)
    assert small["statistical_decision"] == "INCONCLUSIVE"
    protocol, rows = _paired(437)
    for row in rows:
        if row["benchmark"] == "hellaswag":
            row.update(baseline_pass=False, candidate_pass=False)
    summary = results.summarize_primary(rows, protocol)
    assert summary["statistical_decision"] == "NONINFERIOR"
    assert summary["decision"] == "INCONCLUSIVE_UNINFORMATIVE_FLOOR"
    assert len(summary["cells"]) == 40
    assert len(summary["floor_cells"]) == 10
    assert all(cell["pointwise_bounds"]["upper"] < 0.01 for cell in summary["cells"])
    assert all(
        cell["informative"] is False
        for cell in summary["cells"]
        if cell["benchmark"] == "hellaswag"
    )


def test_excessive_degradation_precedes_floor_veto_through_real_statistics() -> None:
    protocol, rows = _paired()
    for row in rows:
        if row["benchmark"] == "hellaswag":
            row.update(baseline_pass=False, candidate_pass=False)
        if row["benchmark"] == "arc":
            row["candidate_pass"] = False
    summary = results.summarize_primary(rows, protocol)
    assert summary["statistical_decision"] == summary["decision"] == "EXCESSIVE_DEGRADATION"
    assert len(summary["cells"]) == 40
    assert len(summary["floor_cells"]) == 10


def test_floor_threshold_equality_is_informative_without_changing_chance() -> None:
    protocol, rows = _paired(20)
    for row in rows:
        if row["benchmark"] == "boolq":
            row["baseline_pass"] = row["candidate_pass"] = (
                int(row["task_id"].rsplit("-", 1)[1]) < 11
            )
    summary = results.summarize_primary(rows, protocol)
    assert all(cell["informative"] for cell in summary["cells"])
    assert all(
        cell["baseline_accuracy"] == 0.55
        for cell in summary["cells"]
        if cell["benchmark"] == "boolq"
    )


def test_clustered_success_averages_seeds_within_task_before_standard_error() -> None:
    tasks = {"fixture": ["harm", "both-fail", "both-pass", "gain"]}
    patterns = [(True, False), (False, False), (True, True), (False, True)]
    rows = [
        {
            "benchmark": "fixture",
            "length": 1024,
            "seed": seed,
            "task_id": task_id,
            "baseline_pass": baseline,
            "candidate_pass": candidate,
        }
        for seed in SEEDS
        for task_id, (baseline, candidate) in zip(tasks["fixture"], patterns)
    ]
    cell = results.summarize_clustered_success(
        rows, expected_tasks=tasks, seeds=SEEDS, lengths=[1024]
    )[0]
    assert cell["n_tasks"] == 4
    assert cell["seed_count"] == 5
    assert cell["mean_difference"] == 0.0
    assert cell["standard_error"] == pytest.approx(math.sqrt(1 / 6))
    assert cell["degrees_of_freedom"] == 3
    assert cell["confidence_interval"] == pytest.approx([-1.299228263625, 1.299228263625])
    assert cell["descriptive_only"] is True


def test_clustered_single_task_does_not_treat_five_seeds_as_five_clusters() -> None:
    rows = [
        {
            "benchmark": "fixture",
            "length": 1024,
            "seed": seed,
            "task_id": "only",
            "baseline_pass": False,
            "candidate_pass": True,
        }
        for seed in SEEDS
    ]
    cell = results.summarize_clustered_success(
        rows, expected_tasks={"fixture": ["only"]}, seeds=SEEDS, lengths=[1024]
    )[0]
    assert cell["mean_difference"] == 1.0
    assert cell["standard_error"] is None
    assert cell["confidence_interval"] is None
    assert cell["status"] == "INSUFFICIENT_TASKS"


def _nll_rows(baseline: tuple[float, ...], candidate: tuple[float, ...]) -> list[dict]:
    return [
        {
            "endpoint": "fixture",
            "length": 1024,
            "seed": 23,
            "arm": arm,
            "task_id": f"task-{index}",
            "nll_mean": mean,
            "n_targets": 1 if index == 0 else 9,
            "nll_sum": mean * (1 if index == 0 else 9),
        }
        for arm, means in zip(ARMS, (baseline, candidate))
        for index, mean in enumerate(means)
    ]


def test_nll_paired_t_interval_uses_task_means_and_retains_token_totals() -> None:
    cell = results.summarize_nll(
        _nll_rows((1.0, 3.0), (2.0, 6.0)),
        expected_tasks={"fixture": ["task-0", "task-1"]},
        seeds=[23],
        lengths=[1024],
    )[0]
    assert cell["baseline_mean"] == 2.0
    assert cell["candidate_mean"] == 4.0
    assert cell["mean_difference"] == 2.0
    assert cell["standard_error"] == pytest.approx(1.0)
    assert cell["confidence_interval"] == pytest.approx([-10.706204736432, 14.706204736432])
    assert cell["relative_mean_change"] == 1.0
    assert cell["baseline_nll_sum"] == 28.0
    assert cell["candidate_nll_sum"] == 56.0
    assert cell["baseline_n_targets"] == cell["candidate_n_targets"] == 10
    assert cell["baseline_token_weighted_mean"] == 2.8
    assert cell["candidate_token_weighted_mean"] == 5.6


def test_zero_baseline_nll_does_not_fabricate_relative_improvement() -> None:
    cell = results.summarize_nll(
        _nll_rows((0.0, 0.0), (1.0, 1.0)),
        expected_tasks={"fixture": ["task-0", "task-1"]},
        seeds=[23],
        lengths=[1024],
    )[0]
    assert cell["mean_difference"] == 1.0
    assert cell["relative_mean_change"] is None
    assert cell["relative_mean_change_status"] == "UNDEFINED_BASELINE_ZERO"
    assert cell["confidence_interval"] == [1.0, 1.0]


def test_incomplete_nll_does_not_summarize_the_remaining_tasks() -> None:
    rows = _nll_rows((1.0, 3.0), (2.0, 6.0))[:-1]
    cell = results.summarize_nll(
        rows, expected_tasks={"fixture": ["task-0", "task-1"]}, seeds=[23], lengths=[1024]
    )[0]
    assert cell["status"] == "INCOMPLETE_NLL"
    assert cell["mean_difference"] is None
    assert cell["missing_identities"] == [{"task_id": "task-1", "missing_arms": [ARMS[1]]}]


def test_cli_assembles_incomplete_evidence_with_explicit_exit_status(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    evidence = _Evidence(tmp_path)
    evidence.run["pairs"].pop(0)
    evidence.save_run()
    code = results.main(
        [
            "--protocol-path",
            str(evidence.protocol_path),
            "--protocol-sha256",
            evidence.protocol_hash,
            "--data-root",
            str(evidence.data),
            "--controller-root",
            str(evidence.controller),
            "--output",
            str(evidence.output),
        ]
    )
    assert code == 2
    assert "BLOCKED_INCOMPLETE_COVERAGE" in capsys.readouterr().out
    assert (evidence.output / "manifest.json").is_file()


def test_manifest_payload_hash_is_checked_even_with_matching_controller_receipt(
    tmp_path: Path,
) -> None:
    evidence = _Evidence(tmp_path)
    receipt = evidence.receipt()
    path = evidence.controller / receipt["manifest"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["requested_rows"] = 1
    receipt["sha256"] = _save(path, manifest)
    evidence.save_run()
    with pytest.raises(results.AdmissionError, match="manifest payload hash"):
        evidence.assemble()


def test_shard_file_descriptor_cannot_escape_its_manifest_directory(tmp_path: Path) -> None:
    evidence = _Evidence(tmp_path)
    evidence.mutate_shard(
        lambda manifest, _rows: manifest["files"].update(
            {
                "../outside.jsonl": {"bytes": 0, "sha256": _digest(b"")},
            }
        )
    )
    with pytest.raises(results.AdmissionError, match="path|contain"):
        evidence.assemble()


def test_changed_data_task_file_is_rejected(tmp_path: Path) -> None:
    evidence = _Evidence(tmp_path)
    path = evidence.data / "hellaswag/tasks.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(results.AdmissionError, match="file.*hash|bytes.*hash"):
        evidence.assemble()


def test_self_hashed_manifest_checkpoint_or_source_change_is_rejected(tmp_path: Path) -> None:
    evidence = _Evidence(tmp_path)
    evidence.mutate_shard(lambda manifest, _rows: manifest.update(checkpoint_sha256="f" * 64))
    with pytest.raises(results.AdmissionError, match="checkpoint"):
        evidence.assemble()


def test_self_hashed_source_subset_must_agree_to_frozen_sources(tmp_path: Path) -> None:
    evidence = _Evidence(tmp_path)
    evidence.mutate_shard(
        lambda manifest, _rows: manifest.update(
            source_files_sha256={"benchmarks/harness/p3_stage2v2_loglik.py": "f" * 64}
        )
    )
    with pytest.raises(results.AdmissionError, match="source hash"):
        evidence.assemble()


def test_s7_independently_rejects_identical_payloads_with_different_verdicts() -> None:
    task = _task("hellaswag", 0)
    baseline = {
        "endpoint": "hellaswag",
        "length": 1024,
        "task_id": task["task_id"],
        "source_ref": task["source_ref"],
        "source_row_sha256": task["source_row_sha256"],
        "results": [
            {"logprob": value, "greedy": index == 0, "n_tokens": 1}
            for index, value in enumerate([-1.0, -5.0, -5.0, -5.0])
        ],
        "metric": "acc_norm",
        "score": 1.0,
        "pred": 0,
        "pred_norm": 0,
        "pass": True,
    }
    candidate = {**baseline, "pred": 1, "pred_norm": 1, "score": 0.0, "pass": False}
    with pytest.raises(ValueError, match="identical likelihood.*inconsistent verdict"):
        results._likelihood_consistency(
            [baseline, candidate], {"hellaswag": {task["task_id"]: task}}
        )


def test_failed_pair_completed_arm_is_hash_checked_but_not_used_for_statistics(
    tmp_path: Path,
) -> None:
    evidence = _Evidence(tmp_path)
    evidence.run.update(status="FAILED", error="candidate process failed")
    evidence.run["pairs"][0].update(status="STARTED", arms=[evidence.receipt()])
    evidence.save_run()
    summary = evidence.assemble()
    failed_batches = [
        batch
        for batch in summary["resources"]["batches"]
        if batch["admitted_for_analysis"] is False
    ]
    assert len(failed_batches) == 1
    assert failed_batches[0]["arm"] == ARMS[0]
    assert summary["coverage"]["observed_paired_rows"] == 156


def test_tampered_completed_arm_in_failed_pair_is_not_silently_ignored(tmp_path: Path) -> None:
    evidence = _Evidence(tmp_path)
    score = (evidence.controller / evidence.receipt()["manifest"]).parent / "scores.jsonl"
    with score.open("a", encoding="utf-8") as stream:
        stream.write(" ")
    evidence.run.update(status="FAILED", error="candidate process failed")
    evidence.run["pairs"][0].update(status="STARTED", arms=[evidence.receipt()])
    evidence.save_run()
    with pytest.raises(results.AdmissionError, match="file.*hash|bytes.*hash"):
        evidence.assemble()


def test_clustered_success_missing_seed_does_not_average_available_seeds() -> None:
    rows = [
        {
            "benchmark": "fixture",
            "length": 1024,
            "seed": seed,
            "task_id": "only",
            "baseline_pass": False,
            "candidate_pass": True,
        }
        for seed in SEEDS[:-1]
    ]
    cell = results.summarize_clustered_success(
        rows, expected_tasks={"fixture": ["only"]}, seeds=SEEDS, lengths=[1024]
    )[0]
    assert cell["status"] == "INCOMPLETE_PRIMARY_PAIRS"
    assert cell["mean_difference"] is None
    assert cell["missing_identities"] == [{"task_id": "only", "seed": 113}]


def test_one_task_nll_has_no_t_interval_and_does_not_invent_precision() -> None:
    cell = results.summarize_nll(
        _nll_rows((1.0,), (2.0,)),
        expected_tasks={"fixture": ["task-0"]},
        seeds=[23],
        lengths=[1024],
    )[0]
    assert cell["mean_difference"] == 1.0
    assert cell["status"] == "INSUFFICIENT_TASKS"
    assert cell["degrees_of_freedom"] == 0
    assert cell["standard_error"] is None
    assert cell["confidence_interval"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_targets", True),
        ("n_targets", 0),
        ("n_targets", 1.5),
        ("nll_mean", math.nan),
        ("nll_mean", -1.0),
        ("nll_sum", math.inf),
        ("seed", 23.0),
        ("length", 1024.0),
    ],
)
def test_nll_rejects_nonfinite_negative_or_coerced_observations(
    field: str,
    value: object,
) -> None:
    rows = _nll_rows((1.0,), (2.0,))
    rows[0][field] = value
    with pytest.raises(ValueError):
        results.summarize_nll(
            rows, expected_tasks={"fixture": ["task-0"]}, seeds=[23], lengths=[1024]
        )


def test_frozen_full_source_order_hash_must_match_admitted_data(tmp_path: Path) -> None:
    evidence = _Evidence(tmp_path)
    evidence.protocol["data"]["source_ordered_task_ids_sha256"]["hellaswag"] = "f" * 64
    evidence.protocol_hash = _save(evidence.protocol_path, evidence.protocol)
    with pytest.raises(results.AdmissionError, match="source ordered task hash"):
        evidence.assemble()


def test_unknown_null_source_entry_cannot_pass_subset_hash_agreement(tmp_path: Path) -> None:
    evidence = _Evidence(tmp_path)
    evidence.mutate_shard(
        lambda manifest, _rows: manifest.update(source_files_sha256={"unknown-source.py": None})
    )
    with pytest.raises(results.AdmissionError, match="source hash"):
        evidence.assemble()


def test_complete_shard_requires_explicit_successful_cleanup(tmp_path: Path) -> None:
    evidence = _Evidence(tmp_path)
    evidence.mutate_shard(lambda manifest, _rows: manifest.update(cleanup={"complete": False}))
    with pytest.raises(results.AdmissionError, match="cleanup"):
        evidence.assemble()


@pytest.mark.parametrize(
    "name",
    [
        "benchmarks/harness/p3_stage2v2_results.py",
        "benchmarks/harness/p3_stage2v2_decision.py",
        "benchmarks/harness/p3_stage2_statistics.py",
        "benchmarks/harness/p3_stage2_dataset.py",
        "benchmarks/harness/p3_stage2_math.py",
    ],
)
def test_changed_analysis_or_dependency_source_refuses_unchanged_frozen_pin(
    tmp_path: Path,
    name: str,
) -> None:
    evidence = _Evidence(tmp_path)
    source = results._SOURCE_ROOT / name
    source.write_bytes(source.read_bytes() + b"\n# changed after freeze\n")
    with pytest.raises(results.AdmissionError, match="analysis source.*hash"):
        evidence.assemble()
    assert not (evidence.output / "manifest.json").exists()


def test_missing_analysis_source_pin_cannot_be_replaced_by_scorer_only_inventory(
    tmp_path: Path,
) -> None:
    evidence = _Evidence(tmp_path)
    evidence.protocol["source_files_sha256"].pop("benchmarks/harness/p3_stage2_statistics.py")
    evidence.protocol_hash = _save(evidence.protocol_path, evidence.protocol)
    with pytest.raises(results.AdmissionError, match="analysis source.*inventory"):
        evidence.assemble()


def test_analysis_sources_are_rechecked_after_statistics_before_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _Evidence(tmp_path)
    original = results._write

    def write_and_change_source(root: Path, name: str, text: str) -> dict:
        descriptor = original(root, name, text)
        if name == "results-v2.md":
            source = results._SOURCE_ROOT / "benchmarks/harness/p3_stage2v2_decision.py"
            source.write_bytes(source.read_bytes() + b"\n# changed during reporting\n")
        return descriptor

    monkeypatch.setattr(results, "_write", write_and_change_source)
    with pytest.raises(results.AdmissionError, match="analysis source.*hash"):
        evidence.assemble()
    assert (evidence.output / "summary.json").exists()
    assert not (evidence.output / "manifest.json").exists()
    assert (evidence.output / "failure-manifest.json").exists()


@pytest.mark.parametrize(
    "name",
    [
        "stage2v2_eval_runner.py",
        "execution-admission.json",
        "source-overlay.zip",
        "p3-stage2-v2-protocol.json",
    ],
)
def test_missing_existing_controller_producer_artifact_is_rejected(
    tmp_path: Path,
    name: str,
) -> None:
    evidence = _Evidence(tmp_path)
    (evidence.controller / name).unlink()
    with pytest.raises(results.AdmissionError, match=name):
        evidence.assemble()


def test_copied_runner_must_match_the_frozen_runner_source(tmp_path: Path) -> None:
    evidence = _Evidence(tmp_path)
    path = evidence.controller / "stage2v2_eval_runner.py"
    path.write_bytes(path.read_bytes() + b"\n# unpinned copied runner\n")
    with pytest.raises(results.AdmissionError, match="runner.*hash"):
        evidence.assemble()


@pytest.mark.parametrize(
    "change",
    [
        lambda admission: admission.update(protocol_sha256="f" * 64),
        lambda admission: admission.update(source_overlay_sha256="f" * 64),
        lambda admission: admission["inputs"]["PROTOCOL"].update(sha256="f" * 64),
        lambda admission: admission["inputs"]["AUDIT"].update(sha256="f" * 64),
        lambda admission: admission["inputs"]["DATA_MANIFEST"].update(sha256="f" * 64),
    ],
)
def test_rehashed_execution_admission_must_join_run_protocol_and_inputs(
    tmp_path: Path,
    change: Callable[[dict], None],
) -> None:
    evidence = _Evidence(tmp_path)
    evidence.mutate_controller_admission(change)
    with pytest.raises(results.AdmissionError, match="controller|execution|overlay|input|protocol"):
        evidence.assemble()


def test_run_input_bindings_cannot_contradict_hash_bound_execution_inputs(tmp_path: Path) -> None:
    evidence = _Evidence(tmp_path)
    evidence.run["input_bindings"]["PROTOCOL"]["sha256"] = "f" * 64
    evidence.save_run()
    with pytest.raises(results.AdmissionError, match="input binding"):
        evidence.assemble()


@pytest.mark.parametrize("rebind_member", [False, True])
def test_rehashed_source_overlay_cannot_change_declared_or_frozen_source_bytes(
    tmp_path: Path,
    rebind_member: bool,
) -> None:
    evidence = _Evidence(tmp_path)
    evidence.mutate_overlay(
        "benchmarks/harness/p3_stage2v2_decision.py", rebind_member=rebind_member
    )
    with pytest.raises(results.AdmissionError, match="overlay.*hash|overlay.*source"):
        evidence.assemble()


def test_rehashed_corrupt_source_overlay_is_rejected(tmp_path: Path) -> None:
    evidence = _Evidence(tmp_path)
    path = evidence.controller / "source-overlay.zip"
    path.write_bytes(b"not a ZIP archive")
    digest = _digest(path.read_bytes())
    evidence.run["source_overlay_sha256"] = digest
    evidence.mutate_controller_admission(
        lambda admission: admission.update(source_overlay_sha256=digest)
    )
    with pytest.raises(results.AdmissionError, match="zip|ZIP|archive|overlay"):
        evidence.assemble()


def test_a_missing_mandatory_scorer_source_cannot_be_hidden_by_rehashing(tmp_path: Path) -> None:
    evidence = _Evidence(tmp_path)
    evidence.mutate_shard(
        lambda manifest, _rows: manifest["source_files_sha256"].pop(
            "benchmarks/harness/p3_stage2_training_audit.py"
        )
    )
    with pytest.raises(results.AdmissionError, match="source.*inventory|source hash"):
        evidence.assemble()


def test_canonical_shard_admission_is_mandatory_even_if_other_files_are_valid(
    tmp_path: Path,
) -> None:
    evidence = _Evidence(tmp_path)
    evidence.mutate_shard(lambda manifest, _rows: manifest["files"].pop("admission.json"))
    with pytest.raises(results.AdmissionError, match="admission"):
        evidence.assemble()


@pytest.mark.parametrize(
    "change",
    [
        lambda admission: admission.update(protocol_sha256="f" * 64),
        lambda admission: admission["protocol_file"].update(bytes=1),
        lambda admission: admission["protocol_file"].update(sha256="f" * 64),
        lambda admission: admission.update(data_manifest_sha256="f" * 64),
        lambda admission: admission.update(selected_rows_sha256="f" * 64),
        lambda admission: admission.update(training_run_key="unbound-checkpoint"),
        lambda admission: admission["checkpoint"].update(sha256="f" * 64),
        lambda admission: admission["training_audit"].update(sha256="f" * 64),
        lambda admission: admission["tasks_file"].update(rows=1),
        lambda admission: admission["ordered_task_ids"].reverse(),
        lambda admission: admission["source_files_sha256"].pop(
            "benchmarks/harness/p3_stage2_training_audit.py"
        ),
    ],
)
def test_rehashed_canonical_shard_admission_must_join_frozen_selected_inputs(
    tmp_path: Path,
    change: Callable[[dict], None],
) -> None:
    evidence = _Evidence(tmp_path)
    evidence.mutate_admission(change)
    with pytest.raises(results.AdmissionError, match="admission"):
        evidence.assemble()


@pytest.mark.parametrize("raw,normalized", [(0, 1), (1, 0)])
def test_unique_rehashed_likelihood_payload_must_derive_both_recorded_predictions(
    tmp_path: Path,
    raw: int,
    normalized: int,
) -> None:
    evidence = _Evidence(tmp_path)

    def change(_manifest: dict, rows: list[dict]) -> None:
        row = rows[0]
        for result, value in zip(row["results"], [-2.73, -0.123, -6.0, -7.0]):
            result["logprob"] = value
        row.update(
            pred=raw,
            pred_norm=normalized,
            score=float(normalized == 0),
            **{"pass": normalized == 0},
        )

    evidence.mutate_shard(change)
    with pytest.raises(results.AdmissionError, match="prediction"):
        evidence.assemble()


def _choice_observation(
    task: dict, logprobs: list[float], raw: int, normalized: int
) -> tuple[dict, dict]:
    identity = {
        "endpoint": task["endpoint"],
        "arm": ARMS[0],
        "seed": 23,
        "length": 1024,
        "protocol_sha256": "c" * 64,
        "checkpoint_sha256": "d" * 64,
        "audit_sha256": AUDIT,
    }
    passed = normalized == task["gold"]
    row = {
        "schema": "p3-stage2-v2-loglik-row-v1",
        **identity,
        "task_index": 0,
        **{
            name: task[name]
            for name in ("task_id", "source_ref", "source_row_sha256", "metric", "gold")
        },
        "pred": raw,
        "pred_norm": normalized,
        "pass": passed,
        "score": float(passed),
        "results": [{"logprob": value, "greedy": False, "n_tokens": 1} for value in logprobs],
    }
    return _payload_hash(row, "row_sha256"), identity


@pytest.mark.parametrize(
    "choices,logs,gold,raw,normalized",
    [
        (["a", "longword", "other"], [-1.0, -2.0, -3.0], 1, 0, 1),
        (["é", "ab"], [-1.2, -2.0], 1, 0, 1),
        (["aa", "bb", "cc"], [-1.0, -1.0, -3.0], 0, 0, 0),
    ],
)
def test_derived_predictions_use_character_lengths_and_first_argmax_ties(
    choices: list[str],
    logs: list[float],
    gold: int,
    raw: int,
    normalized: int,
) -> None:
    task = {**_task("hellaswag", 0), "choices": choices, "gold": gold}
    row, identity = _choice_observation(task, logs, raw, normalized)
    results._validate_score(row, task=task, task_index=0, identity=identity)


def test_recorded_later_index_cannot_win_an_exact_argmax_tie() -> None:
    task = _task("hellaswag", 0)
    row, identity = _choice_observation(task, [-1.0, -1.0, -3.0, -4.0], 1, 1)
    with pytest.raises(ValueError, match="prediction"):
        results._validate_score(row, task=task, task_index=0, identity=identity)


@pytest.mark.parametrize(
    "change",
    [
        lambda _manifest, batches: batches.clear(),
        lambda _manifest, batches: batches[0].update(schema="unbound-batch"),
        lambda _manifest, batches: batches[0].update(batch_index=1),
        lambda _manifest, batches: batches[0].update(request_indices=[0, 0, *range(2, 16)]),
        lambda _manifest, batches: batches[0].update(request_indices=list(range(15))),
        lambda _manifest, batches: batches[0].update(request_indices=[*range(15), 16]),
        lambda _manifest, batches: batches[0].update(scored_tokens=15),
        lambda manifest, _batches: manifest.update(batch_count=2),
        lambda manifest, _batches: manifest.update(request_count=15),
        lambda manifest, _batches: manifest.update(scored_tokens=17),
    ],
)
def test_rehashed_batches_must_partition_requests_and_reconcile_all_existing_totals(
    tmp_path: Path,
    change: Callable[[dict, list[dict]], None],
) -> None:
    evidence = _Evidence(tmp_path)
    evidence.mutate_batches(change)
    with pytest.raises(results.AdmissionError, match="batch|request|token"):
        evidence.assemble()


def test_valid_length_sorted_multibatch_request_partition_need_not_be_sorted(
    tmp_path: Path,
) -> None:
    evidence = _Evidence(tmp_path)
    first = [15, 3, 11, 7, 0, 12, 2, 6]
    second = [14, 1, 13, 5, 4, 10, 8, 9]

    def split(manifest: dict, batches: list[dict]) -> None:
        template = batches.pop()
        for index, indices in enumerate((first, second)):
            batches.append(
                {
                    **template,
                    "batch_index": index,
                    "request_indices": indices,
                    "scored_tokens": 8,
                    "input_tokens": 16,
                    "padded_input_tokens": 16,
                }
            )
        manifest["batch_count"] = 2

    evidence.mutate_batches(split)
    summary = evidence.assemble()
    selected = [
        batch
        for batch in summary["resources"]["batches"]
        if batch["endpoint"] == "hellaswag"
        and batch["arm"] == ARMS[0]
        and batch["length"] == 1024
        and batch["seed"] == 23
    ]
    assert [batch["request_indices"] for batch in selected] == [first, second]
    assert len(summary["resources"]["batches"]) == 121


@pytest.mark.parametrize("filename", ["summary.json", "manifest.staging.json"])
def test_fallible_artifact_or_staged_marker_verification_cannot_publish_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
) -> None:
    evidence = _Evidence(tmp_path)
    original = results._file_hash

    def fail_read(path: Path) -> tuple[int, str]:
        if path.parent == evidence.output and path.name == filename:
            raise OSError("injected staged verification failure")
        return original(path)

    monkeypatch.setattr(results, "_file_hash", fail_read)
    with pytest.raises(results.AdmissionError, match="injected"):
        evidence.assemble()
    assert not (evidence.output / "manifest.json").exists()
    failed = json.loads((evidence.output / "failure-manifest.json").read_text(encoding="utf-8"))
    assert failed["status"] == "FAILED"


def test_no_overwrite_publication_failure_preserves_staging_and_failure_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _Evidence(tmp_path)

    def fail_link(_source: Path, _destination: Path) -> None:
        raise OSError("injected final publication failure")

    monkeypatch.setattr(results.os, "link", fail_link)
    with pytest.raises(results.AdmissionError, match="final publication"):
        evidence.assemble()
    assert (evidence.output / "manifest.staging.json").is_file()
    assert not (evidence.output / "manifest.json").exists()
    assert (evidence.output / "failure-manifest.json").is_file()


def test_published_success_has_no_fallible_final_marker_hash_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _Evidence(tmp_path)
    original = results._file_hash

    def reject_post_commit_read(path: Path) -> tuple[int, str]:
        if path.parent == evidence.output and path.name == "manifest.json":
            raise OSError("post-commit reads are forbidden")
        return original(path)

    monkeypatch.setattr(results, "_file_hash", reject_post_commit_read)
    summary = evidence.assemble()
    assert summary["coverage"]["complete"] is True
    assert (evidence.output / "manifest.json").exists()
    assert not (evidence.output / "failure-manifest.json").exists()


def test_terminal_publication_does_not_replace_a_destination_created_during_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _Evidence(tmp_path)
    original = results._file_hash
    sentinel = b"existing destination, not an assembly commit\n"

    def create_destination_before_commit(path: Path) -> tuple[int, str]:
        if path.parent == evidence.output and path.name == "manifest.staging.json":
            (evidence.output / "manifest.json").write_bytes(sentinel)
        return original(path)

    monkeypatch.setattr(results, "_file_hash", create_destination_before_commit)
    with pytest.raises(results.AdmissionError):
        evidence.assemble()
    assert (evidence.output / "manifest.json").read_bytes() == sentinel
    failed = json.loads((evidence.output / "failure-manifest.json").read_text(encoding="utf-8"))
    assert failed["status"] == "FAILED"


def test_overlay_tests_and_contracts_need_not_be_runtime_source_map_members(tmp_path: Path) -> None:
    evidence = _Evidence(tmp_path)
    admission = json.loads(
        (evidence.controller / "execution-admission.json").read_text(encoding="utf-8")
    )
    extra = set(admission["source_files_sha256"]) - set(evidence.protocol["source_files_sha256"])
    assert "tests/test_p3_stage2v2_results.py" in extra
    assert "benchmarks/results/p3-stage2-quality/frozen-study-protocol-v1.json" in extra
    assert evidence.assemble()["coverage"]["complete"] is True


def test_unchanged_pinned_base_dependency_need_not_be_a_source_overlay_member(
    tmp_path: Path,
) -> None:
    evidence = _Evidence(tmp_path)
    path = evidence.controller / "source-overlay.zip"
    omitted = "benchmarks/harness/p3_realdata_baseline.py"
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist() if name != omitted}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    path.write_bytes(buffer.getvalue())
    digest = _digest(path.read_bytes())
    evidence.run["source_overlay_sha256"] = digest

    def change(admission: dict) -> None:
        admission["source_overlay_sha256"] = digest
        admission["source_files_sha256"].pop(omitted)

    evidence.mutate_controller_admission(change)
    assert evidence.assemble()["coverage"]["complete"] is True


def test_selected_hellaswag_count_is_distinct_from_its_full_physical_task_file(
    tmp_path: Path,
) -> None:
    evidence = _Evidence(tmp_path, hellaswag_n=2)
    assert evidence.protocol["data"]["retained_evaluation_tasks"]["hellaswag"] == 2
    assert evidence.protocol["data"]["full_retained_evaluation_tasks"]["hellaswag"] == 4
    summary = evidence.assemble()
    assert summary["coverage"]["complete"] is True
    assert summary["coverage"]["observed_paired_rows"] == 140
    assert all(cell["n"] == 2 for cell in summary["cells"] if cell["benchmark"] == "hellaswag")


def test_full_physical_count_cannot_replace_selected_endpoint_count_in_both_admissions(
    tmp_path: Path,
) -> None:
    evidence = _Evidence(tmp_path, hellaswag_n=2)
    evidence.mutate_shard(lambda manifest, _rows: manifest.update(complete_endpoint_row_count=4))
    evidence.mutate_admission(lambda admission: admission.update(complete_endpoint_row_count=4))
    with pytest.raises(results.AdmissionError, match="admission.*complete_endpoint_row_count"):
        evidence.assemble()


@pytest.mark.parametrize("exception", [KeyboardInterrupt, OSError])
def test_interruption_after_real_terminal_link_cannot_create_a_failed_terminal_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception: type[BaseException],
) -> None:
    evidence = _Evidence(tmp_path)
    original_link = results.os.link

    def link_then_interrupt(source: Path, destination: Path) -> None:
        original_link(source, destination)
        raise exception("injected immediately after successful link")

    monkeypatch.setattr(results.os, "link", link_then_interrupt)
    if exception is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            evidence.assemble()
    else:
        assert evidence.assemble()["coverage"]["complete"] is True
    committed = json.loads((evidence.output / "manifest.json").read_text(encoding="utf-8"))
    assert committed["status"] == "COMPLETE"
    assert not (evidence.output / "failure-manifest.json").exists()


def test_ambiguous_commit_inspection_failure_never_writes_a_contradictory_failed_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _Evidence(tmp_path)
    original_link, original_hash = results.os.link, results._file_hash

    def link_then_interrupt(source: Path, destination: Path) -> None:
        original_link(source, destination)
        raise OSError("interrupted terminal publication")

    def unreadable_terminal(path: Path) -> tuple[int, str]:
        if path.parent == evidence.output and path.name == "manifest.json":
            raise OSError("cannot inspect terminal publication")
        return original_hash(path)

    monkeypatch.setattr(results.os, "link", link_then_interrupt)
    monkeypatch.setattr(results, "_file_hash", unreadable_terminal)
    with pytest.raises(results.AdmissionError, match="publication.*unknown|publication.*uncertain"):
        evidence.assemble()
    assert (evidence.output / "manifest.json").exists()
    assert not (evidence.output / "failure-manifest.json").exists()
