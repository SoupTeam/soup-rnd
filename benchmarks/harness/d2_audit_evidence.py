"""Replay saved D2 tensors and review sampled ABBA telemetry without a GPU.

The timing review is descriptive: post-arm telemetry cannot prove stable clocks
throughout an arm, and same-process absolute peaks are not per-model VRAM savings.
Original artifacts are read only. Failures and excluded rounds remain in the audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console

try:
    from .d2_replay_tensors import replay
except ImportError:
    from d2_replay_tensors import replay


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _telemetry(row: dict[str, Any]) -> tuple[str, str, str]:
    validity = row["validity"]
    if validity.get("status") != "UNVERIFIED" or validity.get("available") is not True:
        raise ValueError("missing or previously void arm telemetry")
    gpu = validity["gpu_samples"]
    if len(gpu) != 1 or len(gpu[0]) != 6:
        raise ValueError("ambiguous GPU telemetry")
    _index, uuid, driver, clock, state, power = gpu[0]
    if uuid.removeprefix("GPU-") != validity["target_uuid"].removeprefix("GPU-"):
        raise ValueError("wrong GPU")
    if float(clock) <= 0 or not math.isfinite(float(power)) or float(power) <= 0:
        raise ValueError("missing clock or power")
    if not state.startswith("P") or not driver:
        raise ValueError("missing power state or driver")
    peers = validity["process_samples"]
    if not peers or any(
        sample[0] == uuid and sample[1] != str(validity["own_pid"]) for sample in peers
    ) or validity.get("peers"):
        raise ValueError("GPU peers or absent process evidence")
    if not any(sample[0] == uuid and sample[1] == str(validity["own_pid"]) for sample in peers):
        raise ValueError("own process absent from GPU telemetry")
    if row.get("correctness_passed") is not True:
        raise ValueError("failed correctness gate")
    for key in ("forward_ms", "backward_ms"):
        if not math.isfinite(row[key]) or row[key] <= 0:
            raise ValueError("invalid timing")
    return uuid, clock, state


def audit_timing(path: Path) -> dict[str, Any]:
    report = _read(path)
    grouped: dict[tuple[str, bool, int], list[Any]] = defaultdict(list)
    for row in report["rows"]:
        grouped[(row["path"], row["request_dX"], row["round"])].append(row)
    rounds = []
    for (projection, dx, number), rows in grouped.items():
        record: dict[str, Any] = {
            "path": projection, "request_dX": dx, "round": number,
            "sampled_telemetry_passed": False,
        }
        try:
            rows = sorted(rows, key=lambda row: row["arm_index"])
            if [row["arm_index"] for row in rows] != [1, 2, 3, 4] or [
                row["implementation"] for row in rows
            ] != ["PEFT", "fast", "fast", "PEFT"]:
                raise ValueError("incomplete ABBA round")
            observed = [_telemetry(row) for row in rows]
            if len(set(observed)) != 1:
                raise ValueError("clock or power state changed between arms")
            for implementation in ("PEFT", "fast"):
                samples = [row for row in rows if row["implementation"] == implementation]
                record[implementation] = {
                    key: statistics.mean(row[key] for row in samples)
                    for key in ("forward_ms", "backward_ms")
                }
                record[implementation]["total_ms"] = sum(record[implementation].values())
            record["observed_gpu_clock_state"] = observed[0]
            record["sampled_telemetry_passed"] = True
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            record["exclusion_reason"] = str(exc)
        rounds.append(record)
    summaries = []
    for projection in ("single", "qkv", "mlp"):
        for dx in (True, False):
            valid = [
                row for row in rounds if row["path"] == projection
                and row["request_dX"] == dx and row["sampled_telemetry_passed"]
            ]
            summary: dict[str, Any] = {
                "path": projection, "request_dX": dx, "included_rounds": len(valid),
                "excluded_rounds": sum(
                    row["path"] == projection and row["request_dX"] == dx
                    and not row["sampled_telemetry_passed"] for row in rounds
                ),
            }
            if valid:
                for implementation in ("PEFT", "fast"):
                    summary[implementation] = {
                        key: statistics.median(row[implementation][key] for row in valid)
                        for key in ("forward_ms", "backward_ms", "total_ms")
                    }
                summary["median_paired_peft_over_fast_total_ratio"] = statistics.median(
                    row["PEFT"]["total_ms"] / row["fast"]["total_ms"] for row in valid
                )
            summaries.append(summary)
    return {
        "file": str(path), "sha256": _hash(path), "dtype": report["dtype"],
        "verdict": "NO VERDICT: sampled post-arm telemetry, same-process timings only",
        "memory_scope": report["memory_scope"], "rounds": rounds, "summaries": summaries,
    }


def audit(evidence: Path, aligned: list[Path]) -> dict[str, Any]:
    report: dict[str, Any] = {"D2": "INCOMPLETE", "tensor_replays": [], "timing": []}
    for dtype in ("fp32", "fp16", "bf16"):
        for dx in (0, 1):
            directory = evidence / f"parity-{dtype}-dx{dx}"
            paths = [directory / f"{projection}-dx{dx}.safetensors"
                     for projection in ("single", "qkv", "mlp")]
            report["tensor_replays"].extend(
                replay(path, dtype) for path in paths
            )
    for directory in [evidence / f"nf4-{dtype}" for dtype in ("bf16", "fp16")] + aligned:
        original = _read(directory / "report.json")
        cases = {(row["path"], row["request_dX"], row["checkpoint"])
                 for row in original["rows"]}
        expected = {(projection, dx, checkpoint) for projection in ("single", "qkv", "mlp")
                    for dx in (True, False) for checkpoint in (True, False)}
        if len(original["rows"]) != 12 or cases != expected:
            raise ValueError(f"incomplete NF4 case matrix: {directory}")
        for row in original["rows"]:
            path = directory / row["tensor_file"]
            if _hash(path) != row["tensor_sha256"]:
                raise ValueError(f"tensor hash mismatch: {path}")
            report["tensor_replays"].append(replay(path, original["dtype"]))
    for dtype in ("bf16", "fp16"):
        directory = evidence / f"stream-{dtype}"
        original = _read(directory / "report.json")
        path = directory / "streaming.safetensors"
        if _hash(path) != original["tensor_sha256"]:
            raise ValueError(f"streaming tensor hash mismatch: {path}")
        report["tensor_replays"].append(replay(path, dtype))
        report["timing"].append(audit_timing(evidence / f"timing-{dtype}.json"))
    report["failed_quantities"] = [
        {"file": item["file"], "dtype": item["dtype"], **row}
        for item in report["tensor_replays"] for row in item["rows"] if not row["passed"]
    ]
    report["counts"] = {
        "tensor_files": len(report["tensor_replays"]),
        "quantities": sum(len(item["rows"]) for item in report["tensor_replays"]),
        "failed_quantities": len(report["failed_quantities"]),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("--aligned-dir", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.evidence_dir, args.aligned_dir)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    Console().print(result["counts"], f"Evidence audit: {args.output}")
    return 1 if result["failed_quantities"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
