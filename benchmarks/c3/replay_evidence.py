"""Replay the C3 CUDA evidence gates from a committed evidence packet. No GPU needed.

    python benchmarks/c3/replay_evidence.py benchmarks/c3/evidence/20261031_attempt017

The packet holds the archived per-arm records, calibration and foreign-reader
receipts, CUDA control XML, input receipts and the frozen source manifest. This
script checks every member against the archive manifest, checks that the gate code
it uses is byte-identical to the measured source, then recomputes the numerical,
F1/clock, ownership, residency and I/O checks. It decides nothing new: thresholds
are the frozen ones, and performance stays NO_VERDICT unless the frozen
repetitions rule is met. Prints a JSON report; exit 0 means the recomputed result
matches what the packet's run recorded (including recorded failures).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import statistics
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
GATE_FILES = (
    "benchmarks/c3/io_calibration.py",
    "benchmarks/c3/foreign_read_control.py",
    "benchmarks/harness/arm_validity.py",
)
TOKENS = 255
GiB = 1 << 30


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


class Checks:
    def __init__(self) -> None:
        self.results: dict[str, bool] = {}

    def __call__(self, name: str, condition: Any) -> None:
        self.results[name] = bool(condition)

    @property
    def failed(self) -> list[str]:
        return [name for name, ok in self.results.items() if not ok]


def verify_packet(packet: Path, check: Checks) -> dict[str, Any]:
    meta = _read(packet / "PACKET.json")
    members = _read(packet / "archive_manifest.json")["files"]
    excluded = meta["excluded_members"]
    renamed = meta.get("renamed_members", {})
    for rel, entry in members.items():
        if rel in excluded:
            check(f"excluded_listed:{rel}", excluded[rel]["sha256"] == entry["sha256"])
            continue
        path = packet / "evidence" / renamed.get(rel, rel)
        check(f"member:{rel}", path.is_file() and _sha(path) == entry["sha256"])
    rental = _read(packet / "evidence" / "rental_public.json")
    check(
        "rental_public_derivation",
        rental["_derived_from"]["sha256"] == members["rental.json"]["sha256"],
    )
    return {"members": len(members), "excluded": sorted(excluded), "rental": rental}


def load_gates(packet: Path, check: Checks) -> dict[str, Any]:
    manifest = _read(packet / "evidence" / "source_manifest.json")["files"]
    provenance = {}
    frozen_paths = {}
    for rel in GATE_FILES:
        frozen = packet / "frozen_gate" / (Path(rel).name + ".txt")
        expected = manifest[rel]
        same = (frozen.is_file() and frozen.stat().st_size == expected["bytes"]
                and _sha(frozen) == expected["sha256"])
        provenance[rel] = (
            "frozen measured copy from the packet" if same else "DIFFERS from measured source"
        )
        check(f"gate_code:{rel}", same)
        frozen_paths[rel] = frozen
    frozen = packet / "frozen_gate" / "isolated_compare.py"
    same = _sha(frozen) == manifest["benchmarks/c3/isolated_compare.py"]["sha256"]
    provenance["benchmarks/c3/isolated_compare.py"] = (
        "frozen measured copy from the packet" if same else "DIFFERS from measured source"
    )
    check("gate_code:frozen_isolated_compare", same)
    if any(not ok for name, ok in check.results.items() if name.startswith("gate_code:")):
        raise ValueError("frozen gate source hash or size mismatch; refusing to execute it")
    sys.path.insert(0, str(REPO))

    def load(path: Path):
        name = "_c3_frozen_" + path.name.replace(".", "_") + "_" + _sha(path)[:16]
        loader = importlib.machinery.SourceFileLoader(name, str(path))
        spec = importlib.util.spec_from_file_location(name, path, loader=loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclasses resolves annotations through this entry
        spec.loader.exec_module(module)
        return module

    calibration = load(frozen_paths[GATE_FILES[0]])
    # The measured foreign-reader gate imports its calibration constant. Keep
    # that dependency frozen too, and restore the caller's module cache afterward.
    dependency = "benchmarks.c3.io_calibration"
    previous = sys.modules.get(dependency)
    sys.modules[dependency] = calibration
    try:
        foreign = load(frozen_paths[GATE_FILES[1]])
    finally:
        if previous is None:
            sys.modules.pop(dependency, None)
        else:
            sys.modules[dependency] = previous
    validity = load(frozen_paths[GATE_FILES[2]])
    comparison = load(frozen)

    return dict(
        provenance=provenance,
        compare_results=comparison.compare_results,
        assess_foreign=foreign.assess,
        assess_calibration=calibration.assess,
        CHECKS=validity.CHECKS,
        Run=validity.Run,
        arm_outcome=validity.arm_outcome,
        blocks_to_add=validity.blocks_to_add,
        repeats=validity.repeats,
    )


def replay(packet: Path) -> dict[str, Any]:
    check = Checks()
    packet_info = verify_packet(packet, check)
    gates = load_gates(packet, check)
    e = packet / "evidence"
    rental = packet_info["rental"]
    expected_model = _read(REPO / "benchmarks/c3/representative_model.json")

    phases = ["fetch", "setup", "controls", "prepare", "measurement"]
    for name in phases:
        row = _read(e / f"{name}_supervised" / "result.json")
        if name == "measurement":
            # A stopped series records its own failure; it is reported, not passed.
            phase_ok = row["group_cleanup_confirmed"] is True
        else:
            phase_ok = row["status"] == "PASS" and row["returncode"] == 0
            phase_ok = phase_ok and row["group_cleanup_confirmed"] is True
        check(f"{name}_supervised", phase_ok)
    cases = ET.parse(e / "cuda_tests.xml").findall(".//testcase")
    clean = [
        c for c in cases if not any(c.find(k) is not None for k in ("failure", "error", "skipped"))
    ]
    check("cuda_controls_all_passed", cases and len(clean) == len(cases))
    fetch = _read(e / "artifact_fetch.json")
    check(
        "artifacts_readback",
        fetch["status"] == "PASS_ALL_ARTIFACTS_SHA_READBACK"
        and fetch["verified_bytes"] == fetch["expected_bytes"],
    )
    model = _read(e / "input_receipts/model/C3_DOWNLOAD_RECEIPT.json")
    check(
        "model_sha_sizes",
        model["status"] == "PASS_DOWNLOAD_SHA256" and model["files"] == expected_model["files"],
    )

    m = e / "comparison"
    result = _read(m / "result.json")
    order = result["order"]
    planned_blocks = result.get("planned_abba_blocks", len(order) // 4)
    check("registered_order", order == list("ABBA") * planned_blocks)
    check("cleanup_reserve", result["outer_deadline_unix_s"] - result["work_deadline_unix_s"] == 40)
    prepared = m.parent / "input_receipts/repo/.c3_prepared/prepared.json"
    check(
        "prepared_identity",
        _sha(prepared) == result["prepared_sha256"]
        and _read(prepared)["identity"] == expected_model,
    )

    calibration = _read(m / "io_calibration/receipt.json")
    check(
        "io_calibration",
        gates["assess_calibration"](calibration["samples"])["status"] == "PASS_COUNTER_CALIBRATION"
        and calibration["boot_id"] == rental["boot_id"]
        and calibration["device"] == "nvme0n1",
    )
    for i, s in enumerate(calibration["samples"]):
        check(
            f"io_calibration_raw_{i}",
            s["raw_device_end"] - s["raw_device_start"] == s["device_bytes"]
            and s["raw_process_end"] - s["raw_process_start"] == s["process_bytes"],
        )
    foreign = _read(m / "foreign_read_control/receipt.json")
    check(
        "foreign_reader_control",
        gates["assess_foreign"](foreign["f1"], foreign["completed_bytes"], foreign["hashes_match"])[
            "status"
        ]
        == "PASS_FOREIGN_READER_DETECTED"
        and foreign["f1"]["observer_pid"] != foreign["f1"]["reader_pid"],
    )

    frozen_checks, arm_outcome = gates["CHECKS"], gates["arm_outcome"]
    arm_dirs = sorted(p for p in m.glob("arm_*_*") if p.is_dir())
    completed = result["arms"]
    items, rows, f1_failures, partial = [], [], [], []
    for directory in arm_dirs:
        index = int(directory.name.split("_")[1])
        arm = directory.name.split("_")[2]
        item = _read(directory / "result.json")
        if index >= len(completed):
            partial.append(
                {
                    "index": index,
                    "arm": arm,
                    "status": item.get("status"),
                    "records": len(item.get("records", [])),
                }
            )
            continue
        check(f"arm{index}_aggregate_copy", item == completed[index])
        check(f"arm{index}_registered_position", order[index] == arm)
        sup = _read(m / f"supervisor_{index:02d}" / "result.json")
        check(
            f"arm{index}_supervision",
            sup["status"] == "PASS"
            and sup["returncode"] == 0
            and sup["group_cleanup_confirmed"] is True,
        )
        r = item["records"][0]
        check(
            f"arm{index}_workload",
            item["workload_sha256"] == result["prepared_sha256"]
            and item["effective_tokens_per_step"] == TOKENS
            and item["weight_sha256"] == expected_model,
        )
        f1, clock = r["f1"], r["f1"]["gpu_clock"]
        outcome = arm_outcome(f1["checks"], frozen_checks)
        if outcome != "ok":
            f1_failures.append(
                {
                    "index": index,
                    "arm": arm,
                    "outcome": outcome,
                    "foreign_read_bytes": f1["checks"]["foreign_reads"]["evidence"][
                        "foreign_read_bytes"
                    ],
                }
            )
        check(
            f"arm{index}_clock",
            clock["outcome"] == "ok" and clock["samples"] >= 2 and clock["unknown_samples"] == 0,
        )
        # Frozen rule (benchmarks/harness/l2l_box.confirm_gpu_owner): an exclusive local CUDA
        # initialization on the sole GPU. NVML may report a host PID, not the container PID.
        owner = item["gpu_ownership"]
        check(
            f"arm{index}_ownership",
            owner["before"] == []
            and len(owner["after"]) == 1
            and owner["context_active"] is True
            and owner["single_device"] is True
            and owner["method"] == "exclusive-local-CUDA-initialization",
        )
        stats = r["initial_stats"]
        check(
            f"arm{index}_residency",
            (r["resident_indices"] == [])
            if arm == "A"
            else (0 < len(r["resident_indices"]) < 32 and stats["resident_bytes"] <= 4 * GiB),
        )
        check(f"arm{index}_direct_io", r["direct_io"] is True)
        if arm == "B":
            check(f"arm{index}_accounted_peak", stats["accounted_peak_bytes"] <= 16 * GiB)
            check(f"arm{index}_cache_served", r["final_stats"]["resident_tensor_bytes_served"] > 0)
        steps = [s for s in r["updates"] if not s["warmup"]]
        check(f"arm{index}_steps", len(steps) == 4 and len(r["updates"]) == 6)
        items.append(item)
        rows.append(
            {
                "index": index,
                "arm": arm,
                "f1": f1["outcome"],
                "median_window_seconds": statistics.median(
                    s["instrumented_window_seconds"] for s in steps
                ),
                "mean_process_read_bytes": statistics.mean(
                    s["physical_io"]["process_bytes"] for s in steps
                ),
                "mean_device_read_bytes": statistics.mean(
                    s["physical_io"]["device_bytes"] for s in steps
                ),
                "resident_layers": len(r["resident_indices"]),
                "accounted_peak_bytes": stats["accounted_peak_bytes"],
                "peak_cuda_allocated_bytes": r["peak_cuda_allocated_bytes"],
            }
        )

    numerical = gates["compare_results"](items, order[: len(items)])
    check("numerical_exact_completed_arms", numerical["status"] == "NUMERICAL_PASS")
    complete_series = len(items) == len(order)
    if complete_series:
        blocks = [
            [
                gates["Run"](rows[i + j]["median_window_seconds"], rows[i + j]["f1"])
                for j in range(4)
            ]
            for i in range(0, len(rows), 4)
        ]
        try:
            rep = gates["repeats"](blocks)
            repetition = {
                "sigma": rep.sigma,
                "mean_a": rep.mean_a,
                "blocks_needed": rep.blocks_needed,
                "blocks_to_add": gates["blocks_to_add"](blocks),
            }
        except ValueError as error:
            repetition = {"error": str(error)}
    else:
        repetition = {"error": "series incomplete; repetitions rule not evaluated"}

    summary = {}
    for arm in ("A", "B"):
        group = [row for row in rows if row["arm"] == arm and row["f1"] == "ok"]
        summary[arm] = {
            "valid_arms": len(group),
            "mean_median_window_seconds": statistics.mean(
                r["median_window_seconds"] for r in group
            ),
            "mean_process_read_bytes": statistics.mean(r["mean_process_read_bytes"] for r in group),
            "mean_device_read_bytes": statistics.mean(r["mean_device_read_bytes"] for r in group),
        }
        summary[arm]["tokens_per_second"] = TOKENS / summary[arm]["mean_median_window_seconds"]
    read_reduction = 100 * (
        1 - summary["B"]["mean_process_read_bytes"] / summary["A"]["mean_process_read_bytes"]
    )

    if f1_failures:
        status, verdict = "BLOCKED_FOREIGN_READS", "NO_VERDICT_INVALID_EVIDENCE"
    elif repetition.get("blocks_to_add", 1) != 0:
        status, verdict = "NUMERICAL_PASS_GATES_PASS", "NO_VERDICT_INSUFFICIENT_REPETITIONS"
    else:
        status, verdict = "NUMERICAL_PASS_GATES_PASS", "GATES_PASS_PENDING_REVIEW"
    if check.failed:
        status = "REPLAY_CHECKS_FAILED"
    return {
        "packet": str(packet),
        "status": status,
        "performance_verdict": verdict,
        "failed_checks": check.failed,
        "checks_run": len(check.results),
        "gate_code_provenance": gates["provenance"],
        "packet_members": packet_info["members"],
        "excluded_members": packet_info["excluded"],
        "cuda_controls_passed": len(clean),
        "planned_arms": len(order),
        "completed_arms": len(items),
        "partial_arms": partial,
        "f1_failures": f1_failures,
        "numerical_completed_arms": numerical,
        "repetitions": repetition,
        "summary_valid_arms": summary,
        "descriptive_process_read_reduction_pct": read_reduction,
        "per_arm": rows,
        "scope": (
            "Descriptive timing only; dense Llama 3.1 8B BF16 controlled text; "
            "non-L2L timed path."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("packet", type=Path)
    parser.add_argument("--summary", action="store_true", help="omit per-arm rows")
    args = parser.parse_args()
    report = replay(args.packet.resolve())
    if args.summary:
        report.pop("per_arm")
    print(json.dumps(report, indent=2))
    return 0 if not report["failed_checks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
