"""Rebuild the complete D1 measurement index from every retained watched arm."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

from d1_full_study import cases, records
from d1_rules import settings_digest, summarize, write_new_json
from rich.console import Console

DATA_SHA256 = "65025efbd8b11960995d241fb7bb164c9d448abcb2085b9bb66ba2c74dfd8d22"


def qa_contract(case: dict, row: dict) -> None:
    settings = row["settings"]
    expected = {
        "steps": 256,
        "warmup": 20,
        "seq": 1024,
        "seed": 842,
        "rank": 16,
        "lr": 2e-4,
        "dtype": "bfloat16",
        "dataset_sha256": DATA_SHA256,
    }
    if any(settings.get(name) != value for name, value in expected.items()) or (
        row["settings_sha256"] != settings_digest(settings)
    ):
        raise ValueError("QA settings differ from the fixed study")
    if row["updates"] != 276 or row["warmup_updates"] != 20 or len(row["losses"]) != 276:
        raise ValueError("thermal conditioning changed the measured QA update budget")
    if case.get("preheat_steps"):
        elapsed = row.get("thermal_prelude_elapsed_seconds", 0)
        total = row.get("fresh_arm_elapsed_seconds", 0)
        if (
            not math.isfinite(elapsed)
            or elapsed <= 0
            or not math.isfinite(total)
            or total <= elapsed
        ):
            raise ValueError("missing separate thermal-prelude/fresh-arm elapsed time")


def projection_contract(case: dict, row: dict) -> None:
    expected = {key: case[key] for key in ("dtype", "m", "n", "k", "steps", "warmup")}
    expected.update(no_nested=False, warmup_seconds=float(case["warmup_seconds"]))
    if row["settings_sha256"] != settings_digest(expected):
        raise ValueError("projection settings differ from the preregistered full plan")
    if row["shape"] != [case[key] for key in ("m", "n", "k")]:
        raise ValueError("projection geometry differs from its plan")
    for field in ("forward_cuda_ms", "backward_cuda_ms", "step_cuda_ms", "step_wall_ms"):
        values = row.get(field, [])
        if len(values) != case["steps"] or any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            for value in values
        ):
            raise ValueError(f"missing/nonfinite full projection timing array: {field}")
    if (
        row.get("actual_warmup_steps", 0) < case["warmup"]
        or row.get("actual_warmup_seconds", 0) < case["warmup_seconds"]
    ):
        raise ValueError("projection warm-up is shorter than its frozen plan")


def describe(case: dict, rows: list[dict]) -> dict:
    result = {"plan": case, "attempted_arms": len(rows), **summarize(rows)}
    if not rows:
        return result
    comparison = result["comparisons"][0]
    accepted = [row for row in rows if row["block"] in comparison["accepted_block_ids"]]
    if case["kind"] == "projection":
        # Invalid/interrupted arms remain in raw evidence; only accepted blocks
        # contribute to reported timings and the numerical timing contract.
        for row in accepted:
            projection_contract(case, row)
    elif case["kind"] == "train":
        for row in accepted:
            qa_contract(case, row)
    result["accepted_arms"] = len(accepted)
    result["all_attempt_clock_range_mhz"] = [
        min((value for row in rows for value in row.get("clock_sm_mhz", [])), default=None),
        max((value for row in rows for value in row.get("clock_sm_mhz", [])), default=None),
    ]
    fields = (
        (
            "forward_cuda_ms",
            "backward_cuda_ms",
            "step_cuda_ms",
            "dense_gemm_ms",
            "dense_gemm_tflops",
        )
        if case["kind"] == "projection"
        else ("nonpadding_tokens_s", "supervised_tokens_s")
        if case["kind"] == "train"
        else ()
    )
    fields += ("memory_util_pct", "clock_sm_mhz")
    result["descriptive_metrics"] = {
        arm: {
            field: statistics.median(
                statistics.median(row[field]) if isinstance(row[field], list) else row[field]
                for row in accepted
                if row["arm"] == arm
            )
            for field in fields
        }
        for arm in "AB"
        if accepted
    }
    return result


def build_index(root: Path) -> dict:
    plan = json.loads((root / "plan.json").read_text())
    if plan["cases"] != cases():
        raise ValueError("stored plan is not the complete preregistered 98-case cohort")
    described, identities, packages = [], set(), set()
    for case in plan["cases"]:
        rows = records(root / case["name"])
        for row in rows:
            if row.get("status") == "MEASURED":
                identities.add(
                    (
                        row["source_sha"],
                        row["gpu_uuid"],
                        row["rule_sha256"],
                        row["runtime_sha256"],
                        row["harness_sha256"],
                    )
                )
                packages.add(json.dumps(row["packages"], sort_keys=True))
        described.append(describe(case, rows))
    if len(identities) > 1 or len(packages) > 1:
        raise ValueError("full cohort mixes source/GPU/runtime/harness/environment identities")
    if identities and next(iter(identities))[0] != plan["source_sha"]:
        raise ValueError("executed source differs from the frozen plan")
    complete = [
        row
        for row in described
        if len(row["comparisons"]) == 1
        and row["comparisons"][0]["status"] in ("MEASURED", "SAME_PATH")
    ]
    correctness = {}
    for dtype in ("float16", "bfloat16"):
        path = root / ("correctness-" + dtype) / "result.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record["source_sha"] != plan["source_sha"] or record["device"] != "cuda":
                raise ValueError("correctness record differs from the CUDA source cohort")
            correctness[dtype] = record["status"]
        else:
            correctness[dtype] = "NOT_RUN"
    test_log = root / "gpu-tests.log"
    text = test_log.read_text() if test_log.exists() else ""
    tests = {
        name: int(match.group(1)) if (match := re.search(rf"(\d+) {name}", text)) else 0
        for name in ("passed", "skipped", "failed", "deselected")
    }
    tests["errors"] = int(match.group(1)) if (match := re.search(r"(\d+) errors?", text)) else 0
    test_status = root / "gpu-tests.json"
    tests["exit_code"] = (
        json.loads(test_status.read_text())["exit_code"] if test_status.exists() else None
    )
    numerical_ready = all(value == "PASS" for value in correctness.values()) and (
        tests["passed"] == 12
        and tests["failed"] == tests["skipped"] == tests["errors"] == tests["exit_code"] == 0
    )
    return {
        "status": "COMPLETE_MEASUREMENTS_REVIEW_REQUIRED"
        if len(complete) == 98 and numerical_ready
        else "INCOMPLETE",
        "d1_complete": False,
        "independent_acceptance": False,
        "expected_cases": 98,
        "completed_cases": len(complete),
        "expected_projection_cases": 96,
        "completed_projection_cases": sum(row["plan"]["kind"] == "projection" for row in complete),
        "attempted_arms": sum(row["attempted_arms"] for row in described),
        "accepted_arms": sum(row.get("accepted_arms", 0) for row in described),
        "source_sha": plan["source_sha"],
        "identities": sorted(identities),
        "packages": json.loads(next(iter(packages))) if packages else {},
        "cuda_correctness": correctness,
        "existing_gpu_tests": tests,
        "cases": described,
        "scope": "Recomputed F1/clock/identity acceptance and timing contracts for retained "
        "records. Numerical tensor/QA replays and independent review are separate checks. "
        "Ratios and ranges are descriptive; no multiple-comparison significance claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_index(args.input)
    write_new_json(args.output, result)
    Console().print(
        {
            key: result[key]
            for key in ("status", "completed_cases", "completed_projection_cases", "attempted_arms")
        }
    )


if __name__ == "__main__":
    main()
