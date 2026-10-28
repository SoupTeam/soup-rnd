"""Fresh-process C3 numerical comparison. Physical performance acceptance stays separate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import sys
import time
from pathlib import Path
from typing import Any

from benchmarks.c3.compare import save
from benchmarks.c3.setup_supervisor import supervise


def compare_results(results: list[dict[str, Any]], order: list[str]) -> dict[str, Any]:
    failures = []
    if not results or len(results) != len(order) or set(order) != {"A", "B"}:
        failures.append("incomplete paired order")
    reference = results[0] if results else {}
    for position, item in enumerate(results):
        if (item.get("status") != "ARM_COMPLETE"
                or item.get("order") != order[position:position + 1]):
            failures.append(f"arm {position}: incomplete or wrong order")
            continue
        if item.get("weight_sha256") != reference.get("weight_sha256"):
            failures.append(f"arm {position}: weights differ")
        for field in ("config", "steps_per_arm", "warmup_steps", "mode", "torch"):
            if item.get(field) != reference.get(field):
                failures.append(f"arm {position}: {field} differs")
        rows, ref_rows = item.get("records", []), reference.get("records", [])
        if len(rows) != 1 or len(ref_rows) != 1:
            failures.append(f"arm {position}: single record required")
            continue
        row, ref = rows[0], ref_rows[0]
        if row.get("status") != "COMPLETED" or row.get("changed_adapter") is not True:
            failures.append(f"arm {position}: no completed update")
        for field in ("initial_adapter_sha256", "final_adapter_sha256"):
            if (not re.fullmatch(r"[0-9a-f]{64}", str(row.get(field, "")))
                    or row.get(field) != ref.get(field)):
                failures.append(f"arm {position}: {field} mismatch")
        updates = row.get("updates", [])
        if len(updates) != item.get("steps_per_arm") or not updates:
            failures.append(f"arm {position}: missing steps")
        for step, update in enumerate(updates):
            loss = update.get("loss")
            if (type(loss) not in (int, float) or not math.isfinite(loss)
                    or update.get("step") != step
                    or not re.fullmatch(r"[0-9a-f]{64}", str(update.get("gradient_sha256", "")))):
                failures.append(f"arm {position}: invalid loss, step or gradient evidence")
        if [(x.get("loss"), x.get("gradient_sha256")) for x in updates] != [
            (x.get("loss"), x.get("gradient_sha256")) for x in ref.get("updates", [])
        ]:
            failures.append(f"arm {position}: numerical mismatch")
    return {"status": "NUMERICAL_PASS" if not failures else "BLOCKED",
            "failures": failures, "performance_verdict": "NO_VERDICT"}


CLEANUP_RESERVE_SECONDS = 40


def run(output: Path, mode: str, deadline_unix_s: float | None = None) -> dict[str, Any]:
    if mode not in ("cpu-smoke", "cuda-pilot"):
        raise ValueError("unknown mode")
    if os.name != "posix":
        raise RuntimeError("POSIX supervisor required")
    now = time.time()
    bound = 1800 if mode == "cuda-pilot" else 180
    if deadline_unix_s is None:
        if mode == "cuda-pilot":
            raise ValueError("CUDA requires the outer supervisor's absolute deadline")
        deadline_unix_s = now + bound
    if (type(deadline_unix_s) not in (int, float)
            or not math.isfinite(deadline_unix_s)
            or deadline_unix_s <= now + CLEANUP_RESERVE_SECONDS):
        raise ValueError("Unexpired finite deadline with 40s cleanup reserve required")
    deadline = min(deadline_unix_s, now + bound) - CLEANUP_RESERVE_SECONDS
    output = Path(os.path.realpath(output))
    output.mkdir(parents=True, exist_ok=False)
    order = list("ABBA") * 5 if mode == "cuda-pilot" else list("AB")
    result: dict[str, Any] = {"status": "RUNNING", "order": order, "arms": [],
                              "outer_deadline_unix_s": deadline_unix_s,
                              "work_deadline_unix_s": deadline,
                              "cleanup_reserve_seconds": CLEANUP_RESERVE_SECONDS,
                              "performance_verdict": "NO_VERDICT"}
    try:
        if mode == "cuda-pilot":
            from benchmarks.c3.foreign_read_control import run as foreign_control
            from benchmarks.c3.io_calibration import run as calibrate

            calibration = calibrate(output / "io_calibration")
            if calibration["status"] != "PASS_COUNTER_CALIBRATION":
                raise RuntimeError("physical read calibration blocked; no training arms")
            control = foreign_control(output / "io_calibration", output / "foreign_read_control")
            if control["status"] != "PASS_FOREIGN_READER_DETECTED":
                raise RuntimeError("foreign-reader rejection not proven; no training arms")
        for index, arm in enumerate(order):
            if time.time() >= deadline:
                raise TimeoutError("shared work deadline expired; no new arm")
            directory = output / f"arm_{index:02d}_{arm}"
            command = [sys.executable, "-m", "benchmarks.c3.isolated_arm", "--output",
                       str(directory), "--mode", mode, "--arm", arm]
            script = output / f"arm_{index:02d}.sh"
            script.write_text("#!/bin/sh\nset -eu\nexport CUBLAS_WORKSPACE_CONFIG=:4096:8\nexec "
                              + shlex.join(command) + "\n")
            supervision = supervise(script, hashlib.sha256(script.read_bytes()).hexdigest(),
                                    output / f"supervisor_{index:02d}", 300, deadline)
            if supervision["status"] != "PASS" or not supervision["group_cleanup_confirmed"]:
                raise RuntimeError(f"arm {index} process failed; no retry")
            item = json.loads((directory / "result.json").read_text())
            result["arms"].append(item)
            if item.get("status") != "ARM_COMPLETE":
                raise RuntimeError(f"arm {index} incomplete")
            if index and compare_results(result["arms"], order[:index + 1])["status"] != (
                "NUMERICAL_PASS"
            ):
                raise RuntimeError(f"arm {index} numerical mismatch; stop series")
            save(output / "result.json", result)
        result.update(compare_results(result["arms"], order))
    except Exception as error:
        result.update(status="BLOCKED", error=repr(error))
    finally:
        save(output / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["cpu-smoke", "cuda-pilot"], default="cpu-smoke")
    parser.add_argument("--deadline-unix-s", type=float,
                        help="Same absolute deadline as outer supervisor; required for CUDA")
    args = parser.parse_args()
    result = run(args.output, args.mode, args.deadline_unix_s)
    from rich.console import Console

    Console().print(result["status"])
    return 0 if result["status"] == "NUMERICAL_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
