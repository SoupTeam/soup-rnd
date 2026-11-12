"""Observe one existing CLI invocation; no retries, scheduling or provider actions."""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path

from rich.console import Console

from benchmarks.harness.arm_validity import ArmWatch
from benchmarks.harness.p3_deepseek_quality_prepare import cgroup_observation, memory_observation
from benchmarks.harness.p3_deepseek_quality_train import _terminal_signals

receipt = Path(sys.argv[1])
module_name = sys.argv[2]
arguments = sys.argv[3:]
if receipt.exists():
    raise FileExistsError(receipt)
receipt.parent.mkdir(parents=True, exist_ok=True)
sys.argv = [module_name, *arguments]
record = {
    "module": module_name, "argv": arguments, "status": "FAILED",
    "scope": "Whole command, covering all nested pass/step/batch timing intervals",
    "cgroup_before": cgroup_observation(),
}
watch = ArmWatch(
    arm=receipt.parent.name, round=1, run=1, model_path=os.environ["ROOT"],
    required=("suspend", "power", "foreign_reads"),
)
memory = memory_observation("cuda", scope=record["scope"])
started = time.monotonic()
returncode = 1
try:
    with _terminal_signals(), watch, memory:
        try:
            result = importlib.import_module(module_name).main()
            returncode = 0 if result is None else int(result)
        except SystemExit as error:
            if error.code is None or isinstance(error.code, int):
                returncode = int(error.code or 0)
            else:
                raise
    record["status"] = "COMPLETE" if returncode == 0 else "FAILED"
except BaseException as error:
    record["error"] = {"type": type(error).__name__, "message": str(error)}
    raise
finally:
    record.update(
        returncode=returncode, wall_seconds=time.monotonic() - started,
        f1=watch.record, timing_valid=bool(watch.record and watch.record["outcome"] == "ok"),
        memory=memory.result, cgroup_after=cgroup_observation(),
    )
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    Console().print({"command_observation": record["status"], "receipt": str(receipt)})
raise SystemExit(returncode)
