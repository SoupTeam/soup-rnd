"""Bound one Linux subprocess group and preserve its logs, including on timeout.

This is a process deadline, not a rental controller. It never rents, deletes,
checks a provider balance, uploads evidence or grants scientific admission.
The operator still owns price approval, cumulative accounting and teardown.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console


def _write(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def run_stage(argv: list[str], *, output: Path, seconds: float) -> dict[str, Any]:
    """Run once; kill the original child group. Commands must not detach children."""
    if os.name != "posix":
        raise ValueError("pilot process deadlines require a POSIX host")
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("positive finite stage deadline required")
    if not argv or any(not isinstance(part, str) or not part or "\x00" in part for part in argv):
        raise ValueError("nonempty command arguments required")
    output = Path(os.path.realpath(output))
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + seconds
    record: dict[str, Any] = {
        "schema": "p3-deepseek-quality-stage-v1", "status": "FAILED", "argv": argv,
        "started_utc": datetime.now(timezone.utc).isoformat(), "deadline_seconds": seconds,
        "returncode": None, "scientific_admission": False, "provider_cleanup_performed": False,
        "scope": "subprocess exit and deadline only; inspect the child's scientific receipt",
    }
    process = None
    handlers = {}

    def interrupted(number: int, _frame: Any) -> None:
        raise InterruptedError(f"stage interrupted by signal {number}")

    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            handlers[number] = signal.signal(number, interrupted)
        with (output / "stdout.log").open("xb") as log:
            # Block the handoff race; the child unblocks before executing the command.
            previous_mask = signal.pthread_sigmask(
                signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM},
            )
            try:
                launcher = (
                    "import os,signal,sys;signal.pthread_sigmask(signal.SIG_SETMASK,[]);"
                    "os.execvpe(sys.argv[1],sys.argv[1:],os.environ)"
                )
                process = subprocess.Popen(
                    [sys.executable, "-c", launcher, *argv], stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True,
                )
                record["child_pid"] = process.pid
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            grace = min(5.0, seconds / 5)
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic() - grace))
                record["status"] = "COMPLETE" if process.returncode == 0 else "FAILED"
            except subprocess.TimeoutExpired:
                record["status"] = "TIMED_OUT"
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    pass
    except BaseException as error:
        record["status"] = (
            "INTERRUPTED" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "FAILED"
        )
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        # Do not let a second termination interrupt child cleanup and evidence publication.
        for number in handlers:
            signal.signal(number, signal.SIG_IGN)
        try:
            record["cleanup"] = {"child_reaped": False, "errors": [],
                                 "scope": "original process group; detached children unsupported"}
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    record["cleanup"]["group_signal"] = "SIGKILL sent"
                except ProcessLookupError:
                    record["cleanup"]["group_signal"] = "group already absent"
                except BaseException as error:
                    record["cleanup"]["errors"].append({
                        "operation": "killpg", "type": type(error).__name__, "message": str(error)})
                try:
                    process.wait(timeout=10)
                    record["cleanup"]["child_reaped"] = True
                except BaseException as error:
                    record["cleanup"]["errors"].append({
                        "operation": "wait", "type": type(error).__name__, "message": str(error)})
                record["returncode"] = process.returncode
            if record["cleanup"]["errors"] and record["status"] == "COMPLETE":
                record["status"] = "FAILED"
            record["evidence_snapshot_only"] = bool(record["cleanup"]["errors"])
            record["wall_seconds"] = time.monotonic() - started
            record["finished_utc"] = datetime.now(timezone.utc).isoformat()
            _write(output / "result.json", record)
            _write(output / "manifest.json", {
                "schema": record["schema"], "status": record["status"],
                "files": {path.name: {"bytes": path.stat().st_size, "sha256": _digest(path)}
                          for path in sorted(output.iterdir()) if path.is_file()},
            })
        finally:
            for number, handler in handlers.items():
                signal.signal(number, handler)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    result = run_stage(command, output=args.output, seconds=args.seconds)
    Console().print_json(data=result)
    if result["status"] == "TIMED_OUT":
        return 124
    if result["status"] == "INTERRUPTED":
        return 130
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
