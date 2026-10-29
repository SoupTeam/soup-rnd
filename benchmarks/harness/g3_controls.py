"""G3 bounded disk-sensor controls, adapted from the preserved guest calibration.

No model, training, sensor overrides or cleanup.

The foreign mode must be invoked through a second, independent SSH connection.
JSON receipts are the output; stdout carries only a machine-readable handshake.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HARNESS_SHA = "ae7e6e5f4d4df460202c8229deb167c43a350f41538c25dc571f21a5791190f9"
MIB = 1024 * 1024
FIXTURE_BYTES = 1280 * MIB


def write(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)  # Atomic publication, refuses to replace any prior receipt.


def ancestors(pid: int) -> list[int]:
    result = []
    while pid:
        if pid in result or len(result) > 1024:
            raise ValueError("Invalid process ancestry")
        result.append(pid)
        status = Path(f"/proc/{pid}/status").read_text()
        pid = int(next(line.split()[1] for line in status.splitlines()
                       if line.startswith("PPid:")))
    return result


def independent(chain: list[int], watcher: int) -> bool:
    return bool(chain) and watcher > 0 and watcher not in chain


def direct_read(fixture: Path, mib: int) -> dict:
    if mib not in (64, 128, 192, 1280):
        raise ValueError("Read exceeds frozen calibration choices")
    command = ["dd", f"if={fixture}", "of=/dev/null", "bs=4M", f"count={mib // 4}",
               "iflag=direct", "status=none"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f"Direct I/O failed; no fallback: {result.stderr[:2048]}")
    return {"command": command, "exit_code": result.returncode,
            "requested_bytes": mib * MIB, "stderr": result.stderr}


def mapping_ok(rows: list[dict]) -> bool:
    # This only tests path response. Default F1 controls decide foreign-I/O validity.
    return len(rows) == 3 and all(
        isinstance(row.get("before"), int) and isinstance(row.get("after"), int)
        and row["requested_bytes"] > 0
        and row["after"] - row["before"] >= row["requested_bytes"] for row in rows
    )


def controls_ok(rows: dict) -> bool:
    for name, expected in (("clean", "ok"), ("own", "ok"), ("foreign", "void")):
        record = rows.get(name, {})
        checks = record.get("checks", {})
        if record.get("outcome") != expected:
            return False
        if checks.get("foreign_reads", {}).get("outcome") != expected:
            return False
        if any(checks.get(key, {}).get("outcome") != "ok" for key in ("suspend", "power")):
            return False
        if record.get("gpu_clock", {}).get("outcome") != "ok":
            return False
    return True


def foreign(out: Path) -> None:
    ready = json.loads((out / "foreign_ready.json").read_text())
    if not 0 <= time.time() - ready["unix_s"] <= 180:
        raise ValueError("Stale foreign-reader handshake")
    watcher = ready["watcher_pid"]
    chain = ancestors(os.getpid())
    if not independent(chain, watcher):
        raise ValueError("Foreign reader is inside watcher tree")
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != ready["boot_id"]:
        raise ValueError("Boot identity changed")
    started = time.time()
    read = direct_read(out / "fixture.bin", 1280)
    write(out / "foreign_done.json", {
        "watcher_pid": watcher, "reader_pid": os.getpid(), "ancestors": chain,
        "started_unix_s": started, "ended_unix_s": time.time(), "read": read,
    })


def calibrate(out: Path, harness_path: Path, expected_device: str) -> None:
    if hashlib.sha256(harness_path.read_bytes()).hexdigest() != HARNESS_SHA:
        raise ValueError("Frozen F1 differs")
    if shutil.disk_usage(out.parent).free < FIXTURE_BYTES + 2 * 1024**3:
        raise ValueError("Insufficient fixture plus output reserve")
    out.mkdir(exist_ok=False)
    spec = importlib.util.spec_from_file_location("g3_disk_calibration_f1", harness_path)
    harness = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = harness
    spec.loader.exec_module(harness)
    result = {"status": "STARTED", "scope": "declared disk only; physical-host isolation unproven",
              "harness_sha256": HARNESS_SHA, "controls": {}, "mapping": [],
              "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
              "unix_s": time.time(),
              "physical_host_isolation_certified": False, "training_or_timing_accepted": False}
    try:
        device = harness.disk_device(str(out))
        if device != expected_device:
            raise ValueError(f"Expected predeclared device {expected_device}, got {device}")
        result["device"] = device
        fixture = out / "fixture.bin"
        # No sparse allocation, page-cache dropping, privileges or shared-file edits.
        with fixture.open("xb") as handle:
            chunk = bytes(4 * MIB)
            for _ in range(320):
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        for mib in (64, 192, 64):
            before = harness.disk_reads(device)
            read = direct_read(fixture, mib)
            after = harness.disk_reads(device)
            result["mapping"].append({**read, "before": before, "after": after})
        if not mapping_ok(result["mapping"]):
            raise ValueError("Direct reads undercount or lack counters; preserve and block")
        for name in ("clean", "own", "foreign"):
            watch = harness.ArmWatch(arm=name, round=1, run=1, model_path=str(out))
            watch.start()
            try:
                if name == "clean":
                    time.sleep(0.25)
                elif name == "own":
                    result["own_read"] = direct_read(fixture, 128)
                else:
                    started = time.time()
                    write(out / "foreign_ready.json", {
                        "watcher_pid": os.getpid(), "unix_s": started,
                        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                    })
                    sys.stdout.write("FOREIGN_READY\n")
                    sys.stdout.flush()
                    deadline = time.monotonic() + 180
                    while not (out / "foreign_done.json").exists():
                        if time.monotonic() > deadline:
                            raise TimeoutError("Independent foreign reader did not finish")
                        time.sleep(0.1)
                    done = json.loads((out / "foreign_done.json").read_text())
                    if (done["watcher_pid"] != os.getpid()
                            or not independent(done["ancestors"], os.getpid())
                            or done["started_unix_s"] < started):
                        raise ValueError("Invalid independent-reader evidence")
                    result["foreign_reader"] = done
            finally:
                result["controls"][name] = watch.stop()
        result["status"] = "PASS_DISK_SENSOR_CONTROLS" if controls_ok(
            result["controls"]
        ) else "BLOCKED_CONTROL_MISMATCH"
    except Exception as error:
        result["status"] = "BLOCKED_EXCEPTION"
        result["error"] = repr(error)
        raise
    finally:
        write(out / "calibration.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("calibrate", "foreign"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--harness", type=Path)
    parser.add_argument("--expected-device")
    args = parser.parse_args()
    if sys.platform != "linux":
        raise ValueError("Live controls require Linux")
    signal.alarm(900)
    if args.mode == "foreign":
        foreign(args.out)
    elif args.harness and args.expected_device:
        calibrate(args.out, args.harness, args.expected_device)
    else:
        parser.error("calibrate requires --harness and --expected-device")


if __name__ == "__main__":
    main()
