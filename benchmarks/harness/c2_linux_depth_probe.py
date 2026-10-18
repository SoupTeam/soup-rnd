#!/usr/bin/env python3
"""Linux C2 physical-device admission and depth probe; no buffered fallback."""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from c2_depth_probe import ByteMismatchError, pipeline, save, verdict

PROTOCOL = Path(__file__).parents[1] / "c2-linux-nvme-20261018.md"
SPAN = 400 * 2**20


def counters() -> dict:
    result = {}
    for path in sorted(Path("/sys/block").glob("nvme*n*")):
        fields = list(map(int, (path / "stat").read_text().split()))
        result[path.name] = {"read": fields[2] * 512, "write": fields[6] * 512}
    return result


def identify_device(before: dict, after: dict, expected: int) -> str:
    """Reject RAID/multiple devices and a provider name unsupported by real counters."""
    deltas = {name: after[name]["read"] - old["read"]
              for name, old in before.items() if name in after}
    matches = [name for name, amount in deltas.items() if 0.95 <= amount / expected <= 1.05]
    if len(matches) != 1 or any(
        amount >= expected * 0.10 for name, amount in deltas.items() if name not in matches
    ):
        raise ValueError(f"no unique physical NVMe: {deltas}")
    return matches[0]


def arm_reasons(delta: dict, seconds: float, expected: int) -> list[str]:
    if (seconds <= 0 or expected <= 0 or any(
        key not in delta or not math.isfinite(delta[key]) or delta[key] < 0
        for key in ("read", "write")
    )):
        return ["invalid physical counters"]
    reasons = []
    if not 0.8 <= delta["read"] / expected <= 1.2:
        reasons.append("physical read bytes outside 80-120%")
    if delta["write"] / seconds > 2**20:
        reasons.append("foreign writes >1 MiB/s")
    return reasons


def memory() -> dict:
    return {line.split(":")[0]: int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith(("MemTotal:", "MemAvailable:"))}


def gpu_clients() -> list[str]:
    done = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    return sorted(line for line in done.stdout.splitlines() if line.strip())


def temperature_readings(root: Path = Path("/sys/class/hwmon")) -> dict:
    result = {}
    for sensor in root.glob("hwmon*"):
        try:
            name = (sensor / "name").read_text().strip()
            if name not in {"nvme", "coretemp"}:
                continue
            result[str(sensor)] = {
                "name": name,
                "millidegrees_c": {path.name: int(path.read_text())
                                   for path in sensor.glob("temp*_input")},
            }
        except (OSError, ValueError) as error:
            result[str(sensor)] = {"error": repr(error)}
    return result


def snapshot() -> dict:
    done = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,clocks.sm,"
         "memory.used,utilization.gpu,power.draw", "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    return {"time": time.time(), "memory": memory(), "loadavg": os.getloadavg(),
            "gpu": done.stdout, "gpu_clients": gpu_clients(),
            "temperatures": temperature_readings()}


def quiet() -> dict:
    first = counters()
    start = time.monotonic()
    time.sleep(5)
    last = counters()
    seconds = time.monotonic() - start
    delta = {name: {key: (last[name][key] - row[key]) / seconds for key in row}
             for name, row in first.items()}
    if not first or any(value > 2**20 for row in delta.values() for value in row.values()):
        raise RuntimeError(f"NVMe not idle: {delta}")
    return delta


def prepare(path: Path) -> None:
    """Allocated random bytes, never sparse/zero-filled; no throughput result."""
    if path.exists():
        raise ValueError("fixture exists; refusing overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    size = math.ceil(memory()["MemTotal"] * 1.25 / SPAN) * SPAN
    free = os.statvfs(path.parent).f_bavail * os.statvfs(path.parent).f_frsize
    if size + 16 * 2**30 > free:
        raise ValueError("insufficient capacity for a >physical-RAM fixture")
    with path.open("xb") as handle:
        remaining = size
        while remaining:
            chunk = os.urandom(min(4 * 2**20, remaining))
            handle.write(chunk)
            remaining -= len(chunk)
        handle.flush()
        os.fsync(handle.fileno())


def run(args: argparse.Namespace) -> int:
    import torch

    import soup_cli
    from soup_cli.utils.safetensors_reader import open_direct, plan_ranges, read_range_into

    out = Path(args.out)
    if out.exists():
        raise ValueError("sequence exists; use another output name")
    out.parent.mkdir(parents=True, exist_ok=True)
    fixture = Path(args.fixture)
    record: dict[str, Any] = {
        "kind": "Linux physical-NVMe primitive / synthetic random fixture",
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "core_sha256": hashlib.sha256(
            Path(__file__).with_name("c2_depth_probe.py").read_bytes()
        ).hexdigest(),
        "python": sys.version, "torch": torch.__version__, "soup_cli_file": soup_cli.__file__,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "correctness_ok": None, "complete": False, "invalid_reasons": [], "rounds": [],
        "workers": 4, "allocated_slots": 2, "span_bytes": SPAN,
        "allocated_bytes": 2 * SPAN,
    }
    arenas = []
    try:
        record["box_start"] = snapshot()
        if record["box_start"]["gpu_clients"]:
            raise RuntimeError("GPU already has a client before C2 initialization")
        if args.prepare:
            prepare(fixture)
        stat = fixture.stat()
        record["fixture"] = {"path": str(fixture), "size": stat.st_size,
                             "allocated": stat.st_blocks * 512, "mtime_ns": stat.st_mtime_ns}
        if stat.st_size <= memory()["MemTotal"] or stat.st_blocks * 512 < stat.st_size:
            raise ValueError("fixture is sparse or not larger than physical host RAM")
        torch.cuda.init()
        arenas = [torch.empty(SPAN, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
        for arena in arenas:
            arena.zero_()
            if not arena.is_pinned() or arena.data_ptr() % 4096:
                raise RuntimeError("staging not pinned/aligned")
        own_clients = gpu_clients()
        if len(own_clients) != 1:
            raise RuntimeError(f"cannot identify the sole new C2 CUDA client: {own_clients}")

        def works(offset: int, slot: int) -> list:
            def read(start: int, end: int) -> None:
                import fcntl

                with open_direct(str(fixture)) as handle:
                    if not fcntl.fcntl(handle.fileno(), fcntl.F_GETFL) & os.O_DIRECT:
                        raise RuntimeError("buffered fallback is outside this protocol")
                    read_range_into(handle, start, arenas[slot][start - offset : end - offset],
                                    end - start)
            return [lambda lo=lo, hi=hi: read(lo, hi)
                    for lo, hi in plan_ranges(offset, offset + SPAN, 4)]

        def verify(offset: int, slot: int) -> None:
            with fixture.open("rb") as handle:
                handle.seek(offset)
                for position in range(0, SPAN, 2**20):
                    expected = handle.read(2**20)
                    actual = arenas[slot][position : position + 2**20].numpy().tobytes()
                    if len(expected) != 2**20 or actual != expected:
                        raise ByteMismatchError(f"{fixture}:{offset + position}")

        # 1 GiB mapping read, split into the same direct-I/O calls without a rate result.
        record["mapping_idle"] = quiet()
        before = counters()
        with open_direct(str(fixture)) as handle:
            read_range_into(handle, 0, arenas[0], SPAN)
            read_range_into(handle, SPAN, arenas[0], SPAN)
            read_range_into(handle, 2 * SPAN, arenas[0][:224 * 2**20], 224 * 2**20)
        after = counters()
        record["mapping"] = {"before": before, "after": after, "bytes": 2**30}
        device = identify_device(before, after, 2**30)
        record["physical_device"] = device
        record["physical_model"] = (Path("/sys/block") / device / "device/model").read_text()
        offsets = [index * SPAN for index in range(24)]
        record["offsets"] = offsets
        record["correctness_passes"] = []
        save(out, record)
        for depth in (1, 2):
            checked = pipeline(offsets, depth, 4, works, verify)
            record["correctness_passes"].append({"depth": depth, **checked})
            save(out, record)
        record["correctness_ok"] = True
        record["direct_io_verified"] = True
        if args.admit_only:
            record["decision"] = {"verdict": "NO VERDICT", "reason": "admission only"}
            return 0

        orders = ([[args.single_depth]] if getattr(args, "single_depth", None)
                  else [[1, 2] if index % 2 == 0 else [2, 1] for index in range(6)])
        for order in orders:
            entry: dict = {"order": order, "arms": {}}
            record["rounds"].append(entry)
            for depth in entry["order"]:
                arm: dict = {"idle_rates": quiet(), "box_before": snapshot()}
                entry["arms"][str(depth)] = arm
                if (arm["box_before"]["memory"]["MemAvailable"] < 4e9
                        or arm["box_before"]["gpu_clients"] != own_clients):
                    raise RuntimeError("host memory/GPU boundary admission failed")
                pipeline(offsets[:1], depth, 4, works)
                samples = []
                sample_errors = []
                stop = threading.Event()

                def observe() -> None:
                    try:
                        while not stop.wait(0.25):
                            samples.append((time.monotonic(), counters()[device]))
                    except BaseException as error:
                        sample_errors.append(repr(error))

                observer = threading.Thread(target=observe, daemon=True)
                before = counters()[device]
                start = time.monotonic()
                arm["started_unix"] = time.time()
                observer.start()
                try:
                    arm.update(pipeline(offsets, depth, 4, works))
                finally:
                    elapsed = time.monotonic() - start
                    after = counters()[device]
                    stop.set()
                    observer.join()
                    arm["samples"] = samples
                    arm["sample_errors"] = sample_errors
                    record["invalid_reasons"].extend(sample_errors)
                    arm["physical_delta"] = {key: after[key] - before[key] for key in before}
                total = 24 * SPAN
                arm["gb_s"] = total / arm["seconds"] / 1e9
                record["invalid_reasons"].extend(arm_reasons(arm["physical_delta"], elapsed, total))
                stamps = [start, *(stamp for stamp, _ in samples), start + elapsed]
                if len(samples) < 4 or any(b - a > 2 for a, b in zip(stamps, stamps[1:])):
                    record["invalid_reasons"].append("missing samples or suspend gap")
                for position in range(24 - depth, 24):
                    verify(offsets[position], position % depth)
                arm["retained_bytes_ok"] = True
                arm["box_after"] = snapshot()
                if (arm["box_after"]["memory"]["MemAvailable"] < 4e9
                        or arm["box_after"]["gpu_clients"] != own_clients):
                    record["invalid_reasons"].append("host memory/GPU boundary admission failed")
                save(out, record)
                if record["invalid_reasons"]:
                    raise RuntimeError("invalid arm; stop sequence")
        record["complete"] = True
    except ByteMismatchError as error:
        record["correctness_ok"] = False
        record["invalid_reasons"].append(repr(error))
    except BaseException as error:
        record["invalid_reasons"].append(repr(error))
    finally:
        if fixture.exists() and "fixture" in record:
            stat = fixture.stat()
            record["fixture_after"] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            if (stat.st_size != record["fixture"]["size"]
                    or stat.st_mtime_ns != record["fixture"]["mtime_ns"]):
                record["invalid_reasons"].append("fixture changed")
        record["decision"] = verdict(record, reviewed=True)
        save(out, record)
    return 0 if record["complete"] and not record["invalid_reasons"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--admit-only", action="store_true")
    parser.add_argument("--single-depth", type=int, choices=[1, 2],
                        help="one fresh-process F1 arm; no standalone six-pair verdict")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Linux physical-NVMe probe")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
