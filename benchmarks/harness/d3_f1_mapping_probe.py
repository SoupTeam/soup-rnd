"""Read-only D3 overlay-to-device discovery; not timing or host certification.

Associate each explicitly named immutable shard with exposed cgroup/block
counters using two exact 1 MiB O_DIRECT reads. Ambiguity is a refusal, never
an arbitrary disk override. Separate real F1 controls must still pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import sys
import traceback
from pathlib import Path

READ_BYTES = 1024 * 1024


class MappingFailureError(ValueError):
    def __init__(self, message: str, evidence: dict) -> None:
        super().__init__(message)
        self.evidence = evidence


def io_reads(text: str) -> dict[str, int]:
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        device = fields[0]
        if (len(device.split(":")) != 2 or not all(part.isdecimal() for part in device.split(":"))
                or device in result):
            raise ValueError("invalid or duplicate cgroup device")
        values = dict(field.split("=", 1) for field in fields[1:])
        count = int(values["rbytes"])
        if count < 0:
            raise ValueError("negative cgroup reads")
        result[device] = count
    if not result:
        raise ValueError("no readable cgroup I/O devices")
    return result


def exact_read_device(before: dict[str, int], after: dict[str, int], count: int) -> str:
    if type(count) is not int or count <= 0:
        raise ValueError("positive direct-read byte count required")
    deltas = {device: after.get(device, 0) - before.get(device, 0)
              for device in before.keys() | after.keys()}
    if any(value < 0 for value in deltas.values()):
        raise ValueError("cgroup I/O counters reset")
    changed = [device for device, value in deltas.items() if value]
    if len(changed) != 1 or deltas[changed[0]] != count:
        raise ValueError("ambiguous backing device: not one exact direct-read delta")
    return changed[0]


def direct_read(path: Path, *, rounds: int = 1) -> int:
    if type(rounds) is not int or not 1 <= rounds <= 1024:
        raise ValueError("direct-read rounds must be from 1 through 1024")
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    try:
        buffer = mmap.mmap(-1, READ_BYTES)
        try:
            total = 0
            for _ in range(rounds):
                os.lseek(fd, 0, os.SEEK_SET)
                read = os.readv(fd, [buffer])
                if read != READ_BYTES:
                    raise RuntimeError("short direct read during mapping/control")
                total += read
            return total
        finally:
            buffer.close()
    finally:
        os.close(fd)


def _mount_type(path: Path) -> str | None:
    """Return the longest matching mount type from mountinfo."""
    target = os.path.realpath(path)
    matches = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, separator, right = line.partition(" - ")
        fields = left.split()
        if not separator or len(fields) < 6:
            continue
        mountpoint = fields[4]
        for escaped, character in (("\\040", " "), ("\\011", "\t"),
                                   ("\\012", "\n"), ("\\134", "\\")):
            mountpoint = mountpoint.replace(escaped, character)
        mountpoint = os.path.realpath(mountpoint)
        if target == mountpoint or target.startswith(mountpoint.rstrip("/") + "/"):
            matches.append((len(mountpoint), right.split()[0]))
    return max(matches)[1] if matches else None


def _is_overlay_loop(name: str, mount_type: str | None, allowed: bool) -> bool:
    return allowed and name.startswith("loop") and mount_type == "overlay"


def discover(paths: list[Path], *, allow_overlay_loop: bool = False) -> dict:
    if sys.platform != "linux":
        raise ValueError("Linux cgroup v2 and O_DIRECT required")
    cg = Path("/sys/fs/cgroup/io.stat")
    result = {"status": "requires_review", "kind": "instrument mapping, not D3 results",
              "scope": "empirical exposed-device association; not physical-host isolation",
              "cgroup": Path("/proc/self/cgroup").read_text(), "files": []}
    if allow_overlay_loop:
        result["scope"] = ("container-visible overlay loop-device association; "
                           "physical-host storage and isolation are not certified")
    for path in paths:
        start = path.stat()
        if start.st_size < READ_BYTES:
            raise ValueError("mapping requires a shard of at least 1 MiB")
        records = []
        devices = []
        row = {"path": os.path.realpath(path), "st_dev": start.st_dev, "inode": start.st_ino,
               "size": start.st_size, "mtime_ns": start.st_mtime_ns, "reads": records}
        result["files"].append(row)
        # Prime filesystem extent/header metadata BEFORE the exact association.
        # This read is retained but contributes no device verdict by itself.
        row["metadata_warm_read_bytes"] = direct_read(path)
        for _ in range(2):
            before = io_reads(cg.read_text())
            count = direct_read(path)
            after = io_reads(cg.read_text())
            records.append({"before": before, "after": after, "direct_bytes": count})
            try:
                device = exact_read_device(before, after, count)
            except ValueError as exc:
                raise MappingFailureError(str(exc), result) from exc
            devices.append(device)
        if devices[0] != devices[1]:
            raise ValueError("backing-device association changed between reads")
        device = devices[0]
        sys_path = Path("/sys/dev/block") / device
        name = os.path.basename(os.path.realpath(sys_path))
        virtual_overlay = _is_overlay_loop(name, _mount_type(path), allow_overlay_loop)
        if (not sys_path.exists() or not name
                or (name.startswith(("loop", "ram", "zram")) and not virtual_overlay)):
            raise ValueError("associated device is not an exposed disk")
        disk_rows = [line.split() for line in Path("/proc/diskstats").read_text().splitlines()]
        if not any(f"{row[0]}:{row[1]}" == device and row[2] == name for row in disk_rows):
            raise ValueError("sysfs and diskstats device identities disagree")
        end = path.stat()
        if (start.st_dev, start.st_ino, start.st_size, start.st_mtime_ns) != (
            end.st_dev, end.st_ino, end.st_size, end.st_mtime_ns,
        ):
            raise ValueError("shard identity changed during discovery")
        row.update(device=device, disk_name=name)
        if virtual_overlay:
            row["filesystem"] = "overlay"
            row["device_scope"] = "container-visible loop device"
    result["status"] = "mapping_observed_controls_still_required"
    return result


def main(argv: list[str] | None = None) -> int:
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-overlay-loop", action="store_true",
                        help="explicitly scope mapping to an overlay-backed loop device")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    result = {"status": "requires_review", "d3_accepted": False}
    try:
        result.update(discover(args.paths, allow_overlay_loop=args.allow_overlay_loop))
    except (Exception, KeyboardInterrupt) as exc:
        result["traceback"] = traceback.format_exc()
        if isinstance(exc, MappingFailureError):
            result["partial_evidence"] = exc.evidence
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.out / "mapping.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    Console().print(f"{result['status']}: {args.out}")
    return 0 if result["status"] == "mapping_observed_controls_still_required" else 1


if __name__ == "__main__":
    raise SystemExit(main())
