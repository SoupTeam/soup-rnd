#!/usr/bin/env python3
"""Validity checks for one benchmark arm (F1).

Wrap an arm in ``ArmWatch`` and read ``watch.record`` afterwards: the arm's
identity, one entry per check under ``checks`` and the arm's overall
``outcome``. Each check ends in ok, void (the sensor answered and the arm is
contaminated) or unknown (the sensor gave no answer). A sensor that raises or
returns None makes its check unknown, never ok (ADR 0001).

Sensors are callables passed in by the caller, with real Linux defaults that
need no root and no third-party packages.

Benchmark harness code, not shipped. Standard library only.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Collection, Dict, List, NamedTuple, Optional, Tuple, TypedDict, TypeVar

_T = TypeVar("_T")

OK = "ok"
VOID = "void"
UNKNOWN = "unknown"

#: Boot-time minus monotonic time grows by the time spent suspended. Growth past
#: this many seconds during an arm means the machine slept.
SLEEP_OFFSET_LIMIT_S = 1.0

SUSPEND_SUCCESS_PATH = "/sys/power/suspend_stats/success"

POWER_SUPPLY_DIR = "/sys/class/power_supply"

#: Seconds between power samples during an arm.
POWER_INTERVAL_S = 2.0

#: Foreign reads of this many bytes or more from the disk under test void an arm
#: (validity row V6).
FOREIGN_READ_LIMIT_BYTES = 1_000_000_000

PROC_DIR = "/proc"

DISKSTATS_PATH = "/proc/diskstats"

#: /proc/diskstats counts 512-byte sectors whatever the hardware sector size.
DISKSTATS_SECTOR_BYTES = 512

#: Every check an arm record carries, in record order.
CHECKS = ("suspend", "power", "foreign_reads")


def sleep_offset() -> Optional[float]:
    """Boot-time minus monotonic time in seconds; None where the OS has no boot-time clock."""
    boottime = getattr(time, "CLOCK_BOOTTIME", None)
    if boottime is None:
        return None
    return time.clock_gettime(boottime) - time.clock_gettime(time.CLOCK_MONOTONIC)


def suspend_count() -> Optional[int]:
    """The kernel's count of successful suspends since boot."""
    with open(SUSPEND_SUCCESS_PATH, encoding="ascii") as handle:
        return int(handle.read().strip())


class PowerReading(TypedDict):
    """One read of the power supplies: whether a battery is present, and which
    non-battery supplies are online."""

    battery: bool
    online: Dict[str, bool]


def _supply_attribute(supply: str, name: str) -> Optional[str]:
    """One sysfs attribute of a power supply, or None if the supply lacks it."""
    try:
        with open(os.path.join(supply, name), encoding="ascii") as handle:
            return handle.read().strip()
    except FileNotFoundError:
        return None


def power_supplies() -> PowerReading:
    """A ``PowerReading`` from the kernel's power supplies.

    Follows the kernel's ``power_supply_is_system_supplied()``: supplies with
    scope "Device" (a mouse battery, say) are skipped, and ``online`` covers
    every remaining supply that is not a battery and has an ``online`` value.
    ``battery`` says whether any remaining supply is a battery.
    """
    battery = False
    online: Dict[str, bool] = {}
    for name in sorted(os.listdir(POWER_SUPPLY_DIR)):
        supply = os.path.join(POWER_SUPPLY_DIR, name)
        if _supply_attribute(supply, "scope") == "Device":
            continue
        kind = _supply_attribute(supply, "type")
        if kind == "Battery":
            battery = True
            continue
        state = _supply_attribute(supply, "online")
        if kind is not None and state is not None:
            online[name] = int(state) != 0
    return {"battery": battery, "online": online}


def _whole_disks(sys_path: str) -> List[str]:
    """Whole disks under a /sys/devices block entry: a partition's parent, or the
    disks under a device-mapper entry's slaves."""
    if os.path.exists(os.path.join(sys_path, "partition")):
        sys_path = os.path.dirname(sys_path)
    slaves = os.path.join(sys_path, "slaves")
    names = sorted(os.listdir(slaves)) if os.path.isdir(slaves) else []
    if not names:
        return [os.path.basename(sys_path)]
    disks: List[str] = []
    for name in names:
        disks += _whole_disks(os.path.realpath(os.path.join(slaves, name)))
    return sorted(set(disks))


def disk_device(path: str) -> Optional[str]:
    """The whole disk that holds ``path``, as named in /proc/diskstats.

    None if the file system has no block device (tmpfs, btrfs subvolumes,
    network mounts). Raises ValueError if the device spans several disks.
    """
    st_dev = os.stat(path).st_dev
    entry = f"/sys/dev/block/{os.major(st_dev)}:{os.minor(st_dev)}"
    if not os.path.exists(entry):
        return None
    disks = _whole_disks(os.path.realpath(entry))
    if len(disks) != 1:
        raise ValueError(f"{path} spans several disks: {', '.join(disks)}")
    return disks[0]


def disk_reads(device: str) -> Optional[int]:
    """Bytes read from ``device`` since boot, from /proc/diskstats; None if it is not listed."""
    with open(DISKSTATS_PATH, encoding="ascii") as handle:
        for line in handle:
            fields = line.split()
            if fields[2] == device:
                return int(fields[5]) * DISKSTATS_SECTOR_BYTES
    return None


class ProcessRead(TypedDict):
    name: str
    read_bytes: int


class ProcessReads(TypedDict):
    """One scan of /proc.

    ``own`` is the storage bytes read by this process (its waited-for children
    included) and by its live descendants. ``others`` holds every other process
    of this user whose /proc/<pid>/io is readable, by pid. ``uninspectable``
    counts the processes whose counters were denied or belong to other users.
    """

    own: int
    others: Dict[int, ProcessRead]
    uninspectable: int


def _io_read_bytes(pid: int) -> int:
    with open(os.path.join(PROC_DIR, str(pid), "io"), encoding="ascii") as handle:
        for line in handle:
            if line.startswith("read_bytes:"):
                return int(line.split()[1])
    raise ValueError(f"{PROC_DIR}/{pid}/io has no read_bytes")


def _name_and_parent(pid: int) -> Tuple[str, int]:
    path = os.path.join(PROC_DIR, str(pid), "stat")
    with open(path, encoding="utf-8", errors="replace") as handle:
        stat = handle.read()
    name_end = stat.rindex(")")
    return stat[stat.index("(") + 1 : name_end], int(stat[name_end + 2 :].split()[1])


def process_reads() -> ProcessReads:
    """A ``ProcessReads`` scan of /proc. Needs no root; denied counters are counted, not skipped.

    Raises PermissionError if a process of our own tree is unreadable: Linux
    denies an exited child's counter until it is reaped, and its reads would
    otherwise count as foreign.
    """
    names: Dict[int, str] = {}
    parents: Dict[int, int] = {}
    for entry in os.listdir(PROC_DIR):
        if entry.isdigit():
            try:
                names[int(entry)], parents[int(entry)] = _name_and_parent(int(entry))
            except (FileNotFoundError, ProcessLookupError):
                continue  # exited during the scan
    own_tree = {os.getpid()}
    grew = True
    while grew:
        children = {pid for pid, parent in parents.items() if parent in own_tree}
        grew = not children <= own_tree
        own_tree |= children
    own = 0
    others: Dict[int, ProcessRead] = {}
    uninspectable = 0
    uid = os.getuid()
    for pid, name in names.items():
        try:
            if pid not in own_tree and os.stat(os.path.join(PROC_DIR, str(pid))).st_uid != uid:
                uninspectable += 1
                continue
            read_bytes = _io_read_bytes(pid)
        except PermissionError:
            if pid in own_tree:
                raise PermissionError(
                    f"cannot read the counter of own child {pid} ({name}); "
                    "reap child processes before the arm stops"
                ) from None
            uninspectable += 1
            continue
        except (FileNotFoundError, ProcessLookupError):
            continue
        if pid in own_tree:
            own += read_bytes
        else:
            others[pid] = {"name": name, "read_bytes": read_bytes}
    return {"own": own, "others": others, "uninspectable": uninspectable}


def _read(
    name: str, sensor: Callable[[], Optional[_T]], none_problem: Optional[str] = None
) -> Tuple[Optional[_T], Optional[str]]:
    """``(value, problem)``: the reading, or None and why there is none."""
    try:
        value = sensor()
    except Exception as exc:  # noqa: BLE001 - any sensor failure is an unknown
        return None, f"{name} sensor raised {type(exc).__name__}: {exc}"
    if value is None:
        return None, none_problem or f"{name} sensor returned None"
    return value, None


def _check_result(outcome: str, reason: str, evidence: Dict[str, object]) -> Dict[str, object]:
    return {"outcome": outcome, "reason": reason, "evidence": evidence}


@dataclass
class SleepReadings:
    """One reading of both sleep sensors, and why either gave nothing."""

    offset: Optional[float]
    count: Optional[int]
    problems: List[str]

    @classmethod
    def take(
        cls,
        offset_sensor: Callable[[], Optional[float]],
        count_sensor: Callable[[], Optional[int]],
    ) -> "SleepReadings":
        offset, offset_problem = _read("boot-time offset", offset_sensor)
        count, count_problem = _read("suspend count", count_sensor)
        return cls(offset, count, [p for p in (offset_problem, count_problem) if p])


def suspend_check_result(start: SleepReadings, end: SleepReadings) -> Dict[str, object]:
    """Void if boot-time offset grew past the limit or the suspend count rose.

    Either sensor seeing sleep is enough to void. Otherwise a silent sensor
    makes the check unknown. There is no wall-clock gap rule on Linux.
    """
    evidence: Dict[str, object] = {
        "sleep_offset_start_s": start.offset,
        "sleep_offset_end_s": end.offset,
        "suspend_count_start": start.count,
        "suspend_count_end": end.count,
    }
    slept = []
    growth = None
    if start.offset is not None and end.offset is not None:
        growth = end.offset - start.offset
        if growth > SLEEP_OFFSET_LIMIT_S:
            slept.append(f"boot-time offset grew {growth:.3f} s (limit {SLEEP_OFFSET_LIMIT_S} s)")
    if start.count is not None and end.count is not None and end.count > start.count:
        slept.append(f"suspend count rose from {start.count} to {end.count}")
    if slept:
        return _check_result(VOID, "; ".join(slept), evidence)
    problems = start.problems + end.problems
    if problems:
        return _check_result(UNKNOWN, "; ".join(problems), evidence)
    return _check_result(
        OK,
        f"boot-time offset grew {growth:.3f} s (limit {SLEEP_OFFSET_LIMIT_S} s); "
        f"suspend count stayed at {end.count}",
        evidence,
    )


class PowerSample(NamedTuple):
    """One power sample. ``problem`` says why ``reading`` is None."""

    at_s: float
    reading: Optional[PowerReading]
    problem: Optional[str]


class PowerSampler:
    """Reads the power sensor at start, every ``interval_s`` seconds on a thread, and at stop."""

    def __init__(self, sensor: Callable[[], Optional[PowerReading]], interval_s: float) -> None:
        self._sensor = sensor
        self._interval_s = interval_s
        self._stopped = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0
        self.samples: List[PowerSample] = []

    def _sample(self) -> None:
        reading, problem = _read("power", self._sensor)
        self.samples.append(PowerSample(time.monotonic() - self._t0, reading, problem))

    def _loop(self) -> None:
        while not self._stopped.wait(self._interval_s):
            self._sample()

    def start(self) -> None:
        self._t0 = time.monotonic()
        self._sample()
        self._thread = threading.Thread(target=self._loop, name="arm-power-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()


def _on_mains(reading: PowerReading) -> bool:
    """A machine with no battery runs on mains; otherwise some non-battery supply must be online."""
    return not reading["battery"] or any(reading["online"].values())


def power_check_result(samples: List[PowerSample]) -> Dict[str, object]:
    """Void if any sample shows the machine off mains.

    Otherwise a sample with no reading makes the check unknown.
    """
    off_mains = [
        {"sample": index, "at_s": round(sample.at_s, 3), "online": sample.reading["online"]}
        for index, sample in enumerate(samples)
        if sample.reading is not None and not _on_mains(sample.reading)
    ]
    readings = [sample.reading for sample in samples if sample.reading is not None]
    problems = [sample.problem for sample in samples if sample.problem]
    evidence: Dict[str, object] = {
        "samples": len(samples),
        "off_mains_samples": off_mains,
        "battery_present": any(reading["battery"] for reading in readings),
        "unknown_samples": len(problems),
        "last_reading": readings[-1] if readings else None,
    }
    if off_mains:
        return _check_result(
            VOID, f"{len(off_mains)} of {len(samples)} power samples off mains", evidence
        )
    if problems:
        return _check_result(
            UNKNOWN,
            f"{len(problems)} of {len(samples)} power samples gave no reading; "
            f"first: {problems[0]}",
            evidence,
        )
    if not evidence["battery_present"]:
        return _check_result(
            OK, f"no battery in any of {len(samples)} power samples, so on mains", evidence
        )
    return _check_result(OK, f"on mains in all {len(samples)} power samples", evidence)


@dataclass
class DiskReadings:
    """One reading of the disk counter and the process scan, and why either gave nothing."""

    disk: Optional[int]
    processes: Optional[ProcessReads]
    problems: List[str]

    @classmethod
    def take(
        cls,
        device: Optional[str],
        disk_sensor: Callable[[str], Optional[int]],
        process_sensor: Callable[[], Optional[ProcessReads]],
    ) -> "DiskReadings":
        if device is None:
            disk, disk_problem = None, None
        else:
            disk, disk_problem = _read("disk read", lambda: disk_sensor(device))
        processes, process_problem = _read("process read", process_sensor)
        return cls(disk, processes, [p for p in (disk_problem, process_problem) if p])


class Reader(TypedDict):
    pid: int
    name: str
    read_bytes: int


def _readers(start: ProcessReads, end: ProcessReads) -> List[Reader]:
    """Other processes that read during the arm, most bytes first. A pid that is
    new, or now has another name, counts its whole counter."""
    readers: List[Reader] = []
    for pid, now in end["others"].items():
        before = start["others"].get(pid)
        base = before["read_bytes"] if before and before["name"] == now["name"] else 0
        if now["read_bytes"] > base:
            read = now["read_bytes"] - base
            readers.append({"pid": pid, "name": now["name"], "read_bytes": read})
    return sorted(readers, key=lambda reader: (-reader["read_bytes"], reader["pid"]))


def foreign_reads_check_result(
    model_path: Optional[str],
    device: Optional[str],
    device_problem: Optional[str],
    start: DiskReadings,
    end: DiskReadings,
) -> Dict[str, object]:
    """Void if the disk under test read 1 GB or more beyond this process tree's reads (ADR 0002).

    ``readers`` names the other processes this user may inspect that read during
    the arm; ``unattributed_bytes`` is the foreign bytes they do not explain,
    floored at 0 because a reader's counter covers every disk.
    """
    evidence: Dict[str, object] = {
        "model_path": model_path,
        "device": device,
        "disk_read_bytes_start": start.disk,
        "disk_read_bytes_end": end.disk,
        "own_read_bytes": None,
        "foreign_read_bytes": None,
        "uninspectable_processes": None,
    }
    problems = ([device_problem] if device_problem else []) + start.problems + end.problems
    if problems:
        result = _check_result(UNKNOWN, "; ".join(problems), evidence)
        result.update({"readers": [], "unattributed_bytes": None})
        return result
    assert start.disk is not None and end.disk is not None
    assert start.processes is not None and end.processes is not None
    own = end.processes["own"] - start.processes["own"]
    foreign = end.disk - start.disk - own
    readers = _readers(start.processes, end.processes)
    evidence.update(
        own_read_bytes=own,
        foreign_read_bytes=foreign,
        uninspectable_processes=end.processes["uninspectable"],
    )
    result = _check_result(
        VOID if foreign >= FOREIGN_READ_LIMIT_BYTES else OK,
        f"{foreign} bytes of foreign reads on {device} (limit {FOREIGN_READ_LIMIT_BYTES})",
        evidence,
    )
    named_bytes = sum(reader["read_bytes"] for reader in readers)
    result.update({"readers": readers, "unattributed_bytes": max(foreign - named_bytes, 0)})
    return result


def arm_outcome(checks: Dict[str, Dict[str, object]], required: Collection[str]) -> str:
    """Void if any check is void, else unknown if any required check is unknown, else ok."""
    if any(check["outcome"] == VOID for check in checks.values()):
        return VOID
    if any(checks[name]["outcome"] == UNKNOWN for name in required):
        return UNKNOWN
    return OK


class ArmWatch:
    """Watches one arm. Use as a context manager, then read ``record``.

    ``required`` names the checks whose unknown makes the arm unknown; it
    defaults to every check. A void check voids the arm whether required or not.
    ``model_path`` is the path to the model files; the foreign reads check
    watches the disk that holds it, and is unknown without it.
    """

    def __init__(
        self,
        *,
        arm: str,
        round: int,
        run: int,
        label: Optional[str] = None,
        required: Optional[Collection[str]] = None,
        sleep_offset: Callable[[], Optional[float]] = sleep_offset,
        suspend_count: Callable[[], Optional[int]] = suspend_count,
        power_supplies: Callable[[], Optional[PowerReading]] = power_supplies,
        power_interval_s: float = POWER_INTERVAL_S,
        model_path: Optional[str] = None,
        disk_device: Callable[[str], Optional[str]] = disk_device,
        disk_reads: Callable[[str], Optional[int]] = disk_reads,
        process_reads: Callable[[], Optional[ProcessReads]] = process_reads,
    ):
        self.arm = arm
        self.round = round
        self.run = run
        self.label = label
        self.required = list(CHECKS if required is None else required)
        unknown_names = sorted(set(self.required) - set(CHECKS))
        if unknown_names:
            raise ValueError(f"no such checks: {', '.join(unknown_names)}")
        self._sleep_offset = sleep_offset
        self._suspend_count = suspend_count
        self._start: Optional[SleepReadings] = None
        self._power = PowerSampler(power_supplies, power_interval_s)
        self.model_path = model_path
        self._disk_device = disk_device
        self._disk_reads = disk_reads
        self._process_reads = process_reads
        self._device: Optional[str] = None
        self._device_problem: Optional[str] = None
        self._disk_start: Optional[DiskReadings] = None
        self.record: Optional[Dict[str, object]] = None

    def _find_device(self) -> None:
        if self.model_path is None:
            self._device_problem = "no model path given"
            return
        path = self.model_path
        self._device, self._device_problem = _read(
            "disk device", lambda: self._disk_device(path), f"{path} maps to no block device"
        )

    def _disk_readings(self) -> DiskReadings:
        return DiskReadings.take(self._device, self._disk_reads, self._process_reads)

    def start(self) -> None:
        self._find_device()
        self._disk_start = self._disk_readings()
        self._start = SleepReadings.take(self._sleep_offset, self._suspend_count)
        self._power.start()

    def stop(self) -> Dict[str, object]:
        if self._start is None or self._disk_start is None:
            raise RuntimeError("stop() called before start()")
        self._power.stop()
        end = SleepReadings.take(self._sleep_offset, self._suspend_count)
        checks = {
            "suspend": suspend_check_result(self._start, end),
            "power": power_check_result(self._power.samples),
            "foreign_reads": foreign_reads_check_result(
                self.model_path,
                self._device,
                self._device_problem,
                self._disk_start,
                self._disk_readings(),
            ),
        }
        self.record = {
            "arm": self.arm,
            "round": self.round,
            "run": self.run,
            "label": self.label,
            "required": self.required,
            "outcome": arm_outcome(checks, self.required),
            "checks": checks,
        }
        return self.record

    def __enter__(self) -> "ArmWatch":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
