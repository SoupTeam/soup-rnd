"""Real F1 controls for a read-verified overlay mapping; not D3 timing.

The foreign mode must be launched by an INDEPENDENT process, not a child of
the watched process tree. A sibling SSH launch is suitable. Both modes retain
their PIDs/parents, bytes and markers; bounded waits prevent orphan readers.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path


def verified_selector(mapping: dict):
    if mapping.get("status") != "mapping_observed_controls_still_required":
        raise ValueError("read-verified backing mapping required")
    files = {row["path"]: row for row in mapping["files"]}

    def select(path: str) -> str:
        canonical = os.path.realpath(path)
        row = files[canonical]
        stat = os.stat(canonical)
        if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != (
            row["st_dev"], row["inode"], row["size"], row["mtime_ns"],
        ):
            raise ValueError("mapped shard identity changed; discovery must be repeated")
        sys_path = Path("/sys/dev/block") / row["device"]
        if os.path.basename(os.path.realpath(sys_path)) != row["disk_name"]:
            raise ValueError("exposed backing device changed")
        return row["disk_name"]

    return select


def wait_marker(path: Path, timeout: float = 30) -> None:
    start = time.monotonic()
    while not path.is_file():
        if time.monotonic() - start > timeout:
            raise TimeoutError(f"control marker not received: {path.name}")
        time.sleep(0.05)


def foreign(session: Path) -> None:
    from d3_f1_mapping_probe import direct_read

    wait_marker(session / "source.json")
    config = json.loads((session / "source.json").read_text())
    state = {"pid": os.getpid(), "ppid": os.getppid(), "direct_bytes": 0}
    (session / "foreign-ready.json").write_text(json.dumps(state) + "\n")
    try:
        wait_marker(session / "foreign-go")
        state["direct_bytes"] = direct_read(Path(config["path"]), rounds=1024)
        (session / "foreign-done.json").write_text(json.dumps(state) + "\n")
        wait_marker(session / "foreign-release")
    except BaseException:
        state["traceback"] = traceback.format_exc()
        (session / "foreign-error.json").write_text(json.dumps(state, indent=2) + "\n")
        raise
    finally:
        (session / "foreign-final.json").write_text(json.dumps(state, indent=2) + "\n")


def watch(mapping_path: Path, session: Path) -> dict:
    from arm_validity import ArmWatch, cgroup_disk_reads, cgroup_other_disk_reads
    from d3_f1_mapping_probe import direct_read

    mapping = json.loads(mapping_path.read_text())
    selector = verified_selector(mapping)
    source = mapping["files"][0]["path"]
    for row in mapping["files"]:
        selector(row["path"])
    session.mkdir(parents=True, exist_ok=False)
    (session / "source.json").write_text(json.dumps({"path": source}) + "\n")
    result = {"status": "controls_failed", "kind": "F1 instrument controls, not D3 results",
              "scope": ("current-container cgroup-v2 io.stat counters; host-wide I/O and "
                        "physical-host isolation are not certified"),
              "pid": os.getpid(), "ppid": os.getppid(), "controls": [], "mapping": mapping}
    try:
        for index, (case, expected) in enumerate((("baseline", "ok"), ("own", "ok"),
                                                 ("foreign", "void"))):
            if case == "foreign":
                wait_marker(session / "foreign-ready.json")
            with ArmWatch(arm=case, round=0, run=index, model_path=source,
                          disk_device=selector, disk_reads=cgroup_disk_reads,
                          other_disk_reads=cgroup_other_disk_reads) as monitor:
                if case == "own":
                    count = direct_read(Path(source), rounds=1024)
                elif case == "foreign":
                    (session / "foreign-go").write_text("go\n")
                    wait_marker(session / "foreign-done.json")
                    count = json.loads((session / "foreign-done.json").read_text())["direct_bytes"]
                else:
                    count = 0
            row = {"case": case, "expected": expected,
                   "observed": monitor.record["outcome"], "direct_bytes": count,
                   "record": monitor.record}
            result["controls"].append(row)
            (session / "progress.json").write_text(json.dumps(result, indent=2) + "\n")
            if row["observed"] != expected:
                raise RuntimeError(f"control {case} did not produce {expected}")
        result["status"] = "controls_passed"
    except BaseException:
        result["traceback"] = traceback.format_exc()
    finally:
        (session / "foreign-release").write_text("release\n")
        (session / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("watch", "foreign"))
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--session", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "foreign":
        foreign(args.session)
        return 0
    if args.mapping is None:
        parser.error("watch requires --mapping")
    result = watch(args.mapping, args.session)
    Console().print(f"{result['status']}: {args.session}")
    return 0 if result["status"] == "controls_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
