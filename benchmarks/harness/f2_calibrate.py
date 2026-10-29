"""Calibrate F1 on the F2 box without changing system or GPU settings.

For foreign mode, issue a >=1 GB direct read through a separate SSH connection
while this watch is active; it must not be a child of the watching process.
Record a direct-read diskstats probe before using --physical-disk on overlayfs.
"""

import argparse
import os
import re
import time
from pathlib import Path

from arm_validity import ArmWatch
from f2_reference import write_json
from rich.console import Console


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("clean", "foreign"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--physical-disk")
    parser.add_argument("--seconds", type=float, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.seconds < 2:
        raise ValueError("at least two seconds required")
    kwargs = {}
    if args.physical_disk:
        if not re.fullmatch(r"nvme\d+n\d+", args.physical_disk):
            raise ValueError("expected a recorded whole NVMe device name")
        kwargs["disk_device"] = lambda path: args.physical_disk
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    with ArmWatch(
        arm=args.mode,
        round=0,
        run=0,
        label="F2 sensor calibration",
        model_path=args.model_path,
        **kwargs,
    ) as watch:
        Console().print("F2_CALIBRATION_ACTIVE")
        time.sleep(args.seconds)
    record = watch.record
    record["physical_disk_mapping"] = args.physical_disk
    write_json(args.output, record)
    actual = (
        record["outcome"] if args.mode == "clean" else record["checks"]["foreign_reads"]["outcome"]
    )
    expected = "ok" if args.mode == "clean" else "void"
    Console().print(f"{args.mode}: {actual}; expected {expected}")
    if actual != expected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
