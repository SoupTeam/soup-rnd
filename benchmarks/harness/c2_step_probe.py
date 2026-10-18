#!/usr/bin/env python3
"""C2 benchmark-only depth switch around the existing stream_probe step path."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

import stream_probe
from c2_linux_depth_probe import (
    DIAGNOSTIC_PROTOCOL,
    PROTOCOL,
    arm_reasons,
    counters,
    memory,
    quiet,
    snapshot,
)
from c2_source_depth import depth_source


def io_window(samples: list, start: float, end: float) -> tuple[dict, float]:
    """Enclose a timed GPU step in adjacent physical-counter samples."""
    before = [row for row in samples if row[0] <= start]
    after = [row for row in samples if row[0] >= end]
    if not before or not after:
        raise ValueError("counter samples do not enclose the timed step")
    left, right = before[-1], after[0]
    window = [row for row in samples if left[0] <= row[0] <= right[0]]
    if len(window) < 4 or any(b[0] - a[0] > 2 for a, b in zip(window, window[1:])):
        raise ValueError("missing counter samples or suspend gap")
    delta = {key: right[1][key] - left[1][key] for key in ("read", "write")}
    return delta, right[0] - left[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--c2-depth", type=int, choices=[1, 2], required=True)
    parser.add_argument("--c2-device", required=True)
    parser.add_argument("--c2-diagnostic-writes", action="store_true")
    parser.add_argument("--out", required=True)
    args, remaining = parser.parse_known_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("existing arm: preserve it and select a new sequence")
    sys.argv = [sys.argv[0], *remaining, "--out", args.out]
    standard = stream_probe.parse_args()
    required = {"quant": "nf4", "tier": "disk", "batch": 1, "seq": 256,
                "steps": 3, "warmup": 1, "read_ahead": 3, "buffers": 2,
                "seed": 3, "input_seed": 17, "lora_r": 8}
    if any(getattr(standard, key) != value for key, value in required.items()):
        raise SystemExit("configuration differs from the frozen C2 step protocol")
    if not standard.step or not standard.skip_events or standard.no_pin:
        raise SystemExit("C2 requires pinned uninstrumented step mode")
    shape = json.loads((Path(standard.weights) / "config.json").read_text())
    if any(shape.get(key) != value for key, value in {
        "hidden_size": 8192, "num_hidden_layers": 80, "vocab_size": 32768
    }.items()):
        raise SystemExit("weights differ from the frozen synthetic 70B shape")

    from soup_cli.utils import async_disk_source

    source_type = depth_source(args.c2_depth)
    original_source = async_disk_source.AsyncDiskSource
    async_disk_source.AsyncDiskSource = source_type
    original_sink = stream_probe.Sink
    sinks = []
    idle = None
    idle_checked_unix = None

    class C2Sink(original_sink):
        def __init__(self, path: str, meta: dict) -> None:
            self.samples = []
            self.sample_errors = []
            self.stop_event = threading.Event()
            self.idle = idle
            self.box_before = snapshot()
            self.samples.append((time.time(), counters()[args.c2_device]))
            self.thread = threading.Thread(target=self.observe, daemon=True)
            observed = source_type.last_admission
            if (observed is None or not observed["pinned"] or not observed["direct_io"]
                    or observed["read_ranges"] != 4 or observed["read_ahead"] != 3
                    or list(observed["workers_per_root"].values()) != [4]):
                raise RuntimeError(f"unexpected source configuration: {observed}")
            if len(self.box_before["gpu_clients"]) != 1:
                raise RuntimeError("unknown GPU workload")
            cache_bytes = sum(path.stat().st_size for path in
                              Path(meta["shard_dir"]).glob("*.safetensors"))
            if cache_bytes <= memory()["MemTotal"]:
                raise RuntimeError("NF4 cache is not larger than physical RAM")
            if self.box_before["memory"]["MemAvailable"] < 4e9:
                raise RuntimeError("less than 4 GB MemAvailable")
            meta.update({
                "c2_depth": args.c2_depth, "c2_benchmark_only": True,
                "c2_idle_checked_unix": idle_checked_unix,
                "c2_idle_phase": "before model construction and source prefetch",
                "c2_diagnostic_writes": args.c2_diagnostic_writes,
                "c2_diagnostic_protocol_sha256": (
                    hashlib.sha256(DIAGNOSTIC_PROTOCOL.read_bytes()).hexdigest()
                    if args.c2_diagnostic_writes else None
                ),
                "c2_protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                "c2_adapter_sha256": hashlib.sha256(
                    Path(__file__).with_name("c2_source_depth.py").read_bytes()
                ).hexdigest(),
                "c2_wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "c2_source_admission": observed, "c2_physical_device": args.c2_device,
                "c2_cache_bytes": cache_bytes,
                "c2_git_sha": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
            })
            super().__init__(path, meta)
            sinks.append(self)
            self.thread.start()

        def observe(self) -> None:
            try:
                while not self.stop_event.wait(0.25):
                    self.samples.append((time.time(), counters()[args.c2_device]))
            except BaseException as error:
                self.sample_errors.append(repr(error))

        def finish(self) -> None:
            self.stop_event.set()
            self.thread.join()
            self.samples.append((time.time(), counters()[args.c2_device]))
            after = snapshot()
            reasons = self.sample_errors[:]
            if (after["gpu_clients"] != self.box_before["gpu_clients"]
                    or after["memory"]["MemAvailable"] < 4e9):
                reasons.append("host memory/GPU state changed")
            checked = []
            points = [point for point in self.payload["records"]
                      if point.get("kind") == "step" and point.get("label") == "step_plain"]
            if len(points) != 1 or len(points[0]["records"]) != 3:
                reasons.append("incomplete uninstrumented step block")
            else:
                point = points[0]
                for step in point["records"]:
                    try:
                        delta, duration = io_window(
                            self.samples, step["started_unix"],
                            step["started_unix"] + step["step_s"]
                        )
                        reasons.extend(arm_reasons(
                            delta, duration, point["bytes_moved_per_step"],
                            args.c2_diagnostic_writes,
                        ))
                        checked.append({"physical_delta": delta, "seconds": duration})
                    except ValueError as error:
                        reasons.append(str(error))
                    if step["loss"] is None or not math.isfinite(step["loss"]):
                        reasons.append("nonfinite loss")
            self.payload["c2_validity"] = {
                "reasons": reasons, "idle_rates": self.idle,
                "box_before": self.box_before, "box_after": after,
                "step_physical_io": checked,
            }
            self.payload["c2_disk_samples"] = self.samples
            self.flush()

    stream_probe.Sink = C2Sink
    result = 1
    try:
        # Construction starts this process's own asynchronous prefetch. Check the
        # host before starting it, rather than misclassifying it as foreign I/O.
        idle = quiet(args.c2_diagnostic_writes)
        idle_checked_unix = time.time()
        result = stream_probe.main()
    except BaseException as error:
        out.parent.mkdir(parents=True, exist_ok=True)
        partial = json.loads(out.read_text()) if out.exists() else {"records": []}
        partial["c2_error"] = repr(error)
        out.write_text(json.dumps(partial, indent=2) + "\n")
        raise
    finally:
        for sink in sinks:
            sink.finish()
        async_disk_source.AsyncDiskSource = original_source
        stream_probe.Sink = original_sink
    if not sinks or any(sink.payload["c2_validity"]["reasons"] for sink in sinks):
        return 1
    return result


if __name__ == "__main__":
    sys.exit(main())
