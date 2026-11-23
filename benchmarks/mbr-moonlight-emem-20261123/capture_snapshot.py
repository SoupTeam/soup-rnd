"""Capture allocator state for one normal-mode Moonlight C or E run.

The snapshot shows holdings after the last step, before the copy probe and runtime
cleanup. A probe OOM can release cached segments, so stop recording before probing.
"""

import argparse
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from rich.console import Console
from run_mem import (
    measurement_identity,
    measurement_parser,
    power_state,
    run_measurement,
    validate_run,
)


def _memory_stats(cuda: Any) -> dict[str, int]:
    return {
        "memory_allocated": cuda.memory_allocated(),
        "memory_reserved": cuda.memory_reserved(),
        "max_memory_allocated": cuda.max_memory_allocated(),
        "max_memory_reserved": cuda.max_memory_reserved(),
    }


def _copy_probe(torch: Any, row: dict[str, Any], before: dict[str, int]) -> None:
    free = before["memory_reserved"] - before["memory_allocated"]
    size = max(0, free) // 512 * 512
    row["copy_peak_bytes"] = size
    probe = None
    try:
        probe = torch.empty(size, device="cuda", dtype=torch.uint8)
        probe.fill_(1)
        torch.cuda.synchronize()
        row["copy_ok"] = True
    except torch.cuda.OutOfMemoryError as error:
        row.update(copy_ok=False, copy_error=str(error))
    finally:
        del probe


def _capture_cuda(
    args: argparse.Namespace, torch: Any, path: Path, row: dict[str, Any],
) -> dict[str, Any]:
    from benchmarks.harness.mbr_measure import PEAK_KEYS
    from soup_cli.utils import l2l_step, layer_stream_runtime

    cuda = torch.cuda
    memory = cuda.memory
    original_build = layer_stream_runtime.build_streamed_model
    original_run = l2l_step.L2LStep.run
    step = 0
    recording = False

    def build(*model_args: Any, **kwargs: Any) -> Any:
        nonlocal recording
        memory._record_memory_history(
            enabled="all", context="all", stacks="all", max_entries=args.max_entries,
        )
        recording = True
        return original_build(*model_args, **kwargs)

    def run(*step_args: Any, **kwargs: Any) -> Any:
        nonlocal recording, step
        result = original_run(*step_args, **kwargs)
        step += 1
        if step == args.steps:
            cuda.synchronize()
            row["pre_copy"] = _memory_stats(cuda)
            # run_one closes the expert pool and drops the model before it returns.
            memory._dump_snapshot(str(path))
            memory._record_memory_history(enabled=None)
            recording = False
            _copy_probe(torch, row, row["pre_copy"])
        return result

    try:
        with (
            patch.object(layer_stream_runtime, "build_streamed_model", build),
            patch.object(l2l_step.L2LStep, "run", run),
        ):
            measurement = run_measurement(args)
        row.update(_memory_stats(cuda))
    finally:
        if recording:
            memory._record_memory_history(enabled=None)
    return {
        "status": measurement["status"],
        "harness_peaks": {key: measurement.get(key) for key in PEAK_KEYS},
    }


def main(argv: list[str] | None = None) -> int:
    parser = measurement_parser(__doc__, steps=2)
    parser.add_argument("--max-entries", type=int, default=1_000_000)
    args = parser.parse_args(argv)
    validate_run(parser, args)
    if args.max_entries < 1:
        parser.error("--max-entries must be positive")

    import torch

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f"snapshot-{args.arm}-k{args.k}-s{args.seq}"
    identity = measurement_identity(args, torch)
    row = {
        **identity, "ac_online": power_state(), "copy_peak_bytes": None, "status": "cpu",
    }
    if identity["card"] != "CPU":
        row.update(_capture_cuda(args, torch, stem.with_suffix(".pickle"), row))
    text = json.dumps(row)
    stem.with_suffix(".json").write_text(text + "\n", encoding="utf-8")
    Console(soft_wrap=True, markup=False, highlight=False).print(text)
    return 0 if row["status"] in ("ok", "cpu") else 1


if __name__ == "__main__":
    import sys

    repo = Path(__file__).resolve().parents[2]
    sys.path[:0] = (str(repo / "src"), str(repo))
    raise SystemExit(main())
