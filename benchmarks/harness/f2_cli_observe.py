"""Observe the shipped CLI without replacing its trainer or training step.

For train, attach the same collector bench train uses. For bench, retain its
own collector and expose its raw evidence. No model, loss, optimizer or batch
is substituted. This callback synchronises step boundaries on CUDA; both
commands therefore have observation overhead, recorded in the F2 protocol.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from dataclasses import asdict
from pathlib import Path


def observe(output: Path, arguments: list[str]) -> None:
    import torch

    import soup_cli
    import soup_cli.bench.collector as collectors
    from soup_cli.trainer.sft import SFTTrainerWrapper

    collected = []
    original_collector = collectors.BenchCollector
    original_setup = SFTTrainerWrapper.setup
    is_train = arguments[0] == "train"
    complete = False
    error = None
    record = {"soup_cli_file": soup_cli.__file__, "physical_vram_bytes": None}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        record.update(physical_vram_bytes=props.total_memory, gpu=props.name)

    class ObservedCollector(original_collector):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.counted_fingerprint_first = None
            collected.append(self)

        def on_train_begin(self, *args, **kwargs):
            result = super().on_train_begin(*args, **kwargs)
            if self.warmup_steps == 0:
                self.counted_fingerprint_first = self._fingerprint_first
            return result

        def on_step_end(self, *args, **kwargs):
            if self.warmup_steps and len(self.steps) + 1 == self.warmup_steps:
                # Attribute this extra fingerprint to the last discarded warmup
                # step: the native collector ends that step AFTER the copy/hash.
                self._sync()
                self.counted_fingerprint_first = collectors.fingerprint_trainable(kwargs["model"])
            return super().on_step_end(*args, **kwargs)

    collectors.BenchCollector = ObservedCollector

    def setup(wrapper, *args, **kwargs):
        result = original_setup(wrapper, *args, **kwargs)
        if torch.cuda.is_available():
            record.update(
                setup_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                setup_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
        if is_train:
            collector = ObservedCollector(warmup_steps=4)
            if torch.cuda.is_available():
                collector._sync = torch.cuda.synchronize
                torch.cuda.reset_peak_memory_stats()
            trainer = wrapper.trainer
            trainer.add_callback(collector)
            original_step = trainer.training_step

            def training_step(model, inputs, *step_args, **step_kwargs):
                collector.observe_batch(inputs)
                return original_step(model, inputs, *step_args, **step_kwargs)

            trainer.training_step = training_step
        return result

    SFTTrainerWrapper.setup = setup
    original_argv = sys.argv
    try:
        sys.argv = ["soup", *arguments]
        runpy.run_module("soup_cli.cli", run_name="__main__")
        complete = True
    except SystemExit as exc:
        complete = exc.code in (None, 0)
        error = None if complete else f"SystemExit({exc.code})"
        raise
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        sys.argv = original_argv
        SFTTrainerWrapper.setup = original_setup
        collectors.BenchCollector = original_collector
        record.update(complete=complete, error=error, steps=[])
        if collected:
            collector = collected[-1]
            record.update(asdict(collector.param_snapshot()))
            record.update(
                steps=[asdict(step) for step in collector.steps],
                fingerprint_counted_first=collector.counted_fingerprint_first,
                counted_window_started=collector.counted_window_started,
                counted_window_ended=collector.counted_window_ended,
            )
        if torch.cuda.is_available():
            record.update(
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
        output.write_text(json.dumps(record, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if not arguments or arguments[0] not in ("train", "bench"):
        parser.error("expected -- train ... or -- bench train ...")
    observe(args.audit, arguments)


if __name__ == "__main__":
    main()
