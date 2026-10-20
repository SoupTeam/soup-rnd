#!/usr/bin/env python3
"""Attach the `soup bench train` collector to a Trainer someone else built.

F3 seam 2. The collector and `build_train_report` are the ones the native trainer uses,
so the three checks stay the same functions. The adapter adds two things a foreign
script does not do for us: it adds the collector as a callback, and it wraps the
trainer's data collator so the collector sees each batch's labels.

Benchmark harness code, not shipped. Imports `soup_cli`; `transformers` is only needed
by the trainer the caller passes in.
"""

from __future__ import annotations

from typing import Any

from soup_cli.bench import collector as _collector_module


def attach_collector(trainer: Any, *, warmup_steps: int = 0) -> Any:
    """Add a `BenchCollector` to `trainer` and return it. Call before `trainer.train()`."""
    collector = _collector_module.BenchCollector(warmup_steps=warmup_steps)
    inner = trainer.data_collator

    def observing(features):
        batch = inner(features)
        collector.observe_batch(batch)
        return batch

    trainer.data_collator = observing
    trainer.add_callback(collector)
    return collector


def build_report(
    collector: Any,
    *,
    steps_requested: int,
    warmup_steps: int = 0,
    provenance: dict | None = None,
    memory: dict | None = None,
) -> dict:
    """Hand the collector's records to `build_train_report`."""
    return collector.build_report(
        warmup_steps=warmup_steps,
        provenance=provenance or {},
        memory=memory or {},
        steps_requested=steps_requested,
    )
