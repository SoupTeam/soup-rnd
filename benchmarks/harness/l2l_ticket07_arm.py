#!/usr/bin/env python3
"""A5 ticket 07: one measured arm, in a fresh process, of the mixed-length L2L window.

Runs the trainer's L2L step (no optimizer) over the windows of the ticket-03 rows
(lengths 256, 192, 128, 64, 230, 153, 256; k=4, so windows of 4 and 3), one warm-up pass
and ``--passes`` timed passes, each window through ``l2l_measure.measure_step``.

The arm is chosen by the source tree on ``PYTHONPATH``, not by a flag:

* a tree that has ``run_window`` in ``soup_cli.trainer.l2l_trainer`` runs rows at their own
  length (after the ticket-07 change);
* an older tree runs the padded path it shipped (``_window_tensors`` pads to a multiple of
  64, one ``LayerCall`` and one store shape per window). That path is reproduced here from
  the code the harness used before ticket 07.

Benchmark harness code, not shipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _old_window(state: Dict[str, Any], model: Any, runtime: Any, window: Any, dtype: Any,
                device: str) -> Any:

    from l2l_fwdbwd import n_items

    from soup_cli.trainer.l2l_trainer import _window_tensors
    from soup_cli.utils.l2l_activations import ActivationStore
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    ids_list, label_list = _window_tensors(
        [{"input_ids": ids, "labels": labels} for ids, labels in window], device
    )
    shape = tuple(int(dim) for dim in ids_list[0].shape)
    step = state["steps"].get(shape)
    if step is None:
        step = L2LStep(model, runtime, capture_layer_call(model, ids_list[0]))
        state["steps"][shape] = step
    hidden = int(step.causal.config.hidden_size)
    k = len(window)
    store = state.get("store")
    if store is None or store.shape[:2] != shape or store.k < k:
        if store is not None:
            store.close()
        store = state["store"] = ActivationStore(
            n_layers=len(step.layers), k=k, chunk_shape=(shape[0], shape[1], hidden),
            dtype=dtype, device=device, pin=str(device).startswith("cuda"),
        )
    step.run(ids_list, store, n_items=n_items(window), labels=label_list)


def _new_window(state: Dict[str, Any], model: Any, runtime: Any, window: Any, dtype: Any,
                device: str) -> Any:
    from l2l_fwdbwd import n_items

    from soup_cli.trainer.l2l_trainer import _window_tensors, run_window

    ids_list, label_list = _window_tensors(
        [{"input_ids": ids, "labels": labels} for ids, labels in window], device
    )
    run_window(state["l2l"], model, ids_list, label_list, n_items(window), device)


def _pinned_bytes(state: Dict[str, Any], dtype: Any) -> int:
    """Logical bytes of the activation store (pinned blocks round up to powers of two)."""
    import math

    import torch

    store = state["l2l"].store if "l2l" in state else state.get("store")
    if store is None:
        return 0
    itemsize = torch.empty((), dtype=dtype).element_size()
    if hasattr(store, "mb_shapes"):
        return store.n_layers * sum(math.prod(shape) for shape in store.mb_shapes) * itemsize
    return store.stats.resident_chunks * store.chunk_bytes


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--data-seed", type=int, default=5)
    parser.add_argument("--weights", default="mistralai/Mistral-7B-v0.1")
    parsed = parser.parse_args(argv)

    import l2l_measure
    import l2l_mistral_stand as stand
    import l2l_probe
    import stream_probe
    import torch
    from l2l_fwdbwd import _autocast, cut_windows, make_rows

    import soup_cli.trainer.l2l_trainer as trainer_module
    from soup_cli.utils.layer_stream import resolve_stream_dtype

    device = "cuda"
    dtype_name = resolve_stream_dtype(device)
    dtype = getattr(torch, dtype_name)
    args = l2l_probe.parse_args([
        "--mode", "correctness", "--weights", parsed.weights, "--quant", "nf4",
        "--seq", "256", "--tier", "ram", "--out", str(Path(parsed.out).parent / "unused.json"),
    ])
    model, runtime, config, *_ = stream_probe.build(args, device, dtype_name)
    new_code = hasattr(trainer_module, "run_window")
    try:
        import l2l_schedule

        l2l_schedule.make_non_vacuous(model)
        rows = make_rows(7, 256, int(config.vocab_size), seed=parsed.data_seed)
        windows = cut_windows(rows, parsed.k)
        state: Dict[str, Any] = {"steps": {}}
        if new_code:
            state["l2l"] = trainer_module.L2LState(runtime=runtime, dtype=dtype)
        run = _new_window if new_code else _old_window
        model.train()

        def one(window: Any) -> None:
            l2l_schedule.zero_grads(model)
            with _autocast(device, dtype):
                run(state, model, runtime, window, dtype, device)

        def counters() -> Dict[str, int]:
            stats = runtime.stats()
            return {
                "layer_loads": stats["layer_loads"],
                "nf4_restores": stats["layer_restores"],
                "spill_written_bytes": 0,
                "spill_read_bytes": 0,
                "pinned_store_bytes": _pinned_bytes(state, dtype),
            }

        for window in windows:  # warm-up: layer-call capture, store allocation, kernels
            one(window)
        steps = []
        for _ in range(parsed.passes):
            for window in windows:
                mask = torch.ones(1, sum(int(ids.shape[1]) for ids, _ in window))
                steps.append(l2l_measure.measure_step(
                    lambda window=window: one(window), attention_mask=mask, device=device,
                    counters=counters, read_bytes=l2l_measure.process_read_counter,
                    write_bytes=l2l_measure.process_write_counter,
                ))
    finally:
        runtime.close()
    record = {
        "code": "new (rows at their own length)" if new_code else "old (padded to 64)",
        "data_seed": parsed.data_seed,
        "k": parsed.k,
        "lengths": [int(ids.shape[1]) for ids, _ in rows],
        "dtype": dtype_name,
        "gpu": l2l_measure.gpu_facts(),
        "run": l2l_measure.run_record(steps),
        "steps": steps,
        "stand": stand.DEFAULT_WEIGHTS,
    }
    Path(parsed.out).write_text(json.dumps(record, indent=1, default=str))
    print(f"{record['code']}: {record['run']['tokens_per_s']:.1f} tokens/s, "
          f"peak vram {record['run']['peak_vram_allocated_bytes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
