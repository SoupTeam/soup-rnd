#!/usr/bin/env python3
"""A5 ticket 09: the ticket-04 arm with the capture of the layer call measured per step.

Runs ``l2l_ticket04_arm.main`` after two additions that change no behaviour (``--bucket``
aside), so that the same measurement reads the tree before and the tree after the change:

* ``capture_layer_call`` is wrapped. Each call records its shape, wall time (the CUDA stream
  synchronised before and after) and the layer loads the runtime counted during it.
* ``run_window`` is wrapped. Each call is one optimizer step: it records the shapes of the
  window, how many of them the process had not seen, the capture events inside it and the
  layer loads of the whole call.

With ``--bucket N`` each micro-batch of a window is padded on the right to a multiple of N
(ids with 0, labels with -100), the padding ticket 07 removed. That is option (b).

The events are merged into the arm's record as ``steps[i]["l2l_window"]`` and the run
record gets the per-step means. Benchmark harness code, not shipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))


def pad_to_multiple(length: int, bucket: int) -> int:
    return length if bucket <= 1 else -(-length // bucket) * bucket


def window_summary(
    shapes: Sequence[Sequence[int]], seen: set, events: Sequence[Dict[str, Any]], loads: int
) -> Dict[str, Any]:
    """One step's record. ``seen`` (shapes met so far) is updated with this window's."""
    keys = [tuple(int(n) for n in shape) for shape in shapes]
    new = {key for key in keys if key not in seen}
    seen.update(keys)
    return {
        "shapes": [list(key) for key in keys],
        "new_shapes": len(new),
        "captures": len(events),
        "capture_layer_loads": sum(e["layer_loads"] for e in events),
        "capture_seconds": sum(e["seconds"] for e in events),
        "layer_loads": loads,
    }


def merge_windows(record: Dict[str, Any], windows: Sequence[Dict[str, Any]]) -> None:
    """Attach window ``i`` to step ``i`` and the per-step means of the warm steps to the run."""
    steps = record["steps"]
    for step, window in zip(steps, windows):
        step["l2l_window"] = window
    warm = [w for w in windows[1:len(steps)]]
    if record.get("run") and warm:
        n = len(warm)
        record["run"].update(
            new_shapes_per_step=sum(w["new_shapes"] for w in warm) / n,
            captures_per_step=sum(w["captures"] for w in warm) / n,
            capture_layer_loads_per_step=sum(w["capture_layer_loads"] for w in warm) / n,
            capture_seconds_per_step=sum(w["capture_seconds"] for w in warm) / n,
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--bucket", type=int, default=0)
    own, rest = parser.parse_known_args(argv)
    out = rest[rest.index("--out") + 1]

    import torch

    import soup_cli.trainer.l2l_trainer as trainer_module
    import soup_cli.utils.l2l_step as step_module

    cuda = torch.cuda.is_available()

    def sync() -> None:
        if cuda:
            torch.cuda.synchronize()

    box: Dict[str, Any] = {"events": [], "runtime": None}
    seen: set = set()
    windows: List[Dict[str, Any]] = []

    real_capture = step_module.capture_layer_call

    def capture(model: Any, ids: Any, **kwargs: Any) -> Any:
        runtime = box["runtime"]
        before = runtime.stats()["layer_loads"]
        sync()
        start = time.perf_counter()
        call = real_capture(model, ids, **kwargs)
        sync()
        box["events"].append({
            "shape": [int(n) for n in ids.shape],
            "seconds": time.perf_counter() - start,
            "layer_loads": int(runtime.stats()["layer_loads"] - before),
        })
        return call

    step_module.capture_layer_call = capture

    real_run = trainer_module.run_window

    def run_window(state: Any, model: Any, ids_list: Any, *args: Any, **kwargs: Any) -> Any:
        box["runtime"] = state.runtime
        box["events"] = []
        before = state.runtime.stats()["layer_loads"]
        result = real_run(state, model, ids_list, *args, **kwargs)
        windows.append(window_summary(
            [tuple(ids.shape) for ids in ids_list], seen, box["events"],
            int(state.runtime.stats()["layer_loads"] - before),
        ))
        return result

    trainer_module.run_window = run_window

    if own.bucket > 1:
        real_tensors = trainer_module._window_tensors

        def padded(window: Any, device: Any, **kwargs: Any) -> Any:
            ids_list, label_list = real_tensors(window, device, **kwargs)
            ids_out, label_out = [], []
            for ids, labels in zip(ids_list, label_list):
                extra = pad_to_multiple(int(ids.shape[1]), own.bucket) - int(ids.shape[1])
                ids_out.append(torch.nn.functional.pad(ids, (0, extra), value=0))
                label_out.append(torch.nn.functional.pad(labels, (0, extra), value=-100))
            return ids_out, label_out

        trainer_module._window_tensors = padded

    import l2l_ticket04_arm

    status = l2l_ticket04_arm.main(rest)
    path = Path(out)
    record = json.loads(path.read_text())
    record["bucket"] = own.bucket
    merge_windows(record, windows)
    path.write_text(json.dumps(record, indent=1, default=str))
    return status


if __name__ == "__main__":
    sys.exit(main())
