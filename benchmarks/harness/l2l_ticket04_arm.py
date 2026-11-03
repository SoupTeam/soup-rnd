#!/usr/bin/env python3
"""A5 ticket 04: one run of ``soup train`` in a fresh process, measured per optimizer step.

The process calls the ``soup train`` command of ``soup_cli.cli`` in-process, so the run goes
through the shipped CLI, trainer and streaming code. Three hooks watch it and change nothing:

* ``build_streamed_model`` is wrapped to keep the streaming runtime, which holds the layer
  load and NF4 restore counters;
* ``Trainer.get_batch_samples`` is wrapped to count the tokens of each window that are not
  padding (attention mask);
* a ``TrainerCallback`` is added to every ``Trainer``. It records, around each optimizer
  step, the time, the VRAM and RSS peaks of that step, the bytes the process read and wrote,
  the counters, and each logged loss.

The numbers before the first step (reading and quantising the weights) are kept apart as
``load``. Step 0 is kept but left out of the run record, because it pays the layer-call
capture and kernel set-up once.

``--sdpa`` pins one attention backend for the whole run, and the determinism switches of the
A3 stand are set, so that two arms differ in nothing but the config. Benchmark harness
code, not shipped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: A3's tolerance: bf16 loss error at most this many times the noise floor, step by step.
FLOOR_FACTOR = 1.5


# ==========================================================================
# pure parts (tested on the CPU)
# ==========================================================================
def relative_errors(ref: Sequence[Optional[float]], cand: Sequence[Optional[float]]) -> List[float]:
    """Per logged loss: ``|cand - ref| / |ref|``. A missing value or a zero reference is inf."""
    out: List[float] = []
    for left, right in zip(ref, cand):
        if left is None or right is None:
            out.append(math.inf)
        elif left == right:
            out.append(0.0)
        else:
            out.append(abs(right - left) / abs(left) if left else math.inf)
    return out


def judge_loss_curve(
    ref: Sequence[Optional[float]],
    cand: Sequence[Optional[float]],
    floor: Optional[Sequence[Optional[float]]],
    *,
    factor: float = FLOOR_FACTOR,
) -> Dict[str, Any]:
    """The A3-07 rule on a loss curve: at every step the error of ``cand`` against ``ref`` is
    at most ``factor`` x the error of ``floor`` (the reference on a second SDPA backend)
    against ``ref``. A curve of another length, or a floor with an infinite error, gives no
    verdict (an infinite floor error would make a limit every candidate passes).
    """
    out: Dict[str, Any] = {
        "factor": factor, "steps": len(ref), "judged": False, "pass": False,
        "error": [], "floor_error": [], "limit": [], "failing": [], "problem": None,
    }
    if floor is None:
        out["problem"] = "no noise-floor run"
        return out
    if not (len(ref) == len(cand) == len(floor)) or not ref:
        out["problem"] = (
            f"curve lengths differ: ref {len(ref)}, cand {len(cand)}, floor {len(floor)}"
        )
        return out
    floor_error = relative_errors(ref, floor)
    if any(math.isinf(value) or math.isnan(value) for value in floor_error):
        out["problem"] = "the floor has a missing or infinite error"
        return out
    error = relative_errors(ref, cand)
    limit = [factor * value for value in floor_error]
    out.update(judged=True, error=error, floor_error=floor_error, limit=limit)
    for step, (err, lim) in enumerate(zip(error, limit)):
        if not err <= lim:
            out["failing"].append({"step": step, "error": err, "limit": lim})
    out["pass"] = not out["failing"]
    return out


class StepMeter:
    """Records one measurement per optimizer step; the hooks of ``main`` call it.

    ``sources`` is a mapping of callables so tests can supply fakes. ``counters()`` returns
    running totals ``layer_loads``, ``nf4_restores``, ``large_loads`` and the level
    ``pinned_store_bytes``. ``io()`` returns the process's ``read_bytes`` and ``write_bytes``.
    ``peaks()`` returns the step's VRAM peaks (allocated, reserved) and ``reset_peaks()``
    starts a new window for them. ``rss()`` returns the step's RSS peak, ``reset_rss()``
    returns False where it cannot be reset. ``clock()`` and ``tokens()`` complete the set.
    """

    def __init__(self, sources: Mapping[str, Any]):
        self.s = sources
        self.steps: List[Dict[str, Any]] = []
        self.losses: List[Dict[str, Any]] = []
        self.load: Optional[Dict[str, Any]] = None
        self._begin: Optional[Dict[str, Any]] = None
        self._rss_reset = False

    def train_begin(self) -> None:
        io = self.s["io"]()
        self.load = {
            "read_bytes_since_process_start": io["read_bytes"],
            "write_bytes_since_process_start": io["write_bytes"],
            "peak_vram_allocated_bytes": self.s["peaks"]()[0],
            "peak_rss_bytes": self.s["rss"](),
            "seconds_since_process_start": self.s["clock"]() - self.s["t0"],
            "counters": dict(self.s["counters"]()),
        }

    def step_begin(self) -> None:
        self.s["reset_peaks"]()
        self._rss_reset = self.s["reset_rss"]()
        self._begin = {
            "t": self.s["clock"](),
            "io": dict(self.s["io"]()),
            "counters": dict(self.s["counters"]()),
        }

    def step_end(self) -> Dict[str, Any]:
        if self._begin is None:
            raise RuntimeError("step_end without step_begin")
        begin, self._begin = self._begin, None
        seconds = self.s["clock"]() - begin["t"]
        io, counters = self.s["io"](), dict(self.s["counters"]())
        allocated, reserved = self.s["peaks"]()
        reasons: Dict[str, str] = {}
        rss = self.s["rss"]() if self._rss_reset else None
        if not self._rss_reset:
            reasons["peak_rss_bytes"] = "peak RSS could not be reset, so a step peak is unknown"

        def delta(name: str, key: str) -> Optional[int]:
            before, after = begin["io"].get(key), io.get(key)
            if before is None or after is None or after < before:
                reasons[name] = "the process I/O counter is missing or went backwards"
                return None
            return after - before

        tokens = int(self.s["tokens"]())
        record = {
            "step": len(self.steps),
            "step_s": seconds,
            "real_tokens": tokens,
            "tokens_per_s": tokens / seconds if seconds > 0 and tokens > 0 else None,
            "peak_vram_allocated_bytes": allocated,
            "peak_vram_reserved_bytes": reserved,
            "peak_rss_bytes": rss,
            "pinned_store_bytes": counters.get("pinned_store_bytes"),
            "os_read_bytes": delta("os_read_bytes", "read_bytes"),
            "os_write_bytes": delta("os_write_bytes", "write_bytes"),
            "layer_loads": counters["layer_loads"] - begin["counters"]["layer_loads"],
            "nf4_restores": counters["nf4_restores"] - begin["counters"]["nf4_restores"],
            "large_loads": counters["large_loads"] - begin["counters"]["large_loads"],
            "unavailable": reasons,
        }
        self.steps.append(record)
        return record

    def log(self, step: int, logs: Mapping[str, Any]) -> None:
        if "loss" in logs:
            self.losses.append({"step": step, "loss": float(logs["loss"]),
                                "learning_rate": logs.get("learning_rate")})

    def run_record(self) -> Dict[str, Any]:
        """The run from the steps after step 0: tokens over time, largest peaks, summed counts."""
        warm = self.steps[1:]
        if not warm:
            raise ValueError("a run record needs at least one step after step 0")
        seconds = sum(step["step_s"] for step in warm)
        tokens = sum(step["real_tokens"] for step in warm)

        def peak(name: str) -> Optional[int]:
            values = [step[name] for step in warm]
            return None if any(value is None for value in values) else max(values)

        def total(name: str) -> Optional[int]:
            values = [step[name] for step in warm]
            return None if any(value is None for value in values) else sum(values)

        return {
            "steps": len(warm),
            "step_s_mean": seconds / len(warm),
            "real_tokens": tokens,
            "tokens_per_s": tokens / seconds,
            "peak_vram_allocated_bytes": peak("peak_vram_allocated_bytes"),
            "peak_vram_reserved_bytes": peak("peak_vram_reserved_bytes"),
            "peak_rss_bytes": peak("peak_rss_bytes"),
            "pinned_store_bytes": peak("pinned_store_bytes"),
            "os_read_bytes": total("os_read_bytes"),
            "os_write_bytes": total("os_write_bytes"),
            "layer_loads": total("layer_loads"),
            "nf4_restores": total("nf4_restores"),
            "large_loads": total("large_loads"),
            "layer_loads_per_step": total("layer_loads") / len(warm),
        }


# ==========================================================================
# the process: hooks and the call into soup train
# ==========================================================================
def _pinned_bytes(trainer: Any) -> int:
    """Logical bytes of the L2L activation store (pinned blocks round up to powers of two)."""
    import torch

    state = getattr(trainer, "_soup_l2l", None)
    store = getattr(state, "store", None)
    if store is None:
        return 0
    itemsize = torch.empty((), dtype=state.dtype).element_size()
    return store.n_layers * sum(math.prod(shape) for shape in store.mb_shapes) * itemsize


def read_counters(runtime: Any, trainer: Any) -> Dict[str, int]:
    """Running totals of the streaming runtime; all zero for a resident run (no runtime)."""
    if runtime is None:
        return {"layer_loads": 0, "nf4_restores": 0, "large_loads": 0, "pinned_store_bytes": 0}
    stats = runtime.stats()
    return {
        "layer_loads": int(stats["layer_loads"]),
        "nf4_restores": int(stats["layer_restores"]),
        "large_loads": int(stats["large_loads"]),
        "pinned_store_bytes": _pinned_bytes(trainer),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="the JSON record of this run")
    parser.add_argument("--name", default=None)
    parser.add_argument("--sdpa", default="MATH", choices=["MATH", "EFFICIENT_ATTENTION"])
    parser.add_argument("--no-determinism", action="store_true")
    parsed = parser.parse_args(argv)
    t0 = time.perf_counter()

    import l2l_measure
    import l2l_mistral_stand as stand

    if not parsed.no_determinism:
        stand.set_deterministic_env()
    import host_box
    import torch
    import transformers
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if not parsed.no_determinism:
        stand.enable_deterministic()

    import soup_cli.utils.layer_stream_runtime as runtime_module

    box: Dict[str, Any] = {"runtime": None, "trainer": None, "tokens": 0, "window_tokens": 0}

    real_build = runtime_module.build_streamed_model

    def build(*args: Any, **kwargs: Any) -> Any:
        model, runtime = real_build(*args, **kwargs)
        box["runtime"] = runtime
        return model, runtime

    runtime_module.build_streamed_model = build

    real_samples = transformers.Trainer.get_batch_samples

    def samples(self: Any, epoch_iterator: Any, num_batches: int, device: Any) -> Any:
        batches, num_items = real_samples(self, epoch_iterator, num_batches, device)
        box["window_tokens"] = sum(
            l2l_measure.count_real_tokens(batch["attention_mask"])
            if "attention_mask" in batch else int(batch["input_ids"].numel())
            for batch in batches
        )
        return batches, num_items

    transformers.Trainer.get_batch_samples = samples

    def counters() -> Dict[str, int]:
        return read_counters(box["runtime"], box["trainer"])

    cuda = torch.cuda.is_available()

    def peaks() -> Any:
        if not cuda:
            return (None, None)
        torch.cuda.synchronize()
        return (int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved()))

    def reset_peaks() -> None:
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    def clock() -> float:
        if cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    meter = StepMeter({
        "counters": counters, "io": l2l_measure.process_io, "peaks": peaks,
        "reset_peaks": reset_peaks, "reset_rss": host_box.reset_peak_rss,
        "rss": lambda: host_box.process_memory()["peak_rss"], "clock": clock,
        "tokens": lambda: box["window_tokens"], "t0": t0,
    })

    class Callback(transformers.TrainerCallback):
        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            meter.train_begin()

        def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            meter.step_begin()

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            meter.step_end()

        def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kw: Any) -> None:
            if logs:
                meter.log(int(state.global_step), logs)

    real_init = transformers.Trainer.__init__

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        real_init(self, *args, **kwargs)
        box["trainer"] = self
        self.add_callback(Callback())

    transformers.Trainer.__init__ = init

    from soup_cli.cli import app

    command = ["train", "--config", parsed.config, "--yes"]
    if parsed.name:
        command += ["--name", parsed.name]
    failure: Optional[str] = None
    started = time.time()
    try:
        with sdpa_kernel(getattr(SDPBackend, parsed.sdpa)):
            app(command, standalone_mode=False)
    except SystemExit as exc:
        if exc.code not in (0, None):
            failure = f"soup train exited with status {exc.code}"
    except Exception as exc:  # the record of a failed run is kept, then the exit code says so
        failure = f"{type(exc).__name__}: {exc}"
    wall = time.time() - started

    record: Dict[str, Any] = {
        "config": parsed.config,
        "sdpa": parsed.sdpa,
        "deterministic": not parsed.no_determinism,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "failure": failure,
        "wall_s": wall,
        "gpu": l2l_measure.gpu_facts() if cuda else None,
        "load": meter.load,
        "steps": meter.steps,
        "losses": meter.losses,
        "process_peak_rss_bytes": host_box.process_memory()["peak_rss"],
        "process_peak_vram_allocated_bytes": None,
        "run": meter.run_record() if len(meter.steps) > 1 else None,
    }
    if record["load"] and meter.steps:
        vram = [step["peak_vram_allocated_bytes"] for step in meter.steps]
        if cuda and record["load"]["peak_vram_allocated_bytes"] is not None:
            record["process_peak_vram_allocated_bytes"] = max(
                [record["load"]["peak_vram_allocated_bytes"], *vram]
            )
    Path(parsed.out).write_text(json.dumps(record, indent=1, default=str))
    print(f"{parsed.config}: {len(meter.steps)} steps, failure={failure}")
    return 1 if failure else 0


if __name__ == "__main__":
    sys.exit(main())
