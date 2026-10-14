#!/usr/bin/env python3
"""Profile harness for G7, the fixed cost of a streamed layer visit on short steps.

This file holds the fixture, the bare arm, the timer arm and the profile arm. Three
subcommands, run on the RTX 3050 laptop on mains after a reboot::

    python benchmarks/harness/visit_profile.py fixture --root /var/tmp/g7
    python benchmarks/harness/visit_profile.py arm --root /var/tmp/g7 --seq 16 \\
        --out-dir /var/tmp/g7/out [--arm timer|profile]
    python benchmarks/harness/visit_profile.py run ...    # the worker ``arm`` starts

``fixture`` writes the synthetic ``llama-8b`` checkpoint with the GPU hidden (the
writer runs out of GPU memory on a 4 GB card when it draws the embedding) and puts it
on a disk under ``--root``. The step probe accepts a shard folder only under the home
directory, the working directory or the temp directory, and the system temp directory
here is RAM-backed. So ``arm`` starts its worker with ``TMPDIR`` set to ``--root``.

``arm`` runs one arm in a fresh process: ``bare`` (the default) has no instruments,
``timer`` installs ``visit_timer``'s patches inside the process, and ``profile`` installs
``visit_ranges``' named ranges and runs the PyTorch profiler (CPU and CUDA, no stack) over its
steps. The profile arm profiles ``PROFILE_STEPS`` steps by default, because eight steps of
a 32-layer model make a trace of hundreds of megabytes. ``G7ArmWatch`` watches it
with the label "RTX 3050, synthetic", the shard folder as the model path and five required
checks. The worker writes ``<arm>-<seq>.run.json`` with the step times and the fused and old
dequantization path counts, and the timer arm adds its per-visit, per-backward and per-step
times and whether LoRA gradient accumulation falls inside the layer backward window. The
profile worker exports the trace, and the driver, after the arm watch has closed, cuts it with
``trace_analyser`` into the ``profile`` field of the run JSON (parts, kernel time, launches,
reconciliation) and notes, then deletes the trace unless ``--keep-trace``. The driver writes
``<arm>-<seq>.arm.json`` with the arm record. A bare and a timer arm of one round leave
separate files, so the validity row "timer is free" reads both JSONs.

Benchmark harness code, not shipped. Heavy dependencies are imported inside the worker.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
)

from arm_validity import disk_device as find_disk_device
from g7_checks import G7_REQUIRED, G7ArmWatch
from rich.console import Console

HERE = Path(__file__).resolve().parent

#: Every arm of G7 carries this label, so synthetic and real results are never mixed.
LABEL = "RTX 3050, synthetic"

#: Tokens per step. Batch is 1, so they equal the sequence length.
TOKEN_COUNTS = (16, 32, 64, 128)
BATCH = 1

#: The arms this file runs. The control arm comes with ticket 05.
ARMS = ("bare", "timer", "profile")

#: The step probe's defaults: 8 timed steps after 3 warm-up steps.
TIMED_STEPS = 8
WARMUP_STEPS = 3
#: Profiled steps of the profile arm.
PROFILE_STEPS = 3

#: The step probe's LoRA and seed defaults (``stream_probe.py``).
LORA_R = 8
LORA_TARGETS = "q_proj,k_proj,v_proj,o_proj"
ADAPTER_SEED = 3
INPUT_SEED = 17

SHAPE = "llama-8b"
WEIGHTS_DIR = "llama-8b-synth"
SHARDS_DIR = "shards"

console = Console()

#: Starts a command in a fresh process with an environment and waits for it.
Runner = Callable[[Sequence[str], Mapping[str, str]], None]


class Watched(Protocol):
    """An arm watch once its arm has run: ``record`` holds the arm record."""

    record: Optional[Dict[str, object]]


def run_process(command: Sequence[str], env: Mapping[str, str]) -> None:
    subprocess.run(list(command), env=dict(env), check=True)


@dataclass(frozen=True)
class Fixture:
    """Where the synthetic checkpoint and its shards live."""

    weights: Path
    shards: Path


def fixture_paths(root: Path) -> Fixture:
    return Fixture(weights=root / WEIGHTS_DIR, shards=root / SHARDS_DIR)


def build_fixture(
    root: Path,
    *,
    runner: Runner = run_process,
    disk_device: Callable[[str], Optional[str]] = find_disk_device,
) -> Fixture:
    """Write the synthetic ``llama-8b`` checkpoint under ``root`` with the GPU hidden.

    Raises ValueError if ``root`` is on no block device (a RAM-backed file system),
    where 16 GB of shards cannot live and the foreign reads check could not find the disk.
    """
    root.mkdir(parents=True, exist_ok=True)
    if disk_device(str(root)) is None:
        raise ValueError(f"{root} is on no block device; use a directory on the NVMe disk")
    fixture = fixture_paths(root)
    command = [
        sys.executable, str(HERE / "synth_checkpoint.py"),
        "--shape", SHAPE,
        "--out", str(fixture.weights),
    ]  # fmt: skip
    runner(command, {**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    return fixture


# --- one run, in the worker process ----------------------------------------------------


def run_arm(
    step: Callable[[], float],
    count_paths: Callable[[], Dict[str, int]],
    *,
    warmup: int,
    timed: int,
) -> Dict[str, Any]:
    """Run ``warmup`` untimed steps and ``timed`` timed ones, then count the paths.

    ``step`` runs one training step and returns its seconds. ``count_paths`` runs after
    the timed steps, so counting never touches a timed step.
    """
    if timed < 1:
        raise ValueError(f"need at least one timed step, got {timed}")
    seconds = [step() for _ in range(warmup + timed)][warmup:]
    return {
        "median_step_s": statistics.median(seconds),
        "step_s": seconds,
        "timed_steps": timed,
        "warmup_steps": warmup,
        "linear_paths": count_paths(),
    }


def run_timer_arm(
    step: Callable[[], float],
    count_paths: Callable[[], Dict[str, int]],
    recorder: Any,
    check_lora: Callable[[], Dict[str, Any]],
    *,
    warmup: int,
    timed: int,
) -> Dict[str, Any]:
    """A bare run's fields plus the recorder's summary of the timed steps.

    ``check_lora`` runs one more step with the LoRA gradient hooks on and reports whether the
    accumulation fell inside the layer backward window. It runs last, so those hooks never
    touch a timed step.
    """
    result = run_arm(step, count_paths, warmup=warmup, timed=timed)
    result["timer"] = recorder.summary(warmup, timed)
    result["lora_in_window"] = check_lora()
    return result


def run_profile_arm(
    step: Callable[[], float],
    count_paths: Callable[[], Dict[str, int]],
    capture: Callable[[Callable[[], List[float]]], List[float]],
    *,
    warmup: int,
    timed: int,
    trace_file: str,
) -> Dict[str, Any]:
    """A bare run's fields for a profile arm: ``warmup`` untimed steps, then ``timed`` steps
    inside ``capture``, which runs them under the profiler and exports the trace to
    ``trace_file``. The analysis happens later, in the driver."""
    if timed < 1:
        raise ValueError(f"need at least one timed step, got {timed}")
    for _ in range(warmup):
        step()
    seconds = capture(lambda: [step() for _ in range(timed)])
    return {
        "median_step_s": statistics.median(seconds),
        "step_s": seconds,
        "timed_steps": timed,
        "warmup_steps": warmup,
        "linear_paths": count_paths(),
        "trace_file": trace_file,
    }


def analyse_trace_file(path: Path) -> Dict[str, Any]:
    """Cut an exported chrome trace with the trace analyser."""
    import trace_analyser

    events = json.loads(Path(path).read_text(encoding="utf-8"))["traceEvents"]
    return trace_analyser.analyse_trace(events)


#: Found on 2026-10-14 with a Python-stack trace of one step (``clone``, ``contiguous`` and
#: ``copy_`` calls caught with their callers), on the streamed ``llama-8b`` fixture, peft 0.21.2.
COPY_SOURCE_NOTE = (
    "the 4 device-to-device copies per visit come from `result = result.clone()` in peft's "
    "LoRA Linear4bit forward (peft/tuners/lora/bnb.py line 533, peft 0.21.2), once each for "
    "q_proj, k_proj, v_proj and o_proj. They run in the call into the layer, inside functional_call, not in its own machinery."
)


def profile_notes(analysis: Mapping[str, Any]) -> List[str]:
    """Notes for the run JSON, read off the analysis."""
    notes = []
    sources = analysis.get("memcpy_sources") or {}
    if sources:
        listed = "; ".join(f"{count:g} x {chain}" for chain, count in sources.items())
        notes.append(f"device-to-device copies per visit, by the ops that issued them: {listed}")
    else:
        notes.append("device-to-device copies per visit: none found in the trace")
    if any("aten::clone" in chain for chain in sources):
        notes.append(COPY_SOURCE_NOTE)
    return notes


def _count_linear_paths(model: Any, step: Callable[[], float]) -> Dict[str, int]:
    """Run one more step and count the Linear layers that took the fused path and the old one.

    The runtime asks ``_can_use_checkpoint_visible_nf4_gemm`` once per Linear call. This
    wraps it for this one step only and reads the calling module off the caller's frame.
    """
    from soup_cli.utils import layer_stream_runtime as runtime

    original = runtime._can_use_checkpoint_visible_nf4_gemm
    fused: set = set()
    old: set = set()
    calls = {"fused": 0, "old": 0}

    def counting(x: Any, quant_state: Any) -> bool:
        took_fused = original(x, quant_state)
        owner = id(sys._getframe(1).f_locals.get("self"))
        (fused if took_fused else old).add(owner)
        calls["fused" if took_fused else "old"] += 1
        return took_fused

    runtime._can_use_checkpoint_visible_nf4_gemm = counting
    try:
        step()
    finally:
        runtime._can_use_checkpoint_visible_nf4_gemm = original
    return {
        "fused_layers": len(fused),
        "old_layers": len(old),
        "fused_calls": calls["fused"],
        "old_calls": calls["old"],
    }


def worker(args: argparse.Namespace) -> int:
    import bitsandbytes as bnb
    import stream_probe
    import torch
    import transformers
    import visit_ranges
    import visit_timer

    from soup_cli.utils.layer_stream import resolve_stream_dtype
    from soup_cli.utils.layer_stream_runtime import decoder_owner

    if not torch.cuda.is_available():
        console.print("SKIP: CUDA is required")
        return 0
    device = "cuda"
    dtype = resolve_stream_dtype(device)
    build_args = argparse.Namespace(
        weights=str(args.weights),
        shards=str(args.shards),
        quant="nf4",
        tier="ram",
        buffers=2,
        read_ahead=2,
        no_pin=False,
        seed=ADAPTER_SEED,
        lora_r=LORA_R,
        lora_targets=LORA_TARGETS,
        lazy_shard_handles=False,
        control_sync_source=False,
        stripe_dir=[],
    )
    model, runtime, config, *_rest = stream_probe.build(build_args, device, dtype)
    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(trainable, lr=1e-4)
    generator = torch.Generator(device=device).manual_seed(INPUT_SEED)
    ids = torch.randint(
        0, int(config.vocab_size), (BATCH, args.seq), generator=generator, device=device
    )

    timer = args.arm == "timer"
    recorder = visit_timer.Recorder() if timer else None

    profiling = args.arm == "profile"

    def step() -> float:
        torch.cuda.synchronize()
        if recorder is None:
            started = time.perf_counter()
        else:
            recorder.begin_step()
        # The step range opens after the first synchronize and closes after the last one, so
        # the final GPU wait is the last synchronize inside it.
        with visit_ranges.step_range() if profiling else contextlib.nullcontext():
            out = model(input_ids=ids, labels=ids)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
        if recorder is None:
            return time.perf_counter() - started
        return recorder.end_step()

    def count() -> Dict[str, int]:
        return _count_linear_paths(model, step)

    if profiling:
        from torch.profiler import ProfilerActivity, profile

        out_file = Path(args.out)
        trace_file = out_file.with_name(out_file.name.replace(".run.json", "") + ".trace.json")

        def capture(body: Callable[[], List[float]]) -> List[float]:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                seconds = body()
            prof.export_chrome_trace(str(trace_file))
            return seconds

        layers = list(decoder_owner(model).layers)
        with visit_ranges.install_ranges(layers):
            result = run_profile_arm(
                step, count, capture, warmup=args.warmup, timed=args.steps,
                trace_file=str(trace_file),
            )  # fmt: skip
    elif recorder is None:
        result = run_arm(step, count, warmup=args.warmup, timed=args.steps)
    else:
        layers = list(decoder_owner(model).layers)

        def check_lora() -> Dict[str, Any]:
            with visit_timer.watch_lora_gradients(layers, recorder):
                step()
            return recorder.lora_window_report()

        with visit_timer.install_patches(layers, visit_timer.timer_wrappers(recorder)):
            result = run_timer_arm(
                step, count, recorder, check_lora, warmup=args.warmup, timed=args.steps
            )
    result.update(
        {
            "arm": args.arm,
            "label": LABEL,
            "tokens_per_step": BATCH * args.seq,
            "batch": BATCH,
            "seq": args.seq,
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "bitsandbytes": bnb.__version__,
            "transformers": transformers.__version__,
            "weights": str(args.weights),
            "shards": str(args.shards),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    runtime.close()
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    console.print(f"wrote {args.out}")
    return 0


# --- one bare arm, in the driver -------------------------------------------------------


def run_arm_process(
    *,
    arm: str,
    seq: int,
    fixture: Fixture,
    root: Path,
    out_dir: Path,
    round: int = 1,
    run: int = 1,
    steps: Optional[int] = None,
    warmup: int = WARMUP_STEPS,
    runner: Runner = run_process,
    watch: Callable[..., ContextManager[Watched]] = G7ArmWatch,
    analyse: Callable[[Path], Dict[str, Any]] = analyse_trace_file,
    keep_trace: bool = False,
) -> Dict[str, Any]:
    """Run one ``arm`` at ``seq`` tokens in a fresh process under the arm watch.

    The worker's ``TMPDIR`` is ``root``, a directory on the NVMe disk. Writes
    ``<arm>-<seq>.run.json`` (the worker's) and ``<arm>-<seq>.arm.json`` (the arm record)
    into ``out_dir`` and returns both. ``steps`` defaults to ``PROFILE_STEPS`` for the profile
    arm and ``TIMED_STEPS`` for the others. For the profile arm, ``analyse`` cuts the trace
    once the watch has closed, and the run JSON gets the ``profile`` field and ``notes``.
    """
    if arm not in ARMS:
        raise ValueError(f"the arm is one of {', '.join(ARMS)}; got {arm}")
    if seq not in TOKEN_COUNTS:
        raise ValueError(f"the sequence length is 16, 32, 64 or 128; got {seq}")
    if steps is None:
        steps = PROFILE_STEPS if arm == "profile" else TIMED_STEPS
    out_dir.mkdir(parents=True, exist_ok=True)
    # The foreign reads check finds the disk from this folder, so it must exist at the start.
    fixture.shards.mkdir(parents=True, exist_ok=True)
    stem = f"{arm}-{seq}"
    run_file = out_dir / f"{stem}.run.json"
    command = [
        sys.executable, str(HERE / "visit_profile.py"), "run",
        "--arm", arm,
        "--weights", str(fixture.weights),
        "--shards", str(fixture.shards),
        "--seq", str(seq),
        "--batch", str(BATCH),
        "--steps", str(steps),
        "--warmup", str(warmup),
        "--out", str(run_file),
    ]  # fmt: skip
    with watch(
        arm=stem,
        round=round,
        run=run,
        label=LABEL,
        model_path=str(fixture.shards),
        required=list(G7_REQUIRED),
    ) as watched:
        runner(command, {**os.environ, "TMPDIR": str(root)})
    record = watched.record
    assert record is not None
    arm_file = out_dir / f"{stem}.arm.json"
    arm_file.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    result: Dict[str, Any] = json.loads(run_file.read_text(encoding="utf-8"))
    if arm == "profile":
        trace_file = Path(result["trace_file"])
        result["profile"] = analyse(trace_file)
        result["notes"] = profile_notes(result["profile"])
        if not keep_trace:
            trace_file.unlink()
            del result["trace_file"]
        run_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return {"run": result, "arm": record}


def run_bare_arm(**kwargs: Any) -> Dict[str, Any]:
    """``run_arm_process`` for the bare arm, with no instruments."""
    return run_arm_process(arm="bare", **kwargs)


def run_profile_process(**kwargs: Any) -> Dict[str, Any]:
    """``run_arm_process`` for the profile arm: named ranges and the PyTorch profiler."""
    return run_arm_process(arm="profile", **kwargs)


def run_timer_process(**kwargs: Any) -> Dict[str, Any]:
    """``run_arm_process`` for the timer arm: the nanosecond clock, no profiler."""
    return run_arm_process(arm="timer", **kwargs)


# --- command line ----------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    fixture = sub.add_parser("fixture", help="write the synthetic llama-8b checkpoint")
    fixture.add_argument("--root", type=Path, required=True, help="a directory on the NVMe disk")

    arm = sub.add_parser("arm", help="run one bare arm under the arm watch")
    arm.add_argument("--root", type=Path, required=True, help="the fixture root")
    arm.add_argument("--seq", type=int, required=True, choices=TOKEN_COUNTS)
    arm.add_argument("--out-dir", type=Path, required=True, help="where the two JSON files go")
    arm.add_argument("--arm", choices=ARMS, default="bare")
    arm.add_argument("--round", type=int, default=1)
    arm.add_argument("--run", type=int, default=1)
    arm.add_argument("--steps", type=int, default=None, help="timed steps (profile arm: fewer)")
    arm.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    arm.add_argument("--keep-trace", action="store_true", help="keep the profile arm's trace")

    work = sub.add_parser("run", help="the worker: one run in this process")
    work.add_argument("--arm", choices=ARMS, default="bare")
    work.add_argument("--weights", type=Path, required=True)
    work.add_argument("--shards", type=Path, required=True)
    work.add_argument("--seq", type=int, required=True, choices=TOKEN_COUNTS)
    work.add_argument("--batch", type=int, default=BATCH, choices=(BATCH,))
    work.add_argument("--steps", type=int, default=TIMED_STEPS)
    work.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    work.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "fixture":
        fixture = build_fixture(args.root.resolve())
        console.print(f"fixture written to {fixture.weights}")
        return 0
    if args.command == "run":
        return worker(args)
    root = args.root.resolve()
    result = run_arm_process(
        arm=args.arm,
        seq=args.seq,
        fixture=fixture_paths(root),
        root=root,
        out_dir=args.out_dir,
        round=args.round,
        run=args.run,
        steps=args.steps,
        warmup=args.warmup,
        keep_trace=args.keep_trace,
    )
    console.print(
        f"{args.arm} arm at {args.seq} tokens: median step {result['run']['median_step_s']:.4f} s, "
        f"outcome {result['arm']['outcome']}"
    )
    for name, check in result["arm"]["checks"].items():
        console.print(f"  {name}: {check['outcome']} ({check['reason']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
