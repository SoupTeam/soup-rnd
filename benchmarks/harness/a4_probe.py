#!/usr/bin/env python3
"""A4 probe: L2L activations under a host-RAM budget, spilled to a drive past it.

The rule this runs against is §3 of benchmarks/probe-a4-l2l-ram-budget-spill.md,
fixed before the GPU run, to be committed before the first GPU arm. One process per invocation:

  --mode preflight    spill volume (filesystem, drive, aligned unbuffered round
                      trip, free space), cgroup RAM limit, box stamp. Exit 3 on
                      a refusal.
  --mode correctness  GA twice, all-pinned L2L twice, then L2L at every budget
                      (1, 0.5, 0.25, min): per-micro-batch losses (the forward)
                      and LoRA grads (the backward) compared with torch.equal,
                      plus a corrupted-spill-read control expected to differ.
  --mode arm          ONE timed arm (a fresh process): build, plan the budget,
                      warm up, time the steps under an F1 ArmWatch; records step
                      times, spill write/read GB/s on the drive, the hidden share
                      of the spill I/O, the cap check, the host allocator's peak
                      and (disk tier) the weight read GB/s while busy.
  --mode drive        sequential chunk-sized unbuffered write and read GB/s on the
                      spill drive, under an ArmWatch.
  --mode cleanup      Ctrl-C mid-backward, an exception mid-forward, ENOSPC mid-step
                      (injected; plus real with --enospc-dir): the spill file is
                      gone, the spill threads are gone, and the host allocator is
                      back at its baseline after each.

``--device cpu`` runs the same code on CPU (float32, unquantised, pageable) for the
SYNTHETIC / CPU PILOT; the rule refuses its records.
"""

from __future__ import annotations

import argparse
import errno
import os
import shutil
import signal
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import a4_activation_budget as budget  # noqa: E402
import a4_box  # noqa: E402
import a4_spill_store  # noqa: E402
import arm_validity  # noqa: E402
import l2l_activations  # noqa: E402
import l2l_schedule  # noqa: E402
import stream_probe  # noqa: E402

REQUIRED_CHECKS = ("suspend", "power", "foreign_reads")
#: Staging buffers per direction. 4, not l2l_activations' 8: at Mistral-7B's 4 MiB chunk
#: a ring buffer pins 8 MiB (#1531 slack + rounding), so 8 per ring would make the
#: every-layer-spilled minimum (128 MiB) equal the k=2 half budget and erase an arm.
DEFAULT_RING = 4
#: The OS reserve of a declared host total; V3's 4 GB margin of the A0 probe.
DEFAULT_RESERVE_BYTES = 4_000_000_000
CPU_PILOT_LABEL = "SYNTHETIC/CPU PILOT"
#: The arm budgets of the rule, as fractions of the all-pinned store, or its minimum.
BUDGET_SPECS = budget.ARM_BUDGETS


# ==========================================================================
# arguments
# ==========================================================================
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", required=True,
                        choices=("preflight", "correctness", "arm", "drive", "cleanup"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--weights")
    parser.add_argument("--shards", default=None)
    parser.add_argument("--quant", choices=("none", "nf4"), default="nf4")
    parser.add_argument("--tier", choices=("ram", "disk"), default="ram")
    parser.add_argument("--buffers", type=int, default=2)
    parser.add_argument("--read-ahead", type=int, default=2)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--input-seed", type=int, default=17)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--budget", default="1", help="1, 0.5, 0.25 or min (arm mode)")
    parser.add_argument("--ring", type=int, default=DEFAULT_RING)
    parser.add_argument("--spill-dir")
    parser.add_argument("--backend", choices=a4_spill_store.BACKENDS, default="direct")
    parser.add_argument("--accept-network", action="store_true",
                        help="spill to a network volume, labelled network scope (no verdict)")
    parser.add_argument("--host-total-bytes", type=int, default=None,
                        help="declared host RAM total; default: the cgroup limit, if any")
    parser.add_argument("--reserve-bytes", type=int, default=DEFAULT_RESERVE_BYTES)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--comparison", default="")
    parser.add_argument("--arm", default="A")
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--position", type=int, default=0)
    parser.add_argument("--watch-path", default=None,
                        help="the drive the foreign-read check watches; default the spill dir")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--drive-bytes", type=int, default=4 << 30)
    parser.add_argument("--drive-chunk-bytes", type=int, default=512 * 4096 * 2,
                        help="one spill chunk; default Mistral-7B at seq 512 in bf16 (4 MiB)")
    parser.add_argument("--drive-passes", type=int, default=3)
    parser.add_argument("--enospc-dir", default=None,
                        help="a SMALL dedicated filesystem for the real ENOSPC case (cleanup)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)
    if args.mode != "preflight" and args.mode != "drive" and not args.weights:
        parser.error(f"--mode {args.mode} needs --weights")
    if args.mode in ("preflight", "drive", "cleanup") and not args.spill_dir:
        parser.error(f"--mode {args.mode} needs --spill-dir")
    if args.enospc_dir:
        # Before any model is built: the directory, mount and volume-size rules. The
        # free-space rule needs the plan's spill file and is checked again in cleanup.
        problem = enospc_dir_problem(args.enospc_dir, 1 << 62)
        if problem:
            parser.error(problem)
    if args.device == "cpu":
        args.quant = "none"
        args.label = f"{CPU_PILOT_LABEL} {args.label}".strip()
    # The fields stream_probe.build() reads and this probe never varies.
    args.lazy_shard_handles = False
    args.control_sync_source = False
    args.no_pin = args.device == "cpu"
    return args


# ==========================================================================
# device helpers
# ==========================================================================
def _cuda(args: argparse.Namespace) -> bool:
    return args.device == "cuda"


def _sync(args: argparse.Namespace) -> None:
    if _cuda(args):
        import torch

        torch.cuda.synchronize()


def host_peak_bytes() -> Optional[int]:
    """The caching host allocator's peak since the last reset (rounded blocks)."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    stats = torch.cuda.host_memory_stats()
    value = stats.get("allocated_bytes.peak", stats.get("allocated_bytes.all.peak"))
    return None if value is None else int(value)


def reset_host_peak() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_host_memory_stats()


def watch(args: argparse.Namespace, path: str) -> arm_validity.ArmWatch:
    kwargs: Dict[str, Any] = {}
    if not _cuda(args):
        # The CPU pilot never touches the box's GPU, not even through nvidia-smi.
        kwargs.update(gpu_memory_used=lambda: None, gpu_processes=lambda: None,
                      gpu_clock=lambda: None)
    return arm_validity.ArmWatch(arm=args.arm, round=args.block, run=args.position,
                                 label=args.label or None, required=REQUIRED_CHECKS,
                                 model_path=path, **kwargs)


def _spill_threads() -> List[str]:
    return [t.name for t in threading.enumerate() if t.name.startswith("l2l-spill-")]


# ==========================================================================
# build and plan
# ==========================================================================
def build(args: argparse.Namespace) -> Dict[str, Any]:
    import torch

    import soup_cli
    from soup_cli.utils.layer_stream import resolve_stream_dtype

    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    dtype = resolve_stream_dtype(args.device)
    torch_dtype = getattr(torch, dtype)
    model, runtime, config, _index, weights_dir, shard_dir, shard_s, build_s = (
        stream_probe.build(args, args.device, dtype))
    stats = runtime.stats()
    element = torch.empty((), dtype=torch_dtype).element_size()
    meta = {
        **vars(args), "soup_cli_file": soup_cli.__file__, "dtype": dtype,
        "arch": str(config.model_type), "n_layers": int(config.num_hidden_layers),
        "hidden": int(config.hidden_size), "vocab": int(config.vocab_size),
        "chunk_bytes": args.seq * int(config.hidden_size) * element,
        "weights_dir": weights_dir, "shard_dir": shard_dir, "shard_seconds": shard_s,
        "build_seconds": build_s, "runtime": {k: v for k, v in stats.items() if k != "device"},
        "source_class": type(runtime.source).__name__,
        "direct_io": getattr(runtime.source, "direct_io", None),
        "gpu": stream_probe.gpu_facts(args.device) if _cuda(args) else None,
        "cgroup": a4_box.ram_limit(), "process_memory_after_build": a4_box.process_memory(),
        "torch": torch.__version__, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return {"model": model, "runtime": runtime, "config": config, "dtype": torch_dtype,
            "meta": meta}


def layout_for(args: argparse.Namespace, meta: Dict[str, Any]) -> budget.StoreLayout:
    return budget.StoreLayout(n_layers=meta["n_layers"], k=args.k,
                              chunk_bytes=meta["chunk_bytes"], ring=args.ring, pin=_cuda(args))


def weight_pinned_bytes(meta: Dict[str, Any]) -> int:
    """The shipped runtime's own figure (layer_stream_runtime.py:2966). A source that does
    not account for it (None) is charged 0 here; on Linux its page-locked bytes are then
    still charged through ``RssShmem`` by ``a4_box.non_activation_bytes``."""
    return int(meta["runtime"].get("pinned_bytes") or 0)


def host_total_bytes(declared: Optional[int], cgroup_limit: Optional[int]) -> Optional[int]:
    """The host total a split plans inside: the tighter of a declared total and the cgroup
    limit the process is under. A declaration above the cgroup limit cannot raise it."""
    known = [int(value) for value in (declared, cgroup_limit) if value]
    return min(known) if known else None


def plan_arm(args: argparse.Namespace, meta: Dict[str, Any], spec: str
             ) -> Tuple[budget.ActivationPlan, Optional[Dict[str, Any]], Optional[int]]:
    """The arm's plan: its budget spec, inside the host total when there is one."""
    layout = layout_for(args, meta)
    cap = budget.arm_budget_bytes(layout, spec)
    free = a4_spill_store.free_bytes(args.spill_dir) if args.spill_dir else None
    total = host_total_bytes(args.host_total_bytes, (meta.get("cgroup") or {}).get("limit_bytes"))
    if total:
        weights = weight_pinned_bytes(meta)
        process = a4_box.non_activation_bytes(a4_box.process_memory(), weights)
        if process is None:
            raise budget.BudgetRefusalError(
                budget.RAM_OVER_BUDGET, f"a {total:,} B host total is set, but this "
                "process's resident memory cannot be read, so its share cannot be charged")
        split = budget.split_host_ram(total, reserve_bytes=args.reserve_bytes,
                                      process_bytes=process, weight_pinned_bytes=weights,
                                      layout=layout, activation_cap_bytes=cap,
                                      spill_free_bytes=free)
        return split.plan, split.as_dict(), free
    return budget.plan_activations(cap, n_layers=layout.n_layers, k=layout.k,
                                   chunk_bytes=layout.chunk_bytes, ring=layout.ring,
                                   pin=layout.pin, spill_free_bytes=free), None, free


def make_store(args: argparse.Namespace, built: Dict[str, Any], plan: budget.ActivationPlan,
               free: Optional[int], *, spill_dir: Optional[str] = None,
               opener: Callable[..., Any] = a4_spill_store.open_spill,
               cls: type = a4_spill_store.BudgetedStore) -> a4_spill_store.BudgetedStore:
    meta = built["meta"]
    return cls(plan, chunk_shape=(1, args.seq, meta["hidden"]), dtype=built["dtype"],
               device=args.device, spill_dir=(spill_dir or args.spill_dir)
               if plan.spill_layers else None, backend=args.backend, opener=opener,
               free_bytes_at_plan=free)


def micro_batches(args: argparse.Namespace, vocab: int, k: int) -> List[Any]:
    import torch

    gen = torch.Generator(device=args.device).manual_seed(args.input_seed)
    return [torch.randint(0, vocab, (1, args.seq), generator=gen, device=args.device)
            for _ in range(k)]


# ==========================================================================
# the weight reads (disk tier)
# ==========================================================================
class WeightReadMeter:
    """Times every range read of the shipped ``AsyncDiskSource`` (installed on the live
    instance, as stream_probe.Instruments installs on the pools). ``rate`` is bytes over
    the UNION of the busy intervals: the drive's read rate while it was reading
    weights, which is what a concurrent spill write can slow down."""

    def __init__(self, source: Any):
        self.source = source
        self._lock = threading.Lock()
        self.intervals: List[Tuple[float, float]] = []
        self.bytes = 0
        original = source._range_works

        def timed_works(idx: int, region: Any) -> List[Any]:
            works = original(idx, region)
            with self._lock:
                self.bytes += int(source._plans[idx].span)

            def wrap(work: Callable[[], None]) -> Callable[[], None]:
                def run() -> None:
                    started = time.perf_counter()
                    try:
                        work()
                    finally:
                        with self._lock:
                            self.intervals.append((started, time.perf_counter()))
                return run

            return [wrap(work) for work in works]

        source._range_works = timed_works

    def snapshot(self) -> Tuple[int, int]:
        with self._lock:
            return self.bytes, len(self.intervals)

    def since(self, mark: Tuple[int, int]) -> Dict[str, Any]:
        with self._lock:
            spans = sorted(self.intervals[mark[1]:])
            moved = self.bytes - mark[0]
        busy, end = 0.0, None
        for start, stop in spans:
            if end is None or start > end:
                busy += stop - start
                end = stop
            elif stop > end:
                busy += stop - end
                end = stop
        return {"bytes": moved, "busy_s": busy,
                "gb_per_s": moved / busy / 1e9 if busy else None}


# ==========================================================================
# modes
# ==========================================================================
def mode_preflight(args: argparse.Namespace) -> Tuple[Dict[str, Any], int]:
    volume = a4_box.spill_volume(args.spill_dir, accept_network=args.accept_network)
    record = {"kind": "preflight", "volume": volume, "cgroup": a4_box.ram_limit(),
              "process_memory": a4_box.process_memory(),
              "leftover_spill_files": a4_spill_store.leftover_spill_files(args.spill_dir)}
    if args.weights:
        record["weights_volume"] = {k: v for k, v in a4_box.spill_volume(
            args.shards or args.weights, accept_network=True).items()
            if k in ("path", "fstype", "drive", "scope", "direct_io", "free_bytes")}
    print(f"spill volume  {volume['fstype']} {volume['scope']} direct_io={volume['direct_io']} "
          f"free={volume['free_bytes'] / 1e9:.1f} GB -> {'OK' if volume['ok'] else 'REFUSED'}")
    for reason in volume["refusals"]:
        print(f"  refused: {reason}")
    return record, 0 if volume["ok"] else 3


def _run_l2l(built: Dict[str, Any], step: Any, mbs: List[Any], store: Any
             ) -> Tuple[List[Any], Dict[str, Any]]:
    model = built["model"]
    l2l_schedule.zero_grads(model)
    with store:
        losses = step.run(mbs, store)
    return losses, l2l_schedule.collect_grads(model)


class CorruptFirstRead(l2l_activations.SpillBackend):
    """The control: the real spill backend, with the lowest mantissa bit of EVERY element of
    the first chunk read back (layer 0, micro-batch 0) flipped, once. One flipped bit in one
    element can round away (it did on SmolLM2-135M in the CPU pilot); a 1-ulp error in
    every element of a chunk cannot."""

    def __init__(self, inner: l2l_activations.SpillBackend, element_size: int):
        self.inner = inner
        self.direct = inner.direct
        self.path = inner.path
        self.element_size = int(element_size)
        self.flipped = False

    @property
    def left_behind(self) -> Optional[str]:
        return self.inner.left_behind

    def write(self, offset: int, buf: Any) -> None:
        self.inner.write(offset, buf)

    def read(self, offset: int, buf: Any) -> None:
        self.inner.read(offset, buf)
        if offset == 0 and not self.flipped:
            low = buf[:: self.element_size]  # little-endian: byte 0 holds the low mantissa bits
            low.bitwise_xor_(1)
            self.flipped = True

    def close(self, remove: bool = True) -> None:
        self.inner.close(remove=remove)


def mode_correctness(args: argparse.Namespace, built: Dict[str, Any]) -> Dict[str, Any]:
    model, runtime, meta = built["model"], built["runtime"], built["meta"]
    l2l_schedule.make_non_vacuous(model)
    mbs = micro_batches(args, meta["vocab"], args.k)
    l2l_schedule.zero_grads(model)
    ga1 = l2l_schedule.ga_step(model, mbs)
    g1 = l2l_schedule.collect_grads(model)
    l2l_schedule.zero_grads(model)
    ga2 = l2l_schedule.ga_step(model, mbs)
    g2 = l2l_schedule.collect_grads(model)
    step = l2l_schedule.L2LStep(model, runtime, l2l_schedule.capture_layer_call(model, mbs[0]))
    layout = layout_for(args, meta)
    free = a4_spill_store.free_bytes(args.spill_dir) if args.spill_dir else None

    def plan(spec: str) -> budget.ActivationPlan:
        return budget.plan_activations(budget.arm_budget_bytes(layout, spec),
                                       n_layers=layout.n_layers, k=layout.k,
                                       chunk_bytes=layout.chunk_bytes, ring=layout.ring,
                                       pin=layout.pin, spill_free_bytes=free)

    p1_losses, p1 = _run_l2l(built, step, mbs, make_store(args, built, plan("1"), free))
    p2_losses, p2 = _run_l2l(built, step, mbs, make_store(args, built, plan("1"), free))
    record: Dict[str, Any] = {
        "kind": "correctness", "k": args.k, "deterministic": args.deterministic,
        "ga_vs_ga": {"backward": l2l_schedule.compare(g1, g2),
                     "forward_equal": l2l_schedule.losses_equal(ga1, ga2)},
        "pinned_vs_pinned": {"backward": l2l_schedule.compare(p1, p2),
                             "forward_equal": l2l_schedule.losses_equal(p1_losses, p2_losses)},
        "pinned_vs_ga": {"backward": l2l_schedule.compare(g1, p1),
                         "forward_equal": l2l_schedule.losses_equal(ga1, p1_losses)},
        "losses_ga": [float(x) for x in ga1], "losses_pinned": [float(x) for x in p1_losses],
        "budgets": {},
    }
    deepest: Optional[budget.ActivationPlan] = None
    for spec in BUDGET_SPECS[1:]:
        try:
            chosen = plan(spec)
        except budget.BudgetRefusalError as exc:
            record["budgets"][spec] = {"refused": str(exc), "code": exc.code}
            continue
        losses, grads = _run_l2l(built, step, mbs, make_store(args, built, chosen, free))
        record["budgets"][spec] = {
            "plan": chosen.as_dict(),
            "vs_pinned": {"backward": l2l_schedule.compare(p1, grads),
                          "forward_equal": l2l_schedule.losses_equal(p1_losses, losses)},
            "vs_ga": {"backward": l2l_schedule.compare(g1, grads),
                      "forward_equal": l2l_schedule.losses_equal(ga1, losses)},
        }
        if chosen.spill_layers and (deepest is None or chosen.spill_layers > deepest.spill_layers):
            deepest = chosen
    if deepest is None:
        record["control"] = {"skipped": "no budget spills at this k and ring"}
    else:
        element = built["dtype"].itemsize

        def corrupt(kind: str, path: str, size: int) -> CorruptFirstRead:
            return CorruptFirstRead(a4_spill_store.open_spill(kind, path, size), element)

        losses, grads = _run_l2l(built, step, mbs,
                                 make_store(args, built, deepest, free, opener=corrupt))
        record["control"] = {
            "spill_layers": deepest.spill_layers,
            "vs_pinned": {"backward": l2l_schedule.compare(p1, grads),
                          "forward_equal": l2l_schedule.losses_equal(p1_losses, losses)},
        }
    record["leftover_spill_files"] = (a4_spill_store.leftover_spill_files(args.spill_dir)
                                      if args.spill_dir else [])
    for spec, row in record["budgets"].items():
        if "plan" in row:
            print(f"k={args.k} budget {spec:<4} spill {row['plan']['spill_layers']:>3} layers: "
                  f"forward {row['vs_pinned']['forward_equal']} "
                  f"backward {row['vs_pinned']['backward']['equal']}")
        else:
            print(f"k={args.k} budget {spec:<4} REFUSED {row['code']}")
    control = record["control"]
    if "vs_pinned" in control:
        backward = control["vs_pinned"]["backward"]["equal"]
        print(f"control (corrupted read): backward equal {backward} (expected False), "
              f"forward equal {control['vs_pinned']['forward_equal']}")
    return record


def run_timed(args: argparse.Namespace, built: Dict[str, Any], step: Any, store: Any,
              mbs: List[Any], meter: Optional[WeightReadMeter]) -> Dict[str, Any]:
    import torch

    model = built["model"]
    trainable = [p for p in model.parameters() if p.requires_grad]
    if _cuda(args):
        import bitsandbytes as bnb

        optimizer = bnb.optim.PagedAdamW8bit(trainable, lr=1e-4)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    records: List[Dict[str, Any]] = []
    for index in range(args.warmup + args.steps):
        _sync(args)
        if index == args.warmup:
            if _cuda(args):
                torch.cuda.reset_peak_memory_stats()
                reset_host_peak()
            a4_box.reset_peak_rss()
        stats = store.stats
        io0 = store.io()
        waits0 = (stats.put_wait_s, stats.get_wait_s)
        mark = meter.snapshot() if meter is not None else None
        started = time.perf_counter()
        losses = step.run(mbs, store)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        _sync(args)
        wall = time.perf_counter() - started
        if index < args.warmup:
            continue
        io1 = store.io()
        rec = {
            "step_s": wall,
            "losses": [stream_probe.finite(float(x)) for x in losses],
            "put_wait_s": stats.put_wait_s - waits0[0],
            "get_wait_s": stats.get_wait_s - waits0[1],
            "spill_write_s": io1["write_s"] - io0["write_s"],
            "spill_read_s": io1["read_s"] - io0["read_s"],
            "spill_written": io1["bytes_written"] - io0["bytes_written"],
            "spill_read": io1["bytes_read"] - io0["bytes_read"],
        }
        if meter is not None:
            rec["weight_read"] = meter.since(mark)
        records.append(rec)
    return {"records": records}


def summarise(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    def total(key: str) -> float:
        return float(sum(rec[key] for rec in records))

    busy = total("spill_write_s") + total("spill_read_s")
    exposed = total("put_wait_s") + total("get_wait_s")
    weights = [rec["weight_read"] for rec in records if "weight_read" in rec]
    weight_busy = sum(w["busy_s"] for w in weights)
    weight_bytes = sum(w["bytes"] for w in weights)
    return {
        "value": statistics.median(rec["step_s"] for rec in records),
        "step_s_mean": statistics.fmean(rec["step_s"] for rec in records),
        "step_s": [rec["step_s"] for rec in records],
        "spill_write_gb_per_s": total("spill_written") / total("spill_write_s") / 1e9
        if total("spill_write_s") else None,
        "spill_read_gb_per_s": total("spill_read") / total("spill_read_s") / 1e9
        if total("spill_read_s") else None,
        "spill_busy_s": busy, "spill_exposed_s": exposed,
        "hidden_share": None if not busy else max(0.0, 1.0 - exposed / busy),
        "weight_read_gb_per_s": weight_bytes / weight_busy / 1e9 if weight_busy else None,
        "weight_read_bytes_per_step": weight_bytes / len(weights) if weights else None,
        "finite": all(x is not None for rec in records for x in rec["losses"]),
    }


def mode_arm(args: argparse.Namespace, built: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    model, runtime, meta = built["model"], built["runtime"], built["meta"]
    l2l_schedule.make_non_vacuous(model)
    plan, split, free = plan_arm(args, meta, args.budget)
    volume = None
    if plan.spill_layers:
        volume = a4_box.spill_volume(args.spill_dir, accept_network=args.accept_network)
        if not volume["ok"]:
            return {"kind": "arm", "skipped": "spill volume refused: "
                    + "; ".join(volume["refusals"]), "volume": volume, "plan": plan.as_dict()}
    mbs = micro_batches(args, meta["vocab"], args.k)
    step = l2l_schedule.L2LStep(model, runtime, l2l_schedule.capture_layer_call(model, mbs[0]))
    meter = WeightReadMeter(runtime.source) if args.tier == "disk" else None
    a4_spill_store.release_pinned_cache()
    baseline = a4_spill_store.host_allocator_bytes()
    shmem_before = a4_box.process_memory()["rss_shmem"]
    watch_path = args.watch_path or args.spill_dir or meta["shard_dir"]
    store = make_store(args, built, plan, free)
    shmem_store = a4_box.process_memory()["rss_shmem"]
    path = store.path
    try:
        with store:
            with watch(args, watch_path) as watched:
                timed = run_timed(args, built, step, store, mbs, meter)
            peak = host_peak_bytes()
            stats = vars(store.stats).copy()
            cap = dict(store.cap)
            io = store.io()
    finally:
        store.close()
    summary = summarise(timed["records"])
    return {
        "kind": "arm", "skipped": None, "comparison": args.comparison, "arm": args.arm,
        "block": args.block, "position": args.position, "k": args.k, "budget": args.budget,
        "tier": args.tier, "ring": args.ring, "backend": args.backend,
        "plan": plan.as_dict(), "split": split, "volume": volume,
        # Where the spill and the weights really live, for G5's same-drive claim.
        "placement": {
            "spill": a4_box.placement(args.spill_dir) if args.spill_dir else None,
            "weights": a4_box.placement(meta["shard_dir"]),
        },
        "ids": stream_probe.ids_fingerprint(torch.cat(mbs)),
        "watch": watched.record, "outcome": watched.record["outcome"],
        **summary, "records": timed["records"], "store": stats, "io": io,
        "cap": cap,
        "pinned": {
            "allocator_baseline_bytes": baseline, "allocator_peak_bytes": peak,
            "store_budget_bytes": plan.budget_bytes, "store_host_bytes": plan.host_bytes,
            "runtime_pinned_bytes": meta["runtime"].get("pinned_bytes"),
            "rss_shmem_store_delta": None if shmem_before is None or shmem_store is None
            else shmem_store - shmem_before,
        },
        "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1e9 if _cuda(args) else None,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9 if _cuda(args) else None,
        "process_memory": a4_box.process_memory(), "cgroup": a4_box.ram_limit(),
        "spill_path": path, "spill_file_exists_after": bool(path) and os.path.exists(path),
        "leftover_spill_files": a4_spill_store.leftover_spill_files(args.spill_dir)
        if args.spill_dir else [],
        "spill_threads_after": _spill_threads(),
    }


def mode_drive(args: argparse.Namespace) -> Tuple[Dict[str, Any], int]:
    """Sequential chunk-sized unbuffered write then read of ``--drive-bytes`` on the
    spill drive, one chunk at a time, the way the store's writer and reader do it."""
    volume = a4_box.spill_volume(args.spill_dir, accept_network=args.accept_network)
    if not volume["ok"]:
        return {"kind": "drive", "skipped": "; ".join(volume["refusals"]), "volume": volume}, 3
    chunk = args.drive_chunk_bytes
    size = max(chunk, args.drive_bytes // chunk * chunk)
    path = os.path.join(args.spill_dir, f"a4-spill-drive-{os.getpid()}.bin")
    started = time.perf_counter()
    spill = l2l_activations.DirectSpillFile(path, size)
    fill_s = time.perf_counter() - started
    passes = []
    try:
        buf = l2l_activations.aligned_buffer(chunk, pin=_cuda(args))
        buf.fill_(7)
        with watch(args, args.spill_dir) as watched:
            for _ in range(args.drive_passes):
                t0 = time.perf_counter()
                for offset in range(0, size, chunk):
                    spill.write(offset, buf)
                t1 = time.perf_counter()
                for offset in range(0, size, chunk):
                    spill.read(offset, buf)
                t2 = time.perf_counter()
                passes.append({"write_gb_per_s": size / (t1 - t0) / 1e9,
                               "read_gb_per_s": size / (t2 - t1) / 1e9})
    finally:
        spill.close(remove=True)
    record = {"kind": "drive", "skipped": None, "volume": volume, "chunk_bytes": chunk,
              "bytes": size, "fill_gb_per_s": size / fill_s / 1e9, "passes": passes,
              "watch": watched.record, "outcome": watched.record["outcome"],
              "left_behind": os.path.exists(path)}
    for row in passes:
        print(f"drive write {row['write_gb_per_s']:.2f} GB/s  read {row['read_gb_per_s']:.2f} GB/s")
    return record, 0


#: The largest volume ``--enospc-dir`` may name: the real-ENOSPC case fills it.
ENOSPC_DIR_MAX_BYTES = 1 << 30


def enospc_dir_problem(directory: str, spill_file_bytes: int) -> Optional[str]:
    """Why ``directory`` cannot hold the real-ENOSPC case, or None.

    The case creates a spill file of ``spill_file_bytes`` there and expects the volume
    to refuse it, so the volume must be dedicated (its own mount point), small (at most
    ``ENOSPC_DIR_MAX_BYTES``: a wrong path must not fill a real drive) and have less free
    space than the file needs (else nothing is refused and the case proves nothing)."""
    path = os.path.realpath(directory)
    if not os.path.isdir(path):
        return f"--enospc-dir {directory} is not a directory"
    usage = shutil.disk_usage(path)
    if sys.platform != "win32" and not os.path.ismount(path):
        return (f"--enospc-dir {path} is not a mount point: give the root of a small "
                f"dedicated filesystem (runbook §8)")
    if usage.total > ENOSPC_DIR_MAX_BYTES:
        return (f"--enospc-dir {path} is on a {usage.total:,} B volume, over the "
                f"{ENOSPC_DIR_MAX_BYTES:,} B cap: the case fills it")
    if usage.free >= spill_file_bytes:
        return (f"--enospc-dir {path} has {usage.free:,} B free, enough for the "
                f"{spill_file_bytes:,} B spill file: the volume would not refuse it")
    return None


def mode_cleanup(args: argparse.Namespace, built: Dict[str, Any]) -> Dict[str, Any]:
    model, runtime, meta = built["model"], built["runtime"], built["meta"]
    k = max(2, args.k)
    mbs = micro_batches(args, meta["vocab"], k)
    step = l2l_schedule.L2LStep(model, runtime, l2l_schedule.capture_layer_call(model, mbs[0]))
    layout = budget.StoreLayout(n_layers=meta["n_layers"], k=k, chunk_bytes=meta["chunk_bytes"],
                                ring=args.ring, pin=_cuda(args))
    middle = meta["n_layers"] // 2

    class Interrupt(a4_spill_store.BudgetedStore):
        def get(self, layer: int, mb: int) -> Any:
            if layer == middle:
                signal.raise_signal(signal.SIGINT)
            return super().get(layer, mb)

    class Error(a4_spill_store.BudgetedStore):
        def put(self, layer: int, mb: int, tensor: Any) -> None:
            if layer == middle:
                raise RuntimeError("injected failure in the middle of the forward")
            super().put(layer, mb, tensor)

    def full_after(writes: int) -> Callable[..., Any]:
        class Full(l2l_activations.SpillBackend):
            def __init__(self, inner: Any):
                self.inner, self.direct, self.path, self.count = inner, inner.direct, inner.path, 0

            @property
            def left_behind(self) -> Optional[str]:
                return self.inner.left_behind

            def write(self, offset: int, buf: Any) -> None:
                if self.count >= writes:
                    raise OSError(errno.ENOSPC, "No space left on device (injected)")
                self.count += 1
                self.inner.write(offset, buf)

            def read(self, offset: int, buf: Any) -> None:
                self.inner.read(offset, buf)

            def close(self, remove: bool = True) -> None:
                self.inner.close(remove=remove)

        return lambda kind, path, size: Full(a4_spill_store.open_spill(kind, path, size))

    plan = budget.plan_activations(layout.minimum_bytes, n_layers=layout.n_layers, k=k,
                                   chunk_bytes=layout.chunk_bytes, ring=layout.ring,
                                   pin=layout.pin)
    scenarios: List[Tuple[str, type, Callable[..., Any], str, Optional[str]]] = [
        ("ctrl_c_mid_backward", Interrupt, a4_spill_store.open_spill, "KeyboardInterrupt", None),
        ("exception_mid_forward", Error, a4_spill_store.open_spill, "RuntimeError", None),
        ("enospc_injected_mid_step", a4_spill_store.BudgetedStore,
         full_after(middle * k), "SpillDiskFullError", None),
    ]
    if args.enospc_dir:
        problem = enospc_dir_problem(args.enospc_dir, plan.spill_file_bytes)
        if problem:
            raise ValueError(problem)
        scenarios.append(("enospc_real", a4_spill_store.BudgetedStore,
                          a4_spill_store.open_spill, "SpillDiskFullError", args.enospc_dir))
    a4_spill_store.release_pinned_cache()
    baseline = a4_spill_store.host_allocator_bytes()
    rows = []
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        for name, cls, opener, expected, directory in scenarios:
            spill_dir = directory or args.spill_dir
            caught: Optional[BaseException] = None
            store = None
            try:
                store = make_store(args, built, plan, None, spill_dir=spill_dir, opener=opener,
                                   cls=cls)
                with store:
                    step.run(mbs, store)
            except BaseException as exc:  # noqa: BLE001 — the scenario's outcome
                caught = exc
            path = store.path if store is not None else None
            a4_spill_store.release_pinned_cache()
            after = a4_spill_store.host_allocator_bytes()
            row = {
                "scenario": name, "expected": expected, "spill_dir": spill_dir,
                "raised": type(caught).__name__ if caught else None,
                "message": str(caught)[:2000] if caught else None,
                "spill_path": path, "file_exists_after": bool(path) and os.path.exists(path),
                "leftover_spill_files": a4_spill_store.leftover_spill_files(spill_dir),
                "spill_threads_after": _spill_threads(),
                "allocator_baseline_bytes": baseline, "allocator_after_bytes": after,
                "checkpoint_flags_restored": all(layer.use_checkpoint for layer in step.layers),
            }
            row["ok"] = (row["raised"] == expected and not row["file_exists_after"]
                         and not row["leftover_spill_files"] and not row["spill_threads_after"]
                         and (baseline is None or after == baseline))
            rows.append(row)
            returned = "unmeasured" if baseline is None or after is None else after == baseline
            print(f"{name:<28} raised {row['raised']:<18} file left {row['file_exists_after']}  "
                  f"pinned back {returned}  -> {'OK' if row['ok'] else 'FAIL'}")
            del caught, store
    finally:
        signal.signal(signal.SIGINT, previous)
    return {"kind": "cleanup", "k": k, "plan": plan.as_dict(), "scenarios": rows,
            "ok": all(row["ok"] for row in rows)}


# ==========================================================================
# main
# ==========================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    if out.exists():
        print(f"refusing to overwrite {out}")
        return 2
    if args.mode == "preflight":
        record, code = mode_preflight(args)
        sink = stream_probe.Sink(str(out), {**vars(args), "box": _box()})
        sink.add("preflight", record)
        return code
    if args.mode == "drive":
        record, code = mode_drive(args)
        sink = stream_probe.Sink(str(out), {**vars(args), "box": _box()})
        sink.add("drive", record)
        return code
    if _cuda(args) and not stream_probe.cuda_available():
        print("SKIP: CUDA is required for a GPU arm; --device cpu runs the CPU pilot")
        stream_probe.Sink(str(out), {**vars(args), "skipped": "no CUDA"})
        return 3
    built = build(args)
    sink = stream_probe.Sink(str(out), built["meta"])
    try:
        if args.mode == "correctness":
            sink.add("correctness", mode_correctness(args, built))
        elif args.mode == "cleanup":
            result = mode_cleanup(args, built)
            sink.add("cleanup", result)
        else:
            result = mode_arm(args, built)
            sink.add("arm", result)
            if not result.get("skipped"):
                print(f"{args.comparison or 'arm'} {args.arm} k={args.k} budget {args.budget}: "
                      f"median step {result['value']:.3f} s, spill {result['plan']['spill_layers']}"
                      f" layers, hidden {result['hidden_share']}, outcome {result['outcome']}")
    finally:
        built["runtime"].close()
    print(f"wrote {out}")
    return 0


def _box() -> Dict[str, Any]:
    return {"cgroup": a4_box.ram_limit(), "process_memory": a4_box.process_memory(),
            "unix": time.time()}


if __name__ == "__main__":
    sys.exit(main())
