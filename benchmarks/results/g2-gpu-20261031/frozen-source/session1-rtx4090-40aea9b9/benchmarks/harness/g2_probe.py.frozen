#!/usr/bin/env python3
"""G2: recompute versus storing each decoder layer's inner activations, on the
SHIPPED streamed runtime.

Modes (one fresh process each; ``g2_session.py`` runs them in order):

``shapes``       saved bytes per token per layer for the Mistral-7B and Llama-70B
                 decoder shapes (CPU kernels, zero-stride weights: an estimate
                 for the CUDA kernels, labelled so). Needs no weights.
``correctness``  GA vs GA, recompute L2L vs GA, G2 (pinned RAM records) and G2
                 (records through the spill file) vs GA and vs L2L, losses and
                 LoRA grads with ``torch.equal``; the identity-hook control
                 (scratch exp2: no reload, must raise or differ) and the
                 corrupted-record control (must change grads, not losses);
                 the record bytes by class and the weight-alias check.
``profile``      per-phase seconds of the recompute step and of the store step
                 with the device synchronised at each phase boundary: recompute
                 seconds per step (what G2 could save), store overhead, record
                 bytes per step, weight bytes per step. Input to the break-even.
``arm``          one timed arm of a comparison (A = recompute L2L, B = G2 store),
                 under F1's ArmWatch, the token rows fingerprinted.

Benchmark harness code, not shipped. torch is imported inside functions.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arm_validity  # noqa: E402
import g2_disk  # noqa: E402
import g2_inner_activations as g2  # noqa: E402
import l2l_activations  # noqa: E402
import l2l_box  # noqa: E402
import l2l_schedule  # noqa: E402
import stream_probe  # noqa: E402

CPU_PILOT_LABEL = "SYNTHETIC/CPU PILOT"
REQUIRED_CHECKS = ("suspend", "power", "foreign_reads")
VARIANTS = {"A": "recompute", "B": "store"}


# ==========================================================================
# arguments
# ==========================================================================
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", required=True,
                        choices=("shapes", "correctness", "profile", "arm"))
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
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--store", choices=("ram", "spill"), default="ram",
                        help="where the G2 records go (arm B / profile)")
    parser.add_argument("--spill-layers", type=int, default=None,
                        help="with --store spill: records of the lowest N layers spill "
                             "(default all)")
    parser.add_argument("--spill-dir", default=None)
    parser.add_argument("--ring", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--comparison", default="")
    parser.add_argument("--arm", choices=tuple(VARIANTS), default="A",
                        help="arm mode: A = recompute L2L, B = G2 store")
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--position", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--allow-buffered", action="store_true",
                        help="CPU only: page-cache spill backend; its records never count")
    parser.add_argument("--shapes", default="mistral-7b,llama-70b")
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)
    if args.mode != "shapes" and not args.weights:
        parser.error(f"--mode {args.mode} needs --weights")
    spills = args.mode == "correctness" or (
        args.store == "spill" and args.mode in ("arm", "profile"))
    if spills and not args.spill_dir:
        parser.error(f"--mode {args.mode} needs --spill-dir")
    if args.allow_buffered and args.device == "cuda":
        parser.error("--allow-buffered is for the CPU pilot only")
    if args.device == "cpu":
        args.label = f"{CPU_PILOT_LABEL} {args.label}".strip()
    # The fields stream_probe.build() reads and this probe never varies.
    args.lazy_shard_handles = False
    args.control_sync_source = False
    args.no_pin = args.device == "cpu"
    return args


def _cuda(args: argparse.Namespace) -> bool:
    return args.device == "cuda"


def _sync(args: argparse.Namespace) -> None:
    if _cuda(args):
        import torch

        torch.cuda.synchronize()


def write_json(path: str, payload: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp"
    Path(tmp).write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n",
                         encoding="utf-8")
    os.replace(tmp, path)


# ==========================================================================
# build
# ==========================================================================
def build(args: argparse.Namespace) -> Dict[str, Any]:
    import torch

    import soup_cli
    from soup_cli.utils.layer_stream import resolve_stream_dtype

    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    # Must precede build: install_dequant_forward binds dequantize_4bit per layer.
    tracking = g2.install_derived_tracking() if args.quant == "nf4" else False
    dtype = resolve_stream_dtype(args.device)
    model, runtime, config, _index, weights_dir, shard_dir, shard_s, build_s = (
        stream_probe.build(args, args.device, dtype))
    stats = runtime.stats()
    meta = {
        **vars(args), "argv": sys.argv, "soup_cli_file": soup_cli.__file__, "dtype": dtype,
        "arch": str(config.model_type), "n_layers": int(config.num_hidden_layers),
        "hidden": int(config.hidden_size), "vocab": int(config.vocab_size),
        "weights_dir": weights_dir, "shard_dir": shard_dir, "shard_seconds": shard_s,
        "build_seconds": build_s,
        "runtime": {key: value for key, value in stats.items() if key != "device"},
        "source_class": type(runtime.source).__name__,
        "direct_io": getattr(runtime.source, "direct_io", None),
        "derived_tracking": tracking, "torch": torch.__version__,
        "synthetic": not _cuda(args),
        "gpu": stream_probe.gpu_facts(args.device) if _cuda(args) else None,
        "box": l2l_box.box_stamp(), "host": g2_disk.host_identity(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return {"model": model, "runtime": runtime, "config": config,
            "dtype": getattr(torch, dtype), "meta": meta}


def micro_batches(args: argparse.Namespace, vocab: int, k: int) -> List[Any]:
    import torch

    gen = torch.Generator(device=args.device).manual_seed(args.input_seed)
    return [torch.randint(0, vocab, (1, args.seq), generator=gen, device=args.device)
            for _ in range(k)]


def boundary_store(args: argparse.Namespace, meta: Dict[str, Any], dtype: Any, k: int) -> Any:
    """The recompute reference's all-pinned boundary store (A0's ActivationStore)."""
    return l2l_activations.ActivationStore(
        n_layers=meta["n_layers"], k=k, chunk_shape=(1, args.seq, meta["hidden"]),
        dtype=dtype, device=args.device, pin=_cuda(args))


def spill_prefix(tag: str) -> str:
    return f"g2-records-{tag}-"


def record_store_factory(args: argparse.Namespace, *, spill: bool, tag: str) -> Any:
    if not spill:
        return g2.ram_store_factory(args.device, pin=_cuda(args))
    Path(args.spill_dir).mkdir(parents=True, exist_ok=True)
    path = str(Path(args.spill_dir) / f"{spill_prefix(tag)}{os.getpid()}.bin")
    return g2.spill_store_factory(args.device, pin=_cuda(args), spill_layers=args.spill_layers,
                                  path=path, direct=not args.allow_buffered, ring=args.ring)


def left_behind(args: argparse.Namespace, tag: str) -> List[str]:
    if not args.spill_dir or not os.path.isdir(args.spill_dir):
        return []
    return sorted(name for name in os.listdir(args.spill_dir)
                  if name.startswith(spill_prefix(tag)))


def _pair(ref_grads: Any, ref_losses: Any, grads: Any, losses: Any) -> Dict[str, Any]:
    return {"backward": l2l_schedule.compare(ref_grads, grads),
            "forward_equal": l2l_schedule.losses_equal(ref_losses, losses)}


def byte_record(step: Any, args: argparse.Namespace, k: int) -> Dict[str, Any]:
    """The record bytes one step stored, and what the store actually moved."""
    st = step.store.stats
    rows = list(step.record_stats.values())
    return {
        **step.summary(args.seq),
        "records": len(rows),
        "payload_bytes_per_step": sum(row["payload_bytes"] for row in rows),
        "weight_bytes_stored": sum(row["weight_bytes_stored"] for row in rows),
        "weight_like_items": sum(row["weight_like_items"] for row in rows),
        "view_bytes_per_step": sum(row["view_bytes"] for row in rows),
        "record_capacity_bytes": step.capacity, "store_chunk_bytes": st.chunk_bytes,
        "resident_chunks": st.resident_chunks, "spilled_chunks": st.spilled_chunks,
        "spill_layers": st.spill_layers, "planned_spill_bytes": st.spill_layers * k * step.capacity,
        "spill_written": st.bytes_written, "spill_read": st.bytes_read,
        "spill_direct": st.direct, "put_wait_s": st.put_wait_s, "get_wait_s": st.get_wait_s,
    }


# ==========================================================================
# modes
# ==========================================================================
def mode_shapes(args: argparse.Namespace) -> Dict[str, Any]:
    targets = [name.strip() for name in args.lora_targets.split(",") if name.strip()]
    rows = []
    for shape in [name.strip() for name in args.shapes.split(",") if name.strip()]:
        for attn in ("sdpa", "eager"):
            row = g2.shape_estimate(shape, seq=args.seq, lora_r=args.lora_r, targets=targets,
                                    attn=attn)
            rows.append(row)
            print(f"{shape:<11} {attn:<5} seq {args.seq}: {row['bytes_per_token']:>9.0f} B/token/"
                  f"layer saved ({row['inner_bytes_per_token']:.0f} inner, "
                  f"{row['boundary_bytes_per_token']:.0f} boundary)")
    return {"kind": "shapes", "estimate": "CPU kernels; CUDA SDPA saves different tensors",
            "rows": rows}


def mode_correctness(args: argparse.Namespace, built: Dict[str, Any]) -> Dict[str, Any]:
    model, runtime, meta = built["model"], built["runtime"], built["meta"]
    l2l_schedule.make_non_vacuous(model)
    k = args.k
    mbs = micro_batches(args, meta["vocab"], k)
    record: Dict[str, Any] = {"kind": "correctness", "k": k,
                              "deterministic": bool(args.deterministic)}
    l2l_schedule.zero_grads(model)
    ga1 = l2l_schedule.ga_step(model, mbs)
    g1 = l2l_schedule.collect_grads(model)
    l2l_schedule.zero_grads(model)
    ga2 = l2l_schedule.ga_step(model, mbs)
    record["ga_vs_ga"] = _pair(g1, ga1, l2l_schedule.collect_grads(model), ga2)
    call = l2l_schedule.capture_layer_call(model, mbs[0])
    store = boundary_store(args, meta, built["dtype"], k)
    try:
        l2l_schedule.zero_grads(model)
        l2l_losses = l2l_schedule.L2LStep(model, runtime, call).run(mbs, store)
        l2l = l2l_schedule.collect_grads(model)
    finally:
        store.close()
    record["l2l_vs_ga"] = _pair(g1, ga1, l2l, l2l_losses)
    record["losses_ga"] = [float(x) for x in ga1]
    for name, spill in (("g2_ram", False), ("g2_spill", True)):
        # verify=True keeps a clone of every weight view to compare on unpack: it
        # copies weights, so it is a CPU-pilot cross-check only.
        step = g2.G2Step(model, runtime, call, verify=not _cuda(args),
                         store_factory=record_store_factory(args, spill=spill, tag=name))
        try:
            l2l_schedule.zero_grads(model)
            losses = step.run(mbs)
            grads = l2l_schedule.collect_grads(model)
            bytes_ = byte_record(step, args, k)
        finally:
            step.close()
        record[f"{name}_vs_ga"] = _pair(g1, ga1, grads, losses)
        record[f"{name}_vs_l2l"] = _pair(l2l, l2l_losses, grads, losses)
        bytes_["left_behind"] = left_behind(args, name)
        record[f"{name}_bytes"] = bytes_
    controls: Dict[str, Any] = {}
    naive = g2.G2Step(model, runtime, call, store_factory=None, naive_identity=True)
    l2l_schedule.zero_grads(model)
    try:
        naive_losses = naive.run(mbs)
        controls["naive_identity"] = {
            "raised": None, **_pair(g1, ga1, l2l_schedule.collect_grads(model), naive_losses)}
    except RuntimeError as exc:
        controls["naive_identity"] = {"raised": str(exc).splitlines()[0][:300]}
    corrupt = g2.G2Step(model, runtime, call,
                        store_factory=record_store_factory(args, spill=True, tag="corrupt"))
    corrupt.corrupt = (meta["n_layers"] // 2, 0)
    try:
        l2l_schedule.zero_grads(model)
        bad_losses = corrupt.run(mbs)
        controls["corrupt"] = {"record": list(corrupt.corrupt),
                               **_pair(g1, ga1, l2l_schedule.collect_grads(model), bad_losses)}
    finally:
        corrupt.close()
    record["controls"] = controls
    for name in ("g2_ram", "g2_spill"):
        print(f"k={k} {name}: vs GA backward {record[name + '_vs_ga']['backward']['equal']} "
              f"forward {record[name + '_vs_ga']['forward_equal']}; vs L2L backward "
              f"{record[name + '_vs_l2l']['backward']['equal']}; streamed-weight bytes in "
              f"records {record[name + '_bytes']['weight_bytes_stored']}")
    naive_row = controls["naive_identity"]
    print(f"control identity hooks: raised={bool(naive_row.get('raised'))} backward equal "
          f"{(naive_row.get('backward') or {}).get('equal')} (expected a raise or False)")
    print(f"control corrupt record: backward equal {controls['corrupt']['backward']['equal']} "
          f"(expected False), forward equal {controls['corrupt']['forward_equal']}")
    return record


def _profile_steps(step: Any, run: Callable[[], Any], args: argparse.Namespace,
                   pool: Any, stats: Callable[[], Any] = lambda: None) -> Dict[str, float]:
    """Medians over the timed steps of the step's phase clocks, the wall time, the layer
    loads and, from ``stats()`` (the live store's ``StoreStats``), the seconds ``put``
    and ``get`` spent waiting on the spill drive (ring buffer free / chunk read back)."""
    totals: Dict[str, List[float]] = {}
    for index in range(args.warmup + args.steps):
        _sync(args)
        loads = pool.loads
        before = stats()
        waits = (before.put_wait_s, before.get_wait_s) if before is not None else (0.0, 0.0)
        started = time.perf_counter()
        run()
        _sync(args)
        wall = time.perf_counter() - started
        after = stats()
        if index < args.warmup:
            continue
        io = {"store_put_wait_s": after.put_wait_s - waits[0] if after is not None else 0.0,
              "store_get_wait_s": after.get_wait_s - waits[1] if after is not None else 0.0}
        for name, value in {**step.timing, **io, "step_s": wall,
                            "layer_loads": float(pool.loads - loads)}.items():
            totals.setdefault(name, []).append(value)
    return {name: statistics.median(values) for name, values in totals.items()}


def mode_profile(args: argparse.Namespace, built: Dict[str, Any]) -> Dict[str, Any]:
    model, runtime, meta = built["model"], built["runtime"], built["meta"]
    l2l_schedule.make_non_vacuous(model)
    k = args.k
    mbs = micro_batches(args, meta["vocab"], k)
    call = l2l_schedule.capture_layer_call(model, mbs[0])
    pool = runtime.pool
    store = boundary_store(args, meta, built["dtype"], k)
    ref = g2.ProfiledL2LStep(model, runtime, call, profile=True)
    try:
        recompute = _profile_steps(
            ref, lambda: (l2l_schedule.zero_grads(model), ref.run(mbs, store)), args, pool,
            stats=lambda: store.stats)
    finally:
        store.close()
    step = g2.G2Step(model, runtime, call, profile=True,
                     store_factory=record_store_factory(args, spill=args.store == "spill",
                                                        tag="profile"))
    try:
        stored = _profile_steps(
            step, lambda: (l2l_schedule.zero_grads(model), step.run(mbs)), args, pool,
            stats=lambda: step.store.stats if step.store is not None else None)
        bytes_ = byte_record(step, args, k)
    finally:
        step.close()
    per_layer = pool.nbytes // pool.n
    # store_put_s / store_get_s time the whole put/get, which includes waiting on the
    # spill drive (a free ring buffer, a chunk read back). break_even charges the drive
    # separately from its sustained rates, so the overhead here is the store's own work
    # (packing, host copies) with the measured waits taken out; the waits are reported.
    waits = {name: stored.get(f"store_{name}_wait_s", 0.0) - recompute.get(
        f"store_{name}_wait_s", 0.0) for name in ("put", "get")}
    overhead = (stored.get("store_put_s", 0.0) + stored.get("store_get_s", 0.0)
                - recompute.get("store_put_s", 0.0) - recompute.get("store_get_s", 0.0)
                - waits["put"] - waits["get"])
    row = {
        "kind": "profile", "k": k, "tier": args.tier, "store": args.store,
        "recompute": recompute, "stored": stored,
        "recompute_s_per_step": recompute.get("recompute_s", 0.0),
        "store_overhead_s_per_step": overhead,
        "store_overhead_excludes": "spill I/O waits (charged by break_even from the rates)",
        "store_io_wait_s_per_step": waits,
        "forward_compute_s_per_step": stored.get("forward_layers_s"),
        "backward_compute_s_per_step": stored.get("backward_layers_s"),
        "record_capacity_bytes": bytes_["record_capacity_bytes"],
        "records_per_step": meta["n_layers"] * k,
        "bytes": bytes_,
        "weight_bytes_per_step": int(stored["layer_loads"] * per_layer)
        if args.tier == "disk" else 0,
        "tokens_per_step": k * args.seq,
        "left_behind": left_behind(args, "profile"),
        # where this profile ran and which disks it used: the break-even only takes
        # sustained rates measured on the same host and the same spill / weight disks
        "host": meta["host"],
        "placement": {
            "spill": g2_disk.volume_info(args.spill_dir) if args.spill_dir else None,
            "weights": g2_disk.volume_info(meta["shard_dir"]),
        },
        "note": "device synchronised at every phase boundary: these per-phase seconds are "
                "inputs to the break-even, not the timed arms' step time",
    }
    print(f"profile k={k}: recompute {row['recompute_s_per_step']:.3f} s/step, store overhead "
          f"{overhead:.3f} s/step, synchronised step {recompute['step_s']:.3f} -> "
          f"{stored['step_s']:.3f} s, records "
          f"{row['record_capacity_bytes'] * row['records_per_step'] / 1e6:.1f} MB/step")
    return row


def watch(args: argparse.Namespace, path: Optional[str]) -> arm_validity.ArmWatch:
    kwargs: Dict[str, Any] = {}
    if not _cuda(args):
        # The CPU pilot never touches the box's GPU, not even through nvidia-smi.
        kwargs.update(gpu_memory_used=lambda: None, gpu_processes=lambda: None,
                      gpu_clock=lambda: None)
    return arm_validity.ArmWatch(arm=args.arm, round=args.block, run=args.position,
                                 label=args.label or None, required=REQUIRED_CHECKS,
                                 model_path=path, **kwargs)


def mode_arm(args: argparse.Namespace, built: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    model, runtime, meta = built["model"], built["runtime"], built["meta"]
    l2l_schedule.make_non_vacuous(model)
    k = args.k
    mbs = micro_batches(args, meta["vocab"], k)
    call = l2l_schedule.capture_layer_call(model, mbs[0])
    trainable = [p for p in model.parameters() if p.requires_grad]
    if _cuda(args):
        import bitsandbytes as bnb

        optimizer = bnb.optim.PagedAdamW8bit(trainable, lr=1e-4)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    storing = VARIANTS[args.arm] == "store"
    tag = f"arm{args.block}-{args.position}"
    step = boundary = None
    if storing:
        step = g2.G2Step(model, runtime, call,
                         store_factory=record_store_factory(args, spill=args.store == "spill",
                                                            tag=tag))
    else:
        boundary = boundary_store(args, meta, built["dtype"], k)
        reference = l2l_schedule.L2LStep(model, runtime, call)
    pool = runtime.pool
    steps: List[Dict[str, Any]] = []
    watch_path = args.spill_dir if storing and args.store == "spill" else meta["shard_dir"]
    try:
        with watch(args, watch_path) as watched:
            for index in range(args.warmup + args.steps):
                live = step.store if storing else boundary
                written = live.stats.bytes_written if live is not None else 0
                read = live.stats.bytes_read if live is not None else 0
                loads = pool.loads
                _sync(args)
                if index == args.warmup and _cuda(args):
                    torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                losses = step.run(mbs) if storing else reference.run(mbs, boundary)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                _sync(args)
                wall = time.perf_counter() - started
                live = step.store if storing else boundary
                if index < args.warmup:
                    continue
                steps.append({
                    "step_s": wall,
                    "losses": [stream_probe.finite(float(x)) for x in losses],
                    "layer_loads": pool.loads - loads,
                    "spill_written": live.stats.bytes_written - written,
                    "spill_read": live.stats.bytes_read - read,
                })
        planned = step.store.stats.spill_layers * k * step.capacity if storing else 0
        direct = step.store.stats.direct if storing else None
        capacity = step.capacity if storing else None
    finally:
        if storing:
            step.close()
        else:
            boundary.close()
    fingerprint = stream_probe.ids_fingerprint(torch.cat(mbs))
    # Where the records' spill and the streamed weights really live (V10): two
    # direct-I/O flags say nothing about which drive either is on.
    placement = {
        "spill": g2_disk.volume_info(args.spill_dir) if args.spill_dir else None,
        "weights": g2_disk.volume_info(meta["shard_dir"]),
    }
    placement["same_physical_disk"] = g2_disk.same_disk(placement["spill"],
                                                        placement["weights"])
    return {
        "kind": "arm", "comparison": args.comparison, "arm": args.arm,
        "variant_role": args.arm, "variant": VARIANTS[args.arm], "block": args.block,
        "position": args.position, "k": k, "tier": args.tier, "store": args.store,
        "ids_sha256": "|".join(fingerprint["row_sha256"]), "steps": steps,
        "losses": [x for row in steps for x in row["losses"]],
        "planned_spill_bytes_per_step": planned, "spill_direct": direct,
        "record_capacity_bytes": capacity, "placement": placement,
        "F1": watched.record, "outcome": watched.record["outcome"],
        "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1e9 if _cuda(args) else None,
        "spill_threads_after": [t.name for t in threading.enumerate()
                                if t.name.startswith("l2l-spill-")],
        "left_behind": left_behind(args, tag),
    }


# ==========================================================================
# main
# ==========================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.mode == "shapes":
        write_json(args.out, {"meta": {"mode": "shapes", "device": "cpu", "synthetic": True,
                                       "label": "shape estimate (CPU kernels)",
                                       "argv": sys.argv},
                              "records": [mode_shapes(args)]})
        print(f"wrote         {args.out}")
        return 0
    if _cuda(args) and not stream_probe.cuda_available():
        print("REFUSED: --device cuda but CUDA is not available")
        return 3
    built = build(args)
    payload: Dict[str, Any] = {"meta": built["meta"], "records": []}
    modes = {"correctness": mode_correctness, "profile": mode_profile, "arm": mode_arm}
    exit_code = 0
    try:
        payload["records"].append(modes[args.mode](args, built))
    except Exception as exc:  # noqa: BLE001 — a failed block is evidence: written, not dropped
        traceback.print_exc()
        payload["records"].append({"kind": args.mode, "error": f"{type(exc).__name__}: {exc}"})
        exit_code = 1
    finally:
        payload["meta"]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        write_json(args.out, payload)
        built["runtime"].close()
    print(f"wrote         {args.out}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
