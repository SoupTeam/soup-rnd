"""Ordinary against complete streaming: runs one arm of the Kanana plan and judges accuracy.

Plan: ``benchmarks/streaming-ordinary-vs-complete-l2l-plan-20261114.md``. Rule and protocol:
``benchmarks/streaming-ordinary-vs-complete-l2l-protocol-20261114.md`` (sections 2 to 4). This file
is the launcher the protocol lists as missing. It is benchmark harness code, not shipped. Heavy
imports sit inside the functions.

Three subcommands.

``prepare``  Shards the source checkpoint into one ``per_expert`` cache (NF4 on a card, no
             quantisation on CPU). Every arm reads the same cache.
``run``      One arm in one fresh process: ``--arm`` (ga_ordinary, ga_complete, l2l_ordinary,
             l2l_complete, ga_floor), ``--k`` micro-batches per optimizer update, ``--updates``
             updates (the first ``--warmup`` do not count). Writes ``<out>/record.json`` and, with
             ``--capture``, ``<out>/capture.pt``. The ABBA driver runs this inside ``ArmWatch``.
``judge``    Reads the ``capture.pt`` of the five arms and writes the accuracy verdict (protocol
             3.3): tolerance A3 on a card (1.5 times the noise floor), 1e-4 on CPU, bit equality
             reported next to it.

Arms. The complete arms switch on the four levers through ``build_streamed_model``: two-read,
grouped routes, packed NF4 transfer and the prepacked host cache. On CPU only the first two run
(packed needs CUDA and NF4), and the record says so (``levers_applied``). Under L2L two-read
is disabled for the historical ``l2l_complete`` arm A. ``l2l_two_read`` is arm B,
with all four levers enabled (ticket 08).

Data. Random token ids of ``--seq`` tokens, batch of one. Window ``u``, micro-batch ``j`` is
seeded ``SEED + 1000 * u + j``, so every arm sees the same rows. The record's ``kind`` is
``synthetic``. Both modes run without autocast, as ``b5_real_moe.py`` does: the base is bf16 and
the LoRA is fp32.

Reads. ``--hits-probe`` runs one no-grad forward over window 0 before training. Without a
checkpoint recompute or a backward each hit expert is read once, so the fetches of that pass are
the hits of the window, and ``reads_per_hit`` is the training fetches of update 0 over them. Use
it on capture runs, not on timed runs: the pass warms the host cache.

Capture. ``--capture`` sets ``CUBLAS_WORKSPACE_CONFIG`` before CUDA starts, turns strict
deterministic mode on (``--deterministic``) and pins one SDPA backend (MATH, and
EFFICIENT_ATTENTION for ``ga_floor``), as ``l2l_mistral_stand.py`` does. Strict mode raises on a
non-deterministic op. If a MoE kernel does that on the card, use ``--deterministic warn`` and
say so in the report: the record keeps the choice.

    python benchmarks/harness/oc_arm_run.py prepare --source SRC --cache CACHE --device cuda
    python benchmarks/harness/oc_arm_run.py run --arm ga_complete --k 2 --source SRC \\
        --cache CACHE --out OUT --device cuda
    python benchmarks/harness/oc_arm_run.py judge --runs RUNS --k 2
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rich.console import Console

try:
    from benchmarks.harness import b5_real_moe, l2l_mistral_stand
except ImportError:  # run as a script from benchmarks/harness
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import b5_real_moe  # type: ignore[no-redef]
    import l2l_mistral_stand  # type: ignore[no-redef]

console = Console()

SEED = 20261114
CLIP_NORM = 1.0
GIB = 1 << 30
ARM_NAMES = ("ga_ordinary", "ga_complete", "l2l_ordinary", "l2l_complete",
             "l2l_two_read", "ga_floor")
#: arm -> mode. ``ga_floor`` is GA on a second SDPA backend, the noise floor of tolerance A3.
ARM_MODE = {
    "ga_ordinary": "ga", "ga_complete": "ga", "l2l_ordinary": "l2l", "l2l_complete": "l2l",
    "ga_floor": "ga", "l2l_two_read": "l2l",
}
COMPLETE_ARMS = ("ga_complete", "l2l_complete", "l2l_two_read")
#: Recipe of the plan (section 4.1) and PR #5.
DEFAULT_TARGETS = ("q_proj", "kv_b_proj", "o_proj")
PAIRS = (
    ("ga_ordinary", "ga_complete"),
    ("l2l_ordinary", "l2l_complete"),
    ("ga_ordinary", "l2l_ordinary"),
)


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------
def levers_for(arm: str, *, device: str, host_cache_bytes: int) -> Tuple[Dict[str, Any], List[str]]:
    """The ``build_streamed_model`` options of an arm, and the levers left off on this device."""
    if arm not in ARM_MODE:
        raise ValueError(f"unknown arm {arm!r}, expected one of {ARM_NAMES}")
    if arm not in COMPLETE_ARMS:
        return {}, []
    levers: Dict[str, Any] = {
        "two_read_experts": arm != "l2l_complete", "grouped_expert_routes": True,
    }
    if device != "cuda":
        return levers, ["expert_transfer", "expert_host_cache_bytes"]
    levers["expert_transfer"] = "packed"
    if host_cache_bytes:
        levers["expert_host_cache_bytes"] = host_cache_bytes
    return levers, []


def window_ids(vocab_size: int, seq: int, update: int, k: int) -> List[Any]:
    """The ``k`` micro-batches of one update: the same rows in every arm."""
    import torch

    rows = []
    for j in range(k):
        generator = torch.Generator().manual_seed(SEED + 1000 * update + j)
        rows.append(torch.randint(0, vocab_size, (1, seq), generator=generator))
    return rows


def _device_dtype(device: str) -> Tuple[str, str]:
    """``(dtype, quant)``: bf16 NF4 on a card, fp32 unquantised on CPU."""
    return ("bfloat16", "nf4") if device == "cuda" else ("float32", "none")


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------
def prepare(args: argparse.Namespace) -> None:
    """Shard the source into the shared ``per_expert`` cache. Refuses an existing cache."""
    import shutil

    from soup_cli.utils.expert_cache_budget import new_expert_cache_budget
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import (
        build_meta_skeleton,
        quantised_layer_suffixes,
        shard_layout_of,
    )

    source, cache = Path(args.source), Path(args.cache)
    if cache.exists() and any(cache.iterdir()):
        raise ValueError(f"the cache folder {cache} is not empty")
    dtype, quant = _device_dtype(args.device)
    started = time.perf_counter()
    if quant == "nf4":
        with b5_real_moe._meta_construction():
            skeleton = build_meta_skeleton(str(source), dtype=dtype, quant="nf4",
                                           double_quant=args.double_quant)
        suffixes, layout = quantised_layer_suffixes(skeleton), shard_layout_of(skeleton)
        del skeleton
        gc.collect()
        budget = new_expert_cache_budget([str(path) for path in source.glob("*.safetensors")])
        free = shutil.disk_usage(cache.parent if cache.parent.exists() else Path(".")).free
        if free < budget:
            raise ValueError(f"the cache needs {budget} free bytes, {free} available")
        index = shard_checkpoint(
            str(source), str(cache), dtype=dtype, arch="deepseek_v3", layout=layout, quant="nf4",
            quant_suffixes=suffixes, quant_device=args.device, double_quant=args.double_quant,
            expert_layout="per_expert")
    else:
        index = shard_checkpoint(str(source), str(cache), dtype=dtype, arch="deepseek_v3",
                                 expert_layout="per_expert")
    console.print({"prepared": str(cache), "seconds": time.perf_counter() - started,
                   "parameters": index.total_params,
                   "cache_bytes": b5_real_moe._folder_bytes(cache)})


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
def _repo_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def _expert_pool_stats(runtime: Any) -> Dict[str, Any]:
    pool = getattr(runtime, "expert_pool", None)
    stats = getattr(pool, "stats", None)
    return _json(stats()) if callable(stats) else {}


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _ga_update(model: Any, ids_list: List[Any], runtime: Any) -> Tuple[List[Any], Any]:
    """Plain accumulation on the streamed model: k forward+backward calls over one window."""
    from soup_cli.utils import l2l_step

    total = l2l_step.label_tokens(ids_list)
    losses = []
    with l2l_step.count_loads(runtime) as counters:
        for ids in ids_list:
            loss = model(input_ids=ids, labels=ids.clone(), num_items_in_batch=total).loss
            loss.backward()
            losses.append(loss.detach())
    return losses, counters


def _l2l_update(stepper: Any, store: Any, ids_list: List[Any]) -> Tuple[List[Any], Any]:
    from soup_cli.utils import l2l_step

    total = l2l_step.label_tokens(ids_list)
    losses = stepper.run(ids_list, store, n_items=total, labels=[i.clone() for i in ids_list])
    return [loss.detach() for loss in losses], stepper.counters


def _hits_of_window(model: Any, runtime: Any, ids_list: List[Any]) -> int:
    """Expert fetches of a no-grad forward over the window: one read per hit expert."""
    import torch

    before = runtime.stats()["expert_fetches"]
    with torch.no_grad():
        for ids in ids_list:
            model(input_ids=ids)
    return int(runtime.stats()["expert_fetches"] - before)


def run_arm(args: argparse.Namespace) -> Dict[str, Any]:
    """Train ``args.updates`` updates of one arm and write its record (and capture)."""
    if args.capture:
        l2l_mistral_stand.set_deterministic_env()
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    import torch
    from peft import LoraConfig, TaskType

    from soup_cli.utils.layer_shard import read_shard_index
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    arm, device = args.arm, args.device
    if args.updates <= args.warmup:
        raise ValueError("need more updates than warm-up updates")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    source, cache = Path(args.source), Path(args.cache)
    source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if source_config.get("model_type") != "deepseek_v3":
        raise ValueError("this launcher accepts a deepseek_v3 checkpoint only")
    dtype, quant = _device_dtype(device)
    if device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("the run needs exactly one CUDA device")
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("the plan needs bf16 without emulation (compute capability 8.0+)")
    levers, off = levers_for(arm, device=device,
                             host_cache_bytes=int(args.host_cache_gib * GIB))
    torch.manual_seed(SEED)
    if args.capture and args.deterministic != "off":
        torch.use_deterministic_algorithms(True, warn_only=args.deterministic == "warn")
    sdpa = (l2l_mistral_stand.FLOOR_SDPA_BACKEND if arm == "ga_floor"
            else l2l_mistral_stand.PINNED_SDPA_BACKEND)

    run_started = time.perf_counter()
    b5_real_moe._reset_peak_rss()
    index = read_shard_index(str(cache))
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
                      target_modules=list(args.targets), task_type=TaskType.CAUSAL_LM)
    with b5_real_moe._meta_construction():
        model, runtime = build_streamed_model(
            model_id=str(source), shard_dir=str(cache), index=index, lora_config=lora,
            device=device, dtype=dtype, buffers=2, pin=False, seed=SEED, quant=quant,
            double_quant=args.double_quant, tier=args.tier, read_ahead=1, **levers)
    try:
        record = _train(args, model, runtime, levers, off, source_config, sdpa, run_started)
    finally:
        runtime.close()
    (out / "record.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n",
                                     encoding="utf-8")
    console.print({"arm": arm, "k": args.k, "median_step_seconds": record["median_step_seconds"],
                   "reads_per_hit": record["reads_per_hit"]})
    return record


def _train(
    args: argparse.Namespace, model: Any, runtime: Any, levers: Dict[str, Any], off: List[str],
    source_config: Dict[str, Any], sdpa: str, run_started: float,
) -> Dict[str, Any]:
    from contextlib import nullcontext

    import torch

    from soup_cli.utils.l2l_activations import ActivationStore
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    arm, k, device = args.arm, args.k, args.device
    mode = ARM_MODE[arm]
    cuda = device == "cuda"
    vocab = int(source_config["vocab_size"])
    out = Path(args.out)
    model.train()
    params = {name.replace(".inner.", "."): p
              for name, p in model.named_parameters() if p.requires_grad}
    params = dict(sorted(params.items()))
    optimizer = torch.optim.AdamW(list(params.values()), lr=args.lr)

    def sync() -> None:
        if cuda:
            torch.cuda.synchronize()

    def windows(update: int) -> List[Any]:
        return [ids.to(device) for ids in window_ids(vocab, args.seq, update, k)]

    pin_ctx = (l2l_mistral_stand.pinned_sdpa(sdpa) if args.capture else nullcontext())
    stepper = store = None
    with pin_ctx:
        if mode == "l2l":
            first = windows(0)[0]
            stepper = L2LStep(model, runtime, capture_layer_call(model, first))
            dtype = torch.bfloat16 if cuda else torch.float32
            store = ActivationStore(
                n_layers=len(stepper.layers), k=k,
                chunk_shape=(1, args.seq, int(stepper.causal.config.hidden_size)),
                dtype=dtype, device=device, pin=cuda)
        hits = None
        if args.hits_probe:
            hits = _hits_of_window(model, runtime, windows(0))
        setup_seconds = time.perf_counter() - run_started
        rss = {"after_setup": b5_real_moe._peak_rss()}
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        sampler = None
        if cuda:
            try:
                from benchmarks.harness import arm_validity
            except ImportError:  # run as a script from benchmarks/harness
                import arm_validity  # type: ignore[no-redef]
            sampler = arm_validity.Sampler("gpu_clock", arm_validity.gpu_clock, 1.0)
            sampler.start()
        rows: List[Dict[str, Any]] = []
        captured: List[Dict[str, Any]] = []
        cache_bytes: Dict[str, Any] = {}
        try:
            for update in range(args.updates):
                ids_list = windows(update)
                optimizer.zero_grad(set_to_none=True)
                before = runtime.stats()
                sync()
                started = time.perf_counter()
                if mode == "ga":
                    losses, counters = _ga_update(model, ids_list, runtime)
                else:
                    losses, counters = _l2l_update(stepper, store, ids_list)
                sync()
                step_seconds = time.perf_counter() - started
                after = runtime.stats()
                if args.capture:
                    captured.append({
                        "loss": float(sum(float(loss) for loss in losses)),
                        "grads": {n: p.grad.detach().float().cpu().clone()
                                  for n, p in params.items() if p.grad is not None}})
                torch.nn.utils.clip_grad_norm_(list(params.values()), CLIP_NORM)
                optimizer.step()
                sync()
                rows.append({
                    "update": update, "step_seconds": step_seconds,
                    "update_seconds": time.perf_counter() - started,
                    "loss": float(sum(float(loss) for loss in losses)),
                    "layer_loads": int(counters.layer_loads),
                    "expert_fetches": int(after["expert_fetches"] - before["expert_fetches"]),
                    "expert_bytes_read": int(after["expert_bytes_read"]
                                             - before["expert_bytes_read"]),
                    "peak_rss_bytes": b5_real_moe._peak_rss()})
                console.print(rows[-1])
                if update == args.warmup - 1:
                    cache_bytes["after_warmup"] = _expert_pool_stats(runtime).get("host_cache")
                    rss["after_warmup"] = b5_real_moe._peak_rss()
        finally:
            if sampler is not None:
                sampler.stop()
        cache_bytes["after_last"] = _expert_pool_stats(runtime).get("host_cache")
        rss["after_last"] = b5_real_moe._peak_rss()
    measured = rows[args.warmup:]
    fetches0 = rows[0]["expert_fetches"]
    record: Dict[str, Any] = {
        "arm": arm, "mode": mode, "k": k, "kind": "synthetic", "seed": SEED, "device": device,
        "updates": args.updates, "warmup": args.warmup, "seq": args.seq,
        "levers_requested": sorted(levers), "levers_applied": _json(levers),
        "levers_off_on_this_device": off, "sdpa_backend": sdpa if args.capture else "default",
        "deterministic": args.deterministic if args.capture else "off",
        "median_step_seconds": statistics.median(row["step_seconds"] for row in measured),
        "measured_step_seconds": [row["step_seconds"] for row in measured],
        "setup_seconds": setup_seconds, "run_seconds": time.perf_counter() - run_started,
        "rows": rows, "hits_window0": hits,
        "reads_per_hit": (fetches0 / hits) if hits else None,
        "expert_fetches_window0": fetches0,
        "expert_fetches_per_window": [row["expert_fetches"] for row in rows],
        "layer_loads_per_window": [row["layer_loads"] for row in rows],
        "tokens_per_second": args.seq * k / statistics.median(
            row["step_seconds"] for row in measured),
        "vram_peak_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None,
        "vram_peak_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else None,
        "rss_bytes": rss, "host_cache_bytes": cache_bytes,
        "gpu_clock": (b5_real_moe._clock_summary(sampler.samples) if sampler else None),
        "model_revision": args.revision, "repo_commit": _repo_commit(),
        "versions": {name: b5_real_moe._version(name) for name in
                     ("torch", "transformers", "peft", "bitsandbytes", "safetensors", "soup-cli")},
        "python": platform.python_version(), "ram_limit": b5_real_moe._ram_limit(),
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "targets": list(args.targets)},
        "tier": args.tier, "double_quant": args.double_quant,
        "gpu": torch.cuda.get_device_name(0) if cuda else "cpu",
    }
    if args.capture:
        torch.save({
            "arm": arm, "mode": mode, "k": k, "device": device, "lr": args.lr,
            "losses": [c["loss"] for c in captured], "grads": [c["grads"] for c in captured],
            "params": {n: p.detach().float().cpu().clone() for n, p in params.items()},
            "adam": {f"{n}::{key}": value.detach().float().cpu().clone()
                     for n, p in params.items() for key, value in optimizer.state[p].items()},
        }, out / "capture.pt")
    del optimizer, params
    gc.collect()
    return record


# --------------------------------------------------------------------------
# judge
# --------------------------------------------------------------------------
def _as_run(capture: Dict[str, Any]) -> Any:
    rates = [capture["lr"]] * len(capture["grads"])
    logs = [{"loss": loss, "learning_rate": lr} for loss, lr in zip(capture["losses"], rates)]
    return l2l_mistral_stand.RunRecord(logs=logs, rates=rates, grads=capture["grads"],
                                       steps=len(capture["grads"]))


def _bit_equal(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    import torch

    def same(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        return a.keys() == b.keys() and all(torch.equal(a[n], b[n]) for n in a)

    return (left["losses"] == right["losses"]
            and len(left["grads"]) == len(right["grads"])
            and all(same(a, b) for a, b in zip(left["grads"], right["grads"]))
            and same(left["params"], right["params"]) and same(left["adam"], right["adam"]))


def judge_pair(
    ref: Any, cand: Any, *, mode: str, floor_grad: Optional[List[float]],
    floor_loss: Optional[List[float]],
) -> Dict[str, Any]:
    """Tolerance A3 of ``cand`` against ``ref`` at every update.

    ``fp32``: relative error at most 1e-4. ``bf16``: at most 1.5 times the noise floor, which is
    the error between GA on MATH and GA on EFFICIENT_ATTENTION (``floor_*``, both measured
    against ``ga_ordinary``). Without a floor a bf16 pair is not judged.
    """
    stand = l2l_mistral_stand
    grad, loss = stand.rel_l2_by_step(ref, cand), stand.loss_rel_by_step(ref, cand)
    out: Dict[str, Any] = {"mode": mode, "grad_rel": grad, "loss_rel": loss, "judged": True,
                           "failing": [], "pass": False, "floor_grad": floor_grad,
                           "floor_loss": floor_loss}
    if mode == "fp32":
        grad_limit = [stand.FP32_REL_LIMIT] * len(grad)
        loss_limit = [stand.FP32_REL_LIMIT] * len(loss)
    else:
        if floor_grad is None or floor_loss is None:
            out["judged"] = False
            return out
        if not all(math.isfinite(e) for e in floor_grad + floor_loss):
            out["judged"] = False
            out["floor_problem"] = "the floor errors are not finite"
            return out
        grad_limit = [stand.FLOOR_FACTOR * e for e in floor_grad]
        loss_limit = [stand.FLOOR_FACTOR * e for e in floor_loss]
    out["grad_limit"], out["loss_limit"] = grad_limit, loss_limit
    for what, errors, limits in (("gradient", grad, grad_limit), ("loss", loss, loss_limit)):
        if len(errors) != len(limits):
            out["failing"].append({"what": what, "step": None, "error": None, "limit": None})
        for step, (err, lim) in enumerate(zip(errors, limits)):
            if not err <= lim:
                out["failing"].append({"what": what, "step": step, "error": err, "limit": lim})
    out["pass"] = not out["failing"]
    return out


def judge(args: argparse.Namespace) -> Dict[str, Any]:
    """Accuracy verdict for the arms of one ``k`` under ``--runs/<arm>-k<k>/capture.pt``."""
    import torch

    root = Path(args.runs)
    captures: Dict[str, Dict[str, Any]] = {}
    for arm in ARM_NAMES:
        path = root / f"{arm}-k{args.k}" / "capture.pt"
        if path.is_file():
            captures[arm] = torch.load(path, map_location="cpu", weights_only=False)
    device = next(iter(captures.values()))["device"] if captures else "cpu"
    mode = "bf16" if device == "cuda" else "fp32"
    runs = {arm: _as_run(c) for arm, c in captures.items()}
    floor_grad = floor_loss = None
    if "ga_floor" in runs and "ga_ordinary" in runs:
        floor_grad = l2l_mistral_stand.rel_l2_by_step(runs["ga_ordinary"], runs["ga_floor"])
        floor_loss = l2l_mistral_stand.loss_rel_by_step(runs["ga_ordinary"], runs["ga_floor"])
    pairs: Dict[str, Any] = {}
    pairs_to_judge = PAIRS
    if args.two_read:
        pairs_to_judge = tuple(pair for pair in PAIRS if pair[1] != "ga_complete")
        pairs_to_judge += (("l2l_complete", "l2l_two_read"),
                           ("l2l_ordinary", "l2l_two_read"))
    for ref_arm, cand_arm in pairs_to_judge:
        name = f"{cand_arm}_vs_{ref_arm}"
        if ref_arm not in runs or cand_arm not in runs:
            pairs[name] = {"judged": False, "pass": False, "missing": [
                a for a in (ref_arm, cand_arm) if a not in runs]}
            continue
        result = judge_pair(runs[ref_arm], runs[cand_arm], mode=mode, floor_grad=floor_grad,
                            floor_loss=floor_loss)
        result["bit_equal"] = _bit_equal(captures[ref_arm], captures[cand_arm])
        pairs[name] = result
    verdict = {"k": args.k, "device": device, "mode": mode, "arms_found": sorted(captures),
               "pairs": pairs, "all_pass": all(p["pass"] for p in pairs.values())}
    out = root / f"accuracy-k{args.k}.json"
    out.write_text(json.dumps(verdict, indent=2, allow_nan=False, default=str) + "\n",
                   encoding="utf-8")
    console.print({"accuracy": str(out), "all_pass": verdict["all_pass"]})
    return verdict


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--source", required=True, help="the checkpoint folder")
        cmd.add_argument("--cache", required=True, help="the per_expert cache folder")
        cmd.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        cmd.add_argument("--double-quant", action=argparse.BooleanOptionalAction, default=True)
    run = sub.choices["run"]
    run.add_argument("--arm", choices=ARM_NAMES, required=True)
    run.add_argument("--k", type=int, required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--updates", type=int, default=8)
    run.add_argument("--warmup", type=int, default=2)
    run.add_argument("--seq", type=int, default=512)
    run.add_argument("--lr", type=float, default=1e-4)
    run.add_argument("--lora-r", type=int, default=4)
    run.add_argument("--lora-alpha", type=int, default=8)
    run.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    run.add_argument("--tier", choices=("ram", "disk"), default="disk")
    run.add_argument("--host-cache-gib", type=float, default=8.0,
                     help="expert_host_cache_bytes of the complete arms on a card (0 = off)")
    run.add_argument("--hits-probe", action="store_true")
    run.add_argument("--capture", action="store_true")
    run.add_argument("--deterministic", choices=("strict", "warn", "off"), default="strict")
    run.add_argument("--revision", help="model revision, written into the record")
    judge_cmd = sub.add_parser("judge")
    judge_cmd.add_argument("--runs", required=True)
    judge_cmd.add_argument("--k", type=int, required=True)
    judge_cmd.add_argument("--two-read", action="store_true",
                           help="also require B versus A and ordinary L2L accuracy")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        if args.k < 1:
            raise SystemExit("--k must be at least 1")
        run_arm(args)
    else:
        return 0 if judge(args)["all_pass"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
