#!/usr/bin/env python3
"""G4 probe: compressed L2L boundary activations on the SHIPPED streamed runtime.

The rule this runs against is in benchmarks/probe-g4-l2l-activation-compression.md,
fixed before the GPU run; to be committed before the first GPU arm. One mode per process:

  --mode error   One process, deterministic. For each of ``--error-sets`` sets of k
                 real micro-batches: gradient accumulation, exact L2L through the
                 plain ``ActivationStore``, the same through a recording store that
                 measures every codec at every boundary, the identity codec (null
                 control) and each lossy codec (int8, fp8_e4m3, int4). Records the
                 restore error per layer, the one-recompute propagation per layer,
                 LoRA gradient agreement against exact L2L, per-micro-batch loss
                 equality, and a per-chunk compress/restore/copy micro-benchmark.
  --mode train   One training arm: L2L at k micro-batches per step over the packed
                 Dolly train split, one codec, one seed; the loss curve, checkpoint
                 panel losses, the held-out task loss at the end, and the adapter
                 saved next to the JSON (``<stem>-adapter``, hashed in the JSON).
  --mode timing  One timing arm (fresh process per run, driven by g4_timing.py):
                 warm-up and timed steps over fixed rows, then two steps with the
                 codec's CUDA-event timers on.
  --mode suites  Soup's four mini gate suites (greedy, ``--max-new-tokens``) on one
                 arm's saved adapter over the resident base, loaded NF4 as trained.
                 The streamed model has no KV cache, so generation runs resident;
                 the adapter is the trained one, byte for byte. Before scoring, the
                 base-parity check (validity row V13): the live adapter tensors must
                 equal the saved file, and the resident model's response NLL on the
                 first ``--parity-rows`` panel examples is compared with the NLL the
                 streamed model recorded for the same adapter at the end of training.

``--device cpu`` exists for the CPU pilot only: every record it writes is labelled
SYNTHETIC/CPU PILOT and g4_rule.py refuses it.

Benchmark harness code, not shipped. torch is imported inside functions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import g4_activation_codec as codecs  # noqa: E402
import g4_data  # noqa: E402
import l2l_activations  # noqa: E402
import l2l_schedule  # noqa: E402
import stream_probe  # noqa: E402

CPU_LABEL = "SYNTHETIC/CPU PILOT"
ERROR_CODECS = (codecs.INT8, codecs.FP8, codecs.COARSE)
TRAIN_CODECS = (codecs.IDENTITY,) + codecs.CANDIDATES
HARNESS = Path(__file__).resolve().parent
ROOT = HARNESS.parents[1]
PACKAGES = ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "safetensors",
            "numpy", "psutil")
#: Panel examples the suites mode re-scores on the resident base for V13 (base parity).
PARITY_ROWS = 16
#: Exit code for "CUDA requested but absent": nonzero, so g4-run.sh never marks a step
#: done without its JSON.
NO_CUDA_EXIT = 3


# ==========================================================================
# pure helpers (CPU-tested)
# ==========================================================================
def lr_at(step: int, *, total: int, warmup: int, peak: float) -> float:
    """HF ``get_cosine_schedule_with_warmup`` at 0-based ``step``: linear warm-up from 0,
    then a half cosine to 0. As in HF, the first step of a warm-up runs at lr 0."""
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return peak * max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def warmup_steps(total: int, ratio: float) -> int:
    return int(math.ceil(total * ratio))


def task_loss(per_example: Sequence[Dict[str, Any]]) -> Optional[float]:
    """Token-weighted NLL: sum of NLL over sum of scored tokens."""
    tokens = sum(int(ex["tokens"]) for ex in per_example)
    if not tokens:
        return None
    return sum(float(ex["nll"]) for ex in per_example) / tokens


def summarize_times(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"n": 0, "median_s": None, "mean_s": None, "min_s": None, "max_s": None}
    return {"n": len(values), "median_s": statistics.median(values),
            "mean_s": statistics.fmean(values), "min_s": min(values), "max_s": max(values)}


def adapter_dir_for(out: str, adapter_out: Optional[str] = None) -> Path:
    """Where a training arm saves its adapter: ``adapter_out`` if given, else next to
    its JSON as ``<stem>-adapter``."""
    if adapter_out:
        return Path(adapter_out)
    path = Path(out)
    return path.with_name(f"{path.stem}-adapter")


def recorded_adapter_dir(arm: Path, record: Dict[str, Any]) -> Path:
    """The adapter directory a training record names, relative to the record."""
    return (arm.parent / record["adapter"]["dir"]).resolve()


def tree_sha256(root: Path) -> Dict[str, Optional[str]]:
    return {path.relative_to(root).as_posix(): _sha256(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


# ==========================================================================
# arguments
# ==========================================================================
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=("error", "train", "timing", "suites"), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--weights", required=True, help="model directory or Hub id")
    parser.add_argument("--tokenizer", default=None, help="default: --weights")
    parser.add_argument("--model-revision", default="", help="recorded, not resolved")
    parser.add_argument("--shards", default=None)
    parser.add_argument("--quant", choices=("none", "nf4"), default="nf4")
    parser.add_argument("--tier", choices=("ram", "disk"), default="ram")
    parser.add_argument("--buffers", type=int, default=2)
    parser.add_argument("--read-ahead", type=int, default=2)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--seed", type=int, default=1, help="LoRA initialisation seed")
    parser.add_argument("--dolly", default=None, help=f"local {g4_data.FILENAME}")
    parser.add_argument("--codec", choices=TRAIN_CODECS, default=codecs.IDENTITY)
    parser.add_argument("--error-sets", type=int, default=2)
    parser.add_argument("--bench-reps", type=int, default=50)
    parser.add_argument("--steps", type=int, default=0,
                        help="train: 0 = every packed row once; timing: default 8")
    parser.add_argument("--warmup", type=int, default=2, help="timing: untimed steps")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--curve-panel", type=int, default=128)
    parser.add_argument("--panel-limit", type=int, default=0, help="0 = the whole panel")
    parser.add_argument("--panel-max-length", type=int, default=g4_data.PANEL_MAX_LENGTH)
    parser.add_argument("--arm", default=None, help="suites: the training arm's JSON")
    parser.add_argument("--adapter-out", default=None,
                        help="train: adapter directory (default: <out stem>-adapter)")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="suites: greedy budget")
    parser.add_argument("--parity-rows", type=int, default=PARITY_ROWS,
                        help="suites: panel examples for the resident-vs-streamed check (V13)")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None,
                        help="default: on for error and train, off for timing")
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)
    if args.deterministic is None:
        args.deterministic = args.mode != "timing"
    if args.mode == "timing" and not args.steps:
        args.steps = 8
    if args.device == "cpu":
        args.label = f"{CPU_LABEL} {args.label}".strip()
    if args.k < 1 or args.seq < 2:
        parser.error("--k must be >= 1 and --seq >= 2")
    if args.mode == "suites" and not args.arm:
        parser.error("--mode suites needs --arm <train-*.json>")
    if args.parity_rows < 1:
        parser.error("--parity-rows must be >= 1")
    return args


# ==========================================================================
# provenance
# ==========================================================================
def _sha256(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


#: The shipped modules the step actually executes; the harness's l2l_activations.py and
#: l2l_schedule.py only re-export them, so their own hashes do not cover the running code.
EXECUTED_SOURCES = ("l2l_step.py", "l2l_rng.py", "l2l_activations.py")


def harness_hashes() -> Dict[str, Optional[str]]:
    """SHA-256 of the harness files and of the shipped L2L modules they run.

    The shipped modules are hashed where the imported ``soup_cli`` package really is (the
    record's ``soup_cli_file``), keyed by their package path (``soup_cli/utils/l2l_step.py``)
    so the two ``l2l_activations.py`` never collide. Records written before 2026-10-25 (the
    GPU run at b195da29) hold the harness names only."""
    import soup_cli

    names = sorted(HARNESS.glob("g4_*.py")) + [
        HARNESS / name for name in ("l2l_activations.py", "l2l_schedule.py", "stream_probe.py")
    ]
    out = {path.name: _sha256(path) for path in names}
    utils = Path(soup_cli.__file__).resolve().parent / "utils"
    out.update({f"soup_cli/utils/{name}": _sha256(utils / name) for name in EXECUTED_SOURCES})
    return out


def versions() -> Dict[str, str]:
    import importlib.metadata

    out = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def git_head() -> Optional[str]:
    import subprocess

    try:
        result = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def config_record(args: argparse.Namespace, *, dtype: str, steps: int) -> Dict[str, Any]:
    """Everything a training arm's result depends on. Arms may differ in codec and seed only."""
    return {
        "codec": args.codec, "seed": args.seed, "k": args.k, "seq": args.seq,
        "lora_r": args.lora_r, "lora_alpha": 2 * args.lora_r, "lora_dropout": 0.0,
        "lora_targets": args.lora_targets, "optimizer": "torch.optim.AdamW",
        "lr": args.lr, "schedule": "linear warm-up + cosine (HF semantics)",
        "warmup_ratio": args.warmup_ratio, "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm, "steps": steps, "quant": args.quant,
        "tier": args.tier, "dtype": dtype, "deterministic": args.deterministic,
        "eval_every": args.eval_every, "curve_panel": args.curve_panel,
        "panel_limit": args.panel_limit, "panel_max_length": args.panel_max_length,
        "model_revision": args.model_revision, "device": args.device,
    }


# ==========================================================================
# build
# ==========================================================================
class Context:
    """The streamed model and everything a mode needs from the build."""

    def __init__(self, args: argparse.Namespace):
        import torch

        from soup_cli.utils.layer_stream import resolve_stream_dtype

        if args.deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
        self.args = args
        self.device = args.device
        self.is_cuda = args.device == "cuda"
        self.dtype = resolve_stream_dtype(args.device)
        self.torch_dtype = getattr(torch, self.dtype)
        build_args = argparse.Namespace(
            weights=args.weights, shards=args.shards, quant=args.quant, tier=args.tier,
            buffers=args.buffers, read_ahead=args.read_ahead, lora_r=args.lora_r,
            lora_targets=args.lora_targets, seed=args.seed, lazy_shard_handles=False,
            control_sync_source=False, no_pin=not self.is_cuda,
        )
        import soup_cli

        self.facts = stream_probe.gpu_facts(args.device) if self.is_cuda else {
            "cpu": platform.processor() or platform.machine(), "torch": torch.__version__,
            "soup_cli_file": soup_cli.__file__,
        }
        (self.model, self.runtime, self.config, _index, self.weights_dir, self.shard_dir,
         self.shard_s, self.build_s) = stream_probe.build(build_args, args.device, self.dtype)
        self.n_layers = int(self.config.num_hidden_layers)
        self.hidden = int(self.config.hidden_size)
        self.chunk_shape = (1, int(args.seq), self.hidden)

    def tokenizer(self) -> Any:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.args.tokenizer or self.weights_dir)

    def meta(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        weights = Path(self.weights_dir)
        meta = {
            **self.facts, "label": self.args.label, "mode": self.args.mode,
            "device": self.device, "verdict_eligible": self.is_cuda,
            "args": vars(self.args), "dtype": self.dtype, "arch": str(self.config.model_type),
            "n_layers": self.n_layers, "hidden": self.hidden,
            "chunk_shape": list(self.chunk_shape), "weights_dir": str(weights),
            "shard_dir": str(self.shard_dir), "shard_seconds": self.shard_s,
            "build_seconds": self.build_s, "versions": versions(), "git_head": git_head(),
            "harness_sha256": harness_hashes(),
            "model_files_sha256": {name: _sha256(weights / name) for name in (
                "config.json", "tokenizer.json", "tokenizer_config.json")},
            "platform": platform.platform(), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        meta.update(extra or {})
        return meta

    def sync(self) -> None:
        if self.is_cuda:
            import torch

            torch.cuda.synchronize()

    def store(self, codec: str, k: int) -> Any:
        return codecs.CodecActivationStore(
            codec=codec, n_layers=self.n_layers, k=k, chunk_shape=self.chunk_shape,
            dtype=self.torch_dtype, device=self.device, pin=self.is_cuda,
        )

    def plain_store(self, k: int) -> Any:
        return l2l_activations.ActivationStore(
            n_layers=self.n_layers, k=k, chunk_shape=self.chunk_shape,
            dtype=self.torch_dtype, device=self.device, pin=self.is_cuda,
        )

    def ids(self, rows: Sequence[Sequence[int]]) -> List[Any]:
        import torch

        return [torch.tensor([row], dtype=torch.long, device=self.device) for row in rows]


def memory_record(ctx: Context, store: Any = None) -> Dict[str, Any]:
    import torch

    out: Dict[str, Any] = {"store": store.host_bytes() if store is not None else None}
    try:
        import psutil

        out["rss_bytes"] = psutil.Process().memory_info().rss
        out["ram_available_bytes"] = psutil.virtual_memory().available
        out["ram_total_bytes"] = psutil.virtual_memory().total
    except ImportError:
        out["rss_bytes"] = None
    if ctx.is_cuda:
        out["peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 1e9
        out["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
        stats = torch.cuda.host_memory_stats()
        out["host_allocator"] = {key: stats[key] for key in stats
                                 if key.startswith(("allocated_bytes", "reserved_bytes"))}
    return out


def load_data(ctx: Context, *, rows_needed: Optional[int], panel: bool) -> Dict[str, Any]:
    args = ctx.args
    data = g4_data.load_splits(args.dolly)
    tokenizer = ctx.tokenizer()
    rows = g4_data.pack_train(tokenizer, data["splits"]["train"], seq=args.seq,
                              limit=rows_needed)
    out: Dict[str, Any] = {"provenance": data["provenance"], "rows": rows}
    if panel:
        examples = g4_data.encode_panel(tokenizer, data["splits"]["heldout"],
                                        max_length=args.panel_max_length)
        if args.panel_limit:
            examples = examples[:args.panel_limit]
        out["panel"] = examples
        out["provenance"]["panel_ids_sha256"] = g4_data.ids_digest([ex["ids"] for ex in examples])
        out["provenance"]["panel_examples"] = len(examples)
        out["provenance"]["panel_response_tokens"] = sum(ex["response_tokens"] for ex in examples)
    return out


# ==========================================================================
# error mode
# ==========================================================================
def call_layer(step: Any, idx: int, hidden: Any) -> Any:
    out = step.layers[idx](hidden, *step.call.args, **step.call.kwargs)
    return out[0] if isinstance(out, tuple) else out


class RecordingStore:
    """A plain ``ActivationStore`` that, on every ``put``, measures each codec at that
    boundary: the restore error, and the layer's output from the restored input
    against its output from the exact input (one recompute). The layer is the one the
    forward is about to call, so the extra calls load nothing (see l2l_schedule)."""

    def __init__(self, inner: Any, step: Any, names: Sequence[str],
                 parts: Dict[str, Dict[str, List[List[Dict[str, float]]]]]):
        self._inner = inner
        self._step = step
        self._names = tuple(names)
        self.parts = parts
        self.k, self.n_layers, self.chunk_bytes = inner.k, inner.n_layers, inner.chunk_bytes
        self.stats = inner.stats
        self.sample: Optional[Any] = None

    def put(self, layer: int, mb: int, tensor: Any) -> None:
        import torch

        self._inner.put(layer, mb, tensor)
        if layer == self.n_layers // 2 and mb == 0 and self.sample is None:
            self.sample = tensor.detach().clone()
        with torch.no_grad():
            out = call_layer(self._step, layer, tensor)
            for name in self._names:
                x_hat = codecs.round_trip(tensor, name)
                out_hat = call_layer(self._step, layer, x_hat)
                metrics = self.parts[name]
                metrics["restore"][layer].append(codecs.restore_error(tensor, x_hat))
                metrics["output"][layer].append(codecs.restore_error(out, out_hat))
                metrics["branch"][layer].append(codecs.restore_error(
                    out.double() - tensor.double(), out_hat.double() - x_hat.double()))

    def get(self, layer: int, mb: int) -> Any:
        return self._inner.get(layer, mb)

    def begin_backward(self) -> None:
        self._inner.begin_backward()

    def end_step(self) -> None:
        self._inner.end_step()

    def close(self) -> None:
        self._inner.close()


def _timed(ctx: Context, body: Callable[[], Any]) -> Tuple[Any, float]:
    import torch

    if ctx.is_cuda:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = body()
        end.record()
        end.synchronize()
        return out, start.elapsed_time(end) / 1000.0
    started = time.perf_counter()
    out = body()
    return out, time.perf_counter() - started


def microbench(ctx: Context, x: Any, reps: int) -> Dict[str, Any]:
    """Per-chunk compress, restore and device-to-host copy time of one real chunk."""
    import torch

    results: Dict[str, Any] = {"chunk_shape": list(x.shape), "reps": reps,
                               "timer": "cuda-event" if ctx.is_cuda else "perf_counter"}
    for name in (codecs.IDENTITY,) + ERROR_CODECS:
        payload, scale = codecs.encode(x, name)
        host = torch.empty(payload.shape, dtype=payload.dtype, pin_memory=ctx.is_cuda)
        compress, restore, copy = [], [], []
        for rep in range(reps + 5):
            (p, s), c_s = _timed(ctx, lambda: codecs.store_compress(x, name)[:2]
                                 if name != codecs.IDENTITY else codecs.encode(x, name))
            _, r_s = _timed(ctx, lambda: codecs.decode(p, s, name, x.dtype, int(x.shape[-1])))
            _, d_s = _timed(ctx, lambda: host.copy_(payload, non_blocking=True))
            if rep >= 5:
                compress.append(c_s)
                restore.append(r_s)
                copy.append(d_s)
        results[name] = {
            "payload_bytes": int(payload.numel() * payload.element_size()),
            "scale_bytes": int(scale.numel() * 4) if scale is not None else 0,
            "compress": summarize_times(compress if name != codecs.IDENTITY else []),
            "restore": summarize_times(restore if name != codecs.IDENTITY else []),
            "d2h_copy": summarize_times(copy),
        }
    return results


def _empty_parts(names: Sequence[str], n_layers: int) -> Dict[str, Any]:
    return {name: {kind: [[] for _ in range(n_layers)] for kind in ("restore", "output", "branch")}
            for name in names}


def _layer_table(parts: Dict[str, Any], n_layers: int) -> Dict[str, List[Dict[str, Any]]]:
    table: Dict[str, List[Dict[str, Any]]] = {}
    for name, kinds in parts.items():
        rows = []
        for layer in range(n_layers):
            row: Dict[str, Any] = {"layer": layer}
            for kind in ("restore", "output", "branch"):
                row[kind] = codecs.merge_errors(kinds[kind][layer])
            restore = row["restore"]["rms_row_rel"]
            row["branch_amplification"] = (
                row["branch"]["rms_row_rel"] / restore if restore > 0 else None)
            rows.append(row)
        table[name] = rows
    return table


def mode_error(ctx: Context, sink: Any) -> None:
    args = ctx.args
    model, runtime = ctx.model, ctx.runtime
    data = load_data(ctx, rows_needed=args.error_sets * args.k, panel=False)
    rows = data["rows"]
    if len(rows) < args.error_sets * args.k:
        raise RuntimeError(f"only {len(rows)} packed rows; need {args.error_sets * args.k}")
    sink.payload["meta"]["data"] = {**data["provenance"], "error_rows_sha256":
                                    g4_data.ids_digest(rows)}
    # The setup the codecs are measured in; g4_rule V14 ties it to the training arms.
    sink.payload["meta"]["config"] = config_record(args, dtype=ctx.dtype, steps=0)
    l2l_schedule.make_non_vacuous(model)
    parts = _empty_parts(ERROR_CODECS, ctx.n_layers)
    exact_by_set: List[Dict[str, Any]] = []
    sample = None
    for index in range(args.error_sets):
        mbs = ctx.ids(rows[index * args.k:(index + 1) * args.k])
        step = l2l_schedule.L2LStep(model, runtime, l2l_schedule.capture_layer_call(model, mbs[0]))

        def run(store: Any) -> Tuple[List[Any], Dict[str, Any]]:
            l2l_schedule.zero_grads(model)
            try:
                losses = step.run(mbs, store)
            finally:
                store.close()
            return losses, l2l_schedule.collect_grads(model)

        l2l_schedule.zero_grads(model)
        ga_losses = l2l_schedule.ga_step(model, mbs)
        ga = l2l_schedule.collect_grads(model)
        plain_losses, plain = run(ctx.plain_store(args.k))
        recorder = RecordingStore(ctx.plain_store(args.k), step, ERROR_CODECS, parts)
        rec_losses, recorded = run(recorder)
        if sample is None:
            sample = recorder.sample
        ident_losses, ident = run(ctx.store(codecs.IDENTITY, args.k))
        record: Dict[str, Any] = {
            "set": index,
            "ga_vs_l2l": l2l_schedule.compare(ga, plain),
            "ga_vs_l2l_losses_equal": l2l_schedule.losses_equal(ga_losses, plain_losses),
            "a_vs_a": l2l_schedule.compare(plain, recorded),
            "a_vs_a_losses_equal": l2l_schedule.losses_equal(plain_losses, rec_losses),
            "null_control": l2l_schedule.compare(plain, ident),
            "null_control_losses_equal": l2l_schedule.losses_equal(plain_losses, ident_losses),
            "losses_exact": [float(x) for x in plain_losses],
            "codecs": {},
        }
        for name in ERROR_CODECS:
            losses, grads = run(ctx.store(name, args.k))
            agreement = codecs.grad_agreement(plain, grads)
            record["codecs"][name] = {
                "losses": [float(x) for x in losses],
                "losses_equal": l2l_schedule.losses_equal(plain_losses, losses),
                "grads_equal": l2l_schedule.compare(plain, grads)["equal"],
                "grad": agreement,
            }
            print(f"set {index} {name:<9} grad global rel L2 {agreement['global_rel_l2']:.4g}  "
                  f"min cos {agreement['min_cosine']:.6f}  losses equal "
                  f"{record['codecs'][name]['losses_equal']}")
        print(f"set {index} GA==L2L {record['ga_vs_l2l']['equal']}  A==A "
              f"{record['a_vs_a']['equal']}  identity==plain {record['null_control']['equal']}")
        exact_by_set.append(plain)
        sink.add("error_set", record)
    if len(exact_by_set) > 1:
        between = codecs.grad_agreement(exact_by_set[0], exact_by_set[1])
        between.pop("per_tensor")
        sink.add("between_sets", {"grad": between})
    table = _layer_table(parts, ctx.n_layers)
    sink.add("layers", {"codecs": table})
    for name, rows_ in table.items():
        worst = max(rows_, key=lambda row: row["restore"]["rms_row_rel"])
        branch = max(rows_, key=lambda row: row["branch"]["rms_row_rel"])
        print(f"{name:<9} restore rms row rel max {worst['restore']['rms_row_rel']:.4g} "
              f"(layer {worst['layer']})  branch rms row rel max "
              f"{branch['branch']['rms_row_rel']:.4g} (layer {branch['layer']})")
    sink.add("microbench", microbench(ctx, sample, args.bench_reps))
    sink.add("memory", memory_record(ctx))


# ==========================================================================
# train mode
# ==========================================================================
def panel_nll(ctx: Context, examples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-example response NLL through the shipped forward, batch 1, no padding."""
    import torch
    from torch.nn import functional

    out = []
    with torch.no_grad():
        for ex in examples:
            start = max(int(ex["response_from"]), 1)
            if start >= len(ex["ids"]):
                out.append({"row": ex["row"], "nll": 0.0, "tokens": 0})
                continue
            ids = torch.tensor([ex["ids"]], dtype=torch.long, device=ctx.device)
            logits = ctx.model(input_ids=ids).logits
            nll = functional.cross_entropy(
                logits[0, start - 1:-1].float(), ids[0, start:], reduction="sum")
            out.append({"row": ex["row"], "nll": float(nll), "tokens": int(ids.shape[1] - start)})
    return out


def mode_train(ctx: Context, sink: Any) -> None:
    import torch

    args = ctx.args
    model, runtime = ctx.model, ctx.runtime
    rows_needed = args.steps * args.k if args.steps else None
    data = load_data(ctx, rows_needed=rows_needed, panel=True)
    steps = args.steps or len(data["rows"]) // args.k
    if steps < 1 or steps * args.k > len(data["rows"]):
        raise RuntimeError(f"{len(data['rows'])} packed rows cannot fill {steps} steps")
    rows = data["rows"][:steps * args.k]
    panel = data["panel"]
    meta = sink.payload["meta"]
    meta["data"] = {**data["provenance"], "train_rows": len(rows),
                    "train_ids_sha256": g4_data.ids_digest(rows)}
    meta["config"] = config_record(args, dtype=ctx.dtype, steps=steps)
    ids = ctx.ids(rows)
    step = l2l_schedule.L2LStep(model, runtime, l2l_schedule.capture_layer_call(model, ids[0]))
    store = ctx.store(args.codec, args.k)
    params = [param for _name, param in sorted(l2l_schedule.lora_parameters(model).items())]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    warm = warmup_steps(steps, args.warmup_ratio)
    log: List[Dict[str, Any]] = []
    checkpoints: List[Dict[str, Any]] = []
    curve = panel[:args.curve_panel]
    started = time.perf_counter()
    if ctx.is_cuda:
        torch.cuda.reset_peak_memory_stats()
    try:
        for index in range(steps):
            lr = lr_at(index, total=steps, warmup=warm, peak=args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            before = (store.codec_stats.compress_s, store.codec_stats.restore_s)
            ctx.sync()
            t0 = time.perf_counter()
            try:
                losses = step.run(ids[index * args.k:(index + 1) * args.k], store)
            except codecs.NonFiniteActivationError as exc:
                # The step is refused before the optimizer sees it; the arm stops and
                # V7 reads "stopped", like a non-finite loss.
                sink.payload["stopped"] = f"step {index}: {exc}"
                print(f"stopped at step {index}: {exc}")
                break
            norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ctx.sync()
            wall = time.perf_counter() - t0
            store.flush_timing()
            entry = {
                "step": index, "loss": float(sum(float(x) for x in losses)),
                "grad_norm": float(norm), "lr": lr, "step_s": wall,
                "compress_s": store.codec_stats.compress_s - before[0],
                "restore_s": store.codec_stats.restore_s - before[1],
            }
            if index == 0:
                entry["mb_losses"] = [float(x) for x in losses]
            log.append(entry)
            if index % 10 == 0 or index == steps - 1:
                print(f"step {index:>4}/{steps} loss {entry['loss']:.5f} "
                      f"gnorm {entry['grad_norm']:.4f} lr {lr:.3e} {wall:.2f} s")
            if args.eval_every and (index + 1) % args.eval_every == 0 and index + 1 < steps:
                per = panel_nll(ctx, curve)
                checkpoints.append({"step": index + 1, "examples": len(per),
                                    "task_loss": task_loss(per)})
                print(f"checkpoint {index + 1}: curve-panel loss {checkpoints[-1]['task_loss']}")
                sink.payload["checkpoints"] = checkpoints
            if not (math.isfinite(entry["loss"]) and math.isfinite(entry["grad_norm"])):
                sink.payload["stopped"] = f"non-finite loss or grad norm at step {index}"
                break
            sink.payload["log"] = log
            if index % 25 == 0:
                sink.flush()
        train_s = time.perf_counter() - started
        memory = memory_record(ctx, store)
        codec_stats = {key: value for key, value in vars(store.codec_stats).items()
                       if not key.endswith("_chunk_s")}
    finally:
        store.close()
    t0 = time.perf_counter()
    per = panel_nll(ctx, panel)
    adapter_dir = adapter_dir_for(args.out, args.adapter_out)
    ctx.model.save_pretrained(str(adapter_dir))
    adapter_rel = os.path.relpath(adapter_dir.resolve(), Path(args.out).resolve().parent)
    sink.payload.update({
        "log": log, "checkpoints": checkpoints, "completed_steps": len(log),
        "final": {"task_loss": task_loss(per), "examples": len(per),
                  "tokens": sum(ex["tokens"] for ex in per), "per_example": per},
        "memory": memory, "codec_stats": codec_stats,
        "adapter": {"dir": Path(adapter_rel).as_posix(), "sha256": tree_sha256(adapter_dir)},
        "durations": {"train_s": train_s, "final_eval_s": time.perf_counter() - t0},
    })
    print(f"final task loss {sink.payload['final']['task_loss']:.6f} over "
          f"{sink.payload['final']['tokens']} response tokens")


# ==========================================================================
# suites mode
# ==========================================================================
def score_suites(model: Any, tokenizer: Any, device: str, max_new_tokens: int
                 ) -> Dict[str, Any]:
    """Soup's four mini gate suites, greedy, every output kept (as C5 scores them)."""
    from soup_cli.eval.forgetting import MINI_BENCHMARKS, build_mcq_prompt, score_answer
    from soup_cli.eval.gate_suites import score_bundled_suite
    from soup_cli.utils.live_eval import make_generator

    generate = make_generator("in-memory", loaded=(model, tokenizer, device),
                              max_new_tokens=max_new_tokens)
    suites: Dict[str, Any] = {}
    for name, items in MINI_BENCHMARKS.items():
        outputs: Dict[str, str] = {}

        def capture(prompt: str) -> str:
            outputs[prompt] = generate(prompt)
            return outputs[prompt]

        accuracy = score_bundled_suite(name, capture)
        scored = []
        for item in items:
            prompt = build_mcq_prompt(item["question"], item["answer"])
            scored.append({"question": item["question"], "answer": item["answer"],
                           "output": outputs[prompt],
                           "correct": score_answer(outputs[prompt], item["answer"])})
        correct = sum(item["correct"] for item in scored)
        if correct != round(accuracy * len(items)):
            raise RuntimeError(f"{name}: per-item scores disagree with Soup's suite score")
        suites[name] = {"n": len(items), "correct": correct, "items": scored}
    return suites


def adapter_mismatch(model: Any, adapter: Path) -> Dict[str, Any]:
    """The live LoRA tensors of a loaded PeftModel against its saved file: names that
    are missing on either side or whose values differ. A widening cast on load (PEFT
    upcasts a bf16 adapter to float32) is undone before the exact comparison."""
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file

    saved = load_file(str(adapter / "adapter_model.safetensors"))
    live = get_peft_model_state_dict(model)
    bad = sorted(set(saved) ^ set(live))
    bad += [name for name in sorted(set(saved) & set(live)) if not torch.equal(
        live[name].detach().to("cpu").to(saved[name].dtype), saved[name])]
    return {"tensors": len(saved), "mismatch": bad[:8], "mismatched": len(bad)}


def base_parity(resident: Sequence[Dict[str, Any]], streamed: Sequence[Dict[str, Any]],
                *, adapter: Dict[str, Any],
                disabled: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """V13's record: the resident base's response NLL on panel rows against the NLL
    the streamed model recorded for the same rows and the same adapter at the end of
    training. ``disabled`` (the resident base with the adapter switched off) is
    informational: how large an adapter-sized difference is on these rows."""
    by_row = {ex["row"]: ex for ex in streamed}
    missing = [ex["row"] for ex in resident if ex["row"] not in by_row]
    if missing:
        raise RuntimeError(f"the training record has no streamed NLL for panel rows {missing[:4]}")
    paired = [by_row[ex["row"]] for ex in resident]
    per_token = [abs(a["nll"] - b["nll"]) / a["tokens"]
                 for a, b in zip(resident, paired) if a["tokens"]]
    return {
        "rows": len(resident), "tokens": sum(int(ex["tokens"]) for ex in resident),
        "tokens_match": all(a["tokens"] == b["tokens"] for a, b in zip(resident, paired)),
        "task_loss_resident": task_loss(resident), "task_loss_streamed": task_loss(paired),
        "max_abs_nll_per_token_diff": max(per_token, default=0.0),
        "task_loss_resident_adapter_disabled": task_loss(disabled) if disabled else None,
        "adapter": dict(adapter), "per_example_resident": list(resident),
    }


def suites_main(args: argparse.Namespace) -> int:
    """Score one training arm's saved adapter on the resident base, NF4 as trained,
    after checking that this resident base reproduces the streamed model the arm
    trained on (V13)."""
    import torch

    import soup_cli
    from soup_cli.utils.layer_stream import resolve_stream_dtype
    from soup_cli.utils.live_eval import load_model_and_tokenizer

    arm = Path(args.arm)
    record = json.loads(arm.read_text(encoding="utf-8"))
    adapter = recorded_adapter_dir(arm, record)
    if record.get("adapter", {}).get("sha256") != tree_sha256(adapter):
        raise RuntimeError(f"{adapter} does not match the adapter hashes in {arm}")
    config = record["meta"]["config"]
    dtype = resolve_stream_dtype(args.device)
    model, tokenizer, device = load_model_and_tokenizer(
        args.weights, adapter=str(adapter), device=args.device, dtype=dtype,
        quantization="4bit" if args.quant == "nf4" else None)
    # The panel the arm was scored on, rebuilt with this tokenizer and checked by digest.
    data = g4_data.load_splits(args.dolly or record["meta"]["args"].get("dolly"))
    panel = g4_data.encode_panel(tokenizer, data["splits"]["heldout"],
                                 max_length=config["panel_max_length"])
    if config.get("panel_limit"):
        panel = panel[:config["panel_limit"]]
    if g4_data.ids_digest([ex["ids"] for ex in panel]) != record["meta"]["data"][
            "panel_ids_sha256"]:
        raise RuntimeError(f"the panel rebuilt from {args.weights} is not the one {arm} scored")
    rows = panel[:args.parity_rows]
    resident = argparse.Namespace(model=model, device=device)
    with model.disable_adapter():
        disabled = panel_nll(resident, rows)
    parity = base_parity(panel_nll(resident, rows), record["final"]["per_example"],
                         adapter=adapter_mismatch(model, adapter), disabled=disabled)
    print(f"base parity on {parity['rows']} panel rows: resident "
          f"{parity['task_loss_resident']:.6f}, streamed {parity['task_loss_streamed']:.6f}, "
          f"adapter off {parity['task_loss_resident_adapter_disabled']:.6f}, "
          f"adapter tensors {parity['adapter']['tensors']} "
          f"({parity['adapter']['mismatched']} mismatched)")
    torch.manual_seed(0)
    suites = score_suites(model, tokenizer, device, args.max_new_tokens)
    payload = {
        "meta": {"label": args.label, "device": args.device,
                 "verdict_eligible": args.device == "cuda", "arm": arm.name,
                 "adapter_sha256": record["adapter"]["sha256"],
                 "max_new_tokens": args.max_new_tokens, "quant": args.quant, "dtype": dtype,
                 "args": vars(args), "versions": versions(), "git_head": git_head(),
                 "soup_cli_file": soup_cli.__file__, "harness_sha256": harness_hashes()},
        "base_parity": parity,
        "suites": suites,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(" ".join(f"{name} {entry['correct']}/{entry['n']}" for name, entry in suites.items()))
    print(f"wrote {args.out}")
    return 0


# ==========================================================================
# timing mode
# ==========================================================================
def mode_timing(ctx: Context, sink: Any) -> None:
    import torch

    args = ctx.args
    model, runtime = ctx.model, ctx.runtime
    data = load_data(ctx, rows_needed=args.k, panel=False)
    mbs = ctx.ids(data["rows"][:args.k])
    sink.payload["meta"]["data"] = {**data["provenance"],
                                    "timing_rows_sha256": g4_data.ids_digest(data["rows"][:args.k])}
    step = l2l_schedule.L2LStep(model, runtime, l2l_schedule.capture_layer_call(model, mbs[0]))
    store = ctx.store(args.codec, args.k)
    params = [param for _name, param in sorted(l2l_schedule.lora_parameters(model).items())]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=args.weight_decay)
    tokens = args.k * args.seq
    times: List[float] = []
    try:
        store.timing_on = False
        for index in range(args.warmup + args.steps):
            ctx.sync()
            if index == args.warmup and ctx.is_cuda:
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            step.run(mbs, store)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ctx.sync()
            if index >= args.warmup:
                times.append(time.perf_counter() - t0)
        memory = memory_record(ctx, store)
        store.timing_on = True
        for _ in range(2):
            step.run(mbs, store)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ctx.sync()
            store.flush_timing()
        stats = store.codec_stats
        instrumented = {
            "steps": 2, "compress_s_per_step": stats.compress_s / 2,
            "restore_s_per_step": stats.restore_s / 2,
            "compress_s_per_chunk": stats.compress_s / stats.compress_calls
            if stats.compress_calls else None,
            "restore_s_per_chunk": stats.restore_s / stats.restore_calls
            if stats.restore_calls else None,
            "chunks_per_step": ctx.n_layers * args.k,
        }
    finally:
        store.close()
    median = statistics.median(times)
    sink.payload.update({
        "timing": {"median_seconds": median, **summarize_times(times), "steps": times,
                   "tokens_per_step": tokens, "tok_per_s": tokens / median},
        "codec_timing": instrumented, "memory": memory,
    })
    print(f"timing {args.codec}: median {median:.4f} s/step, {tokens / median:.1f} tok/s")


# ==========================================================================
# main
# ==========================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.device == "cuda" and not stream_probe.cuda_available():
        print("ERROR: CUDA is required for g4_probe.py (--device cpu is the labelled pilot)",
              file=sys.stderr)
        return NO_CUDA_EXIT
    if args.mode == "suites":
        return suites_main(args)
    ctx = Context(args)
    sink = stream_probe.Sink(args.out, ctx.meta())
    try:
        if args.mode == "error":
            mode_error(ctx, sink)
        elif args.mode == "train":
            mode_train(ctx, sink)
        else:
            mode_timing(ctx, sink)
        sink.payload["meta"]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    finally:
        sink.flush()
        ctx.runtime.close()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
