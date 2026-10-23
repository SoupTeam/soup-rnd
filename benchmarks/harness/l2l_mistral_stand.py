#!/usr/bin/env python3
"""Ticket A3-04: the Mistral-7B NF4 stand, gradient accumulation (GA) against ``training.l2l``.

One window sequence, trained three times through the real HF trainer with the same seed,
adapters and rows: GA twice (the A-vs-A control) and the L2L trainer once. The rows have
different lengths, masked labels (-100) and an incomplete last window. The stand compares
every optimizer step's LoRA gradients with ``torch.equal``, the logged losses, the number
of optimizer steps, the lr of each step and the log. Deterministic mode is on and
documented: ``CUBLAS_WORKSPACE_CONFIG`` is set before torch and CUDA start, strict
``torch.use_deterministic_algorithms(True)`` follows, and one SDPA backend (MATH) is
pinned in both arms. A bitwise mismatch prints the largest gap and exits 1; the
threshold is never loosened. If A-vs-A already differs, the bitwise criterion cannot be
judged and the script says the criterion has to move to a tolerance, which is the
user's decision.

The RAM plan (weights + workspace + activation cache / 0.7) is printed before anything is
loaded, and the stand refuses to start without the RAM. Raw logs, the command, package
versions and the GPU are written to ``--out``. Results carry their data kind:
random-token rows are "synthetic" even on real weights.

Benchmark code, not shipped. torch is imported inside functions, after the environment is set.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_WEIGHTS = "mistralai/Mistral-7B-v0.1"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
PINNED_SDPA_BACKEND = "MATH"
SDPA_BACKENDS = ("MATH", "FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION")
#: Mistral-7B NF4 weights (3.6 GB of linear layers + 0.5 GB of bf16 embeddings), from the research.
DEFAULT_WEIGHTS_BYTES = 4.1e9
DEFAULT_WORKSPACE_BYTES = 1_000_000_000
#: NF4 with double quant, bytes per parameter (``layer_stream.py``).
NF4_BYTES_PER_PARAM = 0.5 + 1 / 64 + 4 / (64 * 256)
PACKAGES = (
    "torch", "transformers", "peft", "trl", "accelerate", "bitsandbytes", "numpy", "psutil",
)


class StandRefusedError(RuntimeError):
    """The stand will not start or write: a missing precondition, not a result."""


# --------------------------------------------------------------------------
# RAM plan
# --------------------------------------------------------------------------
def ram_plan(
    *, n_layers: int, hidden: int, k: int, batch: int, seq: int, itemsize: int = 2,
    pinned: bool = True, weights_bytes: float = DEFAULT_WEIGHTS_BYTES,
    workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
) -> Dict[str, Any]:
    """Host RAM the run needs, by ``check_activation_budget``'s formula.

    The cache is ``n_layers x k`` chunks of ``batch x seq x hidden``; a pinned chunk takes a
    power-of-two block. The trainer's check wants the cache below 0.7 of the RAM the weights
    leave free, so the whole plan is weights + workspace + cache / 0.7.
    """
    from soup_cli.utils.layer_stream import RAM_TIER_HEADROOM
    from soup_cli.utils.layer_stream_runtime import _next_power_of_two

    chunk = batch * seq * hidden * itemsize
    block = _next_power_of_two(chunk) if pinned else chunk
    cache = n_layers * k * block
    need_cache = cache / RAM_TIER_HEADROOM
    return {
        "n_layers": n_layers, "hidden": hidden, "k": k, "batch": batch, "seq": seq,
        "chunk_bytes": chunk, "block_bytes": block, "cache_bytes": cache,
        "need_free_for_cache": need_cache, "weights_bytes": weights_bytes,
        "workspace_bytes": workspace_bytes,
        "need_total": weights_bytes + workspace_bytes + need_cache,
    }


def check_ram(plan: Dict[str, Any], free_ram: Optional[int]) -> None:
    """Refuse when the free RAM is below the plan; an unknown amount is also a refusal."""
    if free_ram is None:
        raise StandRefusedError("free RAM is unknown (psutil is missing); the stand will not guess")
    if free_ram < plan["need_total"]:
        raise StandRefusedError(
            f"the plan needs {plan['need_total'] / 2**30:.1f} GiB of free RAM (weights "
            f"{plan['weights_bytes'] / 2**30:.1f} + workspace "
            f"{plan['workspace_bytes'] / 2**30:.1f} + activation cache "
            f"{plan['cache_bytes'] / 2**30:.1f} / 0.7), but only {free_ram / 2**30:.1f} GiB of "
            f"free RAM is available"
        )


def weights_bytes_estimate(config: Any, quant: str) -> float:
    """Resident weight bytes from the model config: NF4 linear layers, bf16 embeddings."""
    hidden = int(config.hidden_size)
    heads = int(config.num_attention_heads)
    kv_dim = int(getattr(config, "num_key_value_heads", None) or heads) * (hidden // heads)
    linear = int(config.num_hidden_layers) * (
        2 * hidden * hidden + 2 * hidden * kv_dim + 3 * hidden * int(config.intermediate_size)
    )
    tied = bool(getattr(config, "tie_word_embeddings", False))
    embed = int(config.vocab_size) * hidden * (1 if tied else 2)
    per_linear = NF4_BYTES_PER_PARAM if quant == "nf4" else 2
    return linear * per_linear + embed * 2


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------
def _cuda_started() -> bool:
    torch = sys.modules.get("torch")
    return bool(torch is not None and torch.cuda.is_initialized())


def set_deterministic_env(
    environ: Any = None, cuda_initialized: Callable[[], bool] = _cuda_started
) -> None:
    """Set ``CUBLAS_WORKSPACE_CONFIG``; it has no effect once CUDA is initialised, so refuse."""
    environ = os.environ if environ is None else environ
    if environ.get("CUBLAS_WORKSPACE_CONFIG") == CUBLAS_WORKSPACE_CONFIG:
        return
    if cuda_initialized():
        raise StandRefusedError(
            "CUBLAS_WORKSPACE_CONFIG must be set before CUDA starts, and CUDA is already "
            "initialised in this process; start the stand in a fresh process"
        )
    environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG


def enable_deterministic() -> None:
    """Strict mode: a non-deterministic op raises. ``warn_only`` stays off."""
    import torch

    torch.use_deterministic_algorithms(True)


@contextmanager
def pinned_sdpa() -> Iterator[None]:
    """Run attention on one SDPA backend, the same one in both arms."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(getattr(SDPBackend, PINNED_SDPA_BACKEND)):
        yield


def probe_sdpa_backends(
    *, device: str, dtype: str, batch: int = 1, heads: int = 32, seq: int = 512,
    head_dim: int = 128,
) -> Dict[str, Dict[str, Any]]:
    """Try each SDPA backend alone on the model's attention shape: forward, backward twice.

    Records whether it runs and whether two backward passes give equal gradients, so the
    choice of backend is measured on this card rather than assumed.
    """
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch_dtype = getattr(torch, dtype)
    gen = torch.Generator().manual_seed(0)

    def make() -> Any:
        return torch.randn((batch, heads, seq, head_dim), generator=gen).to(device, torch_dtype)

    q, k, v = make(), make(), make()
    mask = torch.ones(seq, seq, dtype=torch.bool, device=device).tril()
    out: Dict[str, Dict[str, Any]] = {}
    for name in SDPA_BACKENDS:
        record: Dict[str, Any] = {"ok": False, "backward_repeatable": None, "error": None}
        try:
            grads = []
            with sdpa_kernel(getattr(SDPBackend, name)):
                for _ in range(2):
                    qs, ks, vs = (t.clone().requires_grad_() for t in (q, k, v))
                    res = torch.nn.functional.scaled_dot_product_attention(
                        qs, ks, vs, attn_mask=mask
                    )
                    res.sum().backward()
                    grads.append((qs.grad.clone(), ks.grad.clone(), vs.grad.clone()))
            record["ok"] = True
            record["backward_repeatable"] = all(
                torch.equal(a, b) for a, b in zip(grads[0], grads[1])
            )
        except Exception as exc:  # a backend that cannot run here is a result, not a crash
            record["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
        out[name] = record
    return out


# --------------------------------------------------------------------------
# gemm_4bit path counter
# --------------------------------------------------------------------------
_PATH_FUNCTIONS = {"_gemm_4bit_kernel_impl": "custom", "_dequant_linear_fallback": "dequant"}


class GemmPathCounter:
    """Counts bitsandbytes' ``gemm_4bit`` calls by M (rows) and path (own kernel or dequant).

    Wraps the two path functions of ``bitsandbytes.backends.cuda.ops`` for the length of a
    ``with`` block and restores them. ``ops=None`` (no bitsandbytes CUDA ops) counts nothing.
    """

    def __init__(self, ops: Any):
        self.ops = ops
        self.counts: Optional[Dict[int, Dict[str, int]]] = None if ops is None else {}
        self._saved: Dict[str, Any] = {}

    def __enter__(self) -> "GemmPathCounter":
        if self.ops is None:
            return self
        for name, path in _PATH_FUNCTIONS.items():
            real = getattr(self.ops, name)
            self._saved[name] = real
            setattr(self.ops, name, self._wrap(real, path))
        return self

    def _wrap(self, real: Callable[..., Any], path: str) -> Callable[..., Any]:
        def counted(act: Any, *args: Any, **kwargs: Any) -> Any:
            rows = int(act.numel() // act.shape[-1])
            entry = self.counts.setdefault(rows, {"custom": 0, "dequant": 0})
            entry[path] += 1
            return real(act, *args, **kwargs)

        return counted

    def __exit__(self, *exc: Any) -> None:
        for name, real in self._saved.items():
            setattr(self.ops, name, real)
        self._saved = {}


def paths_match(
    left: Optional[Dict[int, Dict[str, int]]], right: Optional[Dict[int, Dict[str, int]]]
) -> Optional[bool]:
    """True when both arms used the same set of gemm_4bit paths; None if uncounted.

    Not keyed by M: GA runs rows at their own length and an L2L window is padded to a
    multiple of 64, so the M values differ by design. The per-M counts stay in the report.
    """
    if left is None or right is None:
        return None

    def kinds(counts: Dict[int, Dict[str, int]]) -> set:
        return {path for paths in counts.values() for path, n in paths.items() if n}

    return kinds(left) == kinds(right)


# --------------------------------------------------------------------------
# bitsandbytes #2034: a recycled slot buffer must not change the gradients
# --------------------------------------------------------------------------
def judge_aliasing(*, shipped_max_abs: float, raw_max_abs: float) -> Dict[str, Any]:
    """Verdict of one M. The shipped path must not move; the raw path is the control."""
    if shipped_max_abs > 0:
        return {
            "status": "active", "pass": False,
            "note": (
                f"bnb #2034 reaches the shipped path: recycled-buffer gradients differ from "
                f"private-buffer ones by {shipped_max_abs:.3e}. An L2L/GA gap on this stand "
                "could come from the aliasing, not from the schedule."
            ),
        }
    if raw_max_abs > 0:
        return {"status": "neutralized", "pass": True,
                "note": "raw matmul_4bit shows the defect; the shipped path is unaffected"}
    return {
        "status": "not_reproduced", "pass": True,
        "note": "the raw control is also equal at this M, so the probe cannot show the defect",
    }


def _aliasing_grads(
    path: str, recycle: bool, *, device: str, dtype: Any, dim: int, n_layers: int,
    slots: int, tokens: int,
) -> List[Any]:
    """Gradients of per-layer scales upstream of a checkpointed NF4 matmul, as bnb_repro.py does.

    ``path`` is ``shipped`` (a Linear4bit patched by ``install_dequant_forward``) or ``raw``
    (``bitsandbytes.matmul_4bit``). With ``recycle`` the packed weight is copied into a
    round-robin slot buffer inside the checkpointed region, as the streaming runtime does.
    """
    import bitsandbytes as bnb
    import torch
    from bitsandbytes.functional import quantize_4bit
    from torch.utils.checkpoint import checkpoint

    from soup_cli.utils.layer_stream_runtime import install_dequant_forward

    torch.manual_seed(0)
    weights = [torch.randn(dim, dim, device=device, dtype=dtype) / 32 for _ in range(n_layers)]
    quantised = [
        quantize_4bit(w, blocksize=64, quant_type="nf4", compress_statistics=True)
        for w in weights
    ]
    sources = [packed for packed, _ in quantised]
    states = [state for _, state in quantised]
    buffers = [torch.empty_like(sources[0]) for _ in range(slots)]
    cursor = [0]
    lin = None
    if path == "shipped":
        lin = bnb.nn.Linear4bit(dim, dim, bias=False, compute_dtype=dtype, quant_type="nf4")
        lin.weight = bnb.nn.Params4bit(
            data=sources[0], requires_grad=False, quant_state=states[0], blocksize=64,
            compress_statistics=True, quant_type="nf4", bnb_quantized=True,
        )
        install_dequant_forward(lin)
    gen = torch.Generator().manual_seed(1234)
    scales = [
        (1 + 0.01 * torch.randn(dim, generator=gen)).to(device, dtype).requires_grad_()
        for _ in range(n_layers)
    ]

    def layer(idx: int) -> Callable[..., Any]:
        def fn(x: Any, scale: Any) -> Any:
            hidden = x * scale
            weight = sources[idx]
            if recycle:
                weight = buffers[cursor[0] % slots]
                cursor[0] += 1
                weight.copy_(sources[idx])
            if lin is not None:
                lin.weight.data = weight
                lin.weight.quant_state = states[idx]
                return lin(hidden)
            return bnb.matmul_4bit(hidden, weight.t(), quant_state=states[idx])

        return fn

    x = torch.randn(tokens, dim, device=device, dtype=dtype, requires_grad=True)
    for idx in range(n_layers):
        x = checkpoint(layer(idx), x, scales[idx], use_reentrant=False)
    x.float().pow(2).mean().backward()
    return [scale.grad.detach().float().cpu().clone() for scale in scales]


def run_aliasing_probe(
    *, device: str, dtype: str, m_values: Sequence[int], dim: int, n_layers: int = 8,
    slots: int = 2,
) -> Dict[str, Any]:
    """Recycled against private buffers on the shipped NF4 path, at each M the stand uses."""
    import torch

    torch_dtype = getattr(torch, dtype)
    by_m: Dict[int, Dict[str, Any]] = {}
    for rows in sorted(set(int(m) for m in m_values)):
        diffs = {}
        for path in ("shipped", "raw"):
            grads = [
                _aliasing_grads(
                    path, recycle, device=device, dtype=torch_dtype, dim=dim,
                    n_layers=n_layers, slots=slots, tokens=rows,
                )
                for recycle in (True, False)
            ]
            diffs[path] = max(
                (a - b).abs().max().item() for a, b in zip(grads[0], grads[1])
            )
        by_m[rows] = {
            "shipped_max_abs": diffs["shipped"], "raw_max_abs": diffs["raw"],
            **judge_aliasing(shipped_max_abs=diffs["shipped"], raw_max_abs=diffs["raw"]),
        }
    ok = all(rec["pass"] for rec in by_m.values())
    statuses = {rec["status"] for rec in by_m.values()}
    status = "active" if "active" in statuses else (
        "neutralized" if "neutralized" in statuses else "not_reproduced"
    )
    return {"pass": ok, "status": status, "by_m": by_m, "dim": dim, "n_layers": n_layers,
            "slots": slots}


def aliasing_block(probe: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The verdict that stops the stand when #2034 is active on the shipped path, else None."""
    if probe["pass"]:
        return None
    note = next(rec["note"] for rec in probe["by_m"].values() if not rec["pass"])
    return {"pass": False, "status": "aliasing_active", "note": note,
            "a_vs_a": None, "l2l_vs_ga": None}


# --------------------------------------------------------------------------
# runs and their comparison
# --------------------------------------------------------------------------
@dataclass
class RunRecord:
    """What one run leaves: its log rows, and the lr and LoRA gradient of each optimizer step."""

    logs: List[dict]
    rates: List[float]
    grads: List[dict]
    steps: int
    l2l: bool = False


def compare_runs(ref: RunRecord, cand: RunRecord) -> Dict[str, Any]:
    """Bit-compare two runs: gradients, losses, step counts, lr per step and the log."""
    import torch

    unequal, diffs = False, []
    if len(ref.grads) != len(cand.grads):
        unequal = True
    for step, (left, right) in enumerate(zip(ref.grads, cand.grads)):
        if left.keys() != right.keys():
            unequal = True
            continue
        for name in sorted(left):
            if not torch.equal(left[name], right[name]):
                unequal = True
                gap = (left[name].float() - right[name].float()).abs().max().item()
                diffs.append({"step": step, "name": name, "max_abs": float(gap)})
    diffs.sort(key=lambda row: row["max_abs"], reverse=True)
    losses_equal = [r.get("loss") for r in ref.logs] == [r.get("loss") for r in cand.logs]
    rates_equal = ref.rates == cand.rates and [
        r.get("learning_rate") for r in ref.logs
    ] == [r.get("learning_rate") for r in cand.logs]
    steps_equal = ref.steps == cand.steps and len(ref.grads) == len(cand.grads)
    return {
        "grads_equal": not unequal,
        "losses_equal": losses_equal,
        "rates_equal": rates_equal,
        "steps_equal": steps_equal,
        "steps": (ref.steps, cand.steps),
        "max_abs": diffs[0]["max_abs"] if diffs else 0.0,
        "worst": diffs[:5],
        "equal": bool(not unequal and losses_equal and rates_equal and steps_equal),
    }


def _vacuous(run: RunRecord) -> bool:
    return not run.grads or any(
        not any(bool(g.abs().sum() > 0) for g in step.values()) for step in run.grads
    )


def decide(ga1: RunRecord, ga2: RunRecord, l2l: RunRecord) -> Dict[str, Any]:
    """The verdict. A-vs-A comes first: if GA does not repeat, L2L cannot be judged in bits."""
    a_vs_a = compare_runs(ga1, ga2)
    out: Dict[str, Any] = {"a_vs_a": a_vs_a, "l2l_vs_ga": None, "pass": False, "note": ""}
    if not a_vs_a["equal"]:
        out["status"] = "a_vs_a_failed"
        out["note"] = (
            f"two gradient-accumulation runs differ (max |d| {a_vs_a['max_abs']:.3e}), so a "
            "bitwise comparison of L2L against GA is meaningless on this stand. The "
            "criterion has to move to a tolerance; that decision is the user's, and the "
            "stand does not loosen it."
        )
        return out
    out["l2l_vs_ga"] = compare = compare_runs(ga1, l2l)
    if _vacuous(ga1) or _vacuous(l2l):
        out["status"] = "vacuous"
        out["note"] = "a step has all-zero LoRA gradients, so equality proves nothing"
    elif not compare["equal"]:
        out["status"] = "l2l_differs"
        flags = [k for k in ("grads_equal", "losses_equal", "rates_equal", "steps_equal")
                 if not compare[k]]
        out["note"] = (
            f"L2L differs from GA in {', '.join(flags)}; largest gradient gap "
            f"{compare['max_abs']:.6g}; worst {compare['worst'][:3]}"
        )
    else:
        out["status"] = "equal"
        out["pass"] = True
    return out


class SyntheticRows:
    """Rows of random ids; ``masked_spans[i]`` lists the (start, stop) label slices set to -100."""

    def __init__(self, lengths: Sequence[int], masked_spans: Sequence[Sequence[tuple]],
                 vocab: int, seed: int = 5):
        import torch

        gen = torch.Generator().manual_seed(seed)
        self.rows = []
        for length, spans in zip(lengths, masked_spans):
            ids = torch.randint(0, vocab, (length,), generator=gen)
            labels = ids.clone()
            for start, stop in spans:
                labels[start:stop] = -100
            self.rows.append(
                {"input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids)}
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return self.rows[i]


def train_arm(
    model: Any, runtime: Any, rows: Any, work_dir: Path, *, name: str, l2l: bool, k: int,
    dtype: Any, seed: int, trainer_options: Optional[dict] = None,
) -> RunRecord:
    """One run through the HF trainer (L2L subclass or plain); records each step's gradient, lr."""
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments

    from soup_cli.trainer.l2l_trainer import attach_l2l_state, make_l2l_trainer_class

    grads: List[dict] = []
    rates: List[float] = []

    class Recorder(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
            grads.append({
                n: p.grad.detach().to("cpu", copy=True)
                for n, p in model.named_parameters()
                if "lora_" in n and p.grad is not None
            })

        def on_step_end(self, args, state, control, optimizer=None, **kw):
            rates.append(optimizer.param_groups[0]["lr"])

    options: Dict[str, Any] = dict(
        output_dir=str(work_dir / name), per_device_train_batch_size=1,
        gradient_accumulation_steps=k, learning_rate=1e-3, optim="sgd",
        lr_scheduler_type="linear", warmup_steps=1, num_train_epochs=1, logging_steps=1,
        save_strategy="no", report_to=[], seed=seed, data_seed=seed, max_grad_norm=0.0,
        disable_tqdm=True,
    )
    options.update(trainer_options or {})
    cls = make_l2l_trainer_class(Trainer) if l2l else Trainer
    trainer = cls(
        model=model, args=TrainingArguments(**options), train_dataset=rows,
        callbacks=[Recorder()],
    )
    state = attach_l2l_state(trainer, runtime, dtype=dtype or torch.float32) if l2l else None
    try:
        trainer.train()
    finally:
        if state is not None:
            state.close()
    logs = [row for row in trainer.state.log_history if "loss" in row]
    return RunRecord(
        logs=logs, rates=rates, grads=grads, steps=int(trainer.state.global_step), l2l=l2l
    )


def _log_text(run: RunRecord) -> str:
    rows = [json.dumps(row, sort_keys=True) for row in run.logs]
    return "\n".join(rows + [json.dumps({"lr_per_step": run.rates})]) + "\n"


@dataclass
class StandResult:
    runs: Dict[str, RunRecord]
    verdict: Dict[str, Any]
    logs: Dict[str, str]
    gemm_paths: Dict[str, Any] = field(default_factory=dict)


def run_stand(
    model: Any, runtime: Any, rows: Any, *, k: int, work_dir: Path,
    trainer_options: Optional[dict] = None, dtype: Any = None, seed: int = 1,
    bnb_ops: Any = None, sdpa: Callable[[], Any] = pinned_sdpa,
) -> StandResult:
    """GA, GA again, then L2L over the same rows from the same adapters; return the verdict."""
    import l2l_schedule
    import torch

    start = l2l_schedule.snapshot_adapters(model)
    arms = (("ga1", False), ("ga2", False), ("l2l", True))
    runs: Dict[str, RunRecord] = {}
    counts: Dict[str, Any] = {}
    with sdpa():
        for name, is_l2l in arms:
            l2l_schedule.restore_adapters(model, start)
            torch.manual_seed(seed)
            with GemmPathCounter(bnb_ops) as counter:
                runs[name] = train_arm(
                    model, runtime, rows, Path(work_dir), name=name, l2l=is_l2l, k=k,
                    dtype=dtype, seed=seed, trainer_options=trainer_options,
                )
            counts[name] = counter.counts
    verdict = decide(runs["ga1"], runs["ga2"], runs["l2l"])
    gemm = {**counts, "match": paths_match(counts["ga1"], counts["l2l"]),
            "a_vs_a_match": paths_match(counts["ga1"], counts["ga2"])}
    return StandResult(
        runs=runs, verdict=verdict, logs={f"{n}.log": _log_text(r) for n, r in runs.items()},
        gemm_paths=gemm,
    )


# --------------------------------------------------------------------------
# facts and report
# --------------------------------------------------------------------------
def environment_facts(packages: Sequence[str] = PACKAGES) -> Dict[str, Any]:
    from importlib import metadata

    versions: Dict[str, Optional[str]] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"packages": versions, "python": sys.version.split()[0], "platform": platform.platform()}


def render_report(payload: Dict[str, Any]) -> str:
    verdict = payload["verdict"]
    kind = payload["data_kind"]
    other = "Real" if kind == "synthetic" else "Synthetic"
    heading = "Synthetic data (random tokens, real weights)" if kind == "synthetic" else \
        "Real data"
    env = payload.get("env", {})
    lines = [
        f"# L2L against gradient accumulation: {payload.get('model', '?')}",
        "",
        f"**Verdict: {'PASS' if verdict['pass'] else 'FAIL'}** ({verdict['status']})",
    ]
    if verdict.get("note"):
        lines += ["", verdict["note"]]
    lines += ["", "## Command", "", "```", " ".join(payload.get("command", [])), "```", ""]
    lines += ["## Environment", ""]
    lines += [f"- {n} {v}" for n, v in env.get("packages", {}).items() if v]
    lines += [f"- GPU: {env.get('gpu', '?')}", f"- Python: {env.get('python', '?')}",
              f"- Platform: {env.get('platform', '?')}", ""]
    det = payload.get("deterministic", {})
    lines += ["## Determinism", ""]
    lines += [f"- {key}: {value}" for key, value in det.items()]
    sdpa = payload.get("sdpa", {})
    lines += ["", "## SDPA backend", "", f"Pinned in both arms: {sdpa.get('pinned')}", ""]
    for name, rec in sdpa.get("probe", {}).items():
        lines.append(f"- {name}: {rec}")
    gemm = payload.get("gemm_paths", {})
    lines += ["", "## gemm_4bit path (calls by M)", ""]
    for arm in ("ga1", "ga2", "l2l"):
        lines.append(f"- {arm}: {gemm.get(arm, 'not counted')}")
    lines.append(f"- same path in GA and L2L: {gemm.get('match')}")
    if gemm.get("match") is False:
        lines.append("- WARNING: the arms did not use the same set of gemm_4bit paths")
    elif gemm.get("match") is None:
        lines.append("- WARNING: gemm_4bit paths were not counted (no bitsandbytes CUDA ops)")
    lines += ["", "## RAM plan", ""]
    lines += [f"- {key}: {value}" for key, value in payload.get("ram_plan", {}).items()]
    alias = payload.get("aliasing")
    lines += ["", "## bitsandbytes #2034 (recycled slot buffers)", ""]
    if alias is None:
        lines.append("- not run")
    else:
        lines.append(f"- status: {alias['status']}")
        for rows, rec in sorted(alias["by_m"].items()):
            lines.append(
                f"- M={rows}: shipped max |d| {rec['shipped_max_abs']}, "
                f"raw control max |d| {rec['raw_max_abs']} ({rec['status']})"
            )
    a_vs_a = verdict.get("a_vs_a") or {}
    lines += ["", f"## {heading}", "", f"- A-vs-A (GA twice) bit-equal: {a_vs_a.get('equal')}"]
    if a_vs_a.get("max_abs") is not None:
        lines.append(f"- A-vs-A max |d|: {a_vs_a['max_abs']}")
    compare = verdict.get("l2l_vs_ga")
    if compare is None:
        lines.append("- L2L against GA: not judged")
    else:
        lines += [f"- L2L against GA bit-equal: {compare.get('equal')}",
                  f"- max |d| of LoRA gradients: {compare.get('max_abs')}",
                  f"- optimizer steps (GA, L2L): {compare.get('steps')}"]
    lines += ["", f"## {other} data: not run", ""]
    return "\n".join(lines)


def write_outputs(out_dir: Path, payload: Dict[str, Any], raw_logs: Dict[str, str]) -> None:
    """Write report.md, records.json and the raw logs; an existing directory is never touched."""
    out = Path(out_dir)
    if out.exists():
        raise StandRefusedError(f"{out} exists; attempts are never overwritten")
    (out / "raw").mkdir(parents=True)
    (out / "report.md").write_text(render_report(payload), encoding="utf-8")
    (out / "records.json").write_text(
        json.dumps(payload, indent=1, default=str), encoding="utf-8"
    )
    for name, text in raw_logs.items():
        (out / "raw" / name).write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------
def make_rows(count: int, seq: int, vocab: int) -> SyntheticRows:
    """Rows of mixed lengths: odd rows mask a prefix, every third row masks a middle span."""
    fractions = (1.0, 0.75, 0.5, 0.25, 0.9, 0.6)
    lengths = [max(8, int(seq * fractions[i % len(fractions)])) for i in range(count)]
    spans: List[List[tuple]] = []
    for i, length in enumerate(lengths):
        row: List[tuple] = []
        if i % 2:
            row.append((0, length // 4))
        if i % 3 == 2:
            row.append((length // 2, length // 2 + max(1, length // 8)))
        spans.append(row)
    return SyntheticRows(lengths, spans, vocab)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--quant", choices=("none", "nf4"), default="nf4")
    parser.add_argument("--k", type=int, default=4, help="gradient_accumulation_steps")
    parser.add_argument("--rows", type=int, default=10, help="must leave an incomplete last window")
    parser.add_argument("--seq", type=int, default=256, help="longest row, in tokens")
    parser.add_argument("--rng-seed", type=int, default=1)
    parser.add_argument("--out", required=True)
    parsed = parser.parse_args(argv)

    set_deterministic_env()  # first, before torch and CUDA start
    try:
        if parsed.rows % parsed.k == 0:
            raise StandRefusedError(
                f"--rows {parsed.rows} fills every window of {parsed.k}; use a count that "
                "leaves an incomplete last window"
            )
        if Path(parsed.out).exists():
            raise StandRefusedError(f"{parsed.out} exists; attempts are never overwritten")
        import stream_probe

        if not stream_probe.cuda_available():
            print("SKIP: CUDA is required for l2l_mistral_stand.py")
            return 0
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(parsed.weights)
        padded = -(-parsed.seq // 64) * 64  # an L2L window pads to a multiple of 64
        plan = ram_plan(
            n_layers=int(config.num_hidden_layers), hidden=int(config.hidden_size),
            k=parsed.k, batch=1, seq=padded,
            weights_bytes=weights_bytes_estimate(config, parsed.quant),
        )
        for key, value in plan.items():
            print(f"ram_plan {key:<22}{value}")
        from soup_cli.utils.layer_stream import free_ram_bytes

        check_ram(plan, free_ram_bytes())
    except StandRefusedError as exc:
        print(f"REFUSED: {exc}")
        return 2

    import l2l_probe
    import l2l_schedule
    import torch

    from soup_cli.utils.layer_stream import resolve_stream_dtype

    enable_deterministic()
    device = "cuda"
    dtype = resolve_stream_dtype(device)
    args = l2l_probe.parse_args([
        "--mode", "correctness", "--weights", parsed.weights, "--quant", parsed.quant,
        "--seq", str(parsed.seq), "--tier", "ram", "--out", str(Path(parsed.out) / "unused.json"),
    ])
    facts = stream_probe.gpu_facts(device)
    print("RUN: L2L vs gradient accumulation,", parsed.weights, parsed.quant)
    model, runtime, config, *_ = stream_probe.build(args, device, dtype)
    try:
        l2l_schedule.make_non_vacuous(model)
        heads = int(config.num_attention_heads)
        probe = probe_sdpa_backends(
            device=device, dtype=dtype, heads=heads, seq=padded,
            head_dim=int(config.hidden_size) // heads,
        )
        try:
            from bitsandbytes.backends.cuda import ops as bnb_ops
        except Exception:
            bnb_ops = None
        rows = make_rows(parsed.rows, parsed.seq, int(config.vocab_size))
        lengths = [len(rows[i]["input_ids"]) for i in range(len(rows))]
        aliasing = run_aliasing_probe(
            device=device, dtype=dtype, dim=int(config.hidden_size),
            m_values={min(lengths), max(lengths), padded},
        )
        print(f"bnb #2034 probe: {aliasing['status']}")
        blocked = aliasing_block(aliasing)
        import tempfile

        if blocked is not None:
            result = StandResult(runs={}, verdict=blocked, logs={})
        else:
            with tempfile.TemporaryDirectory() as work:
                result = run_stand(
                    model, runtime, rows, k=parsed.k, work_dir=Path(work),
                    trainer_options={"bf16": dtype == "bfloat16"},
                    dtype=getattr(torch, dtype), seed=parsed.rng_seed, bnb_ops=bnb_ops,
                )
    finally:
        runtime.close()
    env = environment_facts()
    env["gpu"] = facts.get("gpu")
    env["gpu_facts"] = facts
    payload = {
        "data_kind": "synthetic", "model": parsed.weights, "quant": parsed.quant,
        "command": [sys.executable, *sys.argv], "env": env,
        "deterministic": {
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "algorithms": "strict (warn_only off)",
            "attn_implementation": getattr(config, "_attn_implementation", None),
        },
        "sdpa": {"pinned": PINNED_SDPA_BACKEND, "probe": probe},
        "gemm_paths": result.gemm_paths, "ram_plan": plan, "verdict": result.verdict,
        "aliasing": aliasing,
        "k": parsed.k, "rows": parsed.rows, "seq": parsed.seq,
    }
    try:
        write_outputs(Path(parsed.out), payload, result.logs)
    except StandRefusedError as exc:
        print(f"REFUSED: {exc}")
        return 2
    print(f"verdict {result.verdict['status']}: {result.verdict['note'] or 'bit-equal'}")
    print(f"wrote {Path(parsed.out) / 'report.md'}")
    return 0 if result.verdict["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
