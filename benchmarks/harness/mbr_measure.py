"""Measurement runner pieces for the exact-MBR campaign on Kanana (harness only, not shipped).

Pure logic that does not need a GPU: the four windows, the bit ladder, the mode guard (speed only
in normal mode, bits only in deterministic mode), OOM turned into a table row with its peaks,
MBPP-train records with a dataset hash, and the median of timed steps after warm-up. The GPU
driver calls these around ``L2LStep.run``.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from typing import Any, Callable, Dict, List, Sequence, Tuple

WINDOWS: Tuple[Tuple[int, int], ...] = ((8, 64), (8, 512), (64, 64), (64, 512))
BIT_LADDER: Tuple[int, ...] = (1, 2, 4, 8)
A0_MAX_K = 8
PEAK_KEYS = ("peak_allocated", "peak_reserved", "peak_nvml", "peak_ram")


class MixedModeError(RuntimeError):
    """Raised when a measurement kind is requested in the wrong torch mode."""


def check_mode(kind: str, *, deterministic: bool) -> str:
    """The mode label for ``kind`` ("speed" or "bits"); a mixed mode is refused."""
    if kind == "speed" and deterministic:
        raise MixedModeError("speed must be measured in normal mode, not deterministic")
    if kind == "bits" and not deterministic:
        raise MixedModeError("bits must be measured in deterministic mode, not normal")
    if kind not in ("speed", "bits"):
        raise ValueError(f"unknown measurement kind {kind!r}")
    return "deterministic" if deterministic else "normal"


def arms_for(*, k: int) -> Tuple[str, ...]:
    """A0 only exists for k <= 8 (its micro-batches no longer fit above that)."""
    return ("A0", "B0", "C") if k <= A0_MAX_K else ("B0", "C")


def run_guarded(run: Callable[[], Dict[str, Any]], *, peaks: Callable[[], Dict[str, Any]],
                is_oom: Callable[[BaseException], bool]) -> Dict[str, Any]:
    """Run one window; an OOM becomes ``status: OOM`` with the peaks seen, other errors raise."""
    try:
        out = dict(run())
        status = "ok"
    except Exception as error:  # noqa: BLE001 - re-raised unless it is an OOM
        if not is_oom(error):
            raise
        out, status = {"error": str(error).splitlines()[0][:200]}, "OOM"
    out.update(peaks())
    out["status"] = status
    return out


def make_row(*, arm: str, k: int, seq: int, kind: str, deterministic: bool,
             result: Dict[str, Any], dataset_sha256: str) -> Dict[str, Any]:
    """One report row; the mode is always written and checked against the kind."""
    row = {"arm": arm, "k": k, "seq": seq, "kind": kind,
           "mode": check_mode(kind, deterministic=deterministic),
           "dataset_sha256": dataset_sha256}
    row.update(result)
    row.setdefault("status", "ok")
    return row


def median_after_warmup(step_seconds: Sequence[float]) -> float:
    """Median of the steps after the first (warm) one."""
    if len(step_seconds) < 2:
        raise ValueError("need a warm-up step and at least one timed step")
    return statistics.median(step_seconds[1:])


def mbpp_records(rows: Sequence[Dict[str, Any]], tokenizer: Any, *, seq: int) -> Dict[str, Any]:
    """Right-truncated, padded token rows of MBPP text + code, and the sha256 of all of them.

    The hash covers the final padded ids and the sequence length, so any change of the text,
    the tokenizer or ``seq`` changes it."""
    pad = int(tokenizer.eos_token_id or 0)
    input_ids: List[List[int]] = []
    for row in rows:
        text = f"{str(row['text']).strip()}\n{str(row['code']).strip()}"
        ids = list(tokenizer(text, add_special_tokens=True)["input_ids"])[:seq]
        input_ids.append(ids + [pad] * (seq - len(ids)))
    digest = hashlib.sha256(json.dumps({"seq": seq, "ids": input_ids}).encode()).hexdigest()
    return {"input_ids": input_ids, "sha256": digest, "seq": seq}


def take_window(input_ids: Sequence[Sequence[int]], *, k: int) -> List[List[int]]:
    """``k`` rows, cycling through the dataset when it has fewer than ``k``."""
    return [list(input_ids[i % len(input_ids)]) for i in range(k)]


# --------------------------------------------------------------------------
# Arms, plans and verdicts
# --------------------------------------------------------------------------
def arm_options(arm: str, *, k: int) -> Dict[str, Any]:
    """``L2LStep.run`` keywords of an arm: A0 plain L2L, B0 route-preserving MBR, C B0 plus the
    ordered backward (the exact GEMM joins C with ticket 02)."""
    if arm not in ("A0", "B0", "C"):
        raise ValueError(f"unknown arm {arm!r}")
    if arm not in arms_for(k=k):
        raise ValueError(f"arm {arm} does not exist at k={k} (A0 only at k<={A0_MAX_K})")
    if arm == "A0":
        return {}
    options: Dict[str, Any] = {"route_preserving_mbr": True, "mbr_group_size": k}
    if arm == "C":
        options["mbr_ordered_backward"] = True
    return options


def speed_plan(*, blocks: int = 5) -> List[Dict[str, Any]]:
    """Normal-mode runs: per window, ``blocks`` blocks in A-B-C-C-B-A order (A0 left out when
    k > 8, which leaves B-C-C-B)."""
    plan: List[Dict[str, Any]] = []
    for k, seq in WINDOWS:
        arms = arms_for(k=k)
        order = [a for a in ("A0", "B0", "C", "C", "B0", "A0") if a in arms]
        for block in range(blocks):
            for arm in order:
                plan.append({"arm": arm, "k": k, "seq": seq, "block": block, "kind": "speed",
                             "deterministic": False})
    return plan


def bit_plan(*, seq: int) -> List[Dict[str, Any]]:
    """Deterministic-mode ladder: every arm at k = 1, 2, 4, 8."""
    return [{"arm": arm, "k": k, "seq": seq, "block": 0, "kind": "bits", "deterministic": True}
            for k in BIT_LADDER for arm in arms_for(k=min(k, A0_MAX_K))]


def grad_digest(grads: Dict[str, Sequence[float]]) -> str:
    """sha256 over the gradients in name order, bit-exact on the floats (``repr`` round trip)."""
    text = json.dumps({name: [float(v).hex() for v in grads[name]] for name in sorted(grads)})
    return hashlib.sha256(text.encode()).hexdigest()


def ladder_verdict(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, str], str]:
    """``equal`` / ``differs`` / ``no A0`` for every non-A0 row against the A0 row of its k."""
    ref = {r["k"]: r for r in rows if r["arm"] == "A0" and r.get("status") == "ok"}
    out: Dict[Tuple[int, str], str] = {}
    for row in rows:
        if row["arm"] == "A0":
            continue
        base = ref.get(row["k"])
        if base is None or row.get("status") != "ok":
            out[(row["k"], row["arm"])] = "no A0" if base is None else row.get("status", "?")
        else:
            same = base["grad_sha256"] == row["grad_sha256"] and base["loss"] == row["loss"]
            out[(row["k"], row["arm"])] = "equal" if same else "differs"
    return out


def parse_nvml_used(samples: Sequence[str]) -> int:
    """Largest ``memory.used`` sample of ``nvidia-smi --query-gpu=memory.used`` in bytes."""
    return max(int(s.split()[0]) for s in samples) * 2**20


# --------------------------------------------------------------------------
# GPU driver: one run in one fresh process, and the sweep that spawns them
# --------------------------------------------------------------------------
def _nvml_used_now() -> int:
    import subprocess

    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    return parse_nvml_used(out.split("\n")[:1])


class _NvmlSampler:
    """Peak of ``nvidia-smi`` memory.used, sampled on a thread (NVML sees the whole process,
    CUDA context included, which allocated/reserved do not)."""

    def __init__(self, interval: float = 0.5) -> None:
        import threading

        self.peak, self._stop, self._interval = 0, threading.Event(), interval
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, _nvml_used_now())
            except Exception:  # noqa: BLE001 - nvidia-smi missing or busy: skip the sample
                pass
            self._stop.wait(self._interval)

    def __enter__(self) -> "_NvmlSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join()


def _is_cuda_oom(error: BaseException) -> bool:
    import torch

    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def run_one(args: Any) -> Dict[str, Any]:  # pragma: no cover - needs the model and a CUDA host
    """One window of one arm. Mirrors ``oc_arm_run.run_arm``: streamed model from a prepared
    ``per_expert`` cache, ``L2LStep.run`` over MBPP-train rows, grad digest of the LoRA grads."""
    import os
    import time

    from benchmarks.harness import b5_real_moe, l2l_mistral_stand

    check_mode(args.kind, deterministic=args.deterministic)
    if args.deterministic:
        l2l_mistral_stand.set_deterministic_env()
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoTokenizer

    from soup_cli.utils.l2l_activations import ActivationStore
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call, label_tokens
    from soup_cli.utils.layer_shard import read_shard_index
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    options = arm_options(args.arm, k=args.k)
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.source)
    rows = list(load_dataset("google-research-datasets/mbpp", "full", split="train"))
    data = mbpp_records(rows, tokenizer, seq=args.seq)
    cuda = torch.cuda.is_available() and args.device != "cpu"
    device = "cuda" if cuda else "cpu"
    dtype = torch.bfloat16 if cuda else torch.float32
    b5_real_moe._reset_peak_rss()
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    def peaks() -> Dict[str, Any]:
        return {"peak_allocated": torch.cuda.max_memory_allocated() if cuda else None,
                "peak_reserved": torch.cuda.max_memory_reserved() if cuda else None,
                "peak_nvml": sampler.peak if cuda else None,
                "peak_ram": b5_real_moe._peak_rss()}

    def work() -> Dict[str, Any]:
        index = read_shard_index(args.cache)
        lora = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                          bias="none", target_modules=["q_proj", "kv_b_proj", "o_proj"],
                          task_type=TaskType.CAUSAL_LM)
        with b5_real_moe._meta_construction():
            model, runtime = build_streamed_model(
                model_id=args.source, shard_dir=args.cache, index=index, lora_config=lora,
                device=device, dtype="bfloat16" if cuda else "float32", buffers=2, pin=False,
                seed=0, quant="nf4" if cuda else "none", double_quant=False, tier="ram",
                read_ahead=1, two_read_experts=True, grouped_expert_routes=True,
                **({"expert_transfer": "packed"} if cuda else {}))
        try:
            model.train()
            ids = [torch.tensor([r], device=device) for r in take_window(data["input_ids"],
                                                                           k=args.k)]
            stepper = L2LStep(model, runtime, capture_layer_call(model, ids[0]))
            store = ActivationStore(
                n_layers=len(stepper.layers), k=args.k,
                chunk_shape=(1, args.seq, int(stepper.causal.config.hidden_size)),
                dtype=dtype, device=device, pin=cuda)
            params = {n: p for n, p in model.named_parameters() if p.requires_grad}
            seconds: List[float] = []
            losses: List[Any] = []
            for _ in range(args.steps):
                for p in params.values():
                    p.grad = None
                if cuda:
                    torch.cuda.synchronize()
                started = time.perf_counter()
                losses = stepper.run(ids, store, n_items=label_tokens(ids),
                                     labels=[i.clone() for i in ids], **options)
                if cuda:
                    torch.cuda.synchronize()
                seconds.append(time.perf_counter() - started)
            grads = {n: p.grad.detach().float().flatten().cpu().tolist()
                     for n, p in sorted(params.items()) if p.grad is not None}
            return {"step_seconds": median_after_warmup(seconds), "all_step_seconds": seconds,
                    "loss": float(sum(float(x) for x in losses)),
                    "grad_sha256": grad_digest(grads)}
        finally:
            runtime.close()

    class _Null:
        peak = 0

    sampler: Any = _NvmlSampler() if cuda else _Null()
    with (sampler if cuda else _NullContext()):
        result = run_guarded(work, peaks=peaks, is_oom=_is_cuda_oom)
    return make_row(arm=args.arm, k=args.k, seq=args.seq, kind=args.kind,
                    deterministic=args.deterministic, result=result,
                    dataset_sha256=data["sha256"])


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None


def sweep(args: Any, plan: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:  # pragma: no cover
    """Run each plan item in a fresh process; append every row to ``<out>/rows.jsonl``."""
    import subprocess
    import sys
    from pathlib import Path

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for item in plan:
        command = [sys.executable, __file__, "run", "--arm", item["arm"], "--k", str(item["k"]),
                   "--seq", str(item["seq"]), "--kind", item["kind"], "--source", args.source,
                   "--cache", args.cache, "--steps", str(args.steps), "--out", str(out),
                   "--device", args.device]
        if args.tokenizer:
            command += ["--tokenizer", args.tokenizer]
        if item["deterministic"]:
            command.append("--deterministic")
        proc = subprocess.run(command, capture_output=True, text=True)
        line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith("{")), None)
        row = json.loads(line) if line else {
            "arm": item["arm"], "k": item["k"], "seq": item["seq"], "kind": item["kind"],
            "mode": "deterministic" if item["deterministic"] else "normal",
            "status": "CRASH", "error": proc.stderr[-300:]}
        row["block"] = item["block"]
        rows.append(row)
        with open(out / "rows.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
    return rows


def main(argv: Any = None) -> int:  # pragma: no cover - thin CLI
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "speed", "bits"):
        p = sub.add_parser(name)
        p.add_argument("--source", required=True)
        p.add_argument("--cache", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--steps", type=int, default=4, help="first step is warm-up")
        p.add_argument("--lora-r", type=int, default=16)
        p.add_argument("--tokenizer", default=None, help="default: the --source folder")
        p.add_argument("--device", choices=("auto", "cpu"), default="auto")
        if name == "run":
            p.add_argument("--arm", required=True)
            p.add_argument("--k", type=int, required=True)
            p.add_argument("--seq", type=int, required=True)
            p.add_argument("--kind", choices=("speed", "bits"), required=True)
            p.add_argument("--deterministic", action="store_true")
        if name == "speed":
            p.add_argument("--blocks", type=int, default=5)
        if name == "bits":
            p.add_argument("--seq", type=int, default=64)
    args = parser.parse_args(argv)
    if args.command == "run":
        print(json.dumps(run_one(args)))
        return 0
    plan = speed_plan(blocks=args.blocks) if args.command == "speed" else bit_plan(seq=args.seq)
    rows = sweep(args, plan)
    if args.command == "bits":
        for (k, arm), verdict in sorted(ladder_verdict(rows).items()):
            print(f"k={k} {arm}: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
