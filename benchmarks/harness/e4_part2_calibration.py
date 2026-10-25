"""E4 part 2 calibration harness (calibration rule sections 2-8).

    python benchmarks/harness/e4_part2_calibration.py split  --data-root R --out OUT
    python benchmarks/harness/e4_part2_calibration.py run    --stage F|T|W|N|P --arm A_on ...
    python benchmarks/harness/e4_part2_calibration.py drive  --stage F ...   (fresh process per run)

Every run is one process and one journaled attempt; its record (result, box stamps, provenance,
input hashes, void reasons) is written whether or not it is void. Heavy libraries are imported
inside functions. Nothing here scores a DECISION row.
Calibration boxes install `.[calibration]` with `.github/constraints/e4-part2-pins.txt`, which
provides nvidia-ml-py for the pynvml sampler.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import e4_part2_common as common
import e4_part2_quant as quant
import e4_part2_split as split
import e4_part2_transfer as transfer
import e4_part2_wref as wref

# --------------------------------------------------------------------------
# recipe (calibration rule section 3, the main rule's recipe in full)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Recipe:
    r: int = 16
    alpha: int = 32
    microbatch: int = 1
    accum: int = 16
    max_length: int = 512
    lr: float = 1e-4
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    warmup_updates: int = 16
    total_updates: int = 256
    clip_norm: float = 1.0
    f_updates: int = 16
    seed: int = 42

    def adamw_kwargs(self) -> Dict[str, Any]:
        return {"lr": self.lr, "betas": self.betas, "eps": self.eps,
                "weight_decay": self.weight_decay}


RECIPE = Recipe()
ARMS = ("A_off", "A_on", "A_bf16")
F_MEMORY_LIMIT_BYTES = 36_000_000_000
T_SELECTION_SEED = common.CAL_SEED
CALLS_PER_UNIT = 4  # SMOKE rows per unit in T; the rule does not say, pinned here
# Fallback targeting (--fallback-gateup, off by default). bitsandbytes
# backends/cuda/ops.py:584-678: on A100 (sm80) gate_proj and up_proj (N=1024, K=2048) take the
# dequant + F.linear fallback for M = 129..512 tokens and the fused kernel for M <= 128. No SMOKE
# unit reached M > 128 (calibration T records: gate/up 1-88 tokens), so these cells have no tau.
FALLBACK_GATEUP_KINDS = ("gate_proj", "up_proj")
FALLBACK_GATEUP_M_FLOOR = (129, 256)  # fused up to M=128, fallback from 129 (see above)
FALLBACK_M_CAP = 512


def fallback_m_targets(pool_rows: int) -> tuple:
    """The fallback call sizes for a pool of ``pool_rows`` distinct SMOKE rows: 129 and 256, then
    the whole pool when it is above 256 (capped at 512). Cells are keyed on kind, needs_grad and
    kernel, not on M, so any M in the fallback range calibrates the same cell."""
    if pool_rows < FALLBACK_GATEUP_M_FLOOR[1]:
        raise common.CalibrationStopped(
            f"fallback targets: cannot reach M=256 from {pool_rows} distinct SMOKE rows; no row "
            "is repeated, so the target is not calibrated")
    extra = (min(pool_rows, FALLBACK_M_CAP),) if pool_rows > FALLBACK_GATEUP_M_FLOOR[1] else ()
    return FALLBACK_GATEUP_M_FLOOR + extra
W_LAYERS = (0, 8)
W_ROWS = 32
N_REPEATS = 5
MAX_NEW_TOKENS = 256
KINDS = quant.KIND_ORDER
ATTENTION_KINDS = ("q_proj", "k_proj", "v_proj", "o_proj")


def lr_factor(update: int, warmup: int, total: int) -> float:
    """Linear warmup then linear decay to 0 at ``total``; HF ``get_linear_schedule_with_warmup``.

    ``update`` is the index of the optimizer step about to run (0-based), so update 0 runs at
    lr 0 and update ``warmup`` at the full rate. The rule says "warmup 16 updates, linear decay
    to 0 at update 256" without a formula; this is the Trainer's.
    """
    if update < warmup:
        return update / max(1, warmup)
    return max(0.0, (total - update) / max(1, total - warmup))


def shuffled_order(n: int, seed: int) -> List[int]:
    """Seed-dependent shuffle of TRAIN, identical across arms within a seed."""
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order


def expected_trainable_params(config, r: int = 16) -> Dict[str, int]:
    """Adapter parameter count the placement of main rule section 4 must produce."""
    h, i, e, layers = (config.hidden_size, config.intermediate_size, config.num_experts,
                       config.num_hidden_layers)
    heads_dim = config.num_attention_heads * (h // config.num_attention_heads)
    kv_dim = config.num_key_value_heads * (h // config.num_attention_heads)
    attn = r * ((h + heads_dim) + (h + kv_dim) * 2 + (heads_dim + h))
    gate_up = r * e * h + 2 * i * r * e
    down = r * e * i + h * r * e
    return {"attention": attn * layers, "experts": (gate_up + down) * layers,
            "total": (attn + gate_up + down) * layers}


# --------------------------------------------------------------------------
# data: encoded rows from the manifests
# --------------------------------------------------------------------------


def encode_manifest_rows(manifest_rows: Sequence[dict], dolly_rows: Sequence[dict], tokenizer
                         ) -> List[Dict[str, Any]]:
    """Encode the rows a manifest names; any identity or token-count mismatch is an input error."""
    out = []
    for row in manifest_rows:
        raw = dolly_rows[row["row_id"]]
        fields = [split.normalize_field(raw[k] or "") for k in ("instruction", "context",
                                                                 "response")]
        if split.identity(*fields) != row["identity"]:
            raise ValueError(f"row {row['row_id']}: identity differs from the manifest")
        enc = split.encode_example(tokenizer, *fields)
        if (len(enc["prompt_ids"]), len(enc["response_ids"])) != (
                row["prompt_tokens"], row["response_tokens"]):
            raise ValueError(f"row {row['row_id']}: token counts differ from the manifest")
        out.append({**enc, "row_id": row["row_id"]})
    return out


def read_manifest(path: Path, expect_sha256: Optional[str] = None) -> List[dict]:
    data = Path(path).read_bytes()
    if expect_sha256 is not None and common.sha256_bytes(data) != expect_sha256:
        raise ValueError(f"{path}: SHA-256 differs from the recorded one")
    return [json.loads(line) for line in data.decode().splitlines() if line]


# --------------------------------------------------------------------------
# model and training (sections 1, 3)
# --------------------------------------------------------------------------


def resolve_arm_dir(data_root: Path, arm: str) -> Path:
    return {"A_off": data_root / "writeback-dqoff", "A_on": data_root / "writeback-dqon",
            "A_bf16": data_root / "source"}[arm]


def resolve_device(device: str, gpu_index: int) -> str:
    """The device string every stage uses. A bare ``cuda`` means GPU 0 to torch, so it is
    mapped to ``cuda:{gpu_index}``: the model, NVML sampling, the box stamp and the peer probe
    then all look at the same GPU. A device that names another index is an error."""
    if device == "cuda":
        return f"cuda:{gpu_index}"
    if device.startswith("cuda:"):
        if device != f"cuda:{gpu_index}":
            raise ValueError(f"--device {device} disagrees with --gpu-index {gpu_index}")
        return device
    if device == "cpu" and gpu_index == 0:
        return "cpu"
    raise ValueError(f"--device {device} with --gpu-index {gpu_index} is not supported")


def parameters_device(model) -> str:
    devices = sorted({str(p.device) for p in model.parameters()})
    if len(devices) != 1:
        raise RuntimeError(f"model parameters are on several devices: {devices}")
    return devices[0]


def assert_on_device(model, device: str) -> str:
    """Every parameter must be on ``device``; returns the device actually used."""
    import torch

    wanted = torch.device(device)
    wrong = sorted({str(p.device) for p in model.parameters() if p.device != wanted})
    if wrong:
        raise RuntimeError(f"model parameters are on {wrong}, not on the requested {device}")
    return str(wanted)


def load_base(model_dir: Path, device: str, experts_implementation: Optional[str]):
    import torch
    from transformers import AutoModelForCausalLM

    kwargs = {} if experts_implementation is None else {
        "experts_implementation": experts_implementation}
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation=common.ATTENTION_IMPLEMENTATION,
        **kwargs)
    model.config.use_cache = False
    model = model.to(device)
    assert_on_device(model, device)
    return model


def build_lora_model(model, recipe: Recipe = RECIPE, gradient_checkpointing: bool = True):
    """Main rule section 4 placement; fp32 trainable factors (peft casts them on wrap)."""
    from peft import LoraConfig, get_peft_model

    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    config = LoraConfig(
        r=recipe.r, lora_alpha=recipe.alpha, lora_dropout=0.0, bias="none",
        target_modules=list(ATTENTION_KINDS),
        target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
    )
    peft_model = get_peft_model(model, config)
    peft_model.train()
    return peft_model


def trainable_report(peft_model, config, r: int) -> Dict[str, Any]:
    import torch

    params = [(n, p) for n, p in peft_model.named_parameters() if p.requires_grad]
    total = sum(p.numel() for _, p in params)
    non_fp32 = [n for n, p in params if p.dtype != torch.float32]
    expected = expected_trainable_params(config, r)
    return {"trainable": total, "expected": expected["total"],
            "matches": total == expected["total"], "non_fp32_factors": non_fp32}


def expected_trainable_names(num_layers: int) -> List[str]:
    """Names of the adapter tensors main rule section 4 puts on the model, peft 0.21 naming:
    per layer A and B on each of q, k, v, o, plus one wrapper per fused expert parameter
    (``base_layer`` is ``gate_up_proj``, the outer wrapper ``down_proj``)."""
    names = []
    for layer in range(num_layers):
        prefix = f"base_model.model.model.layers.{layer}."
        for kind in ATTENTION_KINDS:
            names += [f"{prefix}self_attn.{kind}.lora_{ab}.default.weight" for ab in "AB"]
        names += [f"{prefix}mlp.experts.base_layer.lora_{ab}.default.weight" for ab in "AB"]
        names += [f"{prefix}mlp.experts.lora_{ab}.default.weight" for ab in "AB"]
    return names


def placement_report(peft_model, config) -> Dict[str, Any]:
    """Which tensors train, against the placement of main rule section 4 (rule section 3)."""
    trained = {n for n, p in peft_model.named_parameters() if p.requires_grad}
    expected = set(expected_trainable_names(config.num_hidden_layers))
    return {"matches": trained == expected, "missing": sorted(expected - trained),
            "unexpected": sorted(trained - expected)}


def recipe_mismatches(trainable: Dict[str, Any], placement: Dict[str, Any]) -> List[str]:
    """Why F fails besides a non-finite loss: trainable count, fp32 factors, adapter placement."""
    out = []
    if not trainable["matches"]:
        out.append(f"trainable parameters {trainable['trainable']} != expected "
                   f"{trainable['expected']}")
    if trainable["non_fp32_factors"]:
        out.append(f"fp32 factors: {len(trainable['non_fp32_factors'])} trainable tensor(s) "
                   f"are not fp32, e.g. {trainable['non_fp32_factors'][0]}")
    if not placement["matches"]:
        out.append(f"placement: {len(placement['missing'])} adapter tensor(s) missing "
                   f"(e.g. {(placement['missing'] or ['-'])[0]}), "
                   f"{len(placement['unexpected'])} unexpected "
                   f"(e.g. {(placement['unexpected'] or ['-'])[0]})")
    return out


def response_loss_sum(logits, labels):
    """Sum of fp32 cross-entropy over the response targets of one sequence (shifted)."""
    import torch

    shift_logits = logits[:, :-1, :].float().reshape(-1, logits.shape[-1])
    shift_labels = labels[:, 1:].reshape(-1)
    return torch.nn.functional.cross_entropy(
        shift_logits, shift_labels, ignore_index=split.IGNORE_INDEX, reduction="sum")


def n_targets(example: Dict[str, Any]) -> int:
    return sum(1 for t in example["labels"][1:] if t != split.IGNORE_INDEX)


class DeviceMemorySampler:
    """Whole-device used and free memory, sampled once a second on a thread (rule section 3)."""

    def __init__(self, read: Callable[[], Dict[str, int]], interval_s: float = 1.0) -> None:
        self._read, self._interval = read, interval_s
        self.samples: List[Dict[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                reading = self._read()
                self.samples.append({"t": time.time(), **reading})
            except Exception:  # noqa: BLE001 - a missed sample is not a crash
                pass
            self._stop.wait(self._interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        return False

    def summary(self) -> Dict[str, Any]:
        if not self.samples:
            return {"n": 0, "max_used_bytes": None, "min_free_bytes": None}
        return {"n": len(self.samples),
                "max_used_bytes": max(s["used"] for s in self.samples),
                "min_free_bytes": min(s["free"] for s in self.samples)}


def nvml_reader(gpu: Optional[str] = None) -> Callable[[], Dict[str, int]]:
    """Whole-device memory of ``gpu`` (an index or UUID); default: the GPU the run sees as cuda:0.
    NVML ignores CUDA_VISIBLE_DEVICES, so the physical id is looked up, not assumed to be 0."""
    import pynvml

    gpu = common.workload_gpu_id() if gpu is None else str(gpu)
    if gpu is None:
        raise RuntimeError("no GPU is visible to this run")
    pynvml.nvmlInit()
    handle = (pynvml.nvmlDeviceGetHandleByIndex(int(gpu)) if gpu.isdigit()
              else pynvml.nvmlDeviceGetHandleByUUID(gpu))

    def read() -> Dict[str, int]:
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {"used": int(info.used), "free": int(info.free)}

    return read


def train_updates(peft_model, examples: Sequence[Dict[str, Any]], recipe: Recipe,
                  updates: int, seed: int, device: str, pad_id: int,
                  sync: Callable[[], None] = lambda: None) -> Dict[str, Any]:
    """``updates`` optimizer steps of the recipe. Records the loss and wall time per update."""
    import torch

    peft_model.train()
    params = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, **recipe.adamw_kwargs())
    order = shuffled_order(len(examples), seed)
    per_update = []
    for step in range(updates):
        for group in optimizer.param_groups:
            group["lr"] = recipe.lr * lr_factor(step, recipe.warmup_updates, recipe.total_updates)
        batch = [examples[i] for i in order[step * recipe.accum:(step + 1) * recipe.accum]]
        total_targets = sum(n_targets(e) for e in batch)
        sync()
        started = time.perf_counter()
        loss_sum = 0.0
        for example in batch:
            ids = torch.tensor([example["input_ids"]], device=device)
            labels = torch.tensor([example["labels"]], device=device)
            logits = peft_model(input_ids=ids, use_cache=False).logits
            micro = response_loss_sum(logits, labels)
            (micro / total_targets).backward()
            loss_sum += float(micro.detach())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, recipe.clip_norm))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        sync()
        elapsed = time.perf_counter() - started
        per_update.append({
            "update": step, "lr": optimizer.param_groups[0]["lr"],
            "loss": loss_sum / total_targets, "targets": total_targets,
            "grad_norm": grad_norm, "seconds": elapsed,
            "finite": math.isfinite(loss_sum) and math.isfinite(grad_norm),
        })
    return {"updates": per_update, "all_finite": all(u["finite"] for u in per_update)}


def smoke_nll(model, examples: Sequence[Dict[str, Any]], device: str) -> Dict[str, Any]:
    """Token-weighted response NLL on a set of rows; per-sequence sums in fp64."""
    import torch

    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for example in examples:
            ids = torch.tensor([example["input_ids"]], device=device)
            labels = torch.tensor([example["labels"]], device=device)
            logits = model(input_ids=ids, use_cache=False).logits
            total += float(response_loss_sum(logits, labels).double())
            count += n_targets(example)
    return {"nll": total / count, "targets": count, "rows": len(examples),
            "finite": math.isfinite(total)}


def adapter_state(peft_model) -> Dict[str, Any]:
    return {n: p.detach().cpu().clone() for n, p in peft_model.named_parameters()
            if p.requires_grad}


def adapters_bit_identical(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return set(a) == set(b) and all(quant.bitwise_equal(a[k], b[k]) for k in a)


def frozen_base_hash(peft_model) -> str:
    """One SHA-256 over every frozen parameter, in name order (rule section 8, Frozen base)."""
    digest = common.hashlib.sha256()
    for name, param in sorted(peft_model.named_parameters(), key=lambda kv: kv[0]):
        if not param.requires_grad:
            digest.update(name.encode() + b"=" + common.sha256_tensor(param.data).encode())
    return digest.hexdigest()


def run_f(model_dir: Path, arm: str, examples: Sequence[Dict[str, Any]],
          smoke_examples: Sequence[Dict[str, Any]], recipe: Recipe, device: str, pad_id: int,
          memory_reader: Optional[Callable[[], Dict[str, int]]], adapter_out: Optional[Path],
          experts_implementation: Optional[str] = common.EXPERTS_IMPLEMENTATION) -> Dict[str, Any]:
    """Feasibility of one arm (rule section 3): memory, wall time per update, finiteness."""
    import torch

    cuda = device.startswith("cuda")
    sync = (lambda: torch.cuda.synchronize(device)) if cuda else (lambda: None)
    result: Dict[str, Any] = {"arm": arm, "updates_requested": recipe.f_updates, "oom": False}
    try:
        model = load_base(model_dir, device, experts_implementation)
        result["experts_implementation_resolved"] = model.config._experts_implementation
        result["attention_implementation_resolved"] = model.config._attn_implementation
        result["device_used"] = parameters_device(model)
        peft_model = build_lora_model(model, recipe)
        result["trainable"] = trainable_report(peft_model, model.config, recipe.r)
        result["placement"] = placement_report(peft_model, model.config)
        result["recipe_mismatch"] = recipe_mismatches(result["trainable"], result["placement"])
        base_before = frozen_base_hash(peft_model)
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        sampler_cm = DeviceMemorySampler(memory_reader) if memory_reader else _NoSampler()
        with sampler_cm as sampler:
            trained = train_updates(peft_model, examples, recipe, recipe.f_updates, recipe.seed,
                                    device, pad_id, sync)
        result.update(trained)
        if cuda:
            result["max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
            result["max_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        result["device_memory"] = sampler.summary()
        result["seconds_per_update"] = [u["seconds"] for u in trained["updates"]]
        result["median_seconds_per_update"] = sorted(result["seconds_per_update"])[
            len(result["seconds_per_update"]) // 2]
        result["smoke_nll"] = smoke_nll(peft_model, smoke_examples, device)
        result["frozen_base_unchanged"] = frozen_base_hash(peft_model) == base_before
        state = adapter_state(peft_model)
        if adapter_out is not None:
            peft_model.save_pretrained(adapter_out)
            result["adapter_files"] = {p.name: common.sha256_file(p)
                                       for p in sorted(adapter_out.glob("*")) if p.is_file()}
        result["adapter_tensor_hashes"] = {k: common.sha256_tensor(v) for k, v in state.items()}
        result["failed"] = not (trained["all_finite"] and result["smoke_nll"]["finite"]
                                and result["frozen_base_unchanged"]
                                and not result["recipe_mismatch"])
    except torch.cuda.OutOfMemoryError as exc:
        result.update({"oom": True, "oom_message": str(exc)[:500], "failed": False})
    return result


class _NoSampler:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def summary(self):
        return {"n": 0, "max_used_bytes": None, "min_free_bytes": None}


def decide_final_card(f_40: Dict[str, Dict[str, Any]],
                      f_80: Optional[Dict[str, Dict[str, Any]]] = None,
                      limit: int = F_MEMORY_LIMIT_BYTES) -> Dict[str, Any]:
    """The final-card rule (rule section 3). Inputs: arm -> run result for each card.

    A non-finite loss, a changed frozen base or a recipe mismatch fails F, whatever the card.
    An arm "exceeds" if it OOMed or its peak
    reserved memory is above ``limit`` bytes.
    """
    def missing(results: Dict[str, Dict[str, Any]]) -> List[str]:
        return [a for a in ARMS if a not in results]

    if missing(f_40):
        return {"decision": "INCOMPLETE", "final_card": None, "missing": missing(f_40),
                "reason": "every arm needs a non-void F record before a card is chosen"}

    def exceeds(results: Dict[str, Dict[str, Any]]) -> List[str]:
        return [a for a, r in results.items()
                if r.get("oom") or r.get("max_memory_reserved_bytes", 0) > limit]

    def nonfinite(results: Dict[str, Dict[str, Any]]) -> List[str]:
        return [a for a, r in results.items() if not r.get("oom") and r.get("failed")]

    if nonfinite(f_40):
        return {"decision": "F_FAILS", "reason": f"F failed on {nonfinite(f_40)}",
                "final_card": None}
    over = exceeds(f_40)
    if not over:
        return {"decision": "FINAL_A100_40GB", "final_card": "A100 40 GB", "over": []}
    if f_80 is None:
        return {"decision": "REPEAT_F_ON_A100_80GB", "final_card": None, "over": over}
    if missing(f_80):
        return {"decision": "INCOMPLETE", "final_card": None, "missing": missing(f_80),
                "reason": "the A100 80 GB repeat needs a non-void F record for every arm"}
    if nonfinite(f_80):
        return {"decision": "F_FAILS", "reason": f"F failed on {nonfinite(f_80)}",
                "final_card": None}
    if any(r.get("oom") for r in f_80.values()):
        return {"decision": "STOP_RECIPE_TO_OWNER", "final_card": None,
                "reason": "an arm OOMed on the A100 80 GB"}
    return {"decision": "FINAL_A100_80GB", "final_card": "A100 80 GB", "over": over}


# --------------------------------------------------------------------------
# T: base-weight tolerances (section 4)
# --------------------------------------------------------------------------

#: Cell strings are "<kind>/needs_grad=<bool>/kernel=<fused|fallback|cpu>"; the input-gradient
#: backward is its own operation and has its own per-kind cell "<kind>/backward".


def cell_name(kind: str, needs_grad: bool, kernel: str) -> str:
    return f"{kind}/needs_grad={needs_grad}/kernel={kernel}"


def backward_cell(kind: str) -> str:
    return f"{kind}/backward"


def select_units(units: Sequence[quant.Unit], seed: int = T_SELECTION_SEED, per_kind: int = 8
                 ) -> Dict[str, List[quant.Unit]]:
    """8 units of each kind from one ``random.Random(seed)`` drawn before any is read.

    Population of a kind: its units in enumeration order (layer, then expert). Kinds are drawn
    in the order q, k, v, o, gate, up, down from the one generator.
    """
    rng = random.Random(seed)
    chosen = {}
    for kind in KINDS:
        population = [u for u in units if u.kind == kind]
        chosen[kind] = rng.sample(population, per_kind)
    return chosen


def make_g(shape, device: str):
    import torch

    gen = torch.Generator().manual_seed(common.CAL_SEED)
    return torch.randn(tuple(shape), generator=gen, dtype=torch.float32).to(device)


def max_abs(a, b) -> float:
    return float((a.detach().double().cpu() - b.detach().double().cpu()).abs().max())


def reference_linear32(x, w):
    """fp32 reference of ``x @ w.T`` (and its gradients) under the section 1 settings."""
    import torch

    x32 = x.detach().float().clone().requires_grad_(True)
    w32 = w.float()
    with torch.autocast(device_type=x.device.type, enabled=False):
        y = x32 @ w32.T
    g = make_g(y.shape, str(y.device))
    loss = (y * g).sum()
    loss.backward()
    flags = common.assert_fp32_reference(y, x32.grad, where="T linear fp32")
    return {"output": y.detach(), "input_grad": x32.grad.detach(),
            "loss": float(loss.detach())}, flags


def linear_forward_backward(module, x, grad: bool, probe=None):
    """Run ``module`` on ``x``; with ``grad`` also backward. Returns y, input grad, loss, info."""
    import torch

    if probe is not None:
        probe.reset()
    if grad:
        xin = x.detach().clone().requires_grad_(True)
        y = module(xin)
    else:
        xin = x.detach().clone()
        with torch.no_grad():
            y = module(xin)
    info = probe.take(1) if probe is not None else {
        "kernels": ["cpu"], "shapes": [], "needs_grad_observed": None}
    g = make_g(y.shape, str(y.device))
    loss = (y.float() * g).sum()
    in_grad = None
    if grad:
        loss.backward()
        in_grad = xin.grad.detach()
    return y.detach(), in_grad, float(loss.detach()), info


def run_t_unit_calls(unit: quant.Unit, dq: str, w_dq, make_r, inputs: Sequence[Any],
                     probe=None, device: str = "cuda") -> List[Dict[str, Any]]:
    """Forward and input-gradient records for one unit, one DQ setting, both grad modes."""
    import torch

    out_features, in_features = w_dq.shape
    plain = torch.nn.Linear(in_features, out_features, bias=False).to(device, torch.bfloat16)
    with torch.no_grad():
        plain.weight.copy_(w_dq.to(device))
    plain.weight.requires_grad_(False)
    r_module = make_r()
    records: List[Dict[str, Any]] = []
    for call, x in enumerate(inputs):
        x = x.to(device)
        for grad in (False, True):
            ref, flags = reference_linear32(x, w_dq.to(device))
            ref_finite = bool(torch.isfinite(ref["output"]).all()
                              and torch.isfinite(ref["input_grad"]).all()
                              and math.isfinite(ref["loss"]))
            f_y, f_gx, f_loss, _ = linear_forward_backward(plain, x, grad)
            r_y, r_gx, r_loss, info = linear_forward_backward(r_module, x, grad, probe)
            kernel = info["kernels"][0]
            observed = info["needs_grad_observed"]
            base = {"unit": unit.key, "kind": unit.kind, "dq": dq, "call": call,
                    "mode": "grad" if grad else "nograd", "tokens": int(x.shape[0]),
                    "kernel": kernel, "needs_grad": bool(grad),
                    "needs_grad_matches_probe": observed is None or observed == bool(grad),
                    "ref_finite": ref_finite, "fp32_flags": flags}
            records.append({**base, "quantity": "output", "cell": cell_name(
                unit.kind, bool(grad), kernel),
                "e_F": max_abs(f_y, ref["output"]), "e_R": max_abs(r_y, ref["output"]),
                "L_F": f_loss, "L_R": r_loss, "L_32": ref["loss"]})
            if grad:
                records.append({**base, "quantity": "input_grad",
                                "cell": backward_cell(unit.kind),
                                "forward_cell": cell_name(unit.kind, True, kernel),
                                "e_F": max_abs(f_gx, ref["input_grad"]),
                                "e_R": max_abs(r_gx, ref["input_grad"])})
    return records


def lora_holder(module):
    import torch

    class Holder(torch.nn.Module):
        def __init__(self, proj):
            super().__init__()
            self.proj = proj

        def forward(self, x):
            return self.proj(x)

    return Holder(module)


def run_t_lora(unit: quant.Unit, dq: str, w_dq, make_r, x, recipe: Recipe, probe=None,
               device: str = "cuda") -> List[Dict[str, Any]]:
    """LoRA gradients and one AdamW step on F and R against an fp32 reference (section 4)."""
    import torch
    from peft import LoraConfig, get_peft_model

    out_features, in_features = w_dq.shape
    gen = torch.Generator().manual_seed(common.CAL_SEED)
    a0 = torch.empty(recipe.r, in_features)
    torch.nn.init.kaiming_uniform_(a0, a=math.sqrt(5), generator=gen)
    b0 = torch.randn(out_features, recipe.r, generator=gen) * wref.B_STD
    scaling = recipe.alpha / recipe.r
    records: List[Dict[str, Any]] = []
    for grad_input in (False, True):
        xin = x.to(device)
        # fp32 reference
        a32, b32 = a0.to(device).requires_grad_(True), b0.to(device).requires_grad_(True)
        x32 = xin.float()
        with torch.autocast(device_type=xin.device.type, enabled=False):
            y32 = x32 @ w_dq.to(device).float().T + scaling * (x32 @ a32.T) @ b32.T
        g = make_g(y32.shape, device)
        (y32 * g).sum().backward()
        opt32 = torch.optim.AdamW([a32, b32], **recipe.adamw_kwargs())
        ref = {"A_grad": a32.grad.clone(), "B_grad": b32.grad.clone()}
        opt32.step()
        ref["A_after"], ref["B_after"] = a32.detach().clone(), b32.detach().clone()
        common.assert_fp32_reference(y32, ref["A_grad"], where="T LoRA fp32")
        ref_finite = all(bool(torch.isfinite(ref[k]).all()) for k in ref)
        measured: Dict[str, Dict[str, Any]] = {}
        for which in ("F", "R"):
            if which == "F":
                base = torch.nn.Linear(in_features, out_features, bias=False)
                base = base.to(device, torch.bfloat16)
                with torch.no_grad():
                    base.weight.copy_(w_dq.to(device))
            else:
                base = make_r()
            model = get_peft_model(lora_holder(base), LoraConfig(
                r=recipe.r, lora_alpha=recipe.alpha, lora_dropout=0.0, bias="none",
                target_modules=["proj"]))
            layer = model.base_model.model.proj
            if layer.lora_A["default"].weight.dtype != torch.float32:
                raise RuntimeError("peft did not keep the LoRA factors in fp32")
            with torch.no_grad():
                layer.lora_A["default"].weight.copy_(a0)
                layer.lora_B["default"].weight.copy_(b0)
            xr = xin.detach().clone().requires_grad_(grad_input)
            if probe is not None and which == "R":
                probe.reset()
            y = model(xr)
            info = probe.take(1) if (probe is not None and which == "R") else {
                "kernels": ["cpu" if which == "R" else "dense"], "needs_grad_observed": None}
            (y.float() * make_g(y.shape, device)).sum().backward()
            params = [layer.lora_A["default"].weight, layer.lora_B["default"].weight]
            grads = [p.grad.detach().clone() for p in params]
            torch.optim.AdamW(params, **recipe.adamw_kwargs()).step()
            measured[which] = {"A_grad": grads[0], "B_grad": grads[1],
                               "A_after": params[0].detach().clone(),
                               "B_after": params[1].detach().clone(), "info": info}
        kernel = measured["R"]["info"]["kernels"][0]
        cell = cell_name(unit.kind, bool(grad_input), kernel)
        e = {q: {w: max_abs(measured[w][key], ref[key]) for w in ("F", "R")}
             for q, key in (("A_grad", "A_grad"), ("B_grad", "B_grad"))}
        e["update"] = {w: common.max_or_nan([max_abs(measured[w]["A_after"], ref["A_after"]),
                                             max_abs(measured[w]["B_after"], ref["B_after"])])
                       for w in ("F", "R")}
        for quantity, errs in e.items():
            records.append({"unit": unit.key, "kind": unit.kind, "dq": dq, "mode": "lora",
                            "ref_finite": ref_finite,
                            "grad_input": grad_input, "quantity": quantity, "cell": cell,
                            "kernel": kernel, "needs_grad": grad_input,
                            "e_F": errs["F"], "e_R": errs["R"]})
    return records


finite_max = common.finite_max


def t_nonfinite_reasons(records: Sequence[Dict[str, Any]]) -> List[str]:
    """Void reasons: any non-finite e_F, e_R or reference value voids the T run (section 4)."""
    reasons = []
    for rec in records:
        where = f"{rec.get('unit')} {rec.get('dq')} {rec.get('quantity')}"
        for key in ("e_F", "e_R"):
            if not math.isfinite(rec[key]):
                reasons.append(f"finiteness: T {where} {key} is not finite")
        if rec.get("ref_finite") is False:
            reasons.append(f"finiteness: T {where} reference value is not finite")
    return reasons


def compute_tau(records: Sequence[Dict[str, Any]], exclude_kinds: Sequence[str] = ()
                ) -> Dict[str, Dict[str, float]]:
    """tau(cell, quantity) = 2 x max over calls and both DQ settings of max(e_F, e_R).

    A cell that never occurs gets no entry, and neither does a kind in ``exclude_kinds``
    (fewer than the required contributing units). A non-finite error raises. Returns
    {cell: {quantity: tau}}.
    """
    by_cell: Dict[str, Dict[str, float]] = {}
    for rec in records:
        if rec.get("kind") in exclude_kinds:
            continue
        err = finite_max([rec["e_F"], rec["e_R"]])
        cell = by_cell.setdefault(rec["cell"], {})
        cell[rec["quantity"]] = max(cell.get(rec["quantity"], 0.0), err)
    return {cell: {q: 2.0 * v for q, v in sorted(qs.items())}
            for cell, qs in sorted(by_cell.items())}


def capture_unit_inputs(model, units_by_kind: Dict[str, List[quant.Unit]],
                        token_rows: Sequence[Sequence[int]], calls_per_unit: int, device: str
                        ) -> Dict[str, List[Any]]:
    """Activations at the chosen units while SMOKE rows run through the bf16 model.

    Attention: the full sequence entering the projection. Experts: the tokens routed to that
    expert; the down projection sees ``act(gate(x)) * up(x)`` computed from the bf16 weights.
    A unit that receives no tokens in a row gets no call from that row. Returns {unit key:
    [x (tokens, in) bf16 on CPU]}.
    """
    import torch

    wanted = [u for us in units_by_kind.values() for u in us]
    inputs: Dict[str, List[Any]] = {u.key: [] for u in wanted}
    by_layer: Dict[int, List[quant.Unit]] = {}
    for unit in wanted:
        by_layer.setdefault(unit.layer, []).append(unit)
    handles = []

    def need(unit) -> bool:
        return len(inputs[unit.key]) < calls_per_unit

    for layer_idx, units in by_layer.items():
        layer = model.model.layers[layer_idx]
        for unit in (u for u in units if u.expert is None):
            module = getattr(layer.self_attn, unit.kind)

            def attn_hook(_m, args, _u=unit):
                if need(_u):
                    inputs[_u.key].append(args[0].detach().reshape(-1, args[0].shape[-1])
                                          .cpu().clone())
            handles.append(module.register_forward_pre_hook(attn_hook))
        expert_units = [u for u in units if u.expert is not None]
        if expert_units:
            experts = layer.mlp.experts

            def expert_hook(module, args, _units=expert_units):
                hidden, index = args[0], args[1]
                for unit in _units:
                    if not need(unit):
                        continue
                    rows = (index == unit.expert).any(dim=-1)
                    if not bool(rows.any()):
                        continue
                    x = hidden[rows]
                    if unit.kind == "down_proj":
                        gate, up = torch.nn.functional.linear(
                            x, module.gate_up_proj[unit.expert]).chunk(2, dim=-1)
                        x = module.act_fn(gate) * up
                    inputs[unit.key].append(x.detach().cpu().clone())
            handles.append(experts.register_forward_pre_hook(expert_hook))
    try:
        with torch.no_grad():
            for ids in token_rows:
                if all(not need(u) for u in wanted):
                    break
                model(input_ids=torch.tensor([list(ids)], device=device), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return inputs


class UnitFiles:
    """Reads a unit's written-back weight and packed codes from the Mac-side layout."""

    def __init__(self, data_root: Path) -> None:
        self.root = Path(data_root)
        self._index: Dict[str, Dict[str, str]] = {}

    def _wb_index(self, tag: str) -> Dict[str, str]:
        if tag not in self._index:
            path = self.root / f"writeback-dq{tag}" / "model.safetensors.index.json"
            self._index[tag] = json.loads(path.read_text())["weight_map"]
        return self._index[tag]

    def weight(self, tag: str, unit: quant.Unit):
        from safetensors import safe_open

        fname = self._wb_index(tag)[unit.key]
        with safe_open(str(self.root / f"writeback-dq{tag}" / fname), framework="pt") as h:
            return h.get_tensor(unit.key)

    def codes(self, tag: str, unit: quant.Unit):
        from safetensors import safe_open

        path = self.root / f"codes-dq{tag}" / f"layer-{unit.layer:02d}.safetensors"
        with safe_open(str(path), framework="pt") as handle:
            packed = handle.get_tensor(unit.key)
            qs = {k[len(unit.key) + 2:]: handle.get_tensor(k)
                  for k in handle.keys() if k.startswith(unit.key + "::")}
        return packed, qs


def make_r_factory(packed, qs, in_features: int, out_features: int, device: str):
    """Linear4bit from the transferred codes and QuantState (``Params4bit.from_prequantized``)."""
    import bitsandbytes as bnb
    import torch

    def make():
        module = bnb.nn.Linear4bit(in_features, out_features, bias=False,
                                   compute_dtype=torch.bfloat16, quant_type="nf4")
        module.weight = bnb.nn.Params4bit.from_prequantized(
            data=packed, quantized_stats=qs, requires_grad=False, device=device, module=module)
        return module.to(device)

    return make


def e0_check(r_module, w_dq) -> Dict[str, Any]:
    """E0: CUDA ``dequantize_4bit`` of R's codes against the transferred written-back weight."""
    from bitsandbytes.functional import dequantize_4bit

    back = dequantize_4bit(r_module.weight.data, r_module.weight.quant_state).cpu()
    same = quant.bitwise_equal(back, w_dq.cpu())
    return {"identical": same,
            "max_abs_diff": 0.0 if same else max_abs(back, w_dq)}


MIN_UNITS_FOR_TAU = 4


def summarize_contribution(chosen: Dict[str, List[quant.Unit]], inputs: Dict[str, List[Any]],
                           min_units: int = MIN_UNITS_FOR_TAU) -> Dict[str, Any]:
    """Which chosen units contributed calls (rule section 4): listed, never replaced.

    A kind with fewer than ``min_units`` contributing units gets no tau.
    """
    per_kind = {kind: sum(bool(inputs.get(u.key)) for u in units)
                for kind, units in chosen.items()}
    return {
        "non_contributing": [u.key for units in chosen.values() for u in units
                             if not inputs.get(u.key)],
        "contributing_per_kind": per_kind,
        "min_units_for_tau": min_units,
        "kinds_without_tau": [k for k in KINDS if k in per_kind and per_kind[k] < min_units],
    }


def concat_sources(order: Sequence[quant.Unit], inputs: Dict[str, List[Any]], m: int):
    """Rows for one fallback call: the first captured SMOKE input of distinct sources, in
    ``order``, concatenated until ``m`` rows and truncated in the last source.

    A source is one (layer, expert): gate and up of the same expert see the same routed tokens,
    so only the first of them contributes. No row is used twice. Returns (rows or None, source
    unit keys used, rows obtained). Fewer than ``m`` rows means the pool is too small.
    """
    import torch

    parts, used, seen, total = [], [], set(), 0
    for unit in order:
        if total == m:
            break
        source = (unit.layer, unit.expert)
        captured = inputs.get(unit.key) or []
        if source in seen or not captured:
            continue
        take = min(captured[0].shape[0], m - total)
        parts.append(captured[0][:take])
        used.append(unit.key)
        seen.add(source)
        total += take
    return (torch.cat(parts) if parts else None), used, total


def fallback_targets(chosen: Dict[str, List[quant.Unit]], inputs: Dict[str, List[Any]]):
    """Restrict a T selection to the gate/up units. Every such unit gets the same calls, one per
    target M from ``fallback_m_targets`` (fixed by the pool size), built by ``concat_sources``
    over the gate/up units in (layer, expert) order.

    Returns (kept units, {unit key: [rows per call]}, {unit key: [per-call record of M and
    source units]}). A target M the pool cannot reach stops the run.
    """
    kept = {kind: chosen[kind] for kind in FALLBACK_GATEUP_KINDS}
    order = sorted((u for kind in FALLBACK_GATEUP_KINDS for u in kept[kind]),
                   key=lambda u: (u.layer, u.expert))
    _, _, pool = concat_sources(order, inputs, FALLBACK_M_CAP)
    calls, meta = [], []
    for m in fallback_m_targets(pool):
        x, used, total = concat_sources(order, inputs, m)
        if total < m:
            raise common.CalibrationStopped(
                f"fallback targets: cannot reach M={m} from {total} distinct SMOKE rows; "
                "no row is repeated, so the target is not calibrated")
        calls.append(x)
        meta.append({"M": m, "source_units": used, "rows": total, "pool_rows": pool})
    tiled = {unit.key: list(calls) for kind in FALLBACK_GATEUP_KINDS for unit in kept[kind]}
    sources = {unit.key: [dict(m) for m in meta] for kind in FALLBACK_GATEUP_KINDS
               for unit in kept[kind]}
    return kept, tiled, sources


def run_t(units, files: UnitFiles, bf16_model, smoke_token_rows, recipe: Recipe, device: str,
          probe, per_kind: int = 8, calls_per_unit: int = CALLS_PER_UNIT,
          min_units: int = MIN_UNITS_FOR_TAU, fallback_gateup: bool = False) -> Dict[str, Any]:
    chosen = select_units(units, T_SELECTION_SEED, per_kind)
    inputs = capture_unit_inputs(bf16_model, chosen, smoke_token_rows, calls_per_unit, device)
    target_sources: Dict[str, List[Dict[str, Any]]] = {}
    if fallback_gateup:
        chosen, inputs, target_sources = fallback_targets(chosen, inputs)
    records: List[Dict[str, Any]] = []
    e0: List[Dict[str, Any]] = []
    contribution = summarize_contribution(chosen, inputs, min_units)
    fewer = [k for k, v in inputs.items() if 0 < len(v) < calls_per_unit]
    for kind in KINDS:
        for unit in chosen.get(kind, []):
            for tag in ("off", "on"):
                w_dq = files.weight(tag, unit)
                packed, qs = files.codes(tag, unit)
                make_r = make_r_factory(packed, qs, w_dq.shape[1], w_dq.shape[0], device)
                if not inputs[unit.key]:
                    continue
                new = run_t_unit_calls(unit, tag, w_dq, make_r, inputs[unit.key], probe, device)
                for rec in new:
                    if target_sources:
                        rec["source_units"] = target_sources[unit.key][rec["call"]]["source_units"]
                records += new
                e0.append({"unit": unit.key, "dq": tag, **e0_check(make_r(), w_dq)})
                if kind in ATTENTION_KINDS:
                    for x in inputs[unit.key][:1]:
                        records += run_t_lora(unit, tag, w_dq, make_r, x, recipe, probe, device)
    reasons = t_nonfinite_reasons(records)
    tau = {} if reasons else compute_tau(records, contribution["kinds_without_tau"])
    return {"void_reasons": reasons, **contribution, "tau": tau,
            "device_used": parameters_device(bf16_model),
            "selected_units": {k: [u.key for u in v] for k, v in chosen.items()},
            "calls_per_unit": calls_per_unit, "units_with_fewer_calls": fewer,
            "records": records, "e0": e0,
            "fallback_gateup": fallback_gateup, "fallback_sources": target_sources,
            "target_misses": [r["cell"] for r in records
                              if fallback_gateup and r.get("kernel") != "fallback"],
            "needs_grad_probe_mismatches": [r["unit"] for r in records
                                            if not r.get("needs_grad_matches_probe", True)]}


# --------------------------------------------------------------------------
# W: expert wrapper tolerances (section 5)
# --------------------------------------------------------------------------


def hash_base(experts) -> str:
    return common.sha256_bytes((common.sha256_tensor(experts.gate_up_proj.data)
                                + common.sha256_tensor(experts.down_proj.data)).encode())


def run_w(arm_dirs: Dict[str, Path], smoke_token_rows, recipe: Recipe, device: str,
          layers: Sequence[int] = W_LAYERS, rows: int = W_ROWS,
          experts_implementation: Optional[str] = common.EXPERTS_IMPLEMENTATION
          ) -> Dict[str, Any]:
    import torch

    flags = common.set_fp32_reference_mode()
    per_call: List[Dict[str, Any]] = []
    frozen_failures: List[str] = []
    impls = set()
    attention = set()
    devices = set()
    fp32_void: List[str] = []
    nonfinite_ref: List[str] = []
    for arm, directory in arm_dirs.items():
        if fp32_void:
            break
        model = load_base(directory, device, experts_implementation)
        attention.add(model.config._attn_implementation)
        devices.add(parameters_device(model))
        captured = wref.capture_routed(
            model, tuple(layers),
            [torch.tensor(list(ids), device=device) for ids in smoke_token_rows[:rows]])
        for layer in layers:
            experts = model.model.layers[layer].mlp.experts
            base = {"gate_up_proj": experts.gate_up_proj.data,
                    "down_proj": experts.down_proj.data}
            state = wref.make_adapter_state(
                experts, torch.Generator().manual_seed(common.CAL_SEED))
            state = {k: ({n: t.to(device) for n, t in v.items()} if isinstance(v, dict) else v)
                     for k, v in state.items()}
            for call, routed in enumerate(captured[layer]):
                before = hash_base(experts)
                bf = wref.run_reference(experts, base, routed, state,
                                        make_g_for(routed, device), "bf16", recipe.adamw_kwargs())
                try:
                    f32 = wref.run_reference(experts, base, routed, state,
                                             make_g_for(routed, device), "fp32",
                                             recipe.adamw_kwargs())
                except Exception as exc:  # noqa: BLE001 - any failure of the reference voids W
                    # No silent change of implementation: a tolerance measured against a
                    # different experts implementation than the bf16 reference is not tol_W.
                    fp32_void.append(
                        f"fp32 reference: {arm} layer {layer} call {call}: "
                        f"{type(exc).__name__}: {str(exc)[:200]}")
                    break
                after = hash_base(experts)
                if before != after:
                    frozen_failures.append(f"{arm} layer {layer} call {call}")
                impls.add((bf["experts_implementation"], f32["experts_implementation"]))
                diffs = wref.compare_references(bf, f32)
                for term in wref.nonfinite_terms(bf, f32):
                    nonfinite_ref.append(f"{arm} L{layer} call {call} {term}")
                per_call.append({"arm": arm, "layer": layer, "call": call, "tokens":
                                 int(routed["hidden"].shape[0]),
                                 **diffs})
            if fp32_void:
                break
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    reasons = fp32_void + [f"finiteness: W reference {t} is not finite" for t in nonfinite_ref]
    reasons += nonfinite_reasons(
        "W", {f"{c['arm']} L{c['layer']} call {c['call']} {q}": c[q]
              for c in per_call for q in wref.W_QUANTITIES})
    # A changed frozen base fails W (rule sections 5 and 8): the result is reported, not rerun.
    failed = bool(frozen_failures)
    return {"per_call": per_call, "tol_W": {} if (reasons or failed) else wref.tol_w(per_call),
            "void_reasons": reasons, "failed": failed, "frozen_base_changed": frozen_failures,
            "fp32_flags": flags, "experts_implementations_bf16_fp32": sorted(impls),
            "attention_implementation_resolved": sorted(attention),
            "device_used": sorted(devices), "layers": list(layers), "rows": rows}


def nonfinite_reasons(stage: str, values: Dict[str, float]) -> List[str]:
    """Void reasons for non-finite numbers (rule section 8, Finiteness: T/W/N/P run void)."""
    bad = sorted(k for k, v in values.items() if not math.isfinite(v))
    return [f"finiteness: {stage} {k} is not finite" for k in bad]


def make_g_for(routed, device: str):
    return make_g(routed["hidden"].shape, device)


# --------------------------------------------------------------------------
# N and P (sections 6, 7)
# --------------------------------------------------------------------------


def suite_item_counts() -> Dict[str, int]:
    from soup_cli.eval import gate_suites as gs

    counts = {}
    for name in gs.DEFAULT_GENERAL_SUITE:
        counts[name] = (len(gs.MINI_BENCHMARKS[name]) if name in gs.MINI_BENCHMARKS
                        else len(gs.load_suite_items(name)))
    return counts


def make_gate_generator(model, tokenizer, device: str, max_new_tokens: int = MAX_NEW_TOKENS
                        ) -> Callable[[str], str]:
    """gen(prompt) -> text: the gate-suite wrapper, greedy, no chat template."""
    import torch

    model.eval()

    def gen(prompt: str) -> str:
        text = split.gate_wrapper(prompt)
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

    return gen


def run_n(base_dir: Path, adapter_dir: Path, tokenizer, device: str,
          experts_implementation: Optional[str] = common.EXPERTS_IMPLEMENTATION
          ) -> Dict[str, Any]:
    """One scoring of DEFAULT_GENERAL_SUITE for A_on + the seed-42 adapter."""
    from peft import PeftModel

    base = load_base(base_dir, device, experts_implementation)
    resolved = {"experts_implementation_resolved": base.config._experts_implementation,
                "attention_implementation_resolved": base.config._attn_implementation,
                "device_used": parameters_device(base)}
    model = PeftModel.from_pretrained(base, adapter_dir)
    return score_n(make_gate_generator(model, tokenizer, device), resolved)


def score_n(gen: Callable[[str], str], resolved: Dict[str, Any]) -> Dict[str, Any]:
    """The N result for a generator: scores of ``DEFAULT_GENERAL_SUITE`` and what was scored."""
    from soup_cli.eval import gate_suites as gs

    scores = {name: float(gs.score_bundled_suite(name, gen))
              for name in sorted(gs.DEFAULT_GENERAL_SUITE)}
    return {"void_reasons": nonfinite_reasons("N", scores), "scores": scores, **resolved,
            "scorer_revision": gs.BUNDLED_SCORER_REVISION,
            "n_items": suite_item_counts(), "max_new_tokens": MAX_NEW_TOKENS,
            "wrapper_sha256": common.sha256_bytes(split.WRAPPER_NO_CONTEXT.encode())}


def aggregate_n(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """floor(suite) = max - min of the scores; epsilon = max(2 x floor, 1 / n_items)."""
    if len(runs) != N_REPEATS:
        raise ValueError(f"N needs {N_REPEATS} scorings, got {len(runs)}")
    out: Dict[str, Any] = {}
    for suite in sorted(runs[0]["scores"]):
        scores = [r["scores"][suite] for r in runs]
        n_items = runs[0]["n_items"][suite]
        floor = finite_max(scores) - min(scores)
        out[suite] = {"scores": scores, "floor": floor, "n_items": n_items,
                      "epsilon": max(2 * floor, 1.0 / n_items), "comparison": "strict >"}
    return out


def finalize_p_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """P (rule sections 7, 8): a non-finite loss voids the run (one rerun); only a changed
    frozen base fails it."""
    reasons = []
    if not result.get("all_finite", True):
        reasons.append("finiteness: P an update loss or gradient norm is not finite")
    if not result.get("smoke_nll", {}).get("finite", True):
        reasons.append("finiteness: P the SMOKE NLL is not finite")
    result["void_reasons"] = reasons
    result["failed"] = not result.get("frozen_base_unchanged", True)
    return result


def compare_repeat(first: Dict[str, Any], second: Dict[str, Any]) -> Dict[str, Any]:
    """P: SMOKE NLL of both runs, their difference, adapter bit-identity."""
    return {"nll_first": first["smoke_nll"]["nll"], "nll_second": second["smoke_nll"]["nll"],
            "difference": second["smoke_nll"]["nll"] - first["smoke_nll"]["nll"],
            "adapters_bit_identical": first["adapter_tensor_hashes"] == second[
                "adapter_tensor_hashes"],
            "note": "one paired observation, not an estimate of GPU nondeterminism"}


# --------------------------------------------------------------------------
# one attempt = one process = one journal entry
# --------------------------------------------------------------------------


def verify_files(root: Path, expected: Dict[str, Dict[str, Any]], relative_names: Sequence[str]
                 ) -> List[str]:
    """SHA-256 of the named files under ``root`` against the manifest; void reasons."""
    reasons = []
    for name in relative_names:
        want = expected.get(name)
        path = root / name
        if want is None:
            reasons.append(f"inputs: {name} is not in the manifest")
        elif not path.exists():
            reasons.append(f"inputs: {name} is missing")
        elif common.sha256_file(path) != want["sha256"]:
            reasons.append(f"inputs: {name} SHA-256 differs from the manifest")
    return reasons


MODEL_FILE_NAMES = ("model.safetensors.index.json", "config.json", "generation_config.json")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def _model_files(manifest: Dict[str, Any], directory: str) -> List[tuple]:
    """Files ``from_pretrained(directory)`` reads: shards, index, config, generation config."""
    out = []
    if directory == "source":
        for name in manifest["source_files"]:
            if name.endswith(".safetensors") or name in MODEL_FILE_NAMES:
                out.append(("source_files", f"source/{name}"))
        return out
    for name in manifest["files"]:
        head, _, rest = name.partition("/")
        if head == directory and (rest.endswith(".safetensors") or rest in MODEL_FILE_NAMES):
            out.append(("files", name))
    return out


def loaded_files(stage: str, arm: Optional[str], manifest: Dict[str, Any]) -> List[tuple]:
    """(manifest section, path under the data root) of every file this run loads."""
    dirs = {"A_off": ["writeback-dqoff"], "A_on": ["writeback-dqon"], "A_bf16": ["source"]}
    if stage == "T":
        wanted = ["source", "writeback-dqoff", "writeback-dqon"]
    elif stage == "W":
        wanted = ["writeback-dqoff", "writeback-dqon"]
    else:
        wanted = dirs["A_on" if stage in ("N", "P") else (arm or "A_on")]
    found: List[tuple] = []
    for directory in wanted:
        found += _model_files(manifest, directory)
    if stage == "T":
        found += [("files", n) for n in manifest["files"] if n.startswith(
            ("codes-dqoff/", "codes-dqon/"))]
    found += [("source_files", f"source/{n}") for n in TOKENIZER_FILES
              if n in manifest["source_files"]]
    return sorted(set(found), key=lambda t: t[1])


def run_input_check(data_root: Path, manifest: Dict[str, Any], stage: str, arm: Optional[str],
                    manifest_sha256: str, pinned_sha256: str = common.QUANT_MANIFEST_SHA256
                    ) -> List[str]:
    """Void reasons from the inputs.

    ``manifest`` is the parsed manifest the run was given and ``manifest_sha256`` the hash of
    its bytes; that manifest decides which files are rehashed, so it must be the one pinned in
    the rule and the one the pull verified. Then the instance-wide record must exist (the pull
    verified all files once); only the files this run loads are hashed again.
    """
    if manifest_sha256 != pinned_sha256:
        return [f"inputs: the supplied manifest (SHA-256 {manifest_sha256}) is not the pinned "
                f"manifest {pinned_sha256}"]
    record_path = data_root / transfer.VERIFIED_NAME
    if not record_path.exists():
        return [f"inputs: {transfer.VERIFIED_NAME} is missing; the instance was not verified "
                "by the transfer pull"]
    record = json.loads(record_path.read_text())
    if record.get("manifest_sha256") != manifest_sha256:
        return ["inputs: the verification record is for another manifest than the supplied one"]
    manifest_path = data_root / transfer.MANIFEST_NAME
    if manifest_path.exists() and common.sha256_file(manifest_path) != manifest_sha256:
        return ["inputs: the manifest in the data root differs from the supplied one"]
    reasons = []
    for section, rel in loaded_files(stage, arm, manifest):
        key = rel[len("source/"):] if section == "source_files" else rel
        want = manifest[section].get(key)
        path = data_root / rel
        if want is None:
            reasons.append(f"inputs: {rel} is not in the manifest")
        elif not path.is_file():
            reasons.append(f"inputs: {rel} is missing")
        elif common.sha256_file(path) != want["sha256"]:
            reasons.append(f"inputs: {rel} SHA-256 differs from the manifest")
    return reasons


def execute_attempt(journal: common.AttemptsJournal, run_id: str, kind: str,
                    out_dir: Path, body: Callable[[], Dict[str, Any]], *,
                    input_check: Callable[[], List[str]] = lambda: [],
                    require_gpu: bool = True, gpu_index: int = 0,
                    instance: Optional[str] = None,
                    stamp: Callable[..., Dict[str, Any]] = common.box_stamp,
                    prov: Callable[[], Dict[str, Any]] = common.provenance,
                    identify_own: Optional[Callable[[], Dict[str, Any]]] = None,
                    expected_instance: Optional[Dict[str, str]] = None,
                    dataset_revision: Optional[str] = None,
                    card_prefix: Optional[str] = None,
                    require_identity: bool = False,
                    dry_run: Optional[Dict[str, Any]] = None
                    ) -> Dict[str, Any]:
    """Run ``body`` as one journaled attempt and write its record, void or not.

    ``expected_instance`` ({"gpu_uuid", "instance_id"}, from decision.json) is the instance of
    the chosen F records; a run whose box stamps name another one is a void (rule section 3).
    ``card_prefix`` ("F" or "F80") makes a box stamp outside that card class a void whose
    reason starts with ``card:``; ``drive_run`` turns that into ``CalibrationStopped``.
    ``require_identity`` makes a missing field of ``record["identity"]`` (GPU name, compute
    capability, memory.total, driver, CUDA, torch, bnb, transformers, commit, dataset revision,
    instance id) a void. The record always carries the identity; the CLI paths set the flag.

    The record is written to ``out_dir/<attempt_id>.json`` before the journal is closed, so a
    void run always keeps its record. A crash is journaled as a void and re-raised.

    ``dry_run`` ({"fake_card", ...}) marks the record as a dry run. A dry run does not void on
    a dirty tree or on library versions that differ from the rule, and lists both as skipped.
    """
    attempt = journal.start(run_id, kind)
    out_dir.mkdir(parents=True, exist_ok=True)
    peers: Dict[str, Any] = {"own_pids": [os.getpid()], "method": "not applicable (no GPU)"}
    peers_error: Optional[str] = None
    if require_gpu:
        try:  # before the stamp: the probe creates the CUDA context the stamp then sees
            peers = (identify_own or (lambda: common.identify_own_gpu_pids_cuda(gpu_index)))()
        except common.PeersUnidentifiableError as exc:
            peers_error = str(exc)
            peers = {"own_pids": [], "method": "allocation probe failed", "error": peers_error}
    provenance = prov()
    versions = common.library_versions()
    stamp_before = stamp(gpu_index, instance)
    record: Dict[str, Any] = {
        "run_id": run_id, "kind": kind, "attempt_id": attempt,
        "attempt_number": len(journal.attempts(run_id)),
        "record_source": "run", "provenance": provenance, "stack": versions, "peers": peers,
        "box_stamp_before": stamp_before,
        "identity": common.run_identity(stamp_before, versions, provenance, dataset_revision),
    }
    if dry_run is not None:
        record["dry_run"] = dry_run
    reasons = common.validate_provenance(provenance)
    if peers_error:
        reasons.append(f"peers: cannot identify this process's GPU context: {peers_error}")
    if versions["mismatches"]:
        reasons.append(f"provenance: library versions differ from the rule: "
                       f"{versions['mismatches']}")
    if dry_run is not None:
        skipped = [r for r in reasons if r.startswith((
            "provenance: the repository tree", "provenance: library versions"))]
        reasons = [r for r in reasons if r not in skipped]
        dry_run["skipped_checks"] = skipped or ["none triggered"]
    if require_gpu and require_identity:
        reasons += common.identity_problems(record["identity"])
    if card_prefix is not None:
        reasons += [f"card: {problem} (before the run)"
                    for problem in common.card_class_problems(stamp_before, card_prefix)]
    reasons += input_check()
    result: Dict[str, Any] = {}
    try:
        if not reasons:
            result = body()
        record["box_stamp_after"] = stamp(gpu_index, instance)
    except BaseException as exc:
        record["box_stamp_after"] = None
        record["crash"] = f"{type(exc).__name__}: {str(exc)[:2000]}"
        record["failure_kind"] = "oom" if type(exc).__name__ == "OutOfMemoryError" else "crash"
        crash_reasons = [f"crash: {type(exc).__name__}: {exc}"[:500]]
        record.update({"outcome": common.OUTCOME_VOID, "void_reasons": crash_reasons})
        (out_dir / f"{attempt}.json").write_text(json.dumps(record, indent=1, sort_keys=True,
                                                             default=str))
        journal.end(attempt, common.OUTCOME_VOID, crash_reasons)
        raise
    own = [os.getpid()] + list(peers["own_pids"])
    reasons += common.validate_run_stamps(record["box_stamp_before"], record["box_stamp_after"],
                                          own, require_gpu)
    if card_prefix is not None and record["box_stamp_after"] is not None:
        reasons += [f"card: {problem} (after the run)" for problem in
                    common.card_class_problems(record["box_stamp_after"], card_prefix)]
    if expected_instance is not None:
        reasons += instance_mismatches(record["box_stamp_before"], record["box_stamp_after"],
                                       expected_instance)
    reasons += list(result.pop("void_reasons", []))
    outcome = (common.OUTCOME_VOID if reasons else
               common.OUTCOME_FAILED if result.get("failed") else common.OUTCOME_OK)
    record.update({"result": result, "outcome": outcome, "void_reasons": reasons})
    (out_dir / f"{attempt}.json").write_text(json.dumps(record, indent=1, sort_keys=True,
                                                         default=str))
    journal.end(attempt, outcome, reasons)
    return record


def instance_mismatches(before: Dict[str, Any], after: Optional[Dict[str, Any]],
                        expected: Dict[str, str]) -> List[str]:
    """Void reasons for a stamp that names another GPU or instance than the chosen F's."""
    reasons = []
    for label, stamp in (("before", before), ("after", after)):
        for key in ("gpu_uuid", "instance_id"):
            found = (stamp or {}).get(key)
            if found != expected.get(key):
                reasons.append(f"provenance: {key} {found!r} {label} the run is not the "
                               f"chosen F's {expected.get(key)!r}")
    return reasons


def _signal_name(returncode: int) -> Optional[str]:
    import signal

    if returncode >= 0:
        return None
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return f"signal {-returncode}"


def write_driver_record(journal: common.AttemptsJournal, attempt_id: str, run_id: str,
                        runs_dir: Path, returncode: int,
                        stamp: Callable[[], Dict[str, Any]],
                        identity: Callable[[Dict[str, Any]], Dict[str, Any]],
                        dry_run: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The record of an attempt whose process died before it wrote its own (SIGKILL, the host's
    OOM killer, a preemption, an exit during start-up). Written by the driver, from the
    driver's own box stamp, so no attempt lacks a record. ``record_source`` says who wrote it."""
    kind = next((e["kind"] for e in journal.events()
                 if e["event"] == "start" and e["attempt_id"] == attempt_id), None)
    now = stamp()
    signal_name = _signal_name(returncode)
    reasons = next((e["reasons"] for e in reversed(journal.events())
                    if e["event"] == "end" and e["attempt_id"] == attempt_id), [])
    record: Dict[str, Any] = {
        "run_id": run_id, "kind": kind, "attempt_id": attempt_id,
        "attempt_number": len(journal.attempts(run_id)),
        "record_source": "driver", "outcome": common.OUTCOME_VOID, "void_reasons": reasons,
        "killed": {"returncode": returncode, "signal": signal_name,
                   "note": "the child process ended without closing its attempt"},
        "failure_kind": "killed", "box_stamp_before": None, "box_stamp_after": now,
        "identity": identity(now),
    }
    if dry_run is not None:
        record["dry_run"] = dry_run
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"{attempt_id}.json").write_text(json.dumps(record, indent=1, sort_keys=True,
                                                            default=str))
    return record


def drive_child(journal: common.AttemptsJournal, run_id: str, argv: List[str],
                runs_dir: Optional[Path] = None,
                stamp: Callable[[], Dict[str, Any]] = lambda: common.box_stamp(),
                identity: Callable[[Dict[str, Any]], Dict[str, Any]] = lambda s: {},
                dry_run: Optional[Dict[str, Any]] = None) -> int:
    """Run one child process for ``run_id``; a child that dies without closing its attempt is
    a void (the journal closes it) and, with ``runs_dir``, gets a record from the driver."""
    journal.close_interrupted(run_id)
    journal.assert_may_start(run_id)
    code = subprocess.run([sys.executable, *argv], check=False).returncode
    for attempt_id in journal.close_interrupted(run_id):
        print(f"{run_id}: attempt {attempt_id} ended without a record (exit code {code}); "
              "journaled as a void", file=sys.stderr)
        if runs_dir is not None and not (runs_dir / f"{attempt_id}.json").exists():
            write_driver_record(journal, attempt_id, run_id, runs_dir, code, stamp, identity,
                                dry_run)
    return code


def completed(journal: common.AttemptsJournal, run_id: str) -> bool:
    """True when ``run_id`` already has an attempt that finished ok or failed."""
    return any(a["outcome"] in (common.OUTCOME_OK, common.OUTCOME_FAILED)
               for a in journal.attempts(run_id))


def drive_run(journal: common.AttemptsJournal, run_id: str, argv: List[str],
              stop_on_failed: bool = False, **child) -> Dict[str, Any]:
    """Run ``run_id`` in fresh processes: a void is rerun once, a second void stops calibration.

    A failed run is never rerun. With ``stop_on_failed`` (T, W, N, P) it also stops
    calibration, which has failed (rule section 8); F reports its failures through the
    final-card decision instead. A void whose reason is a card mismatch (``card:``) is never
    rerun: it stops calibration at once (calibration rule, clarification 2026-10-20). The one
    rerun of a void is announced on stderr; it is the rule's, not a silent retry.
    """
    while True:
        drive_child(journal, run_id, argv, **child)
        attempts = journal.attempts(run_id)
        if not attempts:
            raise RuntimeError(f"{run_id}: the child wrote no journal entry")
        last = attempts[-1]
        if last["outcome"] == common.OUTCOME_VOID and any(
                r.startswith("card:") for r in last["reasons"]):
            raise common.CalibrationStopped(
                f"{run_id}: the card is not the rule's ({last['reasons']}); calibration stops")
        if last["outcome"] == common.OUTCOME_FAILED and stop_on_failed:
            raise common.CalibrationStopped(
                f"{run_id}: failed ({last.get('reasons') or 'see its record'}); "
                "calibration fails and is reported")
        if last["outcome"] in (common.OUTCOME_OK, common.OUTCOME_FAILED):
            return last
        if journal.void_count(run_id) >= 2:
            raise common.CalibrationStopped(
                f"{run_id}: second void ({last['reasons']}); calibration stops")
        print(f"{run_id}: void ({last['reasons']}); the rule allows one rerun, "
              "starting it now", file=sys.stderr)


def write_sha256sums(directory: Path) -> int:
    lines = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{common.sha256_file(path)}  {path.relative_to(directory).as_posix()}\n")
    (directory / "SHA256SUMS").write_text("".join(lines))
    return len(lines)


# --------------------------------------------------------------------------
# records, decision, report
# --------------------------------------------------------------------------


def find_record(out: Path, run_id: str) -> Optional[Dict[str, Any]]:
    """The record of the last attempt of ``run_id`` that finished ok or failed (not void)."""
    journal = common.AttemptsJournal(out / "attempts.jsonl")
    for attempt in reversed(journal.attempts(run_id)):
        if attempt["outcome"] in (common.OUTCOME_OK, common.OUTCOME_FAILED):
            return json.loads((out / "runs" / f"{attempt['attempt_id']}.json").read_text())
    return None


def f_results(out: Path, prefix: str = "F") -> Dict[str, Dict[str, Any]]:
    found = {}
    for arm in ARMS:
        record = find_record(out, f"{prefix}_{arm}")
        if record is not None:
            found[arm] = record["result"]
    return found


def expected_run_ids(prefix: str = "F") -> List[str]:
    return ([f"{prefix}_{a}" for a in ARMS] + ["T", "W"]
            + [f"N_{i}" for i in range(1, N_REPEATS + 1)] + ["P"])


MAIN_PLAN_RUNS = 9
MAIN_PLAN_RERUNS = 1
MAIN_PLAN_UPDATES = 256
MAIN_PLAN_SEEDS = 3


def project_main_plan(medians: Dict[str, Optional[float]]) -> Optional[Dict[str, Any]]:
    """Wall time of the main plan from F's median seconds per update (rule section 3).

    Nine runs (three seeds of each arm) plus the on/42 rerun, 256 updates each. Training
    only: loading, scoring and the DECISION pass are not in it.
    """
    if any(medians.get(arm) is None for arm in ARMS):
        return None
    per_seed = sum(medians[arm] for arm in ARMS)
    seconds = MAIN_PLAN_UPDATES * (MAIN_PLAN_SEEDS * per_seed + medians["A_on"])
    return {"runs": MAIN_PLAN_RUNS, "reruns": MAIN_PLAN_RERUNS,
            "updates_per_run": MAIN_PLAN_UPDATES, "median_seconds_per_update": dict(medians),
            "rerun_arm": "A_on", "total_seconds": seconds, "total_hours": seconds / 3600,
            "basis": "median seconds per update of the F runs; training updates only"}


def assemble_report(out: Path, prefix: str = "F") -> Dict[str, Any]:
    """The values the main rule copies (rule section 9), from the records on disk."""
    journal = common.AttemptsJournal(out / "attempts.jsonl")
    f = f_results(out, prefix)
    report: Dict[str, Any] = {"f": {a: {k: r.get(k) for k in (
        "oom", "failed", "max_memory_allocated_bytes", "max_memory_reserved_bytes",
        "median_seconds_per_update", "device_memory", "experts_implementation_resolved",
        "attention_implementation_resolved")} for a, r in f.items()}}
    report["projected_main_plan"] = project_main_plan(
        {arm: f.get(arm, {}).get("median_seconds_per_update") for arm in ARMS})
    report["final_card_decision"] = (cal_decision(out) if (out / "decision.json").exists()
                                     else None)
    t = find_record(out, "T")
    if t is not None:
        stamp = t["box_stamp_before"]
        report["final_card"] = {"gpu_name": stamp.get("gpu_name"),
                                "gpu_uuid": stamp.get("gpu_uuid"),
                                "instance_id": stamp.get("instance_id")}
        report["tau"] = t["result"]["tau"]
        report["T_calls_per_unit"] = t["result"]["calls_per_unit"]
    w = find_record(out, "W")
    if w is not None:
        report["tol_W"] = w["result"]["tol_W"]
    n_records = [find_record(out, f"N_{i}") for i in range(1, N_REPEATS + 1)]
    if all(n_records):
        report["noise_floor"] = aggregate_n([r["result"] for r in n_records])
    p = find_record(out, "P")
    if p is not None:
        report["repeatability"] = p["result"].get("comparison")
    report["failed_runs"] = [
        {"run_id": rid, **a} for rid in expected_run_ids(prefix)
        for a in journal.attempts(rid) if a["outcome"] == common.OUTCOME_FAILED]
    report["voids"] = [
        {"run_id": rid, **a} for rid in expected_run_ids(prefix)
        for a in journal.attempts(rid) if a["outcome"] == common.OUTCOME_VOID]
    report["provenance_gaps"] = journal.provenance_gap(expected_run_ids(prefix))
    return report


def cal_decision(out: Path) -> Dict[str, Any]:
    return json.loads((out / "decision.json").read_text())


FINAL_CARD_PREFIX = {"FINAL_A100_40GB": "F", "FINAL_A100_80GB": "F80"}


def final_instance(out: Path, prefix: str) -> Dict[str, str]:
    """The GPU UUID and instance id of the F records of ``prefix`` (rule section 3: T, W, N and
    P run on the final card's instance only). The three arms must agree."""
    seen = set()
    for arm in ARMS:
        record = find_record(out, f"{prefix}_{arm}")
        if record is None:
            raise common.CalibrationStopped(f"{prefix}_{arm}: no F record to take the instance "
                                            "from")
        stamp = record["box_stamp_before"]
        seen.add((stamp.get("gpu_uuid"), stamp.get("instance_id")))
    if len(seen) != 1:
        raise common.CalibrationStopped(
            f"the {prefix} arms ran on different GPUs or instances: {sorted(seen, key=str)}")
    gpu_uuid, instance_id = seen.pop()
    if not gpu_uuid or not instance_id:
        raise common.CalibrationStopped(f"the {prefix} records carry no GPU UUID or instance id")
    return {"gpu_uuid": gpu_uuid, "instance_id": instance_id}


def resolve_final_instance(out: Path) -> Dict[str, str]:
    """The instance the rest stages must run on, from ``decision.json``."""
    instance = json.loads((Path(out) / "decision.json").read_text()).get("final_instance")
    if not instance or not instance.get("gpu_uuid") or not instance.get("instance_id"):
        raise common.CalibrationStopped(
            "decision.json has no final_instance; run decide again on the F records")
    return instance


def resolve_f_prefix(out: Path, explicit: Optional[str]) -> str:
    """The F record prefix of the final card, from ``decision.json`` (rule section 3).

    T, W, N and P run on the final card's instance and N and P load its seed-42 adapter, so
    they must read ``F`` after a 40 GB decision and ``F80`` after the 80 GB repeat. No decision,
    a decision that is not a final card, or an explicit prefix that disagrees is an error.
    """
    path = Path(out) / "decision.json"
    if not path.exists():
        raise common.CalibrationStopped(
            f"{path} is missing: run `drive --stage F` (and `decide`) before the other stages")
    decision = json.loads(path.read_text()).get("decision")
    if decision not in FINAL_CARD_PREFIX:
        raise common.CalibrationStopped(
            f"decision.json says {decision}, which is not a final card; no stage after F runs")
    derived = FINAL_CARD_PREFIX[decision]
    if explicit is not None and explicit != derived:
        raise common.CalibrationStopped(
            f"--f-prefix {explicit} disagrees with decision.json ({decision} means {derived})")
    return derived


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _cmd_split(args) -> int:
    record = split.run_split(args.out, args.data_root / "source", args.dataset_revision)
    return 0 if record["funnel"]["eligible"] >= split.MIN_ELIGIBLE else 1


def _load_rows(args, tokenizer):
    record = json.loads((args.split_dir / "split-record.json").read_text())
    dolly = split.load_dolly(record["dataset_revision"])
    rows = {}
    for name in ("SMOKE", "TRAIN"):
        meta = record["manifests"][name]
        manifest = read_manifest(args.split_dir / meta["file"], meta["sha256"])
        rows[name] = encode_manifest_rows(manifest, dolly, tokenizer)
    return rows


def _body(args, run_id, tokenizer, data_root, device, out, attempt_hint):
    import torch

    rows = _load_rows(args, tokenizer)
    torch.manual_seed(args.recipe_seed)
    if args.stage in ("F", "P"):
        arm = "A_on" if args.stage == "P" else args.arm
        memory = nvml_reader() if device.startswith("cuda") else None
        adapter_out = out / "adapters" / f"{run_id}-{attempt_hint}"
        result = run_f(resolve_arm_dir(data_root, arm), arm, rows["TRAIN"], rows["SMOKE"], RECIPE,
                       device, tokenizer.pad_token_id, memory, adapter_out)
        result["adapter_dir"] = str(adapter_out)
        if args.stage == "P":
            finalize_p_result(result)
            first = find_record(out, f"{args.f_prefix}_A_on")
            if first is None:
                raise RuntimeError("P needs the F A_on record to compare with")
            result["comparison"] = compare_repeat(first["result"], result)
        return result
    token_rows = [r["input_ids"] for r in rows["SMOKE"]]
    if args.stage == "T":
        from e4_part2_gemm_probe import GemmProbe

        common.set_fp32_reference_mode()
        bf16 = load_base(resolve_arm_dir(data_root, "A_bf16"), device,
                         common.EXPERTS_IMPLEMENTATION)
        with GemmProbe() as probe:
            result = run_t(quant.enumerate_units(), UnitFiles(data_root), bf16, token_rows,
                           RECIPE, device, probe,
                           fallback_gateup=bool(getattr(args, "fallback_gateup", False)))
        result["experts_implementation_resolved"] = bf16.config._experts_implementation
        result["attention_implementation_resolved"] = bf16.config._attn_implementation
        return result
    if args.stage == "W":
        return run_w({"A_off": resolve_arm_dir(data_root, "A_off"),
                      "A_on": resolve_arm_dir(data_root, "A_on")}, token_rows, RECIPE, device)
    if args.stage == "N":
        first = find_record(out, f"{args.f_prefix}_A_on")
        if first is None:
            raise RuntimeError("N needs the seed-42 A_on adapter from the F run")
        return run_n(resolve_arm_dir(data_root, "A_on"), Path(first["result"]["adapter_dir"]),
                     tokenizer, device)
    raise SystemExit(f"stage {args.stage} is not runnable")


def is_dry(args) -> bool:
    return bool(getattr(args, "dry_run", False))


def guard_out(out: Path, dry_run: bool) -> None:
    """Dry-run and real records never share an ``--out``."""
    problem = common.dry_run_mix_problem(out, dry_run)
    if problem:
        raise common.CalibrationStopped(problem)


def out_is_dry(out: Path) -> bool:
    return common.dry_run_mix_problem(out, False) is not None


def dataset_revision_of(split_dir: Path) -> Optional[str]:
    path = Path(split_dir) / "split-record.json"
    return json.loads(path.read_text()).get("dataset_revision") if path.exists() else None


def _cmd_run(args) -> int:
    if is_dry(args):
        import e4_part2_dryrun as dryrun

        return dryrun.cmd_run(sys.modules[__name__], args)
    from transformers import AutoTokenizer

    out = args.out
    guard_out(out, dry_run=False)
    if args.stage != "F":
        args.f_prefix = resolve_f_prefix(out, args.f_prefix)
    expected_instance = resolve_final_instance(out) if args.stage != "F" else None
    device = resolve_device(args.device, args.gpu_index)
    problem = nvml_problem() if device.startswith("cuda") else None
    if problem:
        print(f"calibration stopped: {problem}", file=sys.stderr)
        return 6
    problem = setup_problem(args.data_root, args.manifest)
    if problem:
        print(f"calibration stopped: {problem}", file=sys.stderr)
        return 6
    if device.startswith("cuda"):
        import torch

        common.require_single_gpu(args.gpu_index)
        torch.cuda.set_device(args.gpu_index)  # default-device calls agree with the model
    journal = common.AttemptsJournal(out / "attempts.jsonl")
    suffix = {"F": f"{args.f_prefix}_{args.arm}", "N": f"N_{args.repeat}"}.get(args.stage,
                                                                          args.stage)
    run_id = args.run_id or suffix
    manifest_bytes = args.manifest.read_bytes()
    manifest_sha256 = common.sha256_bytes(manifest_bytes)
    manifest = json.loads(manifest_bytes)
    tokenizer = AutoTokenizer.from_pretrained(args.data_root / "source")

    def input_check() -> List[str]:
        return run_input_check(args.data_root, manifest, args.stage, args.arm, manifest_sha256)

    cuda = device.startswith("cuda")
    record = execute_attempt(
        journal, run_id, args.stage, out / "runs",
        lambda: _body(args, run_id, tokenizer, args.data_root, device, out, int(time.time())),
        input_check=input_check, require_gpu=cuda,
        gpu_index=args.gpu_index, instance=args.instance_id,
        expected_instance=expected_instance,
        dataset_revision=dataset_revision_of(args.split_dir),
        card_prefix=args.f_prefix if cuda else None, require_identity=True)
    print(json.dumps({"run_id": run_id, "outcome": record["outcome"],
                      "void_reasons": record["void_reasons"]}))
    return 3 if record["outcome"] == common.OUTCOME_VOID else 0


def _child_argv(args, stage: str, arm: Optional[str] = None, repeat: Optional[int] = None
                ) -> List[str]:
    argv = [str(Path(__file__).resolve()), "run", "--stage", stage, "--out", str(args.out),
            "--gpu-index", str(args.gpu_index), "--f-prefix", args.f_prefix]
    if is_dry(args):
        argv += ["--dry-run", "--fake-card", args.fake_card,
                 "--fake-reserved-bytes", str(args.fake_reserved_bytes)]
        if args.inject:
            argv += ["--inject", args.inject]
    else:
        argv += ["--data-root", str(args.data_root), "--split-dir", str(args.split_dir),
                 "--manifest", str(args.manifest), "--device", args.device]
    if arm:
        argv += ["--arm", arm]
    if repeat:
        argv += ["--repeat", str(repeat)]
    if args.instance_id:
        argv += ["--instance-id", args.instance_id]
    return argv


def _drive_kwargs(args) -> Dict[str, Any]:
    """What ``drive_run`` needs to write a record for an attempt whose process was killed."""
    if is_dry(args):
        import e4_part2_dryrun as dryrun

        stamp = lambda: dryrun.fake_stamp(args.fake_card, args.instance_id)  # noqa: E731
        revision = dryrun.DRY_DATASET_REVISION
        marker = dryrun.dry_marker(args.fake_card, args.inject)
    else:
        stamp = lambda: common.box_stamp(args.gpu_index, args.instance_id)  # noqa: E731
        revision = dataset_revision_of(args.split_dir)
        marker = None

    def identity(now: Dict[str, Any]) -> Dict[str, Any]:
        return common.run_identity(now, common.library_versions(), common.provenance(),
                                   revision)

    return {"runs_dir": args.out / "runs", "stamp": stamp, "identity": identity,
            "dry_run": marker}


def _drive_stage(args, journal: common.AttemptsJournal, stage: str, run_id: str,
                 arm: Optional[str] = None, repeat: Optional[int] = None,
                 skip_completed: bool = False) -> None:
    if skip_completed and completed(journal, run_id):
        print(f"{run_id}: already has a completed attempt; not rerun")
        return
    drive_run(journal, run_id, _child_argv(args, stage, arm, repeat),
              stop_on_failed=stage != "F", **_drive_kwargs(args))


def _rest_plan() -> List[tuple]:
    plan: List[tuple] = [("T", None, None), ("W", None, None)]
    plan += [("N", None, i) for i in range(1, N_REPEATS + 1)] + [("P", None, None)]
    return plan


def _cmd_drive(args) -> int:
    guard_out(args.out, is_dry(args))
    journal = common.AttemptsJournal(args.out / "attempts.jsonl")
    if args.stage != "F":
        args.f_prefix = resolve_f_prefix(args.out, args.f_prefix)
        resolve_final_instance(args.out)
    if not is_dry(args) and resolve_device(args.device, args.gpu_index).startswith("cuda"):
        problem = nvml_problem() or setup_problem(args.data_root, args.manifest)
        if problem:
            print(f"calibration stopped: {problem}", file=sys.stderr)
            return 6
        common.require_single_gpu(args.gpu_index)
    if args.stage == "F":
        for arm in ARMS:  # A_off, A_on, A_bf16, in that order, a fresh process each
            _drive_stage(args, journal, "F", f"{args.f_prefix}_{arm}", arm)
        return _cmd_decide(args)
    for stage, arm, repeat in _rest_plan():
        _drive_stage(args, journal, stage, f"N_{repeat}" if stage == "N" else stage, arm, repeat)
    return _cmd_report(args)


def nvml_problem() -> Optional[str]:
    """Why ``nvml_reader`` cannot run (pynvml not importable), or None."""
    try:
        import pynvml  # noqa: F401
    except ImportError as exc:
        return f"pynvml is not importable ({exc}); install nvidia-ml-py"
    return None


# Exit codes of `calibrate`: 0 complete, 4 no final card (REPEAT_F_ON_A100_80GB, F_FAILS,
# STOP_RECIPE_TO_OWNER or INCOMPLETE; decision.json says which), 6 calibration stopped or
# refused before measurement (CalibrationStopped: wrong card, no instance id, pynvml /
# nvidia-ml-py not importable, a second void, a failed T/W/N/P).
def setup_problem(data_root: Path, manifest: Path) -> Optional[str]:
    """Why the data root is not set up for a run, or None.

    The transfer pull writes ``verified.json`` only after every file matched the manifest. A
    missing record, or one for another manifest, is a setup error: it is refused before the first
    attempt so it does not use the section 8 rerun (run_input_check still checks every file).
    """
    record_path = Path(data_root) / transfer.VERIFIED_NAME
    if not record_path.exists():
        return (f"{transfer.VERIFIED_NAME} is missing in {data_root}: run the transfer pull on "
                "this instance first (e4_part2_transfer.py pull)")
    record = json.loads(record_path.read_text())
    if record.get("manifest_sha256") != common.sha256_file(Path(manifest)):
        return ("the verification record is for another manifest than the supplied one; "
                "rerun the transfer pull")
    return None


def preflight(args, out: Path) -> str:
    """Read the card before any attempt and refuse unless it is in the F or F80 class.

    Writes ``preflight-<unix>.json`` (the stamp and the verdict, refusals included) and
    returns the F record prefix the card runs: ``F`` for the 40 GB class, ``F80`` for 80 GB.
    """
    instance = common.instance_id(args.instance_id)
    if not instance:
        raise common.CalibrationStopped(
            "no instance id: pass --instance-id (every record carries it)")
    args.instance_id = instance
    if is_dry(args):
        import e4_part2_dryrun as dryrun

        stamp = dryrun.fake_stamp(args.fake_card, instance, args.gpu_index)
    else:
        problem = nvml_problem() if args.device.startswith("cuda") else None
        if problem:
            raise common.CalibrationStopped(problem)
        problem = setup_problem(args.data_root, args.manifest)
        if problem:
            raise common.CalibrationStopped(problem)
        common.require_single_gpu(args.gpu_index)
        stamp = common.box_stamp(args.gpu_index, instance)
    problems = {prefix: common.card_class_problems(stamp, prefix) for prefix in ("F", "F80")}
    matching = [prefix for prefix, found in problems.items() if not found]
    out.mkdir(parents=True, exist_ok=True)
    verdict = {"stamp": stamp, "problems": problems, "card_class": matching[0] if matching
               else None, "dry_run": is_dry(args)}
    (out / f"preflight-{int(time.time() * 1000)}.json").write_text(
        json.dumps(verdict, indent=1, sort_keys=True, default=str) + "\n")
    if not matching:
        raise common.CalibrationStopped(
            "the card is neither the rule's A100 40 GB nor A100 80 GB; refusing to run. "
            + "; ".join(f"{p}: {', '.join(found)}" for p, found in problems.items()))
    return matching[0]


def _cmd_calibrate(args) -> int:
    """One command per instance: preflight, F (or the F80 repeat), decide, T, W, N x5, P, report.

    Every attempt is journaled and has a record, void, OOM, killed and failed ones included.
    Nothing is retried except the rule's one rerun of a void, announced on stderr.
    """
    out = args.out
    args.f_prefix = None
    try:
        guard_out(out, is_dry(args))
        prefix = preflight(args, out)
        args.f_prefix = prefix
        journal = common.AttemptsJournal(out / "attempts.jsonl")
        if prefix == "F80" and decide_final_card(f_results(out, "F"))["decision"] != (
                "REPEAT_F_ON_A100_80GB"):
            raise common.CalibrationStopped(
                "this is an A100 80 GB card, which the rule uses only after the 40 GB F "
                "records say to repeat F there; copy those records into --out first")
        for arm in ARMS:
            _drive_stage(args, journal, "F", f"{prefix}_{arm}", arm, skip_completed=True)
        code = _cmd_decide(args)
        if code != 0:
            return code
        decided = json.loads((out / "decision.json").read_text())["decision"]
        if FINAL_CARD_PREFIX[decided] != prefix:
            raise common.CalibrationStopped(
                f"the final card is {decided}; the rest runs on that instance, not this one")
        for stage, arm, repeat in _rest_plan():
            _drive_stage(args, journal, stage, f"N_{repeat}" if stage == "N" else stage, arm,
                         repeat, skip_completed=True)
        return _cmd_report(args)
    except common.CalibrationStopped as exc:
        print(f"calibration stopped: {exc}", file=sys.stderr)
        return 6


def verify_f_cards(out: Path) -> None:
    """Every F record names the card of its prefix in both box stamps (rule sections 1 and 3):
    an A100 of 40 GB under F, of 80 GB under F80, compute capability 8.0."""
    for prefix in ("F", "F80"):
        for arm in ARMS:
            record = find_record(out, f"{prefix}_{arm}")
            if record is None:
                continue
            for which in ("box_stamp_before", "box_stamp_after"):
                stamp = record.get(which)
                if stamp is None:
                    continue
                problems = common.card_class_problems(stamp, prefix)
                if problems:
                    raise common.CalibrationStopped(
                        f"{prefix}_{arm} {which}: not the rule's "
                        f"{'A100 40 GB' if prefix == 'F' else 'A100 80 GB'} card: "
                        + "; ".join(problems))


def _cmd_decide(args) -> int:
    try:
        verify_f_cards(args.out)
    except common.CalibrationStopped as exc:
        print(f"calibration stopped: {exc}", file=sys.stderr)
        return 6
    f40 = f_results(args.out, "F")
    f80 = f_results(args.out, "F80") or None
    decision = decide_final_card(f40, f80)
    if out_is_dry(args.out):
        decision["dry_run"] = True
    if decision["decision"] in FINAL_CARD_PREFIX:
        decision["final_instance"] = final_instance(args.out,
                                                    FINAL_CARD_PREFIX[decision["decision"]])
    (args.out / "decision.json").write_text(json.dumps(decision, indent=1, sort_keys=True) + "\n")
    print(json.dumps(decision))
    return 0 if decision["decision"].startswith("FINAL") else 4


def _cmd_report(args) -> int:
    args.f_prefix = resolve_f_prefix(args.out, args.f_prefix)
    report = assemble_report(args.out, args.f_prefix)
    if out_is_dry(args.out):
        report["dry_run"] = True
    (args.out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True,
                                                     default=str) + "\n")
    write_sha256sums(args.out)
    print(f"report written; {len(report['voids'])} void(s); gaps: {report['provenance_gaps']}")
    return 0


def _cmd_peers_selftest(args) -> int:
    """Run the own-context probe once on this machine and print what it found."""
    common.require_single_gpu(args.gpu_index)
    try:
        found = common.identify_own_gpu_pids_cuda(args.gpu_index)
    except common.PeersUnidentifiableError as exc:
        print(json.dumps({"identified": False, "error": str(exc)}))
        return 5
    print(json.dumps({"identified": True, **found}))
    return 0


def _add_dry_run_options(p) -> None:
    p.add_argument("--dry-run", action="store_true",
                   help="CPU, tiny random model, fake box stamp; records have the real schema "
                        "(see e4_part2_dryrun.py)")
    p.add_argument("--fake-card", default="a100-40", choices=sorted(FAKE_CARD_NAMES),
                   help="dry run: the card the fake box stamp names")
    p.add_argument("--fake-reserved-bytes", type=int, default=20_000_000_000,
                   help="dry run: peak reserved memory the F results report (fake)")
    p.add_argument("--inject", help="dry run: RUN_ID=KIND[@all],... with KIND oom, kill, void "
                                    "or fail")


#: The names of ``e4_part2_dryrun.FAKE_CARDS``, kept here so `--help` needs no import.
FAKE_CARD_NAMES = ("a100-40", "a100-80", "t4", "a100-odd")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("peers-selftest", help="check the peers probe works on this instance")
    p.add_argument("--gpu-index", type=int, default=0)
    p = sub.add_parser("split")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dataset-revision", default=None)
    for name in ("run", "drive", "decide", "report"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--f-prefix", default=None, choices=["F", "F80"],
                       help="F stage: F, or F80 for the repeat on the A100 80 GB (required). "
                            "Other stages derive it from decision.json; a value that "
                            "disagrees is an error")
        if name in ("run", "drive"):
            p.add_argument("--data-root", type=Path)
            p.add_argument("--split-dir", type=Path)
            p.add_argument("--manifest", type=Path)
            p.add_argument("--device", default="cuda")
            p.add_argument("--gpu-index", type=int, default=0)
            p.add_argument("--instance-id")
            _add_dry_run_options(p)
        if name == "run":
            p.add_argument("--fallback-gateup", action="store_true",
                           help="T only: calibrate the gate/up fallback cells (M 129-512) "
                                "instead of the full T selection")
            p.add_argument("--stage", choices=["F", "T", "W", "N", "P"], required=True)
            p.add_argument("--arm", choices=list(ARMS))
            p.add_argument("--repeat", type=int, default=1)
            p.add_argument("--run-id")
            p.add_argument("--recipe-seed", type=int, default=RECIPE.seed)
        if name == "drive":
            p.add_argument("--stage", choices=["F", "rest"], required=True)
    p = sub.add_parser(
        "calibrate", help="the whole calibration on this instance: F, decide, T, W, N x5, P, "
                          "report; refuses a card outside the F or F80 class")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--split-dir", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--instance-id", help="the vast.ai instance id (or set VAST_CONTAINERLABEL)")
    _add_dry_run_options(p)
    args = parser.parse_args(argv)
    if args.cmd in ("run", "drive", "calibrate") and not args.dry_run:
        missing = [flag for flag in ("data_root", "split_dir", "manifest")
                   if getattr(args, flag) is None]
        if missing:
            parser.error("--" + ", --".join(m.replace("_", "-") for m in missing)
                         + " required without --dry-run")
    if args.cmd in ("run", "drive") and args.stage == "F" and args.f_prefix is None:
        parser.error("--f-prefix F or F80 is required for the F stage")
    if args.cmd == "peers-selftest":
        return _cmd_peers_selftest(args)
    return {"split": _cmd_split, "run": _cmd_run, "drive": _cmd_drive, "decide": _cmd_decide,
            "report": _cmd_report, "calibrate": _cmd_calibrate}[args.cmd](args)


if __name__ == "__main__":
    sys.modules.setdefault("e4_part2_calibration", sys.modules["__main__"])
    raise SystemExit(main())
