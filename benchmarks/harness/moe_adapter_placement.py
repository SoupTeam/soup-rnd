"""E1: what a MoE LoRA loses when it skips routed experts (probe-moe-adapter-placement.md).

Subcommands:
  prepare  choose calibration / train / eval rows from the pinned MetaMathQA file
  profile  route the calibration rows through a base model; hottest quarter per layer
  run      train one arm (or score the base) and evaluate it into one JSON

The adapters are built here rather than through peft, so every arm shares one
implementation and one numeric path. They are PEFT-shaped: ``LoRALinear`` is
``lora.Linear`` and ``ExpertLoRA`` is ``lora.ParamWrapper`` on ``gate_up_proj`` and
``down_proj`` (per-expert ``A`` of ``(rank, in)`` and ``B`` of ``(out, rank)``,
``B`` zero at init); tests/test_moe_adapter_placement.py checks both against peft.
The rule that turns run JSONs into verdicts is moe_adapter_rule.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import platform
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as fn

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MODELS: Dict[str, Dict[str, Any]] = {
    "olmoe": {
        "repo": "allenai/OLMoE-1B-7B-0125-Instruct",
        "revision": "b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e",
        "chat": {},
    },
    "qwen35": {
        "repo": "Qwen/Qwen3.5-35B-A3B",
        "revision": "59d61f3ce65a6d9863b86d2e96597125219dc754",
        "chat": {"enable_thinking": False},
    },
}
DATA_FILE = "MetaMathQA-395K.json"
DATA_SHA256 = "fb39a5d8c05c042ece92eae37dfd5ea414a5979df2bf3ad3b86411bef8205725"
SPLIT_SEED = 1234
MAX_TOKENS = 512
N_CAL, N_TRAIN, N_EVAL = 512, 4096, 256
RANK, ALPHA = 16, 32
BATCH_ROWS = 16
WARMUP_STEPS = 16
CLIP_NORM = 1.0
HOT_FRACTION = 0.25
TASK_NEW_TOKENS = 384
GATE_NEW_TOKENS = 256
V4_ROWS = 16
V4_TOLERANCE = 1e-3
ANSWER_MARK = "The answer is:"
ARMS = ("base", "A", "AS", "AT", "AE")
ATTENTION = {
    "self_attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "linear_attn": ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"),
}
SHARED = ("gate_proj", "up_proj", "down_proj")


class NotApplicableError(ValueError):
    """The arm asks for a module this architecture does not have."""


# =====================================================================
# Adapters
# =====================================================================
def _uniform_a(shape: Sequence[int], fan_in: int, generator: torch.Generator) -> torch.Tensor:
    """peft's ``kaiming_uniform_(a=sqrt(5))`` on ``lora_A``: U(-1/sqrt(fan_in), +)."""
    bound = 1.0 / math.sqrt(fan_in)
    return (torch.rand(tuple(shape), generator=generator) * 2 - 1) * bound


class LoRALinear(nn.Module):
    """``base(x) + scale * B(A(x))`` with fp32 masters, computed in the base dtype."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int, generator: torch.Generator):
        super().__init__()
        device = base.weight.device
        self.base = base
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(_uniform_a((rank, base.in_features), base.in_features,
                                              generator).to(device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=device))
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            return out
        dtype = out.dtype
        delta = fn.linear(fn.linear(x.to(dtype), self.lora_A.to(dtype)), self.lora_B.to(dtype))
        return out + delta * self.scale


def grouped_mm(x: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Rows of ``x`` grouped by ``offs`` times per-group ``(in, out)`` weights.

    CUDA takes transformers' own call, ``fn.grouped_mm(x.to(w.dtype), w, offs=offs)``,
    so a zero adapter leaves the base path bit-identical. CPU loops: torch 2.14's
    CPU ``grouped_mm`` has no float32 input gradient ("Invalid strides/sizes").
    """
    if x.is_cuda:
        return fn.grouped_mm(x.to(weight.dtype), weight, offs=offs)
    outs, start = [], 0
    for group, end in enumerate(offs.tolist()):
        if end > start:
            outs.append(x[start:end].to(weight.dtype) @ weight[group])
        start = end
    if not outs:
        return x.new_zeros((0, weight.shape[-1]), dtype=weight.dtype)
    return torch.cat(outs)


class ExpertLoRA(nn.Module):
    """Per-expert LoRA on fused ``gate_up_proj`` / ``down_proj`` for ``selected`` experts.

    Mirrors ``transformers.integrations.moe.grouped_mm_experts_forward`` (gated,
    not transposed, no bias) and adds ``scale * B_e A_e x`` to each projection of
    each token-expert pair routed to a selected expert. Unselected experts carry
    no parameters at all, so trainable counts and optimizer state are real.
    """

    def __init__(self, base: nn.Module, selected: Sequence[int], rank: int, alpha: int,
                 generator: torch.Generator):
        super().__init__()
        experts, two_inter, hidden = base.gate_up_proj.shape
        inter = base.down_proj.shape[-1]
        chosen = sorted({int(e) for e in selected})
        if not chosen or chosen[0] < 0 or chosen[-1] >= experts:
            raise ValueError(f"selected experts must be a non-empty subset of 0..{experts - 1}")
        device = base.gate_up_proj.device
        self.base = base
        self.num_experts = experts
        self.scale = alpha / rank
        slot = torch.full((experts,), -1, dtype=torch.long)
        slot[chosen] = torch.arange(len(chosen))
        self.register_buffer("selected", torch.tensor(chosen, device=device), persistent=False)
        self.register_buffer("slot", slot.to(device), persistent=False)
        n = len(chosen)
        self.gate_up_A = nn.Parameter(_uniform_a((n, rank, hidden), hidden, generator).to(device))
        self.gate_up_B = nn.Parameter(torch.zeros(n, two_inter, rank, device=device))
        self.down_A = nn.Parameter(_uniform_a((n, rank, inter), inter, generator).to(device))
        self.down_B = nn.Parameter(torch.zeros(n, hidden, rank, device=device))
        self.enabled = True

    def _delta(self, x: torch.Tensor, offs: torch.Tensor, a: torch.Tensor,
               b: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        low = grouped_mm(x, a.to(dtype).transpose(-2, -1), offs)
        return grouped_mm(low, b.to(dtype).transpose(-2, -1), offs) * self.scale

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        base = self.base
        num_top_k = top_k_index.size(-1)
        num_tokens, hidden_dim = hidden_states.size(0), hidden_states.size(-1)
        expert_ids, perm = torch.sort(top_k_index.reshape(-1))
        x = hidden_states[perm // num_top_k]
        weights = top_k_weights.reshape(-1)[perm]
        histc_in = expert_ids.float() if x.device.type in ("cpu", "mps") else expert_ids.int()
        counts = torch.histc(histc_in, bins=self.num_experts, min=0, max=self.num_experts - 1)
        offs = torch.cumsum(counts, dim=0, dtype=torch.int32)
        rows = None
        if self.enabled:
            rows = (self.slot[expert_ids] >= 0).nonzero().squeeze(1)
            offs_sel = torch.cumsum(counts[self.selected], dim=0, dtype=torch.int32)
            if rows.numel() == 0:
                rows = None
        proj = grouped_mm(x, base.gate_up_proj.transpose(-2, -1), offs)
        if rows is not None:
            proj = proj.index_add(0, rows, self._delta(x[rows], offs_sel, self.gate_up_A,
                                                       self.gate_up_B).to(proj.dtype))
        proj = base._apply_gate(proj)
        out = grouped_mm(proj, base.down_proj.transpose(-2, -1), offs)
        if rows is not None:
            out = out.index_add(0, rows, self._delta(proj[rows], offs_sel, self.down_A,
                                                     self.down_B).to(out.dtype))
        out = out * weights.unsqueeze(-1)
        inverse = torch.empty_like(perm)
        inverse[perm] = torch.arange(perm.size(0), device=perm.device)
        out = out[inverse].view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
        return out.to(hidden_states.dtype)


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    return model.model.layers


def hot_experts(counts: Sequence[int], fraction: float = HOT_FRACTION) -> List[int]:
    """The busiest ``fraction`` of experts by assignment count, ties to the lower index."""
    keep = max(1, int(round(len(counts) * fraction)))
    order = sorted(range(len(counts)), key=lambda e: (-counts[e], e))
    return sorted(order[:keep])


def attach_adapters(model: nn.Module, arm: str, *, hot: Optional[Sequence[Sequence[int]]] = None,
                    rank: int = RANK, alpha: int = ALPHA, seed: int = 0) -> List[str]:
    """Freeze ``model`` and wrap the arm's targets; return the adapted module paths."""
    if arm not in ARMS or arm == "base":
        raise ValueError(f"no adapters for arm {arm!r}")
    generator = torch.Generator().manual_seed(seed)
    for param in model.parameters():
        param.requires_grad_(False)
    targets: List[str] = []
    shared_seen = False
    for index, layer in enumerate(decoder_layers(model)):
        prefix = f"model.layers.{index}"
        for owner, names in ATTENTION.items():
            module = getattr(layer, owner, None)
            if module is None:
                continue
            for name in names:
                linear = getattr(module, name)
                if not isinstance(linear, nn.Linear):
                    raise TypeError(f"{prefix}.{owner}.{name} is {type(linear).__name__}")
                setattr(module, name, LoRALinear(linear, rank, alpha, generator))
                targets.append(f"{prefix}.{owner}.{name}")
        mlp = layer.mlp
        shared = getattr(mlp, "shared_expert", None)
        if arm != "A" and shared is not None:
            shared_seen = True
            for name in SHARED:
                setattr(shared, name, LoRALinear(getattr(shared, name), rank, alpha, generator))
                targets.append(f"{prefix}.mlp.shared_expert.{name}")
        if arm in ("AT", "AE"):
            experts = mlp.experts.gate_up_proj.shape[0]
            if arm == "AE":
                chosen: Sequence[int] = range(experts)
            else:
                if hot is None:
                    raise ValueError("arm AT needs the calibration hot set")
                chosen = hot[index]
            mlp.experts = ExpertLoRA(mlp.experts, chosen, rank, alpha, generator)
            targets.append(f"{prefix}.mlp.experts[{len(list(chosen))}/{experts}]")
    if arm == "AS" and not shared_seen:
        raise NotApplicableError("arm AS needs a shared expert; this model has none")
    return targets


def adapter_parameters(model: nn.Module) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def set_adapters_enabled(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, (LoRALinear, ExpertLoRA)):
            module.enabled = enabled


# =====================================================================
# Data
# =====================================================================
def reference_answer(response: str) -> Optional[str]:
    from soup_cli.utils.final_answer import parse_reference

    if ANSWER_MARK not in response:
        return None
    text = response.rsplit(ANSWER_MARK, 1)[1].strip()
    return text if text and parse_reference(text) is not None else None


def encode_row(tokenizer: Any, chat: Dict[str, Any], query: str,
               response: str) -> Optional[Tuple[List[int], List[int]]]:
    """``(prompt_ids, full_ids)`` through the chat template, or None if not a token prefix."""
    user = [{"role": "user", "content": query}]
    prompt = tokenizer.apply_chat_template(user, tokenize=False, add_generation_prompt=True,
                                           **chat)
    full = tokenizer.apply_chat_template(user + [{"role": "assistant", "content": response}],
                                         tokenize=False, **chat)
    if not full.startswith(prompt):
        return None
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if full_ids[:len(prompt_ids)] != prompt_ids or len(full_ids) <= len(prompt_ids):
        return None
    return prompt_ids, full_ids


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: str) -> List[Dict[str, Any]]:
    actual = file_sha256(path)
    if actual != DATA_SHA256:
        raise ValueError(f"{path} has SHA-256 {actual}, expected {DATA_SHA256}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_tokenizer(model_key: str):
    from transformers import AutoTokenizer

    spec = MODELS[model_key]
    return AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])


def group_key(row: Dict[str, Any]) -> str:
    """MetaMathQA's source question: every augmentation of one GSM8K/MATH item shares it."""
    return " ".join(str(row["original_question"]).split())


def select_rows(rows: Sequence[Dict[str, Any]], order: Sequence[int], wanted: int,
                reject) -> Tuple[List[int], Dict[str, int]]:
    """First ``wanted`` rows in ``order`` that ``reject`` passes, one per source question.

    ``reject(row)`` returns a drop reason or None. Taking one row per group makes
    every split group-disjoint: no augmentation of an evaluated question is trained on.
    """
    accepted: List[int] = []
    seen: set = set()
    dropped: Dict[str, int] = {"same_group": 0}
    for index in order:
        row = rows[index]
        key = group_key(row)
        if key in seen:
            dropped["same_group"] += 1
            continue
        reason = reject(row)
        if reason:
            dropped[reason] = dropped.get(reason, 0) + 1
            continue
        seen.add(key)
        accepted.append(index)
        if len(accepted) == wanted:
            break
    if len(accepted) < wanted:
        raise ValueError(f"only {len(accepted)} rows qualified; need {wanted}")
    return accepted, dropped


def check_group_disjoint(rows: Sequence[Dict[str, Any]], split: Dict[str, Any]) -> None:
    """Refuse a split in which a source question reaches two splits or twice into one."""
    owner: Dict[str, str] = {}
    for name in ("calibration", "train", "eval"):
        for index in split[name]:
            key = group_key(rows[index])
            if key in owner:
                raise ValueError(f"row {index} ({name}) shares its source question with "
                                 f"a {owner[key]} row; the split is not group-disjoint")
            owner[key] = name


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    rows = load_rows(args.data)
    order = list(range(len(rows)))
    random.Random(SPLIT_SEED).shuffle(order)
    tokenizers = {key: load_tokenizer(key) for key in MODELS}

    def reject(row: Dict[str, Any]) -> Optional[str]:
        if reference_answer(row["response"]) is None:
            return "no_reference"
        for key, tokenizer in tokenizers.items():
            encoded = encode_row(tokenizer, MODELS[key]["chat"], row["query"], row["response"])
            if encoded is None:
                return "not_prefix"
            if len(encoded[1]) > MAX_TOKENS:
                return "too_long"
        return None

    accepted, dropped = select_rows(rows, order, N_CAL + N_TRAIN + N_EVAL, reject)
    split = {
        "data_file": DATA_FILE, "data_sha256": DATA_SHA256, "split_seed": SPLIT_SEED,
        "max_tokens": MAX_TOKENS, "grouping": "one row per original_question",
        "examined": len(accepted) + sum(dropped.values()), "dropped": dropped,
        "calibration": accepted[:N_CAL],
        "train": accepted[N_CAL:N_CAL + N_TRAIN],
        "eval": accepted[N_CAL + N_TRAIN:],
    }
    check_group_disjoint(rows, split)
    return split


# =====================================================================
# Model, routing, training, evaluation
# =====================================================================
def load_model(model_key: str, device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM

    spec = MODELS[model_key]
    model = AutoModelForCausalLM.from_pretrained(
        spec["repo"], revision=spec["revision"], dtype=dtype, attn_implementation="eager",
        experts_implementation="grouped_mm", low_cpu_mem_usage=True,
    )
    return model.to(device)


def route_counts(model: nn.Module, sequences: Sequence[Sequence[int]],
                 device: str) -> List[List[int]]:
    """Top-k assignments per expert per layer, one unpadded sequence at a time."""
    layers = decoder_layers(model)
    counts: List[Optional[torch.Tensor]] = [None] * len(layers)
    handles = []
    for index, layer in enumerate(layers):
        def capture(_module, _args, output, index=index):
            experts = _module.num_experts
            hits = torch.bincount(output[2].reshape(-1), minlength=experts)
            counts[index] = hits if counts[index] is None else counts[index] + hits
        handles.append(layer.mlp.gate.register_forward_hook(capture))
    try:
        with torch.no_grad():
            for ids in sequences:
                model(input_ids=torch.tensor([list(ids)], device=device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return [c.tolist() for c in counts]


def _pad_batch(items: Sequence[Tuple[List[int], List[int]]], pad_id: int,
               device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(full) for _prompt, full in items)
    ids = torch.full((len(items), width), pad_id, dtype=torch.long)
    labels = torch.full((len(items), width), -100, dtype=torch.long)
    mask = torch.zeros((len(items), width), dtype=torch.long)
    for row, (prompt, full) in enumerate(items):
        ids[row, :len(full)] = torch.tensor(full)
        labels[row, len(prompt):len(full)] = torch.tensor(full[len(prompt):])
        mask[row, :len(full)] = 1
    return ids.to(device), labels.to(device), mask.to(device)


def nll_sum(model: nn.Module, items, pad_id: int, device: str) -> torch.Tensor:
    """Summed response-token NLL of ``items`` (right-padded, loss on response only)."""
    ids, labels, mask = _pad_batch(items, pad_id, device)
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    return fn.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                           labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")


def targets_of(items) -> int:
    return sum(len(full) - len(prompt) for prompt, full in items)


def evaluate_nll(model, items, pad_id, device, micro) -> Dict[str, float]:
    total, tokens = 0.0, targets_of(items)
    with torch.no_grad():
        for start in range(0, len(items), micro):
            total += nll_sum(model, items[start:start + micro], pad_id, device).item()
    return {"nll": total / tokens, "tokens": tokens}


def generate(model, tokenizer, prompts: Sequence[List[int]], max_new_tokens: int,
             batch: int, device: str) -> List[str]:
    """Greedy, left-padded batches sorted by length; outputs in input order."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None \
        else tokenizer.eos_token_id
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outputs: List[Optional[str]] = [None] * len(prompts)
    for start in range(0, len(order), batch):
        chunk = order[start:start + batch]
        width = max(len(prompts[i]) for i in chunk)
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for row, i in enumerate(chunk):
            ids[row, width - len(prompts[i]):] = torch.tensor(prompts[i])
            mask[row, width - len(prompts[i]):] = 1
        with torch.no_grad():
            out = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device),
                                 max_new_tokens=max_new_tokens, do_sample=False, top_p=None,
                                 top_k=None, temperature=None, pad_token_id=pad_id)
        for row, i in enumerate(chunk):
            outputs[i] = tokenizer.decode(out[row, width:], skip_special_tokens=True)
    return [o or "" for o in outputs]


def gate_prompts() -> Dict[str, List[str]]:
    """Every prompt Soup's leg 2 sends, per bundled suite, in scoring order."""
    from soup_cli.eval.gate_suites import DEFAULT_GENERAL_SUITE, score_bundled_suite

    seen: Dict[str, List[str]] = {}
    for name in DEFAULT_GENERAL_SUITE:
        seen[name] = []
        score_bundled_suite(name, lambda prompt, name=name: seen[name].append(prompt) or "")
    return seen


def score_gate(model, tokenizer, batch: int, device: str) -> Dict[str, Any]:
    """Soup's bundled leg-2 suites, prompts rendered as ``soup ship`` renders them."""
    from soup_cli.eval.gate_suites import (
        BEHAVIOURAL_MAX_NEW_TOKENS,
        BUNDLED_SCORER_REVISION,
        score_bundled_suite,
    )
    from soup_cli.utils.live_eval import _MAX_PROMPT_TOKENS, _render_prompt_template
    from soup_cli.utils.vllm import encode_rendered_prompt

    if BEHAVIOURAL_MAX_NEW_TOKENS != GATE_NEW_TOKENS:
        raise ValueError("Soup's behavioural generation budget moved; the rule pins 256")
    prompts = gate_prompts()
    flat = [p for name in prompts for p in prompts[name]]
    if len(set(flat)) != len(flat):
        raise ValueError("gate prompts are not unique; a lookup generator would mix them")
    encoded = []
    for prompt in flat:
        text, templated = _render_prompt_template(tokenizer, prompt)
        encoded.append(list(encode_rendered_prompt(tokenizer, text, templated=templated,
                                                   truncation=True,
                                                   max_length=_MAX_PROMPT_TOKENS)["input_ids"]))
    outputs = dict(zip(flat, generate(model, tokenizer, encoded, GATE_NEW_TOKENS, batch, device)))
    scores = {name: score_bundled_suite(name, outputs.__getitem__) for name in prompts}
    return {"scorer_revision": BUNDLED_SCORER_REVISION, "scores": scores, "outputs": outputs}


def score_task(model, tokenizer, prompts, references, batch, device) -> Dict[str, Any]:
    from soup_cli.utils.final_answer import answers_match, parse_completion, parse_reference

    outputs = generate(model, tokenizer, prompts, TASK_NEW_TOKENS, batch, device)
    correct = [bool(answers_match(parse_completion(out), parse_reference(ref)))
               for out, ref in zip(outputs, references)]
    return {"accuracy": sum(correct) / len(correct), "correct": correct, "outputs": outputs}


def train(model, items, *, lr: float, seed: int, micro: int, pad_id: int,
          device: str) -> Dict[str, Any]:
    params = adapter_parameters(model)
    optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / WARMUP_STEPS))
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    steps = len(order) // BATCH_ROWS
    log: List[Dict[str, float]] = []
    model.train()
    for step in range(steps):
        batch = [items[i] for i in order[step * BATCH_ROWS:(step + 1) * BATCH_ROWS]]
        tokens = targets_of(batch)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for start in range(0, len(batch), micro):
            loss = nll_sum(model, batch[start:start + micro], pad_id, device)
            (loss / tokens).backward()
            total += loss.item()
        norm = torch.nn.utils.clip_grad_norm_(params, CLIP_NORM).item()
        optimizer.step()
        schedule.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        log.append({"step": step, "loss": total / tokens, "grad_norm": norm,
                    "seconds": time.perf_counter() - started, "tokens": tokens})
        if not math.isfinite(total):
            break
    model.eval()
    return {"steps": len(log), "log": log}


# =====================================================================
# Records
# =====================================================================
def host_state() -> Dict[str, Any]:
    import transformers

    state: Dict[str, Any] = {
        "platform": platform.platform(), "python": platform.python_version(),
        "cpus": os.cpu_count(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        state.update(gpu=props.name, gpu_bytes=props.total_memory,
                     capability=list(torch.cuda.get_device_capability(0)))
    try:
        import psutil

        memory = psutil.virtual_memory()
        state.update(ram_total=memory.total, ram_available=memory.available)
    except ImportError:
        pass
    try:
        import source_provenance

        state["soup_source"] = source_provenance.source_sha(dirty_pathspec="src")
    except Exception as exc:  # noqa: BLE001 - a stamp must not break the run
        state["soup_source"] = f"unknown: {type(exc).__name__}"
    return state


def harness_sha256() -> str:
    return file_sha256(__file__)


def adamw_fp32_bytes(trainable: int) -> int:
    """B4's ``torch_adamw_fp32`` persistent budget: fp32 weight, gradient, two moments."""
    return 16 * trainable


def encode_split(rows, indices, tokenizer, chat) -> Tuple[list, List[str]]:
    items, references = [], []
    for index in indices:
        row = rows[index]
        encoded = encode_row(tokenizer, chat, row["query"], row["response"])
        if encoded is None:
            raise ValueError(f"row {index} no longer encodes as a template prefix")
        items.append(encoded)
        references.append(reference_answer(row["response"]))
    return items, references


def write_json(path: str, payload: Dict[str, Any]) -> None:
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    partial.replace(target)


def profile(args: argparse.Namespace) -> Dict[str, Any]:
    split = json.loads(pathlib.Path(args.split).read_text(encoding="utf-8"))
    rows = load_rows(args.data)
    check_group_disjoint(rows, split)
    tokenizer = load_tokenizer(args.model)
    items, _refs = encode_split(rows, split["calibration"], tokenizer, MODELS[args.model]["chat"])
    model = load_model(args.model, args.device, getattr(torch, args.dtype)).eval()
    counts = route_counts(model, [full for _prompt, full in items], args.device)
    return {"model": args.model, "rows": len(items), "fraction": HOT_FRACTION,
            "counts": counts, "hot": [hot_experts(layer) for layer in counts],
            "host": host_state(), "harness_sha256": harness_sha256()}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.time()
    split = json.loads(pathlib.Path(args.split).read_text(encoding="utf-8"))
    rows = load_rows(args.data)
    check_group_disjoint(rows, split)
    spec = MODELS[args.model]
    tokenizer = load_tokenizer(args.model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None \
        else tokenizer.eos_token_id
    train_items, _ = encode_split(rows, split["train"], tokenizer, spec["chat"])
    eval_items, references = encode_split(rows, split["eval"], tokenizer, spec["chat"])
    record: Dict[str, Any] = {
        "model": args.model, "repo": spec["repo"], "revision": spec["revision"],
        "arm": args.arm, "lr": args.lr, "seed": args.seed, "rank": RANK, "alpha": ALPHA,
        "batch_rows": BATCH_ROWS, "harness_sha256": harness_sha256(), "host": host_state(),
        "split_sha256": file_sha256(args.split), "status": "started",
    }
    torch.manual_seed(args.seed)
    model = load_model(args.model, args.device, getattr(torch, args.dtype)).eval()
    model.config.use_cache = False
    if args.arm != "base":
        hot = None
        if args.arm == "AT":
            hot_record = json.loads(pathlib.Path(args.hot).read_text(encoding="utf-8"))
            if hot_record["model"] != args.model:
                raise ValueError("hot-set file was profiled on another model")
            hot = hot_record["hot"]
            record["hot_sha256"] = file_sha256(args.hot)
        probe = eval_items[:V4_ROWS]
        base_probe = evaluate_nll(model, probe, pad_id, args.device, args.micro)["nll"]
        try:
            record["targets"] = attach_adapters(model, args.arm, hot=hot, seed=args.seed)
        except NotApplicableError as exc:
            record.update(status="not_applicable", reason=str(exc))
            return record
        zero_probe = evaluate_nll(model, probe, pad_id, args.device, args.micro)["nll"]
        record["v4"] = {"base": base_probe, "zero_adapter": zero_probe,
                        "passed": abs(base_probe - zero_probe) <= V4_TOLERANCE}
        if not record["v4"]["passed"]:
            record["status"] = "void_v4"
            return record
        trainable = sum(p.numel() for p in adapter_parameters(model))
        record.update(trainable_parameters=trainable,
                      adapter_bytes_adamw_fp32=adamw_fp32_bytes(trainable))
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        record["train"] = train(model, train_items, lr=args.lr, seed=args.seed, micro=args.micro,
                                pad_id=pad_id, device=args.device)
        if args.device.startswith("cuda"):
            record["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
    record["eval"] = evaluate_nll(model, eval_items, pad_id, args.device, args.micro)
    prompts = [prompt for prompt, _full in eval_items]
    record["task"] = score_task(model, tokenizer, prompts, references, args.gen_batch,
                                args.device)
    record["gate"] = score_gate(model, tokenizer, args.gen_batch, args.device)
    finite = all(math.isfinite(entry["loss"]) for entry in record.get("train", {}).get("log", []))
    record.update(status="ok" if finite else "void_nonfinite", seconds=time.time() - started)
    return record


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--data", required=True)
    prep.add_argument("--out", required=True)
    for name in ("profile", "run"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--model", choices=sorted(MODELS), required=True)
        cmd.add_argument("--data", required=True)
        cmd.add_argument("--split", required=True)
        cmd.add_argument("--out", required=True)
        cmd.add_argument("--device", default="cuda")
        cmd.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    runner = sub.choices["run"]
    runner.add_argument("--arm", choices=ARMS, required=True)
    runner.add_argument("--lr", type=float, default=0.0)
    runner.add_argument("--seed", type=int, default=0)
    runner.add_argument("--hot", default=None)
    runner.add_argument("--micro", type=int, default=4)
    runner.add_argument("--gen-batch", type=int, default=32)
    args = parser.parse_args(argv)
    if args.command == "run" and args.arm != "base" and args.lr <= 0:
        parser.error("--lr is required for a trained arm")
    if args.command == "run" and args.arm == "AT" and not args.hot:
        parser.error("--hot is required for arm AT")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    from rich.console import Console

    console = Console()
    args = parse_args(argv)
    handler = {"prepare": prepare, "profile": profile, "run": run}[args.command]
    result = handler(args)
    write_json(args.out, result)
    console.print(f"{args.command}: wrote {args.out} status={result.get('status', 'ok')}",
                  markup=False)
    return 0 if result.get("status", "ok") in ("ok", "not_applicable") else 1


if __name__ == "__main__":
    sys.exit(main())
