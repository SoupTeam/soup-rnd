"""E1 in Soup's RAM tier: A and AS trained by Soup's streamed trainer, scored by E1's harness.

Subcommands:
  labels     Soup's assistant-only labels against E1's on every train, E and C row
  preflight  what Soup's streamed adapter pre-flight makes of A, AS and three AE spellings,
             plus why AT has no Soup spelling at all
  parity     streamed vs resident on the same base numerics and the same batches: loss,
             adapter gradients and several AdamW steps (pre-registered pass rule below)
  noise      post hoc, after a parity run did not pass: each side repeated against itself
             (step-0 gradients, AdamW trajectory) and once against the other side
  train      one arm and seed through ``soup train``'s in-process calls (RAM tier)
  score      Soup's saved adapter copied into E1's measured harness and scored as E1 scores
             (E, task, leg 2, then C and per-row E/C via moe_adapter_confirm)

Training, data loading, row formatting and the streamed save are Soup's own; NLL,
generation, leg 2 and the confirmation set are the measured E1 harness's and
moe_adapter_confirm's. Score records (``ramtier-*.json``) are what
moe_adapter_confirm_rule.py's ``ram_verdicts`` / ``ram_memory`` read.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import pathlib
import sys
import time
import traceback
from collections import Counter
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import moe_adapter_confirm as confirm  # noqa: E402

MODEL = "qwen35"
ATTENTION_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z",
                     "in_proj_a", "in_proj_b", "out_proj")
SHARED_TARGETS = ("gate_proj", "up_proj", "down_proj")
ARM_TARGETS = {"A": ATTENTION_TARGETS, "AS": ATTENTION_TARGETS + SHARED_TARGETS}
E1_TRAINABLE = {"A": 16_250_880, "AS": 21_166_080}
LR = 2e-4
RANK, ALPHA = 16, 32
# E1 scored every Qwen3.5 record with --micro 4 --gen-batch 16 (record section 10 plans), and
# the resident confirmation runs use the same; left-padded batches of another size can change
# greedy outputs, so the RAM-tier records are scored identically.
MICRO, GEN_BATCH = 4, 16
MAX_LENGTH = 512
MAX_MISMATCHES = 5
IGNORE_INDEX = -100
ENV_KEYS = ("PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE",
            "SOUP_LAYER_STREAM_CACHE_DIR")
NONFINITE_MARKERS = ("non-finite training metric", "non-finite trainable parameter")
FORWARD_KEYS = ("input_ids", "attention_mask", "labels", "position_ids")

# Parity pass rule, pre-registered: fixed here before any parity run exists.
PARITY_LOSS_RELATIVE = 1e-3
PARITY_GRAD_RELATIVE_L2 = 1e-2
PARITY_GRAD_COSINE = 0.99
PARITY_ADAMW = {"lr": 2e-4, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0}
PARITY_LORA_B_SEED = 23
PARITY_LORA_B_STD = 0.02

# Qwen3.5's own template (chat_template.jinja at the pinned revision) for one user turn and
# one assistant turn with enable_thinking=False, its content trimming and think-block
# handling copied; ``content + '<|im_end|>\n'`` is the generation span, so Soup's
# assistant-only labels are E1's ``full_ids[len(prompt_ids):]``.
CHAT_TEMPLATE = (
    r"{%- for message in messages %}"
    r"{%- if message['role'] == 'user' %}"
    r"{{- '<|im_start|>user\n' + message['content']|trim + '<|im_end|>\n' }}"
    r"{%- elif message['role'] == 'assistant' %}"
    r"{%- set content = message['content']|trim %}"
    r"{%- set reasoning_content = '' %}"
    r"{%- if '</think>' in content %}"
    r"{%- set reasoning_content = content.split('</think>')[0].rstrip('\n')"
    r".split('<think>')[-1].lstrip('\n') %}"
    r"{%- set content = content.split('</think>')[-1].lstrip('\n') %}"
    r"{%- endif %}"
    r"{{- '<|im_start|>assistant\n<think>\n' + reasoning_content|trim + '\n</think>\n\n' }}"
    r"{%- generation %}{{- content + '<|im_end|>\n' }}{%- endgeneration %}"
    r"{%- else %}"
    r"{{- raise_exception('this template renders user and assistant turns only') }}"
    r"{%- endif %}"
    r"{%- endfor %}"
    r"{%- if add_generation_prompt %}"
    r"{{- '<|im_start|>assistant\n<think>\n\n</think>\n\n' }}"
    r"{%- endif %}"
)

# Why AT (a per-layer subset of the fused routed experts) has no Soup spelling. Each entry
# is re-read at preflight time and recorded with the text it cites.
AT_CITATIONS = (
    {"root": "repo", "path": "src/soup_cli/config/schema.py", "lines": (94, 94),
     "expect": "target_parameters",
     "says": "Soup's only raw-parameter target is a list of parameter names; no field "
             "selects experts within a fused tensor (full field list in soup_lora_fields)"},
    {"root": "repo", "path": "src/soup_cli/config/schema.py", "lines": (6318, 6322),
     "expect": "is not yet supported with",
     "says": "lora.target_parameters is refused at config load with stream_layers=true"},
    {"root": "repo", "path": "src/soup_cli/utils/moe.py", "lines": (222, 226),
     "expect": "moe_targets = get_moe_target_modules(model)",
     "says": "training.moe_lora resolves to module-name patterns: every expert of every "
             "layer, never a subset"},
    {"root": "peft", "path": "tuners/lora/layer.py", "lines": (2335, 2335),
     "expect": "class ParamWrapper",
     "says": "PEFT adapts a targeted nn.Parameter through ParamWrapper"},
    {"root": "peft", "path": "tuners/lora/layer.py", "lines": (2397, 2400),
     "expect": "num_experts, in_features, out_features = shape",
     "says": "the expert count is the fused tensor's whole leading dimension"},
    {"root": "peft", "path": "tuners/lora/layer.py", "lines": (2455, 2456),
     "expect": "r * self.num_experts",
     "says": "lora_A/lora_B are sized for all experts of the fused tensor at once"},
)


# =====================================================================
# Shared pieces
# =====================================================================
def text_sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def env_state() -> Dict[str, Any]:
    return {key: os.environ.get(key) for key in ENV_KEYS}


def host(harness) -> Dict[str, Any]:
    """The measured harness's host stamp, the run's env vars and the visible GPU count."""
    import torch

    state = harness.host_state()
    state["env"] = env_state()
    state["visible_gpus"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return state


def stamp(record: Dict[str, Any], harness, split: Optional[str]) -> None:
    record.update(harness_sha256=confirm.file_sha256(__file__),
                  measured_harness_sha256=confirm.file_sha256(str(confirm.MEASURED_HARNESS)),
                  host=host(harness),
                  split_sha256=confirm.file_sha256(split) if split else None)


def error_fields(exc: BaseException) -> Dict[str, Any]:
    return {"error": {"type": type(exc).__name__, "message": str(exc),
                      "traceback": "".join(traceback.format_exception(
                          type(exc), exc, exc.__traceback__))}}


def jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def load_json(path: str) -> Dict[str, Any]:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


@contextlib.contextmanager
def working_directory(path: pathlib.Path) -> Iterator[None]:
    """Soup keeps data and output under the cwd, so every run happens inside its run dir."""
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def recipe(base: str, targets: Sequence[str], *, seed: int, lr: float, quantization: str,
           lora_extra: Optional[Dict[str, Any]] = None,
           training_extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The Soup config of one arm/seed (the settled recipe)."""
    lora = {"r": RANK, "alpha": ALPHA, "dropout": 0.0, "target_modules": list(targets),
            **(lora_extra or {})}
    training = {"epochs": 1, "lr": lr, "batch_size": 16, "gradient_accumulation_steps": 1,
                "quantization": quantization, "stream_layers": True, "stream_source": "ram",
                "optimizer": "adamw_torch", "scheduler": "constant_with_warmup",
                "warmup_ratio": 0.0625, "weight_decay": 0.0, "max_grad_norm": 1.0,
                "seed": seed, "logging_steps": 1, "save_steps": 100000,
                **(training_extra or {}), "lora": lora}
    return {"base": base, "task": "sft", "backend": "transformers", "modality": "text",
            "data": {"train": "./train.jsonl", "format": "chatml", "val_split": 0,
                     "max_length": MAX_LENGTH, "chat_template": CHAT_TEMPLATE},
            "training": training, "output": "./out"}


def recipe_text(config: Dict[str, Any]) -> str:
    import yaml

    text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True, width=100)
    if yaml.safe_load(text) != config:
        raise ValueError("the recipe does not survive a YAML round trip")
    return text


def row_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    return [{"role": "user", "content": row["query"]},
            {"role": "assistant", "content": row["response"]}]


def prepare_run_dir(run_dir: pathlib.Path, config: Dict[str, Any],
                    rows: Sequence[Dict[str, Any]], indices: Sequence[int]) -> str:
    """Write ``soup.yaml`` and ``train.jsonl`` (E1's train rows in split order); LF bytes."""
    run_dir.mkdir(parents=True, exist_ok=True)
    text = recipe_text(config)
    with open(run_dir / "soup.yaml", "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    with open(run_dir / "train.jsonl", "w", encoding="utf-8", newline="\n") as handle:
        for index in indices:
            handle.write(json.dumps({"messages": row_messages(rows[index])},
                                    ensure_ascii=False) + "\n")
    return text


def soup_load(config_path: str, device: str):
    """``soup train``'s calls up to the wrapper: config, dataset, ``build_trainer``."""
    from soup_cli.config.loader import load_config, load_config_from_string
    from soup_cli.data.loader import (
        data_config_for_task,
        load_dataset,
        task_preserves_source_columns,
    )
    from soup_cli.trainer.dispatch import build_trainer
    from soup_cli.utils.eval_schedule import loader_data_config

    try:
        cfg = load_config(config_path)
    except SystemExit as exc:
        try:
            load_config_from_string(pathlib.Path(config_path).read_text(encoding="utf-8"))
        except ValueError as why:
            raise ValueError(f"Soup refused the recipe: {why}") from None
        raise ValueError(f"load_config exited with {exc.code}") from None
    dataset = load_dataset(data_config_for_task(loader_data_config(cfg), cfg.task),
                           preserve_source_columns=task_preserves_source_columns(cfg.task))
    wrapper = build_trainer(cfg, device=device, report_to="none", deepspeed_config=None,
                            fsdp_config=None, trust_remote_code=False)
    return cfg, dataset, wrapper


# =====================================================================
# Labels
# =====================================================================
def soup_format_row(cfg) -> Tuple[Any, Callable[[dict], dict]]:
    """A fresh tokenizer for ``cfg.base`` and Soup's ``format_row`` (template override on)."""
    from transformers import AutoTokenizer

    from soup_cli.data.sft_format import build_format_row

    tokenizer = AutoTokenizer.from_pretrained(cfg.base, trust_remote_code=False)
    return tokenizer, build_format_row(tokenizer=tokenizer, data_cfg=cfg.data,
                                       training_cfg=cfg.training)


def _first_difference(left: Sequence[int], right: Sequence[int]) -> int:
    for position, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return position
    return min(len(left), len(right))


def _window(tokenizer, ids: Sequence[int], position: int) -> List[str]:
    return [tokenizer.decode([t]) if t != IGNORE_INDEX else "<masked>"
            for t in ids[max(0, position - 8):position + 8]]


def compare_row(tokenizer, prompt_ids: Sequence[int], full_ids: Sequence[int],
                soup_row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """None when Soup's row is E1's: same ids, loss on exactly E1's response tokens."""
    ids, labels = list(soup_row["input_ids"]), list(soup_row["labels"])
    expected = [IGNORE_INDEX] * len(prompt_ids) + list(full_ids[len(prompt_ids):])
    for field, soup, e1 in (("input_ids", ids, list(full_ids)), ("labels", labels, expected)):
        if soup != e1:
            at = _first_difference(soup, e1)
            return {"field": field, "position": at, "soup_length": len(soup),
                    "e1_length": len(e1), "soup": _window(tokenizer, soup, at),
                    "e1": _window(tokenizer, e1, at)}
    return None


def check_labels(harness, e1_tokenizer, soup_tokenizer, format_row,
                 rows: Sequence[Dict[str, Any]], indices: Sequence[int],
                 examples: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Soup's formatted row against E1's ``encode_row`` on every row of one set.

    ``examples``: the rows as Soup loaded them (train); otherwise built from MetaMathQA.
    Also checks that the generation prompt renders as Qwen3.5's own (enable_thinking off).
    """
    chat = harness.MODELS[MODEL]["chat"]
    if examples is not None and len(examples) != len(indices):
        raise ValueError(f"Soup loaded {len(examples)} rows for {len(indices)} split rows")
    identical, mismatches = 0, []
    for position, index in enumerate(indices):
        row = rows[index]
        user = [{"role": "user", "content": row["query"]}]
        problem: Optional[Dict[str, Any]]
        encoded = harness.encode_row(e1_tokenizer, chat, row["query"], row["response"])
        e1_prompt = e1_tokenizer.apply_chat_template(user, tokenize=False,
                                                     add_generation_prompt=True, **chat)
        soup_prompt = soup_tokenizer.apply_chat_template(user, tokenize=False,
                                                         add_generation_prompt=True)
        if encoded is None:
            problem = {"field": "e1_encoding", "reason": "E1's prompt is not a token prefix"}
        elif soup_prompt != e1_prompt:
            problem = {"field": "generation_prompt", "soup": soup_prompt[-80:],
                       "e1": e1_prompt[-80:]}
        else:
            example = examples[position] if examples is not None \
                else {"messages": row_messages(row)}
            problem = compare_row(e1_tokenizer, encoded[0], encoded[1], format_row(example))
        if problem is None:
            identical += 1
        elif len(mismatches) < MAX_MISMATCHES:
            mismatches.append({"row": index, "position_in_set": position, **problem})
    return {"rows": len(indices), "identical": identical,
            "all_identical": identical == len(indices), "mismatches": mismatches}


def config_from_recipe(config: Dict[str, Any]):
    from soup_cli.config.loader import load_config_from_string

    return load_config_from_string(recipe_text(config))


def labels(args: argparse.Namespace, record: Dict[str, Any]) -> None:
    harness = confirm.load_measured_harness()
    stamp(record, harness, args.split)
    rows = harness.load_rows(args.data)
    split = load_json(args.split)
    confirm_split = load_json(args.confirm_split)
    harness.check_group_disjoint(rows, split)
    confirm.check_confirmation(harness, rows, split, confirm_split)
    cfg = config_from_recipe(recipe(args.base, ARM_TARGETS["A"], seed=0, lr=LR,
                                    quantization="4bit"))
    soup_tokenizer, format_row = soup_format_row(cfg)
    e1_tokenizer = harness.load_tokenizer(MODEL)
    record.update(template_sha256=text_sha256(CHAT_TEMPLATE), template=CHAT_TEMPLATE,
                  base=args.base, confirm_split_sha256=confirm.file_sha256(args.confirm_split))
    sets = {}
    for name, indices in (("train", split["train"]), ("E", split["eval"]),
                          ("C", confirm_split["eval"])):
        sets[name] = check_labels(harness, e1_tokenizer, soup_tokenizer, format_row, rows,
                                  indices)
    record["sets"] = sets
    record["status"] = "ok" if all(s["all_identical"] for s in sets.values()) \
        else "void_labels"


# =====================================================================
# Preflight
# =====================================================================
def preflight_cases(base: str) -> Dict[str, Dict[str, Any]]:
    targets_as = ARM_TARGETS["AS"]
    common = {"seed": 0, "lr": LR, "quantization": "4bit"}
    return {
        "A": recipe(base, ARM_TARGETS["A"], **common),
        "AS": recipe(base, targets_as, **common),
        "AE_target_parameters": recipe(base, targets_as, **common, lora_extra={
            "target_parameters": ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]}),
        "AE_moe_lora": recipe(base, targets_as, **common, training_extra={"moe_lora": True}),
        "AE_target_modules_experts": recipe(base, (*targets_as, "experts"), **common),
    }


def _tensor_kind(name: str) -> str:
    module = name.split(".lora_")[0]
    leaf = module.rsplit(".", 1)[-1]
    if ".shared_expert." in module:
        return "shared_expert." + leaf
    if ".experts" in module:
        return "experts." + leaf
    return leaf


def _part(part) -> Optional[Dict[str, int]]:
    return None if part is None else {"cpu_bytes": part.cpu_bytes, "cuda_bytes": part.cuda_bytes}


def preflight_case(base: str, text: str, model_config) -> Dict[str, Any]:
    """Load the config as Soup does, then Soup's streamed adapter plan and B4 budget (meta)."""
    import io

    from rich.console import Console

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.utils.fp8_source import stream_double_quant
    from soup_cli.utils.layer_shard import QUANT_NF4
    from soup_cli.utils.stream_adapter_preflight import (
        build_stream_adapter_plan,
        check_stream_adapter_budget,
    )

    out: Dict[str, Any] = {"config": text, "config_sha256": text_sha256(text)}
    try:
        cfg = load_config_from_string(text)
    except ValueError as exc:
        out.update(outcome="refused_at_load", message=str(exc))
        return out
    buffer = io.StringIO()
    console = Console(file=buffer, width=400)
    try:
        tcfg = cfg.training
        double_quant = stream_double_quant(tcfg, model_config, quant=QUANT_NF4)
        plan = build_stream_adapter_plan(base, model_config, tcfg, dtype="bfloat16",
                                         quant=QUANT_NF4, double_quant=double_quant,
                                         trust_remote_code=False, on_cuda=True,
                                         console=console)
        check_stream_adapter_budget(plan, available_cuda_bytes=None, console=console)
    except Exception as exc:  # noqa: BLE001 - the refusal is the measurement
        out.update(outcome="error", exception=type(exc).__name__, message=str(exc),
                   console=buffer.getvalue())
        return out
    budget = plan.budget
    out.update(
        outcome="plan_built", double_quant=double_quant,
        trainable_parameters=sum(math.prod(t.shape) for t in plan.tensors if t.trainable),
        lora_tensors=len(plan.tensors),
        lora_tensors_by_kind=dict(sorted(Counter(_tensor_kind(t.name)
                                                 for t in plan.tensors).items())),
        routed_expert_adapter=any(".mlp.experts" in t.name for t in plan.tensors),
        tensor_dtypes=sorted({t.dtype for t in plan.tensors}),
        target_modules=sorted(plan.lora_config.target_modules or []),
        target_parameters=plan.lora_config.target_parameters,
        budget={"device": "cuda", "profile": budget.profile,
                "stored_parameters": budget.stored_parameters,
                "trainable_parameters": budget.trainable_parameters,
                "weights": _part(budget.weights), "gradients": _part(budget.gradients),
                "optimizer": _part(budget.optimizer), "total": _part(budget.total),
                "unsupported_reason": budget.unsupported_reason},
        console=buffer.getvalue())
    return out


def at_entry() -> Dict[str, Any]:
    import peft

    from soup_cli.config.schema import LoraConfig

    roots = {"repo": REPO, "peft": pathlib.Path(peft.__file__).resolve().parent}
    citations = []
    for cite in AT_CITATIONS:
        path = roots[cite["root"]] / cite["path"]
        first, last = cite["lines"]
        lines = path.read_text(encoding="utf-8").splitlines()[first - 1:last]
        citations.append({"file": f"{'peft/' if cite['root'] == 'peft' else ''}{cite['path']}",
                          "lines": f"{first}-{last}", "says": cite["says"],
                          "text": "\n".join(lines),
                          "verified": any(cite["expect"] in line for line in lines)})
    return {"outcome": "no_mechanism",
            "reason": "no Soup mechanism selects a per-layer subset of the fused routed "
                      "experts: there is no such field, target_parameters is refused under "
                      "streaming, and PEFT's ParamWrapper adapts a whole fused tensor",
            "soup_lora_fields": sorted(LoraConfig.model_fields), "peft": peft.__version__,
            "citations": citations,
            "verified": all(c["verified"] for c in citations)}


def preflight(args: argparse.Namespace, record: Dict[str, Any]) -> None:
    from transformers import AutoConfig

    harness = confirm.load_measured_harness()
    stamp(record, harness, None)
    record["base"] = args.base
    model_config = AutoConfig.from_pretrained(args.base, trust_remote_code=False)
    cases = {name: preflight_case(args.base, recipe_text(config), model_config)
             for name, config in preflight_cases(args.base).items()}
    cases["AT"] = at_entry()
    record["cases"] = cases
    record["status"] = "ok" if cases["AT"]["verified"] else "stale_citation"


# =====================================================================
# Parity
# =====================================================================
def lora_parameters(model) -> Dict[str, Any]:
    from soup_cli.utils.layer_stream_runtime import canonical_named_parameters

    return {name: p for name, p in canonical_named_parameters(model)
            if p.requires_grad and "lora_" in name}


def make_non_vacuous(params: Dict[str, Any]) -> None:
    """LoRA-B to N(0, 0.02^2) as bitexact.py does, so parity cannot pass on zero deltas."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(PARITY_LORA_B_SEED)
    with torch.no_grad():
        for name in sorted(params):
            if "lora_B" in name:
                p = params[name]
                p.copy_((torch.randn(p.shape, generator=generator) * PARITY_LORA_B_STD)
                        .to(device=p.device, dtype=p.dtype))


def resident_reference(base: str, streamed, *, device: str, quant: str, dtype: str,
                       double_quant: bool):
    """Resident model on the streamed model's numerics: same NF4 config, attention, experts."""
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM

    from soup_cli.utils.layer_stream_runtime import build_nf4_config
    from soup_cli.utils.peft_wiring import apply_pre_lora_patches

    config = streamed.get_base_model().config
    kwargs: Dict[str, Any] = {"device_map": {"": device}, "dtype": getattr(torch, dtype)}
    for key, attr in (("attn_implementation", "_attn_implementation"),
                      ("experts_implementation", "_experts_implementation")):
        if getattr(config, attr, None):
            kwargs[key] = getattr(config, attr)
    if quant == "nf4":
        kwargs["quantization_config"] = build_nf4_config(dtype, double_quant=double_quant)
    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad = False
    apply_pre_lora_patches(model, base)
    model = get_peft_model(model, copy.deepcopy(streamed.peft_config["default"]))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, {k: str(v) for k, v in kwargs.items()}


def gradient_comparison(streamed: Dict[str, Any], reference: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    per: Dict[str, Dict[str, Any]] = {}
    missing = sorted(n for n in streamed if streamed[n].grad is None or reference[n].grad is None)
    num = den = 0.0
    for name in sorted(set(streamed) - set(missing)):
        gs = streamed[name].grad.detach().double().cpu().flatten()
        gr = reference[name].grad.detach().double().cpu().flatten()
        diff = gs - gr
        num += float(diff.square().sum())
        den += float(gr.square().sum())
        norms = float(gs.norm()) * float(gr.norm())
        per[name] = {"max_abs": float(diff.abs().max()),
                     "relative_l2": float(diff.norm()) / float(gr.norm())
                     if float(gr.norm()) > 0 else math.inf,
                     "cosine": float(gs @ gr) / norms if norms > 0 else math.nan,
                     "bit_exact": bool(torch.equal(streamed[name].grad, reference[name].grad))}
    cosines = [v["cosine"] for v in per.values()]
    return {"tensors": per, "missing": missing,
            "global_relative_l2": math.sqrt(num / den) if den > 0 else math.inf,
            "min_cosine": min(cosines) if cosines else math.nan,
            "max_abs": max((v["max_abs"] for v in per.values()), default=math.nan),
            "bit_exact": bool(per) and not missing and all(v["bit_exact"] for v in per.values())}


def collated_batches(trainer, *, steps: int, rows_per_batch: int, device: str) -> List[dict]:
    """The trainer's first formatted rows through the trainer's own data collator."""
    dataset = trainer.train_dataset
    need = steps * rows_per_batch
    if len(dataset) < need:
        raise ValueError(f"the trainer's dataset holds {len(dataset)} rows; parity needs {need}")
    batches = []
    for step in range(steps):
        rows = [dataset[i] for i in range(step * rows_per_batch, (step + 1) * rows_per_batch)]
        collated = trainer.data_collator(rows)
        batches.append({k: collated[k].to(device) for k in FORWARD_KEYS if k in collated})
    return batches


def parity_passed(result: Dict[str, Any]) -> bool:
    gradient = result["gradient_step0"]
    return (result["streamed_meta_parameters"] > 0 and result["tier"] == "ram"
            and all(s["relative"] <= PARITY_LOSS_RELATIVE for s in result["steps"])
            and not gradient["missing"]
            and gradient["global_relative_l2"] <= PARITY_GRAD_RELATIVE_L2
            and gradient["min_cosine"] >= PARITY_GRAD_COSINE)


def parity(args: argparse.Namespace, record: Dict[str, Any]) -> None:
    import torch
    from transformers import AutoConfig

    from soup_cli.utils.fp8_source import stream_double_quant
    from soup_cli.utils.layer_shard import QUANT_NF4, QUANT_NONE
    from soup_cli.utils.layer_stream import resolve_stream_dtype

    harness = confirm.load_measured_harness()
    stamp(record, harness, args.split)
    record.update(quant=args.quant, device=args.device, rows_per_batch=args.rows_per_batch,
                  steps=args.steps, arm="AS", seed=0,
                  rule={"loss_relative": PARITY_LOSS_RELATIVE,
                        "grad_relative_l2": PARITY_GRAD_RELATIVE_L2,
                        "grad_cosine": PARITY_GRAD_COSINE, "adamw": PARITY_ADAMW,
                        "lora_b_seed": PARITY_LORA_B_SEED, "lora_b_std": PARITY_LORA_B_STD})
    if args.quant == "nf4" and not args.device.startswith("cuda"):
        raise ValueError("parity --quant nf4 runs on CUDA; CPU is allowed with --quant none only")
    rows = harness.load_rows(args.data)
    split = load_json(args.split)
    out = pathlib.Path(args.out).resolve()
    run_dir = out.parent / f"{out.stem}-run"
    quantization = "4bit" if args.quant == "nf4" else "none"
    text = prepare_run_dir(run_dir, recipe(args.base, ARM_TARGETS["AS"], seed=0, lr=LR,
                                           quantization=quantization),
                           rows, split["train"][:args.steps * args.rows_per_batch])
    record["recipe"] = {"run_dir": str(run_dir), "text": text, "sha256": text_sha256(text)}
    with working_directory(run_dir):
        cfg, dataset, wrapper = soup_load("soup.yaml", args.device)
        try:
            wrapper.setup(dataset)
            trainer = wrapper.trainer
            streamed = trainer.model
            stats = wrapper._stream_runtime.stats()
            dtype = resolve_stream_dtype(args.device)
            quant = QUANT_NF4 if args.quant == "nf4" else QUANT_NONE
            double_quant = stream_double_quant(
                cfg.training, AutoConfig.from_pretrained(args.base), quant=quant)
            named_s = lora_parameters(streamed)
            adapter_dtypes = sorted({str(p.dtype) for p in named_s.values()})
            if len(adapter_dtypes) != 1:
                raise ValueError(f"streamed adapters carry several dtypes: {adapter_dtypes}")
            record.update(tier=stats["tier"], stream=jsonable(stats), store_dtype=dtype,
                          double_quant=double_quant, adapter_dtype=adapter_dtypes[0],
                          streamed_meta_parameters=sum(1 for p in streamed.parameters()
                                                       if p.is_meta),
                          lora_tensors=len(named_s))
            make_non_vacuous(named_s)
            reference, record["reference_kwargs"] = resident_reference(
                args.base, streamed, device=args.device, quant=args.quant, dtype=dtype,
                double_quant=double_quant)
            named_r = lora_parameters(reference)
            if set(named_s) != set(named_r):
                raise ValueError(
                    "LoRA tensors do not match by canonical name: streamed-only "
                    f"{sorted(set(named_s) - set(named_r))[:4]}, reference-only "
                    f"{sorted(set(named_r) - set(named_s))[:4]}")
            with torch.no_grad():
                for name, param in named_r.items():
                    param.data = param.data.to(named_s[name].dtype)
                    param.copy_(named_s[name])
            batches = collated_batches(trainer, steps=args.steps,
                                       rows_per_batch=args.rows_per_batch, device=args.device)
            record["batch_tokens"] = [int((b["labels"] != IGNORE_INDEX).sum()) for b in batches]
            names = sorted(named_s)
            optimizers = [torch.optim.AdamW([params[n] for n in names], **PARITY_ADAMW)
                          for params in (named_s, named_r)]
            streamed.train()
            reference.train()
            steps = []
            for step, batch in enumerate(batches):
                losses = []
                for model, optimizer in zip((streamed, reference), optimizers):
                    optimizer.zero_grad(set_to_none=True)
                    loss = model(**batch).loss
                    loss.backward()
                    losses.append(float(loss.detach()))
                if step == 0:
                    record["gradient_step0"] = gradient_comparison(named_s, named_r)
                for optimizer in optimizers:
                    optimizer.step()
                steps.append({"step": step, "streamed": losses[0], "resident": losses[1],
                              "relative": abs(losses[0] - losses[1]) / abs(losses[1]),
                              "bit_exact": losses[0] == losses[1]})
            record["steps"] = steps
        finally:
            wrapper._close_stream_runtime()
    record["bit_exact"] = {"losses": all(s["bit_exact"] for s in record["steps"]),
                           "gradients_step0": record["gradient_step0"]["bit_exact"]}
    record["passed"] = parity_passed(record)
    record["status"] = "ok" if record["passed"] else "not_passed"


# =====================================================================
# Noise (post hoc)
# =====================================================================
NOISE_NOTE = ("added 2026-10-13 after parity-qwen35 did not pass; a diagnostic only, it never "
              "changes the parity verdict")


def _layer_of(name: str) -> Optional[int]:
    _, found, rest = name.partition(".layers.")
    head = rest.split(".", 1)[0]
    return int(head) if found and head.isdigit() else None


def compare_gradient_sets(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Two gradient sets over the same adapter tensors: global, per layer and per tensor."""
    import torch

    shared = sorted(set(left) & set(right))
    per: Dict[str, Dict[str, Any]] = {}
    layers: Dict[int, List[float]] = {}
    num = den = 0.0
    for name in shared:
        a = left[name].detach().double().cpu().flatten()
        b = right[name].detach().double().cpu().flatten()
        diff = a - b
        num += float(diff.square().sum())
        den += float(b.square().sum())
        diff_norm, a_norm, b_norm = float(diff.norm()), float(a.norm()), float(b.norm())
        relative = diff_norm / b_norm if b_norm > 0 else (0.0 if diff_norm == 0 else math.inf)
        per[name] = {"max_abs": float(diff.abs().max()) if diff.numel() else 0.0,
                     "relative_l2": relative,
                     "cosine": float(a @ b) / (a_norm * b_norm) if a_norm * b_norm > 0 else None,
                     "bit_exact": bool(torch.equal(left[name], right[name]))}
        layer = _layer_of(name)
        if layer is not None:
            layers.setdefault(layer, []).append(relative)
    cosines = [v["cosine"] for v in per.values() if v["cosine"] is not None]
    missing = sorted(set(left) ^ set(right))
    return {
        "tensors_compared": len(shared), "missing": missing,
        "global_relative_l2": math.sqrt(num / den) if den > 0 else (0.0 if num == 0 else math.inf),
        "min_cosine": min(cosines) if cosines else None,
        "max_abs": max((v["max_abs"] for v in per.values()), default=0.0),
        "bit_exact": bool(per) and not missing and all(v["bit_exact"] for v in per.values()),
        "per_layer": {str(layer): {"tensors": len(values),
                                   "mean_relative_l2": sum(values) / len(values),
                                   "max_relative_l2": max(values)}
                      for layer, values in sorted(layers.items())},
        "tensors": per}


def compare_losses(left: Sequence[float], right: Sequence[float]) -> Dict[str, Any]:
    steps = [{"step": i, "left": a, "right": b,
              "relative": abs(a - b) / abs(b) if b else (0.0 if a == b else math.inf),
              "bit_exact": a == b} for i, (a, b) in enumerate(zip(left, right))]
    return {"steps": steps, "max_relative": max((s["relative"] for s in steps), default=0.0),
            "bit_exact": len(left) == len(right) and all(s["bit_exact"] for s in steps)}


def _restore(params: Dict[str, Any], snapshot: Dict[str, Any]) -> None:
    import torch

    with torch.no_grad():
        for name, param in params.items():
            param.copy_(snapshot[name])
            param.grad = None


def step0_gradients(model, params: Dict[str, Any], snapshot: Dict[str, Any],
                    batch: dict) -> Tuple[float, Dict[str, Any]]:
    """Loss and adapter gradients of one forward+backward from ``snapshot``."""
    _restore(params, snapshot)
    loss = model(**batch).loss
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in params.items() if p.grad is not None}
    value = float(loss.detach())
    _restore(params, snapshot)
    return value, grads


def adamw_trajectory(model, params: Dict[str, Any], snapshot: Dict[str, Any],
                     batches: Sequence[dict]) -> List[float]:
    """Per-step losses of PARITY_ADAMW steps from ``snapshot`` with a fresh optimizer."""
    import torch

    _restore(params, snapshot)
    optimizer = torch.optim.AdamW([params[n] for n in sorted(params)], **PARITY_ADAMW)
    losses = []
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        losses.append(float(loss.detach()))
        optimizer.step()
    del optimizer
    _restore(params, snapshot)
    return losses


def _noise_comparison(left: Dict[str, Any], right: Dict[str, Any], i: int,
                      j: int) -> Dict[str, Any]:
    return {"gradient_step0": compare_gradient_sets(left["step0"][i][1], right["step0"][j][1]),
            "loss_step0": compare_losses([left["step0"][i][0]], [right["step0"][j][0]]),
            "trajectory": compare_losses(left["trajectories"][i], right["trajectories"][j])}


def noise(args: argparse.Namespace, record: Dict[str, Any]) -> None:
    """Parity's build, then each side against itself and repeat 1 of each against the other."""
    import torch
    from transformers import AutoConfig

    from soup_cli.utils.fp8_source import stream_double_quant
    from soup_cli.utils.layer_shard import QUANT_NF4, QUANT_NONE
    from soup_cli.utils.layer_stream import resolve_stream_dtype

    harness = confirm.load_measured_harness()
    stamp(record, harness, args.split)
    record.update(post_hoc=NOISE_NOTE, quant=args.quant, device=args.device,
                  rows_per_batch=args.rows_per_batch, steps=args.steps, arm="AS", seed=0,
                  repeats=2, adamw=PARITY_ADAMW, lora_b_seed=PARITY_LORA_B_SEED,
                  lora_b_std=PARITY_LORA_B_STD,
                  determinism={"deterministic_algorithms":
                               torch.are_deterministic_algorithms_enabled(),
                               "cudnn_deterministic": torch.backends.cudnn.deterministic,
                               "cudnn_benchmark": torch.backends.cudnn.benchmark})
    if args.quant == "nf4" and not args.device.startswith("cuda"):
        raise ValueError("noise --quant nf4 runs on CUDA; CPU is allowed with --quant none only")
    rows = harness.load_rows(args.data)
    split = load_json(args.split)
    out = pathlib.Path(args.out).resolve()
    run_dir = out.parent / f"{out.stem}-run"
    quantization = "4bit" if args.quant == "nf4" else "none"
    text = prepare_run_dir(run_dir, recipe(args.base, ARM_TARGETS["AS"], seed=0, lr=LR,
                                           quantization=quantization),
                           rows, split["train"][:args.steps * args.rows_per_batch])
    record["recipe"] = {"run_dir": str(run_dir), "text": text, "sha256": text_sha256(text)}
    runs: Dict[str, Dict[str, Any]] = {}
    with working_directory(run_dir):
        cfg, dataset, wrapper = soup_load("soup.yaml", args.device)
        try:
            wrapper.setup(dataset)
            trainer = wrapper.trainer
            streamed = trainer.model
            stats = wrapper._stream_runtime.stats()
            dtype = resolve_stream_dtype(args.device)
            quant = QUANT_NF4 if args.quant == "nf4" else QUANT_NONE
            double_quant = stream_double_quant(
                cfg.training, AutoConfig.from_pretrained(args.base), quant=quant)
            named_s = lora_parameters(streamed)
            adapter_dtypes = sorted({str(p.dtype) for p in named_s.values()})
            if len(adapter_dtypes) != 1:
                raise ValueError(f"streamed adapters carry several dtypes: {adapter_dtypes}")
            record.update(tier=stats["tier"], stream=jsonable(stats), store_dtype=dtype,
                          double_quant=double_quant, adapter_dtype=adapter_dtypes[0],
                          streamed_meta_parameters=sum(1 for p in streamed.parameters()
                                                       if p.is_meta),
                          lora_tensors=len(named_s))
            make_non_vacuous(named_s)
            reference, record["reference_kwargs"] = resident_reference(
                args.base, streamed, device=args.device, quant=args.quant, dtype=dtype,
                double_quant=double_quant)
            named_r = lora_parameters(reference)
            if set(named_s) != set(named_r):
                raise ValueError(
                    "LoRA tensors do not match by canonical name: streamed-only "
                    f"{sorted(set(named_s) - set(named_r))[:4]}, reference-only "
                    f"{sorted(set(named_r) - set(named_s))[:4]}")
            with torch.no_grad():
                for name, param in named_r.items():
                    param.data = param.data.to(named_s[name].dtype)
                    param.copy_(named_s[name])
            batches = collated_batches(trainer, steps=args.steps,
                                       rows_per_batch=args.rows_per_batch, device=args.device)
            record["batch_tokens"] = [int((b["labels"] != IGNORE_INDEX).sum()) for b in batches]
            sides = {"streamed": (streamed, named_s), "resident": (reference, named_r)}
            snapshots = {side: {n: p.detach().clone() for n, p in params.items()}
                         for side, (_model, params) in sides.items()}
            record["snapshots_identical"] = all(
                torch.equal(snapshots["streamed"][n], snapshots["resident"][n]) for n in named_s)
            streamed.train()
            reference.train()
            for side, (model, params) in sides.items():
                snap = snapshots[side]
                runs[side] = {
                    "step0": [step0_gradients(model, params, snap, batches[0])
                              for _ in range(2)],
                    "trajectories": [adamw_trajectory(model, params, snap, batches)
                                     for _ in range(2)]}
        finally:
            wrapper._close_stream_runtime()
    record["step0_losses"] = {side: [loss for loss, _grads in run["step0"]]
                              for side, run in runs.items()}
    record["trajectories"] = {side: run["trajectories"] for side, run in runs.items()}
    streamed_run, resident_run = runs["streamed"], runs["resident"]
    record["comparisons"] = {
        "streamed_vs_streamed": _noise_comparison(streamed_run, streamed_run, 0, 1),
        "resident_vs_resident": _noise_comparison(resident_run, resident_run, 0, 1),
        "streamed_vs_resident": _noise_comparison(streamed_run, resident_run, 0, 0)}
    record["status"] = "ok"


# =====================================================================
# Train
# =====================================================================
def bytes_per_parameter(named_parameters: Iterable[Tuple[str, Any]],
                        optimizer) -> Dict[str, Any]:
    """Persistent bytes per trainable element, read off the live tensors after a step.

    Weight + gradient + every optimizer-state tensor shaped like the parameter (AdamW's
    exp_avg and exp_avg_sq); scalar state such as ``step`` is listed apart. Every
    parameter must give the same layout, else ``bytes_per_parameter`` is None and
    ``problem`` says why.
    """
    import torch

    inner = getattr(optimizer, "optimizer", optimizer)
    groups: Dict[tuple, Dict[str, Any]] = {}
    scalar: Dict[str, set] = {}
    for name, param in named_parameters:
        state = inner.state.get(param, {})
        shaped = {k: v for k, v in state.items()
                  if torch.is_tensor(v) and tuple(v.shape) == tuple(param.shape)}
        for key, value in state.items():
            if key not in shaped:
                scalar.setdefault(key, set()).add(
                    f"{value.dtype} {tuple(value.shape)} {value.device}"
                    if torch.is_tensor(value) else type(value).__name__)
        grad = param.grad
        layout = (str(param.dtype), None if grad is None else str(grad.dtype),
                  tuple(sorted((k, str(v.dtype)) for k, v in shaped.items())))
        group = groups.setdefault(layout, {
            "weight_dtype": layout[0], "grad_dtype": layout[1], "state_dtypes": dict(layout[2]),
            "bytes": param.element_size() + (0 if grad is None else grad.element_size())
            + sum(v.element_size() for v in shaped.values()),
            "parameters": 0, "elements": 0, "example": name})
        group["parameters"] += 1
        group["elements"] += param.numel()
    breakdown = list(groups.values())
    problem = None
    if not breakdown:
        problem = "no trainable parameters"
    elif len(breakdown) > 1:
        problem = f"{len(breakdown)} different per-parameter layouts"
    elif breakdown[0]["grad_dtype"] is None:
        problem = "no gradient at the optimizer step"
    elif not breakdown[0]["state_dtypes"]:
        problem = "no optimizer state shaped like the parameters"
    return {"bytes_per_parameter": None if problem else breakdown[0]["bytes"],
            "breakdown": breakdown, "scalar_state": {k: sorted(v) for k, v in scalar.items()},
            "problem": problem}


def make_recorder(sync: Callable[[], None]):
    """A TrainerCallback: per-step log and the first optimizer step's byte layout."""
    from transformers import TrainerCallback

    class StepRecorder(TrainerCallback):
        def __init__(self):
            self.entries: Dict[int, Dict[str, Any]] = {}
            self.bytes: Optional[Dict[str, Any]] = None
            self.pending_tokens = 0
            self.started: Optional[float] = None

        def observe(self, inputs) -> None:
            found = inputs.get("labels") if hasattr(inputs, "get") else None
            if found is not None:
                self.pending_tokens += int((found != IGNORE_INDEX).sum().item())

        def entry(self, step: int) -> Dict[str, Any]:
            return self.entries.setdefault(step, {"step": step})

        def on_step_begin(self, args, state, control, **kwargs):
            sync()
            self.started = time.perf_counter()

        def on_optimizer_step(self, args, state, control, **kwargs):
            if self.bytes is None:
                model = kwargs["model"]
                self.bytes = bytes_per_parameter(
                    [(n, p) for n, p in model.named_parameters() if p.requires_grad],
                    kwargs["optimizer"])

        def on_step_end(self, args, state, control, **kwargs):
            sync()
            self.entry(state.global_step - 1).update(
                seconds=time.perf_counter() - self.started, tokens=self.pending_tokens)
            self.pending_tokens = 0

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                self.entry(state.global_step - 1).update(
                    loss=float(logs["loss"]), grad_norm=logs.get("grad_norm"),
                    learning_rate=logs.get("learning_rate"))

        def log(self) -> List[Dict[str, Any]]:
            keys = ("step", "loss", "grad_norm", "learning_rate", "seconds", "tokens")
            return [{k: self.entries[s].get(k) for k in keys} for s in sorted(self.entries)]

    return StepRecorder()


def ram_state() -> Dict[str, Optional[int]]:
    try:
        import psutil
    except ImportError:
        return {"total": None, "available": None}
    memory = psutil.virtual_memory()
    return {"total": memory.total, "available": memory.available}


def host_peak_rss() -> Dict[str, Any]:
    try:
        import resource
    except ImportError:
        import psutil

        peak = getattr(psutil.Process().memory_info(), "peak_wset", None)
        return {"bytes": peak, "source": "psutil peak_wset"}
    scale = 1 if sys.platform == "darwin" else 1024
    return {"bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale,
            "source": "resource.getrusage ru_maxrss"}


def safetensors_summary(path: pathlib.Path) -> Dict[str, Any]:
    """Tensor count and dtypes from the file header; no tensor is loaded."""
    with open(path, "rb") as handle:
        size = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(size))
    header.pop("__metadata__", None)
    return {"sha256": confirm.file_sha256(str(path)), "tensors": len(header),
            "dtypes": dict(sorted(Counter(v["dtype"] for v in header.values()).items())),
            "parameters": sum(math.prod(v["shape"]) for v in header.values())}


def train(args: argparse.Namespace, record: Dict[str, Any]) -> None:
    import torch

    harness = confirm.load_measured_harness()
    stamp(record, harness, args.split)
    record.update(model=MODEL, arm=args.arm, seed=args.seed, lr=args.lr,
                  targets=list(ARM_TARGETS[args.arm]), quantization=args.quantization,
                  device=args.device, limit_rows=args.limit_rows, base=args.base)
    rows = harness.load_rows(args.data)
    split = load_json(args.split)
    harness.check_group_disjoint(rows, split)
    indices = split["train"][:args.limit_rows] if args.limit_rows else split["train"]
    run_dir = pathlib.Path(args.run_dir).resolve()
    text = prepare_run_dir(run_dir, recipe(args.base, ARM_TARGETS[args.arm], seed=args.seed,
                                           lr=args.lr, quantization=args.quantization),
                           rows, indices)
    record["recipe"] = {"run_dir": str(run_dir), "text": text, "sha256": text_sha256(text)}
    on_cuda = args.device.startswith("cuda")
    sync = torch.cuda.synchronize if on_cuda else (lambda: None)
    with working_directory(run_dir):
        cfg, dataset, wrapper = soup_load("soup.yaml", args.device)
        soup_tokenizer, format_row = soup_format_row(cfg)
        record["label_check"] = check_labels(harness, harness.load_tokenizer(MODEL),
                                             soup_tokenizer, format_row, rows, indices,
                                             examples=dataset["train"])
        if not record["label_check"]["all_identical"]:
            record["status"] = "void_labels"
            return
        started_training = False
        recorder = make_recorder(sync)
        try:
            wrapper.setup(dataset)
            trainer = wrapper.trainer
            trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
            record.update(trainable_parameters=trainable,
                          e1_trainable_parameters=E1_TRAINABLE[args.arm])
            stats = wrapper._stream_runtime.stats()
            record.update(tier=stats["tier"], stream={"after_setup": jsonable(stats)})
            if (args.limit_rows is None and args.quantization == "4bit"
                    and trainable != E1_TRAINABLE[args.arm]):
                record["status"] = "void_targets"
                return
            trainer.add_callback(recorder)
            original_step = trainer.training_step

            def training_step(model, inputs, *rest, **kwargs):
                recorder.observe(inputs)
                return original_step(model, inputs, *rest, **kwargs)

            trainer.training_step = training_step
            record["ram"] = {"before_train": ram_state()}
            if on_cuda:
                record["allocated_before_train_bytes"] = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            started_training = True
            try:
                result = wrapper.train()
            except RuntimeError as exc:
                if not any(marker in str(exc) for marker in NONFINITE_MARKERS):
                    raise
                record.update(status="void_nonfinite", **error_fields(exc))
                result = None
            if on_cuda:
                record.update(peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved())
            else:
                record.update(peak_cuda_bytes=None, peak_reserved_bytes=None,
                              allocated_before_train_bytes=None)
            record["ram"]["after_train"] = ram_state()
            record["host_peak_rss"] = host_peak_rss()
            try:
                record["stream"]["after_train"] = jsonable(wrapper._stream_runtime.stats())
            except Exception as exc:  # noqa: BLE001 - the runtime is closed after training
                record["stream"]["after_train"] = f"unavailable: {type(exc).__name__}: {exc}"
        finally:
            log = recorder.log()
            record["train"] = {"steps": len(log), "log": log}
            if not started_training:
                wrapper._close_stream_runtime()
    if record.get("status") == "void_nonfinite":
        return
    record["soup_result"] = jsonable(result)
    layout = recorder.bytes or {"bytes_per_parameter": None, "breakdown": [],
                                "scalar_state": {}, "problem": "no optimizer step observed"}
    record.update(bytes_per_parameter=layout["bytes_per_parameter"], bytes_breakdown=layout,
                  adapter_dtype=layout["breakdown"][0]["weight_dtype"]
                  if len(layout["breakdown"]) == 1 else None)
    out_dir = run_dir / "out"
    record["adapter"] = {
        "dir": str(out_dir),
        "model": safetensors_summary(out_dir / "adapter_model.safetensors"),
        "config_sha256": confirm.file_sha256(str(out_dir / "adapter_config.json")),
        "config": load_json(str(out_dir / "adapter_config.json"))}
    if not all(isinstance(e["loss"], float) and math.isfinite(e["loss"]) for e in log):
        record["status"] = "void_nonfinite"
    elif layout["problem"]:
        record["status"] = "void_bytes"
    else:
        record["status"] = "ok"


# =====================================================================
# Score
# =====================================================================
def adapter_config_problems(config: Dict[str, Any], arm: str, harness,
                            scales: Iterable[float]) -> List[str]:
    """What in Soup's adapter_config.json would make the harness compute something else."""
    problems = []
    r, alpha = config.get("r"), config.get("lora_alpha")
    scales = set(scales)
    if r != harness.RANK:
        problems.append(f"r {r!r} is not the harness rank {harness.RANK}")
    if not isinstance(r, int) or not isinstance(alpha, (int, float)) or r <= 0 \
            or len(scales) != 1 or alpha / r not in scales:
        problems.append(f"lora_alpha/r = {alpha!r}/{r!r} is not the harness scale "
                        f"{sorted(scales)}")
    for key, wanted in (("lora_dropout", 0.0), ("use_rslora", False), ("use_dora", False),
                        ("bias", "none")):
        if config.get(key, wanted) != wanted:
            problems.append(f"{key} {config.get(key)!r}, not {wanted!r}")
    for key in ("target_parameters", "modules_to_save", "rank_pattern", "alpha_pattern"):
        if config.get(key):
            problems.append(f"{key} {config.get(key)!r} is set")
    targets = config.get("target_modules")
    if not isinstance(targets, list) or sorted(targets) != sorted(ARM_TARGETS[arm]):
        problems.append(f"target_modules {targets!r} are not arm {arm}'s "
                        f"{sorted(ARM_TARGETS[arm])}")
    return problems


def copy_adapter_into_harness(model, harness, adapter_dir: str, arm: str) -> Dict[str, Any]:
    """Copy Soup's saved LoRA tensors into the harness's attached adapters, exactly.

    ``base_model.model.<module>.lora_A.weight`` / ``.lora_B.weight`` go to the harness
    ``LoRALinear`` at ``<module>``. Refuses a config the harness would compute differently,
    a missing or unexpected key and a shape mismatch; values are upcast to the harness's
    dtype (its fp32 masters) and must arrive unchanged.
    """
    import torch
    from safetensors.torch import load_file

    directory = pathlib.Path(adapter_dir)
    modules = {name: m for name, m in model.named_modules() if isinstance(m, harness.LoRALinear)}
    config = load_json(str(directory / "adapter_config.json"))
    problems = adapter_config_problems(config, arm, harness, (m.scale for m in modules.values()))
    if problems:
        raise ValueError("adapter_config.json: " + "; ".join(problems))
    tensors = load_file(str(directory / "adapter_model.safetensors"))
    expected = {f"base_model.model.{path}.{part}.weight": getattr(module, part)
                for path, module in modules.items() for part in ("lora_A", "lora_B")}
    missing = sorted(set(expected) - set(tensors))
    unexpected = sorted(set(tensors) - set(expected))
    if missing or unexpected:
        raise ValueError(f"adapter keys: {len(missing)} missing {missing[:3]}, "
                         f"{len(unexpected)} unexpected {unexpected[:3]}")
    shapes = [f"{k} {tuple(tensors[k].shape)} vs {tuple(p.shape)}"
              for k, p in expected.items() if tuple(tensors[k].shape) != tuple(p.shape)]
    if shapes:
        raise ValueError(f"adapter shape mismatch: {shapes[:3]}")
    with torch.no_grad():
        for key, param in expected.items():
            param.copy_(tensors[key].to(device=param.device, dtype=param.dtype))
    changed = [k for k, p in expected.items()
               if not torch.equal(p.detach().cpu(), tensors[k].to(p.dtype))]
    if changed:
        raise ValueError(f"the copy is not exact for {changed[:3]}")
    return {"tensors": len(expected), "modules": len(modules),
            "source_dtypes": sorted({str(t.dtype) for t in tensors.values()}),
            "harness_dtypes": sorted({str(p.dtype) for p in expected.values()})}


def load_scoring_base(harness, args: argparse.Namespace):
    import torch

    if args.base_path is None:
        return harness.load_model(MODEL, args.device, torch.bfloat16)
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.base_path, dtype=torch.bfloat16, attn_implementation="eager",
        experts_implementation="grouped_mm", low_cpu_mem_usage=True)
    return model.to(args.device)


def score(args: argparse.Namespace, record: Dict[str, Any]) -> None:
    import torch

    harness = confirm.load_measured_harness()
    spec = harness.MODELS[MODEL]
    record.update(model=MODEL, repo=spec["repo"], revision=spec["revision"], arm=args.arm,
                  lr=args.lr, seed=args.seed, rank=harness.RANK, alpha=harness.ALPHA,
                  batch_rows=harness.BATCH_ROWS, regime="ram_tier",
                  limit_rows=args.limit_rows, base_path=args.base_path)
    stamp(record, harness, args.split)
    trained = load_json(args.train_record)
    if args.base_path is not None and args.limit_rows is None:
        raise ValueError("--base-path is for the smoke run only (with --limit-rows)")
    status = trained.get("status")
    if status == "smoke" and args.limit_rows:
        status = trained.get("underlying_status")
    if status != "ok":
        raise ValueError(f"train record status {trained.get('status')!r} "
                         f"(underlying {trained.get('underlying_status')!r}) is not ok")
    for key in ("arm", "seed", "lr", "split_sha256"):
        if trained.get(key) != record[key]:
            raise ValueError(f"train record {key} {trained.get(key)!r} != {record[key]!r}")
    adapter_file = pathlib.Path(args.adapter_dir) / "adapter_model.safetensors"
    if confirm.file_sha256(str(adapter_file)) != trained["adapter"]["model"]["sha256"]:
        raise ValueError(f"{adapter_file} is not the adapter the train record wrote")
    rows = harness.load_rows(args.data)
    split = load_json(args.split)
    confirm_split = load_json(args.confirm_split)
    harness.check_group_disjoint(rows, split)
    confirm.check_confirmation(harness, rows, split, confirm_split)
    e_indices = split["eval"][:args.limit_rows] if args.limit_rows else split["eval"]
    c_indices = confirm_split["eval"][:args.limit_rows] if args.limit_rows \
        else confirm_split["eval"]
    tokenizer = harness.load_tokenizer(MODEL)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None \
        else tokenizer.eos_token_id
    eval_items, references = harness.encode_split(rows, e_indices, tokenizer, spec["chat"])
    torch.manual_seed(args.seed)
    model = load_scoring_base(harness, args).eval()
    model.config.use_cache = False
    probe = eval_items[:harness.V4_ROWS]
    base_probe = harness.evaluate_nll(model, probe, pad_id, args.device, MICRO)["nll"]
    record["targets"] = harness.attach_adapters(model, args.arm, seed=args.seed)
    zero_probe = harness.evaluate_nll(model, probe, pad_id, args.device, MICRO)["nll"]
    record["v4"] = {"base": base_probe, "zero_adapter": zero_probe,
                    "passed": abs(base_probe - zero_probe) <= harness.V4_TOLERANCE}
    if not record["v4"]["passed"]:
        record["status"] = "void_v4"
        return
    trainable = sum(p.numel() for p in harness.adapter_parameters(model))
    record.update(trainable_parameters=trainable,
                  adapter_bytes_adamw_fp32=harness.adamw_fp32_bytes(trainable))
    if trainable != trained.get("trainable_parameters"):
        record["status"] = "void_targets"
        return
    record["adapter_copy"] = copy_adapter_into_harness(model, harness, args.adapter_dir,
                                                       args.arm)
    record["train"] = {"steps": trained["train"]["steps"], "log": trained["train"]["log"]}
    record["peak_cuda_bytes"] = trained.get("peak_cuda_bytes")
    model.config.use_cache = True
    record["eval"] = harness.evaluate_nll(model, eval_items, pad_id, args.device, MICRO)
    record["task"] = harness.score_task(model, tokenizer, [p for p, _full in eval_items],
                                        references, GEN_BATCH, args.device)
    record["gate"] = harness.score_gate(model, tokenizer, GEN_BATCH, args.device)
    record["confirm"] = {
        "split_sha256": confirm.file_sha256(args.confirm_split),
        "wrapper_sha256": confirm.file_sha256(confirm.__file__),
        **confirm.score_confirmation(harness, model, tokenizer, rows=rows, chat=spec["chat"],
                                     e_indices=e_indices, c_indices=c_indices,
                                     device=args.device, micro=MICRO, gen_batch=GEN_BATCH)}
    record["ram_tier"] = {
        "peak_cuda_bytes": trained.get("peak_cuda_bytes"),
        "peak_reserved_bytes": trained.get("peak_reserved_bytes"),
        "allocated_before_train_bytes": trained.get("allocated_before_train_bytes"),
        "trainable_parameters": trained["trainable_parameters"],
        "bytes_per_parameter": trained.get("bytes_per_parameter"),
        "bytes_breakdown": trained.get("bytes_breakdown"),
        "adapter_dtype": trained.get("adapter_dtype"),
        "train_record_sha256": confirm.file_sha256(args.train_record),
        "adapter_sha256": trained["adapter"]["model"]["sha256"],
        "adapter_config_sha256": trained["adapter"]["config_sha256"],
        "config_sha256": trained["recipe"]["sha256"],
        "stream": trained.get("stream"), "tier": trained.get("tier"),
        "label_check": trained.get("label_check"),
        "soup_source": trained.get("host", {}).get("soup_source"),
        "train_host": trained.get("host"),
        "scoring_base": "bf16 resident, measured harness load_model"
        if args.base_path is None else f"bf16 resident from --base-path {args.base_path}"}
    finite = all(isinstance(e.get("loss"), float) and math.isfinite(e["loss"])
                 for e in record["train"]["log"])
    record["status"] = "ok" if finite else "void_nonfinite"


# =====================================================================
# CLI
# =====================================================================
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    lab = sub.add_parser("labels")
    lab.add_argument("--data", required=True)
    lab.add_argument("--split", required=True)
    lab.add_argument("--confirm-split", required=True)
    lab.add_argument("--base", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--base", required=True)
    par = sub.add_parser("parity")
    par.add_argument("--base", required=True)
    par.add_argument("--data", required=True)
    par.add_argument("--split", required=True)
    par.add_argument("--rows-per-batch", type=int, default=4)
    par.add_argument("--steps", type=int, default=4)
    par.add_argument("--quant", choices=("nf4", "none"), default="nf4")
    par.add_argument("--device", default="cuda")
    noi = sub.add_parser("noise")
    noi.add_argument("--base", required=True)
    noi.add_argument("--data", required=True)
    noi.add_argument("--split", required=True)
    noi.add_argument("--rows-per-batch", type=int, default=4)
    noi.add_argument("--steps", type=int, default=4)
    noi.add_argument("--quant", choices=("nf4", "none"), default="nf4")
    noi.add_argument("--device", default="cuda")
    tra = sub.add_parser("train")
    tra.add_argument("--arm", choices=sorted(ARM_TARGETS), required=True)
    tra.add_argument("--seed", type=int, required=True)
    tra.add_argument("--lr", type=float, required=True)
    tra.add_argument("--base", required=True)
    tra.add_argument("--data", required=True)
    tra.add_argument("--split", required=True)
    tra.add_argument("--run-dir", required=True)
    tra.add_argument("--limit-rows", type=int, default=None)
    tra.add_argument("--device", default="cuda")
    tra.add_argument("--quantization", choices=("4bit", "none"), default="4bit")
    sco = sub.add_parser("score")
    sco.add_argument("--arm", choices=sorted(ARM_TARGETS), required=True)
    sco.add_argument("--seed", type=int, required=True)
    sco.add_argument("--lr", type=float, required=True)
    sco.add_argument("--train-record", required=True)
    sco.add_argument("--adapter-dir", required=True)
    sco.add_argument("--data", required=True)
    sco.add_argument("--split", required=True)
    sco.add_argument("--confirm-split", required=True)
    sco.add_argument("--base-path", default=None)
    sco.add_argument("--limit-rows", type=int, default=None)
    sco.add_argument("--device", default="cuda")
    for cmd in (lab, pre, par, noi, tra, sco):
        cmd.add_argument("--out", required=True)
    return parser.parse_args(argv)


HANDLERS = {"labels": labels, "preflight": preflight, "parity": parity, "noise": noise,
            "train": train, "score": score}


def main(argv: Optional[Sequence[str]] = None) -> int:
    from rich.console import Console

    args = parse_args(argv)
    started = time.time()
    record: Dict[str, Any] = {"command": args.command, "status": "started"}
    try:
        HANDLERS[args.command](args, record)
    except Exception as exc:  # noqa: BLE001 - every outcome is written as a record
        record.update(status="error", **error_fields(exc))
    if getattr(args, "limit_rows", None) is not None:
        record.update(underlying_status=record["status"], status="smoke")
    record["seconds"] = time.time() - started
    confirm.write_json(args.out, record)
    detail = f" ({record['error']['type']}: {record['error']['message'][:200]})" \
        if "error" in record else ""
    Console().print(f"{args.command}: wrote {args.out} status={record['status']}{detail}",
                    markup=False)
    return 0 if record["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
