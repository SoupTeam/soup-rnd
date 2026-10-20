"""C5 ticket 04: one LoRA run over a MoE whose experts are NF4, Q2_0 or bf16.

Spec: .scratch/c5-q2-0-experts/issues/04-lora-nf4-vs-q2-0.md. Rule (binding,
in Russian): benchmarks/gate-c5-q2-0-experts.md. Experts are fake-quant
(docs/adr/0001-c5-fake-quant-training.md) through ``c5_expert_swap.py``.

One process trains one arm on one seed and writes ``run.json``:

    python benchmarks/harness/c5_train.py --arm q2_0 --seed 42 \\
        --model allenai/OLMoE-1B-7B-0924 --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 \\
        --data-dir DATA --device cuda --out-dir RUNS/q2_0-s42-a1
    python benchmarks/harness/c5_train.py --arm nf4 --seed 42 --fixture granite \\
        --data-dir DATA --train-limit 64 --panel-limit 16 --out-dir RUNS/nf4-s42-a1

``DATA`` is the output of ``c5_data.py``. The fixture is a tiny random-init
granitemoe with a byte-level tokenizer, and its record is labelled SYNTHETIC.

Choices the rule leaves open, fixed before the first arm:

- OLMoE-0924 ships no chat template, so Soup's response-only path would fall
  back to training on every token. The data goes through Soup itself: the Dolly
  row becomes an alpaca row (``context`` is the input), ``format_to_messages``
  makes the messages and ``build_format_row`` with ``chat_template: chatml``
  makes the input ids and labels. Training loss is on Soup's assistant mask.
- Task loss counts the answer tokens only: the tokens whose first character is
  inside the answer text. The ``<|im_start|>assistant`` header and the
  ``<|im_end|>`` marker are fixed text with near-zero loss after training, and
  OLMoE-0924 splits each into 7 to 9 plain tokens (the median panel answer is
  44 tokens), so counting them would pull every ratio towards 1. A token that
  merges the answer's last character with the marker (``.<``) counts.
- The training loop is written here, with Soup's defaults (AdamW, warmup 0.03,
  weight decay 0.01, grad clip 1.0, cosine). Each step's loss is the summed NLL
  of its 16 examples over their label tokens, as transformers' Trainer does
  with gradient accumulation. The batch order comes only from the seed.
- Gate suites are scored by Soup's ``score_bundled_suite`` with Soup's
  ``make_generator`` (greedy, 64 new tokens) on the trained model in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm_validity  # noqa: E402
import c5_expert_swap as swap  # noqa: E402
import l2l_box  # noqa: E402
import q2_0_codec as codec  # noqa: E402

if TYPE_CHECKING:
    import torch

console = Console()

# Split hashes from benchmarks/gate-c5-q2-0-experts.md; a real run refuses others.
PANEL_SHA256 = "b4b2ab938ca72a52c660d810045e3f31f842e3107e00738f827abb3f932d2a37"
TRAIN_SHA256 = {
    "a915bbbf18ebd1fae1dd869bdc8e2a0efbd2ab522d3a0c733cc05b212befa0b7": 4000,
    "806910717344ff5c4545dde76e5ae50367fb9d69365d59a11e23cf32463d7f7c": 2000,
}
ARM_CODECS: dict[str, str | None] = {"nf4": "nf4", "q2_0": "q2_0", "bf16": None}
IGNORE_INDEX = -100
CHAT_TEMPLATE = "chatml"
MAX_LENGTH = 1024
EFFECTIVE_BATCH = 16
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


def batch_order(
    n: int, seed: int, micro_batch: int, accumulation: int
) -> list[list[list[int]]]:
    """Steps of micro batches of example indices, one epoch.

    The order depends only on ``n`` and ``seed``: ``random.Random(seed)``
    shuffles the indices, which are cut into steps of ``micro_batch *
    accumulation`` and each step into micro batches. A last step may be short.
    """
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    effective = micro_batch * accumulation
    return [
        [indices[start:start + micro_batch]
         for start in range(step, min(step + effective, n), micro_batch)]
        for step in range(0, n, effective)
    ]


def byte_tokenizer() -> Any:
    """A byte-level BPE tokenizer with no merges, built in memory. For the fixture."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {char: i for i, char in enumerate(alphabet)}
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, eos_token="<|endoftext|>", pad_token="<|padding|>"
    )


def _messages(row: dict[str, Any]) -> list[dict[str, str]]:
    from soup_cli.data.formats import format_to_messages

    alpaca = {"instruction": row["instruction"], "input": row["context"],
              "output": row["response"]}
    return format_to_messages(alpaca, "alpaca")["messages"]


def encode_example(row: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, Any]:
    """Input ids and training labels by Soup, and the task-loss mask.

    ``labels`` is Soup's response-only mask with ``chat_template: chatml``.
    ``response_mask`` keeps the labelled tokens whose first character lies in
    the answer text of the rendered conversation, by the tokenizer's offsets.
    """
    from soup_cli.config.schema import DataConfig
    from soup_cli.data.sft_format import build_format_row

    data_config = DataConfig(train="dolly", chat_template=CHAT_TEMPLATE, max_length=max_length)
    messages = _messages(row)
    encoded = build_format_row(tokenizer, data_config)({"messages": messages})
    input_ids, labels = list(encoded["input_ids"]), list(encoded["labels"])
    prompt = tokenizer.apply_chat_template(
        messages[:1], tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    shared = min(len(prompt_ids), len(input_ids))
    if input_ids[:shared] != prompt_ids[:shared]:
        raise ValueError(f"row {row['row']}: the prompt does not tokenize as a prefix")
    answer = messages[1]["content"]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False)
    if rendered[len(prompt):len(prompt) + len(answer)] != answer:
        raise ValueError(f"row {row['row']}: the answer does not follow the prompt")
    tokens = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if tokens["input_ids"][:len(input_ids)] != input_ids:
        raise ValueError(f"row {row['row']}: Soup's ids differ from the rendered text's")
    answer_span = range(len(prompt), len(prompt) + len(answer))
    response_mask = [
        label != IGNORE_INDEX and start in answer_span
        for label, (start, _end) in zip(labels, tokens["offset_mapping"])
    ]
    return {"row": row["row"], "input_ids": input_ids, "labels": labels,
            "response_mask": response_mask}


def run_config(args: argparse.Namespace) -> dict[str, Any]:
    """The full config of a run. Rule 03 values are written here; the rest are
    Soup's schema defaults, read at run time."""
    from soup_cli.config.schema import LoraConfig, TrainingConfig
    from soup_cli.eval.forgetting import MINI_BENCHMARKS

    def soup(model: Any, field: str) -> Any:
        return model.model_fields[field].default

    return {
        "codec": ARM_CODECS[args.arm],
        "seed": args.seed,
        "model": {"id": "fixture:granite" if args.fixture else args.model,
                  "revision": args.revision},
        "data": {
            "source": "Dolly row as alpaca: instruction, context as input, response as output",
            "chat_template": CHAT_TEMPLATE,
            "max_length": MAX_LENGTH,
            "train_on_responses_only": True,
            "task_loss_tokens": "tokens starting inside the answer text; no header, no marker",
            "train_limit": args.train_limit,
            "panel_limit": args.panel_limit,
        },
        "lora": {
            "r": 16, "alpha": 32, "target_modules": list(LORA_TARGETS),
            "dropout": soup(LoraConfig, "dropout"), "bias": "none",
        },
        "training": {
            "epochs": 1,
            "effective_batch": EFFECTIVE_BATCH,
            "micro_batch": args.micro_batch,
            "gradient_accumulation": EFFECTIVE_BATCH // args.micro_batch,
            "lr": 2e-4,
            "scheduler": soup(TrainingConfig, "scheduler"),
            "warmup_ratio": soup(TrainingConfig, "warmup_ratio"),
            "weight_decay": soup(TrainingConfig, "weight_decay"),
            "max_grad_norm": soup(TrainingConfig, "max_grad_norm"),
            "optimizer": soup(TrainingConfig, "optimizer"),
            "adam_betas": [0.9, 0.999],
            "adam_eps": 1e-8,
            "gradient_checkpointing": args.gradient_checkpointing,
            "model_dtype": "bfloat16",
            "batch_order": "random.Random(seed).shuffle, then steps, then micro batches",
            "loss": "summed NLL of a step's label tokens over their count",
        },
        "eval": {
            "panel_micro_batch": args.micro_batch,
            "gate_suites": list(MINI_BENCHMARKS),
            "gate_max_new_tokens": args.gate_max_new_tokens,
            "gate_decode": "greedy, soup_cli.utils.live_eval.make_generator",
        },
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_split(path: Path, limit: int | None) -> tuple[list[dict[str, Any]], str]:
    """Rows of a ``c5_data.py`` split, and the SHA-256 of the whole file."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return rows[:limit] if limit else rows, _sha256_file(path)


def load_tokenizer(args: argparse.Namespace) -> Any:
    from soup_cli.data.chat_templates import apply_chat_template_override

    if args.fixture:
        tokenizer = byte_tokenizer()
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    apply_chat_template_override(tokenizer, CHAT_TEMPLATE)
    return tokenizer


def load_model(args: argparse.Namespace, tokenizer: Any) -> tuple[torch.nn.Module, dict]:
    import torch

    if args.fixture:
        model = swap.granite_fixture(seed=0, vocab_size=len(tokenizer))
        info = {
            "id": "fixture:granite", "revision": None, "synthetic": True,
            "note": (
                "SYNTHETIC: a tiny random-init granitemoe with a byte-level tokenizer. "
                "Its losses and scores say nothing about a real model and never enter "
                "the verdict."
            ),
        }
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        )
        info = {
            "id": args.model,
            "revision": getattr(model.config, "_commit_hash", None) or args.revision,
            "revision_requested": args.revision, "synthetic": False, "note": "",
        }
    info["model_type"] = model.config.model_type
    info["output_router_logits"] = getattr(model.config, "output_router_logits", None)
    return model.to(args.device), info


def prepare_experts(model: torch.nn.Module, arm: str) -> dict[str, Any]:
    """Fake-quant the experts (none for bf16); fingerprints and errors for the record."""
    codec_name = ARM_CODECS[arm]
    experts = swap.expert_tensor_names(model)
    if codec_name is None:
        after = swap.fingerprints(model)
        record = {
            "codec": None, "experts": experts, "tensors": {},
            "non_expert_fingerprints": {n: d for n, d in after.items() if n not in experts},
            "checks": {"expert_rules_agree": bool(experts) and sorted(experts) == sorted(
                swap.expert_module_tensor_names(model))},
        }
    else:
        record = swap.run_arm(model, codec_name)
    params = dict(model.named_parameters())
    expert_fingerprints = {name: swap._sha256(params[name]) for name in experts}
    return {
        "checks": record["checks"],
        "experts": experts,
        "rel_mse": {
            name: {"error": entry["error"]["rel_mse"],
                   "error_in_model": entry["error_in_model"]["rel_mse"]}
            for name, entry in record["tensors"].items()
        },
        "non_expert_fingerprints": record["non_expert_fingerprints"],
        "expert_digest": hashlib.sha256(
            json.dumps(expert_fingerprints, sort_keys=True).encode()
        ).hexdigest(),
    }


def add_lora(model: torch.nn.Module, config: dict[str, Any], seed: int) -> torch.nn.Module:
    import torch
    from peft import LoraConfig, get_peft_model

    if config["training"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lora = config["lora"]
    _seed_everything(seed)
    return get_peft_model(model, LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"], bias=lora["bias"], task_type="CAUSAL_LM",
    )).to(torch.device(next(model.parameters()).device))


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate(
    examples: list[dict[str, Any]], key: str, pad_id: int, device: str
) -> dict[str, torch.Tensor]:
    """Right-padded tensors. ``key`` picks the targets: Soup's ``labels`` for
    training or the ``response_mask`` for the task loss."""
    import torch

    width = max(len(e["input_ids"]) for e in examples)
    ids, attention, targets = [], [], []
    for e in examples:
        pad = width - len(e["input_ids"])
        if key == "labels":
            target = e["labels"]
        else:
            target = [t if keep else IGNORE_INDEX for t, keep in zip(e["input_ids"], e[key])]
        ids.append(e["input_ids"] + [pad_id] * pad)
        attention.append([1] * len(e["input_ids"]) + [0] * pad)
        targets.append(list(target) + [IGNORE_INDEX] * pad)
    return {name: torch.tensor(value, device=device) for name, value in
            (("input_ids", ids), ("attention_mask", attention), ("targets", targets))}


def token_nll(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> tuple[Any, Any]:
    """Summed NLL and target-token count per example (a target at position 0 has
    no prediction and does not count)."""
    from torch.nn import functional

    logits = model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False
    ).logits[:, :-1].float()
    targets = batch["targets"][:, 1:]
    nll = functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
        ignore_index=IGNORE_INDEX, reduction="none",
    ).view(targets.shape)
    return nll.sum(dim=1), (targets != IGNORE_INDEX).sum(dim=1)


def _label_tokens(example: dict[str, Any]) -> int:
    return sum(1 for label in example["labels"][1:] if label != IGNORE_INDEX)


def train(
    model: torch.nn.Module, examples: list[dict[str, Any]], order: list[list[list[int]]],
    config: dict[str, Any], device: str, pad_id: int, log: list[dict[str, Any]],
) -> str:
    """One epoch in ``order``; appends one entry per optimizer step to ``log``.

    Returns ``complete``, or ``non_finite`` after the first step whose loss or
    grad norm is NaN or Inf (training stops there).
    """
    import torch
    from transformers import get_cosine_schedule_with_warmup

    training = config["training"]
    if (training["optimizer"], training["scheduler"]) != ("adamw_torch", "cosine"):
        raise ValueError(
            "this loop runs AdamW with a cosine schedule; Soup's defaults are now "
            f"{training['optimizer']} and {training['scheduler']}"
        )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=training["lr"], betas=tuple(training["adam_betas"]),
        eps=training["adam_eps"], weight_decay=training["weight_decay"],
    )
    warmup = math.ceil(training["warmup_ratio"] * len(order))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, len(order))
    model.train()
    start = time.perf_counter()
    for number, step in enumerate(order, 1):
        tokens = sum(_label_tokens(examples[i]) for micro in step for i in micro)
        step_nll = 0.0
        for micro in step:
            nll, _ = token_nll(model, collate([examples[i] for i in micro], "labels",
                                              pad_id, device))
            (nll.sum() / max(tokens, 1)).backward()
            step_nll += nll.sum().item()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, training["max_grad_norm"]).item()
        lr = scheduler.get_last_lr()[0]
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        loss = step_nll / max(tokens, 1)
        log.append({"step": number, "loss": loss, "grad_norm": grad_norm, "lr": lr,
                    "tokens": tokens, "elapsed_s": time.perf_counter() - start})
        if number == 1 or number % 25 == 0 or number == len(order):
            console.print(f"  step {number}/{len(order)} loss {loss:.4f} "
                          f"grad_norm {grad_norm:.4f} lr {lr:.2e}")
        if not (math.isfinite(loss) and math.isfinite(grad_norm)):
            return "non_finite"
    return "complete"


def evaluate_panel(
    model: torch.nn.Module, examples: list[dict[str, Any]], micro_batch: int, device: str,
    pad_id: int,
) -> dict[str, Any]:
    """Task loss: NLL on the response mask over the panel, weighted by tokens."""
    import torch

    model.eval()
    per_example = []
    with torch.no_grad():
        for start in range(0, len(examples), micro_batch):
            chunk = examples[start:start + micro_batch]
            nll, tokens = token_nll(model, collate(chunk, "response_mask", pad_id, device))
            per_example += [
                {"row": e["row"], "tokens": int(t), "nll": float(n)}
                for e, n, t in zip(chunk, nll.tolist(), tokens.tolist())
            ]
    total = sum(e["tokens"] for e in per_example)
    return {
        "task_loss": sum(e["nll"] for e in per_example) / total if total else float("nan"),
        "tokens": total,
        "examples_without_tokens": sum(1 for e in per_example if e["tokens"] == 0),
        "per_example": per_example,
    }


def score_gate_suites(
    model: torch.nn.Module, tokenizer: Any, device: str, max_new_tokens: int
) -> dict[str, Any]:
    """Soup's four mini suites on the model in memory, every output kept."""
    from soup_cli.eval.forgetting import MINI_BENCHMARKS, build_mcq_prompt, score_answer
    from soup_cli.eval.gate_suites import score_bundled_suite
    from soup_cli.utils.live_eval import make_generator

    model.eval()
    generate = make_generator(
        "in-memory", loaded=(model, tokenizer, device), max_new_tokens=max_new_tokens
    )
    suites = {}
    for name, items in MINI_BENCHMARKS.items():
        outputs: dict[str, str] = {}

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


def _versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in ("torch", "transformers", "peft", "bitsandbytes", "datasets", "tokenizers",
                    "soup-cli"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True,
        cwd=Path(__file__).resolve().parent,
    ).stdout.strip()


def _commit() -> dict[str, Any]:
    return {"sha": _git("rev-parse", "HEAD"),
            "dirty": [line for line in _git("status", "--porcelain").splitlines() if line]}


def gpu_info(device: str) -> dict[str, Any] | None:
    """Name, UUID, driver and memory of the GPU the run sees; None on CPU."""
    if not device.startswith("cuda"):
        return None
    import torch

    gpu = arm_validity.workload_gpu()
    fields = ("name", "uuid", "driver_version", "memory.total", "clocks.max.sm")
    try:
        line = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader",
             "-i", str(gpu["id"])], capture_output=True, text=True, check=True,
        ).stdout.strip()
        smi: dict[str, Any] = dict(zip(fields, (v.strip() for v in line.split(","))))
    except (OSError, subprocess.CalledProcessError) as error:
        smi = {"error": str(error)}
    return {"workload_gpu": gpu, **smi, "torch_name": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda}


def _write(out_dir: Path, record: dict[str, Any]) -> None:
    (out_dir / "run.json").write_text(json.dumps(record, indent=1), encoding="utf-8")


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arm", required=True, choices=tuple(ARM_CODECS))
    parser.add_argument("--seed", required=True, type=int)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="model id or local path")
    source.add_argument("--fixture", choices=("granite",),
                        help="tiny random-init model, labelled synthetic")
    parser.add_argument("--revision", help="commit to load; required with --model")
    parser.add_argument("--data-dir", required=True, help="output of c5_data.py")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", required=True, help="writes run.json here")
    parser.add_argument("--attempt", type=int, default=1,
                        help="2 for the one repeat of a spoiled run")
    parser.add_argument("--micro-batch", type=int, default=4,
                        help=f"must divide {EFFECTIVE_BATCH}; the same in every arm")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--gate-max-new-tokens", type=int, default=64)
    parser.add_argument("--train-limit", type=int, help="first N train rows (fixture only)")
    parser.add_argument("--panel-limit", type=int, help="first N panel rows (fixture only)")
    args = parser.parse_args(argv)
    if args.model and not args.revision:
        parser.error("--model needs --revision: the record must name the exact weights")
    if args.model and (args.train_limit or args.panel_limit):
        parser.error("--train-limit and --panel-limit are for the fixture only")
    if EFFECTIVE_BATCH % args.micro_batch:
        parser.error(f"--micro-batch must divide {EFFECTIVE_BATCH}")
    return args


def main(argv: list[str] | None = None) -> int:
    import torch

    args = _parse(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = run_config(args)
    record: dict[str, Any] = {
        "synthetic": bool(args.fixture),
        "arm": args.arm, "seed": args.seed, "attempt": args.attempt,
        "status": "running",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "started_unix": time.time(),
        "command": [sys.executable, *sys.argv] if argv is None else ["c5_train.py", *argv],
        "commit": _commit(),
        "versions": _versions(),
        "device": args.device,
        "gpu": gpu_info(args.device),
        "harness": {"sha256": _sha256_file(Path(__file__)),
                    "swap_sha256": swap.harness_sha256(),
                    "codec_sha256": codec.codec_sha256()},
        "config": config,
        "box_stamp_before": l2l_box.box_stamp(),
        "train_log": [],
        "phases_s": {},
    }
    _write(out_dir, record)
    try:
        record["status"] = _run(args, config, record, out_dir)
    except Exception:  # noqa: BLE001 - a crashed run keeps its record (rule: kept, marked)
        record["status"] = "failed"
        record["error"] = traceback.format_exc()
        console.print(f"[red]run failed[/]: {record['error'].splitlines()[-1]}")
    record["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    record["box_stamp_after"] = l2l_box.box_stamp()
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        record["max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
    _write(out_dir, record)
    label = "SYNTHETIC " if record["synthetic"] else ""
    console.print(f"{label}{args.arm} seed {args.seed}: {record['status']} -> {out_dir}")
    return 0 if record["status"] == "complete" else 1


def _phase(record: dict[str, Any], name: str, start: float) -> None:
    record["phases_s"][name] = time.perf_counter() - start


def _run(
    args: argparse.Namespace, config: dict[str, Any], record: dict[str, Any], out_dir: Path
) -> str:
    data_dir = Path(args.data_dir)
    panel_rows, panel_sha = load_split(data_dir / "heldout.jsonl", args.panel_limit)
    train_rows, train_sha = load_split(data_dir / "train.jsonl", args.train_limit)
    if not args.fixture and (panel_sha != PANEL_SHA256 or train_sha not in TRAIN_SHA256):
        raise ValueError(f"splits do not match the rule: panel {panel_sha}, train {train_sha}")
    record["data"] = {"heldout_sha256": panel_sha, "train_sha256": train_sha,
                      "panel_rows": len(panel_rows), "train_rows": len(train_rows)}

    start = time.perf_counter()
    tokenizer = load_tokenizer(args)
    panel = [encode_example(row, tokenizer, MAX_LENGTH) for row in panel_rows]
    examples = [encode_example(row, tokenizer, MAX_LENGTH) for row in train_rows]
    record["data"].update({
        "train_truncated": sum(1 for e in examples if len(e["input_ids"]) == MAX_LENGTH),
        "train_without_labels": sum(1 for e in examples if _label_tokens(e) == 0),
        "train_label_tokens": sum(_label_tokens(e) for e in examples),
    })
    order = batch_order(len(examples), args.seed, args.micro_batch,
                        EFFECTIVE_BATCH // args.micro_batch)
    record["batch_order_sha256"] = hashlib.sha256(json.dumps(order).encode()).hexdigest()
    record["total_steps"] = len(order)
    _phase(record, "data", start)

    start = time.perf_counter()
    model, record["model"] = load_model(args, tokenizer)
    _phase(record, "load", start)
    if record["model"]["synthetic"]:
        console.print(f"  {record['model']['note']}")

    start = time.perf_counter()
    swapped = prepare_experts(model, args.arm)
    record["non_expert_fingerprints"] = swapped.pop("non_expert_fingerprints")
    record["expert_digest"] = swapped.pop("expert_digest")
    record["swap"] = swapped
    _phase(record, "swap", start)
    if not all(swapped["checks"].values()):
        raise RuntimeError(f"expert swap checks failed: {swapped['checks']}")
    _write(out_dir, record)

    model = add_lora(model, config, args.seed)
    record["trainable_parameters"] = [
        name for name, p in model.named_parameters() if p.requires_grad
    ]
    pad_id = tokenizer.pad_token_id
    start = time.perf_counter()
    status = train(model, examples, order, config, args.device, pad_id, record["train_log"])
    _phase(record, "train", start)
    _write(out_dir, record)
    if status != "complete":
        return status

    start = time.perf_counter()
    record["panel"] = evaluate_panel(model, panel, args.micro_batch, args.device, pad_id)
    _phase(record, "panel", start)
    console.print(f"  task loss {record['panel']['task_loss']:.4f} "
                  f"over {record['panel']['tokens']} tokens")
    _write(out_dir, record)

    start = time.perf_counter()
    record["gate_suites"] = score_gate_suites(model, tokenizer, args.device,
                                              args.gate_max_new_tokens)
    _phase(record, "gate", start)
    console.print("  gate: " + ", ".join(
        f"{name} {s['correct']}/{s['n']}" for name, s in record["gate_suites"].items()))
    return "complete"


if __name__ == "__main__":
    sys.exit(main())
