"""G4 ticket 04: one LoRA run over a frozen Mistral-7B base in NVFP4, NF4 or bf16.

Spec: .scratch/g4-nvfp4/issues/04-mistral-harness-and-gate-rule.md. Rule (binding, in
Russian): benchmarks/gate-g4-nvfp4.md. Temporary harness, not a product path (ADR 0003).

One process trains one arm on one seed and writes ``run.json``:

    python benchmarks/harness/g4_train.py --arm nvfp4 --seed 42 --data-dir DATA \\
        --device cuda --out-dir RUNS/nvfp4-s42-a1
    python benchmarks/harness/g4_train.py --arm nf4 --seed 42 --fixture mistral \\
        --data-dir DATA --train-limit 64 --panel-limit 16 --device cuda --out-dir RUNS/x

``DATA`` is the output of ``c5_data.py`` (the rule reuses the C5 splits). The fixture is a
tiny random-init Mistral with a byte-level tokenizer; its record is labelled SYNTHETIC, only
shows that the harness starts, and never enters the verdict.

What this file adds to ``c5_train.py`` (data, training loop, panel loss are reused from it):

- The base is Mistral's decoder linears (``q/k/v/o/gate/up/down`` of every layer, 224 tensors
  for Mistral-7B) replaced one by one by NVFP4 ``te.Linear`` or bitsandbytes NF4 layers. The
  embedding, norms and ``lm_head`` stay bf16 in every arm.
- LoRA comes from ``g4_lora_nvfp4.LoRALinear`` in every arm (peft does not wrap ``te.Linear``),
  so the arms share the adapter code and its initialisation for a given seed.
- Every micro batch is right-padded to a multiple of 16 tokens and the gate suites decode
  greedily without a KV cache, because the native FP4 GEMM refuses a token count that is not a
  multiple of 16. A shape the GEMM cannot run is refused, never moved to another format.
- A real run refuses to start unless the rule file is committed and unchanged.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
import traceback
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest import mock

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c5_train as c5  # noqa: E402
import g4_lora_nvfp4  # noqa: E402
import l2l_box  # noqa: E402
from g4_fp4_layer import FP4_MULTIPLES, UnsupportedShapeError, check_fp4_shape  # noqa: E402,F401

if TYPE_CHECKING:
    import torch

console = Console()

#: C5's collate, bound before ``padded_batches`` can replace it.
_c5_collate = c5.collate

REPO = Path(__file__).resolve().parents[2]
RULE = REPO / "benchmarks" / "gate-g4-nvfp4.md"
MODEL = "mistralai/Mistral-7B-v0.1"
REVISION = "27d67f1b5f57dc0953326b2601d68371d40ea8da"
#: The arm's base storage format. The only thing the three arm configs differ in.
ARM_FORMATS: dict[str, str] = {"nvfp4": "nvfp4", "nf4": "nf4", "bf16": "bf16"}
#: Type name of a quantized layer's weight, per arm (a validity row of the rule).
EXPECTED_STORAGE = {"nvfp4": "NVFP4Tensor", "nf4": "Params4bit", "bf16": "Parameter"}
FORMAT_DETAIL = {
    "nvfp4": "TransformerEngine NVFP4BlockScaling, frozen packed weight, native FP4 GEMM",
    "nf4": "bitsandbytes nf4, double quant on, bf16 compute",
    "bf16": "no quantization",
}
PROJECTIONS = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)
LORA_DROPOUT = 0.0
PAD_MULTIPLE = 16
FIXTURE_DIMS = dict(hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048)


# ---------------------------------------------------------------- shapes and storage


def _decoder_layers(model: torch.nn.Module) -> Any:
    return model.model.layers


def _projection(layer: torch.nn.Module, path: str) -> torch.nn.Module:
    module = layer
    for part in path.split("."):
        module = getattr(module, part)
    return module


def quantized_linear_shapes(model: torch.nn.Module) -> dict[str, tuple[int, int]]:
    """``name -> (in_features, out_features)`` of every linear the arms store in 4 bits."""
    return {
        f"model.layers.{i}.{path}": (
            _projection(layer, path).in_features, _projection(layer, path).out_features)
        for i, layer in enumerate(_decoder_layers(model)) for path in PROJECTIONS
    }


def check_model_shapes(model: torch.nn.Module, tokens: int) -> None:
    """Refuse a model with a layer the FP4 GEMM cannot run, naming the layer. No fallback."""
    for name, (in_features, out_features) in quantized_linear_shapes(model).items():
        try:
            check_fp4_shape(tokens=tokens, in_features=in_features, out_features=out_features)
        except UnsupportedShapeError as error:
            raise UnsupportedShapeError(f"{name}: {error}") from error


def replace_linears(
    model: torch.nn.Module, factory: Callable[[torch.Tensor], torch.nn.Module],
    tokens: int = PAD_MULTIPLE,
) -> list[str]:
    """Replace each quantized linear by ``factory(weight)``; refuse the shapes first.

    The check runs before the first replacement, so a refused model is left untouched.
    """
    check_model_shapes(model, tokens)
    names = []
    for i, layer in enumerate(_decoder_layers(model)):
        for path in PROJECTIONS:
            parent_path, _, leaf = path.rpartition(".")
            parent = _projection(layer, parent_path)
            old = getattr(parent, leaf)
            setattr(parent, leaf, factory(old.weight.data))
            names.append(f"model.layers.{i}.{path}")
    return names


def storage_counts(model: torch.nn.Module) -> dict[str, int]:
    """Quantized layers counted by the type of their weight (looks through adapters)."""
    counts: Counter[str] = Counter()
    for layer in _decoder_layers(model):
        for path in PROJECTIONS:
            module = _projection(layer, path)
            module = getattr(module, "base", module)
            counts[type(module.weight).__name__] += 1
    return dict(counts)


def _sha256_tensor(tensor: torch.Tensor) -> str:
    import torch

    flat = tensor.detach().cpu().contiguous().reshape(-1)
    return hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest()


def unquantized_fingerprints(model: torch.nn.Module) -> dict[str, str]:
    """SHA-256 of every weight the arms keep in bf16 (embedding, norms, ``lm_head``)."""
    quantized = {f"{name}.weight" for name in quantized_linear_shapes(model)}
    return {
        name: _sha256_tensor(param) for name, param in model.named_parameters()
        if name not in quantized
    }


# ---------------------------------------------------------------- padding and decoding


def padded_width(length: int) -> int:
    return -(-length // PAD_MULTIPLE) * PAD_MULTIPLE


def padded_collate(
    examples: list[dict[str, Any]], key: str, pad_id: int, device: str
) -> dict[str, torch.Tensor]:
    """``c5_train.collate`` with the width rounded up to a multiple of 16.

    The extra columns are masked from attention and from every target.
    """
    from torch.nn import functional

    batch = _c5_collate(examples, key, pad_id, device)
    extra = padded_width(batch["input_ids"].shape[1]) - batch["input_ids"].shape[1]
    if extra:
        batch["input_ids"] = functional.pad(batch["input_ids"], (0, extra), value=pad_id)
        batch["attention_mask"] = functional.pad(batch["attention_mask"], (0, extra), value=0)
        batch["targets"] = functional.pad(batch["targets"], (0, extra), value=c5.IGNORE_INDEX)
    return batch


@contextlib.contextmanager
def padded_batches(autocast: Callable[[], Any] | None = None) -> Iterator[None]:
    """Make ``c5_train``'s loop and panel loss use padded batches, and the forward the
    ``autocast`` context (TransformerEngine's FP4 autocast). The backward stays outside it."""
    original_nll = c5.token_nll

    def nll(model: Any, batch: Any) -> Any:
        with (autocast or contextlib.nullcontext)():
            return original_nll(model, batch)

    with mock.patch.object(c5, "collate", lambda *a, **k: padded_collate(*a, **k)), \
            mock.patch.object(c5, "token_nll", nll):
        yield


def greedy_ids(
    model: torch.nn.Module, input_ids: torch.Tensor, max_new_tokens: int,
    stop_id: int | None, pad_id: int, autocast: Callable[[], Any] | None = None,
) -> list[int]:
    """Greedy decode with no KV cache. Each step runs the whole sequence, padded on the right
    to a multiple of 16, and reads the logits at the last real position. The stop token ends
    decoding and is not returned."""
    import torch
    from torch.nn import functional

    new: list[int] = []
    ids = input_ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            length = ids.shape[1]
            extra = padded_width(length) - length
            padded = functional.pad(ids, (0, extra), value=pad_id)
            mask = functional.pad(torch.ones_like(ids), (0, extra), value=0)
            with (autocast or contextlib.nullcontext)():
                logits = model(input_ids=padded, attention_mask=mask, use_cache=False).logits
            token = int(logits[0, length - 1].argmax())
            if token == stop_id:
                break
            new.append(token)
            ids = torch.cat([ids, torch.tensor([[token]], device=ids.device)], dim=1)
    return new


def make_generator(
    model: torch.nn.Module, tokenizer: Any, device: str, max_new_tokens: int,
    stop_id: int | None, autocast: Callable[[], Any] | None = None,
) -> Callable[[str], str]:
    """Soup's prompt rendering and tokenisation (as ``make_generator`` in live_eval), with the
    padded no-cache greedy loop above in place of ``model.generate``."""
    from soup_cli.utils.live_eval import _MAX_PROMPT_TOKENS, _render_prompt_template
    from soup_cli.utils.vllm import encode_rendered_prompt

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def generate(prompt: str) -> str:
        text, templated = _render_prompt_template(tokenizer, prompt)
        inputs = encode_rendered_prompt(
            tokenizer, text, templated=templated, return_tensors="pt", truncation=True,
            max_length=_MAX_PROMPT_TOKENS,
        ).to(device)
        new = greedy_ids(model, inputs["input_ids"], max_new_tokens, stop_id, pad_id, autocast)
        return tokenizer.decode(new, skip_special_tokens=True)

    return generate


def score_gate_suites(generate: Callable[[str], str]) -> dict[str, Any]:
    """Soup's four mini suites through ``generate``, every output kept."""
    from soup_cli.eval.forgetting import MINI_BENCHMARKS, build_mcq_prompt, score_answer
    from soup_cli.eval.gate_suites import score_bundled_suite

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


# ---------------------------------------------------------------- adapters


def add_lora(model: torch.nn.Module, config: dict[str, Any], seed: int) -> torch.nn.Module:
    """Freeze everything, then wrap the attention projections in ``LoRALinear``.

    The seed is set just before the wrapping, so the adapter initialisation depends on the seed
    only, whatever the base storage format is. Adapters take the dtype of the embedding (bf16).
    """
    if config["training"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    for param in model.parameters():
        param.requires_grad_(False)
    lora = config["lora"]
    embedding = model.get_input_embeddings().weight
    c5._seed_everything(seed)
    for layer in _decoder_layers(model):
        for path in PROJECTIONS:
            parent_path, _, leaf = path.rpartition(".")
            if leaf not in lora["target_modules"]:
                continue
            parent = _projection(layer, parent_path)
            setattr(parent, leaf, g4_lora_nvfp4.LoRALinear(
                getattr(parent, leaf), rank=lora["r"], alpha=lora["alpha"],
                dtype=embedding.dtype, device=embedding.device))
    return model


# ---------------------------------------------------------------- config and rule


def flatten(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flat.update(flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def run_config(args: argparse.Namespace) -> dict[str, Any]:
    """The full config of a run: C5's config (Soup defaults read at run time) with the C5
    codec replaced by the base format and the rule's changes (dropout 0, padding to 16)."""
    config = c5.run_config(argparse.Namespace(**{**vars(args), "arm": "bf16"}))
    del config["codec"]
    config["base"] = {"format": ARM_FORMATS[args.arm]}
    config["model"] = {"id": "fixture:mistral" if args.fixture else MODEL,
                       "revision": None if args.fixture else REVISION}
    config["lora"]["dropout"] = LORA_DROPOUT
    config["training"]["pad_to_multiple"] = PAD_MULTIPLE
    config["eval"]["gate_decode"] = (
        "greedy, no KV cache, whole sequence padded to a multiple of 16 each step")
    return config


def rule_state(repo: Path, rule: Path) -> dict[str, Any]:
    """The commit that added the rule file, its time, and whether it is unchanged."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=repo
                              ).stdout.strip()

    added = git("log", "--diff-filter=A", "--format=%H %cI", "--", str(rule)).splitlines()
    commit, _, when = added[-1].partition(" ") if added else (None, "", None)
    return {
        "added_commit": commit or None,
        "added_at": when or None,
        "clean": bool(commit) and not git("status", "--porcelain", "--", str(rule)),
    }


def require_committed_rule(state: dict[str, Any]) -> None:
    if not state["added_commit"] or not state["clean"]:
        raise RuntimeError(
            "the rule file benchmarks/gate-g4-nvfp4.md must be committed and unchanged before "
            f"a real run: {state}")


# ---------------------------------------------------------------- bases (GPU)


def _nvfp4_factory(device: str) -> tuple[Callable[[torch.Tensor], torch.nn.Module],
                                         Callable[[], Any]]:
    """Frozen NVFP4 ``te.Linear`` layers holding a given bf16 weight, and their autocast."""
    import torch
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import NVFP4BlockScaling

    recipe = NVFP4BlockScaling()

    def make(weight: torch.Tensor) -> torch.nn.Module:
        out_features, in_features = weight.shape
        with te.quantized_model_init(enabled=True, recipe=recipe):
            layer = te.Linear(in_features, out_features, bias=False,
                              params_dtype=torch.bfloat16, device=device)
        layer.weight.copy_(weight.to(device=device, dtype=torch.bfloat16))
        layer.weight.requires_grad_(False)
        return layer

    return make, lambda: te.autocast(enabled=True, recipe=recipe)


def _nf4_factory(device: str) -> Callable[[torch.Tensor], torch.nn.Module]:
    """Frozen bitsandbytes NF4 layers (double quant on, bf16 compute) holding a bf16 weight."""
    import bitsandbytes as bnb
    import torch

    def make(weight: torch.Tensor) -> torch.nn.Module:
        out_features, in_features = weight.shape
        layer = bnb.nn.Linear4bit(
            in_features, out_features, bias=False, compute_dtype=torch.bfloat16,
            compress_statistics=True, quant_type="nf4", device="meta")
        layer.weight = bnb.nn.Params4bit(
            weight.detach().cpu().clone(), requires_grad=False, compress_statistics=True,
            quant_type="nf4")
        return layer.to(device)

    return make


def prepare_base(model: torch.nn.Module, arm: str, device: str, tokens: int
                 ) -> tuple[torch.nn.Module, Callable[[], Any] | None]:
    """Move the model to ``device`` with the arm's storage; returns it and the forward's
    autocast (None except for NVFP4)."""
    autocast = None
    if arm == "nvfp4":
        factory, autocast = _nvfp4_factory(device)
        replace_linears(model, factory, tokens)
    elif arm == "nf4":
        replace_linears(model, _nf4_factory(device), tokens)
    return model.to(device), autocast


# ---------------------------------------------------------------- run


def load_model(args: argparse.Namespace, tokenizer: Any) -> tuple[torch.nn.Module, dict]:
    """The bf16 model on the CPU, so quantizing layer by layer never holds two copies on the GPU."""
    import torch

    if args.fixture:
        import transformers

        torch.manual_seed(0)
        model = transformers.MistralForCausalLM(transformers.MistralConfig(
            vocab_size=len(tokenizer), tie_word_embeddings=False, **FIXTURE_DIMS,
        )).to(torch.bfloat16)
        info = {"id": "fixture:mistral", "revision": None, "synthetic": True,
                "note": ("SYNTHETIC: a tiny random-init Mistral with a byte-level tokenizer. "
                         "Its losses and scores say nothing about a real model and never "
                         "enter the verdict.")}
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, dtype=torch.bfloat16, low_cpu_mem_usage=True)
        info = {"id": MODEL, "revision": getattr(model.config, "_commit_hash", None) or REVISION,
                "revision_requested": REVISION, "synthetic": False, "note": ""}
    info["model_type"] = model.config.model_type
    return model, info


def load_tokenizer(args: argparse.Namespace) -> Any:
    from soup_cli.data.chat_templates import apply_chat_template_override

    if args.fixture:
        tokenizer = c5.byte_tokenizer()
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    apply_chat_template_override(tokenizer, c5.CHAT_TEMPLATE)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    return tokenizer


def _versions() -> dict[str, str | None]:
    versions = c5._versions()
    try:
        versions["transformer_engine"] = importlib.metadata.version("transformer_engine")
    except importlib.metadata.PackageNotFoundError:
        versions["transformer_engine"] = None
    return versions


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arm", required=True, choices=tuple(ARM_FORMATS))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--fixture", choices=("mistral",),
                        help="tiny random-init model, labelled synthetic")
    parser.add_argument("--data-dir", required=True, help="output of c5_data.py")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", required=True, help="writes run.json here")
    parser.add_argument("--attempt", type=int, default=1,
                        help="2 for the one repeat of a spoiled run")
    parser.add_argument("--micro-batch", type=int, default=4,
                        help=f"must divide {c5.EFFECTIVE_BATCH}; the same in every arm")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--gate-max-new-tokens", type=int, default=64)
    parser.add_argument("--train-limit", type=int, help="first N train rows (fixture only)")
    parser.add_argument("--panel-limit", type=int, help="first N panel rows (fixture only)")
    args = parser.parse_args(argv)
    args.model, args.revision = (None, None) if args.fixture else (MODEL, REVISION)
    if not args.fixture and (args.train_limit or args.panel_limit):
        parser.error("--train-limit and --panel-limit are for the fixture only")
    if c5.EFFECTIVE_BATCH % args.micro_batch:
        parser.error(f"--micro-batch must divide {c5.EFFECTIVE_BATCH}")
    return args


def main(argv: list[str] | None = None) -> int:
    import torch

    args = _parse(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = run_config(args)
    rule = rule_state(REPO, RULE)
    record: dict[str, Any] = {
        "synthetic": bool(args.fixture),
        "arm": args.arm, "seed": args.seed, "attempt": args.attempt,
        "status": "running",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "started_unix": time.time(),
        "command": [sys.executable, *sys.argv] if argv is None else ["g4_train.py", *argv],
        "commit": c5._commit(),
        "rule": rule,
        "versions": _versions(),
        "device": args.device,
        "gpu": c5.gpu_info(args.device),
        "harness": {"sha256": c5._sha256_file(Path(__file__))},
        "base_format": {"name": ARM_FORMATS[args.arm], "detail": FORMAT_DETAIL[args.arm]},
        "config": config,
        "box_stamp_before": l2l_box.box_stamp(),
        "train_log": [],
        "phases_s": {},
    }
    c5._write(out_dir, record)
    try:
        if not args.fixture:
            require_committed_rule(rule)
        record["status"] = _run(args, config, record, out_dir)
    except Exception:  # noqa: BLE001 - a crashed run keeps its record (rule: kept, marked)
        record["status"] = "failed"
        record["error"] = traceback.format_exc()
        console.print(f"[red]run failed[/]: {record['error'].splitlines()[-1]}")
    record["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    record["box_stamp_after"] = l2l_box.box_stamp()
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        record["max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
    c5._write(out_dir, record)
    label = "SYNTHETIC " if record["synthetic"] else ""
    console.print(f"{label}{args.arm} seed {args.seed}: {record['status']} -> {out_dir}")
    return 0 if record["status"] == "complete" else 1


def _run(
    args: argparse.Namespace, config: dict[str, Any], record: dict[str, Any], out_dir: Path
) -> str:
    data_dir = Path(args.data_dir)
    panel_rows, panel_sha = c5.load_split(data_dir / "heldout.jsonl", args.panel_limit)
    train_rows, train_sha = c5.load_split(data_dir / "train.jsonl", args.train_limit)
    if not args.fixture and (
            panel_sha != c5.PANEL_SHA256 or train_sha not in c5.TRAIN_SHA256):
        raise ValueError(f"splits do not match the rule: panel {panel_sha}, train {train_sha}")
    record["data"] = {"heldout_sha256": panel_sha, "train_sha256": train_sha,
                      "panel_rows": len(panel_rows), "train_rows": len(train_rows)}

    start = time.perf_counter()
    tokenizer = load_tokenizer(args)
    record["pad_token_id"] = tokenizer.pad_token_id
    panel = [c5.encode_example(row, tokenizer, c5.MAX_LENGTH) for row in panel_rows]
    examples = [c5.encode_example(row, tokenizer, c5.MAX_LENGTH) for row in train_rows]
    record["data"].update({
        "train_truncated": sum(1 for e in examples if len(e["input_ids"]) == c5.MAX_LENGTH),
        "train_without_labels": sum(1 for e in examples if c5._label_tokens(e) == 0),
        "train_label_tokens": sum(c5._label_tokens(e) for e in examples),
    })
    order = c5.batch_order(len(examples), args.seed, args.micro_batch,
                           c5.EFFECTIVE_BATCH // args.micro_batch)
    record["batch_order_sha256"] = hashlib.sha256(json.dumps(order).encode()).hexdigest()
    record["total_steps"] = len(order)
    c5._phase(record, "data", start)

    start = time.perf_counter()
    model, record["model"] = load_model(args, tokenizer)
    if record["model"]["synthetic"]:
        console.print(f"  {record['model']['note']}")
    record["unquantized_fingerprints"] = unquantized_fingerprints(model)
    c5._phase(record, "load", start)

    start = time.perf_counter()
    model, autocast = prepare_base(model, args.arm, args.device,
                                   tokens=args.micro_batch * c5.MAX_LENGTH)
    record["storage_counts"] = storage_counts(model)
    c5._phase(record, "base", start)
    expected = {EXPECTED_STORAGE[args.arm]: len(quantized_linear_shapes(model))}
    if record["storage_counts"] != expected:
        raise RuntimeError(f"layers are not all in the arm's format: "
                           f"{record['storage_counts']}, expected {expected}")
    c5._write(out_dir, record)

    model = add_lora(model, config, args.seed)
    record["trainable_parameters"] = [n for n, p in model.named_parameters() if p.requires_grad]
    pad_id = tokenizer.pad_token_id
    start = time.perf_counter()
    with padded_batches(autocast):
        status = c5.train(model, examples, order, config, args.device, pad_id,
                          record["train_log"])
    c5._phase(record, "train", start)
    c5._write(out_dir, record)
    if status != "complete":
        return status

    start = time.perf_counter()
    with padded_batches(autocast):
        record["panel"] = c5.evaluate_panel(model, panel, args.micro_batch, args.device, pad_id)
    c5._phase(record, "panel", start)
    console.print(f"  task loss {record['panel']['task_loss']:.4f} "
                  f"over {record['panel']['tokens']} tokens")
    c5._write(out_dir, record)

    start = time.perf_counter()
    model.eval()
    generate = make_generator(model, tokenizer, args.device, args.gate_max_new_tokens,
                              tokenizer.eos_token_id, autocast)
    record["gate_suites"] = score_gate_suites(generate)
    c5._phase(record, "gate", start)
    console.print("  gate: " + ", ".join(
        f"{name} {s['correct']}/{s['n']}" for name, s in record["gate_suites"].items()))
    return "complete"


if __name__ == "__main__":
    sys.exit(main())
