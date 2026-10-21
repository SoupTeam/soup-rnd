#!/usr/bin/env python3
"""One F3 run in one fresh process: train a model for N steps with the collector attached.

The driver (``f3_rehearsal.py``) starts this script once per run, in the venv that belongs
to the framework: ``native`` runs a plain HF ``Trainer``, ``unsloth`` runs Unsloth's
``FastLanguageModel`` with TRL's ``SFTTrainer``. Both attach the same collector through
``f3_trainer_adapter`` and write a ``build_train_report`` report as JSON.

``--control`` masks every label (``labels == -100``) and sets ``weight_decay`` to 0: the
zero-gradient control. ``--tokens`` is the file ``f3_setup.py data`` writes. The trainer
does not see the dataset's own labels, so a masked batch cannot come from the data.

Weights load in float32 and ``bf16=True`` turns on autocast: the collector's parameter
fingerprint calls ``.numpy()``, which fails on bfloat16 tensors, and ``src/soup_cli/bench/``
stays unchanged. Record this as a changed condition if a published arm holds bf16 weights.

Benchmark harness code, not shipped. Heavy imports are lazy and the foreign framework is
imported first, as Unsloth asks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(HARNESS.parents[1] / "src"))


def make_collate(pad_id: int, mask_all: bool, max_length: int):
    """A collator for pre-tokenized rows: pad, and mask padding (or everything) in labels."""
    import torch

    def collate(features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        rows: List[List[int]] = [list(f["input_ids"])[:max_length] for f in features]
        width = max(len(row) for row in rows)
        ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        labels = torch.full((len(rows), width), -100, dtype=torch.long)
        for i, row in enumerate(rows):
            ids[i, : len(row)] = torch.tensor(row)
            mask[i, : len(row)] = 1
            if not mask_all:
                labels[i, : len(row)] = torch.tensor(row)
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    return collate


def _training_args(args: argparse.Namespace, out: Path, extra: Optional[dict] = None) -> dict:
    values = dict(
        output_dir=str(out / "trainer"),
        max_steps=args.steps,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        weight_decay=0.0 if args.control else args.weight_decay,
        logging_steps=1,
        report_to=[],
        save_strategy="no",
        bf16=True,
        disable_tqdm=True,
        seed=0,
        remove_unused_columns=False,
    )
    values.update(extra or {})
    return values


def run_native(args: argparse.Namespace, rows: list, out: Path) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.to("cuda")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**_training_args(args, out)),
        train_dataset=[{"input_ids": r} for r in rows],
        data_collator=make_collate(tokenizer.pad_token_id or 0, args.control, args.max_length),
    )
    return _train_and_report(trainer, args)


def run_unsloth(args: argparse.Namespace, rows: list, out: Path) -> dict:
    from unsloth import FastLanguageModel  # noqa: I001 - Unsloth must be imported first

    import torch
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_length,
        dtype=torch.float32,
        load_in_4bit=False,
        full_finetuning=True,
    )
    config = SFTConfig(
        **_training_args(args, out, {"dataset_kwargs": {"skip_prepare_dataset": True}})
    )
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=[{"input_ids": r} for r in rows],
        data_collator=make_collate(pad, args.control, args.max_length),
        processing_class=tokenizer,
    )
    return _train_and_report(trainer, args)


def _train_and_report(trainer: Any, args: argparse.Namespace) -> dict:
    import f3_trainer_adapter
    import torch

    collector = f3_trainer_adapter.attach_collector(trainer)
    if torch.cuda.is_available():
        collector._sync = torch.cuda.synchronize
    trainer.train()
    memory = {}
    if torch.cuda.is_available():
        memory = {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated())}
    return f3_trainer_adapter.build_report(
        collector,
        steps_requested=args.steps,
        provenance={
            "path": args.path,
            "model": args.model,
            "control": args.control,
            "weight_decay": 0.0 if args.control else args.weight_decay,
        },
        memory=memory,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", choices=("native", "unsloth"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="report JSON to write")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--control", action="store_true")
    args = parser.parse_args(argv)

    rows = json.loads(args.tokens.read_text(encoding="utf-8"))[: args.steps * args.batch]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    runner = run_unsloth if args.path == "unsloth" else run_native
    report = runner(args, rows, args.out.parent)
    report["report_path"] = str(args.out)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
