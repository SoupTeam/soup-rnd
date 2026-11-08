"""Experimental public-TRL integration smoke for academic-ds-9B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from benchmarks.harness.b13_moe_fixture import trainables
from benchmarks.harness.d3_academic_nf4_training_probe import REVISION, _load_model


def run(model_dir: Path, shard_dir: Path, output: Path, tokens: int, backend: str) -> None:
    if tokens < 2 or tokens > 512 or output.exists():
        raise ValueError("tokens must be 2..512 and output must be new")
    if backend not in ("sequential", "grouped"):
        raise ValueError("backend must be sequential or grouped")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(
        including_emulation=False
    ):
        raise ValueError("native BF16 CUDA is required")

    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, default_data_collator
    from trl import SFTConfig, SFTTrainer

    output.mkdir(parents=True)
    torch.manual_seed(102)
    model, runtime = _load_model(model_dir, shard_dir, backend)
    try:
        # build_streamed_model initializes DeepSeek RoPE through the production path.
        params = trainables(model)
        if len(params) != 160 or any("lora_" not in name for name in params):
            raise ValueError("unexpected trainable parameter set")
        before = {
            name: parameter.detach().cpu().contiguous().clone()
            for name, parameter in sorted(params.items())
        }
        # Keep token IDs in the tiny tokenizer vocabulary; dataset processing is skipped.
        row = [2 + ((index * 17 + 7) % 61) for index in range(tokens)]
        dataset = Dataset.from_dict({
            "input_ids": [row],
            "labels": [row],
            "attention_mask": [[1] * tokens],
        })
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({str(index): index for index in range(64)},
                                                  unk_token="0")),
            pad_token="0",
            eos_token="1",
        )
        args = SFTConfig(
            output_dir=str(output / "trainer-output"),
            use_cpu=False,
            bf16=True,
            fp16=False,
            max_steps=1,
            per_device_train_batch_size=1,
            learning_rate=0.01,
            optim="sgd",
            report_to="none",
            save_strategy="no",
            logging_strategy="no",
            disable_tqdm=True,
            gradient_checkpointing=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            max_grad_norm=0.0,
        )
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=default_data_collator,
        )
        result = trainer.train()
        after = {
            name: parameter.detach().cpu().contiguous()
            for name, parameter in sorted(params.items())
        }
        changed = sum(not torch.equal(before[name], after[name]) for name in before)
        if result.global_step != 1 or changed == 0:
            raise ValueError(
                "trainer did not complete a real update: "
                f"step={result.global_step}, changed={changed}"
            )
        loss = float(result.training_loss)
        if not torch.isfinite(torch.tensor(loss)):
            raise ValueError("nonfinite trainer loss")

        save_file({f"before.{name}": value for name, value in before.items()},
                  str(output / "before.safetensors"))
        save_file({f"after.{name}": value for name, value in after.items()},
                  str(output / "after.safetensors"))
        runtime_stats = runtime.stats()
        gemm_stats = runtime_stats.get("expert_gemm") or {}
        from soup_cli.utils.layer_shard import read_shard_index

        shard_index = read_shard_index(str(shard_dir))
        meta = {
            "scope": "experimental full-model public SFTTrainer smoke, one synthetic batch "
            "and one SGD step",
            "revision": REVISION,
            "tokens": tokens,
            "seed": 102,
            "backend": backend,
            "codec": "nf4-nested" if shard_index.double_quant else "nf4",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "global_step": result.global_step,
            "training_loss": loss,
            "trainable_tensors": len(params),
            "changed_trainable_tensors": changed,
            "grouped_matrices": gemm_stats.get("grouped_matrices", 0),
            "runtime": runtime_stats,
            "limitations": [
                "synthetic tokens",
                "one optimizer step",
                "test-only eager-attention selection",
                "not L2L, convergence, throughput, or general DeepSeek qualification",
            ],
        }
        for name in ("before.safetensors", "after.safetensors"):
            meta[f"{name}_sha256"] = hashlib.sha256((output / name).read_bytes()).hexdigest()
        (output / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        print(json.dumps(meta, default=str), flush=True)
    finally:
        if runtime is not None:
            runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--backend", choices=("sequential", "grouped"), default="grouped")
    args = parser.parse_args()
    run(args.model, args.shards, args.output, args.tokens, args.backend)


if __name__ == "__main__":
    main()
