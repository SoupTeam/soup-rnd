"""E2 quality pilot using the real Soup SFT trainer."""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from soup_cli.config.loader import load_config
from soup_cli.data.loader import load_dataset
from soup_cli.trainer.sft import SFTTrainerWrapper


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False

    cfg = load_config(args.config)

    wrapper = SFTTrainerWrapper(cfg, device="cuda")
    wrapper.setup(load_dataset(cfg.data))

    trainer = wrapper.trainer

    assert trainer.eval_dataset is not None
    assert len(trainer.eval_dataset) == 200

    trainer.args.max_steps = args.steps
    trainer.args.save_strategy = "no"
    trainer.args.eval_strategy = "no"

    # E2_TWO_EPOCH_AUDIT_V1
    from soup_cli.bench.collector import BenchCollector

    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("Deterministic algorithms must be enabled")
    if trainer.args.world_size != 1:
        raise RuntimeError("Expected one process, not distributed training")
    if (
        trainer.args.per_device_train_batch_size != 1
        or trainer.args.gradient_accumulation_steps != 1
        or len(trainer.train_dataset) != 2000
    ):
        raise RuntimeError("Expected 2000 train rows, batch=1, accumulation=1")

    initial_digest = hashlib.sha256()
    for name, parameter in sorted(trainer.model.named_parameters()):
        if parameter.requires_grad:
            initial_digest.update(name.encode())
            initial_digest.update(
                parameter.detach().float().cpu().contiguous().numpy().tobytes()
            )
    initial_fingerprint = initial_digest.hexdigest()

    collector = BenchCollector(warmup_steps=0)
    collector.e2_runner_getter = lambda: getattr(wrapper, "_e2_runner", None)

    def synchronize_gpus():
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)

    collector._sync = synchronize_gpus
    trainer.add_callback(collector)

    batch_digest = hashlib.sha256()
    original_training_step = trainer.training_step

    def observed_training_step(model, inputs, *positional, **keyword):
        collector.observe_batch(inputs)
        snapshot = {
            key: (
                inputs[key].detach().cpu().tolist()
                if inputs.get(key) is not None else None
            )
            for key in ("input_ids", "attention_mask", "position_ids", "labels")
        }
        batch_digest.update(
            (json.dumps(snapshot, sort_keys=True) + "\n").encode("utf-8")
        )
        return original_training_step(model, inputs, *positional, **keyword)

    trainer.training_step = observed_training_step

    print("Starting training...", flush=True)
    wrapper.train()

    assert trainer.state.global_step == args.steps

    if trainer.lr_scheduler is not None:
        print(
            "Scheduler:",
            type(trainer.lr_scheduler).__name__,
            "Last epoch:",
            trainer.lr_scheduler.last_epoch,
        )
    assert getattr(wrapper, "_e2_runner", None) is None

    print("Evaluating final model...", flush=True)
    metrics = trainer.evaluate()

    loss = float(metrics["eval_loss"])
    assert np.isfinite(loss)

    trainable = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad
    }

    weights_path = Path(args.output).with_suffix(".weights.pt")
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trainable, weights_path)

    digest = hashlib.sha256()
    for name in sorted(trainable):
        digest.update(name.encode())
        digest.update(trainable[name].float().numpy().tobytes())

    report = {
        "epoch_timing": collector.epoch_records,
        "train_examples": len(trainer.train_dataset),
        "initial_trainable_fingerprint": initial_fingerprint,
        "training_batches_sha256": batch_digest.hexdigest(),
        "parameters_changed": initial_fingerprint != digest.hexdigest(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "config": args.config,
        "steps": args.steps,
        "validation_examples": len(trainer.eval_dataset),
        "validation_loss": loss,
        "trainable_fingerprint": digest.hexdigest(),
        "dataset_sha256": file_sha256(cfg.data.train),
        "seed": 42,
        "device": "cuda",
        "torch_version": torch.__version__,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
