"""E2 quality pilot using the real Soup SFT trainer."""

import argparse
import hashlib
import json
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

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    cfg = load_config(args.config)

    wrapper = SFTTrainerWrapper(cfg, device="cuda")
    wrapper.setup(load_dataset(cfg))

    trainer = wrapper.trainer

    assert trainer.eval_dataset is not None
    assert len(trainer.eval_dataset) == 200

    trainer.args.max_steps = args.steps
    trainer.args.save_strategy = "no"
    trainer.args.eval_strategy = "no"

    print("Starting training...", flush=True)
    wrapper.train()

    assert trainer.state.global_step == args.steps
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

    digest = hashlib.sha256()
    for name in sorted(trainable):
        digest.update(name.encode())
        digest.update(trainable[name].float().numpy().tobytes())

    report = {
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
