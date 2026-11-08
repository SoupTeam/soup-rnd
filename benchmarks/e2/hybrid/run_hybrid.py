"""E2 hybrid quality run (hybrid/PROTOCOL.txt section 3).

benchmarks/e2/quality/run_quality.py with one change: after optimizer step
--freeze-at-step, the LoRA tensors of every decoder layer below the top --keep-top
layers stop training (requires_grad=False). The report adds a "hybrid" block with
hashes of the frozen and kept tensors, and fingerprints cover every LoRA tensor.

--seed (default 42) seeds Python, NumPy and torch; without --keep-top nothing is frozen
and the run is an all-layer run (hybrid_seeds/PROTOCOL.txt section 2).

--cache-dir (with --keep-top) installs the E2 frozen-prefix activation cache right after
the freeze, keyed to the frozen bottom, in that new empty directory
(hybrid_cache/PROTOCOL.txt section 3). Without it nothing about the run changes.
"""

import argparse
import hashlib
import json
import os
import random
import re
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from transformers import TrainerCallback

from soup_cli.config.loader import load_config
from soup_cli.data.loader import load_dataset
from soup_cli.trainer.sft import SFTTrainerWrapper

LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def lora_parameters(model):
    """Every LoRA tensor with its decoder layer index, sorted by name."""
    out = []
    for name, parameter in sorted(model.named_parameters()):
        if ".lora_" in name:
            match = LAYER_RE.search(name)
            if match is None:
                raise RuntimeError(f"LoRA tensor outside a decoder layer: {name}")
            out.append((name, int(match.group(1)), parameter))
    return out


def lora_digest(entries):
    h = hashlib.sha256()
    for name, _, parameter in entries:
        h.update(name.encode())
        h.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def split_layers(model, keep_top):
    entries = lora_parameters(model)
    layers = max(layer for _, layer, _ in entries) + 1
    if not 1 <= keep_top < layers:
        raise ValueError(f"keep_top must be in [1, {layers - 1}]")
    cutoff = layers - keep_top
    bottom = [e for e in entries if e[1] < cutoff]
    top = [e for e in entries if e[1] >= cutoff]
    return layers, cutoff, bottom, top


class FreezeBottomAtStep(TrainerCallback):
    """After optimizer step `step`, stop training LoRA below the top `keep_top` layers."""

    def __init__(self, model, keep_top, step, cache_dir=None, model_revision=None,
                 on_install=None):
        self.model, self.keep_top, self.step = model, keep_top, step
        self.record = {"freeze_at_step": step, "keep_top": keep_top, "frozen": False}
        self.cache_dir, self.model_revision = cache_dir, model_revision
        self.on_install = on_install
        self.stack, self.runner = None, None

    def install_cache(self):
        """Install the E2 cache over the now-frozen bottom (as SFTTrainerWrapper.train)."""
        from soup_cli.utils.frozen_prefix_cache import (
            FrozenPrefixCache,
            build_cache_metadata,
            fingerprint_prefix_configuration,
        )
        from soup_cli.utils.frozen_prefix_forward import install_frozen_prefix_cache
        from soup_cli.utils.layer_stream_runtime import decoder_owner

        started = time.perf_counter()
        decoder = decoder_owner(self.model)
        cutoff = len(decoder.layers) - self.keep_top
        cache = FrozenPrefixCache(self.cache_dir)
        config_fingerprint = fingerprint_prefix_configuration(decoder, cutoff)

        def metadata_factory(*, input_ids, attention_mask, position_ids):
            return build_cache_metadata(
                model_revision=self.model_revision,
                frozen_prefix_fingerprint="runner-computed",
                config_fingerprint=config_fingerprint,
                cutoff=cutoff,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        self.stack = ExitStack()
        self.runner = self.stack.enter_context(install_frozen_prefix_cache(
            model=self.model, cache=cache, cutoff=cutoff, metadata_factory=metadata_factory,
        ))
        if self.on_install is not None:
            self.on_install(self.runner)
        self.record["cache"] = {
            "installed_at_step": self.step,
            "cutoff": cutoff,
            "install_seconds": time.perf_counter() - started,
        }

    def close_cache(self):
        if self.stack is None:
            return
        files = [p for p in Path(self.cache_dir).iterdir() if p.is_file()]
        self.record["cache"].update({
            "hits": self.runner.hits,
            "misses": self.runner.misses,
            "files": len(files),
            "bytes": sum(p.stat().st_size for p in files),
        })
        self.stack.close()
        self.stack = None
        if self.on_install is not None:
            self.on_install(None)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step != self.step or self.record["frozen"]:
            return
        layers, cutoff, bottom, top = split_layers(self.model, self.keep_top)
        for _, _, parameter in bottom:
            parameter.requires_grad_(False)
            parameter.grad = None
        self.record.update({
            "frozen": True,
            "layers": layers,
            "frozen_layers": sorted({layer for _, layer, _ in bottom}),
            "frozen_tensors": [name for name, _, _ in bottom],
            "bottom_at_freeze": lora_digest(bottom),
            "top_at_freeze": lora_digest(top),
        })
        if self.cache_dir is not None:
            self.install_cache()

    def finish(self, initial_bottom, initial_top):
        _, _, bottom, top = split_layers(self.model, self.keep_top)
        record = self.record
        record["bottom_at_end"] = lora_digest(bottom)
        record["top_at_end"] = lora_digest(top)
        record["bottom_changed_before_freeze"] = record.get("bottom_at_freeze") != initial_bottom
        record["bottom_unchanged_after_freeze"] = (
            record.get("bottom_at_freeze") == record["bottom_at_end"]
        )
        record["top_changed_after_freeze"] = record.get("top_at_freeze") != record["top_at_end"]
        record["top_changed_before_freeze"] = record.get("top_at_freeze") != initial_top
        record["requires_grad_after"] = sorted(
            {str(p.requires_grad) for _, _, p in bottom}
        )
        return record


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--output", required=True)
    parser.add_argument("--keep-top", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--freeze-at-step", type=int, default=2000)
    args = parser.parse_args()

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False

    cfg = load_config(args.config)
    if (cfg.training.seed, cfg.training.data_seed) != (args.seed, args.seed):
        raise RuntimeError("Config seed/data_seed differ from --seed")

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

    if len(lora_parameters(trainer.model)) != sum(
        1 for name, p in trainer.model.named_parameters() if p.requires_grad
    ):
        raise RuntimeError("Expected exactly the LoRA tensors to be trainable at start")
    freezer = None
    if args.keep_top is not None:
        _, _, initial_bottom, initial_top = split_layers(trainer.model, args.keep_top)
        initial_bottom, initial_top = lora_digest(initial_bottom), lora_digest(initial_top)
        if args.cache_dir is not None:
            args.cache_dir.mkdir(parents=True, exist_ok=False)
        freezer = FreezeBottomAtStep(
            trainer.model, args.keep_top, args.freeze_at_step,
            cache_dir=args.cache_dir, model_revision=str(cfg.base),
            on_install=lambda runner: setattr(wrapper, "_e2_runner", runner),
        )
        trainer.add_callback(freezer)
    elif args.cache_dir is not None:
        raise ValueError("--cache-dir needs --keep-top")

    print("Starting training...", flush=True)
    train_started = time.perf_counter()
    try:
        wrapper.train()
    finally:
        if freezer is not None:
            freezer.close_cache()
    train_call_seconds = time.perf_counter() - train_started

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

    hybrid = None
    if freezer is not None:
        hybrid = freezer.finish(initial_bottom, initial_top)
        if not hybrid["frozen"]:
            raise RuntimeError("The freeze step was never reached")
    # Every LoRA tensor, frozen or not: the saved adapter holds all of them.
    trainable = {
        name: parameter.detach().cpu().contiguous()
        for name, _, parameter in lora_parameters(trainer.model)
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
        "seed": args.seed,
        "device": "cuda",
        "torch_version": torch.__version__,
        "hybrid": hybrid,
        "train_call_seconds": train_call_seconds,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
