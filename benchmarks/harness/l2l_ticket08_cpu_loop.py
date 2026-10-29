#!/usr/bin/env python3
"""A5 ticket 08: L2L against gradient accumulation through ``L2LTrainer``, on the CPU.

One command, seconds, no GPU and no 7B::

    python benchmarks/harness/l2l_ticket08_cpu_loop.py

A tiny untied Llama (3 layers) trains for 16 optimizer steps with the ticket-04 settings
(AdamW, linear schedule with warmup, clip 1.0, bf16 autocast, k = 4, batch size 1). The
script runs it twice from the same adapters, once as plain ``Trainer`` and once as the
``L2LTrainer`` subclass, and prints per step the relative loss error and the largest
difference of the LoRA gradients and parameters. Exit code 1 if any loss differs.

``--model-dtype`` is the dtype of the streamed layers and the shards, ``--store-dtype`` the
dtype of the L2L activation store. ``sft.py`` gives both the same dtype on CUDA (bf16), so
the default is ``bfloat16`` for both. ``--model-dtype float32 --store-dtype bfloat16`` is the
control: the store rounds an fp32 residual stream to bf16, the recompute starts from other
inputs than the forward used, and the loop goes red (loss error up to 0.4%). It is a
misconfiguration, not what ``soup train`` builds.

Benchmark harness code, not shipped; it imports the tiny-model builders from ``tests/``.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # this checkout, not an editable install elsewhere
sys.path.insert(0, str(Path(__file__).resolve().parent))


def build(tmp: Path, dtype: str):
    from peft import LoraConfig, TaskType

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model
    from tests.test_v07204 import _tiny_llama_dir

    weights, _model, _config = _tiny_llama_dir(tmp, n_layers=3, tie=False)
    shards = str(tmp / "shards")
    index = shard_checkpoint(weights, shards, dtype=dtype, arch="llama")
    lora = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type=TaskType.CAUSAL_LM,
    )
    return build_streamed_model(
        model_id=weights, shard_dir=shards, index=index, lora_config=lora, device="cpu",
        dtype=dtype, buffers=2, pin=False, seed=3,
    )


def train(model, runtime, out: Path, *, l2l: bool, k: int, steps: int, store_dtype: str):
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments

    from soup_cli.trainer.l2l_trainer import attach_l2l_state, make_l2l_trainer_class
    from tests.test_l2l_trainer_mode import _Rows

    losses, grads, params = [], [], []

    class Record(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                losses.append(float(logs["loss"]))

        def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
            grads.append({n: p.grad.detach().clone() for n, p in model.named_parameters()
                          if p.grad is not None and "lora_" in n})

        def on_optimizer_step(self, args, state, control, model=None, **kw):
            params.append({n: p.detach().clone() for n, p in model.named_parameters()
                           if p.requires_grad and "lora_" in n})

    args = TrainingArguments(
        output_dir=str(out), per_device_train_batch_size=1, gradient_accumulation_steps=k,
        learning_rate=2e-2, optim="adamw_torch", lr_scheduler_type="linear", warmup_steps=0.1,
        num_train_epochs=1, logging_steps=1, save_strategy="no", report_to=[], use_cpu=True,
        seed=1, data_seed=1, max_grad_norm=1.0, weight_decay=0.0, disable_tqdm=True, bf16=True,
    )
    cls = make_l2l_trainer_class(Trainer) if l2l else Trainer
    trainer = cls(model=model, args=args, train_dataset=_Rows(k * steps), callbacks=[Record()])
    if l2l:
        attach_l2l_state(trainer, runtime, dtype=getattr(torch, store_dtype))
    trainer.train()
    return losses, grads, params


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--store-dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16)
    opts = parser.parse_args(argv)

    from tests.test_l2l_probe_harness import _load

    sched = _load("l2l_schedule")
    tmp = Path(tempfile.mkdtemp(prefix="l2l-ticket08-"))
    model, runtime = build(tmp, opts.model_dtype)
    try:
        sched.make_non_vacuous(model)
        start = sched.snapshot_adapters(model)
        ga = train(model, runtime, tmp / "ga", l2l=False, k=opts.k, steps=opts.steps,
                   store_dtype=opts.store_dtype)
        sched.restore_adapters(model, start)
        l2l = train(model, runtime, tmp / "l2l", l2l=True, k=opts.k, steps=opts.steps,
                    store_dtype=opts.store_dtype)
    finally:
        runtime.close()

    print(f"model {opts.model_dtype}, store {opts.store_dtype}, k={opts.k}, steps={opts.steps}")
    red = False
    for i, (la, lb, ga_g, lb_g, ga_p, lb_p) in enumerate(zip(ga[0], l2l[0], ga[1], l2l[1],
                                                              ga[2], l2l[2])):
        err = abs(lb - la) / abs(la) if la else float("inf")
        dg = max((ga_g[n] - lb_g[n]).abs().max().item() for n in ga_g)
        dp = max((ga_p[n] - lb_p[n]).abs().max().item() for n in ga_p)
        red |= la != lb
        print(f"step {i:2d}: loss err {err:.2e}  grad maxdiff {dg:.2e}  param maxdiff {dp:.2e}")
    print("RED: L2L differs from GA" if red else "GREEN: L2L equals GA bit for bit")
    return 1 if red else 0


if __name__ == "__main__":
    raise SystemExit(main())
