#!/usr/bin/env python3
"""A5 ticket 08: one L2L step against gradient accumulation, without a trainer.

Usage: ``l2l_ticket08_step_isolation.py DEVICE DTYPE`` (``mps`` or ``cpu``, ``bfloat16``).

A tiny untied Llama (3 layers, layers and activation store in DTYPE, LoRA in fp32) runs k = 4
micro-batches as plain accumulation and as ``L2LStep.run``, with and without autocast around
the call, and the script prints the largest LoRA-gradient difference. HF's ``Trainer`` calls
the accumulation backward after the autocast block, so ``GA`` here does the same; the control
is GA with the backward inside autocast, which is how the L2L step used to run its backward.

Benchmark harness code, not shipped; it builds the tiny model with helpers from ``tests/``.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # this checkout, not an editable install elsewhere
sys.path.insert(0, str(Path(__file__).resolve().parent))

K = 4


def build(device: str, dtype: str):
    from peft import LoraConfig, TaskType

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model
    from tests.test_v07204 import _tiny_llama_dir

    tmp = Path(tempfile.mkdtemp(prefix="l2l-ticket08-iso-"))
    weights, _model, _config = _tiny_llama_dir(tmp, n_layers=3, tie=False)
    shards = str(tmp / "shards")
    index = shard_checkpoint(weights, shards, dtype=dtype, arch="llama")
    lora = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type=TaskType.CAUSAL_LM,
    )
    return build_streamed_model(
        model_id=weights, shard_dir=shards, index=index, lora_config=lora, device=device,
        dtype=dtype, buffers=2, pin=False, seed=3,
    )


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    device, dtype = argv

    import torch

    from tests.test_l2l_probe_harness import _load, _micro_batches

    sched, acts = _load("l2l_schedule"), _load("l2l_activations")
    model, runtime = build(device, dtype)
    try:
        sched.make_non_vacuous(model)
        model.train()
        mbs = [ids.to(device) for ids in _micro_batches(K)]
        total = sched.label_tokens(mbs)

        def autocast():
            return torch.autocast(device, dtype=torch.bfloat16)

        def grads():
            return {n: g.float().cpu() for n, g in sched.collect_grads(model).items()}

        def ga(*, use_autocast: bool, backward_inside: bool = False):
            sched.zero_grads(model)
            for ids in mbs:
                if use_autocast:
                    with autocast():
                        loss = model(input_ids=ids, labels=ids, num_items_in_batch=total).loss
                        if backward_inside:
                            loss.backward()
                    if not backward_inside:
                        loss.backward()
                else:
                    model(input_ids=ids, labels=ids, num_items_in_batch=total).loss.backward()
            return grads()

        def l2l(*, use_autocast: bool):
            sched.zero_grads(model)
            call = sched.capture_layer_call(model, mbs[0])
            step = sched.L2LStep(model, runtime, call)
            store = acts.ActivationStore(
                n_layers=3, k=K, chunk_shape=(1, mbs[0].shape[1], 64),
                dtype=getattr(torch, dtype), device=device, pin=False,
            )
            if use_autocast:
                with autocast():
                    step.run(mbs, store, n_items=total)
            else:
                step.run(mbs, store, n_items=total)
            return grads()

        def diff(left, right):
            return max((left[n] - right[n]).abs().max().item() for n in left)

        for use_autocast in (False, True):
            reference = ga(use_autocast=use_autocast)
            repeat = diff(reference, ga(use_autocast=use_autocast))
            against_l2l = diff(reference, l2l(use_autocast=use_autocast))
            print(
                f"dev={device} model={dtype} autocast={use_autocast}: "
                f"GA vs GA {repeat:.2e}   GA vs L2L {against_l2l:.2e}"
            )
        control = diff(ga(use_autocast=True, backward_inside=True), l2l(use_autocast=True))
        outside = diff(ga(use_autocast=True), ga(use_autocast=True, backward_inside=True))
        print(
            f"dev={device}: GA(backward inside autocast) vs L2L {control:.2e};"
            f"  GA(backward outside) vs GA(backward inside) {outside:.2e}"
        )
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
