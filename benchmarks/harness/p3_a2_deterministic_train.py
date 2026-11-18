"""Run Soup's actual train command with deterministic algorithms required from startup."""

import hashlib
import json
import os
import sys
from pathlib import Path


def tensor_sha(name, value):
    """Frame the name, dtype and shape, then stream the native bytes without casting."""
    import torch

    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name}: missing or non-tensor value")
    cpu = value.detach().cpu().contiguous()
    if cpu.is_floating_point() and not torch.isfinite(cpu).all().item():
        raise ValueError(f"{name}: non-finite value")
    header = json.dumps([name, str(cpu.dtype), list(cpu.shape)], separators=(",", ":"))
    digest = hashlib.sha256(header.encode() + b"\0")
    digest.update(cpu.reshape(-1).view(torch.uint8).numpy())
    return digest.hexdigest()


def install_trajectory_observer():
    """Observe the CLI's real window *before* HF clips and each completed optimizer step."""
    import torch
    from transformers import TrainerCallback

    from soup_cli.trainer import l2l_trainer

    class Trace(TrainerCallback):
        def __init__(self):
            self.pending = None
            self.completed = 0
            self.params = {}
            self.path = None

        def write(self, row, mode):
            row["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
            if not row["deterministic_algorithms"]:
                raise ValueError("deterministic algorithms disabled during training")
            with self.path.open(mode, encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")

        def on_train_begin(self, args, state, control, model, **kwargs):
            if (state.global_step, args.seed, args.data_seed) != (0, 42, 42):
                raise ValueError("expected fresh seed-42 training at step 0")
            self.params = dict(sorted((n, p) for n, p in model.named_parameters()
                                      if p.requires_grad))
            if len(self.params) != 216 or not all("lora_" in n for n in self.params):
                raise ValueError("expected 216 trainable attention LoRA tensors")
            self.path = Path(args.output_dir) / "trajectory.jsonl"
            self.write({"step": 0, "seed": args.seed, "data_seed": args.data_seed,
                        "initial_adapters": {n: tensor_sha(n, p) for n, p in self.params.items()}},
                       "x")

        def on_step_end(self, args, state, control, optimizer, **kwargs):
            if self.pending is None or state.global_step != self.completed + 1:
                raise ValueError("missing window or optimizer-step order mismatch")
            moments = {}
            for name, param in self.params.items():
                slots = optimizer.state.get(param, {})
                if set(slots) != {"step", "exp_avg", "exp_avg_sq"}:
                    raise ValueError(f"{name}: unexpected AdamW slots {list(slots)}")
                if slots["step"].item() != state.global_step:
                    raise ValueError(f"{name}: AdamW and trainer step disagree")
                for slot, value in sorted(slots.items()):
                    key = f"{name}/{slot}"
                    moments[key] = tensor_sha(key, value)
            self.pending["post_adapters"] = {n: tensor_sha(n, p) for n, p in self.params.items()}
            self.pending["adamw"] = moments
            self.write({"step": state.global_step, **self.pending}, "a")
            self.pending = None
            self.completed = state.global_step

        def on_train_end(self, args, state, control, **kwargs):
            if self.completed != 40 or self.pending is not None:
                raise ValueError("incomplete 40-step trajectory")

    trace = Trace()
    attach = l2l_trainer.attach_l2l_state
    run_window = l2l_trainer.run_window

    def attach_with_trace(trainer, *args, **kwargs):
        state = attach(trainer, *args, **kwargs)
        trainer.add_callback(trace)
        return state

    def observed_window(state, model, ids_list, label_list, n_items, device):
        losses, counters = run_window(state, model, ids_list, label_list, n_items, device)
        # This is the return boundary of the real backward. No optimizer callback, gradient
        # clipping or update has run. Copy the actual gradients, not zero_grad remnants.
        if trace.pending is not None or {len(ids_list), len(losses), len(label_list)} != {8}:
            raise ValueError("expected exactly one eight-microbatch window per optimizer step")
        grads = {n: tensor_sha(n, p.grad) for n, p in trace.params.items()}
        counts = [int((labels[:, 1:] != -100).sum().item()) for labels in label_list]
        if n_items != sum(counts) or min(counts) < 1 or any(x.numel() != 1 for x in losses):
            raise ValueError("invalid loss shape or shifted target count")
        trace.pending = {
            "input_ids": [tensor_sha(f"input_ids/{i}", t) for i, t in enumerate(ids_list)],
            "labels": [tensor_sha(f"labels/{i}", t) for i, t in enumerate(label_list)],
            "targets_per_microbatch": counts, "target_count": n_items,
            "target_count_sha256": tensor_sha(
                "target_count", torch.tensor(n_items, dtype=torch.long)
            ),
            "micro_loss_bits": [tensor_sha(f"loss/{i}", x) for i, x in enumerate(losses)],
            "micro_loss_values": [x.detach().item() for x in losses],
            "raw_lora_gradients": grads,
        }
        return losses, counters

    # Both names are looked up locally by the stock trainer at setup/run time.
    l2l_trainer.attach_l2l_state = attach_with_trace
    l2l_trainer.run_window = observed_window


def main() -> None:
    # Must precede torch import and any CUDA context creation, not just the first step.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from rich.console import Console

    torch.use_deterministic_algorithms(True, warn_only=False)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected one visible CUDA device for the stage-2 pair")
    Console().print(f"Deterministic algorithms: {torch.are_deterministic_algorithms_enabled()}; "
                    f"cuBLAS workspace: :4096:8; GPU: {torch.cuda.get_device_name(0)}")
    install_trajectory_observer()
    from soup_cli.cli import run

    sys.argv = ["soup", "train", *sys.argv[1:]]
    run()


if __name__ == "__main__":
    main()
