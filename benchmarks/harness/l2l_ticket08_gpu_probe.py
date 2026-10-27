#!/usr/bin/env python3
"""A5 ticket 08: where does L2L leave GA in the first optimizer steps of ``soup train``?

``run`` calls the ``soup train`` command in-process (as ticket 04's arm does) and records,
without changing the run:

* the LoRA gradients at every optimizer step, right before the optimizer step, with the
  clip already applied (full tensors for the first ``--grad-steps`` steps, a SHA-256 of
  the bytes for the rest);
* the LoRA parameters after each of the first ``--grad-steps`` steps;
* every logged loss;
* the dtypes the L2L path meets: the activation store, the hidden state that goes into it
  at each layer, the LoRA parameters and the streamed layers.

``compare`` takes two such directories (GA and L2L) on a machine without a GPU and prints,
per step, the largest gradient difference and the parameter that has it. The first step
with a difference tells which half of the optimizer step to look in: a difference at
step 0, where the weights are the same in both arms, is in forward/backward, not in the
optimizer or the schedule.

Benchmark harness code, not shipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # this checkout


# ==========================================================================
# pure parts (tested on the CPU)
# ==========================================================================
def compare_gradients(
    ref: Sequence[Dict[str, Any]], cand: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Per step both runs have: are the gradients equal, and where is the largest difference.

    A parameter present in only one run is listed under ``missing`` and makes the step unequal.
    """
    report: List[Dict[str, Any]] = []
    for step, (left, right) in enumerate(zip(ref, cand)):
        missing = sorted(set(left) ^ set(right))
        worst, worst_diff, largest = None, 0.0, 0.0
        for name in sorted(set(left) & set(right)):
            diff = float((left[name].double() - right[name].double()).abs().max())
            largest = max(largest, float(left[name].double().abs().max()))
            if diff > worst_diff:
                worst, worst_diff = name, diff
        report.append({
            "step": step,
            "equal": worst_diff == 0.0 and not missing,
            "max_abs_diff": worst_diff,
            "max_abs_ref": largest,
            "worst_param": worst,
            "missing": missing,
        })
    return report


class DtypeLog:
    """The dtypes the L2L path meets. ``store_rounds_hidden`` is true when the activation
    store is narrower than the hidden state put into it (bf16 store, fp32 residual stream):
    the recompute in the backward then starts from other inputs than the forward used."""

    def __init__(self) -> None:
        self.store: Optional[str] = None
        self.hidden: set = set()
        self.params: Dict[str, int] = {}
        self.layers: Dict[str, int] = {}
        self._store_dtype: Any = None
        self._hidden_dtypes: set = set()

    def note_store(self, dtype: Any) -> None:
        self._store_dtype = dtype
        self.store = str(dtype)

    def note_hidden(self, dtype: Any) -> None:
        self._hidden_dtypes.add(dtype)
        self.hidden.add(str(dtype))

    def note_params(self, model: Any) -> None:
        for name, param in model.named_parameters():
            bucket = self.params if "lora_" in name else self.layers
            bucket[str(param.dtype)] = bucket.get(str(param.dtype), 0) + param.numel()

    def summary(self) -> Dict[str, Any]:
        rounds = False
        if self._store_dtype is not None:
            rounds = any(
                dtype.itemsize > self._store_dtype.itemsize for dtype in self._hidden_dtypes
            )
        return {
            "store_dtype": self.store,
            "hidden_dtypes_into_store": sorted(self.hidden),
            "lora_param_elements_by_dtype": self.params,
            "other_param_elements_by_dtype": self.layers,
            "store_rounds_hidden": rounds,
        }


def tensor_sha(tensor: Any) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(-1).numpy().tobytes()).hexdigest()


# ==========================================================================
# the process
# ==========================================================================
def run(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="directory for this run's record")
    parser.add_argument("--name", default=None)
    parser.add_argument("--sdpa", default="MATH", choices=["MATH", "EFFICIENT_ATTENTION"])
    parser.add_argument("--grad-steps", type=int, default=4)
    parsed = parser.parse_args(argv)
    out = Path(parsed.out)
    out.mkdir(parents=True, exist_ok=True)

    import l2l_mistral_stand as stand

    stand.set_deterministic_env()
    import torch
    import transformers
    from torch.nn.attention import SDPBackend, sdpa_kernel

    stand.enable_deterministic()

    dtypes = DtypeLog()
    grads_full: List[Dict[str, Any]] = []
    params_full: List[Dict[str, Any]] = []
    grad_sha: List[Dict[str, str]] = []
    losses: List[Dict[str, Any]] = []

    from soup_cli.utils import l2l_activations

    real_init_store = l2l_activations.ActivationStore.__init__
    real_put = l2l_activations.ActivationStore.put

    def init_store(self: Any, *args: Any, **kwargs: Any) -> None:
        real_init_store(self, *args, **kwargs)
        dtypes.note_store(self.dtype)

    def put(self: Any, layer: int, mb: int, tensor: Any) -> None:
        dtypes.note_hidden(tensor.dtype)
        return real_put(self, layer, mb, tensor)

    l2l_activations.ActivationStore.__init__ = init_store
    l2l_activations.ActivationStore.put = put

    def lora_items(model: Any) -> Dict[str, Any]:
        return {
            name: param for name, param in model.named_parameters()
            if "lora_" in name and param.requires_grad
        }

    class Callback(transformers.TrainerCallback):
        def on_train_begin(self, args: Any, state: Any, control: Any, model: Any = None,
                           **kw: Any) -> None:
            dtypes.note_params(model)

        def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, model: Any = None,
                                  **kw: Any) -> None:
            step = len(grad_sha)
            items = {n: p.grad for n, p in lora_items(model).items() if p.grad is not None}
            grad_sha.append({n: tensor_sha(g) for n, g in items.items()})
            if step < parsed.grad_steps:
                grads_full.append({n: g.detach().float().cpu().clone() for n, g in items.items()})

        def on_optimizer_step(self, args: Any, state: Any, control: Any, model: Any = None,
                              **kw: Any) -> None:
            if len(params_full) < parsed.grad_steps:
                params_full.append({
                    n: p.detach().float().cpu().clone() for n, p in lora_items(model).items()
                })

        def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kw: Any) -> None:
            if logs and "loss" in logs:
                losses.append({"step": int(state.global_step), "loss": float(logs["loss"]),
                               "learning_rate": logs.get("learning_rate")})

    real_init = transformers.Trainer.__init__

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        real_init(self, *args, **kwargs)
        self.add_callback(Callback())

    transformers.Trainer.__init__ = init

    from soup_cli.cli import app

    command = ["train", "--config", parsed.config, "--yes"]
    if parsed.name:
        command += ["--name", parsed.name]
    failure: Optional[str] = None
    started = time.time()
    try:
        with sdpa_kernel(getattr(SDPBackend, parsed.sdpa)):
            app(command, standalone_mode=False)
    except SystemExit as exc:
        if exc.code not in (0, None):
            failure = f"soup train exited with status {exc.code}"
    except Exception as exc:  # the record of a failed run is kept, then the exit code says so
        failure = f"{type(exc).__name__}: {exc}"

    torch.save(grads_full, out / "grads.pt")
    torch.save(params_full, out / "params.pt")
    record = {
        "config": parsed.config,
        "sdpa": parsed.sdpa,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "failure": failure,
        "wall_s": time.time() - started,
        "losses": losses,
        "grad_sha256_by_step": grad_sha,
        "dtypes": dtypes.summary(),
    }
    (out / "run.json").write_text(json.dumps(record, indent=1, default=str))
    print(f"{parsed.config}: {len(losses)} losses, {len(grad_sha)} grad steps, failure={failure}")
    return 1 if failure else 0


def compare(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="compare")
    parser.add_argument("ref", help="directory of the GA run")
    parser.add_argument("cand", help="directory of the L2L run")
    parsed = parser.parse_args(argv)

    import torch

    ref_dir, cand_dir = Path(parsed.ref), Path(parsed.cand)
    ref_run = json.loads((ref_dir / "run.json").read_text())
    cand_run = json.loads((cand_dir / "run.json").read_text())
    grads = compare_gradients(
        torch.load(ref_dir / "grads.pt"), torch.load(cand_dir / "grads.pt")
    )
    params = compare_gradients(
        torch.load(ref_dir / "params.pt"), torch.load(cand_dir / "params.pt")
    )
    print("L2L dtypes:", json.dumps(cand_run["dtypes"], indent=1))
    for row in grads:
        print(
            f"step {row['step']:2d}: grad equal={row['equal']} maxdiff={row['max_abs_diff']:.3e} "
            f"(max|g|={row['max_abs_ref']:.3e}) worst={row['worst_param']}"
        )
    for row in params:
        print(f"step {row['step']:2d}: param after step maxdiff={row['max_abs_diff']:.3e}")
    same_sha = [
        left == right
        for left, right in zip(ref_run["grad_sha256_by_step"], cand_run["grad_sha256_by_step"])
    ]
    print("gradient bytes equal per step:", same_sha)
    ref_losses = [row["loss"] for row in ref_run["losses"]]
    cand_losses = [row["loss"] for row in cand_run["losses"]]
    errors = [abs(b - a) / abs(a) if a else float("inf") for a, b in zip(ref_losses, cand_losses)]
    print("loss relative error per step:", " ".join(f"{e:.2e}" for e in errors))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("run", "compare"):
        print("usage: l2l_ticket08_gpu_probe.py run|compare ...", file=sys.stderr)
        return 2
    return (run if argv[0] == "run" else compare)(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
