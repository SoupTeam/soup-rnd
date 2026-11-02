#!/usr/bin/env python3
"""G1 ticket 03: LoRA on a frozen NVFP4 base, on a small synthetic model.

The base is a stack of ``te.Linear`` layers built under ``te.quantized_model_init`` with the
``NVFP4BlockScaling`` recipe: weights are held packed in NVFP4 and frozen. Each one is wrapped by
a bf16 LoRA adapter (``LoRALinear``). The forward multiplies on the packed weights (native FP4
GEMM). The backward sends gradients to the adapters through the same frozen FP4 layers.

The check is whether that works, not how fast: the first-step gradient norm is above zero, the
adapter weights move, the loss falls, and nothing is NaN or Inf (see ``verdict``).

All numbers are synthetic: a small random model, random tokens, a fixed random token mapping as
the task. Temporary harness code, not a product path (ADR 0003). Heavy dependencies are
imported inside functions.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path
from typing import Callable, ContextManager, Dict, Iterator, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from g1_fp4_layer import UnsupportedShapeError, check_fp4_shape  # noqa: E402,F401

REPORT_LABEL = "RTX 5090, synthetic (small random model, random tokens, fixed token mapping)"

#: The task's token mapping is fixed so every run and every step learns the same function.
TASK_SEED = 12345


def check_model_shape(*, tokens: int, hidden: int, ffn: int, vocab: int) -> None:
    """Refuse a model whose linear layers the FP4 GEMM cannot run. Never falls back to NF4.

    The layers are ``hidden -> ffn``, ``ffn -> hidden`` and the head ``hidden -> vocab``.
    """
    for in_features, out_features in ((hidden, ffn), (ffn, hidden), (hidden, vocab)):
        check_fp4_shape(tokens=tokens, in_features=in_features, out_features=out_features)


#: Defined by ``_define_modules`` once torch is importable (torch is a lazy dependency).
LoRALinear = None
TinyLoRALM = None


def _torch():
    import torch

    return torch


def _define_modules() -> None:
    """Define the torch modules at import time when torch exists, so tests import them directly."""
    torch = _torch()
    nn = torch.nn
    functional = torch.nn.functional

    class _LoRALinear(nn.Module):
        """``base(x) + (alpha / rank) * x A^T B^T`` with ``base`` frozen and A, B trainable.

        ``B`` starts at zero, so the adapter starts as the identity on the base.
        """

        def __init__(self, base, rank: int, alpha: float, dtype=None, device=None):
            super().__init__()
            dtype = dtype or torch.bfloat16
            self.base = base
            for param in self.base.parameters():
                param.requires_grad_(False)
            in_features, out_features = _features(base)
            self.scale = alpha / rank
            self.lora_a = nn.Parameter(torch.empty(rank, in_features, dtype=dtype, device=device))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            self.lora_b = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype, device=device))

        def forward(self, x):
            delta = (x @ self.lora_a.T) @ self.lora_b.T
            return self.base(x) + delta * self.scale

    class _TinyLoRALM(nn.Module):
        """Embedding, residual feed-forward blocks and a head. Every linear carries an adapter."""

        def __init__(
            self, *, vocab, hidden, ffn, layers, rank, alpha, linear_factory, dtype, device="cpu"
        ):
            super().__init__()
            self.hidden = hidden
            self.embed = nn.Embedding(vocab, hidden, dtype=dtype, device=device)
            self.embed.weight.requires_grad_(False)

            def wrap(i, o):
                return LoRALinear(
                    linear_factory(i, o), rank=rank, alpha=alpha, dtype=dtype, device=device
                )

            self.up = nn.ModuleList([wrap(hidden, ffn) for _ in range(layers)])
            self.down = nn.ModuleList([wrap(ffn, hidden) for _ in range(layers)])
            self.head = wrap(hidden, vocab)

        def forward(self, tokens):
            h = self.embed(tokens)
            for up, down in zip(self.up, self.down):
                normed = functional.layer_norm(h, (self.hidden,))
                h = h + down(functional.gelu(up(normed)))
            return self.head(functional.layer_norm(h, (self.hidden,)))

        def adapters(self) -> Iterator[Tuple[str, object]]:
            for name, module in self.named_modules():
                if isinstance(module, LoRALinear):
                    yield name, module

    global LoRALinear, TinyLoRALM
    LoRALinear = _LoRALinear
    TinyLoRALM = _TinyLoRALM


def _features(base) -> Tuple[int, int]:
    return base.in_features, base.out_features


try:
    _define_modules()
except ModuleNotFoundError:  # pragma: no cover - torch is a lazy dependency of the harness
    pass


def synthetic_batch(*, vocab: int, batch: int, seq: int, seed: int):
    """Random tokens and their targets under the fixed mapping (the whole task)."""
    torch = _torch()
    mapping = torch.randperm(vocab, generator=torch.Generator().manual_seed(TASK_SEED))
    tokens = torch.randint(0, vocab, (batch, seq), generator=torch.Generator().manual_seed(seed))
    return tokens, mapping[tokens]


def loss_fn(model, tokens, targets):
    logits = model(tokens)
    return _torch().nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1)
    )


def adapter_grad_norms(model) -> Dict[str, Dict[str, float]]:
    """Gradient norm of each adapter's A and B (0.0 where no gradient arrived)."""
    out = {}
    for name, module in model.adapters():
        out[name] = {
            key: (0.0 if p.grad is None else p.grad.float().norm().item())
            for key, p in (("lora_a", module.lora_a), ("lora_b", module.lora_b))
        }
    return out


def _total_grad_norm(params) -> float:
    total = sum(p.grad.float().pow(2).sum().item() for p in params if p.grad is not None)
    return math.sqrt(total)


def train(
    model,
    *,
    vocab: int,
    batch: int,
    seq: int,
    steps: int,
    lr: float,
    device: str,
    seed: int,
    autocast: Optional[Callable[[], ContextManager]] = None,
) -> dict:
    """Train the adapters for ``steps`` AdamW steps and return the record ``verdict`` reads.

    ``autocast`` makes the context the forward and the loss run in (TransformerEngine's FP4
    autocast on the GPU); the backward runs outside it.
    """
    torch = _torch()
    trainable = [p for p in model.parameters() if p.requires_grad]
    initial = {
        name: (m.lora_a.detach().clone(), m.lora_b.detach().clone()) for name, m in model.adapters()
    }
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    context = autocast or contextlib.nullcontext
    losses: List[float] = []
    first_norm = float("nan")
    first_per_adapter: Dict[str, Dict[str, float]] = {}
    for step in range(steps):
        tokens, targets = synthetic_batch(vocab=vocab, batch=batch, seq=seq, seed=seed + step)
        tokens, targets = tokens.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with context():
            loss = loss_fn(model, tokens, targets)
        loss.backward()
        if step == 0:
            first_norm = _total_grad_norm(trainable)
            first_per_adapter = adapter_grad_norms(model)
        optimizer.step()
        losses.append(loss.item())
    changed_a = all(
        not torch.equal(m.lora_a.detach().cpu(), initial[n][0].cpu()) for n, m in model.adapters()
    )
    changed_b = all(
        not torch.equal(m.lora_b.detach().cpu(), initial[n][1].cpu()) for n, m in model.adapters()
    )
    outside = sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and not name.endswith(("lora_a", "lora_b"))
    )
    record = {
        "label": REPORT_LABEL,
        "data_kind": "synthetic",
        "steps": steps,
        "lr": lr,
        "losses": losses,
        "first_step_grad_norm": first_norm,
        "first_step_adapter_grad_norms": first_per_adapter,
        "adapters_changed": {"a": changed_a, "b": changed_b},
        "adapters_finite": all(
            bool(torch.isfinite(p).all()) for m in model.modules() if isinstance(m, LoRALinear)
            for p in (m.lora_a, m.lora_b)
        ),
        "trainable_outside_adapter": outside,
    }  # fmt: skip
    record["verdict"] = verdict(record)
    return record


def verdict(record: dict) -> dict:
    """The ticket's pass conditions, as written before the GPU run.

    - ``grad_norm_positive``: the first step's total gradient norm is finite and above 0.
    - ``adapters_changed``: every adapter's A and B differ from their initial values.
    - ``loss_decreased``: the mean of the last 5 losses is below the mean of the first 5.
    - ``finite``: no NaN or Inf in any loss, in the first gradient norm or in the adapters.
    - ``base_frozen``: no parameter outside the adapters has ``requires_grad``.
    """
    losses = record["losses"]
    window = max(1, min(5, len(losses) // 2))
    first, last = losses[:window], losses[-window:]
    norm = record["first_step_grad_norm"]
    checks = {
        "grad_norm_positive": math.isfinite(norm) and norm > 0,
        "adapters_changed": all(record["adapters_changed"].values()),
        "loss_decreased": sum(last) / window < sum(first) / window,
        "finite": all(math.isfinite(v) for v in losses)
        and math.isfinite(norm)
        and record.get("adapters_finite", True),
        "base_frozen": record["trainable_outside_adapter"] == 0,
    }
    return {"ok": all(checks.values()), "checks": checks}


def _te_factory():
    """Frozen NVFP4 ``te.Linear`` layers (packed, ``requires_grad`` off) and their autocast."""
    import torch
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import NVFP4BlockScaling

    recipe = NVFP4BlockScaling()

    def make(in_features: int, out_features: int):
        with te.quantized_model_init(enabled=True, recipe=recipe):
            layer = te.Linear(in_features, out_features, bias=False, params_dtype=torch.bfloat16)
        layer.weight.requires_grad_(False)
        return layer

    return make, lambda: te.autocast(enabled=True, recipe=recipe)


def run(args: argparse.Namespace) -> int:
    import torch
    import transformer_engine

    check_model_shape(
        tokens=args.batch * args.seq, hidden=args.hidden, ffn=args.ffn, vocab=args.vocab
    )
    torch.manual_seed(args.seed)
    factory, autocast = _te_factory()
    model = TinyLoRALM(
        vocab=args.vocab, hidden=args.hidden, ffn=args.ffn, layers=args.layers, rank=args.rank,
        alpha=args.alpha, linear_factory=factory, dtype=torch.bfloat16, device="cuda",
    )  # fmt: skip
    storage = sorted({type(m.base.weight).__name__ for _, m in model.adapters()})
    report = train(
        model, vocab=args.vocab, batch=args.batch, seq=args.seq, steps=args.steps, lr=args.lr,
        device="cuda", seed=args.seed, autocast=autocast,
    )  # fmt: skip
    report["environment"] = {
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "transformer_engine": getattr(transformer_engine, "__version__", "unknown"),
        "base_weight_storage": storage,
        "adapter_dtype": "bfloat16",
    }
    report["model"] = {
        "vocab": args.vocab, "hidden": args.hidden, "ffn": args.ffn, "layers": args.layers,
        "rank": args.rank, "alpha": args.alpha, "batch": args.batch, "seq": args.seq,
    }  # fmt: skip
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"], "storage": storage}, indent=2))
    return 0 if report["verdict"]["ok"] else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--vocab", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--ffn", type=int, default=512)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
