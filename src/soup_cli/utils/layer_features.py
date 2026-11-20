"""Per-layer features of a LoRA run after a short stretch of training (ZERO-2: choosing the cut).

``training.lora_layers.feature_step`` records, at that optimizer step, what an adaptive selector
(a rule, a small model or an LLM) can read to decide how many bottom layers to freeze. For every
decoder layer ``i``:

* ``update``: ``sum over its LoRA modules of ||scaling * B @ A||_F`` (how far the adapters moved);
* ``update_rank``: the mean participation ratio of ``B @ A``'s singular values, in ``[1, r]``
  (how many directions the update uses);
* ``grad_mean``, ``grad_cv``: mean and coefficient of variation of the layer's summed adapter
  gradient norm over the windows recorded so far (``Selection.norms``);
* ``cka_change``: ``1 - linear CKA`` between the layer's output on a few training rows at the
  start and at ``feature_step`` (how much its representation changed), and ``cka_own``, the part
  not inherited from the layer below (``max(0, d_i - d_{i-1})``);
* ``turn``: mean angle between the residual entering and leaving the layer, ``arccos(cos) / pi``
  (how much the layer rewrites the stream), at ``feature_step``.

The probe rows are the first ``feature_rows`` training rows, read in order on their own loader;
up to ``tokens_per_row`` tokens per row (evenly spaced) are kept per layer, in fp32 on the host.
The forwards run in eval mode under ``no_grad`` with the RNG state restored, so training is
unchanged. ``update`` and ``update_rank`` use only the adapter tensors, so they are exact under
layer streaming and L2L as well.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


def adapter_update_features(model: Any, n_layers: int,
                            layer_of: Any) -> Dict[int, Dict[str, float]]:
    """``update`` and ``update_rank`` of every decoder layer from its LoRA tensors."""
    import torch

    acc: Dict[int, Dict[str, Any]] = {}
    for name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if lora_a is None or lora_b is None or "default" not in lora_a:
            continue
        idx = layer_of(name)
        if idx is None:
            continue
        a = lora_a["default"].weight.detach().float()
        b = lora_b["default"].weight.detach().float()
        scale = float(module.scaling["default"])
        with torch.no_grad():
            gram = (b.T @ b) @ (a @ a.T)  # r x r, eigenvalues are the squared singular values
            eig = torch.linalg.eigvals(gram).real.clamp_min(0)
            total = float(eig.sum())
            sq = float((eig ** 2).sum())
        entry = acc.setdefault(idx, {"update": 0.0, "ranks": []})
        entry["update"] += scale * math.sqrt(max(total, 0.0))
        if sq > 0:
            entry["ranks"].append(total * total / sq)
    out = {}
    for idx in range(n_layers):
        entry = acc.get(idx)
        if entry is None:
            continue
        ranks = entry["ranks"]
        out[idx] = {"update": entry["update"],
                    "update_rank": sum(ranks) / len(ranks) if ranks else 0.0}
    return out


def grad_features(norms: Sequence[Any], n_layers: int) -> Dict[int, Dict[str, float]]:
    """Mean and coefficient of variation of each layer's windowed gradient norm."""
    out: Dict[int, Dict[str, float]] = {}
    for idx in range(n_layers):
        values = [w[idx] for _step, w in norms if idx in w]
        if not values:
            continue
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        out[idx] = {"grad_mean": mean, "grad_cv": math.sqrt(var) / mean if mean else 0.0}
    return out


class LayerProbe:
    """Records each decoder layer's input and output on fixed rows (a few tokens per row)."""

    def __init__(self, batches: Sequence[Any], tokens_per_row: int = 32) -> None:
        self.batches = list(batches)
        self.tokens_per_row = int(tokens_per_row)

    def capture(self, model: Any, device: Any) -> Dict[str, List[Any]]:
        import torch

        from soup_cli.utils.prefix_cache import find_decoder

        decoder = find_decoder(model)
        n = len(decoder.layers)
        ins: List[List[Any]] = [[] for _ in range(n)]
        outs: List[List[Any]] = [[] for _ in range(n)]
        pick: Dict[str, Any] = {}

        def make(i):
            def hook(_module, args, output):
                hidden_in = args[0] if args else None
                hidden_out = output[0] if isinstance(output, tuple) else output
                index = pick["index"]
                if torch.is_tensor(hidden_in):
                    ins[i].append(hidden_in[0, index].detach().float().cpu())
                outs[i].append(hidden_out[0, index].detach().float().cpu())
            return hook

        handles = [layer.register_forward_hook(make(i)) for i, layer in enumerate(decoder.layers)]
        was_training = model.training
        cuda = [torch.device(device)] if torch.device(device).type == "cuda" else []
        try:
            model.eval()
            with torch.random.fork_rng(devices=cuda), torch.no_grad():
                for batch in self.batches:
                    ids = batch["input_ids"][:1].to(device)
                    mask = batch.get("attention_mask")
                    t = int(mask[0].sum()) if mask is not None else ids.shape[1]
                    ids = ids[:, :t]
                    k = min(self.tokens_per_row, t)
                    pick["index"] = torch.linspace(0, t - 1, k).round().long().to(ids.device)
                    model(input_ids=ids)
        finally:
            for h in handles:
                h.remove()
            model.train(was_training)
        return {"in": [torch.cat(x) if x else None for x in ins],
                "out": [torch.cat(x) for x in outs]}


def linear_cka(x: Any, y: Any) -> float:
    import torch

    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    num = torch.linalg.matrix_norm(x.T @ y) ** 2
    den = torch.linalg.matrix_norm(x.T @ x) * torch.linalg.matrix_norm(y.T @ y)
    return float(num / den) if float(den) > 0 else 1.0


def representation_features(before: Dict[str, List[Any]],
                            after: Dict[str, List[Any]]) -> Dict[int, Dict[str, float]]:
    import torch

    out: Dict[int, Dict[str, float]] = {}
    prev = 0.0
    for i, (b, a) in enumerate(zip(before["out"], after["out"])):
        change = max(0.0, 1.0 - linear_cka(b, a))
        entry = {"cka_change": change, "cka_own": max(0.0, change - prev)}
        prev = change
        x = after["in"][i]
        if x is not None:
            cos = torch.nn.functional.cosine_similarity(x, a, dim=-1).clamp(-1.0, 1.0)
            entry["turn"] = float((torch.arccos(cos) / math.pi).mean())
        out[i] = entry
    return out


def merge(*parts: Dict[int, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    layers: Dict[int, Dict[str, float]] = {}
    for part in parts:
        for idx, values in part.items():
            layers.setdefault(idx, {}).update(values)
    return {str(i): {k: round(v, 6) for k, v in sorted(vals.items())}
            for i, vals in sorted(layers.items())}


def loss_summary(losses: Sequence[float]) -> Optional[Dict[str, float]]:
    if len(losses) < 4:
        return None
    q = max(1, len(losses) // 4)
    first, last = losses[:q], losses[-q:]
    return {"first_quarter": sum(first) / len(first), "last_quarter": sum(last) / len(last),
            "drop": (sum(first) / len(first) - sum(last) / len(last)) / (sum(first) / len(first)),
            "steps": len(losses)}
