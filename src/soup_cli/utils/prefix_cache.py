"""Reuse the output of a frozen bottom of the decoder across epochs (ZERO-2 x E2).

When every parameter below decoder layer ``cutoff`` is frozen (the input embedding, the base
weights and the LoRA adapters of layers ``0 .. cutoff-1``) and nothing in those layers is random,
the hidden state leaving layer ``cutoff-1`` is a function of the row alone. The first time a row
is seen it is computed and kept in host RAM; when the same row comes back (the next epoch) the
bottom layers are skipped and the kept tensor is used. ``training.lora_layers`` already stops the
backward at the lowest trainable layer; this removes the forward below it as well.

Unlike the E2 research runner (``research/e2-recipe-check``), this does not re-implement the
decoder's forward. It replaces the ``forward`` of the bottom layers only, so the decoder builds
its masks, rotary embeddings and per-layer arguments exactly as it always does (the hybrid
Qwen3.5 decoder passes a different mask to its DeltaNet and attention layers) and works with
gradient checkpointing (the checkpointed layer calls the replaced ``forward``):

* layer 0 looks the batch up. If every row is kept, it returns the kept tensor, and layers
  ``1 .. cutoff-1`` return their input unchanged.
* otherwise the layers run as usual and layer ``cutoff-1`` stores the rows that were missing.

A row's key is its token ids and padding mask (and position ids, when the batch has them). Padding
positions of the stored output are zeroed (in both cases, so an epoch that computes and an epoch
that reads give the same tensor). With right padding and causal layers nothing real attends to
them; an unpadded batch is passed through untouched, so with batch 1 a cached step is bit-identical
to an uncached one.

Guards: the cache acts only while the decoder is in training mode (evaluation runs the full
model); it refuses a bottom with a trainable parameter or an active dropout; it raises if a frozen
parameter is modified in place after install (its version counter moved). Entries are kept until
``max_bytes``; beyond that rows are computed every time. In-process only: nothing is written to
disk, so no cross-run fingerprint is needed.
"""

from __future__ import annotations

import hashlib
import logging
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def find_decoder(model: Any) -> Any:
    """The module holding the decoder layers (``.layers``) and the input embedding."""
    import torch.nn as nn

    for module in model.modules():
        layers = getattr(module, "layers", None)
        if isinstance(layers, nn.ModuleList) and hasattr(module, "embed_tokens"):
            return module
    raise ValueError("prefix cache: no decoder with .layers and .embed_tokens in the model")


@dataclass
class CacheStats:
    cutoff: int
    hits: int = 0
    misses: int = 0
    rows_stored: int = 0
    rows_not_stored: int = 0
    bytes_stored: int = 0
    passthrough: int = 0

    def to_json(self) -> Dict[str, int]:
        return dict(vars(self))


class _Batch:
    """What the current decoder call is doing (set by layer 0, read by the layers above)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.input_ids = None
        self.attention_mask = None
        self.position_ids = None
        self.mode = "off"  # off | hit | miss
        self.keys: List[str] = []
        self.missing: List[int] = []
        self.real = None  # bool [B, T] or None when unpadded


class PrefixCache:
    def __init__(self, model: Any, cutoff: int, max_bytes: int) -> None:
        self.decoder = find_decoder(model)
        n_layers = len(self.decoder.layers)
        if not 0 < cutoff < n_layers:
            raise ValueError(f"prefix cache: cutoff {cutoff} outside 1..{n_layers - 1}")
        self.cutoff = int(cutoff)
        self.max_bytes = int(max_bytes)
        self.store: Dict[str, Any] = {}
        self.stats = CacheStats(cutoff=self.cutoff)
        self.batch = _Batch()
        self._handles: List[Any] = []
        self._originals: List[Any] = []
        self._frozen = self._frozen_parameters()
        self._check_frozen()
        self._versions = tuple(p._version for p in self._frozen)
        self._pin = False

    # ----------------------------------------------------------------- checks

    def _bottom_modules(self) -> List[Any]:
        return [self.decoder.embed_tokens, *self.decoder.layers[: self.cutoff]]

    def _frozen_parameters(self) -> List[Any]:
        return [p for m in self._bottom_modules() for p in m.parameters()]

    def _check_frozen(self) -> None:
        if any(p.requires_grad for p in self._frozen):
            raise ValueError(
                f"prefix cache: a parameter of the embedding or of layers 0..{self.cutoff - 1} "
                "trains; the cached bottom must be frozen"
            )

    def _check_deterministic(self) -> None:
        import torch.nn as nn

        for module in self._bottom_modules():
            for sub in module.modules():
                if isinstance(sub, nn.modules.dropout._DropoutNd) and sub.training and sub.p > 0:
                    raise ValueError(
                        "prefix cache: an active dropout in the frozen bottom makes its output "
                        "random (set lora.dropout to 0)"
                    )
        if getattr(self.decoder.config, "attention_dropout", 0.0):
            raise ValueError("prefix cache: attention_dropout must be 0")

    # ------------------------------------------------------------ install

    def install(self) -> "PrefixCache":
        import torch

        self._check_deterministic()
        self._pin = torch.cuda.is_available()

        def on_embedding(_module: Any, args: Any) -> None:
            if args and torch.is_tensor(args[0]):
                self.batch.input_ids = args[0]

        def on_decoder(_module: Any, args: Any, kwargs: Dict[str, Any]) -> None:
            self.batch.reset()
            mask = kwargs.get("attention_mask")
            self.batch.attention_mask = mask if torch.is_tensor(mask) and mask.ndim == 2 else None
            pos = kwargs.get("position_ids")
            self.batch.position_ids = pos if torch.is_tensor(pos) else None
            ids = kwargs.get("input_ids")
            if torch.is_tensor(ids):
                self.batch.input_ids = ids

        self._handles.append(self.decoder.register_forward_pre_hook(on_decoder, with_kwargs=True))
        self._handles.append(self.decoder.embed_tokens.register_forward_pre_hook(on_embedding))
        for idx in range(self.cutoff):
            layer = self.decoder.layers[idx]
            original = layer.forward
            self._originals.append((layer, vars(layer).get("forward")))
            layer.forward = types.MethodType(self._wrap(idx, original), layer)
        return self

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for layer, own in self._originals:
            if own is None:
                vars(layer).pop("forward", None)
            else:
                layer.forward = own
        self._originals.clear()
        self.store.clear()
        self.batch.reset()

    # ------------------------------------------------------------ forward

    def _wrap(self, idx: int, original: Any) -> Any:
        cache = self
        last = self.cutoff - 1

        def forward(layer_self: Any, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
            batch = cache.batch
            if idx == 0:
                batch.mode = "off"
                if layer_self.training:
                    hit = cache._lookup(hidden_states)
                    if hit is not None:
                        return hit
            elif batch.mode == "hit":
                return hidden_states
            out = original(hidden_states, *args, **kwargs)
            if idx == last and batch.mode == "miss":
                out = cache._finish(out)
            return out

        return forward

    def _row_keys(self, hidden: Any) -> Optional[List[str]]:
        import torch

        batch = self.batch
        ids = batch.input_ids
        b, t = hidden.shape[0], hidden.shape[1]
        if ids is None or tuple(ids.shape) != (b, t):
            return None
        mask = batch.attention_mask
        if mask is not None and tuple(mask.shape) != (b, t):
            return None
        real = None
        if mask is not None and not bool(mask.all()):
            real = mask.to(torch.bool)
        batch.real = real
        ids_cpu = ids.detach().to("cpu", torch.int64)
        real_cpu = real.to("cpu") if real is not None else None
        pos_cpu = None
        if batch.position_ids is not None:
            pos = batch.position_ids.detach().to("cpu", torch.int64)
            pos_cpu = pos.reshape(-1, b, t)[-1] if pos.numel() >= b * t else None
        keys = []
        for i in range(b):
            h = hashlib.sha256()
            h.update(str(t).encode())
            h.update(ids_cpu[i].numpy().tobytes())
            if real_cpu is not None:
                h.update(real_cpu[i].numpy().tobytes())
            if pos_cpu is not None:
                h.update(pos_cpu[i].numpy().tobytes())
            keys.append(h.hexdigest())
        return keys

    def _lookup(self, hidden: Any) -> Any:
        import torch

        current = tuple(p._version for p in self._frozen)
        if current != self._versions:
            raise RuntimeError(
                "prefix cache: a frozen parameter of the cached bottom changed in place after "
                "the cache was installed"
            )
        keys = self._row_keys(hidden)
        batch = self.batch
        if keys is None:
            self.stats.passthrough += 1
            return None
        batch.keys = keys
        missing = [i for i, k in enumerate(keys) if k not in self.store]
        if missing:
            batch.mode = "miss"
            batch.missing = missing
            self.stats.misses += 1
            return None
        batch.mode = "hit"
        self.stats.hits += 1
        rows = [self.store[k] for k in keys]
        device = hidden.device
        non_blocking = device.type == "cuda"
        if batch.real is None:
            return torch.stack([r.to(device, non_blocking=non_blocking) for r in rows])
        out = hidden.new_zeros(hidden.shape)
        for i, row in enumerate(rows):
            out[i, batch.real[i]] = row.to(device, non_blocking=non_blocking)
        return out

    def _finish(self, out: Any) -> Any:
        """Zero the padding of a computed bottom output and keep its missing rows."""
        batch = self.batch
        real = batch.real
        if real is not None:
            out = out.masked_fill(~real.to(out.device).unsqueeze(-1), 0)
        for i in batch.missing:
            row = out[i] if real is None else out[i, real[i].to(out.device)]
            size = row.numel() * row.element_size()
            if self.stats.bytes_stored + size > self.max_bytes:
                self.stats.rows_not_stored += 1
                continue
            kept = row.detach().to("cpu", copy=True)
            if self._pin:
                try:
                    kept = kept.pin_memory()
                except RuntimeError:  # pinned RAM exhausted: keep it pageable
                    pass
            self.store[batch.keys[i]] = kept
            self.stats.rows_stored += 1
            self.stats.bytes_stored += size
        return out
