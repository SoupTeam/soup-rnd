"""Reuse the output of a frozen bottom of the decoder across epochs (ZERO-2 x E2).

When every parameter below decoder layer ``cutoff`` is frozen (the input embedding, the base
weights and the LoRA adapters of layers ``0 .. cutoff-1``) and nothing in those layers is random,
the hidden state leaving layer ``cutoff-1`` is a function of the row alone. The first time a row
is seen it is computed and kept in host RAM; when the same row comes back (the next epoch) the
bottom layers are skipped and the kept tensor is used. ``training.lora_layers`` already stops the
backward at the lowest trainable layer; this removes the forward below it as well.

Unlike the E2 research runner (``research/e2-recipe-check``), this does not re-implement the
decoder's forward. Three paths share one store:

* **Resident model** (and gradient checkpointing). A forward pre-hook on the decoder looks the
  batch up before anything runs; the ``forward`` of the bottom layers is replaced, so on a hit
  layer 0 returns the kept tensor and layers ``1 .. cutoff-1`` return their input, while on a
  miss they run and layer ``cutoff-1`` stores the missing rows. The decoder still builds its
  masks, rotary embeddings and per-layer arguments itself (the hybrid Qwen3.5 decoder passes a
  different mask to its DeltaNet and attention layers).
* **Layer streaming** (``stream_layers``). The same, and on a hit the step's first weight load
  is layer ``cutoff`` instead of layer 0 (``StreamPrefetcher.prime_start``): the bottom layers
  are neither read nor computed.
* **L2L** (``training.l2l``). The step walks the layers itself and never calls the decoder, so
  the replaced forwards stay inert (they act only inside a decoder call). ``L2LStep`` asks
  :meth:`PrefixCache.l2l_lookup`: if every micro-batch of the window is kept, the window starts
  at layer ``cutoff`` and the bottom is not loaded; otherwise it runs and stores through
  :meth:`PrefixCache.l2l_store`.

A row's key is its length, token ids and padding mask (and position ids, when the batch has
them). Padding positions of the stored output are zeroed (in both cases, so an epoch that
computes and an epoch that reads give the same tensor). With right padding and causal layers
nothing real attends to them; an unpadded batch is passed through untouched, so with one row per
batch a cached step is bit-identical to an uncached one.

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
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: Attribute on the decoder through which ``L2LStep`` finds the installed cache.
OWNER_ATTR = "_soup_prefix_cache"


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
    l2l_hits: int = 0
    l2l_misses: int = 0

    def to_json(self) -> Dict[str, int]:
        return dict(vars(self))


class _Batch:
    """What the current decoder call is doing (set by the pre-hook, read by the bottom layers)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.armed = False  # inside a training-mode decoder call
        self.mode = "off"  # off | hit | miss
        self.shape = None  # (B, T) the keys were made for
        self.keys: List[str] = []
        self.missing: List[int] = []
        self.real = None  # bool [B, T] or None when unpadded


def _row_key(t: int, ids: Any, real: Any = None, pos: Any = None) -> str:
    h = hashlib.sha256()
    h.update(str(int(t)).encode())
    h.update(ids.numpy().tobytes())
    if real is not None:
        h.update(real.numpy().tobytes())
    if pos is not None:
        h.update(pos.numpy().tobytes())
    return h.hexdigest()


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
        self._fresh_ids = None
        self.prefetcher = getattr(self.decoder.layers[0], "prefetcher", None)

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

    def _check_versions(self) -> None:
        if tuple(p._version for p in self._frozen) != self._versions:
            raise RuntimeError(
                "prefix cache: a frozen parameter of the cached bottom changed in place after "
                "the cache was installed"
            )

    # ------------------------------------------------------------ install

    def install(self) -> "PrefixCache":
        import torch

        self._check_deterministic()
        self._pin = torch.cuda.is_available()

        def on_embedding(_module: Any, args: Any) -> None:
            # A multimodal wrapper embeds the ids itself and hands the decoder inputs_embeds;
            # the ids it embedded just before the decoder call are the batch's.
            if args and torch.is_tensor(args[0]):
                self._fresh_ids = args[0]

        def on_decoder(module: Any, args: Any, kwargs: Dict[str, Any]) -> None:
            ids = kwargs.get("input_ids")
            if not torch.is_tensor(ids):
                ids = self._fresh_ids
            self._fresh_ids = None
            self._decide(module, ids, kwargs)

        def after_decoder(*_args: Any) -> None:
            self.disarm()

        # Prepended: on a streamed model it must decide before the runtime's own pre-hook
        # primes the first weight load.
        self._handles.append(self.decoder.register_forward_pre_hook(
            on_decoder, with_kwargs=True, prepend=True))
        # always_call: a forward cut short by an exception (L2L's capture_layer_call stops the
        # decoder once it has seen every layer type) must not leave the batch armed, or the
        # bottom layers would act on a later call with another row's keys.
        self._handles.append(self.decoder.register_forward_hook(after_decoder, always_call=True))
        self._handles.append(self.decoder.embed_tokens.register_forward_pre_hook(on_embedding))
        for idx in range(self.cutoff):
            layer = self.decoder.layers[idx]
            original = layer.forward
            self._originals.append((layer, vars(layer).get("forward")))
            layer.forward = types.MethodType(self._wrap(idx, original), layer)
        setattr(self.decoder, OWNER_ATTR, self)
        if self.prefetcher is not None and hasattr(self.prefetcher, "backward_floor"):
            self.prefetcher.backward_floor = self.cutoff
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
        if getattr(self.decoder, OWNER_ATTR, None) is self:
            delattr(self.decoder, OWNER_ATTR)
        self._set_prime_start(0)
        if self.prefetcher is not None and hasattr(self.prefetcher, "backward_floor"):
            self.prefetcher.backward_floor = 0
        self.store.clear()
        self.batch.reset()

    def disarm(self) -> None:
        """Forget the current decoder call: the bottom layers run plainly until the next one."""
        self.batch.reset()
        self._set_prime_start(0)

    def _set_prime_start(self, idx: int) -> None:
        if self.prefetcher is not None and hasattr(self.prefetcher, "prime_start"):
            self.prefetcher.prime_start = idx

    # ------------------------------------------------------- decoder path

    def _decide(self, module: Any, ids: Any, kwargs: Dict[str, Any]) -> None:
        import torch

        batch = self.batch
        batch.reset()
        self._set_prime_start(0)
        if not module.training:
            return
        mask = kwargs.get("attention_mask")
        mask = mask if torch.is_tensor(mask) and mask.ndim == 2 else None
        pos = kwargs.get("position_ids")
        pos = pos if torch.is_tensor(pos) else None
        batch.armed = True
        keys = self._batch_keys(ids, mask, pos)
        if keys is None:
            self.stats.passthrough += 1
            return
        self._check_versions()
        batch.keys = keys
        batch.missing = [i for i, k in enumerate(keys) if k not in self.store]
        if batch.missing:
            batch.mode = "miss"
            self.stats.misses += 1
        else:
            batch.mode = "hit"
            self.stats.hits += 1
            self._set_prime_start(self.cutoff)

    def _batch_keys(self, ids: Any, mask: Any, pos: Any) -> Optional[List[str]]:
        import torch

        if ids is None or ids.ndim != 2:
            return None
        b, t = ids.shape
        if mask is not None and tuple(mask.shape) != (b, t):
            return None
        real = None
        if mask is not None and not bool(mask.all()):
            real = mask.to(torch.bool)
        self.batch.real = real
        self.batch.shape = (b, t)
        ids_cpu = ids.detach().to("cpu", torch.int64)
        real_cpu = real.to("cpu") if real is not None else None
        pos_cpu = None
        if pos is not None and pos.numel() >= b * t:
            pos_cpu = pos.detach().to("cpu", torch.int64).reshape(-1, b, t)[-1]
        return [
            _row_key(t, ids_cpu[i], None if real_cpu is None else real_cpu[i],
                     None if pos_cpu is None else pos_cpu[i])
            for i in range(b)
        ]

    def _wrap(self, idx: int, original: Any) -> Any:
        cache = self
        last = self.cutoff - 1

        def forward(layer_self: Any, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
            batch = cache.batch
            if batch.armed and batch.mode == "hit":
                if idx == 0:
                    return cache._kept(hidden_states)
                return hidden_states
            out = original(hidden_states, *args, **kwargs)
            if idx == last and batch.armed and batch.mode == "miss":
                out = cache._finish(out)
            return out

        return forward

    def _kept(self, hidden: Any) -> Any:
        import torch

        batch = self.batch
        if tuple(hidden.shape[:2]) != batch.shape:
            raise RuntimeError(
                f"prefix cache: layer 0 got hidden states of shape {tuple(hidden.shape)} for a "
                f"batch looked up as {batch.shape}"
            )
        rows = [self.store[k] for k in batch.keys]
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
        if tuple(out.shape[:2]) != batch.shape:
            raise RuntimeError(
                f"prefix cache: layer {self.cutoff - 1} returned shape {tuple(out.shape)} for a "
                f"batch looked up as {batch.shape}; nothing is kept"
            )
        real = batch.real
        if real is not None:
            out = out.masked_fill(~real.to(out.device).unsqueeze(-1), 0)
        for i in batch.missing:
            row = out[i] if real is None else out[i, real[i].to(out.device)]
            self._keep(batch.keys[i], row)
        return out

    def _keep(self, key: str, row: Any) -> None:
        size = row.numel() * row.element_size()
        if self.stats.bytes_stored + size > self.max_bytes:
            self.stats.rows_not_stored += 1
            return
        kept = row.detach().to("cpu", copy=True)
        if self._pin:
            try:
                kept = kept.pin_memory()
            except RuntimeError:  # pinned RAM exhausted: keep it pageable
                pass
        self.store[key] = kept
        self.stats.rows_stored += 1
        self.stats.bytes_stored += size

    # ---------------------------------------------------------- L2L path

    def l2l_keys(self, micro_batches: Sequence[Any]) -> Optional[List[str]]:
        """One key per micro-batch (each one unpadded row ``[1, T]``), or None if not keyable."""
        import torch

        keys = []
        for ids in micro_batches:
            if not torch.is_tensor(ids) or ids.ndim != 2 or ids.shape[0] != 1:
                return None
            keys.append(_row_key(ids.shape[1], ids[0].detach().to("cpu", torch.int64)))
        return keys

    def l2l_lookup(self, keys: Sequence[str], device: Any) -> Optional[List[Any]]:
        """The kept bottom output of every micro-batch (``[1, T, H]`` on ``device``), or None.

        None unless every micro-batch is kept: a window either skips the bottom for all of its
        micro-batches or computes it for all of them, since each layer is loaded once per window.
        """
        self._check_versions()
        if any(k not in self.store for k in keys):
            self.stats.l2l_misses += 1
            return None
        self.stats.l2l_hits += 1
        non_blocking = getattr(device, "type", str(device)) == "cuda"
        return [self.store[k].to(device, non_blocking=non_blocking).unsqueeze(0) for k in keys]

    def l2l_store(self, keys: Sequence[str], hidden: Sequence[Any]) -> None:
        """Keep each micro-batch's output of layer ``cutoff-1`` that is not kept yet."""
        for key, h in zip(keys, hidden):
            if key not in self.store:
                self._keep(key, h[0])
