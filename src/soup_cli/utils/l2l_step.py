"""One layer-major (L2L) step over the streamed model.

L2L (Pudipeddi et al., arXiv 2002.05645): load a decoder layer once per step, run all
k micro-batches through it, park each micro-batch's boundary activation in host RAM,
and walk the backward the same way in reverse.

* ``StreamedDecoderLayer._body`` calls ``pool.wait(idx)`` and then
  ``prefetcher.advance(idx)``. A second call on the same index leaves the
  direction alone and finds the next layer already owned, so k calls on a loaded
  layer load nothing. The load sequence (forward 0..L-1, then L-1..0) is the
  streamed step's, and ``AsyncDiskSource`` infers its direction from that order.
* The decoder container's forward pre-hook (``runtime.hook``, the shipped ``_prime``)
  is called directly, because the step bypasses ``LlamaModel.forward``.
* All k embedding lookups run before layer 0's first ``advance`` queues the head
  into the shared large slot; ``LargeLayerBufferPool.load_async`` orders that
  refill after them on the compute stream.
* The backward turns the wrapper's checkpoint off for its per-(layer,
  micro-batch) forward+backward, which would otherwise pay a third forward.
* The top-level forward is re-implemented for Llama/Mistral only:
  ``embed_tokens`` -> layers with the kwargs HF passed to layer 0 -> ``norm`` ->
  ``lm_head(h[:, slice(0, None), :])`` -> the model's own ``loss_function``
  with ``num_items_in_batch``. ``capture_layer_call`` records the layer kwargs
  from a real forward instead of rebuilding the rotary embeddings and the mask.

No optimizer step. torch is imported inside functions.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

SUPPORTED_ARCHS = ("llama", "mistral")


@dataclass(frozen=True)
class LayerCall:
    """What HF passes to every decoder layer after ``hidden_states``."""

    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]


@dataclass
class LoadCounters:
    """Layer loads and weight restores over a span, read from the runtime's own counters.

    ``layer_loads`` and ``large_loads`` are the pools' ``loads``; ``restores`` is the
    NF4 rebuilds of ``Params4bit`` views (always 0 without quantization). Nothing is
    patched: the runtime already counts, this reads the difference.
    """

    layer_loads: int = 0
    large_loads: int = 0
    restores: int = 0

    def add(self, other: "LoadCounters") -> None:
        self.layer_loads += other.layer_loads
        self.large_loads += other.large_loads
        self.restores += other.restores


def _counter_values(runtime: Any) -> Tuple[int, int, int]:
    stats = runtime.stats()
    return stats["layer_loads"], stats["large_loads"], stats["layer_restores"]


@contextmanager
def count_loads(runtime: Any) -> Iterator[LoadCounters]:
    """Yield a ``LoadCounters`` that holds the span's deltas once the block exits."""
    counters = LoadCounters()
    before = _counter_values(runtime)
    try:
        yield counters
    finally:
        after = _counter_values(runtime)
        counters.layer_loads, counters.large_loads, counters.restores = (
            b - a for a, b in zip(before, after)
        )


def model_parts(model: Any) -> Tuple[Any, Any]:
    """(the CausalLM under any PEFT wrapper, the container that owns ``.layers``)."""
    from soup_cli.utils.layer_stream_runtime import decoder_owner

    owner = decoder_owner(model)
    causal = model.get_base_model() if hasattr(model, "get_base_model") else model
    arch = str(getattr(getattr(causal, "config", None), "model_type", ""))
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(
            f"the L2L step re-implements the top-level forward of {SUPPORTED_ARCHS} only; "
            f"got {arch!r}"
        )
    for name in ("lm_head", "loss_function"):
        if getattr(causal, name, None) is None:
            raise ValueError(f"the causal LM has no {name!r}")
    return causal, owner


def label_tokens(micro_batches: Sequence[Any]) -> int:
    """HF's ``num_items_in_batch`` when labels == input_ids: seq - 1 per row."""
    return sum(int(ids.shape[0]) * (int(ids.shape[1]) - 1) for ids in micro_batches)


def capture_layer_call(model: Any, input_ids: Any) -> LayerCall:
    """Run one forward without grad and record what layer 0 receives."""
    import torch

    _causal, owner = model_parts(model)
    seen: Dict[str, Any] = {}

    def hook(_module: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        if "call" not in seen:
            seen["call"] = (tuple(args[1:]), dict(kwargs))

    handle = owner.layers[0].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        handle.remove()
    if "call" not in seen:
        raise RuntimeError("decoder layer 0 was never called by the forward")
    args, kwargs = seen["call"]
    kwargs.pop("num_items_in_batch", None)
    return LayerCall(args=args, kwargs=kwargs)


class L2LStep:
    """One layer-major step: forward, head, backward. No optimizer step."""

    def __init__(self, model: Any, runtime: Any, call: LayerCall):
        self.causal, self.owner = model_parts(model)
        self.layers = list(self.owner.layers)
        self.call = call
        hook = getattr(runtime, "hook", None)
        prime = self.owner._forward_pre_hooks.get(getattr(hook, "id", None))
        if prime is None:
            raise RuntimeError(
                "the streaming prime hook is not installed on the decoder container; "
                "the L2L step cannot load the embedding and prime the prefetcher"
            )
        self._prime = prime
        self.runtime = runtime
        #: Loads and restores of the last ``run``; ``None`` before the first.
        self.counters: Optional[LoadCounters] = None

    def _layer(self, layer: Any, hidden: Any) -> Any:
        out = layer(hidden, *self.call.args, **self.call.kwargs)
        return out[0] if isinstance(out, tuple) else out

    def run(
        self,
        micro_batches: Sequence[Any],
        store: Any,
        *,
        n_items: Optional[int] = None,
        labels: Optional[Sequence[Any]] = None,
    ) -> List[Any]:
        """``labels`` defaults to the input ids; pass them for SFT rows with ``-100``."""
        with count_loads(self.runtime) as counters:
            losses = self._run(micro_batches, store, n_items, labels)
        self.counters = counters
        return losses

    def _run(
        self,
        micro_batches: Sequence[Any],
        store: Any,
        n_items: Optional[int],
        labels: Optional[Sequence[Any]],
    ) -> List[Any]:
        import torch

        k = len(micro_batches)
        if k < 1:
            raise ValueError("an L2L step needs at least one micro-batch")
        if k > store.k:
            raise ValueError(f"{k} micro-batches, but the activation store holds {store.k}")
        if labels is None:
            labels = micro_batches
        elif len(labels) != k:
            raise ValueError(f"{len(labels)} label tensors for {k} micro-batches")
        total = label_tokens(micro_batches) if n_items is None else int(n_items)
        vocab = int(self.causal.config.vocab_size)

        self._prime(self.owner, ())
        with torch.no_grad():
            hidden = [self.owner.embed_tokens(ids) for ids in micro_batches]
            for idx, layer in enumerate(self.layers):
                for mb in range(k):
                    store.put(idx, mb, hidden[mb])
                    hidden[mb] = self._layer(layer, hidden[mb])

        losses: List[Any] = []
        grads: List[Any] = []
        for mb in range(k):
            last = hidden[mb].detach().requires_grad_(True)
            hidden[mb] = None
            with torch.enable_grad():
                normed = self.owner.norm(last)
                logits = self.causal.lm_head(normed[:, slice(0, None), :])
                loss = self.causal.loss_function(
                    logits=logits, labels=labels[mb], vocab_size=vocab, num_items_in_batch=total
                )
            loss.backward()
            losses.append(loss.detach())
            grads.append(last.grad)

        store.begin_backward()
        for idx in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[idx]
            saved = layer.use_checkpoint
            layer.use_checkpoint = False
            try:
                for mb in range(k):
                    inputs = store.get(idx, mb).requires_grad_(True)
                    with torch.enable_grad():
                        out = self._layer(layer, inputs)
                    torch.autograd.backward(out, grad_tensors=grads[mb])
                    grads[mb] = inputs.grad
            finally:
                layer.use_checkpoint = saved
        store.end_step()
        return losses
