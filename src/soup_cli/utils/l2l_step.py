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
* Trainable embedding adapters get a final one-micro-batch recompute and VJP after layer 0,
  through the same streamed embedding wrapper. Its RNG is restored for noise hooks; the
  large slot is not refilled between that recompute and its immediate backward.
* The backward turns the wrapper's checkpoint off for its per-(layer,
  micro-batch) forward+backward, which would otherwise pay a third forward.
* The top-level forward is re-implemented for Llama/Mistral and DeepSeek-V3:
  ``embed_tokens`` -> layers with the kwargs HF passed to layer 0 -> ``norm`` ->
  ``lm_head(h[:, slice(0, None), :])`` -> the model's own ``loss_function``
  with ``num_items_in_batch``. DeepSeek-V3's top-level forward is that same sequence (it
  adds no auxiliary routing loss). ``capture_layer_call`` records the layer kwargs from a
  real forward instead of rebuilding the rotary embeddings and the mask.

* With ``lora.dropout`` > 0 the step replays the random stream of plain gradient
  accumulation: every decoder and head call runs at the generator state accumulation
  reaches for that micro-batch, including head draws between consecutive micro-batches.
  Recompute restores the same states without advancing the outer stream. See ``l2l_rng.py``.

* A layer with streamed routed experts (a per-expert cache, ``utils/expert_stream.py``) reads
  every hit expert through the one-slot arena in the no-grad forward, again in the grad
  recompute and again in the expert's own backward; ``LoadCounters`` counts those reads. The
  router draws no randomness, so the replay above needs nothing more for it, and
  ``RoutingGuard`` checks that the recompute routes like the forward.

No optimizer step. torch is imported inside functions.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

#: Architectures whose top-level forward this step re-implements. What ``training.l2l`` admits
#: at launch is narrower: see ``trainer.l2l_trainer.PUBLIC_ARCHS``.
SUPPORTED_ARCHS = ("llama", "mistral", "deepseek_v3")


@dataclass(frozen=True)
class LayerCall:
    """What HF passes to every decoder layer after ``hidden_states``."""

    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    #: Per layer, the dropouts one call draws (``l2l_rng.Draw``); empty without dropout.
    draws: Tuple[Tuple[Any, ...], ...] = ()
    #: Dropouts the native output head draws after the decoder layers of each micro-batch.
    head_draws: Tuple[Any, ...] = ()


#: ``LoadCounters`` field -> the ``runtime.stats()`` key whose growth it counts.
_STAT_KEYS = {
    "layer_loads": "layer_loads",
    "large_loads": "large_loads",
    "restores": "layer_restores",
    "expert_fetches": "expert_fetches",
    "expert_bytes_read": "expert_bytes_read",
}


@dataclass
class LoadCounters:
    """Loads and weight restores over a span, read from the runtime's own counters.

    ``layer_loads`` and ``large_loads`` are the pools' ``loads``; ``restores`` is the
    NF4 rebuilds of ``Params4bit`` views (always 0 without quantization). ``expert_fetches``
    is the reads of one routed expert into the expert arena, whether the forward, the
    recompute or the expert's own backward asked for it, and ``expert_bytes_read`` the stored
    bytes those reads took; both are 0 without a per-expert cache. Nothing is patched: the
    runtime already counts, this reads the difference.
    """

    layer_loads: int = 0
    large_loads: int = 0
    restores: int = 0
    expert_fetches: int = 0
    expert_bytes_read: int = 0

    def add(self, other: "LoadCounters") -> None:
        for name in _STAT_KEYS:
            setattr(self, name, getattr(self, name) + getattr(other, name))


def _counter_values(runtime: Any) -> Dict[str, int]:
    stats = runtime.stats()
    return {name: int(stats[key]) for name, key in _STAT_KEYS.items()}


@contextmanager
def count_loads(runtime: Any) -> Iterator[LoadCounters]:
    """Yield a ``LoadCounters`` that holds the span's deltas once the block exits."""
    counters = LoadCounters()
    before = _counter_values(runtime)
    try:
        yield counters
    finally:
        for name, value in _counter_values(runtime).items():
            setattr(counters, name, value - before[name])


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
    """Run one forward without grad and record what layer 0 receives.

    The random state is restored afterwards, so the capture draws nothing from it. The same
    forward records which dropouts each decoder layer and the native output head draw.
    """
    import torch

    from soup_cli.utils.l2l_rng import record_draws

    causal, owner = model_parts(model)
    seen: Dict[str, Any] = {}

    def hook(_module: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        if "call" not in seen:
            seen["call"] = (tuple(args[1:]), dict(kwargs))

    handle = owner.layers[0].register_forward_pre_hook(hook, with_kwargs=True)
    # The capture forward runs in training mode, so NEFTune would draw noise from it.
    cuda = [input_ids.device] if input_ids.device.type == "cuda" else []
    try:
        with torch.no_grad(), torch.random.fork_rng(devices=cuda):
            with record_draws([*owner.layers, causal.lm_head]) as plan:
                model(input_ids=input_ids)
    finally:
        handle.remove()
    if "call" not in seen:
        raise RuntimeError("decoder layer 0 was never called by the forward")
    args, kwargs = seen["call"]
    kwargs.pop("num_items_in_batch", None)
    return LayerCall(
        args=args, kwargs=kwargs, draws=tuple(tuple(layer) for layer in plan[:-1]),
        head_draws=tuple(plan[-1]),
    )


class RoutingGuard:
    """Checks that the recompute of each (layer, micro-batch) routes tokens like its forward.

    The no-grad forward and the grad recompute of a layer run the router on the same stored
    activation, so they choose the same experts as long as the router's weights, its
    correction bias and its arithmetic stay put in between. Nothing else verifies that, and a
    recompute that chose other experts would backpropagate through experts the forward never
    ran. Forward pre-hooks on the layers' streamed experts modules, alive for one
    :meth:`L2LStep.run`, record each call's ``top_k_index`` on the host while a pair is
    forwarded and compare it with ``torch.equal`` while the pair is recomputed, before its
    experts are read and before its backward. A pair's record is freed as soon as it has
    been compared, and ``close`` drops what a failed step left. A layer without streamed
    experts (a dense model, a fused cache) gets no hook, so nothing is recorded for it.
    """

    def __init__(self, layers: Sequence[Any]) -> None:
        from soup_cli.utils.expert_stream import streamed_experts_class

        experts = streamed_experts_class()
        self._records: Dict[Tuple[int, int], Any] = {}
        #: (layer, micro-batch, recording) inside ``record`` or ``check``; ``None`` otherwise.
        self._visit: Optional[Tuple[int, int, bool]] = None
        self._handles: List[Any] = []
        for layer in layers:
            for module in layer.modules():
                if isinstance(module, experts):
                    self._handles.append(
                        module.register_forward_pre_hook(self._watch, with_kwargs=True)
                    )

    def _watch(self, _module: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        import torch

        if self._visit is None:
            return
        layer, mb, recording = self._visit
        routed = (kwargs["top_k_index"] if "top_k_index" in kwargs else args[1]).detach()
        pair = (layer, mb)
        if recording:
            self._records[pair] = routed.to("cpu", copy=True)
            return
        before = self._records.pop(pair, None)
        if before is None:
            raise RuntimeError(
                f"routing guard: the recompute of layer {layer} micro-batch {mb} called its "
                "streamed experts more often than the forward did"
            )
        now = routed.to("cpu")
        if not torch.equal(before, now):
            raise RuntimeError(
                f"routing guard: layer {layer} micro-batch {mb} was routed differently in the "
                f"recompute than in the forward ({_routing_difference(before, now)}), so its "
                "gradients would not be those of the forward. The router's weights, its "
                "correction bias or its input changed between the two."
            )

    @contextmanager
    def record(self, layer: int, mb: int) -> Iterator[None]:
        """Record the routing of the streamed experts called inside the block (the forward)."""
        if not self._handles:
            yield
            return
        self._visit = (layer, mb, True)
        try:
            yield
        except BaseException:
            self._records.pop((layer, mb), None)
            raise
        finally:
            self._visit = None

    @contextmanager
    def check(self, layer: int, mb: int) -> Iterator[None]:
        """Compare the routing of the streamed experts called inside the block with the
        forward's (the recompute).

        Raises ``RuntimeError`` naming the layer and micro-batch on a difference. The pair's
        record is freed on success and on exceptions, without hiding the original exception.
        """
        if not self._handles:
            yield
            return
        self._visit = (layer, mb, False)
        try:
            yield
            if (layer, mb) in self._records:
                raise RuntimeError(
                    f"routing guard: the forward of layer {layer} micro-batch {mb} called its "
                    "streamed experts more often than the recompute did"
                )
        finally:
            self._records.pop((layer, mb), None)
            self._visit = None

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._records.clear()
        self._visit = None

    def __enter__(self) -> "RoutingGuard":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def _routing_difference(before: Any, now: Any) -> str:
    if before.shape != now.shape:
        return f"routing went from shape {tuple(before.shape)} to {tuple(now.shape)}"
    changed = int((before != now).any(dim=-1).sum())
    return f"{changed} of {before.shape[0]} tokens chose other experts"


class L2LStep:
    """One layer-major step: forward, head, backward. No optimizer step."""

    def __init__(
        self,
        model: Any,
        runtime: Any,
        call: LayerCall,
    ) -> None:
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

    def _layer(self, module: Any, hidden: Any, state: Any) -> Any:
        # ``state`` is where accumulation's stream stands before this (micro-batch, layer)
        # pair, so the forward and the recompute in the backward draw the same mask in any
        # visiting order. ``None`` without dropout.
        from soup_cli.utils import l2l_rng

        with l2l_rng.state_at(state) if state is not None else nullcontext():
            out = module(hidden, *self.call.args, **self.call.kwargs)
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
        with count_loads(self.runtime) as counters, RoutingGuard(self.layers) as guard:
            losses = self._run(micro_batches, store, n_items, labels, guard)
        self.counters = counters
        return losses

    def _run(
        self,
        micro_batches: Sequence[Any],
        store: Any,
        n_items: Optional[int],
        labels: Optional[Sequence[Any]],
        guard: RoutingGuard,
    ) -> List[Any]:
        import torch

        from soup_cli.utils import l2l_rng

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

        replayed = None
        plan = (*self.call.draws, self.call.head_draws)
        if l2l_rng.has_draws(plan):
            replayed = l2l_rng.replay(plan, k)
        states = replayed[0] if replayed else None
        train_embedding = any(param.requires_grad for param in self.owner.embed_tokens.parameters())
        embedding_states: List[Any] = []

        self._prime(self.owner, ())
        with torch.no_grad():
            hidden = []
            for ids in micro_batches:
                if train_embedding:
                    embedding_states.append(l2l_rng.snapshot())
                hidden.append(self.owner.embed_tokens(ids))
            for idx, layer in enumerate(self.layers):
                for mb in range(k):
                    store.put(idx, mb, hidden[mb])
                    with guard.record(idx, mb):
                        hidden[mb] = self._layer(
                            layer, hidden[mb], states[mb][idx] if states else None
                        )

        losses: List[Any] = []
        grads: List[Any] = []
        for mb in range(k):
            last = hidden[mb].detach().requires_grad_(True)
            hidden[mb] = None
            head_state = states[mb][-1] if states and self.call.head_draws else None
            with (
                torch.enable_grad(),
                l2l_rng.state_at(head_state) if head_state is not None else nullcontext(),
            ):
                normed = self.owner.norm(last)
                logits = self.causal.lm_head(normed[:, slice(0, None), :])
                loss = self.causal.loss_function(
                    logits=logits, labels=labels[mb], vocab_size=vocab, num_items_in_batch=total
                )
            # HF's Trainer leaves autocast after the forward, before backward. The trainer
            # calls this whole step inside autocast, so suspend it for the actual VJP only.
            with torch.autocast("cpu", enabled=False), torch.autocast("cuda", enabled=False):
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
                    with torch.enable_grad(), guard.check(idx, mb):
                        out = self._layer(
                            layer, inputs, states[mb][idx] if states else None
                        )
                    with (
                        torch.autocast("cpu", enabled=False),
                        torch.autocast("cuda", enabled=False),
                    ):
                        torch.autograd.backward(out, grad_tensors=grads[mb])
                    grads[mb] = inputs.grad
            finally:
                layer.use_checkpoint = saved
        if train_embedding:
            # The reverse walk normally prefetches this at layer 0. Explicitly request the
            # same load for a one-layer decoder too, whose walk never changes direction.
            refill = self.runtime.prefetcher.backward_tail_prefetch
            if refill is not None:
                refill()
            for mb, ids in enumerate(micro_batches):
                with torch.enable_grad(), l2l_rng.state_at(embedding_states[mb]):
                    embedded = self.owner.embed_tokens(ids)
                with (
                    torch.autocast("cpu", enabled=False),
                    torch.autocast("cuda", enabled=False),
                ):
                    torch.autograd.backward(embedded, grad_tensors=grads[mb])
                grads[mb] = None
        store.end_step()
        if replayed is not None:
            l2l_rng.leave_stream_at(replayed[1])
        return losses
