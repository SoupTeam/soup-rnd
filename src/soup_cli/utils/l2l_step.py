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

* A preference loss with a reference model passes one or more ``ReferencePass``es: after each
  layer's policy micro-batches the same loaded layer runs the k micro-batches again, without
  grad and without writing to the activation store, then the pass's head turns each last
  hidden state into log-probabilities. A pass runs under a named adapter (DPO's frozen
  ``ref``), with every adapter off (KTO's reference) or under the policy as it is (KTO's KL
  estimate), and walks either the policy's ids or ids of its own (KTO's KL completions have
  their own width). It adds no layer load: the extra calls land on a layer the pool already
  owns.
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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

#: Architectures whose top-level forward this step re-implements. ``training.l2l`` admits them
#: at launch under the rules in ``trainer.l2l_trainer.refuse_unsupported_l2l``.
SUPPORTED_ARCHS = ("llama", "mistral", "deepseek_v3", "qwen3_5", "qwen3_5_text")


@dataclass(frozen=True)
class LayerCall:
    """What HF passes to every decoder layer after ``hidden_states``."""

    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    #: Per layer, the dropouts one call draws (``l2l_rng.Draw``); empty without dropout.
    draws: Tuple[Tuple[Any, ...], ...] = ()
    #: Dropouts the native output head draws after the decoder layers of each micro-batch.
    head_draws: Tuple[Any, ...] = ()
    #: Hybrid decoders (``qwen3_5``: Gated DeltaNet + gated attention) pass each layer type
    #: different arguments. ``layer_types[i]`` names layer ``i``'s type and ``typed`` holds the
    #: ``(args, kwargs)`` captured at the first layer of each type; empty for uniform decoders.
    layer_types: Tuple[str, ...] = ()
    typed: Dict[str, Tuple[Tuple[Any, ...], Dict[str, Any]]] = field(default_factory=dict)

    def for_layer(self, idx: int) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """The ``(args, kwargs)`` HF passes to decoder layer ``idx``."""
        if self.typed and self.layer_types:
            return self.typed[self.layer_types[idx]]
        return self.args, self.kwargs


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


#: ``(causal, owner, hidden, mb) -> loss``. ``hidden`` is micro-batch ``mb``'s output of the
#: last decoder layer, a leaf that requires grad. The head runs ``norm``, ``lm_head`` and the
#: loss, and returns the scalar the step backpropagates, already scaled for the window.
LossHead = Callable[[Any, Any, Any, int], Any]


def sft_head(labels: Sequence[Any], n_items: int) -> LossHead:
    """The SFT head: ``norm``, ``lm_head``, the model's ``loss_function`` over ``n_items``."""

    def head(causal: Any, owner: Any, hidden: Any, mb: int) -> Any:
        normed = owner.norm(hidden)
        logits = causal.lm_head(normed[:, slice(0, None), :])
        return causal.loss_function(
            logits=logits,
            labels=labels[mb],
            vocab_size=int(causal.config.vocab_size),
            num_items_in_batch=n_items,
        )

    return head


@dataclass
class ReferencePass:
    """A gradient-free pass over the same layers, after each layer's policy micro-batches.

    ``head`` has the shape of a :data:`LossHead`; the step calls it under ``no_grad``, inside
    the pass's adapter state, on the last hidden state of micro-batch ``mb`` and stores the
    result in ``outputs[mb]``. The policy head reads ``outputs`` after the step has filled it.

    The adapter state is one of: ``adapter`` (a PEFT adapter name, TRL's ``ref`` for DPO)
    active for the pass; ``adapters_off`` (every adapter switched off, TRL's
    ``null_ref_context`` for KTO); neither, the policy as it stands.

    ``micro_batches`` are the ids the pass walks; without them it walks the policy's. Ids of
    their own need one :class:`LayerCall` per micro-batch in ``calls``, captured at that
    micro-batch's shape.
    """

    head: LossHead
    adapter: Optional[str] = None
    adapters_off: bool = False
    micro_batches: Optional[Sequence[Any]] = None
    calls: Optional[Sequence[LayerCall]] = None
    outputs: Dict[int, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.adapter is not None and self.adapters_off:
            raise ValueError(
                "a reference pass names an adapter or switches the adapters off, not both: "
                f"adapter={self.adapter!r}, adapters_off=True"
            )


class _CapturedError(Exception):
    """Raised from layer 0's pre-hook to end a capture forward before any layer runs."""


def capture_layer_call(model: Any, input_ids: Any, *, draws: bool = True) -> LayerCall:
    """Run one forward without grad and record what layer 0 receives.

    The random state is restored afterwards, so the capture draws nothing from it.

    With ``draws`` (the default) the forward runs through every layer and the native output
    head and records which dropouts each draws, for the replay in :meth:`L2LStep.run`; on a
    streamed model that walk loads every layer once. With ``draws=False`` the forward stops at
    layer 0's pre-hook, which already holds the arguments, so no layer loads or runs. Use it
    when the model draws no dropout; the returned call has no draws.
    """
    import torch

    from soup_cli.utils.l2l_rng import record_draws

    causal, owner = model_parts(model)
    layer_types = _layer_types(causal, len(owner.layers))
    # The first layer of each type; a uniform decoder has one, layer 0.
    firsts: Dict[str, int] = {}
    for i, kind in enumerate(layer_types or ("layer",)):
        firsts.setdefault(kind, i)
    seen: Dict[str, Any] = {}

    def make_hook(kind: str) -> Callable[..., None]:
        def hook(_module: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
            if kind not in seen:
                seen[kind] = (tuple(args[1:]), dict(kwargs))
                if not draws and len(seen) == len(firsts):
                    raise _CapturedError

        return hook

    handles = [
        owner.layers[i].register_forward_pre_hook(make_hook(kind), with_kwargs=True)
        for kind, i in firsts.items()
    ]
    # The capture forward runs in training mode, so NEFTune would draw noise from it.
    cuda = [input_ids.device] if input_ids.device.type == "cuda" else []
    plan: List[List[Any]] = []
    try:
        with torch.no_grad(), torch.random.fork_rng(devices=cuda):
            if draws:
                with record_draws([*owner.layers, causal.lm_head]) as plan:
                    model(input_ids=input_ids)
            else:
                try:
                    model(input_ids=input_ids)
                except _CapturedError:
                    pass
    finally:
        for handle in handles:
            handle.remove()
    missing = [kind for kind in firsts if kind not in seen]
    if missing:
        layers = [firsts[k] for k in missing]
        raise RuntimeError(f"decoder layer(s) {layers} never called by the forward")
    for _args, kwargs in seen.values():
        kwargs.pop("num_items_in_batch", None)
    first_kind = (layer_types or ("layer",))[0]
    args, kwargs = seen[first_kind]
    typed = dict(seen) if layer_types and len(firsts) > 1 else {}
    return LayerCall(
        args=args,
        kwargs=kwargs,
        draws=tuple(tuple(layer) for layer in plan[:-1]),
        head_draws=tuple(plan[-1]) if plan else (),
        layer_types=tuple(layer_types) if typed else (),
        typed=typed,
    )


def _layer_types(causal: Any, n_layers: int) -> Tuple[str, ...]:
    """``config.layer_types`` of a hybrid decoder (text config for a VLM wrapper), else ()."""
    config = getattr(causal, "config", None)
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    types = tuple(getattr(text, "layer_types", None) or ())
    if types and len(types) < n_layers:
        raise ValueError(
            f"config.layer_types names {len(types)} layers, the decoder has {n_layers}"
        )
    return types[:n_layers]


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


def as_passes(
    reference: Union[ReferencePass, Sequence[ReferencePass], None],
) -> List[ReferencePass]:
    """The extra passes of a window as a list: none, one pass, or the given sequence."""
    if reference is None:
        return []
    if isinstance(reference, ReferencePass):
        return [reference]
    return list(reference)


def _pass_micro_batches(spec: ReferencePass, policy_ids: Sequence[Any], k: int) -> Sequence[Any]:
    """The ids a pass walks: its own, checked against ``k``, or the policy's."""
    if spec.micro_batches is None:
        return policy_ids
    if len(spec.micro_batches) != k:
        raise ValueError(
            f"a reference pass has {len(spec.micro_batches)} micro-batches for a window of {k}"
        )
    return spec.micro_batches


def _pass_calls(spec: ReferencePass, policy_calls: Sequence[LayerCall], k: int) -> Sequence[Any]:
    """The layer call of each micro-batch a pass walks; ids of its own bring their own."""
    if spec.micro_batches is None:
        return policy_calls
    if spec.calls is None or len(spec.calls) != k:
        raise ValueError(
            "a reference pass over ids of its own needs one layer call per micro-batch "
            "(captured at that micro-batch's shape); got "
            f"{None if spec.calls is None else len(spec.calls)} layer calls for {k}"
        )
    return spec.calls


class L2LStep:
    """One layer-major step: forward, head, backward. No optimizer step."""

    def __init__(
        self,
        model: Any,
        runtime: Any,
        call: LayerCall,
    ) -> None:
        self.causal, self.owner = model_parts(model)
        self.model = model
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

    def _layer(self, module: Any, hidden: Any, state: Any, call: LayerCall, idx: int = 0) -> Any:
        # ``state`` is where accumulation's stream stands before this (micro-batch, layer)
        # pair, so the forward and the recompute in the backward draw the same mask in any
        # visiting order. ``None`` without dropout.
        from soup_cli.utils import l2l_rng

        with l2l_rng.state_at(state) if state is not None else nullcontext():
            args, kwargs = call.for_layer(idx)
            out = module(hidden, *args, **kwargs)
        return out[0] if isinstance(out, tuple) else out

    @contextmanager
    def _adapter(self, name: str) -> Iterator[None]:
        """The named PEFT adapter active inside, the previous one after.

        The semantics of TRL's ``use_adapter(model, name)``, which the DPO trainer wraps its
        reference forward in. ``set_adapter`` moves ``requires_grad`` to the active adapter
        and back; the reference pass runs under ``no_grad``, so no graph keeps the flag.
        """
        previous = self.model.active_adapter
        self.model.set_adapter(name)
        try:
            yield
        finally:
            self.model.set_adapter(previous)

    def _pass_adapters(self, spec: ReferencePass) -> Any:
        """The adapter state of an extra pass: a named adapter, every adapter off, or none."""
        if spec.adapters_off:
            # ``PeftModel.disable_adapter``: what TRL's ``null_ref_context`` enters for KTO.
            return self.model.disable_adapter()
        if spec.adapter is not None:
            return self._adapter(spec.adapter)
        return nullcontext()

    @staticmethod
    def _backward_context(ids: Any) -> Any:
        """Autocast off around a backward call, the state HF's accumulation runs it in.

        The caller wraps the whole step in autocast, forward and recompute included, but HF
        calls ``backward`` after the autocast block has exited. Autograd runs the gradient
        kernels under the autocast state of the calling thread, so a backward inside the
        block takes other dtype paths than accumulation's and the gradients differ in bf16.
        """
        import torch

        return torch.autocast(device_type=ids.device.type, enabled=False)

    def run(
        self,
        micro_batches: Sequence[Any],
        store: Any,
        *,
        n_items: Optional[int] = None,
        labels: Optional[Sequence[Any]] = None,
        calls: Optional[Sequence[LayerCall]] = None,
        head: Optional[LossHead] = None,
        reference: Union[ReferencePass, Sequence[ReferencePass], None] = None,
        scaler: Any = None,
    ) -> List[Any]:
        """``labels`` defaults to the input ids; pass them for SFT rows with ``-100``.

        ``calls`` is one ``LayerCall`` per micro-batch, captured at that micro-batch's own
        shape; micro-batches of different lengths run unpadded. Without it every
        micro-batch uses the call the step was built with.

        ``head`` replaces the SFT head and loss (see :data:`LossHead`); it owns its labels
        and its scale, so it excludes ``labels`` and ``n_items``.

        ``scaler`` is the Trainer's fp16 ``GradScaler`` (``accelerator.scaler``), or None. The
        head loss is backpropagated as ``scaler.scale(loss)``; the scale then travels down every
        layer through ``grad_tensors``, as it does through accumulation's
        ``accelerator.backward``. Unscaling, clipping and skipping an inf step stay with
        accelerate's optimizer step. The returned losses are unscaled.

        ``reference`` adds the gradient-free passes (one :class:`ReferencePass` or a sequence
        of them). It refuses a model whose layers draw dropout: accumulation's extra forwards
        would draw masks too, and the replay of ``l2l_rng`` covers the policy's draws only.
        """
        if head is not None and (labels is not None or n_items is not None):
            raise ValueError("a custom head owns its labels and scale: pass no labels or n_items")
        with count_loads(self.runtime) as counters, RoutingGuard(self.layers) as guard:
            losses = self._run(
                micro_batches, store, n_items, labels, calls, guard, head, reference, scaler
            )
        self.counters = counters
        return losses

    def _run(
        self,
        micro_batches: Sequence[Any],
        store: Any,
        n_items: Optional[int],
        labels: Optional[Sequence[Any]],
        calls: Optional[Sequence[LayerCall]],
        guard: RoutingGuard,
        head: Optional[LossHead],
        reference: Union[ReferencePass, Sequence[ReferencePass], None] = None,
        scaler: Any = None,
    ) -> List[Any]:
        import torch

        from soup_cli.utils import l2l_rng

        k = len(micro_batches)
        if k < 1:
            raise ValueError("an L2L step needs at least one micro-batch")
        if k > store.k:
            raise ValueError(f"{k} micro-batches, but the activation store holds {store.k}")
        if calls is None:
            calls = [self.call] * k
        elif len(calls) != k:
            raise ValueError(f"{len(calls)} layer calls for {k} micro-batches")
        if head is None:
            if labels is None:
                labels = micro_batches
            elif len(labels) != k:
                raise ValueError(f"{len(labels)} label tensors for {k} micro-batches")
            total = label_tokens(micro_batches) if n_items is None else int(n_items)
            head = sft_head(labels, total)

        passes = as_passes(reference)
        if passes and l2l_rng.has_draws(self.call.draws):
            raise ValueError(
                "the L2L reference pass does not support dropout: the reference forward "
                "of gradient accumulation draws dropout masks too, and the replay covers "
                "the policy's draws only. Set attention_dropout to 0."
            )
        pass_ids = [_pass_micro_batches(spec, micro_batches, k) for spec in passes]
        pass_calls = [_pass_calls(spec, calls, k) for spec in passes]
        replayed = None
        plan = (*self.call.draws, self.call.head_draws)
        if l2l_rng.has_draws(plan):
            replayed = l2l_rng.replay(plan, k)
        states = replayed[0] if replayed else None
        train_embedding = any(param.requires_grad for param in self.owner.embed_tokens.parameters())
        embedding_states: List[Any] = []

        self._prime(self.owner, ())
        pass_hidden: List[List[Any]] = []
        with torch.no_grad():
            hidden = []
            for ids in micro_batches:
                if train_embedding:
                    embedding_states.append(l2l_rng.snapshot())
                hidden.append(self.owner.embed_tokens(ids))
            for spec, ids in zip(passes, pass_ids):
                with self._pass_adapters(spec):
                    pass_hidden.append([self.owner.embed_tokens(x) for x in ids])
            for idx, layer in enumerate(self.layers):
                for mb in range(k):
                    store.put(idx, mb, hidden[mb])
                    with guard.record(idx, mb):
                        hidden[mb] = self._layer(
                            layer, hidden[mb], states[mb][idx] if states else None, calls[mb],
                            idx,
                        )
                for spec, current, spec_calls in zip(passes, pass_hidden, pass_calls):
                    # Same loaded layer: nothing is stored, nothing loads. Outside
                    # ``guard.record``, so the routing guard sees the policy forward only.
                    with self._pass_adapters(spec):
                        for mb in range(k):
                            current[mb] = self._layer(
                                layer, current[mb], None, spec_calls[mb], idx
                            )
            for spec, current in zip(passes, pass_hidden):
                spec.outputs.clear()
                with self._pass_adapters(spec):
                    for mb in range(k):
                        spec.outputs[mb] = spec.head(self.causal, self.owner, current[mb], mb)
                        current[mb] = None

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
                loss = head(self.causal, self.owner, last, mb)
            with self._backward_context(micro_batches[mb]):
                (loss if scaler is None else scaler.scale(loss)).backward()
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
                            layer, inputs, states[mb][idx] if states else None, calls[mb], idx
                        )
                    with self._backward_context(micro_batches[mb]):
                        torch.autograd.backward(out, grad_tensors=grads[mb])
                    grads[mb] = inputs.grad
            finally:
                layer.use_checkpoint = saved
        if train_embedding:
            # The reverse walk prefetches this at layer 0 and records it on the prefetcher. A
            # one-layer decoder never changes direction, so only then is it still to be asked.
            prefetcher = self.runtime.prefetcher
            refill = prefetcher.backward_tail_prefetch
            if refill is not None and not prefetcher.backward_tail_prefetched:
                refill()
                prefetcher.backward_tail_prefetched = True
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
