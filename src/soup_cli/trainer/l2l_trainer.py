"""SFT trainer mode for L2L (``training.l2l``, ADR 0002).

HF's loop prefetches ``gradient_accumulation_steps`` batches per optimizer step
(``Trainer.get_batch_samples``) and then calls ``training_step`` once per batch. The
subclass keeps that loop and everything it owns (``num_items_in_batch``, lr schedule,
logging, checkpoints, resume) and changes two calls:

* ``get_batch_samples`` remembers the window of k batches HF just built;
* ``training_step`` runs one layer-major step over the whole window on the window's
  first call and returns the summed loss. The other k-1 calls of the window return a
  zero and compute nothing, so HF's accumulation of the returned losses and its one
  optimizer step per window stay as they are.

Each micro-batch runs at its own length, as in gradient accumulation: nothing is padded.
The layer arguments (position ids, mask) are captured once per distinct shape and each
micro-batch gets the call of its shape. The activation store sizes every micro-batch's
chunks to that micro-batch, so a short row does not pay for the longest row of its window.

torch is imported inside functions.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from soup_cli.utils.l2l_step import SUPPORTED_ARCHS, LoadCounters

#: Captured layer arguments kept per shape; the cache restarts when a run exceeds it. A new
#: shape costs one forward to capture them.
MAX_CACHED_SHAPES = 64


def refuse_unsupported_l2l(
    model_config: Any, *, n_gpu: int, fp16: bool = False
) -> None:
    """Refuse at launch what the schema cannot see: the model, the GPU count, fp16."""
    arch = str(getattr(model_config, "model_type", ""))
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(
            f"training.l2l does not support model_type {arch!r}: the L2L step "
            f"re-implements the top-level forward of {' and '.join(SUPPORTED_ARCHS)} only."
        )
    if n_gpu > 1:
        raise ValueError(
            f"training.l2l is single-GPU, but {n_gpu} GPUs are visible. Re-run with "
            f"one card visible, e.g. CUDA_VISIBLE_DEVICES=0."
        )
    if fp16:
        raise ValueError(
            "training.l2l does not support fp16: the L2L step runs backward itself, "
            "so the fp16 grad scaler would never scale the loss. Use bf16 or fp32."
        )


@dataclass
class L2LState:
    """What the trainer subclass needs besides the HF trainer."""

    runtime: Any
    dtype: Any
    window: Optional[List[Any]] = None
    position: int = 0
    steps: int = 0
    has_dropout: Optional[bool] = None
    totals: LoadCounters = field(default_factory=LoadCounters)
    store: Any = None
    stepper: Any = None
    calls_by_shape: Dict[Tuple[int, int], Any] = field(default_factory=dict)
    #: Capture forwards run, the layer loads they cost and their wall time (running totals).
    captures: int = 0
    capture_loads: int = 0
    capture_seconds: float = 0.0

    def log_fields(self) -> Dict[str, Any]:
        return {
            "l2l_steps": self.steps,
            "l2l_captures": self.captures,
            "l2l_capture_layer_loads": self.capture_loads,
            "l2l_capture_seconds": self.capture_seconds,
            "l2l_layer_loads": self.totals.layer_loads,
            "l2l_large_loads": self.totals.large_loads,
            "l2l_weight_restores": self.totals.restores,
        }

    def close(self) -> None:
        if self.store is not None:
            self.store.close()
            self.store = None


def _window_tensors(
    window: List[Any], device: Any, *, neftune: bool = False, has_dropout: bool = False
) -> Tuple[List[Any], List[Any]]:
    """Input ids and labels of each micro-batch, each at its own shape.

    A window whose micro-batches differ in shape is refused with NEFTune or dropout on.
    Dropout: ``l2l_rng.replay`` builds the generator states from one draw plan shared by
    all micro-batches, and rows of different lengths draw different masks. NEFTune: the
    noise scales with ``1 / sqrt(seq * hidden)`` and the combination has not been checked
    against gradient accumulation on rows of different lengths.
    """
    ids_list, label_list = [], []
    for inputs in window:
        ids = inputs["input_ids"].to(device)
        labels = inputs.get("labels")
        labels = ids if labels is None else labels.to(device)
        mask = inputs.get("attention_mask")
        if mask is not None:
            mask = mask.to(device)
            if bool((mask[:, 1:] > mask[:, :-1]).any()):
                raise ValueError(
                    "training.l2l needs right-padded rows: a padding token precedes "
                    "a real token in the attention mask."
                )
        ids_list.append(ids)
        label_list.append(labels)
    shapes = {tuple(ids.shape) for ids in ids_list}
    mixed = len(shapes) > 1
    if mixed and neftune:
        raise ValueError(
            "training.l2l with training.neftune_alpha needs every micro-batch of a window "
            f"to have one shape, but this window has {sorted(shapes)}: the L2L step has "
            "not been checked against gradient accumulation for NEFTune noise on rows of "
            "different lengths. Use rows of one length, or turn neftune_alpha off."
        )
    if mixed and has_dropout:
        raise ValueError(
            "training.l2l with dropout (training.lora.dropout or attention_dropout above 0) "
            f"needs every micro-batch of a window to have one shape, but this window has "
            f"{sorted(shapes)}: rows of different lengths draw different dropout masks, and "
            "the replay of l2l_rng uses one draw plan for all micro-batches. Use rows of "
            "one length, or set the dropout to 0."
        )
    return ids_list, label_list


def _has_dropout(model: Any) -> bool:
    """True when a forward in training draws dropout masks: any ``nn.Dropout`` with p > 0
    (LoRA dropout) or a non-zero ``attention_dropout`` on the config or a module."""
    import torch

    config = getattr(model, "config", None)
    if float(getattr(config, "attention_dropout", 0.0) or 0.0) != 0:
        return True
    return any(
        (isinstance(module, torch.nn.Dropout) and module.p > 0)
        or float(getattr(module, "attention_dropout", 0.0) or 0.0) != 0
        for module in model.modules()
    )


@contextmanager
def _timed_capture(state: L2LState) -> Iterator[None]:
    """Add one capture, its layer loads and its wall time to the state's totals."""
    import time

    import torch

    from soup_cli.utils.l2l_step import count_loads

    def sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    sync()
    start = time.perf_counter()
    with count_loads(state.runtime) as counters:
        try:
            yield
        finally:
            sync()
            state.captures += 1
            state.capture_seconds += time.perf_counter() - start
    state.capture_loads += counters.layer_loads


def _l2l_step_for(state: L2LState, model: Any, ids_list: List[Any]) -> Tuple[Any, List[Any]]:
    """The window's L2L step and one captured ``LayerCall`` per micro-batch."""
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    if state.has_dropout is None:
        state.has_dropout = _has_dropout(model)
    calls = []
    for ids in ids_list:
        shape = (int(ids.shape[0]), int(ids.shape[1]))
        call = state.calls_by_shape.get(shape)
        if call is None:
            if len(state.calls_by_shape) >= MAX_CACHED_SHAPES:
                state.calls_by_shape.clear()
            # Only a model with dropout needs the full forward, to record the draws.
            with _timed_capture(state):
                call = capture_layer_call(model, ids, draws=state.has_dropout)
            state.calls_by_shape[shape] = call
        calls.append(call)
    if state.stepper is None:
        state.stepper = L2LStep(model, state.runtime, calls[0])
    return state.stepper, calls


def check_activation_budget(
    *,
    free_ram: Optional[int],
    n_layers: int,
    k: int,
    chunk_bytes: int,
    pinned: bool,
    mb_bytes: Optional[Sequence[int]] = None,
) -> None:
    """Refuse an activation store that would not fit the free host RAM, before allocating it.

    The weight store and staging are already resident when this runs, so ``free_ram``
    is what they left over. A pinned chunk is its own allocation, and torch's host
    allocator rounds it up to the next power of two (#901). ``mb_bytes`` is the chunk
    size of each micro-batch when they differ (rows of different lengths); the store
    then needs the sum of their blocks, not ``k`` times the largest. Without psutil
    (``free_ram`` is None) nothing is checked.
    """
    from soup_cli.utils.layer_stream import RAM_TIER_HEADROOM
    from soup_cli.utils.layer_stream_runtime import _next_power_of_two

    sizes = [chunk_bytes] * k if mb_bytes is None else list(mb_bytes)
    blocks = [_next_power_of_two(size) if pinned else size for size in sizes]
    nbytes = n_layers * sum(blocks)
    if free_ram is None or nbytes < free_ram * RAM_TIER_HEADROOM:
        return
    kind = "pinned " if pinned else ""
    largest, block = max(sizes), max(blocks)
    rounding = (
        f" (a {largest / 1e6:.1f} MB chunk takes a {block / 1e6:.1f} MB pinned block)"
        if block != largest
        else ""
    )
    raise ValueError(
        f"training.l2l keeps {n_layers} layers x {len(sizes)} micro-batches of activations in "
        f"host RAM: {nbytes / 1e9:.2f} GB of {kind}blocks{rounding}, but only "
        f"{free_ram / 1e9:.1f} GB is free and {RAM_TIER_HEADROOM:.0%} of that is usable. "
        f"Lower gradient_accumulation_steps, data.max_length or batch_size, or free RAM."
    )


def _store_for(state: L2LState, stepper: Any, ids_list: List[Any], device: Any) -> Any:
    """The activation store for the window, grown only when a micro-batch does not fit.

    Micro-batch ``i`` of every window uses chunk ``i``, so the store keeps the largest
    size each position has needed. A window that fits is served by the store as it is.
    """
    import math

    import torch

    from soup_cli.utils.l2l_activations import ActivationStore
    from soup_cli.utils.layer_stream import free_ram_bytes

    hidden = int(stepper.causal.config.hidden_size)
    need = [int(ids.shape[0]) * int(ids.shape[1]) * hidden for ids in ids_list]
    have = [math.prod(shape) for shape in state.store.mb_shapes] if state.store else []
    if state.store is not None and len(need) <= len(have) and all(
        n <= h for n, h in zip(need, have)
    ):
        return state.store
    caps = [max(n, h) for n, h in zip(need, have)] + need[len(have):] + have[len(need):]
    state.close()
    n_layers = len(stepper.layers)
    itemsize = torch.empty((), dtype=state.dtype).element_size()
    pin = str(device).startswith("cuda")
    check_activation_budget(
        free_ram=free_ram_bytes(),
        n_layers=n_layers,
        k=len(caps),
        chunk_bytes=max(caps) * itemsize,
        pinned=pin,
        mb_bytes=[cap * itemsize for cap in caps],
    )
    state.store = ActivationStore(
        n_layers=n_layers,
        k=len(caps),
        chunk_shape=(max(caps),),
        mb_shapes=[(cap,) for cap in caps],
        dtype=state.dtype,
        device=device,
        pin=pin,
    )
    return state.store


def run_window(
    state: L2LState,
    model: Any,
    ids_list: List[Any],
    label_list: List[Any],
    n_items: int,
    device: Any,
) -> Tuple[List[Any], Any]:
    """One L2L step over the window; returns the losses and the step's load counters."""
    stepper, calls = _l2l_step_for(state, model, ids_list)
    store = _store_for(state, stepper, ids_list, device)
    losses = stepper.run(ids_list, store, n_items=n_items, labels=label_list, calls=calls)
    return losses, stepper.counters


@functools.lru_cache(maxsize=None)
def make_l2l_trainer_class(base_cls: type) -> type:
    """Return a subclass of ``base_cls`` that runs each accumulation window as one L2L step.

    Cached, so two calls with the same ``base_cls`` return the same subclass. Without
    :func:`attach_l2l_state` the subclass behaves as ``base_cls``.
    """

    class L2LTrainer(base_cls):  # type: ignore[misc, valid-type]
        soup_l2l: bool = True
        _soup_l2l: Optional[L2LState] = None

        def get_batch_samples(self, epoch_iterator, num_batches, device):  # type: ignore[override]
            samples, num_items = super().get_batch_samples(epoch_iterator, num_batches, device)
            if self._soup_l2l is not None:
                self._soup_l2l.window = list(samples)
                self._soup_l2l.position = 0
            return samples, num_items

        def training_step(  # type: ignore[override]
            self, model, inputs, num_items_in_batch=None
        ):
            import torch

            state = self._soup_l2l
            if state is None or state.window is None:
                return super().training_step(model, inputs, num_items_in_batch)
            first = state.position == 0
            state.position += 1
            window = state.window
            if state.position >= len(window):
                state.window = None
            if not first:
                return torch.zeros((), device=self.args.device)
            if num_items_in_batch is None:
                raise RuntimeError(
                    "training.l2l needs num_items_in_batch from the HF loop to scale "
                    "the loss over the window; this trainer did not receive one."
                )
            model.train()
            if state.has_dropout is None:
                state.has_dropout = _has_dropout(model)
            if self.args.neftune_noise_alpha and state.has_dropout:
                raise ValueError(
                    "training.l2l cannot combine neftune_alpha with dropout "
                    "(training.lora.dropout or attention_dropout above 0): accumulation "
                    "draws the NEFTune noise from the same stream as the dropout masks, "
                    "between the layers, and a layer-major pass cannot replay that. Turn "
                    "neftune_alpha off or set the dropout to 0."
                )
            device = state.runtime.device
            ids_list, label_list = _window_tensors(
                window,
                device,
                neftune=bool(self.args.neftune_noise_alpha),
                has_dropout=state.has_dropout,
            )
            # The step bypasses the wrapped forward, where the Trainer applies its mixed
            # precision; forward, head and the recompute in backward all run inside it.
            with self.accelerator.autocast():
                losses, counters = run_window(
                    state, model, ids_list, label_list, int(num_items_in_batch), device
                )
            state.steps += 1
            state.totals.add(counters)
            return torch.stack(losses).sum().to(self.args.device)

        def log(self, logs, *args, **kwargs):  # type: ignore[override]
            state = self._soup_l2l
            if state is not None and state.steps and "loss" in logs:
                logs = {**logs, **state.log_fields()}
            return super().log(logs, *args, **kwargs)

    return L2LTrainer


def attach_l2l_state(trainer: Any, runtime: Any, *, dtype: Any) -> L2LState:
    """Give a trainer from :func:`make_l2l_trainer_class` its streaming runtime."""
    state = L2LState(runtime=runtime, dtype=dtype)
    trainer._soup_l2l = state
    return state
