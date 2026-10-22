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

The L2L step takes one shape for all k micro-batches. A window with rows of different
lengths is right-padded to the longest one (rounded up to ``SHAPE_ROUND`` tokens) with
labels ``-100``: the causal mask keeps real tokens from seeing the padding and the loss
ignores it, so the result equals gradient accumulation.

torch is imported inside functions.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from soup_cli.utils.l2l_step import SUPPORTED_ARCHS, LoadCounters

#: Windows with mixed lengths pad to a multiple of this, so a run sees few distinct
#: shapes. Each new shape costs one forward to capture the layer arguments.
SHAPE_ROUND = 64

#: Captured layer arguments kept per shape; the cache restarts when a run exceeds it.
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
    totals: LoadCounters = field(default_factory=LoadCounters)
    store: Any = None
    steps_by_shape: Dict[Tuple[int, int], Any] = field(default_factory=dict)

    def log_fields(self) -> Dict[str, int]:
        return {
            "l2l_steps": self.steps,
            "l2l_layer_loads": self.totals.layer_loads,
            "l2l_large_loads": self.totals.large_loads,
            "l2l_weight_restores": self.totals.restores,
        }

    def close(self) -> None:
        if self.store is not None:
            self.store.close()
            self.store = None


def _window_tensors(
    window: List[Any], device: Any, *, neftune: bool = False, dropout: bool = False
) -> Tuple[List[Any], List[Any]]:
    """Input ids and labels of each micro-batch, right-padded to one common shape.

    With NEFTune on, a window that needs padding is refused: the noise scales with
    ``1 / sqrt(seq * hidden)`` and is drawn per element, so padding the ids before
    the embedding changes both its size and the RNG sequence of gradient accumulation.
    Dropout is refused the same way: padding changes the shape of the layer input and so
    the shape and values of every mask drawn from ``l2l_rng.layer_seed``.
    """
    import torch

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
    if len(shapes) == 1:
        return ids_list, label_list
    if neftune:
        raise ValueError(
            "training.l2l with training.neftune_alpha needs every micro-batch of a window "
            f"to have one shape, but this window has {sorted(shapes)}: padding them would "
            "change the NEFTune noise. Use rows of one length, or turn neftune_alpha off."
        )
    if dropout:
        raise ValueError(
            "training.l2l with dropout (training.lora.dropout or attention_dropout above 0) "
            f"needs every micro-batch of a window to have one shape, but this window has "
            f"{sorted(shapes)}: padding them would change the dropout masks. Use rows of "
            "one length, or set the dropout to 0."
        )
    rows = max(ids.shape[0] for ids in ids_list)
    longest = max(ids.shape[1] for ids in ids_list)
    seq = -(-longest // SHAPE_ROUND) * SHAPE_ROUND
    padded_ids, padded_labels = [], []
    for ids, labels in zip(ids_list, label_list):
        pad = (0, seq - ids.shape[1], 0, rows - ids.shape[0])
        padded_ids.append(torch.nn.functional.pad(ids, pad, value=0))
        padded_labels.append(torch.nn.functional.pad(labels, pad, value=-100))
    return padded_ids, padded_labels


def _has_dropout(model: Any) -> bool:
    """True when the model draws dropout masks in training: LoRA dropout or attention dropout."""
    import torch

    config = getattr(model, "config", None)
    if float(getattr(config, "attention_dropout", 0.0) or 0.0) != 0:
        return True
    return any(
        (isinstance(module, torch.nn.Dropout) and module.p > 0)
        or float(getattr(module, "attention_dropout", 0.0) or 0.0) != 0
        for module in model.modules()
    )


def _l2l_step_for(
    state: L2LState, model: Any, shape: Tuple[int, int], ids: Any, seed: int
) -> Any:
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    stepper = state.steps_by_shape.get(shape)
    if stepper is None:
        if len(state.steps_by_shape) >= MAX_CACHED_SHAPES:
            state.steps_by_shape.clear()
        stepper = L2LStep(model, state.runtime, capture_layer_call(model, ids), seed=seed)
        state.steps_by_shape[shape] = stepper
    return stepper


def check_activation_budget(
    *, free_ram: Optional[int], n_layers: int, k: int, chunk_bytes: int, pinned: bool
) -> None:
    """Refuse an activation store that would not fit the free host RAM, before allocating it.

    The weight store and staging are already resident when this runs, so ``free_ram``
    is what they left over. A pinned chunk is its own allocation, and torch's host
    allocator rounds it up to the next power of two (#901). Without psutil
    (``free_ram`` is None) nothing is checked.
    """
    from soup_cli.utils.layer_stream import RAM_TIER_HEADROOM
    from soup_cli.utils.layer_stream_runtime import _next_power_of_two

    block = _next_power_of_two(chunk_bytes) if pinned else chunk_bytes
    nbytes = n_layers * k * block
    if free_ram is None or nbytes < free_ram * RAM_TIER_HEADROOM:
        return
    kind = "pinned " if pinned else ""
    rounding = (
        f" (a {chunk_bytes / 1e6:.1f} MB chunk takes a {block / 1e6:.1f} MB pinned block)"
        if block != chunk_bytes
        else ""
    )
    raise ValueError(
        f"training.l2l keeps {n_layers} layers x {k} micro-batches of activations in host "
        f"RAM: {nbytes / 1e9:.2f} GB of {kind}blocks{rounding}, but only "
        f"{free_ram / 1e9:.1f} GB is free and {RAM_TIER_HEADROOM:.0%} of that is usable. "
        f"Lower gradient_accumulation_steps, data.max_length or batch_size, or free RAM."
    )


def _store_for(state: L2LState, stepper: Any, shape: Tuple[int, int], k: int, device: Any) -> Any:
    import torch

    from soup_cli.utils.l2l_activations import ActivationStore
    from soup_cli.utils.layer_stream import free_ram_bytes

    hidden = int(stepper.causal.config.hidden_size)
    if state.store is None or state.store.shape[:2] != shape or state.store.k < k:
        state.close()
        n_layers = len(stepper.layers)
        itemsize = torch.empty((), dtype=state.dtype).element_size()
        chunk_bytes = shape[0] * shape[1] * hidden * itemsize
        pin = str(device).startswith("cuda")
        check_activation_budget(
            free_ram=free_ram_bytes(),
            n_layers=n_layers,
            k=k,
            chunk_bytes=chunk_bytes,
            pinned=pin,
        )
        state.store = ActivationStore(
            n_layers=n_layers,
            k=k,
            chunk_shape=(shape[0], shape[1], hidden),
            dtype=state.dtype,
            device=device,
            pin=pin,
        )
    return state.store


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
            device = state.runtime.device
            ids_list, label_list = _window_tensors(
                window,
                device,
                neftune=bool(self.args.neftune_noise_alpha),
                dropout=_has_dropout(model),
            )
            shape = tuple(int(dim) for dim in ids_list[0].shape)
            # The step bypasses the wrapped forward, where the Trainer applies its mixed
            # precision; forward, head and the recompute in backward all run inside it.
            with self.accelerator.autocast():
                stepper = _l2l_step_for(state, model, shape, ids_list[0], int(self.args.seed))
                store = _store_for(state, stepper, shape, len(window), device)
                losses = stepper.run(
                    ids_list,
                    store,
                    n_items=int(num_items_in_batch),
                    labels=label_list,
                    step=int(self.state.global_step),
                )
            state.steps += 1
            state.totals.add(stepper.counters)
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
