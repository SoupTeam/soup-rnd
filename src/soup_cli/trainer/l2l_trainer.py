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
    attention_dropout = float(getattr(model_config, "attention_dropout", 0.0) or 0.0)
    if attention_dropout != 0:
        raise ValueError(
            f"training.l2l requires attention_dropout=0 on the model (got "
            f"{attention_dropout}): a layer-major pass draws dropout masks in a "
            f"different order than gradient accumulation."
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


def _window_tensors(window: List[Any], device: Any) -> Tuple[List[Any], List[Any]]:
    """Input ids and labels of each micro-batch, right-padded to one common shape."""
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
    rows = max(ids.shape[0] for ids in ids_list)
    longest = max(ids.shape[1] for ids in ids_list)
    seq = -(-longest // SHAPE_ROUND) * SHAPE_ROUND
    padded_ids, padded_labels = [], []
    for ids, labels in zip(ids_list, label_list):
        pad = (0, seq - ids.shape[1], 0, rows - ids.shape[0])
        padded_ids.append(torch.nn.functional.pad(ids, pad, value=0))
        padded_labels.append(torch.nn.functional.pad(labels, pad, value=-100))
    return padded_ids, padded_labels


def _l2l_step_for(state: L2LState, model: Any, shape: Tuple[int, int], ids: Any) -> Any:
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    step = state.steps_by_shape.get(shape)
    if step is None:
        if len(state.steps_by_shape) >= MAX_CACHED_SHAPES:
            state.steps_by_shape.clear()
        step = L2LStep(model, state.runtime, capture_layer_call(model, ids))
        state.steps_by_shape[shape] = step
    return step


def _store_for(state: L2LState, step: Any, shape: Tuple[int, int], k: int, device: Any) -> Any:
    from soup_cli.utils.l2l_activations import ActivationStore

    hidden = int(step.causal.config.hidden_size)
    if state.store is None or state.store.shape[:2] != shape or state.store.k < k:
        state.close()
        state.store = ActivationStore(
            n_layers=len(step.layers),
            k=k,
            chunk_shape=(shape[0], shape[1], hidden),
            dtype=state.dtype,
            device=device,
            pin=str(device).startswith("cuda"),
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
            ids_list, label_list = _window_tensors(window, device)
            shape = tuple(int(dim) for dim in ids_list[0].shape)
            step = _l2l_step_for(state, model, shape, ids_list[0])
            store = _store_for(state, step, shape, len(window), device)
            losses = step.run(
                ids_list, store, n_items=int(num_items_in_batch), labels=label_list
            )
            state.steps += 1
            state.totals.add(step.counters)
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
