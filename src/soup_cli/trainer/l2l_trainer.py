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

The activation store is built once, for the largest window the run can produce
(``per_device_train_batch_size`` rows of ``max_seq`` tokens, ``gradient_accumulation_steps``
micro-batches), and serves every smaller window. When it does not fit the host-RAM budget,
the lowest layers spill to one file (``soup_cli.utils.l2l_spill``); ``train`` removes it on
every exit it sees and sweeps files of dead runs at start.

torch is imported inside functions.
"""

from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from soup_cli.utils.l2l_step import SUPPORTED_ARCHS, LoadCounters

logger = logging.getLogger(__name__)

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
    has_dropout: Optional[bool] = None
    totals: LoadCounters = field(default_factory=LoadCounters)
    store: Any = None
    steps_by_shape: Dict[Tuple[int, int], Any] = field(default_factory=dict)
    #: ``training.l2l_spill_dir``; None means ``<output_dir>/_l2l_spill``.
    spill_dir: Optional[str] = None
    #: Directories the weights stream from (layer cache and stripe folders).
    weight_dirs: Tuple[str, ...] = ()
    output_dir: Optional[str] = None
    #: The largest window: rows, sequence length (None: unknown) and micro-batches.
    max_rows: int = 0
    max_seq: Optional[int] = None
    max_k: int = 0
    #: The plan of the current store (kept after it is closed); None before the first step.
    spill_plan: Optional[Dict[str, Any]] = None
    #: Files of dead runs the last ``train`` removed from the spill directory.
    stale_removed: List[str] = field(default_factory=list)

    def log_fields(self) -> Dict[str, int]:
        return {
            "l2l_steps": self.steps,
            "l2l_layer_loads": self.totals.layer_loads,
            "l2l_large_loads": self.totals.large_loads,
            "l2l_weight_restores": self.totals.restores,
            "l2l_spill_layers": int(self.spill_plan["spill_layers"]) if self.spill_plan else 0,
        }

    def spill_directory(self) -> str:
        """The configured spill directory, or the default one under the output directory."""
        if self.spill_dir is not None:
            return self.spill_dir
        from soup_cli.utils.l2l_spill import SPILL_DIR_NAME

        return os.path.join(self.output_dir or ".", SPILL_DIR_NAME)

    def close(self) -> None:
        if self.store is not None:
            self.store.close()
            self.store = None


def _window_tensors(
    window: List[Any], device: Any, *, neftune: bool = False, has_dropout: bool = False
) -> Tuple[List[Any], List[Any]]:
    """Input ids and labels of each micro-batch, right-padded to one common shape.

    With NEFTune on, a window that needs padding is refused: the noise scales with
    ``1 / sqrt(seq * hidden)`` and is drawn per element, so padding the ids before
    the embedding changes both its size and the RNG sequence of gradient accumulation.
    Dropout is refused the same way: padding changes the shape of the layer input, so the
    replay of ``l2l_rng`` would no longer consume the stream the unpadded rows consume.
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
    if has_dropout:
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


def longest_row(dataset: Any) -> Optional[int]:
    """The longest ``input_ids`` row of a map-style HF ``datasets.Dataset``, or None.

    Read from the Arrow table under the dataset (no Python lists), so after a ``select`` or
    ``shuffle`` it is still an upper bound. None for an iterable or streaming dataset, one
    without an ``input_ids`` column, or anything that cannot be read that way.
    """
    try:
        import pyarrow.compute as pc
        from datasets import Dataset

        if not isinstance(dataset, Dataset) or "input_ids" not in dataset.column_names:
            return None
        longest = pc.max(pc.list_value_length(dataset.data.column("input_ids"))).as_py()
    except Exception:  # noqa: BLE001 - an unknown length falls back to data.max_length
        logger.debug("could not read the longest input_ids row", exc_info=True)
        return None
    return int(longest) if longest else None


def _l2l_step_for(
    state: L2LState, model: Any, shape: Tuple[int, int], ids: Any
) -> Any:
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    stepper = state.steps_by_shape.get(shape)
    if stepper is None:
        if len(state.steps_by_shape) >= MAX_CACHED_SHAPES:
            state.steps_by_shape.clear()
        stepper = L2LStep(model, state.runtime, capture_layer_call(model, ids))
        state.steps_by_shape[shape] = stepper
    return stepper


def _capacity(state: L2LState, shape: Tuple[int, int], k: int) -> Tuple[int, int, int]:
    """``(rows, seq, k)`` of the store for the largest window this run can produce.

    A mixed-length window pads to a multiple of ``SHAPE_ROUND``, so a known ``max_seq`` is
    rounded up to it. Without ``max_seq`` the store is sized to the largest shape seen so
    far, and only a window larger than that rebuilds it.
    """
    rows = max(int(shape[0]), int(state.max_rows or 0))
    seq = int(shape[1])
    if state.max_seq:
        seq = max(seq, -(-int(state.max_seq) // SHAPE_ROUND) * SHAPE_ROUND)
    k = max(int(k), int(state.max_k or 0))
    store = state.store
    if store is not None:
        rows, seq, k = max(rows, store.shape[0]), max(seq, store.shape[1]), max(k, store.k)
    return rows, seq, k


def _spill_line(plan: Dict[str, Any], budget: Any) -> str:
    from soup_cli.utils.terminal import for_terminal

    return (
        f"[yellow]L2L:[/] activations do not fit host RAM; spilling "
        f"{plan['spill_layers']} of {plan['n_layers']} layers "
        f"({plan['spill_file_bytes'] / 1e9:.2f} GB) to {for_terminal(plan['spill_path'])}; "
        f"{for_terminal(budget.describe())}"
    )


def _store_for(state: L2LState, stepper: Any, shape: Tuple[int, int], k: int, device: Any) -> Any:
    """The activation store, built once for the largest window and reused for smaller ones.

    The plan keeps as many layers resident as the RAM budget holds and spills the rest;
    it is decided, and the spill directory checked, before anything is allocated.
    """
    import torch

    from soup_cli.utils import l2l_spill
    from soup_cli.utils.l2l_activations import ActivationStore

    hidden = int(stepper.causal.config.hidden_size)
    store = state.store
    if store is not None and shape[0] * shape[1] * hidden <= store.capacity and k <= store.k:
        return store
    rows, seq, k_cap = _capacity(state, shape, k)
    pin = str(device).startswith("cuda")
    rebuilding_cuda = store is not None and pin
    state.close()
    # close() releases the old chunks/rings; do not retain the old store while replanning.
    del store
    if rebuilding_cuda:
        empty_host_cache = getattr(getattr(torch, "accelerator", None), "empty_host_cache", None)
        if empty_host_cache is not None:
            empty_host_cache()
        else:
            # Older torch has no supported cache-release API. Do not invent reusable-byte
            # credit: a larger window can need different pinned allocation size buckets.
            logger.warning(
                "training.l2l is growing its CUDA activation store, but this torch has no "
                "torch.accelerator.empty_host_cache(); idle pinned blocks may remain charged "
                "to host RAM, so the new plan may need extra free RAM."
            )
    n_layers = len(stepper.layers)
    itemsize = torch.empty((), dtype=state.dtype).element_size()
    budget = l2l_spill.ram_budget()
    logger.info(
        "training.l2l activation budget: %s [limiting level %s: %s]",
        budget.describe(), budget.limiting_level, budget.limiting_numbers,
    )
    plan = l2l_spill.plan_spill(
        budget.budget_bytes,
        n_layers=n_layers,
        k=k_cap,
        chunk_bytes=rows * seq * hidden * itemsize,
        ring=l2l_spill.TRAINER_RING,
        pin=pin,
        budget_source=budget.describe(),
    )
    record = {
        **plan.as_dict(),
        "available_bytes": budget.available_bytes,
        "limiting_level": budget.limiting_level,
        "limiting_numbers": dict(budget.limiting_numbers),
        "spill_dir": state.spill_directory(),
        "spill_path": None,
        "explicit_dir": state.spill_dir is not None,
    }
    factory = None
    if plan.spill_layers:
        directory = l2l_spill.prepare_spill_dir(
            record["spill_dir"],
            explicit=record["explicit_dir"],
            spill_layers=plan.spill_layers,
            file_bytes=plan.spill_file_bytes,
            tier=str(getattr(state.runtime, "tier", "ram")),
            weight_dirs=state.weight_dirs,
        )
        record["spill_dir"] = directory
        record["spill_path"] = os.path.join(directory, l2l_spill.spill_file_name())
        factory = functools.partial(l2l_spill.open_spill_file, record["spill_path"])
    state.store = ActivationStore(
        n_layers=n_layers,
        k=k_cap,
        chunk_shape=(rows, seq, hidden),
        dtype=state.dtype,
        device=device,
        pin=pin,
        spill_layers=plan.spill_layers,
        spill_factory=factory,
        ring=l2l_spill.TRAINER_RING,
    )
    state.spill_plan = record
    if plan.spill_layers:
        l2l_spill.say(_spill_line(record, budget))
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

        def train(self, *args, **kwargs):  # type: ignore[override]
            """HF's ``train`` with the spill directory swept before it and the activation
            store (and its spill file) closed after it, on a normal end, an exception or
            Ctrl-C alike."""
            state = self._soup_l2l
            if state is None:
                return super().train(*args, **kwargs)
            from soup_cli.utils.l2l_spill import sweep_stale

            state.stale_removed = sweep_stale(state.spill_directory())
            try:
                return super().train(*args, **kwargs)
            finally:
                state.close()

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
            shape = tuple(int(dim) for dim in ids_list[0].shape)
            # The step bypasses the wrapped forward, where the Trainer applies its mixed
            # precision; forward, head and the recompute in backward all run inside it.
            with self.accelerator.autocast():
                stepper = _l2l_step_for(state, model, shape, ids_list[0])
                store = _store_for(state, stepper, shape, len(window), device)
                losses = stepper.run(
                    ids_list,
                    store,
                    n_items=int(num_items_in_batch),
                    labels=label_list,
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


def attach_l2l_state(
    trainer: Any,
    runtime: Any,
    *,
    dtype: Any,
    spill_dir: Optional[str] = None,
    weight_dirs: Sequence[str] = (),
    max_seq: Optional[int] = None,
) -> L2LState:
    """Give a trainer from :func:`make_l2l_trainer_class` its streaming runtime.

    ``spill_dir`` is ``training.l2l_spill_dir`` (None: ``<output_dir>/_l2l_spill``),
    ``weight_dirs`` the directories the weights stream from on the disk tier, and
    ``max_seq`` the longest row (``data.max_length``; None: the trainer's ``max_length``
    if it has one, else the longest window seen so far).
    """
    args = getattr(trainer, "args", None)
    if max_seq is None:
        max_seq = getattr(args, "max_length", None)
    state = L2LState(
        runtime=runtime,
        dtype=dtype,
        spill_dir=spill_dir,
        weight_dirs=tuple(str(path) for path in weight_dirs),
        output_dir=getattr(args, "output_dir", None),
        max_rows=int(getattr(args, "per_device_train_batch_size", 0) or 0),
        max_seq=int(max_seq) if max_seq else None,
        max_k=int(getattr(args, "gradient_accumulation_steps", 0) or 0),
    )
    trainer._soup_l2l = state
    return state
