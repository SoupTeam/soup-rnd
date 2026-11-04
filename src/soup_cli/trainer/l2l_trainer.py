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

Each micro-batch position keeps its largest capacity seen so far. A larger window grows
the store after releasing its old allocations. When the store does not fit the host-RAM
budget, the lowest layers spill to one file (``soup_cli.utils.l2l_spill``); ``train`` removes
it on every exit it sees and sweeps files of dead runs at start.

torch is imported inside functions.
"""

from __future__ import annotations

import functools
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from soup_cli.utils.expert_shard import EXPERT_LAYOUT_PER_EXPERT
from soup_cli.utils.l2l_step import LoadCounters

#: Architectures ``training.l2l`` admits at launch with any weight layout.
PUBLIC_ARCHS = ("llama", "mistral")

#: Architectures ``training.l2l`` admits only with ``training.stream_expert_layout``
#: set to ``EXPERT_LAYOUT_PER_EXPERT``: the DeepSeek-V3 step reads each routed expert from its
#: own cache entry, which a fused layer cache does not have.
PER_EXPERT_ARCHS = ("deepseek_v3",)

logger = logging.getLogger(__name__)

#: Captured layer arguments kept per shape; the cache restarts when a run exceeds it. A new
#: shape costs one forward to capture them.
MAX_CACHED_SHAPES = 64


def refuse_unsupported_l2l(
    model_config: Any, *, n_gpu: int, fp16: bool = False, expert_layout: str = "fused"
) -> None:
    """Refuse at launch what the schema cannot see: the model, the GPU count, fp16.

    ``expert_layout`` is ``training.stream_expert_layout``; DeepSeek-V3 needs ``per_expert``.
    """
    arch = str(getattr(model_config, "model_type", ""))
    if arch in PER_EXPERT_ARCHS:
        if expert_layout != EXPERT_LAYOUT_PER_EXPERT:
            raise ValueError(
                f"training.l2l with model_type {arch!r} requires "
                f"training.stream_expert_layout={EXPERT_LAYOUT_PER_EXPERT!r}; "
                f"got {expert_layout!r}."
            )
    elif arch not in PUBLIC_ARCHS:
        raise ValueError(
            f"training.l2l does not support model_type {arch!r}: it is enabled for "
            f"{' and '.join(PUBLIC_ARCHS)}, and for {' and '.join(PER_EXPERT_ARCHS)} "
            f"with training.stream_expert_layout={EXPERT_LAYOUT_PER_EXPERT!r}."
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
    #: ``training.l2l_spill_dir``; None means ``<output_dir>/_l2l_spill``.
    spill_dir: Optional[str] = None
    #: Directories the weights stream from (layer cache and stripe folders).
    weight_dirs: Tuple[str, ...] = ()
    output_dir: Optional[str] = None
    #: The plan of the current store (kept after it is closed); None before the first step.
    spill_plan: Optional[Dict[str, Any]] = None
    #: Files of dead runs the last ``train`` removed from the spill directory.
    stale_removed: List[str] = field(default_factory=list)
    c3_plan: Any = None

    def log_fields(self) -> Dict[str, Any]:
        fields = {
            "l2l_steps": self.steps,
            "l2l_captures": self.captures,
            "l2l_capture_layer_loads": self.capture_loads,
            "l2l_capture_seconds": self.capture_seconds,
            "l2l_layer_loads": self.totals.layer_loads,
            "l2l_large_loads": self.totals.large_loads,
            "l2l_weight_restores": self.totals.restores,
            "l2l_spill_layers": int(self.spill_plan["spill_layers"]) if self.spill_plan else 0,
        }
        # Only a runtime that streams routed experts has them to count; a dense run's log
        # keeps the fields it always had.
        if getattr(self.runtime, "expert_pool", None) is not None:
            fields["l2l_expert_fetches"] = self.totals.expert_fetches
            fields["l2l_expert_bytes_read"] = self.totals.expert_bytes_read
        return fields

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


def _spill_line(plan: Dict[str, Any], budget: Any) -> str:
    from soup_cli.utils.terminal import for_terminal

    return (
        f"[yellow]L2L:[/] activations do not fit host RAM; spilling "
        f"{plan['spill_layers']} of {plan['n_layers']} layers "
        f"({plan['spill_file_bytes'] / 1e9:.2f} GB) to {for_terminal(plan['spill_path'])}; "
        f"{for_terminal(budget.describe())}"
    )


def _store_for(state: L2LState, stepper: Any, ids_list: List[Any], device: Any) -> Any:
    """The activation store for the window, grown only when a micro-batch does not fit.

    Micro-batch ``i`` of every window uses chunk ``i``, so the store keeps the largest
    size each position has needed. The plan sums those allocations for resident layers;
    spill slots and staging buffers hold the largest chunk. Replanning and directory
    validation happen after the old store is released and before the new allocation.
    """
    import math

    import torch

    from soup_cli.utils import l2l_spill
    from soup_cli.utils.l2l_activations import ActivationStore

    hidden = int(stepper.causal.config.hidden_size)
    store = state.store
    n_layers = len(stepper.layers)
    pin = str(device).startswith("cuda")
    if state.c3_plan is not None:
        # Before any release: a window outside the reserved plan keeps the current store.
        state.c3_plan.validate_microbatches(
            n_layers=n_layers,
            shapes=[(int(ids.shape[0]), int(ids.shape[1]), hidden) for ids in ids_list],
            dtype=str(state.dtype).removeprefix("torch."),
            pin=pin,
        )
    need = [int(ids.shape[0]) * int(ids.shape[1]) * hidden for ids in ids_list]
    have = [math.prod(shape) for shape in store.mb_shapes] if store is not None else []
    if store is not None and len(need) <= len(have) and all(
        n <= h for n, h in zip(need, have)
    ):
        return store
    caps = [max(n, h) for n, h in zip(need, have)] + need[len(have):] + have[len(need):]
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
    if state.c3_plan is not None:
        from soup_cli.utils.budgeted_activations import BudgetedActivationStore

        # The lease charges every chunk at the largest capacity: an upper bound of the
        # per-micro-batch allocations, inside the partition reserved before the cache.
        state.store = BudgetedActivationStore(
            host_budget=state.c3_plan.activations,
            n_layers=n_layers,
            k=len(caps),
            chunk_shape=(max(caps),),
            mb_shapes=[(cap,) for cap in caps],
            dtype=state.dtype,
            device=device,
            pin=pin,
        )
        return state.store
    budget = l2l_spill.ram_budget()
    logger.info(
        "training.l2l activation budget: %s [limiting level %s: %s]",
        budget.describe(), budget.limiting_level, budget.limiting_numbers,
    )
    plan = l2l_spill.plan_spill(
        budget.budget_bytes,
        n_layers=n_layers,
        k=len(caps),
        chunk_bytes=max(caps) * itemsize,
        mb_bytes=[cap * itemsize for cap in caps],
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
        k=len(caps),
        chunk_shape=(max(caps),),
        mb_shapes=[(cap,) for cap in caps],
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


def attach_l2l_state(
    trainer: Any,
    runtime: Any,
    *,
    dtype: Any,
    c3_plan: Any = None,
    spill_dir: Optional[str] = None,
    weight_dirs: Sequence[str] = (),
) -> L2LState:
    """Give a trainer from :func:`make_l2l_trainer_class` its streaming runtime.

    ``spill_dir`` is ``training.l2l_spill_dir`` (None: ``<output_dir>/_l2l_spill``), and
    ``weight_dirs`` lists the directories the weights stream from on the disk tier.
    """
    args = getattr(trainer, "args", None)
    state = L2LState(
        runtime=runtime,
        dtype=dtype,
        c3_plan=c3_plan,
        spill_dir=spill_dir,
        weight_dirs=tuple(str(path) for path in weight_dirs),
        output_dir=getattr(args, "output_dir", None),
    )
    trainer._soup_l2l = state
    return state
