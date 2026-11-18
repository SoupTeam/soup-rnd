"""Train the LoRA adapters of a subset of decoder layers (``training.lora_layers``).

ZERO-2 (Qwen3.5-9B in 8 and 4 GB): which layers of a model need training for a given
dataset, and what training only those costs in quality. LoRA is attached to every layer
the target modules match; this module decides which layers' adapters are trainable and
freezes the rest (``requires_grad`` off, so they get no gradient and no optimizer step).

Methods (``LayerSelectPolicy.method``):

* ``fixed``: the ``layers`` given.
* ``random``: ``k`` layers drawn once with ``seed`` (the control for the scored methods).
* ``grad``: before training, ``probe_rows`` rows of the training set run forward and
  backward with every adapter trainable; a layer's score is the mean squared gradient of
  its adapter parameters (with peft's zero-initialised ``lora_B`` only B carries it). The
  ``k`` highest train.
* ``angular``: the same rows run forward only; a layer's score is the mean angular
  distance between the residual entering and leaving it (``arccos(cos) / pi``, as
  ``soup shrink`` scores blocks for removal). The ``k`` layers that change the residual
  most train.
* ``lisa``: LISA (arXiv 2403.17919) over the adapters: ``k`` layers drawn at random every
  ``interval_steps`` optimizer steps, the optimizer state of adapters that go inactive
  cleared. Unlike ``training.lisa_enabled`` (full fine-tuning) it trains only adapters, so
  it works on a quantized base.

The probe runs on its own sequential loader over the training set with the RNG state
saved and restored, so the training run's data order and random draws are unchanged.
The choice and the scores are written to ``<output>/lora_layers.json``.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

METHODS = ("fixed", "random", "grad", "angular", "lisa")
#: A decoder layer's index in a parameter name (``model.layers.3.``,
#: ``model.language_model.layers.3.``, GPT-2's ``h.3.``).
_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)\.")
RECORD_NAME = "lora_layers.json"
#: Set on an adapter parameter this module froze: it is still part of the saved adapter
#: (``layer_stream_runtime.assert_streamed_adapter_saved`` counts it).
FROZEN_BY_SELECTION = "_soup_frozen_by_layer_select"


@dataclass(frozen=True)
class LayerSelectPolicy:
    method: str
    k: Optional[int] = None
    layers: Tuple[int, ...] = ()
    probe_rows: int = 64
    interval_steps: int = 20
    seed: int = 0

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(
                f"layer selection method must be one of {METHODS}, got {self.method!r}"
            )
        if self.method == "fixed":
            if not self.layers:
                raise ValueError("method 'fixed' needs layers")
        elif self.k is None or self.k < 1:
            raise ValueError(f"method {self.method!r} needs k >= 1")
        if self.probe_rows < 1 or self.interval_steps < 1 or self.seed < 0:
            raise ValueError("probe_rows and interval_steps must be >= 1, seed >= 0")


def adapter_params_by_layer(model: Any) -> Dict[int, List[Tuple[str, Any]]]:
    """Every LoRA parameter of a decoder layer, keyed by the layer's index.

    Only parameters that are trainable when this is called count: the adapters as peft
    created them. Raises when the model has no per-layer adapter.
    """
    by_layer: Dict[int, List[Tuple[str, Any]]] = {}
    for name, param in model.named_parameters():
        if "lora_" not in name or not param.requires_grad:
            continue
        match = _LAYER_RE.search(name)
        if match is None:
            continue
        by_layer.setdefault(int(match.group(1)), []).append((name, param))
    if not by_layer:
        raise ValueError(
            "training.lora_layers: the model has no trainable LoRA parameter inside a "
            "numbered decoder layer ('layers.N.'), so there is no layer to select"
        )
    return by_layer


def apply_layer_mask(by_layer: Dict[int, List[Tuple[str, Any]]], active: Iterable[int]) -> int:
    """Make the adapters of ``active`` layers trainable and freeze the others.

    Returns the number of trainable adapter parameters (elements).
    """
    keep = set(active)
    unknown = keep - set(by_layer)
    if unknown:
        raise ValueError(
            f"training.lora_layers: layer(s) {sorted(unknown)} have no adapter; the model's "
            f"adapted layers are {min(by_layer)}..{max(by_layer)}"
        )
    count = 0
    for idx, params in by_layer.items():
        on = idx in keep
        for _name, param in params:
            param.requires_grad_(on)
            setattr(param, FROZEN_BY_SELECTION, not on)
            if not on:
                param.grad = None
            else:
                count += param.numel()
    return count


def top_k(scores: Dict[int, float], k: int) -> List[int]:
    """The ``k`` highest-scoring layers, ties broken by the lower index; sorted."""
    ranked = sorted(scores, key=lambda idx: (-scores[idx], idx))
    return sorted(ranked[: min(k, len(ranked))])


def draw(layers: Sequence[int], k: int, rng: random.Random) -> List[int]:
    return sorted(rng.sample(sorted(layers), min(k, len(layers))))


def _model_inputs(batch: Dict[str, Any], device: Any) -> Dict[str, Any]:
    keep = ("input_ids", "attention_mask", "labels", "position_ids")
    return {k: v.to(device) for k, v in batch.items() if k in keep and hasattr(v, "to")}


def grad_scores(model: Any, by_layer: Dict[int, List[Tuple[str, Any]]],
                batches: Iterable[Dict[str, Any]], device: Any) -> Dict[int, float]:
    """Mean squared gradient of each layer's adapters over ``batches`` (every adapter on).

    Leaves no gradient behind.
    """
    import torch

    sums = {idx: 0.0 for idx in by_layer}
    sizes = {idx: sum(p.numel() for _n, p in params) for idx, params in by_layer.items()}
    for params in by_layer.values():
        for _n, p in params:
            p.grad = None
    n = 0
    with torch.enable_grad():
        for batch in batches:
            out = model(**_model_inputs(batch, device))
            out.loss.backward()
            for idx, params in by_layer.items():
                for _n, p in params:
                    if p.grad is not None:
                        sums[idx] += float(p.grad.detach().float().pow(2).sum())
                        p.grad = None
            n += 1
    if n == 0:
        raise ValueError("training.lora_layers: the probe saw no batch")
    return {idx: sums[idx] / max(1, sizes[idx]) / n for idx in by_layer}


def angular_scores(model: Any, n_layers: int, batches: Iterable[Dict[str, Any]],
                   device: Any) -> Dict[int, float]:
    """Mean angular distance between each layer's input and output residual."""
    import torch

    sums = [0.0] * n_layers
    counts = [0] * n_layers
    with torch.no_grad():
        for batch in batches:
            inputs = _model_inputs(batch, device)
            inputs.pop("labels", None)
            hidden = model(**inputs, output_hidden_states=True).hidden_states
            if len(hidden) != n_layers + 1:
                raise ValueError(f"expected {n_layers + 1} hidden states, got {len(hidden)}")
            mask = inputs.get("attention_mask")
            for idx in range(n_layers):
                a = hidden[idx].float()
                b = hidden[idx + 1].float()
                cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).clamp(-1.0, 1.0)
                dist = torch.arccos(cos) / torch.pi
                if mask is not None:
                    dist = dist[mask.to(torch.bool)]
                sums[idx] += float(dist.sum())
                counts[idx] += int(dist.numel())
    return {idx: sums[idx] / max(1, counts[idx]) for idx in range(n_layers)}


def probe_batches(dataset: Any, collator: Any, rows: int, batch_size: int) -> List[Any]:
    """The first ``rows`` rows of ``dataset`` in order, collated in batches."""
    n = min(int(rows), len(dataset))
    items = [dataset[i] for i in range(n)]
    return [collator(items[i: i + batch_size]) for i in range(0, n, batch_size)]


@dataclass
class Selection:
    method: str
    layers: List[int]
    scores: Dict[int, float] = field(default_factory=dict)
    history: List[Tuple[int, List[int]]] = field(default_factory=list)
    backward_cut: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "layers": self.layers,
            "backward_cut": self.backward_cut,
            "scores": {str(k): v for k, v in sorted(self.scores.items())},
            "history": [{"step": s, "layers": ls} for s, ls in self.history],
        }


def choose(policy: LayerSelectPolicy, model: Any, by_layer: Dict[int, List[Tuple[str, Any]]],
           batches: Optional[Sequence[Any]], device: Any, rng: random.Random) -> Selection:
    """The first selection of a run (``lisa`` draws its first set here)."""
    import torch

    layers = sorted(by_layer)
    if policy.method == "fixed":
        return Selection("fixed", sorted(set(policy.layers)))
    if policy.method in ("random", "lisa"):
        return Selection(policy.method, draw(layers, policy.k, rng))
    if not batches:
        raise ValueError(f"method {policy.method!r} needs probe batches")
    cuda = [torch.device(device)] if torch.device(device).type == "cuda" else []
    was_training = model.training
    with torch.random.fork_rng(devices=cuda):
        if policy.method == "grad":
            model.train()
            scores = grad_scores(model, by_layer, batches, device)
        else:
            model.eval()
            scores = angular_scores(model, max(layers) + 1, batches, device)
            scores = {idx: scores[idx] for idx in layers}
    model.train(was_training)
    return Selection(policy.method, top_k(scores, policy.k), scores)


_INPUT_GRAD_HOOK_NAMES = ("make_inputs_require_grads", "make_inputs_require_grad")


def cut_input_grads(model: Any) -> str:
    """Let autograd stop at the lowest trainable layer on the resident path.

    transformers (``gradient_checkpointing_enable``), PEFT and TRL all hook the input embedding so
    its output requires grad: reentrant checkpointing needs a grad-requiring input to rebuild a
    layer's graph. That hook also makes every backward run down to the embedding, through every
    frozen layer below the lowest one whose adapters train, and with ``lora_layers`` choosing,
    say, the top 8 of 32 layers that is most of the backward for a gradient nobody reads. With
    non-reentrant checkpointing (Soup's and transformers' default) the hook is not needed, so it
    is removed and the backward ends at the lowest layer with a trainable parameter, the same cut
    L2L makes by hand. Returns what was done, for the record."""
    for module in model.modules():
        func = getattr(module, "_gradient_checkpointing_func", None)
        if func is None or not getattr(module, "gradient_checkpointing", False):
            continue
        if getattr(func, "keywords", {}).get("use_reentrant", True) is not False:
            return "kept: reentrant gradient checkpointing needs the input grads"
    getter = getattr(model, "get_input_embeddings", None)
    embedding = getter() if callable(getter) else None
    if embedding is None:
        return "kept: no input embedding"
    weight = getattr(embedding, "weight", None)
    if weight is not None and weight.requires_grad:
        return "kept: the input embedding trains"
    for module in model.modules():
        disable = getattr(module, "disable_input_require_grads", None)
        if callable(disable) and hasattr(module, "_require_grads_hooks"):
            disable()
    for key, hook in list(embedding._forward_hooks.items()):  # hooks registered by hand (PEFT)
        if getattr(hook, "__name__", "") in _INPUT_GRAD_HOOK_NAMES:
            del embedding._forward_hooks[key]
    return "cut: the backward stops at the lowest trainable layer"


def write_record(selection: Selection, output_dir: Optional[str]) -> Optional[str]:
    if not output_dir:
        return None
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, RECORD_NAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(selection.to_json(), handle, indent=2)
    return path


def clear_optimizer_state(optimizer: Any, params: Iterable[Any]) -> None:
    state = getattr(optimizer, "state", None)
    if state is None:
        return
    for param in params:
        if param in state:
            state[param] = {}


def _callback_base():
    try:
        from transformers import TrainerCallback

        return TrainerCallback
    except Exception:  # noqa: BLE001 — transformers is optional in slim environments
        return object


class _LayerSelectBody:  # noqa: N801
    """HF callback: choose at ``on_train_begin``, re-draw for ``lisa`` at ``on_step_end``."""

    def __init__(self, policy: LayerSelectPolicy, *, dataset: Any = None, collator: Any = None,
                 batch_size: int = 1, output_dir: Optional[str] = None,
                 console: Any = None) -> None:
        self.policy = policy
        self.dataset = dataset
        self.collator = collator
        self.batch_size = int(batch_size)
        self.output_dir = output_dir
        self.console = console
        self.rng = random.Random(policy.seed)
        self.by_layer: Optional[Dict[int, List[Tuple[str, Any]]]] = None
        self.selection: Optional[Selection] = None

    def on_train_begin(self, args: Any, state: Any, control: Any = None, **kwargs: Any) -> Any:
        model = kwargs.get("model")
        if model is None or self.selection is not None:
            return control
        self.by_layer = adapter_params_by_layer(model)
        batches = None
        if self.policy.method in ("grad", "angular"):
            batches = probe_batches(self.dataset, self.collator, self.policy.probe_rows,
                                    self.batch_size)
        device = getattr(args, "device", None) or next(model.parameters()).device
        self.selection = choose(self.policy, model, self.by_layer, batches, device, self.rng)
        trainable = apply_layer_mask(self.by_layer, self.selection.layers)
        self.selection.backward_cut = cut_input_grads(model)
        self.selection.history.append((0, list(self.selection.layers)))
        write_record(self.selection, self.output_dir)
        if self.console is not None:
            self.console.print(
                f"[green]LoRA layers ({self.policy.method}):[/] training "
                f"{len(self.selection.layers)} of {len(self.by_layer)} layers "
                f"{self.selection.layers} ({trainable:,} adapter parameters); "
                f"backward {self.selection.backward_cut}"
            )
        return control

    def on_step_end(self, args: Any, state: Any, control: Any = None, **kwargs: Any) -> Any:
        if self.policy.method != "lisa" or self.selection is None or self.by_layer is None:
            return control
        step = int(getattr(state, "global_step", 0) or 0)
        if step <= 0 or step % self.policy.interval_steps != 0:
            return control
        before = set(self.selection.layers)
        layers = draw(sorted(self.by_layer), self.policy.k, self.rng)
        apply_layer_mask(self.by_layer, layers)
        gone = [p for idx in before - set(layers) for _n, p in self.by_layer[idx]]
        clear_optimizer_state(kwargs.get("optimizer"), gone)
        self.selection.layers = layers
        self.selection.history.append((step, list(layers)))
        write_record(self.selection, self.output_dir)
        return control


def __getattr__(name: str) -> Any:  # PEP 562: the HF base is resolved lazily
    if name == "LayerSelectCallback":
        ns = {k: v for k, v in vars(_LayerSelectBody).items()
              if k not in ("__dict__", "__weakref__")}
        cls = type(name, (_callback_base(),), ns)
        cls.__module__ = __name__
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def attach_layer_select_callback(trainer: Any, tcfg: Any, output_dir: Optional[str],
                                 console: Any = None) -> bool:
    """Attach the callback when ``training.lora_layers`` is set; True when attached."""
    cfg = getattr(tcfg, "lora_layers", None)
    if cfg is None:
        return False
    from soup_cli.utils import layer_select

    policy = LayerSelectPolicy(
        method=cfg.method,
        k=cfg.k,
        layers=tuple(cfg.layers or ()),
        probe_rows=int(cfg.probe_rows),
        interval_steps=int(cfg.interval_steps),
        seed=int(cfg.seed),
    )
    args = getattr(trainer, "args", None)
    trainer.add_callback(layer_select.LayerSelectCallback(
        policy,
        dataset=getattr(trainer, "train_dataset", None),
        collator=getattr(trainer, "data_collator", None),
        batch_size=int(getattr(args, "per_device_train_batch_size", 1) or 1),
        output_dir=output_dir,
        console=console,
    ))
    return True
