"""Timer instruments for G7: host time per layer visit, per layer backward and per step.

``visit_profile.py`` installs these inside the arm's process, so no file in the shipped
source changes. Three parts:

- ``install_patches`` replaces up to seven points of the streamed layer runtime and puts
  every original back on ``remove``. The profile arm of ticket 04 reuses it with its own
  wrappers.
- ``Recorder`` keeps the visits, the layer backward windows and the step times of each step,
  and summarises the timed steps.
- ``timer_wrappers`` are the timer arm's wrappers: a monotonic nanosecond clock around the
  layer body, and gradient hooks on the layer's output and input. They add no synchronize.

A layer backward is the time between the gradient hook on the layer's output and the one on
its input. Under the non-reentrant checkpoint the recompute runs inside that window, so the
recompute visit is subtracted from it: a layer backward belongs to no visit.

Benchmark harness code, not shipped. ``torch`` is imported inside the functions that need it.
"""

from __future__ import annotations

import contextlib
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Mapping, Sequence, Set, Tuple

#: A wrapper takes the original callable and returns the one that replaces it.
Wrapper = Callable[[Callable[..., Any]], Callable[..., Any]]

#: Where ``install_patches`` can reach. The first three are methods on the cached layer class,
#: the next two are free variables of those methods, the last two are instance attributes of
#: the shared pool and prefetcher.
PATCH_POINTS = (
    "body",
    "substituted_weights",
    "forward",
    "functional_call",
    "checkpoint",
    "pool_wait",
    "prefetcher_advance",
)

_METHODS = {"body": "_body", "substituted_weights": "_substituted_weights", "forward": "forward"}
#: point -> (method whose closure holds the variable, the variable)
_POINT_OF_METHOD = {name: point for point, name in _METHODS.items()}
_CELLS = {
    "functional_call": ("_body", "functional_call"),
    "checkpoint": ("forward", "checkpoint"),
}
_OBJECTS = {"pool_wait": ("pool", "wait"), "prefetcher_advance": ("prefetcher", "advance")}

#: Classes and objects under a patch now, so a second install fails instead of stacking.
_PATCHED: Set[int] = set()


class Patches:
    """The undo list of one ``install_patches`` call."""

    def __init__(self) -> None:
        self._undo: List[Callable[[], None]] = []

    def remove(self) -> None:
        """Put every original back. Safe to call twice."""
        while self._undo:
            self._undo.pop()()

    def __enter__(self) -> "Patches":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.remove()


def _closure_cell(function: Any, name: str) -> Any:
    names = function.__code__.co_freevars
    if name not in names:
        raise RuntimeError(f"{function.__qualname__} has no free variable {name!r}")
    return function.__closure__[names.index(name)]


def _restorer(obj: Any, name: str, original: Any, had_own: bool) -> Callable[[], None]:
    """An undo step: put ``original`` back, or delete the attribute if the object had none."""

    def undo() -> None:
        if had_own:
            setattr(obj, name, original)
        else:
            delattr(obj, name)

    return undo


def install_patches(layers: Sequence[Any], wrappers: Mapping[str, Wrapper]) -> Patches:
    """Wrap the runtime points named in ``wrappers`` for every layer in ``layers``.

    ``layers`` are the streamed decoder layers, reached through the base model's layer
    container (the wrapper overrides module iteration). The layer class is a cached singleton,
    so patching it reaches every layer. The shared pool and prefetcher are found on the layers
    and patched once each. Raises ValueError for an unknown point and RuntimeError if the
    class or an object is already patched. A failed install leaves nothing patched.
    """
    unknown = sorted(set(wrappers) - set(PATCH_POINTS))
    if unknown:
        raise ValueError(f"unknown patch points {unknown}; choose from {list(PATCH_POINTS)}")
    if not layers:
        raise ValueError("no layers to patch")
    cls = type(layers[0])
    patches = Patches()
    claimed: List[int] = []

    def claim(obj: Any) -> None:
        if id(obj) in _PATCHED:
            raise RuntimeError(f"{obj!r} is already patched; remove the first patches")
        _PATCHED.add(id(obj))
        claimed.append(id(obj))

    def release() -> None:
        for key in claimed:
            _PATCHED.discard(key)

    patches._undo.append(release)
    try:
        claim(cls)
        # Read every original before any patch lands, so a closure lookup never sees a wrapper.
        originals = {point: cls.__dict__[name] for point, name in _METHODS.items()}
        for point, wrap in wrappers.items():
            if point in _METHODS:
                name = _METHODS[point]
                setattr(cls, name, wrap(originals[point]))
                patches._undo.append(_restorer(cls, name, originals[point], True))
            elif point in _CELLS:
                method, variable = _CELLS[point]
                cell = _closure_cell(originals[_POINT_OF_METHOD[method]], variable)
                original = cell.cell_contents
                cell.cell_contents = wrap(original)
                patches._undo.append(_restorer(cell, "cell_contents", original, True))
            else:
                attr, method = _OBJECTS[point]
                seen: Dict[int, Any] = {}
                for layer in layers:
                    seen.setdefault(id(getattr(layer, attr)), getattr(layer, attr))
                for obj in seen.values():
                    claim(obj)
                    had_own = method in vars(obj)
                    original = getattr(obj, method)
                    setattr(obj, method, wrap(original))
                    patches._undo.append(_restorer(obj, method, original, had_own))
    except BaseException:
        patches.remove()
        raise
    return patches


# --- the recorder -----------------------------------------------------------------------


@dataclass
class StepRecord:
    """One training step: visits as (kind, layer, start ns, end ns), windows as (layer, start
    ns, end ns), LoRA gradient accumulations as (layer, ns), and the step time in ns."""

    visits: List[Tuple[str, int, int, int]] = field(default_factory=list)
    windows: List[Tuple[int, int, int]] = field(default_factory=list)
    lora_marks: List[Tuple[int, int]] = field(default_factory=list)
    started: int = 0
    ns: int = 0


def _ms_stats(values_ns: Sequence[float]) -> Dict[str, Any]:
    values = [round(value / 1e6, 6) for value in values_ns]
    return {"median": statistics.median(values), "values": values}


class Recorder:
    """Collects the timings of one run. ``clock`` returns monotonic nanoseconds."""

    def __init__(self, clock: Callable[[], int] = time.perf_counter_ns) -> None:
        self.clock = clock
        self.steps: List[StepRecord] = []
        #: Above zero while a layer's ``forward`` is on the stack. A body visit made then is a
        #: forward visit, and one made from the backward pass is a recompute visit.
        self.in_forward = 0
        self._open: Dict[int, int] = {}

    def new_step_record(self) -> StepRecord:
        step = StepRecord()
        self.steps.append(step)
        self._open.clear()
        return step

    def begin_step(self) -> None:
        self.new_step_record().started = self.clock()

    def end_step(self) -> float:
        """Close the step and return its seconds."""
        step = self.steps[-1]
        step.ns = self.clock() - step.started
        return step.ns / 1e9

    def add_visit(self, kind: str, idx: int, start: int, end: int) -> None:
        self.steps[-1].visits.append((kind, idx, start, end))

    def window_start(self, idx: int) -> None:
        self._open[idx] = self.clock()

    def window_end(self, idx: int) -> None:
        end = self.clock()
        start = self._open.pop(idx, None)
        if start is not None:
            self.steps[-1].windows.append((idx, start, end))

    def mark_lora(self, idx: int) -> None:
        if self.steps:
            self.steps[-1].lora_marks.append((idx, self.clock()))

    def summary(self, warmup: int, timed: int) -> Dict[str, Any]:
        """Per-visit, per-backward and per-step times in ms for the ``timed`` steps after
        ``warmup``: the median over layers and steps, and the raw values."""
        if len(self.steps) < warmup + timed:
            raise ValueError(f"need {warmup + timed} steps, recorded {len(self.steps)}")
        steps = self.steps[warmup : warmup + timed]
        forward: List[int] = []
        recompute: List[int] = []
        backward: List[int] = []
        two_visits = True
        for step in steps:
            counts: Dict[Tuple[str, int], int] = {}
            for kind, idx, start, end in step.visits:
                (forward if kind == "forward" else recompute).append(end - start)
                counts[(kind, idx)] = counts.get((kind, idx), 0) + 1
            layers = {idx for _, idx in counts}
            two_visits = two_visits and all(
                counts.get(("forward", idx)) == 1 and counts.get(("recompute", idx)) == 1
                for idx in layers
            )
            for idx, start, end in step.windows:
                inside = sum(
                    e - s
                    for kind, i, s, e in step.visits
                    if kind == "recompute" and i == idx and start <= s and e <= end
                )
                backward.append(end - start - inside)
        return {
            "forward_visit_ms": _ms_stats(forward),
            "recompute_visit_ms": _ms_stats(recompute),
            "layer_backward_ms": _ms_stats(backward),
            "step_ms": _ms_stats([step.ns for step in steps]),
            "two_visits_per_layer": two_visits,
        }

    def lora_window_report(self) -> Dict[str, Any]:
        """Whether the last step's LoRA gradient accumulations fell inside the layer backward
        windows. A mark counts as inside when its layer has a window that contains its time."""
        step = self.steps[-1]
        inside = 0
        for idx, at in step.lora_marks:
            if any(i == idx and start <= at <= end for i, start, end in step.windows):
                inside += 1
        outside = len(step.lora_marks) - inside
        if not step.lora_marks or outside == 0:
            state = "inside"
        elif inside == 0:
            state = "outside"
        else:
            state = "mixed"
        return {"inside": inside, "outside": outside, "state": state}


# --- the timer arm's wrappers -----------------------------------------------------------


def _first_tensor(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, torch.Tensor):
                return item
    return None


def timer_wrappers(recorder: Recorder) -> Dict[str, Wrapper]:
    """The timer arm: the clock around the layer body, and gradient hooks around the layer.

    The ``forward`` wrapper marks forward visits and registers a hook on the layer's output
    (the layer backward starts when its gradient arrives) and on its input (it ends when the
    input gradient is ready). A layer input that needs no gradient gets no input hook, so
    that layer has no backward time.
    """
    import torch

    clock = recorder.clock

    def wrap_body(orig: Callable[..., Any]) -> Callable[..., Any]:
        def body(self: Any, *args: Any, **kwargs: Any) -> Any:
            kind = "forward" if recorder.in_forward else "recompute"
            start = clock()
            try:
                return orig(self, *args, **kwargs)
            finally:
                recorder.add_visit(kind, self.idx, start, clock())

        return body

    def wrap_forward(orig: Callable[..., Any]) -> Callable[..., Any]:
        def forward(self: Any, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
            recorder.in_forward += 1
            try:
                out = orig(self, hidden_states, *args, **kwargs)
            finally:
                recorder.in_forward -= 1
            if torch.is_grad_enabled():
                idx = self.idx
                head = _first_tensor(out)
                if (
                    isinstance(hidden_states, torch.Tensor)
                    and hidden_states.requires_grad
                    and head is not None
                    and head.requires_grad
                ):
                    head.register_hook(lambda grad: recorder.window_start(idx))
                    hidden_states.register_hook(lambda grad: recorder.window_end(idx))
            return out

        return forward

    return {"body": wrap_body, "forward": wrap_forward}


@contextlib.contextmanager
def watch_lora_gradients(layers: Sequence[Any], recorder: Recorder) -> Iterator[None]:
    """Mark the time each trainable parameter of each layer finishes accumulating its gradient.

    Used for one extra step after the timed ones, so the hooks never touch a timed step.
    """
    handles = []
    for layer in layers:
        for param in layer.inner.parameters():
            if param.requires_grad:
                handles.append(
                    param.register_post_accumulate_grad_hook(
                        lambda _param, idx=layer.idx: recorder.mark_lora(idx)
                    )
                )
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
