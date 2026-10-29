"""Named profiler ranges for G7's profile arm.

``install_ranges`` puts ``torch.profiler.record_function`` ranges around the parts of a
streamed layer visit, with ``visit_timer.install_patches`` (no shipped source changes).
``trace_analyser.analyse_trace`` cuts the exported trace by these names:

- ``g7::forward_visit`` and ``g7::recompute_visit``: the layer body;
- ``g7::weight_prep``: the pool wait, the prefetcher advance and the substitution mapping;
- ``g7::functional_call``: the whole call, and ``g7::layer_call`` the call into the layer
  inside it (a wrapper on the inner module's ``forward``, installed once, so it costs nothing
  inside ``functional_call``'s own machinery; a ``with`` block, because the non-reentrant
  checkpoint stops a recompute by raising out of the layer and a forward hook would not run);
- ``g7::layer_backward``: from the gradient hook on the layer's output to the one on its
  input. The recompute visit runs inside it.
- ``g7::step`` is made by the worker (``step_range``).

Benchmark harness code, not shipped. ``torch`` is imported inside the functions.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence

import trace_analyser
import visit_timer

RANGES = trace_analyser.RANGES


def _record_function() -> Callable[[str], Any]:
    from torch.profiler import record_function

    return record_function


@contextlib.contextmanager
def step_range() -> Iterator[None]:
    """The ``g7::step`` range of one training step."""
    with _record_function()(RANGES.step):
        yield


class RangeState:
    """What the wrappers share: the forward depth, and the ranges opened by one hook that the
    other hook closes."""

    def __init__(self) -> None:
        self.in_forward = 0
        self.backward_open: Dict[int, Any] = {}

    def close_all(self) -> None:
        for held in list(self.backward_open.values()):
            held.__exit__(None, None, None)
        self.backward_open.clear()


def range_wrappers(state: RangeState) -> Dict[str, visit_timer.Wrapper]:
    """Wrappers for the patch points that carry a range."""
    import torch

    record_function = _record_function()

    def around(name: str) -> visit_timer.Wrapper:
        def wrap(orig: Callable[..., Any]) -> Callable[..., Any]:
            def ranged(*args: Any, **kwargs: Any) -> Any:
                with record_function(name):
                    return orig(*args, **kwargs)

            return ranged

        return wrap

    def wrap_body(orig: Callable[..., Any]) -> Callable[..., Any]:
        def body(self: Any, *args: Any, **kwargs: Any) -> Any:
            name = RANGES.forward_visit if state.in_forward else RANGES.recompute_visit
            with record_function(name):
                return orig(self, *args, **kwargs)

        return body

    def start_backward(idx: int) -> None:
        held = record_function(RANGES.layer_backward)
        held.__enter__()
        state.backward_open[idx] = held

    def end_backward(idx: int) -> None:
        held = state.backward_open.pop(idx, None)
        if held is not None:
            held.__exit__(None, None, None)

    def wrap_forward(orig: Callable[..., Any]) -> Callable[..., Any]:
        def forward(self: Any, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
            state.in_forward += 1
            try:
                out = orig(self, hidden_states, *args, **kwargs)
            finally:
                state.in_forward -= 1
            if torch.is_grad_enabled():
                idx = self.idx
                head = visit_timer._first_tensor(out)
                if (
                    isinstance(hidden_states, torch.Tensor)
                    and hidden_states.requires_grad
                    and head is not None
                    and head.requires_grad
                ):
                    head.register_hook(lambda grad: start_backward(idx))
                    hidden_states.register_hook(lambda grad: end_backward(idx))
            return out

        return forward

    return {
        "body": wrap_body,
        "forward": wrap_forward,
        "substituted_weights": around(RANGES.weight_prep),
        "pool_wait": around(RANGES.weight_prep),
        "prefetcher_advance": around(RANGES.weight_prep),
        "functional_call": around(RANGES.functional_call),
    }


class RangePatches:
    """The patches of ``install_ranges``. ``remove`` undoes all of them."""

    def __init__(self, patches: visit_timer.Patches, inners: Sequence[Any], state: RangeState):
        self._patches = patches
        self._inners = list(inners)
        self._state = state

    def remove(self) -> None:
        while self._inners:
            del self._inners.pop().forward
        self._state.close_all()
        self._patches.remove()

    def __enter__(self) -> "RangePatches":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.remove()


def _layered(
    ranges: Mapping[str, visit_timer.Wrapper],
    outer: Mapping[str, visit_timer.Wrapper],
    inner: Mapping[str, visit_timer.Wrapper],
) -> Dict[str, visit_timer.Wrapper]:
    """One wrapper per point: ``outer`` around the range around ``inner`` around the original.
    A point named only in ``outer`` or ``inner`` gets that wrapper alone."""

    def stack(*wrappers: Optional[visit_timer.Wrapper]) -> visit_timer.Wrapper:
        chain = [wrap for wrap in wrappers if wrap is not None]

        def wrap(orig: Callable[..., Any]) -> Callable[..., Any]:
            for one in reversed(chain):
                orig = one(orig)
            return orig

        return wrap

    points = [*ranges, *(p for p in (*outer, *inner) if p not in ranges)]
    return {p: stack(outer.get(p), ranges.get(p), inner.get(p)) for p in points}


def install_ranges(
    layers: Sequence[Any],
    *,
    outer: Optional[Mapping[str, visit_timer.Wrapper]] = None,
    inner: Optional[Mapping[str, visit_timer.Wrapper]] = None,
) -> RangePatches:
    """Install the profile arm's ranges on every layer in ``layers``.

    The control arm adds its own wrappers through ``outer`` (the timer's, around the ranges)
    and ``inner`` (the injections, inside them). Both are ``install_patches`` wrappers.
    """
    record_function = _record_function()
    state = RangeState()
    patches = visit_timer.install_patches(
        layers, _layered(range_wrappers(state), outer or {}, inner or {})
    )
    inners: List[Any] = []

    def ranged_forward(original: Callable[..., Any]) -> Callable[..., Any]:
        def forward(*args: Any, **kwargs: Any) -> Any:
            with record_function(RANGES.layer_call):
                return original(*args, **kwargs)

        return forward

    try:
        for layer in layers:
            module = layer.inner
            if "forward" in vars(module):
                raise RuntimeError(f"{module!r} already has an instance forward")
            module.forward = ranged_forward(module.forward)
            inners.append(module)
    except BaseException:
        RangePatches(patches, inners, state).remove()
        raise
    return RangePatches(patches, inners, state)
