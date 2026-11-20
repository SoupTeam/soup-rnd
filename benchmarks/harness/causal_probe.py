"""Causal profiling by delay injection (harness only): which CPU time is on the critical path.

``CausalProbe.wrap`` replaces ``owner.attr`` by a wrapper that counts calls and, while a delay is
set for its point, spins ``delay`` us before the call (``spin`` should release the GIL, as the C++
``spin_us`` of the extension does, so only the calling thread is held up). The step is run with the
delays off and on in turn; the extra seconds per injected second say how much of the time at that
point the step waits for.
"""

from __future__ import annotations

import contextlib
import functools
from typing import Any, Callable, Dict, Iterator, List, Tuple


def parse_causal_spec(spec: str) -> Dict[str, float]:
    """``"expert_fwd=250,layer=40"`` -> ``{"expert_fwd": 250.0, "layer": 40.0}``."""
    out: Dict[str, float] = {}
    for item in filter(None, (part.strip() for part in spec.split(","))):
        name, sep, value = item.partition("=")
        if not sep or not name:
            raise ValueError(f"causal spec items look like name=microseconds, got {item!r}")
        delay = float(value)
        if delay < 0:
            raise ValueError(f"delay of {name!r} must not be negative")
        out[name] = delay
    return out


class CausalProbe:
    def __init__(self, spin: Callable[[float], None]) -> None:
        self._spin = spin
        self._active: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
        self._undo: List[Tuple[Any, str, Any]] = []
        self._points: Dict[str, bool] = {}

    def wrap(self, name: str, owner: Any, attr: str) -> None:
        if name in self._points:
            raise ValueError(f"point {name!r} is already wrapped")
        original = owner.__dict__[attr]
        function = original.__func__ if isinstance(original, staticmethod) else original
        probe = self

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            probe._counts[name] += 1
            delay = probe._active.get(name, 0.0)
            if delay > 0:
                probe._spin(delay)
            return function(*args, **kwargs)

        self._counts[name] = 0
        self._points[name] = True
        setattr(owner, attr, staticmethod(wrapper) if isinstance(original, staticmethod)
                else wrapper)
        self._undo.append((owner, attr, original))

    @contextlib.contextmanager
    def delays(self, spec: Dict[str, float]) -> Iterator[None]:
        unknown = sorted(set(spec) - set(self._points))
        if unknown:
            raise ValueError(f"no such causal point: {unknown}")
        self._active = dict(spec)
        try:
            yield
        finally:
            self._active = {}

    def counts(self) -> Dict[str, int]:
        return dict(self._counts)

    def restore(self) -> None:
        for owner, attr, original in reversed(self._undo):
            setattr(owner, attr, original)
        self._undo.clear()
        self._points.clear()
