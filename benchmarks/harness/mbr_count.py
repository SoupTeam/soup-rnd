"""Routed-expert ``F.linear`` counter for one run of the ticket 15 campaign (harness only).

``torch.profiler`` cannot count the routed-expert GEMMs of a whole step on this host: a CPU
activity trace with shapes of the (8,64) step takes 7.9 GiB of RSS and is OOM-killed there, and
the (64,64) step killed even the CUDA-only trace at about 1.1M kernel events. A wrapper around
``torch.nn.functional.linear`` counts the same calls in constant memory: measured 6.6k calls per
step for C and 34.8k for S at (8,64), 15.0k and 538.7k at (64,64).

Pure apart from ``sys._getframe``: the tests drive it with fake tensors and real frames, so they
need no torch and no GPU. The summmarizers price the counted calls with the measured per-M table
of ``sweep_rows.py``. Benchmark code, not shipped.
"""

from __future__ import annotations

import functools
import statistics
import sys
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence

from benchmarks.harness.mbr_trace import EXPERT_WEIGHT_SHAPES

# Rows of one routed expert per projection: the same two shapes the trace selects on.
EXPERT_ROWS: Dict[str, tuple] = EXPERT_WEIGHT_SHAPES
# The walk from the wrapper to ``expert_mlp``: two frames in the forward, three in the backward
# recompute. Six covers both and stops a same-named function further up the stack.
FRAME_LIMIT = 6
PHASES = ("forward", "backward")


def frame_names(frame: Any, limit: int = FRAME_LIMIT) -> List[str]:
    """The code names of at most ``limit`` frames, innermost first, starting at ``frame``."""
    names: List[str] = []
    while frame is not None and len(names) < limit:
        names.append(str(frame.f_code.co_name))
        frame = frame.f_back
    return names


def _projection(weight_shape: Any) -> Optional[str]:
    """The routed-expert projection of a weight, or None for every other weight."""
    try:
        shape = tuple(int(size) for size in weight_shape)
    except (TypeError, ValueError):
        return None
    for projection, expert_shape in EXPERT_ROWS.items():
        if shape == expert_shape:
            return projection
    return None


def _rows(x: Any) -> int:
    """Rows of a ``F.linear`` input: every axis but the feature one, so ``(1,M,K)`` gives ``M``."""
    feature = int(x.shape[-1])
    return int(x.numel()) // feature if feature else 0


def _phase(names: Sequence[str]) -> str:
    """The phase of the node: the nearest enclosing ``forward``/``backward`` frame, else ``?``."""
    return next((name for name in names if name in PHASES), "?")


def _operand(args: Sequence[Any], kwargs: Dict[str, Any], position: int, name: str) -> Any:
    """One ``F.linear`` operand, whether the caller passed it by position or by keyword."""
    if name in kwargs:
        return kwargs[name]
    return args[position] if len(args) > position else None


class ExpertCounter:
    """Counts routed-expert ``F.linear`` calls: rows, projection and the phase of the node."""

    def __init__(self) -> None:
        self.calls = 0
        self.rows: Counter = Counter()
        self.phases: Counter = Counter()
        self.projections: Counter = Counter()

    def is_expert(self, weight_shape: Any, function_name: str,
                  frame_names: Sequence[str]) -> bool:
        """True when the call came from ``expert_mlp`` and the weight is one of the two shapes."""
        if "expert_mlp" not in (function_name, *frame_names):
            return False
        return _projection(weight_shape) is not None

    def record(self, *, rows: int, projection: str, phase: str) -> None:
        """One routed-expert GEMM: its row count, its projection and the phase it ran in."""
        self.calls += 1
        self.rows[int(rows)] += 1
        self.phases[phase] += 1
        self.projections[projection] += 1

    def snapshot(self) -> Dict[str, Any]:
        """The counters of one step: calls, rows ascending, phases (``projections`` stays live)."""
        return {"calls": int(self.calls),
                "rows": {int(rows): int(count) for rows, count in sorted(self.rows.items())},
                "phases": dict(self.phases)}

    def reset(self) -> None:
        """Start the next step's counters from zero."""
        self.calls = 0
        self.rows.clear()
        self.phases.clear()
        self.projections.clear()

    def wrap(self, original: Callable[..., Any]) -> Callable[..., Any]:
        """``torch.nn.functional.linear`` replacement that counts the routed-expert calls.

        It walks at most ``FRAME_LIMIT`` frames for ``expert_mlp``, takes the phase from the
        nearest enclosing ``forward``/``backward`` frame and records the call when the weight
        shape is one of the two routed-expert projections, then calls ``original`` unchanged.
        """

        @functools.wraps(original)
        def counting_linear(*args: Any, **kwargs: Any) -> Any:
            names = frame_names(sys._getframe(1))
            weight = _operand(args, kwargs, 1, "weight")
            if self.is_expert(getattr(weight, "shape", ()), names[0] if names else "",
                              names[1:]):
                x = _operand(args, kwargs, 0, "input")
                projection = _projection(weight.shape)
                rows = _rows(x)
                if projection is not None and rows:
                    self.record(rows=rows, projection=projection, phase=_phase(names))
            return original(*args, **kwargs)

        return counting_linear


def median_calls(snapshots: Sequence[Dict[str, Any]]) -> int:
    """Median of the per-step call counts, rounded to an int (0 without steps)."""
    values = [int(snapshot["calls"]) for snapshot in snapshots]
    return int(round(statistics.median(values))) if values else 0


def merged_rows(snapshots: Sequence[Dict[str, Any]]) -> Dict[int, int]:
    """Rows -> calls over the steps, ascending by row count."""
    merged: Counter = Counter()
    for snapshot in snapshots:
        for rows, count in (snapshot.get("rows") or {}).items():
            merged[int(rows)] += int(count)
    return dict(sorted(merged.items()))


def expert_seconds(rows: Dict[int, int], per_row_us: Dict[int, float]) -> Optional[float]:
    """Σ calls(M) × the microsecond price of one expert GEMM at M.

    ``per_row_us`` prices one call at row count M (the per-M table of ``sweep_rows.py``), not one
    row: ``expert_seconds({8: 3}, {8: 120.0})`` is 3 × 120 µs. None when a row count has no
    measured price: the model reports the missing rows rather than guessing a price for them.
    """
    prices = {int(row_count): float(price) for row_count, price in per_row_us.items()}
    total_us = 0.0
    for row_count, calls in rows.items():
        price = prices.get(int(row_count))
        if price is None:
            return None
        total_us += int(calls) * price
    return total_us / 1e6
