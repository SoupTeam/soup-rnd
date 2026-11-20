"""Phase profiler for one L2L step (PR #72 diagnostics). Timing wrappers only.

``install(clock)`` replaces a fixed list of methods with wrappers that call the original with
the same arguments and return its result unchanged, so the arithmetic is untouched. Each
wrapper pushes a named frame on a ``PhaseClock``. A frame's time is exclusive: while a child
frame runs, the parent's clock stops. The frames of one step therefore add up to the step wall
time, and the part no frame covers shows up as ``unattributed_run_glue`` instead of hiding.

On CUDA every push and pop calls ``torch.cuda.synchronize`` so asynchronous copies and
kernels are charged to the frame that issued them. That serialises work the real step
overlaps, so GPU phase times answer "what does each phase cost" and the sum of phases is
longer than an uninstrumented step. The harness reports that ratio as ``instrumentation_ratio``.

Frames from other threads are accepted only while the main thread is blocked inside
``autograd_backward`` (CUDA runs autograd on a worker thread). Anything else is counted
under ``background`` and stays out of the sum.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

ROOT = "unattributed_run_glue"


class _Frame:
    __slots__ = ("label", "last", "excl", "rows")

    def __init__(self, label: str, now: float) -> None:
        self.label = label
        self.last = now
        self.excl = 0.0
        self.rows: Optional[int] = None


class PhaseClock:
    """``now`` is the time source (``time.perf_counter``); tests pass a clock they move by hand."""

    def __init__(self, device: str = "cpu", now: Callable[[], float] = time.perf_counter) -> None:
        import torch

        self._torch = torch
        self._now = now
        self.cuda = str(device).startswith("cuda")
        self.main = threading.get_ident()
        self.stage = "fwd"
        self._lock = threading.Lock()
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.totals: Dict[str, float] = defaultdict(float)
        self.counts: Dict[str, int] = defaultdict(int)
        self.background: Dict[str, float] = defaultdict(float)
        self.gemm_rows: List[Tuple[str, int, float]] = []  # (kind, rows M, seconds)
        self.stage = "fwd"
        self._stack: List[_Frame] = []

    def _sync(self) -> None:
        if self.cuda:
            self._torch.cuda.synchronize()

    def _accepts(self) -> bool:
        if threading.get_ident() == self.main:
            return True
        # The prefetch worker runs beside the step, not inside it: its time overlaps the
        # others, so it is recorded under ``background`` and kept out of the sum.
        if threading.current_thread().name.startswith("expert-prefetch"):
            return False
        # Main is blocked inside the autograd engine, so every frame of the engine's worker
        # thread nests under it, however deep (disk read inside a load inside a lease).
        return any(frame.label.endswith("autograd_backward") for frame in self._stack)

    # ----------------------------------------------------------------- frames
    def push(self, label: str) -> Optional[_Frame]:
        if not self._accepts():
            return None
        self._sync()
        now = self._now()
        with self._lock:
            if self._stack:
                top = self._stack[-1]
                top.excl += now - top.last
            frame = _Frame(label, now)
            self._stack.append(frame)
        return frame

    def pop(self, frame: Optional[_Frame]) -> None:
        if frame is None:
            return
        self._sync()
        now = self._now()
        with self._lock:
            if not self._stack or self._stack[-1] is not frame:
                raise RuntimeError(f"phase frame {frame.label!r} closed out of order")
            self._stack.pop()
            frame.excl += now - frame.last
            self.totals[frame.label] += frame.excl
            self.counts[frame.label] += 1
            if frame.label.startswith("expert_gemm_"):
                self.gemm_rows.append((frame.label, frame.rows or 0, frame.excl))
            if self._stack:
                self._stack[-1].last = now

    @contextmanager
    def frame(self, label: str) -> Iterator[None]:
        handle = self.push(label)
        if handle is None:
            start = self._now()
            try:
                yield
            finally:
                self.background[label] += self._now() - start
            return
        try:
            yield
        finally:
            self.pop(handle)

    @contextmanager
    def run(self) -> Iterator[None]:
        """The root frame: everything inside it that no other frame claims."""
        self.reset()
        root = self.push(ROOT)
        try:
            yield
        finally:
            self.pop(root)

    def summary(self) -> Dict[str, Dict[str, float]]:
        return {
            label: {"seconds": self.totals[label], "calls": self.counts[label]}
            for label in sorted(self.totals, key=lambda name: -self.totals[name])
        }


def _wrap(
    clock: PhaseClock, owner: Any, name: str, label: Callable[..., str]
) -> Tuple[Any, str, Any]:
    original = getattr(owner, name)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with clock.frame(label(clock)):
            return original(*args, **kwargs)

    wrapper.__wrapped__ = original  # type: ignore[attr-defined]
    setattr(owner, name, wrapper)
    return owner, name, original


@contextmanager
def install(clock: PhaseClock) -> Iterator[None]:
    """Wrap the step's phase boundaries; restore every original on exit."""
    import torch
    import torch.nn.functional as functional

    from soup_cli.utils import expert_stream, l2l_activations, l2l_step, layer_stream_runtime
    from soup_cli.utils.expert_shard import ExpertShardReader

    undo: List[Tuple[Any, str, Any]] = []

    def fixed(text: str) -> Callable[[PhaseClock], str]:
        return lambda _clock: text

    def staged(text: str) -> Callable[[PhaseClock], str]:
        return lambda c: f"{c.stage}/{text}"

    pool_cls = expert_stream.ExpertBufferPool
    experts_cls = expert_stream.streamed_experts_class()

    # --- expert path -------------------------------------------------------------------
    undo.append(_wrap(clock, ExpertShardReader, "read_host", fixed("expert_disk_read")))
    undo.append(_wrap(clock, pool_cls, "_dequantize", fixed("expert_dequant_nf4")))
    undo.append(_wrap(clock, pool_cls, "_load", fixed("expert_load_h2d_and_misc")))
    if hasattr(pool_cls, "_read_host"):  # prefetch build: the wait for the background read
        undo.append(_wrap(clock, pool_cls, "_read_host", fixed("expert_read_wait")))
    undo.append(_wrap(clock, pool_cls, "_synchronize", fixed("expert_lease_sync")))
    undo.append(_wrap(clock, experts_cls, "forward", staged("expert_dispatch_overhead")))

    import soup_cli.utils.expert_transfer as transfer

    undo.append(_wrap(clock, transfer, "pack_into", fixed("expert_host_pack")))
    # ``_load_packed`` imports pack_into inside the function, so the module attribute wins.

    original_lease = pool_cls.lease

    def lease(self: Any, reader: Any, expert: int, **kwargs: Any) -> Any:
        @contextmanager
        def managed() -> Iterator[Any]:
            with original_lease(self, reader, expert, **kwargs) as buffers:
                if torch._C._current_autograd_node() is not None:
                    kind = "bwd_vjp"
                elif clock.stage == "bwd":
                    kind = "recompute"
                else:
                    kind = "fwd"
                handle = clock.push(f"expert_gemm_{kind}")
                try:
                    yield buffers
                finally:
                    clock.pop(handle)

        return managed()

    undo.append((pool_cls, "lease", original_lease))
    pool_cls.lease = lease  # type: ignore[assignment]

    # Rows (M) of every expert GEMM: the first ``linear`` of an expert is ``hidden @ gate_up.T``.
    original_linear = functional.linear

    def counting_linear(input: Any, weight: Any, bias: Any = None) -> Any:  # noqa: A002
        stack = clock._stack
        if stack and stack[-1].rows is None and stack[-1].label.startswith("expert_gemm_"):
            stack[-1].rows = int(input.reshape(-1, input.shape[-1]).shape[0])
        return original_linear(input, weight, bias)

    undo.append((functional, "linear", original_linear))
    functional.linear = counting_linear  # type: ignore[assignment]

    # --- layer path --------------------------------------------------------------------
    undo.append(_wrap(clock, l2l_step.L2LStep, "_layer", staged("attn_router_shared_other")))
    window = "_route_preserving_layer_window"
    undo.append(_wrap(clock, l2l_step.L2LStep, window, staged("mbr_join_split_glue")))
    for pool_name in ("LayerBufferPool", "LargeLayerBufferPool"):
        pool_type = getattr(layer_stream_runtime, pool_name)
        undo.append(_wrap(clock, pool_type, "load_async", fixed("layer_weight_load")))

    # --- activations, head, backward --------------------------------------------------
    store_cls = l2l_activations.ActivationStore
    undo.append(_wrap(clock, store_cls, "put", staged("activation_store")))
    undo.append(_wrap(clock, store_cls, "get", staged("activation_store")))
    original_begin = store_cls.begin_backward

    def begin_backward(self: Any) -> Any:
        clock.stage = "bwd"
        return original_begin(self)

    undo.append((store_cls, "begin_backward", original_begin))
    store_cls.begin_backward = begin_backward  # type: ignore[assignment]
    undo.append(_wrap(clock, torch.autograd, "backward", staged("autograd_backward")))

    try:
        yield
    finally:
        for owner, name, original in reversed(undo):
            setattr(owner, name, original)
        clock.stage = "fwd"
