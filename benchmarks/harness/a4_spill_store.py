#!/usr/bin/env python3
"""A4: an L2L activation store that obeys an ``ActivationPlan``, and cleans up after itself.

``BudgetedStore`` builds the existing ``l2l_activations.ActivationStore`` with
the plan's ``spill_layers``, ring depth and pinning, and adds what A4 needs
around it without changing that class:

* the cap check: the host bytes the store really holds (each pinned
  allocation at its power-of-two block, rings at chunk + one sector) must equal
  the plan and stay at or below its budget; on CUDA the caching host
  allocator's own counter is read before and after construction as well;
* a named ``SpillDiskFullError`` when the spill volume runs out of space (ENOSPC,
  or EDQUOT), with the arithmetic in the message, raised only after the spill
  file is removed and the page-locked blocks are returned;
* release on every exit: ``close()`` from ``__exit__`` on an exception, a
  KeyboardInterrupt (Ctrl-C) or a normal end. Before the cache release, the
  traceback frames that still hold a buffer are cleared: those of the exception
  leaving the ``with`` block (a Ctrl-C inside ``put``/``get`` holds a staging
  buffer there) and of a failed writer or reader thread (any error, not only
  ENOSPC). A failed constructor closes the spill file it already opened
  (``ActivationStore.__init__`` opens the file before it pins the rings, and
  would leave it behind if a ring failed), and removes a file its opener created
  before failing (on Windows ``truncate`` extends a file by writing zeros, so a
  full volume fails there). ``release_cached_pinned_memory``
  (layer_stream_runtime.py:3351, shipped for #901) then returns the freed blocks
  torch's host cache would otherwise keep;
* on POSIX the spill file is unlinked as soon as its handles are open, so even
  a SIGKILL from the cgroup OOM killer leaves no file: the kernel frees the
  blocks when the process dies. Windows cannot unlink an open file; there the
  file is removed on close and a failure to remove it is reported by path;
* ``TimedSpill`` times every write and read on the drive, so spill write and
  read GB/s come from the step itself.

The store interface is the one ``l2l_schedule.L2LStep.run`` drives: ``put``,
``get``, ``begin_backward``, ``end_step``, ``close``, ``stats``, ``k``,
``n_layers``, ``chunk_bytes``.

Benchmark harness code, not shipped. Standard library at the top level; torch
inside functions.
"""

from __future__ import annotations

import errno
import itertools
import os
import shutil
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import a4_activation_budget as budget
import l2l_activations

SPILL_ENOSPC = "SPILL_ENOSPC"
PLAN_MISMATCH = "PLAN_MISMATCH"
BACKENDS = ("direct", "buffered")
_DISK_FULL_ERRNOS = {errno.ENOSPC} | ({errno.EDQUOT} if hasattr(errno, "EDQUOT") else set())
#: ERROR_HANDLE_DISK_FULL and ERROR_DISK_FULL; CPython maps both to ENOSPC as well.
_DISK_FULL_WINERRORS = {39, 112}
_COUNTER = itertools.count()


class SpillDiskFullError(RuntimeError):
    """The spill volume refused a write. The message carries the arithmetic."""

    code = SPILL_ENOSPC


class PlanMismatchError(RuntimeError):
    """The store holds other bytes than its plan says."""

    code = PLAN_MISMATCH


def is_disk_full(exc: BaseException) -> Optional[OSError]:
    """The OSError in ``exc``'s cause chain that means "no space", or None."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, OSError) and (
            exc.errno in _DISK_FULL_ERRNOS or getattr(exc, "winerror", None) in _DISK_FULL_WINERRORS
        ):
            return exc
        exc = exc.__cause__ or exc.__context__
    return None


def _clear_chain(exc: Optional[BaseException]) -> None:
    """Drop the locals of every finished frame in ``exc``'s chain. The spill writer's
    frame holds the staging buffer it failed to write, and a frame of a failed
    constructor holds the chunks it pinned; neither can return to the OS while an
    exception that is still being handled keeps those frames alive."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        traceback.clear_frames(exc.__traceback__)
        exc = exc.__cause__ or exc.__context__


def _b(value: Optional[int]) -> str:
    return "unknown" if value is None else f"{value:,} B ({value / 2**30:,.2f} GiB)"


def free_bytes(directory: str) -> Optional[int]:
    try:
        return int(shutil.disk_usage(directory).free)
    except OSError:
        return None


def _remove_partial(path: str) -> Optional[str]:
    """Remove a spill file its opener left behind; the path when it cannot be removed."""
    if not os.path.exists(path):
        return None
    try:
        os.remove(path)
    except OSError:
        return path
    return None


def open_file_bytes(backend: Any) -> Optional[int]:
    """``fstat`` size of a spill backend's open writer handle; None when it has none.

    The name is not used: on POSIX ``open_spill`` unlinks it as soon as the handles
    are open, and the file then lives only through them."""
    handle = getattr(backend, "_writer", None)
    if handle is None or handle.closed:
        return None
    return int(os.fstat(handle.fileno()).st_size)


class TimedSpill(l2l_activations.SpillBackend):
    """A spill backend with every write and read timed and counted.

    ``write_s``/``read_s`` are the seconds the writer and the reader thread
    spent inside the drive calls, so ``bytes / seconds`` is the drive's rate
    while it was busy with spill I/O, not an average over the step.
    """

    def __init__(self, inner: l2l_activations.SpillBackend):
        self.inner = inner
        self.direct = bool(inner.direct)
        self.path = inner.path
        self.write_s = 0.0
        self.read_s = 0.0
        self.bytes_written = 0
        self.bytes_read = 0
        self.failed: Optional[Tuple[str, int, int]] = None
        self._lock = threading.Lock()

    @property
    def left_behind(self) -> Optional[str]:
        return self.inner.left_behind

    def open_file_bytes(self) -> Optional[int]:
        """Size of the open spill file read through its own handle (``fstat``), so it
        holds on POSIX, where the name is unlinked right after opening. None when no
        handle is open."""
        return open_file_bytes(self.inner)

    def _timed(self, op: str, offset: int, buf: Any) -> None:
        started = time.perf_counter()
        try:
            getattr(self.inner, op)(offset, buf)
        except BaseException:
            self.failed = (op, int(offset), int(buf.numel()))
            raise
        elapsed = time.perf_counter() - started
        with self._lock:
            if op == "write":
                self.write_s += elapsed
                self.bytes_written += int(buf.numel())
            else:
                self.read_s += elapsed
                self.bytes_read += int(buf.numel())

    def write(self, offset: int, buf: Any) -> None:
        self._timed("write", offset, buf)

    def read(self, offset: int, buf: Any) -> None:
        self._timed("read", offset, buf)

    def close(self, remove: bool = True) -> None:
        self.inner.close(remove=remove)

    def io(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "write_s": self.write_s, "read_s": self.read_s,
                "bytes_written": self.bytes_written, "bytes_read": self.bytes_read,
                "write_gb_per_s": self.bytes_written / self.write_s / 1e9 if self.write_s else None,
                "read_gb_per_s": self.bytes_read / self.read_s / 1e9 if self.read_s else None,
            }


def open_spill(kind: str, path: str, size: int) -> l2l_activations.SpillBackend:
    """A spill file of ``size`` bytes at ``path``: ``direct`` (unbuffered, the measured
    arms) or ``buffered`` (page cache; CPU tests and the labelled CPU pilot only).

    On POSIX the name is unlinked once the handles are open, so the blocks go back
    to the volume when the process exits however it exits.
    """
    if kind not in BACKENDS:
        raise ValueError(f"spill backend must be one of {BACKENDS}; got {kind!r}")
    made = (l2l_activations.DirectSpillFile if kind == "direct"
            else l2l_activations.BufferedSpillFile)(path, size)
    if os.name == "posix":
        try:
            os.unlink(path)
        except OSError:
            made.close(remove=True)
            raise
    return made


def host_allocator_bytes() -> Optional[int]:
    """Bytes torch's caching host allocator holds from CUDA (rounded blocks, active +
    cached), or None without CUDA."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    stats = torch.cuda.host_memory_stats()
    for key in ("allocated_bytes.current", "allocated_bytes.all.current"):
        if key in stats:
            return int(stats[key])
    return None


def release_pinned_cache() -> int:
    """The shipped #901 release of torch's cached page-locked blocks; 0 without CUDA."""
    from soup_cli.utils.layer_stream_runtime import release_cached_pinned_memory

    return int(release_cached_pinned_memory())


class BudgetedStore:
    """An ``ActivationStore`` built from an ``ActivationPlan``, with the cap check,
    named disk-full handling and release on every exit. Use as a context manager."""

    def __init__(
        self,
        plan: budget.ActivationPlan,
        *,
        chunk_shape: Tuple[int, ...],
        dtype: Any,
        device: Any,
        spill_dir: Optional[str] = None,
        backend: str = "direct",
        opener: Callable[[str, str, int], l2l_activations.SpillBackend] = open_spill,
        release: Callable[[], int] = release_pinned_cache,
        free_bytes_at_plan: Optional[int] = None,
    ):
        import torch

        if plan.spill_layers and spill_dir is None:
            raise ValueError("a plan that spills needs a spill_dir")
        element = torch.empty((), dtype=dtype).element_size()
        chunk = element
        for dim in chunk_shape:
            chunk *= int(dim)
        if chunk != plan.chunk_bytes:
            raise PlanMismatchError(
                f"{PLAN_MISMATCH}: a {tuple(chunk_shape)} {dtype} chunk is {chunk:,} B, "
                f"but the plan was made for {plan.chunk_bytes:,} B"
            )
        self.plan = plan
        self.spill_dir = spill_dir
        self.backend = backend
        self.free_bytes_at_plan = free_bytes_at_plan
        self._release = release
        self.spill: Optional[TimedSpill] = None
        self.released_pinned_bytes = 0
        self.closed = False
        self.left_behind: Optional[str] = None
        self.path: Optional[str] = None
        if spill_dir is not None:
            Path(spill_dir).mkdir(parents=True, exist_ok=True)
            self.path = str(Path(spill_dir) / f"a4-spill-{os.getpid()}-{next(_COUNTER)}.bin")

        def factory(size: int) -> TimedSpill:
            if size != plan.spill_file_bytes:
                raise PlanMismatchError(
                    f"{PLAN_MISMATCH}: the store asked for a {size:,} B spill file, "
                    f"the plan for {plan.spill_file_bytes:,} B"
                )
            self.spill = TimedSpill(opener(backend, self.path, size))
            return self.spill

        allocator_before = host_allocator_bytes()
        try:
            self._store = l2l_activations.ActivationStore(
                n_layers=plan.n_layers, k=plan.k, chunk_shape=tuple(chunk_shape), dtype=dtype,
                device=device, pin=plan.pin, spill_layers=plan.spill_layers,
                spill_factory=factory if plan.spill_layers else None, ring=plan.ring,
            )
        except BaseException as exc:
            # ActivationStore opens the spill file before it pins the rings; a ring that
            # fails to pin would otherwise leave the file (and its blocks) behind. The
            # finished constructor frame still holds the chunks it pinned: clear it, so
            # the release below can hand those blocks back.
            _clear_chain(exc)
            if self.spill is not None:
                self.spill.close(remove=True)
                self.left_behind = self.spill.left_behind
            elif self.path is not None:
                # The opener raised before it handed back a backend. On Windows the
                # file's truncate() extends it by WRITING zeros, outside
                # DirectSpillFile's own cleanup guard, so a full volume raises out of
                # it with the partly written file still on disk.
                self.left_behind = _remove_partial(self.path)
            self.released_pinned_bytes = self._release()
            full = is_disk_full(exc)
            if full is not None:
                raise SpillDiskFullError(self._full_message(full, "creating")) from exc
            raise
        allocator_after = host_allocator_bytes()
        self.k = self._store.k
        self.n_layers = self._store.n_layers
        self.chunk_bytes = self._store.chunk_bytes
        self.held_bytes = self._held_bytes()
        self.allocator_delta_bytes = (
            None if allocator_before is None or allocator_after is None
            else allocator_after - allocator_before
        )
        self.cap = self.cap_check()
        if not self.cap["ok"]:
            self.close()
            raise PlanMismatchError(f"{PLAN_MISMATCH}: {self.cap['reason']}")

    # -- accounting -------------------------------------------------------------
    def _held_bytes(self) -> int:
        """What the store's allocations cost the host, each at its real block size."""
        held = 0
        for tensor in self._store._resident.values():
            held += budget.allocation_cost(tensor.untyped_storage().nbytes(), pin=self.plan.pin)
        if self.plan.spill_layers:
            for ring in (self._store._write_ring, self._store._read_ring):
                for buf, _event in list(ring._free.queue):
                    held += budget.allocation_cost(buf.untyped_storage().nbytes(),
                                                   pin=self.plan.pin)
        return held

    def cap_check(self) -> Dict[str, Any]:
        """Validity row C1: the store holds exactly the plan's bytes, within its budget,
        and (CUDA) the host allocator grew by no more than the budget."""
        problems = []
        if self.held_bytes != self.plan.host_bytes:
            problems.append(f"the store holds {self.held_bytes:,} B, the plan says "
                            f"{self.plan.host_bytes:,} B")
        if self.held_bytes > self.plan.budget_bytes:
            problems.append(f"the store holds {self.held_bytes:,} B over its "
                            f"{self.plan.budget_bytes:,} B budget")
        delta = self.allocator_delta_bytes
        if delta is not None and delta > self.plan.budget_bytes:
            problems.append(f"the CUDA host allocator grew {delta:,} B, over the "
                            f"{self.plan.budget_bytes:,} B budget")
        return {
            "ok": not problems, "reason": "; ".join(problems) or "within the plan",
            "held_bytes": self.held_bytes, "plan_host_bytes": self.plan.host_bytes,
            "budget_bytes": self.plan.budget_bytes, "allocator_delta_bytes": delta,
            "allocator_matches_plan": None if delta is None else delta == self.plan.host_bytes,
        }

    # -- disk full ----------------------------------------------------------------
    def _full_message(self, exc: OSError, during: str) -> str:
        plan = self.plan
        written = self.spill.bytes_written if self.spill is not None else 0
        failed = self.spill.failed if self.spill is not None else None
        where = (f" at offset {failed[1]:,} ({failed[2]:,} B {failed[0]})" if failed else "")
        free_now = free_bytes(self.spill_dir) if self.spill_dir else None
        gone = (f"the spill file could NOT be removed and is left at {self.left_behind}"
                if self.left_behind else "the spill file was removed")
        return (
            f"{SPILL_ENOSPC}: the volume holding {self.spill_dir} is out of space while "
            f"{during} the spill file: the plan spills {plan.spill_layers} layers x k={plan.k} "
            f"x {plan.chunk_bytes:,} B = {_b(plan.spill_file_bytes)}; {_b(written)} were "
            f"written before the drive refused{where}; {_b(free_now)} free now, "
            f"{_b(self.free_bytes_at_plan)} when the plan was made; {gone} and "
            f"{self.released_pinned_bytes:,} B of cached page-locked memory were returned. "
            f"OS error: {exc}"
        )

    def _guard(self, call: Callable[[], Any]) -> Any:
        try:
            return call()
        except RuntimeError as exc:
            full = is_disk_full(exc)
            if full is None:
                raise
            _clear_chain(exc)
            self.close()
            raise SpillDiskFullError(self._full_message(full, "writing")) from exc

    # -- the store interface ------------------------------------------------------
    @property
    def stats(self) -> Any:
        return self._store.stats

    def put(self, layer: int, mb: int, tensor: Any) -> None:
        self._guard(lambda: self._store.put(layer, mb, tensor))

    def get(self, layer: int, mb: int) -> Any:
        return self._guard(lambda: self._store.get(layer, mb))

    def begin_backward(self) -> None:
        self._store.begin_backward()

    def end_step(self) -> None:
        self._guard(self._store.end_step)

    def io(self) -> Dict[str, Any]:
        if self.spill is not None:
            return self.spill.io()
        return {"write_s": 0.0, "read_s": 0.0, "bytes_written": 0, "bytes_read": 0,
                "write_gb_per_s": None, "read_gb_per_s": None}

    def close(self) -> None:
        """Join the spill threads, remove the file, drop every buffer, return the
        page-locked blocks to the OS. Idempotent."""
        if self.closed:
            return
        self.closed = True
        store = getattr(self, "_store", None)
        if store is not None:
            store.close()
            self.left_behind = store.stats.spill_left_behind
            # A writer or reader that failed (any error, not only ENOSPC) left its
            # exception on the store; its frame holds the staging buffer it was moving.
            _clear_chain(store._error)
            store._resident.clear()
            if store.spill_layers:
                # A failed writer leaves queued (key, staging buffer, event) items behind;
                # they hold ring buffers, which must go before the cache release.
                for pending in (store._writes, store._reads):
                    while not pending.empty():
                        pending.get_nowait()
                store._write_ring = store._read_ring = None
                store._ready.clear()
        self.released_pinned_bytes += self._release()

    def __enter__(self) -> "BudgetedStore":
        return self

    def __exit__(self, exc_type: object, exc: Optional[BaseException], tb: object) -> None:
        # An exception (a Ctrl-C included) raised inside ActivationStore.put or get after
        # it took a staging buffer keeps that buffer in a finished frame of its traceback.
        if exc is not None:
            _clear_chain(exc)
        self.close()


def leftover_spill_files(spill_dir: str) -> List[str]:
    """Spill files of this harness in ``spill_dir`` (a crash on Windows, or a dead run)."""
    root = Path(spill_dir)
    if not root.is_dir():
        return []
    return sorted(str(path) for path in root.glob("a4-spill-*.bin"))
