"""Which loader feeds streamed routed experts: ``async_lease`` by default, C++ two-slot opt-in.

``async_lease`` is the existing single-slot ``ExpertBufferPool`` lease with a narrowed device
synchronisation; it needs nothing from this module. ``cpp_async_2slot`` selects the C++ loader
vendored from PR #77 in ``utils/_native_expert_async``; it is never replaced by the Python
lease when it cannot run (D3/ADR 0002). The extension is CUDA-only and is not built in a clean
checkout, so selecting it raises ``CppAsyncLoaderUnavailableError`` with the build command.

When it can run, ``open_native_stream`` binds the compiled ``PackedStreamAsync`` to the
per-expert readers and the expert pool reads every lease through it (``ExpertBufferPool``'s
``stream`` seam) instead of ``ExpertShardReader.read_host``.

``PackedSequenceLoader`` is the host-side controller for a packed stream's sequence contract:
one reusable destination slot, ordered expert ids, and a finished sequence on every error or
cancelled run so no ring slot stays reserved. The compiled extension implements the stream; the
tests drive the same controller with a threaded CPU stand.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

EXPERT_LOADER_CHOICES = ("async_lease", "cpp_async_2slot")
DEFAULT_EXPERT_LOADER = "async_lease"
EXTENSION_MODULE = "_native_expert_stream_async"
EXTENSION_BUILD_SCRIPT = "src/soup_cli/utils/_native_expert_async/build_ext.py"

#: The two-slot ring and its reader depth, for the wired pool.
NATIVE_RING_SLOTS = 2
NATIVE_IO_WORKERS = 2


class CppAsyncLoaderUnavailableError(RuntimeError):
    """``stream_expert_loader='cpp_async_2slot'`` cannot run here; never answered by a lease."""


@dataclass(frozen=True)
class LoaderEnvironment:
    """What the C++ loader depends on, read from this machine (nothing is built)."""

    extension_available: bool
    cuda_available: bool
    detail: Optional[str] = None


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001 - a missing/broken torch is simply "no CUDA"
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - a broken CUDA install counts as no CUDA
        return False


def detect_loader_environment() -> LoaderEnvironment:
    """Probe the compiled module and CUDA; importing ``soup_cli`` never calls this."""
    try:
        found = importlib.util.find_spec(EXTENSION_MODULE)
    except (ImportError, ValueError) as failure:
        return LoaderEnvironment(False, _cuda_available(), str(failure))
    if found is None:
        return LoaderEnvironment(False, _cuda_available(), "the compiled module is absent")
    try:
        module = importlib.import_module(EXTENSION_MODULE)
    except Exception as failure:  # noqa: BLE001 - an ABI or link error names itself
        return LoaderEnvironment(
            False, _cuda_available(), f"{type(failure).__name__}: {failure}"
        )
    if not hasattr(module, "PackedStreamAsync"):
        return LoaderEnvironment(
            False, _cuda_available(), "the module has no PackedStreamAsync"
        )
    return LoaderEnvironment(True, _cuda_available(), None)


def require_cpp_async_2slot(environment: Optional[LoaderEnvironment] = None) -> None:
    """Raise ``CppAsyncLoaderUnavailableError`` for the first failed requirement."""
    found = environment if environment is not None else detect_loader_environment()
    flag = "stream_expert_loader='cpp_async_2slot'"
    _require_extension(found, option=flag)
    if not found.cuda_available:
        raise CppAsyncLoaderUnavailableError(f"{flag} needs CUDA; none is available")


@contextmanager
def expert_loader_context(
    loader: str = DEFAULT_EXPERT_LOADER,
) -> Iterator[Any]:
    """``async_lease`` (default): nothing to install, the existing pool keeps its behaviour.
    ``cpp_async_2slot``: require the extension and CUDA; the pool then reads through it."""
    resolve_expert_loader(loader)
    yield None


def resolve_expert_loader(loader: str = DEFAULT_EXPERT_LOADER) -> str:
    """Return the selected loader or raise; called before a streamed run allocates anything.

    ``async_lease`` returns without probing the machine. ``cpp_async_2slot`` names the missing
    extension/CUDA and otherwise returns itself: the setup path hands it to the expert pool,
    which reads every lease through the compiled stream. An option that validated but silently
    left the Python lease in place would be the substitution this exists to stop.
    """
    if loader == "async_lease":
        return loader
    if loader == "cpp_async_2slot":
        require_cpp_async_2slot()
        return loader
    raise ValueError(f"expert loader must be one of {EXPERT_LOADER_CHOICES}, got {loader!r}")


def _expert_ids(experts: Sequence[int]) -> List[int]:
    ids: List[int] = []
    for expert in experts:
        if type(expert) is not int or expert < 0:
            raise ValueError(f"expert ids must be non-negative integers, got {expert!r}")
        ids.append(expert)
    return ids


class PackedSequenceLoader:
    """Drive one packed stream's ordered expert sequence into one reusable host slot.

    ``stream`` is the ``PackedStreamAsync`` contract: ``start_sequence(layer, ids)``,
    ``consume_next(dst_ptr) -> expert``, ``finish_sequence()``, ``enqueue_next_sequence``,
    ``cancel_next_sequence()`` and ``close()``. The controller owns the destination buffer and
    the error path: a failed or cancelled sequence finishes on the error path, and a partially
    consumed one finishes when its generator is closed or collected, so no ring slot stays
    reserved. A stream that hands back an expert other than the one the sequence asked for is
    refused instead of feeding the wrong weights to the node.
    """

    def __init__(self, stream: Any, *, packed_bytes: int) -> None:
        if type(packed_bytes) is not int or packed_bytes < 1:
            raise ValueError("packed_bytes must be a positive integer")
        import torch

        self.stream = stream
        self.packed_bytes = packed_bytes
        self._slot = torch.empty(packed_bytes, dtype=torch.uint8)
        self.sequences = 0
        self.consumed = 0

    def fill_sync(self, layer: int, expert: int) -> Any:
        """Read one expert synchronously into the slot (the no-overlap control path)."""
        (expert_id,) = _expert_ids([expert])
        if type(layer) is not int or layer < 0:
            raise ValueError("layer must be a non-negative integer")
        self.stream.fill_sync(layer, expert_id, self._slot.data_ptr())
        return self._slot

    def prefetch(self, layer: int, experts: Sequence[int]) -> None:
        """Hint the next sequence; a mismatched or cancelled hint reads nothing."""
        if type(layer) is not int or layer < 0:
            raise ValueError("layer must be a non-negative integer")
        self.stream.enqueue_next_sequence(layer, _expert_ids(experts))

    def cancel_prefetch(self) -> None:
        self.stream.cancel_next_sequence()

    def sequence(self, layer: int, experts: Sequence[int]) -> Iterator[Tuple[int, Any]]:
        """Yield ``(expert, slot)`` in request order; the slot is reused by the next item.

        An iterator that is neither drained nor closed keeps its sequence active: the next
        ``sequence()`` call raises until the caller closes the iterator (or drops it for the
        garbage collector), which finishes the sequence and frees the ring slot.
        """
        if type(layer) is not int or layer < 0:
            raise ValueError("layer must be a non-negative integer")
        ids = _expert_ids(experts)
        if not ids:
            return
        self.stream.start_sequence(layer, ids)
        self.sequences += 1
        try:
            for expected in ids:
                expert = int(self.stream.consume_next(self._slot.data_ptr()))
                if expert != expected:
                    raise RuntimeError(
                        f"packed stream returned expert {expert} where the sequence order "
                        f"expects {expected}; refusing the wrong weights"
                    )
                self.consumed += 1
                yield expert, self._slot
        except BaseException:
            self.stream.finish_sequence()
            raise
        self.stream.finish_sequence()

    def close(self) -> None:
        close = getattr(self.stream, "close", None)
        if close is not None:
            close()


# ---------------------------------------------------------------- the compiled stream


def _require_extension(
    found: Optional[LoaderEnvironment] = None, *, option: str = "the C++ expert loader"
) -> None:
    """Name the compiled module and its build command, without requiring CUDA."""
    environment = found if found is not None else detect_loader_environment()
    if environment.extension_available:
        return
    detail = f": {environment.detail}" if environment.detail else ""
    raise CppAsyncLoaderUnavailableError(
        f"{option} needs the extension {EXTENSION_MODULE}; it is not built{detail}. "
        f"Build it with `python {EXTENSION_BUILD_SCRIPT} build_ext --build-lib "
        "/tmp/native-ext` (add --with-cuda-nf4 for the NF4 decoder) and put that directory "
        "on PYTHONPATH. The Python async_lease path is not substituted."
    )


@dataclass(frozen=True)
class PackedStreamLayer:
    """One expert file registered with the compiled stream.

    ``layer`` is the position inside the extension's layer list (the stream addresses layers
    by index, not by model layer number). ``rows`` is the ``expert_transfer.layout_rows`` of
    every stored component in file order, and ``experts[i]`` is the ``(start, end,
    packed_offset)`` list of expert ``i`` in that same order.
    """

    layer: int
    reader: Any
    rows: Tuple[Tuple[str, int, int, Any, Tuple[int, ...]], ...]
    experts: Tuple[Tuple[Tuple[int, int, int], ...], ...]
    staging_bytes: int


@dataclass(frozen=True)
class PackedStreamPlan:
    """What ``PackedStreamAsync`` is constructed from, shared with the CPU stand."""

    layers: Tuple[PackedStreamLayer, ...]
    packed_bytes: int

    def entry_for(self, reader: Any) -> PackedStreamLayer:
        for entry in self.layers:
            if entry.reader.path == reader.path:
                return entry
        raise ValueError(
            f"expert reader {getattr(reader, 'path', reader)!r} is not registered in this "
            "packed stream"
        )


def plan_packed_stream(readers: Mapping[int, Any]) -> PackedStreamPlan:
    """Lay out every stored expert component as one packed block, per expert file.

    Reads headers only. The rows are ``expert_transfer.layout_rows`` order, which is also
    the order ``ExpertBufferPool(transfer="packed")`` packs its pinned region, so the same
    block serves both the host cache and the copy to the device.
    """
    from soup_cli.utils.expert_transfer import layout_rows
    from soup_cli.utils.safetensors_reader import SECTOR_BYTES

    if not isinstance(readers, Mapping) or not readers:
        raise ValueError("a packed stream needs at least one expert reader")
    layers: List[PackedStreamLayer] = []
    packed_bytes: Optional[int] = None
    for position, key in enumerate(sorted(readers)):
        reader = readers[key]
        components = reader.stored_components(0)
        rows, size = layout_rows(
            [(name, tuple(shape), dtype) for name, shape, dtype, _start, _end in components]
        )
        if packed_bytes is None:
            packed_bytes = size
        elif size != packed_bytes:
            raise ValueError("expert shard files disagree on their packed block size")
        experts: List[Tuple[Tuple[int, int, int], ...]] = []
        staging = SECTOR_BYTES
        for expert in range(reader.count):
            stored = reader.stored_components(expert)
            if len(stored) != len(rows):
                raise ValueError(f"expert {expert} has unexpected stored components")
            ranges: List[Tuple[int, int, int]] = []
            for row, (name, shape, dtype, start, end) in zip(rows, stored):
                if (tuple(shape), dtype) != (row[4], row[3]) or end - start != row[2]:
                    raise ValueError(
                        f"expert {expert} component {name!r} does not match the first expert"
                    )
                ranges.append((start, end, row[1]))
                low = start - start % SECTOR_BYTES
                high = -(-end // SECTOR_BYTES) * SECTOR_BYTES
                staging = max(staging, high - low)
            experts.append(tuple(ranges))
        layers.append(
            PackedStreamLayer(position, reader, tuple(rows), tuple(experts), staging)
        )
    assert packed_bytes is not None
    return PackedStreamPlan(tuple(layers), packed_bytes)


class CppPackedStream:
    """``PackedStreamAsync`` bound to per-expert readers; the pool's read path.

    ``read`` runs a one-expert sequence through the two-slot ring and refuses an expert other
    than the one asked for. ``read_host`` is the same bytes as a dict keyed by the reader's
    stored tensor names, which is exactly what the pool's existing copy loop consumes.
    """

    def __init__(
        self,
        handle: Any,
        plan: PackedStreamPlan,
        *,
        ring_slots: int,
        io_workers: int,
        host_cache_bytes: int,
    ) -> None:
        self.handle = handle
        self.plan = plan
        self.ring_slots = ring_slots
        self.io_workers = io_workers
        self.host_cache_bytes = host_cache_bytes
        self.packed_bytes = plan.packed_bytes
        self.layouts = {entry.layer: entry.rows for entry in plan.layers}

    def read(self, reader: Any, expert: int, destination: int) -> None:
        entry = self.plan.entry_for(reader)
        if (
            isinstance(expert, bool)
            or not isinstance(expert, int)
            or not 0 <= expert < reader.count
        ):
            raise IndexError(f"expert {expert!r} outside [0, {reader.count})")
        self.handle.start_sequence(entry.layer, [expert])
        try:
            got = int(self.handle.consume_next(destination))
        finally:
            self.handle.finish_sequence()
        if got != expert:
            raise RuntimeError(
                f"packed stream returned expert {got} where the sequence order expects "
                f"{expert}; refusing the wrong weights"
            )

    def read_host(self, reader: Any, expert: int, staging: Any) -> Dict[str, Any]:
        if staging.numel() < self.packed_bytes:
            raise ValueError(
                f"packed staging holds {staging.numel()} bytes, the stream writes "
                f"{self.packed_bytes}"
            )
        self.read(reader, expert, staging.data_ptr())
        rows = self.layouts[self.plan.entry_for(reader).layer]
        stored = reader.stored_components(expert)
        return {
            stored[index][0]: staging[offset : offset + count].view(dtype).view(shape)
            for index, (_name, offset, count, dtype, shape) in enumerate(rows)
        }

    def stats(self) -> Dict[str, Any]:
        return dict(self.handle.stats())

    def close(self) -> None:
        close = getattr(self.handle, "close", None)
        if close is not None:
            close()


def open_native_stream(
    readers: Mapping[int, Any],
    *,
    ring_slots: int = NATIVE_RING_SLOTS,
    io_workers: int = NATIVE_IO_WORKERS,
    host_cache_bytes: int = 0,
) -> CppPackedStream:
    """Register the readers with the compiled stream; the caller owns the result.

    Only the extension is required here; the CUDA gate belongs to selecting the option
    (``require_cpp_async_2slot``), so the same stream can be exercised on a CPU-only host.
    """
    for name, value, low, high in (
        ("ring_slots", ring_slots, 2, 64),
        ("io_workers", io_workers, 1, 16),
        ("host_cache_bytes", host_cache_bytes, 0, None),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < low:
            raise ValueError(f"{name} must be an integer >= {low}, got {value!r}")
        if high is not None and value > high:
            raise ValueError(f"{name} must be at most {high}, got {value!r}")
    _require_extension()
    module = importlib.import_module(EXTENSION_MODULE)
    plan = plan_packed_stream(readers)
    registrations: List[Tuple[Any, int, List[List[Tuple[int, int, int]]]]] = []
    descriptors: List[int] = []
    try:
        for entry in plan.layers:
            descriptor = os.open(entry.reader.path, os.O_RDONLY)
            descriptors.append(descriptor)
            registrations.append(
                (entry.reader, descriptor, [list(ranges) for ranges in entry.experts])
            )
        handle = module.PackedStreamAsync(
            registrations,
            max(entry.staging_bytes for entry in plan.layers),
            plan.packed_bytes,
            host_cache_bytes,
            io_workers,
            ring_slots,
        )
    finally:
        for descriptor in descriptors:
            os.close(descriptor)
    return CppPackedStream(
        handle,
        plan,
        ring_slots=ring_slots,
        io_workers=io_workers,
        host_cache_bytes=host_cache_bytes,
    )
