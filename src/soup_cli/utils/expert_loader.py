"""Which loader feeds streamed routed experts: ``async_lease`` by default, C++ two-slot opt-in.

``async_lease`` is the existing single-slot ``ExpertBufferPool`` lease with a narrowed device
synchronisation; it needs nothing from this module. ``cpp_async_2slot`` selects the C++ loader
vendored from PR #77 in ``utils/_native_expert_async``; it is never replaced by the Python
lease when it cannot run (D3/ADR 0002). The extension is CUDA-only and is not built in a clean
checkout, so selecting it raises ``CppAsyncLoaderUnavailableError`` with the build command.

``PackedSequenceLoader`` is the host-side controller for a packed stream's sequence contract:
one reusable destination slot, ordered expert ids, and a finished sequence on every error or
cancelled run so no ring slot stays reserved. The compiled extension implements the stream; the
tests drive the same controller with a threaded CPU stand.
"""

from __future__ import annotations

import importlib
import importlib.util
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Sequence, Tuple

EXPERT_LOADER_CHOICES = ("async_lease", "cpp_async_2slot")
DEFAULT_EXPERT_LOADER = "async_lease"
EXTENSION_MODULE = "_native_expert_stream_async"
EXTENSION_BUILD_SCRIPT = "src/soup_cli/utils/_native_expert_async/build_ext.py"


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
    if not found.extension_available:
        detail = f": {found.detail}" if found.detail else ""
        raise CppAsyncLoaderUnavailableError(
            f"{flag} needs the C++ extension {EXTENSION_MODULE}; it is not built{detail}. "
            f"Build it with `python {EXTENSION_BUILD_SCRIPT} build_ext --build-lib "
            "/tmp/native-ext` (add --with-cuda-nf4 for the NF4 decoder) and put that "
            "directory on PYTHONPATH. The Python async_lease path is not substituted."
        )
    if not found.cuda_available:
        raise CppAsyncLoaderUnavailableError(f"{flag} needs CUDA; none is available")


@contextmanager
def expert_loader_context(
    loader: str = DEFAULT_EXPERT_LOADER,
) -> Iterator[Any]:
    """``async_lease`` (default): nothing to install, the existing pool keeps its behaviour.
    ``cpp_async_2slot``: require the extension, then refuse until the pool swap lands, so a
    selected arm never runs as the Python lease by accident."""
    resolve_expert_loader(loader)
    yield None


def resolve_expert_loader(loader: str = DEFAULT_EXPERT_LOADER) -> str:
    """Return the selected loader or raise; called before a streamed run allocates anything.

    ``async_lease`` returns without probing the machine. ``cpp_async_2slot`` first names the
    missing extension/CUDA, then refuses until the pool swap is wired: an option that validated
    but silently left the Python lease in place would be the substitution this exists to stop.
    """
    if loader == "async_lease":
        return loader
    if loader == "cpp_async_2slot":
        require_cpp_async_2slot()
        raise CppAsyncLoaderUnavailableError(
            "stream_expert_loader='cpp_async_2slot' is selected and the extension is built, "
            "but the loader is not wired into the expert pool yet (ticket 04 GPU part); "
            "the Python async_lease path is never substituted for it."
        )
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
