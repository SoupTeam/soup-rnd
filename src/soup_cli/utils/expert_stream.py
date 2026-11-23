"""Internal MoE streaming with a bounded reusable expert arena and per-expert backward.

The eager expert loop is preserved. One reusable weight arena replaces the original fetcher's
independently allocated tensors. Each hit expert is one autograd node: forward and backward
lease it through the same reader and the same bounded pool, so backward re-reads exactly the
experts it needs. Routed expert weights stay frozen constants, adapters belong outside routed
experts and backward admits only the validated stateless SiLU activation. This module is the
expert primitive only; public YAML/CLI/planner enablement lives elsewhere.
"""

from __future__ import annotations

import functools
import math
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from types import MethodType
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

from soup_cli.utils.expert_shard import DOWN, EXPERT_PARTS, GATE_UP, ExpertShardReader


@dataclass(frozen=True)
class ExpertBufferPlan:
    """Tensor storage in a single expert slot; independent of expert count.

    Weight bytes include arena padding, dense output, packed tensors, statistics,
    shared code tables and explicit nested-statistics scratch. Activations,
    backend workspaces and CUDA context/allocator reservations are separate.
    """

    dense_bytes: int
    packed_and_stats_bytes: int
    code_bytes: int
    scratch_bytes: int
    padding_bytes: int
    weight_bytes: int
    host_staging_bytes: int
    #: Bytes of the one contiguous region at the front of the arena that a packed transfer
    #: fills with a single copy; zero for the default per-component copies.
    packed_region_bytes: int = 0
    #: Pinned host bytes a packed transfer stages that region in (outside ``weight_bytes``).
    host_pinned_bytes: int = 0


TRANSFERS = ("sync", "packed")


def _check_transfer(transfer: Any) -> None:
    if transfer not in TRANSFERS:
        raise ValueError("transfer must be 'sync' or 'packed'")


def _buffer_layout(
    reader: ExpertShardReader, dtype: str, transfer: str = "sync"
) -> Tuple[ExpertBufferPlan, list]:
    import torch

    from soup_cli.utils.expert_shard import _expected_tensors
    from soup_cli.utils.layer_stream_runtime import _torch_dtype
    from soup_cli.utils.safetensors_reader import SECTOR_BYTES

    _check_transfer(transfer)
    if dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError("expert compute dtype must be float32, float16 or bfloat16")
    if transfer == "packed" and any(reader.quant_specs.get(part) is None for part in EXPERT_PARTS):
        raise ValueError("packed transfer needs NF4 in every expert matrix")
    target = _torch_dtype(dtype)
    entries = []
    counts = {"dense": 0, "packed": 0, "code": 0, "scratch": 0}
    offset = 0

    def add(name: str, shape: tuple, kind: Any, category: str) -> None:
        nonlocal offset
        size = math.prod(shape) * kind.itemsize
        offset = (offset + 255) // 256 * 256
        entries.append((name, shape, kind, offset, size))
        offset += size
        counts[category] += size

    blocks = 0
    dense_items, packed_items = [], []
    for part in EXPERT_PARTS:
        shape = tuple(reader.shapes[part])
        dense_items.append((part, shape, target, "dense"))
        spec = reader.quant_specs.get(part)
        if spec is None:
            continue
        if spec.dtype != dtype or spec.quant_type != "nf4":
            raise ValueError("NF4 expert compute dtype must match its stored NF4 spec")
        if spec.blocksize not in (32, 64, 128, 256, 512, 1024, 2048, 4096):
            raise ValueError("unsupported NF4 expert blocksize")
        for idx, (stored_shape, stored_dtype) in enumerate(_expected_tensors(part, shape, spec)):
            packed_items.append((f"{part}:{idx}", tuple(stored_shape), stored_dtype, "packed"))
        if spec.nested:
            blocks = max(blocks, -(-math.prod(shape) // spec.blocksize))
    if transfer == "packed":
        # One contiguous region at the front, so one copy fills every stored component.
        for item in packed_items:
            add(*item)
        region = offset
        for item in dense_items:
            add(*item)
    else:
        region = 0
        by_part = {part: [item for item in dense_items + packed_items
                          if item[0].split(":")[0] == part] for part in EXPERT_PARTS}
        for part in EXPERT_PARTS:
            for item in by_part[part]:
                add(*item)
    if reader.quant_specs:
        add("code", (16,), torch.float32, "code")
        if blocks:
            add("nested_code", (256,), torch.float32, "code")
            add("scratch", (blocks,), torch.float32, "scratch")
    total = (offset + 511) // 512 * 512
    plan = ExpertBufferPlan(
        dense_bytes=counts["dense"],
        packed_and_stats_bytes=counts["packed"],
        code_bytes=counts["code"],
        scratch_bytes=counts["scratch"],
        padding_bytes=total - sum(counts.values()),
        weight_bytes=total,
        host_staging_bytes=reader.max_span + SECTOR_BYTES,
        packed_region_bytes=region,
        host_pinned_bytes=region,
    )
    return plan, entries


def plan_expert_buffer(
    reader: ExpertShardReader, *, dtype: str, transfer: str = "sync"
) -> ExpertBufferPlan:
    """Plan storage without reading any expert or allocating the weight arena."""
    return _buffer_layout(reader, dtype, transfer)[0]


def expert_allocation_budget(plan: ExpertBufferPlan, *, device: str) -> int:
    """Include an explicit CUDA allocator allowance above logical tensor storage.

    Reused allocator blocks can be larger than the requested arena. The pool
    still checks the observed allocation against this finite budget; this is a
    reserve, not an assumption that every allocator configuration fits it.
    """
    return plan.weight_bytes + (32 * 2**20 if str(device).startswith("cuda") else 0)


class ExpertBufferPool:
    """One synchronous expert lease at a time, backed by one bounded arena.

    Returned weights are valid only inside ``lease``; they alias the reusable
    arena, never the host staging. Concurrent/reentrant leases fail explicitly,
    including a backward lease requested while another lease is active.
    CUDA work is synchronized before releasing a lease or recycling staging, unless
    ``async_lease`` narrows that for the packed transfer (see below).
    Readers belong to the caller; close them separately (including on success).
    """

    #: opt-in (class default, or set on an instance). Off: every lease ends with
    #: ``torch.cuda.synchronize``, so the host waits for all queued work before the next visit.
    #: On, packed transfer on CUDA only: a lease ends without a device-wide wait, and the only
    #: host wait left is for the H2D copy that reads the pinned staging region, before that
    #: region is written again. Weights, dequantisation and GEMMs stay ordered on their stream,
    #: so no bit changes; the host can prepare the next expert while the card computes. The one
    #: writer of the pinned staging region is ``_load_packed`` (after ``_wait_staging_free``); the
    #: prefetch worker writes its own staging buffer, never this region.
    async_lease = False
    _h2d_event: Any = None
    _resolved_paths: Dict[str, str]

    def __init__(
        self,
        reader: ExpertShardReader,
        *,
        device: str,
        dtype: str,
        weight_budget_bytes: int,
        codes: Optional[Mapping[str, Any]] = None,
        transfer: str = "sync",
        host_cache_bytes: int = 0,
        prefetch: bool = False,
        stream: Optional[Any] = None,
    ) -> None:
        """``prefetch`` (off by default) adds a second host staging buffer and one worker
        thread. ``prefetch(reader, expert)`` then reads the named expert in the background
        while the current one computes; a ``lease`` for that expert uses the bytes, any other
        lease discards them and reads synchronously. One read is in flight at a time.

        ``transfer="packed"`` (CUDA, NF4 in both matrices) stages each expert's stored
        components in one pinned block and copies it to the arena once instead of once per
        component; the dequantised weights are the same. ``host_cache_bytes`` (packed only)
        also keeps whole packed blocks in pageable RAM up to that many bytes, first come first
        kept; a hit reads nothing from the experts file. The cache is not charged to any other
        RAM budget. Both are off by default.

        ``stream`` (off by default) is the selected alternative loader: an object with a
        positive ``packed_bytes`` and ``read(reader, expert, destination_ptr)`` /
        ``read_host(reader, expert, staging)``. Every lease then takes its bytes from it
        instead of ``ExpertShardReader.read_host``; ``cpp_async_2slot`` builds a
        ``CppPackedStream`` over the readers. The pool closes it with the rest of its state.
        """
        import torch

        from soup_cli.utils.expert_shard import aligned_staging
        from soup_cli.utils.layer_shard import NF4_CODE_KEY, NF4_NESTED_CODE_KEY

        if (
            isinstance(weight_budget_bytes, bool)
            or not isinstance(weight_budget_bytes, int)
            or weight_budget_bytes < 1
        ):
            raise ValueError("weight_budget_bytes must be a positive integer")
        _check_transfer(transfer)
        if (
            isinstance(host_cache_bytes, bool)
            or not isinstance(host_cache_bytes, int)
            or host_cache_bytes < 0
        ):
            raise ValueError("host_cache_bytes must be a non-negative integer")
        if host_cache_bytes and transfer != "packed":
            raise ValueError("host_cache_bytes needs transfer='packed'")
        if type(prefetch) is not bool:
            raise ValueError("prefetch must be a bool")
        stream_packed = getattr(stream, "packed_bytes", None) if stream is not None else None
        if stream is not None:
            if (
                isinstance(stream_packed, bool)
                or not isinstance(stream_packed, int)
                or stream_packed < 1
            ):
                raise ValueError("stream.packed_bytes must be a positive integer")
            if not callable(getattr(stream, "read", None)) or not callable(
                getattr(stream, "read_host", None)
            ):
                raise ValueError("stream must provide read() and read_host()")
        self.plan, layout = _buffer_layout(reader, dtype, transfer)
        if self.plan.weight_bytes > weight_budget_bytes:
            raise ValueError(
                f"expert slot needs {self.plan.weight_bytes} bytes; budget is {weight_budget_bytes}"
            )
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("expert forward supports CPU or CUDA only")
        if transfer == "packed" and self.device.type != "cuda":
            raise ValueError("packed expert transfer requires CUDA")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = dtype
        self.transfer = transfer
        self.weight_budget_bytes = weight_budget_bytes
        self._shapes = dict(reader.shapes)
        self._specs = dict(reader.quant_specs)
        self._stored_dtype = reader.dtype
        self._max_span = reader.max_span
        tables = {}
        for name, key in (("code", NF4_CODE_KEY), ("nested_code", NF4_NESTED_CODE_KEY)):
            if not any(entry[0] == name for entry in layout):
                continue
            value = (codes or {}).get(key)
            count = 16 if name == "code" else 256
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.float32
                or tuple(value.shape) != (count,)
                or not torch.isfinite(value).all().item()
            ):
                raise ValueError(f"missing or invalid NF4 code table: {key}")
            tables[name] = value
        self._lock = threading.Lock()
        self._phase_lock = threading.Lock()
        self._closed = False
        self.fetches = 0
        self.logical_bytes_read = 0
        self.phase_read_tracking = False
        self._phase_read_counters: Dict[int, Dict[str, Dict[str, int]]] = {}
        self.h2d_copies = 0
        self._resolved_paths = {}
        self._arena = None
        self._buffers: dict = {}
        self._staging_owner = self._staging = None
        self._pinned = None
        self._stream = stream
        self._stream_staging_owner = self._stream_staging = None
        #: Prefetch state. ``prefetch_enabled`` may be switched at run time; it has an effect
        #: only on a pool built with ``prefetch=True`` (the staging and the worker exist).
        self.prefetch_enabled = prefetch
        self._prefetch_staging_owner = self._prefetch_staging = None
        self._prefetch_executor: Any = None
        self._prefetch_lock = threading.Lock()
        self._pending: Optional[Tuple[Any, int, Any]] = None
        self.prefetch_hits = 0
        self.prefetch_discards = 0
        #: Stored bytes the latest lease took from its reader (0 for a host-cache hit).
        self.last_lease_bytes = 0
        self._packed_rows: list = []
        self._host_cache = None
        if transfer == "packed":
            from soup_cli.utils.expert_transfer import PackedHostHotset

            self._packed_rows = [
                (name, offset, size, kind, shape)
                for name, shape, kind, offset, size in layout if ":" in name
            ]
            self._packed_bytes = self.plan.packed_region_bytes
            if host_cache_bytes:
                self._host_cache = PackedHostHotset(host_cache_bytes)
        if self._stream is not None:
            if transfer == "packed":
                if stream_packed != self._packed_bytes:
                    raise ValueError(
                        f"the packed stream writes {stream_packed} bytes, the pool's pinned "
                        f"region is {self._packed_bytes} — build both from the same readers"
                    )
            else:
                self._stream_staging_owner, self._stream_staging = aligned_staging(
                    stream_packed
                )
        before = torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else 0
        try:
            self._arena = torch.empty(self.plan.weight_bytes, dtype=torch.uint8, device=self.device)
            for name, shape, kind, offset, size in layout:
                self._buffers[name] = self._arena[offset : offset + size].view(kind).view(shape)
            for name, value in tables.items():
                self._buffers[name].copy_(value, non_blocking=False)
            self._staging_owner, self._staging = aligned_staging(reader.max_span)
            if prefetch:
                self._prefetch_staging_owner, self._prefetch_staging = aligned_staging(
                    reader.max_span
                )
            if transfer == "packed":
                self._pinned = torch.empty(self._packed_bytes, dtype=torch.uint8, pin_memory=True)
            self._synchronize()
            self.cuda_allocation_delta = (
                torch.cuda.memory_allocated(self.device) - before
                if self.device.type == "cuda" else 0
            )
            if self.cuda_allocation_delta > weight_budget_bytes:
                raise ValueError(
                    "CUDA allocator rounded the expert slot above the weight budget: "
                    f"observed {self.cuda_allocation_delta}, budget {weight_budget_bytes}, "
                    f"logical storage {self.plan.weight_bytes} bytes"
                )
        except BaseException:
            self._buffers.clear()
            self._arena = self._staging_owner = self._staging = self._pinned = None
            self._prefetch_staging_owner = self._prefetch_staging = None
            self._stream_staging_owner = self._stream_staging = None
            self._closed = True
            raise

    @property
    def stream(self) -> Optional[Any]:
        """The selected alternative loader feeding every lease, or None for the reader."""
        return self._stream

    def validate_reader(self, reader: ExpertShardReader) -> None:
        """All layers sharing this slot must have the same storage contract."""
        if self._closed:
            raise RuntimeError("expert buffer pool is closed")
        if (
            dict(reader.shapes) != self._shapes
            or dict(reader.quant_specs) != self._specs
            or reader.dtype != self._stored_dtype
            or reader.max_span > self._max_span
        ):
            raise ValueError("expert reader does not match the buffer pool; build a matching pool")

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            import torch

            torch.cuda.synchronize(self.device)

    def _host_key(self, reader: ExpertShardReader, expert: int) -> Tuple[str, int]:
        """The host-cache key of ``expert``: the resolved path of its reader and the index.

        ``os.path.realpath`` is a chain of system calls that every load and every prefetch hint
        used to repeat. Under ``async_lease`` a path is resolved once per pool; by default it is
        resolved on every call, as before."""
        import os

        if not self.async_lease:
            return (os.path.realpath(reader.path), expert)
        resolved = self._resolved_paths.get(reader.path)
        if resolved is None:
            resolved = self._resolved_paths[reader.path] = os.path.realpath(reader.path)
        return (resolved, expert)

    def _lease_end(self) -> None:
        """The wait a finished lease owes: the whole device, or none under ``async_lease``."""
        if self.async_lease and self.transfer == "packed" and self.device.type == "cuda":
            return
        self._synchronize()

    def _wait_staging_free(self) -> None:
        """Wait for the last H2D copy that reads the pinned staging region (``async_lease``)."""
        event = self._h2d_event
        if event is not None:
            event.synchronize()
            self._h2d_event = None

    def _mark_staging_busy(self) -> None:
        """Record that an H2D copy now reads the pinned staging region (``async_lease``)."""
        if self.async_lease and self.device.type == "cuda":
            import torch

            event = torch.cuda.Event()
            event.record()
            self._h2d_event = event

    def _copy_host(self, expert: int, host: Mapping[str, Any]) -> None:
        """Copy an expert's stored tensors into its arena views and dequantise NF4 parts."""
        from soup_cli.utils.expert_shard import expert_key
        from soup_cli.utils.layer_stream_runtime import quant_sidecar_keys

        for part in EXPERT_PARTS:
            key = expert_key(expert, part)
            spec = self._specs.get(part)
            if spec is None:
                self._buffers[part].copy_(host[key], non_blocking=False)
                continue
            for idx, name in enumerate(quant_sidecar_keys(key, spec)):
                self._buffers[f"{part}:{idx}"].copy_(host[name], non_blocking=False)
            self._dequantize(part, spec)

    def _load(
        self,
        reader: ExpertShardReader,
        expert: int,
        *,
        phase: Optional[str] = None,
        layer_idx: Optional[int] = None,
    ) -> None:
        fetch_succeeded = False
        self.last_lease_bytes = 0
        try:
            if self.transfer == "packed":
                self._load_packed(reader, expert)
            else:
                host = (
                    self._stream.read_host(reader, expert, self._stream_staging)
                    if self._stream is not None
                    else self._read_host(reader, expert)
                )
                if self._stream is not None:
                    self.last_lease_bytes = reader.stored_bytes(expert)
                self._copy_host(expert, host)
            self.fetches += 1
            fetch_succeeded = True
        finally:
            bytes_read = self.last_lease_bytes
            self.logical_bytes_read += bytes_read
            if self.phase_read_tracking and phase is not None and layer_idx is not None:
                with self._phase_lock:
                    row = self._phase_read_counters.setdefault(layer_idx, {}).setdefault(
                        phase,
                        {"expert_fetches": 0, "expert_bytes_read": 0},
                    )
                    row["expert_fetches"] += int(fetch_succeeded)
                    row["expert_bytes_read"] += bytes_read

    def _load_packed(self, reader: ExpertShardReader, expert: int) -> None:
        """Fill the packed region with one H2D copy (from the host cache on a hit), then decode."""
        from soup_cli.utils.expert_shard import expert_key
        from soup_cli.utils.expert_transfer import pack_into
        from soup_cli.utils.layer_stream_runtime import quant_sidecar_keys

        if self._stream is not None:
            # The compiled stream is the reader: it fills the pinned block with one sequence,
            # then the pool copies it to the arena once exactly like the Python packed path.
            self._wait_staging_free()
            self._stream.read(reader, expert, self._pinned.data_ptr())
            self.last_lease_bytes = reader.stored_bytes(expert)
            self._arena[: self._packed_bytes].copy_(self._pinned, non_blocking=True)
            self._mark_staging_busy()
            self.h2d_copies += 1
            for part in EXPERT_PARTS:
                self._dequantize(part, self._specs[part])
            return

        self._wait_staging_free()
        region = self._pinned[: self._packed_bytes]
        key = self._host_key(reader, expert)
        block = self._host_cache.get(key) if self._host_cache is not None else None
        if block is not None:
            region.copy_(block)
        else:
            host = self._read_host(reader, expert)
            stored = {}
            for part in EXPERT_PARTS:
                names = quant_sidecar_keys(expert_key(expert, part), self._specs[part])
                for idx, name in enumerate(names):
                    stored[f"{part}:{idx}"] = host[name]
            pack_into(region, self._packed_rows, stored)
            if self._host_cache is not None:
                self._host_cache.put(key, region)
        # One copy for every stored component; the arena's dequantised part is rebuilt below.
        self._arena[: self._packed_bytes].copy_(region, non_blocking=True)
        self._mark_staging_busy()
        self.h2d_copies += 1
        for part in EXPERT_PARTS:
            self._dequantize(part, self._specs[part])

    # ------------------------------------------------------------------ prefetch
    def _cached(self, reader: ExpertShardReader, expert: int) -> bool:
        return self._host_cache is not None and self._host_cache.holds(
            self._host_key(reader, expert)
        )

    def _read_background(self, reader: ExpertShardReader, expert: int) -> Tuple[Any, int]:
        before = reader.bytes_read
        host = reader.read_host(expert, self._prefetch_staging)
        return host, reader.bytes_read - before

    def _discard_pending(self) -> None:
        """Wait for the read in flight (it owns the reader and the staging) and drop it."""
        with self._prefetch_lock:
            pending, self._pending = self._pending, None
        if pending is not None:
            self.prefetch_discards += 1
            try:
                pending[2].result()
            except BaseException:  # noqa: BLE001 - nobody asked for these bytes
                pass

    def prefetch(self, reader: ExpertShardReader, expert: int) -> None:
        """Start reading ``expert`` in the background; a hint, never required.

        Does nothing when prefetch is off, the pool is closed, the expert is already in the
        host cache, or the same read is already pending. A pending read of another expert is
        waited for and dropped first, so at most one read is ever in flight.
        """
        if (
            not self.prefetch_enabled
            or self._prefetch_staging is None
            or self._closed
            or expert is None
            or self._cached(reader, expert)
        ):
            return
        with self._prefetch_lock:
            pending = self._pending
            if pending is not None and pending[0] is reader and pending[1] == expert:
                return
        self._discard_pending()
        if self._prefetch_executor is None:
            from concurrent.futures import ThreadPoolExecutor

            self._prefetch_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="expert-prefetch"
            )
        future = self._prefetch_executor.submit(self._read_background, reader, expert)
        with self._prefetch_lock:
            self._pending = (reader, expert, future)

    def _read_host(self, reader: ExpertShardReader, expert: int) -> Dict[str, Any]:
        """The expert's stored tensors on the host: the prefetched bytes, or a direct read.

        The views stay valid until the next read into the same staging. Both callers copy
        them out before the lease yields, which is also before the next hint can be given.
        ``last_lease_bytes`` is set to what the reader read for this lease.
        """
        with self._prefetch_lock:
            pending, self._pending = self._pending, None
        if pending is not None:
            if pending[0] is reader and pending[1] == expert:
                host, nbytes = pending[2].result()  # raises the background read's error here
                self.prefetch_hits += 1
                self.last_lease_bytes = nbytes
                return host
            self._pending = pending
            self._discard_pending()
        before = reader.bytes_read
        host = reader.read_host(expert, self._staging)
        self.last_lease_bytes = reader.bytes_read - before
        return host

    def _dequantize(self, part: str, spec: Any) -> None:
        from bitsandbytes.functional import QuantState, dequantize_4bit, dequantize_blockwise

        from soup_cli.utils.layer_stream_runtime import _torch_dtype

        absmax = self._buffers[f"{part}:1"]
        if spec.nested:
            nested = QuantState(
                absmax=self._buffers[f"{part}:2"],
                code=self._buffers["nested_code"],
                blocksize=spec.nested_blocksize,
                dtype=_torch_dtype("float32"),
            )
            scratch = self._buffers["scratch"][: absmax.numel()]
            if self.device.type == "cpu":
                # bnb 0.50 CPU has no .out kernel. Its temporary is CPU-only;
                # do not silently take this allocating path on CUDA.
                scratch.copy_(dequantize_blockwise(absmax, quant_state=nested))
            else:
                dequantize_blockwise(absmax, quant_state=nested, out=scratch)
            scratch.add_(self._buffers[f"{part}:3"])
            absmax = scratch
        state = QuantState(
            absmax=absmax,
            code=self._buffers["code"],
            shape=self._buffers[part].shape,
            dtype=self._buffers[part].dtype,
            blocksize=spec.blocksize,
            quant_type=spec.quant_type,
        )
        if self.device.type == "cpu":
            self._buffers[part].copy_(
                dequantize_4bit(self._buffers[f"{part}:0"], quant_state=state)
            )
        else:
            dequantize_4bit(self._buffers[f"{part}:0"], quant_state=state, out=self._buffers[part])

    @contextmanager
    def lease(
        self,
        reader: ExpertShardReader,
        expert: int,
        *,
        phase: Optional[str] = None,
        layer_idx: Optional[int] = None,
    ) -> Iterator[Tuple[Any, Any]]:
        """Hold one expert until all its computation has finished."""
        if phase is not None and phase not in (
            "init_forward",
            "recompute_forward",
            "backward_dX",
        ):
            raise ValueError(f"unsupported expert-read phase: {phase!r}")
        if (phase is None) != (layer_idx is None):
            raise ValueError("phase and layer_idx must be provided together")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("expert buffer pool already has an active lease")
        try:
            self.validate_reader(reader)
            failed = False
            try:
                self._load(reader, expert, phase=phase, layer_idx=layer_idx)
                yield self._buffers[GATE_UP], self._buffers[DOWN]
            except BaseException:
                failed = True
                self._discard_pending()  # the worker may be reading through this handle
                reader.close()
                raise
            finally:
                if failed:
                    self._synchronize()      # an error keeps the full wait
                else:
                    self._lease_end()
        finally:
            self._lock.release()

    def phase_read_snapshot(self) -> Dict[int, Dict[str, Dict[str, int]]]:
        """Return actual expert leases tagged by the caller's forward/backward phase."""
        with self._phase_lock:
            return {
                layer: {phase: dict(counts) for phase, counts in phases.items()}
                for layer, phases in self._phase_read_counters.items()
            }

    def stats(self) -> dict:
        return {
            "slot_count": 1,
            "weight_budget_bytes": self.weight_budget_bytes,
            "plan": asdict(self.plan),
            "cuda_allocation_delta": self.cuda_allocation_delta,
            "fetches": self.fetches,
            "logical_bytes_read": self.logical_bytes_read,
            "transfer": self.transfer,
            "h2d_copies": self.h2d_copies,
            "prefetch": {
                "enabled": bool(self.prefetch_enabled and self._prefetch_staging is not None),
                "hits": self.prefetch_hits,
                "discards": self.prefetch_discards,
            },
            "host_cache": self._host_cache.stats() if self._host_cache is not None else None,
            "stream": (
                None
                if self._stream is None
                else {
                    "name": type(self._stream).__name__,
                    "packed_bytes": self._stream.packed_bytes,
                }
            ),
            "closed": self._closed,
            "cpu_nf4_temporaries": self.device.type == "cpu" and bool(self._specs),
        }

    def close(self) -> None:
        """Release this pool's buffers; it does not own its readers."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("cannot close an expert pool with an active lease")
        try:
            try:
                self._synchronize()
            finally:
                self._discard_pending()
                if self._prefetch_executor is not None:
                    self._prefetch_executor.shutdown(wait=True)
                    self._prefetch_executor = None
                self._prefetch_staging_owner = self._prefetch_staging = None
                self._stream_staging_owner = self._stream_staging = None
                if self._stream is not None:
                    close_stream = getattr(self._stream, "close", None)
                    if callable(close_stream):
                        close_stream()
                    self._stream = None
                self._buffers.clear()
                self._arena = self._staging_owner = self._staging = self._pinned = None
                self._host_cache = None
                self._closed = True
        finally:
            self._lock.release()


#: Per thread, the two-read checkpoint scopes entered and not yet left, innermost
#: last. The forward enters one on the caller's thread, a recompute on the thread
#: that runs the backward.
_SCOPES = threading.local()

#: Decoder classes whose MoE output reaches the rest of the layer only through views and
#: additions, so a recompute that returns zeros for the routed experts changes nothing the
#: backward reads. Each has a gradient comparison with the resident model.
TWO_READ_DECODERS = frozenset({"DeepseekV3DecoderLayer"})


class _TwoReadRecord:
    """What one checkpointed call of a decoder layer keeps for its recompute:
    each top-k result its routers took in the forward, in call order."""

    __slots__ = ("topk", "cursor")

    def __init__(self) -> None:
        #: ``(signature, (values, indices))`` per top-k call.
        self.topk: list = []
        self.cursor = 0


class _TwoReadScope:
    """Makes ``record`` the active scope: the forward's, or the recompute's."""

    def __init__(
        self, record: _TwoReadRecord, replay: bool, guard: Optional["_TwoReadHookGuard"] = None
    ) -> None:
        self.record = record
        self.replay = replay
        self.guard = guard

    def __enter__(self) -> "_TwoReadScope":
        if self.guard is not None:
            # Before the layer runs, in the forward and again in every recompute, so a
            # hook registered at any time is refused before it sees a recompute output.
            self.guard()
        if self.replay:
            # Every recompute replays from the first decision.
            self.record.cursor = 0
        stack = getattr(_SCOPES, "stack", None)
        if stack is None:
            stack = _SCOPES.stack = []
        stack.append(self)
        return self

    def __exit__(self, *exc: Any) -> bool:
        _SCOPES.stack.pop()
        return False


def _active_scope() -> Optional[_TwoReadScope]:
    stack = getattr(_SCOPES, "stack", None)
    return stack[-1] if stack else None


class _TwoReadHookGuard:
    """Refuses a two-read checkpoint call while a forward hook would see a recompute output.

    In the recompute the routed experts return zeros, so the outputs of the experts module,
    of the MoE block holding it and of the decoder layer differ from the forward's; every
    other module inside the layer (attention, norms, router, shared expert) sees the
    forward's values. A forward hook on one of those three modules, or a global forward
    hook, is refused; no hook is judged safe by what it does. Forward pre-hooks and hooks
    on any other module are not checked, and neither is a patched or replaced ``forward``.
    When given the streamed wrapper for L2L, both that wrapper and the native decoder
    are watched: L2L calls the wrapper during its recompute too.
    """

    __slots__ = ("layer_idx", "watched")

    def __init__(self, layer: Any) -> None:
        cls = streamed_experts_class()
        watched: Dict[int, Tuple[str, Any]] = {id(layer): ("the decoder layer", layer)}
        layer_idx = None
        for name, module in layer.named_modules():
            if type(module).__name__ in TWO_READ_DECODERS:
                watched.setdefault(id(module), (f"the decoder layer `{name}`", module))
            if not (isinstance(module, cls) and module.two_reads):
                continue
            layer_idx = module.layer_idx
            parent_name = name.rpartition(".")[0]
            parent = layer.get_submodule(parent_name) if parent_name else layer
            watched.setdefault(id(parent), (f"the MoE block `{parent_name}`", parent))
            watched.setdefault(id(module), (f"the routed experts `{name}`", module))
        self.layer_idx = layer_idx
        self.watched = tuple(watched.values())

    def __call__(self) -> None:
        import torch.nn.modules.module as nn_module

        where = None
        hooks = getattr(nn_module, "_global_forward_hooks", None)
        if hooks:
            where = "a global forward hook (register_module_forward_hook), which runs on it,"
        else:
            for label, module in self.watched:
                hooks = module._forward_hooks
                if hooks:
                    where = f"{label}, which has a forward hook,"
                    break
        if where is None:
            return
        hook = next(iter(hooks.values()))
        name = getattr(hook, "__qualname__", type(hook).__name__)
        raise RuntimeError(
            f"two-read expert streaming refuses decoder layer {self.layer_idx}: {where} "
            f"({name}). Its checkpoint recompute runs the routed experts at zero, so a "
            f"forward hook on the decoder layer, its MoE block or its routed experts would "
            f"see a different output there and could produce wrong gradients without an "
            f"error. Remove the hook, move it to the router or the attention, "
            f"or build with two_read_experts=False."
        )


def two_read_checkpoint_contexts(
    guard: Optional[_TwoReadHookGuard] = None,
) -> Tuple[_TwoReadScope, _TwoReadScope]:
    """``context_fn`` for ``torch.utils.checkpoint.checkpoint`` around a decoder
    layer holding two-read experts: one fresh record, entered by the forward and
    by every recompute of this call. :func:`checkpoint_context_fn` binds ``guard``."""
    record = _TwoReadRecord()
    return (_TwoReadScope(record, replay=False, guard=guard),
            _TwoReadScope(record, replay=True, guard=guard))


def checkpoint_context_fn(layer: Any) -> Optional[Any]:
    """The ``context_fn`` a checkpointed ``layer`` needs: :func:`two_read_checkpoint_contexts`
    with ``layer``'s hook guard when it holds two-read ``StreamedExperts``, else ``None``."""
    cls = streamed_experts_class()
    for module in layer.modules():
        if isinstance(module, cls) and module.two_reads:
            return functools.partial(two_read_checkpoint_contexts, _TwoReadHookGuard(layer))
    return None


def _build_topk_decisions_class() -> Any:
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    topk = torch.ops.aten.topk.default

    class TopkDecisions(TorchDispatchMode):
        """Active for one router call. In the forward it keeps every top-k result;
        in a recompute it hands them back in order and never selects. Below
        autograd, so the recompute's top-k node still saves the indices the
        checkpoint expects; every other op runs unchanged."""

        def __init__(self, scope: _TwoReadScope) -> None:
            super().__init__()
            self.scope = scope

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func is not topk:
                return func(*args, **kwargs)
            source = args[0]
            signature = (
                tuple(source.shape), source.dtype, source.device, tuple(args[1:]),
                tuple(sorted(kwargs.items())),
            )
            record = self.scope.record
            if not self.scope.replay:
                out = func(*args, **kwargs)
                # Owned copies: nothing the router's caller does to its outputs
                # can change what the recompute receives.
                record.topk.append((signature, tuple(tensor.clone() for tensor in out)))
                return out
            cursor = record.cursor
            if cursor >= len(record.topk) or record.topk[cursor][0] != signature:
                raise RuntimeError(
                    f"two-read experts: the checkpoint recompute asked a router for top-k "
                    f"#{cursor} {signature[:2]}, which the forward did not take "
                    f"({len(record.topk)} recorded)"
                )
            record.cursor = cursor + 1
            return tuple(tensor.clone() for tensor in record.topk[cursor][1])

    return TopkDecisions


_TOPK_DECISIONS_CLASS = None


def _topk_decisions(scope: _TwoReadScope) -> Any:
    global _TOPK_DECISIONS_CLASS
    if _TOPK_DECISIONS_CLASS is None:
        _TOPK_DECISIONS_CLASS = _build_topk_decisions_class()
    return _TOPK_DECISIONS_CLASS(scope)


def _routing_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
    """Forward of a two-read MoE block's router: under a two-read scope its top-k
    decisions are kept (forward) or replayed (recompute); scoring always runs."""
    forward = type(self).forward
    scope = _active_scope()
    if scope is None:
        return forward(self, *args, **kwargs)
    with _topk_decisions(scope):
        return forward(self, *args, **kwargs)


def _validate_routing(hidden: Any, index: Any, weights: Any, pool: ExpertBufferPool) -> None:
    import torch

    from soup_cli.utils.layer_stream_runtime import _torch_dtype

    if hidden.ndim != 2 or index.ndim != 2 or weights.shape != index.shape:
        raise ValueError("expected hidden [tokens, hidden] and routing indices/weights [tokens, k]")
    if index.shape[0] != hidden.shape[0] or not 1 <= index.shape[1]:
        raise ValueError("routing token count or top-k does not match hidden states")
    if index.dtype != torch.int64 or not weights.is_floating_point():
        raise ValueError("routing indices must be int64 and weights must be floating point")
    if hidden.dtype != _torch_dtype(pool.dtype):
        raise ValueError("hidden dtype must match the expert pool compute dtype")
    if any(value.device != pool.device for value in (hidden, index, weights)):
        raise ValueError("hidden and routing tensors must be on the expert pool device")
    if hidden.shape[1] != pool._shapes[GATE_UP][1]:
        raise ValueError("hidden dimension does not match the expert pool")
    if not torch.isfinite(weights).all().item():
        raise ValueError("routing weights must be finite")


def _validate_training_activation(act_fn: Any) -> None:
    """Recomputation is only valid for the native stateless SiLU implementations."""
    import torch.nn as nn
    from torch.nn.modules import module as module_hooks
    from transformers.activations import SiLUActivation

    if type(act_fn) not in (nn.SiLU, SiLUActivation):
        raise ValueError("expert backward requires an unmodified stateless SiLU activation")
    if (
        getattr(act_fn, "inplace", False)
        or "forward" in vars(act_fn)
        or any(True for _ in act_fn.parameters())
        or any(True for _ in act_fn.buffers())
        or any(True for _ in act_fn.children())
        or act_fn._forward_hooks
        or act_fn._forward_pre_hooks
        or act_fn._backward_hooks
        or act_fn._backward_pre_hooks
        or module_hooks._global_forward_hooks
        or module_hooks._global_forward_pre_hooks
        or module_hooks._global_backward_hooks
        or module_hooks._global_backward_pre_hooks
    ):
        raise ValueError("expert backward requires an unmodified stateless SiLU activation")


def _default_expert_routes(expert_mask: Any, expert_hit: Any) -> Iterator[Tuple[int, Any, Any]]:
    """The per-expert ``where`` walk the module used before the grouped lever."""
    import torch

    for expert_idx in expert_hit:
        expert = int(expert_idx[0])
        top_k_pos, token_idx = torch.where(expert_mask[expert])
        yield expert, top_k_pos, token_idx


def _grouped_expert_routes(expert_mask: Any) -> Iterator[Tuple[int, Any, Any]]:
    """Hit experts with their (top-k slot, token) rows from one nonzero and one count read.

    ``expert_mask`` is the one-hot routing ordered (expert, top-k slot, token). ``nonzero`` walks
    it in C order, so the experts come out ascending and each expert's rows in the order
    ``torch.where`` gives. The slices are views of one coordinate tensor made for this call.
    """
    import torch

    coordinates = torch.nonzero(expert_mask, as_tuple=False)
    counts = expert_mask.sum(dim=(-1, -2)).tolist()
    offset = 0
    for expert, count in enumerate(counts):
        if count == 0:
            continue
        if count < 0 or offset + count > coordinates.shape[0]:
            raise ValueError("grouped expert routes require a binary one-hot mask")
        group = coordinates.narrow(0, offset, count)
        offset += count
        yield expert, group[:, 1], group[:, 2]
    if offset != coordinates.shape[0]:
        raise ValueError("grouped expert routes omitted nonzero coordinates")


def _build_streamed_experts_class() -> Any:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    def expert_mlp(hidden: Any, gate_up: Any, down: Any, act_fn: Any) -> Any:
        gate, up = functional.linear(hidden, gate_up).chunk(2, dim=-1)
        return functional.linear(act_fn(gate) * up, down)

    class FrozenExpert(torch.autograd.Function):
        """One hit expert as one autograd node that leases its weights again in backward.

        Forward computes under a pool lease and saves only the gathered hidden slice (activation
        memory, never an arena view). Backward leases the same expert through the same reader
        and pool, rebuilds the body under the forward's autocast state and differentiates it
        with respect to the hidden slice only. The routing weight is multiplied outside this
        node, so its gradient needs the saved output rather than a second read. The activation
        is validated as the stateless SiLU again before any expert is read, so the recompute
        draws nothing the forward did not. Backward is first-order only: the gradients carry no
        graph over the arena weights, which the next lease overwrites, so a higher-order
        derivative is refused rather than taken against stale memory.
        """

        @staticmethod
        def forward(
            ctx: Any, hidden: Any, owner: Any, expert: int, nxt: Any = None, prv: Any = None
        ) -> Any:
            ctx.owner = owner
            ctx.act_fn = owner.act_fn
            ctx.expert = expert
            ctx.hint_prev = prv  # the backward visits experts in the reverse order
            ctx.device_type = hidden.device.type
            ctx.autocast = {
                "enabled": torch.is_autocast_enabled(ctx.device_type),
                "dtype": torch.get_autocast_dtype(ctx.device_type),
                "cache_enabled": torch.is_autocast_cache_enabled(),
            }
            ctx.save_for_backward(hidden)
            with owner.pool.lease(owner.reader, expert) as (gate_up, down):
                owner.pool.prefetch(owner.reader, nxt)
                return expert_mlp(hidden, gate_up, down, owner.act_fn)

        @staticmethod
        def backward(ctx: Any, grad_out: Any) -> Tuple[Any, None, None, None, None]:
            if torch.is_grad_enabled():
                raise RuntimeError("expert streaming does not support higher-order gradients")
            if not ctx.needs_input_grad[0]:
                return None, None, None, None, None
            _validate_training_activation(ctx.act_fn)
            (hidden,) = ctx.saved_tensors
            with ctx.owner.pool.lease(ctx.owner.reader, ctx.expert) as (gate_up, down):
                ctx.owner.pool.prefetch(ctx.owner.reader, ctx.hint_prev)
                local = hidden.detach().requires_grad_(True)
                # Only the recompute replays the forward's autocast state. The VJP itself runs
                # in the caller's backward state, as eager autograd does, so a backward inside
                # or outside an autocast region matches the resident module either way.
                with torch.enable_grad(), torch.autocast(ctx.device_type, **ctx.autocast):
                    result = expert_mlp(local, gate_up, down, ctx.act_fn)
                (grad_hidden,) = torch.autograd.grad(result, local, grad_out)
            return grad_hidden, None, None, None, None

    class WeightedFrozenExpert(torch.autograd.Function):
        """``expert_mlp(hidden) * weights`` as one node, for two reads.

        It saves only the gathered hidden slice and the routing weights, never the expert's
        output, so a checkpoint recompute can run it with ``replay=True``: no lease, a zero
        output, and the same two saved inputs the backward needs. Backward leases the expert
        once, rebuilds the multiply under the forward's autocast state and differentiates with
        respect to the hidden slice and the routing weights. First-order only, like
        ``FrozenExpert``. The forward's routing decisions reach a recompute through the router
        replay of ``_routing_forward``; a recompute that routed elsewhere would be a bug there.
        """

        @staticmethod
        def forward(
            ctx: Any, hidden: Any, weights: Any, owner: Any, expert: int, replay: bool,
            nxt: Any = None, prv: Any = None,
        ) -> Any:
            ctx.owner = owner
            ctx.act_fn = owner.act_fn
            ctx.expert = expert
            ctx.hint_prev = prv  # the backward visits experts in the reverse order
            ctx.device_type = hidden.device.type
            ctx.autocast = {
                "enabled": torch.is_autocast_enabled(ctx.device_type),
                "dtype": torch.get_autocast_dtype(ctx.device_type),
                "cache_enabled": torch.is_autocast_cache_enabled(),
            }
            ctx.save_for_backward(hidden, weights)
            if replay:
                # The consumers of this product (the cast and index_add_) keep it for its
                # shape alone, so the backward never reads this zero.
                return hidden.new_zeros(()).expand_as(hidden)
            scope = _active_scope()
            phase = "recompute_forward" if scope is not None and scope.replay else "init_forward"
            with owner.pool.lease(
                owner.reader,
                expert,
                phase=phase,
                layer_idx=owner.layer_idx,
            ) as (gate_up, down):
                owner.pool.prefetch(owner.reader, nxt)
                return expert_mlp(hidden, gate_up, down, owner.act_fn) * weights

        @staticmethod
        def backward(
            ctx: Any, grad_out: Any
        ) -> Tuple[Any, Any, None, None, None, None, None]:
            if torch.is_grad_enabled():
                raise RuntimeError("expert streaming does not support higher-order gradients")
            need_hidden, need_weights = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
            if not (need_hidden or need_weights):
                return None, None, None, None, None, None, None
            _validate_training_activation(ctx.act_fn)
            hidden, weights = ctx.saved_tensors
            with ctx.owner.pool.lease(
                ctx.owner.reader,
                ctx.expert,
                phase="backward_dX",
                layer_idx=ctx.owner.layer_idx,
            ) as (gate_up, down):
                ctx.owner.pool.prefetch(ctx.owner.reader, ctx.hint_prev)
                local_hidden = hidden.detach().requires_grad_(need_hidden)
                local_weights = weights.detach().requires_grad_(need_weights)
                with torch.enable_grad(), torch.autocast(ctx.device_type, **ctx.autocast):
                    result = expert_mlp(local_hidden, gate_up, down, ctx.act_fn) * local_weights
                wanted = [
                    tensor for tensor in (local_hidden, local_weights) if tensor.requires_grad
                ]
                grads = list(torch.autograd.grad(result, wanted, grad_out))
            grad_hidden = grads.pop(0) if need_hidden else None
            grad_weights = grads.pop(0) if need_weights else None
            return grad_hidden, grad_weights, None, None, None, None, None

    class StreamedExperts(nn.Module):
        """Eager expert loop over one-slot streamed weights, differentiable in its inputs.

        Hit experts are visited in ascending order. Every visit goes through ``FrozenExpert``,
        which leases the expert from the pool. It records a backward node only with autograd
        enabled and a grad-bearing hidden slice; that node's backward leases the same expert
        again through the same reader and pool. Without a graph the visit is one plain lease
        with identical arithmetic. The routing weight multiplies the node's output with
        ordinary autograd. A grad-bearing hidden state or routing weight additionally makes
        ``forward`` validate the stateless-SiLU activation before any expert is read. Routed
        weights are frozen constants and never receive a gradient.

        ``last_trace`` holds one row per expert visited by the latest ``forward`` call, so it
        is bounded by the expert count; a checkpoint recompute is a forward call and replaces
        it. Backward leases never append rows: they are counted only by ``reader.fetches``,
        ``reader.bytes_read``, ``pool.fetches`` and ``pool.logical_bytes_read``.
        """

        #: opt-in (class default, or set on an instance): light gathers for a visit's rows and
        #: route weights, see :func:`fast_gather_functions`
        fast_row_gather = False

        def __init__(
            self, reader: ExpertShardReader, pool: ExpertBufferPool, act_fn: Any, layer_idx: int,
            grouped_expert_routes: bool = False, two_reads: bool = False,
        ) -> None:
            super().__init__()
            if type(grouped_expert_routes) is not bool:
                raise ValueError("grouped_expert_routes must be a bool")
            if type(two_reads) is not bool:
                raise ValueError("two_reads must be a bool")
            self.two_reads = two_reads
            pool.validate_reader(reader)
            self.grouped_expert_routes = grouped_expert_routes
            if not callable(act_fn) or (
                isinstance(act_fn, nn.Module) and any(True for _ in act_fn.parameters())
            ):
                raise ValueError("expert activation must be callable without parameters")
            self.reader = reader
            self.pool = pool
            self.act_fn = act_fn
            self.layer_idx = layer_idx
            self.num_experts = reader.count
            self.last_trace: list = []
            self._forward_lock = threading.Lock()
            # MBR may need a real recompute output for a deferred decoder hook.
            self._mbr_materialize_recompute = False

        def backward_from_inputs(
            self,
            hidden_states: Any,
            top_k_index: Any,
            top_k_weights: Any,
            grad_output: Any,
            *,
            autocast_cfg: Mapping[str, Any],
            need_hidden: bool = True,
            need_weights: bool = True,
            segment_counts: Optional[Sequence[int]] = None,
            mm: Optional[Any] = None,
        ) -> Tuple[Optional[Any], Optional[Any]]:
            """Streaming per-expert backward straight from detached window inputs (M2+).

            One visit per hit expert, in reverse expert order like autograd's reverse
            topological order, re-reading each expert's weights for the visit. Nothing of the
            forward is kept: no ``top_k``-fold gathered ``current_state`` per
            ``WeightedFrozenExpert`` node (448 MiB at k=64 S=64 and 3.50 GiB at k=64 S=512 on
            DeepSeek-V3 671B) and no full-window sorted activation buffers.
            ``direct_expert_backward.expert_vjp`` runs the same ops as the node's backward, per
            segment, without a graph.

            ``segment_counts`` is the row count of every micro-batch the inputs were
            concatenated from. Each expert's rows are then gathered micro-batch by micro-batch
            in their original route order and differentiated per segment, so every matmul has
            the row count of the per-micro-batch call A0 makes, not one call over the window
            (the shapes of native GEMM bits). ``mm`` replaces the per-segment matmul.
            """
            import torch

            from soup_cli.utils import direct_expert_backward

            if grad_output.requires_grad:
                raise RuntimeError("expert streaming does not support higher-order gradients")
            if not (need_hidden or need_weights):
                return None, None
            sizes = None if segment_counts is None else [int(value) for value in segment_counts]
            if sizes is not None and (any(value < 0 for value in sizes)
                                      or sum(sizes) != hidden_states.shape[0]):
                raise ValueError(
                    f"expert segments {sizes} do not cover the {hidden_states.shape[0]} rows"
                )
            boundaries = None
            if sizes is not None and len(sizes) > 1:
                boundaries = torch.tensor(sizes, device=hidden_states.device).cumsum(0)[:-1]
            matmul = direct_expert_backward.segment_mm if mm is None else mm
            if not self._forward_lock.acquire(blocking=False):
                raise RuntimeError("expert forward is already active")
            try:
                self.last_trace = []
                self.pool.validate_reader(self.reader)
                _validate_routing(hidden_states, top_k_index, top_k_weights, self.pool)
                _validate_training_activation(self.act_fn)
                if grad_output.shape != hidden_states.shape:
                    raise ValueError("expert backward grad_output shape must match hidden_states")
                if top_k_index.shape[1] > self.num_experts:
                    raise ValueError("routing top-k exceeds the expert count")
                if top_k_index.numel() and (
                    top_k_index.min().item() < 0
                    or top_k_index.max().item() >= self.num_experts
                ):
                    raise ValueError("routing contains an expert outside the reader")

                expert_mask = functional.one_hot(top_k_index, num_classes=self.num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                if self.grouped_expert_routes:
                    routes = list(_grouped_expert_routes(expert_mask))
                else:
                    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
                    routes = list(_default_expert_routes(expert_mask, expert_hit))
                del expert_mask

                grad_hidden = torch.zeros_like(hidden_states) if need_hidden else None
                grad_weights = torch.zeros_like(top_k_weights) if need_weights else None
                device_type = hidden_states.device.type
                autocast_dtype = (
                    autocast_cfg.get("dtype") if autocast_cfg.get("enabled", False) else None
                )
                # Ascending for the trace, so the visited-expert telemetry keeps forward order;
                # the leases below run in reverse, as the forward's nodes are visited.
                for expert, _top_k_pos, token_idx in routes:
                    self.last_trace.append(
                        {
                            "layer": self.layer_idx,
                            "expert": expert,
                            "token_routes": token_idx.numel(),
                            "logical_bytes_read": 0,
                            "status": "replay",
                        }
                    )
                with torch.no_grad(), torch.autocast(device_type, enabled=False):
                    for row, (expert, top_k_pos, token_idx) in zip(
                        reversed(self.last_trace), reversed(routes), strict=True
                    ):
                        if boundaries is not None:
                            segment = torch.bucketize(token_idx, boundaries, right=True)
                            order = torch.argsort(segment, stable=True)
                            tokens, slots = token_idx[order], top_k_pos[order]
                            counts = [
                                count
                                for count in torch.bincount(
                                    segment, minlength=len(sizes)
                                ).tolist()
                                if count
                            ]
                        else:
                            tokens, slots = token_idx, top_k_pos
                            counts = [int(token_idx.numel())]
                        fetches = self.pool.fetches
                        try:
                            with self.pool.lease(
                                self.reader,
                                expert,
                                phase="backward_dX",
                                layer_idx=self.layer_idx,
                            ) as (gate_up, down):
                                grad_rows, grad_row_weights = direct_expert_backward.expert_vjp(
                                    hidden_states[tokens].detach(),
                                    top_k_weights[tokens, slots, None].detach(),
                                    gate_up,
                                    down,
                                    grad_output[tokens],
                                    counts,
                                    need_hidden=need_hidden,
                                    need_weights=need_weights,
                                    autocast_dtype=autocast_dtype,
                                    mm=matmul,
                                )
                        finally:
                            # One semantics with the counter: the row takes the lease's bytes,
                            # and only a visit that read nothing is a replay.
                            bytes_read = (
                                self.pool.last_lease_bytes
                                if self.pool.fetches != fetches
                                else 0
                            )
                            row["logical_bytes_read"] = bytes_read
                            row["status"] = "ok" if bytes_read else "replay"
                        if grad_hidden is not None and grad_rows is not None:
                            grad_hidden.index_add_(
                                0, tokens, grad_rows.to(grad_hidden.dtype)
                            )
                        if grad_weights is not None and grad_row_weights is not None:
                            grad_weights[tokens, slots] = grad_row_weights.squeeze(-1).to(
                                grad_weights.dtype
                            )
                        del grad_rows, grad_row_weights
                return grad_hidden, grad_weights
            finally:
                self._forward_lock.release()

        def forward(self, hidden_states: Any, top_k_index: Any, top_k_weights: Any) -> Any:
            if not self._forward_lock.acquire(blocking=False):
                raise RuntimeError("expert forward is already active")
            try:
                self.last_trace = []
                self.pool.validate_reader(self.reader)
                _validate_routing(hidden_states, top_k_index, top_k_weights, self.pool)
                differentiable = torch.is_grad_enabled() and (
                    hidden_states.requires_grad or top_k_weights.requires_grad
                )
                if differentiable:
                    _validate_training_activation(self.act_fn)
                if top_k_index.shape[1] > self.num_experts:
                    raise ValueError("routing top-k exceeds the expert count")
                if top_k_index.numel() and (
                    top_k_index.min().item() < 0 or top_k_index.max().item() >= self.num_experts
                ):
                    raise ValueError("routing contains an expert outside the reader")
                final_hidden_states = torch.zeros_like(hidden_states)
                expert_mask = functional.one_hot(top_k_index, num_classes=self.num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                if self.grouped_expert_routes:
                    routes = _grouped_expert_routes(expert_mask)
                else:
                    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
                    routes = _default_expert_routes(expert_mask, expert_hit)
                # Inside a two-read checkpoint recompute no expert is read.
                scope = _active_scope() if self.two_reads else None
                replay = (
                    scope is not None and scope.replay and not self._mbr_materialize_recompute
                )
                # A hint names the next expert of this visit order (and the previous one for the
                # backward). Hints never cross a layer: the next layer's routes are unknown.
                hinting = bool(self.pool.prefetch_enabled)
                visit = list(routes) if hinting else None
                for position, (expert, top_k_pos, token_idx) in enumerate(
                    visit if hinting else routes
                ):
                    nxt = visit[position + 1][0] if hinting and position + 1 < len(visit) else None
                    prv = visit[position - 1][0] if hinting and position > 0 else None
                    row = {
                        "layer": self.layer_idx, "expert": expert,
                        "token_routes": token_idx.numel(),
                        "logical_bytes_read": 0, "status": "loading",
                    }
                    self.last_trace.append(row)
                    fetches = self.pool.fetches
                    try:
                        if self.fast_row_gather:
                            gather_rows, gather_weights = fast_gather_functions()
                            current_state = gather_rows.apply(hidden_states, token_idx)
                            weights = gather_weights.apply(top_k_weights, token_idx, top_k_pos)
                        else:
                            current_state = hidden_states[token_idx]
                            weights = top_k_weights[token_idx, top_k_pos, None]
                        if self.two_reads:
                            current = WeightedFrozenExpert.apply(
                                current_state, weights, self, expert, replay, nxt, prv
                            )
                        else:
                            current = FrozenExpert.apply(current_state, self, expert, nxt, prv)
                            current = current * weights
                        row["status"] = "replay" if replay else "ok"
                    except BaseException:
                        row["status"] = "error"
                        raise
                    finally:
                        # The same figure the counter took. ``reader.bytes_read`` is not it on
                        # the cpp path (the compiled stream reads through its own descriptors)
                        # and it also moves for the expert read ahead under a prefetch hint.
                        row["logical_bytes_read"] = (
                            self.pool.last_lease_bytes if self.pool.fetches != fetches else 0
                        )
                    # Outside the load status on purpose: the lease is over, and a non-reentrant
                    # checkpoint recompute may stop here (it raises once every saved tensor has
                    # been recomputed), which is not an expert load failure.
                    final_hidden_states.index_add_(
                        0, token_idx, current.to(final_hidden_states.dtype)
                    )
                return final_hidden_states
            finally:
                self._forward_lock.release()

    return StreamedExperts


_FAST_GATHER_FUNCTIONS = None


def fast_gather_functions() -> Any:
    """``(GatherRows, GatherRouteWeights)``: the expert loop's gathers under ``fast_row_gather``.

    ``hidden_states[token_idx]`` and ``top_k_weights[token_idx, top_k_pos, None]`` backpropagate
    through ``index_put_(accumulate=True)``, which sorts and launches about a dozen kernels. Each
    token reaches an expert once, so those indices are unique and the backward is a scatter into
    zeros, one ``index_add_``: ``0 + g`` gives the same bits as the accumulate (the sign of a zero
    included), and nothing is summed. The indices must be unique; duplicates would be summed in
    an undefined order by the atomics.
    """
    global _FAST_GATHER_FUNCTIONS
    if _FAST_GATHER_FUNCTIONS is not None:
        return _FAST_GATHER_FUNCTIONS
    import torch

    class GatherRows(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, source: Any, index: Any) -> Any:
            ctx.save_for_backward(index)
            ctx.source_shape = source.shape
            return source.index_select(0, index)

        @staticmethod
        def backward(ctx: Any, grad: Any) -> Any:
            if not ctx.needs_input_grad[0]:
                return None, None
            (index,) = ctx.saved_tensors
            out = grad.new_zeros(ctx.source_shape)
            return out.index_add_(0, index, grad), None

    class GatherRouteWeights(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, weights: Any, token_idx: Any, top_k_pos: Any) -> Any:
            ctx.save_for_backward(token_idx, top_k_pos)
            ctx.weights_shape = weights.shape
            return weights[token_idx, top_k_pos, None]

        @staticmethod
        def backward(ctx: Any, grad: Any) -> Any:
            if not ctx.needs_input_grad[0]:
                return None, None, None
            token_idx, top_k_pos = ctx.saved_tensors
            out = grad.new_zeros(ctx.weights_shape)
            flat = token_idx * ctx.weights_shape[1] + top_k_pos
            out.view(-1).index_add_(0, flat, grad.reshape(-1))
            return out, None, None

    _FAST_GATHER_FUNCTIONS = (GatherRows, GatherRouteWeights)
    return _FAST_GATHER_FUNCTIONS


_STREAMED_EXPERTS_CLASS = None


def streamed_experts_class() -> Any:
    global _STREAMED_EXPERTS_CLASS
    if _STREAMED_EXPERTS_CLASS is None:
        _STREAMED_EXPERTS_CLASS = _build_streamed_experts_class()
    return _STREAMED_EXPERTS_CLASS


def _find_experts(layer: Any) -> Optional[Tuple[Any, str, Any]]:
    found = []
    for name, module in layer.named_modules():
        parent_name, _, leaf = name.rpartition(".")
        if leaf == "experts":
            parent = layer.get_submodule(parent_name) if parent_name else layer
            found.append((parent, leaf, module))
    if len(found) > 1:
        raise ValueError("multiple routed-experts modules in one layer are unsupported")
    return found[0] if found else None


def install_streamed_experts(
    layers: Sequence[Any], readers: Mapping[int, ExpertShardReader], pool: ExpertBufferPool,
    *, grouped_expert_routes: bool = False, two_reads: bool = False,
) -> int:
    """Replace only routed experts after validating ALL layers.

    Router, shared experts and dense layers stay with their original MoE code.
    Routed experts become differentiable-through but never trainable: their weights must be
    frozen and carry no adapters or extra parameters. Backward re-leases each hit expert and
    requires the validated stateless SiLU activation. Readers and pool are caller-owned.
    Invalid installs do not alter the model. ``grouped_expert_routes`` (off by default) makes
    each replacement find its hit experts and their rows with one ``nonzero``.

    ``two_reads`` (off by default) makes a step read each hit expert twice instead of three
    times: the layer's checkpoint gets :func:`checkpoint_context_fn`, inside which the MoE
    block's router (``.gate``) keeps its top-k decisions in the forward and replays them in
    the recompute, and the recompute reads no expert. Only ``TWO_READ_DECODERS`` are
    admitted. In the recompute the routed experts output zeros, so a forward hook on the
    decoder layer, its MoE block or its routed experts, or a global forward hook, is refused
    with a ``RuntimeError`` when the layer's forward or recompute starts. A patched
    ``forward`` is not detected and is unsupported.
    """
    import torch.nn as nn

    if type(grouped_expert_routes) is not bool:
        raise ValueError("grouped_expert_routes must be a bool")
    if type(two_reads) is not bool:
        raise ValueError("two_reads must be a bool")
    if any(
        isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < len(layers)
        for idx in readers
    ):
        raise ValueError("expert reader layer index is outside the supplied layers")
    replacements = []
    seen = set()
    cls = streamed_experts_class()
    for idx, layer in enumerate(layers):
        found = _find_experts(layer)
        reader = readers.get(idx)
        if reader is None and found is None:
            continue
        if reader is None or found is None:
            raise ValueError(f"layer {idx}: model/cache experts disagree; reshard the checkpoint")
        pool.validate_reader(reader)
        parent, attr, module = found
        # An adapter (peft ParamWrapper) wraps ``experts`` and hides gate_up_proj/down_proj, so
        # the parameter set is checked before the layout: a wrapper must read as an unsupported
        # adapter, not as a shape mismatch. Nothing is read here; installation never reads.
        parameters = dict(module.named_parameters())
        if set(parameters) != {"gate_up_proj", "down_proj"}:
            held = sorted(parameters)
            shown = (", ".join(held[:3]) + (", ..." if len(held) > 3 else "")) if held else "none"
            raise ValueError(
                f"layer {idx}: routed experts must hold exactly the frozen gate_up_proj and "
                f"down_proj parameters but hold {shown}; adapters or extra parameters on routed "
                "experts are unsupported (B1.3 allows adapters outside routed experts only)"
            )
        expected = (
            (reader.count, *reader.shapes[GATE_UP]), (reader.count, *reader.shapes[DOWN])
        )
        actual = tuple(
            tuple(getattr(module, name, ()).shape) if hasattr(module, name) else ()
            for name in ("gate_up_proj", "down_proj")
        )
        if actual != expected or not callable(getattr(module, "act_fn", None)):
            raise ValueError(f"layer {idx}: experts shape/activation mismatch; reshard checkpoint")
        if any(parameter.requires_grad for parameter in parameters.values()):
            raise ValueError(
                f"layer {idx}: routed-expert weights must stay frozen; B1.3 backpropagates "
                "through streamed routed experts but never trains them"
            )
        router = getattr(parent, "gate", None) if two_reads else None
        if two_reads and type(layer).__name__ not in TWO_READ_DECODERS:
            raise ValueError(
                f"decoder layer {idx} is a {type(layer).__name__}: two reads are admitted "
                f"only where the MoE output is consumed by views and additions "
                f"({', '.join(sorted(TWO_READ_DECODERS))})"
            )
        if two_reads and not isinstance(router, nn.Module):
            raise ValueError(
                f"decoder layer {idx}: two reads replay the router's top-k decisions, but "
                "the MoE block has no router module at `.gate`"
            )
        identity = (id(parent), attr)
        if identity in seen:
            raise ValueError("aliased expert modules cannot share multiple layer replacements")
        seen.add(identity)
        replacements.append((
            parent, attr,
            cls(
                reader, pool, module.act_fn, idx,
                grouped_expert_routes=grouped_expert_routes, two_reads=two_reads,
            ),
            router,
        ))
    for parent, attr, replacement, router in replacements:
        setattr(parent, attr, replacement)
        if router is not None:
            router.forward = MethodType(_routing_forward, router)
    return len(replacements)
