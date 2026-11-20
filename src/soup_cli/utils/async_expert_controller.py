"""Two-slot async expert controller bridging C++17 NVMe ring prefetch and CUDA DMA streams."""

from __future__ import annotations

import os
from contextlib import ExitStack, contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

LOADER_MODES = ("python_sync", "cpp_sync", "cpp_async_2slot")


def cpu_packed_control(pool: Any, reader: Any, budget: int) -> None:
    """Exercise the packed expert layout on CPU using pageable byte buffers."""
    import torch

    from soup_cli.utils.expert_stream import _buffer_layout
    from soup_cli.utils.expert_transfer import PackedHostHotset

    if pool.device.type != "cpu" or pool.transfer != "sync" or pool.fetches:
        raise ValueError("CPU packing simulation requires a fresh CPU NF4 pool")
    pool.plan, layout = _buffer_layout(reader, pool.dtype, "packed")
    pool.weight_budget_bytes = pool.plan.weight_bytes
    pool._packed_bytes = pool.plan.packed_region_bytes
    pool._packed_rows = [
        (name, offset, size, dtype, shape)
        for name, shape, dtype, offset, size in layout
        if ":" in name
    ]
    arena = torch.empty(pool.plan.weight_bytes, dtype=torch.uint8)
    buffers = {
        name: arena[offset : offset + size].view(dtype).view(shape)
        for name, shape, dtype, offset, size in layout
    }
    for name in ("code", "nested_code"):
        if name in buffers:
            buffers[name].copy_(pool._buffers[name])
    pool._arena, pool._buffers = arena, buffers
    pool._pinned = torch.empty(pool._packed_bytes, dtype=torch.uint8)
    pool._host_cache = PackedHostHotset(budget) if budget else None
    pool.transfer = "packed"


def _build_registrations(
    pool: Any, readers: Tuple[Any, ...]
) -> List[Tuple[Any, int, List[Any]]]:
    from soup_cli.utils.expert_shard import EXPERT_PARTS, _stored_torch_dtype, expert_key
    from soup_cli.utils.layer_stream_runtime import quant_sidecar_keys

    layouts = {
        name: (offset, size, dtype, shape) for name, offset, size, dtype, shape in pool._packed_rows
    }
    rows = []
    for reader in readers:
        pool.validate_reader(reader)
        experts = []
        for expert, entries in enumerate(reader._entries):
            fields = {
                field: f"{part}:{index}"
                for part in EXPERT_PARTS
                for index, field in enumerate(
                    quant_sidecar_keys(expert_key(expert, part), reader.quant_specs[part])
                )
            }
            if set(fields) != {key for key, _ in entries}:
                raise ValueError("native reader fields differ from validated packed layout")
            ranges = []
            for key, entry in entries:
                offset, size, dtype, shape = layouts[fields[key]]
                actual = (entry.nbytes, _stored_torch_dtype(entry.dtype), tuple(entry.shape))
                if actual != (size, dtype, shape):
                    raise ValueError(f"{key} does not match its packed layout row")
                ranges.append((entry.start, entry.end, offset))
            experts.append(ranges)
        rows.append((reader, reader._open().fileno(), experts))
    return rows


def _make_companion_slot(primary: Any, readers: Tuple[Any, ...]) -> Any:
    from soup_cli.utils.expert_stream import ExpertBufferPool, expert_allocation_budget
    from soup_cli.utils.layer_shard import NF4_CODE_KEY, NF4_NESTED_CODE_KEY

    biggest = max(readers, key=lambda shard: shard.max_span)
    codes = {
        key: primary._buffers[name]
        for key, name in ((NF4_CODE_KEY, "code"), (NF4_NESTED_CODE_KEY, "nested_code"))
        if name in primary._buffers
    }
    if primary.device.type == "cuda":
        budget = expert_allocation_budget(primary.plan, device="cuda")
        return ExpertBufferPool(
            biggest,
            device=str(primary.device),
            dtype=primary.dtype,
            codes=codes,
            transfer="packed",
            weight_budget_bytes=budget,
            host_cache_bytes=0,
        )
    slot = ExpertBufferPool(
        biggest,
        device="cpu",
        dtype=primary.dtype,
        codes=codes,
        transfer="sync",
        weight_budget_bytes=primary.plan.weight_bytes,
        host_cache_bytes=0,
    )
    cpu_packed_control(slot, biggest, 0)
    return slot


def _nf4_slot_args(slot: Any) -> Tuple[int, ...]:
    """Validate the packed views once before binding their device addresses."""
    import torch

    from soup_cli.utils.expert_shard import EXPERT_PARTS

    lookup = slot._buffers["code"]
    if (
        lookup.device != slot.device
        or lookup.dtype != torch.float32
        or lookup.numel() != 16
        or not lookup.is_contiguous()
    ):
        raise ValueError("CUDA NF4 decode requires a contiguous FP32 codebook of 16 values")
    args = [lookup.data_ptr()]
    for part in EXPERT_PARTS:
        spec = slot._specs[part]
        codes = slot._buffers[f"{part}:0"]
        scales = slot._buffers[f"{part}:1"]
        output = slot._buffers[part]
        packed_bytes = codes.numel()
        if (
            spec.quant_type != "nf4"
            or spec.nested
            or spec.blocksize != 64
            or codes.dtype != torch.uint8
            or scales.dtype != torch.float32
            or output.dtype != torch.bfloat16
            or any(
                tensor.device != slot.device or not tensor.is_contiguous()
                for tensor in (codes, scales, output)
            )
            or packed_bytes == 0
            or output.numel() != 2 * packed_bytes
            or scales.numel() != (packed_bytes + 31) // 32
            or output.data_ptr() % 4
        ):
            raise ValueError(f"CUDA NF4 decode requires aligned block-64 NF4 -> BF16 for {part}")
        args.extend((codes.data_ptr(), scales.data_ptr(), output.data_ptr(), packed_bytes))
    return tuple(args)


class AsyncExpertController:
    """Executes expert sequences via stock 1-slot Python, 1-slot C++, or 2-slot async C++."""

    def __init__(
        self,
        runtime: Any,
        mode: str = "python_sync",
        *,
        cuda_nf4_decode: bool = False,
        sample_overlap: bool = True,
        sample_stride: int = 16,
        io_workers: int = 1,
        ring_slots: int = 4,
        cross_layer_prefetch: bool = False,
    ) -> None:
        import torch

        from soup_cli.utils.safetensors_reader import SECTOR_BYTES

        if mode not in LOADER_MODES:
            raise ValueError(f"mode must be one of {LOADER_MODES}; got {mode!r}")
        if type(io_workers) is not int or not 1 <= io_workers <= 16:
            raise ValueError("io_workers must be an integer in [1, 16]")
        if type(ring_slots) is not int or not 2 <= ring_slots <= 64:
            raise ValueError("ring_slots must be an integer in [2, 64]")
        if type(cross_layer_prefetch) is not bool:
            raise ValueError("cross_layer_prefetch must be a bool")
        self.runtime = runtime
        self.primary = runtime.expert_pool
        self.readers = tuple(runtime.expert_readers)
        self.reader_to_idx = {reader: idx for idx, reader in enumerate(self.readers)}
        self.mode = mode
        self.io_workers = io_workers
        self.ring_slots = ring_slots
        self.cross_layer_prefetch = cross_layer_prefetch and mode == "cpp_async_2slot"
        self.cuda = self.primary.device.type == "cuda"
        if cuda_nf4_decode and (mode != "cpp_async_2slot" or not self.cuda):
            raise ValueError("cuda_nf4_decode requires CUDA and the cpp_async_2slot loader")
        self.sample_overlap = sample_overlap and self.cuda
        self.sample_stride = max(1, int(sample_stride))
        self.closed = False

        self.native = None
        self.companion = None
        self.slots = [self.primary]
        self.copy_stream = None
        self._nf4_launch = None
        self._nf4_args: Tuple[Tuple[int, ...], ...] = ()
        self._nf4_launches = 0
        self._decode_events: List[Tuple[Any, Any]] = []
        self._overlap_pairs: List[Tuple[Any, Any, Any, Any, Any]] = []
        self._sequence_counter = 0
        self._pending_next_sequence: Optional[Tuple[Any, List[int]]] = None
        self._orig_load_packed = self.primary._load_packed
        self._orig_host_cache = self.primary._host_cache

        if mode in ("cpp_sync", "cpp_async_2slot"):
            import _native_expert_stream_async as extension

            if self.primary.transfer != "packed" or SECTOR_BYTES != 4096:
                raise ValueError("C++ expert stream requires packed transfer and 4096B sectors")
            if cuda_nf4_decode:
                self._nf4_launch = getattr(extension, "decode_nf4_bf16_pair", None)
                if self._nf4_launch is None:
                    raise RuntimeError("build the C++ extension with --with-cuda-nf4")
            if len({os.path.realpath(r.path) for r in self.readers}) != len(self.readers):
                raise ValueError("duplicate expert shard path in runtime")
            budget = 0
            if self.primary._host_cache is not None:
                cache_state = self.primary._host_cache.stats()
                if cache_state["entries"]:
                    raise ValueError("attach AsyncExpertController before first expert request")
                budget = cache_state["budget_bytes"]
            regs = _build_registrations(self.primary, self.readers)
            self.native = extension.PackedStreamAsync(
                regs,
                int(self.primary.plan.host_staging_bytes),
                int(self.primary._packed_bytes),
                int(budget),
                io_workers,
                ring_slots,
            )

            if mode == "cpp_async_2slot":
                try:
                    self.companion = _make_companion_slot(self.primary, self.readers)
                    self.slots = [self.primary, self.companion]
                    if self.cuda:
                        self.copy_stream = torch.cuda.Stream(device=self.primary.device)
                    if cuda_nf4_decode:
                        self._nf4_args = tuple(_nf4_slot_args(slot) for slot in self.slots)
                except BaseException:
                    self.close()
                    raise

            def _native_load_packed(reader: Any, expert: int) -> None:
                layer_idx = self.reader_to_idx[reader]
                region = self.primary._pinned[: self.primary._packed_bytes]
                self.native.fill_sync(layer_idx, int(expert), int(region.data_ptr()))
                self.primary._arena[: self.primary._packed_bytes].copy_(
                    region, non_blocking=self.cuda
                )
                self.primary.h2d_copies += 1
                self._decode_slot(0)

            self.primary._load_packed = _native_load_packed
            if self._orig_host_cache is not None:
                self.primary._host_cache = self.native

    def _decode_slot(self, index: int, stream: Any = None) -> None:
        slot = self.slots[index]
        if self._nf4_launch is not None:
            if stream is None:
                import torch

                stream = torch.cuda.current_stream(slot.device)
            self._nf4_launch(stream.device.index, stream.cuda_stream, *self._nf4_args[index])
            self._nf4_launches += 1
        else:
            from soup_cli.utils.expert_shard import EXPERT_PARTS

            for part in EXPERT_PARTS:
                slot._dequantize(part, slot._specs[part])

    def _record_phase_read(
        self, phase: Optional[str], layer_idx: Optional[int], bytes_read: int
    ) -> None:
        primary = self.primary
        if primary.phase_read_tracking and phase is not None and layer_idx is not None:
            with primary._phase_lock:
                row = primary._phase_read_counters.setdefault(layer_idx, {}).setdefault(
                    phase,
                    {"expert_fetches": 0, "expert_bytes_read": 0},
                )
                row["expert_fetches"] += 1
                row["expert_bytes_read"] += int(bytes_read)

    def cache_stats(self) -> Optional[Dict[str, Any]]:
        if self.native is not None:
            return dict(self.native.stats())
        if self.primary._host_cache is not None:
            return dict(self.primary._host_cache.stats())
        return None

    def native_telemetry(self) -> Optional[Dict[str, Any]]:
        if self.native is None:
            return None
        data = dict(self.native.native_stats())
        data["cuda_nf4_launches"] = self._nf4_launches
        if self._decode_events:
            data["sampled_nf4_decodes"] = len(self._decode_events)
            data["sampled_nf4_decode_ms"] = sum(
                start.elapsed_time(end) for start, end in self._decode_events
            )
        if self._overlap_pairs:
            total_h2d_ms = 0.0
            overlap_ms = 0.0
            for base_ev, copy_a, copy_b, comp_a, comp_b in self._overlap_pairs:
                copy_b.synchronize()
                comp_b.synchronize()
                dur = copy_a.elapsed_time(copy_b)
                total_h2d_ms += dur
                lo = max(base_ev.elapsed_time(copy_a), base_ev.elapsed_time(comp_a))
                hi = min(base_ev.elapsed_time(copy_b), base_ev.elapsed_time(comp_b))
                overlap_ms += max(0.0, hi - lo)
            data["sampled_h2d_pairs"] = len(self._overlap_pairs)
            data["sampled_h2d_ms"] = total_h2d_ms
            data["sampled_h2d_overlap_fraction"] = (
                overlap_ms / total_h2d_ms if total_h2d_ms > 0.0 else 0.0
            )
        return data

    def reset_telemetry(self) -> None:
        self._overlap_pairs.clear()
        self._decode_events.clear()
        self._nf4_launches = 0
        self._sequence_counter = 0
        self._pending_next_sequence = None
        if self.native is not None:
            self.native.reset_native_stats()

    def set_next_hint(self, reader: Optional[Any], expert_ids: Optional[List[int]]) -> None:
        """Schedule the upcoming MoE layer's expert list for cross-layer prefetching."""
        if not self.cross_layer_prefetch or self.native is None:
            return
        if reader is None or not expert_ids:
            self._pending_next_sequence = None
            return
        self._pending_next_sequence = (reader, [int(eid) for eid in expert_ids])

    def clear_next_hint(self) -> None:
        self._pending_next_sequence = None
        if self.native is not None:
            self.native.cancel_next_sequence()

    @contextmanager
    def lease_one(
        self,
        reader: Any,
        expert_id: int,
        *,
        phase: Optional[str] = None,
        layer_idx: Optional[int] = None,
    ) -> Iterator[Tuple[Any, Any]]:
        """Single-expert synchronous lease (used by python_sync or cpp_sync)."""
        with self.primary.lease(
            reader, int(expert_id), phase=phase, layer_idx=layer_idx
        ) as weights:
            yield weights

    def iter_experts(
        self,
        reader: Any,
        expert_ids: List[int],
        *,
        phase: Optional[str] = None,
        layer_idx: Optional[int] = None,
    ) -> Iterator[Tuple[int, Any, Any]]:
        """Yield ``(expert_id, gate_up, down)`` across an ordered phase schedule."""
        if not expert_ids:
            return
        if self.mode in ("python_sync", "cpp_sync"):
            for eid in expert_ids:
                with self.lease_one(
                    reader, int(eid), phase=phase, layer_idx=layer_idx
                ) as (gate_up, down):
                    yield int(eid), gate_up, down
            return

        yield from self._iter_async_two_slot(
            reader, [int(eid) for eid in expert_ids], phase=phase, layer_idx=layer_idx
        )

    def _iter_async_two_slot(
        self,
        reader: Any,
        expert_ids: List[int],
        *,
        phase: Optional[str] = None,
        layer_idx: Optional[int] = None,
    ) -> Iterator[Tuple[int, Any, Any]]:
        with ExitStack() as locks:
            if not self.primary._lock.acquire(blocking=False):
                raise RuntimeError("primary expert buffer pool already has an active lease")
            locks.callback(self.primary._lock.release)
            if not self.companion._lock.acquire(blocking=False):
                raise RuntimeError("companion expert buffer pool already has an active lease")
            locks.callback(self.companion._lock.release)
            yield from self._run_async_sequence(
                reader, expert_ids, phase=phase, layer_idx=layer_idx
            )

    def _run_async_sequence(
        self,
        reader: Any,
        expert_ids: List[int],
        *,
        phase: Optional[str] = None,
        layer_idx: Optional[int] = None,
    ) -> Iterator[Tuple[int, Any, Any]]:
        import torch

        from soup_cli.utils.expert_shard import DOWN, GATE_UP

        primary = self.primary
        reader_idx = self.reader_to_idx[reader]
        packed_bytes = primary._packed_bytes
        self._sequence_counter += 1
        do_sample = self.sample_overlap and (self._sequence_counter - 1) % self.sample_stride == 0

        if self.cuda:
            compute_stream = torch.cuda.current_stream(primary.device)
            h2d_done = [torch.cuda.Event(), torch.cuda.Event()]
            compute_done = [torch.cuda.Event(), torch.cuda.Event()]
            for slot_idx in (0, 1):
                h2d_done[slot_idx].record(compute_stream)
                compute_done[slot_idx].record(compute_stream)
            base_ev = torch.cuda.Event(enable_timing=True) if do_sample else None
            if base_ev is not None:
                base_ev.record(compute_stream)
                self.copy_stream.wait_event(base_ev)
        else:
            compute_stream = None
            h2d_done = compute_done = None
            base_ev = None

        timing_copy: Dict[int, Tuple[Any, Any]] = {}
        timing_comp: Dict[int, Tuple[Any, Any]] = {}

        def _stage(pos: int) -> None:
            slot_idx = pos % 2
            slot = self.slots[slot_idx]
            if self.cuda:
                h2d_done[slot_idx].synchronize()
            before_bytes = reader.bytes_read
            got = self.native.consume_next(int(slot._pinned.data_ptr()))
            delta_bytes = reader.bytes_read - before_bytes
            primary.logical_bytes_read += delta_bytes
            if int(got) != expert_ids[pos]:
                raise RuntimeError(f"prefetch order mismatch: {got} != {expert_ids[pos]}")
            region = slot._pinned[:packed_bytes]
            if self.cuda:
                with torch.cuda.stream(self.copy_stream):
                    self.copy_stream.wait_event(compute_done[slot_idx])
                    if do_sample:
                        ca = torch.cuda.Event(enable_timing=True)
                        cb = torch.cuda.Event(enable_timing=True)
                        ca.record(self.copy_stream)
                        slot._arena[:packed_bytes].copy_(region, non_blocking=True)
                        cb.record(self.copy_stream)
                        timing_copy[pos] = (ca, cb)
                    else:
                        slot._arena[:packed_bytes].copy_(region, non_blocking=True)
                    h2d_done[slot_idx].record(self.copy_stream)
            else:
                slot._arena[:packed_bytes].copy_(region)
            primary.fetches += 1
            primary.h2d_copies += 1
            self._record_phase_read(phase, layer_idx, delta_bytes)

        self.native.start_sequence(reader_idx, expert_ids)
        if self.cross_layer_prefetch and self._pending_next_sequence is not None:
            next_reader, next_ids = self._pending_next_sequence
            self._pending_next_sequence = None
            next_reader_idx = self.reader_to_idx.get(next_reader)
            if next_reader_idx is not None and next_ids:
                self.native.enqueue_next_sequence(next_reader_idx, next_ids)
        try:
            _stage(0)
            total = len(expert_ids)
            for pos, eid in enumerate(expert_ids):
                slot_idx = pos % 2
                slot = self.slots[slot_idx]
                if self.cuda:
                    compute_stream.wait_event(h2d_done[slot_idx])
                    if do_sample:
                        ka = torch.cuda.Event(enable_timing=True)
                        kb = torch.cuda.Event(enable_timing=True)
                        ka.record(compute_stream)
                self._decode_slot(slot_idx, compute_stream)
                if do_sample:
                    decode_end = torch.cuda.Event(enable_timing=True)
                    decode_end.record(compute_stream)
                    self._decode_events.append((ka, decode_end))
                yield eid, slot._buffers[GATE_UP], slot._buffers[DOWN]
                if self.cuda:
                    if do_sample:
                        kb.record(compute_stream)
                        timing_comp[pos] = (ka, kb)
                    compute_done[slot_idx].record(compute_stream)
                if pos + 1 < total:
                    _stage(pos + 1)
        except BaseException:
            self.native.cancel_next_sequence()
            reader.close()
            raise
        finally:
            self.native.finish_sequence()
            if self.cuda:
                self.copy_stream.synchronize()
                compute_stream.synchronize()
                if do_sample and base_ev is not None:
                    for pos in range(1, len(expert_ids)):
                        if pos in timing_copy and (pos - 1) in timing_comp:
                            ca, cb = timing_copy[pos]
                            ka, kb = timing_comp[pos - 1]
                            self._overlap_pairs.append((base_ev, ca, cb, ka, kb))

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.primary._load_packed = self._orig_load_packed
        self.primary._host_cache = self._orig_host_cache
        try:
            if self.native is not None:
                self.native.close()
        finally:
            if self.companion is not None:
                self.companion.close()
