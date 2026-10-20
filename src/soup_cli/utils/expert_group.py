"""Internal D3 grouped weight restore, with the unchanged B1.2 eager GEMMs.

Forward only. CPU NF4 is a reference decoder; CUDA BF16 uses grouped Triton
restore. This is not fused grouped GEMM, a public training path or D3 acceptance.
"""

from __future__ import annotations

import math
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Iterator, Mapping

from soup_cli.utils.expert_shard import EXPERT_PARTS, ExpertShardReader
from soup_cli.utils.expert_stream import ExpertBufferPool, _buffer_layout


@dataclass(frozen=True)
class GroupBufferPlan:
    slot_count: int
    slot_weight_bytes: int
    metadata_bytes: int
    weight_bytes: int
    host_staging_bytes: int


def plan_grouped_expert_buffer(
    reader: ExpertShardReader, *, dtype: str, group_width: int,
) -> GroupBufferPlan:
    """Count full B1.2 slots and padded pointer tables; no reads or allocations."""
    if type(group_width) is not int or not 1 <= group_width <= 4:
        raise ValueError("group width must be an integer from 1 through 4")
    slot, _layout = _buffer_layout(reader, dtype)
    # Two parts, five pointer tables each; each table is rounded to a 512-byte bin.
    metadata = 10 * 512 if reader.quant_specs else 0
    return GroupBufferPlan(
        group_width, slot.weight_bytes, metadata,
        group_width * slot.weight_bytes + metadata, slot.host_staging_bytes,
    )


class GroupedExpertPool:
    """One group lease, one arena and one blocking host staging buffer.

    Weights alias arena storage and must not escape their lease. CPU NF4 decoder
    temporaries, activations, GEMM workspaces and context/reservations are separate.
    """

    def __init__(
        self, reader: ExpertShardReader, *, device: str, dtype: str,
        group_width: int, weight_budget_bytes: int, codes: Mapping | None = None,
    ) -> None:
        import torch

        from soup_cli.utils.expert_shard import aligned_staging
        from soup_cli.utils.layer_shard import NF4_CODE_KEY, NF4_NESTED_CODE_KEY

        if type(weight_budget_bytes) is not int or weight_budget_bytes < 1:
            raise ValueError("weight budget must be a positive integer")
        self.plan = plan_grouped_expert_buffer(reader, dtype=dtype, group_width=group_width)
        if self.plan.weight_bytes > weight_budget_bytes:
            raise ValueError(
                f"group needs {self.plan.weight_bytes} bytes; budget is {weight_budget_bytes}"
            )
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("grouped restore supports CPU or CUDA only")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if reader.quant_specs and self.device.type == "cuda" and dtype != "bfloat16":
            raise ValueError("grouped CUDA NF4 restore requires BF16")
        self.dtype = dtype
        self.group_width = group_width
        self.weight_budget_bytes = weight_budget_bytes
        self._shapes = dict(reader.shapes)
        self._specs = dict(reader.quant_specs)
        self._stored_dtype = reader.dtype
        self._max_span = reader.max_span
        self._lock = threading.Lock()
        self._closed = False
        self._arena = self._staging_owner = self._staging = None
        self._slots: list = []
        self._metadata: dict = {}
        self._decoder = None
        self.fetches = 0
        self.logical_bytes_read = 0
        self.groups_loaded = 0
        self.last_loads: list = []
        _slot_plan, layout = _buffer_layout(reader, dtype)
        tables = {}
        for name, key in (("code", NF4_CODE_KEY), ("nested_code", NF4_NESTED_CODE_KEY)):
            if not any(row[0] == name for row in layout):
                continue
            value = (codes or {}).get(key)
            count = 16 if name == "code" else 256
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                    or tuple(value.shape) != (count,) or not torch.isfinite(value).all().item()):
                raise ValueError(f"missing or invalid NF4 code table: {key}")
            tables[name] = value
        for spec in self._specs.values():
            block = spec.nested_blocksize
            if spec.nested and (block < 1 or block > 4096 or block & (block - 1)):
                raise ValueError("unsupported nested NF4 blocksize")
        if self._specs and self.device.type == "cuda":
            with torch.cuda.device(self.device):
                if (not torch.cuda.is_bf16_supported(including_emulation=False)
                        or torch.cuda.get_device_capability(self.device)[0] < 8):
                    raise ValueError("grouped CUDA NF4 requires sm_80+ and native BF16")
            from soup_cli.utils.expert_group_decode import build_group_decoder

            self._decoder = build_group_decoder()
        before = torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else 0
        try:
            self._arena = torch.empty(self.plan.weight_bytes, dtype=torch.uint8, device=self.device)
            for index in range(group_width):
                buffers = {}
                start = index * self.plan.slot_weight_bytes
                for name, shape, kind, offset, size in layout:
                    buffers[name] = self._arena[start + offset:start + offset + size].view(
                        kind
                    ).view(shape)
                for name, value in tables.items():
                    buffers[name].copy_(value, non_blocking=False)
                self._slots.append(SimpleNamespace(device=self.device, _buffers=buffers))
            self._staging_owner, self._staging = aligned_staging(reader.max_span)
            if self._specs and self.device.type == "cuda":
                self._initialize_metadata()
            self._synchronize()
            self.cuda_allocation_delta = (
                torch.cuda.memory_allocated(self.device) - before
                if self.device.type == "cuda" else 0
            )
            if self.cuda_allocation_delta > weight_budget_bytes:
                raise ValueError("CUDA allocator rounded group above the weight budget")
        except BaseException:
            self._release()
            raise

    def _initialize_metadata(self) -> None:
        import torch

        offset = self.group_width * self.plan.slot_weight_bytes
        for part in EXPERT_PARTS:
            spec = self._specs.get(part)
            if spec is None:
                continue
            names = [f"{part}:0", f"{part}:1"]
            names += [f"{part}:2", f"{part}:3"] if spec.nested else [part, part]
            names += [part]
            tables = []
            for name in names:
                table = self._arena[offset:offset + self.group_width * 8].view(torch.int64)
                addresses = torch.tensor(
                    [slot._buffers[name].data_ptr() for slot in self._slots], dtype=torch.int64,
                )
                table.copy_(addresses, non_blocking=False)
                tables.append(table)
                offset += 512
            self._metadata[part] = tables

    def validate_reader(self, reader: ExpertShardReader) -> None:
        if self._closed:
            raise RuntimeError("group pool is closed")
        if (dict(reader.shapes) != self._shapes or dict(reader.quant_specs) != self._specs
                or reader.dtype != self._stored_dtype or reader.max_span > self._max_span):
            raise ValueError("expert reader does not match grouped buffer pool")

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            import torch

            torch.cuda.synchronize(self.device)

    def _load_raw(self, reader: ExpertShardReader, expert: int, slot: Any) -> None:
        from soup_cli.utils.expert_shard import expert_key
        from soup_cli.utils.layer_stream_runtime import quant_sidecar_keys

        before = reader.bytes_read
        row = {"expert": expert, "logical_bytes_read": 0, "status": "loading"}
        self.last_loads.append(row)
        try:
            host = reader.read_host(expert, self._staging)
            for part in EXPERT_PARTS:
                key = expert_key(expert, part)
                spec = self._specs.get(part)
                if spec is None:
                    slot._buffers[part].copy_(host[key], non_blocking=False)
                else:
                    for index, name in enumerate(quant_sidecar_keys(key, spec)):
                        slot._buffers[f"{part}:{index}"].copy_(host[name], non_blocking=False)
            row["status"] = "loaded"
            self.fetches += 1
        except BaseException:
            row["status"] = "error"
            raise
        finally:
            row["logical_bytes_read"] = reader.bytes_read - before
            self.logical_bytes_read += row["logical_bytes_read"]

    def _restore(self, count: int) -> None:
        for part, spec in self._specs.items():
            if self.device.type == "cpu":
                for slot in self._slots[:count]:
                    ExpertBufferPool._dequantize(slot, part, spec)
            else:
                import torch

                elements = math.prod(self._shapes[part])
                first = self._slots[0]._buffers
                with torch.cuda.device(self.device):
                    self._decoder[(-(-elements // 1024), count)](
                        *self._metadata[part], first["code"],
                        first["nested_code"] if spec.nested else first["code"],
                        elements, spec.blocksize, spec.nested_blocksize if spec.nested else 1,
                        spec.nested, 1024, num_warps=4, enable_fp_fusion=False,
                    )

    @contextmanager
    def lease(
        self, reader: ExpertShardReader, experts: list[int], *, trace: list | None = None,
    ) -> Iterator[tuple]:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("group pool already has an active lease")
        try:
            self.validate_reader(reader)
            if (not 1 <= len(experts) <= self.group_width
                    or any(type(expert) is not int or not 0 <= expert < reader.count
                           for expert in experts)
                    or experts != sorted(set(experts))):
                raise ValueError("group IDs must be unique, ascending and within width/reader")
            if trace:
                raise ValueError("group load trace must be empty on entry")
            self.last_loads = [] if trace is None else trace
            try:
                try:
                    for expert, slot in zip(experts, self._slots):
                        self._load_raw(reader, expert, slot)
                    self._restore(len(experts))
                    self.groups_loaded += 1
                    yield tuple((slot._buffers["gate_up"], slot._buffers["down"])
                                for slot in self._slots[:len(experts)])
                finally:
                    try:
                        self._synchronize()
                    except BaseException:
                        # Completion is unknown: never allow another lease to reuse this arena.
                        self._closed = True
                        raise
            except BaseException:
                reader.close()
                raise
        finally:
            self._lock.release()

    def stats(self) -> dict:
        return {
            "plan": asdict(self.plan), "weight_budget_bytes": self.weight_budget_bytes,
            "cuda_allocation_delta": self.cuda_allocation_delta,
            "fetches": self.fetches, "logical_bytes_read": self.logical_bytes_read,
            "groups_loaded": self.groups_loaded, "closed": self._closed,
            "decode_backend": (
                "none" if not self._specs else
                "cpu-reference" if self.device.type == "cpu" else "triton-grouped"
            ),
            "gemm_backend": "unchanged-eager",
            "cpu_nf4_temporaries": self.device.type == "cpu" and bool(self._specs),
        }

    def _release(self) -> None:
        self._metadata.clear()
        for slot in self._slots:
            slot._buffers.clear()
        self._slots.clear()
        self._arena = self._staging_owner = self._staging = None
        self._closed = True

    def close(self) -> None:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("cannot close group pool with an active lease")
        try:
            try:
                self._synchronize()
            finally:
                self._release()
        finally:
            self._lock.release()


_GROUPED_EXPERTS_CLASS = None


def grouped_experts_class() -> Any:
    global _GROUPED_EXPERTS_CLASS
    if _GROUPED_EXPERTS_CLASS is not None:
        return _GROUPED_EXPERTS_CLASS
    import torch
    import torch.nn.functional as functional

    from soup_cli.utils.expert_stream import _validate_routing, streamed_experts_class

    class GroupedExperts(streamed_experts_class()):
        """Grouped weight restoration only; mathematical expert operations stay eager."""

        def _compute_group(self, weights, routes, rows, hidden_states, top_k_weights, output):
            for pair, (position, token), row in zip(weights, routes, rows):
                gate_up, down = pair
                gate, up = functional.linear(hidden_states[token], gate_up).chunk(2, dim=-1)
                current = functional.linear(self.act_fn(gate) * up, down)
                current = current * top_k_weights[token, position, None]
                output.index_add_(0, token, current.to(output.dtype))
                row["status"] = "computed"

        def forward(self, hidden_states: Any, top_k_index: Any, top_k_weights: Any) -> Any:
            if not self._forward_lock.acquire(blocking=False):
                raise RuntimeError("expert forward is already active")
            try:
                self.last_trace = []
                self.pool.validate_reader(self.reader)
                _validate_routing(hidden_states, top_k_index, top_k_weights, self.pool)
                if top_k_index.shape[1] > self.num_experts:
                    raise ValueError("routing top-k exceeds the expert count")
                if top_k_index.numel() and (
                    top_k_index.min().item() < 0 or top_k_index.max().item() >= self.num_experts
                ):
                    raise ValueError("routing contains an expert outside the reader")
                output = torch.zeros_like(hidden_states)
                mask = functional.one_hot(top_k_index, num_classes=self.num_experts)
                mask = mask.permute(2, 1, 0)
                active = (mask.sum(dim=(-1, -2)) > 0).nonzero().flatten().tolist()
                for start in range(0, len(active), self.pool.group_width):
                    experts = active[start:start + self.pool.group_width]
                    routes = [torch.where(mask[expert]) for expert in experts]
                    rows = [{
                        "layer": self.layer_idx, "expert": expert,
                        "token_routes": route[1].numel(), "status": "loading",
                        "logical_bytes_read": 0,
                    } for expert, route in zip(experts, routes)]
                    self.last_trace.extend(rows)
                    load_trace: list = []
                    try:
                        with self.pool.lease(self.reader, experts, trace=load_trace) as weights:
                            self._compute_group(
                                weights, routes, rows, hidden_states, top_k_weights, output,
                            )
                        # The lease also waits for device completion before reporting success.
                        for row in rows:
                            row["status"] = "ok"
                    except BaseException:
                        for row in rows:
                            if row["status"] in ("loading", "computed"):
                                row["status"] = "error"
                        raise
                    finally:
                        loads = {row["expert"]: row for row in load_trace}
                        for row in rows:
                            row["logical_bytes_read"] = loads.get(row["expert"], {}).get(
                                "logical_bytes_read", 0,
                            )
                return output
            finally:
                self._forward_lock.release()

    _GROUPED_EXPERTS_CLASS = GroupedExperts
    return _GROUPED_EXPERTS_CLASS
