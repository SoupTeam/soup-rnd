"""Internal MoE streaming with a bounded reusable expert arena.

The eager expert loop is preserved. One reusable weight arena replaces the
original fetcher's independently allocated tensors. Backward refetches each
frozen expert; public training integration remains gated in layer_stream_runtime.
"""

from __future__ import annotations

import math
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple

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


def _buffer_layout(reader: ExpertShardReader, dtype: str) -> Tuple[ExpertBufferPlan, list]:
    import torch

    from soup_cli.utils.expert_shard import _expected_tensors
    from soup_cli.utils.layer_stream_runtime import _torch_dtype
    from soup_cli.utils.safetensors_reader import SECTOR_BYTES

    if dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError("expert compute dtype must be float32, float16 or bfloat16")
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
    for part in EXPERT_PARTS:
        shape = tuple(reader.shapes[part])
        add(part, shape, target, "dense")
        spec = reader.quant_specs.get(part)
        if spec is None:
            continue
        if spec.dtype != dtype or spec.quant_type != "nf4":
            raise ValueError("NF4 expert compute dtype must match its stored NF4 spec")
        if spec.blocksize not in (32, 64, 128, 256, 512, 1024, 2048, 4096):
            raise ValueError("unsupported NF4 expert blocksize")
        for idx, (stored_shape, stored_dtype) in enumerate(_expected_tensors(part, shape, spec)):
            add(f"{part}:{idx}", tuple(stored_shape), stored_dtype, "packed")
        if spec.nested:
            blocks = max(blocks, -(-math.prod(shape) // spec.blocksize))
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
    )
    return plan, entries


def plan_expert_buffer(reader: ExpertShardReader, *, dtype: str) -> ExpertBufferPlan:
    """Plan storage without reading any expert or allocating the weight arena."""
    return _buffer_layout(reader, dtype)[0]


class ExpertBufferPool:
    """One synchronous expert lease at a time, backed by one bounded arena.

    Returned weights are valid only inside ``lease``; they alias the reusable
    arena, never the host staging. Concurrent/reentrant leases fail explicitly.
    CUDA work is synchronized before releasing a lease or recycling staging.
    Readers belong to the caller; close them separately (including on success).
    """

    def __init__(
        self,
        reader: ExpertShardReader,
        *,
        device: str,
        dtype: str,
        weight_budget_bytes: int,
        codes: Optional[Mapping[str, Any]] = None,
    ) -> None:
        import torch

        from soup_cli.utils.expert_shard import aligned_staging
        from soup_cli.utils.layer_shard import NF4_CODE_KEY, NF4_NESTED_CODE_KEY

        if (
            isinstance(weight_budget_bytes, bool)
            or not isinstance(weight_budget_bytes, int)
            or weight_budget_bytes < 1
        ):
            raise ValueError("weight_budget_bytes must be a positive integer")
        self.plan, layout = _buffer_layout(reader, dtype)
        if self.plan.weight_bytes > weight_budget_bytes:
            raise ValueError(
                f"expert slot needs {self.plan.weight_bytes} bytes; budget is {weight_budget_bytes}"
            )
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("expert forward supports CPU or CUDA only")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = dtype
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
        self._closed = False
        self.fetches = 0
        self.logical_bytes_read = 0
        self._arena = None
        self._buffers: dict = {}
        self._staging_owner = self._staging = None
        before = torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else 0
        try:
            self._arena = torch.empty(self.plan.weight_bytes, dtype=torch.uint8, device=self.device)
            for name, shape, kind, offset, size in layout:
                self._buffers[name] = self._arena[offset : offset + size].view(kind).view(shape)
            for name, value in tables.items():
                self._buffers[name].copy_(value, non_blocking=False)
            self._staging_owner, self._staging = aligned_staging(reader.max_span)
            self._synchronize()
            self.cuda_allocation_delta = (
                torch.cuda.memory_allocated(self.device) - before
                if self.device.type == "cuda" else 0
            )
            if self.cuda_allocation_delta > weight_budget_bytes:
                raise ValueError("CUDA allocator rounded the expert slot above the weight budget")
        except BaseException:
            self._buffers.clear()
            self._arena = self._staging_owner = self._staging = None
            self._closed = True
            raise

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

    def _load(self, reader: ExpertShardReader, expert: int) -> None:
        from soup_cli.utils.expert_shard import expert_key
        from soup_cli.utils.layer_stream_runtime import quant_sidecar_keys

        before = reader.bytes_read
        try:
            host = reader.read_host(expert, self._staging)
            for part in EXPERT_PARTS:
                key = expert_key(expert, part)
                spec = self._specs.get(part)
                if spec is None:
                    self._buffers[part].copy_(host[key], non_blocking=False)
                    continue
                for idx, name in enumerate(quant_sidecar_keys(key, spec)):
                    self._buffers[f"{part}:{idx}"].copy_(host[name], non_blocking=False)
                self._dequantize(part, spec)
            self.fetches += 1
        finally:
            self.logical_bytes_read += reader.bytes_read - before

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
    def lease(self, reader: ExpertShardReader, expert: int) -> Iterator[Tuple[Any, Any]]:
        """Hold one expert until all its computation has finished."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("expert buffer pool already has an active lease")
        try:
            self.validate_reader(reader)
            try:
                self._load(reader, expert)
                yield self._buffers[GATE_UP], self._buffers[DOWN]
            except BaseException:
                reader.close()
                raise
            finally:
                self._synchronize()
        finally:
            self._lock.release()

    def stats(self) -> dict:
        return {
            "slot_count": 1,
            "weight_budget_bytes": self.weight_budget_bytes,
            "plan": asdict(self.plan),
            "cuda_allocation_delta": self.cuda_allocation_delta,
            "fetches": self.fetches,
            "logical_bytes_read": self.logical_bytes_read,
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
                self._buffers.clear()
                self._arena = self._staging_owner = self._staging = None
                self._closed = True
        finally:
            self._lock.release()


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


def _build_streamed_experts_class() -> Any:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    def expert_mlp(hidden: Any, gate_up: Any, down: Any, act_fn: Any) -> Any:
        gate, up = functional.linear(hidden, gate_up).chunk(2, dim=-1)
        return functional.linear(act_fn(gate) * up, down)

    class FrozenExpert(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, hidden: Any, owner: Any, expert: int) -> Any:
            ctx.owner = owner
            ctx.act_fn = owner.act_fn
            ctx.expert = expert
            ctx.device_type = hidden.device.type
            ctx.autocast = {
                "enabled": torch.is_autocast_enabled(ctx.device_type),
                "dtype": torch.get_autocast_dtype(ctx.device_type),
                "cache_enabled": torch.is_autocast_cache_enabled(),
            }
            ctx.save_for_backward(hidden)
            with owner.pool.lease(owner.reader, expert) as (gate_up, down):
                return expert_mlp(hidden, gate_up, down, owner.act_fn)

        @staticmethod
        def backward(ctx: Any, grad_out: Any) -> Tuple[Any, None, None]:
            if torch.is_grad_enabled():
                raise RuntimeError("expert streaming does not support higher-order gradients")
            if not ctx.needs_input_grad[0]:
                return None, None, None
            _validate_training_activation(ctx.act_fn)
            (hidden,) = ctx.saved_tensors
            with ctx.owner.pool.lease(ctx.owner.reader, ctx.expert) as (gate_up, down):
                with torch.enable_grad(), torch.autocast(ctx.device_type, **ctx.autocast):
                    local = hidden.detach().requires_grad_(True)
                    result = expert_mlp(local, gate_up, down, ctx.act_fn)
                    (grad_hidden,) = torch.autograd.grad(result, local, grad_out)
            return grad_hidden, None, None

    class StreamedExperts(nn.Module):
        """Internal frozen-expert forward and first-order backward replacement."""

        def __init__(
            self, reader: ExpertShardReader, pool: ExpertBufferPool, act_fn: Any, layer_idx: int
        ) -> None:
            super().__init__()
            pool.validate_reader(reader)
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

        def forward(self, hidden_states: Any, top_k_index: Any, top_k_weights: Any) -> Any:
            if not self._forward_lock.acquire(blocking=False):
                raise RuntimeError("expert forward is already active")
            try:
                self.last_trace = []
                self.pool.validate_reader(self.reader)
                _validate_routing(hidden_states, top_k_index, top_k_weights, self.pool)
                if torch.is_grad_enabled() and (
                    hidden_states.requires_grad or top_k_weights.requires_grad
                ):
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
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
                for expert_idx in expert_hit:
                    expert = int(expert_idx[0])
                    top_k_pos, token_idx = torch.where(expert_mask[expert])
                    row = {
                        "layer": self.layer_idx, "expert": expert,
                        "token_routes": token_idx.numel(),
                        "logical_bytes_read": 0, "status": "loading",
                    }
                    self.last_trace.append(row)
                    before = self.reader.bytes_read
                    try:
                        current_state = hidden_states[token_idx]
                        current = FrozenExpert.apply(current_state, self, expert)
                        current = current * top_k_weights[token_idx, top_k_pos, None]
                        final_hidden_states.index_add_(
                            0, token_idx, current.to(final_hidden_states.dtype)
                        )
                        row["status"] = "ok"
                    except BaseException:
                        row["status"] = "error"
                        raise
                    finally:
                        row["logical_bytes_read"] = self.reader.bytes_read - before
                return final_hidden_states
            finally:
                self._forward_lock.release()

    return StreamedExperts


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


def install_expert_forward(
    layers: Sequence[Any], readers: Mapping[int, ExpertShardReader], pool: ExpertBufferPool
) -> int:
    """Replace frozen routed experts after validating every layer.

    Router, shared experts and dense layers stay with their original MoE code.
    Readers and pool are caller-owned. Invalid installs do not alter the model.
    First-order backward refetches weights within a lease and requires the
    validated stateless SiLU activation; routed-expert adapters are refused.
    """
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
        expected = (
            (reader.count, *reader.shapes[GATE_UP]), (reader.count, *reader.shapes[DOWN])
        )
        actual = tuple(
            tuple(getattr(module, name, ()).shape) if hasattr(module, name) else ()
            for name in ("gate_up_proj", "down_proj")
        )
        if actual != expected or not callable(getattr(module, "act_fn", None)):
            raise ValueError(f"layer {idx}: experts shape/activation mismatch; reshard checkpoint")
        parameters = dict(module.named_parameters())
        if set(parameters) != {"gate_up_proj", "down_proj"}:
            raise ValueError(
                f"layer {idx}: expert adapters/extra parameters are unsupported in B1.2"
            )
        if any(parameter.requires_grad for parameter in parameters.values()):
            raise ValueError(f"layer {idx}: B1.2 requires frozen routed-expert weights")
        identity = (id(parent), attr)
        if identity in seen:
            raise ValueError("aliased expert modules cannot share multiple layer replacements")
        seen.add(identity)
        replacements.append((parent, attr, cls(reader, pool, module.act_fn, idx)))
    for parent, attr, replacement in replacements:
        setattr(parent, attr, replacement)
    return len(replacements)
