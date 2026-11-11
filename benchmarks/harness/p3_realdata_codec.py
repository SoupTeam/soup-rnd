"""Bounded storage policies for the trained-Granite benchmark, not production codecs.

The none and plain E4M3 policies delegate to unchanged production stores. Other
policies pad channel tails explicitly, quantize groups in at most 256-row tiles,
and undo static smoothing in FP32 before the final BF16 cast. Calls are serialized
and CUDA work is drained before returning, including on errors.

``allocation`` is pure byte arithmetic. A store reserves its entire host bound
before allocating any tensors. The bound includes owned persistent storage,
individual pinned-allocation rounding, and one operation's host scratch/output.
CUDA persistent copies and scratch/output are separate. Caller inputs, outputs
retained after get returns, Python objects, allocator caches, and framework
workspaces are not owned tensor storage and are not a process-RSS bound.

``round_trip`` is an uncharged, tiled calibration diagnostic. Only that diagnostic
allows format=none with a smoothing vector to test the reversible transform.
Training none policies cannot request smoothing or grouping.
"""

from __future__ import annotations

import math
import struct
import traceback
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from typing import Any

_TILE_ROWS = 256
_FORMATS = {"e4m3": ("float8_e4m3fn", 448), "e5m2": ("float8_e5m2", 57344)}


@dataclass(frozen=True)
class CodecPolicy:
    name: str
    format: str
    group_size: int | None = None
    smooth: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("a nonempty policy name is required")
        if self.format not in ("none", "e4m3", "e5m2"):
            raise ValueError("format must be none, e4m3, or e5m2")
        if self.group_size is not None and (
            type(self.group_size) is not int or self.group_size < 1
        ):
            raise ValueError("group_size must be a positive integer or None")
        if type(self.smooth) is not bool:
            raise ValueError("smooth must be a boolean")
        if self.format == "none" and (self.group_size is not None or self.smooth):
            raise ValueError("none policies cannot group or smooth")


def _shapes(mb_shapes: Sequence[Sequence[int]]) -> list[tuple[int, int, int]]:
    shapes = [tuple(shape) for shape in mb_shapes]
    if not shapes or any(
        len(shape) != 3 or any(type(dim) is not int or dim < 1 for dim in shape)
        for shape in shapes
    ):
        raise ValueError("mb_shapes must contain positive integer [B, L, H] shapes")
    if any(shape[-1] != shapes[0][-1] for shape in shapes):
        raise ValueError("all micro-batches must have the same hidden dimension")
    return shapes


def _rounded(size: int, pin: bool) -> int:
    return 1 << (size - 1).bit_length() if pin and size else size


def _baseline(policy: CodecPolicy) -> bool:
    return policy.format == "none" or (
        policy.format == "e4m3" and policy.group_size is None and not policy.smooth
    )


def allocation(
    policy: CodecPolicy, *, n_layers: int, mb_shapes: Sequence[Sequence[int]],
    dtype: Any, pin: bool, device_type: str,
) -> dict[str, Any]:
    """Return a JSON-safe conservative bound without importing torch.

    ``payload_bytes`` excludes separately listed padded channels. The uncompressed
    production store reserves max-sized slots even when its allocations are
    ragged; ``reservation_slack_bytes`` preserves that reservation honestly.
    Experimental scratch allows eight padded FP32 matrices, eight group FP32
    matrices, a BF16 input tile, FP8 transfer payload, finite masks, and a complete
    decoded BF16 output. Static-vector construction also gets a host FP32 row.
    """
    if not isinstance(policy, CodecPolicy):
        raise ValueError("policy must be a CodecPolicy")
    if type(n_layers) is not int or n_layers < 1:
        raise ValueError("n_layers must be a positive integer")
    if str(dtype) not in ("bfloat16", "torch.bfloat16"):
        raise ValueError("the benchmark compute dtype must be bfloat16")
    if type(pin) is not bool:
        raise ValueError("pin must be a boolean")
    if device_type not in ("cpu", "cuda"):
        raise ValueError("the benchmark supports cpu and cuda devices only")
    shapes = _shapes(mb_shapes)
    hidden = shapes[0][-1]
    rows = [shape[0] * shape[1] for shape in shapes]
    max_rows = max(rows)
    tile_rows = min(max_rows, _TILE_ROWS)
    output = max_rows * hidden * 2
    common = {
        "payload_bytes": 0, "scale_bytes": 0, "padding_bytes": 0,
        "pinned_rounding_bytes": 0, "calibration_vector_bytes": 0,
        "reservation_slack_bytes": 0, "host_scratch_bytes": 0,
        "device_persistent_bytes": 0, "device_scratch_bytes": 0,
        "production_reservation_bytes": 0, "decoded_output_peak_bytes": output,
        "tile_rows": tile_rows, "groups_per_row": 0, "padded_hidden": hidden,
    }
    if policy.format == "none":
        from soup_cli.utils.budgeted_activations import activation_allocation_bytes

        maximum = output
        production = activation_allocation_bytes(
            n_layers=n_layers, k=len(shapes), chunk_bytes=maximum,
            spill_layers=0, ring=1, pin=pin,
        )
        payload = n_layers * sum(rows) * hidden * 2
        max_requested = n_layers * len(shapes) * maximum
        scratch = 10 * tile_rows * hidden + 1 + output
        common.update(
            payload_bytes=payload, reservation_slack_bytes=max_requested - payload,
            pinned_rounding_bytes=production - max_requested,
            host_scratch_bytes=scratch if device_type == "cpu" else 1,
            device_scratch_bytes=scratch if device_type == "cuda" else 0,
            production_reservation_bytes=production,
        )
    elif _baseline(policy):
        from soup_cli.utils.fp8_activations import fp8_activation_allocation

        original = fp8_activation_allocation(
            n_layers=n_layers, mb_shapes=shapes, pin=pin,
            compute_dtype="bfloat16", device_type=device_type,
        )
        common.update(
            payload_bytes=original.payload_bytes, scale_bytes=original.scale_bytes,
            pinned_rounding_bytes=original.pinned_overhead_bytes,
            host_scratch_bytes=original.host_transient_bytes,
            device_scratch_bytes=original.device_transient_bytes,
            production_reservation_bytes=original.host_bytes, groups_per_row=1,
        )
    else:
        width = policy.group_size or hidden
        groups = (hidden + width - 1) // width
        padded = groups * width
        payload = n_layers * sum(rows) * hidden
        padding = n_layers * sum(rows) * (padded - hidden)
        scales = n_layers * sum(rows) * groups * 4
        vectors = n_layers * hidden * 4 if policy.smooth else 0
        requested = payload + padding + scales + vectors
        persistent = n_layers * sum(_rounded(count * padded, pin) for count in rows)
        persistent += _rounded(scales, pin) + _rounded(vectors, pin)
        # Includes conversion/transfer/finite temporaries rather than assuming
        # in-place arithmetic eliminates every backend allocation.
        scratch = (
            8 * 4 * tile_rows * padded + 8 * 4 * tile_rows * groups
            + 2 * tile_rows * hidden + tile_rows * padded
            + tile_rows * padded + tile_rows * groups + output + 1
        )
        common.update(
            payload_bytes=payload, scale_bytes=scales, padding_bytes=padding,
            pinned_rounding_bytes=persistent - requested, calibration_vector_bytes=vectors,
            host_scratch_bytes=(scratch if device_type == "cpu" else 1)
            + (hidden * 4 if policy.smooth else 0),
            device_persistent_bytes=vectors if device_type == "cuda" else 0,
            device_scratch_bytes=scratch if device_type == "cuda" else 0,
            groups_per_row=groups, padded_hidden=padded,
        )
    common["charged_host_bytes"] = sum(
        common[key] for key in (
            "payload_bytes", "scale_bytes", "padding_bytes", "pinned_rounding_bytes",
            "calibration_vector_bytes", "reservation_slack_bytes", "host_scratch_bytes",
        )
    )
    return common


def _positive_vector(values: Any, hidden: int) -> None:
    if not isinstance(values, list) or len(values) != hidden:
        raise ValueError("channel scales must be Python numeric lists of hidden width")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("channel scales must be Python numbers")
        try:
            rounded = struct.unpack("f", struct.pack("f", value))[0]
        except (OverflowError, struct.error) as error:
            raise ValueError("channel scales must be representable positive finite FP32") from error
        if not math.isfinite(rounded) or rounded <= 0:
            raise ValueError("channel scales must be representable positive finite FP32")


def _validate_tensor(tensor: Any, shape: tuple[int, ...] | None = None) -> None:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise ValueError("activation must be a torch tensor")
    _shapes([tuple(tensor.shape)])
    if shape is not None and tuple(tensor.shape) != shape:
        raise ValueError("activation shape differs from its declared [B, L, H] shape")
    if tensor.dtype != torch.bfloat16:
        raise ValueError("activation dtype must be bfloat16")
    if tensor.device.type not in ("cpu", "cuda"):
        raise ValueError("activation must be on the declared CPU or CUDA device")
    if tensor.layout != torch.strided:
        raise ValueError("activation must have strided tensor layout")


def _tiles(tensor: Any) -> Iterator[tuple[int, Any]]:
    """View row tiles where possible, otherwise stage only one BF16 row tile."""
    import torch

    rows = tensor.shape[0] * tensor.shape[1]
    hidden = tensor.shape[2]
    flat = staging = None
    try:
        try:
            flat = tensor.view(rows, hidden)
        except RuntimeError:
            pass
        if flat is not None:
            for start in range(0, rows, _TILE_ROWS):
                yield start, flat[start:start + _TILE_ROWS]
        else:
            staging = torch.empty(
                (min(rows, _TILE_ROWS), hidden), dtype=tensor.dtype, device=tensor.device,
            )
            for start in range(0, rows, _TILE_ROWS):
                count = min(rows - start, _TILE_ROWS)
                for offset in range(count):
                    batch, position = divmod(start + offset, tensor.shape[1])
                    staging[offset].copy_(tensor[batch, position])
                yield start, staging[:count]
    finally:
        flat = staging = tensor = None


def _finite(tensor: Any, context: str) -> None:
    import torch

    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{context} contains a nonfinite value; the step is refused")


def _finite_activation(tensor: Any) -> None:
    tiles = _tiles(tensor)
    values = None
    try:
        for _, values in tiles:
            _finite(values, "activation")
    finally:
        tiles.close()
        tiles = values = tensor = None


def _encode_tile(values: Any, policy: CodecPolicy, channel: Any) -> tuple[Any, Any]:
    import torch

    hidden = values.shape[-1]
    width = policy.group_size or hidden
    groups = (hidden + width - 1) // width
    staged = torch.zeros(
        (values.shape[0], groups * width), dtype=torch.float32, device=values.device,
    )
    staged[:, :hidden].copy_(values)
    if channel is not None:
        staged[:, :hidden].div_(channel)
    grouped = staged.view(values.shape[0], groups, width)
    absmax = grouped.abs().amax(dim=-1)
    _finite(absmax, "smoothed activation absmax")
    safe = torch.where(absmax > 0, absmax, torch.ones_like(absmax))
    dtype_name, maximum = _FORMATS[policy.format]
    grouped.div_(safe.unsqueeze(-1)).mul_(maximum).clamp_(-maximum, maximum)
    return staged.to(getattr(torch, dtype_name)), absmax


def _decode_tile(
    payload: Any, scales: Any, policy: CodecPolicy, channel: Any, hidden: int,
) -> Any:
    import torch

    width = policy.group_size or hidden
    groups = (hidden + width - 1) // width
    _, maximum = _FORMATS[policy.format]
    values = payload.float().view(payload.shape[0], groups, width)
    values.div_(maximum).mul_(scales.unsqueeze(-1))
    logical = values.view(payload.shape[0], groups * width)[:, :hidden]
    if channel is not None:
        logical.mul_(channel)
    _finite(logical, "decoded FP32 activation")
    restored = logical.to(torch.bfloat16)
    _finite(restored, "decoded BF16 activation")
    return restored


def round_trip(
    x: Any, *, format: str, group_size: int | None = None, channel_scale: Any = None,
) -> Any:
    """Uncharged diagnostic with the same tile math as experimental storage."""
    import torch

    _validate_tensor(x)
    if format == "none" and group_size is not None:
        raise ValueError("none cannot use channel groups")
    policy = CodecPolicy(
        "diagnostic", format, group_size, channel_scale is not None and format != "none",
    )
    # The none+smooth diagnostic is permitted without permitting a training policy.
    channel = out = flat = values = payload = scales = restored = None
    tiles = None
    try:
        if channel_scale is not None:
            _positive_vector(channel_scale, x.shape[-1])
        with torch.no_grad():
            channel = None if channel_scale is None else torch.tensor(
                channel_scale, dtype=torch.float32, device=x.device,
            )
            _finite_activation(x)
            out = torch.empty_like(x, memory_format=torch.contiguous_format)
            flat = out.view(-1, x.shape[-1])
            tiles = _tiles(x)
            for start, values in tiles:
                if format == "none":
                    if channel is None:
                        restored = values
                    else:
                        restored = values.float().div_(channel)
                        _finite(restored, "smoothed identity activation")
                        restored.mul_(channel)
                        _finite(restored, "restored identity activation")
                        restored = restored.to(torch.bfloat16)
                elif _baseline(policy):
                    from soup_cli.utils import activation_codec

                    payload, scales = activation_codec.encode(values, activation_codec.FP8)
                    restored = activation_codec.decode(
                        payload, scales, activation_codec.FP8, torch.bfloat16, x.shape[-1],
                    )
                else:
                    payload, scales = _encode_tile(values, policy, channel)
                    restored = _decode_tile(payload, scales, policy, channel, x.shape[-1])
                _finite(restored, "round-trip activation")
                flat[start:start + values.shape[0]].copy_(restored)
                values = payload = scales = restored = None
        return out
    except BaseException as error:
        out = None
        traceback.clear_frames(error.__traceback__)
        raise
    finally:
        if tiles is not None:
            tiles.close()
        channel = flat = values = payload = scales = restored = tiles = x = None


def _observations(tensors: list[tuple[str, Any]]) -> list[dict[str, Any]]:
    observations = []
    seen = set()
    for name, tensor in tensors:
        if tensor is None:
            continue
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        if key in seen:
            continue
        seen.add(key)
        observations.append({
            "name": name, "device": str(tensor.device), "dtype": str(tensor.dtype),
            "shape": list(tensor.shape), "storage_bytes": int(storage.nbytes()),
            "pinned": bool(tensor.is_pinned()) if tensor.device.type == "cpu" else False,
        })
    return observations


class _Store:
    """Common admission, strict boundaries, synchronous lifetime, and reporting."""

    def __init__(
        self, policy: CodecPolicy, *, host_budget: Any, n_layers: int,
        mb_shapes: Sequence[Sequence[int]], dtype: Any, device: Any, pin: bool,
    ) -> None:
        import torch

        self.policy = policy
        self.device = torch.device(device)
        self.allocation = allocation(
            policy, n_layers=n_layers, mb_shapes=mb_shapes, dtype=dtype, pin=pin,
            device_type=self.device.type,
        )
        if dtype != torch.bfloat16:
            raise ValueError("make_store requires torch.bfloat16, not a dtype name")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = dtype
        self.n_layers = n_layers
        self.mb_shapes = _shapes(mb_shapes)
        self.hidden = self.mb_shapes[0][-1]
        self.pin = pin
        self._closed = self._closing = False
        self._cuda_used = False
        self._written: set[tuple[int, int]] = set()
        self._put_calls = self._get_calls = 0
        self._lease = host_budget.reserve(
            f"realdata.codec.{id(self)}", self.allocation["charged_host_bytes"],
        )

    def _key(self, layer: int, mb: int) -> tuple[int, int]:
        if self._closed or self._closing:
            raise RuntimeError("benchmark activation store is closed or closing")
        if type(layer) is not int or type(mb) is not int:
            raise ValueError("layer and micro-batch indices must be integers")
        if not 0 <= layer < self.n_layers or not 0 <= mb < len(self.mb_shapes):
            raise IndexError("activation index exceeds declared capacity")
        return layer, mb

    def _prepare_put(self, layer: int, mb: int, x: Any) -> tuple[int, int]:
        key = self._key(layer, mb)
        _validate_tensor(x, self.mb_shapes[mb])
        if x.device != self.device:
            raise ValueError("activation is not on the reserved decoder device")
        self._written.discard(key)
        self._cuda_used = self.device.type == "cuda"
        _finite_activation(x)
        return key

    def _drain(self) -> None:
        if self._cuda_used:
            import torch

            torch.cuda.synchronize(self.device)
            self._cuda_used = False

    def begin_backward(self) -> None:
        self._key(0, 0)
        self._drain()

    def end_step(self) -> None:
        self._key(0, 0)
        self._drain()
        self._written.clear()

    def _owned(self) -> list[tuple[str, Any]]:
        raise NotImplementedError

    def report(self) -> dict[str, Any]:
        import torch

        owned = _observations(self._owned())
        live = not self._closed
        result = dict(self.allocation)
        for name in ("charged_host_bytes", "host_scratch_bytes", "device_persistent_bytes",
                     "device_scratch_bytes"):
            if not live:
                result[name] = 0
        result.update({
            "policy": asdict(self.policy), "allocation": dict(self.allocation),
            "underlying_store_class": self._underlying_class,
            "actual_owned_storages": owned,
            "owned_host_storage_bytes": sum(
                item["storage_bytes"] for item in owned if item["device"] == "cpu"
            ),
            "owned_device_storage_bytes": sum(
                item["storage_bytes"] for item in owned if item["device"].startswith("cuda")
            ),
            "cleanup": {"closed": self._closed, "closing": self._closing,
                        "host_lease_released": self._closed},
            "put_calls": self._put_calls, "get_calls": self._get_calls,
            "synchronous_transfers": True,
            "torch_version": str(torch.__version__),
            "payload_dtype": _FORMATS[self.policy.format][0]
            if self.policy.format != "none" else "bfloat16",
            "scale_semantics": "FP32 group absmax; divide then multiply by format maximum",
            "rounding": "PyTorch deterministic FP8 cast, finite clamp, no stochastic rounding",
            "format_maximum": _FORMATS[self.policy.format][1]
            if self.policy.format != "none" else None,
            "scratch_scope": "one operation including decoded output until returned to caller",
            "production_lease_is_partition_of_outer_lease": _baseline(self.policy),
        })
        return result


class _ProductionStore(_Store):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._store = None
        super().__init__(*args, **kwargs)
        try:
            from soup_cli.utils.host_memory_budget import HostMemoryBudget

            if self.policy.format == "none":
                from soup_cli.utils.budgeted_activations import BudgetedActivationStore

                store_type = BudgetedActivationStore
            else:
                from soup_cli.utils.fp8_activations import FP8ActivationStore

                store_type = FP8ActivationStore
            self._underlying_class = f"{store_type.__module__}.{store_type.__qualname__}"
            self._store = store_type(
                host_budget=HostMemoryBudget(self.allocation["production_reservation_bytes"]),
                n_layers=self.n_layers, k=len(self.mb_shapes),
                chunk_shape=max(self.mb_shapes, key=math.prod), mb_shapes=self.mb_shapes,
                dtype=self.dtype, device=self.device, pin=self.pin, spill_layers=0, ring=1,
            )
        except BaseException as error:
            traceback.clear_frames(error.__traceback__)
            self.close()
            raise

    def put(self, layer: int, mb: int, x: Any) -> None:
        import torch

        try:
            with torch.no_grad():
                key = self._prepare_put(layer, mb, x)
                self._store.put(layer, mb, x)
                self._drain()
                self._written.add(key)
                self._put_calls += 1
        except BaseException as error:
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            self._drain()
            x = None

    def get(self, layer: int, mb: int) -> Any:
        import torch

        out = None
        try:
            key = self._key(layer, mb)
            if key not in self._written:
                raise KeyError(f"activation {key} was not put this step")
            self._cuda_used = self.device.type == "cuda"
            with torch.no_grad():
                out = self._store.get(layer, mb)
                _validate_tensor(out, self.mb_shapes[mb])
                if out.device != self.device:
                    raise ValueError("production decoder returned a different device")
                _finite_activation(out)
            self._drain()
            self._get_calls += 1
            return out
        except BaseException as error:
            out = None
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            self._drain()

    def begin_backward(self) -> None:
        super().begin_backward()
        self._store.begin_backward()

    def end_step(self) -> None:
        try:
            super().end_step()
            self._store.end_step()
        finally:
            self._written.clear()

    def _owned(self) -> list[tuple[str, Any]]:
        if self._store is None:
            return []
        if self.policy.format == "none":
            return [(f"payload.{layer}.{mb}", tensor)
                    for (layer, mb), tensor in self._store._resident.items()]
        return [(f"payload.{layer}.{mb}", tensor)
                for (layer, mb), tensor in self._store._payload.items()] + [
                    ("scales", self._store._scales),
                ]

    def close(self) -> None:
        if self._closed:
            return
        self._closing = True
        self._drain()
        if self._store is not None:
            self._store.close()
        self._written.clear()
        self._lease.close()
        self._closed = True
        self._closing = False


class _ExperimentalStore(_Store):
    def __init__(self, *args: Any, channel_scales: Any, **kwargs: Any) -> None:
        import torch

        self._payload: dict[tuple[int, int], Any] = {}
        self._scales = self._channel_host = self._channel_device = None
        super().__init__(*args, **kwargs)
        self._underlying_class = f"{type(self).__module__}.{type(self).__qualname__}"
        try:
            self._row_offsets = [0]
            for shape in self.mb_shapes:
                self._row_offsets.append(self._row_offsets[-1] + shape[0] * shape[1])
            self._rows_per_layer = self._row_offsets[-1]
            self._groups = self.allocation["groups_per_row"]
            self._padded = self.allocation["padded_hidden"]
            payload_dtype = getattr(torch, _FORMATS[self.policy.format][0])
            for layer in range(self.n_layers):
                for mb, shape in enumerate(self.mb_shapes):
                    self._payload[layer, mb] = torch.empty(
                        (shape[0] * shape[1], self._padded), dtype=payload_dtype,
                        device="cpu", pin_memory=self.pin,
                    )
            self._scales = torch.empty(
                (self.n_layers * self._rows_per_layer, self._groups), dtype=torch.float32,
                device="cpu", pin_memory=self.pin,
            )
            if self.policy.smooth:
                self._channel_host = torch.empty(
                    (self.n_layers, self.hidden), dtype=torch.float32,
                    device="cpu", pin_memory=self.pin,
                )
                for layer in range(self.n_layers):
                    self._channel_host[layer].copy_(torch.tensor(
                        channel_scales[layer], dtype=torch.float32, device="cpu",
                    ))
                if self.device.type == "cuda":
                    self._cuda_used = True
                    self._channel_device = self._channel_host.to(self.device, non_blocking=False)
                    self._drain()
        except BaseException as error:
            traceback.clear_frames(error.__traceback__)
            self.close()
            raise
        finally:
            channel_scales = None

    def _scale_view(self, layer: int, mb: int) -> Any:
        offset = layer * self._rows_per_layer + self._row_offsets[mb]
        return self._scales[offset:offset + self.mb_shapes[mb][0] * self.mb_shapes[mb][1]]

    def _channel(self, layer: int) -> Any:
        slab = self._channel_device if self.device.type == "cuda" else self._channel_host
        return None if slab is None else slab[layer]

    def put(self, layer: int, mb: int, x: Any) -> None:
        import torch

        values = payload = scales = host_scales = channel = tiles = None
        try:
            with torch.no_grad():
                key = self._prepare_put(layer, mb, x)
                host_scales = self._scale_view(layer, mb)
                channel = self._channel(layer)
                tiles = _tiles(x)
                for start, values in tiles:
                    payload, scales = _encode_tile(values, self.policy, channel)
                    stop = start + values.shape[0]
                    self._payload[key][start:stop].copy_(payload, non_blocking=False)
                    host_scales[start:stop].copy_(scales, non_blocking=False)
                    values = payload = scales = None
                self._drain()
                self._written.add(key)
                self._put_calls += 1
        except BaseException as error:
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            if tiles is not None:
                tiles.close()
            values = payload = scales = host_scales = channel = tiles = x = None
            self._drain()

    def get(self, layer: int, mb: int) -> Any:
        import torch

        out = flat = host_scales = channel = payload = scales = restored = None
        try:
            key = self._key(layer, mb)
            if key not in self._written:
                raise KeyError(f"activation {key} was not put this step")
            self._cuda_used = self.device.type == "cuda"
            with torch.no_grad():
                shape = self.mb_shapes[mb]
                rows = shape[0] * shape[1]
                out = torch.empty(shape, dtype=self.dtype, device=self.device)
                flat = out.view(rows, self.hidden)
                host_scales = self._scale_view(layer, mb)
                channel = self._channel(layer)
                for start in range(0, rows, _TILE_ROWS):
                    stop = min(start + _TILE_ROWS, rows)
                    payload = self._payload[key][start:stop].to(self.device, non_blocking=False)
                    scales = host_scales[start:stop].to(self.device, non_blocking=False)
                    restored = _decode_tile(payload, scales, self.policy, channel, self.hidden)
                    flat[start:stop].copy_(restored)
                    payload = scales = restored = None
            self._drain()
            self._get_calls += 1
            return out
        except BaseException as error:
            out = None
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            flat = host_scales = channel = payload = scales = restored = None
            self._drain()

    def _owned(self) -> list[tuple[str, Any]]:
        return [(f"payload.{layer}.{mb}", tensor)
                for (layer, mb), tensor in self._payload.items()] + [
                    ("scales", self._scales), ("channel_scales.host", self._channel_host),
                    ("channel_scales.device", self._channel_device),
                ]

    def close(self) -> None:
        if self._closed:
            return
        self._closing = True
        self._drain()
        self._payload.clear()
        self._scales = self._channel_host = self._channel_device = None
        self._written.clear()
        self._lease.close()
        self._closed = True
        self._closing = False


def make_store(
    policy: CodecPolicy, *, host_budget: Any, n_layers: int,
    mb_shapes: Sequence[Sequence[int]], dtype: Any, device: Any, pin: bool,
    channel_scales: Any = None,
) -> Any:
    """Admit the whole bound, then construct an unchanged or experimental store.

    A baseline's private production lease is a partition of the outer reservation,
    not a second charge. This admits the full bound before even the production
    BF16 constructor's scalar dtype probe. The caller must close the returned store.
    """
    shapes = _shapes(mb_shapes)
    if not isinstance(policy, CodecPolicy):
        raise ValueError("policy must be a CodecPolicy")
    if policy.smooth:
        if not isinstance(channel_scales, list) or len(channel_scales) != n_layers:
            raise ValueError("smoothing requires one Python channel-scale list per layer")
        for vector in channel_scales:
            _positive_vector(vector, shapes[0][-1])
    elif channel_scales is not None:
        raise ValueError("channel_scales were supplied for a policy without smoothing")
    args = dict(
        host_budget=host_budget, n_layers=n_layers, mb_shapes=shapes,
        dtype=dtype, device=device, pin=pin,
    )
    if _baseline(policy):
        return _ProductionStore(policy, **args)
    return _ExperimentalStore(policy, channel_scales=channel_scales, **args)
