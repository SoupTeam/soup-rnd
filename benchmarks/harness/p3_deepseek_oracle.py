"""Saved original-forward sigmoid Jacobians for the tiny DeepSeek smoke.

Only native DeepseekV3MoE consumers are supported. They discard the first gate
return, so replay returns ``(None, FP32 weights, ordered int64 IDs)``. Router-logit
consumers, auxiliary router losses, router adapters, autocast and higher-order
VJPs are refused. This forced replay measures no natural routing stability.

One probe owns one optimizer window. Enter ``visit`` around each layer call,
then call ``require_complete`` before updating adapters and ``close`` in finally.
Recompute must have grad enabled; the original L2L forward must have it disabled.
The graph keeps selected sigmoid scores, normalizers and the current substituted
frozen weight, never a hidden-state primal or an original-forward pool view.

Reservations bound owned tensor storage, not RSS. Native model outputs/experts,
caller tensors, allocator caches, Python bookkeeping and kernel workspaces are
excluded. CPU operation scratch is charged to the shared host budget. CUDA
scratch is admitted separately and permits only one pending replay at a time.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import traceback
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MethodType
from typing import Any, Iterator, Sequence

__all__ = ["DeepseekOracleProbe"]

_NATIVE_MODULE = "transformers.models.deepseek_v3.modeling_deepseek_v3"
_HASH_ELEMENTS = 4096
_EPSILON = 1e-20


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"DeepSeek Oracle: {message}")


def _stamp(tensor: Any) -> tuple[Any, ...]:
    return (id(tensor), tensor._version, tensor.data_ptr(), str(tensor.dtype),
            str(tensor.device), tuple(tensor.shape), tuple(tensor.stride()))


def _settings(gate: Any) -> tuple[Any, ...]:
    return (gate.hidden_dim, gate.num_experts, gate.top_k, gate.num_group,
            gate.topk_group, gate.norm_topk_prob, gate.routed_scaling_factor)


def _source_identity(target: Any) -> dict[str, str]:
    filename = inspect.getsourcefile(target)
    _require(filename is not None, "native source identity is unavailable")
    digest = hashlib.sha256()
    with Path(filename).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": filename, "sha256": digest.hexdigest()}


def _allocation_bound(
    shapes: Sequence[Sequence[int]], moe_layers: int, experts: int, topk: int,
    device_type: str,
) -> dict[str, Any]:
    tokens = [shape[0] * shape[1] for shape in shapes]
    largest, width = max(tokens), shapes[0][2]
    selected = largest * topk
    largest_tensor = max(largest * width, experts * width, largest * experts, selected)
    operation = {
        "native_sigmoid_recovery_fp32": 4 * largest * experts,
        "selected_capture_fp32": 4 * selected,
        "normalizer_capture_fp32": 8 * largest,
        "native_weight_validation_fp32": 12 * selected,
        "replay_ids": 8 * selected,
        "replay_scores_fp32": 4 * selected,
        "replay_normalizers_fp32": 4 * largest,
        "returned_ids": 8 * selected,
        "returned_weights_fp32": 8 * selected,
        "incoming_contiguous_fp32": 4 * selected,
        "scaled_incoming_fp32": 4 * selected,
        "normalization_backward_fp32": 24 * selected + 8 * largest,
        "sigmoid_backward_fp32": 12 * selected,
        "scatter_logit_gradient_fp32": 4 * largest * experts,
        "restored_router_weight_fp32": 4 * experts * width,
        "hidden_gradient_fp32": 4 * largest * width,
        "returned_hidden_gradient": 4 * largest * width,
        "finite_validation": 8 * min(_HASH_ELEMENTS, largest_tensor) + 16,
    }
    scratch = sum(operation.values())
    rows = moe_layers * sum(tokens)
    host = {
        "ids": 8 * rows * topk,
        "selected_sigmoid_scores": 4 * rows * topk,
        "normalizers": 4 * rows,
        # D2H digest tiles are synchronous, unpinned, and never copied to Python bytes.
        "hash_staging": 8 * min(_HASH_ELEMENTS, largest_tensor),
        "cpu_operation_scratch": scratch if device_type == "cpu" else 0,
    }
    return {
        "host": host, "operation_buffer_bound": operation,
        "device_scratch_bound_bytes": scratch if device_type == "cuda" else 0,
        "maximum_simultaneous_replays": 1, "host_pinned": False,
        "max_tokens": largest, "tensor_storage_only": True,
        "exclusions": ["native model outputs and expert execution", "caller-owned tensors",
                       "Python bookkeeping", "framework allocator caches", "kernel workspaces"],
    }


@dataclass
class _Binding:
    gate: Any
    wrapper: Any
    moe: Any
    native_moe_forward: Any
    settings: tuple[Any, ...]
    device: Any
    offset: int


@dataclass
class _ForwardRecord:
    ids: Any
    scores: Any
    normalizer: Any
    hidden_shape: tuple[int, ...]
    hidden_dtype: Any
    settings: tuple[Any, ...]
    weight_digest: str
    bias_digest: str
    weight_stamp: tuple[Any, ...] | None
    bias_stamp: tuple[Any, ...] | None

    def metadata(self) -> tuple[Any, ...]:
        return (self.hidden_shape, str(self.hidden_dtype), self.settings,
                self.weight_digest, self.bias_digest, self.weight_stamp, self.bias_stamp)

    def clear(self) -> None:
        self.ids = self.scores = self.normalizer = None


class _Replay:
    """Clearable graph state, including when the caller retains output/exception objects."""

    def __init__(self, owner: DeepseekOracleProbe, pair: tuple[int, int]) -> None:
        self.owner = weakref.ref(owner)
        self.pair = pair
        self.ids = self.scores = self.normalizer = None
        self.weight = self.bias = None
        self.weight_stamp = self.bias_stamp = None
        self.digest = ""
        self.shape: tuple[int, ...] = ()
        self.dtype: Any = None
        self.settings: tuple[Any, ...] = ()
        self.closed = False

    def clear(self) -> None:
        self.ids = self.scores = self.normalizer = self.weight = self.bias = None
        self.closed = True


@lru_cache(maxsize=1)
def _oracle_function() -> Any:
    import torch

    class _OriginalSigmoidJacobian(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, hidden: Any, state: _Replay) -> tuple[Any, Any]:
            owner = state.owner()
            ids = weights = normalized = None
            try:
                _require(owner is not None, "probe owner is unavailable")
                ctx.set_materialize_grads(False)
                ctx.state = state
                # The returned IDs must not alias clearable graph-owned copies.
                ids = owner._keep(state.ids.clone())
                if state.settings[5]:
                    normalized = owner._keep(state.scores / state.normalizer)
                else:
                    normalized = state.scores
                weights = owner._keep(normalized * state.settings[6])
                ctx.mark_non_differentiable(ids)
                return weights, ids
            finally:
                del hidden
                ids = weights = normalized = None

        @staticmethod
        def backward(ctx: Any, incoming: Any, _ids_gradient: Any) -> tuple[Any, ...]:
            state = ctx.state
            owner = state.owner()
            grad = scaled = normalized = denominator_term = selected = None
            sigmoid_gradient = full = weight32 = gradient32 = hidden_gradient = None
            try:
                _require(not state.closed and owner is not None
                         and not owner._closed and not owner._stopping, "replay is closed")
                _require(not torch.is_grad_enabled(), "only first-order VJPs are supported")
                _require(incoming is not None, "routing weights received no gradient")
                owner._touch_device(state.weight.device)
                owner._check_replay(state)
                _require(incoming.dtype == torch.float32 and incoming.shape == state.scores.shape
                         and incoming.device == state.scores.device,
                         "incoming routing-weight gradient has changed shape/dtype/device")
                grad = (owner._keep(incoming.contiguous())
                        if not incoming.is_contiguous() else incoming)
                owner._finite(grad, "incoming routing-weight gradient")
                scaled = owner._keep(grad * state.settings[6])
                if state.settings[5]:
                    # Native div backward includes the denominator's dependence on every
                    # selected sigmoid. This order also stays finite at s=0, D=1e-20.
                    normalized = owner._keep(state.scores / state.normalizer)
                    denominator_term = owner._keep(-scaled)
                    denominator_term = owner._keep(denominator_term * normalized)
                    denominator_term = owner._keep(denominator_term / state.normalizer)
                    denominator_term = owner._keep(denominator_term.sum(dim=-1, keepdim=True))
                    selected = owner._keep(scaled / state.normalizer)
                    selected = owner._keep(selected + denominator_term)
                else:
                    selected = scaled
                sigmoid_gradient = owner._keep(torch.ops.aten.sigmoid_backward.default(
                    selected, state.scores,
                ))
                full = owner._keep(torch.zeros(
                    (state.scores.shape[0], state.weight.shape[0]),
                    dtype=torch.float32, device=state.scores.device,
                ))
                full.scatter_(1, state.ids, sigmoid_gradient)
                # gate.weight is meta again after functional_call. Use the real restored
                # recompute tensor, checked against the pool owner and original content.
                weight32 = (state.weight if state.weight.dtype == torch.float32
                            else owner._keep(state.weight.float()))
                gradient32 = owner._keep(full.mm(weight32))
                hidden_gradient = gradient32.to(dtype=state.dtype).view(state.shape)
                if hidden_gradient.dtype != gradient32.dtype:
                    owner._keep(hidden_gradient)
                owner._finite(hidden_gradient, "returned hidden gradient")
                owner._drain()
                owner._scratch.clear()
                owner._complete_replay(state)
                return hidden_gradient, None
            except BaseException as error:
                # Owned temporaries stay in owner._scratch until close drains their stream.
                incoming = _ids_gradient = grad = scaled = normalized = denominator_term = None
                selected = sigmoid_gradient = full = weight32 = gradient32 = hidden_gradient = None
                if owner is not None:
                    owner._abort(error)
                raise
            finally:
                incoming = _ids_gradient = grad = scaled = normalized = denominator_term = None
                selected = sigmoid_gradient = full = weight32 = gradient32 = hidden_gradient = None

    return _OriginalSigmoidJacobian


class DeepseekOracleProbe:
    """Native-gate interception and bounded, first-order original-forward input VJPs."""

    def __init__(
        self, model: Any, mb_shapes: Sequence[Sequence[int]], host_budget: Any, *,
        device_scratch_limit: int = 128 * 1024**2,
    ) -> None:
        self._closed = self._stopping = False
        self._lease: Any = None
        self._bindings: dict[int, _Binding] = {}
        self._originals: list[tuple[Any, bool, Any]] = []
        self._handles: list[Any] = []
        self._configs: list[Any] = []
        self._records: dict[tuple[int, int], _ForwardRecord] = {}
        self._record_digests: dict[tuple[int, int], str] = {}
        self._pending: dict[tuple[int, int], _Replay] = {}
        self._seen_forward: set[tuple[int, int]] = set()
        self._seen_recompute: set[tuple[int, int]] = set()
        self._seen_vjp: set[tuple[int, int]] = set()
        self._visit: tuple[int, int, str] | None = None
        self._visit_calls = 0
        self._ids = self._scores = self._normalizers = None
        self._scratch: list[Any] = []
        self._borrowed: list[Any] = []
        self._streams: dict[int, Any] = {}
        self._touched_streams: dict[int, Any] = {}
        self._syncs = 0
        self._counts = {"native_forward": 0, "oracle_recompute": 0, "oracle_vjp": 0}
        self._failure: dict[str, str] | None = None
        self._cleanup_errors: list[dict[str, str]] = []
        self._charged = self._expected_pairs = self._n_layers = 0
        self._allocation: dict[str, Any] = {}
        self._source_hashes: dict[str, Any] = {}
        self._runtime: dict[str, str] = {}
        try:
            self._initialize(model, mb_shapes, host_budget, device_scratch_limit)
        except BaseException as error:
            model = host_budget = mb_shapes = None
            self._abort(error)
            raise

    def _initialize(
        self, model: Any, mb_shapes: Sequence[Sequence[int]], host_budget: Any,
        device_scratch_limit: int,
    ) -> None:
        import torch
        import transformers
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
            DeepseekV3DecoderLayer,
            DeepseekV3ForCausalLM,
            DeepseekV3MLP,
            DeepseekV3MoE,
            DeepseekV3TopkRouter,
        )

        from soup_cli.utils.l2l_step import model_parts
        from soup_cli.utils.layer_stream_runtime import StreamedDecoderLayer

        _require(type(device_scratch_limit) is int and device_scratch_limit >= 0,
                 "device_scratch_limit must be nonnegative integer bytes")
        causal, owner = model_parts(model)
        _require(type(causal) is DeepseekV3ForCausalLM, "expected native DeepSeek causal LM")
        _require(type(causal).__module__ == _NATIVE_MODULE,
                 "expected native DeepSeek implementation")
        config = causal.config
        self._n_layers = len(owner.layers)
        _require(self._n_layers == config.num_hidden_layers and self._n_layers > 0,
                 "decoder-layer count differs from config")
        for candidate in (getattr(model, "config", None), config, getattr(owner, "config", None)):
            if candidate is not None and all(candidate is not item for item in self._configs):
                self._configs.append(candidate)
        self._check_config()
        self._shapes = tuple(tuple(shape) for shape in mb_shapes)
        _require(bool(self._shapes) and all(
            len(shape) == 3 and all(type(value) is int and value > 0 for value in shape)
            for shape in self._shapes
        ), "mb_shapes must contain positive integer [B,L,H] shapes")
        self._width = config.hidden_size
        _require(all(shape[2] == self._width for shape in self._shapes),
                 "microbatch hidden width differs from native config")
        self._tokens = tuple(shape[0] * shape[1] for shape in self._shapes)
        self._total_tokens = sum(self._tokens)
        self._offsets = []
        offset = 0
        for tokens in self._tokens:
            self._offsets.append(offset)
            offset += tokens
        _require(type(config.first_k_dense_replace) is int
                 and 0 <= config.first_k_dense_replace < self._n_layers,
                 "at least one native MoE layer is required")
        for layer, module in enumerate(owner.layers):
            wrapper = module if isinstance(module, StreamedDecoderLayer) else None
            inner = module.inner if wrapper is not None else module
            _require(type(inner) is DeepseekV3DecoderLayer, "non-native decoder layer")
            if layer < config.first_k_dense_replace:
                _require(type(inner.mlp) is DeepseekV3MLP, "dense layer is not native DeepSeek")
                continue
            _require(type(inner.mlp) is DeepseekV3MoE,
                     "non-native MoE/logit consumers are unsupported")
            _require(getattr(inner.mlp.forward, "__func__", None) is DeepseekV3MoE.forward,
                     "non-native MoE forward may consume unsupported router logits")
            gate = inner.mlp.gate
            _require(type(gate) is DeepseekV3TopkRouter, "expected native DeepseekV3TopkRouter")
            _require(getattr(gate.forward, "__func__", None) is DeepseekV3TopkRouter.forward,
                     "native gate forward is already intercepted or replaced")
            settings = _settings(gate)
            self._validate_settings(settings)
            _require(settings[:3] == (self._width, config.num_local_experts,
                                      config.num_experts_per_tok),
                     "gate dimensions differ from config")
            _require(all(existing.settings == settings for existing in self._bindings.values()),
                     "nonuniform router settings are unsupported")
            self._check_hooks(gate)
            self._validate_tensor(gate.weight, (gate.num_experts, self._width), "router weight",
                                  allow_meta=wrapper is not None)
            self._validate_tensor(gate.e_score_correction_bias, (gate.num_experts,), "router bias",
                                  allow_meta=wrapper is not None)
            if wrapper is not None:
                _require(wrapper.idx == layer, "streamed wrapper layer index differs")
                device = torch.device(wrapper.pool.device)
            else:
                device = gate.weight.device
            if device.type == "cuda" and device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            _require(device.type in {"cpu", "cuda"}, "only CPU/CUDA gates are supported")
            _require(all(existing.device == device for existing in self._bindings.values()),
                     "all gates must execute on one device")
            self._bindings[layer] = _Binding(
                gate, wrapper, inner.mlp, DeepseekV3MoE.forward, settings, device,
                len(self._bindings) * self._total_tokens,
            )
        first = next(iter(self._bindings.values()))
        self._device = first.device
        self._experts, self._topk = first.settings[1:3]
        self._expected_pairs = len(self._bindings) * len(self._shapes)
        self._allocation = _allocation_bound(
            self._shapes, len(self._bindings), self._experts, self._topk, self._device.type,
        )
        scratch = self._allocation["device_scratch_bound_bytes"]
        if scratch > device_scratch_limit:
            raise MemoryError(f"DeepSeek Oracle scratch needs {scratch} bytes; "
                              f"limit is {device_scratch_limit}")
        self._charged = sum(self._allocation["host"].values())
        self._lease = host_budget.reserve(f"deepseek-oracle-{id(self)}", self._charged)
        rows = len(self._bindings) * self._total_tokens
        self._ids = torch.empty((rows, self._topk), dtype=torch.int64, device="cpu")
        self._scores = torch.empty((rows, self._topk), dtype=torch.float32, device="cpu")
        self._normalizers = torch.empty((rows, 1), dtype=torch.float32, device="cpu")
        self._runtime = {"torch_version": str(torch.__version__),
                         "transformers_version": str(transformers.__version__)}
        self._source_hashes = {"native_router": _source_identity(DeepseekV3TopkRouter),
                               "native_moe": _source_identity(DeepseekV3MoE),
                               "benchmark": _source_identity(DeepseekOracleProbe)}
        for layer, binding in self._bindings.items():
            gate = binding.gate
            self._originals.append((gate, "forward" in gate.__dict__, gate.__dict__.get("forward")))
            gate.forward = self._wrapper(layer, gate, gate.forward)
        hooked: set[int] = set()
        for target in (model, causal, owner):
            if id(target) not in hooked:
                self._handles.append(target.register_forward_pre_hook(
                    self._model_pre_hook, with_kwargs=True,
                ))
                hooked.add(id(target))

    @staticmethod
    def _validate_settings(settings: tuple[Any, ...]) -> None:
        width, experts, topk, groups, chosen_groups, normalize, scaling = settings
        _require(all(type(value) is int and value > 0
                     for value in (width, experts, topk, groups, chosen_groups)),
                 "native grouped top-k dimensions must be positive integers")
        _require(experts % groups == 0 and experts // groups >= 2
                 and chosen_groups <= groups and topk <= chosen_groups * (experts // groups),
                 "unsupported native grouped top-k dimensions")
        _require(type(normalize) is bool, "norm_topk_prob must be bool")
        _require(type(scaling) in {int, float} and math.isfinite(scaling) and scaling > 0,
                 "routed scaling factor must be finite and positive")

    @staticmethod
    def _validate_tensor(tensor: Any, shape: tuple[int, ...], name: str, *,
                         allow_meta: bool = False) -> None:
        import torch

        _require(isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == shape,
                 f"{name} shape changed")
        _require(not tensor.requires_grad and tensor.dtype in {torch.bfloat16, torch.float32},
                 f"{name} must be frozen BF16 or FP32")
        _require(tensor.is_contiguous() and (allow_meta or not tensor.is_meta),
                 f"{name} must be contiguous restored storage")

    @staticmethod
    def _check_hooks(gate: Any) -> None:
        _require(not any(getattr(gate, name, {}) for name in (
            "_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks",
        )), "gate hooks may consume unsupported full router logits or alter native routing")

    def _check_config(self) -> None:
        for config in self._configs:
            _require(getattr(config, "model_type", None) == "deepseek_v3", "model_type changed")
            _require(getattr(config, "output_router_logits", False) in (False, None),
                     "full router-logit consumers are unsupported")
            for field in ("router_aux_loss_coef", "aux_loss_alpha"):
                _require(getattr(config, field, 0) in (0, None),
                         "auxiliary router-logit loss is unsupported")

    def _model_pre_hook(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        signature = flags = source = None
        try:
            self._check_config()
            signature = inspect.signature(type(module).forward)
            flags = signature.bind_partial(module, *args, **kwargs).arguments
            for source in (flags, flags.get("kwargs", {}), kwargs):
                _require(source.get("output_router_logits") in (False, None),
                         "full router-logit output override is unsupported")
                for field in ("router_aux_loss_coef", "aux_loss_alpha"):
                    _require(source.get(field) in (0, None), "auxiliary router loss is unsupported")
        except BaseException as error:
            module = args = kwargs = signature = flags = source = None
            self._abort(error)
            raise
        finally:
            module = args = kwargs = signature = flags = source = None

    def _wrapper(self, layer: int, gate: Any, native: Any) -> Any:
        reference = weakref.ref(self)

        def wrapped(_gate: Any, *args: Any, **kwargs: Any) -> Any:
            owner = reference()
            try:
                _require(owner is not None, "intercepted gate has no live probe")
                return owner._dispatch(layer, native, args, kwargs)
            finally:
                args = kwargs = None

        return MethodType(wrapped, gate)

    def _touch_device(self, device: Any) -> None:
        if device.type == "cuda":
            import torch

            stream = torch.cuda.current_stream(device)
            self._streams[stream.cuda_stream] = stream
            self._touched_streams[stream.cuda_stream] = stream

    def _keep(self, tensor: Any) -> Any:
        self._scratch.append(tensor)
        return tensor

    def _drain(self) -> None:
        # A failed drain deliberately keeps storage and its lease for an explicit retry.
        for key, stream in tuple(self._streams.items()):
            stream.synchronize()
            self._syncs += 1
            del self._streams[key]

    def _finite(self, tensor: Any, name: str) -> None:
        import torch

        flat = mask = reduction = None
        try:
            flat = tensor.detach().view(-1)
            for offset in range(0, flat.numel(), _HASH_ELEMENTS):
                start = len(self._scratch)
                mask = self._keep(torch.isfinite(flat[offset:offset + _HASH_ELEMENTS]))
                reduction = self._keep(mask.all())
                finite = bool(reduction.item())
                _require(finite, f"nonfinite {name}")
                # .item() completed these validation operations on CPU or CUDA.
                del self._scratch[start:]
                mask = reduction = None
        finally:
            tensor = flat = mask = reduction = None

    def _digest(self, values: Sequence[Any], metadata: Any = ()) -> str:
        import torch

        digest = hashlib.sha256(repr(metadata).encode("utf-8"))
        tensor = flat = host = array = view = None
        try:
            for tensor in values:
                _require(tensor.is_contiguous(),
                         "integrity hashing needs contiguous native tensors")
                digest.update(repr((str(tensor.dtype), tuple(tensor.shape))).encode("utf-8"))
                flat = tensor.detach().view(-1)
                for offset in range(0, flat.numel(), _HASH_ELEMENTS):
                    host = flat[offset:offset + _HASH_ELEMENTS]
                    transferred = host.device.type != "cpu"
                    if transferred:
                        host = self._keep(host.to(device="cpu", copy=True, non_blocking=False))
                    # uint8 supports BF16 bytes without a NumPy BF16 dtype or an extra copy.
                    array = host.view(dtype=torch.uint8).numpy()
                    view = memoryview(array).cast("B")
                    digest.update(view)
                    if transferred:
                        self._scratch.pop()
                    if view is not None:
                        view.release()
                    host = array = view = None
            return digest.hexdigest()
        finally:
            if view is not None:
                view.release()
            values = tensor = flat = host = array = view = None

    def _check_slot(self, layer: int, weight: Any, bias: Any) -> None:
        binding = self._bindings[layer]
        wrapper = binding.wrapper
        if wrapper is None:
            _require(binding.gate.weight is weight and binding.gate.e_score_correction_bias is bias,
                     "resident router tensor identity changed")
            return
        slot = wrapper.pool.slot_for(layer)
        _require(wrapper.pool.owner[slot] == layer, "streamed router slot has another layer owner")
        for local, tensor in (("mlp.gate.weight", weight),
                              ("mlp.gate.e_score_correction_bias", bias)):
            source = wrapper.name_map.get(local)
            if source is not None:
                _require(wrapper.pool.buffers[slot][source] is tensor,
                         "restored router tensor differs from its streamed slot")
            elif local.endswith("correction_bias"):
                _require(binding.gate.e_score_correction_bias is tensor,
                         "resident correction bias identity changed")
            else:
                _require(binding.gate.weight is tensor,
                         "unmapped resident router weight identity changed")

    def _check_gate(self, layer: int, weight: Any, bias: Any) -> None:
        self._check_config()
        binding = self._bindings[layer]
        self._check_hooks(binding.gate)
        _require(getattr(binding.moe.forward, "__func__", None) is binding.native_moe_forward,
                 "MoE forward changed to an unsupported router-logit consumer")
        _require(_settings(binding.gate) == binding.settings, "router scalar settings changed")
        self._validate_tensor(weight, (self._experts, self._width), "router weight")
        self._validate_tensor(bias, (self._experts,), "router bias")
        _require(weight.device == bias.device == binding.device, "restored router device changed")
        self._check_slot(layer, weight, bias)
        self._finite(weight, "router weight")
        self._finite(bias, "router bias")

    def _record_digest(self, record: _ForwardRecord) -> str:
        return self._digest((record.ids, record.scores, record.normalizer), record.metadata())

    def _check_record(self, pair: tuple[int, int], weight: Any, bias: Any) -> _ForwardRecord:
        record = self._records[pair]
        _require(self._record_digest(record) == self._record_digests[pair],
                 "saved original-forward record was mutated")
        _require(self._digest((weight,)) == record.weight_digest,
                 "restored router weight changed from original forward")
        _require(self._digest((bias,)) == record.bias_digest,
                 "restored router bias changed from original forward")
        if record.weight_stamp is not None:
            _require(_stamp(weight) == record.weight_stamp, "resident router weight changed")
        if record.bias_stamp is not None:
            _require(_stamp(bias) == record.bias_stamp, "resident router bias changed")
        return record

    def _capture(self, pair: tuple[int, int], hidden: Any, output: Any) -> None:
        import torch

        logits = weights = ids = scores = selected = normalizer = expected = record = None
        weight = bias = None
        try:
            _require(isinstance(output, tuple) and len(output) == 3, "native gate return changed")
            logits, weights, ids = output
            self._borrowed.extend((logits, weights, ids))
            rows = self._tokens[pair[1]]
            _require(logits.dtype == weights.dtype == torch.float32
                     and tuple(logits.shape) == (rows, self._experts)
                     and tuple(weights.shape) == (rows, self._topk),
                     "native gate shape/dtype changed")
            _require(ids.dtype == torch.int64 and ids.shape == weights.shape,
                     "native ordered IDs shape/dtype changed")
            _require(logits.device == weights.device == ids.device == hidden.device,
                     "native gate output device changed")
            self._finite(logits, "native logits")
            self._finite(weights, "native routing weights")
            # Repeat the native full sigmoid kernel before gathering. No inversion of
            # rounded/scaled weights, no correction-bias score, and no softmax.
            scores = self._keep(logits.detach().sigmoid())
            selected = self._keep(scores.gather(1, ids))
            binding = self._bindings[pair[0]]
            normalizer = self._keep(selected.sum(dim=-1, keepdim=True))
            normalizer = self._keep(normalizer + _EPSILON)
            expected = (self._keep(selected / normalizer) if binding.settings[5] else selected)
            expected = self._keep(expected * binding.settings[6])
            _require(torch.equal(expected, weights),
                     "native sigmoid/normalization arithmetic changed")
            offset = binding.offset + self._offsets[pair[1]]
            record = _ForwardRecord(
                self._ids[offset:offset + rows], self._scores[offset:offset + rows],
                self._normalizers[offset:offset + rows], tuple(hidden.shape), hidden.dtype,
                binding.settings, "", "", None, None,
            )
            record.ids.copy_(ids.detach(), non_blocking=False)
            record.scores.copy_(selected, non_blocking=False)
            record.normalizer.copy_(normalizer, non_blocking=False)
            for row_index in range(rows):
                row = record.ids[row_index].tolist()
                _require(len(set(row)) == self._topk
                         and all(0 <= expert < self._experts for expert in row),
                         "native route IDs must be unique in-range experts")
            weight, bias = binding.gate.weight, binding.gate.e_score_correction_bias
            record.weight_digest = self._digest((weight,))
            record.bias_digest = self._digest((bias,))
            if binding.wrapper is None:
                record.weight_stamp = _stamp(weight)
            if (binding.wrapper is None
                    or "mlp.gate.e_score_correction_bias" not in binding.wrapper.name_map):
                record.bias_stamp = _stamp(bias)
            self._records[pair] = record
            self._record_digests[pair] = self._record_digest(record)
            self._seen_forward.add(pair)
            self._counts["native_forward"] += 1
        finally:
            hidden = output = logits = weights = ids = scores = selected = normalizer = None
            expected = record = weight = bias = None

    def _replay(self, pair: tuple[int, int], hidden: Any, weight: Any,
                bias: Any) -> tuple[Any, ...]:
        record = state = weights = ids = None
        try:
            record = self._check_record(pair, weight, bias)
            _require(
                tuple(hidden.shape) == record.hidden_shape
                and hidden.dtype == record.hidden_dtype, "recompute hidden shape/dtype changed",
            )
            state = _Replay(self, pair)
            self._pending[pair] = state
            state.weight, state.bias = weight, bias
            state.weight_stamp, state.bias_stamp = _stamp(weight), _stamp(bias)
            state.shape, state.dtype = record.hidden_shape, record.hidden_dtype
            state.settings = record.settings
            state.ids = record.ids.detach().to(
                device=hidden.device, copy=True, non_blocking=False,
            )
            state.scores = record.scores.detach().to(
                device=hidden.device, copy=True, non_blocking=False,
            )
            state.normalizer = record.normalizer.detach().to(
                device=hidden.device, copy=True, non_blocking=False,
            )
            state.digest = self._digest((state.ids, state.scores, state.normalizer), state.settings)
            weights, ids = _oracle_function().apply(hidden, state)
            self._seen_recompute.add(pair)
            self._counts["oracle_recompute"] += 1
            return None, weights, ids
        finally:
            hidden = weight = bias = record = state = weights = ids = None

    def _check_replay(self, state: _Replay) -> None:
        _require(self._pending.get(state.pair) is state, "missing or duplicate pending replay")
        self._check_gate(state.pair[0], state.weight, state.bias)
        _require(_stamp(state.weight) == state.weight_stamp, "router weight changed before VJP")
        _require(_stamp(state.bias) == state.bias_stamp, "router bias changed before VJP")
        record = self._check_record(state.pair, state.weight, state.bias)
        _require(state.settings == record.settings and state.shape == record.hidden_shape
                 and state.dtype == record.hidden_dtype, "replay scalar state changed")
        replay_digest = self._digest((state.ids, state.scores, state.normalizer), state.settings)
        _require(replay_digest == state.digest, "pending replay tensor state was mutated")

    def _complete_replay(self, state: _Replay) -> None:
        pair = state.pair
        _require(pair not in self._seen_vjp, "duplicate Oracle VJP")
        self._seen_vjp.add(pair)
        self._counts["oracle_vjp"] += 1
        self._pending.pop(pair)
        self._records.pop(pair).clear()
        self._record_digests.pop(pair)
        state.clear()

    def _dispatch(self, layer: int, native: Any, args: tuple[Any, ...],
                  kwargs: dict[str, Any]) -> Any:
        import torch

        hidden = output = weight = bias = None
        try:
            _require(not self._closed and not self._stopping, "probe is closed")
            _require(self._visit is not None, "native gate call requires an explicit layer visit")
            expected_layer, mb, phase = self._visit
            _require(layer == expected_layer, "visit layer differs from native gate")
            _require(self._visit_calls == 0, "duplicate native gate call inside visit")
            _require((len(args) == 1 and not kwargs)
                     or (not args and set(kwargs) == {"hidden_states"}),
                     "native gate arguments changed")
            hidden = args[0] if args else kwargs["hidden_states"]
            _require(isinstance(hidden, torch.Tensor) and tuple(hidden.shape) in (
                self._shapes[mb], (self._tokens[mb], self._width),
            ), "hidden shape differs from declared microbatch")
            _require(hidden.dtype in {torch.bfloat16, torch.float32}
                     and hidden.device == self._device and hidden.is_contiguous(),
                     "hidden must be contiguous BF16/FP32 on the declared device")
            _require(not torch.is_autocast_enabled(hidden.device.type),
                     "autocast is unsupported for the native FP32 router linear")
            binding = self._bindings[layer]
            pair = (layer, mb)
            if phase == "forward":
                _require(not torch.is_grad_enabled(), "original L2L forward must disable grad")
                _require(pair not in self._seen_forward, "duplicate original-forward pair")
                _require(not self._pending, "cannot forward with an unfinished pending replay")
            else:
                _require(pair in self._seen_forward, "recompute is missing original forward")
                _require(pair not in self._seen_recompute, "duplicate Oracle recompute pair")
                _require(not self._pending, "only one unfinished pending replay fits scratch bound")
                _require(torch.is_grad_enabled() and hidden.requires_grad,
                         "Oracle recompute requires grad and a differentiable hidden input")
            self._touch_device(hidden.device)
            weight, bias = binding.gate.weight, binding.gate.e_score_correction_bias
            self._borrowed.extend((weight, bias))
            self._check_gate(layer, weight, bias)
            self._finite(hidden, "router hidden input")
            if phase == "forward":
                output = native(*args, **kwargs)
                self._capture(pair, hidden, output)
            else:
                output = self._replay(pair, hidden, weight, bias)
            self._visit_calls += 1
            self._drain()
            self._scratch.clear()
            self._borrowed.clear()
            return output
        except BaseException as error:
            hidden = output = weight = bias = args = kwargs = None
            self._abort(error)
            raise
        finally:
            hidden = output = weight = bias = args = kwargs = None

    @contextmanager
    def visit(self, layer: int, microbatch: int, phase: str) -> Iterator[None]:
        try:
            _require(not self._closed and not self._stopping, "probe is closed")
            _require(self._visit is None, "nested visits are unsupported")
            _require(type(layer) is int and 0 <= layer < self._n_layers, "unknown layer index")
            _require(type(microbatch) is int and 0 <= microbatch < len(self._shapes),
                     "unknown microbatch index")
            _require(phase in {"forward", "recompute"}, "phase must be forward or recompute")
            self._visit, self._visit_calls = (layer, microbatch, phase), 0
            yield
            _require(self._visit_calls == int(layer in self._bindings),
                     "incomplete layer visit: expected native gate call was missing")
        except BaseException as error:
            self._abort(error)
            raise
        finally:
            self._visit, self._visit_calls = None, 0

    def _complete(self) -> bool:
        return (self._expected_pairs > 0 and self._failure is None and not self._cleanup_errors
                and len(self._seen_forward) == len(self._seen_recompute) == len(self._seen_vjp)
                == self._expected_pairs and not self._records and not self._pending)

    def require_complete(self) -> None:
        """Refuse an optimizer update unless every declared MoE pair completed its VJP."""
        try:
            _require(self._complete(), "incomplete original-forward/recompute/VJP window")
        except BaseException as error:
            self._abort(error)
            raise

    def _owned_storage(self) -> tuple[int, int]:
        tensors = [self._ids, self._scores, self._normalizers, *self._scratch]
        for state in self._pending.values():
            tensors.extend((state.ids, state.scores, state.normalizer))
        tensor = raw = None
        try:
            seen = set()
            host = device = 0
            for tensor in tensors:
                if tensor is None:
                    continue
                raw = tensor.untyped_storage()
                key = (str(tensor.device), raw.data_ptr())
                if key in seen:
                    continue
                seen.add(key)
                if tensor.device.type == "cpu":
                    host += raw.nbytes()
                else:
                    device += raw.nbytes()
            return host, device
        finally:
            tensors = tensor = raw = None

    def report(self) -> dict[str, Any]:
        """Return detached scalar evidence, including after a failed or successful close."""
        host, device = self._owned_storage()
        return {
            "complete": self._complete(), "expected_pairs": self._expected_pairs,
            "counts": dict(self._counts), "host_reserved_bytes": (
                self._lease.nbytes if self._lease is not None else 0
            ),
            "host_reservation_bound_bytes": self._charged,
            "device_scratch_bound_bytes": self._allocation.get("device_scratch_bound_bytes", 0),
            "records_remaining": len(self._records), "pending_replays": len(self._pending),
            "closed": self._closed, "forced_routing": True, "natural_routing_measured": False,
            "oracle_semantics": "selected-sigmoid-Jacobian-at-original-forward",
            "full_router_logits_replayed": False, "first_order_only": True,
            "failure": dict(self._failure) if self._failure else None,
            "cleanup_errors": [dict(error) for error in self._cleanup_errors],
            "owned_host_storage_bytes": host, "owned_device_storage_bytes": device,
            "runtime": dict(self._runtime), "source_hashes": {
                key: dict(value) for key, value in self._source_hashes.items()
            },
            "allocation": {**self._allocation,
                           "host": dict(self._allocation.get("host", {})),
                           "operation_buffer_bound": dict(
                               self._allocation.get("operation_buffer_bound", {}),
                           )},
            "cleanup": {"original_forwards_restored": not self._originals,
                        "hooks_removed": not self._handles,
                        "owned_buffers_dropped": host == device == 0,
                        "synchronous_device_drains": self._syncs},
        }

    def _abort(self, error: BaseException) -> None:
        if self._failure is None:
            self._failure = {"kind": type(error).__name__, "message": str(error)}
        traceback.clear_frames(error.__traceback__)
        self.close()

    def close(self) -> None:
        """Restore methods, drain outstanding work, drop owned state, then release the lease.

        A drain failure leaves the probe stopping, its storage and lease intact.
        The caller can retry close; the cleanup error stays in its report.
        """
        if self._closed:
            return
        self._stopping = True
        self._streams.update(self._touched_streams)
        try:
            while self._originals:
                gate, had_forward, previous = self._originals[-1]
                if had_forward:
                    gate.forward = previous
                elif "forward" in gate.__dict__:
                    del gate.forward
                self._originals.pop()
            while self._handles:
                self._handles[-1].remove()
                self._handles.pop()
            self._drain()
            self._scratch.clear()
            self._borrowed.clear()
            for state in self._pending.values():
                state.clear()
            self._pending.clear()
            for record in self._records.values():
                record.clear()
            self._records.clear()
            self._record_digests.clear()
            self._ids = self._scores = self._normalizers = None
            self._bindings.clear()
            self._configs.clear()
            self._touched_streams.clear()
            self._visit = None
            if self._lease is not None:
                self._lease.close()
                self._lease = None
            self._closed = True
        except BaseException as error:
            self._cleanup_errors.append({"kind": type(error).__name__, "message": str(error)})
            raise

    def __enter__(self) -> DeepseekOracleProbe:
        _require(not self._closed and not self._stopping, "probe is closed")
        return self

    def __exit__(self, _kind: Any, error: BaseException | None, tb: Any) -> None:
        if error is not None:
            traceback.clear_frames(tb)
            self._abort(error)
        else:
            self.close()
