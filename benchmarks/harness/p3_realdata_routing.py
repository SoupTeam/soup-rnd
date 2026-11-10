"""Bounded native Granite route observations and a saved-Jacobian Oracle.

This benchmark does not change production model admission or the routing guard.
Natural diagnostics never replace native decisions. Oracle replay substitutes the
original ordered IDs and FP32 pre-cast softmax Jacobian only during layer-major
recompute. Its forced equality is not a measurement of natural route stability.
Oracle recompute under no_grad is output-only diagnostic replay. It creates no
autograd state, does not claim a VJP, and cannot receive training admission.

Host reservations cover tensor storage, including CPU operation scratch. They are
not process-RSS limits. Device bounds exclude the caller's native model/experts,
framework allocator caches and kernel workspaces. No hidden-state primal is saved.
Dependencies used for tensor work are imported only when the probe is constructed.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import sys
import traceback
import weakref
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from types import MethodType
from typing import Any, Iterator, Sequence

__all__ = ["RouteProbe", "RoutingMismatchError", "routing_allocation_bound"]

_NATIVE_MODULE = "transformers.models.granitemoe.modeling_granitemoe"
_TILE_ROWS = 256


class RoutingMismatchError(RuntimeError):
    """A natural recompute changed at least one native ordered expert-ID row."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Granite route probe: {message}")


def routing_allocation_bound(
    *, n_layers: int, mb_shapes: Sequence[Sequence[int]], experts: int, topk: int,
    oracle: bool, device_type: str,
) -> dict[str, Any]:
    """Plan the probe's existing tensor bound without importing or allocating torch."""
    if type(n_layers) is not int or n_layers < 1:
        raise ValueError("n_layers must be a positive integer")
    if type(experts) is not int or experts < 1:
        raise ValueError("experts must be a positive integer")
    if type(topk) is not int or not 0 < topk <= experts:
        raise ValueError("topk must be a positive integer no greater than experts")
    if type(oracle) is not bool or device_type not in ("cpu", "cuda"):
        raise ValueError("oracle must be bool and device_type must be cpu or cuda")
    shapes = tuple(tuple(shape) for shape in mb_shapes)
    if not shapes or any(
        len(shape) != 3 or any(type(value) is not int or value < 1 for value in shape)
        for shape in shapes
    ):
        raise ValueError("mb_shapes must contain positive integer [B,L,H] shapes")
    width = shapes[0][-1]
    if any(shape[-1] != width for shape in shapes):
        raise ValueError("all microbatches must have the same hidden width")
    rows = [shape[0] * shape[1] for shape in shapes]
    tokens = max(rows)
    tile = min(_TILE_ROWS, tokens)
    # isfinite may use a float temporary, several boolean tiles, and scalar reductions.
    finite = 8 * min(_TILE_ROWS, max(tokens, experts)) * max(width, experts, topk) + 16
    operation = {"finite_validation": finite}
    if oracle:
        operation.update({
            "transferred_ids": 8 * tokens * topk,
            "transferred_probabilities": 4 * tokens * topk,
            "returned_weights_bf16": 2 * tokens * topk,
            "contiguous_incoming_bf16": 2 * tokens * topk,
            "incoming_gradient_fp32": 4 * tokens * topk,
            "native_softmax_product_fp32": 4 * tokens * topk,
            "selected_gradient_fp32": 4 * tokens * topk,
            "scatter_fp32": 4 * tokens * experts,
            "logit_gradient_bf16": 2 * tokens * experts,
            "returned_hidden_gradient_bf16": 2 * tokens * width,
        })
    bound = sum(operation.values())
    records = n_layers * sum(rows) * topk
    host = {
        "ids": 8 * records,
        "probabilities": 4 * records if oracle else 0,
        "comparison": 8 * tokens * topk,
        "expert_counts": 8 * n_layers * experts,
        "top1_counts": 8 * n_layers * experts,
        "comparison_bool": tile * (topk + 1),
        "comparison_reduction": 8,
        "cpu_operation_scratch": bound if device_type == "cpu" else 0,
    }
    return {
        "host": host, "host_pinned": False, "pinned_rounding_bytes": 0,
        "device_buffer_bound_bytes": bound if device_type == "cuda" else 0,
        "operation_buffer_bound": operation,
        "original_probability_recovery_bound_bytes": (
            10 * tokens * topk + finite if oracle else finite
        ),
        "max_tokens": tokens, "max_hidden": width, "experts": experts, "topk": topk,
        "maximum_simultaneous_oracle_replays": 1,
        "tensor_storage_only": True,
        "exclusions": ["Python scalar bookkeeping", "caller native model/experts",
                       "pre-existing caller tensors", "framework allocator caches",
                       "kernel workspaces"],
    }


def _source_hash(target: Any) -> dict[str, Any]:
    filename = inspect.getsourcefile(target)
    if filename is None:
        return {"path": None, "sha256": None, "available": False}
    digest = hashlib.sha256()
    with Path(filename).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": filename, "sha256": digest.hexdigest(), "available": True}


def _finite(values: Any, name: str) -> None:
    """Keep validation temporaries within one tile, never a full hidden matrix."""
    import torch

    tile = None
    try:
        with torch.no_grad():
            for start in range(0, values.shape[0], _TILE_ROWS):
                tile = values[start:start + _TILE_ROWS]
                _require(bool(torch.isfinite(tile).all().item()), f"nonfinite {name}")
    finally:
        tile = values = None


def _recover_probabilities(hidden: Any, output: Sequence[Any]) -> Any:
    """Recover pre-cast p from native FP32 logits on the original device/stream."""
    import torch

    ids = weights = logits = selected = probabilities = cast = None
    try:
        ids, weights, logits = output
        _require(logits.dtype == torch.float32, "native router logits must be FP32")
        _require(ids.dtype == torch.int64, "native router IDs must be int64")
        _require(weights.dtype == hidden.dtype == torch.bfloat16,
                 "native router weights/input must be BF16")
        _require(ids.device == weights.device == logits.device == hidden.device,
                 "probability recovery must use the native device")
        with torch.no_grad():
            selected = logits.detach().gather(-1, ids.detach()).contiguous()
            probabilities = torch.softmax(selected, dim=-1)
            cast = probabilities.to(dtype=hidden.dtype)
            _require(torch.equal(cast, weights.detach()),
                     "recovered p cast differs from native weights")
            _finite(probabilities, "recovered probabilities")
        return probabilities
    finally:
        hidden = output = ids = weights = logits = selected = probabilities = cast = None


class _OracleReplay:
    """One replay's saved tensors, clearable even if an exception/graph survives."""

    def __init__(self, owner: Any, pair: tuple[int, int]) -> None:
        self.owner = weakref.ref(owner)
        self.pair = pair
        self.ids: Any = None
        self.probabilities: Any = None
        self.weight: Any = None
        self.weight_version = -1
        self.closed = False

    def clear(self) -> None:
        self.ids = self.probabilities = self.weight = None
        self.closed = True


@lru_cache(maxsize=1)
def _oracle_function() -> Any:
    import torch

    class _SavedRouterJacobian(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, hidden: Any, weight: Any, ids: Any,
                    probabilities: Any, state: _OracleReplay) -> tuple[Any, Any]:
            try:
                ctx.set_materialize_grads(False)
                ctx.state = state
                state.weight = weight
                state.weight_version = weight._version
                # copy=True matters for CPU tests too: the graph must not alias host caches.
                state.ids = ids.to(device=hidden.device, copy=True, non_blocking=False)
                state.probabilities = probabilities.to(
                    device=hidden.device, copy=True, non_blocking=False,
                )
                ctx.mark_non_differentiable(state.ids)
                return state.ids, state.probabilities.to(dtype=torch.bfloat16)
            finally:
                hidden = weight = ids = probabilities = None

        @staticmethod
        def backward(ctx: Any, _ids_gradient: Any, incoming: Any) -> tuple[Any, ...]:
            state = ctx.state
            owner = state.owner()
            failure = None
            contiguous = grad32 = selected = full_logits = None
            linear_gradient = hidden_gradient = None
            try:
                _require(not state.closed, "Oracle replay is closed")
                _require(owner is not None and not owner._closed, "Oracle probe is closed")
                _require(not torch.is_grad_enabled(), "Oracle supports first-order gradients only")
                _require(incoming is not None, "Oracle routing weights received no gradient")
                owner._check_router(state.pair[0])
                _require(state.weight._version == state.weight_version,
                         "router weight changed before Oracle backward")
                _require(incoming.dtype == torch.bfloat16, "Oracle incoming gradient must be BF16")
                _require(incoming.shape == state.probabilities.shape,
                         "Oracle incoming gradient shape changed")
                _finite(incoming, "Oracle incoming gradient")
                contiguous = incoming.contiguous()
                grad32 = contiguous.to(dtype=torch.float32)
                selected = torch.ops.aten._softmax_backward_data.default(
                    grad32, state.probabilities, -1, torch.float32,
                )
                full_logits = torch.zeros(
                    (selected.shape[0], state.weight.shape[0]),
                    dtype=torch.float32, device=selected.device,
                )
                full_logits.scatter_(1, state.ids, selected)
                linear_gradient = full_logits.to(dtype=torch.bfloat16)
                hidden_gradient = linear_gradient.mm(state.weight)
                _finite(hidden_gradient, "Oracle returned input gradient")
                owner._counts["oracle_vjp"] += 1
                return hidden_gradient, None, None, None, None
            except BaseException as error:
                failure = error
                raise
            finally:
                # Do not let a retained autograd exception keep caches or transfer copies.
                incoming = contiguous = grad32 = selected = full_logits = None
                linear_gradient = hidden_gradient = _ids_gradient = None
                if failure is not None:
                    traceback.clear_frames(failure.__traceback__)
                state.clear()
                if owner is not None:
                    owner._pending.pop(state.pair, None)
                    if failure is not None:
                        owner._abort(failure)

    return _SavedRouterJacobian


def _oracle_weights(hidden: Any, weight: Any, ids: Any, probabilities: Any,
                    state: _OracleReplay) -> tuple[Any, Any]:
    """Private first-order helper; the owning probe supplies validated native caches."""
    try:
        return _oracle_function().apply(hidden, weight, ids, probabilities, state)
    finally:
        hidden = weight = ids = probabilities = None


class RouteProbe:
    """Observe one declared window, or replay its original Granite gate Jacobians.

    Install on construction, use ``visit`` for every native forward/recompute,
    then close in a finally block or use this object as a context manager. Only
    explicit layer-indexed recomputes may use Oracle. ``strict=False`` preserves
    all natural routes and gradients, but is diagnostic-only even when routes match.
    Oracle under ``no_grad`` supports output-only diagnostics without a pending
    replay. Its ``routing_forward_complete`` is distinct from ``complete``, which
    continues to require every training VJP. It never receives training admission.
    Reports remain scalar/JSON-safe after close, including failures and charges.
    """

    def __init__(self, model: Any, mb_shapes: Sequence[Sequence[int]], host_budget: Any,
                 *, oracle: bool = False, strict: bool = True) -> None:
        import torch

        self._closed = False
        self._lease: Any = None
        self._model: Any = model
        self._model_handle: Any = None
        self._routers: list[Any] = []
        self._originals: list[tuple[Any, bool, Any]] = []
        self._pending: dict[tuple[int, int], _OracleReplay] = {}
        self._visit: tuple[int | None, int, str] | None = None
        self._ids = self._probabilities = self._comparison = None
        self._slot_diff = self._row_diff = self._reduction = None
        self._expert_counts = self._top1_counts = None
        self._frozen_expert_counts: list[list[int]] = []
        self._frozen_top1_counts: list[list[int]] = []
        self._seen_forward: set[tuple[int, int]] = set()
        self._seen_recompute: set[tuple[int, int]] = set()
        self._digests: dict[tuple[int, int], str] = {}
        self._first_mismatch: dict[str, Any] | None = None
        self._failure: dict[str, str] | None = None
        self._syncs = 0
        self._cuda_used = False
        self._stats: list[dict[str, int]] = []
        self._counts = {
            "native_forward": 0, "native_recompute": 0, "oracle_forward": 0,
            "oracle_vjp": 0, "probability_reconstruction": 0,
            "oracle_diagnostic_forward": 0, "oracle_vjp_required": 0,
            "unobserved_native_calls": 0, "id_d2h_bytes": 0, "probability_d2h_bytes": 0,
            "id_h2d_bytes": 0, "probability_h2d_bytes": 0,
        }
        try:
            _require(type(oracle) is bool and type(strict) is bool, "flags must be bool")
            self.oracle = oracle
            self.strict = strict
            _require(
                type(model).__name__ == "GraniteMoeForCausalLM"
                and type(model).__module__ == _NATIVE_MODULE, "expected native Granite causal LM",
            )
            _require(model.config.model_type == "granitemoe", "unsupported model_type")
            _require(model.config.output_router_logits is False,
                     "output_router_logits must remain False; no auxiliary/logit loss is supported")
            _require(sys.byteorder == "little", "route digests require little-endian int64 storage")
            layers = model.model.layers
            _require(len(layers) > 0 and len(layers) == model.config.num_hidden_layers,
                     "decoder layer count does not match config")
            self._routers = [layer.block_sparse_moe.router for layer in layers]
            first = self._routers[0]
            self._width = model.config.hidden_size
            self._experts = first.num_experts
            self._topk = first.top_k
            self._device = first.weight.device
            self._n_layers = len(layers)
            _require(self._device.type in {"cpu", "cuda"}, "unsupported router device")
            _require(type(self._topk) is int and 0 < self._topk <= self._experts,
                     "unsupported router top-k")
            _require(self._experts == model.config.num_local_experts
                     and self._topk == model.config.num_experts_per_tok, "router config differs")
            self._weight_signatures = []
            for router in self._routers:
                _require(type(router).__name__ == "GraniteMoeTopKRouter"
                         and type(router).__module__ == _NATIVE_MODULE, "non-native Granite router")
                _require(router.num_experts == self._experts and router.top_k == self._topk,
                         "nonuniform router dimensions are unsupported")
                weight = router.weight
                _require(tuple(weight.shape) == (self._experts, self._width),
                         "router weight shape differs")
                _require(weight.dtype == torch.bfloat16 and not weight.requires_grad,
                         "router weight must be frozen BF16")
                _require(weight.device == self._device and weight.is_contiguous(),
                         "router weights must be contiguous on one device")
                self._weight_signatures.append((id(weight), weight._version, weight.data_ptr()))
            self._shapes = tuple(tuple(shape) for shape in mb_shapes)
            _require(bool(self._shapes), "at least one microbatch shape is required")
            for shape in self._shapes:
                _require(
                    len(shape) == 3
                    and all(type(value) is int and value > 0 for value in shape),
                    "microbatch shapes must be positive integer [B,L,H]",
                )
                _require(shape[-1] == self._width, "microbatch hidden shape differs")
            self._tokens = tuple(shape[0] * shape[1] for shape in self._shapes)
            self._offsets = []
            offset = 0
            for tokens in self._tokens:
                self._offsets.append(offset)
                offset += tokens
            self._total_tokens = offset
            self._max_tokens = max(self._tokens)
            self._tile = min(_TILE_ROWS, self._max_tokens)
            self._expected_pairs = self._n_layers * len(self._tokens)
            self._allocation = routing_allocation_bound(
                n_layers=self._n_layers, mb_shapes=self._shapes, experts=self._experts,
                topk=self._topk, oracle=self.oracle, device_type=self._device.type,
            )
            self._charged = sum(self._allocation["host"].values())
            self._lease = host_budget.reserve(f"granite-route-probe-{id(self)}", self._charged)
            # Every owned persistent tensor and comparison scratch is allocated after admission.
            elements = self._n_layers * self._total_tokens * self._topk
            self._ids = torch.empty(elements, dtype=torch.int64, device="cpu")
            if oracle:
                self._probabilities = torch.empty(elements, dtype=torch.float32, device="cpu")
            self._comparison = torch.empty(
                (self._max_tokens, self._topk), dtype=torch.int64, device="cpu",
            )
            self._slot_diff = torch.empty(
                (self._tile, self._topk), dtype=torch.bool, device="cpu",
            )
            self._row_diff = torch.empty(self._tile, dtype=torch.bool, device="cpu")
            self._reduction = torch.empty((), dtype=torch.int64, device="cpu")
            self._expert_counts = torch.zeros(
                (self._n_layers, self._experts), dtype=torch.int64, device="cpu",
            )
            self._top1_counts = torch.zeros_like(self._expert_counts)
            self._stats = [
                {"forward_tokens": 0, "compared_tokens": 0, "mismatched_tokens": 0,
                 "mismatched_slots": 0, "set_mismatched_tokens": 0,
                 "forward_calls": 0, "recompute_calls": 0}
                for _ in layers
            ]
            self._source_hashes = {
                "native_router": _source_hash(type(first)),
                "native_router_forward": _source_hash(type(first).forward),
                "benchmark": _source_hash(RouteProbe),
            }
            for index, router in enumerate(self._routers):
                _finite(router.weight, "router weight")
                had_forward = "forward" in router.__dict__
                previous = router.__dict__.get("forward")
                self._originals.append((router, had_forward, previous))
                router.forward = self._wrapper(index, router.forward, router)
            self._model_handle = model.register_forward_pre_hook(
                self._model_pre_hook, with_kwargs=True,
            )
        except BaseException as error:
            # Partial constructor tracebacks may include a local reference to an allocated view.
            layers = first = router = weight = model = None
            traceback.clear_frames(error.__traceback__)
            self.close()
            raise

    def _wrapper(self, index: int, original: Any, router: Any) -> Any:
        owner_ref = weakref.ref(self)

        def wrapped(_router: Any, *args: Any, **kwargs: Any) -> Any:
            owner = owner_ref()
            if owner is None:
                return original(*args, **kwargs)
            return owner._dispatch(index, original, args, kwargs)

        return MethodType(wrapped, router)

    def _check_router(self, index: int) -> None:
        import torch

        _require(self._model.config.output_router_logits is False,
                 "output_router_logits must remain False")
        router = self._routers[index]
        weight = router.weight
        _require(not weight.requires_grad and weight.dtype == torch.bfloat16,
                 "router weight must remain frozen BF16")
        _require((id(weight), weight._version, weight.data_ptr()) == self._weight_signatures[index],
                 "router weight changed since the original forward")
        _require(
            weight.device == self._device and weight.is_contiguous(),
            "router weight layout changed",
        )
        _require(router.num_experts == self._experts and router.top_k == self._topk,
                 "router dimensions changed")

    def _model_pre_hook(self, _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        try:
            _require(self._model.config.output_router_logits is False,
                     "output_router_logits must remain False")
            _require(kwargs.get("output_router_logits") in (False, None),
                     "output_router_logits override enables an unsupported auxiliary loss")
            signature = inspect.signature(type(self._model).forward)
            bound = signature.bind_partial(self._model, *args, **kwargs)
            flags = bound.arguments
            _require(flags.get("output_router_logits") in (False, None),
                     "output_router_logits positional override is unsupported")
            _require(flags.get("kwargs", {}).get("output_router_logits") in (False, None),
                     "output_router_logits override is unsupported")
        except BaseException as error:
            args = kwargs = None
            self._abort(error)
            raise

    def _id_record(self, pair: tuple[int, int]) -> Any:
        layer, mb = pair
        offset = (layer * self._total_tokens + self._offsets[mb]) * self._topk
        return self._ids.narrow(0, offset, self._tokens[mb] * self._topk).view(-1, self._topk)

    def _probability_record(self, pair: tuple[int, int]) -> Any:
        layer, mb = pair
        offset = (layer * self._total_tokens + self._offsets[mb]) * self._topk
        return self._probabilities.narrow(
            0, offset, self._tokens[mb] * self._topk,
        ).view(-1, self._topk)

    def _dispatch(self, layer: int, original: Any, args: tuple[Any, ...],
                  kwargs: dict[str, Any]) -> Any:
        import torch

        hidden = output = ids = probabilities = state = None
        try:
            _require(not self._closed, "probe is closed")
            if self._visit is None:
                self._counts["unobserved_native_calls"] += 1
                return original(*args, **kwargs)
            expected_layer, mb, phase = self._visit
            _require(
                expected_layer is None or expected_layer == layer,
                "visit layer differs from router",
            )
            self._check_router(layer)
            _require(
                (len(args) == 1 and not kwargs) or (not args and set(kwargs) == {"hidden_states"}),
                "unsupported router arguments",
            )
            hidden = args[0] if args else kwargs["hidden_states"]
            _require(tuple(hidden.shape) == (self._tokens[mb], self._width),
                     "router input shape exceeds or differs from declared microbatch shape")
            _require(hidden.device == self._device and hidden.dtype == torch.bfloat16,
                     "router input must be BF16 on the original device")
            _require(hidden.is_contiguous(), "noncontiguous router inputs are unsupported")
            _require(not torch.is_autocast_enabled(self._device.type),
                     "router autocast must be disabled to preserve the native BF16 linear output")
            _finite(hidden, "router input")
            self._cuda_used = self._cuda_used or hidden.device.type == "cuda"
            pair = (layer, mb)
            if phase == "recompute" and self.oracle:
                _require(
                    expected_layer is not None, "Oracle recompute requires an explicit L2L layer",
                )
                _require(
                    pair in self._seen_forward, "Oracle recompute has no original forward record",
                )
                _require(pair not in self._seen_recompute, "duplicate Oracle replay")
                _require(
                    not self._pending, "only one unfinished Oracle replay fits the buffer bound",
                )
                ids = self._id_record(pair)
                probabilities = self._probability_record(pair)
                if torch.is_grad_enabled():
                    _require(hidden.requires_grad,
                             "Oracle recompute requires live differentiable router input")
                    state = _OracleReplay(self, pair)
                    self._pending[pair] = state
                    output = _oracle_weights(
                        hidden, self._routers[layer].weight, ids, probabilities, state,
                    )
                    self._counts["oracle_vjp_required"] += 1
                else:
                    # Output-only diagnostics have no graph and retain no replay state.
                    output = (
                        ids.to(device=hidden.device, copy=True, non_blocking=False),
                        probabilities.to(
                            device=hidden.device, copy=True, non_blocking=False,
                        ).to(dtype=torch.bfloat16),
                    )
                    self._counts["oracle_diagnostic_forward"] += 1
                self._counts["oracle_forward"] += 1
                if self._device.type == "cuda":
                    self._counts["id_h2d_bytes"] += ids.numel() * 8
                    self._counts["probability_h2d_bytes"] += probabilities.numel() * 4
                self._seen_recompute.add(pair)
                self._stats[layer]["recompute_calls"] += 1
                self._stats[layer]["compared_tokens"] += self._tokens[mb]
                return output[0], output[1], None
            _require(
                pair not in (self._seen_forward if phase == "forward" else self._seen_recompute),
                f"duplicate native router {phase} call",
            )
            self._counts["native_forward" if phase == "forward" else "native_recompute"] += 1
            output = original(*args, **kwargs)
            self._observe(pair, hidden, output, phase)
            return output
        except BaseException as error:
            # Nonreentrant checkpoint uses this private exception as normal control flow.
            if (type(error).__name__ == "_StopRecomputationError"
                    and type(error).__module__ == "torch.utils.checkpoint"):
                raise
            hidden = output = ids = probabilities = state = args = kwargs = None
            self._abort(error)
            raise
        finally:
            hidden = output = ids = probabilities = state = args = kwargs = None

    def _observe(self, pair: tuple[int, int], hidden: Any, output: Any, phase: str) -> None:
        import torch

        ids = weights = logits = record = probabilities = host_probabilities = current = None
        try:
            _require(
                isinstance(output, tuple) and len(output) == 3, "unsupported native router output",
            )
            ids, weights, logits = output
            tokens = self._tokens[pair[1]]
            _require(ids.dtype == torch.int64 and tuple(ids.shape) == (tokens, self._topk),
                     "native IDs shape/dtype differs")
            _require(weights.dtype == torch.bfloat16 and weights.shape == ids.shape,
                     "native weights shape/dtype differs")
            _require(
                logits.dtype == torch.float32 and tuple(logits.shape) == (tokens, self._experts),
                "native logits shape/dtype differs",
            )
            _require(ids.device == weights.device == logits.device == self._device,
                     "native router outputs moved device")
            _finite(logits, "native router logits")
            _finite(weights, "native routing weights")
            record = self._id_record(pair)
            if phase == "forward":
                record.copy_(ids.detach(), non_blocking=False)
                if self._device.type == "cuda":
                    self._counts["id_d2h_bytes"] += ids.numel() * 8
                self._accumulate_counts(pair[0], record)
                if self.oracle:
                    probabilities = _recover_probabilities(hidden, output)
                    host_probabilities = self._probability_record(pair)
                    host_probabilities.copy_(probabilities, non_blocking=False)
                    self._counts["probability_reconstruction"] += 1
                    if self._device.type == "cuda":
                        self._counts["probability_d2h_bytes"] += probabilities.numel() * 4
                # numpy shares the existing contiguous CPU storage; memoryview does not copy it.
                self._digests[pair] = hashlib.sha256(
                    memoryview(record.numpy()).cast("B"),
                ).hexdigest()
                self._seen_forward.add(pair)
                self._stats[pair[0]]["forward_calls"] += 1
                self._stats[pair[0]]["forward_tokens"] += tokens
            else:
                _require(
                    pair in self._seen_forward, "native recompute has no original forward record",
                )
                current = self._comparison[:tokens]
                current.copy_(ids.detach(), non_blocking=False)
                if self._device.type == "cuda":
                    self._counts["id_d2h_bytes"] += ids.numel() * 8
                self._compare(pair, record, current)
                self._seen_recompute.add(pair)
                self._stats[pair[0]]["recompute_calls"] += 1
                if self.strict and self._stats[pair[0]]["mismatched_tokens"]:
                    raise RoutingMismatchError(
                        f"native Granite ordered routing changed at layer {pair[0]} "
                        f"microbatch {pair[1]}"
                    )
        finally:
            hidden = output = ids = weights = logits = record = probabilities = None
            host_probabilities = current = None

    def _accumulate_counts(self, layer: int, record: Any) -> None:
        for row in range(record.shape[0]):
            seen: set[int] = set()
            for column in range(self._topk):
                expert = int(record[row, column].item())
                _require(0 <= expert < self._experts and expert not in seen,
                         "native route IDs must be unique in-range experts")
                seen.add(expert)
                self._expert_counts[layer, expert] += 1
                if column == 0:
                    self._top1_counts[layer, expert] += 1

    def _compare(self, pair: tuple[int, int], before: Any, current: Any) -> None:
        import torch

        mask = rows = expected = observed = None
        stats = self._stats[pair[0]]
        try:
            for start in range(0, before.shape[0], self._tile):
                count = min(self._tile, before.shape[0] - start)
                mask, rows = self._slot_diff[:count], self._row_diff[:count]
                torch.ne(before[start:start + count], current[start:start + count], out=mask)
                torch.any(mask, dim=1, out=rows)
                torch.sum(mask, dim=(0, 1), dtype=torch.int64, out=self._reduction)
                stats["mismatched_slots"] += int(self._reduction.item())
                torch.sum(rows, dim=0, dtype=torch.int64, out=self._reduction)
                stats["mismatched_tokens"] += int(self._reduction.item())
                for row in range(count):
                    if not bool(rows[row].item()):
                        continue
                    token = start + row
                    expected = [int(before[token, column].item()) for column in range(self._topk)]
                    observed = [int(current[token, column].item()) for column in range(self._topk)]
                    stats["set_mismatched_tokens"] += int(set(expected) != set(observed))
                    if self._first_mismatch is None:
                        self._first_mismatch = {
                            "layer": pair[0], "mb": pair[1], "token": token,
                            "slot": next(
                                index for index, values in enumerate(zip(expected, observed))
                                if values[0] != values[1]
                            ),
                            "expected_ids": expected, "observed_ids": observed,
                        }
                stats["compared_tokens"] += count
        finally:
            before = current = mask = rows = None

    def _abort(self, error: BaseException) -> None:
        if self._failure is None:
            self._failure = {"kind": type(error).__name__, "message": str(error)}
        traceback.clear_frames(error.__traceback__)
        self.close()

    @contextmanager
    def visit(self, layer: int | None, mb: int, phase: str) -> Iterator[None]:
        previous = self._visit
        try:
            _require(not self._closed, "probe is closed")
            _require(previous is None, "nested routing visits are unsupported")
            _require(type(mb) is int and 0 <= mb < len(self._tokens), "unknown microbatch index")
            _require(layer is None or type(layer) is int and 0 <= layer < self._n_layers,
                     "unknown routing layer index")
            _require(phase in {"forward", "recompute"}, "phase must be forward or recompute")
            _require(not (self.oracle and phase == "recompute" and layer is None),
                     "Oracle recompute requires an explicit L2L layer")
            self._visit = (layer, mb, phase)
            yield
        except BaseException as error:
            self._abort(error)
            raise
        finally:
            self._visit = previous if not self._closed else None

    def _owned_storage(self) -> tuple[int, int]:
        tensors = [self._ids, self._probabilities, self._comparison, self._slot_diff,
                   self._row_diff, self._reduction, self._expert_counts, self._top1_counts]
        raw = tensor = None
        try:
            for state in self._pending.values():
                tensors.extend((state.ids, state.probabilities))
            storage: set[tuple[str, int]] = set()
            host = device = 0
            for tensor in tensors:
                if tensor is None:
                    continue
                raw = tensor.untyped_storage()
                key = (str(tensor.device), raw.data_ptr())
                if key in storage:
                    continue
                storage.add(key)
                if tensor.device.type == "cpu":
                    host += raw.nbytes()
                else:
                    device += raw.nbytes()
            return host, device
        finally:
            tensors = raw = tensor = None

    def report(self) -> dict[str, Any]:
        layers = []
        for index, stats in enumerate(self._stats):
            counts = (self._frozen_expert_counts[index] if self._closed
                      else self._expert_counts[index].tolist())
            top1 = (self._frozen_top1_counts[index] if self._closed
                    else self._top1_counts[index].tolist())
            assignments = sum(counts)
            positive = [count for count in counts if count]
            entropy = -sum((count / assignments) * math.log(count / assignments)
                           for count in positive) if assignments else None
            compared = stats["compared_tokens"]
            layer_complete = (stats["forward_calls"] == len(self._tokens)
                              and stats["recompute_calls"] == len(self._tokens))
            observed_rmr = 1 - stats["mismatched_tokens"] / compared if compared else None
            layers.append({
                "layer": index, **stats, "experts": self._experts, "topk": self._topk,
                "forward_expert_counts": counts, "forward_top1_counts": top1,
                "assignments": assignments, "expert_coverage": len(positive) / self._experts,
                "zero_hit_experts": [expert for expert, count in enumerate(counts) if not count],
                "entropy": entropy,
                "normalized_entropy": entropy / math.log(self._experts)
                if entropy is not None and self._experts > 1 else None,
                "effective_experts": math.exp(entropy) if entropy is not None else None,
                "maximum_assignment_share": max(counts) / assignments if assignments else None,
                "complete": layer_complete,
                "natural_rmr": observed_rmr if layer_complete and not self.oracle else None,
                "observed_natural_rmr": observed_rmr if not self.oracle else None,
                "forced_rmr": 1.0 if layer_complete and self.oracle else None,
            })
        routing_complete = (
            len(self._seen_forward) == len(self._seen_recompute) == self._expected_pairs
        )
        complete = routing_complete and (
            not self.oracle or self._counts["oracle_vjp"] == self._expected_pairs
        )
        compared = sum(stats["compared_tokens"] for stats in self._stats)
        mismatches = sum(stats["mismatched_tokens"] for stats in self._stats)
        slot_mismatches = sum(stats["mismatched_slots"] for stats in self._stats)
        set_mismatches = sum(stats["set_mismatched_tokens"] for stats in self._stats)
        observed = 1 - mismatches / compared if compared else None
        routing_pass = complete and not mismatches and self._failure is None
        output_only = self._counts["oracle_diagnostic_forward"] > 0
        output_only_complete = (
            routing_complete and self.oracle and self._failure is None
            and self._counts["oracle_diagnostic_forward"] == self._expected_pairs
        )
        host, device = self._owned_storage()
        return {
            "oracle": self.oracle, "strict": self.strict,
            "diagnostic_only": not self.strict or output_only,
            "admissible": routing_pass and self.strict and not output_only,
            "natural_admissible": routing_pass and self.strict and not self.oracle,
            "routing_pass": routing_pass, "complete": complete,
            "routing_forward_complete": routing_complete,
            "output_only_diagnostic_complete": output_only_complete,
            "oracle_output_only": output_only,
            "vjp_required": self.oracle and (
                not output_only or self._counts["oracle_vjp_required"] > 0
            ),
            "natural_rmr": observed if routing_complete and not self.oracle else None,
            "ordered_topk_rmr": observed if routing_complete and not self.oracle else None,
            "observed_natural_rmr": observed if not self.oracle else None,
            "natural_slot_equality": 1 - slot_mismatches / (compared * self._topk)
            if routing_complete and compared and not self.oracle else None,
            "natural_set_rmr": 1 - set_mismatches / compared
            if routing_complete and compared and not self.oracle else None,
            "forced_rmr": (
                1.0 if routing_complete and self.oracle and self._failure is None else None
            ),
            "forced_routing": self.oracle, "native_router_replay": not self.oracle,
            "oracle_semantics": "saved-Jacobian-at-original-forward" if self.oracle else None,
            "first_mismatch": self._first_mismatch, "failure": self._failure,
            "admission_refusal": self._failure or (
                {"kind": "RoutingMismatchError", "message": "natural ordered route mismatch"}
                if mismatches else {"kind": "DiagnosticOnly",
                                    "message": "output-only Oracle diagnostic; no training VJP"}
                if output_only else {"kind": "DiagnosticOnly", "message": "strict=False diagnostic"}
                if not self.strict else None
            ),
            "layers": layers, "counts": dict(self._counts),
            "native_forward_calls": (
                self._counts["native_forward"] + self._counts["native_recompute"]
            ),
            "oracle_forward_calls": self._counts["oracle_forward"],
            "oracle_vjp_calls": self._counts["oracle_vjp"],
            "oracle_diagnostic_forward_calls": self._counts["oracle_diagnostic_forward"],
            "oracle_vjp_required_calls": self._counts["oracle_vjp_required"],
            "expected_pairs": self._expected_pairs, "forward_records": len(self._seen_forward),
            "recomputed_records": len(self._seen_recompute),
            "forward_digests": [{"layer": pair[0], "mb": pair[1], "sha256": digest}
                                for pair, digest in sorted(self._digests.items())],
            "source_hashes": self._source_hashes,
            "charged_host_bytes": self._charged,
            "live_reserved_host_bytes": self._lease.nbytes if self._lease is not None else 0,
            "device_scratch_bytes": self._allocation["device_buffer_bound_bytes"],
            "allocation": self._allocation,
            "owned_host_storage_bytes": host, "owned_device_storage_bytes": device,
            "pending_oracle_replays": len(self._pending), "closed": self._closed,
            "cleanup": {"original_forwards_restored": not self._originals,
                        "hooks_removed": self._model_handle is None,
                        "owned_buffers_dropped": host == device == 0,
                        "synchronous_device_drains": self._syncs},
        }

    def close(self) -> None:
        """Restore interception, drain CUDA, drop tensor ownership, then release the lease."""
        if self._closed:
            return
        for router, had_forward, previous in reversed(self._originals):
            if had_forward:
                router.forward = previous
            else:
                del router.forward
        self._originals.clear()
        if self._model_handle is not None:
            self._model_handle.remove()
            self._model_handle = None
        if self._cuda_used:
            import torch

            # A failed drain leaves ownership and the lease intact for an explicit close retry.
            torch.cuda.current_stream(self._device).synchronize()
            self._syncs += 1
            self._cuda_used = False
        if self._expert_counts is not None:
            self._frozen_expert_counts = self._expert_counts.tolist()
        if self._top1_counts is not None:
            self._frozen_top1_counts = self._top1_counts.tolist()
        for state in self._pending.values():
            state.clear()
        self._pending.clear()
        self._ids = self._probabilities = self._comparison = None
        self._slot_diff = self._row_diff = self._reduction = None
        self._expert_counts = self._top1_counts = None
        self._routers.clear()
        self._model = None
        self._visit = None
        if self._lease is not None:
            self._lease.close()
            self._lease = None
        self._closed = True

    def __enter__(self) -> RouteProbe:
        _require(not self._closed, "probe is closed")
        return self

    def __exit__(self, exc_type: Any, error: BaseException | None, tb: Any) -> None:
        if error is not None:
            traceback.clear_frames(tb)
            self._abort(error)
        else:
            self.close()
