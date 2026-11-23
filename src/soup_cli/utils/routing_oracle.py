"""Production DeepSeek-V3 routing Oracle for lossy L2L boundary activations.

Ported from PR #77, commit 5e3fa12c (author kok-o), without the FP8 codec and the
decoupled-MBR extension of that commit; the local MBR path passes ``routing_oracle``
to :meth:`soup_cli.utils.l2l_step.L2LStep.run`.

Captures native ``DeepseekV3TopkRouter`` selections ``(ids, selected_sigmoid_scores,
normalizer)`` during the no-grad L2L forward walk and replays those exact routes
during backward recompute with the exact first-order original-forward sigmoid and
normalization Jacobian into the router input (identical numerical semantics to
``benchmarks/harness/p3_deepseek_oracle.py``).

Supports three explicit visit phases per ``(layer, microbatch)``:
- ``"forward"`` (``torch.no_grad``): runs the native router, records compact CPU
  ``(ids, scores, normalizer)`` asynchronously on pinned memory when on CUDA, and
  returns the native ``(router_logits, topk_weights, topk_indices)`` untouched.
- ``"peek"`` (``torch.no_grad``): read-only replay used by decoupled MBR Phase R1
  to supply exact original-forward ``(None, weights, ids)`` without consuming the
  differentiable recompute slot or building an autograd graph.
- ``"recompute"`` (``torch.enable_grad``): returns ``(None, weights, ids)`` backed
  by the exact first-order ``_OriginalSigmoidJacobian`` autograd Function so router
  weight gradients propagate back through attention and LoRA adapters.

Top-level imports are stdlib-only; ``torch`` and ``transformers`` stay lazy.
"""

from __future__ import annotations

import math
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from types import MethodType
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

__all__ = ["DeepseekRoutingOracle"]

_NATIVE_MODULE = "transformers.models.deepseek_v3.modeling_deepseek_v3"
_EPSILON = 1e-20


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"DeepSeek Routing Oracle: {message}")


def _router_settings(gate: Any) -> Tuple[Any, ...]:
    return (
        gate.hidden_dim,
        gate.num_experts,
        gate.top_k,
        gate.num_group,
        gate.topk_group,
        gate.norm_topk_prob,
        gate.routed_scaling_factor,
    )


@dataclass
class _LayerBinding:
    gate: Any
    wrapper: Any
    moe: Any
    native_moe_forward: Any
    settings: Tuple[Any, ...]
    device: Any
    moe_index: int


@dataclass
class _ForwardRecord:
    ids: Any
    scores: Any
    normalizer: Any
    hidden_shape: Tuple[int, ...]
    hidden_dtype: Any
    settings: Tuple[Any, ...]


class _ReplayState:
    """Clearable autograd context state for one pending router VJP."""

    __slots__ = (
        "owner",
        "pair",
        "ids",
        "scores",
        "normalizer",
        "weight",
        "shape",
        "dtype",
        "settings",
        "closed",
    )

    def __init__(self, owner: "DeepseekRoutingOracle", pair: Tuple[int, int]) -> None:
        self.owner = weakref.ref(owner)
        self.pair = pair
        self.ids: Any = None
        self.scores: Any = None
        self.normalizer: Any = None
        self.weight: Any = None
        self.shape: Tuple[int, ...] = ()
        self.dtype: Any = None
        self.settings: Tuple[Any, ...] = ()
        self.closed: bool = False

    def clear(self) -> None:
        self.ids = None
        self.scores = None
        self.normalizer = None
        self.weight = None
        self.closed = True


@lru_cache(maxsize=1)
def _oracle_function() -> Any:
    import torch

    class _OriginalSigmoidJacobian(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, hidden: Any, state: _ReplayState) -> Tuple[Any, Any]:
            owner = state.owner()
            _require(owner is not None and not owner._closed, "probe owner is unavailable")
            ctx.set_materialize_grads(False)
            ctx.state = state
            ids = state.ids.clone()
            if state.settings[5]:
                normalized = state.scores / state.normalizer
            else:
                normalized = state.scores
            weights = normalized * state.settings[6]
            ctx.mark_non_differentiable(ids)
            return weights, ids

        @staticmethod
        def backward(ctx: Any, incoming: Any, _ids_gradient: Any) -> Tuple[Any, Any]:
            state = ctx.state
            owner = state.owner()
            try:
                _require(
                    not state.closed and owner is not None and not owner._closed,
                    "replay is closed",
                )
                _require(not torch.is_grad_enabled(), "only first-order VJPs are supported")
                _require(incoming is not None, "routing weights received no gradient")
                owner._verify_replay_slot(state.pair[0], state.weight)
                _require(
                    incoming.dtype == torch.float32
                    and incoming.shape == state.scores.shape
                    and incoming.device == state.scores.device,
                    "incoming routing-weight gradient has unexpected shape/dtype/device",
                )
                grad = incoming if incoming.is_contiguous() else incoming.contiguous()
                scaled = grad * state.settings[6]
                if state.settings[5]:
                    normalized = state.scores / state.normalizer
                    denominator_term = -scaled
                    denominator_term = denominator_term * normalized
                    denominator_term = denominator_term / state.normalizer
                    denominator_term = denominator_term.sum(dim=-1, keepdim=True)
                    selected = scaled / state.normalizer
                    selected = selected + denominator_term
                else:
                    selected = scaled
                sigmoid_gradient = torch.ops.aten.sigmoid_backward.default(
                    selected, state.scores
                )
                full = torch.zeros(
                    (state.scores.shape[0], state.weight.shape[0]),
                    dtype=torch.float32,
                    device=state.scores.device,
                )
                full.scatter_(1, state.ids, sigmoid_gradient)
                weight32 = (
                    state.weight
                    if state.weight.dtype == torch.float32
                    else state.weight.float()
                )
                gradient32 = full.mm(weight32)
                hidden_gradient = gradient32.to(dtype=state.dtype).view(state.shape)
                owner._complete_replay(state)
                return hidden_gradient, None
            except BaseException:
                state.clear()
                raise

    return _OriginalSigmoidJacobian


class DeepseekRoutingOracle:
    """Reusable multi-step routing Oracle for native DeepSeek-V3 MoE models.

    Install around a training run with ``with DeepseekRoutingOracle(model) as oracle:``
    and pass ``routing_oracle=oracle`` to :meth:`soup_cli.utils.l2l_step.L2LStep.run`.
    """

    def __init__(
        self,
        model: Any,
        *,
        host_budget: Any = None,
        pin: Optional[bool] = None,
    ) -> None:
        self._closed = False
        self._host_budget = host_budget
        self._lease: Any = None
        self._bindings: Dict[int, _LayerBinding] = {}
        self._originals: List[Tuple[Any, bool, Any]] = []
        self._n_layers = 0
        self._width = 0
        self._experts = 0
        self._topk = 0
        self._device: Any = None
        self._pin = False
        self._id_dtype: Any = None
        self._id_element_bytes = 2

        # Persistent CPU storage reused across steps of equal or smaller token count.
        self._capacity_rows = 0
        self._ids: Any = None
        self._scores: Any = None
        self._normalizers: Any = None

        # Per-step state initialized in begin_step().
        self._step_active = False
        self._shapes: Tuple[Tuple[int, int, int], ...] = ()
        self._tokens: Tuple[int, ...] = ()
        self._offsets: List[int] = []
        self._total_tokens = 0
        self._expected_pairs = 0
        self._records: Dict[Tuple[int, int], _ForwardRecord] = {}
        self._pending: Dict[Tuple[int, int], _ReplayState] = {}
        self._seen_forward: Set[Tuple[int, int]] = set()
        self._seen_recompute: Set[Tuple[int, int]] = set()
        self._seen_vjp: Set[Tuple[int, int]] = set()
        self._visit: Optional[Tuple[int, int, str]] = None
        self._visit_calls = 0
        self._steps_completed = 0
        self._counts = {
            "native_forward": 0,
            "oracle_peek": 0,
            "oracle_recompute": 0,
            "oracle_vjp": 0,
        }
        self._step_counts = dict(self._counts)

        try:
            self._initialize(model, pin=pin)
        except BaseException:
            self.close()
            raise

    def _initialize(self, model: Any, *, pin: Optional[bool]) -> None:
        import torch
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
            DeepseekV3DecoderLayer,
            DeepseekV3ForCausalLM,
            DeepseekV3MLP,
            DeepseekV3MoE,
            DeepseekV3TopkRouter,
        )

        from soup_cli.utils.expert_stream import _routing_forward
        from soup_cli.utils.l2l_step import model_parts
        from soup_cli.utils.layer_stream_runtime import StreamedDecoderLayer

        causal, owner = model_parts(model)
        _require(type(causal) is DeepseekV3ForCausalLM, "expected native DeepseekV3ForCausalLM")
        _require(type(causal).__module__ == _NATIVE_MODULE, "expected native DeepSeek module")
        config = causal.config
        _require(getattr(config, "model_type", None) == "deepseek_v3", "unsupported model_type")
        _require(
            getattr(config, "output_router_logits", False) in (False, None),
            "full router-logit consumers are unsupported",
        )
        for field in ("router_aux_loss_coef", "aux_loss_alpha"):
            _require(getattr(config, field, 0) in (0, None), "auxiliary router loss is unsupported")

        self._n_layers = len(owner.layers)
        _require(
            self._n_layers == int(config.num_hidden_layers) and self._n_layers > 0,
            "decoder layer count differs from config",
        )
        self._width = int(config.hidden_size)
        first_k = int(config.first_k_dense_replace)
        _require(0 <= first_k < self._n_layers, "at least one native MoE layer is required")

        moe_index = 0
        for layer_idx, module in enumerate(owner.layers):
            wrapper = module if isinstance(module, StreamedDecoderLayer) else None
            inner = module.inner if wrapper is not None else module
            _require(type(inner) is DeepseekV3DecoderLayer, "non-native decoder layer")
            if layer_idx < first_k:
                _require(
                    type(inner.mlp) is DeepseekV3MLP, "dense layer is not native DeepseekV3MLP"
                )
                continue
            _require(type(inner.mlp) is DeepseekV3MoE, "non-native MoE module is unsupported")
            _require(
                getattr(inner.mlp.forward, "__func__", None) is DeepseekV3MoE.forward,
                "non-native MoE forward is unsupported",
            )
            gate = inner.mlp.gate
            _require(type(gate) is DeepseekV3TopkRouter, "expected native DeepseekV3TopkRouter")
            gate_func = getattr(gate.forward, "__func__", None)
            _require(
                gate_func in (DeepseekV3TopkRouter.forward, _routing_forward),
                "native gate forward is already intercepted",
            )
            settings = _router_settings(gate)
            width, experts, topk, groups, chosen_groups, normalize, scaling = settings
            _require(
                all(type(v) is int and v > 0 for v in (width, experts, topk, groups, chosen_groups))
                and experts % groups == 0
                and experts // groups >= 2
                and chosen_groups <= groups
                and topk <= chosen_groups * (experts // groups)
                and type(normalize) is bool
                and type(scaling) in {int, float}
                and math.isfinite(scaling)
                and scaling > 0,
                "unsupported native router settings",
            )
            _require(
                (width, experts, topk)
                == (self._width, int(config.num_local_experts), int(config.num_experts_per_tok)),
                "gate dimensions differ from config",
            )
            _require(
                not gate.weight.requires_grad and not gate.e_score_correction_bias.requires_grad,
                "router weights must be frozen",
            )
            _require(
                all(existing.settings == settings for existing in self._bindings.values()),
                "nonuniform router settings are unsupported",
            )
            if wrapper is not None:
                _require(wrapper.idx == layer_idx, "streamed wrapper layer index mismatch")
                device = torch.device(wrapper.pool.device)
            else:
                device = gate.weight.device
            if device.type == "cuda" and device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            _require(device.type in {"cpu", "cuda"}, "only CPU/CUDA gates are supported")
            _require(
                all(existing.device == device for existing in self._bindings.values()),
                "all gates must execute on one device",
            )
            self._bindings[layer_idx] = _LayerBinding(
                gate=gate,
                wrapper=wrapper,
                moe=inner.mlp,
                native_moe_forward=DeepseekV3MoE.forward,
                settings=settings,
                device=device,
                moe_index=moe_index,
            )
            moe_index += 1

        first = next(iter(self._bindings.values()))
        self._device = first.device
        self._experts, self._topk = int(first.settings[1]), int(first.settings[2])
        self._pin = (self._device.type == "cuda") if pin is None else bool(pin)
        if self._experts <= 32767:
            self._id_dtype = torch.int16
            self._id_element_bytes = 2
        else:
            self._id_dtype = torch.int32
            self._id_element_bytes = 4

        for layer_idx, binding in self._bindings.items():
            gate = binding.gate
            self._originals.append(
                (gate, "forward" in gate.__dict__, gate.__dict__.get("forward"))
            )
            native_bound = MethodType(DeepseekV3TopkRouter.forward, gate)
            gate.forward = self._make_wrapper(layer_idx, gate, native_bound)

    def _make_wrapper(self, layer_idx: int, gate: Any, native: Any) -> Any:
        ref = weakref.ref(self)

        def wrapped(_gate: Any, *args: Any, **kwargs: Any) -> Any:
            owner = ref()
            _require(owner is not None and not owner._closed, "intercepted gate has no live oracle")
            return owner._dispatch(layer_idx, native, args, kwargs)

        return MethodType(wrapped, gate)

    def _ensure_buffers(self, needed_rows: int) -> None:
        import torch

        if needed_rows <= self._capacity_rows and self._ids is not None:
            return
        if self._device.type == "cuda" and self._ids is not None:
            torch.cuda.current_stream(self._device).synchronize()
        self._ids = None
        self._scores = None
        self._normalizers = None
        if self._lease is not None:
            self._lease.close()
            self._lease = None

        id_bytes = needed_rows * self._topk * self._id_element_bytes
        score_bytes = needed_rows * self._topk * 4
        norm_bytes = needed_rows * 4
        total_bytes = id_bytes + score_bytes + norm_bytes
        if self._host_budget is not None:
            self._lease = self._host_budget.reserve(
                f"deepseek-routing-oracle-{id(self)}", total_bytes
            )
        try:
            self._ids = torch.empty(
                (needed_rows, self._topk),
                dtype=self._id_dtype,
                device="cpu",
                pin_memory=self._pin,
            )
            self._scores = torch.empty(
                (needed_rows, self._topk),
                dtype=torch.float32,
                device="cpu",
                pin_memory=self._pin,
            )
            self._normalizers = torch.empty(
                (needed_rows, 1),
                dtype=torch.float32,
                device="cpu",
                pin_memory=self._pin,
            )
            self._capacity_rows = needed_rows
        except BaseException:
            self._ids = self._scores = self._normalizers = None
            self._capacity_rows = 0
            if self._lease is not None:
                self._lease.close()
                self._lease = None
            raise

    def begin_step(self, mb_shapes: Sequence[Sequence[int]]) -> None:
        """Prepare storage and counters for one L2L optimizer window."""
        _require(not self._closed, "oracle is closed")
        _require(not self._step_active, "previous step was not ended or aborted")
        shapes = tuple(tuple(int(d) for d in shape) for shape in mb_shapes)
        _require(
            bool(shapes)
            and all(len(s) == 3 and all(d > 0 for d in s) and s[2] == self._width for s in shapes),
            "mb_shapes must contain positive [B, L, H] shapes matching hidden_size",
        )
        self._shapes = shapes
        self._tokens = tuple(s[0] * s[1] for s in shapes)
        self._offsets = []
        offset = 0
        for count in self._tokens:
            self._offsets.append(offset)
            offset += count
        self._total_tokens = offset
        self._expected_pairs = len(self._bindings) * len(shapes)
        needed_rows = len(self._bindings) * self._total_tokens
        self._ensure_buffers(needed_rows)

        self._records.clear()
        self._pending.clear()
        self._seen_forward.clear()
        self._seen_recompute.clear()
        self._seen_vjp.clear()
        self._visit = None
        self._visit_calls = 0
        self._step_counts = {
            "native_forward": 0,
            "oracle_peek": 0,
            "oracle_recompute": 0,
            "oracle_vjp": 0,
        }
        self._step_active = True

    def _verify_slot(self, layer_idx: int, weight: Any) -> None:
        binding = self._bindings[layer_idx]
        wrapper = binding.wrapper
        _require(
            getattr(binding.moe.forward, "__func__", None) is binding.native_moe_forward,
            "MoE forward changed to an unsupported router-logit consumer",
        )
        _require(
            not weight.is_meta and weight.device == binding.device,
            "restored router weight is not materialized on the declared device",
        )
        if wrapper is not None:
            slot = wrapper.pool.slot_for(layer_idx)
            _require(wrapper.pool.owner[slot] == layer_idx, "streamed slot owner changed")

    def _verify_replay_slot(self, layer_idx: int, weight: Any) -> None:
        _require(weight is not None, "pending replay lost router weight reference")
        self._verify_slot(layer_idx, weight)

    def _capture(self, pair: Tuple[int, int], hidden: Any, output: Any) -> None:
        import torch

        _require(isinstance(output, tuple) and len(output) == 3, "native gate return changed")
        logits, _weights, ids = output
        layer_idx, mb = pair
        rows = self._tokens[mb]
        _require(
            tuple(logits.shape) == (rows, self._experts)
            and tuple(ids.shape) == (rows, self._topk)
            and ids.dtype == torch.int64,
            "native gate output shapes/dtypes changed",
        )
        binding = self._bindings[layer_idx]
        scores = logits.detach().sigmoid()
        selected = scores.gather(1, ids)
        normalizer = selected.sum(dim=-1, keepdim=True) + _EPSILON

        base = binding.moe_index * self._total_tokens + self._offsets[mb]
        ids_slot = self._ids[base : base + rows]
        scores_slot = self._scores[base : base + rows]
        norm_slot = self._normalizers[base : base + rows]
        ids_slot.copy_(ids.detach().to(dtype=self._id_dtype), non_blocking=self._pin)
        scores_slot.copy_(selected, non_blocking=self._pin)
        norm_slot.copy_(normalizer, non_blocking=self._pin)

        self._records[pair] = _ForwardRecord(
            ids=ids_slot,
            scores=scores_slot,
            normalizer=norm_slot,
            hidden_shape=tuple(int(d) for d in hidden.shape),
            hidden_dtype=hidden.dtype,
            settings=binding.settings,
        )
        self._seen_forward.add(pair)
        self._counts["native_forward"] += 1
        self._step_counts["native_forward"] += 1

    def _peek(self, pair: Tuple[int, int], hidden: Any) -> Tuple[Any, Any, Any]:
        import torch

        record = self._records.get(pair)
        _require(record is not None, f"missing forward record for peek on {pair}")
        _require(
            tuple(int(d) for d in hidden.shape) == record.hidden_shape
            and hidden.dtype == record.hidden_dtype,
            "peek hidden shape/dtype changed",
        )
        device = hidden.device
        ids = record.ids.to(device=device, dtype=torch.int64, non_blocking=self._pin)
        scores = record.scores.to(device=device, non_blocking=self._pin)
        normalizer = record.normalizer.to(device=device, non_blocking=self._pin)
        normalized = scores / normalizer if record.settings[5] else scores
        weights = normalized * record.settings[6]
        self._counts["oracle_peek"] += 1
        self._step_counts["oracle_peek"] += 1
        return None, weights, ids

    def _replay(self, pair: Tuple[int, int], hidden: Any, weight: Any) -> Tuple[Any, Any, Any]:
        import torch

        record = self._records.get(pair)
        _require(record is not None, f"missing forward record for recompute on {pair}")
        _require(
            tuple(int(d) for d in hidden.shape) == record.hidden_shape
            and hidden.dtype == record.hidden_dtype,
            "recompute hidden shape/dtype changed",
        )
        state = _ReplayState(self, pair)
        state.weight = weight
        state.shape = record.hidden_shape
        state.dtype = record.hidden_dtype
        state.settings = record.settings
        device = hidden.device
        state.ids = record.ids.to(device=device, dtype=torch.int64, non_blocking=self._pin)
        state.scores = record.scores.to(device=device, non_blocking=self._pin)
        state.normalizer = record.normalizer.to(device=device, non_blocking=self._pin)
        self._pending[pair] = state
        weights, ids = _oracle_function().apply(hidden, state)
        self._seen_recompute.add(pair)
        self._counts["oracle_recompute"] += 1
        self._step_counts["oracle_recompute"] += 1
        return None, weights, ids

    def _complete_replay(self, state: _ReplayState) -> None:
        pair = state.pair
        _require(self._pending.get(pair) is state, "missing or duplicate pending replay")
        _require(pair not in self._seen_vjp, "duplicate Oracle VJP")
        self._seen_vjp.add(pair)
        self._counts["oracle_vjp"] += 1
        self._step_counts["oracle_vjp"] += 1
        self._pending.pop(pair, None)
        self._records.pop(pair, None)
        state.clear()

    def _dispatch(
        self,
        layer_idx: int,
        native: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        import torch

        _require(not self._closed, "oracle is closed")
        _require(self._step_active, "native gate call outside an active oracle step")
        _require(self._visit is not None, "native gate call requires an explicit layer visit")
        expected_layer, mb, phase = self._visit
        _require(layer_idx == expected_layer, "visit layer differs from native gate")
        _require(self._visit_calls == 0, "duplicate native gate call inside visit")
        _require(
            (len(args) == 1 and not kwargs)
            or (not args and set(kwargs) == {"hidden_states"}),
            "native gate arguments changed",
        )
        hidden = args[0] if args else kwargs["hidden_states"]
        _require(
            isinstance(hidden, torch.Tensor)
            and tuple(int(d) for d in hidden.shape)
            in (self._shapes[mb], (self._tokens[mb], self._width)),
            "hidden shape differs from declared microbatch",
        )
        _require(
            hidden.dtype in {torch.bfloat16, torch.float32} and hidden.device == self._device,
            "hidden must be BF16/FP32 on the declared device",
        )
        _require(
            not torch.is_autocast_enabled(hidden.device.type),
            "autocast is unsupported for native FP32 router linear",
        )
        binding = self._bindings[layer_idx]
        pair = (layer_idx, mb)
        weight = binding.gate.weight
        self._verify_slot(layer_idx, weight)

        if phase == "forward":
            _require(not torch.is_grad_enabled(), "forward visit must disable grad")
            _require(pair not in self._seen_forward, "duplicate forward pair")
            output = native(*args, **kwargs)
            self._capture(pair, hidden, output)
        elif phase == "peek":
            _require(not torch.is_grad_enabled(), "peek visit must disable grad")
            _require(pair in self._seen_forward, "peek missing original forward")
            _require(pair not in self._seen_recompute, "peek after recompute")
            output = self._peek(pair, hidden)
        else:
            _require(pair in self._seen_forward, "recompute missing original forward")
            _require(pair not in self._seen_recompute, "duplicate recompute pair")
            _require(
                torch.is_grad_enabled() and hidden.requires_grad,
                "recompute requires grad and differentiable hidden input",
            )
            output = self._replay(pair, hidden, weight)

        self._visit_calls += 1
        return output

    @contextmanager
    def visit(self, layer: int, microbatch: int, phase: str) -> Iterator[None]:
        _require(not self._closed, "oracle is closed")
        _require(self._step_active, "visit requires an active step via begin_step()")
        _require(self._visit is None, "nested oracle visits are unsupported")
        _require(type(layer) is int and 0 <= layer < self._n_layers, "unknown layer index")
        _require(
            type(microbatch) is int and 0 <= microbatch < len(self._shapes),
            "unknown microbatch index",
        )
        _require(phase in {"forward", "peek", "recompute"}, f"invalid oracle phase {phase!r}")
        self._visit = (layer, microbatch, phase)
        self._visit_calls = 0
        try:
            yield
            _require(
                self._visit_calls == int(layer in self._bindings),
                f"incomplete layer visit ({layer}, {microbatch}, {phase}): "
                f"expected {int(layer in self._bindings)} gate calls, got {self._visit_calls}",
            )
        finally:
            self._visit = None
            self._visit_calls = 0

    def _step_complete(self) -> bool:
        return (
            self._expected_pairs > 0
            and len(self._seen_forward) == self._expected_pairs
            and len(self._seen_recompute) == self._expected_pairs
            and len(self._seen_vjp) == self._expected_pairs
            and not self._pending
            and not self._records
        )

    def require_complete(self) -> None:
        """Refuse an optimizer step unless every MoE (layer, mb) completed its VJP."""
        _require(
            self._step_complete(),
            f"incomplete oracle step: fwd={len(self._seen_forward)}, "
            f"recompute={len(self._seen_recompute)}, vjp={len(self._seen_vjp)}, "
            f"expected={self._expected_pairs}, pending={len(self._pending)}",
        )

    def end_step(self) -> None:
        """Finalize the current step and drain device copies before next step reuse."""
        if not self._step_active:
            return
        if self._device is not None and self._device.type == "cuda":
            import torch

            torch.cuda.current_stream(self._device).synchronize()
        for state in self._pending.values():
            state.clear()
        self._pending.clear()
        self._records.clear()
        self._steps_completed += 1
        self._step_active = False

    def abort_step(self) -> None:
        """Drop any in-flight replay tensors after an aborted step."""
        for state in self._pending.values():
            state.clear()
        self._pending.clear()
        self._records.clear()
        self._visit = None
        self._visit_calls = 0
        self._step_active = False

    def host_bytes(self) -> Dict[str, int]:
        """Exact owned host tensor bytes and budget reservation."""
        owned = 0
        for tensor in (self._ids, self._scores, self._normalizers):
            if tensor is not None:
                owned += int(tensor.untyped_storage().nbytes())
        return {
            "owned_host_storage_bytes": owned,
            "host_reserved_bytes": self._lease.nbytes if self._lease is not None else owned,
            "pinned": bool(self._pin),
        }

    def report(self) -> Dict[str, Any]:
        """Return JSON-safe telemetry for the oracle."""
        mem = self.host_bytes()
        return {
            "complete": self._step_complete(),
            "expected_pairs": self._expected_pairs,
            "steps_completed": self._steps_completed,
            "step_counts": dict(self._step_counts),
            "total_counts": dict(self._counts),
            "id_dtype": str(self._id_dtype),
            "moe_layers": len(self._bindings),
            "owned_host_storage_bytes": mem["owned_host_storage_bytes"],
            "host_reserved_bytes": mem["host_reserved_bytes"],
            "pinned": mem["pinned"],
            "closed": self._closed,
            "oracle_semantics": "selected-sigmoid-Jacobian-at-original-forward",
        }

    def close(self) -> None:
        """Restore native gate methods, drain stream, drop buffers and release lease."""
        if self._closed:
            return
        try:
            while self._originals:
                gate, had_forward, previous = self._originals.pop()
                if had_forward:
                    gate.forward = previous
                elif "forward" in gate.__dict__:
                    del gate.forward
            if self._device is not None and self._device.type == "cuda" and self._ids is not None:
                import torch

                torch.cuda.current_stream(self._device).synchronize()
            for state in self._pending.values():
                state.clear()
            self._pending.clear()
            self._records.clear()
            self._ids = None
            self._scores = None
            self._normalizers = None
            self._capacity_rows = 0
            self._bindings.clear()
            self._visit = None
            self._step_active = False
            if self._lease is not None:
                self._lease.close()
                self._lease = None
        finally:
            self._closed = True

    def __enter__(self) -> "DeepseekRoutingOracle":
        _require(not self._closed, "oracle is closed")
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()
