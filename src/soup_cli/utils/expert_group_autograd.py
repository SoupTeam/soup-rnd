"""First-order D3 frozen-expert derivatives, shared by tests and opt-in SFT.

Reuses original Soup's reload/recompute design from expert_stream._ExpertMLP at
4a78a7ba74d1b52c820fb156647ef1e78f3105a6. Forward groups weight restoration and,
when supplied, uses the explicit D3 GEMM backend. Backward currently recomputes
one expert at a time with native eager autograd. Only SiLU and first derivatives
are supported. GPU qualification is specific to the caller's source and recipe.
"""

from __future__ import annotations

import os
import threading
from typing import Any


def _identity(path: str) -> tuple:
    stat = os.stat(path)
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


_CLASS = None


def grouped_autograd_experts_class() -> Any:
    """Build the backward module without relaxing B1.2's forward-only API."""
    global _CLASS
    if _CLASS is not None:
        return _CLASS
    import torch
    import torch.nn.functional as functional

    from soup_cli.utils.expert_group_compute import plan_grouped_forward
    from soup_cli.utils.expert_group_gemm import SUPPORTED_SHAPES, require_reference_policy
    from soup_cli.utils.expert_stream import _validate_routing

    class ReloadExpert(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden, computed, owner, expert, identity):
            if computed.requires_grad:
                raise ValueError("precomputed expert output must have no autograd graph")
            ctx.owner = owner
            ctx.expert = expert
            ctx.identity = identity
            ctx.autocast = {
                "enabled": torch.is_autocast_enabled(hidden.device.type),
                "dtype": torch.get_autocast_dtype(hidden.device.type),
                "cache_enabled": torch.is_autocast_cache_enabled(),
            }
            # Save the gathered input, never the reusable weight-slot views.
            ctx.save_for_backward(hidden)
            return computed

        @staticmethod
        def backward(ctx, grad_output):
            if torch.is_grad_enabled():
                raise RuntimeError("grouped expert reload supports first derivatives only")
            if not ctx.needs_input_grad[0]:
                return None, None, None, None, None
            (hidden,) = ctx.saved_tensors
            owner = ctx.owner
            if not owner._lock.acquire(blocking=False):
                raise RuntimeError("expert forward/backward is already active")
            try:
                owner._check_identity(ctx.identity)
                owner.pool.validate_reader(owner.reader)
                if owner.gemm is not None:
                    require_reference_policy()
                row = {"expert": ctx.expert, "status": "loading"}
                owner.backward_trace.append(row)
                try:
                    with owner.pool.lease(owner.reader, [ctx.expert]) as weights:
                        owner._check_identity(ctx.identity)
                        gate_up, down = weights[0]
                        with torch.enable_grad(), torch.autocast(
                            hidden.device.type, **ctx.autocast,
                        ):
                            local = hidden.detach().requires_grad_(True)
                            gate, up = functional.linear(local, gate_up).chunk(2, dim=-1)
                            out = functional.linear(owner.act_fn(gate) * up, down)
                            (grad_hidden,) = torch.autograd.grad(out, local, grad_output)
                    row["status"] = "ok"
                except BaseException:
                    row["status"] = "error"
                    raise
                return grad_hidden, None, None, None, None
            finally:
                owner._lock.release()

    class GroupedAutogradExperts(torch.nn.Module):
        """Frozen-base, first-order SiLU primitive, using a caller-owned group pool.

        Pool/reader/backend must outlive every graph. Closing them before backward
        produces a refusal. Routed expert adapters and stochastic expert activations
        are not supported. Routing weights remain in native autograd outside the
        expert function. The opt-in SFT installer owns the pool/reader lifetime;
        the ordinary B1.2 forward-only installer remains unchanged.
        """

        def __init__(self, reader, pool, act_fn, *, activation_budget_bytes, gemm=None,
                     activation_backend="native"):
            super().__init__()
            from transformers.activations import SiLUActivation

            if type(activation_budget_bytes) is not int or activation_budget_bytes < 1:
                raise ValueError("positive integer activation budget required")
            if not (type(act_fn) in (torch.nn.SiLU, SiLUActivation)
                    or act_fn is functional.silu):
                raise ValueError("development expert derivatives support stateless SiLU only")
            if isinstance(act_fn, torch.nn.SiLU) and act_fn.inplace:
                raise ValueError("in-place SiLU is unsupported")
            pool.validate_reader(reader)
            if activation_backend not in ("native", "triton"):
                raise ValueError("activation backend must be native or triton")
            if activation_backend == "triton" and (
                pool.device.type != "cuda" or pool.dtype != "bfloat16"
            ):
                raise ValueError("Triton activation requires a CUDA BF16 pool")
            if gemm is not None:
                if pool.device.type != "cuda" or pool.dtype != "bfloat16":
                    raise ValueError("grouped GEMM training primitive requires CUDA BF16")
                if gemm.device != pool.device:
                    raise ValueError("GEMM and expert pool devices must match")
                if any((shape[1], shape[0]) not in SUPPORTED_SHAPES
                       for shape in reader.shapes.values()):
                    raise ValueError("unvalidated expert geometry for grouped GEMM")
                require_reference_policy()
            self.reader, self.pool, self.gemm = reader, pool, gemm
            self.act_fn = act_fn
            self.activation_backend = activation_backend
            self.num_experts = reader.count
            self.activation_budget_bytes = activation_budget_bytes
            self._lock = threading.Lock()
            self.forward_trace: list = []
            self.backward_trace: list = []
            self.last_plan = None

        def _check_identity(self, expected):
            if _identity(self.reader.path) != expected:
                raise ValueError("expert storage changed between forward and backward")

        def forward(self, hidden, indices, route_weights):
            if not self._lock.acquire(blocking=False):
                raise RuntimeError("expert forward/backward is already active")
            try:
                self.pool.validate_reader(self.reader)
                if ((self.gemm is not None or self.activation_backend == "triton")
                        and torch.is_autocast_enabled("cuda")
                        and torch.get_autocast_dtype("cuda") != torch.bfloat16):
                    raise ValueError("CUDA grouped/Triton path requires BF16 autocast or none")
                # Reuse shape/dtype checks, without changing the old forward-only API.
                with torch.no_grad():
                    _validate_routing(hidden, indices, route_weights, self.pool)
                if indices.shape[1] > self.num_experts or (
                    indices.numel() and
                    (indices.min().item() < 0 or indices.max().item() >= self.num_experts)
                ):
                    raise ValueError("invalid routing expert IDs or top-k")
                plan = plan_grouped_forward(
                    tokens=hidden.shape[0], top_k=indices.shape[1], experts=self.num_experts,
                    hidden=hidden.shape[1], intermediate=self.reader.shapes["down"][1],
                    width=self.pool.group_width,
                )
                if plan["activation_bytes"] > self.activation_budget_bytes:
                    raise ValueError("expert derivatives exceed the activation budget")
                if self.gemm is not None:
                    require_reference_policy()
                    rows = hidden.shape[0] * indices.shape[1]
                    columns = max(shape[0] for shape in self.reader.shapes.values())
                    bound = rows * columns * 2 + self.pool.group_width * 512 + 1536
                    if bound > self.gemm.call_budget_bytes:
                        raise ValueError("expert derivatives exceed GEMM budget before reads")
                self.last_plan = plan
                self.forward_trace, self.backward_trace = [], []
                identity = _identity(self.reader.path)
                with torch.no_grad():
                    mask = functional.one_hot(indices, self.num_experts).permute(2, 1, 0)
                    active = (mask.sum(dim=(-1, -2)) > 0).nonzero().flatten().tolist()
                output = torch.zeros_like(hidden)
                for start in range(0, len(active), self.pool.group_width):
                    experts = active[start:start + self.pool.group_width]
                    routes = [torch.where(mask[expert]) for expert in experts]
                    # Keep native gather and scatter in autograd, in expert order.
                    gathered = [hidden[token] for _position, token in routes]
                    with self.pool.lease(self.reader, experts) as weights, torch.no_grad():
                        self._check_identity(identity)
                        first = self._linear(gathered, [pair[0] for pair in weights])
                        intermediate = [self._activate(x) for x in first]
                        del first
                        computed = self._linear(intermediate, [pair[1] for pair in weights])
                        del intermediate
                    for expert, state, current, (position, token) in zip(
                        experts, gathered, computed, routes,
                    ):
                        current = ReloadExpert.apply(state, current, self, expert, identity)
                        current = current * route_weights[token, position, None]
                        output.index_add_(0, token, current.to(output.dtype))
                        self.forward_trace.append({"expert": expert, "status": "ok"})
                return output
            finally:
                self._lock.release()

        def _linear(self, inputs, weights):
            if self.gemm is not None:
                # Keep gathered originals for ReloadExpert, but the explicit cuBLAS
                # forward has no graph and must receive non-autograd operand views.
                return self.gemm.linear([hidden.detach() for hidden in inputs], weights)
            return [functional.linear(hidden, weight) for hidden, weight in zip(inputs, weights)]

        def _activate(self, matrix):
            if self.activation_backend == "triton":
                from soup_cli.utils.expert_group_activation import fused_swiglu

                return fused_swiglu(matrix, output_budget_bytes=self.activation_budget_bytes)
            gate, up = matrix.chunk(2, dim=-1)
            return self.act_fn(gate) * up

    _CLASS = GroupedAutogradExperts
    return _CLASS
