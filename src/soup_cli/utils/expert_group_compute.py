"""Internal D3 grouped restore and hybrid grouped GEMM; no trainer integration."""

from __future__ import annotations

from typing import Any


def plan_grouped_forward(
    *, tokens: int, top_k: int, experts: int, hidden: int, intermediate: int, width: int,
) -> dict:
    """Conservative live tensor storage, NOT CUDA contexts/allocator reservations.

    Uses all token routes as the per-group bound, even with repeated top-k IDs.
    Charges four bytes per temporary element although compute is BF16; includes
    routing masks/indices, output, pointer bins and per-tensor allocation padding.
    Weights and the separately declared cuBLAS workspace are not activation RAM.
    """
    if (type(tokens) is not int or tokens < 0
            or any(type(value) is not int or value < 1
                   for value in (top_k, experts, hidden, intermediate, width))
            or top_k > experts or width > 4):
        raise ValueError("invalid forward workspace dimensions")
    routes = tokens * top_k
    output = (tokens * hidden * 2 + 511) // 512 * 512
    routing = routes * experts * 8 + routes * 16 + experts * 32
    temporaries = routes * (3 * hidden + 4 * intermediate) * 4
    padding_and_pointers = width * 8192
    return {"output_bytes": output, "routing_bytes": routing,
            "temporary_bytes": temporaries, "padding_and_pointers": padding_and_pointers,
            "activation_bytes": output + routing + temporaries + padding_and_pointers,
            "scope": "conservative tensor storage; weights, BLAS workspace and contexts separate"}


_COMPUTE_CLASS = None


def grouped_compute_experts_class() -> Any:
    global _COMPUTE_CLASS
    if _COMPUTE_CLASS is not None:
        return _COMPUTE_CLASS
    import torch

    from soup_cli.utils.expert_group import grouped_experts_class
    from soup_cli.utils.expert_group_gemm import (
        SUPPORTED_SHAPES,
        require_reference_policy,
    )

    class GroupedComputeExperts(grouped_experts_class()):
        """Each group retains its weight lease across BOTH GEMMs and ordered scatter."""

        def __init__(self, reader, pool, act_fn, layer_idx, *, gemm, activation_budget_bytes):
            if type(activation_budget_bytes) is not int or activation_budget_bytes < 1:
                raise ValueError("positive integer activation budget required")
            if pool.device.type != "cuda" or pool.dtype != "bfloat16" or gemm.device != pool.device:
                raise ValueError("grouped compute requires a matching CUDA BF16 pool/backend")
            if any((shape[1], shape[0]) not in SUPPORTED_SHAPES
                   for shape in reader.shapes.values()):
                raise ValueError("unvalidated expert geometry for grouped GEMM")
            require_reference_policy()
            super().__init__(reader, pool, act_fn, layer_idx)
            self.gemm = gemm
            self.activation_budget_bytes = activation_budget_bytes
            self.last_workspace_plan = None

        def forward(self, hidden_states, top_k_index, top_k_weights):
            require_reference_policy()
            if hidden_states.ndim != 2 or top_k_index.ndim != 2:
                raise ValueError("hidden and routing inputs must be two-dimensional")
            if top_k_weights.dtype != torch.bfloat16:
                raise ValueError("grouped compute recipe requires BF16 routing weights")
            plan = plan_grouped_forward(
                tokens=hidden_states.shape[0], top_k=top_k_index.shape[1], experts=self.num_experts,
                hidden=self.pool._shapes["gate_up"][1],
                intermediate=self.pool._shapes["down"][1], width=self.pool.group_width,
            )
            if plan["activation_bytes"] > self.activation_budget_bytes:
                raise ValueError("grouped forward exceeds declared activation budget")
            # All token routes in one group is conservative even for repeated IDs.
            rows = hidden_states.shape[0] * top_k_index.shape[1]
            largest_columns = max(shape[0] for shape in self.pool._shapes.values())
            call_bound = rows * largest_columns * 2 + self.pool.group_width * 512 + 1536
            if call_bound > self.gemm.call_budget_bytes:
                raise ValueError("grouped forward exceeds GEMM call budget before expert reads")
            self.last_workspace_plan = plan
            return super().forward(hidden_states, top_k_index, top_k_weights)

        def _compute_group(self, weights, routes, rows, hidden_states, top_k_weights, output):
            gathered = [hidden_states[token] for _position, token in routes]
            first = self.gemm.linear(gathered, [pair[0] for pair in weights])

            def activate(matrix):
                gate, up = matrix.chunk(2, dim=-1)
                return self.act_fn(gate) * up

            intermediate = [activate(matrix) for matrix in first]
            del first, gathered
            second = self.gemm.linear(intermediate, [pair[1] for pair in weights])
            for current, (position, token), row in zip(second, routes, rows):
                current = current * top_k_weights[token, position, None]
                output.index_add_(0, token, current.to(output.dtype))
                row["status"] = "computed"

    _COMPUTE_CLASS = GroupedComputeExperts
    return _COMPUTE_CLASS
