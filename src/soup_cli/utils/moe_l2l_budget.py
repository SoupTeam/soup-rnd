"""Host allocation planning for uncompressed MoE L2L windows."""

from dataclasses import dataclass

from soup_cli.utils.budgeted_activations import activation_allocation_bytes
from soup_cli.utils.host_memory_budget import HostMemoryBudget
from soup_cli.utils.layer_stream import dtype_bytes


@dataclass(frozen=True)
class MoEL2LMemoryPlan:
    """Peak host tensor bytes for the selected accumulation window."""

    requested_k: int
    effective_k: int
    available_host_bytes: int
    activation_bytes: int
    routing_record_bytes: int
    routing_comparison_bytes: int

    @property
    def host_bytes(self) -> int:
        return self.activation_bytes + self.routing_record_bytes + self.routing_comparison_bytes


def plan_moe_l2l(
    host_budget: HostMemoryBudget,
    *,
    n_layers: int,
    n_moe_layers: int,
    max_shape: tuple[int, int, int],
    experts_per_token: int,
    requested_k: int,
    dtype: str,
    pin: bool,
) -> MoEL2LMemoryPlan:
    """Select the largest RAM-fitting ``k`` for a fixed upper bound on micro-batch shape.

    ``max_shape`` is ``(batch, sequence, hidden)``. Charge every position at that
    capacity using the C3 resident store's FP32/BF16 allocation calculation. Charge
    pageable int64 routing records for the streamed MoE layers, plus one temporary
    record for the recompute comparison in B1.3.

    The caller accounts for other host consumers and safety headroom in ``host_budget``.
    The result estimates the owned host tensors listed in the plan. Device workspace
    and process memory need separate bounds. Planning reads the available budget once;
    allocation must still acquire leases through ``HostMemoryBudget``.
    """
    dimensions = (n_layers, n_moe_layers, *max_shape, experts_per_token, requested_k)
    if len(max_shape) != 3 or any(type(value) is not int or value < 1 for value in dimensions):
        raise ValueError("MoE L2L needs a positive (batch, sequence, hidden) shape and counts")
    if n_moe_layers > n_layers:
        raise ValueError("MoE layer count must not exceed the total layer count")
    if dtype not in ("float32", "bfloat16"):
        raise ValueError("MoE L2L host planning supports float32 or bfloat16 activations")

    batch_size, sequence_length, hidden_size = max_shape
    tokens = batch_size * sequence_length
    activation_bytes_per_batch = activation_allocation_bytes(
        n_layers=n_layers,
        k=1,
        chunk_bytes=tokens * hidden_size * dtype_bytes(dtype),
        spill_layers=0,
        ring=1,
        pin=pin,
    )
    comparison_bytes = tokens * experts_per_token * 8  # one int64 top-k index tensor
    routing_bytes_per_batch = n_moe_layers * comparison_bytes
    host_bytes_per_batch = activation_bytes_per_batch + routing_bytes_per_batch
    available = host_budget.available_bytes
    effective_k = min(requested_k, (available - comparison_bytes) // host_bytes_per_batch)
    if effective_k < 1:
        raise MemoryError(
            f"MoE L2L k=1 needs {host_bytes_per_batch + comparison_bytes} host bytes "
            f"for activations and routing, but only {available} available"
        )

    return MoEL2LMemoryPlan(
        requested_k=requested_k,
        effective_k=effective_k,
        available_host_bytes=available,
        activation_bytes=effective_k * activation_bytes_per_batch,
        routing_record_bytes=effective_k * routing_bytes_per_batch,
        routing_comparison_bytes=comparison_bytes,
    )
