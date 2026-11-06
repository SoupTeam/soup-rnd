"""Static C3/L2L host partition for boundary activations and routing metadata.

Reserve the largest configured window before optional weight caching.
The activation allocator can reuse its partition across shapes, but never borrow
from the weights. This bounds owned allocations, not process RSS or device RAM.
"""

from dataclasses import dataclass

from soup_cli.utils.host_memory_budget import HostMemoryBudget
from soup_cli.utils.partial_ram_source import PartialRamBudget

#: Sequence rounding of the reserved window. L2L no longer pads micro-batches to a
#: bucket, so this only over-reserves slightly; it keeps the CUDA-tested reservation.
RESERVATION_ROUND = 64


@dataclass(frozen=True)
class C3L2LPlan:
    weights: PartialRamBudget
    activations: HostMemoryBudget
    n_layers: int
    max_shape: tuple[int, int, int]
    max_microbatches: int
    dtype: str
    pinned: bool
    codec: str = "none"
    device_type: str = "cuda"
    routing_bytes: int = 0

    def validate_window(
        self, *, n_layers: int, shape: tuple[int, int, int], k: int,
        dtype: str, pin: bool, codec: str = "none", device_type: str | None = None,
    ) -> None:
        dimensions = (n_layers, k, *shape)
        if len(shape) != 3 or any(type(x) is not int or x < 1 for x in dimensions):
            raise ValueError("C3 L2L window dimensions must be positive integers")
        if (
            n_layers != self.n_layers
            or shape[2] != self.max_shape[2]
            or any(x > cap for x, cap in zip(shape, self.max_shape))
            or k > self.max_microbatches
            or dtype != self.dtype
            or pin != self.pinned
            or codec != self.codec
            or (
                codec != "none" and device_type is not None
                and device_type != self.device_type
            )
        ):
            raise ValueError("C3 L2L window exceeds or differs from the reserved activation plan")

    def validate_microbatches(
        self, *, n_layers: int, shapes: list[tuple[int, int, int]],
        dtype: str, pin: bool, codec: str = "none", device_type: str | None = None,
    ) -> None:
        """Check a window whose micro-batches may differ in shape (rows of other lengths)."""
        if not shapes:
            raise ValueError("C3 L2L window needs at least one micro-batch")
        for shape in shapes:
            self.validate_window(
                n_layers=n_layers, shape=shape, k=len(shapes), dtype=dtype, pin=pin,
                codec=codec, device_type=device_type,
            )


def plan_c3_l2l(
    weights: PartialRamBudget,
    *,
    n_layers: int,
    batch_size: int,
    max_length: int,
    hidden_size: int,
    microbatches: int,
    dtype: str,
    pin: bool,
    shape_round: int,
    codec: str = "none",
    device_type: str = "cuda",
    routing_bytes: int = 0,
) -> C3L2LPlan:
    """Partition once, before weight allocation; charge actual pinned rounding.

    Reserve rounded maximum length even when the initial window needs no padding.
    A later mixed-length window therefore cannot consume optional weight-cache RAM.
    Other consumers remain covered by the caller's existing fixed reservation.
    ``routing_bytes`` comes from an already-resolved MoE plan; this partitioner
    does not select accumulation k or recompute routing requirements.
    """
    from soup_cli.utils.budgeted_activations import activation_allocation_bytes
    from soup_cli.utils.layer_stream import dtype_bytes

    dimensions = (n_layers, batch_size, max_length, hidden_size, microbatches, shape_round)
    if any(type(x) is not int or x < 1 for x in dimensions) or type(pin) is not bool:
        raise ValueError("C3 L2L needs fixed positive dimensions and boolean pin")
    if dtype not in ("float32", "bfloat16"):
        raise ValueError("C3 L2L supports float32 or bfloat16 activations")
    if codec not in ("none", "fp8_e4m3"):
        raise ValueError("C3 L2L codec must be none or fp8_e4m3")
    if type(routing_bytes) is not int or routing_bytes < 0:
        raise ValueError("C3 L2L routing_bytes must be a nonnegative integer")
    sequence = -(-max_length // shape_round) * shape_round
    if codec == "fp8_e4m3":
        from soup_cli.utils.fp8_activations import fp8_activation_allocation

        allocation = fp8_activation_allocation(
            n_layers=n_layers,
            mb_shapes=[(batch_size, sequence, hidden_size)] * microbatches,
            pin=pin, compute_dtype=dtype, device_type=device_type,
        ).host_bytes
    else:
        chunk_bytes = batch_size * sequence * hidden_size * dtype_bytes(dtype)
        allocation = activation_allocation_bytes(
            n_layers=n_layers,
            k=microbatches,
            chunk_bytes=chunk_bytes,
            spill_layers=0,
            ring=1,
            pin=pin,
        )
    allocation += routing_bytes
    partition = PartialRamBudget(
        weights.limit_bytes, weights.reserved_bytes + allocation, weights.cache_bytes
    )
    return C3L2LPlan(
        partition,
        HostMemoryBudget(allocation),
        n_layers,
        (batch_size, sequence, hidden_size),
        microbatches,
        dtype,
        pin,
        codec,
        device_type,
        routing_bytes,
    )
