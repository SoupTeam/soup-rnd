"""C3/A4 integration proposal for RAM-only L2L activations.

An explicit shared allocator wraps the existing store. Spill scheduling and the
public L2L trainer are deliberately not enabled by this experimental bridge.
"""

from typing import Any

from soup_cli.utils.host_memory_budget import HostMemoryBudget
from soup_cli.utils.l2l_activations import ActivationStore


class BudgetedActivationStore(ActivationStore):
    def __init__(
        self,
        *,
        host_budget: HostMemoryBudget,
        n_layers: int,
        k: int,
        chunk_shape: tuple[int, ...],
        dtype: Any,
        device: Any,
        pin: bool,
    ) -> None:
        import math

        import torch

        from soup_cli.utils.layer_stream_runtime import _next_power_of_two

        dimensions = (n_layers, k, *chunk_shape)
        if not chunk_shape or any(type(value) is not int or value < 1 for value in dimensions):
            raise ValueError("activation dimensions must be positive integers")
        chunk_bytes = math.prod(chunk_shape) * torch.empty((), dtype=dtype).element_size()
        block_bytes = _next_power_of_two(chunk_bytes) if pin else chunk_bytes
        self._budget_closed = False
        self._budget_lease = host_budget.reserve("a4.activations", n_layers * k * block_bytes)
        try:
            super().__init__(
                n_layers=n_layers, k=k, chunk_shape=chunk_shape, dtype=dtype, device=device, pin=pin
            )
        except BaseException:
            if hasattr(self, "_resident"):
                self._resident.clear()
            self._budget_lease.close()
            raise

    def _raise_if_failed(self) -> None:
        if self._budget_closed:
            raise RuntimeError("budgeted activation store is closed")
        super()._raise_if_failed()

    def close(self) -> None:
        if self._budget_closed:
            return
        if self.is_cuda:
            import torch

            # Resident activation copies may still be in flight on the device.
            # On failure do not release the accounting lease prematurely.
            torch.cuda.synchronize(self.device)
        super().close()
        self._resident.clear()
        self._budget_lease.close()
        self._budget_closed = True
