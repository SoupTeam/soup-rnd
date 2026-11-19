"""Batch-invariant expert GEMM for the PR #72 MBR bridge (plan A2, harness only).

The routed experts' ``F.linear`` is swapped for one call per row (M=1), forward and backward
(the expert weights are frozen, so autograd only needs the input gradient and differentiates
the per-row calls). The result of a row then no longer depends on how many rows share a call,
which is what grouping micro-batches changes. Only calls made from ``expert_mlp`` are touched:
the router, attention and LoRA layers keep their native GEMMs.

The expert node also multiplies the expert output by the routing weight (``[M, 1]``), and the
gradient of that weight is a sum over the hidden axis whose order CUDA picks from M (found by the
ticket 06 diagnosis: fp32 rows differ between M=1 and M=2). Inside the expert node that multiply is
done one row per call as well.

This alone does not make grouped MBR bit-equal to L2L: LoRA gradients must also add up in the
same order (``L2LStep.run(mbr_split_backward=True)``), and in bf16 autocast the weight cache
must not share a cast across micro-batches. Plan section 6, row "Факт".
"""

from __future__ import annotations

import sys
from contextlib import contextmanager, nullcontext
from typing import Any, ContextManager, Iterator, List

EXPERT_GEMM_CHOICES = ("native", "invariant")
_FRAME_DEPTH = 4  # other patches of F.linear (the phase clock) may sit between us and the caller


def _called_from_expert_mlp() -> bool:
    frame = sys._getframe(2)
    for _ in range(_FRAME_DEPTH):
        if frame is None:
            return False
        if frame.f_code.co_name == "expert_mlp":
            return True
        frame = frame.f_back
    return False


_EXPERT_NODE_MODULE = "soup_cli.utils.expert_stream"
_EXPERT_NODE_FUNCTIONS = ("forward", "backward")


def _called_from_expert_node() -> bool:
    """True when the multiply is written in ``forward``/``backward`` of ``expert_stream``."""
    frame = sys._getframe(2)  # the patched __mul__ is frame 1, its caller frame 2
    return (
        frame.f_globals.get("__name__") == _EXPERT_NODE_MODULE
        and frame.f_code.co_name in _EXPERT_NODE_FUNCTIONS
    )


@contextmanager
def rowwise_expert_linear() -> Iterator[List[int]]:
    """Swap the experts' ``F.linear`` and route-weight multiply for one call per row.

    Yields ``[linear rows, multiply rows]``, the rows served by each swap."""
    import torch
    import torch.nn.functional as functional

    original = functional.linear
    original_mul = torch.Tensor.__mul__
    rows: List[int] = [0, 0]

    def linear(input: Any, weight: Any, bias: Any = None) -> Any:  # noqa: A002
        if not _called_from_expert_mlp():
            return original(input, weight, bias)
        flat = input.reshape(-1, input.shape[-1])
        rows[0] += int(flat.shape[0])
        out = [original(flat[i : i + 1], weight, bias) for i in range(flat.shape[0])]
        return torch.cat(out).reshape(*input.shape[:-1], -1)

    def mul(self: Any, other: Any) -> Any:
        if (
            isinstance(other, torch.Tensor)
            and self.dim() == 2
            and other.dim() == 2
            and other.shape[1] == 1
            and self.shape[0] == other.shape[0]
            and _called_from_expert_node()
        ):
            rows[1] += int(self.shape[0])
            return torch.cat([original_mul(self[i : i + 1], other[i : i + 1])
                              for i in range(self.shape[0])])
        return original_mul(self, other)

    functional.linear = linear  # type: ignore[assignment]
    torch.Tensor.__mul__ = mul  # type: ignore[method-assign]
    try:
        yield rows
    finally:
        functional.linear = original  # type: ignore[assignment]
        torch.Tensor.__mul__ = original_mul  # type: ignore[method-assign]


def expert_gemm_context(mode: str) -> ContextManager[Any]:
    """``native``: no change. ``invariant``: the row-wise expert linear layer."""
    if mode == "native":
        return nullcontext([0, 0])
    if mode == "invariant":
        return rowwise_expert_linear()
    raise ValueError(f"expert gemm must be one of {EXPERT_GEMM_CHOICES}, got {mode!r}")
