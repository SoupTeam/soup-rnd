"""Batch-invariant expert GEMM for the PR #72 MBR bridge (plan A2, harness only).

The routed experts' ``F.linear`` is swapped for one call per row (M=1), forward and backward
(the expert weights are frozen, so autograd only needs the input gradient and differentiates
the per-row calls). The result of a row then no longer depends on how many rows share a call,
which is what grouping micro-batches changes. Only calls made from ``expert_mlp`` are touched:
the router, attention and LoRA layers keep their native GEMMs.

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


@contextmanager
def rowwise_expert_linear() -> Iterator[List[int]]:
    """Swap the experts' ``F.linear`` for one call per row; yields the count of rows served."""
    import torch
    import torch.nn.functional as functional

    original = functional.linear
    rows: List[int] = [0]

    def linear(input: Any, weight: Any, bias: Any = None) -> Any:  # noqa: A002
        if not _called_from_expert_mlp():
            return original(input, weight, bias)
        flat = input.reshape(-1, input.shape[-1])
        rows[0] += int(flat.shape[0])
        out = [original(flat[i : i + 1], weight, bias) for i in range(flat.shape[0])]
        return torch.cat(out).reshape(*input.shape[:-1], -1)

    functional.linear = linear  # type: ignore[assignment]
    try:
        yield rows
    finally:
        functional.linear = original  # type: ignore[assignment]


def expert_gemm_context(mode: str) -> ContextManager[Any]:
    """``native``: no change. ``invariant``: the row-wise expert linear layer."""
    if mode == "native":
        return nullcontext([0])
    if mode == "invariant":
        return rowwise_expert_linear()
    raise ValueError(f"expert gemm must be one of {EXPERT_GEMM_CHOICES}, got {mode!r}")
