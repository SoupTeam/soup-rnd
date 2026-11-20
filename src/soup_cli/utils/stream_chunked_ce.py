"""PR #63's streamed loss functions at the L2L SFT head seam."""

from __future__ import annotations

from typing import Any, Sequence

from soup_cli.utils.l2l_step import LossHead, sft_head

CE_CHUNK = 256
IGNORE_INDEX = -100


def _divide(total: Any, shifted: Any, n_items: Any) -> Any:
    import torch

    divisor = (shifted != IGNORE_INDEX).sum() if n_items is None else n_items
    return total / (divisor.to(total.device) if torch.is_tensor(divisor) else divisor)


def chunked_causal_lm_loss(
    hidden: Any, lm_head: Any, labels: Any, n_items: Any = None, *, chunk: int = CE_CHUNK
) -> Any:
    """Recompute each position tile during backward, keeping the head slot live."""
    import torch
    from torch.nn import functional as F  # noqa: N812
    from torch.utils.checkpoint import checkpoint

    if chunk < 1:
        raise ValueError("chunk must be positive")
    shifted = F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:].reshape(-1).to(hidden.device)
    hidden = hidden.reshape(-1, hidden.shape[-1])

    def piece(states: Any, targets: Any) -> Any:
        return F.cross_entropy(
            lm_head(states).float(), targets, ignore_index=IGNORE_INDEX, reduction="sum"
        )

    total = torch.zeros((), device=hidden.device, dtype=torch.float32)
    for start in range(0, shifted.numel(), chunk):
        # An all-ignored tile contributes a differentiable zero. In particular,
        # a fully masked micro-batch in a nonempty window still needs a VJP.
        span = slice(start, start + chunk)
        total = total + checkpoint(piece, hidden[span], shifted[span], use_reentrant=False)
    return _divide(total, shifted, n_items)


def _cce_weight(head: Any) -> Any:
    from torch import nn

    pool, key = getattr(head, "pool", None), getattr(head, "key", None)
    linear = head if pool is None else head.inner
    if type(linear) is not nn.Linear or linear.bias is not None:
        raise ValueError(
            "streamed CCE requires a plain bias-free linear head"
        )
    return linear.weight if pool is None else pool.wait(key)


def cce_causal_lm_loss(hidden: Any, lm_head: Any, labels: Any, n_items: Any = None) -> Any:
    """PR #63's CCE kernel semantics, including the default gradient filtering."""
    import torch
    from torch.nn import functional as F  # noqa: N812

    weight = _cce_weight(lm_head)
    from cut_cross_entropy import linear_cross_entropy

    shifted = F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:].contiguous().to(hidden.device)
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        total = linear_cross_entropy(
            hidden.to(weight.dtype), weight, shifted,
            ignore_index=IGNORE_INDEX, reduction="sum",
        ).float()
    return _divide(total, shifted, n_items)


def l2l_sft_head(
    labels: Sequence[Any], n_items: int, *, impl: str = "chunked", chunk: int = CE_CHUNK
) -> LossHead:
    """Use one window-wide divisor, including with uneven or all-masked micro-batches."""
    if impl == "full":
        return sft_head(labels, n_items)
    if impl not in ("chunked", "cce"):
        raise ValueError(f"unknown L2L SFT loss implementation {impl!r}")

    def head(causal: Any, owner: Any, hidden: Any, mb: int) -> Any:
        hidden = owner.norm(hidden)
        if impl == "cce":
            return cce_causal_lm_loss(hidden, causal.lm_head, labels[mb], n_items)
        return chunked_causal_lm_loss(hidden, causal.lm_head, labels[mb], n_items, chunk=chunk)

    return head
