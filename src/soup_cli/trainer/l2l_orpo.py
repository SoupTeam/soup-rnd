"""ORPO window loss for L2L (ticket A6-02).

TRL's ``ORPOTrainer`` builds one micro-batch from separate tensors ``chosen_*`` and
``rejected_*`` (right-padded) and has no ``input_ids``. Its ``concatenated_forward`` stacks
chosen over rejected, pads both to the longer width and runs the policy once. The loss is

    NLL - mean(beta * logsigmoid(log_odds))

* ``NLL`` is a cross entropy over the whole chosen sequence, prompt included, averaged over
  the tokens of the micro-batch (``orpo_trainer.py`` ``concatenated_forward``).
* ``log_odds`` comes from the per-row mean log-probability of the response tokens.

There is no reference model, so the layers see the policy's ``[chosen; rejected]`` batch
and nothing else. HF divides a preference loss by ``k = len(window)`` (``divide_by``
``"micro_batches"``); the head returns the loss already divided.

The layers run without an attention mask, as for SFT. With right padding and causal
attention, the output at a real position does not depend on the padding after it, and the
loss ignores padded positions. TRL's own static methods build the micro-batch and the
log-probabilities, so the L2L loss follows the installed TRL.

torch is imported inside functions.
"""

from __future__ import annotations

from typing import Any, List, Tuple

from soup_cli.trainer.l2l_trainer import (
    WindowLoss,
    refuse_mixed_shapes,
    require_right_padded,
)
from soup_cli.utils.l2l_step import LossHead


def _orpo_trainer_class() -> Any:
    from soup_cli.trainer._trl_compat import resolve_trl_symbol

    return resolve_trl_symbol("ORPOTrainer", "trl.experimental.orpo")


def _log1mexp(value: Any) -> Any:
    """``log(1 - exp(value))``, branched at -ln 2 as TRL does."""
    import torch

    threshold = -0.6931471805599453
    return torch.where(
        value < threshold, torch.log1p(-torch.exp(value)), torch.log(-torch.expm1(value))
    )


def orpo_loss(
    logits: Any, labels: Any, nll_labels: Any, len_chosen: int, beta: float, trainer_cls: Any
) -> Any:
    """TRL's ORPO loss of one micro-batch: ``NLL - mean(beta * logsigmoid(log_odds))``.

    ``logits`` are for the ``2 * len_chosen`` stacked rows; ``labels`` mask the prompt
    (``-100``), ``nll_labels`` mask only padding.
    """
    import torch
    import torch.nn.functional as functional

    chosen = logits[:len_chosen][..., :-1, :].contiguous()
    shifted = nll_labels[:len_chosen][..., 1:].contiguous()
    nll = torch.nn.CrossEntropyLoss()(chosen.view(-1, chosen.shape[-1]), shifted.view(-1))

    all_logps = trainer_cls.get_batch_logps(logits, labels, average_log_prob=True)
    chosen_logps = all_logps[:len_chosen].float()
    rejected_logps = all_logps[len_chosen:].float()
    log_odds = (chosen_logps - rejected_logps) - (
        _log1mexp(chosen_logps) - _log1mexp(rejected_logps)
    )
    return nll - (beta * functional.logsigmoid(log_odds)).mean()


def orpo_window_loss(*, beta: float, padding_value: int) -> WindowLoss:
    """The window loss of ``task: orpo``; ``padding_value`` is the tokenizer's pad id."""

    def read_window(
        window: List[Any], device: Any, *, neftune: bool = False, has_dropout: bool = False
    ) -> Tuple[List[Any], List[Tuple[Any, Any, int]]]:
        import torch

        trainer_cls = _orpo_trainer_class()
        ids_list: List[Any] = []
        extras: List[Tuple[Any, Any, int]] = []
        for inputs in window:
            moved = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in inputs.items()
            }
            stacked = trainer_cls.concatenated_inputs(
                moved, is_encoder_decoder=False, padding_value=padding_value, device=device
            )
            ids = stacked["concatenated_input_ids"]
            mask = stacked["concatenated_attention_mask"]
            require_right_padded(mask)
            nll_labels = torch.where(mask == 1, ids, -100)
            ids_list.append(ids)
            extras.append(
                (stacked["concatenated_labels"], nll_labels, int(moved["chosen_labels"].shape[0]))
            )
        refuse_mixed_shapes(ids_list, neftune=neftune, has_dropout=has_dropout)
        return ids_list, extras

    def make_head(extras: List[Tuple[Any, Any, int]], divisor: int) -> LossHead:
        trainer_cls = _orpo_trainer_class()

        def head(causal: Any, owner: Any, hidden: Any, mb: int) -> Any:
            logits = causal.lm_head(owner.norm(hidden)[:, slice(0, None), :])
            labels, nll_labels, len_chosen = extras[mb]
            loss = orpo_loss(logits, labels, nll_labels, len_chosen, beta, trainer_cls)
            return loss / divisor

        return head

    return WindowLoss(divide_by="micro_batches", read_window=read_window, make_head=make_head)
