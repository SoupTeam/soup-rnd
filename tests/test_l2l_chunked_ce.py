"""ZERO: training.l2l with training.stream_chunked_ce.

The L2L window's head computes the chunked causal-LM loss (the logits of one chunk of
positions at a time, recomputed in backward) instead of the model's full-logits loss, so a
248k-vocabulary model never holds one micro-batch's logits. An L2L step with the chunked
head must equal gradient accumulation with the same chunked loss, on the hybrid qwen3_5
decoder, and stay close to the full-logits loss.
"""
# ruff: noqa: F811  (the qwen35_streamed fixture is imported and taken as an argument)
from __future__ import annotations

import pytest

from tests.test_l2l_probe_harness import _load
from tests.test_l2l_qwen35 import _micro_batches, _store, qwen35_streamed  # noqa: F401


def _ga_chunked(model, micro_batches, chunk):
    """Gradient accumulation through the streamed forward with the chunked loss."""
    from soup_cli.utils.l2l_step import label_tokens, model_parts
    from soup_cli.utils.stream_chunked_ce import chunked_causal_lm_loss

    causal, owner = model_parts(model)
    total = label_tokens(micro_batches)
    losses = []
    for ids in micro_batches:
        hidden = owner(input_ids=ids).last_hidden_state
        loss = chunked_causal_lm_loss(hidden, causal.lm_head, ids, total, chunk=chunk)
        loss.backward()
        losses.append(loss.detach())
    return losses


def _l2l(model, runtime, micro_batches, head):
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    step = L2LStep(model, runtime, capture_layer_call(model, micro_batches[0]))
    return step.run(micro_batches, _store(len(micro_batches)), head=head)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_l2l_chunked_equals_accumulation_with_chunked_loss(qwen35_streamed, k):
    from soup_cli.utils.l2l_step import label_tokens, sft_chunked_head

    harness = _load("l2l_schedule")
    model, runtime, _config = qwen35_streamed
    harness.make_non_vacuous(model)
    mbs = _micro_batches(k)
    assert mbs[0].shape[1] > 2 * 3  # several chunks of 3 positions per row

    harness.zero_grads(model)
    ga_losses = _ga_chunked(model, mbs, chunk=3)
    ga = harness.collect_grads(model)

    harness.zero_grads(model)
    n_items = label_tokens(mbs)
    l2l_losses = _l2l(model, runtime, mbs, sft_chunked_head(mbs, n_items, chunk=3))
    l2l = harness.collect_grads(model)

    result = harness.compare(ga, l2l)
    assert result["all_zero"] == []
    assert result["equal"], result
    assert harness.losses_equal(ga_losses, l2l_losses)


def test_l2l_chunked_head_matches_full_logits_head(qwen35_streamed):
    import torch

    from soup_cli.utils.l2l_step import label_tokens, sft_chunked_head, sft_head

    harness = _load("l2l_schedule")
    model, runtime, _config = qwen35_streamed
    harness.make_non_vacuous(model)
    mbs = _micro_batches(2)
    n_items = label_tokens(mbs)

    harness.zero_grads(model)
    full_losses = _l2l(model, runtime, mbs, sft_head(mbs, n_items))
    full = harness.collect_grads(model)
    harness.zero_grads(model)
    chunked_losses = _l2l(model, runtime, mbs, sft_chunked_head(mbs, n_items, chunk=3))
    chunked = harness.collect_grads(model)

    for a, b in zip(full_losses, chunked_losses):
        torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-6)
    for name in full:
        torch.testing.assert_close(chunked[name], full[name], rtol=1e-4, atol=1e-6)


def test_trainer_picks_the_chunked_window_loss():
    from soup_cli.trainer.l2l_trainer import SFT_CHUNKED_WINDOW_LOSS, SFT_WINDOW_LOSS
    from soup_cli.utils.l2l_step import sft_chunked_head, sft_head

    assert SFT_CHUNKED_WINDOW_LOSS.make_head is sft_chunked_head
    assert SFT_WINDOW_LOSS.make_head is sft_head
    assert SFT_CHUNKED_WINDOW_LOSS.divide_by == SFT_WINDOW_LOSS.divide_by == "items"
