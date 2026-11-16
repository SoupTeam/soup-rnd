"""ZERO-2: an L2L step stops its backward at the lowest decoder layer that trains.

With ``training.lora_layers`` only some layers' adapters train. Gradient accumulation's
autograd never goes below the lowest trainable layer (nothing there needs a gradient); an
L2L step walked every layer back anyway, re-reading and recomputing each. It now stores no
activation and runs no backward below that layer: the gradients are unchanged and the step
loads fewer layers.
"""
# ruff: noqa: F811
from __future__ import annotations

import pytest

from tests.test_l2l_probe_harness import _load
from tests.test_l2l_qwen35 import _micro_batches, _store, qwen35_streamed  # noqa: F401


def _select(model, layers):
    from soup_cli.utils.layer_select import adapter_params_by_layer, apply_layer_mask

    by_layer = adapter_params_by_layer(model)
    apply_layer_mask(by_layer, layers)
    return by_layer


def _grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None}


@pytest.mark.parametrize("layers", [[2, 3], [3], [1, 3]])
@pytest.mark.parametrize("split", [(), ("cpu",)])
def test_cut_step_equals_accumulation_and_loads_less(qwen35_streamed, layers, split):
    import torch

    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    harness = _load("l2l_schedule")
    model, runtime, _config = qwen35_streamed
    harness.make_non_vacuous(model)
    mbs = _micro_batches(2)
    call = capture_layer_call(model, mbs[0])

    by_layer = _select(model, range(4))
    full = L2LStep(model, runtime, call, split_devices=split)
    harness.zero_grads(model)
    full.run(mbs, _store(2))
    full_loads = full.counters.layer_loads

    _select(model, layers)
    harness.zero_grads(model)
    ga_losses = harness.ga_step(model, mbs)
    ga = _grads(model)
    for p in model.parameters():
        p.grad = None
    step = L2LStep(model, runtime, call, split_devices=split)
    losses = step.run(mbs, _store(2))
    got = _grads(model)
    assert set(got) == set(ga) and got
    for name in ga:
        torch.testing.assert_close(got[name], ga[name], rtol=1e-5, atol=1e-7)
    assert harness.losses_equal(ga_losses, losses)
    # The backward stops at the lowest trainable layer. The pool keeps the last two layers
    # of the forward, so on this 4-layer model the saving shows from a cut at layer 2 up.
    assert step.counters.layer_loads <= full_loads
    if min(layers) >= 2:
        assert step.counters.layer_loads < full_loads
    for idx, params in by_layer.items():
        assert all((p.grad is not None) == (idx in layers) for _n, p in params)


def test_nothing_trainable_in_the_layers_runs_no_layer_backward(qwen35_streamed):
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    model, runtime, _config = qwen35_streamed
    mbs = _micro_batches(2)
    by_layer = _select(model, [0])
    for params in by_layer.values():
        for _n, p in params:
            p.requires_grad_(False)
    step = L2LStep(model, runtime, capture_layer_call(model, mbs[0]))
    losses = step.run(mbs, _store(2))
    assert len(losses) == 2
    assert all(p.grad is None for p in model.parameters())
