"""G1 on CPU: the shipped L2L step equals gradient accumulation bit for bit.

The seam is ``soup_cli.utils.l2l_step.L2LStep.run`` plus the shipped
``ActivationStore``. The gradient-accumulation reference and the comparison
helpers come from the A0 harness, which is not shipped.
"""

from __future__ import annotations

import pytest

from tests.test_l2l_probe_harness import SEQ, _load, _micro_batches, streamed  # noqa: F401


@pytest.fixture()
def harness():
    return _load("l2l_schedule")


def _store(k, n_layers=3, hidden=64):
    import torch

    from soup_cli.utils.l2l_activations import ActivationStore

    return ActivationStore(
        n_layers=n_layers,
        k=k,
        chunk_shape=(1, SEQ, hidden),
        dtype=torch.float32,
        device="cpu",
        pin=False,
    )


@pytest.mark.parametrize("k", [1, 2, 4])
def test_shipped_l2l_step_equals_gradient_accumulation(harness, streamed, k):  # noqa: F811
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    model, runtime = streamed
    harness.make_non_vacuous(model)
    mbs = _micro_batches(k)

    harness.zero_grads(model)
    ga_losses = harness.ga_step(model, mbs)
    ga = harness.collect_grads(model)

    step = L2LStep(model, runtime, capture_layer_call(model, mbs[0]))
    harness.zero_grads(model)
    l2l_losses = step.run(mbs, _store(k))
    l2l = harness.collect_grads(model)

    result = harness.compare(ga, l2l)
    assert result["tensors"] == 3 * 4 * 2  # 3 layers x q/k/v/o x A/B
    assert result["all_zero"] == []
    assert result["equal"], result
    assert harness.losses_equal(ga_losses, l2l_losses)


def test_harness_reexports_the_shipped_core(harness):
    from soup_cli.utils import l2l_step

    assert harness.L2LStep is l2l_step.L2LStep
    assert harness.LayerCall is l2l_step.LayerCall
