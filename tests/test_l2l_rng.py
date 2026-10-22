"""A2 ticket 01: dropout masks of the L2L step equal gradient accumulation's, on CPU.

Seams: ``soup_cli.utils.l2l_rng`` (the state function) and
``soup_cli.utils.l2l_step.L2LStep.run`` against a gradient-accumulation reference that
installs the same function as forward pre-hooks on the layers.

The reference is not a stock HF Trainer. A stock Trainer draws every mask from one CPU
stream that runs micro-batch after micro-batch, layer after layer; a layer-major pass
cannot visit the (micro-batch, layer) pairs in that order. Equality with a stock Trainer
at ``lora.dropout`` > 0 is therefore not promised (see the A2 report, ticket 04).
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from soup_cli.utils.l2l_rng import layer_seed


def test_layer_seed_is_deterministic_and_fits_manual_seed():
    first = layer_seed(7, 3, 2, 1)
    assert first == layer_seed(7, 3, 2, 1)
    assert 0 <= first < 2**63


@pytest.mark.parametrize("position", range(4))
def test_layer_seed_changes_with_each_argument(position):
    base = [7, 3, 2, 1]
    other = list(base)
    other[position] += 1
    assert layer_seed(*base) != layer_seed(*other)


def test_layer_seed_does_not_collide_when_arguments_swap():
    assert layer_seed(7, 1, 2, 3) != layer_seed(7, 2, 1, 3)
    assert layer_seed(7, 1, 2, 3) != layer_seed(7, 1, 3, 2)


# --------------------------------------------------------------------------
# L2L step against gradient accumulation
# --------------------------------------------------------------------------
SEED = 11
STEP = 5


@pytest.fixture()
def make_model(tmp_path):
    from tests.test_l2l_probe_harness import _load, build_streamed

    harness = _load("l2l_schedule")
    runtimes = []

    def make(dropout, device="cpu"):
        root = tmp_path / f"model{len(runtimes)}"
        root.mkdir()
        model, runtime = build_streamed(root, lora_dropout=dropout, device=device)
        runtimes.append(runtime)
        harness.make_non_vacuous(model)
        model.train()
        return model, runtime

    yield make
    for runtime in runtimes:
        runtime.close()


@pytest.fixture()
def harness():
    from tests.test_l2l_probe_harness import _load

    return _load("l2l_schedule")


def _store(k):
    from tests.test_l2l_step import _store as store

    return store(k)


def ga_reference(model, harness, mbs, *, seed=SEED, step=STEP, inside_checkpoint=False):
    """Gradient accumulation with the state function installed as a pre-hook on every layer.

    The hook sets the generator from ``layer_seed(seed, step, mb, layer)`` before each
    layer call, which is what the L2L step does around its own calls. A streamed layer
    always runs under gradient checkpointing here (its weights live in shared pool slots),
    so this reference is the "accumulation with gradient checkpointing" case. The hook sits
    on the wrapper (outside the checkpoint) or, with ``inside_checkpoint``, on the inner
    layer, where the recompute in the backward fires it a second time.
    """
    from soup_cli.utils.l2l_rng import set_layer_state
    from soup_cli.utils.layer_stream_runtime import decoder_owner

    layers = list(decoder_owner(model).layers)
    current = {"mb": 0}
    handles = [
        (layer.inner if inside_checkpoint else layer).register_forward_pre_hook(
            lambda _module, _args, idx=idx: set_layer_state(seed, step, current["mb"], idx)
        )
        for idx, layer in enumerate(layers)
    ]
    for layer in layers:
        assert layer.use_checkpoint
    harness.zero_grads(model)
    try:
        total = harness.label_tokens(mbs)
        losses = []
        for mb, ids in enumerate(mbs):
            current["mb"] = mb
            out = model(input_ids=ids, labels=ids, num_items_in_batch=total)
            out.loss.backward()
            losses.append(out.loss.detach())
        return losses, harness.collect_grads(model)
    finally:
        for handle in handles:
            handle.remove()


def l2l_run(model, runtime, harness, mbs, *, seed=SEED, step=STEP, store=None):
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    stepper = L2LStep(model, runtime, capture_layer_call(model, mbs[0]), seed=seed)
    harness.zero_grads(model)
    losses = stepper.run(mbs, _store(len(mbs)) if store is None else store, step=step)
    return losses, harness.collect_grads(model)


def _same(harness, left, right):
    return harness.losses_equal(left[0], right[0]) and harness.compare(left[1], right[1])["equal"]


@pytest.mark.parametrize("dropout", [0.0, 0.05])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_l2l_equals_accumulation(make_model, harness, dropout, k):
    from tests.test_l2l_probe_harness import _micro_batches

    model, runtime = make_model(dropout)
    mbs = _micro_batches(k)
    ga = ga_reference(model, harness, mbs)
    l2l = l2l_run(model, runtime, harness, mbs)
    assert _same(harness, ga, l2l)
    assert harness.compare(ga[1], l2l[1])["all_zero"] == []


@pytest.mark.parametrize("k", [1, 2, 4])
def test_l2l_equals_accumulation_when_the_hook_is_recomputed_inside_checkpoint(
    make_model, harness, k
):
    from tests.test_l2l_probe_harness import _micro_batches

    model, runtime = make_model(0.05)
    mbs = _micro_batches(k)
    ga = ga_reference(model, harness, mbs, inside_checkpoint=True)
    l2l = l2l_run(model, runtime, harness, mbs)
    assert _same(harness, ga, l2l)


def test_dropout_changes_the_gradients(make_model, harness):
    from tests.test_l2l_probe_harness import _micro_batches

    mbs = _micro_batches(2)
    plain = ga_reference(make_model(0.0)[0], harness, mbs)
    dropped = ga_reference(make_model(0.05)[0], harness, mbs)
    assert not _same(harness, plain, dropped)


def test_same_seed_repeats_bit_for_bit_and_another_seed_differs(make_model, harness):
    from tests.test_l2l_probe_harness import _micro_batches

    model, runtime = make_model(0.05)
    mbs = _micro_batches(2)
    first = l2l_run(model, runtime, harness, mbs)
    again = l2l_run(model, runtime, harness, mbs)
    other = l2l_run(model, runtime, harness, mbs, seed=SEED + 1)
    later_step = l2l_run(model, runtime, harness, mbs, step=STEP + 1)
    assert _same(harness, first, again)
    assert not _same(harness, first, other)
    assert not _same(harness, first, later_step)


def test_the_step_leaves_the_outside_generator_where_it_was(make_model, harness):
    import torch

    from tests.test_l2l_probe_harness import _micro_batches

    model, runtime = make_model(0.05)
    torch.manual_seed(1234)
    before = torch.get_rng_state()
    l2l_run(model, runtime, harness, _micro_batches(2))
    assert torch.equal(before, torch.get_rng_state())


# --------------------------------------------------------------------------
# each way of getting the order or the recompute wrong must disagree with accumulation
# --------------------------------------------------------------------------


def _patch_layer_rng(monkeypatch, replacement):
    monkeypatch.setattr("soup_cli.utils.l2l_rng.layer_rng", replacement)


def _disagrees_with_accumulation(make_model, harness):
    from tests.test_l2l_probe_harness import _micro_batches

    model, runtime = make_model(0.05)
    mbs = _micro_batches(2)
    ga = ga_reference(model, harness, mbs)
    return ga, model, runtime, mbs


def test_one_shared_stream_as_in_a1_disagrees_with_accumulation(
    make_model, harness, monkeypatch
):
    import torch

    ga, model, runtime, mbs = _disagrees_with_accumulation(make_model, harness)

    @contextmanager
    def shared_stream(seed, step, mb, layer):
        yield

    _patch_layer_rng(monkeypatch, shared_stream)
    torch.manual_seed(SEED)
    assert not _same(harness, ga, l2l_run(model, runtime, harness, mbs))


def test_states_saved_and_restored_but_forward_in_layer_major_order_disagrees(
    make_model, harness, monkeypatch
):
    import torch

    ga, model, runtime, mbs = _disagrees_with_accumulation(make_model, harness)
    saved = {}

    @contextmanager
    def save_then_restore(seed, step, mb, layer):
        if torch.is_grad_enabled():
            torch.set_rng_state(saved[(mb, layer)])
        else:
            saved[(mb, layer)] = torch.get_rng_state()
        yield

    _patch_layer_rng(monkeypatch, save_then_restore)
    torch.manual_seed(SEED)
    assert not _same(harness, ga, l2l_run(model, runtime, harness, mbs))


def test_the_function_not_called_on_the_recompute_disagrees(make_model, harness, monkeypatch):
    import torch

    from soup_cli.utils.l2l_rng import layer_rng

    ga, model, runtime, mbs = _disagrees_with_accumulation(make_model, harness)

    @contextmanager
    def forward_only(seed, step, mb, layer):
        if torch.is_grad_enabled():
            yield
        else:
            with layer_rng(seed, step, mb, layer):
                yield

    _patch_layer_rng(monkeypatch, forward_only)
    assert not _same(harness, ga, l2l_run(model, runtime, harness, mbs))


def _l2l_layer_under_torch_checkpoint(preserve_rng_state):
    """An ``L2LStep._layer`` that recomputes the layer in the backward with torch's checkpoint.

    The state is set around the call only, so the recompute (which autograd runs later,
    outside it) depends on ``preserve_rng_state`` to see the forward's masks.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    from soup_cli.utils.l2l_rng import layer_rng

    def layer_call(self, module, hidden, layer, mb, step):
        with layer_rng(self.seed, step, mb, layer):
            if torch.is_grad_enabled():
                out = checkpoint(
                    module._body,
                    hidden,
                    *self.call.args,
                    use_reentrant=False,
                    preserve_rng_state=preserve_rng_state,
                    **self.call.kwargs,
                )
            else:
                out = module(hidden, *self.call.args, **self.call.kwargs)
        return out[0] if isinstance(out, tuple) else out

    return layer_call


def test_l2l_recomputed_by_torch_checkpoint_matches_only_with_rng_preservation(
    make_model, harness, monkeypatch
):
    from soup_cli.utils.l2l_step import L2LStep
    from tests.test_l2l_probe_harness import _micro_batches

    model, runtime = make_model(0.05)
    mbs = _micro_batches(2)
    ga = ga_reference(model, harness, mbs)

    monkeypatch.setattr(L2LStep, "_layer", _l2l_layer_under_torch_checkpoint(True))
    assert _same(harness, ga, l2l_run(model, runtime, harness, mbs))  # the mutant is sound

    monkeypatch.setattr(L2LStep, "_layer", _l2l_layer_under_torch_checkpoint(False))
    assert not _same(harness, ga, l2l_run(model, runtime, harness, mbs))


# --------------------------------------------------------------------------
# which generators the call forks (torch.manual_seed seeds every initialised CUDA card)
# --------------------------------------------------------------------------


def test_layer_rng_forks_every_initialised_cuda_card(monkeypatch):
    import torch

    from soup_cli.utils.l2l_rng import layer_rng

    forked = []

    @contextmanager
    def record(devices=None, **_kwargs):
        forked.append(list(devices))
        yield

    monkeypatch.setattr(torch.random, "fork_rng", record)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with layer_rng(SEED, STEP, 0, 0):
        pass
    assert forked == [[0, 1]]


def test_layer_rng_forks_no_card_while_cuda_is_uninitialised(monkeypatch):
    import torch

    from soup_cli.utils.l2l_rng import layer_rng

    forked = []

    @contextmanager
    def record(devices=None, **_kwargs):
        forked.append(list(devices))
        yield

    monkeypatch.setattr(torch.random, "fork_rng", record)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    with layer_rng(SEED, STEP, 0, 0):
        pass
    assert forked == [[]]
