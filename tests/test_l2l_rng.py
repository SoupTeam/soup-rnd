"""A2: dropout masks of the L2L step equal an unmodified gradient accumulation, on CPU.

Seams: ``soup_cli.utils.l2l_step.L2LStep.run`` against plain gradient accumulation (no
hooks, one global generator seeded before the step), and ``soup_cli.utils.l2l_rng`` for
the pieces the controls break.

Accumulation draws every mask from one stream that runs micro-batch after micro-batch,
layer after layer. The L2L step replays that stream: it captures which dropouts a layer
call draws (shape, dtype, probability), replays them in the accumulation order to learn the
generator state before each (micro-batch, layer) pair, and runs every layer call, forward
and recompute, at its state. It then leaves the outer stream where accumulation would.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext

import pytest

SEED = 11


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


def ga_reference(model, harness, mbs, *, seed=SEED, windows=1):
    """Plain accumulation, as HF runs it: no hooks, one generator seeded before the step.

    A streamed layer always runs under gradient checkpointing (its weights live in shared
    pool slots), so this is accumulation with gradient checkpointing. ``windows`` repeats
    the step on the same generator, as consecutive optimizer steps do.
    """
    import torch

    from soup_cli.utils import l2l_rng

    torch.manual_seed(seed)
    out = None
    for _ in range(windows):
        harness.zero_grads(model)
        total = harness.label_tokens(mbs)
        losses = []
        for ids in mbs:
            loss = model(input_ids=ids, labels=ids, num_items_in_batch=total).loss
            loss.backward()
            losses.append(loss.detach())
        out = losses, harness.collect_grads(model), l2l_rng.snapshot()
    return out


def l2l_run(model, runtime, harness, mbs, *, seed=SEED, windows=1, store=None):
    import torch

    from soup_cli.utils import l2l_rng
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    stepper = L2LStep(model, runtime, capture_layer_call(model, mbs[0]))
    torch.manual_seed(seed)
    out = None
    for _ in range(windows):
        harness.zero_grads(model)
        losses = stepper.run(mbs, _store(len(mbs)) if store is None else store)
        out = losses, harness.collect_grads(model), l2l_rng.snapshot()
    return out


def _same(harness, left, right, *, stream=True):
    """Losses and gradients equal, and (with ``stream``) both generators left alike."""
    import torch

    same = harness.losses_equal(left[0], right[0]) and harness.compare(left[1], right[1])["equal"]
    if not stream:
        return same
    cpu = torch.equal(left[2].cpu, right[2].cpu)
    cards = all(
        a[0] == b[0] and torch.equal(a[1], b[1]) for a, b in zip(left[2].cuda, right[2].cuda)
    )
    return same and cpu and cards


@pytest.mark.parametrize("dropout", [0.0, 0.05])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_l2l_equals_unmodified_accumulation(make_model, harness, dropout, k):
    from tests.test_l2l_probe_harness import _micro_batches

    model, runtime = make_model(dropout)
    mbs = _micro_batches(k)
    ga = ga_reference(model, harness, mbs)
    l2l = l2l_run(model, runtime, harness, mbs)
    assert _same(harness, ga, l2l)
    assert harness.compare(ga[1], l2l[1])["all_zero"] == []


def test_the_outer_stream_ends_where_accumulation_leaves_it(make_model, harness):
    """Two windows in a row: the second starts from the stream the first one left."""
    from tests.test_l2l_probe_harness import _micro_batches

    model, runtime = make_model(0.05)
    mbs = _micro_batches(2)
    ga = ga_reference(model, harness, mbs, windows=2)
    l2l = l2l_run(model, runtime, harness, mbs, windows=2)
    assert _same(harness, ga, l2l)


def test_dropout_changes_the_gradients(make_model, harness):
    from tests.test_l2l_probe_harness import _micro_batches

    mbs = _micro_batches(2)
    plain = ga_reference(make_model(0.0)[0], harness, mbs)
    dropped = ga_reference(make_model(0.05)[0], harness, mbs)
    assert not _same(harness, plain, dropped, stream=False)


def test_same_seed_repeats_bit_for_bit_and_another_seed_differs(make_model, harness):
    from tests.test_l2l_probe_harness import _micro_batches

    model, runtime = make_model(0.05)
    mbs = _micro_batches(2)
    first = l2l_run(model, runtime, harness, mbs)
    again = l2l_run(model, runtime, harness, mbs)
    other = l2l_run(model, runtime, harness, mbs, seed=SEED + 1)
    assert _same(harness, first, again)
    assert not _same(harness, first, other, stream=False)


# --------------------------------------------------------------------------
# each way of getting the order, the recompute or the outer stream wrong must disagree
# --------------------------------------------------------------------------


def _reference(make_model, harness):
    from tests.test_l2l_probe_harness import _micro_batches

    model, runtime = make_model(0.05)
    mbs = _micro_batches(2)
    return ga_reference(model, harness, mbs), model, runtime, mbs


def test_one_shared_stream_as_in_a1_disagrees_with_accumulation(
    make_model, harness, monkeypatch
):
    ga, model, runtime, mbs = _reference(make_model, harness)
    monkeypatch.setattr("soup_cli.utils.l2l_rng.state_at", lambda state: nullcontext())
    assert not _same(harness, ga, l2l_run(model, runtime, harness, mbs), stream=False)


def test_states_saved_in_layer_major_order_disagree_with_accumulation(
    make_model, harness, monkeypatch
):
    """The recompute sees the forward's state, but the forward order is not the stream's."""
    import torch

    ga, model, runtime, mbs = _reference(make_model, harness)
    saved = []

    @contextmanager
    def layer_major(_state):
        if torch.is_grad_enabled():
            torch.set_rng_state(saved.pop(0))
        else:
            saved.append(torch.get_rng_state())
        yield

    monkeypatch.setattr("soup_cli.utils.l2l_rng.state_at", layer_major)
    assert not _same(harness, ga, l2l_run(model, runtime, harness, mbs), stream=False)


def test_the_state_not_applied_on_the_recompute_disagrees(make_model, harness, monkeypatch):
    import torch

    from soup_cli.utils import l2l_rng

    ga, model, runtime, mbs = _reference(make_model, harness)
    real = l2l_rng.state_at

    def forward_only(state):
        return nullcontext() if torch.is_grad_enabled() else real(state)

    monkeypatch.setattr("soup_cli.utils.l2l_rng.state_at", forward_only)
    assert not _same(harness, ga, l2l_run(model, runtime, harness, mbs), stream=False)


def test_an_outer_stream_left_untouched_disagrees_with_accumulation(
    make_model, harness, monkeypatch
):
    ga, model, runtime, mbs = _reference(make_model, harness)
    monkeypatch.setattr("soup_cli.utils.l2l_rng.leave_stream_at", lambda state: None)
    l2l = l2l_run(model, runtime, harness, mbs)
    assert _same(harness, ga, l2l, stream=False)  # masks are right ...
    assert not _same(harness, ga, l2l)  # ... but the next window would not be


# --------------------------------------------------------------------------
# what cannot be replayed is refused
# --------------------------------------------------------------------------


def test_sdpa_attention_dropout_is_refused_at_capture(make_model):
    from soup_cli.utils.l2l_step import capture_layer_call
    from tests.test_l2l_probe_harness import _micro_batches

    model, _runtime = make_model(0.0)
    for module in model.modules():
        if hasattr(module, "attention_dropout"):
            module.attention_dropout = 0.1
    with pytest.raises(ValueError, match=r"(?i)scaled_dot_product_attention"):
        capture_layer_call(model, _micro_batches(1)[0])


@pytest.mark.parametrize("k", [1, 2])
def test_eager_attention_dropout_equals_unmodified_accumulation(make_model, harness, k):
    from tests.test_l2l_probe_harness import _micro_batches

    model, runtime = make_model(0.05)
    model.config._attn_implementation = "eager"
    for module in model.modules():
        if hasattr(module, "attention_dropout"):
            module.attention_dropout = 0.1
        if hasattr(module, "config") and hasattr(module.config, "_attn_implementation"):
            module.config._attn_implementation = "eager"
    mbs = _micro_batches(k)
    # control: the same model without attention dropout
    plain = ga_reference(make_model(0.05)[0], harness, mbs)
    ga = ga_reference(model, harness, mbs)
    l2l = l2l_run(model, runtime, harness, mbs)
    assert _same(harness, ga, l2l)
    assert not _same(harness, ga, plain, stream=False)
