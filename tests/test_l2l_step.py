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


# --------------------------------------------------------------------------
# ticket 02: layer loads and weight restores, read from the shipped counters
# --------------------------------------------------------------------------
def _warm_walk(model, ids):
    """Loads depend on what the previous walk left in the slots; start every
    measurement after the same forward-only walk."""
    from soup_cli.utils.l2l_step import capture_layer_call

    capture_layer_call(model, ids)


def _l2l_counters(harness, model, runtime, k, cold=None):
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    mbs = _micro_batches(k)
    step = L2LStep(model, runtime, capture_layer_call(model, mbs[0]))
    harness.zero_grads(model)
    _warm_walk(model, mbs[0])
    if cold:
        cold(model)
    step.run(mbs, _store(k))
    return step.counters


def _ga_counters(harness, model, runtime, k, cold=None):
    from soup_cli.utils.l2l_step import count_loads

    mbs = _micro_batches(k)
    harness.zero_grads(model)
    _warm_walk(model, mbs[0])
    if cold:
        cold(model)
    with count_loads(runtime) as counters:
        harness.ga_step(model, mbs)
    return counters


def test_stats_exposes_the_restore_counter(streamed):  # noqa: F811
    _model, runtime = streamed
    assert runtime.stats()["layer_restores"] == 0


def test_count_loads_reads_the_runtime_counters_without_patching(streamed):  # noqa: F811
    from soup_cli.utils.l2l_step import count_loads

    model, runtime = streamed
    load_async = runtime.pool.load_async
    with count_loads(runtime) as counters:
        _warm_walk(model, _micro_batches(1)[0])
    assert runtime.pool.load_async == load_async
    assert counters.layer_loads > 0


def test_l2l_counters_exist_after_a_step_and_are_zero_before(harness, streamed):  # noqa: F811
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    model, runtime = streamed
    mbs = _micro_batches(1)
    step = L2LStep(model, runtime, capture_layer_call(model, mbs[0]))
    assert step.counters is None
    step.run(mbs, _store(1))
    assert step.counters.layer_loads > 0


def test_k1_l2l_loads_what_gradient_accumulation_loads(harness, streamed):  # noqa: F811
    model, runtime = streamed
    harness.make_non_vacuous(model)
    ga = _ga_counters(harness, model, runtime, 1)
    l2l = _l2l_counters(harness, model, runtime, 1)
    assert l2l.layer_loads == ga.layer_loads
    assert l2l.layer_loads > 0


def test_k4_does_not_load_four_times_what_k1_loads(harness, streamed):  # noqa: F811
    model, runtime = streamed
    harness.make_non_vacuous(model)
    k1 = _l2l_counters(harness, model, runtime, 1)
    k4 = _l2l_counters(harness, model, runtime, 4)
    assert k4.layer_loads == k1.layer_loads
    ga4 = _ga_counters(harness, model, runtime, 4)
    assert ga4.layer_loads > k4.layer_loads  # accumulation reloads per micro-batch


def test_restores_are_zero_without_quantization(harness, streamed):  # noqa: F811
    model, runtime = streamed
    harness.make_non_vacuous(model)
    for k in (1, 2, 4):
        assert _l2l_counters(harness, model, runtime, k).restores == 0
        assert _ga_counters(harness, model, runtime, k).restores == 0


def _cold_views(model):
    """The wrapper caches the rebuilt NF4 views per pool mapping (#841), so a warm
    runtime restores nothing. Drop the cache to measure one step's restores."""
    from soup_cli.utils.layer_stream_runtime import decoder_owner

    for layer in decoder_owner(model).layers:
        layer._cached_substitution_buffers = None
        layer._cached_substitution_weights = None


@pytest.fixture()
def nf4(tmp_path):
    pytest.importorskip("peft")
    pytest.importorskip("bitsandbytes")
    from tests.test_v07202 import _nf4_stream

    model, runtime, *_ = _nf4_stream(tmp_path, n_layers=3, tie=False)
    yield model, runtime
    runtime.close()


@pytest.mark.parametrize("k", [1, 2, 4])
def test_nf4_restores_are_counted_and_do_not_grow_with_k(harness, nf4, k):
    model, runtime = nf4
    harness.make_non_vacuous(model)
    l2l = _l2l_counters(harness, model, runtime, k, cold=_cold_views)
    assert l2l.restores > 0
    # at most once per quantised linear per layer, never once per micro-batch
    assert l2l.restores <= 3 * 7
    assert l2l.layer_loads > 0
    assert runtime.stats()["layer_restores"] >= l2l.restores


def test_nf4_restores_agree_between_l2l_and_accumulation(harness, nf4):
    model, runtime = nf4
    harness.make_non_vacuous(model)
    counts = {}
    for k in (1, 2, 4):
        counts[k] = (
            _l2l_counters(harness, model, runtime, k, cold=_cold_views).restores,
            _ga_counters(harness, model, runtime, k, cold=_cold_views).restores,
        )
    assert len({l2l for l2l, _ga in counts.values()}) == 1, counts
    assert all(l2l == ga for l2l, ga in counts.values()), counts


# --------------------------------------------------------------------------
# ticket 03: SFT labels (-100 on padding / masked tokens) reach the loss
# --------------------------------------------------------------------------
def test_l2l_step_with_masked_labels_equals_gradient_accumulation(harness, streamed):  # noqa: F811
    import torch

    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    model, runtime = streamed
    harness.make_non_vacuous(model)
    mbs = _micro_batches(2)
    labels = [ids.clone() for ids in mbs]
    labels[0][:, :3] = -100
    labels[1][:, -2:] = -100
    n_items = sum(int((lab[:, 1:] != -100).sum()) for lab in labels)

    harness.zero_grads(model)
    for ids, lab in zip(mbs, labels):
        model(input_ids=ids, labels=lab, num_items_in_batch=n_items).loss.backward()
    ga = harness.collect_grads(model)

    step = L2LStep(model, runtime, capture_layer_call(model, mbs[0]))
    harness.zero_grads(model)
    step.run(mbs, _store(2), n_items=n_items, labels=labels)
    l2l = harness.collect_grads(model)

    assert harness.compare(ga, l2l)["equal"]
    unmasked = harness.collect_grads(model)
    assert any(torch.count_nonzero(g) for g in unmasked.values())
