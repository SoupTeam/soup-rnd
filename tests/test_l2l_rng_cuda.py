"""The L2L layer-call state function on CUDA.

``layer_rng`` sets the CPU and the CUDA generator from ``layer_seed(...)`` inside
``torch.random.fork_rng`` over every initialised CUDA card. The tests carry the ``gpu`` marker: the
conftest hook skips them without a GPU, and ``pytest -m gpu`` selects them. They check that
the CUDA state is set and restored separately from the CPU state, and that the L2L step
equals gradient accumulation at dropout 0 and 0.05.

Deterministic mode is needed even at dropout 0, as in the A0 probe (``--deterministic``):
cuBLAS and some backward kernels pick different reductions between runs, so two correct
schedules differ in the last bit without it. The module sets ``CUBLAS_WORKSPACE_CONFIG``
before CUDA starts, which holds only if no earlier test in the session initialised CUDA
(run ``pytest -m gpu --no-cov`` alone). Each test turns on
``torch.use_deterministic_algorithms``.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from tests import test_l2l_rng as _rng  # noqa: E402

SEED, STEP = _rng.SEED, _rng.STEP
_same, ga_reference, l2l_run = _rng._same, _rng.ga_reference, _rng.l2l_run
_patch_layer_rng = _rng._patch_layer_rng
# the CPU suite's fixtures, collected here under the same names
harness, make_model = _rng.harness, _rng.make_model

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu

DEVICE = "cuda"


@pytest.fixture(autouse=True)
def deterministic_kernels():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


def _micro_batches(k):
    from tests.test_l2l_probe_harness import _micro_batches as cpu_batches

    return [ids.to(DEVICE) for ids in cpu_batches(k)]


def _cuda_store(k, n_layers=3, hidden=64):
    from soup_cli.utils.l2l_activations import ActivationStore
    from tests.test_l2l_probe_harness import SEQ

    return ActivationStore(
        n_layers=n_layers,
        k=k,
        chunk_shape=(1, SEQ, hidden),
        dtype=torch.float32,
        device=DEVICE,
        pin=False,
    )


def _l2l(model, runtime, harness, mbs, **kwargs):
    return l2l_run(model, runtime, harness, mbs, store=_cuda_store(len(mbs)), **kwargs)


# --------------------------------------------------------------------------
# fork_rng(devices=[device]) with the function's state
# --------------------------------------------------------------------------


def test_cpu_and_cuda_state_are_set_from_the_function_and_restored_apart():
    from soup_cli.utils.l2l_rng import layer_rng, layer_seed

    torch.manual_seed(1)
    torch.cuda.manual_seed(2)
    cpu_before = torch.get_rng_state()
    cuda_before = torch.cuda.get_rng_state(0)

    with layer_rng(SEED, STEP, 1, 2):
        cpu_inside = torch.rand(4)
        cuda_inside = torch.rand(4, device=DEVICE)
        # draw from the CUDA stream only: the CPU state must not move with it
        cpu_mid = torch.get_rng_state()
        torch.rand(64, device=DEVICE)
        assert torch.equal(torch.get_rng_state(), cpu_mid)

    seed = layer_seed(SEED, STEP, 1, 2)
    cpu_gen = torch.Generator().manual_seed(seed)
    cuda_gen = torch.Generator(device=DEVICE).manual_seed(seed)
    assert torch.equal(cpu_inside, torch.rand(4, generator=cpu_gen))
    assert torch.equal(cuda_inside, torch.rand(4, device=DEVICE, generator=cuda_gen))
    assert not torch.equal(cpu_mid, cpu_before)
    assert torch.equal(torch.get_rng_state(), cpu_before)
    assert torch.equal(torch.cuda.get_rng_state(0), cuda_before)


def test_a_layer_call_restores_every_card_even_when_the_input_is_on_the_cpu():
    from soup_cli.utils.l2l_rng import layer_rng

    torch.cuda.manual_seed_all(5)
    before = [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
    with layer_rng(SEED, STEP, 0, 0):
        torch.rand(4)
    after = [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
    assert all(torch.equal(a, b) for a, b in zip(before, after))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two visible CUDA cards")
def test_a_layer_call_restores_the_second_card_too():
    from soup_cli.utils.l2l_rng import layer_rng

    torch.cuda.manual_seed_all(5)
    before = torch.cuda.get_rng_state(1)
    with layer_rng(SEED, STEP, 0, 0):
        torch.rand(4, device="cuda:0")
        assert not torch.equal(torch.cuda.get_rng_state(1), before)  # manual_seed reached it
    assert torch.equal(torch.cuda.get_rng_state(1), before)


# --------------------------------------------------------------------------
# the L2L step against gradient accumulation on CUDA
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dropout", [0.0, 0.05])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_l2l_equals_accumulation_on_cuda(make_model, harness, dropout, k):
    model, runtime = make_model(dropout, DEVICE)
    mbs = _micro_batches(k)
    ga = ga_reference(model, harness, mbs)
    l2l = _l2l(model, runtime, harness, mbs)
    assert _same(harness, ga, l2l)
    assert harness.compare(ga[1], l2l[1])["all_zero"] == []


def test_dropout_changes_the_gradients_on_cuda(make_model, harness):
    mbs = _micro_batches(2)
    plain = ga_reference(make_model(0.0, DEVICE)[0], harness, mbs)
    dropped = ga_reference(make_model(0.05, DEVICE)[0], harness, mbs)
    assert not _same(harness, plain, dropped)


def test_same_seed_repeats_and_another_seed_differs_on_cuda(make_model, harness):
    model, runtime = make_model(0.05, DEVICE)
    mbs = _micro_batches(2)
    first = _l2l(model, runtime, harness, mbs)
    again = _l2l(model, runtime, harness, mbs)
    other = _l2l(model, runtime, harness, mbs, seed=SEED + 1)
    assert _same(harness, first, again)
    assert not _same(harness, first, other)


def test_the_step_leaves_both_outside_generators_where_they_were(make_model, harness):
    model, runtime = make_model(0.05, DEVICE)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(4321)
    cpu_before = torch.get_rng_state()
    cuda_before = torch.cuda.get_rng_state(0)
    _l2l(model, runtime, harness, _micro_batches(2))
    assert torch.equal(torch.get_rng_state(), cpu_before)
    assert torch.equal(torch.cuda.get_rng_state(0), cuda_before)


def test_the_function_not_called_on_the_recompute_disagrees_on_cuda(
    make_model, harness, monkeypatch
):
    from soup_cli.utils.l2l_rng import layer_rng

    model, runtime = make_model(0.05, DEVICE)
    mbs = _micro_batches(2)
    ga = ga_reference(model, harness, mbs)

    skipped = []

    @contextmanager
    def forward_only(seed, step, mb, layer):
        if torch.is_grad_enabled():
            skipped.append((mb, layer))
            yield
        else:
            with layer_rng(seed, step, mb, layer):
                yield

    _patch_layer_rng(monkeypatch, forward_only)
    assert not _same(harness, ga, _l2l(model, runtime, harness, mbs))
    assert len(skipped) == 3 * len(mbs)  # the recompute ran once per (layer, micro-batch)
