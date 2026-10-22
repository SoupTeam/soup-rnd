"""The L2L replay of the accumulation's random stream on CUDA.

``state_at`` sets the CPU and the CUDA generator inside ``torch.random.fork_rng`` over every
initialised CUDA card; ``replay`` runs the recorded dropouts on zeros on the card, so the
philox offset moves as it does in the accumulation. The tests carry the ``gpu`` marker: the
conftest hook skips them without a GPU, and ``pytest -m gpu`` selects them. They check that
the CUDA state is set and restored separately from the CPU state, and that the L2L step
equals plain gradient accumulation, and leaves both generators alike, at dropout 0 and 0.05.

Deterministic mode is needed even at dropout 0, as in the A0 probe (``--deterministic``):
cuBLAS and some backward kernels pick different reductions between runs, so two correct
schedules differ in the last bit without it. The module sets ``CUBLAS_WORKSPACE_CONFIG``
before CUDA starts, which holds only if no earlier test in the session initialised CUDA
(run ``pytest -m gpu --no-cov`` alone). Each test turns on
``torch.use_deterministic_algorithms``.
"""

from __future__ import annotations

import os
from contextlib import nullcontext

import pytest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from tests import test_l2l_rng as _rng  # noqa: E402

SEED = _rng.SEED
_same, ga_reference, l2l_run = _rng._same, _rng.ga_reference, _rng.l2l_run
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
# fork_rng over the cards with a captured state
# --------------------------------------------------------------------------


def test_cpu_and_cuda_state_are_set_from_the_snapshot_and_restored_apart():
    from soup_cli.utils import l2l_rng

    torch.manual_seed(1)
    torch.cuda.manual_seed(2)
    target = l2l_rng.snapshot()
    expected_cpu = torch.rand(4)
    expected_cuda = torch.rand(4, device=DEVICE)
    torch.manual_seed(3)
    torch.cuda.manual_seed(4)
    cpu_before = torch.get_rng_state()
    cuda_before = torch.cuda.get_rng_state(0)

    with l2l_rng.state_at(target):
        assert torch.equal(torch.rand(4), expected_cpu)
        assert torch.equal(torch.rand(4, device=DEVICE), expected_cuda)
        # draw from the CUDA stream only: the CPU state must not move with it
        cpu_mid = torch.get_rng_state()
        torch.rand(64, device=DEVICE)
        assert torch.equal(torch.get_rng_state(), cpu_mid)

    assert torch.equal(torch.get_rng_state(), cpu_before)
    assert torch.equal(torch.cuda.get_rng_state(0), cuda_before)


def test_replay_moves_the_cuda_offset_like_the_native_dropout():
    """Dropout on zeros of the recorded metadata consumes the philox stream like the real one."""
    from soup_cli.utils.l2l_rng import Draw, replay

    torch.cuda.manual_seed(7)
    start = torch.cuda.get_rng_state(0)
    x = torch.randn(1, 24, 64, device=DEVICE, requires_grad=True)
    torch.cuda.set_rng_state(start, 0)
    torch.nn.functional.dropout(x, 0.05, True)
    native_end = torch.cuda.get_rng_state(0)

    torch.cuda.set_rng_state(start, 0)
    draw = Draw(tuple(x.shape), tuple(x.stride()), x.dtype, str(x.device), 0.05)
    _states, end = replay([[draw]], 1)
    assert torch.equal(dict(end.cuda)[0], native_end)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two visible CUDA cards")
def test_a_state_restores_the_second_card_too():
    from soup_cli.utils import l2l_rng

    torch.cuda.manual_seed_all(5)
    target = l2l_rng.snapshot()
    before = torch.cuda.get_rng_state(1)
    with l2l_rng.state_at(target):
        torch.rand(4, device="cuda:1")
        assert not torch.equal(torch.cuda.get_rng_state(1), before)
    assert torch.equal(torch.cuda.get_rng_state(1), before)


# --------------------------------------------------------------------------
# the L2L step against plain gradient accumulation on CUDA
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


def test_two_windows_in_a_row_equal_accumulation_on_cuda(make_model, harness):
    model, runtime = make_model(0.05, DEVICE)
    mbs = _micro_batches(2)
    ga = ga_reference(model, harness, mbs, windows=2)
    l2l = _l2l(model, runtime, harness, mbs, windows=2)
    assert _same(harness, ga, l2l)


def test_dropout_changes_the_gradients_on_cuda(make_model, harness):
    mbs = _micro_batches(2)
    plain = ga_reference(make_model(0.0, DEVICE)[0], harness, mbs)
    dropped = ga_reference(make_model(0.05, DEVICE)[0], harness, mbs)
    assert not _same(harness, plain, dropped, stream=False)


def test_same_seed_repeats_and_another_seed_differs_on_cuda(make_model, harness):
    model, runtime = make_model(0.05, DEVICE)
    mbs = _micro_batches(2)
    first = _l2l(model, runtime, harness, mbs)
    again = _l2l(model, runtime, harness, mbs)
    other = _l2l(model, runtime, harness, mbs, seed=SEED + 1)
    assert _same(harness, first, again)
    assert not _same(harness, first, other, stream=False)


def test_one_shared_stream_disagrees_with_accumulation_on_cuda(
    make_model, harness, monkeypatch
):
    model, runtime = make_model(0.05, DEVICE)
    mbs = _micro_batches(2)
    ga = ga_reference(model, harness, mbs)
    monkeypatch.setattr("soup_cli.utils.l2l_rng.state_at", lambda state: nullcontext())
    assert not _same(harness, ga, _l2l(model, runtime, harness, mbs), stream=False)
