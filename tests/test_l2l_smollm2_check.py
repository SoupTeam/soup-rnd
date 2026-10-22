"""CPU tests for the ticket-04 CUDA check driver (benchmarks/harness/l2l_smollm2_check.py).

The GPU run cannot be tested here. These pin what it computes: one record per k with
equality flags and load counters, and the verdict over the k=1/2/4 records.
"""

from __future__ import annotations

import pytest

from tests.test_l2l_probe_harness import SEQ, _load, _micro_batches, streamed  # noqa: F401


@pytest.fixture()
def check():
    _load("l2l_schedule")
    return _load("l2l_smollm2_check")


def _store_factory(k, n_layers=3, hidden=64):
    import torch

    from soup_cli.utils.l2l_activations import ActivationStore

    def make():
        return ActivationStore(
            n_layers=n_layers, k=k, chunk_shape=(1, SEQ, hidden),
            dtype=torch.float32, device="cpu", pin=False,
        )

    return make


@pytest.mark.parametrize("k", [1, 2, 4])
def test_measure_k_records_equality_and_counters(check, streamed, k):  # noqa: F811
    model, runtime = streamed
    check.l2l_schedule.make_non_vacuous(model)
    record = check.measure_k(model, runtime, _micro_batches(k), _store_factory(k))
    assert record["k"] == k
    assert record["grads_equal"] is True
    assert record["losses_equal"] is True
    assert record["tensors"] == 3 * 4 * 2
    assert record["all_zero"] == []
    assert record["a_vs_a_equal"] is True
    assert record["l2l"]["layer_loads"] > 0
    assert record["ga"]["layer_loads"] >= record["l2l"]["layer_loads"]
    assert set(record["l2l"]) == {"layer_loads", "large_loads", "restores"}


def test_measure_k_flags_an_unequal_gradient(check, streamed, monkeypatch):  # noqa: F811
    model, runtime = streamed
    real = check.l2l_schedule.compare

    def broken(ref, cand):
        out = real(ref, cand)
        out["equal"] = False
        return out

    monkeypatch.setattr(check.l2l_schedule, "compare", broken)
    record = check.measure_k(model, runtime, _micro_batches(2), _store_factory(2))
    assert record["grads_equal"] is False


def _rec(k, *, equal=True, losses=True, l2l=10, ga=None, zero=()):
    return {
        "k": k, "grads_equal": equal, "losses_equal": losses, "all_zero": list(zero),
        "a_vs_a_equal": True,
        "l2l": {"layer_loads": l2l, "large_loads": 2, "restores": 0},
        "ga": {"layer_loads": ga if ga is not None else 10 * k, "large_loads": 2, "restores": 0},
    }


def test_verdict_passes_when_equal_and_loads_flat(check):
    out = check.verdict([_rec(1), _rec(2), _rec(4)])
    assert out["equal"] and out["loads_flat"] and out["pass"]


def test_verdict_fails_on_unequal_gradients(check):
    out = check.verdict([_rec(1), _rec(2, equal=False), _rec(4)])
    assert not out["equal"] and not out["pass"]


def test_verdict_fails_on_unequal_losses(check):
    assert not check.verdict([_rec(1), _rec(2, losses=False)])["pass"]


def test_verdict_fails_when_loads_grow_k_times(check):
    out = check.verdict([_rec(1), _rec(2, l2l=20), _rec(4, l2l=40)])
    assert not out["loads_flat"] and not out["pass"]


def test_verdict_fails_on_vacuous_zero_gradients(check):
    assert not check.verdict([_rec(1, zero=["a.lora_A"])])["pass"]


def test_verdict_needs_the_k1_baseline(check):
    assert not check.verdict([_rec(2), _rec(4)])["pass"]


def test_verdict_of_nothing_is_not_a_pass(check):
    assert not check.verdict([])["pass"]
