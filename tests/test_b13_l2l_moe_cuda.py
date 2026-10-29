"""CUDA E3 and negative controls through the real per-expert runtime and L2L step.

Run this module in a fresh CUDA process. CUBLAS_WORKSPACE_CONFIG is installed before importing
its CPU helpers (and therefore torch); every case enables deterministic kernels. The resident
reference is independent native eager accumulation. All comparisons remain torch.equal,
including every LoRA gradient and CPU/CUDA generator state; there is no GPU-only tolerance.
"""

from __future__ import annotations

import os
from typing import Callable, Iterator, Sequence

import pytest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from tests import test_b13_l2l_moe as _moe  # noqa: E402

make_rig = _moe.make_rig
pytestmark = [pytest.mark.gpu, _moe.pytestmark]


@pytest.fixture(autouse=True)
def deterministic_kernels() -> Iterator[None]:
    import torch

    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


@pytest.mark.parametrize("dropout", [0.0, 0.05])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_cuda_e3_l2l_equals_resident_eager_accumulation(
    make_rig: Callable[..., _moe.Rig], k: int, dropout: float,
    record_property: Callable[..., None],
) -> None:
    _rig, want, got = _moe.check_l2l_equals_accumulation(
        make_rig, k=k, dropout=dropout, device="cuda"
    )
    _moe.record_comparison(record_property, want, got)


def test_cuda_backward_only_corruption_changes_grads_not_losses(
    make_rig: Callable[..., _moe.Rig], monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[..., None],
) -> None:
    evidence = _moe.check_backward_only_perturbation(make_rig, monkeypatch, device="cuda")
    for key, value in evidence.items():
        record_property(key, value)


def test_cuda_broken_rng_replay_changes_grads_not_losses(
    make_rig: Callable[..., _moe.Rig], monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[..., None],
) -> None:
    evidence = _moe.check_broken_rng_replay(make_rig, monkeypatch, device="cuda")
    for key, value in evidence.items():
        record_property(key, value)


@pytest.mark.parametrize("dropout", [0.0, 0.05])
def test_cuda_consecutive_windows_and_masked_labels_stay_exact(
    make_rig: Callable[..., _moe.Rig], dropout: float, record_property: Callable[..., None],
) -> None:
    rig = make_rig(k=2, dropout=dropout, device="cuda")
    specs = []
    for first in (0, 10, 0):
        ids = _moe.micro_batches(rig, 2, first=first)
        labels = [row.clone() for row in ids]
        labels[0][:, :3] = -100
        labels[1][0, :] = -100
        total = sum(int((row[:, 1:] != -100).sum()) for row in labels)
        specs.append(_moe.Spec(ids, labels, total))
    want = _moe.ga_windows(rig, specs)
    got = _moe.l2l_windows(rig, specs)
    for expected, actual in zip(want, got):
        _moe.assert_signal(expected)
        _moe.assert_routing_is_varied(expected, rig, 2)
        _moe.assert_same(expected, actual)
        _moe.record_comparison(record_property, expected, actual)


@pytest.mark.parametrize("top_k", [1, 3])
def test_cuda_router_top_k_varies_independently_of_accumulation(
    make_rig: Callable[..., _moe.Rig], top_k: int, record_property: Callable[..., None],
) -> None:
    _rig, want, got = _moe.check_l2l_equals_accumulation(
        make_rig,
        k=2,
        dropout=0.05,
        config=_moe.fixture.moe_config(num_experts_per_tok=top_k),
        rows=1,
        seq=4,
        device="cuda",
    )
    assert all(route.shape[1] == top_k for routes in want.routes.values() for route in routes)
    _moe.record_comparison(record_property, want, got)


def test_cuda_fp32_model_under_bf16_forward_amp_matches_resident_ga(
    make_rig: Callable[..., _moe.Rig], record_property: Callable[..., None],
) -> None:
    want, got = _moe.check_bf16_forward_amp(make_rig, k=2, device="cuda")
    for expected, actual in zip(want, got):
        _moe.record_comparison(record_property, expected, actual)


def test_cuda_eager_attention_and_lora_dropout_replay_match_resident_ga(
    make_rig: Callable[..., _moe.Rig], record_property: Callable[..., None],
) -> None:
    want, got = _moe.check_eager_attention_dropout(make_rig, device="cuda")
    for expected, actual in zip(want, got):
        _moe.record_comparison(record_property, expected, actual)


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize(
    "targets,dropout,neftune",
    [
        pytest.param(_moe.fixture.EMBEDDING_TARGETS, 0.0, 0.0, id="embedding32"),
        pytest.param(_moe.fixture.EMBEDDING_TARGETS, 0.0, 0.2, id="embedding32-neftune"),
        pytest.param(_moe.fixture.HEAD_SHARED_TARGETS, 0.05, 0.0, id="head-shared44-dropout"),
        pytest.param(_moe.fixture.ALL_TARGETS, 0.05, 0.0, id="combined46-dropout"),
    ],
)
def test_cuda_supported_trainables_match_repeated_resident_ga(
    make_rig: Callable[..., _moe.Rig], k: int, targets: Sequence[str], dropout: float,
    neftune: float, record_property: Callable[..., None],
) -> None:
    want, got = _moe.check_supported_trainable_boundaries(
        make_rig, k=k, targets=targets, dropout=dropout, neftune=neftune, device="cuda",
    )
    for expected, actual in zip(want, got):
        _moe.record_comparison(record_property, expected, actual)


def test_cuda_embedding_vjp_under_bf16_forward_amp_matches_resident_ga(
    make_rig: Callable[..., _moe.Rig], record_property: Callable[..., None],
) -> None:
    want, got = _moe.check_supported_trainable_boundaries(
        make_rig, k=2, targets=_moe.fixture.EMBEDDING_TARGETS, dropout=0.0,
        amp=True, device="cuda",
    )
    for expected, actual in zip(want, got):
        _moe.record_comparison(record_property, expected, actual)
