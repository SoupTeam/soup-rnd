"""CUDA variants of the B1.3 streamed-expert backward checks.

Same criteria as ``test_b13_expert_backward.py`` on a card: ``torch.equal`` of the output, the
hidden-state gradient and the routing-weight gradient against the resident eager module (for NF4
the independently dequantised copy), the full ``DeepseekV3MoE`` block, an arena churn between
forward and backward, and the negative control that perturbs only the expert weights loaded
inside autograd's engine. Deterministic mode is needed even for a forward/backward comparison:
cuBLAS and some index kernels pick different reductions between runs, so two correct schedules
would otherwise differ in the last bit.

The module sets ``CUBLAS_WORKSPACE_CONFIG`` before CUDA starts, which holds only if no earlier
test in the session initialised CUDA (run ``pytest -m gpu --no-cov`` on this file alone). Each
test turns on ``torch.use_deterministic_algorithms``. The tests carry the ``gpu`` marker: the
conftest hook skips them without a card.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from tests.test_b13_expert_backward import (  # noqa: E402
    _assert_exact,
    _assert_signal,
    _hit,
    check_arena_churn,
    check_block_backward,
    check_negative_control,
    exact_backward_case,
)

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu

DEVICE = "cuda"
#: dense bf16, NF4, NF4 with nested statistics: the B1.2 storage matrix on a card
_STORAGE = {
    "bf16": dict(nf4=False, nested=False),
    "nf4": dict(nf4=True, nested=False),
    "nf4-nested": dict(nf4=True, nested=True),
}


@pytest.fixture(autouse=True)
def deterministic_kernels():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize("storage", _STORAGE)
def test_cuda_backward_equals_the_resident_eager_module(tmp_path, storage):
    if _STORAGE[storage]["nf4"]:
        pytest.importorskip("bitsandbytes")
    with exact_backward_case(
        tmp_path, device=DEVICE, dtype="bfloat16", count=24, **_STORAGE[storage]
    ) as case:
        want, got, reader, pool, args = case
        _assert_signal(want)
        _assert_exact(want, got)
        # forward and backward each read every hit expert once through the one-slot arena
        assert pool.fetches == reader.fetches == 2 * len(_hit(args))
        assert pool.cuda_allocation_delta <= pool.weight_budget_bytes


@pytest.mark.parametrize("storage", _STORAGE)
def test_cuda_full_moe_block_grads_equal_eager(tmp_path, storage):
    if _STORAGE[storage]["nf4"]:
        pytest.importorskip("bitsandbytes")
    check_block_backward(tmp_path, device=DEVICE, dtype="bfloat16", **_STORAGE[storage])


def test_cuda_arena_churn_between_forward_and_backward_keeps_the_gradients(tmp_path):
    check_arena_churn(tmp_path, device=DEVICE, dtype="bfloat16")


@pytest.mark.parametrize("storage", ["bf16", "nf4-nested"])
def test_cuda_perturbing_only_backward_reloads_changes_gradients_not_outputs(
    tmp_path, monkeypatch, storage
):
    if _STORAGE[storage]["nf4"]:
        pytest.importorskip("bitsandbytes")
    check_negative_control(
        tmp_path, monkeypatch, device=DEVICE, dtype="bfloat16", **_STORAGE[storage]
    )
