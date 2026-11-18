"""A2: the batch-invariant routed-expert GEMM makes route-preserving MBR equal per-micro-batch L2L.

Seam: ``L2LStep.run`` over a per-expert DeepSeek-V3 model from ``build_streamed_model`` with
``expert_gemm="batch_invariant"`` (the routed experts' matmuls on
``soup_cli.utils.invariant_gemm``), two reads and grouped routes. The same window runs as plain
two-read L2L (one expert call per micro-batch) and as MBR at several group sizes (one expert call
over the rows of 2 or all micro-batches). With an expert GEMM whose rows do not depend on how many
rows share a call, every arm must give the same per-micro-batch losses and every LoRA gradient,
compared with ``torch.equal``. There is no tolerance.

The kernel needs CUDA; the GPU cases run the real Triton GEMM. The CPU cases check the same
contract through the wiring with a row-at-a-time stand-in for the kernel (its rows cannot depend
on M by construction), plus a control that the native GEMM is wired as before.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List

import pytest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from tests import test_b13_l2l_moe as _moe  # noqa: E402

make_rig = _moe.make_rig
pytestmark = [_moe.pytestmark]

LEVERS = {"two_read_experts": True, "grouped_expert_routes": True}
K = 4


def _cuda_ok() -> bool:
    try:
        import torch
        import triton  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


def _window(rig: Any, k: int) -> List[Any]:
    """Micro-batches of different lengths, so the joined expert calls are ragged."""
    seqs = (3, 6, 4, 5, 2, 6)[:k]
    return [
        _moe.fixture.moe_batch(rig.config, seed=seed, rows=1, seq=seq)["input_ids"].to(rig.device)
        for seed, seq in enumerate(seqs)
    ]


def _arms(rig: Any, ids: List[Any], groups: List[Any]) -> Dict[Any, Any]:
    """group -> (losses, grads); ``None`` is plain two-read L2L."""
    import torch

    from soup_cli.utils import l2l_step

    calls = [l2l_step.capture_layer_call(rig.model, item) for item in ids]
    out = {}
    for group in groups:
        _moe._zero_grads(rig.model)
        torch.manual_seed(_moe.SEED)
        options = {} if group is None else {"route_preserving_mbr": True, "mbr_group_size": group}
        losses = rig.stepper.run(ids, rig.store, calls=calls, **options)
        grads = {
            name: grad.detach().clone()
            for name, grad in _moe._collect(rig.model).items() if grad is not None
        }
        out[group] = ([loss.detach().clone() for loss in losses], grads)
    return out


def _assert_equal(want: Any, got: Any, what: str) -> None:
    import torch

    losses_a, grads_a = want
    losses_b, grads_b = got
    assert len(losses_a) == len(losses_b)
    for mb, (a, b) in enumerate(zip(losses_a, losses_b)):
        assert torch.equal(a, b), f"{what}: loss of micro-batch {mb}: {a.item()} vs {b.item()}"
    assert grads_a and grads_a.keys() == grads_b.keys()
    differ = [name for name in grads_a if not torch.equal(grads_a[name], grads_b[name])]
    assert not differ, f"{what}: {len(differ)} LoRA gradients differ, first {differ[:3]}"


def _rowwise_linear(x: Any, weight: Any) -> Any:
    import torch
    import torch.nn.functional as functional

    flat = x.reshape(-1, x.shape[-1])
    rows = [functional.linear(flat[i:i + 1], weight) for i in range(flat.shape[0])]
    out = torch.cat(rows) if rows else flat.new_empty((0, weight.shape[0]))
    return out.reshape(*x.shape[:-1], weight.shape[0])


# ==========================================================================
# wiring (CPU)
# ==========================================================================
def test_cpu_wiring_runs_every_expert_matmul_through_the_selected_linear(
    make_rig: Callable[..., Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Install binds ``invariant_gemm.linear``; forward and recompute of each hit expert use it."""
    from soup_cli.utils import invariant_gemm

    monkeypatch.setattr(invariant_gemm, "check_supported", lambda device, dtype: None)
    calls = []

    def counting(x: Any, weight: Any) -> Any:
        calls.append(tuple(x.shape))
        return _rowwise_linear(x, weight)

    monkeypatch.setattr(invariant_gemm, "linear", counting)
    rig = make_rig(k=K, rows=1, seq=6, build_options={**LEVERS, "expert_gemm": "batch_invariant"})
    experts = list(_moe.streamed_experts_modules(rig.model).values())
    assert experts and all(module.linear is counting for module in experts)
    _arms(rig, _window(rig, K), [K])
    hits = sum(len(module.last_trace) for module in experts)
    # The last call per layer is the backward recompute: two matmuls per hit expert, and the
    # forward ran at least as many.
    assert hits and len(calls) >= 4 * hits and len(calls) % 2 == 0


@pytest.mark.parametrize("dropout", [0.0, 0.05])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_cpu_rowwise_stand_in_makes_mbr_equal_l2l_at_every_group_size(
    make_rig: Callable[..., Any], monkeypatch: pytest.MonkeyPatch, dtype: str, dropout: float,
) -> None:
    """Shared-expert and attention LoRA, with and without dropout, in fp32 and bf16 storage."""
    from soup_cli.utils import invariant_gemm

    monkeypatch.setattr(invariant_gemm, "check_supported", lambda device, dtype: None)
    monkeypatch.setattr(invariant_gemm, "linear", _rowwise_linear)
    rig = make_rig(
        k=5, rows=1, seq=6, dropout=dropout, targets=_moe.fixture.HEAD_SHARED_TARGETS,
        build_options={**LEVERS, "expert_gemm": "batch_invariant", "dtype": dtype},
    )
    arms = _arms(rig, _window(rig, 5), [None, 1, 2, 5])
    for group in (1, 2, 5):
        _assert_equal(arms[None], arms[group], f"MBR group {group} vs L2L")


def test_cpu_native_mbr_differs_from_l2l_so_the_check_has_teeth(
    make_rig: Callable[..., Any],
) -> None:
    """The M-dependent CPU BLAS makes the same comparison fail on the native GEMM."""
    rig = make_rig(k=K, rows=1, seq=6, build_options=LEVERS)
    arms = _arms(rig, _window(rig, K), [None, K])
    with pytest.raises(AssertionError, match="LoRA gradients differ"):
        _assert_equal(arms[None], arms[K], "native MBR vs L2L")


def test_native_expert_gemm_stays_functional_linear(make_rig: Callable[..., Any]) -> None:
    import torch.nn.functional as functional

    rig = make_rig(k=2, rows=1, seq=6, build_options=LEVERS)
    experts = list(_moe.streamed_experts_modules(rig.model).values())
    assert experts and all(module.linear is functional.linear for module in experts)
    assert all(module.expert_gemm == "native" for module in experts)


@pytest.mark.parametrize(
    ("expert_gemm", "message"),
    [("batch_invariant", "needs a CUDA device"), ("triton", "expert_gemm must be one of")],
)
def test_batch_invariant_refuses_cpu_and_unknown_names(
    make_rig: Callable[..., Any], expert_gemm: str, message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        make_rig(k=2, build_options={**LEVERS, "expert_gemm": expert_gemm})


def test_fused_layout_refuses_the_expert_gemm(make_rig: Callable[..., Any]) -> None:
    with pytest.raises(ValueError, match="expert_gemm needs a cache sharded"):
        make_rig(k=2, layout="fused", build_options={"expert_gemm": "batch_invariant"})


# ==========================================================================
# the real kernel (CUDA)
# ==========================================================================
@pytest.fixture()
def deterministic_kernels() -> Any:
    import torch

    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


def _wide_config() -> Any:
    """Wider than one GEMM tile in every dimension, so M tiles and K loops really vary."""
    return _moe.fixture.moe_config(
        hidden_size=256, moe_intermediate_size=128, intermediate_size=256,
        n_routed_experts=16, num_experts_per_tok=4,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_ok(), reason="needs CUDA and triton")
@pytest.mark.usefixtures("deterministic_kernels")
@pytest.mark.parametrize(
    ("dtype", "amp", "dropout"),
    [("float32", False, 0.0), ("float32", False, 0.05), ("bfloat16", True, 0.05)],
)
def test_cuda_batch_invariant_mbr_equals_l2l_bit_for_bit(
    make_rig: Callable[..., Any], dtype: str, amp: bool, dropout: float,
) -> None:
    import torch

    if dtype == "bfloat16" and torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("bf16 needs compute capability 8.0+")
    rig = make_rig(
        k=6, rows=2, seq=16, device="cuda", dropout=dropout, config=_wide_config(),
        targets=_moe.fixture.HEAD_SHARED_TARGETS,
        build_options={**LEVERS, "expert_gemm": "batch_invariant", "dtype": dtype},
    )
    ids = [
        _moe.fixture.moe_batch(rig.config, seed=seed, rows=2, seq=16)["input_ids"].to("cuda")
        for seed in range(6)
    ]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        arms = _arms(rig, ids, [None, 1, 2, 3, 6])
        again = _arms(rig, ids, [6])
    for group in (1, 2, 3, 6):
        _assert_equal(arms[None], arms[group], f"MBR group {group} vs L2L")
    _assert_equal(arms[6], again[6], "MBR repeat")


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_ok(), reason="needs CUDA and triton")
@pytest.mark.usefixtures("deterministic_kernels")
def test_cuda_native_mbr_differs_from_l2l(make_rig: Callable[..., Any]) -> None:
    """The same comparison on cuBLAS: shows the CUDA check is not vacuous at these widths."""
    rig = make_rig(
        k=6, rows=2, seq=16, device="cuda", config=_wide_config(), build_options=LEVERS,
    )
    ids = [
        _moe.fixture.moe_batch(rig.config, seed=seed, rows=2, seq=16)["input_ids"].to("cuda")
        for seed in range(6)
    ]
    arms = _arms(rig, ids, [None, 6])
    with pytest.raises(AssertionError, match="differ"):
        _assert_equal(arms[None], arms[6], "native MBR vs L2L")


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_ok(), reason="needs CUDA and triton")
@pytest.mark.usefixtures("deterministic_kernels")
def test_cuda_batch_invariant_kernel_matches_native_within_rounding(
    make_rig: Callable[..., Any],
) -> None:
    """Different summation order, same product: L2L losses agree to float32 rounding."""
    import torch

    runs = []
    for expert_gemm in ("native", "batch_invariant"):
        rig = make_rig(
            k=2, rows=1, seq=6, device="cuda",
            build_options={**LEVERS, "expert_gemm": expert_gemm},
        )
        runs.append(_arms(rig, _window(rig, 2), [None])[None])
    (losses_a, grads_a), (losses_b, grads_b) = runs
    for a, b in zip(losses_a, losses_b):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    for name in grads_a:
        torch.testing.assert_close(grads_a[name], grads_b[name], rtol=1e-3, atol=1e-5)
