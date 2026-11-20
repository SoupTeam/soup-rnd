"""``--expert-gemm``: native cuBLAS stays the default, ``lt-fused`` raises instead of falling back.

All CPU. The environment probe is replaced, so each refusal is checked without a card."""

from __future__ import annotations

from typing import Any, Optional

import pytest

torch = pytest.importorskip("torch")

from soup_cli.utils import expert_gemm  # noqa: E402
from soup_cli.utils.expert_gemm import (  # noqa: E402
    DEFAULT_EXPERT_GEMM,
    EXPERT_GEMM_CHOICES,
    ExactGemmUnavailableError,
    GemmEnvironment,
    expert_gemm_context,
    require_lt_fused,
)


def env(**changes: Any) -> GemmEnvironment:
    values: dict = {
        "torch_version": "2.14.1+cu130",
        "cuda_available": True,
        "capability": (8, 9),
        "toolchain_error": None,
    }
    values.update(changes)
    return GemmEnvironment(**values)


def test_a_supported_environment_passes() -> None:
    require_lt_fused(env())


@pytest.mark.parametrize(
    ("changes", "words"),
    [
        ({"cuda_available": False, "capability": None}, "CUDA"),
        ({"toolchain_error": "no nvcc found"}, "no nvcc found"),
        ({"capability": (7, 5)}, "sm_80"),
        ({"capability": (7, 5), "torch_version": "2.14.0"}, "sm_80"),
        ({"torch_version": "2.11.0+cu130"}, "2.14"),
        ({"torch_version": "2.15.0"}, "2.14"),
        ({"torch_version": "2.140.0"}, "2.14"),
    ],
)
def test_lt_fused_names_the_missing_requirement(changes: dict, words: str) -> None:
    with pytest.raises(ExactGemmUnavailableError, match=words):
        require_lt_fused(env(**changes))


def test_the_refusal_is_a_runtime_error_with_the_flag_in_the_message() -> None:
    with pytest.raises(RuntimeError, match="--expert-gemm lt-fused"):
        require_lt_fused(env(cuda_available=False, capability=None))


@pytest.mark.parametrize(
    "changes",
    [
        {"cuda_available": False, "capability": None},
        {"toolchain_error": "no nvcc found"},
        {"capability": (7, 5)},
        {"torch_version": "2.11.0"},
    ],
)
def test_lt_fused_context_never_falls_back_to_native(
    monkeypatch: pytest.MonkeyPatch, changes: dict
) -> None:
    monkeypatch.setattr(expert_gemm, "detect_environment", lambda: env(**changes))
    built: list = []
    monkeypatch.setattr(
        expert_gemm.a0_exact_lt, "load_extension", lambda *a, **k: built.append(1)
    )
    with pytest.raises(ExactGemmUnavailableError):
        with expert_gemm_context("lt-fused"):
            pytest.fail("the body ran: the context swapped in another path")
    assert built == []  # the extension is not even built


def test_native_is_the_default_and_patches_nothing() -> None:
    import torch.nn.functional as functional

    from soup_cli.utils import expert_stream, l2l_step

    assert DEFAULT_EXPERT_GEMM == "native"
    assert EXPERT_GEMM_CHOICES == ("native", "lt-fused")
    before = (
        functional.linear,
        torch.Tensor.__mul__,
        expert_stream._grouped_expert_routes,
        expert_stream._default_expert_routes,
        expert_stream.streamed_experts_class().forward,
        l2l_step._route_preserving_deepseek_forward,
    )
    with expert_gemm_context() as default, expert_gemm_context("native") as named:
        during = (
            functional.linear,
            torch.Tensor.__mul__,
            expert_stream._grouped_expert_routes,
            expert_stream._default_expert_routes,
            expert_stream.streamed_experts_class().forward,
            l2l_step._route_preserving_deepseek_forward,
        )
        assert default is None and named is None
    assert all(a is b for a, b in zip(before, during))


def test_native_does_not_probe_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> Optional[GemmEnvironment]:
        raise AssertionError("native must not look at CUDA or a compiler")

    monkeypatch.setattr(expert_gemm, "detect_environment", boom)
    with expert_gemm_context("native"):
        pass


def test_an_unknown_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="native"):
        expert_gemm_context("segmented")


def test_detect_environment_reports_this_machine() -> None:
    found = expert_gemm.detect_environment()
    assert found.torch_version == torch.__version__
    assert found.cuda_available == torch.cuda.is_available()
    assert (found.capability is not None) == found.cuda_available
