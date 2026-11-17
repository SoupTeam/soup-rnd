"""Fail-closed attention receipts and native Ampere operator/kernel qualification."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import pytest

from benchmarks.harness import p3_deepseek_quality_attention as attention


def _serialized_evidence(backend: str = "EFFICIENT_ATTENTION") -> dict[str, Any]:
    """Serialization-only fixture. These strings never claim a measured GPU run."""
    call = {
        "dtype": "torch.bfloat16", "device": "cuda", "device_index": 0,
        "key_dtype": "torch.bfloat16", "value_dtype": "torch.bfloat16",
        "key_device": "cuda", "value_device": "cuda",
        "key_device_index": 0, "value_device_index": 0,
        "query_shape": [1, 128, 1024, 192], "key_shape": [1, 128, 1024, 192],
        "value_shape": [1, 128, 1024, 128],
    }
    efficient = backend == "EFFICIENT_ATTENTION"
    return {
        "backend_requested": backend, "executed_sdpa_calls": 2, "unique_calls": [call],
        "hardware": {
            "device": "cuda:0", "name": "serialized Ampere fixture", "capability": [8, 0],
            "native_bf16": True, "bf16_supported": True,
            "native_bf16_check": "torch.cuda.is_bf16_supported(including_emulation=False)",
            "cuda_version": "12.8",
        },
        "deterministic_flags": {
            "algorithms_enabled": True, "warn_only": False, "allow_tf32": False,
            "cudnn_allow_tf32": False, "cudnn_benchmark": False, "cudnn_deterministic": True,
        },
        "backend_flags": {"efficient": efficient, "math": not efficient,
                          "flash": False, "cudnn": False},
        "instrumentation": {
            "scope": "bounded_attention_operators", "max_signatures": 16,
            "max_profiled_operations": 80, "profiled_operations": 2,
            "retains_tensors": False, "full_window_trace": False,
            "signature_limit_exceeded": False, "profiling_errors": [],
        },
        "samples": [{
            **copy.deepcopy(call),
            "forward": {
                "operators": ["aten::_scaled_dot_product_efficient_attention" if efficient
                              else "aten::_scaled_dot_product_attention_math"],
                "cuda_kernel_names": ["fmha_cutlassF_bf16_aligned_32x128_gmem_sm80"]
                if efficient else ["ampere_bf16_s16816gemm"],
            },
            "backward": {
                "operators": ["aten::_scaled_dot_product_efficient_attention_backward"]
                if efficient else ["aten::bmm", "aten::_softmax_backward_data"],
                "cuda_kernel_names": ["fmha_cutlassB_bf16_aligned_128x64_k65536_sm80"]
                if efficient else ["ampere_bf16_s16816gemm", "softmax_backward_cuda"],
            },
        }],
    }


def test_policy_admits_only_the_approved_immutable_backend_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = attention.policy_identity()
    assert identity["path"] == "benchmarks/results/p3-deepseek-quality/attention-protocol-v1.json"
    assert identity["sha256"] == (
        "fc887c5a996bd58536ef0325b61405cd73a2a83cf363fcc4156204ab180e1ed6")
    assert identity["contract"]["main_backend"] == "EFFICIENT_ATTENTION"
    original = Path.read_bytes

    def changed_protocol(path: Path) -> bytes:
        if path.as_posix().endswith(identity["path"]):
            changed = copy.deepcopy(identity["contract"])
            changed["main_backend"] = "MATH"
            return json.dumps(changed).encode()
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", changed_protocol)
    with pytest.raises(ValueError, match="protocol"):
        attention.policy_identity()


@pytest.mark.parametrize("backend", ["EFFICIENT_ATTENTION", "MATH"])
def test_serialized_admission_requires_its_own_backend_kernel_evidence(backend: str) -> None:
    receipt = _serialized_evidence(backend)
    attention.require_execution_evidence(receipt, backend=backend, dtype="bfloat16", min_calls=2)
    other = "MATH" if backend == "EFFICIENT_ATTENTION" else "EFFICIENT_ATTENTION"
    with pytest.raises(ValueError, match="attention"):
        attention.require_execution_evidence(receipt, backend=other, dtype="bfloat16")


@pytest.mark.parametrize("damage", [
    "no_calls", "too_few_calls", "no_shapes", "unprofiled_shape", "wrong_dtype", "cpu",
    "wrong_key", "wrong_value", "no_forward", "no_backward", "no_forward_kernels",
    "no_backward_kernels", "autograd_name_only", "math_fallback", "warn_only", "tf32",
    "no_determinism", "emulated_bf16", "t4", "unbounded", "retained_tensors",
    "profiling_failure", "signature_overflow", "mismatched_forward_kernel",
    "mismatched_backward_kernel",
])
def test_serialized_admission_refuses_incomplete_or_inconsistent_execution(damage: str) -> None:
    receipt = _serialized_evidence()
    if damage == "no_calls":
        receipt["executed_sdpa_calls"] = 0
    elif damage == "too_few_calls":
        receipt["executed_sdpa_calls"] = 1
    elif damage == "no_shapes":
        receipt["unique_calls"] = []
    elif damage == "unprofiled_shape":
        second = copy.deepcopy(receipt["unique_calls"][0])
        second["query_shape"][2] = 4096
        second["key_shape"][2] = second["value_shape"][2] = 4096
        receipt["unique_calls"].append(second)
    elif damage == "wrong_dtype":
        receipt["unique_calls"][0]["dtype"] = "torch.float32"
    elif damage == "cpu":
        receipt["unique_calls"][0]["device"] = "cpu"
    elif damage == "wrong_key":
        receipt["unique_calls"][0]["key_shape"][-1] = 128
    elif damage == "wrong_value":
        receipt["samples"][0]["value_shape"][-1] = 192
    elif damage == "no_forward":
        receipt["samples"][0]["forward"]["operators"] = []
    elif damage == "no_backward":
        receipt["samples"][0]["backward"]["operators"] = []
    elif damage in {"no_forward_kernels", "no_backward_kernels"}:
        direction = "forward" if damage == "no_forward_kernels" else "backward"
        receipt["samples"][0][direction]["cuda_kernel_names"] = []
    elif damage == "autograd_name_only":
        receipt["samples"][0]["backward"]["operators"] = [
            "ScaledDotProductEfficientAttentionBackward0"]
    elif damage in {"mismatched_forward_kernel", "mismatched_backward_kernel"}:
        direction = "forward" if damage == "mismatched_forward_kernel" else "backward"
        receipt["samples"][0][direction]["cuda_kernel_names"] = ["ampere_bf16_s16816gemm"]
    elif damage == "math_fallback":
        receipt["backend_flags"]["math"] = True
    elif damage == "warn_only":
        receipt["deterministic_flags"]["warn_only"] = True
    elif damage == "tf32":
        receipt["deterministic_flags"]["allow_tf32"] = True
    elif damage == "no_determinism":
        receipt["deterministic_flags"]["algorithms_enabled"] = False
    elif damage == "emulated_bf16":
        receipt["hardware"]["native_bf16"] = False
    elif damage == "t4":
        receipt["hardware"]["capability"] = [7, 5]
    elif damage == "unbounded":
        receipt["instrumentation"]["full_window_trace"] = True
    elif damage == "retained_tensors":
        receipt["instrumentation"]["retains_tensors"] = True
    elif damage == "profiling_failure":
        receipt["instrumentation"]["profiling_errors"] = ["CUPTI unavailable"]
    else:
        receipt["instrumentation"]["signature_limit_exceeded"] = True
    with pytest.raises(ValueError, match="attention"):
        attention.require_execution_evidence(receipt, backend="EFFICIENT_ATTENTION",
                                             dtype="bfloat16", min_calls=2)


def test_math_floor_refuses_fused_cuda_kernel_identity() -> None:
    receipt = _serialized_evidence("MATH")
    receipt["samples"][0]["forward"]["cuda_kernel_names"] = [
        "fmha_cutlassF_bf16_aligned_32x128_gmem_sm80"]
    with pytest.raises(ValueError, match="attention"):
        attention.require_execution_evidence(receipt, backend="MATH", dtype="bfloat16")


def test_cpu_and_unknown_backend_are_rejected_without_executing_attention() -> None:
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="CUDA"):
        attention.require_ampere_cuda("cpu")
    with pytest.raises(ValueError, match="backend"):
        with attention.attention_execution("FLASH_ATTENTION", device="cuda"):
            pytest.fail("unsupported backend was entered")
    with pytest.raises(ValueError, match="CUDA"):
        with attention.attention_execution("EFFICIENT_ATTENTION", device="cpu"):
            pytest.fail("CPU attention was entered")


@pytest.mark.parametrize("capability,native", [([7, 5], True), ([8, 0], False)])
def test_hardware_admission_refuses_t4_and_emulated_bf16(
    monkeypatch: pytest.MonkeyPatch, capability: list[int], native: bool,
) -> None:
    from contextlib import nullcontext

    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())

    def bf16_support(*, including_emulation: bool = True) -> bool:
        return True if including_emulation else native

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", bf16_support)
    with pytest.raises(ValueError, match="Ampere|native"):
        attention.require_ampere_cuda("cuda")


def _ampere_stack() -> Any:
    # The GPU marker owns no-CUDA skipping. An available but ineligible GPU must fail.
    torch = pytest.importorskip("torch")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    attention.require_ampere_cuda("cuda")
    return torch


def _native_sdpa(torch: Any, length: int, backend: str) -> tuple[list[Any], dict[str, Any]]:
    generator = torch.Generator(device="cuda").manual_seed(23)
    with attention.attention_execution(backend, device="cuda") as observed:
        qkv = [torch.randn((1, 128, length, dimension), generator=generator,
                           dtype=torch.bfloat16, device="cuda", requires_grad=True)
               for dimension in (192, 192, 128)]
        output = torch.nn.functional.scaled_dot_product_attention(*qkv, is_causal=True)
        cotangent = torch.randn(output.shape, generator=generator,
                                dtype=output.dtype, device=output.device)
        output.backward(cotangent)
        observed.require_complete(dtype="bfloat16")
        tensors = [output.detach().cpu(), *(value.grad.detach().cpu() for value in qkv)]
        receipt = observed.report()
    return tensors, receipt


@pytest.mark.gpu
@pytest.mark.parametrize("length", [1024, 4096])
def test_exact_mla192_128_executes_strict_efficient_forward_backward_and_repeats(
    length: int,
) -> None:
    torch = _ampere_stack()
    first, first_receipt = _native_sdpa(torch, length, "EFFICIENT_ATTENTION")
    second, second_receipt = _native_sdpa(torch, length, "EFFICIENT_ATTENTION")
    assert all(torch.equal(left, right) for left, right in zip(first, second, strict=True))
    for receipt in (first_receipt, second_receipt):
        assert receipt["unique_calls"][0]["query_shape"] == [1, 128, length, 192]
        assert receipt["unique_calls"][0]["key_shape"] == [1, 128, length, 192]
        assert receipt["unique_calls"][0]["value_shape"] == [1, 128, length, 128]
        sample = receipt["samples"][0]
        assert "aten::_scaled_dot_product_efficient_attention" in sample["forward"]["operators"]
        assert "aten::_scaled_dot_product_efficient_attention_backward" in (
            sample["backward"]["operators"])
        assert sample["forward"]["cuda_kernel_names"]
        assert sample["backward"]["cuda_kernel_names"]
    root = os.environ.get("P3_QUALITY_EVIDENCE")
    if root:
        target = Path(root) / f"efficient-mla192-128-L{length}.json"
        with target.open("x", encoding="utf-8") as stream:
            json.dump({"repeat_equal": True, "passes": [first_receipt, second_receipt]},
                      stream, indent=2, allow_nan=False)
            stream.write("\n")


@pytest.mark.gpu
def test_math_alternative_floor_has_executed_cuda_forward_and_backward() -> None:
    torch = _ampere_stack()
    with attention.attention_execution("MATH", device="cuda") as observed:
        qkv = [torch.randn(1, 2, 64, 32, device="cuda", dtype=torch.bfloat16,
                           requires_grad=True) for _ in range(3)]
        torch.nn.functional.scaled_dot_product_attention(*qkv, is_causal=True).sum().backward()
        observed.require_complete(dtype="bfloat16")
    sample = observed.report()["samples"][0]
    assert "aten::_scaled_dot_product_attention_math" in sample["forward"]["operators"]
    assert "aten::bmm" in sample["backward"]["operators"]
    assert "aten::_softmax_backward_data" in sample["backward"]["operators"]
    assert sample["forward"]["cuda_kernel_names"] and sample["backward"]["cuda_kernel_names"]


@pytest.mark.gpu
@pytest.mark.parametrize("raise_inside", [False, True])
def test_attention_context_restores_permissions_and_advances_rng_between_windows(
    raise_inside: bool,
) -> None:
    torch = _ampere_stack()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    original = {
        "enabled": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }
    rng = torch.random.get_rng_state()
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        with sdpa_kernel([SDPBackend.MATH]):
            for _ in range(2):
                before = torch.random.get_rng_state()
                try:
                    with attention.attention_execution("EFFICIENT_ATTENTION", device="cuda"):
                        assert torch.are_deterministic_algorithms_enabled()
                        assert not torch.is_deterministic_algorithms_warn_only_enabled()
                        assert not torch.backends.cuda.matmul.allow_tf32
                        assert not torch.backends.cudnn.allow_tf32
                        torch.rand(3)
                        if raise_inside:
                            raise RuntimeError("injected body failure")
                except RuntimeError as error:
                    assert raise_inside and str(error) == "injected body failure"
                assert not torch.equal(before, torch.random.get_rng_state())
                assert torch.is_deterministic_algorithms_warn_only_enabled()
                assert torch.backends.cuda.matmul.allow_tf32
                assert torch.backends.cudnn.allow_tf32
                assert torch.backends.cuda.math_sdp_enabled()
                assert not torch.backends.cuda.mem_efficient_sdp_enabled()
    finally:
        torch.random.set_rng_state(rng)
        torch.use_deterministic_algorithms(original["enabled"], warn_only=original["warn_only"])
        torch.backends.cuda.matmul.allow_tf32 = original["tf32"]
        torch.backends.cudnn.allow_tf32 = original["cudnn_tf32"]


@pytest.mark.gpu
def test_inference_mode_forward_evidence_does_not_require_a_backward_graph() -> None:
    torch = _ampere_stack()
    with torch.inference_mode(), attention.attention_execution(
        "EFFICIENT_ATTENTION", device="cuda",
    ) as observed:
        qkv = [torch.randn(1, 4, 64, dimension, device="cuda", dtype=torch.bfloat16)
               for dimension in (192, 192, 128)]
        output = torch.nn.functional.scaled_dot_product_attention(*qkv, is_causal=True)
        assert output.shape == (1, 4, 64, 128)
        observed.require_complete(dtype="bfloat16", backward=False)
        with pytest.raises(ValueError, match="backward"):
            observed.require_complete(dtype="bfloat16", backward=True)
    receipt = observed.report()
    assert receipt["instrumentation"]["retains_tensors"] is False
    assert receipt["samples"][0]["backward"]["operators"] == []


@pytest.mark.gpu
def test_observer_drops_attention_tensors_and_does_not_seed_cuda_rng() -> None:
    import gc
    import weakref

    torch = _ampere_stack()
    previous_rng = torch.cuda.get_rng_state()
    try:
        with attention.attention_execution("EFFICIENT_ATTENTION", device="cuda") as observed:
            qkv = [torch.randn(1, 4, 64, dimension, device="cuda", dtype=torch.bfloat16,
                               requires_grad=True) for dimension in (192, 192, 128)]
            output = torch.nn.functional.scaled_dot_product_attention(*qkv, is_causal=True)
            output.sum().backward()
            observed.require_complete(dtype="bfloat16")
            references = [weakref.ref(tensor) for tensor in (*qkv, output)]
            del qkv, output
            gc.collect()
            assert all(reference() is None for reference in references)
        before = torch.cuda.get_rng_state()
        with attention.attention_execution("EFFICIENT_ATTENTION", device="cuda"):
            assert torch.equal(before, torch.cuda.get_rng_state())
            torch.rand(3, device="cuda")
        assert not torch.equal(before, torch.cuda.get_rng_state())
    finally:
        torch.cuda.set_rng_state(previous_rng)
