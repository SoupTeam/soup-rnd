"""CPU tests of candidate checkpoint state, not claims about CUDA arithmetic."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks/harness"))


@pytest.fixture
def candidate(monkeypatch):
    import d1_fused_nf4 as module
    import torch

    def oracle(value, saved, metadata, transpose):
        packed, absmax, code, state2_absmax, state2_code, offset = saved
        n, k, blocksize, nested, state2_block, weight_dtype = metadata
        positions = torch.arange(n * k)
        byte = packed.flatten()[positions // 2].long()
        nibble = torch.where(positions % 2 == 0, byte >> 4, byte & 15)
        if nested:
            absmax = (
                state2_code[absmax.long()]
                * state2_absmax[torch.arange(absmax.numel()) // state2_block]
                + offset
            )
        weight = (code[nibble] * absmax[positions // blocksize]).to(weight_dtype).view(n, k)
        return value @ weight if transpose else torch.nn.functional.linear(value, weight)

    # Isolate the state/checkpoint wrapper; the actual Triton kernel has its
    # separate interpreter/physical-CUDA screen and is not mocked as "passed".
    monkeypatch.setattr(module, "_matmul", oracle)
    return module


def state():
    import torch

    return SimpleNamespace(
        quant_type="nf4",
        shape=(4, 8),
        blocksize=8,
        dtype=torch.float32,
        code=torch.arange(16, dtype=torch.float32) / 16,
        absmax=torch.tensor([1, 2, 3, 4], dtype=torch.uint8),
        nested=True,
        state2=SimpleNamespace(
            blocksize=256,
            absmax=torch.tensor([2.0]),
            code=torch.arange(256, dtype=torch.float32) / 256,
        ),
        offset=torch.tensor([0.25]),
    )


@pytest.mark.parametrize(
    "field", ["packed", "absmax", "code", "state2_absmax", "state2_code", "offset"]
)
def test_each_backward_tensor_is_saved_and_version_checked(candidate, field):
    import torch

    quant = state()
    packed = torch.full((16, 1), 0xA5, dtype=torch.uint8)
    value = torch.ones(2, 8, requires_grad=True)
    output = candidate.linear(value, packed, quant)
    tensors = {
        "packed": packed,
        "absmax": quant.absmax,
        "code": quant.code,
        "state2_absmax": quant.state2.absmax,
        "state2_code": quant.state2.code,
        "offset": quant.offset,
    }
    tensors[field].add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.sum().backward()


def test_checkpoint_reloads_recycled_packed_buffer(candidate):
    import torch
    from torch.utils.checkpoint import checkpoint

    quant = state()
    packed = torch.full((16, 1), 0xA5, dtype=torch.uint8)
    reference_x = torch.ones(2, 8, requires_grad=True)
    reference = candidate.linear(reference_x, packed.clone(), quant)
    reference.sum().backward()
    pool = packed.clone()
    value = reference_x.detach().clone().requires_grad_(True)
    calls = []

    def forward(current):
        calls.append(True)
        pool.copy_(packed)
        return candidate.linear(current, pool, quant)

    result = checkpoint(forward, value, use_reentrant=False)
    pool.fill_(0x11)
    result.sum().backward()
    assert len(calls) == 2
    assert torch.equal(result, reference)
    assert torch.equal(value.grad, reference_x.grad)


def test_backward_checks_stream_slot_ownership(candidate):
    import torch

    def stale():
        raise RuntimeError("stream slot reused")

    packed = torch.full((16, 1), 0xA5, dtype=torch.uint8)
    packed._soup_stream_owner_check = stale
    output = candidate.linear(torch.ones(2, 8, requires_grad=True), packed, state())
    with pytest.raises(RuntimeError, match="stream slot reused"):
        output.sum().backward()


def test_invalid_nested_extent_is_rejected_before_kernel(candidate):
    import torch

    quant = state()
    quant.state2.absmax = torch.empty(0)
    with pytest.raises(ValueError, match="invalid nested"):
        candidate.linear(torch.ones(2, 8), torch.empty((16, 1), dtype=torch.uint8), quant)


def test_candidate_never_silently_falls_back_to_cpu(monkeypatch):
    import d1_fused_nf4
    import torch

    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    with pytest.raises(RuntimeError, match="CUDA required"):
        d1_fused_nf4.linear(torch.ones(2, 8), torch.full((16, 1), 0xA5, dtype=torch.uint8), state())


def test_wmma_dispatch_rejects_reduced_precision_without_changing_global_mode(monkeypatch):
    import d1_fused_nf4
    import torch

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", True)
    with pytest.raises(RuntimeError, match="requires full FP32"):
        d1_fused_nf4._dispatch(None, (), (4, 8, 8, True, 256, torch.float32, "wmma"), False)
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is True


def test_backend_is_saved_per_autograd_call_instead_of_global_monkeypatch(candidate, monkeypatch):
    import d1_wmma_nf4
    import torch

    oracle = candidate._matmul
    calls = []

    def wmma_oracle(value, saved, metadata, transpose):
        calls.append(transpose)
        return oracle(value, saved, metadata, transpose)

    def forbidden(*_args):
        pytest.fail("WMMA autograd unexpectedly used the default Triton backend")

    monkeypatch.setattr(d1_wmma_nf4, "matmul", wmma_oracle)
    monkeypatch.setattr(candidate, "_matmul", forbidden)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", False)
    quant = state()
    saved = (
        torch.full((16, 1), 0xA5, dtype=torch.uint8),
        quant.absmax,
        quant.code,
        quant.state2.absmax,
        quant.state2.code,
        quant.offset,
    )
    metadata = (4, 8, 8, True, 256, torch.float32, "wmma")
    value = torch.ones(2, 8, requires_grad=True)
    # CPU oracle checks routing/state only, never CUDA arithmetic.
    output = candidate._function().apply(value, *saved, metadata)
    output.sum().backward()
    assert calls == [False, True]
    assert value.grad is not None


@pytest.mark.parametrize("failed", ["comparison", "decode"])
def test_cuda_screen_stops_at_first_failure_before_alias_or_timing(monkeypatch, failed):
    import d1_kernel_quick_gate as screen
    import torch

    calls = []

    def fail_case(*_args):
        calls.append(True)
        return {
            "comparison": {"passed": failed != "comparison"},
            "decode": {"passed": failed != "decode"},
        }

    def forbidden(*_args):
        pytest.fail("screen continued after a failed numerical gate")

    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace(__version__="unit-test-only"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _i: SimpleNamespace(uuid="fake")
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _i: "unit-test-only")
    monkeypatch.setattr(torch, "set_num_threads", lambda _n: None)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda _flag: None)
    monkeypatch.setattr(screen, "numerical_case", fail_case)
    monkeypatch.setattr(screen, "alias_case", forbidden)
    result = screen.run("cuda")
    assert len(calls) == 1
    assert result["status"] == "SMALL_CUDA_GATE_FAIL"
    assert result["small_alias"] == {"status": "NOT_RUN"}
    assert result["performance_measured"] is False
    assert result["d1_complete"] is False
