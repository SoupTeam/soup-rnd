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
