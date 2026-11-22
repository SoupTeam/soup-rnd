"""The NF4 pair decoder must equal bitsandbytes 0.50.2 on ``uint16``, bit for bit.

Red-capable scaffold: it skips until the vendored ``_native_expert_stream_async`` module is
built with ``--with-cuda-nf4`` on a CUDA host. Run it on the card with
``pytest tests/test_nf4_decoder_bnb.py -m gpu --no-cov -v``. Any differing bit is a failure
that names the code byte, scale and output index; a benchmark never replaces this equality.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Tuple

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu

BLOCKSIZE = 64
BNB_VERSION = "0.50.2"


def _extension_or_skip() -> Any:
    # Skip only when the module does not exist at all. A module that exists but fails to
    # import (broken .so, ABI mismatch, missing libtorch) must fail the gate, not skip it.
    if importlib.util.find_spec("_native_expert_stream_async") is None:
        pytest.skip(
            "_native_expert_stream_async is not built; run "
            "python src/soup_cli/utils/_native_expert_async/build_ext.py "
            "build_ext --build-lib /tmp/native-ext --with-cuda-nf4"
        )
    import _native_expert_stream_async as extension

    if not hasattr(extension, "decode_nf4_bf16_pair"):
        pytest.fail("the extension was built without --with-cuda-nf4 (no NF4 kernel)")
    return extension


def _bitsandbytes() -> Tuple[Any, Any]:
    bitsandbytes = pytest.importorskip("bitsandbytes")
    assert bitsandbytes.__version__ == BNB_VERSION, (
        f"this gate is defined against bitsandbytes {BNB_VERSION}, "
        f"got {bitsandbytes.__version__}"
    )
    from bitsandbytes import functional

    return bitsandbytes, functional


def _lookup(device: Any) -> Any:
    _, functional = _bitsandbytes()
    return functional.get_4bit_type("nf4", device=device)


def _reference(codes: Any, scales: Any) -> Any:
    """bitsandbytes' own BF16 dequantisation, as ``uint16`` bits."""
    _, functional = _bitsandbytes()
    count = int(codes.numel()) * 2
    state = functional.QuantState(
        absmax=scales,
        code=_lookup(codes.device),
        shape=(count, 1),
        dtype=torch.bfloat16,
        blocksize=BLOCKSIZE,
        quant_type="nf4",
    )
    dequantised = functional.dequantize_4bit(codes.view(-1, 1), quant_state=state)
    return dequantised.view(torch.uint16).flatten()


def _decode_pair(
    extension: Any,
    gate_codes: Any,
    gate_scales: Any,
    down_codes: Any,
    down_scales: Any,
) -> Tuple[Any, Any]:
    """Run the extension on one gate block and one down block, or either alone."""
    device = gate_codes.device if gate_codes is not None else down_codes.device
    gate_bits = (
        torch.empty(int(gate_codes.numel()) * 2, dtype=torch.uint16, device=device)
        if gate_codes is not None
        else torch.empty(0, dtype=torch.uint16, device=device)
    )
    down_bits = (
        torch.empty(int(down_codes.numel()) * 2, dtype=torch.uint16, device=device)
        if down_codes is not None
        else torch.empty(0, dtype=torch.uint16, device=device)
    )
    empty = torch.empty(0, dtype=torch.uint8, device=device)
    extension.decode_nf4_bf16_pair(
        device.index,
        torch.cuda.current_stream(device).cuda_stream,
        _lookup(device).data_ptr(),
        gate_codes.data_ptr() if gate_codes is not None else empty.data_ptr(),
        gate_scales.data_ptr() if gate_scales is not None else empty.data_ptr(),
        gate_bits.data_ptr(),
        int(gate_codes.numel()) if gate_codes is not None else 0,
        down_codes.data_ptr() if down_codes is not None else empty.data_ptr(),
        down_scales.data_ptr() if down_scales is not None else empty.data_ptr(),
        down_bits.data_ptr(),
        int(down_codes.numel()) if down_codes is not None else 0,
    )
    torch.cuda.synchronize(device)
    return gate_bits, down_bits


def _first_mismatch(expected: Any, actual: Any) -> str:
    if torch.equal(expected, actual):
        return ""
    differing = (expected != actual).nonzero().flatten()
    index = int(differing[0])
    return (
        f"{differing.numel()} of {expected.numel()} bits differ; first at output index "
        f"{index} (code byte {index // 2}, nibble {'high' if index % 2 == 0 else 'low'}): "
        f"bnb {int(expected[index])} != extension {int(actual[index])}"
    )


def _codes_covering_every_code(count: int, device: Any) -> Any:
    """Random packed bytes with the first 16 bytes spelling every NF4 code in both nibbles."""
    codes = torch.randint(0, 256, ((count + 1) // 2, 1), dtype=torch.uint8, device=device)
    codes[:16, 0] = torch.arange(16, dtype=torch.int64, device=device).to(torch.uint8) * 17
    return codes


def _random_scales(count: int, device: Any) -> Any:
    return torch.rand(count // BLOCKSIZE, dtype=torch.float32, device=device) * 2.0


def test_random_packets_cover_every_nf4_code() -> None:
    extension = _extension_or_skip()
    torch.manual_seed(20261123)
    count = BLOCKSIZE * 16
    codes = _codes_covering_every_code(count, "cuda")
    scales = _random_scales(count, "cuda")
    (gate_bits, _) = _decode_pair(extension, codes, scales, None, None)
    expected = _reference(codes, scales)
    assert _first_mismatch(expected, gate_bits) == ""


def test_boundary_scales_match_bitsandbytes() -> None:
    extension = _extension_or_skip()
    scales = torch.tensor(
        [0.0, -0.0, 1e-45, 1e-38, 1.0, -1.0, float("inf"), float("nan")],
        dtype=torch.float32,
        device="cuda",
    )
    count = int(scales.numel()) * BLOCKSIZE
    codes = _codes_covering_every_code(count, "cuda")
    (gate_bits, _) = _decode_pair(extension, codes, scales, None, None)
    expected = _reference(codes, scales)
    assert _first_mismatch(expected, gate_bits) == ""


def _quantised_matrix(shape: Tuple[int, int], device: Any) -> Tuple[Any, Any]:
    _, functional = _bitsandbytes()
    source = torch.randn(*shape, dtype=torch.float32, device=device)
    codes, state = functional.quantize_4bit(
        source, blocksize=BLOCKSIZE, compress_statistics=False, quant_type="nf4"
    )
    assert not state.nested, "plain absmax is the wired expert format"
    return codes.contiguous(), state.absmax.contiguous()


def test_real_gate_up_and_down_blocks_match_bitsandbytes() -> None:
    extension = _extension_or_skip()
    torch.manual_seed(77)
    hidden = 256
    gate_codes, gate_scales = _quantised_matrix((2 * hidden, hidden), "cuda")
    down_codes, down_scales = _quantised_matrix((hidden, 2 * hidden), "cuda")
    gate_bits, down_bits = _decode_pair(
        extension, gate_codes, gate_scales, down_codes, down_scales
    )
    mismatch = _first_mismatch(_reference(gate_codes, gate_scales), gate_bits)
    assert mismatch == "", f"gate_up: {mismatch}"
    mismatch = _first_mismatch(_reference(down_codes, down_scales), down_bits)
    assert mismatch == "", f"down: {mismatch}"
