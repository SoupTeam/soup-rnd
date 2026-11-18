"""NF4 Triton dequantization (soup_cli.utils.nf4_triton) against bitsandbytes.

On CPU the kernel runs in the Triton interpreter (TRITON_INTERPRET=1, set in a subprocess so it
is in place before triton is imported). The GPU check (EXACT_MUL, mul.rn) runs in the ZERO-2
stage-2b runner.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("bitsandbytes")
pytest.importorskip("triton")

_CHECK = textwrap.dedent(
    """
    import torch, bitsandbytes.functional as F
    from soup_cli.utils import nf4_triton as nt
    torch.manual_seed(0)
    cases = [((96, 128), 64, True, torch.bfloat16), ((64, 192), 64, True, torch.float16),
             ((32, 256), 128, False, torch.bfloat16), ((48, 64), 64, True, torch.float32),
             ((300, 70), 64, True, torch.bfloat16)]
    for shape, bs, nested, dt in cases:
        w = (torch.randn(shape) * 0.05).to(dt)
        packed, state = F.quantize_4bit(w, blocksize=bs, compress_statistics=nested,
                                        quant_type="nf4", quant_storage=torch.uint8)
        ref = F.dequantize_4bit(packed, state)
        assert nt.covered(packed, state, None), shape
        got = nt.dequantize_nf4(packed, state)
        assert got.dtype == ref.dtype and got.shape == ref.shape, (shape, got.dtype, got.shape)
        assert torch.equal(got.view(torch.int16 if dt != torch.float32 else torch.int32),
                           ref.view(torch.int16 if dt != torch.float32 else torch.int32)), shape
        out = torch.empty_like(ref)
        nt.enable()
        try:
            got2 = F.dequantize_4bit(packed, state, out=out)
        finally:
            nt.disable()
        assert got2.data_ptr() == out.data_ptr() and torch.equal(got2, ref), shape
    # the forward entry point: bitsandbytes' _dequant_linear_fallback signature
    w = (torch.randn(96, 128) * 0.05).to(torch.bfloat16)
    packed, st = F.quantize_4bit(w, blocksize=64, compress_statistics=True, quant_type="nf4",
                                 quant_storage=torch.uint8)
    x = torch.randn(5, 128).to(torch.bfloat16)
    ref = torch.nn.functional.linear(x, F.dequantize_4bit(packed, st))
    got = nt.dequant_linear(x, packed.view(-1, 1), st.shape, st.state2.absmax, st.blocksize, "nf4",
                            absmax_8bit=st.absmax, absmax_code=st.state2.code,
                            absmax_offset=st.offset)
    assert torch.equal(got, ref)
    packed, st = F.quantize_4bit(w, blocksize=64, compress_statistics=False, quant_type="nf4")
    ref = torch.nn.functional.linear(x, F.dequantize_4bit(packed, st))
    assert torch.equal(nt.dequant_linear(x, packed, st.shape, st.absmax, 64, "nf4"), ref)
    print("NF4_OK")
    """
)


def test_bit_identical_to_bitsandbytes_in_the_interpreter():
    env = {**os.environ, "TRITON_INTERPRET": "1"}
    p = subprocess.run([sys.executable, "-c", _CHECK], capture_output=True, text=True, env=env,
                       timeout=600)
    assert "NF4_OK" in p.stdout, p.stdout[-2000:] + p.stderr[-4000:]


def test_falls_back_for_fp4_and_cpu_without_interpreter(monkeypatch):
    import torch
    import bitsandbytes.functional as F

    from soup_cli.utils import nf4_triton as nt

    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    w = torch.randn(16, 64).to(torch.bfloat16)
    packed, state = F.quantize_4bit(w, quant_type="fp4")
    assert not nt.covered(packed, state, None)
    packed, state = F.quantize_4bit(w, quant_type="nf4")
    assert not nt.covered(packed, state, None)  # CPU tensor, no interpreter
    nt.enable()
    try:
        assert torch.equal(F.dequantize_4bit(packed, state), nt._original(packed, state))
    finally:
        nt.disable()
    assert F.dequantize_4bit is not nt.dequantize_4bit


def test_enable_from_env(monkeypatch):
    from soup_cli.utils import nf4_triton as nt

    monkeypatch.delenv(nt.ENV, raising=False)
    assert nt.enable_from_env() is False and not nt.enabled()
    monkeypatch.setenv(nt.ENV, "1")
    try:
        assert nt.enable_from_env() is True and nt.enabled()
        assert nt.enable() is True  # idempotent
    finally:
        nt.disable()
    assert not nt.enabled()
