"""C5 ticket 01: the Q2_0 expert-weight codec (benchmarks/harness/q2_0_codec.py).

Spec: .scratch/c5-q2-0-experts/issues/01-q2-0-codec.md. The byte layout is
llama.cpp's ``GGML_TYPE_Q2_0``; the compatibility tests read our blocks back
with gguf-py and skip when the installed ``gguf`` has no Q2_0 (PyPI gguf 0.19.0
predates it).
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"


@pytest.fixture(scope="module")
def codec():
    path = HARNESS / "q2_0_codec.py"
    name = "_q2_0_codec_test_module"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _grid_block(scale: float) -> torch.Tensor:
    """A 64-weight block that uses all four levels {-1, 0, 1, 2} x scale."""
    levels = torch.tensor([-1.0, 0.0, 1.0, 2.0]).repeat(16)
    return levels * scale


class TestRoundTrip:
    @pytest.mark.parametrize("scale", [0.0123, -0.0123, 3.5, -1.0e-3])
    def test_grid_block_restores_exactly(self, codec, scale):
        # The fp16-representable scale is what a block on the grid must carry.
        s = float(torch.tensor(scale, dtype=torch.float16))
        block = _grid_block(s)
        packed = codec.quantize(block, "q2_0")
        assert torch.equal(codec.dequantize(packed), block)
        assert packed.tensors["d"].item() == s

    def test_zero_block_has_zero_scale_and_code_one(self, codec):
        packed = codec.quantize(torch.zeros(64), "q2_0")
        assert packed.tensors["d"].tolist() == [0.0]
        assert packed.tensors["qs"].flatten().tolist() == [0x55] * 16
        assert torch.equal(codec.dequantize(packed), torch.zeros(64))

    def test_shape_not_a_multiple_of_64_is_restored(self, codec):
        # 100 grid weights: a full block, then 36 weights and 28 zeros of padding.
        s = float(torch.tensor(0.75, dtype=torch.float16))
        weight = torch.cat([_grid_block(s), _grid_block(-s)[:36]]).reshape(10, 10)
        restored = codec.dequantize(codec.quantize(weight, "q2_0"))
        assert restored.dtype == torch.float32
        assert torch.equal(restored, weight)

    @pytest.mark.parametrize("shape", [(100,), (64,), (3, 5, 7)])
    @pytest.mark.parametrize("name", ["q2_0", "nf4"])
    def test_every_codec_restores_the_input_shape(self, codec, name, shape):
        if name == "nf4":
            pytest.importorskip("bitsandbytes")
        weight = torch.randn(shape)
        assert codec.dequantize(codec.quantize(weight, name)).shape == shape


class TestRefusals:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    @pytest.mark.parametrize("name", ["q2_0", "q2_0_absmax", "nf4"])
    def test_nan_or_inf_input_is_refused(self, codec, bad, name):
        weight = torch.randn(128)
        weight[70] = bad
        with pytest.raises(ValueError, match="NaN or Inf"):
            codec.quantize(weight, name)

    @pytest.mark.parametrize("big", [65505.0, 1.0e5, 2.0e5])
    @pytest.mark.parametrize("name", ["q2_0", "q2_0_absmax"])
    def test_scale_beyond_fp16_is_refused(self, codec, name, big):
        # absmax is itself a candidate d; the search must not fall back to a
        # smaller candidate that fits fp16 (1e5 / 2 = 50000 would).
        weight = torch.zeros(128)
        weight[100] = big
        with pytest.raises(ValueError, match="fp16"):
            codec.quantize(weight, name)

    @pytest.mark.parametrize("name", ["q2_0", "q2_0_absmax"])
    def test_scale_at_fp16_max_is_accepted(self, codec, name):
        weight = torch.zeros(64)
        weight[0] = -65504.0
        assert codec.dequantize(codec.quantize(weight, name))[0].item() == -65504.0

    def test_unknown_codec_is_refused(self, codec):
        with pytest.raises(ValueError, match="unknown codec"):
            codec.quantize(torch.zeros(64), "q4_0")


class TestAbsmaxReference:
    def test_scale_is_absmax_and_ties_round_away_from_zero(self, codec):
        # llama.cpp: d = absmax = 2, so 2 / 2 = 1 is code 2 (never 3 under
        # absmax), 1 / 2 = 0.5 rounds to 1 (code 2) and -0.5 to -1 (code 0)
        # under roundf; torch.round would give 0 for both halves. Weight 0
        # sits in the low bits.
        block = torch.zeros(64)
        block[:4] = torch.tensor([2.0, 1.0, -1.0, 0.0])
        packed = codec.quantize(block, "q2_0_absmax")
        assert packed.tensors["d"].tolist() == [2.0]
        assert packed.tensors["qs"][0, 0].item() == 0b01_00_10_10
        assert codec.dequantize(packed)[:4].tolist() == [2.0, 2.0, -2.0, 0.0]


@pytest.fixture(scope="module")
def gguf_q2_0():
    gguf = pytest.importorskip("gguf")
    if not hasattr(gguf.GGMLQuantizationType, "Q2_0"):
        pytest.skip("installed gguf has no Q2_0 (added to gguf-py 2026-10-16, PR #30630)")
    return gguf


class TestLlamaCppCompatibility:
    def test_gguf_dequant_of_our_blocks_matches_ours(self, codec, gguf_q2_0):
        # A shape with a tail and blocks of both scale signs.
        weight = torch.randn(7, 100, generator=torch.Generator().manual_seed(1)) * 0.02
        packed = codec.quantize(weight, "q2_0")
        assert (packed.tensors["d"] < 0).any() and (packed.tensors["d"] > 0).any()
        blocks = codec.to_block_q2_0(packed)
        assert blocks.dtype == torch.uint8 and blocks.shape == (11, 18)
        theirs = gguf_q2_0.quants.dequantize(
            blocks.reshape(-1).numpy(), gguf_q2_0.GGMLQuantizationType.Q2_0
        )
        ours = torch.zeros(11 * 64)
        ours[:700] = codec.dequantize(packed).reshape(-1)
        assert torch.equal(torch.from_numpy(theirs).reshape(-1), ours)

    def test_absmax_reference_is_byte_identical_to_gguf_quantize(self, codec, gguf_q2_0):
        weight = torch.randn(32, 64, generator=torch.Generator().manual_seed(2))
        weight[0, :4] = torch.tensor([2.0, 1.0, -1.0, 0.0])  # ties at +-0.5
        weight[1] = 0.0
        ours = codec.to_block_q2_0(codec.quantize(weight, "q2_0_absmax"))
        theirs = gguf_q2_0.quants.quantize(weight.numpy(), gguf_q2_0.GGMLQuantizationType.Q2_0)
        assert ours.reshape(-1).tolist() == theirs.reshape(-1).tolist()


class TestReconstructionError:
    def test_one_measurement_for_every_codec_on_a_gaussian_matrix(self, codec):
        pytest.importorskip("bitsandbytes")
        weight = torch.randn(512, 1024, generator=torch.Generator().manual_seed(3))
        rel = {}
        for name in codec.CODECS:
            error = codec.reconstruction_error(weight, codec.quantize(weight, name))
            assert error["mse"] == pytest.approx(error["rel_mse"] * weight.pow(2).mean().item())
            rel[name] = error["rel_mse"]
        # The research simulation (200 000 Gaussian blocks) gave 0.515 for
        # absmax and 0.127 for a signed search over 122 candidates; ours
        # searches a superset, so it can only do as well or better.
        assert 0.50 < rel["q2_0_absmax"] < 0.53
        assert rel["q2_0"] < 0.13
        assert rel["nf4"] < 0.01
        assert rel["nf4"] < rel["q2_0"] < rel["q2_0_absmax"]


class TestCodecSha256:
    def test_is_the_hash_of_the_file_bytes(self, codec):
        expected = hashlib.sha256((HARNESS / "q2_0_codec.py").read_bytes()).hexdigest()
        assert codec.codec_sha256() == expected

    def test_file_is_checked_out_with_lf_on_every_os(self):
        # The hash goes into the C5 rule (ticket 03); CRLF on Windows would change it.
        attrs = (HARNESS.parents[1] / ".gitattributes").read_text().splitlines()
        assert "benchmarks/harness/q2_0_codec.py text eol=lf" in attrs


class TestBytes:
    @staticmethod
    def _stored_bytes(packed) -> int:
        return sum(t.numel() * t.element_size() for t in packed.tensors.values())

    def test_q2_0_is_2_25_bits_per_weight_when_n_divides_by_64(self, codec):
        packed = codec.quantize(torch.randn(128, 256), "q2_0")
        assert packed.tensors["d"].dtype == torch.float16
        assert packed.tensors["qs"].dtype == torch.uint8
        assert self._stored_bytes(packed) == 9216  # 18 * 32768 / 64
        assert self._stored_bytes(packed) * 8 / (128 * 256) == 2.25

    def test_q2_0_pads_the_tail_to_a_whole_block(self, codec):
        packed = codec.quantize(torch.randn(10, 10), "q2_0")
        assert self._stored_bytes(packed) == 36  # 18 * ceil(100 / 64)

    def test_nf4_bytes_are_the_bitsandbytes_tensors(self, codec):
        bnb = pytest.importorskip("bitsandbytes.functional")
        weight = torch.randn(256, 256)
        packed = codec.quantize(weight, "nf4")
        state = packed.state
        assert state.nested and state.blocksize == 64
        assert packed.tensors["data"] is not None
        # 4-bit codes 32768 + uint8 absmax 1024 + fp32 offset 4 + fp32 absmax
        # of the absmax, one per 256 blocks, 16. The 16- and 256-entry lookup
        # tables are constants of the format and are not per-tensor storage.
        assert self._stored_bytes(packed) == 33812
        restored = codec.dequantize(packed)
        assert restored.shape == (256, 256) and restored.dtype == torch.float32
        assert torch.equal(restored, bnb.dequantize_4bit(packed.tensors["data"], state).float())
