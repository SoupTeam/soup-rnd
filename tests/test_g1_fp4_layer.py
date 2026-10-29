"""Tests for the G1 single-layer harness (benchmarks/harness/g1_fp4_layer.py, G1 ticket 02).

No test touches a GPU, TransformerEngine or bitsandbytes. Numbers are synthetic.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "harness"


@pytest.fixture(scope="module")
def g1():
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    name = "_g1_fp4_layer_test_module"
    spec = importlib.util.spec_from_file_location(name, HARNESS / "g1_fp4_layer.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_shape_multiple_of_16_is_accepted(g1):
    g1.check_fp4_shape(tokens=256, in_features=512, out_features=1024)


@pytest.mark.parametrize(
    "shape, bad",
    [
        ((250, 512, 512), "tokens=250"),
        ((256, 520, 512), "in_features=520"),
        ((256, 4112, 512), "in_features=4112"),
        ((256, 512, 1000), "out_features=1000"),
    ],
)
def test_shape_off_its_multiple_is_refused_with_a_clear_message(g1, shape, bad):
    tokens, k, n = shape
    with pytest.raises(g1.UnsupportedShapeError) as excinfo:
        g1.check_fp4_shape(tokens=tokens, in_features=k, out_features=n)
    message = str(excinfo.value)
    assert bad in message
    assert "multiple of" in message
    assert "NF4" in message and "no fallback" in message


def test_refusal_is_a_value_error(g1):
    assert issubclass(g1.UnsupportedShapeError, ValueError)


def test_padding_cost_of_an_aligned_shape_is_zero(g1):
    cost = g1.padding_cost(tokens=256, in_features=512, out_features=1024)
    assert cost["padded"] == {"tokens": 256, "in_features": 512, "out_features": 1024}
    assert cost["flop_overhead"] == 0.0
    assert cost["weight_overhead"] == 0.0


def test_padding_cost_counts_the_extra_flops_and_weight_elements(g1):
    # 136 -> 256 tokens, 192 -> 256 inputs, 128 stays. Worked by hand:
    # flops 256*256*128 / (136*192*128) = 2.5098..., weights 256*128 / (192*128) = 1.3333...
    cost = g1.padding_cost(tokens=136, in_features=192, out_features=128)
    assert cost["padded"] == {"tokens": 256, "in_features": 256, "out_features": 128}
    assert cost["flop_overhead"] == pytest.approx(65536 / 26112 - 1)
    assert cost["weight_overhead"] == pytest.approx(1 / 3)


def _run(arm, value, outcome="ok", rel_error=0.1, cosine=0.99):
    return {
        "arm": arm,
        "value": value,
        "outcome": outcome,
        "error": {"rel_error": rel_error, "cosine": cosine},
    }


def _block(a1, b1, b2, a2, outcome="ok"):
    return [_run("A", a1, outcome), _run("B", b1), _run("B", b2), _run("A", a2)]


def test_report_averages_valid_blocks_and_is_labelled_synthetic(g1):
    blocks = [_block(1.0, 2.0, 2.0, 1.0), _block(3.0, 4.0, 4.0, 3.0)]
    report = g1.build_report(blocks, shape={"tokens": 256}, padding=[])
    assert report["data_kind"] == "synthetic"
    assert "synthetic" in report["label"]
    assert report["arms"]["A"]["path"] == "native_fp4"
    assert report["arms"]["B"]["path"] == "nf4_dequant_bf16"
    assert report["arms"]["A"]["mean_seconds"] == pytest.approx(2.0)
    assert report["arms"]["B"]["mean_seconds"] == pytest.approx(3.0)
    assert report["arms"]["A"]["runs"] == 4
    assert report["valid_blocks"] == 2 and report["void_blocks"] == 0


def test_report_leaves_out_a_block_with_a_void_run(g1):
    blocks = [_block(1.0, 2.0, 2.0, 1.0), _block(100.0, 2.0, 2.0, 1.0, outcome="void")]
    report = g1.build_report(blocks, shape={}, padding=[])
    assert report["arms"]["A"]["mean_seconds"] == pytest.approx(1.0)
    assert report["valid_blocks"] == 1 and report["void_blocks"] == 1


def test_report_refuses_to_report_without_a_valid_block(g1):
    with pytest.raises(ValueError, match="no block without a void run"):
        g1.build_report([_block(1, 2, 2, 1, outcome="void")], shape={}, padding=[])


def test_report_carries_each_arms_error_against_bf16(g1):
    blocks = [
        [
            _run("A", 1, rel_error=0.12, cosine=0.99),
            _run("B", 2, rel_error=0.09, cosine=0.995),
            _run("B", 2, rel_error=0.09, cosine=0.995),
            _run("A", 1, rel_error=0.12, cosine=0.99),
        ]
    ]
    report = g1.build_report(blocks, shape={}, padding=[])
    assert report["arms"]["A"]["rel_error"] == pytest.approx(0.12)
    assert report["arms"]["B"]["cosine"] == pytest.approx(0.995)


def test_fp4_run_with_a_bad_shape_is_refused_before_any_work(g1):
    # No torch or TransformerEngine is imported on this path, and there is no NF4 fallback.
    with pytest.raises(g1.UnsupportedShapeError, match="no fallback"):
        g1.run_layer(path="fp4", tokens=250, in_features=512, out_features=512, steps=3, warmup=1)


def test_in_features_needs_32_but_the_other_dimensions_only_16(g1):
    # Measured on an RTX 5090 (sm_120): cuBLAS finds no FP4 algorithm for K=4112 but runs
    # tokens=4112 and out_features=4112. TransformerEngine's own check passes all three.
    g1.check_fp4_shape(tokens=4112, in_features=4128, out_features=4112)
    refused = r"in_features=4112 \(needs a multiple of 32\)"
    with pytest.raises(g1.UnsupportedShapeError, match=refused):
        g1.check_fp4_shape(tokens=4096, in_features=4112, out_features=4096)
