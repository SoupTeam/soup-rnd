"""CPU test for the float64 accuracy microtest (ticket 04): native vs row-wise expert GEMM."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks" / "harness"))

from p72_fp64_accuracy import (  # noqa: E402
    classify,
    measure_gemm_accuracy,
    run_report,
)


def _expert(rows=24, d_in=64, d_out=48, dtype=torch.float32):
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(rows, d_in, generator=gen).to(dtype)
    w = (torch.randn(d_out, d_in, generator=gen) / d_in**0.5).to(dtype)
    return x, w


def test_errors_are_computed_against_float64_for_both_variants():
    x, w = _expert()
    res = measure_gemm_accuracy(x, w, portion_rows=8)
    assert set(res) == {"native", "invariant"}
    for arm in res.values():
        assert arm["max_abs"] > 0.0 and arm["mean_abs"] > 0.0
        assert arm["max_abs"] >= arm["mean_abs"]
        assert arm["max_abs"] < 1e-4  # fp32 against fp64, sanity bound


def test_float64_input_has_zero_error_in_both_variants():
    x, w = _expert(dtype=torch.float64)
    res = measure_gemm_accuracy(x, w, portion_rows=8)
    assert res["native"]["max_abs"] < 1e-12
    assert res["invariant"]["max_abs"] < 1e-12


def test_bf16_error_is_larger_than_fp32():
    x, w = _expert()
    fp32 = measure_gemm_accuracy(x, w, portion_rows=8)["native"]["mean_abs"]
    bf16 = measure_gemm_accuracy(x.bfloat16(), w.bfloat16(), portion_rows=8)["native"]["mean_abs"]
    assert bf16 > fp32 * 10


def test_portion_must_be_positive():
    x, w = _expert()
    with pytest.raises(ValueError):
        measure_gemm_accuracy(x, w, portion_rows=0)


@pytest.mark.parametrize(
    "native,invariant,verdict",
    [
        (1.0, 0.5, "invariant точнее"),
        (1.0, 2.0, "invariant не точнее"),
        (1.0, 1.05, "в пределах шума"),
    ],
)
def test_classify(native, invariant, verdict):
    assert classify(native, invariant) == verdict


def test_report_has_metadata_and_a_verdict():
    x, w = _expert()
    report = run_report(x, w, portion_rows=8, source="synthetic")
    assert report["dtype"] == "torch.float32"
    assert report["device"] == "cpu"
    assert report["torch"] == torch.__version__
    assert report["source"] == "synthetic"
    assert report["verdict"] in {"invariant точнее", "invariant не точнее", "в пределах шума"}
