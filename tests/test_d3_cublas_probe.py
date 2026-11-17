"""Column-major binding contract and development-only CLI behavior."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "benchmarks/harness/d3_cublas_grouped_probe.py"
    spec = importlib.util.spec_from_file_location("test_cublas_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("rows,inner,columns", [(1, 33, 47), (17, 32, 64), (3, 11, 7)])
def test_descriptors_preserve_physical_row_major_linear_without_packing(rows, inner, columns):
    import numpy as np

    rule = _module().column_major_problem(rows, inner, columns)
    hidden = np.arange(rows * inner).reshape(rows, inner)
    weight = np.arange(columns * inner).reshape(columns, inner)
    stored_a = weight.reshape(-1).reshape(rule["lda"], rule["m"], order="F")
    stored_b = hidden.reshape(-1).reshape(rule["ldb"], rule["n"], order="F")
    result = stored_a.T @ stored_b
    row_major = result.reshape(-1, order="F").reshape(rows, columns)
    assert np.array_equal(row_major, hidden @ weight.T)
    assert rule["transa"] == 1 and rule["transb"] == 0


@pytest.mark.parametrize("bad", [0, -1, True, 2.0, 2**31])
def test_dimensions_cannot_overflow_c_api(bad):
    with pytest.raises(ValueError, match="32-bit"):
        _module().column_major_problem(bad, 32, 64)


def test_probe_import_does_not_load_cuda_libraries():
    path = _module().__file__
    code = (
        "import importlib.util,sys; "
        f"s=importlib.util.spec_from_file_location('probe', {path!r}); "
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m); "
        "assert not any(x in sys.modules for x in ('torch','triton','bitsandbytes'))"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_reference_policy_restores_blas_and_both_flags_on_failure(monkeypatch):
    from types import SimpleNamespace

    import torch

    class Matmul:
        flags = (True, True)

        @property
        def allow_bf16_reduced_precision_reduction(self):
            return self.flags[0]

        @allow_bf16_reduced_precision_reduction.setter
        def allow_bf16_reduced_precision_reduction(self, value):
            self.flags = value

        @property
        def allow_bf16_reduced_precision_reduction_split_k(self):
            return self.flags[1]

    state = {"blas": "default"}

    def preferred(value=None):
        if value is not None:
            state["blas"] = value
        return state["blas"]

    cuda = SimpleNamespace(matmul=Matmul(), preferred_blas_library=preferred)
    monkeypatch.setattr(torch.backends, "cuda", cuda)
    with pytest.raises(RuntimeError, match="injected"):
        with _module().reference_policy("cublaslt-no-splitk") as policy:
            assert policy["blas"] == "cublaslt"
            assert cuda.matmul.flags == (False, False)
            raise RuntimeError("injected body failure")
    assert state["blas"] == "default" and cuda.matmul.flags == (True, True)
