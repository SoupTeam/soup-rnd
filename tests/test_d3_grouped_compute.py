"""Grouped compute workspace planning and CUDA primitive integration controls."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


def _policy():
    path = Path(__file__).resolve().parents[1] / "benchmarks/harness/d3_cublas_grouped_probe.py"
    spec = importlib.util.spec_from_file_location("d3_math_policy_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reference_policy("cublaslt-no-splitk")


def _library():
    import torch

    if sys.platform != "linux" or not torch.cuda.is_available():
        pytest.skip("Linux native BF16 CUDA required")
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.skip("native BF16 required")
    site = Path(torch.__file__).parent.parent
    for name in ("cu13/lib/libcublas.so.13", "cublas/lib/libcublas.so.12"):
        path = site / "nvidia" / name
        if path.is_file():
            return path
    pytest.skip("explicit cuBLAS grouped API library unavailable")


def test_linear_plan_counts_allocator_bins_and_native_empty_paths():
    from soup_cli.utils.expert_group_gemm import plan_grouped_linear

    plan = plan_grouped_linear([(0, 32, 32), (1, 32, 32), (17, 32, 32)])
    assert plan["output_bytes"] == 512 + 1536
    assert plan["pointer_bytes"] == 1536
    assert plan["native_matrices"] == plan["grouped_matrices"] == 1
    empty = plan_grouped_linear([(0, 2048, 2048)])
    assert empty["call_bytes"] == 0


@pytest.mark.parametrize("shapes", [[], [(1, 33, 47)], [(True, 32, 32)],
                                     [(2**31, 32, 32)], [(1, 32, 32)] * 5])
def test_linear_plan_refuses_unvalidated_or_overflowing_shapes(shapes):
    from soup_cli.utils.expert_group_gemm import plan_grouped_linear

    with pytest.raises(ValueError):
        plan_grouped_linear(shapes)


def test_workspace_includes_mask_output_and_all_routes_without_total_expert_weights():
    from soup_cli.utils.expert_group_compute import plan_grouped_forward

    args = dict(tokens=512, top_k=8, experts=64, hidden=2048, intermediate=1024, width=4)
    plan = plan_grouped_forward(**args)
    assert plan["routing_bytes"] >= 512 * 8 * 64 * 8
    assert plan["output_bytes"] == 512 * 2048 * 2
    assert plan["temporary_bytes"] >= 512 * 8 * (2048 + 2048 + 1024) * 2
    assert plan["activation_bytes"] < 256 * 2**20
    # Changing E changes routing metadata, not a dense stack in the activation plan.
    many = plan_grouped_forward(**dict(args, experts=128))
    assert many["temporary_bytes"] == plan["temporary_bytes"]


def test_compute_imports_do_not_load_heavy_libraries():
    code = (
        "import sys;import soup_cli.utils.expert_group_compute;"
        "import soup_cli.utils.expert_group_gemm;"
        "assert not any(x in sys.modules for x in ('torch','triton','bitsandbytes','transformers'))"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("codec", ["dense", "nf4", "nf4-dq"])
@pytest.mark.parametrize("width", [1, 2, 4])
@pytest.mark.parametrize("case", ["empty", "one", "skew", "all"])
def test_cuda_grouped_compute_preserves_native_and_b12(tmp_path, codec, width, case):
    import torch

    from soup_cli.utils.expert_group_compute import grouped_compute_experts_class
    from soup_cli.utils.expert_group_gemm import GroupedGemm
    from tests.test_b1_expert_forward import _dequantized_reference, _inputs, _streamed
    from tests.test_d3_grouped_expert import _fixture

    library = _library()
    with _policy():
        module, reader, baseline, pool = _fixture(
            tmp_path, dtype="bfloat16", device="cuda", width=width,
            nf4=codec != "dense", nested=codec == "nf4-dq",
        )
        gemm = GroupedGemm(library, call_budget_bytes=16 * 2**20)
        candidate = grouped_compute_experts_class()(
            reader, pool, module.act_fn, 0, gemm=gemm, activation_budget_bytes=256 * 2**20,
        )
        try:
            if codec != "dense":
                _dequantized_reference(module, reader, baseline)
            args = _inputs(count={"empty": 0, "one": 1, "skew": 37, "all": 129}[case],
                           partial=case in ("one", "skew"), top_k=8 if case == "all" else 2,
                           device="cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                want = module(*args)
                prior = _streamed(module, reader, baseline)(*args)
                before = reader.fetches
                got = candidate(*args)
                again = candidate(*args)
                assert torch.equal(want, prior) and torch.equal(want, got)
                assert torch.equal(got, again)
                hit = sorted(set(args[1].flatten().tolist()))
                assert reader.fetches - before == 2 * len(hit)
                assert [row["expert"] for row in candidate.last_trace] == hit
                if hit:
                    module.down_proj[hit[0]].add_(0.125)
                    assert not torch.equal(module(*args), want)
                if case in ("skew", "all"):
                    assert gemm.grouped_matrices > 0
                if case == "one":
                    assert gemm.native_matrices > 0 and gemm.grouped_matrices == 0
                if case == "empty":
                    assert gemm.calls == 0
        finally:
            gemm.close()
            pool.close()
            baseline.close()
            reader.close()


def test_cuda_activation_budget_refusal_precedes_reads(tmp_path):
    from soup_cli.utils.expert_group_compute import grouped_compute_experts_class
    from soup_cli.utils.expert_group_gemm import GroupedGemm
    from tests.test_b1_expert_forward import _inputs
    from tests.test_d3_grouped_expert import _fixture

    library = _library()
    with _policy():
        module, reader, baseline, pool = _fixture(tmp_path, dtype="bfloat16", device="cuda")
        gemm = GroupedGemm(library, call_budget_bytes=16 * 2**20)
        try:
            candidate = grouped_compute_experts_class()(
                reader, pool, module.act_fn, 0, gemm=gemm, activation_budget_bytes=1,
            )
            with pytest.raises(ValueError, match="activation budget"):
                candidate(*_inputs(device="cuda", dtype=__import__("torch").bfloat16))
            assert reader.fetches == 0 and gemm.calls == 0
        finally:
            gemm.close()
            pool.close()
            baseline.close()
            reader.close()
