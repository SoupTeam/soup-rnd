"""Development-only D3 activation fusion gates; CPU tests cannot qualify Triton."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize("rows,width", [(0, 16), (1, 16), (37, 16), (512, 1024)])
def test_activation_storage_model_exposes_one_removed_temporary(rows, width):
    from soup_cli.utils.expert_group_activation import plan_swiglu

    plan = plan_swiglu(rows, width)
    assert plan["eager_live_bytes"] - plan["fused_live_bytes"] == plan["output_bytes"]
    assert plan["removed_temporary_bytes"] == (rows * width * 2 + 511) // 512 * 512
    assert "must be measured" in plan["scope"]


@pytest.mark.parametrize("rows,width", [(-1, 16), (1, 0), (1, -1), (True, 16), (1, 1.5)])
def test_activation_plan_refuses_invalid_geometry(rows, width):
    from soup_cli.utils.expert_group_activation import plan_swiglu

    with pytest.raises(ValueError):
        plan_swiglu(rows, width)


def test_development_imports_remain_lazy():
    subprocess.run([sys.executable, "-c", (
        "import sys; import soup_cli.utils.expert_group_activation;"
        "import soup_cli.utils.expert_group_autograd;"
        "assert not any(x in sys.modules for x in ('torch','triton','transformers','bitsandbytes'))"
    )], check=True)


def test_cpu_is_refused_before_loading_triton(monkeypatch):
    import torch

    from soup_cli.utils import expert_group_activation as activation

    def forbidden():
        pytest.fail("CPU call reached the Triton builder")

    monkeypatch.setattr(activation, "_kernel", forbidden)
    with torch.no_grad(), pytest.raises(ValueError, match="requires CUDA"):
        activation.fused_swiglu(torch.ones(1, 32, dtype=torch.bfloat16), output_budget_bytes=512)


@pytest.mark.gpu
@pytest.mark.parametrize("rows,width", [(0, 16), (1, 16), (37, 17), (129, 1024), (512, 1024)])
def test_cuda_activation_matches_native_bf16_bits_and_preserves_input(rows, width):
    import torch

    from soup_cli.utils.expert_group_activation import fused_swiglu, plan_swiglu

    with torch.no_grad():
        generator = torch.Generator().manual_seed(71)
        matrix = torch.randn(rows, 2 * width, generator=generator).cuda().bfloat16()
        before = matrix.clone()
        gate, up = matrix.chunk(2, -1)
        expected = torch.nn.functional.silu(gate) * up
        actual = fused_swiglu(matrix, output_budget_bytes=max(1, plan_swiglu(rows, width)[
            "output_bytes"
        ]))
        torch.cuda.synchronize()
        assert torch.equal(expected.view(torch.int16), actual.view(torch.int16))
        assert torch.equal(matrix, before)
        assert actual.is_contiguous()


@pytest.mark.gpu
def test_cuda_activation_covers_every_finite_bf16_gate_including_signed_zero():
    import torch

    from soup_cli.utils.expert_group_activation import fused_swiglu

    with torch.no_grad():
        values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        values = values[values.isfinite()].reshape(-1, 256).cuda()
        generator = torch.Generator().manual_seed(93)
        up = torch.randn(values.shape, generator=generator).cuda().bfloat16()
        for multiplier in (torch.ones_like(up), up, -up):
            matrix = torch.cat((values, multiplier), dim=1)
            actual = fused_swiglu(matrix, output_budget_bytes=2**20)
            expected = torch.nn.functional.silu(values) * multiplier
            torch.cuda.synchronize()
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.gpu
def test_cuda_activation_budgets_and_autograd_are_refused_before_launch(monkeypatch):
    import torch

    from soup_cli.utils import expert_group_activation as activation

    matrix = torch.ones(37, 32, dtype=torch.bfloat16, device="cuda")

    def forbidden():
        pytest.fail("invalid call reached the Triton builder")

    monkeypatch.setattr(activation, "_kernel", forbidden)
    with torch.no_grad(), pytest.raises(ValueError, match="exceeds declared budget"):
        activation.fused_swiglu(matrix, output_budget_bytes=1)
    with pytest.raises(ValueError, match="requires no_grad"):
        activation.fused_swiglu(matrix, output_budget_bytes=2**20)
    with torch.no_grad(), pytest.raises(ValueError, match="forward-only"):
        activation.fused_swiglu(matrix.requires_grad_(), output_budget_bytes=2**20)


@pytest.mark.gpu
def test_cuda_activation_reduces_measured_allocated_peak_after_warmup(record_property):
    import torch

    from soup_cli.utils.expert_group_activation import fused_swiglu, plan_swiglu

    with torch.no_grad():
        matrix = torch.randn(1024, 2048, dtype=torch.bfloat16, device="cuda")

        def eager():
            gate, up = matrix.chunk(2, -1)
            return torch.nn.functional.silu(gate) * up

        def fused():
            return fused_swiglu(matrix, output_budget_bytes=4 * 2**20)

        for call in (eager, fused):
            out = call()
            torch.cuda.synchronize()
            del out
        peaks = []
        for call in (eager, fused):
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            out = call()
            torch.cuda.synchronize()
            peaks.append(torch.cuda.max_memory_allocated() - before)
            del out
        record_property("eager_allocated_peak_delta_bytes", peaks[0])
        record_property("fused_allocated_peak_delta_bytes", peaks[1])
        record_property("scope", "activation only, after warm-up; not whole-model VRAM")
        # A local activation-only assertion, not a model VRAM or speed claim.
        assert peaks[0] - peaks[1] >= plan_swiglu(1024, 1024)["removed_temporary_bytes"]
