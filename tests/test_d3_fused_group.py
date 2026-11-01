"""CPU descriptor/budget controls; CUDA arithmetic lives in the separate gate."""

from dataclasses import replace
from pathlib import Path

import pytest

from soup_cli.utils.expert_group_fused import (
    PackedNF4,
    fused_group_linear,
    plan_fused_group,
    validate_fused_group,
)


def fixture(*, nested=False, rows=17):
    import torch

    w = PackedNF4(
        packed=torch.zeros(1024, dtype=torch.uint8),
        scales=torch.ones(32, dtype=torch.uint8 if nested else torch.float32),
        code=torch.linspace(-1, 1, 16), shape=(64, 32),
        nested_scales=torch.ones(1) if nested else None,
        nested_code=torch.linspace(-1, 1, 256) if nested else None,
        offset=torch.zeros(1) if nested else None,
    )
    return torch.ones(rows, 32, dtype=torch.bfloat16), w


@pytest.mark.parametrize("rows", [[0], [0, 1, 17, 512], [512] * 4])
@pytest.mark.parametrize("transpose", [False, True])
def test_output_and_descriptor_bound_includes_empty_experts(rows, transpose):
    plan = plan_fused_group(rows, (64, 32), transpose=transpose)
    cols = 32 if transpose else 64
    expected = sum((m * cols * 2 + 511) // 512 * 512 for m in rows)
    assert plan["output_bytes"] == expected
    assert plan["call_bytes"] == expected + (512 if max(rows) else 0)
    assert plan["dense_weight_bytes"] == 0


@pytest.mark.parametrize("rows", [[], [1] * 5, [True], [-1], [513], [1.5]])
def test_invalid_rows_fail_before_tensor_work(rows):
    with pytest.raises(ValueError):
        plan_fused_group(rows, (64, 32))


def test_hybrid_dispatch_and_single_dense_fallback_reservation():
    plan = plan_fused_group([0, 128, 129, 512], (2048, 2048), native_small_rows=True)
    assert plan["fused_experts"] == 2
    assert plan["fallback_experts"] == 1
    assert plan["dense_weight_bytes"] == 2048 * 2048 * 2
    assert plan["call_bytes"] == plan["output_bytes"] + 512 + plan["dense_weight_bytes"]
    small = plan_fused_group([1, 17, 31, 128], (2048, 1024), native_small_rows=True)
    assert small["fused_experts"] == small["descriptor_bytes"] == 0
    assert small["fallback_experts"] == 4
    # The fallback processes one projection at a time, never a dense expert stack.
    assert small["dense_weight_bytes"] == 2048 * 1024 * 2


@pytest.mark.parametrize("nested", [False, True])
def test_valid_plain_nested_and_exact_budget_boundary(nested):
    x, w = fixture(nested=nested)
    plan = plan_fused_group([17], w.shape)
    assert validate_fused_group([x], [w], call_budget_bytes=plan["call_bytes"]) == plan
    with pytest.raises(ValueError, match="budget"):
        validate_fused_group([x], [w], call_budget_bytes=plan["call_bytes"] - 1)


@pytest.mark.parametrize("field", ["packed", "scales", "code", "nested_scales",
                                  "nested_code", "offset"])
def test_truncated_nf4_state_is_refused(field):
    x, w = fixture(nested=True)
    damaged = replace(w, **{field: getattr(w, field)[:-1]})
    with pytest.raises(ValueError, match="extent"):
        validate_fused_group([x], [damaged], call_budget_bytes=2**20)


def test_partial_nested_state_and_mixed_groups_are_refused():
    x, plain = fixture()
    _, nested = fixture(nested=True)
    with pytest.raises(ValueError, match="partial nested"):
        validate_fused_group([x], [replace(plain, offset=nested.offset)], call_budget_bytes=2**20)
    with pytest.raises(ValueError, match="share"):
        validate_fused_group([x, x], [plain, nested], call_budget_bytes=2**20)


def test_frozen_only_and_transposed_input_contract():
    import torch

    x, w = fixture()
    with pytest.raises(ValueError, match="frozen BF16"):
        validate_fused_group([x.requires_grad_()], [w], call_budget_bytes=2**20)
    with pytest.raises(ValueError, match="dimensions"):
        validate_fused_group([x.detach()], [w], transpose=True, call_budget_bytes=2**20)
    dx = torch.ones(17, 64, dtype=torch.bfloat16)
    assert validate_fused_group([dx], [w], transpose=True,
                                call_budget_bytes=2**20)["output_bytes"] == 1536


def test_noncontiguous_input_and_trainable_code_are_refused():
    import torch

    x, w = fixture()
    with pytest.raises(ValueError, match="contiguous"):
        validate_fused_group([torch.ones(32, 17, dtype=x.dtype).t()], [w],
                             call_budget_bytes=2**20)
    with pytest.raises(ValueError, match="frozen NF4"):
        validate_fused_group([x], [replace(w, code=w.code.requires_grad_())],
                             call_budget_bytes=2**20)


def test_cpu_and_underbudget_launch_refuse_before_output_allocation(monkeypatch):
    import torch

    x, w = fixture()
    monkeypatch.setattr(torch, "empty", lambda *a, **k: pytest.fail("allocated before refusal"))
    with pytest.raises(ValueError, match="budget"):
        fused_group_linear([x], [w], call_budget_bytes=1)
    with pytest.raises(ValueError, match="physical CUDA"):
        fused_group_linear([x], [w], call_budget_bytes=2**20)


def test_cuda_gate_refuses_missing_cells_and_native_mismatch(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "benchmarks/harness"))
    from d3_fused_nf4_gate import evaluate_cells

    assert "incomplete grid" in evaluate_cells([])
    cells = []
    for shape in ((2048, 2048), (2048, 1024)):
        for nested in (False, True):
            for transpose in (False, True):
                for rows in ([0, 1, 17, 129], [31, 128, 481, 512]):
                    cells.append({"shape": shape, "nested": nested, "transpose": transpose,
                                  "rows": rows, "experts": [
                                      {"group_singleton_exact": True, "group_native_exact": True,
                                       "finite": True} for _ in range(4)]})
            cells.append({"shape": shape, "nested": nested, "kind": "control",
                          "negative_effective": True, "restored_exact": True})
    assert evaluate_cells(cells) == []
    assert "incomplete grid" in evaluate_cells(cells[:-1])
    assert "unexpected/duplicate control" in evaluate_cells(cells + [cells[-1]])
    cells[0]["experts"][0]["group_native_exact"] = False
    assert "native/grouped exactness" in evaluate_cells(cells)


def test_profiler_descriptor_copy_is_not_counted_as_second_kernel(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "benchmarks/harness"))
    from d3_fused_nf4_dispatch import compute_kernel_names

    assert compute_kernel_names(["Memcpy HtoD (Pageable -> Device)", "fused"]) == ["fused"]
    assert compute_kernel_names(["fused", "unexpected_compute"]) == ["fused", "unexpected_compute"]


@pytest.mark.parametrize("nested", [False, True])
def test_packed_pool_budget_raw_bytes_and_expired_leases(tmp_path, nested):
    import torch

    from soup_cli.utils.expert_group import GroupedExpertPool, plan_grouped_expert_buffer
    from soup_cli.utils.layer_shard import NF4_CODE_KEY, NF4_NESTED_CODE_KEY
    from tests.test_d3_grouped_expert import _fixture

    _module, reader, baseline, old_pool = _fixture(
        tmp_path, nf4=True, nested=nested, dtype="bfloat16")
    codes = {NF4_CODE_KEY: baseline._buffers["code"]}
    if nested:
        codes[NF4_NESTED_CODE_KEY] = baseline._buffers["nested_code"]
    plan = plan_grouped_expert_buffer(reader, dtype="bfloat16", group_width=4, packed_only=True)
    assert plan.weight_bytes < old_pool.plan.weight_bytes
    with pytest.raises(ValueError, match="budget"):
        GroupedExpertPool(reader, device="cpu", dtype="bfloat16", group_width=4,
                          codes=codes, packed_only=True, weight_budget_bytes=plan.weight_bytes - 1)
    pool = GroupedExpertPool(reader, device="cpu", dtype="bfloat16", group_width=4,
                             codes=codes, packed_only=True, weight_budget_bytes=plan.weight_bytes)
    try:
        assert all("gate_up" not in slot._buffers and "down" not in slot._buffers
                   and "scratch" not in slot._buffers for slot in pool._slots)
        with pool.lease(reader, [0, 1]) as pairs:
            weight = pairs[0][0]
            weight.packed._soup_stream_owner_check()
            first = weight.packed.clone()
            with baseline.lease(reader, 0):
                assert torch.equal(weight.packed, baseline._buffers["gate_up:0"].flatten())
            with pytest.raises(RuntimeError, match="active lease"):
                pool.close()
        with pytest.raises(RuntimeError, match="expired"):
            weight.packed._soup_stream_owner_check()
        with pool.lease(reader, [0]):
            with pytest.raises(RuntimeError, match="expired"):
                weight.packed._soup_stream_owner_check()
        assert first.numel() > 0
    finally:
        pool.close()
        old_pool.close()
        baseline.close()
        reader.close()


def test_public_fused_config_is_explicit_about_row_policy_and_cpu_refusal():
    from types import SimpleNamespace

    from soup_cli.config.schema import ExpertStreamConfig
    from soup_cli.utils.expert_training import validate_expert_training

    with pytest.raises(ValueError, match="native_rows_from"):
        ExpertStreamConfig(backend="fused_nf4")
    options = ExpertStreamConfig(backend="fused_nf4", native_rows_from=None)
    with pytest.raises(ValueError, match="CUDA BF16"):
        validate_expert_training(SimpleNamespace(model_type="olmoe"), options,
                                 device="cpu", dtype="bfloat16")


def test_unqualified_gpu_is_refused_by_public_validation_before_install(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch

    from soup_cli.config.schema import ExpertStreamConfig
    from soup_cli.utils import expert_training

    library = tmp_path / "libcublas.so.13"
    library.touch()
    options = ExpertStreamConfig(backend="fused_nf4", native_rows_from=None,
                                 cublas_library=str(library))
    monkeypatch.setattr(expert_training.sys, "platform", "linux")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    config = SimpleNamespace(model_type="olmoe", hidden_size=2048, intermediate_size=1024)
    with pytest.raises(ValueError, match="sm_86"):
        expert_training.validate_expert_training(config, options, device="cuda", dtype="bfloat16")


def test_preflight_reserves_packed_allocator_ceiling(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from soup_cli.config.schema import ExpertStreamConfig
    from soup_cli.utils import expert_training, layer_shard
    from tests.test_d3_grouped_expert import _fixture

    _module, reader, baseline, old_pool = _fixture(
        tmp_path, nf4=True, nested=True, dtype="bfloat16")
    try:
        monkeypatch.setattr(layer_shard, "expert_paths", lambda *args: {0: reader.path})
        index = SimpleNamespace(expert_count=reader.count, expert_shapes=reader.shapes,
                                expert_quant_specs=reader.quant_specs, dtype=reader.dtype)
        options = ExpertStreamConfig(backend="fused_nf4", native_rows_from=None)
        charge = expert_training.expert_training_reservations(
            str(tmp_path), index, options, dtype="bfloat16")
        assert charge["device_bytes"] == (
            options.weight_budget_bytes + options.activation_budget_bytes
            + options.call_budget_bytes + 4 * 2**20)
        assert charge["host_bytes"] > reader.max_span
    finally:
        old_pool.close()
        baseline.close()
        reader.close()
