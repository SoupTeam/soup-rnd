"""Bounded grouped weight restore with the unchanged B1.2 eager arithmetic."""

from __future__ import annotations

import subprocess
import sys

import pytest


def _fixture(
    tmp_path, *, dtype="float32", nf4=False, nested=False, width=4, count=8, device="cpu",
):
    import torch

    from soup_cli.utils.expert_group import GroupedExpertPool, plan_grouped_expert_buffer
    from soup_cli.utils.layer_shard import NF4_CODE_KEY, NF4_NESTED_CODE_KEY
    from tests.test_b1_expert_forward import _experts, _store

    module, _ = _experts(count=count)
    module = module.to(device=device, dtype=getattr(torch, dtype))
    reader, baseline = _store(tmp_path, module, nf4=nf4, nested=nested, device=device)
    codes = {}
    for name, key in (("code", NF4_CODE_KEY), ("nested_code", NF4_NESTED_CODE_KEY)):
        if name in baseline._buffers:
            codes[key] = baseline._buffers[name].clone()
    plan = plan_grouped_expert_buffer(reader, dtype=dtype, group_width=width)
    pool = GroupedExpertPool(
        reader, device=device, dtype=dtype, group_width=width,
        weight_budget_bytes=plan.weight_bytes, codes=codes,
    )
    return module, reader, baseline, pool


@pytest.mark.parametrize("width", [1, 2, 4])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("partial", [False, True])
def test_grouped_reader_forward_matches_native_and_b12(tmp_path, width, dtype, partial):
    import torch

    from soup_cli.utils.expert_group import grouped_experts_class
    from tests.test_b1_expert_forward import _inputs, _streamed

    module, reader, baseline, pool = _fixture(tmp_path, width=width, dtype=dtype)
    args = _inputs(count=17, dtype=getattr(torch, dtype), partial=partial)
    candidate = grouped_experts_class()(reader, pool, module.act_fn, 0)
    try:
        with torch.no_grad():
            expected = module(*args)
            prior = _streamed(module, reader, baseline)(*args)
            before = reader.fetches
            actual = candidate(*args)
        assert torch.equal(expected, actual) and torch.equal(prior, actual)
        hit = sorted(set(args[1].flatten().tolist()))
        assert reader.fetches - before == len(hit)
        assert [row["expert"] for row in candidate.last_trace] == hit
        assert pool.fetches == len(hit)
        assert sum(row["token_routes"] for row in candidate.last_trace) == args[1].numel()
        assert pool.logical_bytes_read == sum(
            row["logical_bytes_read"] for row in candidate.last_trace
        )
        assert all(row["status"] == "ok" for row in candidate.last_trace)
        assert pool._arena.untyped_storage().nbytes() == pool.plan.weight_bytes
    finally:
        pool.close()
        baseline.close()
        reader.close()


@pytest.mark.parametrize("nested", [False, True])
def test_cpu_nf4_control_uses_identical_stored_bytes(tmp_path, nested):
    import torch

    from soup_cli.utils.expert_group import grouped_experts_class
    from tests.test_b1_expert_forward import _dequantized_reference, _inputs

    module, reader, baseline, pool = _fixture(tmp_path, nf4=True, nested=nested, width=4)
    try:
        _dequantized_reference(module, reader, baseline)
        args = _inputs(count=17)
        with torch.no_grad():
            want = module(*args)
            got = grouped_experts_class()(reader, pool, module.act_fn, 0)(*args)
        assert torch.equal(got, want)
        assert pool.stats()["decode_backend"] == "cpu-reference"
        assert pool.stats()["cpu_nf4_temporaries"] is True
    finally:
        pool.close()
        baseline.close()
        reader.close()


def test_zero_token_input_reads_nothing(tmp_path):
    import torch

    from soup_cli.utils.expert_group import grouped_experts_class
    from tests.test_b1_expert_forward import _inputs

    module, reader, baseline, pool = _fixture(tmp_path)
    try:
        candidate = grouped_experts_class()(reader, pool, module.act_fn, 0)
        with torch.no_grad():
            out = candidate(*_inputs(count=0))
        assert out.shape == (0, 32)
        assert reader.fetches == 0 and pool.fetches == 0 and candidate.last_trace == []
    finally:
        pool.close()
        baseline.close()
        reader.close()


def test_negative_control_differs_with_same_process_source_mutation(tmp_path):
    import torch

    from soup_cli.utils.expert_group import grouped_experts_class
    from tests.test_b1_expert_forward import _inputs

    module, reader, baseline, pool = _fixture(tmp_path)
    args = _inputs(count=17, partial=True)
    try:
        with torch.no_grad():
            want = module(*args)
            module.down_proj[0].add_(2)
            negative = module(*args)
            got = grouped_experts_class()(reader, pool, module.act_fn, 0)(*args)
        assert torch.equal(want, got) and not torch.equal(want, negative)
    finally:
        pool.close()
        baseline.close()
        reader.close()


def test_budget_refusal_precedes_any_allocation_or_read(tmp_path, monkeypatch):
    import torch

    from soup_cli.utils.expert_group import GroupedExpertPool, plan_grouped_expert_buffer

    _module, reader, baseline, pool = _fixture(tmp_path)
    plan = plan_grouped_expert_buffer(reader, dtype="float32", group_width=4)
    pool.close()

    def unexpected(*args, **kwargs):
        raise AssertionError("allocation occurred before budget refusal")

    monkeypatch.setattr(torch, "empty", unexpected)
    try:
        with pytest.raises(ValueError, match="budget"):
            GroupedExpertPool(reader, device="cpu", dtype="float32", group_width=4,
                              weight_budget_bytes=plan.weight_bytes - 1)
        assert reader.fetches == 0
    finally:
        baseline.close()
        reader.close()


def test_slot_plan_does_not_scale_with_total_experts(tmp_path):
    from dataclasses import replace
    from types import SimpleNamespace

    from soup_cli.utils.expert_group import plan_grouped_expert_buffer

    _module, reader, baseline, pool = _fixture(tmp_path)
    try:
        much_larger = SimpleNamespace(
            shapes=reader.shapes, quant_specs=reader.quant_specs,
            max_span=reader.max_span, count=4096,
        )
        one = plan_grouped_expert_buffer(reader, dtype="float32", group_width=4)
        many = plan_grouped_expert_buffer(much_larger, dtype="float32", group_width=4)
        assert replace(one, slot_count=many.slot_count) == many
    finally:
        pool.close()
        baseline.close()
        reader.close()


def test_lease_prevents_reuse_and_close_until_consumers_finish(tmp_path):
    import torch

    _module, reader, baseline, pool = _fixture(tmp_path)
    try:
        with pool.lease(reader, [0, 3]) as values:
            saved = values[0][0].clone()
            assert len(values) == 2
            assert values[0][0].untyped_storage().data_ptr() == pool._arena.data_ptr()
            with pytest.raises(RuntimeError, match="active"):
                with pool.lease(reader, [1]):
                    pass
            with pytest.raises(RuntimeError, match="active"):
                pool.close()
            assert torch.equal(values[0][0], saved)
        with pool.lease(reader, [1]) as values:
            assert not torch.equal(values[0][0], saved)
        pool.close()
        with pytest.raises(RuntimeError, match="closed"):
            with pool.lease(reader, [0]):
                pass
    finally:
        pool.close()
        baseline.close()
        reader.close()


@pytest.mark.parametrize("experts", [[], [1, 0], [1, 1], [True], [8], [0, 1, 2, 3, 4]])
def test_invalid_group_refused_before_read(tmp_path, experts):
    _module, reader, baseline, pool = _fixture(tmp_path)
    try:
        with pytest.raises(ValueError):
            with pool.lease(reader, experts):
                pass
        assert reader.fetches == 0
    finally:
        pool.close()
        baseline.close()
        reader.close()


def test_partial_read_failure_closes_reader_and_unlocks_group(tmp_path, monkeypatch):
    _module, reader, baseline, pool = _fixture(tmp_path)
    original = reader.read_host

    def broken(expert, staging):
        if expert == 1:
            raise OSError("injected second expert failure")
        return original(expert, staging)

    monkeypatch.setattr(reader, "read_host", broken)
    try:
        with pytest.raises(OSError, match="second expert"):
            with pool.lease(reader, [0, 1]):
                raise AssertionError("failed group should never be yielded")
        assert reader._handle is None
        monkeypatch.setattr(reader, "read_host", original)
        with pool.lease(reader, [2, 3]) as result:
            assert len(result) == 2
    finally:
        pool.close()
        baseline.close()
        reader.close()


@pytest.mark.parametrize("width", [0, 5, True, 2.0, "2"])
def test_invalid_width_refused_before_allocation(tmp_path, monkeypatch, width):
    import torch

    from soup_cli.utils.expert_group import GroupedExpertPool

    _module, reader, baseline, pool = _fixture(tmp_path)

    def unexpected(*args, **kwargs):
        raise AssertionError("invalid width allocated storage")

    monkeypatch.setattr(torch, "empty", unexpected)
    try:
        with pytest.raises(ValueError, match="group width"):
            GroupedExpertPool(reader, device="cpu", dtype="float32", group_width=width,
                              weight_budget_bytes=1000000)
        assert reader.fetches == 0
    finally:
        pool.close()
        baseline.close()
        reader.close()


def test_mismatched_storage_contract_is_refused_before_read(tmp_path):
    from types import SimpleNamespace

    _module, reader, baseline, pool = _fixture(tmp_path)
    wrong = SimpleNamespace(
        shapes=reader.shapes, quant_specs=reader.quant_specs,
        dtype=reader.dtype, max_span=reader.max_span + 1,
    )
    try:
        with pytest.raises(ValueError, match="does not match"):
            with pool.lease(wrong, [0]):
                raise AssertionError("mismatched reader was accepted")
        assert reader.fetches == 0
        with pool.lease(reader, [0]):
            pass
    finally:
        pool.close()
        baseline.close()
        reader.close()


def test_decode_failure_never_yields_stale_weights(tmp_path, monkeypatch):
    _module, reader, baseline, pool = _fixture(tmp_path, nf4=True)
    original = pool._restore

    def broken(count):
        raise RuntimeError("injected decoder failure")

    monkeypatch.setattr(pool, "_restore", broken)
    try:
        with pytest.raises(RuntimeError, match="decoder failure"):
            with pool.lease(reader, [0, 1]):
                raise AssertionError("decoder failure yielded stale weights")
        assert reader._handle is None and pool.groups_loaded == 0
        monkeypatch.setattr(pool, "_restore", original)
        with pool.lease(reader, [2]):
            pass
        assert pool.groups_loaded == 1
    finally:
        pool.close()
        baseline.close()
        reader.close()


def test_unknown_device_completion_poisoned_pool_cannot_be_reused(tmp_path, monkeypatch):
    _module, reader, baseline, pool = _fixture(tmp_path)
    original = pool._synchronize

    def broken():
        raise RuntimeError("injected device completion failure")

    monkeypatch.setattr(pool, "_synchronize", broken)
    try:
        with pytest.raises(RuntimeError, match="completion failure"):
            with pool.lease(reader, [0]):
                pass
        assert reader._handle is None and pool.stats()["closed"]
        with pytest.raises(RuntimeError, match="closed"):
            with pool.lease(reader, [1]):
                raise AssertionError("unknown completion allowed arena reuse")
    finally:
        monkeypatch.setattr(pool, "_synchronize", original)
        pool.close()
        baseline.close()
        reader.close()


def test_failed_group_forward_retains_reads_and_marks_no_success(tmp_path, monkeypatch):
    import torch

    from soup_cli.utils.expert_group import grouped_experts_class
    from tests.test_b1_expert_forward import _inputs

    module, reader, baseline, pool = _fixture(tmp_path)
    candidate = grouped_experts_class()(reader, pool, module.act_fn, 0)
    original = reader.read_host

    def broken(expert, staging):
        if expert == 1:
            raise OSError("injected read failure")
        return original(expert, staging)

    monkeypatch.setattr(reader, "read_host", broken)
    try:
        with torch.no_grad(), pytest.raises(OSError, match="read failure"):
            candidate(*_inputs(top_k=8))
        assert all(row["status"] == "error" for row in candidate.last_trace)
        assert sum(row["logical_bytes_read"] for row in candidate.last_trace) > 0
        assert pool.logical_bytes_read == sum(
            row["logical_bytes_read"] for row in candidate.last_trace
        )
        assert pool.last_loads[-1]["status"] == "error"
        assert not candidate._forward_lock.locked()
    finally:
        pool.close()
        baseline.close()
        reader.close()


def test_busy_pool_refusal_does_not_attribute_another_lease_reads(tmp_path):
    import torch

    from soup_cli.utils.expert_group import grouped_experts_class
    from tests.test_b1_expert_forward import _inputs

    module, reader, baseline, pool = _fixture(tmp_path)
    candidate = grouped_experts_class()(reader, pool, module.act_fn, 0)
    try:
        with pool.lease(reader, [0]):
            before = reader.bytes_read
            with torch.no_grad(), pytest.raises(RuntimeError, match="active lease"):
                candidate(*_inputs(partial=True))
            assert reader.bytes_read == before
            assert all(row["logical_bytes_read"] == 0 for row in candidate.last_trace)
    finally:
        pool.close()
        baseline.close()
        reader.close()


@pytest.mark.parametrize("width", [1, 2, 4])
@pytest.mark.parametrize("format_name", ["bf16", "nf4", "nested-nf4"])
@pytest.mark.parametrize("routing", ["empty", "one-partial", "all-experts"])
def test_cuda_grouped_reader_decode_forward_exact(tmp_path, width, format_name, routing):
    """Integration gate, distinct from the standalone decoder's prior CUDA checks."""
    import torch

    from soup_cli.utils.expert_group import grouped_experts_class
    from tests.test_b1_expert_forward import _dequantized_reference, _inputs, _streamed

    if not torch.cuda.is_available():
        pytest.skip("CUDA integration requires an existing native BF16 GPU")
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.skip("native BF16 required")
    if format_name != "bf16":
        pytest.importorskip("triton")
    module, reader, baseline, pool = _fixture(
        tmp_path, dtype="bfloat16", device="cuda", width=width,
        nf4=format_name != "bf16", nested=format_name == "nested-nf4",
    )
    args = _inputs(
        count=0 if routing == "empty" else 1 if routing == "one-partial" else 17,
        partial=routing == "one-partial", top_k=8 if routing == "all-experts" else 2,
        device="cuda", dtype=torch.bfloat16,
    )
    candidate = grouped_experts_class()(reader, pool, module.act_fn, 0)
    try:
        if format_name != "bf16":
            _dequantized_reference(module, reader, baseline)
        with torch.no_grad():
            expected = module(*args)
            prior = _streamed(module, reader, baseline)(*args)
            before = reader.fetches
            actual = candidate(*args)
            again = candidate(*args)
            assert torch.equal(expected, prior)
            assert torch.equal(expected, actual) and torch.equal(actual, again)
            assert torch.isfinite(actual).all().item()
            hit = sorted(set(args[1].flatten().tolist()))
            assert reader.fetches - before == 2 * len(hit)
            assert [row["expert"] for row in candidate.last_trace] == hit
            if hit:
                assert actual.count_nonzero().item() > 0
                module.down_proj[hit[0]].add_(2)
                assert not torch.equal(module(*args), expected)
        assert pool.cuda_allocation_delta <= pool.weight_budget_bytes
        assert pool._arena.untyped_storage().nbytes() == pool.plan.weight_bytes
    finally:
        pool.close()
        baseline.close()
        reader.close()


def test_gradient_path_is_refused_before_read(tmp_path):
    from soup_cli.utils.expert_group import grouped_experts_class
    from tests.test_b1_expert_forward import _inputs

    module, reader, baseline, pool = _fixture(tmp_path)
    hidden, index, weights = _inputs()
    hidden.requires_grad_(True)
    try:
        with pytest.raises(RuntimeError, match="B1.3"):
            grouped_experts_class()(reader, pool, module.act_fn, 0)(hidden, index, weights)
        assert reader.fetches == 0
    finally:
        pool.close()
        baseline.close()
        reader.close()


def test_import_does_not_load_heavy_libraries():
    code = (
        "import sys; import soup_cli.utils.expert_group; "
        "assert not any(x in sys.modules for x in "
        "('torch','transformers','bitsandbytes','triton','peft'))"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
