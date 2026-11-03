"""B1.2 exact forward, explicit weight residency and failure boundaries."""

from __future__ import annotations

import copy
import json
import subprocess
import sys

import pytest


def _experts(seed=5, count=8):
    import torch
    from transformers import DeepseekV3Config
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Experts

    config = DeepseekV3Config(
        hidden_size=32, intermediate_size=64, moe_intermediate_size=16,
        n_routed_experts=count, n_group=2, topk_group=1, num_experts_per_tok=2,
        n_shared_experts=1, norm_topk_prob=True,
    )
    config._experts_implementation = "eager"
    torch.manual_seed(seed)
    module = DeepseekV3Experts(config)
    with torch.no_grad():
        module.gate_up_proj.normal_(0, 0.2)
        module.down_proj.normal_(0, 0.2)
    return module.requires_grad_(False).eval(), config


def _store(tmp_path, module, *, nf4=False, nested=True, device="cpu"):
    from soup_cli.utils.expert_shard import ExpertShardReader, write_expert_shard
    from soup_cli.utils.expert_stream import ExpertBufferPool, plan_expert_buffer
    from soup_cli.utils.layer_shard import NF4_CODE_KEY, NF4_NESTED_CODE_KEY

    path = str(tmp_path / "experts.safetensors")
    info = write_expert_shard(
        path,
        ((gate.detach(), down.detach()) for gate, down in zip(
            module.gate_up_proj, module.down_proj
        )),
        quantise=nf4, double_quant=nested, device=device,
    )
    dtype = str(module.gate_up_proj.dtype).removeprefix("torch.")
    reader = ExpertShardReader(
        path, count=info.count, shapes=info.shapes, quant_specs=info.quant_specs,
        dtype=dtype, direct=False,
    )
    codes = {NF4_CODE_KEY: info.codes[0], NF4_NESTED_CODE_KEY: info.codes[1]}
    plan = plan_expert_buffer(reader, dtype=dtype)
    pool = ExpertBufferPool(
        reader, dtype=dtype, device=device, weight_budget_bytes=plan.weight_bytes, codes=codes,
    )
    return reader, pool


def _inputs(count=24, *, device="cpu", dtype=None, partial=False, top_k=2):
    import torch

    gen = torch.Generator().manual_seed(9)
    hidden = torch.randn(count, 32, generator=gen).to(device=device, dtype=dtype or torch.float32)
    if partial:
        index = torch.tensor([[0, 3]] * count, dtype=torch.int64).reshape(count, 2)
    else:
        index = torch.rand(count, 8, generator=gen).topk(top_k, dim=-1).indices
    weights = torch.rand(count, index.shape[1], generator=gen)
    return hidden, index.to(device), weights.to(device=device, dtype=hidden.dtype)


def _streamed(module, reader, pool):
    from soup_cli.utils.expert_stream import streamed_experts_class

    return streamed_experts_class()(reader, pool, module.act_fn, 0)


def _dequantized_reference(module, reader, pool):
    """Copy the EXACT stored weights into an independent resident reference."""
    import torch
    from bitsandbytes.functional import dequantize_4bit
    from safetensors.torch import load_file

    from soup_cli.utils.expert_shard import EXPERT_PARTS, expert_key
    from soup_cli.utils.layer_shard import NF4_CODE_KEY, NF4_NESTED_CODE_KEY
    from soup_cli.utils.layer_stream_runtime import rebuild_quant_state

    stored = {key: value.to(pool.device) for key, value in load_file(reader.path).items()}
    tables = {NF4_CODE_KEY: pool._buffers["code"]}
    if "nested_code" in pool._buffers:
        tables[NF4_NESTED_CODE_KEY] = pool._buffers["nested_code"]
    with torch.no_grad():
        for expert in range(reader.count):
            for part, target in zip(EXPERT_PARTS, (module.gate_up_proj, module.down_proj)):
                key = expert_key(expert, part)
                state = rebuild_quant_state(key, stored, reader.quant_specs[part], tables)
                target[expert].copy_(dequantize_4bit(stored[key], quant_state=state))


@pytest.mark.parametrize("count,partial", [(0, False), (1, True), (24, False), (24, True)])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_eager_forward_bit_exact_and_load_order(tmp_path, count, partial, dtype):
    import torch

    module, _ = _experts()
    module = module.to(getattr(torch, dtype))
    reader, pool = _store(tmp_path, module)
    streamed = _streamed(module, reader, pool)
    args = _inputs(count, dtype=getattr(torch, dtype), partial=partial)
    try:
        with torch.no_grad():
            want, got = module(*args), streamed(*args)
        assert torch.equal(got, want)
        hit = sorted(set(args[1].flatten().tolist()))
        assert [row["expert"] for row in streamed.last_trace] == hit
        assert all(row["status"] == "ok" for row in streamed.last_trace)
        assert sum(row["token_routes"] for row in streamed.last_trace) == count * 2
        assert pool.fetches == reader.fetches == len(hit)
        assert pool.logical_bytes_read == sum(
            row["logical_bytes_read"] for row in streamed.last_trace
        )
    finally:
        reader.close()
        pool.close()


def test_negative_control_differs_in_the_same_process(tmp_path):
    import torch

    module, _ = _experts()
    reader, pool = _store(tmp_path, module)
    args = _inputs(partial=True)
    streamed = _streamed(module, reader, pool)
    try:
        with torch.no_grad():
            want = module(*args)
            assert want.abs().sum().item() > 0
            module.down_proj[0].add_(2.0)
            assert not torch.equal(module(*args), want)
            assert torch.equal(streamed(*args), want)
    finally:
        reader.close()
        pool.close()


@pytest.mark.parametrize("top_k,seed", [(1, 1), (4, 19), (8, 101)])
def test_different_top_k_and_nontrivial_expert_weights(tmp_path, top_k, seed):
    import torch

    module, _ = _experts(seed)
    reader, pool = _store(tmp_path, module)
    args = _inputs(top_k=top_k)
    try:
        with torch.no_grad():
            assert torch.equal(_streamed(module, reader, pool)(*args), module(*args))
    finally:
        reader.close()
        pool.close()


def test_cpu_autocast_matches_eager(tmp_path):
    import torch

    module, _ = _experts()
    reader, pool = _store(tmp_path, module)
    try:
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            assert torch.equal(_streamed(module, reader, pool)(*_inputs()), module(*_inputs()))
    finally:
        reader.close()
        pool.close()


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_nf4_equals_independently_dequantized_reference(tmp_path, nested, dtype):
    import torch

    pytest.importorskip("bitsandbytes")
    module, _ = _experts()
    module = module.to(getattr(torch, dtype))
    reader, pool = _store(tmp_path, module, nf4=True, nested=nested)
    try:
        _dequantized_reference(module, reader, pool)
        args = _inputs(dtype=getattr(torch, dtype))
        with torch.no_grad():
            assert torch.equal(_streamed(module, reader, pool)(*args), module(*args))
        plan = pool.plan
        assert plan.packed_and_stats_bytes > 0 and plan.code_bytes >= 64
        assert (plan.scratch_bytes > 0) == nested
        assert plan.weight_bytes == (
            plan.dense_bytes + plan.packed_and_stats_bytes + plan.code_bytes
            + plan.scratch_bytes + plan.padding_bytes
        )
    finally:
        reader.close()
        pool.close()


def test_budget_is_checked_before_allocating_or_reading(tmp_path, monkeypatch):
    import torch

    from soup_cli.utils.expert_stream import ExpertBufferPool

    module, _ = _experts()
    reader, pool = _store(tmp_path, module)
    needed = pool.plan.weight_bytes
    pool.close()

    def forbidden(*args, **kwargs):
        pytest.fail("an insufficient budget must fail before allocation or a disk read")

    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(reader, "read_host", forbidden)
    with pytest.raises(ValueError, match="slot needs"):
        ExpertBufferPool(reader, device="cpu", dtype="float32", weight_budget_bytes=needed - 1)
    assert reader.fetches == 0
    reader.close()


def test_pool_reuses_one_arena_across_experts_and_blocks_nested_lease(tmp_path):
    import torch

    module, _ = _experts()
    reader, pool = _store(tmp_path, module)
    arena = pool._arena.data_ptr()
    try:
        with pool.lease(reader, 0) as (gate_up, down):
            assert gate_up.untyped_storage().data_ptr() == arena
            assert down.untyped_storage().data_ptr() == arena
            assert gate_up.untyped_storage().data_ptr() != (
                pool._staging.untyped_storage().data_ptr()
            )
            assert torch.equal(gate_up, module.gate_up_proj[0])
            snapshot = gate_up.clone()
            with pytest.raises(RuntimeError, match="active lease"):
                with pool.lease(reader, 1):
                    pass
            with pytest.raises(RuntimeError, match="active lease"):
                pool.close()
            assert torch.equal(gate_up, snapshot)
        with pool.lease(reader, 1) as (gate_up, down):
            assert pool._arena.data_ptr() == arena
            assert torch.equal(gate_up, module.gate_up_proj[1])
        assert pool.stats()["slot_count"] == 1 and pool.fetches == 2
    finally:
        reader.close()
        pool.close()
    assert pool._arena is None and not pool._buffers
    with pytest.raises(RuntimeError, match="closed"):
        with pool.lease(reader, 1):
            pass


def test_weight_budget_is_independent_of_expert_count(tmp_path):
    from soup_cli.utils.expert_stream import plan_expert_buffer

    plans = []
    for count in (8, 64):
        folder = tmp_path / str(count)
        folder.mkdir()
        module, _ = _experts(count=count)
        reader, pool = _store(folder, module)
        plans.append(plan_expert_buffer(reader, dtype="float32"))
        reader.close()
        pool.close()
    assert plans[0] == plans[1]


def test_pool_close_releases_buffers_even_if_device_sync_fails(tmp_path, monkeypatch):
    module, _ = _experts()
    reader, pool = _store(tmp_path, module)

    def failed_sync():
        raise RuntimeError("injected device synchronization failure")

    monkeypatch.setattr(pool, "_synchronize", failed_sync)
    with pytest.raises(RuntimeError, match="injected"):
        pool.close()
    assert pool.stats()["closed"] and pool._arena is None and not pool._buffers
    reader.close()


def test_load_error_is_recorded_and_lease_can_be_reused(tmp_path, monkeypatch):
    import torch

    module, _ = _experts()
    reader, pool = _store(tmp_path, module)
    streamed = _streamed(module, reader, pool)
    original = reader.read_host

    def broken(*args):
        raise OSError("deliberate short read")

    monkeypatch.setattr(reader, "read_host", broken)
    try:
        with pytest.raises(OSError, match="deliberate"):
            streamed(*_inputs(partial=True))
        assert streamed.last_trace[0]["status"] == "error"
        assert reader._handle is None
        monkeypatch.setattr(reader, "read_host", original)
        with torch.no_grad():
            assert torch.equal(streamed(*_inputs()), module(*_inputs()))
    finally:
        reader.close()
        pool.close()


@pytest.mark.parametrize("which", ["hidden", "weights", "both"])
def test_backward_matches_resident_after_arena_reuse(tmp_path, which):
    import torch

    module, _ = _experts()
    reader, pool = _store(tmp_path, module)
    hidden, index, weights = _inputs()
    hidden.requires_grad_(which in ("hidden", "both"))
    weights.requires_grad_(which in ("weights", "both"))
    ref_hidden = hidden.detach().clone().requires_grad_(hidden.requires_grad)
    ref_weights = weights.detach().clone().requires_grad_(weights.requires_grad)
    streamed = _streamed(module, reader, pool)
    try:
        expected = module(ref_hidden, index, ref_weights)
        actual = streamed(hidden, index, weights)
        assert torch.equal(actual, expected)
        hit_count = len(set(index.flatten().tolist()))
        assert pool.fetches == hit_count
        grad = torch.randn_like(actual)
        expected.backward(grad)
        actual.backward(grad)
        if hidden.requires_grad:
            assert torch.equal(hidden.grad, ref_hidden.grad)
            assert pool.fetches == 2 * hit_count
        else:
            assert pool.fetches == hit_count
        if weights.requires_grad:
            assert torch.equal(weights.grad, ref_weights.grad)
    finally:
        reader.close()
        pool.close()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_backward_restores_forward_autocast(tmp_path, device):
    import torch

    module, _ = _experts()
    module = module.to(device)
    reader, pool = _store(tmp_path, module, device=device)
    hidden, index, weights = _inputs(device=device)
    hidden.requires_grad_(True)
    weights.requires_grad_(True)
    ref_hidden = hidden.detach().clone().requires_grad_(True)
    ref_weights = weights.detach().clone().requires_grad_(True)
    try:
        with torch.autocast(device, dtype=torch.bfloat16):
            expected = module(ref_hidden, index, ref_weights)
            actual = _streamed(module, reader, pool)(hidden, index, weights)
        assert torch.equal(actual, expected)
        grad = torch.randn_like(actual)
        expected.backward(grad)
        actual.backward(grad)
        assert torch.equal(hidden.grad, ref_hidden.grad)
        assert torch.equal(weights.grad, ref_weights.grad)
    finally:
        reader.close()
        pool.close()


def test_backward_rejects_higher_order_gradients(tmp_path):
    import torch

    module, _ = _experts()
    reader, pool = _store(tmp_path, module)
    hidden, index, weights = _inputs(partial=True)
    hidden.requires_grad_(True)
    try:
        actual = _streamed(module, reader, pool)(hidden, index, weights)
        with pytest.raises(RuntimeError, match="higher-order"):
            torch.autograd.grad(actual.sum(), hidden, create_graph=True)
    finally:
        reader.close()
        pool.close()


@pytest.mark.parametrize("activation", ["dropout", "callable", "hooked_silu"])
def test_training_rejects_unproven_activation_but_forward_accepts_it(tmp_path, activation):
    import torch

    module, _ = _experts()
    reader, pool = _store(tmp_path, module)
    streamed = _streamed(module, reader, pool)
    if activation == "dropout":
        streamed.act_fn = torch.nn.Dropout(p=0.5)
    elif activation == "callable":
        del streamed.act_fn
        streamed.act_fn = lambda value: value * torch.rand_like(value)
    else:
        streamed.act_fn.register_forward_hook(lambda _module, _args, output: output + 1)
    hidden, index, weights = _inputs(partial=True)
    hidden.requires_grad_(True)
    try:
        with pytest.raises(ValueError, match="stateless SiLU"):
            streamed(hidden, index, weights)
        assert pool.fetches == 0
        with torch.no_grad():
            assert streamed(hidden, index, weights).shape == hidden.shape
        assert pool.fetches == 2
    finally:
        reader.close()
        pool.close()


def test_backward_rejects_activation_hook_added_after_forward(tmp_path):
    module, _ = _experts()
    reader, pool = _store(tmp_path, module)
    streamed = _streamed(module, reader, pool)
    hidden, index, weights = _inputs(partial=True)
    hidden.requires_grad_(True)
    try:
        result = streamed(hidden, index, weights)
        streamed.act_fn.register_forward_hook(lambda _module, _args, output: output + 1)
        with pytest.raises(ValueError, match="stateless SiLU"):
            result.sum().backward()
        assert pool.fetches == 2
    finally:
        reader.close()
        pool.close()


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_nf4_backward_matches_stored_weight_reference(tmp_path, nested, device):
    import torch

    pytest.importorskip("bitsandbytes")
    module, _ = _experts()
    module = module.to(device=device, dtype=torch.float32)
    reader, pool = _store(tmp_path, module, nf4=True, nested=nested, device=device)
    _dequantized_reference(module, reader, pool)
    hidden, index, weights = _inputs(device=device, partial=True)
    hidden.requires_grad_(True)
    weights.requires_grad_(True)
    ref_hidden = hidden.detach().clone().requires_grad_(True)
    ref_weights = weights.detach().clone().requires_grad_(True)
    try:
        expected = module(ref_hidden, index, ref_weights)
        actual = _streamed(module, reader, pool)(hidden, index, weights)
        assert torch.equal(actual, expected)
        grad = torch.randn_like(actual)
        expected.backward(grad)
        actual.backward(grad)
        assert torch.equal(hidden.grad, ref_hidden.grad)
        assert torch.equal(weights.grad, ref_weights.grad)
        assert pool.fetches == 4
    finally:
        reader.close()
        pool.close()


def test_checkpoint_recompute_preserves_gradients(tmp_path):
    import torch
    from torch.utils.checkpoint import checkpoint

    module, _ = _experts()
    reader, pool = _store(tmp_path, module)
    hidden, index, weights = _inputs(partial=True)
    hidden.requires_grad_(True)
    weights.requires_grad_(True)
    ref_hidden = hidden.detach().clone().requires_grad_(True)
    ref_weights = weights.detach().clone().requires_grad_(True)
    streamed = _streamed(module, reader, pool)
    try:
        expected = checkpoint(module, ref_hidden, index, ref_weights, use_reentrant=False)
        actual = checkpoint(streamed, hidden, index, weights, use_reentrant=False)
        grad = torch.randn_like(actual)
        expected.backward(grad)
        actual.backward(grad)
        assert torch.equal(actual, expected)
        assert torch.equal(hidden.grad, ref_hidden.grad)
        assert torch.equal(weights.grad, ref_weights.grad)
        assert pool.fetches >= 4
    finally:
        reader.close()
        pool.close()


def test_backward_load_failure_releases_lease(tmp_path, monkeypatch):
    import torch

    module, _ = _experts()
    reader, pool = _store(tmp_path, module)
    hidden, index, weights = _inputs(partial=True)
    hidden.requires_grad_(True)
    streamed = _streamed(module, reader, pool)
    actual = streamed(hidden, index, weights)
    original = reader.read_host

    def broken(*args):
        raise OSError("backward read failed")

    try:
        monkeypatch.setattr(reader, "read_host", broken)
        with pytest.raises(OSError, match="backward read failed"):
            actual.sum().backward()
        assert reader._handle is None
        monkeypatch.setattr(reader, "read_host", original)
        with pool.lease(reader, 0) as (gate_up, _down):
            assert torch.equal(gate_up, module.gate_up_proj[0])
    finally:
        reader.close()
        pool.close()


@pytest.mark.parametrize("bad", ["negative", "large", "dtype", "shape", "nan", "hidden"])
def test_malformed_routing_fails_before_a_read(tmp_path, bad):
    import torch

    module, _ = _experts()
    reader, pool = _store(tmp_path, module)
    hidden, index, weights = _inputs()
    if bad == "negative":
        index[0, 0] = -1
    elif bad == "large":
        index[0, 0] = 8
    elif bad == "dtype":
        index = index.to(torch.int32)
    elif bad == "shape":
        weights = weights[:, :1]
    elif bad == "nan":
        weights[0, 0] = float("nan")
    else:
        hidden = hidden[:, :16]
    try:
        with pytest.raises(ValueError):
            _streamed(module, reader, pool)(hidden, index, weights)
        assert reader.fetches == 0
    finally:
        reader.close()
        pool.close()


@pytest.mark.parametrize("kimi", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_full_moe_block_preserves_router_shared_and_dense_parts(tmp_path, kimi, device):
    import torch
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE

    from soup_cli.utils.expert_stream import install_expert_forward

    module, config = _experts()
    if kimi:
        # B2's alias retains the native DeepSeek MoE arithmetic.
        from soup_cli.utils.kimi_k2 import load_stream_config

        folder = tmp_path / "config"
        folder.mkdir()
        raw = config.to_dict()
        raw["model_type"] = "kimi_k2"
        (folder / "config.json").write_text(json.dumps(raw), encoding="utf-8")
        config = load_stream_config(str(folder), trust_remote_code=False)
        config._experts_implementation = "eager"
    torch.manual_seed(7)
    block = DeepseekV3MoE(config).eval().requires_grad_(False).to(device)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.normal_(0, 0.2)
        block.gate.e_score_correction_bias.normal_(0, 0.01)
    reference = copy.deepcopy(block)
    reader, pool = _store(tmp_path, block.experts, device=device)
    holder = torch.nn.Module()
    holder.mlp = block
    dense_layer = torch.nn.Linear(32, 32)
    router, shared = block.gate, block.shared_experts
    calls = []
    hook = shared.register_forward_hook(lambda *args: calls.append(True))
    try:
        assert install_expert_forward([dense_layer, holder], {1: reader}, pool) == 1
        assert block.gate is router and block.shared_experts is shared
        args = _inputs(device=device)[0].reshape(2, 12, 32)
        with torch.no_grad():
            assert torch.equal(block(args), reference(args))
        assert calls == [True] and reader.fetches > 0
    finally:
        hook.remove()
        reader.close()
        pool.close()


def test_invalid_later_layer_does_not_partially_install(tmp_path):
    import torch

    from soup_cli.utils.expert_stream import install_expert_forward

    module, _ = _experts()
    reader, pool = _store(tmp_path, module)
    holder = torch.nn.Module()
    holder.mlp = torch.nn.Module()
    holder.mlp.experts = module
    invalid = torch.nn.Linear(32, 32)
    try:
        with pytest.raises(ValueError, match="model/cache"):
            install_expert_forward([holder, invalid], {0: reader, 1: reader}, pool)
        assert holder.mlp.experts is module
        module.requires_grad_(True)
        with pytest.raises(ValueError, match="frozen"):
            install_expert_forward([holder], {0: reader}, pool)
        assert holder.mlp.experts is module
    finally:
        reader.close()
        pool.close()


def test_new_module_has_lazy_heavy_imports():
    code = (
        "import json, sys; import soup_cli.utils.expert_stream; "
        "print(json.dumps([name for name in ('torch', 'transformers', 'bitsandbytes', 'peft') "
        "if name in sys.modules]))"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []


@pytest.mark.gpu
@pytest.mark.parametrize("nf4,nested", [(False, False), (True, False), (True, True)])
def test_cuda_forward_and_weight_arena_peak(tmp_path, nf4, nested, record_property):
    import torch

    module, _ = _experts()
    module = module.to(device="cuda", dtype=torch.bfloat16)
    reader, pool = _store(tmp_path, module, nf4=nf4, nested=nested, device="cuda")
    streamed = _streamed(module, reader, pool)
    if nf4:
        _dequantized_reference(module, reader, pool)
    args = _inputs(device="cuda", dtype=torch.bfloat16)
    try:
        torch.cuda.synchronize()
        baseline_allocated = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            want = module(*args)
            first_hit = int(args[1].min())
            module.down_proj[first_hit].add_(2.0)
            assert not torch.equal(module(*args), want), "ineffective CUDA negative control"
            got = streamed(*args)
        torch.cuda.synchronize()
        assert torch.equal(got, want)
        assert pool.cuda_allocation_delta <= pool.weight_budget_bytes
        assert pool._arena.untyped_storage().nbytes() == pool.plan.weight_bytes
        # Report actual total peaks separately; activations/workspaces aren't the weight slot.
        streamed.cuda_peak = {
            "baseline_allocated": baseline_allocated, "baseline_reserved": baseline_reserved,
            "max_allocated": torch.cuda.max_memory_allocated(),
            "max_reserved": torch.cuda.max_memory_reserved(),
        }
        assert streamed.cuda_peak["max_allocated"] >= baseline_allocated
        record_property("weight_pool", json.dumps(pool.stats()))
        record_property("cuda_memory", json.dumps(streamed.cuda_peak))
        record_property("expert_trace", json.dumps(streamed.last_trace))
        record_property("negative_control_differs", "true")
    finally:
        reader.close()
        pool.close()
