"""Development gates: no historical CUDA/training result is reused here."""

from __future__ import annotations

import copy
import os

import pytest


def _candidate(reader, pool, module, budget=2**24):
    from soup_cli.utils.expert_group_autograd import grouped_autograd_experts_class

    return grouped_autograd_experts_class()(
        reader, pool, module.act_fn, activation_budget_bytes=budget,
    )


@pytest.mark.parametrize("width", [1, 2, 4])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("partial", [False, True])
def test_exact_input_and_router_derivatives(tmp_path, width, dtype, partial):
    import torch

    from tests.test_b1_expert_forward import _inputs
    from tests.test_d3_grouped_expert import _fixture

    native, reader, baseline, pool = _fixture(tmp_path, width=width, dtype=dtype)
    hidden, indices, routes = _inputs(count=17, partial=partial, dtype=getattr(torch, dtype))
    x = hidden.clone().requires_grad_()
    y = hidden.clone().requires_grad_()
    a = routes.clone().requires_grad_()
    b = routes.clone().requires_grad_()
    candidate = _candidate(reader, pool, native)
    try:
        expected = native(x, indices, a)
        actual = candidate(y, indices, b)
        assert torch.equal(expected, actual)
        probe = torch.arange(expected.numel(), dtype=torch.float32).reshape_as(expected)
        probe = (probe.remainder(13) / 13).to(expected.dtype)
        expected.backward(probe)
        actual.backward(probe)
        assert torch.equal(x.grad, y.grad)
        assert torch.equal(a.grad, b.grad)
        assert all(parameter.grad is None for parameter in native.parameters())
        active = sorted(set(indices.flatten().tolist()))
        assert reader.fetches == 2 * len(active)
        assert sorted(row['expert'] for row in candidate.backward_trace) == active
        assert all(row['status'] == 'ok' for row in candidate.backward_trace)
    finally:
        pool.close()
        baseline.close()
        reader.close()


@pytest.mark.parametrize("nested", [False, True])
def test_quantized_derivatives_use_the_exact_stored_weights(tmp_path, nested):
    import torch

    from tests.test_b1_expert_forward import _dequantized_reference, _inputs
    from tests.test_d3_grouped_expert import _fixture

    native, reader, baseline, pool = _fixture(tmp_path, nf4=True, nested=nested)
    try:
        _dequantized_reference(native, reader, baseline)
        hidden, indices, routes = _inputs(count=17)
        x, y = hidden.clone().requires_grad_(), hidden.clone().requires_grad_()
        a, b = routes.clone().requires_grad_(), routes.clone().requires_grad_()
        expected = native(x, indices, a)
        actual = _candidate(reader, pool, native)(y, indices, b)
        assert torch.equal(expected, actual)
        expected.sum().backward()
        actual.sum().backward()
        assert torch.equal(x.grad, y.grad)
        assert torch.equal(a.grad, b.grad)
    finally:
        pool.close()
        baseline.close()
        reader.close()


def test_real_adapter_updates_match_across_repeated_steps(tmp_path):
    import torch

    from tests.test_b1_expert_forward import _inputs
    from tests.test_d3_grouped_expert import _fixture

    native, reader, baseline, pool = _fixture(tmp_path)
    hidden, indices, routes = _inputs(count=17)
    torch.manual_seed(25)
    adapter = torch.nn.Sequential(torch.nn.Linear(32, 4, bias=False),
                                  torch.nn.Linear(4, 32, bias=False))
    other = copy.deepcopy(adapter)
    before = [p.detach().clone() for p in adapter.parameters()]
    optimizers = [torch.optim.SGD(model.parameters(), lr=0.03) for model in (adapter, other)]
    candidate = _candidate(reader, pool, native)
    try:
        for _step in range(3):
            outputs = [native(hidden + adapter(hidden), indices, routes),
                       candidate(hidden + other(hidden), indices, routes)]
            assert torch.equal(*outputs)
            for out in outputs:
                out.square().mean().backward()
            for a, b in zip(adapter.parameters(), other.parameters()):
                assert torch.equal(a.grad, b.grad)
                assert a.grad.abs().sum().item() > 0
            for opt in optimizers:
                opt.step()
                opt.zero_grad(set_to_none=True)
            assert all(torch.equal(a, b) for a, b in zip(adapter.parameters(), other.parameters()))
        assert any(not torch.equal(a, b) for a, b in zip(before, adapter.parameters()))
    finally:
        pool.close()
        baseline.close()
        reader.close()


def test_graph_never_saves_reusable_weight_views(tmp_path):
    import torch

    from tests.test_b1_expert_forward import _inputs
    from tests.test_d3_grouped_expert import _fixture

    native, reader, baseline, pool = _fixture(tmp_path)
    hidden, indices, routes = _inputs(count=17)
    hidden.requires_grad_()
    saved = []
    try:
        with torch.autograd.graph.saved_tensors_hooks(
            lambda t: (saved.append(t), t)[1], lambda t: t,
        ):
            out = _candidate(reader, pool, native)(hidden, indices, routes)
        arena = pool._arena.untyped_storage().data_ptr()
        assert saved and all(t.untyped_storage().data_ptr() != arena for t in saved)
        out.sum().backward()
        assert hidden.grad is not None
    finally:
        pool.close()
        baseline.close()
        reader.close()


@pytest.fixture
def development_primitive(tmp_path):
    from tests.test_d3_grouped_expert import _fixture

    native, reader, baseline, pool = _fixture(tmp_path)
    try:
        yield native, reader, pool, _candidate(reader, pool, native)
    finally:
        pool.close()
        baseline.close()
        reader.close()


@pytest.mark.parametrize("hidden_grad,router_grad", [(True, False), (False, True), (True, True)])
def test_separate_derivatives_and_no_unnecessary_backward_reads(
    development_primitive, hidden_grad, router_grad,
):
    import torch

    from tests.test_b1_expert_forward import _inputs

    native, reader, _pool, candidate = development_primitive
    hidden, indices, routes = _inputs(count=17, partial=True)
    x = hidden.clone().requires_grad_(hidden_grad)
    y = hidden.clone().requires_grad_(hidden_grad)
    a = routes.clone().requires_grad_(router_grad)
    b = routes.clone().requires_grad_(router_grad)
    expected, actual = native(x, indices, a), candidate(y, indices, b)
    expected.sum().backward()
    actual.sum().backward()
    if hidden_grad:
        assert torch.equal(x.grad, y.grad)
    if router_grad:
        assert torch.equal(a.grad, b.grad)
    assert reader.fetches == (4 if hidden_grad else 2)


def test_autocast_settings_are_restored_for_backward(development_primitive):
    import torch

    from tests.test_b1_expert_forward import _inputs

    native, _reader, _pool, candidate = development_primitive
    hidden, indices, routes = _inputs(count=17)
    x, y = hidden.clone().requires_grad_(), hidden.clone().requires_grad_()
    a, b = routes.clone().requires_grad_(), routes.clone().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected, actual = native(x, indices, a), candidate(y, indices, b)
    expected.sum().backward()
    actual.sum().backward()
    assert torch.equal(expected, actual)
    assert torch.equal(x.grad, y.grad)
    assert torch.equal(a.grad, b.grad)


@pytest.mark.parametrize("failure", ["storage-changed", "pool-closed"])
def test_backward_refuses_invalidated_weight_lifetime(development_primitive, failure):
    from tests.test_b1_expert_forward import _inputs

    _native, reader, pool, candidate = development_primitive
    hidden, indices, routes = _inputs(count=17)
    out = candidate(hidden.requires_grad_(), indices, routes)
    before = reader.fetches
    if failure == "storage-changed":
        stat = os.stat(reader.path)
        os.utime(reader.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
        message = "storage changed"
    else:
        pool.close()
        message = "closed"
    with pytest.raises((ValueError, RuntimeError), match=message):
        out.sum().backward()
    assert reader.fetches == before


def test_higher_derivatives_are_explicitly_refused(development_primitive):
    import torch

    from tests.test_b1_expert_forward import _inputs

    _native, reader, _pool, candidate = development_primitive
    hidden, indices, routes = _inputs(count=17)
    out = candidate(hidden.requires_grad_(), indices, routes)
    before = reader.fetches
    with pytest.raises(RuntimeError, match="first derivatives only"):
        torch.autograd.grad(out.sum(), hidden, create_graph=True)
    assert reader.fetches == before


def test_activation_budget_refusal_precedes_expert_reads(development_primitive):
    from tests.test_b1_expert_forward import _inputs

    native, reader, pool, _candidate_default = development_primitive
    with pytest.raises(ValueError, match="activation budget"):
        _candidate(reader, pool, native, budget=1)(*_inputs())
    assert reader.fetches == 0


def test_triton_activation_refuses_cpu_before_reads(development_primitive):
    from soup_cli.utils.expert_group_autograd import grouped_autograd_experts_class

    native, reader, pool, _candidate_default = development_primitive
    with pytest.raises(ValueError, match="CUDA BF16 pool"):
        grouped_autograd_experts_class()(
            reader, pool, native.act_fn, activation_budget_bytes=2**24,
            activation_backend="triton",
        )
    assert reader.fetches == 0


def test_empty_tokens_match_native_without_reads(development_primitive):
    import torch

    from tests.test_b1_expert_forward import _inputs

    native, reader, _pool, candidate = development_primitive
    args = _inputs(count=0)
    args[0].requires_grad_()
    expected, actual = native(*args), candidate(*args)
    assert torch.equal(expected, actual)
    assert actual.requires_grad == expected.requires_grad
    assert reader.fetches == 0 and not candidate.forward_trace


@pytest.mark.gpu
@pytest.mark.parametrize("codec", ["dense", "nf4", "nf4-dq"])
@pytest.mark.parametrize("width", [1, 4])
def test_cuda_actual_grouped_gemm_derivatives_match_native(tmp_path, codec, width):
    import torch

    from soup_cli.utils.expert_group_autograd import grouped_autograd_experts_class
    from soup_cli.utils.expert_group_gemm import GroupedGemm
    from tests.test_b1_expert_forward import _dequantized_reference, _inputs
    from tests.test_d3_grouped_compute import _library, _policy
    from tests.test_d3_grouped_expert import _fixture

    library = _library()
    with _policy():
        native, reader, baseline, pool = _fixture(
            tmp_path, dtype="bfloat16", device="cuda", width=width,
            nf4=codec != "dense", nested=codec == "nf4-dq",
        )
        gemm = GroupedGemm(library, call_budget_bytes=16 * 2**20)
        try:
            if codec != "dense":
                _dequantized_reference(native, reader, baseline)
            candidate = grouped_autograd_experts_class()(
                reader, pool, native.act_fn, gemm=gemm, activation_budget_bytes=256 * 2**20,
            )
            hidden, indices, routes = _inputs(
                count=129, top_k=8, device="cuda", dtype=torch.bfloat16,
            )
            x, y = hidden.clone().requires_grad_(), hidden.clone().requires_grad_()
            a, b = routes.clone().requires_grad_(), routes.clone().requires_grad_()
            expected, actual = native(x, indices, a), candidate(y, indices, b)
            assert torch.equal(expected, actual)
            expected.square().mean().backward()
            actual.square().mean().backward()
            torch.cuda.synchronize()
            assert torch.equal(x.grad, y.grad)
            assert torch.equal(a.grad, b.grad)
            assert x.grad.abs().sum().item() > 0
            assert a.grad.abs().sum().item() > 0
            assert reader.fetches == 16 and gemm.grouped_matrices > 0
            assert all(parameter.grad is None for parameter in native.parameters())
        finally:
            gemm.close()
            pool.close()
            baseline.close()
            reader.close()


@pytest.mark.gpu
@pytest.mark.parametrize("codec", ["dense", "nf4", "nf4-dq"])
@pytest.mark.parametrize("activation_backend", ["native", "triton"])
def test_cuda_olmoe_geometry_derivatives_and_adapter_updates(
    tmp_path, codec, activation_backend, record_property,
):
    """Real OLMoE matrix sizes, synthetic weights; not a pretrained model gate."""
    import torch
    from transformers import OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import OlmoeExperts

    from soup_cli.utils.expert_group import GroupedExpertPool, plan_grouped_expert_buffer
    from soup_cli.utils.expert_group_autograd import grouped_autograd_experts_class
    from soup_cli.utils.expert_group_gemm import GroupedGemm
    from soup_cli.utils.layer_shard import NF4_CODE_KEY, NF4_NESTED_CODE_KEY
    from tests.test_b1_expert_forward import _dequantized_reference, _store
    from tests.test_d3_grouped_compute import _library, _policy

    library = _library()
    with _policy():
        config = OlmoeConfig(hidden_size=2048, intermediate_size=1024, num_experts=8,
                             num_experts_per_tok=8, hidden_act="silu")
        config._experts_implementation = "eager"
        torch.manual_seed(41)
        native = OlmoeExperts(config).cuda().bfloat16().requires_grad_(False)
        with torch.no_grad():
            native.gate_up_proj.normal_(0, 0.02)
            native.down_proj.normal_(0, 0.02)
        reader, baseline = _store(
            tmp_path, native, nf4=codec != "dense", nested=codec == "nf4-dq", device="cuda",
        )
        codes = {}
        for name, key in (("code", NF4_CODE_KEY), ("nested_code", NF4_NESTED_CODE_KEY)):
            if name in baseline._buffers:
                codes[key] = baseline._buffers[name].clone()
        plan = plan_grouped_expert_buffer(reader, dtype="bfloat16", group_width=4)
        pool = GroupedExpertPool(reader, device="cuda", dtype="bfloat16", group_width=4,
                                 weight_budget_bytes=plan.weight_bytes, codes=codes)
        gemm = GroupedGemm(library, call_budget_bytes=64 * 2**20, native_olmoe_rows_max=128)
        try:
            if codec != "dense":
                _dequantized_reference(native, reader, baseline)
            candidate = grouped_autograd_experts_class()(
                reader, pool, native.act_fn, gemm=gemm, activation_budget_bytes=256 * 2**20,
                activation_backend=activation_backend,
            )
            generator = torch.Generator().manual_seed(819)
            hidden = torch.randn(257, 2048, generator=generator).cuda().bfloat16()
            indices = torch.arange(8, device="cuda").expand(257, -1).contiguous()
            routes = torch.rand(257, 8, generator=generator).cuda().bfloat16()
            adapter = torch.nn.Sequential(torch.nn.Linear(2048, 4, bias=False),
                                          torch.nn.Linear(4, 2048, bias=False)).cuda().bfloat16()
            other = copy.deepcopy(adapter)
            before = [p.detach().clone() for p in adapter.parameters()]
            optimizers = [torch.optim.SGD(model.parameters(), lr=0.1)
                          for model in (adapter, other)]
            for _step in range(2):
                a, b = routes.clone().requires_grad_(), routes.clone().requires_grad_()
                left, right = hidden + adapter(hidden), hidden + other(hidden)
                left.retain_grad()
                right.retain_grad()
                expected, actual = native(left, indices, a), candidate(right, indices, b)
                assert torch.equal(expected, actual)
                expected.float().square().mean().backward()
                actual.float().square().mean().backward()
                assert torch.equal(left.grad, right.grad)
                assert torch.equal(a.grad, b.grad)
                assert a.grad.abs().sum().item() > 0
                for first, second in zip(adapter.parameters(), other.parameters()):
                    assert torch.equal(first.grad, second.grad)
                    assert first.grad.abs().sum().item() > 0
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                assert all(torch.equal(a, b)
                           for a, b in zip(adapter.parameters(), other.parameters()))
            assert any(not torch.equal(a, b) for a, b in zip(before, adapter.parameters()))
            assert gemm.grouped_matrices == 32
            assert reader.fetches == 32
            assert all(parameter.grad is None for parameter in native.parameters())
            record_property("scope", "synthetic OLMoE-size weights, two adapter optimizer steps")
            record_property("activation_backend", activation_backend)
            record_property("grouped_matrices", gemm.grouped_matrices)
            record_property("weight_budget_bytes", pool.weight_budget_bytes)
        finally:
            gemm.close()
            pool.close()
            baseline.close()
            reader.close()


@pytest.mark.gpu
@pytest.mark.parametrize("activation_backend", ["native", "triton"])
def test_cuda_complete_tiny_model_optimizer_steps_share_one_pool(
    tmp_path, activation_backend, record_property,
):
    """Full two-layer model with synthetic weights, not the trained OLMoE gate."""
    from contextlib import ExitStack

    import torch
    from transformers import OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM

    from soup_cli.utils.expert_group import GroupedExpertPool, plan_grouped_expert_buffer
    from soup_cli.utils.expert_group_autograd import grouped_autograd_experts_class
    from soup_cli.utils.expert_group_gemm import GroupedGemm
    from tests.test_b1_expert_forward import _store
    from tests.test_d3_grouped_compute import _library, _policy

    class AdapterEmbedding(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.first = torch.nn.Linear(32, 4, bias=False).cuda().bfloat16()
            self.second = torch.nn.Linear(4, 32, bias=False).cuda().bfloat16()

        def forward(self, ids):
            hidden = self.base(ids)
            return hidden + self.second(self.first(hidden))

    library = _library()
    with _policy(), ExitStack() as stack:
        torch.manual_seed(127)
        config = OlmoeConfig(
            vocab_size=256, hidden_size=32, intermediate_size=16, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=4, num_experts=4,
            num_experts_per_tok=2, max_position_embeddings=256, pad_token_id=0, eos_token_id=1,
        )
        config._attn_implementation = "eager"
        config._experts_implementation = "eager"
        native = OlmoeForCausalLM(config).cuda().bfloat16().requires_grad_(False).eval()
        candidate = copy.deepcopy(native)
        pool = None
        gemm = GroupedGemm(library, call_budget_bytes=16 * 2**20)
        stack.callback(gemm.close)
        wrappers = []
        for index, layer in enumerate(native.model.layers):
            directory = tmp_path / str(index)
            directory.mkdir()
            reader, baseline = _store(directory, layer.mlp.experts, device="cuda")
            stack.callback(reader.close)
            stack.callback(baseline.close)
            if pool is None:
                plan = plan_grouped_expert_buffer(reader, dtype="bfloat16", group_width=4)
                pool = GroupedExpertPool(reader, device="cuda", dtype="bfloat16", group_width=4,
                                         weight_budget_bytes=plan.weight_bytes)
                stack.callback(pool.close)
            wrapper = grouped_autograd_experts_class()(
                reader, pool, layer.mlp.experts.act_fn, gemm=gemm,
                activation_budget_bytes=256 * 2**20, activation_backend=activation_backend,
            )
            wrappers.append(wrapper)
            candidate.model.layers[index].mlp.experts = wrapper
        embedding = AdapterEmbedding(native.model.embed_tokens)
        native.model.embed_tokens = embedding
        candidate.model.embed_tokens = copy.deepcopy(embedding)
        left = [p for p in native.parameters() if p.requires_grad]
        right = [p for p in candidate.parameters() if p.requires_grad]
        before = [p.detach().clone() for p in left]
        optimizers = [torch.optim.SGD(parameters, lr=0.5) for parameters in (left, right)]
        ids = torch.arange(129, device="cuda").reshape(1, -1)
        targets = (ids + 1).remainder(config.vocab_size)
        for _step in range(2):
            expected = native(ids, use_cache=False).logits
            actual = candidate(ids, use_cache=False).logits
            assert torch.equal(expected, actual)
            for logits in (expected, actual):
                torch.nn.functional.cross_entropy(
                    logits.float().reshape(-1, config.vocab_size), targets.flatten(),
                ).backward()
            for a, b in zip(left, right):
                assert torch.equal(a.grad, b.grad)
                assert a.grad.abs().sum().item() > 0
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            assert all(torch.equal(a, b) for a, b in zip(left, right))
        assert any(not torch.equal(a, b) for a, b in zip(before, left))
        assert all(wrapper.pool is pool for wrapper in wrappers)
        assert all(wrapper.forward_trace and wrapper.backward_trace for wrapper in wrappers)
        assert all(row["status"] == "ok" for wrapper in wrappers
                   for row in wrapper.backward_trace)
        assert not any(".mlp.experts." in name for name, _ in candidate.named_parameters())
        assert gemm.grouped_matrices > 0
        record_property("scope", "two complete optimizer steps; two-layer synthetic MoE")
        record_property("activation_backend", activation_backend)
        record_property("shared_weight_budget_bytes", pool.weight_budget_bytes)


@pytest.mark.gpu
@pytest.mark.parametrize("backend,use_gemm", [("native", True), ("triton", True),
                                             ("triton", False)])
def test_cuda_fp16_autocast_is_refused_before_reads(tmp_path, backend, use_gemm):
    import torch

    from soup_cli.utils.expert_group_autograd import grouped_autograd_experts_class
    from soup_cli.utils.expert_group_gemm import GroupedGemm
    from tests.test_b1_expert_forward import _inputs
    from tests.test_d3_grouped_compute import _library, _policy
    from tests.test_d3_grouped_expert import _fixture

    library = _library()
    with _policy():
        native, reader, baseline, pool = _fixture(tmp_path, device="cuda", dtype="bfloat16")
        gemm = GroupedGemm(library, call_budget_bytes=16 * 2**20) if use_gemm else None
        try:
            candidate = grouped_autograd_experts_class()(
                reader, pool, native.act_fn, gemm=gemm, activation_budget_bytes=2**24,
                activation_backend=backend,
            )
            args = _inputs(device="cuda", dtype=torch.bfloat16)
            with torch.autocast("cuda", dtype=torch.float16), pytest.raises(
                ValueError, match="BF16 autocast or none",
            ):
                candidate(*args)
            assert reader.fetches == 0
        finally:
            if gemm is not None:
                gemm.close()
            pool.close()
            baseline.close()
            reader.close()
