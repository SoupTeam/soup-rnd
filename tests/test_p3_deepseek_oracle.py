"""Native DeepSeek saved-Jacobian checks, run by the parent on Kaggle only."""

import gc
import json
import weakref
from typing import Any

import pytest

from soup_cli.utils.host_memory_budget import HostMemoryBudget


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def device(request: Any) -> str:
    return request.param


def _torch() -> Any:
    return pytest.importorskip("torch")


def _model(
    device: str = "cpu", *, normalize: bool = True, scaling: float = 2.75,
    layers: int = 1, dense: int = 0, weight_dtype: str = "float32",
) -> Any:
    torch = _torch()
    pytest.importorskip("transformers")
    from transformers import DeepseekV3ForCausalLM

    from benchmarks.harness.b13_moe_fixture import moe_config

    config = moe_config(
        hidden_size=8, intermediate_size=16, moe_intermediate_size=8,
        num_hidden_layers=layers, first_k_dense_replace=dense, n_routed_experts=8,
        n_group=4, topk_group=2, q_lora_rank=4, kv_lora_rank=4,
        qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=4,
        num_attention_heads=2, num_key_value_heads=2, norm_topk_prob=normalize,
        routed_scaling_factor=scaling, output_router_logits=False,
    )
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    with torch.random.fork_rng(devices=[device] if device == "cuda" else []):
        torch.manual_seed(191)
        model = DeepseekV3ForCausalLM(config).to(device=device, dtype=torch.float32)
    model.requires_grad_(False)
    model.train()
    generator = torch.Generator().manual_seed(751)
    with torch.no_grad():
        for index in range(dense, layers):
            gate = model.model.layers[index].mlp.gate
            gate.weight.data = torch.randn(8, 8, generator=generator).to(
                device=device, dtype=getattr(torch, weight_dtype),
            ) * 0.55
            gate.e_score_correction_bias.copy_(torch.tensor(
                [0.375, -0.125, 0.25, -0.25, -0.375, 0.5, 0.125, -0.5], device=device,
            ))
    return model


def _gate(model: Any, layer: int = 0) -> Any:
    from soup_cli.utils.l2l_step import model_parts

    return model_parts(model)[1].layers[layer].mlp.gate


def _hidden(device: str = "cpu", dtype: str = "float32", rows: int = 3) -> Any:
    torch = _torch()
    return torch.tensor(
        [[[0.75, 0.25, -0.5, 1.0, 0.125, -0.25, 0.5, -0.75],
          [-0.25, 1.25, 0.75, -0.5, 0.25, 0.5, -0.125, 0.25],
          [1.0, -0.5, 0.25, 0.75, -0.75, 0.125, 0.25, -0.25]]],
        device=device, dtype=getattr(torch, dtype),
    )[:, :rows].contiguous()


def _incoming(device: str = "cpu", rows: int = 3) -> Any:
    torch = _torch()
    return torch.tensor(
        [[1.75, -0.625], [-2.25, 0.875], [0.3125, 1.375]],
        device=device, dtype=torch.float32,
    )[:rows].contiguous()


def _probe(model: Any, shapes: Any = None, budget: Any = None, **kwargs: Any) -> Any:
    from benchmarks.harness.p3_deepseek_oracle import DeepseekOracleProbe

    return DeepseekOracleProbe(
        model, shapes if shapes is not None else [(1, 3, 8)],
        budget if budget is not None else HostMemoryBudget(1 << 26), **kwargs,
    )


def _capture(probe: Any, gate: Any, hidden: Any, layer: int = 0, mb: int = 0) -> Any:
    torch = _torch()
    with torch.no_grad(), probe.visit(layer, mb, "forward"):
        return gate(hidden)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("scaling", [0.625, 2.75])
@pytest.mark.parametrize("input_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("weight_dtype", ["float32", "bfloat16"])
def test_original_sigmoid_vjp_matches_native_on_changed_replay_input(
    device: str, normalize: bool, scaling: float, input_dtype: str, weight_dtype: str,
) -> None:
    # A softmax transplant, missing scaling/normalizer term, or detached weight fails here.
    torch = _torch()
    model = _model(device, normalize=normalize, scaling=scaling, weight_dtype=weight_dtype)
    gate = _gate(model)
    original = _hidden(device, input_dtype).requires_grad_(True)
    native_logits, native_weights, native_ids = gate(original)
    native_weights.backward(_incoming(device))
    expected_gradient = original.grad.clone()
    assert native_logits.dtype == native_weights.dtype == torch.float32
    assert torch.isfinite(expected_gradient).all() and torch.count_nonzero(expected_gradient)
    changed = (-original.detach().roll(1, dims=-1) + 0.3125).contiguous()
    changed_logits, changed_weights, changed_ids = gate(changed)
    assert not torch.equal(changed_logits, native_logits)
    assert (not torch.equal(changed_ids, native_ids)
            or not torch.equal(changed_weights, native_weights))
    budget = HostMemoryBudget(1 << 26, reserved_bytes=37)
    probe = _probe(model, budget=budget)
    try:
        captured = _capture(probe, gate, original.detach())
        assert all(torch.equal(left, right) for left, right in zip(captured, (
            native_logits, native_weights, native_ids,
        )))
        replay = changed.requires_grad_(True)
        with probe.visit(0, 0, "recompute"):
            no_logits, weights, ids = gate(replay)
        assert no_logits is None
        assert torch.equal(weights, native_weights) and torch.equal(ids, native_ids)
        assert weights.dtype == torch.float32 and weights.requires_grad
        weights.backward(_incoming(device))
        torch.testing.assert_close(
            replay.grad, expected_gradient,
            rtol=0.008 if input_dtype == "bfloat16" else 3e-6,
            atol=6e-5 if input_dtype == "bfloat16" else 3e-7,
        )
        assert gate.weight.grad is None and not gate.weight.requires_grad
        assert gate.e_score_correction_bias.grad is None
        probe.require_complete()
        report = probe.report()
        assert report["complete"] and report["expected_pairs"] == 1
        assert report["counts"] == {
            "native_forward": 1, "oracle_recompute": 1, "oracle_vjp": 1,
        }
        assert report["forced_routing"] and not report["natural_routing_measured"]
        assert report["records_remaining"] == report["pending_replays"] == 0
        assert report["runtime"]["transformers_version"]
        assert report["source_hashes"]["native_router"]["sha256"]
        json.dumps(report, allow_nan=False)
    finally:
        probe.close()
    assert budget.used_bytes == 37
    assert probe.report()["complete"] and probe.report()["closed"]
    assert probe.report()["host_reserved_bytes"] == 0


def test_gate_gradient_reaches_trainable_upstream_projection_with_frozen_router(
    device: str,
) -> None:
    # Returning native weights detached would silently lose this adapter's gradient.
    torch = _torch()
    model = _model(device, weight_dtype="bfloat16")
    gate = _gate(model)
    projection = torch.nn.Linear(8, 8, bias=False, device=device, dtype=torch.float32)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(8, device=device) * 0.75)
    source = _hidden(device)
    original = projection(source).to(torch.bfloat16)
    _, native_weights, _ = gate(original)
    native_weights.backward(_incoming(device))
    expected = projection.weight.grad.clone()
    assert torch.count_nonzero(expected)
    projection.zero_grad(set_to_none=True)
    with _probe(model) as probe:
        _capture(probe, gate, original.detach())
        with probe.visit(0, 0, "recompute"):
            _, weights, _ = gate(projection(source).to(torch.bfloat16))
        weights.backward(_incoming(device))
        probe.require_complete()
        torch.testing.assert_close(projection.weight.grad, expected, rtol=0.008, atol=6e-5)


def test_forward_primal_is_not_cached_and_gate_replay_saves_no_hidden_tensor() -> None:
    # A hidden-primal cache or ctx.save_for_backward(hidden) fails both ownership checks.
    torch = _torch()
    model = _model()
    gate = _gate(model)
    probe = _probe(model)
    hidden = _hidden()
    primal = weakref.ref(hidden)
    _capture(probe, gate, hidden)
    del hidden
    gc.collect()
    assert primal() is None
    saved = []
    live = _hidden().requires_grad_(True)
    with torch.autograd.graph.saved_tensors_hooks(lambda value: saved.append(value) or value,
                                                 lambda value: value):
        with probe.visit(0, 0, "recompute"):
            _, weights, _ = gate(live)
    assert saved == []
    weights.backward(_incoming())
    probe.require_complete()
    probe.close()


def test_grouped_bias_selects_ids_but_does_not_enter_selected_sigmoid_values() -> None:
    # Using correction-biased scores as Jacobian primals gives a different native VJP.
    torch = _torch()
    model = _model(normalize=False, scaling=1.5)
    gate = _gate(model)
    with torch.no_grad():
        gate.weight.zero_()
        gate.weight[0, 0], gate.weight[1, 1] = 1.0, 1.0
        gate.e_score_correction_bias.copy_(torch.tensor([0, 0, 8, 7, 6, 5, 0, 0]))
    hidden = _hidden().requires_grad_(True)
    logits, expected, ids = gate(hidden)
    assert torch.equal(expected, logits.sigmoid().gather(1, ids) * 1.5)
    assert all(expert in {2, 3, 4, 5} for expert in ids.flatten().tolist())
    expected.backward(_incoming())
    with _probe(model) as probe:
        _capture(probe, gate, hidden.detach())
        replay = (-hidden.detach()).requires_grad_(True)
        with probe.visit(0, 0, "recompute"):
            _, weights, forced = gate(replay)
        weights.backward(_incoming())
        assert torch.equal(forced, ids) and torch.equal(weights, expected)
        torch.testing.assert_close(replay.grad, hidden.grad)
        probe.require_complete()


def test_extreme_sigmoid_scores_and_tied_choices_preserve_native_vjp(device: str) -> None:
    torch = _torch()
    model = _model(device, scaling=0.75)
    gate = _gate(model)
    with torch.no_grad():
        gate.weight.zero_()
        gate.weight[:, 0] = 1.0
        gate.e_score_correction_bias.zero_()
    hidden = torch.zeros((1, 3, 8), device=device)
    hidden[0, :, 0] = torch.tensor([0.0, -1000.0, 1000.0], device=device)
    native = hidden.clone().requires_grad_(True)
    _, native_weights, native_ids = gate(native)
    native_weights.backward(_incoming(device))
    with _probe(model) as probe:
        _capture(probe, gate, hidden)
        replay = (-hidden).requires_grad_(True)
        with probe.visit(0, 0, "recompute"):
            _, weights, ids = gate(replay)
        assert torch.equal(ids, native_ids) and torch.equal(weights, native_weights)
        weights.backward(_incoming(device))
        assert torch.isfinite(replay.grad).all()
        torch.testing.assert_close(replay.grad, native.grad, atol=1e-7, rtol=3e-6)
        probe.require_complete()


def test_dense_visits_are_allowed_and_each_moe_microbatch_requires_its_own_vjp() -> None:
    torch = _torch()
    model = _model(layers=3, dense=1)
    shapes = [(1, 3, 8), (1, 2, 8)]
    with _probe(model, shapes) as probe:
        assert probe.report()["expected_pairs"] == 4
        for layer in range(3):
            for mb, rows in enumerate((3, 2)):
                with torch.no_grad(), probe.visit(layer, mb, "forward"):
                    if layer:
                        _gate(model, layer)(_hidden(rows=rows))
        for layer in (2, 1, 0):
            for mb, rows in enumerate((3, 2)):
                live = (-_hidden(rows=rows)).requires_grad_(True)
                with probe.visit(layer, mb, "recompute"):
                    if layer:
                        _, weights, _ = _gate(model, layer)(live)
                if layer:
                    weights.backward(_incoming(rows=rows))
        probe.require_complete()
        assert probe.report()["counts"] == {
            "native_forward": 4, "oracle_recompute": 4, "oracle_vjp": 4,
        }


class _TensorSource:
    def __init__(self, values: list[dict[str, Any]]) -> None:
        self.values = values

    def get(self, layer: int, name: str) -> Any:
        return self.values[layer][name]


def _streamed_model(
    device: str, *, peft: bool = False, resident_bias: bool = False,
) -> tuple[Any, Any, Any]:
    """Use production wrappers and functional substitution, with real tiny gate slots."""
    torch = _torch()
    from soup_cli.utils.l2l_step import model_parts
    from soup_cli.utils.layer_stream_runtime import (
        LayerBufferPool,
        StreamedDecoderLayer,
        StreamPrefetcher,
    )

    model = _model(device, layers=2, weight_dtype="bfloat16")
    if peft:
        pytest.importorskip("peft")
        from peft import LoraConfig, TaskType, get_peft_model

        model = get_peft_model(model, LoraConfig(
            r=2, lora_alpha=4, target_modules=["q_a_proj"], task_type=TaskType.CAUSAL_LM,
        ))
        model.requires_grad_(False)
    owner = model_parts(model)[1]
    values = [
        {"weight": layer.mlp.gate.weight.detach().clone(),
         "bias": layer.mlp.gate.e_score_correction_bias.detach().clone()}
        for layer in owner.layers
    ]
    source = _TensorSource(values)
    pool = LayerBufferPool({"weight": ((8, 8), "bfloat16"), "bias": ((8,), "float32")},
                           n_buffers=2, device=device)
    prefetcher = StreamPrefetcher(pool, source, 2)
    for index, inner in enumerate(tuple(owner.layers)):
        inner.mlp.gate.weight = torch.nn.Parameter(
            torch.empty((8, 8), device="meta", dtype=torch.bfloat16), requires_grad=False,
        )
        mapping = {"mlp.gate.weight": "weight"}
        if not resident_bias:
            inner.mlp.gate.e_score_correction_bias = torch.empty(8, device="meta")
            mapping["mlp.gate.e_score_correction_bias"] = "bias"
        owner.layers[index] = StreamedDecoderLayer(
            inner, index, pool, prefetcher, name_map=mapping,
            use_checkpoint=False,
        )
    return model, pool, source


def _streamed_call(model: Any, pool: Any, source: Any, hidden: Any, layer: int) -> Any:
    from soup_cli.utils.l2l_step import model_parts

    torch = _torch()
    owner = model_parts(model)[1]
    pool.load_async(layer, source)
    positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
    rotary = owner.rotary_emb(hidden, positions)
    return owner.layers[layer](hidden, position_embeddings=rotary, position_ids=positions)


@pytest.mark.parametrize("peft", [False, True])
def test_native_gates_under_streamed_and_peft_wrappers_use_restored_frozen_weights(
    device: str, peft: bool,
) -> None:
    # Reading gate.weight in backward finds a meta placeholder after functional_call unwinds.
    torch = _torch()
    model, pool, source = _streamed_model(device, peft=peft)
    originals = [_gate(model, index).forward for index in range(2)]
    with _probe(model) as probe:
        for layer in range(2):
            with torch.no_grad(), probe.visit(layer, 0, "forward"):
                _streamed_call(model, pool, source, _hidden(device), layer)
            assert _gate(model, layer).weight.is_meta
        for layer in (1, 0):
            hidden = _hidden(device).requires_grad_(True)
            with probe.visit(layer, 0, "recompute"):
                output = _streamed_call(model, pool, source, hidden, layer)
            assert _gate(model, layer).weight.is_meta
            output.square().sum().backward()
            assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
            assert torch.count_nonzero(hidden.grad)
        probe.require_complete()
    assert all(_gate(model, layer).forward == originals[layer] for layer in range(2))


def test_functional_call_gate_vjp_matches_native_after_parameters_return_to_meta(
    device: str,
) -> None:
    # This isolates the gate derivative from the decoder's independent residual gradient.
    torch = _torch()
    from torch.func import functional_call

    model, pool, source = _streamed_model(device)
    gate = _gate(model)
    pool.load_async(0, source)
    buffers = pool.wait(0)
    substitutions = {"weight": buffers["weight"], "e_score_correction_bias": buffers["bias"]}
    original = _hidden(device, "bfloat16").requires_grad_(True)
    _, native_weights, native_ids = functional_call(gate, substitutions, (original,))
    native_weights.backward(_incoming(device))
    expected = original.grad.clone()
    assert torch.count_nonzero(expected) and gate.weight.is_meta
    with _probe(model, [(1, 3, 8), (1, 2, 8)]) as probe:
        for layer in range(2):
            for mb, rows in enumerate((3, 2)):
                pool.load_async(layer, source)
                current = pool.wait(layer)
                weights = {"weight": current["weight"],
                           "e_score_correction_bias": current["bias"]}
                with torch.no_grad(), probe.visit(layer, mb, "forward"):
                    functional_call(_gate(model, layer), weights,
                                    (_hidden(device, "bfloat16", rows),))
        for layer in (1, 0):
            for mb, rows in enumerate((3, 2)):
                pool.load_async(layer, source)
                current = pool.wait(layer)
                weights = {"weight": current["weight"],
                           "e_score_correction_bias": current["bias"]}
                live = (-_hidden(device, "bfloat16", rows)).requires_grad_(True)
                with probe.visit(layer, mb, "recompute"):
                    _, replay_weights, ids = functional_call(_gate(model, layer), weights, (live,))
                state = probe._pending[(layer, mb)]
                assert state.weight is current["weight"] and not state.weight.requires_grad
                assert state.bias is current["bias"]
                assert _gate(model, layer).weight.is_meta
                replay_weights.backward(_incoming(device, rows))
                assert state.closed and state.weight is None
                assert state.bias is None
                if layer == 0 and mb == 0:
                    assert torch.equal(ids, native_ids)
                    assert torch.equal(replay_weights, native_weights)
                    torch.testing.assert_close(live.grad, expected, rtol=0.008, atol=6e-5)
        probe.require_complete()
    assert not model._forward_pre_hooks and not model.model._forward_pre_hooks


def test_recycled_stream_slot_refuses_backward_instead_of_using_another_layer_weight() -> None:
    torch = _torch()
    model, pool, source = _streamed_model("cpu")
    probe = _probe(model)
    with torch.no_grad(), probe.visit(0, 0, "forward"):
        _streamed_call(model, pool, source, _hidden(), 0)
    hidden = _hidden().requires_grad_(True)
    with probe.visit(0, 0, "recompute"):
        output = _streamed_call(model, pool, source, hidden, 0)
    pool.owner[pool.slot_for(0)] = 9
    with pytest.raises(RuntimeError, match="slot"):
        output.sum().backward()
    assert probe.report()["closed"] and not probe.report()["complete"]


def test_restored_streamed_weight_must_match_original_forward_content() -> None:
    torch = _torch()
    model, pool, source = _streamed_model("cpu")
    probe = _probe(model)
    with torch.no_grad(), probe.visit(0, 0, "forward"):
        _streamed_call(model, pool, source, _hidden(), 0)
    source.values[0]["weight"].add_(0.25)
    with pytest.raises(RuntimeError, match="weight|mutat"):
        with probe.visit(0, 0, "recompute"):
            _streamed_call(model, pool, source, _hidden().requires_grad_(True), 0)
    assert probe.report()["closed"]


@pytest.mark.parametrize("missing", ["forward", "recompute", "vjp", "gate_call"])
def test_incomplete_window_blocks_consumer_update_and_cleans_resources(missing: str) -> None:
    torch = _torch()
    model = _model()
    budget = HostMemoryBudget(1 << 26, reserved_bytes=37)
    gate = _gate(model)
    original = gate.forward
    probe = _probe(model, budget=budget)
    updates = []
    with pytest.raises(RuntimeError, match="complete|forward|gate"):
        if missing != "forward":
            if missing == "gate_call":
                with torch.no_grad(), probe.visit(0, 0, "forward"):
                    pass
            else:
                _capture(probe, gate, _hidden())
        if missing in {"vjp", "forward"}:
            with probe.visit(0, 0, "recompute"):
                _, weights, _ = gate(_hidden().requires_grad_(True))
            assert weights.requires_grad
        probe.require_complete()
        updates.append("optimizer.step")
    assert not updates and probe.report()["closed"]
    assert budget.used_bytes == 37 and gate.forward == original
    assert probe.report()["records_remaining"] == probe.report()["pending_replays"] == 0
    probe.close()


@pytest.mark.parametrize("phase", ["forward", "recompute"])
def test_duplicate_pairs_refuse_and_restore_native_method(phase: str) -> None:
    model = _model()
    gate = _gate(model)
    original = gate.forward
    probe = _probe(model)
    _capture(probe, gate, _hidden())
    if phase == "recompute":
        with probe.visit(0, 0, "recompute"):
            _, weights, _ = gate(_hidden().requires_grad_(True))
        weights.backward(_incoming())
    with pytest.raises(RuntimeError, match="duplicate"):
        if phase == "forward":
            _capture(probe, gate, _hidden())
        else:
            with probe.visit(0, 0, "recompute"):
                gate(_hidden().requires_grad_(True))
    assert probe.report()["closed"] and gate.forward == original


def test_second_unfinished_replay_is_refused_and_first_graph_is_invalidated() -> None:
    model = _model()
    gate = _gate(model)
    probe = _probe(model, [(1, 3, 8), (1, 3, 8)])
    for mb in (0, 1):
        _capture(probe, gate, _hidden(), mb=mb)
    with probe.visit(0, 0, "recompute"):
        _, weights, _ = gate(_hidden().requires_grad_(True))
    with pytest.raises(RuntimeError, match="unfinished|pending"):
        with probe.visit(0, 1, "recompute"):
            gate(_hidden().requires_grad_(True))
    assert probe.report()["pending_replays"] == 0
    with pytest.raises(RuntimeError, match="closed"):
        weights.backward(_incoming())


@pytest.mark.parametrize("mutation", ["weight", "bias", "norm", "scale", "ids", "scores",
                                      "normalizer"])
@pytest.mark.parametrize("when", ["before_recompute", "before_backward"])
def test_mutated_router_or_saved_forward_state_is_rejected(mutation: str, when: str) -> None:
    torch = _torch()
    model = _model()
    gate = _gate(model)
    budget = HostMemoryBudget(1 << 26)
    probe = _probe(model, budget=budget)
    _capture(probe, gate, _hidden())
    if when == "before_backward":
        with probe.visit(0, 0, "recompute"):
            _, weights, _ = gate(_hidden().requires_grad_(True))
    with torch.no_grad():
        if mutation == "weight":
            gate.weight.add_(0.125)
        elif mutation == "bias":
            gate.e_score_correction_bias.add_(0.125)
        elif mutation == "norm":
            gate.norm_topk_prob = False
        elif mutation == "scale":
            gate.routed_scaling_factor += 0.125
        else:
            record = probe._records[(0, 0)]
            tensor = getattr(record, mutation)
            # .data bypasses version counters; content integrity must still reject it.
            tensor.data.view(-1)[0] += 1 if mutation == "ids" else 0.125
    with pytest.raises(RuntimeError, match="changed|mutat"):
        if when == "before_recompute":
            with probe.visit(0, 0, "recompute"):
                gate(_hidden().requires_grad_(True))
        else:
            weights.backward(_incoming())
    assert probe.report()["closed"] and budget.used_bytes == 0
    assert not probe.report()["complete"]


def test_nonfinite_backward_retained_exception_drops_owned_caches_and_graph_copies() -> None:
    torch = _torch()
    model = _model()
    gate = _gate(model)
    budget = HostMemoryBudget(1 << 26)
    probe = _probe(model, budget=budget)
    refs = [weakref.ref(probe._ids), weakref.ref(probe._scores), weakref.ref(probe._normalizers)]
    _capture(probe, gate, _hidden())
    with probe.visit(0, 0, "recompute"):
        no_logits, weights, ids = gate(_hidden().requires_grad_(True))
    state = next(iter(probe._pending.values()))
    refs.extend(weakref.ref(value) for value in (state.ids, state.scores, state.normalizer))
    with pytest.raises(RuntimeError, match="nonfinite") as held:
        weights.backward(torch.full_like(weights, float("nan")))
    gc.collect()
    assert held.value.__traceback__ is not None
    assert no_logits is None and ids.shape == (3, 2) and weights.grad_fn is not None
    assert all(reference() is None for reference in refs)
    assert budget.used_bytes == 0 and probe.report()["closed"]
    assert probe.report()["owned_host_storage_bytes"] == 0
    assert probe.report()["owned_device_storage_bytes"] == 0


def test_exception_after_replay_restores_preexisting_forward_override_and_cancels_graph() -> None:
    model = _model()
    gate = _gate(model)
    native = gate.forward
    gate.forward = native
    budget = HostMemoryBudget(1 << 26)
    probe = _probe(model, budget=budget)
    _capture(probe, gate, _hidden())
    with pytest.raises(ValueError, match="consumer failure") as held:
        with probe.visit(0, 0, "recompute"):
            _, weights, _ = gate(_hidden().requires_grad_(True))
            raise ValueError("consumer failure")
    assert held.value.__traceback__ is not None
    assert gate.__dict__["forward"] is native and budget.used_bytes == 0
    assert probe.report()["closed"]
    probe.close()
    with pytest.raises(RuntimeError, match="closed"):
        weights.backward(_incoming())


def test_constructor_allocation_failure_releases_lease_without_losing_native_methods(
    monkeypatch: Any,
) -> None:
    torch = _torch()
    model = _model(layers=2)
    gates = [_gate(model, layer) for layer in range(2)]
    originals = [gate.forward for gate in gates]
    budget = HostMemoryBudget(1 << 26, reserved_bytes=37)
    native_empty = torch.empty
    refs = []
    attempts = 0

    def fail_third_allocation(*args: Any, **kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 3:
            raise MemoryError("injected normalizer allocation failure")
        result = native_empty(*args, **kwargs)
        refs.append(weakref.ref(result))
        return result

    monkeypatch.setattr(torch, "empty", fail_third_allocation)
    with pytest.raises(MemoryError, match="normalizer allocation") as held:
        _probe(model, budget=budget)
    gc.collect()
    assert held.value.__traceback__ is not None
    assert refs and all(reference() is None for reference in refs)
    assert budget.used_bytes == 37
    assert all(gate.forward == original for gate, original in zip(gates, originals))
    assert not model._forward_pre_hooks and not model.model._forward_pre_hooks


def test_partial_method_install_failure_restores_every_previously_intercepted_gate(
    monkeypatch: Any,
) -> None:
    from benchmarks.harness.p3_deepseek_oracle import DeepseekOracleProbe

    model = _model(layers=2)
    originals = [_gate(model, layer).forward for layer in (0, 1)]
    budget = HostMemoryBudget(1 << 26)
    native_wrapper = DeepseekOracleProbe._wrapper

    def refuse_second(probe: Any, layer: int, gate: Any, native: Any) -> Any:
        if layer == 1:
            raise RuntimeError("injected second method installation failure")
        return native_wrapper(probe, layer, gate, native)

    monkeypatch.setattr(DeepseekOracleProbe, "_wrapper", refuse_second)
    with pytest.raises(RuntimeError, match="method installation"):
        _probe(model, budget=budget)
    assert budget.used_bytes == 0
    assert all(_gate(model, layer).forward == originals[layer] for layer in (0, 1))
    assert all("forward" not in _gate(model, layer).__dict__ for layer in (0, 1))


def test_host_reservation_covers_records_hashing_and_cpu_operation_scratch() -> None:
    model = _model(layers=3, dense=1)
    shapes = [(1, 3, 8), (1, 2, 8)]
    budget = HostMemoryBudget(1 << 26, reserved_bytes=37)
    probe = _probe(model, shapes, budget)
    report = probe.report()
    allocation = report["allocation"]
    # Two MoE layers, five tokens each, top2. No hidden-state primal storage.
    assert allocation["host"]["ids"] == 160
    assert allocation["host"]["selected_sigmoid_scores"] == 80
    assert allocation["host"]["normalizers"] == 40
    assert report["owned_host_storage_bytes"] == 280
    assert allocation["host"]["hash_staging"] > 0
    assert allocation["host"]["cpu_operation_scratch"] > 0
    charge = report["host_reserved_bytes"]
    assert charge == sum(allocation["host"].values()) == budget.used_bytes - 37
    assert report["device_scratch_bound_bytes"] == 0
    probe.close()
    capped = HostMemoryBudget(charge - 1)
    with pytest.raises(MemoryError):
        _probe(model, shapes, capped)
    assert capped.used_bytes == 0 and budget.used_bytes == 37
    assert all("forward" not in _gate(model, layer).__dict__ for layer in (1, 2))


@pytest.mark.gpu
def test_cuda_scratch_is_largest_pair_only_and_enforced_before_patching() -> None:
    model = _model("cuda", layers=2)
    with _probe(model) as first:
        one = first.report()
    with _probe(model, [(1, 3, 8), (1, 2, 8)]) as second:
        many = second.report()
    assert one["device_scratch_bound_bytes"] == many["device_scratch_bound_bytes"] > 0
    assert many["allocation"]["host"]["cpu_operation_scratch"] == 0
    budget = HostMemoryBudget(1 << 26)
    with pytest.raises(MemoryError, match="scratch"):
        _probe(model, budget=budget,
               device_scratch_limit=one["device_scratch_bound_bytes"] - 1)
    assert budget.used_bytes == 0
    assert all("forward" not in _gate(model, layer).__dict__ for layer in (0, 1))


@pytest.mark.gpu
def test_failed_cuda_drain_keeps_owned_storage_and_lease_until_close_retry(
    monkeypatch: Any,
) -> None:
    torch = _torch()
    model = _model("cuda")
    budget = HostMemoryBudget(1 << 26)
    probe = _probe(model, budget=budget)
    _capture(probe, _gate(model), _hidden("cuda"))
    with probe.visit(0, 0, "recompute"):
        _, weights, _ = _gate(model)(_hidden("cuda").requires_grad_(True))
    stream_class = type(torch.cuda.current_stream())
    synchronize = stream_class.synchronize
    attempts = 0

    def fail_once(stream: Any) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("injected CUDA drain failure")
        synchronize(stream)

    monkeypatch.setattr(stream_class, "synchronize", fail_once)
    with pytest.raises(RuntimeError, match="CUDA drain"):
        probe.close()
    report = probe.report()
    assert not report["closed"] and report["pending_replays"] == 1
    assert report["owned_host_storage_bytes"] and report["owned_device_storage_bytes"]
    assert report["host_reserved_bytes"] == budget.used_bytes > 0
    assert "forward" not in _gate(model).__dict__
    probe.close()
    assert probe.report()["closed"] and budget.used_bytes == 0
    with pytest.raises(RuntimeError, match="closed"):
        weights.backward(_incoming("cuda"))


@pytest.mark.parametrize("mode", ["higher_order", "no_grad", "autocast", "nonfinite_input",
                                  "detached"])
def test_unsupported_derivative_modes_fail_closed(mode: str) -> None:
    torch = _torch()
    model = _model()
    gate = _gate(model)
    probe = _probe(model)
    _capture(probe, gate, _hidden())
    expected = {"higher_order": "first-order", "no_grad": "grad", "autocast": "autocast",
                "nonfinite_input": "nonfinite", "detached": "differentiable"}[mode]
    with pytest.raises(RuntimeError, match=expected):
        live = _hidden().requires_grad_(True)
        if mode == "higher_order":
            with probe.visit(0, 0, "recompute"):
                _, weights, _ = gate(live)
            torch.autograd.grad(weights.sum(), live, create_graph=True)
        elif mode == "no_grad":
            with torch.no_grad(), probe.visit(0, 0, "recompute"):
                gate(live)
        elif mode == "autocast":
            with torch.autocast("cpu", dtype=torch.bfloat16), probe.visit(0, 0, "recompute"):
                gate(live)
        elif mode == "detached":
            with probe.visit(0, 0, "recompute"):
                gate(live.detach())
        else:
            with torch.no_grad():
                live[0, 0, 0] = float("nan")
            with probe.visit(0, 0, "recompute"):
                gate(live)
    assert probe.report()["closed"] and probe.report()["pending_replays"] == 0


@pytest.mark.parametrize("mode", ["config", "keyword", "aux_loss", "gate_hook", "moe_forward"])
def test_full_logit_and_auxiliary_consumers_are_explicitly_rejected(mode: str) -> None:
    model = _model()
    gate = _gate(model)
    probe = _probe(model)
    handle = None
    try:
        with pytest.raises(RuntimeError, match="logit|auxiliary|hook"):
            if mode == "config":
                model.config.output_router_logits = True
                _capture(probe, gate, _hidden())
            elif mode == "keyword":
                model(input_ids=_torch().tensor([[1, 3, 7]]), output_router_logits=True)
            elif mode == "aux_loss":
                model.config.router_aux_loss_coef = 0.1
                _capture(probe, gate, _hidden())
            elif mode == "moe_forward":
                model.model.layers[0].mlp.forward = lambda hidden: hidden
                _capture(probe, gate, _hidden())
            else:
                handle = gate.register_forward_hook(lambda _module, _args, output: output)
                _capture(probe, gate, _hidden())
        assert probe.report()["closed"]
    finally:
        if handle is not None:
            handle.remove()
        probe.close()


@pytest.mark.parametrize("visit", [(0, 0, "backward"), (0, -1, "forward"),
                                   (4, 0, "forward"), (None, 0, "forward")])
def test_invalid_visit_metadata_refuses_without_leaking(visit: tuple[Any, ...]) -> None:
    probe = _probe(_model())
    with pytest.raises(RuntimeError):
        with probe.visit(*visit):
            pass
    assert probe.report()["closed"] and probe.report()["host_reserved_bytes"] == 0


def test_non_native_router_and_trainable_router_are_not_silently_detached() -> None:
    torch = _torch()
    model = _model()
    _gate(model).weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="frozen"):
        _probe(model)
    _gate(model).weight.requires_grad_(False)
    model.model.layers[0].mlp.gate = torch.nn.Linear(8, 8, bias=False)
    with pytest.raises(RuntimeError, match="native"):
        _probe(model)


def test_second_probe_cannot_intercept_an_already_owned_native_gate() -> None:
    model = _model()
    first = _probe(model)
    budget = HostMemoryBudget(1 << 26)
    with pytest.raises(RuntimeError, match="native|intercept"):
        _probe(model, budget=budget)
    assert budget.used_bytes == 0 and not first.report()["closed"]
    _capture(first, _gate(model), _hidden())
    with first.visit(0, 0, "recompute"):
        _, weights, _ = _gate(model)(_hidden().requires_grad_(True))
    weights.backward(_incoming())
    first.require_complete()
    first.close()


def test_gate_call_without_visit_cannot_escape_window_accounting() -> None:
    model = _model()
    probe = _probe(model)
    with pytest.raises(RuntimeError, match="visit"):
        _gate(model)(_hidden())
    assert probe.report()["closed"]


def test_replay_output_mutation_cannot_corrupt_saved_original_scores_or_ids() -> None:
    torch = _torch()
    model = _model()
    gate = _gate(model)
    original = _hidden().requires_grad_(True)
    _, native_weights, _ = gate(original)
    native_weights.backward(_incoming())
    expected = original.grad.clone()
    with _probe(model) as probe:
        _capture(probe, gate, original.detach())
        live = (-original.detach()).requires_grad_(True)
        with probe.visit(0, 0, "recompute"):
            _, weights, ids = gate(live)
        with torch.no_grad():
            weights.add_(11)
            ids.zero_()
        weights.backward(_incoming())
        torch.testing.assert_close(live.grad, expected, rtol=3e-6, atol=3e-7)
        probe.require_complete()


def test_noncontiguous_upstream_gradient_preserves_native_input_vjp() -> None:
    torch = _torch()
    model = _model()
    gate = _gate(model)
    original = _hidden().requires_grad_(True)
    incoming = _incoming().t().contiguous().t()
    assert not incoming.is_contiguous()
    _, native_weights, _ = gate(original)
    native_weights.backward(incoming)
    with _probe(model) as probe:
        _capture(probe, gate, original.detach())
        live = (-original.detach()).requires_grad_(True)
        with probe.visit(0, 0, "recompute"):
            _, weights, _ = gate(live)
        weights.backward(incoming)
        torch.testing.assert_close(live.grad, original.grad, rtol=3e-6, atol=3e-7)
        probe.require_complete()


def test_production_style_streamed_weight_with_resident_bias_completes_and_checks_bias() -> None:
    torch = _torch()
    model, pool, source = _streamed_model("cpu", resident_bias=True)
    with _probe(model) as probe:
        for layer in range(2):
            with torch.no_grad(), probe.visit(layer, 0, "forward"):
                _streamed_call(model, pool, source, _hidden(), layer)
        for layer in (1, 0):
            hidden = _hidden().requires_grad_(True)
            with probe.visit(layer, 0, "recompute"):
                output = _streamed_call(model, pool, source, hidden, layer)
            assert _gate(model, layer).weight.is_meta
            assert not _gate(model, layer).e_score_correction_bias.is_meta
            output.square().sum().backward()
        probe.require_complete()
    probe = _probe(model)
    with torch.no_grad(), probe.visit(0, 0, "forward"):
        _streamed_call(model, pool, source, _hidden(), 0)
    _gate(model).e_score_correction_bias.add_(0.25)
    with pytest.raises(RuntimeError, match="bias.*changed"):
        with probe.visit(0, 0, "recompute"):
            _streamed_call(model, pool, source, _hidden().requires_grad_(True), 0)
    assert probe.report()["closed"]


@pytest.mark.parametrize("field", ["weight", "e_score_correction_bias"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_frozen_native_state_is_rejected_before_capture(field: str, value: float) -> None:
    torch = _torch()
    model = _model()
    gate = _gate(model)
    budget = HostMemoryBudget(1 << 26)
    probe = _probe(model, budget=budget)
    with torch.no_grad():
        getattr(gate, field).view(-1)[0] = value
    with pytest.raises(RuntimeError, match="nonfinite"):
        _capture(probe, gate, _hidden())
    assert probe.report()["closed"] and budget.used_bytes == 0


def test_nested_visit_fails_closed_without_restoring_an_obsolete_outer_visit() -> None:
    probe = _probe(_model())
    with pytest.raises(RuntimeError, match="nested"):
        with probe.visit(0, 0, "forward"):
            with probe.visit(0, 0, "forward"):
                pass
    assert probe.report()["closed"] and probe._visit is None


@pytest.mark.parametrize("field", ["ids", "scores", "normalizer"])
def test_pending_replay_tensor_mutation_fails_before_input_vjp(field: str) -> None:
    torch = _torch()
    model = _model()
    gate = _gate(model)
    budget = HostMemoryBudget(1 << 26)
    probe = _probe(model, budget=budget)
    _capture(probe, gate, _hidden())
    live = _hidden().requires_grad_(True)
    with probe.visit(0, 0, "recompute"):
        _, weights, _ = gate(live)
    state = probe._pending[(0, 0)]
    with torch.no_grad():
        getattr(state, field).data.view(-1)[0] += 1
    with pytest.raises(RuntimeError, match="mutat"):
        weights.backward(_incoming())
    assert live.grad is None and probe.report()["counts"]["oracle_vjp"] == 0
    assert state.closed and state.weight is None
    assert probe.report()["closed"] and budget.used_bytes == 0


def test_native_flattened_gate_input_restores_the_original_flattened_gradient_shape() -> None:
    torch = _torch()
    model = _model()
    gate = _gate(model)
    original = _hidden().view(3, 8).requires_grad_(True)
    _, native_weights, _ = gate(original)
    native_weights.backward(_incoming())
    with _probe(model) as probe:
        _capture(probe, gate, original.detach())
        live = (-original.detach()).requires_grad_(True)
        with probe.visit(0, 0, "recompute"):
            _, weights, _ = gate(live)
        weights.backward(_incoming())
        assert live.grad.shape == (3, 8)
        torch.testing.assert_close(live.grad, original.grad, rtol=3e-6, atol=3e-7)
        probe.require_complete()
