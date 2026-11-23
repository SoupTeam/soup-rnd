"""Ticket 07: the routing Oracle on CPU, as a BF16 lever over route-preserving MBR.

Three seams, all CPU:

* one native ``DeepseekV3TopkRouter`` of the tiny fixture, compared against
  ``DeepseekRoutingOracle``: the forward replay must return the native logits, routing
  weights and indices bit for bit, and the saved original-forward sigmoid Jacobian must
  reproduce native autograd's gradient of the routing weights into the router input. The
  known dense-FP32 fact is measured, not hidden: with ``norm_topk_prob`` the two backward
  paths differ at rounding level (~1e-7 relative L2), without it they are bit-equal.
* the whole tiny MoE step through ``L2LStep.run(..., route_preserving_mbr=True)`` with and
  without the Oracle: per-micro-batch losses and every trainable gradient must be bit-equal
  at k = 1, 2, 4, and the recompute must report zero routing-weight drift.
* the Oracle's own telemetry: every MoE (layer, micro-batch) records one native forward and
  one replayed recompute, so exactly half of MBR's router executions are skipped.

The Oracle is only defined for frozen routers: the router parameter gets no gradient in
either arm. "Gate-weight gradient" below is the upstream gradient of the gate's routing
weights, the input the replay consumes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from benchmarks.harness import b13_moe_fixture as fixture
from soup_cli.utils import l2l_step
from soup_cli.utils.routing_oracle import DeepseekRoutingOracle
from tests.test_b13_l2l_moe import Rig, _collect, _zero_grads
from tests.test_b13_l2l_moe import make_rig as make_rig  # noqa: F401  (pytest fixture)
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR
from tests.test_p72_mbr_bitexact import bits_equal, differing

pytestmark = _REQUIRES_TRAINING_FLOOR


def _agreement(reference: Any, candidate: Any) -> Dict[str, Any]:
    """``torch.equal``, max abs and relative L2 of one gradient pair."""
    import torch

    difference = candidate.double() - reference.double()
    scale = float(reference.double().norm())
    return {
        "equal": bool(torch.equal(reference, candidate)),
        "max_abs": float(difference.abs().max()),
        "rel_l2": float(difference.norm()) / scale if scale else float(difference.norm()),
    }


def _resident(tmp_path: Any, *, norm_topk_prob: bool, scaling: float) -> Any:
    weights = fixture.moe_weights_dir(tmp_path)
    resident = fixture.resident_adapter(fixture.resident_base(weights), fixture.moe_lora_config())
    config = resident.get_base_model().config
    for layer in resident.get_base_model().model.layers[config.first_k_dense_replace:]:
        layer.mlp.gate.norm_topk_prob = norm_topk_prob
        layer.mlp.gate.routed_scaling_factor = scaling
    return resident


@pytest.mark.parametrize("scaling", [1.0, 2.5])
@pytest.mark.parametrize("norm_topk_prob", [True, False])
def test_oracle_replay_matches_native_gate_forward_and_input_gradient(
    tmp_path: Any, norm_topk_prob: bool, scaling: float, record_property: Any,
) -> None:
    """One native gate call against the Oracle forward/peek/recompute phases, FP32."""
    import torch

    resident = _resident(tmp_path, norm_topk_prob=norm_topk_prob, scaling=scaling)
    config = resident.get_base_model().config
    owner = resident.get_base_model().model
    layer_idx = config.first_k_dense_replace
    gate = owner.layers[layer_idx].mlp.gate
    hidden = torch.randn(5, config.hidden_size)

    inputs = hidden.clone().requires_grad_(True)
    logits, weights, ids = type(gate).forward(gate, inputs)
    upstream = torch.randn_like(weights)
    (native_gradient,) = torch.autograd.grad(weights, inputs, upstream)

    with DeepseekRoutingOracle(resident, pin=False) as oracle:
        oracle.begin_step([(1, 5, config.hidden_size)])
        with torch.no_grad():
            with oracle.visit(layer_idx, 0, "forward"):
                forward = gate(hidden)
            with oracle.visit(layer_idx, 0, "peek"):
                peek = gate(hidden)
        replay_input = hidden.clone().requires_grad_(True)
        with torch.enable_grad():
            with oracle.visit(layer_idx, 0, "recompute"):
                replay = gate(replay_input)
        (replay_gradient,) = torch.autograd.grad(replay[1], replay_input, upstream)
        report = oracle.report()
        oracle.abort_step()

    assert report["step_counts"] == {
        "native_forward": 1, "oracle_peek": 1, "oracle_recompute": 1, "oracle_vjp": 1
    }
    # The forward phase is the native gate untouched: logits, weights and indices.
    assert torch.equal(forward[0], logits)
    assert torch.equal(forward[1], weights)
    assert torch.equal(forward[2], ids)
    # The saved forward scores reproduce the weights and routes the recompute skips.
    assert peek[0] is None and replay[0] is None
    assert torch.equal(peek[1], weights) and torch.equal(peek[2], ids)
    assert torch.equal(replay[1], weights) and torch.equal(replay[2], ids)

    agreement = _agreement(native_gradient, replay_gradient)
    record_property("norm_topk_prob", norm_topk_prob)
    record_property("routed_scaling_factor", scaling)
    record_property("gate_weight_gradient", upstream.abs().sum().item())
    for name, value in agreement.items():
        record_property(f"input_gradient_{name}", value)
    if norm_topk_prob:
        # Known dense-FP32 fact (main ticket 06): the normalize chain runs in another FP32
        # order, so only rounding-level differences are accepted here.
        assert agreement["max_abs"] < 1e-6, agreement
        assert agreement["rel_l2"] < 1e-6, agreement
    else:
        assert agreement["equal"], agreement


def test_oracle_replay_is_bit_equal_in_bfloat16(tmp_path: Any, record_property: Any) -> None:
    """Moonlight's compute dtype: the same VJP in BF16 is bit-equal with normalization on."""
    import torch

    resident = _resident(tmp_path, norm_topk_prob=True, scaling=2.5).to(torch.bfloat16)
    config = resident.get_base_model().config
    owner = resident.get_base_model().model
    layer_idx = config.first_k_dense_replace
    gate = owner.layers[layer_idx].mlp.gate
    hidden = torch.randn(5, config.hidden_size).to(torch.bfloat16)

    inputs = hidden.clone().requires_grad_(True)
    _logits, weights, _ids = type(gate).forward(gate, inputs)
    upstream = torch.randn_like(weights)
    (native_gradient,) = torch.autograd.grad(weights, inputs, upstream)

    with DeepseekRoutingOracle(resident, pin=False) as oracle:
        oracle.begin_step([(1, 5, config.hidden_size)])
        with torch.no_grad():
            with oracle.visit(layer_idx, 0, "forward"):
                forward = gate(hidden)
        replay_input = hidden.clone().requires_grad_(True)
        with torch.enable_grad():
            with oracle.visit(layer_idx, 0, "recompute"):
                replay = gate(replay_input)
        (replay_gradient,) = torch.autograd.grad(replay[1], replay_input, upstream)
        oracle.abort_step()

    assert torch.equal(forward[1], weights)
    assert torch.equal(replay[1], weights)
    agreement = _agreement(native_gradient, replay_gradient)
    for name, value in agreement.items():
        record_property(f"input_gradient_{name}", value)
    assert agreement["equal"], agreement


def _window(
    rig: Rig,
    ids: List[Any],
    calls: List[Any],
    *,
    group_size: int,
    oracle: Optional[Any],
    decoupled: bool = False,
) -> Dict[str, Any]:
    _zero_grads(rig.model)
    extra = (
        {"mbr_decoupled_subwindow": True, "mbr_recompute_subwindow": 1} if decoupled else {}
    )
    losses = rig.stepper.run(
        ids,
        rig.store,
        calls=calls,
        route_preserving_mbr=True,
        mbr_group_size=group_size,
        collect_phase_stats=True,
        routing_oracle=oracle,
        **extra,
    )
    collected = _collect(rig.model)
    missing = [name for name, grad in collected.items() if grad is None]
    assert collected and not missing, f"LoRA parameters without a gradient: {missing[:3]}"
    return {
        "losses": [loss.detach().clone() for loss in losses],
        "grads": {name: grad.clone() for name, grad in collected.items()},
    }


def _make_rig(tmp_path: Any, k: int, dtype: str) -> Rig:
    """A streamed tiny MoE in one compute dtype, without a resident twin."""
    import torch

    from soup_cli.utils.l2l_activations import ActivationStore
    from tests.test_b13_moe_runtime import build_moe_streamed

    config = fixture.moe_config()
    weights = fixture.moe_weights_dir(tmp_path / "weights", config)
    moe = build_moe_streamed(
        weights,
        tmp_path / "shards",
        dtype=dtype,
        two_read_experts=True,
        grouped_expert_routes=True,
        lora_config=fixture.moe_lora_config(),
    )
    model = moe.model
    model.train()
    first = fixture.moe_batch(config, seed=0, rows=1, seq=6)["input_ids"]
    stepper = l2l_step.L2LStep(model, moe.runtime, l2l_step.capture_layer_call(model, first))
    store = ActivationStore(
        n_layers=config.num_hidden_layers,
        k=k,
        chunk_shape=(1, 6, config.hidden_size),
        dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float32,
        device="cpu",
        pin=False,
    )
    return Rig(config, moe, None, stepper, store, "cpu", 1, 6)


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_tiny_moe_mbr_with_the_oracle_equals_without_it_bitwise(
    tmp_path: Any, k: int, dtype: str, record_property: Any,
) -> None:
    """BF16 is Moonlight's compute dtype and must stay bit-equal; dense FP32 is measured.

    The known dense-FP32 fact (main ticket 06) is that the normalize branch's backward
    rounds differently (~1e-7 relative L2). Losses stay bit-equal in both dtypes.
    """
    rig = _make_rig(tmp_path, k, dtype)
    ids = [
        fixture.moe_batch(rig.config, seed=seed, rows=1, seq=6)["input_ids"]
        for seed in range(k)
    ]
    calls = [l2l_step.capture_layer_call(rig.model, item) for item in ids]

    try:
        without = _window(rig, ids, calls, group_size=k, oracle=None)
        with DeepseekRoutingOracle(rig.model, pin=False) as oracle:
            with_oracle = _window(rig, ids, calls, group_size=k, oracle=oracle)
            stats = rig.stepper.routing_oracle_stats
            report = oracle.report()
        assert stats is not None and report["complete"] is True
        assert stats["step_counts"] == {
            "native_forward": len(rig.moe_layers) * k,
            "oracle_peek": 0,
            "oracle_recompute": len(rig.moe_layers) * k,
            "oracle_vjp": len(rig.moe_layers) * k,
        }
        for layer in rig.moe_layers:
            assert rig.stepper.route_preserving_mbr_stats["layers"][layer][
                "gate_weight_rel_l2"
            ] == 0.0

        assert bits_equal(without["losses"][0], with_oracle["losses"][0])
        bad = differing(without, with_oracle)
        max_abs = max(
            float(
                (
                    without["grads"][name].double() - with_oracle["grads"][name].double()
                ).abs().max()
            )
            for name in without["grads"]
        )
        record_property("dtype", dtype)
        record_property("different_tensors", len(bad))
        record_property("first_different", bad[:3])
        record_property("max_abs_gradient_difference", max_abs)
        record_property("router_calls_skipped", stats["step_counts"]["oracle_recompute"])
        if dtype == "bfloat16":
            assert not bad, f"{len(bad)} tensors differ, first {bad[:3]}"
        else:
            assert max_abs < 1e-6, f"dense FP32 differs by more than rounding: {max_abs}"
    finally:
        rig.close()


@pytest.mark.parametrize("k", [1, 2, 4])
def test_decoupled_mbr_with_the_oracle_matches_without_it_bitwise(
    tmp_path: Any, k: int, record_property: Any,
) -> None:
    """The decoupled backward uses the Oracle's peek in R1 and recompute in R2."""
    rig = _make_rig(tmp_path, k, "bfloat16")
    ids = [
        fixture.moe_batch(rig.config, seed=seed, rows=1, seq=6)["input_ids"]
        for seed in range(k)
    ]
    calls = [l2l_step.capture_layer_call(rig.model, item) for item in ids]
    try:
        without = _window(rig, ids, calls, group_size=k, oracle=None, decoupled=True)
        with DeepseekRoutingOracle(rig.model, pin=False) as oracle:
            with_oracle = _window(
                rig, ids, calls, group_size=k, oracle=oracle, decoupled=True
            )
            stats = rig.stepper.routing_oracle_stats
        assert stats is not None
        assert stats["step_counts"] == {
            "native_forward": len(rig.moe_layers) * k,
            "oracle_peek": len(rig.moe_layers) * k,
            "oracle_recompute": len(rig.moe_layers) * k,
            "oracle_vjp": len(rig.moe_layers) * k,
        }
        bad = differing(without, with_oracle)
        record_property("different_tensors", len(bad))
        record_property("first_different", bad[:3])
        assert not bad, f"{len(bad)} tensors differ, first {bad[:3]}"
    finally:
        rig.close()


def test_trainer_state_carries_the_oracle_switch() -> None:
    import torch

    from soup_cli.trainer.l2l_trainer import L2LState

    state = L2LState(
        runtime=None,
        dtype=torch.float32,
        effective_k=2,
        route_preserving_mbr=True,
        routing_oracle=True,
    )
    fields = state.log_fields()
    assert fields["l2l_route_preserving_mbr"] is True
    assert fields["l2l_routing_oracle"] is True
    assert L2LState(runtime=None, dtype=torch.float32).log_fields().get(
        "l2l_routing_oracle"
    ) is None


def test_oracle_requires_route_preserving_mbr_and_a_frozen_router(
    make_rig: Any,
) -> None:
    rig = make_rig(k=1, rows=1, seq=4)
    ids = [fixture.moe_batch(rig.config, seed=0, rows=1, seq=4)["input_ids"]]
    with DeepseekRoutingOracle(rig.model, pin=False) as oracle:
        with pytest.raises(ValueError, match="route_preserving_mbr"):
            rig.stepper.run(ids, rig.store, routing_oracle=oracle)

    gate = rig.model.get_base_model().model.layers[rig.config.first_k_dense_replace].mlp.gate
    gate.weight.requires_grad_(True)
    try:
        with pytest.raises(RuntimeError, match="frozen"):
            DeepseekRoutingOracle(rig.model, pin=False)
    finally:
        gate.weight.requires_grad_(False)
