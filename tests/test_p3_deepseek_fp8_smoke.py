"""Driver safety checks with native DeepSeek routers and real storage/SGD operations.

These CPU edge tests do not replace the parent-owned CUDA NF4 CLI smoke. No test
runs a resident training arm or substitutes an echo for the Oracle or optimizer.
"""

from __future__ import annotations

import json
import weakref
from pathlib import Path
from typing import Any

import pytest

from benchmarks.harness import p3_deepseek_fp8_smoke as smoke
from soup_cli.utils.host_memory_budget import HostMemoryBudget


def _native() -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    from benchmarks.harness.b13_moe_fixture import moe_config

    config = moe_config(
        hidden_size=128, intermediate_size=128, moe_intermediate_size=64,
        num_hidden_layers=2,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
        model.requires_grad_(False)
        gate = model.model.layers[1].mlp.gate
        with torch.no_grad():
            gate.weight.normal_(0.0, 0.2)
            gate.e_score_correction_bias.copy_(torch.linspace(-0.1, 0.1, 8))
    return model


def _completed_window() -> tuple[Any, Any, Any, Any]:
    import torch

    from benchmarks.harness.p3_deepseek_oracle import DeepseekOracleProbe

    model = _native()
    budget = HostMemoryBudget(16 * 1024**2)
    probe = DeepseekOracleProbe(model, [(1, 4, 128)], budget)
    parameter = torch.nn.Parameter(torch.linspace(-0.3, 0.4, 512).reshape(1, 4, 128))
    gate = model.model.layers[1].mlp.gate
    with torch.no_grad(), probe.visit(1, 0, "forward"):
        gate(parameter)
    with probe.visit(1, 0, "recompute"):
        _, weights, _ = gate(parameter)
        loss = weights.square().sum()
    loss.backward()
    return parameter, loss, probe, budget


def test_incomplete_native_oracle_refuses_optimizer_update() -> None:
    import torch

    from benchmarks.harness.p3_deepseek_oracle import DeepseekOracleProbe

    model = _native()
    budget = HostMemoryBudget(16 * 1024**2)
    probe = DeepseekOracleProbe(model, [(1, 4, 128)], budget)
    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.grad = torch.ones_like(parameter)
    optimizer = torch.optim.SGD([parameter], lr=0.05, momentum=0.9)
    record: dict[str, Any] = {}
    try:
        with pytest.raises((RuntimeError, ValueError)):
            smoke._apply_update(
                {"attention.lora_A": parameter}, optimizer, probe, [torch.tensor(1.0)], record,
            )
        assert torch.equal(parameter.detach(), torch.ones(2))
        assert optimizer.state == {}
        assert record["optimizer_applied"] is False
    finally:
        probe.close()
    assert budget.used_bytes == 0


def test_completed_native_vjp_permits_one_changed_finite_sgd_update() -> None:
    import torch

    parameter, loss, probe, budget = _completed_window()
    before = parameter.detach().clone()
    optimizer = torch.optim.SGD([parameter], lr=0.05)
    record: dict[str, Any] = {}
    try:
        smoke._apply_update({"attention.lora_A": parameter}, optimizer, probe, [loss], record)
        assert not torch.equal(before, parameter)
        assert bool(torch.isfinite(parameter).all())
        assert record["optimizer_applied"] is True
        assert record["changed_adapter_tensors"] == 1
        assert record["before_sha256"] != record["after_sha256"]
        assert record["gradients"]["nonzero_tensors"] == 1
        assert probe.report()["counts"]["oracle_vjp"] == 1
        json.dumps(record, allow_nan=False)
    finally:
        probe.close()
    assert budget.used_bytes == 0


@pytest.mark.parametrize("bad", ["loss", "nonfinite_gradient", "zero_gradient"])
def test_bad_training_values_never_reach_optimizer(bad: str) -> None:
    import torch

    parameter, loss, probe, budget = _completed_window()
    optimizer = torch.optim.SGD([parameter], lr=0.05, momentum=0.9)
    before = parameter.detach().clone()
    if bad == "loss":
        loss = torch.tensor(float("nan"))
    elif bad == "nonfinite_gradient":
        parameter.grad.fill_(float("inf"))
    else:
        parameter.grad.zero_()
    record: dict[str, Any] = {}
    try:
        with pytest.raises(ValueError):
            smoke._apply_update({"attention.lora_A": parameter}, optimizer, probe, [loss], record)
        assert torch.equal(parameter.detach(), before)
        assert optimizer.state == {}
        assert record["optimizer_applied"] is False
    finally:
        probe.close()
    assert budget.used_bytes == 0


def test_retained_failure_closes_store_oracle_and_restores_native_method() -> None:
    import torch

    model = _native()
    gate = model.model.layers[1].mlp.gate
    original = gate.forward.__func__
    budget = HostMemoryBudget(16 * 1024**2)
    record: dict[str, Any] = {}
    source = torch.arange(512).reshape(1, 4, 128).to(torch.bfloat16) / 512
    payload_refs = []
    with pytest.raises(RuntimeError, match="retained failure") as retained:
        with smoke._window_resources(model, [(1, 4, 128)], budget, "cpu", record) as resources:
            store, probe = resources
            store.put(0, 0, source)
            restored = store.get(0, 0)
            assert restored.shape == source.shape
            payload_refs.extend(weakref.ref(value) for value in store._payload.values())
            with torch.no_grad(), probe.visit(1, 0, "forward"):
                gate(source)
            raise RuntimeError("retained failure")
    assert retained.value is not None
    assert payload_refs and all(reference() is None for reference in payload_refs)
    assert gate.forward.__func__ is original
    assert budget.used_bytes == 0
    assert record["store_after_close"]["cleanup"]["closed"] is True
    assert record["store_after_close"]["owned_host_storage_bytes"] == 0
    assert record["oracle_after_close"]["closed"] is True
    assert record["oracle_after_close"]["records_remaining"] == 0
    assert record["oracle_after_close"]["pending_replays"] == 0
    assert record["host_budget_after_close_bytes"] == 0
    json.dumps(record, allow_nan=False)


def test_failed_store_admission_releases_budget() -> None:
    model = _native()
    budget = HostMemoryBudget(1)
    with pytest.raises(MemoryError):
        with smoke._window_resources(model, [(1, 4, 128)], budget, "cpu", {}):
            pytest.fail("unadmitted store entered")
    assert budget.used_bytes == 0


def test_reused_layer_call_is_rejected_before_running_a_window() -> None:
    import torch

    from soup_cli.utils.l2l_step import capture_layer_call

    model = _native()
    call = capture_layer_call(model, torch.zeros((1, 4), dtype=torch.long), draws=False)
    with pytest.raises(ValueError, match="distinct"):
        smoke._stepper(model, None, [call, call], None)


@pytest.mark.parametrize("rate", [0.0, -0.1, float("inf"), float("nan")])
def test_invalid_learning_rate_cannot_create_output(tmp_path: Path, rate: float) -> None:
    output = tmp_path / "new-run"
    with pytest.raises(ValueError):
        smoke.run_smoke(output, learning_rate=rate)
    assert not output.exists()


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        smoke.run_smoke(output)
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert not (output / "result.json").exists()


def test_process_rng_and_backend_flags_restore_when_training_raises() -> None:
    import random

    import torch

    before_random = random.getstate()
    before_torch = torch.get_rng_state().clone()
    before_flags = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
    )
    with pytest.raises(RuntimeError, match="training failure"):
        with smoke._preserve_process_state("cpu") as state:
            random.random()
            torch.rand(7)
            raise RuntimeError("training failure")
    assert random.getstate() == before_random
    assert torch.equal(torch.get_rng_state(), before_torch)
    assert (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
    ) == before_flags
    assert state["restored"] is True


def test_json_writer_rejects_tensor_or_nonfinite_evidence(tmp_path: Path) -> None:
    import torch

    output = tmp_path / "evidence.json"
    with pytest.raises(TypeError):
        smoke._write_json(output, {"hidden": torch.ones(1)})
    with pytest.raises(ValueError):
        smoke._write_json(output, {"loss": float("nan")})
    assert not output.exists()


def test_output_child_cannot_escape_exclusive_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        smoke._child(tmp_path, "../outside.json")


def test_distinct_calls_drive_two_microbatches_through_production_step(tmp_path: Path) -> None:
    """CPU dense shards isolate call/probe wiring; the CUDA CLI must prove NF4."""
    import torch

    from benchmarks.harness import b13_moe_fixture as fixture
    from soup_cli.utils.l2l_step import capture_layer_call
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream import stream_arch_of
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    config = fixture.moe_config(
        hidden_size=128, intermediate_size=128, moe_intermediate_size=64,
        num_hidden_layers=2,
    )
    weights = fixture.moe_weights_dir(tmp_path, config, seed=23)
    shards = tmp_path / "shards"
    index = shard_checkpoint(
        str(weights), str(shards), dtype="bfloat16", arch=stream_arch_of(config),
        expert_layout="per_expert",
    )
    model, runtime = build_streamed_model(
        model_id=str(weights), shard_dir=str(shards), index=index,
        lora_config=fixture.moe_lora_config(), device="cpu", dtype="bfloat16",
        quant="none", buffers=2, pin=False, seed=23,
    )
    record: dict[str, Any] = {}
    budget = HostMemoryBudget(16 * 1024**2)
    try:
        parameters = fixture.trainables(model)
        for parameter in parameters.values():
            parameter.data = parameter.data.float()
        fixture.randomize_adapters(model, seed=23)
        model.train()
        ids = [torch.tensor([[0, 3, 7, 9]]), torch.tensor([[11, 13, 17, 19]])]
        calls = [capture_layer_call(model, value, draws=False) for value in ids]
        optimizer = torch.optim.SGD(parameters.values(), lr=0.05)
        with smoke._window_resources(model, [(1, 4, 128)] * 2, budget, "cpu", record) as pair:
            store, probe = pair
            step = smoke._stepper(model, runtime, calls, probe)
            losses = step.run(ids, store, calls=calls)
            smoke._apply_update(parameters, optimizer, probe, losses, record)
            assert probe.report()["expected_pairs"] == 2
            assert probe.report()["counts"]["oracle_vjp"] == 2
            assert step.counters.expert_fetches > 0
            assert step.counters.expert_bytes_read > 0
        assert record["optimizer_applied"] is True
        assert record["changed_adapter_tensors"] > 0
        assert record["store_before_close"]["put_calls"] == 4
        assert record["store_before_close"]["get_calls"] == 4
        assert budget.used_bytes == 0
        json.dumps(record, allow_nan=False)
    finally:
        runtime.close()
