"""Public L2L window accounting, admission and cleanup for C3."""

from types import SimpleNamespace

import pytest

from soup_cli.utils.c3_l2l_budget import plan_c3_l2l
from soup_cli.utils.partial_ram_source import PartialRamBudget


def _plan(**changes):
    options = dict(
        n_layers=3,
        batch_size=1,
        max_length=65,
        hidden_size=33,
        microbatches=2,
        dtype="float32",
        pin=False,
        shape_round=64,
    )
    options.update(changes)
    return plan_c3_l2l(PartialRamBudget(10_000_000, 1234, 45_000), **options)


@pytest.mark.parametrize("pin,expected", [(False, 3 * 2 * 128 * 33 * 4), (True, 3 * 2 * 32768)])
def test_maximum_window_is_reserved_before_cache_including_pinned_rounding(pin, expected):
    plan = _plan(pin=pin)
    assert plan.max_shape == (1, 128, 33)
    assert plan.weights.reserved_bytes == 1234 + expected
    assert plan.activations.limit_bytes == expected
    assert plan.weights.cache_bytes == 45_000
    assert plan.activations.used_bytes == 0


def test_fp8_partition_covers_cpu_transient_and_variable_shape_storage():
    import torch

    from soup_cli.trainer.l2l_trainer import _store_for

    plan = _plan(codec="fp8_e4m3", device_type="cpu")
    state, stepper = _state_and_stepper(plan)
    state.activation_codec = "fp8_e4m3"
    try:
        store = _store_for(state, stepper, _ids((1, 128), (1, 5)), "cpu")
        for mb, length in enumerate((128, 5)):
            value = torch.ones((1, length, 33), dtype=torch.float32)
            store.put(0, mb, value)
            assert torch.equal(store.get(0, mb), value)
        assert plan.activations.used_bytes <= plan.activations.limit_bytes
        assert plan.weights.reserved_bytes == 1234 + plan.activations.limit_bytes
        with pytest.raises(ValueError, match="plan"):
            plan.validate_window(
                n_layers=3, shape=(1, 128, 33), k=2,
                dtype="float32", pin=False, codec="none",
            )
    finally:
        state.close()
    assert plan.activations.used_bytes == 0


def test_combined_fp8_moe_c3_reserves_routing_before_weights_and_keeps_it_during_growth():
    import torch

    from soup_cli.trainer.l2l_trainer import L2LState, _store_for
    from soup_cli.utils.host_memory_budget import HostMemoryBudget
    from soup_cli.utils.moe_l2l_budget import plan_moe_l2l

    memory_plan = plan_moe_l2l(
        HostMemoryBudget(10_000_000), n_layers=3, n_moe_layers=2,
        max_shape=(1, 65, 33), experts_per_token=2, requested_k=2,
        dtype="float32", pin=False, codec="fp8_e4m3", device_type="cpu",
    )
    routing_bytes = memory_plan.routing_record_bytes + memory_plan.routing_comparison_bytes
    missing_routes = _plan(codec="fp8_e4m3", device_type="cpu", shape_round=1)
    with pytest.raises(ValueError, match="routing"):
        L2LState(
            runtime=None, dtype=torch.float32, activation_codec="fp8_e4m3",
            c3_plan=missing_routes, memory_plan=memory_plan, host_budget=missing_routes.activations,
        )
    assert missing_routes.activations.used_bytes == 0
    plan = _plan(
        codec="fp8_e4m3", device_type="cpu", shape_round=1, routing_bytes=routing_bytes,
    )
    assert plan.weights.reserved_bytes == 1234 + memory_plan.host_bytes
    state = L2LState(
        runtime=None, dtype=torch.float32, activation_codec="fp8_e4m3",
        c3_plan=plan, memory_plan=memory_plan, host_budget=plan.activations,
    )
    stepper = SimpleNamespace(
        causal=SimpleNamespace(config=SimpleNamespace(hidden_size=33)), layers=[None] * 3,
    )
    try:
        _store_for(state, stepper, _ids((1, 5)), "cpu")
        state.close_store()
        assert plan.activations.used_bytes == routing_bytes
        store = _store_for(state, stepper, _ids((1, 65), (1, 65)), "cpu")
        value = torch.ones((1, 65, 33), dtype=torch.float32)
        store.put(2, 1, value)
        assert torch.equal(store.get(2, 1), value)
        assert plan.activations.used_bytes == plan.activations.limit_bytes
    finally:
        state.close()
    assert plan.activations.used_bytes == 0


def test_fp8_device_mismatch_keeps_existing_store_and_lease():
    from soup_cli.trainer.l2l_trainer import _store_for

    plan = _plan(codec="fp8_e4m3", device_type="cpu")
    state, stepper = _state_and_stepper(plan)
    state.activation_codec = "fp8_e4m3"
    wrong_device_plan = _plan(codec="fp8_e4m3", device_type="cuda")
    try:
        first = _store_for(state, stepper, _ids((1, 5)), "cpu")
        reserved = plan.activations.used_bytes
        state.c3_plan = wrong_device_plan
        with pytest.raises(ValueError, match="plan"):
            _store_for(state, stepper, _ids((1, 128), (1, 128)), "cpu")
        assert state.store is first
        assert plan.activations.used_bytes == reserved
        assert wrong_device_plan.activations.used_bytes == 0
    finally:
        state.close()
    assert plan.activations.used_bytes == 0


@pytest.mark.parametrize(
    "changes",
    [
        dict(k=3),
        dict(shape=(2, 128, 33)),
        dict(shape=(1, 129, 33)),
        dict(shape=(1, 64, 32)),
        dict(n_layers=4),
        dict(dtype="bfloat16"),
        dict(pin=True),
        dict(k=True),
    ],
)
def test_windows_cannot_borrow_cache_partition(changes):
    values = dict(n_layers=3, shape=(1, 128, 33), k=2, dtype="float32", pin=False)
    values.update(changes)
    with pytest.raises(ValueError):
        _plan().validate_window(**values)


def test_insufficient_fixed_budget_refuses_before_any_weight_allocation():
    with pytest.raises(ValueError, match="reservations"):
        plan_c3_l2l(
            PartialRamBudget(100, 50, 20),
            n_layers=3,
            batch_size=1,
            max_length=64,
            hidden_size=32,
            microbatches=2,
            dtype="float32",
            pin=False,
            shape_round=64,
        )


def _ids(*shapes):
    import torch

    return [torch.zeros(shape, dtype=torch.long) for shape in shapes]


def _state_and_stepper(plan):
    import torch

    from soup_cli.trainer.l2l_trainer import L2LState

    state = L2LState(runtime=None, dtype=torch.float32, c3_plan=plan)
    stepper = SimpleNamespace(
        causal=SimpleNamespace(config=SimpleNamespace(hidden_size=33)), layers=[None] * 3
    )
    return state, stepper


def test_shape_replacement_releases_old_activation_lease_and_oversize_retains_it():
    from soup_cli.trainer.l2l_trainer import _store_for

    plan = _plan()
    state, stepper = _state_and_stepper(plan)
    try:
        first = _store_for(state, stepper, _ids((1, 32), (1, 32)), "cpu")
        assert plan.activations.used_bytes == 3 * 2 * 32 * 33 * 4
        assert _store_for(state, stepper, _ids((1, 16), (1, 32)), "cpu") is first
        second = _store_for(state, stepper, _ids((1, 128), (1, 128)), "cpu")
        assert first is not second
        assert plan.activations.used_bytes == plan.activations.limit_bytes
        for window in (_ids((1, 129), (1, 1)), _ids((1, 1), (1, 1), (1, 1)), _ids((2, 64))):
            with pytest.raises(ValueError):
                _store_for(state, stepper, window, "cpu")
            assert state.store is second
            assert plan.activations.used_bytes == plan.activations.limit_bytes
    finally:
        state.close()
    assert plan.activations.used_bytes == 0


def test_mixed_length_window_fits_reserved_partition():
    import torch

    from soup_cli.trainer.l2l_trainer import _store_for

    plan = _plan()
    state, stepper = _state_and_stepper(plan)
    try:
        store = _store_for(state, stepper, _ids((1, 128), (1, 5)), "cpu")
        assert store.mb_shapes == [(128 * 33,), (5 * 33,)]
        assert plan.activations.used_bytes <= plan.activations.limit_bytes
        short = torch.arange(5 * 33, dtype=torch.float32).view(1, 5, 33)
        store.put(0, 1, short)
        assert torch.equal(store.get(0, 1), short)
    finally:
        state.close()
    assert plan.activations.used_bytes == 0


def test_failed_activation_drain_prevents_weight_cleanup():
    from soup_cli.trainer.stream_setup import StreamingSetupMixin

    order = []

    def fail():
        order.append("activation")
        raise RuntimeError("drain failed")

    wrapper = SimpleNamespace(
        trainer=SimpleNamespace(_soup_l2l=SimpleNamespace(c3_plan=object(), close=fail)),
        _stream_runtime=SimpleNamespace(close=lambda: order.append("weights")),
    )
    with pytest.raises(RuntimeError, match="drain failed"):
        StreamingSetupMixin._close_stream_runtime(wrapper)
    assert order == ["activation"]
    wrapper.trainer._soup_l2l.close = lambda: order.append("activation retry")
    StreamingSetupMixin._close_stream_runtime(wrapper)
    assert order == ["activation", "activation retry", "weights"]


@pytest.mark.parametrize("codec", ["none", "fp8_e4m3"])
def test_public_setup_and_optimizer_update_use_reserved_l2l_store(tmp_path, monkeypatch, codec):
    import torch
    from transformers import TrainerCallback

    from soup_cli.config.schema import SoupConfig
    from tests.test_v07200 import TestStreamingEndToEndSetup

    wrapper, dataset = TestStreamingEndToEndSetup()._wrapper(tmp_path, monkeypatch)
    config = wrapper.config.model_dump()
    config["training"].update(
        l2l=True,
        l2l_activation_codec=codec,
        gradient_accumulation_steps=2,
        stream_source="disk",
        # Isolate disk admission from the test host; this does not measure NVMe performance.
        stream_disk_kind="nvme",
        stream_pin=False,
        stream_partial_ram=dict(limit_bytes=10_000_000, reserved_bytes=100_000, cache_bytes=45_000),
    )
    wrapper.config = SoupConfig.model_validate(config)
    try:
        wrapper.setup(dataset)
        state = wrapper.trainer._soup_l2l
        plan = wrapper._c3_l2l_plan
        assert state.c3_plan is plan
        source = wrapper._stream_runtime.source
        assert source.budget == plan.weights
        before = {
            n: p.detach().clone()
            for n, p in wrapper.model.named_parameters()
            if "lora_" in n and p.requires_grad
        }
        observed_usage = []

        class ObserveStore(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                assert 0 < plan.activations.used_bytes <= plan.activations.limit_bytes
                observed_usage.append(plan.activations.used_bytes)

        wrapper.trainer.add_callback(ObserveStore())
        wrapper.trainer.args.max_steps = 2
        wrapper.trainer.train()
        assert wrapper.trainer.state.global_step == 2
        assert len(observed_usage) == 2
        assert state.store is None
        assert plan.activations.used_bytes == 0
        assert source.accounted_peak_bytes <= plan.weights.limit_bytes
        assert any(
            not torch.equal(before[n], p)
            for n, p in wrapper.model.named_parameters()
            if n in before
        )
    finally:
        wrapper._close_stream_runtime()
    assert plan.activations.used_bytes == 0


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)])
def test_public_hf_loop_matches_disk_only_l2l(tmp_path, device):
    """Same actual Trainer optimizer steps, with optional decoder residency only."""
    import torch
    from peft import LoraConfig, TaskType
    from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments

    from benchmarks.harness import l2l_schedule
    from soup_cli.trainer.l2l_trainer import attach_l2l_state, make_l2l_trainer_class
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model
    from tests.test_l2l_trainer_mode import _adapters, _Rows

    # The gpu marker skips the CUDA case where no device exists (tests/conftest.py).
    cuda = device.startswith("cuda")
    torch.set_num_threads(2)
    torch.manual_seed(7)
    weights = tmp_path / "weights"
    base = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            tie_word_embeddings=False,
            attention_dropout=0,
        )
    )
    base.save_pretrained(weights, safe_serialization=True)
    del base
    shards = tmp_path / "shards"
    index = shard_checkpoint(str(weights), str(shards), dtype="float32", arch="llama")
    results = []
    for cached in (False, True):
        plan = (
            plan_c3_l2l(
                PartialRamBudget(10_000_000, 500, 65_536 if cuda else 45_000),
                n_layers=4,
                batch_size=1,
                max_length=64,
                hidden_size=32,
                microbatches=2,
                dtype="float32",
                pin=cuda,
                shape_round=64,
            )
            if cached
            else None
        )
        model, runtime = build_streamed_model(
            model_id=str(weights),
            shard_dir=str(shards),
            index=index,
            lora_config=LoraConfig(
                r=4,
                lora_alpha=8,
                lora_dropout=0,
                bias="none",
                target_modules=["q_proj", "v_proj"],
                task_type=TaskType.CAUSAL_LM,
            ),
            device=device,
            dtype="float32",
            pin=cuda,
            seed=3,
            tier="disk",
            read_ahead=1,
            partial_ram_budget=plan.weights if plan else None,
        )
        trainer = None
        try:
            l2l_schedule.make_non_vacuous(model)
            before = _adapters(model)
            args = TrainingArguments(
                output_dir=str(tmp_path / str(cached)),
                per_device_train_batch_size=1,
                gradient_accumulation_steps=2,
                learning_rate=0.01,
                optim="sgd",
                lr_scheduler_type="constant",
                num_train_epochs=1,
                save_strategy="no",
                report_to=[],
                use_cpu=not cuda,
                seed=1,
                data_seed=1,
                max_grad_norm=0,
                disable_tqdm=True,
                logging_steps=1,
            )
            trainer = make_l2l_trainer_class(Trainer)(
                model=model, args=args, train_dataset=_Rows(6, lengths=[32, 32, 64, 64, 32, 32])
            )
            attach_l2l_state(trainer, runtime, dtype=torch.float32, c3_plan=plan)
            trainer.train()
            assert trainer.state.global_step == 3
            after = _adapters(model)
            assert all(not torch.equal(before[n], after[n]) for n in before)
            results.append(
                (
                    {n: p.cpu() for n, p in after.items()},
                    [row["loss"] for row in trainer.state.log_history if "loss" in row],
                )
            )
            if plan:
                assert 0 < len(runtime.source.resident_indices) < 4
                assert runtime.source.accounted_peak_bytes <= plan.weights.limit_bytes
                assert plan.activations.used_bytes <= plan.activations.limit_bytes
        finally:
            if trainer:
                trainer._soup_l2l.close()
            runtime.close()
        if plan:
            assert plan.activations.used_bytes == 0
    assert results[0][1] == results[1][1]
    for name, expected in results[0][0].items():
        torch.testing.assert_close(results[1][0][name], expected, rtol=0, atol=0)
