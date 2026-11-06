"""Opt-in boundary compression through the production SFT window consumer."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string
from soup_cli.trainer.l2l_trainer import L2LState, _store_for
from soup_cli.utils.host_memory_budget import HostMemoryBudget
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR


def _config(**changes):
    fields = {
        "l2l": True,
        "stream_layers": True,
        "quantization": "none",
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "lora": {"r": 4, "alpha": 8, "dropout": 0},
    }
    fields.update(changes.pop("training", {}))
    values = {
        "base": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "task": "sft",
        "data": {"train": "data.jsonl"},
        "training": fields,
    }
    values.update(changes)
    return load_config_from_string(yaml.safe_dump(values))


def _ids(*shapes):
    import torch

    return [torch.zeros(shape, dtype=torch.long) for shape in shapes]


def _stepper(hidden=32, layers=3):
    return SimpleNamespace(
        causal=SimpleNamespace(config=SimpleNamespace(hidden_size=hidden)),
        layers=[None] * layers,
    )


def _state(**options):
    import torch

    values = dict(
        runtime=SimpleNamespace(tier="ram"),
        dtype=torch.float32,
        activation_codec="fp8_e4m3",
        host_budget=HostMemoryBudget(10_000_000),
    )
    values.update(options)
    return L2LState(**values)


def _plan(budget, **changes):
    from soup_cli.utils.moe_l2l_budget import plan_moe_l2l

    options = dict(
        n_layers=3,
        n_moe_layers=2,
        max_shape=(2, 16, 32),
        experts_per_token=2,
        requested_k=4,
        dtype="float32",
        pin=False,
        codec="fp8_e4m3",
        device_type="cpu",
    )
    options.update(changes)
    return plan_moe_l2l(budget, **options)


@pytest.mark.parametrize(
    "changes",
    [
        {"training": {"l2l": False}},
        {"training": {"stream_layers": False}},
        {"task": "dpo"},
        {"task": "orpo"},
        {"backend": "mlx"},
        {"modality": "vision"},
        {"training": {"multipack": True}},
        {"data": {"train": "data.jsonl", "format": "raft"}},
    ],
)
def test_codec_refuses_settings_outside_its_sft_consumer(changes):
    changes = dict(changes)
    training = dict(changes.pop("training", {}), l2l_activation_codec="fp8_e4m3")
    with pytest.raises(ValueError, match="l2l_activation_codec"):
        _config(training=training, **changes)


def test_codec_rejects_unknown_spelling():
    with pytest.raises(ValueError, match="l2l_activation_codec"):
        _config(training={"l2l_activation_codec": "fp8"})


def test_fp8_store_preserves_hidden_rows_and_grows_each_position():
    import torch

    from soup_cli.utils.fp8_activations import FP8ActivationStore

    state = _state()
    budget = state.host_budget
    try:
        first = _store_for(state, _stepper(), _ids((1, 3), (2, 5)), "cpu")
        assert isinstance(first, FP8ActivationStore)
        assert first.mb_shapes == [(3, 32), (10, 32)]
        value = torch.arange(3 * 32, dtype=torch.float32).reshape(1, 3, 32) / 10
        first.put(0, 0, value)
        restored = first.get(0, 0)
        assert restored.shape == value.shape
        assert restored.dtype == value.dtype
        assert torch.isfinite(restored).all()
        assert (restored - value).abs().max().item() < 0.5
        first.end_step()
        value = torch.arange(32, dtype=torch.float32).reshape(1, 1, 32)
        value = value * torch.tensor([0.01, 1.0, 1000.0]).reshape(1, 3, 1)
        first.put(1, 0, value)
        restored = first.get(1, 0)
        relative_error = (restored - value).abs().amax(-1) / value.abs().amax(-1)
        assert bool((relative_error < 0.06).all())
        first.end_step()
        assert _store_for(state, _stepper(), _ids((1, 2)), "cpu") is first
        second = _store_for(state, _stepper(), _ids((2, 4), (1, 3), (1, 7)), "cpu")
        assert second is not first
        assert second.mb_shapes == [(8, 32), (10, 32), (7, 32)]
        assert second.k == 3
        used = budget.used_bytes
        first.close()
        assert budget.used_bytes == used
        assert _store_for(state, _stepper(), _ids((1, 2)), "cpu") is second
        used = budget.used_bytes
        with pytest.raises(ValueError, match="hidden"):
            _store_for(state, _stepper(hidden=16), _ids((1, 2)), "cpu")
        assert state.store is second
        assert budget.used_bytes == used
    finally:
        state.close()
        state.close()
    assert budget.used_bytes == 0


@pytest.mark.gpu
def test_bare_cuda_device_reuses_store_after_put_resolves_device_index():
    import torch

    state = _state()
    ids = torch.zeros((1, 3), dtype=torch.long, device="cuda")
    try:
        store = _store_for(state, _stepper(), [ids], "cuda")
        value = torch.arange(96, dtype=torch.float32, device="cuda").reshape(1, 3, 32)
        store.put(0, 0, value)
        restored = store.get(0, 0)
        assert restored.device == value.device
        assert torch.isfinite(restored).all()
        assert (restored - value).abs().max().item() < 5
        store.end_step()
        assert _store_for(state, _stepper(), [ids[:, :2]], "cuda") is store
        assert _store_for(state, _stepper(), [ids], ids.device) is store
    finally:
        state.close()
    assert state.host_budget.used_bytes == 0


@pytest.mark.parametrize("available", [None, 0, 1])
def test_unknown_or_too_small_ram_budget_never_falls_back(monkeypatch, available):
    from soup_cli.utils import l2l_spill

    monkeypatch.setattr(
        l2l_spill, "ram_budget", lambda: SimpleNamespace(budget_bytes=available)
    )
    state = _state(host_budget=None)
    with pytest.raises((ValueError, MemoryError), match="FP8|fp8|host"):
        _store_for(state, _stepper(), _ids((1, 8)), "cpu")
    assert state.store is None
    state.close()


def test_fp8_uses_c3_activation_partition_without_touching_weights():
    from soup_cli.utils.c3_l2l_budget import plan_c3_l2l
    from soup_cli.utils.partial_ram_source import PartialRamBudget

    plan = plan_c3_l2l(
        PartialRamBudget(10_000_000, 100, 1000),
        n_layers=3,
        batch_size=2,
        max_length=16,
        hidden_size=32,
        microbatches=4,
        dtype="float32",
        pin=False,
        shape_round=1,
        codec="fp8_e4m3",
        device_type="cpu",
    )
    state = _state(host_budget=None, c3_plan=plan)
    try:
        _store_for(state, _stepper(), _ids((1, 5), (2, 8)), "cpu")
        assert 0 < plan.activations.used_bytes <= plan.activations.limit_bytes
        assert plan.weights.cache_bytes == 1000
    finally:
        state.close()
    assert plan.activations.used_bytes == 0


def test_resolved_plan_reserves_routing_across_growth_and_rejects_oversize():
    budget = HostMemoryBudget(10_000_000, reserved_bytes=123)
    plan = _plan(budget)
    state = _state(memory_plan=plan, host_budget=budget)
    routing = plan.routing_record_bytes + plan.routing_comparison_bytes
    try:
        first = _store_for(state, _stepper(), _ids((1, 3), (1, 5)), "cpu")
        assert budget.used_bytes > 123 + routing
        second = _store_for(state, _stepper(), _ids((2, 8), (1, 4)), "cpu")
        assert second is not first
        used = budget.used_bytes
        for window in (
            _ids((3, 8)),
            _ids((1, 17)),
            _ids(*([(1, 3)] * (plan.effective_k + 1))),
        ):
            with pytest.raises(ValueError, match="plan"):
                _store_for(state, _stepper(), window, "cpu")
            assert state.store is second
            assert budget.used_bytes == used
        assert _store_for(state, _stepper(), _ids((1, 2)), "cpu") is second
        first.close()
        assert budget.used_bytes == used
    finally:
        state.close()
    assert budget.used_bytes == 123


def test_plan_allocation_bound_is_checked_before_replacing_store():
    budget = HostMemoryBudget(10_000_000)
    plan = _plan(budget)
    state = _state(memory_plan=plan, host_budget=budget)
    try:
        store = _store_for(state, _stepper(), _ids((1, 3)), "cpu")
        used = budget.used_bytes
        state.memory_plan = replace(plan, activation_bytes=1)
        with pytest.raises(ValueError, match="plan"):
            _store_for(state, _stepper(), _ids((1, 8)), "cpu")
        assert state.store is store
        assert budget.used_bytes == used
    finally:
        state.close()
    assert budget.used_bytes == 0


def test_plan_requires_matching_codec_and_explicit_allocator():
    budget = HostMemoryBudget(10_000_000)
    plan = _plan(budget)
    with pytest.raises(ValueError, match="host_budget"):
        _state(memory_plan=plan, host_budget=None)
    with pytest.raises(ValueError, match="codec"):
        _state(memory_plan=replace(plan, codec="none"), host_budget=budget)
    assert budget.used_bytes == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"dtype": "bfloat16"},
        {"pin": True},
        {"n_layers": 4},
        {"max_shape": (2, 16, 31)},
        {"device_type": "cuda"},
    ],
)
def test_incompatible_plan_metadata_refuses_before_allocation(changes):
    budget = HostMemoryBudget(10_000_000)
    plan = replace(_plan(budget), **changes)
    state = _state(memory_plan=plan, host_budget=budget)
    with pytest.raises(ValueError, match="plan"):
        _store_for(state, _stepper(), _ids((1, 3)), "cpu")
    assert state.store is None
    assert budget.used_bytes == 0
    state.close()


def test_resolved_plan_constructor_failure_releases_routing_lease():
    budget = HostMemoryBudget(10_000_000)
    plan = _plan(budget)
    blocker = budget.reserve("other.consumer", budget.available_bytes - 1)
    state = _state(memory_plan=plan, host_budget=budget)
    # Routing alone cannot fit. No storage or partial lease may survive the refusal.
    with pytest.raises(MemoryError):
        _store_for(state, _stepper(), _ids((1, 3)), "cpu")
    assert state.store is None
    assert budget.available_bytes == 1
    blocker.close()
    # Permit routing, but leave too little room for the activation constructor.
    routing = plan.routing_record_bytes + plan.routing_comparison_bytes
    blocker = budget.reserve("other.consumer", budget.available_bytes - routing - 1)
    with pytest.raises(MemoryError):
        _store_for(state, _stepper(), _ids((1, 3)), "cpu")
    assert state.store is None
    assert budget.available_bytes == routing + 1
    state.close()
    blocker.close()
    assert budget.used_bytes == 0


@pytest.mark.parametrize("shapes", [[], [(1, 0)], [(2,)], [(1, 2, 3)]])
def test_empty_or_malformed_windows_refuse_without_storage(shapes):
    state = _state()
    with pytest.raises(ValueError, match="micro-batches"):
        _store_for(state, _stepper(), _ids(*shapes), "cpu")
    assert state.store is None
    assert state.host_budget.used_bytes == 0
    state.close()


@pytest.mark.parametrize("dtype", ["float16", "int32"])
def test_fp8_rejects_incompatible_compute_dtype_before_allocation(dtype):
    import torch

    with pytest.raises(ValueError, match="dtype|float32|bfloat16"):
        _state(dtype=getattr(torch, dtype))


def test_failed_store_drain_retains_routing_reservation_and_blocks_weight_cleanup(monkeypatch):
    from soup_cli.trainer.stream_setup import StreamingSetupMixin

    budget = HostMemoryBudget(10_000_000)
    state = _state(memory_plan=_plan(budget), host_budget=budget)
    store = _store_for(state, _stepper(), _ids((1, 3)), "cpu")
    close = store.close
    used = budget.used_bytes
    weights_closed = []

    def fail():
        raise RuntimeError("drain failed")

    monkeypatch.setattr(store, "close", fail)
    wrapper = SimpleNamespace(
        trainer=SimpleNamespace(_soup_l2l=state),
        _stream_runtime=SimpleNamespace(close=lambda: weights_closed.append(True)),
    )
    with pytest.raises(RuntimeError, match="drain failed"):
        StreamingSetupMixin._close_stream_runtime(wrapper)
    assert not weights_closed
    assert state.store is store
    assert budget.used_bytes == used
    monkeypatch.setattr(store, "close", close)
    StreamingSetupMixin._close_stream_runtime(wrapper)
    assert weights_closed == [True]
    assert budget.used_bytes == 0


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_training_exit_drains_fp8_store_and_routing_lease(tmp_path, failure):
    from soup_cli.trainer.l2l_trainer import make_l2l_trainer_class

    budget = HostMemoryBudget(10_000_000)
    state = _state(memory_plan=_plan(budget), host_budget=budget, output_dir=str(tmp_path))

    class Base:
        def train(self):
            _store_for(state, _stepper(), _ids((1, 3)), "cpu")
            assert budget.used_bytes > 0
            raise failure("training interrupted")

    trainer = make_l2l_trainer_class(Base)()
    trainer._soup_l2l = state
    with pytest.raises(failure, match="training interrupted"):
        trainer.train()
    assert state.store is None
    assert budget.used_bytes == 0
    state.close()


def test_custom_loss_cannot_silently_use_sft_only_codec():
    from soup_cli.trainer.l2l_trainer import WindowLoss

    loss = WindowLoss("micro_batches", lambda *a, **kw: None, lambda *a: None)
    with pytest.raises(ValueError, match="SFT|sft"):
        _state(loss=loss)


def test_public_builder_refuses_unresolved_accumulation_without_rescheduling(tmp_path):
    from soup_cli.trainer.stream_setup import StreamingSetupMixin

    budget = HostMemoryBudget(10_000_000)
    plan = _plan(budget)
    args = SimpleNamespace(
        n_gpu=1, fp16=False, gradient_accumulation_steps=plan.effective_k + 1,
        output_dir=str(tmp_path),
    )
    wrapper = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(model_type="llama")),
        device="cpu", config=_config(training={"l2l_activation_codec": "fp8_e4m3"}),
        _stream_runtime=SimpleNamespace(device="cpu"), _stream_weight_dirs=(),
    )
    with pytest.raises(ValueError, match="effective_k|accumulation"):
        StreamingSetupMixin._build_l2l_trainer(
            wrapper, object, {}, args, memory_plan=plan, host_budget=budget
        )
    assert args.gradient_accumulation_steps == plan.effective_k + 1
    assert budget.used_bytes == 0


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize(
    "k,expected_counts,expected_stores",
    [
        (1, [(1, 2), (1, 7), (1, 3), (1, 6), (1, 4)], [1, 1, 1, 1, 1]),
        (4, [(4, 18), (1, 4)], [4, 4]),
    ],
)
def test_sft_trainer_mixed_lengths_short_window_normalization_and_cleanup(
    tmp_path, k, expected_counts, expected_stores, device
):
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    from peft import LoraConfig, TaskType
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    from soup_cli.trainer.l2l_trainer import attach_l2l_state, make_l2l_trainer_class
    from soup_cli.utils.fp8_activations import FP8ActivationStore
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    torch.manual_seed(19)
    source = tmp_path / "weights"
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, attention_dropout=0,
        tie_word_embeddings=False,
    ))
    model.save_pretrained(source, safe_serialization=True)
    del model
    shards = tmp_path / "shards"
    index = shard_checkpoint(str(source), str(shards), dtype="float32", arch="llama")
    model, runtime = build_streamed_model(
        model_id=str(source), shard_dir=str(shards), index=index,
        lora_config=LoraConfig(
            r=4, lora_alpha=8, lora_dropout=0, target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
        ), device=device, dtype="float32", pin=device == "cuda",
    )
    seen, counts, expected_losses, holders = [], [], [], {}

    class Record(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            store = holders["state"].store
            assert isinstance(store, FP8ActivationStore)
            assert all(shape[-1] == 32 for shape in store.mb_shapes)
            gradients = [
                param.grad for name, param in model.named_parameters()
                if "lora_" in name and param.grad is not None
            ]
            assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
            assert any(torch.count_nonzero(grad) for grad in gradients)
            seen.append(store.k)
            offset = sum(count for count, _items in counts[:len(seen) - 1])
            count, n_items = counts[len(seen) - 1]
            with torch.no_grad():
                expected_losses.append(sum(
                    model(
                        **{name: value.unsqueeze(0).to(device) for name, value in row.items()},
                        num_items_in_batch=n_items,
                    ).loss.item()
                    for row in rows[offset:offset + count]
                ))

    class OrderedTrainer(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            return torch.utils.data.SequentialSampler(
                self.train_dataset if train_dataset is None else train_dataset
            )

        def get_batch_samples(self, epoch_iterator, num_batches, device):
            samples, n_items = super().get_batch_samples(epoch_iterator, num_batches, device)
            counts.append((len(samples), int(n_items)))
            return samples, n_items

    rows = []
    for length in (3, 8, 4, 7, 5):
        ids = torch.arange(length) % 64
        labels = ids.clone()
        labels[0] = -100
        rows.append(dict(input_ids=ids, labels=labels, attention_mask=torch.ones_like(ids)))
    budget = HostMemoryBudget(10_000_000)
    state = None
    try:
        trainer = make_l2l_trainer_class(OrderedTrainer)(
            model=model,
            args=TrainingArguments(
                output_dir=str(tmp_path / "train"), per_device_train_batch_size=1,
                gradient_accumulation_steps=k, num_train_epochs=1,
                logging_steps=1, save_strategy="no", report_to=[], use_cpu=device == "cpu",
                disable_tqdm=True, optim="sgd", learning_rate=0.01,
            ), train_dataset=rows, callbacks=[Record()],
        )
        state = attach_l2l_state(
            trainer, runtime, dtype=torch.float32, activation_codec="fp8_e4m3",
            host_budget=budget,
        )
        holders["state"] = state
        trainer.train()
        assert trainer.state.global_step == state.steps == len(expected_counts)
        assert seen == expected_stores
        assert counts == expected_counts
        assert state.store is None
        assert budget.used_bytes == 0
        logged = [row["loss"] for row in trainer.state.log_history if "loss" in row]
        assert logged == pytest.approx(expected_losses, abs=1e-4)
    finally:
        if state is not None:
            state.close()
        runtime.close()


@_REQUIRES_TRAINING_FLOOR
@pytest.mark.parametrize("codec", ["none", "fp8_e4m3"])
@pytest.mark.parametrize("contract", ["top_k", "moe_layers", "missing_host", "missing_c3"])
def test_native_routing_contract_refuses_before_expert_reads(tmp_path, codec, contract):
    import torch

    from benchmarks.harness import b13_moe_fixture
    from soup_cli.trainer.l2l_trainer import run_window
    from soup_cli.utils.c3_l2l_budget import plan_c3_l2l
    from soup_cli.utils.partial_ram_source import PartialRamBudget

    rig = b13_moe_fixture.build_fixture(tmp_path)
    budget = HostMemoryBudget(10_000_000)
    plan = _plan(
        budget, max_shape=(2, 6, rig.config.hidden_size), requested_k=1, codec=codec,
        n_moe_layers=1 if contract == "moe_layers" else 2,
        experts_per_token=1 if contract == "top_k" else 2,
    )
    c3 = None
    if contract == "missing_c3":
        c3 = plan_c3_l2l(
            PartialRamBudget(10_000_000, 0, 32768),
            n_layers=3, batch_size=2, max_length=6, hidden_size=rig.config.hidden_size,
            microbatches=1, dtype="float32", pin=False, shape_round=1,
            device_type="cpu", codec=codec,
        )
        budget = c3.activations
    state = None
    before = b13_moe_fixture.snapshot(rig.streamed)
    batches, labels = b13_moe_fixture.micro_batches(rig.config, 1, first=0, device="cpu")
    reads = rig.runtime.expert_pool.fetches
    try:
        with pytest.raises(ValueError, match="routing|memory_plan"):
            state = L2LState(
                runtime=rig.runtime, dtype=torch.float32, activation_codec=codec,
                memory_plan=None if contract.startswith("missing") else plan,
                host_budget=budget, c3_plan=c3, has_dropout=False,
            )
            run_window(
                state, rig.streamed, batches, labels,
                sum(int((label[:, 1:] != -100).sum()) for label in labels), "cpu",
            )
        assert rig.runtime.expert_pool.fetches == reads
        assert all(
            torch.equal(value, b13_moe_fixture.trainables(rig.streamed)[name])
            for name, value in before.items()
        )
        assert state is None or state.store is None
        assert budget.used_bytes == 0
    finally:
        if state is not None:
            state.close()
        rig.close()


def test_replacing_plan_cannot_reuse_an_undersized_routing_lease():
    budget = HostMemoryBudget(10_000_000)
    state = _state(memory_plan=_plan(budget), host_budget=budget)
    try:
        store = _store_for(state, _stepper(), _ids((1, 3)), "cpu")
        used = budget.used_bytes
        state.memory_plan = _plan(budget, experts_per_token=3)
        with pytest.raises(ValueError, match="routing"):
            _store_for(state, _stepper(), _ids((1, 3)), "cpu")
        assert state.store is store
        assert budget.used_bytes == used
    finally:
        state.close()
    assert budget.used_bytes == 0


@_REQUIRES_TRAINING_FLOOR
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("comparison_error", [False, True])
def test_routing_refusal_traceback_releases_owned_records(
    tmp_path, monkeypatch, device, comparison_error
):
    import gc
    import weakref

    import torch

    from benchmarks.harness import b13_moe_fixture
    from soup_cli.utils.l2l_step import RoutingGuard, model_parts
    from tests.test_b13_moe_runtime import streamed_experts_modules

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    rig = b13_moe_fixture.build_fixture(tmp_path, device=device)
    budget = HostMemoryBudget(10_000_000)
    plan = _plan(budget, device_type=device, pin=device == "cuda")
    state = _state(runtime=rig.runtime, memory_plan=plan, host_budget=budget)
    causal, owner = model_parts(rig.streamed)
    stepper = SimpleNamespace(causal=causal, layers=list(owner.layers))
    _store_for(state, stepper, _ids((2, 6)), device)
    layer, module = next(iter(streamed_experts_modules(rig.streamed).items()))
    route = torch.tensor([[0, 1], [2, 3]], device=device)
    hidden = torch.randn(2, rig.config.hidden_size, device=device)
    weights = torch.full(route.shape, 0.5, device=device)
    comparisons = []
    equal = torch.equal

    def observe_comparison(before, now):
        comparisons.append(weakref.ref(now))
        if comparison_error:
            raise RuntimeError("injected routing comparison failure")
        return equal(before, now)
    try:
        with RoutingGuard(stepper.layers) as guard:
            with guard.record(layer, 0), torch.no_grad():
                module(hidden, route, weights)
            owned = weakref.ref(guard._records[(layer, 0)])
            bad = route.clone()
            bad[0, 0] = 4
            reads = module.reader.fetches
            message = "comparison failure" if comparison_error else "routed differently"
            with monkeypatch.context() as observation:
                observation.setattr(torch, "equal", observe_comparison)
                with pytest.raises(RuntimeError, match=message) as caught:
                    with guard.check(layer, 0):
                        module(hidden, bad, weights)
            assert module.reader.fetches == reads
        state.close()
        gc.collect()
        assert caught.value.__traceback__ is not None
        assert owned() is None
        assert comparisons and all(reference() is None for reference in comparisons)
        assert budget.used_bytes == 0
    finally:
        state.close()
        rig.close()


@_REQUIRES_TRAINING_FLOOR
def test_live_routes_cannot_exceed_the_resolved_comparison_budget(tmp_path):
    import torch

    from benchmarks.harness import b13_moe_fixture
    from soup_cli.trainer.l2l_trainer import run_window
    from tests.test_b13_moe_runtime import streamed_experts_modules

    rig = b13_moe_fixture.build_fixture(tmp_path)
    budget = HostMemoryBudget(10_000_000)
    state = _state(
        runtime=rig.runtime, activation_codec="none", host_budget=budget,
        memory_plan=_plan(budget, max_shape=(2, 6, 32), requested_k=1, codec="none"),
        has_dropout=False,
    )
    module = next(iter(streamed_experts_modules(rig.streamed).values()))

    def widen_routes(_module, args, kwargs):
        # Negative control: runtime indices exceed the unchanged model/config top-k.
        if "top_k_index" in kwargs:
            changed = dict(kwargs)
            for name in ("top_k_index", "top_k_weights"):
                value = changed[name]
                changed[name] = torch.cat((value, value[:, :1]), dim=1)
            return args, changed
        changed = list(args)
        for position in (1, 2):
            value = changed[position]
            changed[position] = torch.cat((value, value[:, :1]), dim=1)
        return tuple(changed), kwargs

    hook = module.register_forward_pre_hook(widen_routes, with_kwargs=True, prepend=True)
    batches, labels = b13_moe_fixture.micro_batches(rig.config, 1, first=0, device="cpu")
    reads = rig.runtime.expert_pool.fetches
    try:
        with pytest.raises(ValueError, match="routing.*(budget|plan)"):
            run_window(
                state, rig.streamed, batches, labels,
                sum(int((label[:, 1:] != -100).sum()) for label in labels), "cpu",
            )
        assert rig.runtime.expert_pool.fetches == reads
    finally:
        hook.remove()
        state.close()
        rig.close()
    assert budget.used_bytes == 0


@_REQUIRES_TRAINING_FLOOR
def test_native_resolved_routing_budget_runs_full_then_short_windows(tmp_path):
    import torch

    from benchmarks.harness import b13_moe_fixture
    from soup_cli.trainer.l2l_trainer import run_window

    rig = b13_moe_fixture.build_fixture(tmp_path)
    budget = HostMemoryBudget(10_000_000)
    plan = _plan(budget, max_shape=(2, 6, 32), requested_k=4, codec="none")
    state = _state(
        runtime=rig.runtime, activation_codec="none", memory_plan=plan,
        host_budget=budget, has_dropout=False,
    )
    optimizer = torch.optim.SGD(b13_moe_fixture.trainables(rig.streamed).values(), lr=0.01)
    store = None
    try:
        for k in (4, 1):
            batches, labels = b13_moe_fixture.micro_batches(rig.config, k, first=0, device="cpu")
            if k == 1:
                batches, labels = [batches[0][:1, :3]], [labels[0][:1, :3]]
            before = b13_moe_fixture.snapshot(rig.streamed)
            optimizer.zero_grad(set_to_none=True)
            losses, _ = run_window(
                state, rig.streamed, batches, labels,
                sum(int((label[:, 1:] != -100).sum()) for label in labels), "cpu",
            )
            assert all(torch.isfinite(loss) for loss in losses)
            parameters = b13_moe_fixture.trainables(rig.streamed)
            assert all(
                param.grad is not None and torch.isfinite(param.grad).all()
                and torch.count_nonzero(param.grad) for param in parameters.values()
            )
            optimizer.step()
            assert any(not torch.equal(before[name], value) for name, value in parameters.items())
            assert budget.used_bytes == plan.host_bytes
            if store is not None:
                assert state.store is store
            store = state.store
    finally:
        state.close()
        rig.close()
    assert budget.used_bytes == 0
