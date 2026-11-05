"""Ticket A6-05: KTO on L2L (CPU).

TRL 0.29.1 ``KTOTrainer`` takes the reference from the policy model with its adapter switched
off (a ``PeftModel`` given no ``ref_model``; there is no ``ref`` adapter), and per micro-batch
runs four forwards: the policy and the reference over the KL completions (``KL_completion_*``,
the answers shifted by one inside a block of ``per_device_train_batch_size`` consecutive rows,
no grad), the policy over the completions (with grad) and the reference over the completions.
Its loss is ``nanmean`` over the rows of one micro-batch, with the KL estimate
``clamp(mean(policy_KL - reference_KL), min=0)`` as a constant.

The reference of these tests is that trainer under gradient accumulation, built by the real
``KTOTrainerWrapper.setup()``. The L2L run is the same wrapper with ``l2l: true``. Both start
from the same adapters and the same data order; the LoRA gradient of every optimizer step and
the logged loss must be equal with ``torch.equal``.

Seams, agreed by the ticket:

* ``L2LStep.run(..., reference=[ReferencePass, ...])``: several gradient-free passes through the
  loaded layer, each with its own ids and with the adapters on, off or switched (step level,
  ``test_extra_pass_*``).
* ``kto_window_loss(args=...)``: how a window of TRL collator batches is read.
* The trainer built by ``KTOTrainerWrapper`` with and without ``training.l2l``.
* The start-up refusals, the schema refusals and the short-training example.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_l2l_dpo import _CountingStore, _grads, _ids, _logits_head, _step, _warm_walk
from tests.test_l2l_orpo import _assert_equal_runs, _train
from tests.test_l2l_probe_harness import _load
from tests.test_v07204 import _stream_cfg, _tiny_llama_dir, _write_tiny_tokenizer

ROOT = Path(__file__).resolve().parent.parent


def _rows(completions, labels, prompts=None):
    prompts = prompts or ["hi"] * len(completions)
    return [
        {"prompt": prompt, "completion": completion, "label": label}
        for prompt, completion, label in zip(prompts, completions, labels)
    ]


#: Rows of one length; every micro-batch of 2 holds one desirable and one undesirable row.
UNIFORM_ROWS = _rows([f" good answer {i}" for i in range(10)], [i % 2 == 0 for i in range(10)])
UNEVEN_COMPLETIONS = [
    " good answer the cat sat on mat",
    " bad",
    " good answer",
    " bad answer the soup",
    " good",
    " bad answer the cat sat on",
    " good answer the soup",
    " bad the",
    " good answer the cat sat",
    " bad answer",
]
#: Rows of different lengths, so the KL completions differ in width from the completions.
UNEVEN_ROWS = _rows(
    UNEVEN_COMPLETIONS,
    [i % 2 == 0 for i in range(10)],
    ["hi", "the cat", "hello world yo", "hi", "the cat sat", "yo", "hi", "hello", "soup", "the"],
)
#: Micro-batches of 2: (T,T) (F,F) (T,T) (F,F) (T,F): four of them hold one label only.
SINGLE_LABEL_ROWS = _rows(
    UNEVEN_COMPLETIONS, [True, True, False, False, True, True, False, False, True, False]
)
ALL_DESIRABLE_ROWS = _rows(UNEVEN_COMPLETIONS, [True] * 10)
ALL_UNDESIRABLE_ROWS = _rows(UNEVEN_COMPLETIONS, [False] * 10)
LORA = {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]}


def _wrapper(
    tmp_path, monkeypatch, name, *, l2l, k, rows, batch_size=2, raises=None, **training
):
    """A streamed KTO wrapper through the real ``setup()``, on CPU in float32.

    With ``raises`` (a regex) ``setup()`` must refuse with that message; the streaming runtime
    it had opened is closed, so a refusal leaves nothing running for the next test.
    """
    from soup_cli.trainer.kto import KTOTrainerWrapper

    root = tmp_path / name
    root.mkdir()
    weights, _model, _config = _tiny_llama_dir(root, n_layers=3, tie=False)
    _write_tiny_tokenizer(weights)
    monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(root / "cache"))
    # TrainingArguments would pick MPS or CUDA by itself; the model here is on the CPU.
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    import torch
    from transformers import TrainingArguments

    original_device = TrainingArguments.device.fget

    def cpu_device(self):
        original_device(self)  # still sets up the distributed state
        return torch.device("cpu")

    monkeypatch.setattr(TrainingArguments, "device", property(cpu_device))
    monkeypatch.chdir(root)
    options = {
        "batch_size": batch_size,
        "gradient_accumulation_steps": k,
        "seed": 1,
        "lr": 0.01,
        "optimizer": "sgd",
        "scheduler": "constant",
        "warmup_ratio": 0.0,
        "kto_beta": 0.3,
        "lora": LORA,
    }
    if l2l:
        options["l2l"] = True
    options.update(training)
    cfg = _stream_cfg(weights, root / "out", task="kto", **options)
    wrapper = KTOTrainerWrapper(cfg, device="cpu")
    if raises is not None:
        try:
            with pytest.raises(ValueError, match=raises):
                wrapper.setup({"train": list(rows)})
        finally:
            wrapper._close_stream_runtime()
        return wrapper
    wrapper.setup({"train": list(rows)})
    return wrapper


def _pair(tmp_path, monkeypatch, *, k, rows, **options):
    """The accumulation run and the L2L run from the same start; returns both results."""
    harness = _load("l2l_schedule")
    ga = _wrapper(tmp_path, monkeypatch, "ga", l2l=False, k=k, rows=rows, **options)
    l2l = _wrapper(tmp_path, monkeypatch, "l2l", l2l=True, k=k, rows=rows, **options)
    try:
        harness.make_non_vacuous(ga.model)
        start = harness.snapshot_adapters(ga.model)
        harness.restore_adapters(l2l.model, start)
        ga_result = _train(ga)
        l2l_result = _train(l2l)
        return ga, l2l, ga_result, l2l_result
    finally:
        for wrapper in (ga, l2l):
            wrapper._close_stream_runtime()


# 10 rows at batch 2 are 5 micro-batches: k=2 gives windows of 2, 2, 1; k=4 gives 4, 1
@pytest.mark.parametrize("k,windows", [(1, 5), (2, 3), (4, 2)])
def test_kto_l2l_equals_accumulation_for_rows_of_one_length(tmp_path, monkeypatch, k, windows):
    _ga, l2l, ga_result, l2l_result = _pair(tmp_path, monkeypatch, k=k, rows=UNIFORM_ROWS)
    _assert_equal_runs(ga_result, l2l_result, windows)
    assert l2l.trainer._soup_l2l.steps == windows


@pytest.mark.parametrize("k,windows", [(1, 5), (2, 3), (4, 2)])
def test_kto_l2l_equals_accumulation_for_rows_of_different_lengths(
    tmp_path, monkeypatch, k, windows
):
    _ga, l2l, ga_result, l2l_result = _pair(tmp_path, monkeypatch, k=k, rows=UNEVEN_ROWS)
    _assert_equal_runs(ga_result, l2l_result, windows)
    # the micro-batches really differ in shape, so the calls are per shape
    assert len(l2l.trainer._soup_l2l.calls_by_shape) > 1


def test_the_kl_estimate_is_positive_in_the_runs_that_are_compared(tmp_path, monkeypatch):
    """KTO's KL term is clamped at zero; a run where it is always zero proves little."""
    _ga, _l2l, (_grads_ga, ga_logs), _l2l_result = _pair(
        tmp_path, monkeypatch, k=2, rows=UNEVEN_ROWS
    )
    assert max(row["kl"] for row in ga_logs) > 0


@pytest.mark.parametrize("k,windows", [(1, 5), (2, 3), (4, 2)])
def test_a_micro_batch_with_one_label_gives_no_nan_and_equals_accumulation(
    tmp_path, monkeypatch, k, windows
):
    _ga, _l2l, ga_result, l2l_result = _pair(tmp_path, monkeypatch, k=k, rows=SINGLE_LABEL_ROWS)
    _assert_equal_runs(ga_result, l2l_result, windows)


def test_a_last_block_shorter_than_the_batch_equals_accumulation(tmp_path, monkeypatch):
    """9 rows at batch 2: the last micro-batch is one row, its own KL completion."""
    _ga, _l2l, ga_result, l2l_result = _pair(tmp_path, monkeypatch, k=2, rows=UNEVEN_ROWS[:9])
    _assert_equal_runs(ga_result, l2l_result, 3)


@pytest.mark.parametrize("rows", [ALL_DESIRABLE_ROWS, ALL_UNDESIRABLE_ROWS])
def test_a_window_with_one_label_only_equals_accumulation(tmp_path, monkeypatch, rows):
    _ga, _l2l, ga_result, l2l_result = _pair(tmp_path, monkeypatch, k=2, rows=rows)
    _assert_equal_runs(ga_result, l2l_result, 3)


def test_kto_l2l_equals_accumulation_with_the_config_default_lora_dropout(
    tmp_path, monkeypatch
):
    """lora.dropout 0.05 reaches the adapters, and TRL zeroes it before the first forward."""
    _ga, l2l, ga_result, l2l_result = _pair(
        tmp_path, monkeypatch, k=2, rows=UNEVEN_ROWS, lora={**LORA, "dropout": 0.05}
    )
    _assert_equal_runs(ga_result, l2l_result, 3)
    import torch

    dropouts = [m for m in l2l.model.modules() if isinstance(m, torch.nn.Dropout)]
    assert dropouts and all(m.p == 0 for m in dropouts)


def test_only_the_policy_adapter_gets_gradients_and_it_is_the_active_one_afterwards(
    tmp_path, monkeypatch
):
    _ga, l2l, _ga_result, (grads, _logs) = _pair(tmp_path, monkeypatch, k=2, rows=UNEVEN_ROWS)
    assert grads
    for step in grads:
        assert [name for name in step if ".default." in name]
    assert list(l2l.model.peft_config) == ["default"]  # KTO makes no ``ref`` adapter
    assert l2l.model.active_adapter == "default"
    # the reference passes switched the adapter off and on: the flags are back
    assert all(p.requires_grad for n, p in l2l.model.named_parameters() if "lora_" in n)


def test_the_layer_load_counters_are_in_the_training_log(tmp_path, monkeypatch):
    _ga, _l2l, _ga_result, (_grads, logs) = _pair(tmp_path, monkeypatch, k=2, rows=UNIFORM_ROWS)
    assert [row["l2l_steps"] for row in logs] == [1, 2, 3]
    for key in ("l2l_layer_loads", "l2l_large_loads"):
        assert logs[0][key] > 0
        assert all(later[key] > earlier[key] for earlier, later in zip(logs, logs[1:]))
    assert "l2l_weight_restores" in logs[0]


def test_the_built_trainer_has_the_settings_the_l2l_loss_assumes(tmp_path, monkeypatch):
    """The L2L loss is TRL's plain KTO with the adapter off as the reference."""
    from torch.utils.data import SequentialSampler

    wrapper = _wrapper(tmp_path, monkeypatch, "defaults", l2l=True, k=2, rows=UNIFORM_ROWS)
    try:
        trainer = wrapper.trainer
        assert trainer.loss_type == "kto"
        assert trainer.calculate_KL is True
        assert trainer.ref_model is None
        assert trainer.ref_adapter_name is None
        assert not trainer.precompute_ref_log_probs
        assert trainer.args.dataset_num_proc is None
        assert isinstance(trainer._get_train_sampler(), SequentialSampler)
        assert trainer._soup_l2l.loss.divide_by == "micro_batches"
    finally:
        wrapper._close_stream_runtime()


def test_the_trainer_is_the_l2l_subclass_only_with_l2l_true(tmp_path, monkeypatch):
    from trl import KTOTrainer

    plain = _wrapper(tmp_path, monkeypatch, "plain", l2l=False, k=2, rows=UNIFORM_ROWS)
    streamed = _wrapper(tmp_path, monkeypatch, "streamed", l2l=True, k=2, rows=UNIFORM_ROWS)
    try:
        assert type(plain.trainer) is KTOTrainer
        assert not hasattr(plain.trainer, "soup_l2l")
        assert type(streamed.trainer) is not KTOTrainer
        assert isinstance(streamed.trainer, KTOTrainer)
        assert streamed.trainer.soup_l2l is True
    finally:
        plain._close_stream_runtime()
        streamed._close_stream_runtime()


def test_setup_checks_the_kl_blocks_before_and_after_building_the_trainer(
    tmp_path, monkeypatch
):
    """The args check runs before TRL maps the data (``dataset_num_proc`` forks workers); the
    sampler is only known once the trainer exists."""
    import soup_cli.trainer.l2l_kto as l2l_kto

    calls = []
    original = l2l_kto.refuse_unaligned_kl_blocks

    def spy(args, sampler=None):
        calls.append(sampler)
        return original(args, sampler)

    monkeypatch.setattr(l2l_kto, "refuse_unaligned_kl_blocks", spy)
    wrapper = _wrapper(tmp_path, monkeypatch, "spy", l2l=True, k=2, rows=UNIFORM_ROWS)
    try:
        assert len(calls) == 2 and calls[0] is None and calls[1] is not None
    finally:
        wrapper._close_stream_runtime()


# --------------------------------------------------------------------------
# start-up checks
# --------------------------------------------------------------------------
def test_setup_refuses_a_parallel_data_map_before_the_trainer_is_built(
    tmp_path, monkeypatch
):
    import trl

    real_config = trl.KTOConfig

    def parallel_config(*args, **kwargs):
        return real_config(*args, dataset_num_proc=2, **kwargs)

    monkeypatch.setattr(trl, "KTOConfig", parallel_config)
    _wrapper(
        tmp_path, monkeypatch, "parallel", l2l=True, k=2, rows=UNIFORM_ROWS,
        raises="dataset_num_proc",
    )


def test_setup_refuses_a_trainer_that_shuffles_the_rows(tmp_path, monkeypatch):
    import trl
    from torch.utils.data import RandomSampler

    monkeypatch.setattr(
        trl.KTOTrainer, "_get_train_sampler", lambda self, dataset=None: RandomSampler(range(4))
    )
    _wrapper(
        tmp_path, monkeypatch, "shuffled", l2l=True, k=2, rows=UNIFORM_ROWS,
        raises="SequentialSampler",
    )


def _kl_args(**overrides):
    values = {"dataset_num_proc": None, "world_size": 1}
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("num_proc", [None, 1])
def test_kl_blocks_that_match_the_micro_batches_are_accepted(num_proc):
    from torch.utils.data import SequentialSampler

    from soup_cli.trainer.l2l_kto import refuse_unaligned_kl_blocks

    refuse_unaligned_kl_blocks(_kl_args(dataset_num_proc=num_proc))
    refuse_unaligned_kl_blocks(
        _kl_args(dataset_num_proc=num_proc), SequentialSampler(range(4))
    )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"dataset_num_proc": 2}, "dataset_num_proc"),
        ({"dataset_num_proc": 8}, "dataset_num_proc"),
        ({"world_size": 2}, "process"),
    ],
)
def test_kl_blocks_that_may_not_match_the_micro_batches_are_refused(overrides, match):
    from soup_cli.trainer.l2l_kto import refuse_unaligned_kl_blocks

    with pytest.raises(ValueError, match=match):
        refuse_unaligned_kl_blocks(_kl_args(**overrides))


def test_a_sampler_that_does_not_keep_the_row_order_is_refused():
    from torch.utils.data import RandomSampler

    from soup_cli.trainer.l2l_kto import refuse_unaligned_kl_blocks

    with pytest.raises(ValueError, match="SequentialSampler"):
        refuse_unaligned_kl_blocks(_kl_args(), RandomSampler(range(4)))


@pytest.mark.parametrize(
    "attribute,value,match",
    [
        ("loss_type", "apo_zero_unpaired", "loss_type"),
        ("calculate_KL", False, "calculate_KL"),
        ("ref_model", object(), "ref_model"),
        ("precompute_ref_log_probs", True, "precompute_ref_log_probs"),
        ("ref_adapter_name", "ref", "ref_adapter_name"),
        ("aux_loss_enabled", True, "aux_loss_enabled"),
    ],
)
def test_a_trainer_that_departs_from_plain_kto_is_refused(attribute, value, match):
    from soup_cli.trainer.l2l_kto import refuse_unsupported_kto

    trainer = SimpleNamespace(
        loss_type="kto",
        calculate_KL=True,
        ref_model=None,
        precompute_ref_log_probs=False,
        ref_adapter_name=None,
        aux_loss_enabled=False,
    )
    refuse_unsupported_kto(trainer)
    setattr(trainer, attribute, value)
    with pytest.raises(ValueError, match=match):
        refuse_unsupported_kto(trainer)


# --------------------------------------------------------------------------
# the extra passes inside the L2L step (step level)
# --------------------------------------------------------------------------
@pytest.fixture()
def streamed_kto(tmp_path, monkeypatch):
    wrapper = _wrapper(tmp_path, monkeypatch, "step", l2l=True, k=2, rows=UNIFORM_ROWS)
    harness = _load("l2l_schedule")
    harness.make_non_vacuous(wrapper.model)  # the policy differs from the base model
    try:
        yield wrapper, harness
    finally:
        wrapper._close_stream_runtime()


def _other_ids(k, rows=3, seq=9):
    """Ids of a width the policy micro-batches do not have, as the KL completions can be."""
    import torch

    generator = torch.Generator().manual_seed(5)
    return [torch.randint(4, 60, (rows, seq), generator=generator) for _ in range(k)]


def test_extra_pass_with_the_adapters_off_returns_the_base_model_logits(streamed_kto):
    import torch

    from soup_cli.utils.l2l_step import ReferencePass

    wrapper, _harness = streamed_kto
    ids = _ids(2)
    reference = ReferencePass(adapters_off=True, head=_logits_head)
    _step(wrapper, ids[0]).run(ids, _CountingStore(2), reference=reference)

    with torch.no_grad():
        policy = [wrapper.model(input_ids=x).logits for x in ids]
        with wrapper.model.disable_adapter():
            expected = [wrapper.model(input_ids=x).logits for x in ids]
    assert sorted(reference.outputs) == [0, 1]
    for mb in range(2):
        assert not reference.outputs[mb].requires_grad
        assert torch.equal(reference.outputs[mb], expected[mb])
        assert not torch.equal(reference.outputs[mb], policy[mb])  # the adapter matters


def test_extra_pass_without_a_switch_returns_the_policy_logits_without_grad(streamed_kto):
    import torch

    from soup_cli.utils.l2l_step import ReferencePass

    wrapper, _harness = streamed_kto
    ids = _ids(2)
    reference = ReferencePass(head=_logits_head)
    _step(wrapper, ids[0]).run(ids, _CountingStore(2), reference=reference)
    with torch.no_grad():
        expected = [wrapper.model(input_ids=x).logits for x in ids]
    for mb in range(2):
        assert not reference.outputs[mb].requires_grad
        assert torch.equal(reference.outputs[mb], expected[mb])


def test_extra_pass_walks_its_own_ids_with_its_own_layer_calls(streamed_kto):
    import torch

    from soup_cli.utils.l2l_step import ReferencePass, capture_layer_call

    wrapper, _harness = streamed_kto
    ids, other = _ids(2), _other_ids(2)
    assert other[0].shape != ids[0].shape
    calls = [capture_layer_call(wrapper.model, x) for x in other]
    off = ReferencePass(adapters_off=True, head=_logits_head, micro_batches=other, calls=calls)
    on = ReferencePass(head=_logits_head, micro_batches=other, calls=calls)
    _step(wrapper, ids[0]).run(ids, _CountingStore(2), reference=[off, on])

    with torch.no_grad():
        expected_on = [wrapper.model(input_ids=x).logits for x in other]
        with wrapper.model.disable_adapter():
            expected_off = [wrapper.model(input_ids=x).logits for x in other]
    for mb in range(2):
        assert tuple(off.outputs[mb].shape) == (3, 9, 64)
        assert torch.equal(off.outputs[mb], expected_off[mb])
        assert torch.equal(on.outputs[mb], expected_on[mb])


def test_extra_pass_over_its_own_ids_needs_its_layer_calls(streamed_kto):
    from soup_cli.utils.l2l_step import ReferencePass

    wrapper, _harness = streamed_kto
    ids = _ids(2)
    reference = ReferencePass(head=_logits_head, micro_batches=_other_ids(2))
    with pytest.raises(ValueError, match="layer calls"):
        _step(wrapper, ids[0]).run(ids, _CountingStore(2), reference=reference)


def test_extra_pass_over_the_wrong_number_of_micro_batches_is_refused(streamed_kto):
    from soup_cli.utils.l2l_step import ReferencePass, capture_layer_call

    wrapper, _harness = streamed_kto
    ids, other = _ids(2), _other_ids(3)
    calls = [capture_layer_call(wrapper.model, x) for x in other]
    reference = ReferencePass(head=_logits_head, micro_batches=other, calls=calls)
    with pytest.raises(ValueError, match="micro-batches"):
        _step(wrapper, ids[0]).run(ids, _CountingStore(2), reference=reference)


def test_extra_pass_cannot_name_an_adapter_and_switch_the_adapters_off():
    from soup_cli.utils.l2l_step import ReferencePass

    with pytest.raises(ValueError, match="adapter"):
        ReferencePass(adapter="ref", adapters_off=True, head=_logits_head)


def test_extra_passes_leave_the_policy_losses_and_gradients_unchanged(streamed_kto):
    import torch

    from soup_cli.utils.l2l_step import ReferencePass, capture_layer_call

    wrapper, harness = streamed_kto
    ids, other = _ids(2), _other_ids(2)
    calls = [capture_layer_call(wrapper.model, x) for x in other]
    step = _step(wrapper, ids[0])
    harness.zero_grads(wrapper.model)
    plain_losses = step.run(ids, _CountingStore(2))
    plain = _grads(harness, wrapper.model)

    passes = [
        ReferencePass(head=_logits_head, micro_batches=other, calls=calls),
        ReferencePass(adapters_off=True, head=_logits_head),
        ReferencePass(adapters_off=True, head=_logits_head, micro_batches=other, calls=calls),
    ]
    harness.zero_grads(wrapper.model)
    losses = step.run(ids, _CountingStore(2), reference=passes)
    with_passes = _grads(harness, wrapper.model)

    assert all(torch.equal(a, b) for a, b in zip(plain_losses, losses))
    assert plain.keys() == with_passes.keys()
    assert all(torch.equal(plain[name], with_passes[name]) for name in plain)


def test_extra_passes_write_nothing_to_the_activation_store(streamed_kto):
    from soup_cli.utils.l2l_step import ReferencePass, capture_layer_call

    wrapper, _harness = streamed_kto
    ids, other = _ids(2), _other_ids(2)
    calls = [capture_layer_call(wrapper.model, x) for x in other]
    step = _step(wrapper, ids[0])
    plain = _CountingStore(2)
    step.run(ids, plain)
    with_passes = _CountingStore(2)
    step.run(
        ids,
        with_passes,
        reference=[
            ReferencePass(head=_logits_head, micro_batches=other, calls=calls),
            ReferencePass(adapters_off=True, head=_logits_head),
            ReferencePass(adapters_off=True, head=_logits_head, micro_batches=other, calls=calls),
        ],
    )
    assert plain.puts == with_passes.puts == 3 * 2  # layers x micro-batches


def test_extra_passes_give_the_adapters_back_as_they_found_them(streamed_kto):
    from soup_cli.utils.l2l_step import ReferencePass

    wrapper, harness = streamed_kto
    ids = _ids(2)
    before = {n: p.requires_grad for n, p in wrapper.model.named_parameters()}
    harness.zero_grads(wrapper.model)
    _step(wrapper, ids[0]).run(
        ids, _CountingStore(2), reference=[ReferencePass(adapters_off=True, head=_logits_head)]
    )
    assert wrapper.model.active_adapter == "default"
    assert {n: p.requires_grad for n, p in wrapper.model.named_parameters()} == before
    # the adapters are on again for whoever runs next
    assert not wrapper.model.base_model.model.model.layers[0].self_attn.q_proj.disable_adapters


@pytest.mark.parametrize("k", [1, 2, 4])
def test_a_kto_step_loads_each_layer_as_often_as_an_sft_step(streamed_kto, k):
    """The three extra passes add no layer load and no full read of the weights.

    Both numbers come from the runtime's own counters (``LoadCounters``): the SFT step and
    the KTO step walk the same model with the same k micro-batches.
    """
    from soup_cli.utils.l2l_step import ReferencePass, capture_layer_call

    wrapper, harness = streamed_kto
    ids, other = _ids(k), _other_ids(k)
    calls = [capture_layer_call(wrapper.model, x) for x in other]
    step = _step(wrapper, ids[0])

    harness.zero_grads(wrapper.model)
    _warm_walk(wrapper.model, ids[0])
    step.run(ids, _CountingStore(k))
    sft = step.counters

    harness.zero_grads(wrapper.model)
    _warm_walk(wrapper.model, ids[0])
    step.run(
        ids,
        _CountingStore(k),
        reference=[
            ReferencePass(head=_logits_head, micro_batches=other, calls=calls),
            ReferencePass(adapters_off=True, head=_logits_head),
            ReferencePass(adapters_off=True, head=_logits_head, micro_batches=other, calls=calls),
        ],
    )
    kto = step.counters

    assert sft.layer_loads > 0
    assert kto.layer_loads == sft.layer_loads
    assert kto.large_loads <= sft.large_loads
    assert kto.restores <= sft.restores


# --------------------------------------------------------------------------
# reading a micro-batch: TRL's collator
# --------------------------------------------------------------------------
def _example(prompt, completion, kl_completion, label):
    """One tokenized KTO row as TRL's ``_process_tokens`` leaves it (ids are made up)."""
    def side(answer):
        ids = list(prompt) + list(answer)
        return ids, [1] * len(ids), [-100] * len(prompt) + list(answer)

    ids, mask, labels = side(completion)
    kl_ids, kl_mask, kl_labels = side(kl_completion)
    return {
        "completion_input_ids": ids,
        "completion_attention_mask": mask,
        "completion_labels": labels,
        "KL_completion_input_ids": kl_ids,
        "KL_completion_attention_mask": kl_mask,
        "KL_completion_labels": kl_labels,
        "label": label,
    }


def _collated(*examples):
    """What TRL's ``DPODataCollatorWithPadding`` hands the trainer, built by that class."""
    from trl.experimental.utils import DPODataCollatorWithPadding

    return DPODataCollatorWithPadding(pad_token_id=3)(list(examples))


def _args(beta=0.3, loss_type="kto", desirable_weight=1.0, undesirable_weight=1.0):
    return SimpleNamespace(
        beta=beta,
        loss_type=loss_type,
        desirable_weight=desirable_weight,
        undesirable_weight=undesirable_weight,
    )


def _batch(labels=(True, False)):
    return _collated(
        _example((1, 2), (5, 6, 7), (8,), labels[0]),
        _example((1, 2), (9,), (5, 6, 7, 4), labels[1]),
    )


def test_the_window_reader_walks_the_completions_and_the_kl_completions_at_their_own_widths():
    import torch

    from soup_cli.trainer.l2l_kto import kto_window_loss

    strategy = kto_window_loss(args=_args())
    batch = _batch()
    ids_list, extras = strategy.read_window([batch], "cpu", neftune=False, has_dropout=False)
    assert len(ids_list) == 1
    assert torch.equal(ids_list[0], batch["completion_input_ids"])
    assert tuple(ids_list[0].shape) == (2, 5)

    policy_kl, reference, reference_kl = strategy.make_reference(extras)
    assert torch.equal(policy_kl.micro_batches[0], batch["KL_completion_input_ids"])
    assert tuple(policy_kl.micro_batches[0].shape) == (2, 6)
    assert reference_kl.micro_batches[0] is policy_kl.micro_batches[0]
    assert reference.micro_batches is None  # the completions are the policy's ids


def test_the_window_reader_sets_the_adapters_as_kto_does():
    from soup_cli.trainer.l2l_kto import kto_window_loss

    strategy = kto_window_loss(args=_args())
    _ids_list, extras = strategy.read_window(
        [_batch()], "cpu", neftune=False, has_dropout=False
    )
    policy_kl, reference, reference_kl = strategy.make_reference(extras)
    assert not policy_kl.adapters_off and policy_kl.adapter is None  # policy, no grad
    assert reference.adapters_off and reference_kl.adapters_off  # TRL's null_ref_context


def test_the_window_reader_gives_each_micro_batch_its_own_width():
    from soup_cli.trainer.l2l_kto import kto_window_loss

    strategy = kto_window_loss(args=_args())
    short = _collated(_example((1,), (5,), (6,), True), _example((1,), (7,), (8,), False))
    long = _collated(
        _example((1,), (5, 6, 7, 8), (6,), True), _example((1,), (7,), (8, 9, 4), False)
    )
    ids_list, _ = strategy.read_window([short, long], "cpu", neftune=False, has_dropout=False)
    assert [tuple(ids.shape) for ids in ids_list] == [(2, 2), (2, 5)]


@pytest.mark.parametrize(
    "key,row", [("completion_attention_mask", 1), ("KL_completion_attention_mask", 0)]
)
def test_the_window_reader_refuses_left_padding(key, row):
    """``row`` is the row of ``_batch`` that has padding in that mask."""
    from soup_cli.trainer.l2l_kto import kto_window_loss

    strategy = kto_window_loss(args=_args())
    batch = _batch()
    batch[key][row] = batch[key][row].flip(0)
    with pytest.raises(ValueError, match="right-padded"):
        strategy.read_window([batch], "cpu", neftune=False, has_dropout=False)


def test_the_window_reader_keeps_the_mixed_shape_refusals_of_sft():
    from soup_cli.trainer.l2l_kto import kto_window_loss

    strategy = kto_window_loss(args=_args())
    window = [
        _collated(_example((1,), (5,), (6,), True), _example((1,), (7,), (8,), False)),
        _collated(_example((1,), (5, 6, 7), (6,), True), _example((1,), (7,), (8,), False)),
    ]
    with pytest.raises(ValueError, match="neftune_alpha"):
        strategy.read_window(window, "cpu", neftune=True, has_dropout=False)
    with pytest.raises(ValueError, match="dropout"):
        strategy.read_window(window, "cpu", neftune=False, has_dropout=True)


@pytest.mark.parametrize("labels", [(True, True), (False, False), (True, False)])
def test_the_window_reader_takes_micro_batches_of_one_label_or_two(labels):
    from soup_cli.trainer.l2l_kto import kto_window_loss

    strategy = kto_window_loss(args=_args())
    ids_list, _ = strategy.read_window(
        [_batch(labels)], "cpu", neftune=False, has_dropout=False
    )
    assert len(ids_list) == 1


def test_kto_divides_the_window_by_k_and_needs_no_item_count():
    from soup_cli.trainer.l2l_kto import kto_window_loss

    strategy = kto_window_loss(args=_args())
    assert strategy.divide_by == "micro_batches"
    assert strategy.divisor(3, None) == 3


def test_the_trainer_window_loads_as_often_as_an_sft_window(streamed_kto):
    """The same count through the trainer's own path (``run_window``), four passes included."""
    import torch

    from soup_cli.trainer.l2l_trainer import SFT_WINDOW_LOSS, run_window
    from soup_cli.utils.l2l_step import label_tokens

    wrapper, harness = streamed_kto
    state = wrapper.trainer._soup_l2l
    batches = [_batch() for _ in range(2)]
    device = wrapper._stream_runtime.device
    kto_ids, kto_extras = state.loss.read_window(
        batches, device, neftune=False, has_dropout=False
    )

    harness.zero_grads(wrapper.model)
    _warm_walk(wrapper.model, kto_ids[0])
    kto_losses, kto = run_window(state, wrapper.model, kto_ids, kto_extras, None, device)

    kto_state_loss = state.loss
    state.loss = SFT_WINDOW_LOSS
    try:
        harness.zero_grads(wrapper.model)
        _warm_walk(wrapper.model, kto_ids[0])
        _sft_losses, sft = run_window(
            state, wrapper.model, kto_ids, kto_ids, label_tokens(kto_ids), device
        )
    finally:
        state.loss = kto_state_loss
    assert torch.isfinite(torch.stack(kto_losses)).all()
    assert kto.layer_loads == sft.layer_loads > 0
    assert kto.large_loads <= sft.large_loads
    assert kto.restores <= sft.restores


# --------------------------------------------------------------------------
# refusals in the schema
# --------------------------------------------------------------------------
def _l2l_yaml(task="kto", **training):
    import yaml

    return yaml.safe_dump(
        {
            "base": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "task": task,
            "data": {"train": "data.jsonl", "format": "kto"},
            "training": {
                "batch_size": 2,
                "quantization": "none",
                "stream_layers": True,
                "l2l": True,
                "gradient_accumulation_steps": 2,
                "lora": {"r": 4, "alpha": 8, "dropout": 0.0},
                **training,
            },
        }
    )


def test_the_schema_accepts_kto_with_l2l():
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.config.schema import L2L_SUPPORTED_TASKS

    assert "kto" in L2L_SUPPORTED_TASKS
    cfg = load_config_from_string(_l2l_yaml())
    assert cfg.task == "kto" and cfg.training.l2l is True


def test_the_schema_still_refuses_ipo_with_l2l():
    from soup_cli.config.loader import load_config_from_string

    with pytest.raises(ValueError, match="got task='ipo'"):
        load_config_from_string(_l2l_yaml(task="ipo"))


def test_the_schema_refuses_kto_with_l2l_when_streaming_is_off():
    from soup_cli.config.loader import load_config_from_string

    text = _l2l_yaml().replace("stream_layers: true", "stream_layers: false")
    with pytest.raises(ValueError, match="requires training.stream_layers"):
        load_config_from_string(text)


def test_the_schema_refuses_neftune_with_kto_and_l2l():
    from soup_cli.config.loader import load_config_from_string

    with pytest.raises(ValueError, match="neftune_alpha"):
        load_config_from_string(_l2l_yaml(neftune_alpha=5.0))


def test_the_schema_still_refuses_a_kto_batch_of_one_with_l2l():
    from soup_cli.config.loader import load_config_from_string

    with pytest.raises(ValueError, match="batch_size >= 2"):
        load_config_from_string(_l2l_yaml(batch_size=1))


def test_the_schema_accepts_the_default_lora_dropout_with_kto_and_l2l():
    """TRL's KTOTrainer zeroes every nn.Dropout, so the config default is harmless."""
    from soup_cli.config.loader import load_config_from_string

    text = _l2l_yaml().replace("dropout: 0.0", "dropout: 0.05")
    cfg = load_config_from_string(text)
    assert cfg.training.lora.dropout == 0.05 and cfg.training.l2l is True


def test_kto_without_l2l_keeps_neftune():
    from soup_cli.config.loader import load_config_from_string

    text = _l2l_yaml(neftune_alpha=5.0).replace("l2l: true", "l2l: false")
    cfg = load_config_from_string(text)
    assert cfg.training.neftune_alpha == 5.0


# --------------------------------------------------------------------------
# the short-training config and its synthetic rows
# --------------------------------------------------------------------------
CONFIG = ROOT / "examples" / "configs" / "kto_l2l_short.yaml"
DATA = ROOT / "examples" / "data" / "kto_synthetic.jsonl"


def test_the_short_config_passes_the_schema_check():
    from soup_cli.config.loader import load_config

    cfg = load_config(CONFIG)
    assert cfg.task == "kto"
    assert cfg.training.l2l is True and cfg.training.stream_layers is True
    assert cfg.training.batch_size >= 2
    assert cfg.data.train == "examples/data/kto_synthetic.jsonl"
    assert cfg.data.format == "kto"
    assert cfg.training.epochs == 1


def test_the_short_config_names_a_supported_architecture():
    import yaml

    text = CONFIG.read_text(encoding="utf-8")
    base = yaml.safe_load(text)["base"]
    assert "SmolLM2" in base or "Llama" in base or "llama" in base
    assert "synthetic" in text.lower()


def test_the_rows_are_small_valid_unpaired_and_labelled_synthetic():
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line]
    assert 8 <= len(rows) <= 40
    for row in rows:
        assert set(row) == {"prompt", "completion", "label"}
        assert isinstance(row["label"], bool)
        for key in ("prompt", "completion"):
            assert isinstance(row[key], str) and row[key].strip()
    # both labels, so a micro-batch of two can hold both
    assert {row["label"] for row in rows} == {True, False}
    readme = (ROOT / "examples" / "data" / "README.md").read_text(encoding="utf-8")
    line = next(line for line in readme.splitlines() if "kto_synthetic.jsonl" in line)
    assert "synthetic" in line.lower()
