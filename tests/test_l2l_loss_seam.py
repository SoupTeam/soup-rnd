"""Ticket A6-01: the loss seam of the L2L step and of the L2L trainer (CPU).

Three seams:

* ``L2LStep.run(head=...)``: the head and the loss of one micro-batch are a function the
  caller passes in; the walk over the layers does not change.
* the window-loss strategy of the trainer (``WindowLoss``): how a window is read, which
  head it builds and what the window loss is divided by.
* the schema's list of tasks that L2L supports.
"""

from __future__ import annotations

import pytest

from tests.test_l2l_probe_harness import SEQ, _load, _micro_batches, streamed  # noqa: F401


def _store(k, n_layers=3, hidden=64):
    import torch

    from soup_cli.utils.l2l_activations import ActivationStore

    return ActivationStore(
        n_layers=n_layers,
        k=k,
        chunk_shape=(1, SEQ, hidden),
        dtype=torch.float32,
        device="cpu",
        pin=False,
    )


# --------------------------------------------------------------------------
# seam 1: L2LStep.run(head=...)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("k", [1, 2, 4])
def test_a_custom_head_gives_the_gradients_of_accumulation_with_the_same_loss(streamed, k):  # noqa: F811
    """The loss is not the model's: mean of squared logits, divided by k as HF divides it."""
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call

    harness = _load("l2l_schedule")
    model, runtime = streamed
    harness.make_non_vacuous(model)
    mbs = _micro_batches(k)

    harness.zero_grads(model)
    ga_losses = []
    for ids in mbs:
        loss = model(input_ids=ids).logits.float().pow(2).mean() / k
        loss.backward()
        ga_losses.append(loss.detach())
    ga = harness.collect_grads(model)

    def head(causal, owner, hidden, mb):
        return causal.lm_head(owner.norm(hidden)).float().pow(2).mean() / k

    step = L2LStep(model, runtime, capture_layer_call(model, mbs[0]))
    harness.zero_grads(model)
    l2l_losses = step.run(mbs, _store(k), head=head)
    l2l = harness.collect_grads(model)

    result = harness.compare(ga, l2l)
    assert result["all_zero"] == []
    assert result["equal"], result
    assert harness.losses_equal(ga_losses, l2l_losses)


def test_a_custom_head_excludes_labels_and_n_items(streamed):  # noqa: F811
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call, sft_head

    model, runtime = streamed
    mbs = _micro_batches(2)
    step = L2LStep(model, runtime, capture_layer_call(model, mbs[0]))
    head = sft_head(mbs, 14)
    with pytest.raises(ValueError, match="owns its labels"):
        step.run(mbs, _store(2), head=head, labels=mbs)
    with pytest.raises(ValueError, match="owns its labels"):
        step.run(mbs, _store(2), head=head, n_items=14)


@pytest.mark.parametrize("k", [1, 2, 4])
def test_the_sft_head_passed_explicitly_equals_the_default_bit_for_bit(streamed, k):  # noqa: F811
    from soup_cli.utils.l2l_step import L2LStep, capture_layer_call, label_tokens, sft_head

    harness = _load("l2l_schedule")
    model, runtime = streamed
    harness.make_non_vacuous(model)
    mbs = _micro_batches(k)
    step = L2LStep(model, runtime, capture_layer_call(model, mbs[0]))

    harness.zero_grads(model)
    default_losses = step.run(mbs, _store(k))
    default = harness.collect_grads(model)

    harness.zero_grads(model)
    explicit_losses = step.run(mbs, _store(k), head=sft_head(mbs, label_tokens(mbs)))
    explicit = harness.collect_grads(model)

    assert harness.compare(default, explicit)["equal"]
    assert harness.losses_equal(default_losses, explicit_losses)


# --------------------------------------------------------------------------
# seam 2: the window-loss strategy of the trainer
# --------------------------------------------------------------------------
def _strategy(divide_by):
    from soup_cli.trainer.l2l_trainer import WindowLoss

    return WindowLoss(
        divide_by=divide_by,
        read_window=lambda window, device, **kw: ([w["ids"].to(device) for w in window], None),
        make_head=lambda extras, divisor: None,
    )


@pytest.mark.parametrize("k", [1, 2, 4])
def test_sft_divides_the_window_by_the_hf_item_count_and_preference_by_k(k):
    from soup_cli.trainer.l2l_trainer import SFT_WINDOW_LOSS

    assert SFT_WINDOW_LOSS.divisor(k, 37) == 37
    assert _strategy("micro_batches").divisor(k, 37) == k
    assert _strategy("micro_batches").divisor(k, None) == k


def test_sft_without_num_items_in_batch_is_refused_and_preference_is_not():
    from soup_cli.trainer.l2l_trainer import SFT_WINDOW_LOSS

    with pytest.raises(RuntimeError, match="num_items_in_batch"):
        SFT_WINDOW_LOSS.divisor(2, None)
    assert _strategy("micro_batches").divisor(2, None) == 2


class _IdRows:
    """Rows with no ``input_ids`` and no ``labels``, like a preference batch."""

    def __init__(self, n, vocab=64, seed=5):
        import torch

        gen = torch.Generator().manual_seed(seed)
        self.rows = [{"ids": torch.randint(0, vocab, (SEQ,), generator=gen)} for _ in range(n)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def _synthetic_loss(logits):
    return logits.float().pow(2).mean()


def _synthetic_window_loss():
    from soup_cli.trainer.l2l_trainer import WindowLoss

    def read(window, device, *, neftune=False, has_dropout=False):
        return [row["ids"].to(device) for row in window], None

    def make_head(_extras, divisor):
        def head(causal, owner, hidden, mb):
            return _synthetic_loss(causal.lm_head(owner.norm(hidden))) / divisor

        return head

    return WindowLoss(divide_by="micro_batches", read_window=read, make_head=make_head)


def _run_synthetic(streamed, tmp_path, *, l2l, rows, k):  # noqa: F811
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments

    from soup_cli.trainer.l2l_trainer import attach_l2l_state, make_l2l_trainer_class

    model, runtime = streamed
    grads, seen = [], []

    class Recorder(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
            grads.append(
                {
                    name: p.grad.detach().clone()
                    for name, p in model.named_parameters()
                    if "lora_" in name and p.grad is not None
                }
            )

    class Accumulation(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            seen.append(num_items_in_batch)
            return _synthetic_loss(model(input_ids=inputs["ids"]).logits)

    base = Accumulation
    cls = make_l2l_trainer_class(base) if l2l else base
    args = TrainingArguments(
        output_dir=str(tmp_path / ("l2l" if l2l else "ga")),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=k,
        learning_rate=0.5,
        optim="sgd",
        lr_scheduler_type="constant",
        num_train_epochs=1,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        seed=1,
        data_seed=1,
        max_grad_norm=0.0,
        disable_tqdm=True,
        remove_unused_columns=False,
    )
    trainer = cls(
        model=model,
        args=args,
        train_dataset=_IdRows(rows),
        callbacks=[Recorder()],
        data_collator=lambda batch: {"ids": torch.stack([row["ids"] for row in batch])},
    )
    state = None
    if l2l:
        state = attach_l2l_state(
            trainer, runtime, dtype=torch.float32, loss=_synthetic_window_loss()
        )
    trainer.train()
    logs = [row["loss"] for row in trainer.state.log_history if "loss" in row]
    return grads, logs, seen, state


def test_the_trainer_runs_a_task_without_input_ids_or_item_count_like_accumulation(
    streamed, tmp_path  # noqa: F811
):
    """5 rows at k=2 make windows of 2, 2 and 1: the last divides by 1, not by 2."""
    import torch

    harness = _load("l2l_schedule")
    model, _runtime = streamed
    harness.make_non_vacuous(model)
    start = harness.snapshot_adapters(model)

    ga_grads, ga_logs, ga_seen, _ = _run_synthetic(streamed, tmp_path, l2l=False, rows=5, k=2)
    harness.restore_adapters(model, start)
    l2l_grads, l2l_logs, l2l_seen, state = _run_synthetic(
        streamed, tmp_path, l2l=True, rows=5, k=2
    )

    assert set(ga_seen) == {None}  # HF gives a preference batch no item count
    assert l2l_seen == []  # the L2L trainer never calls compute_loss
    assert state.steps == len(ga_grads) == len(l2l_grads) == 3
    for ga, l2l in zip(ga_grads, l2l_grads):
        assert ga.keys() == l2l.keys()
        assert any(bool(g.abs().sum() > 0) for g in l2l.values())
        for name in ga:
            assert torch.equal(ga[name], l2l[name]), name
    assert l2l_logs == ga_logs


# --------------------------------------------------------------------------
# seam 3: the schema's list of tasks that L2L supports
# --------------------------------------------------------------------------
def _l2l_yaml(task):
    import yaml

    return yaml.safe_dump(
        {
            "base": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "task": task,
            "data": {"train": "data.jsonl"},
            "training": {
                "batch_size": 1,
                "quantization": "none",
                "stream_layers": True,
                "l2l": True,
                "gradient_accumulation_steps": 2,
                "lora": {"r": 4, "alpha": 8},
            },
        }
    )


def test_the_list_of_supported_tasks_holds_sft_only_for_now():
    from soup_cli.config.schema import L2L_SUPPORTED_TASKS

    assert L2L_SUPPORTED_TASKS == ("sft",)


def test_the_refusal_names_the_task_and_the_supported_list():
    from soup_cli.config.loader import load_config_from_string

    with pytest.raises(ValueError) as caught:
        load_config_from_string(_l2l_yaml("dpo"))
    message = str(caught.value)
    assert "training.l2l supports task in ['sft']" in message
    assert "got task='dpo'" in message


def test_the_refusal_reads_the_list_instead_of_a_hardcoded_task(monkeypatch):
    from soup_cli.config import schema
    from soup_cli.config.loader import load_config_from_string

    monkeypatch.setattr(schema, "L2L_SUPPORTED_TASKS", ("sft", "dpo"))
    assert load_config_from_string(_l2l_yaml("dpo")).training.l2l is True
    with pytest.raises(ValueError, match=r"supports task in \['sft', 'dpo'\].*got task='orpo'"):
        load_config_from_string(_l2l_yaml("orpo"))


def _bare_trainer(loss):
    """An L2L trainer with a window pending and no model, to reach the top of training_step."""
    import torch

    from soup_cli.trainer.l2l_trainer import L2LState, make_l2l_trainer_class

    class Base:
        class args:  # noqa: N801
            device = "cpu"

    trainer = make_l2l_trainer_class(Base)()
    trainer._soup_l2l = L2LState(
        runtime=None, dtype=torch.float32, window=[{}, {}], loss=loss
    )
    return trainer


def test_sft_training_step_refuses_a_missing_item_count_before_touching_the_model():
    from soup_cli.trainer.l2l_trainer import SFT_WINDOW_LOSS

    trainer = _bare_trainer(SFT_WINDOW_LOSS)
    with pytest.raises(RuntimeError, match="num_items_in_batch"):
        trainer.training_step(None, {}, None)


def test_the_state_defaults_to_the_sft_window_loss():
    import torch

    from soup_cli.trainer.l2l_trainer import SFT_WINDOW_LOSS, L2LState

    assert L2LState(runtime=None, dtype=torch.float32).loss is SFT_WINDOW_LOSS
