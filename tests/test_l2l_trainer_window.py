"""A last window shorter than k through the real SFT trainer (ticket 02, CPU).

A dataset whose size k does not divide ends in a shorter window. HF builds that window
with its own ``num_items_in_batch``, takes one optimizer step on it and moves the lr
schedule by one. The L2L trainer must match gradient accumulation on the gradients, the
number of optimizer steps, the lr of every step and the logged loss.
"""

from __future__ import annotations

import math

import pytest

from tests.test_l2l_probe_harness import SEQ, _load, streamed  # noqa: F401
from tests.test_l2l_trainer_labels import _MaskedRows


def _train(base_cls, model, runtime, tmp_path, data, *, l2l, k, **overrides):
    """Train and return the logs, the lr and the LoRA gradient of every optimizer step."""
    import torch
    from transformers import TrainerCallback, TrainingArguments

    from soup_cli.trainer.l2l_trainer import attach_l2l_state, make_l2l_trainer_class

    grads, rates = [], []

    class Recorder(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
            grads.append(
                {
                    name: p.grad.detach().clone()
                    for name, p in model.named_parameters()
                    if "lora_" in name and p.grad is not None
                }
            )

        def on_step_end(self, args, state, control, optimizer=None, **kw):
            rates.append(optimizer.param_groups[0]["lr"])

    options = dict(
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
    )
    options.update(overrides)
    cls = make_l2l_trainer_class(base_cls) if l2l else base_cls
    trainer = cls(
        model=model, args=TrainingArguments(**options), train_dataset=data, callbacks=[Recorder()]
    )
    state = attach_l2l_state(trainer, runtime, dtype=torch.float32) if l2l else None
    trainer.train()
    logs = [row for row in trainer.state.log_history if "loss" in row]
    return logs, rates, grads, state, trainer


def _both(streamed, tmp_path, rows, k, **overrides):  # noqa: F811
    from transformers import Trainer

    harness = _load("l2l_schedule")
    model, runtime = streamed
    harness.make_non_vacuous(model)
    start = harness.snapshot_adapters(model)
    data = _MaskedRows([SEQ] * rows, [[(0, i % 3)] if i % 3 else [] for i in range(rows)])
    ga = _train(Trainer, model, runtime, tmp_path, data, l2l=False, k=k, **overrides)
    harness.restore_adapters(model, start)
    l2l = _train(Trainer, model, runtime, tmp_path, data, l2l=True, k=k, **overrides)
    return ga, l2l


def _assert_same_run(ga, l2l, windows):
    import torch

    ga_logs, ga_rates, ga_grads, _, ga_trainer = ga
    l2l_logs, l2l_rates, l2l_grads, state, l2l_trainer = l2l
    assert len(ga_grads) == len(l2l_grads) == windows
    assert ga_trainer.state.global_step == l2l_trainer.state.global_step == windows
    assert state.steps == windows
    for gradient_ga, gradient_l2l in zip(ga_grads, l2l_grads):
        assert gradient_ga.keys() == gradient_l2l.keys()
        assert any(bool(g.abs().sum() > 0) for g in gradient_l2l.values())
        for name in gradient_ga:
            assert torch.isfinite(gradient_l2l[name]).all(), name
            assert torch.equal(gradient_ga[name], gradient_l2l[name]), name
    assert l2l_rates == ga_rates
    assert all(math.isfinite(row["loss"]) for row in l2l_logs)
    assert [row["loss"] for row in l2l_logs] == [row["loss"] for row in ga_logs]
    assert [row["learning_rate"] for row in l2l_logs] == [row["learning_rate"] for row in ga_logs]
    assert l2l_logs[-1]["l2l_steps"] == windows
    for row in l2l_logs:
        assert {"l2l_steps", "l2l_layer_loads", "l2l_large_loads", "l2l_weight_restores"} <= set(
            row
        )


@pytest.mark.parametrize(
    "rows,k,windows", [(7, 3, 3), (8, 3, 3), (5, 4, 2), (3, 4, 1)]
)
def test_a_short_last_window_matches_accumulation(streamed, tmp_path, rows, k, windows):  # noqa: F811
    ga, l2l = _both(streamed, tmp_path, rows, k)
    _assert_same_run(ga, l2l, windows)


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_the_lr_of_every_step_matches_accumulation(streamed, tmp_path, schedule):  # noqa: F811
    ga, l2l = _both(
        streamed,
        tmp_path,
        7,
        3,
        lr_scheduler_type=schedule,
        warmup_steps=1,
        num_train_epochs=2,
    )
    _assert_same_run(ga, l2l, 6)
    assert len(set(l2l[1])) > 2  # the schedule really moves


def test_max_steps_counts_windows_across_epochs(streamed, tmp_path):  # noqa: F811
    ga, l2l = _both(streamed, tmp_path, 7, 3, max_steps=5, num_train_epochs=1)
    _assert_same_run(ga, l2l, 5)


def test_a_full_step_after_a_short_window_reuses_the_store(streamed, tmp_path):  # noqa: F811
    _, l2l = _both(streamed, tmp_path, 7, 3, num_train_epochs=2)
    state = l2l[3]
    # the epoch ends on a window of one micro-batch, the next epoch opens with a full one
    assert state.steps == 6
    assert state.store is not None and state.store.k == 3
    assert state.window is None
    state.close()
    assert state.store is None


def test_resume_does_not_shift_the_windows(streamed, tmp_path, monkeypatch):  # noqa: F811
    import torch
    from transformers import Trainer, TrainingArguments

    from soup_cli.trainer.l2l_trainer import attach_l2l_state, make_l2l_trainer_class

    # peft loads a resumed adapter onto the best device (mps on a Mac); the runtime is CPU
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    harness = _load("l2l_schedule")
    model, runtime = streamed
    harness.make_non_vacuous(model)
    start = harness.snapshot_adapters(model)
    data = _MaskedRows([SEQ] * 7, [[] for _ in range(7)])
    extra = dict(save_strategy="steps", save_steps=1, save_only_model=False, num_train_epochs=2)

    full = _train(Trainer, model, runtime, tmp_path, data, l2l=True, k=3, **extra)
    checkpoint = tmp_path / "l2l" / "checkpoint-4"
    assert checkpoint.is_dir()
    harness.restore_adapters(model, start)

    args = TrainingArguments(
        output_dir=str(tmp_path / "l2l"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=3,
        learning_rate=0.5,
        optim="sgd",
        lr_scheduler_type="constant",
        num_train_epochs=2,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        seed=1,
        data_seed=1,
        max_grad_norm=0.0,
        disable_tqdm=True,
    )
    trainer = make_l2l_trainer_class(Trainer)(model=model, args=args, train_dataset=data)
    state = attach_l2l_state(trainer, runtime, dtype=torch.float32)
    trainer.train(resume_from_checkpoint=str(checkpoint))
    assert trainer.state.global_step == full[4].state.global_step == 6
    # steps 5 and 6 run after the resume: a full window is not cut and the short one stays short
    assert state.steps == 2
    resumed = [row["loss"] for row in trainer.state.log_history if "loss" in row]
    original = [row["loss"] for row in full[0]]
    assert resumed[-2:] == original[-2:]
