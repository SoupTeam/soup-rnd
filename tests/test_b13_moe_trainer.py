"""B1.3 E4/E5: real HF windows against an independent resident eager MoE model.

These checks catch wrong window normalization, missed expert backward reloads, stale
adapter state across optimizer steps, or autocast omitted by the trainer boundary.
Dropout plus uneven lengths is already refused/tested in test_l2l_trainer_mode.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from tests.test_b13_moe_runtime import (
    build_moe_streamed,
    lora_parameters,
    moe_config,
    moe_resident_reference,
    moe_weights_dir,
)
from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR
from tests.test_l2l_trainer_labels import _MaskedRows

pytestmark = _REQUIRES_TRAINING_FLOOR


def _values(model: Any, *, gradients: bool = False) -> dict:
    return {
        name: None if value is None else value.detach().cpu().clone()
        for name, param in lora_parameters(model).items()
        for value in [param.grad if gradients else param]
    }


def _train(model, runtime, path, data, *, k, l2l, bf16):
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments

    from soup_cli.trainer.l2l_trainer import attach_l2l_state, make_l2l_trainer_class

    grads, losses, rates, active_losses = [], [], [], []

    class Recorder(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            grads.append(_values(model, gradients=True))
            losses.append(torch.stack(active_losses).sum())
            active_losses.clear()

        def on_step_end(self, args, state, control, optimizer=None, **kwargs):
            rates.append(optimizer.param_groups[0]["lr"])

    base = make_l2l_trainer_class(Trainer) if l2l else Trainer

    class MeasuredTrainer(base):
        def training_step(self, *args, **kwargs):
            loss = super().training_step(*args, **kwargs)
            active_losses.append(loss.detach().cpu().clone())
            return loss

    args = TrainingArguments(
        output_dir=str(path), per_device_train_batch_size=1,
        gradient_accumulation_steps=k, learning_rate=0.5, optim="sgd",
        lr_scheduler_type="linear", num_train_epochs=1, logging_steps=1,
        save_strategy="no", report_to=[], use_cpu=True, seed=1, data_seed=1,
        max_grad_norm=0.0, disable_tqdm=True, bf16=bf16,
    )
    trainer = MeasuredTrainer(
        model=model, args=args, train_dataset=data, callbacks=[Recorder()],
    )
    state = attach_l2l_state(trainer, runtime, dtype=torch.float32) if l2l else None
    try:
        trainer.train()
        return {
            "grads": grads, "losses": losses, "rates": rates, "final": _values(model),
            "steps": trainer.state.global_step,
            "logs": [row for row in trainer.state.log_history if "loss" in row],
            "l2l_steps": state.steps if state is not None else None,
            "window": state.window if state is not None else None,
        }
    finally:
        if state is not None:
            state.close()


def _compare(ga, streamed, *, windows, atol, rtol, record_property):
    import torch

    assert ga["steps"] == streamed["steps"] == streamed["l2l_steps"] == windows
    assert streamed["window"] is None
    assert len(ga["grads"]) == len(streamed["grads"]) == windows
    assert len(ga["losses"]) == len(streamed["losses"]) == windows
    maxima = {"gradient": 0.0, "loss": 0.0, "trained_state": 0.0}
    failures = []

    def same(left, right, kind, name):
        assert left is not None and right is not None, name
        assert torch.isfinite(left).all() and torch.isfinite(right).all(), name
        maxima[kind] = max(maxima[kind], float((left - right).abs().max()))
        matches = torch.equal(left, right) if atol == rtol == 0 else torch.allclose(
            left, right, atol=atol, rtol=rtol,
        )
        if not matches:
            failures.append((kind, name, float((left - right).abs().max())))

    for step, (want, got) in enumerate(zip(ga["grads"], streamed["grads"])):
        assert want.keys() == got.keys()
        for name in want:
            assert got[name] is not None and bool(got[name].ne(0).any()), (step, name)
            same(want[name], got[name], "gradient", (step, name))
        record_property(
            f"step_{step}_max_abs_gradient_diff",
            max(float((want[name] - got[name]).abs().max()) for name in want),
        )
    for step, (want, got) in enumerate(zip(ga["losses"], streamed["losses"])):
        same(want, got, "loss", step)
    assert ga["final"].keys() == streamed["final"].keys()
    for name in ga["final"]:
        same(ga["final"][name], streamed["final"][name], "trained_state", name)
    for name, value in maxima.items():
        record_property(f"max_abs_{name}_diff", value)
    record_property("atol", atol)
    record_property("rtol", rtol)
    assert ga["rates"] == streamed["rates"]
    assert all(math.isfinite(row["loss"]) for row in streamed["logs"])
    assert [row["loss"] for row in streamed["logs"]] == pytest.approx(
        [row["loss"] for row in ga["logs"]], abs=atol, rel=rtol,
    )
    assert [row["l2l_steps"] for row in streamed["logs"]] == list(range(1, windows + 1))
    for key in ("l2l_layer_loads", "l2l_expert_fetches", "l2l_expert_bytes_read"):
        counts = [row[key] for row in streamed["logs"]]
        assert counts[0] > 0
        assert all(after > before for before, after in zip(counts, counts[1:])), key
    assert not failures, failures[:8]


@pytest.mark.parametrize(
    "k,uneven,bf16",
    [(1, False, False), (2, False, False), (4, False, False),
     (2, True, False), (4, True, False),
     (1, False, True), (2, False, True), (4, False, True)],
)
def test_trainer_moe_windows_match_resident_ga(tmp_path, record_property, k, uneven, bf16):
    """Three real updates (including a short final window), masked labels and right padding."""
    import torch

    config = moe_config()
    weights = moe_weights_dir(tmp_path, config)
    moe = build_moe_streamed(weights, tmp_path / "shards")
    try:
        resident = moe_resident_reference(weights, moe.lora_config, lora_from=moe.model)
        before = _values(moe.model)
        rows = 2 * k + 1
        lengths = ([5, 8, 3, 6, 9] * rows)[:rows] if uneven else [6] * rows
        data = _MaskedRows(lengths, [[(0, 1 + index % 2)] for index in range(rows)])
        if uneven:
            # Exercise already right-padded rows as well as padding between micro-batches.
            data.rows[1]["input_ids"][-2:] = 0
            data.rows[1]["labels"][-2:] = -100
            data.rows[1]["attention_mask"][-2:] = 0
        ga = _train(resident, None, tmp_path / "ga", data, k=k, l2l=False, bf16=bf16)
        actual = _train(moe.model, moe.runtime, tmp_path / "l2l", data, k=k, l2l=True, bf16=bf16)
        atol, rtol = (2e-5, 1e-4) if uneven else ((1e-6, 1e-5) if bf16 else (0.0, 0.0))
        _compare(
            ga, actual, windows=math.ceil(rows / k), atol=atol, rtol=rtol,
            record_property=record_property,
        )
        assert any(not torch.equal(before[name], actual["final"][name]) for name in before)
        assert len(before) == 30  # five MLA projections x A/B x three decoder layers
        assert all("lora_" in name for name in before)
    finally:
        moe.close()
