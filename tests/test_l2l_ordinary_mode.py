"""Ticket 03: without `training.l2l` the SFT trainer runs the ordinary HF path (CPU).

Seams: `SFTTrainerWrapper` (which trainer it builds and what its log carries), the trainer
subclass from `make_l2l_trainer_class` driven by HF's loop, and the recipe catalog. The
recipe config snapshots are pinned by `test_issue621_recipe_snapshot.py`.
"""

from __future__ import annotations

import pytest

from tests.test_l2l_probe_harness import _load, streamed  # noqa: F401
from tests.test_l2l_trainer_mode import _adapters, _train, _wrapper


# --------------------------------------------------------------------------
# the production path without l2l
# --------------------------------------------------------------------------
@pytest.mark.parametrize("stream_layers", [True, False])
def test_wrapper_without_l2l_builds_an_ordinary_trainer(tmp_path, monkeypatch, stream_layers):
    from trl import SFTTrainer

    wrapper = _wrapper(tmp_path, monkeypatch, l2l=False, stream_layers=stream_layers)
    trainer = wrapper.trainer
    assert isinstance(trainer, SFTTrainer)
    assert type(trainer).training_step is SFTTrainer.training_step
    assert not hasattr(trainer, "soup_l2l")


@pytest.mark.parametrize("stream_layers", [True, False])
def test_wrapper_without_l2l_logs_no_l2l_fields(tmp_path, monkeypatch, stream_layers):
    # the ordinary forward moves the batch to the device HF picks, mps on an Apple GPU box
    monkeypatch.setattr("transformers.training_args.is_torch_mps_available", lambda *a, **k: False)
    wrapper = _wrapper(tmp_path, monkeypatch, l2l=False, stream_layers=stream_layers)
    wrapper.trainer.train()
    rows = wrapper.trainer.state.log_history
    assert any("loss" in row for row in rows)
    assert wrapper.trainer.state.global_step == 2
    assert not [key for row in rows for key in row if key.startswith("l2l_")]


# --------------------------------------------------------------------------
# streaming without l2l is still gradient accumulation
# --------------------------------------------------------------------------
def test_streaming_without_l2l_accumulates_like_one_batch_of_the_same_rows(
    streamed, tmp_path  # noqa: F811
):
    """Independent reference: two rows in one batch of 2 equal two micro-batches of 1."""
    import torch
    from transformers import Trainer

    harness = _load("l2l_schedule")
    model, runtime = streamed
    harness.make_non_vacuous(model)
    start = harness.snapshot_adapters(model)

    trainer, _ = _train(Trainer, model, runtime, tmp_path, l2l=False, k=2, rows=4)
    accumulated = _adapters(model)
    acc_losses = [row["loss"] for row in trainer.state.log_history if "loss" in row]

    harness.restore_adapters(model, start)
    trainer, _ = _train(
        Trainer, model, runtime, tmp_path, l2l=False, k=1, rows=4, batch_size=2
    )
    batched = _adapters(model)
    batch_losses = [row["loss"] for row in trainer.state.log_history if "loss" in row]

    assert acc_losses == pytest.approx(batch_losses, abs=1e-5, rel=1e-4)
    for name in accumulated:
        assert torch.allclose(accumulated[name], batched[name], atol=1e-6, rtol=1e-5), name
    assert any(not torch.equal(accumulated[n], start[n]) for n in accumulated)


# --------------------------------------------------------------------------
# l2l at k=1 is plain streaming
# --------------------------------------------------------------------------
def test_l2l_with_k_one_equals_streaming_without_l2l(streamed, tmp_path):  # noqa: F811
    import math

    import torch
    from transformers import Trainer

    harness = _load("l2l_schedule")
    model, runtime = streamed
    harness.make_non_vacuous(model)
    start = harness.snapshot_adapters(model)

    plain_trainer, plain_steps = _train(Trainer, model, runtime, tmp_path, l2l=False, k=1)
    plain = _adapters(model)
    plain_losses = [r["loss"] for r in plain_trainer.state.log_history if "loss" in r]

    harness.restore_adapters(model, start)
    l2l_trainer, l2l_steps = _train(Trainer, model, runtime, tmp_path, l2l=True, k=1)
    l2l = _adapters(model)
    l2l_losses = [r["loss"] for r in l2l_trainer.state.log_history if "loss" in r]

    assert plain_steps == l2l_steps == 4
    # control: the L2L path ran at k=1, the plain run never logs its counters
    assert l2l_trainer.state.log_history[-2]["l2l_steps"] == 4
    assert not [k for r in plain_trainer.state.log_history for k in r if k.startswith("l2l_")]
    assert all(math.isfinite(loss) for loss in l2l_losses)
    assert l2l_losses == pytest.approx(plain_losses, abs=1e-5, rel=1e-4)
    for name in plain:
        assert torch.allclose(plain[name], l2l[name], atol=1e-6, rtol=1e-5), name
    assert any(not torch.equal(l2l[n], start[n]) for n in l2l)


# --------------------------------------------------------------------------
# the subclass without attached state, and the recipes
# --------------------------------------------------------------------------
def test_subclass_without_state_trains_as_the_base_trainer(
    streamed, tmp_path  # noqa: F811
):
    import torch
    from transformers import Trainer

    from soup_cli.trainer.l2l_trainer import make_l2l_trainer_class

    harness = _load("l2l_schedule")
    model, runtime = streamed
    harness.make_non_vacuous(model)
    start = harness.snapshot_adapters(model)

    _train(Trainer, model, runtime, tmp_path, l2l=False, k=2)
    base = _adapters(model)
    harness.restore_adapters(model, start)

    class Stateless(make_l2l_trainer_class(Trainer)):
        pass

    # l2l=False in the helper uses the given class and attaches no state
    _train(Stateless, model, runtime, tmp_path, l2l=False, k=2)
    plain = _adapters(model)
    for name in base:
        assert torch.equal(base[name], plain[name]), name


def test_no_recipe_turns_l2l_on():
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.recipes.catalog import RECIPES

    for name, recipe in RECIPES.items():
        cfg = load_config_from_string(recipe.yaml_str)
        assert cfg.training.l2l is False, name
