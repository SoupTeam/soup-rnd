"""Launch guards of the L2L preference paths (A6 PR review).

* A task that did not pass the readiness rule prints a warning when its trainer is built.
* The trainer checks fail when TRL drops or renames an attribute, instead of passing.
* DPO also checks ``label_smoothing`` and ``use_weighting``; ORPO and SimPO have checks too.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from rich.console import Console

# --------------------------------------------------------------------------
# warning for tasks that are not ready
# --------------------------------------------------------------------------


@pytest.mark.parametrize("task", ["sft", "dpo"])
def test_a_ready_task_gets_no_warning(task):
    from soup_cli.trainer.l2l_trainer import unready_task_warning

    assert unready_task_warning(task) is None


@pytest.mark.parametrize("task", ["orpo", "simpo", "kto"])
def test_a_task_that_did_not_pass_the_readiness_rule_gets_a_warning(task):
    from soup_cli.trainer.l2l_trainer import unready_task_warning

    text = unready_task_warning(task)
    assert text is not None
    assert task in text
    assert "not ready" in text
    assert "docs/performance-and-quantization.md" in text


def test_the_ready_tasks_are_a_subset_of_the_accepted_ones():
    from soup_cli.config.schema import L2L_SUPPORTED_TASKS
    from soup_cli.trainer.l2l_trainer import L2L_READY_TASKS

    assert set(L2L_READY_TASKS) <= set(L2L_SUPPORTED_TASKS)
    assert set(L2L_READY_TASKS) == {"sft", "dpo"}


def _build_with_task(monkeypatch, task):
    """Run the shared ``_build_l2l_trainer`` up to its printed lines, with a fake trainer."""
    from soup_cli.trainer import l2l_trainer, stream_setup
    from soup_cli.trainer.orpo import ORPOTrainerWrapper

    recorder = Console(record=True, width=200, force_terminal=False)
    monkeypatch.setattr(stream_setup, "console", recorder)
    monkeypatch.setattr(l2l_trainer, "make_l2l_trainer_class", lambda base: lambda **kw: object())
    monkeypatch.setattr(l2l_trainer, "attach_l2l_state", lambda *a, **kw: None)

    wrapper = ORPOTrainerWrapper.__new__(ORPOTrainerWrapper)
    wrapper.device = "cpu"
    wrapper.model = SimpleNamespace(config=SimpleNamespace(model_type="llama"))
    wrapper._stream_runtime = object()
    wrapper._stream_weight_dirs = ()
    wrapper.config = SimpleNamespace(task=task, training=SimpleNamespace(l2l_spill_dir=None))
    args = SimpleNamespace(n_gpu=1, fp16=False, gradient_accumulation_steps=2)
    wrapper._build_l2l_trainer(object, {}, args)
    return recorder.export_text()


@pytest.mark.parametrize("task", ["orpo", "simpo", "kto"])
def test_building_the_trainer_of_an_unready_task_prints_the_warning(monkeypatch, task):
    printed = _build_with_task(monkeypatch, task)
    assert "not ready" in printed
    assert task in printed


@pytest.mark.parametrize("task", ["sft", "dpo"])
def test_building_the_trainer_of_a_ready_task_prints_no_warning(monkeypatch, task):
    assert "not ready" not in _build_with_task(monkeypatch, task)


# --------------------------------------------------------------------------
# a missing attribute is a refusal, not a pass
# --------------------------------------------------------------------------


def _dpo_trainer(**overrides):
    values = dict(
        loss_types=["sigmoid"],
        loss_weights=[1.0],
        ref_model=None,
        precompute_ref_logps=False,
        ld_alpha=None,
        f_divergence_type="reverse_kl",
        label_smoothing=0.0,
        use_weighting=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "attribute,value,match",
    [
        ("label_smoothing", 0.1, "label_smoothing"),
        ("use_weighting", True, "use_weighting"),
    ],
)
def test_dpo_refuses_the_settings_that_change_the_sigmoid_loss(attribute, value, match):
    from soup_cli.trainer.l2l_dpo import refuse_unsupported_dpo

    refuse_unsupported_dpo(_dpo_trainer())
    with pytest.raises(ValueError, match=match):
        refuse_unsupported_dpo(_dpo_trainer(**{attribute: value}))


@pytest.mark.parametrize(
    "attribute",
    [
        "loss_types",
        "loss_weights",
        "ref_model",
        "precompute_ref_logps",
        "ld_alpha",
        "f_divergence_type",
        "label_smoothing",
        "use_weighting",
    ],
)
def test_dpo_refuses_a_trainer_without_the_attribute(attribute):
    from soup_cli.trainer.l2l_dpo import refuse_unsupported_dpo

    trainer = _dpo_trainer()
    delattr(trainer, attribute)
    with pytest.raises(ValueError, match=attribute):
        refuse_unsupported_dpo(trainer)


def _kto_trainer(**overrides):
    values = dict(
        loss_type="kto",
        calculate_KL=True,
        ref_model=None,
        precompute_ref_log_probs=False,
        ref_adapter_name=None,
        aux_loss_enabled=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "attribute",
    [
        "loss_type",
        "calculate_KL",
        "ref_model",
        "precompute_ref_log_probs",
        "ref_adapter_name",
        "aux_loss_enabled",
    ],
)
def test_kto_refuses_a_trainer_without_the_attribute(attribute):
    from soup_cli.trainer.l2l_kto import refuse_unsupported_kto

    trainer = _kto_trainer()
    delattr(trainer, attribute)
    with pytest.raises(ValueError, match=attribute):
        refuse_unsupported_kto(trainer)


# --------------------------------------------------------------------------
# ORPO and SimPO
# --------------------------------------------------------------------------


@pytest.mark.parametrize("attribute,value", [("aux_loss_enabled", True)])
def test_orpo_refuses_a_trainer_that_departs_from_plain_orpo(attribute, value):
    from soup_cli.trainer.l2l_orpo import refuse_unsupported_orpo

    trainer = SimpleNamespace(aux_loss_enabled=False)
    refuse_unsupported_orpo(trainer)
    setattr(trainer, attribute, value)
    with pytest.raises(ValueError, match=attribute):
        refuse_unsupported_orpo(trainer)


def test_orpo_refuses_a_trainer_without_the_attribute():
    from soup_cli.trainer.l2l_orpo import refuse_unsupported_orpo

    with pytest.raises(ValueError, match="aux_loss_enabled"):
        refuse_unsupported_orpo(SimpleNamespace())


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("loss_type", "sigmoid"),
        ("label_smoothing", 0.1),
        ("aux_loss_enabled", True),
    ],
)
def test_simpo_refuses_a_trainer_that_departs_from_plain_simpo(attribute, value):
    from soup_cli.trainer.l2l_simpo import refuse_unsupported_simpo

    trainer = SimpleNamespace(loss_type="simpo", label_smoothing=0.0, aux_loss_enabled=False)
    refuse_unsupported_simpo(trainer)
    setattr(trainer, attribute, value)
    with pytest.raises(ValueError, match=attribute):
        refuse_unsupported_simpo(trainer)


@pytest.mark.parametrize("attribute", ["loss_type", "label_smoothing", "aux_loss_enabled"])
def test_simpo_refuses_a_trainer_without_the_attribute(attribute):
    from soup_cli.trainer.l2l_simpo import refuse_unsupported_simpo

    trainer = SimpleNamespace(loss_type="simpo", label_smoothing=0.0, aux_loss_enabled=False)
    delattr(trainer, attribute)
    with pytest.raises(ValueError, match=attribute):
        refuse_unsupported_simpo(trainer)


def test_the_real_trl_trainers_carry_every_attribute_the_guards_read():
    """A guard that raises on a missing attribute must not fire on the installed TRL."""
    import inspect

    from trl.experimental.cpo import CPOConfig, CPOTrainer
    from trl.experimental.orpo import ORPOTrainer

    assert "label_smoothing" in inspect.getsource(CPOTrainer)
    assert "aux_loss_enabled" in inspect.getsource(ORPOTrainer)
    assert "label_smoothing" in CPOConfig.__dataclass_fields__
