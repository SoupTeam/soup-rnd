"""Real HF constructor lifecycle; CPU patch spies are not Liger kernel validation."""
import ast
import inspect
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'benchmarks/harness'))
import h2_operator_patch as patches  # noqa: E402


def control_for(arm):
    control = patches.PatchController.__new__(patches.PatchController)
    control.arm, control.flags = arm, patches.patch_flags(arm)
    control.counts = {'class': 0, 'instance': 0}
    control.capture_prepared = False
    calls = []
    control.patch = SimpleNamespace(apply_liger_kernel_to_llama=lambda **kw: calls.append(kw))
    from transformers.trainer_utils import unwrap_peft_model

    control.unwrap = unwrap_peft_model
    return control, calls


@pytest.mark.parametrize('arm', list(patches.ARMS))
def test_real_trainer_setup_requires_explicit_instance_stage(arm, monkeypatch, tmp_path):
    import torch
    import transformers.trainer as hf_trainer
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments

    control, calls = control_for(arm)
    monkeypatch.setattr(hf_trainer, 'apply_liger_kernel', control.instance_patch)
    monkeypatch.setattr(Trainer, 'train', lambda *a, **k: pytest.fail('train called'))
    monkeypatch.setattr(Trainer, 'create_optimizer', lambda *a: pytest.fail('optimizer created'))
    enabled = arm != 'baseline'
    if enabled:
        control.class_patch('tiny-local-llama')
    model = get_peft_model(LlamaForCausalLM(LlamaConfig(
        vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2,
    )), LoraConfig(r=2, target_modules=['q_proj', 'v_proj']))
    trainer = Trainer(model=model, args=TrainingArguments(
        output_dir=str(tmp_path), use_cpu=True, report_to='none',
        use_liger_kernel=enabled, liger_kernel_config=control.flags,
    ))
    # Reproduces the old assumption against the real HF constructor, not a fake Trainer.
    assert control.counts == {'class': int(enabled), 'instance': 0}
    if enabled:
        with pytest.raises(ValueError, match='patch point'):
            patches.validate_counts(arm, control.counts)
    receipt = control.prepare_capture(trainer, model)
    assert receipt['patch_calls'] == {'class': int(enabled), 'instance': int(enabled)}
    assert len(calls) == 2 * int(enabled)
    if enabled:
        assert calls == [control.flags, dict(control.flags, model=control.unwrap(model))]
        assert calls[-1]['model'] is not model  # Actual PEFT unwrapping.
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    ids = torch.tensor([[1, 2, 3, 4]])
    model(input_ids=ids, labels=ids).loss.backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad)
               for p in model.parameters() if p.requires_grad)
    assert all(torch.equal(initial[name], p) for name, p in model.named_parameters())
    assert trainer.optimizer is None and trainer.lr_scheduler is None
    assert trainer.state.global_step == 0 and not trainer.is_in_train
    with pytest.raises(ValueError, match='already prepared'):
        control.prepare_capture(trainer, model)
    with pytest.raises(ValueError, match='instance patch'):
        control.instance_patch(model, control.flags)


def test_installed_hf_patch_is_in_train_not_constructor():
    from transformers import Trainer

    def count_calls(method):
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        return sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == 'apply_liger_kernel' for node in ast.walk(tree))

    assert count_calls(Trainer.__init__) == 0
    assert count_calls(Trainer.train) == 1


@pytest.mark.parametrize('fault', [
    'model', 'model_wrapped', 'disabled', 'extra_config', 'optimizer', 'scheduler',
    'global_step', 'training', 'missing_class', 'duplicate_instance',
])
def test_invalid_setup_rejected_before_any_instance_mutation(fault):
    control, calls = control_for('ce_only')
    control.counts['class'] = 1
    model = object()
    trainer = SimpleNamespace(
        model=model, model_wrapped=model, optimizer=None, lr_scheduler=None,
        state=SimpleNamespace(global_step=0), is_in_train=False,
        args=SimpleNamespace(use_liger_kernel=True, liger_kernel_config=None),
    )
    if fault in {'model', 'model_wrapped', 'optimizer'}:
        setattr(trainer, fault, object())
    elif fault == 'scheduler':
        trainer.lr_scheduler = object()
    elif fault == 'disabled':
        trainer.args.use_liger_kernel = False
    elif fault == 'extra_config':
        trainer.args.liger_kernel_config = {'rope': True}
    elif fault == 'global_step':
        trainer.state.global_step = 1
    elif fault == 'training':
        trainer.is_in_train = True
    elif fault == 'missing_class':
        control.counts['class'] = 0
    else:
        control.counts['instance'] = 1
    with pytest.raises(ValueError, match='Unexpected'):
        control.prepare_capture(trainer, model)
    assert calls == []
    with pytest.raises(ValueError, match='already prepared'):
        control.prepare_capture(trainer, model)


def test_failed_instance_patch_is_not_retried():
    control, _ = control_for('ce_only')
    control.counts['class'] = 1
    control.unwrap = lambda model: model
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        raise RuntimeError('partial patch')

    control.patch.apply_liger_kernel_to_llama = fail
    model = object()
    trainer = SimpleNamespace(
        model=model, model_wrapped=model, optimizer=None, lr_scheduler=None,
        state=SimpleNamespace(global_step=0), is_in_train=False,
        args=SimpleNamespace(use_liger_kernel=True, liger_kernel_config=None),
    )
    with pytest.raises(RuntimeError, match='partial patch'):
        control.prepare_capture(trainer, model)
    with pytest.raises(ValueError, match='already prepared'):
        control.prepare_capture(trainer, model)
    assert len(calls) == 1
