"""Real tiny CPU optimizer controls, not an 8B or CUDA performance study."""
import copy
import importlib.util
import sys
from pathlib import Path

import pytest

MODULE = Path(__file__).parents[1] / 'benchmarks/harness/h2_update_oracle.py'
sys.path.insert(0, str(MODULE.parent))
spec = importlib.util.spec_from_file_location('h2_update_oracle', MODULE)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def train_case(mode):
    torch = pytest.importorskip('torch')
    torch.manual_seed(42)

    class TinyAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Parameter(torch.eye(3), requires_grad=False)
            self.lora_a = torch.nn.Parameter(torch.ones(3, 1) * 0.1)
            self.lora_b = torch.nn.Parameter(torch.zeros(1, 3))
            self.register_buffer('scale', torch.ones(1))

        def forward(self, value):
            return value @ (self.base + self.lora_a @ self.lora_b) * self.scale

    model = TinyAdapter()
    names = ['lora_a', 'lora_b']
    before = oracle.snapshot(model, names)
    optimizer = torch.optim.SGD([v for v in model.parameters() if v.requires_grad], lr=0.1)
    loss = (model(torch.ones(2, 3)) - 2).square().mean()
    loss.backward()
    row = oracle.gradients(model, names)
    if mode != 'skip_step':
        optimizer.step()
    with torch.no_grad():
        if mode == 'base_changed':
            model.base.add_(1)
        if mode == 'buffer_changed':
            model.scale.add_(1)
        if mode == 'nonfinite_adapter':
            model.lora_b.fill_(float('nan'))
    after = oracle.snapshot(model, names)
    return before, after, row, loss.item()


def test_real_update_passes_even_when_initial_a_gradient_is_zero():
    before, after, row, loss = train_case('normal')
    assert row['nonzero_names'] == ['lora_b']
    result = oracle.audit_update(before, after, [row], 1, [loss])
    assert result['status'] == 'PASS_UPDATE_EVIDENCE'
    assert result['performance_verdict'] == 'NO_VERDICT'


@pytest.mark.parametrize('mode', [
    'skip_step', 'base_changed', 'buffer_changed', 'nonfinite_adapter',
])
def test_negative_controls_rejected_even_with_claimed_optimizer_step(mode):
    before, after, row, loss = train_case(mode)
    assert oracle.audit_update(before, after, [row], 1, [loss])['status'] == 'FAIL'


@pytest.mark.parametrize('problem', ['nan_loss', 'nan_grad', 'no_grad', 'no_steps', 'shape'])
def test_invalid_evidence_rejected(problem):
    before, after, row, loss = train_case('normal')
    steps = 1
    if problem == 'nan_loss':
        loss = float('nan')
    elif problem == 'nan_grad':
        row['finite'] = False
    elif problem == 'no_grad':
        row['nonzero_names'] = []
    elif problem == 'no_steps':
        steps = True
    else:
        after = copy.deepcopy(after)
        after['trainable']['lora_b']['shape'] = [999]
    assert oracle.audit_update(before, after, [row], steps, [loss])['status'] == 'FAIL'


def test_causal_targets_padding_and_prompt_masks_count_separately():
    torch = pytest.importorskip('torch')
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    labels = torch.tensor([[-100, -100, 7, -100], [4, 5, -100, -100]])
    assert oracle.token_counts(mask, labels) == {
        'processed_positions': 8, 'nonpadding_tokens': 5,
        'padding_positions': 3, 'useful_targets': 2,
    }
    labels[0, 3] = 9
    with pytest.raises(ValueError, match='Padding'):
        oracle.token_counts(mask, labels)


def test_manifest_cannot_hide_an_unexpected_trainable_base():
    torch = pytest.importorskip('torch')
    with pytest.raises(ValueError, match='Trainable names'):
        oracle.snapshot(torch.nn.Linear(2, 2), ['weight'])


def test_hash_tracks_dtype_shape_and_bytes():
    torch = pytest.importorskip('torch')
    value = torch.ones(2, 2)
    assert oracle.tensor_sha(value) != oracle.tensor_sha(value.double())
    assert oracle.tensor_sha(value) != oracle.tensor_sha(value.reshape(4))
    assert oracle.tensor_sha(value) != oracle.tensor_sha(value + 1)
