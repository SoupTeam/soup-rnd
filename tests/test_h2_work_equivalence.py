"""Controls for equal token counts hiding different actual training work."""
import copy
import importlib.util
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).parents[1] / 'benchmarks/harness'
sys.path.insert(0, str(HARNESS))
spec = importlib.util.spec_from_file_location(
    'h2_work_equivalence', HARNESS / 'h2_work_equivalence.py',
)
work = importlib.util.module_from_spec(spec)
spec.loader.exec_module(work)


def records():
    torch = pytest.importorskip('torch')
    inputs = {'input_ids': torch.tensor([[1, 2, 3, 0]]),
              'attention_mask': torch.tensor([[1, 1, 1, 0]]),
              'labels': torch.tensor([[-100, 2, 3, -100]])}
    first = {**work.batch_evidence(inputs), 'loss': 1.0}
    inputs['input_ids'][0, 1] = 4
    inputs['labels'][0, 1] = 4
    second = {**work.batch_evidence(inputs), 'loss': 1.1}
    return {'batches': [first, second], 'gradients': [{}, {}],
            'optimizer_events': [{'completed_batches': 2}]}


def test_equal_work_can_have_different_finite_losses_without_claiming_numerical_agreement():
    left = records()
    right = copy.deepcopy(left)
    right['batches'][0]['loss'] = 10.0
    result = work.audit_pair(left, right, expected_batches=2, accumulation=2)
    assert result['status'] == 'PASS_IDENTICAL_COLLATED_WORK'
    assert result['performance_verdict'] == 'NO_VERDICT'
    assert result['arms'][0]['totals'] == {'processed_positions': 8, 'nonpadding_tokens': 6,
                                         'padding_positions': 2, 'useful_targets': 4}


@pytest.mark.parametrize('change', ['tokens', 'labels', 'mask', 'order', 'missing_batch',
                                   'extra_step', 'early_step', 'missing_backward',
                                   'counts', 'bool_count', 'nan_loss', 'missing_hash'])
def test_different_or_incomplete_work_cannot_pass_with_equal_claimed_step_count(change):
    left = records()
    right = copy.deepcopy(left)
    batch = right['batches'][0]
    if change in ('tokens', 'labels', 'mask'):
        key = {'tokens': 'input_ids', 'labels': 'labels', 'mask': 'attention_mask'}[change]
        batch[key + '_sha256'] = 'f' * 64
    elif change == 'order':
        right['batches'].reverse()
    elif change == 'missing_batch':
        right['batches'].pop()
    elif change == 'extra_step':
        right['optimizer_events'].append({'completed_batches': 2})
    elif change == 'early_step':
        right['optimizer_events'][0]['completed_batches'] = 1
    elif change == 'missing_backward':
        right['gradients'].pop()
    elif change == 'counts':
        batch['counts']['useful_targets'] = 10
    elif change == 'bool_count':
        batch['counts']['padding_positions'] = True
    elif change == 'nan_loss':
        batch['loss'] = float('nan')
    else:
        del batch['labels_sha256']
    assert work.audit_pair(left, right, expected_batches=2, accumulation=2)['status'] == 'FAIL'


def test_actual_prompt_mask_change_preserves_counts_but_fails_pair():
    torch = pytest.importorskip('torch')
    ids = torch.tensor([[1, 2, 3, 4]])
    mask = torch.ones_like(ids)
    first = work.batch_evidence({'input_ids': ids, 'attention_mask': mask,
                                'labels': torch.tensor([[-100, -100, 3, 4]])})
    second = work.batch_evidence({'input_ids': ids, 'attention_mask': mask,
                                 'labels': torch.tensor([[-100, 2, -100, 4]])})
    assert first['counts'] == second['counts']
    left = {'batches': [{**first, 'loss': 1.0}], 'gradients': [{}],
            'optimizer_events': [{'completed_batches': 1}]}
    right = {**left, 'batches': [{**second, 'loss': 1.0}]}
    assert work.audit_pair(left, right, expected_batches=1, accumulation=1)['status'] == 'FAIL'


def test_mismatched_labels_rejected_at_capture():
    torch = pytest.importorskip('torch')
    with pytest.raises(ValueError, match='labels must match'):
        work.batch_evidence({'input_ids': torch.tensor([[1, 2]]),
                             'attention_mask': torch.tensor([[1, 1]]),
                             'labels': torch.tensor([[1, 3]])})


@pytest.mark.parametrize('batches,accumulation', [(3, 2), (0, 1), (2, True)])
def test_partial_or_invalid_accumulation_policy_refused(batches, accumulation):
    with pytest.raises(ValueError, match='complete accumulation'):
        work.validate_work({}, expected_batches=batches, accumulation=accumulation)
