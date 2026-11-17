"""Compare actual collated inputs despite TRL's Liger dataset-column pruning."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'benchmarks/harness'))
import h2_ce_confirmation as runner  # noqa: E402


def fixture_rows():
    return [json.loads(line) for line in
            (runner.SPEC / 'prepared_fixture_candidate.jsonl').read_text().splitlines()]


def trainer_for(rows, *, liger=False):
    pytest.importorskip('torch')
    pytest.importorskip('trl')
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    source = copy.deepcopy(rows)
    for row in source:
        row['assistant_masks'] = [int(value != -100) for value in row['labels']]
    if liger:
        # TRL 0.29.1 SFTTrainer._prepare_dataset selects these columns for Liger.
        source = [{key: row[key] for key in ('input_ids', 'assistant_masks')} for row in source]
    return SimpleNamespace(train_dataset=source, data_collator=DataCollatorForLanguageModeling(
        pad_token_id=128009, completion_only_loss=False, padding_free=False))


@pytest.mark.parametrize('liger', [False, True])
def test_real_trl_collator_preserves_all_frozen_rows(liger):
    rows = fixture_rows()
    trainer = trainer_for(rows, liger=liger)
    if liger:
        assert 'attention_mask' not in trainer.train_dataset[0]
        assert 'labels' not in trainer.train_dataset[0]
    assert runner.validate_prepared_batches(trainer, rows)['rows'] == len(rows)


@pytest.mark.parametrize('mutation', ['mask', 'tokens', 'order', 'missing', 'extra'])
def test_changed_workload_rejected(mutation):
    rows = fixture_rows()
    trainer = trainer_for(rows, liger=True)
    if mutation == 'mask':
        trainer.train_dataset[0]['assistant_masks'][0] = 1
    elif mutation == 'tokens':
        trainer.train_dataset[0]['input_ids'][0] += 1
    elif mutation == 'order':
        trainer.train_dataset[:2] = reversed(trainer.train_dataset[:2])
    elif mutation == 'missing':
        trainer.train_dataset.pop()
    else:
        trainer.train_dataset.append(copy.deepcopy(trainer.train_dataset[0]))
    with pytest.raises(ValueError, match='prepared'):
        runner.validate_prepared_batches(trainer, rows)


@pytest.mark.parametrize('mutation', ['missing_mask', 'float', 'rank', 'padding_free'])
def test_invalid_collator_output_rejected(mutation):
    torch = pytest.importorskip('torch')
    rows = fixture_rows()
    trainer = trainer_for(rows, liger=True)
    original = trainer.data_collator

    def broken(items):
        result = original(items)
        if mutation == 'missing_mask':
            del result['attention_mask']
        elif mutation == 'float':
            result['labels'] = result['labels'].to(torch.float32)
        elif mutation == 'rank':
            result['input_ids'] = result['input_ids'][0]
        else:
            result['position_ids'] = result.pop('attention_mask')
        return result
    trainer.data_collator = broken
    with pytest.raises(ValueError, match='prepared'):
        runner.validate_prepared_batches(trainer, rows)
