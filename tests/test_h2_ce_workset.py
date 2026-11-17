"""Workset construction must preserve shifted targets, boundaries and padding."""
import pytest

from benchmarks.harness.h2_ce_workset import collate, repeat_row


def test_repeat_and_shifted_target_mask():
    row = {'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1], 'labels': [-100, 2, 3]}
    assert repeat_row(row, 2)['labels'] == [-100, 2, 3, -100, 2, 3]
    batch = collate([repeat_row(row, 2), row], pad_id=3)
    assert batch['input_ids'][1] == [1, 2, 3, 3, 3, 3]
    assert batch['attention_mask'][1] == [1, 1, 1, 0, 0, 0]
    assert batch['labels'][1] == [-100, 2, 3, -100, -100, -100]
    assert sum(x != -100 for labels in batch['labels'] for x in labels[1:]) == 6


@pytest.mark.parametrize('row', [
    {'input_ids': [1], 'attention_mask': [1, 1], 'labels': [-100]},
    {'input_ids': [1, 2], 'attention_mask': [1, 0], 'labels': [-100, 2]},
    {'input_ids': [1, 2], 'attention_mask': [1, 1], 'labels': [1, 2]},
    {'input_ids': [1, 2], 'attention_mask': [1, 1], 'labels': [-100, 3]},
])
def test_invalid_source_refused(row):
    with pytest.raises(ValueError):
        repeat_row(row, 1)


@pytest.mark.parametrize('count', [0, -1, True, 1.5, 9])
def test_invalid_repetition_refused(count):
    row = {'input_ids': [1, 2], 'attention_mask': [1, 1], 'labels': [-100, 2]}
    with pytest.raises(ValueError):
        repeat_row(row, count)


def test_frozen_workset_is_reproducible_and_work_counts_match():
    import json
    from pathlib import Path

    from benchmarks.harness.h2_ce_workset import build_workset

    root = Path(__file__).parents[1] / 'benchmarks/h2'
    source = [json.loads(line) for line in
              (root / 'prepared_fixture_candidate.jsonl').read_text().splitlines()]
    actual = build_workset(source)
    assert actual == json.loads((root / 'ce_confirmation_workset.json').read_text())
    assert len(set(actual['training_row_indices'])) == 16
    batches = actual['training_batches']
    assert sum(len(row) for b in batches for row in b['input_ids']) == 384
    assert sum(x != -100 for b in batches for row in b['labels'] for x in row[1:]) == 96
    assert sum(x == 0 for b in actual['probes'] for row in b['attention_mask'] for x in row) == 168
