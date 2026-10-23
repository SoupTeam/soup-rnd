"""Candidate recipes share work and parse through the actual configuration schema."""
import hashlib
import json
from pathlib import Path

import pytest

from soup_cli.config.schema import SoupConfig

ROOT = Path(__file__).parents[1] / 'benchmarks/h2'


@pytest.mark.parametrize('arm', ['baseline', 'liger', 'unsloth'])
def test_candidate_has_only_known_fields_and_validates_without_loading_model(arm):
    value = json.loads((ROOT / 'candidate_configs' / (arm + '.json')).read_text())
    assert set(value) <= set(SoupConfig.model_fields)
    config = SoupConfig.model_validate(value)
    assert set(value['training']) <= set(type(config.training).model_fields)
    assert set(value['data']) <= set(type(config.data).model_fields)
    assert config.training.batch_size == 1
    assert config.training.gradient_accumulation_steps == 4
    assert config.data.chat_template == 'llama3'
    assert config.training.lora.dropout == 0
    assert len(config.training.lora.target_modules) == 7


def test_only_declared_treatment_and_output_paths_differ():
    values = []
    for arm in ['baseline', 'liger', 'unsloth']:
        value = json.loads((ROOT / 'candidate_configs' / (arm + '.json')).read_text())
        del value['output']
        value['backend'] = 'transformers'
        value['training']['use_liger'] = False
        values.append(value)
    assert values[0] == values[1] == values[2]


def test_fixture_and_unresolved_pins_are_honestly_declared():
    raw = (ROOT / 'synthetic_fixture.jsonl').read_bytes()
    plan = json.loads((ROOT / 'candidate_plan.json').read_text())
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == 128 and len({row['instruction'] for row in rows}) == 128
    assert all(row['output'] == str(index + 7) for index, row in enumerate(rows))
    assert hashlib.sha256(raw).hexdigest() == plan['fixture_sha256']
    snapshot = json.loads((ROOT / 'snapshot_preparation_result.json').read_text())
    assert plan['model_revision'] == snapshot['revision']
    assert plan['tokenizer_revision'] == snapshot['revision']
    assert plan['model_file_sha256'] == snapshot['files_sha256']
    assert len(snapshot['files_sha256']) == 11
    assert set(snapshot['files_sha256']) == set(snapshot['files_bytes'])
    assert sum(snapshot['files_bytes'].values()) == snapshot['total_bytes']
    assert snapshot['gpu_execution'] is False
    assert snapshot['performance_verdict'] == 'NO_VERDICT'
    assert plan['oracle_integrated_with_real_trainer'] is True
    assert plan['real_8b_runner_validated'] is False
    assert plan['gpu_execution'] is False and plan['rental_authorized'] is False
