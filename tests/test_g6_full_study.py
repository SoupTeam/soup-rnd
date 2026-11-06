"""Full-study admission and completion refuse weaker pilot or partial evidence."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'benchmarks/g6'))
import full_study as full  # noqa: E402
import run  # noqa: E402


def authority():
    launch = {'account': 'fixture', 'instance_id': 1, 'gpu_uuid': 'GPU-fixture',
              'credit_reserve_waived_by_user': True}
    receipt = {**launch, 'mode': 'full', 'arm': 'baseline', 'contract_sha256': 'science',
               'launch_config_sha256': 'private', 'authorized_for_this_stage': True}
    return launch, receipt


def test_exact_engine_reuse_and_explicit_financial_waiver():
    assert full.execute_arm is run.execute_arm
    launch, receipt = authority()
    full.admit(receipt, launch, 'science', 'private', 'baseline')
    del launch['credit_reserve_waived_by_user']
    with pytest.raises(ValueError, match='waiver'):
        full.admit(receipt, launch, 'science', 'private', 'baseline')


@pytest.mark.parametrize('field,value', [('mode', 'pilot'), ('arm', 'coherence_filter'),
                                        ('instance_id', 2), ('contract_sha256', 'changed'),
                                        ('credit_reserve_waived_by_user', False),
                                        ('authorized_for_this_stage', False)])
def test_waiver_does_not_waive_science_or_identity(field, value):
    launch, receipt = authority()
    receipt[field] = value
    with pytest.raises(ValueError):
        full.admit(receipt, launch, 'science', 'private', 'baseline')


def complete_fixture(tmp_path):
    root = tmp_path / 'full_baseline_001'
    directory = root / 'baseline'
    directory.mkdir(parents=True)
    inventory = {'gpu': 'fixture'}
    value = {'status': 'ARM_COMPLETED_REVIEW_PENDING',
             'training': {'actual_useful_tokens': 5_000_000, 'applied_optimizer_updates': 611,
                          'AMP_exercised': True}, 'reload_exact': True,
             'frozen_base_unchanged': True, 'F1_timing_accepted': True,
             'initial_eval': {'mean_ce': 1.0, 'rows': 256, 'useful_targets': 322332},
             'final_eval': {'mean_ce': 1.0, 'rows': 256, 'useful_targets': 322332},
             'raw_reload_eval': {'mean_ce': 1.0, 'rows': 256, 'useful_targets': 322332},
             'peft_reload_eval': {'mean_ce': 1.0, 'rows': 256, 'useful_targets': 322332}}
    (directory / 'receipt.json').write_text(json.dumps(value))
    (root / 'receipt.json').write_text('{}')
    (root / 'prospective_run.json').write_text(json.dumps({
        'arm': 'baseline', 'mode': 'full', 'contract_sha': 'science'}))
    (root / 'inventory.json').write_text(json.dumps(inventory))
    Path(str(root) + '.launcher.json').write_text(json.dumps({
        'exitcode': 0, 'hard_timeout': False, 'bound_s': full.BOUND,
        'contract_sha256': 'science'}))
    for name in full.PHASES:
        (directory / (name + '_phase.json')).write_text(json.dumps({
            'F1': {'outcome': 'ok', 'gpu_clock': {'outcome': 'ok'}}}))
    for name in ['initial_diagnostics', 'final_diagnostics', 'reload_diagnostics']:
        (directory / (name + '.jsonl')).write_text('{}\n' * 24)
    for name in ['initial_eval', 'final_eval', 'raw_reload_eval', 'peft_reload_eval']:
        rows = [{'tokens': {'useful_loss_tokens': count}, 'sum_nll': float(count)}
                for count in [1259] * 255 + [1287]]
        (directory / (name + '.jsonl')).write_text(''.join(json.dumps(row) + '\n' for row in rows))
    steps = [{'event': 'optimizer_step', 'update': index,
              'actual_useful_tokens': 8192 if index < 611 else 2880,
              'consumed_useful_tokens': min(index * 8192, 5_000_000)}
             for index in range(1, 612)]
    (directory / 'actual_trace.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in steps))
    refresh(root)
    return root, inventory


def refresh(root):
    (root / 'files_sha256.json').write_text(json.dumps({
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob('*') if path.is_file() and path.name != 'files_sha256.json'}))


@pytest.mark.parametrize('change', ['pilot', 'overflow', 'clock', 'reload',
                                   'manifest', 'last_step', 'eval', 'mode'])
def test_partial_or_pilot_data_cannot_pass_full_review(tmp_path, change):
    root, inventory = complete_fixture(tmp_path)
    full.completed(root, 'baseline', 'science', inventory, 5_000_000, 611, full.BOUND)
    if change == 'clock':
        (root / 'baseline/train_phase.json').write_text(json.dumps({
            'F1': {'outcome': 'ok', 'gpu_clock': {'outcome': 'failed'}}}))
    elif change == 'reload':
        (root / 'baseline/reload_diagnostics.jsonl').write_text('{}\n' * 23)
    elif change == 'manifest':
        entry = root / 'files_sha256.json'
        manifest = json.loads(entry.read_text())
        del manifest['baseline/receipt.json']
        entry.write_text(json.dumps(manifest))
    elif change == 'last_step':
        entry = root / 'baseline/actual_trace.jsonl'
        entry.write_text(entry.read_text().replace('2880', '8192'))
    elif change == 'eval':
        entry = root / 'baseline/final_eval.jsonl'
        rows = [json.loads(line) for line in entry.read_text().splitlines()]
        rows[0]['sum_nll'] += 1
        entry.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    elif change == 'mode':
        entry = root / 'prospective_run.json'
        value = json.loads(entry.read_text())
        value['mode'] = 'pilot'
        entry.write_text(json.dumps(value))
    else:
        entry = root / 'baseline/receipt.json'
        value = json.loads(entry.read_text())
        value['training']['actual_useful_tokens'] = 65536 if change == 'pilot' else 4_999_999
        entry.write_text(json.dumps(value))
    if change != 'manifest':
        refresh(root)
    with pytest.raises(ValueError):
        full.completed(root, 'baseline', 'science', inventory, 5_000_000, 611, full.BOUND)
