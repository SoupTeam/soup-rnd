"""A fresh replacement cannot relabel or overwrite the preserved paused attempt."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'benchmarks/g6'))
import policy_continuation  # noqa: E402
import policy_recovery as recovery  # noqa: E402
from reload_followup import PHASES  # noqa: E402


def fixture(tmp_path):
    inventory = {'gpu': 'fixture'}
    parent = {'expected_initial_state_sha256': 'initial'}
    phase = {'F1': {'outcome': 'ok', 'gpu_clock': {'outcome': 'ok'}}}
    spec = {'parent': {'sha256': 'parent'}, 'series': {}}
    for arm, code in [('coherence_filter', 0), ('length_curriculum', -15)]:
        root = tmp_path / ('pilot_' + arm + '_001')
        directory = root / arm
        directory.mkdir(parents=True)
        (root / 'prospective_run.json').write_text(json.dumps({
            'arm': arm, 'contract_sha': 'parent',
            'source_baseline_end_to_end_timing_validated': False}))
        (root / 'inventory.json').write_text(json.dumps(inventory))
        (tmp_path / (root.name + '.launcher.json')).write_text(json.dumps({
            'exitcode': code, 'hard_timeout': False, 'bound_s': 4800,
            'contract_sha256': 'parent'}))
        updates = [{'event': 'optimizer_step', 'update': step, 'actual_useful_tokens': 8192,
                    'consumed_useful_tokens': step * 8192} for step in range(1, 9)]
        (directory / 'actual_trace.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in updates))
        names = [*PHASES, 'reload_diagnostics'] if code == 0 else [
            'initial_eval', 'initial_diagnostics', 'train']
        for name in names:
            (directory / (name + '_phase.json')).write_text(json.dumps(phase))
        if code == 0:
            (directory / 'receipt.json').write_text(json.dumps({
                'status': 'ARM_COMPLETED_REVIEW_PENDING', 'initial_state_sha256': 'initial',
                'training': {'actual_useful_tokens': 65536, 'applied_optimizer_updates': 8,
                             'AMP_exercised': True}, 'reload_exact': True,
                'F1_timing_accepted': True}))
        else:
            for name, count in [('initial_eval', 256), ('initial_diagnostics', 24),
                                ('final_eval', 15)]:
                (directory / (name + '.jsonl')).write_text('{}\n' * count)
    refresh(tmp_path, spec)
    return spec, parent, inventory


def refresh(root, spec):
    spec['series']['files_sha256'] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob('*') if path.is_file()}


def test_reuses_exact_worker_and_preserves_original_failure(tmp_path):
    spec, parent, inventory = fixture(tmp_path)
    recovery.verify_series(tmp_path, spec, parent, inventory)
    assert recovery.worker is policy_continuation.worker
    assert not (tmp_path / 'pilot_length_curriculum_001/receipt.json').exists()


@pytest.mark.parametrize('change', ['receipt', 'adapter', 'partial', 'overflow', 'clock',
                                   'supervisor', 'manifest'])
def test_invalid_or_relabelled_evidence_blocks_new_attempt(tmp_path, change):
    spec, parent, inventory = fixture(tmp_path)
    paused = tmp_path / 'pilot_length_curriculum_001/length_curriculum'
    if change == 'receipt':
        (paused.parent / 'receipt.json').write_text('{}')
    elif change == 'adapter':
        (paused / 'adapter.safetensors').write_bytes(b'not-a-checkpoint')
    elif change == 'partial':
        (paused / 'final_eval.jsonl').write_text('{}\n' * 16)
    elif change == 'overflow':
        trace = paused / 'actual_trace.jsonl'
        trace.write_text(trace.read_text().replace('65536', '57344'))
    elif change == 'clock':
        (paused / 'train_phase.json').write_text(json.dumps({
            'F1': {'outcome': 'ok', 'gpu_clock': {'outcome': 'failed'}}}))
    elif change == 'supervisor':
        path = tmp_path / 'pilot_length_curriculum_001.launcher.json'
        value = json.loads(path.read_text())
        value['exitcode'] = 0
        path.write_text(json.dumps(value))
    if change != 'manifest':
        refresh(tmp_path, spec)
    else:
        del spec['series']['files_sha256'][
            'pilot_coherence_filter_001/coherence_filter/receipt.json']
    with pytest.raises(ValueError):
        recovery.verify_series(tmp_path, spec, parent, inventory)
