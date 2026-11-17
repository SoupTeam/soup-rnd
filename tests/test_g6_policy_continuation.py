"""Continuation requires an independently preserved verification-only success."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'benchmarks/g6'))
import policy_continuation as continuation  # noqa: E402


def fixture(tmp_path):
    ref = tmp_path / 'reference'
    old = ref / 'pilot_baseline_001/baseline'
    old.mkdir(parents=True)
    outputs = ''.join(json.dumps({'id': str(index)}) + '\n' for index in range(24))
    (old / 'final_diagnostics.jsonl').write_text(outputs)
    (old / 'reload_diagnostics.jsonl').write_text(outputs.splitlines()[0] + '\n')
    root = tmp_path / 'verification'
    run = root / 'verification_001'
    run.mkdir(parents=True)
    (run / 'reload_diagnostics.jsonl').write_text(outputs)
    receipt = {'status': 'VERIFIED_RELOAD_DIAGNOSTICS_ONLY', 'cases': 24,
               'contract_sha256': 'verification-sha', 'launch_config_sha256': 'private-sha',
               'new_training': False, 'optimizer_resume': False,
               'original_attempt_status': 'USER_PAUSED_INCOMPLETE',
               'original_end_to_end_timing_validated': False,
               'remaining_pilots_unlocked': False, 'inventory': {'gpu': 'example'}}
    (run / 'receipt.json').write_text(json.dumps(receipt))
    (run / 'resumed_reload_diagnostics_phase.json').write_text(json.dumps({
        'F1': {'outcome': 'ok', 'gpu_clock': {'outcome': 'ok'}}}))
    (root / 'verification_001.launcher.json').write_text(json.dumps({
        'exitcode': 0, 'hard_timeout': False, 'contract_sha256': 'verification-sha',
        'bound_s': 2400}))
    spec = {'verification': {'sha256': 'verification-sha', 'launch_config_sha256': 'private-sha'}}
    return ref, root, run, spec


def refresh(root, spec):
    spec['verification']['files_sha256'] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob('*') if path.is_file()}


def test_linked_proof_does_not_relabel_the_interrupted_attempt(tmp_path):
    ref, root, _, spec = fixture(tmp_path)
    refresh(root, spec)
    value = continuation.linked_verification(ref, root, spec)
    assert value == {'gpu': 'example'}
    assert not (ref / 'pilot_baseline_001/receipt.json').exists()


@pytest.mark.parametrize('field,value', [('cases', 1), ('new_training', True),
                                        ('original_attempt_status', 'COMPLETED'),
                                        ('original_end_to_end_timing_validated', True),
                                        ('remaining_pilots_unlocked', True)])
def test_forged_scope_is_rejected_even_with_a_matching_manifest(tmp_path, field, value):
    ref, root, run, spec = fixture(tmp_path)
    receipt = json.loads((run / 'receipt.json').read_text())
    receipt[field] = value
    (run / 'receipt.json').write_text(json.dumps(receipt))
    refresh(root, spec)
    with pytest.raises(ValueError):
        continuation.linked_verification(ref, root, spec)


def test_partial_or_changed_verification_cannot_unlock_training(tmp_path):
    ref, root, run, spec = fixture(tmp_path)
    refresh(root, spec)
    (run / 'reload_diagnostics.jsonl').write_text('{"id":"0"}\n')
    with pytest.raises(ValueError):
        continuation.linked_verification(ref, root, spec)
    refresh(root, spec)
    with pytest.raises(ValueError):
        continuation.linked_verification(ref, root, spec)


def test_missing_manifest_coverage_rejects_unpinned_proof(tmp_path):
    ref, root, _, spec = fixture(tmp_path)
    refresh(root, spec)
    del spec['verification']['files_sha256']['verification_001/receipt.json']
    with pytest.raises(ValueError, match='omits required'):
        continuation.linked_verification(ref, root, spec)


@pytest.mark.parametrize('clock', ['failed', None])
def test_rejected_or_missing_clocks_block_continuation(tmp_path, clock):
    ref, root, run, spec = fixture(tmp_path)
    (run / 'resumed_reload_diagnostics_phase.json').write_text(json.dumps({
        'F1': {'outcome': 'ok', 'gpu_clock': {'outcome': clock}}}))
    refresh(root, spec)
    with pytest.raises(ValueError, match='clocks rejected'):
        continuation.linked_verification(ref, root, spec)
