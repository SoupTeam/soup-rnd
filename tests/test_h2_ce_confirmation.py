"""No GPU launch in unit tests: authority and phase-order failures must fail closed."""
import copy
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'benchmarks/harness'))
import h2_ce_confirmation as runner  # noqa: E402
from h2_gpu_correctness import SOURCES, sha_file  # noqa: E402


def authority(tmp_path):
    study = json.loads((runner.SPEC / 'operator_study.json').read_text())
    return {'schema': 'h2-ce-authority-v1', 'execution_authorized': True,
            'license_and_access_confirmed': True, 'instance_id': 1,
            'gpu_uuid': 'GPU-' + '1' * 8 + '-1111-1111-1111-' + '1' * 12,
            'boot_id': '11111111-1111-1111-1111-111111111111',
            'budget_cap_usd': 5, 'prior_spend_usd': 1, 'hourly_rate_usd': 1,
            'cleanup_traffic_reserve_usd': 1, 'max_seconds': 600,
            'series_started_unix_s': time.time(), 'deadline_unix_s': time.time() + 2400,
            'model_revision': study['model_revision'], 'versions': study['versions'],
            'model_files_sha256': study['model_files_sha256'],
            'numerical_policy_sha256': study['numerical_policy_sha256'],
            'source_commit': 'a' * 40, 'evidence_root': str(tmp_path / 'run'),
            'runner_sha256': {n: sha_file(runner.HARNESS / n) for n in SOURCES},
            'artifacts_sha256': {n: sha_file(runner.REPO / n) for n in runner.ARTIFACTS}}


def test_contract(tmp_path):
    data = authority(tmp_path)
    # Use the exact same start to avoid a test adding fractional time to the 2400s bound.
    data['deadline_unix_s'] = data['series_started_unix_s'] + 2400
    runner.validate_contract(data)


@pytest.mark.parametrize('field,value', [
    ('max_seconds', 601), ('budget_cap_usd', 6), ('prior_spend_usd', 4.5),
    ('model_revision', 'b' * 40), ('execution_authorized', False),
    ('deadline_unix_s', float('nan')), ('series_started_unix_s', float('nan')),
    ('artifacts_sha256', {}), ('versions', {}),
])
def test_invalid_authority(tmp_path, field, value):
    data = authority(tmp_path)
    data['deadline_unix_s'] = data['series_started_unix_s'] + 2400
    data[field] = value
    with pytest.raises(ValueError):
        runner.validate_contract(data)


@pytest.mark.parametrize('failure', ['child', 'probe_review', 'training_review', None])
def test_no_next_phase_after_failure(tmp_path, monkeypatch, failure):
    data = authority(tmp_path)
    data['deadline_unix_s'] = data['series_started_unix_s'] + 2400
    launched = []
    monkeypatch.setattr(runner, 'validate_contract', lambda value: {})

    def child(argv, out, **kwargs):
        phase = argv[argv.index('--phase') + 1]
        launched.append(phase)
        target = Path(argv[argv.index('--out') + 1])
        target.mkdir(parents=True)
        (target / 'evidence.json').write_text(json.dumps({'status': 'COMPLETE_CE_CAPTURE'}))
        return {'status': 'CHILD_EXITED', 'group_cleanup_confirmed': True,
                'child_status': {'returncode': 2 if failure == 'child' else 0}}

    def review(root, *, training):
        if failure == ('training_review' if training else 'probe_review'):
            raise ValueError('Injected independent review failure')
        return {'status': 'PASS'}
    monkeypatch.setattr(runner, 'bounded_process', child)
    monkeypatch.setattr(runner, 'review_pair', review)
    result = runner.run_series(copy.deepcopy(data), Path(data['evidence_root']))
    assert launched == list(runner.PHASES[:1 if failure == 'child'
                                        else 2 if failure == 'probe_review' else 4])
    assert result['status'] == ('COMPLETE_CE_NUMERICAL_SERIES' if failure is None else 'BLOCKED')


@pytest.mark.parametrize('arm', ['baseline', 'ce_only'])
def test_child_uses_actual_patch_receipt_schema(tmp_path, arm):
    from h2_operator_patch import patch_flags

    phase = arm + '_training'
    (tmp_path / 'profile').mkdir()
    kernel = 'liger_cross_entropy_kernel' if arm == 'ce_only' else 'ordinary_gemm'
    (tmp_path / 'profile/kernel_trace.json').write_text(json.dumps(
        {'traceEvents': [{'cat': 'kernel', 'name': kernel}]}))
    (tmp_path / 'evidence.json').write_text(json.dumps({
        'phase': phase, 'status': 'COMPLETE_CE_CAPTURE',
        'patch_after': {'status': 'PASS_EXPLICIT_PATCH_BINDINGS', 'arm': arm,
                        'flags': patch_flags(arm),
                        'patch_calls': {'class': int(arm != 'baseline'),
                                        'instance': int(arm != 'baseline')}}}))
    runner.tree_seal(tmp_path)
    assert runner.read_child(tmp_path, phase)['status'] == 'COMPLETE_CE_CAPTURE'
