"""Prospective pilot authority, immutable inputs and sequential no-retry claims."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'benchmarks/g6'))
import policy_pilot_support as support  # noqa: E402


def fixture():
    spec = {'launch_contract_sha256': 'science', 'arm_order': ['baseline', 'filtered'],
            'limits': {'pilot_total_s': 4800, 'cuda-gate_total_s': 600},
            'supervisor_grace_s': 40}
    launch = {'account': 'example-project', 'instance_id': 321, 'gpu_uuid': 'GPU-example',
              'account_floor_usd': 7.5, 'stop_only_this_instance_after_preservation': True,
              'contract_sha256': 'science', 'runs_root': '/synthetic/runs'}
    admission = {'account': 'example-project', 'instance_id': 321, 'gpu_uuid': 'GPU-example',
                 'mode': 'pilot', 'arm': 'baseline', 'contract_sha256': 'science',
                 'launch_config_sha256': 'private-sha', 'authorized_for_this_stage': True,
                 'observed_unix_s': 1000, 'balance_usd': 12,
                 'total_account_rate_usd_h': .6}
    return spec, launch, admission


@pytest.mark.parametrize('key,value', [
    ('mode', 'full'), ('account', 'wrong'), ('instance_id', 123), ('gpu_uuid', 'GPU-wrong'),
    ('authorized_for_this_stage', False), ('observed_unix_s', 699),
    ('observed_unix_s', 1001), ('balance_usd', float('nan')), ('balance_usd', 7.6),
    ('total_account_rate_usd_h', 0), ('contract_sha256', 'wrong'),
    ('launch_config_sha256', 'wrong'), ('arm', 'filtered'),
])
def test_admission_is_fresh_host_bound_and_reserves_all_remaining_arms(key, value):
    spec, launch, admission = fixture()
    support.admit(admission, spec, launch, 'private-sha', 'pilot', 'baseline', now=1000)
    admission[key] = value
    with pytest.raises(ValueError):
        support.admit(admission, spec, launch, 'private-sha', 'pilot', 'baseline', now=1000)


def test_budget_must_cover_remaining_series_not_just_next_arm():
    spec, launch, admission = fixture()
    admission['balance_usd'] = 9.5  # Fits one arm, does not fit the remaining pair plus reserve.
    with pytest.raises(ValueError, match='remaining'):
        support.admit(admission, spec, launch, 'private-sha', 'pilot', 'baseline', now=1000)
    admission['arm'] = 'filtered'
    support.admit(admission, spec, launch, 'private-sha', 'pilot', 'filtered', now=1000)
    with pytest.raises(ValueError):
        support.admit(admission, spec, launch, 'private-sha', 'full', 'filtered', now=1000)


def test_private_authority_cannot_be_changed_under_existing_hash(tmp_path):
    _, launch, _ = fixture()
    path = tmp_path / 'private.json'
    path.write_text(json.dumps(launch))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert support.load_authority(path, sha, 'science') == launch
    launch['account_floor_usd'] = 0
    path.write_text(json.dumps(launch))
    with pytest.raises(ValueError):
        support.load_authority(path, sha, 'science')


def test_prior_failure_missing_evidence_and_tampering_block_following_arm(tmp_path):
    spec, _, _ = fixture()
    spec['expected_initial_state_sha256'] = 'initial'
    root = tmp_path / 'claims'
    root.mkdir()
    out = tmp_path / 'baseline'
    (out / 'baseline').mkdir(parents=True)
    (root / 'pilot_baseline.json').write_text(json.dumps(
        {'out': str(out), 'launch_config_sha256': 'private-sha'}))
    with pytest.raises((ValueError, FileNotFoundError)):
        support.preceding(root, spec, 'filtered', 'private-sha')
    receipt = out / 'baseline/receipt.json'
    receipt.write_text(json.dumps({'status': 'ARM_COMPLETED_REVIEW_PENDING',
                                  'F1_timing_accepted': True,
                                  'initial_state_sha256': 'initial'}))
    (out / 'files_sha256.json').write_text(json.dumps({
        'baseline/receipt.json': hashlib.sha256(receipt.read_bytes()).hexdigest()}))
    launcher = Path(str(out) + '.launcher.json')
    launcher.write_text(json.dumps({'exitcode': 1, 'hard_timeout': False,
                                   'contract_sha256': 'science', 'bound_s': 4800}))
    with pytest.raises(ValueError, match='supervisor'):
        support.preceding(root, spec, 'filtered', 'private-sha')
    saved = json.loads(launcher.read_text())
    saved['exitcode'] = 0
    launcher.write_text(json.dumps(saved))
    assert support.preceding(root, spec, 'filtered', 'private-sha') == 'initial'
    receipt.write_text('{}')
    with pytest.raises(ValueError):
        support.preceding(root, spec, 'filtered', 'private-sha')
    (root / 'pilot_filtered.json').write_text('{}')
    with pytest.raises(ValueError, match='claim'):
        support.preceding(root, spec, 'filtered', 'private-sha')


def test_private_series_identity_cannot_switch_between_arms(tmp_path):
    spec, _, _ = fixture()
    spec['expected_initial_state_sha256'] = 'initial'
    (tmp_path / 'pilot_baseline.json').write_text(json.dumps(
        {'out': str(tmp_path), 'launch_config_sha256': 'other-host'}))
    with pytest.raises(ValueError, match='configuration'):
        support.preceding(tmp_path, spec, 'filtered', 'private-sha')


def test_new_entry_reuses_the_frozen_training_engine():
    import policy_pilot
    import run

    assert policy_pilot.execute_arm is run.execute_arm


def test_numerical_gate_math_and_tolerances_are_unchanged():
    import ast
    import copy

    base = Path(__file__).parents[1] / 'benchmarks/g6'
    bodies = []
    for name in ['cuda_gate.py', 'policy_cuda_gate.py']:
        tree = ast.parse((base / name).read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'main')
        block = copy.deepcopy(next(node for node in main.body if isinstance(node, ast.Try)))
        for node in ast.walk(block):
            if isinstance(node, ast.Dict):
                pairs = [(key, value) for key, value in zip(node.keys, node.values, strict=True)
                         if not (isinstance(key, ast.Constant)
                                 and key.value == 'launch_config_sha256')]
                node.keys = [key for key, _ in pairs]
                node.values = [value for _, value in pairs]
        bodies.append(ast.dump(block))
    assert bodies[0] == bodies[1]


def test_full_training_is_not_a_cli_mode():
    import subprocess

    path = Path(__file__).parents[1] / 'benchmarks/g6/policy_pilot.py'
    result = subprocess.run([sys.executable, str(path), 'full'], capture_output=True,
                            text=True, timeout=10)
    assert result.returncode != 0 and 'invalid choice' in result.stderr


def test_supervisor_kills_the_entire_group_and_never_retries(tmp_path, monkeypatch):
    import signal
    import subprocess

    import launch_policy_pilot as launcher

    out = tmp_path / 'run'
    calls = []
    killed = []

    class Child:
        pid = 987654
        returncode = -15
        waits = 0

        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired('synthetic child', timeout)
            return self.returncode

    def start(*args, **kwargs):
        calls.append((args, kwargs))
        return Child()

    monkeypatch.setattr(launcher, 'load_contract', lambda *args: {})
    monkeypatch.setattr(launcher, 'bound', lambda *args: 1)
    monkeypatch.setattr(launcher.subprocess, 'Popen', start)
    monkeypatch.setattr(launcher.os, 'killpg', lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(sys, 'argv', ['launch', 'pilot', '--spec', 'science', '--spec-sha', 'sha',
                                    '--artifact-root', str(tmp_path), '--out', str(out)])
    with pytest.raises(SystemExit):
        launcher.main()
    assert len(calls) == 1 and calls[0][1]['start_new_session'] is True
    assert killed == [(Child.pid, signal.SIGTERM)]
    result = json.loads(Path(str(out) + '.launcher.json').read_text())
    assert result['hard_timeout'] is True and result['no_retry'] is True


def test_contract_checks_private_artifact_bytes_and_containment(tmp_path, monkeypatch):
    monkeypatch.setattr(support, 'REPO', tmp_path)
    for name in ['lock', 'suite', 'plan', 'artifact']:
        (tmp_path / name).write_text(name)
    def sha(name):
        return hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    spec = {'scope': 'fresh-matched-policy-pilot-only', 'arm_order': support.ARMS,
            'limits': {'pilot_train_targets': 65536}, 'recipe': {'targets_per_update': 8192},
            'data': {'budget_per_arm': 5_000_000}, 'source_sha256': {},
            'artifact_sha256': {'artifact': sha('artifact')},
            'runtime_wheels': {'path': 'lock', 'sha256': sha('lock')},
            'suite': {'path': 'suite', 'sha256': sha('suite')},
            'input_plan': {'path': 'plan', 'sha256': sha('plan')}}
    path = tmp_path / 'science.json'
    path.write_text(json.dumps(spec))
    science_sha = sha('science.json')
    support.load_contract(path, science_sha, tmp_path)
    (tmp_path / 'artifact').write_text('changed')
    with pytest.raises(ValueError, match='artifact differs'):
        support.load_contract(path, science_sha, tmp_path)
    spec['artifact_sha256'] = {'../outside': 'invalid'}
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='escapes'):
        support.load_contract(path, sha('science.json'), tmp_path)


def test_failed_gate_supervisor_blocks_before_model_or_training(tmp_path, monkeypatch):
    from argparse import Namespace

    import policy_pilot

    spec, launch, _ = fixture()
    spec['runtime_target'] = {'synthetic': 'versions'}
    gate_dir = tmp_path / 'gate'
    gate_dir.mkdir()
    receipt = gate_dir / 'receipt.json'
    receipt.write_text(json.dumps({
        'status': 'PASS_TINY_CUDA_NUMERICAL_GATE', 'contract_sha256': 'science',
        'launch_config_sha256': 'private-sha', 'runtime': spec['runtime_target'],
        'inventory': {'synthetic': 'inventory'},
    }))
    Path(str(gate_dir) + '.launcher.json').write_text(json.dumps({
        'exitcode': 1, 'hard_timeout': False, 'contract_sha256': 'science', 'bound_s': 600,
    }))
    admission = tmp_path / 'admission.json'
    admission.write_text('{}')
    monkeypatch.setattr(policy_pilot, 'load_authority', lambda *args: launch)
    monkeypatch.setattr(policy_pilot, 'admit', lambda *args: None)
    monkeypatch.setattr(policy_pilot, 'verify_runtime', lambda *args: None)
    monkeypatch.setattr(policy_pilot, 'gpu_inventory', lambda *args: {'synthetic': 'inventory'})
    monkeypatch.setenv('SOUP_G6_INSTANCE_ID', str(launch['instance_id']))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', launch['gpu_uuid'])
    monkeypatch.setenv('G6_POLICY_EXTERNAL_BOUND_S', '4800')
    args = Namespace(launch_config=None, launch_config_sha='private-sha', spec_sha='science',
                     protocol_commit='a' * 40, admission=admission, cuda_gate=receipt,
                     arm='baseline')
    with pytest.raises(ValueError, match='gate supervisor'):
        policy_pilot.pilot(args, spec, {})
