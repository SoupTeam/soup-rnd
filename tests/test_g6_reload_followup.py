"""Verification-only follow-up refuses altered evidence and false completion."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'benchmarks/g6'))
import reload_followup as followup  # noqa: E402


def test_reference_bytes_and_containment(tmp_path):
    (tmp_path / 'proof').write_text('preserved')
    manifest = {'proof': hashlib.sha256(b'preserved').hexdigest()}
    followup.verify_files(tmp_path, manifest)
    (tmp_path / 'proof').write_text('changed')
    with pytest.raises(ValueError):
        followup.verify_files(tmp_path, manifest)
    with pytest.raises(ValueError):
        followup.verify_files(tmp_path, {'../escape': 'sha'})


def test_comparison_requires_all_cases_exact_bytes_and_original_prefix(tmp_path):
    expected = tmp_path / 'expected'
    actual = tmp_path / 'actual'
    partial = tmp_path / 'partial'
    expected.write_text('{"id":"a"}\n{"id":"b"}\n')
    actual.write_bytes(expected.read_bytes())
    partial.write_text('{"id":"a"}\n')
    assert followup.compare_outputs(expected, partial, actual, ['a', 'b']) == 2
    actual.write_bytes(partial.read_bytes())
    with pytest.raises(ValueError):
        followup.compare_outputs(expected, partial, actual, ['a', 'b'])
    actual.write_text('{"id": "a"}\n{"id":"b"}\n')
    with pytest.raises(ValueError):
        followup.compare_outputs(expected, partial, actual, ['a', 'b'])


@pytest.mark.parametrize('field,value', [('observed_unix_s', 699), ('mode', 'pilot'),
                                        ('balance_usd', float('nan')),
                                        ('gpu_uuid', 'GPU-other'), ('balance_usd', 8.1)])
def test_fresh_private_authority_required(field, value):
    launch = {'account': 'example', 'instance_id': 123, 'gpu_uuid': 'GPU-example',
              'account_floor_usd': 8, 'stop_only_this_instance_after_preservation': True,
              'contract_sha256': 'science'}
    receipt = {'account': 'example', 'instance_id': 123, 'gpu_uuid': 'GPU-example',
               'mode': 'reload-verification', 'contract_sha256': 'science',
               'launch_config_sha256': 'private', 'authorized_for_this_stage': True,
               'observed_unix_s': 1000, 'balance_usd': 12, 'total_account_rate_usd_h': .6}
    followup.admit(receipt, launch, 'science', 'private', 1000)
    receipt[field] = value
    with pytest.raises(ValueError):
        followup.admit(receipt, launch, 'science', 'private', 1000)


def test_training_or_invalid_clock_phase_never_becomes_success(tmp_path):
    folder = tmp_path / 'baseline'
    folder.mkdir()
    for name in followup.PHASES:
        (folder / (name + '_phase.json')).write_text(json.dumps({
            'F1': {'outcome': 'ok', 'gpu_clock': {'outcome': 'ok'}}}))
    followup.verify_phases(folder)
    (folder / 'train_phase.json').write_text(json.dumps({
        'F1': {'outcome': 'ok', 'gpu_clock': {'outcome': 'unknown'}}}))
    with pytest.raises(ValueError):
        followup.verify_phases(folder)


def test_runtime_entry_has_no_training_call():
    import ast

    tree = ast.parse(Path(followup.__file__).read_text())
    calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(isinstance(node, ast.Name) and node.id in {
        'train', 'execute_arm', 'build_model'} for node in calls)
    assert not any(isinstance(node, ast.Attribute) and node.attr in {
        'backward', 'step'} for node in calls)


def test_supervisor_terminates_group_on_deadline_without_retry(tmp_path, monkeypatch):
    import signal
    import subprocess

    class Child:
        pid = 987654
        returncode = -15
        waits = 0

        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired('synthetic', timeout)
            return -15

    calls = []
    signals = []

    def start(*args, **kwargs):
        calls.append(kwargs)
        return Child()

    monkeypatch.setattr(followup, 'evidence', lambda *args: ({}, {}))
    monkeypatch.setattr(followup.subprocess, 'Popen', start)
    monkeypatch.setattr(followup.os, 'killpg', lambda pid, sig: signals.append((pid, sig)))
    output = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['reload', 'run', '--spec', 'science', '--spec-sha', 'sha',
                                    '--reference', 'reference', '--artifact-root', 'artifacts',
                                    '--data', 'data', '--out', str(output),
                                    '--launch-config', 'cfg',
                                    '--admission', 'admit', '--model-dir', 'model',
                                    '--launch-config-sha', 'cfgsha', '--protocol-commit', 'a' * 40])
    with pytest.raises(SystemExit):
        followup.main()
    assert len(calls) == 1 and calls[0]['start_new_session'] is True
    assert signals == [(Child.pid, signal.SIGTERM)]
    receipt = json.loads(Path(str(output) + '.launcher.json').read_text())
    assert receipt['hard_timeout'] is True and receipt['bound_s'] == 2400
