"""Selection evidence must never become a DMA verdict or execution permission."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks/harness'))
import g3_readiness as gate  # noqa: E402


def command(text):
    return {'status': 'COMPLETED', 'returncode': 0, 'truncated': False, 'stdout': text}


def file_record(text):
    return {'status': 'READ', 'truncated': False, 'text': text}


def candidate():
    return {
        'schema': 'g3-inventory-v1', 'system': 'Linux', 'environment': {},
        'commands': {
            'gpu': command('GPU-00000000-0000-0000-0000-000000000000, NVIDIA A40, 570, 0'),
            'mount': command(json.dumps({'filesystems': [
                {'fstype': 'xfs', 'maj:min': '259:0', 'options': 'rw'}]})),
            'block_devices': command(json.dumps({'blockdevices': [
                {'type': 'disk', 'kname': 'nvme0n1', 'maj:min': '259:0',
                 'tran': 'nvme', 'rota': False}]})),
            'gdscheck': command('NVMe : nvfs, compat\nGPU index 0 NVIDIA A40 supports GDS'),
        },
        'files': {
            'modules': file_record('nvidia_fs 100 0\n'),
            'nvidia_fs_stats': file_record('Read: 0'),
            'boot_id': file_record('00000000-0000-0000-0000-000000000000'),
        },
    }


def test_even_complete_diagnostics_never_authorize_execution():
    result = gate.assess(candidate())
    assert result['status'] == 'CANDIDATE_REQUIRES_LIVE_REVIEW'
    assert result['support_verdict'] == 'NO_VERDICT'
    assert not result['direct_path_verified'] and not result['execution_authorized']


@pytest.mark.parametrize('name', ['GeForce RTX 4090', 'NVIDIA A4000', 'Unknown', 'RTX A40'])
def test_other_models_do_not_match_shortlist_by_substring(name):
    value = candidate()
    value['commands']['gpu']['stdout'] = f'GPU-test, {name}, 570, 0'
    assert gate.assess(value)['status'] == 'REJECT_CANDIDATE'


@pytest.mark.parametrize('section,key', [
    ('files', 'modules'), ('files', 'nvidia_fs_stats'), ('files', 'boot_id'),
    ('commands', 'gpu'), ('commands', 'mount'), ('commands', 'block_devices'),
    ('commands', 'gdscheck'),
])
def test_missing_or_truncated_evidence_is_rejected(section, key):
    for problem in ['missing', 'truncated']:
        value = candidate()
        if problem == 'missing':
            del value[section][key]
        else:
            value[section][key]['truncated'] = True
        assert gate.assess(value)['status'] == 'REJECT_CANDIDATE'


@pytest.mark.parametrize('transport', ['compat', 'Unsupported', 'p2pdma, compat'])
def test_fallback_and_other_transports_are_not_the_frozen_route(transport):
    value = candidate()
    value['commands']['gdscheck']['stdout'] = (
        f'NVMe : {transport}\nGPU index 0 NVIDIA A40 supports GDS')
    assert gate.assess(value)['status'] == 'REJECT_CANDIDATE'


def test_loop_backed_mount_and_ext4_without_explicit_ordering_are_rejected():
    value = candidate()
    value['commands']['block_devices']['stdout'] = json.dumps({'blockdevices': [
        {'type': 'loop', 'kname': 'loop21', 'maj:min': '259:0', 'rota': False}]})
    assert gate.assess(value)['status'] == 'REJECT_CANDIDATE'
    value = candidate()
    value['commands']['mount']['stdout'] = value['commands']['mount']['stdout'].replace(
        'xfs', 'ext4')
    assert gate.assess(value)['status'] == 'REJECT_CANDIDATE'


def test_override_does_not_substitute_for_configured_transport():
    value = candidate()
    value['environment']['CUFILE_USE_PCIP2PDMA'] = 'true'
    assert gate.assess(value)['status'] == 'REJECT_CANDIDATE'


def test_cli_preserves_input_and_refuses_to_overwrite(tmp_path):
    source, output = tmp_path / 'inventory.json', tmp_path / 'result.json'
    raw = json.dumps(candidate()).encode()
    source.write_bytes(raw)
    argv = [sys.executable, str(ROOT / 'benchmarks/harness/g3_readiness.py'),
            '--inventory', str(source), '--out', str(output)]
    subprocess.run(argv, check=True, capture_output=True, timeout=10)
    before = output.read_bytes()
    assert subprocess.run(argv, capture_output=True, timeout=10).returncode != 0
    assert output.read_bytes() == before and source.read_bytes() == raw
