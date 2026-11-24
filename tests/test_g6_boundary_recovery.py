"""Fail closed on driver drift, old boot or a previously observed model response."""
import copy
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'benchmarks/g6'))
from boundary_recovery_preflight import validate_controls, validate_host, verify_prior  # noqa: E402


def admission():
    declaration = {'prior_failure': {'boot_id_sha256': hashlib.sha256(b'old').hexdigest()},
                   'target_gpu_uuid_sha256': hashlib.sha256(b'GPU-fixed').hexdigest(),
                   'expected_driver_version': '580.178.04'}
    host = {'boot_id': 'new', 'gpu_uuid': 'GPU-fixed', 'loaded_driver': '580.178.04',
            'module_on_disk': '580.178.04', 'nvml_reported_driver': '580.178.04',
            'package_writers': [], 'dpkg_audit': ''}
    return declaration, host


def test_matching_new_boot_is_admitted():
    declaration, host = admission()
    validate_host(host, declaration)


@pytest.mark.parametrize('key,value', [
    ('boot_id', 'old'), ('gpu_uuid', 'GPU-other'), ('loaded_driver', '580.95.05'),
    ('module_on_disk', '580.95.05'), ('nvml_reported_driver', '580.95.05'),
    ('package_writers', [123]), ('dpkg_audit', 'package incomplete'),
])
def test_old_boot_drift_and_package_work_block(key, value):
    declaration, host = admission()
    host[key] = value
    with pytest.raises(ValueError):
        validate_host(host, declaration)


def prior_fixture(tmp_path, observed=False, error='Driver/library version mismatch'):
    failure = json.dumps({'error': error, 'retry_allowed': False}).encode()
    launcher = json.dumps({'exitcode': 1, 'hard_timeout': False, 'spec_sha256': 'base'}).encode()
    files = {'runs/initial/failure.json': failure, 'runs/initial.launcher.json': launcher}
    if observed:
        files['runs/initial/confirmation_original.jsonl'] = b'{"answer":"observed"}\n'
    path = tmp_path / 'prior.tar.gz'
    with tarfile.open(path, 'w:gz') as handle:
        for name, raw in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            handle.addfile(member, io.BytesIO(raw))
    declaration = {'base_contract': {'sha256': 'base'}, 'prior_failure': {
        'archive_bytes': path.stat().st_size,
        'archive_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'required_members_sha256': {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}}}
    return path, declaration


def test_exact_zero_output_nvml_failure_is_eligible(tmp_path):
    path, declaration = prior_fixture(tmp_path)
    verify_prior(path, declaration)


def test_any_observed_generation_blocks_recovery_exception(tmp_path):
    path, declaration = prior_fixture(tmp_path, observed=True)
    with pytest.raises(ValueError, match='zero generation'):
        verify_prior(path, declaration)


def test_different_failure_is_not_eligible(tmp_path):
    path, declaration = prior_fixture(tmp_path, error='wrong answer')
    with pytest.raises(ValueError):
        verify_prior(path, declaration)


def test_archive_digest_drift_blocks(tmp_path):
    path, declaration = prior_fixture(tmp_path)
    declaration['prior_failure']['archive_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        verify_prior(path, declaration)


def control_fixture(tmp_path):
    directory = tmp_path / 'controls' / 'output'
    directory.mkdir(parents=True)
    for name in ['calibrate_guest_disk_v2.py', 'arm_validity.py']:
        (directory.parent / name).write_text(name)
    declaration = {'calibration_source_sha256': hashlib.sha256(
        (directory.parent / 'calibrate_guest_disk_v2.py').read_bytes()).hexdigest(),
        'harness_sha256': hashlib.sha256((directory.parent / 'arm_validity.py').read_bytes())
        .hexdigest()}
    done = {'watcher_pid': 10, 'ancestors': [21, 20, 1], 'started_unix_s': 2}
    controls = {'status': 'PASS_GUEST_CONTROLS_ONLY',
                'harness_sha256': declaration['harness_sha256'], 'foreign_reader': done,
                'controls': {}}
    for name, outcome in [('clean', 'ok'), ('own', 'ok'), ('foreign', 'void')]:
        controls['controls'][name] = {'outcome': outcome, 'gpu_clock': {'outcome': 'ok'},
                                     'checks': {'foreign_reads': {'outcome': outcome},
                                                'power': {'outcome': 'ok'},
                                                'suspend': {'outcome': 'ok'}}}
    values = {'calibration.json': controls, 'foreign_ready.json': {
        'boot_id': 'new', 'watcher_pid': 10, 'unix_s': 1}, 'foreign_done.json': done}
    for name, value in values.items():
        (directory / name).write_text(json.dumps(value))
    return directory, declaration, values


def test_fresh_independent_controls_are_required(tmp_path):
    directory, declaration, _ = control_fixture(tmp_path)
    assert len(validate_controls(directory, 'new', declaration)) == 3
    with pytest.raises(ValueError):
        validate_controls(directory, 'old', declaration)


@pytest.mark.parametrize('fault', ['foreign_passed', 'unknown_clock', 'dependent_reader'])
def test_invalid_controls_block(tmp_path, fault):
    directory, declaration, values = control_fixture(tmp_path)
    values = copy.deepcopy(values)
    if fault == 'foreign_passed':
        values['calibration.json']['controls']['foreign']['outcome'] = 'ok'
    elif fault == 'unknown_clock':
        values['calibration.json']['controls']['clean']['gpu_clock']['outcome'] = 'unknown'
    else:
        values['foreign_done.json']['ancestors'] = [21, 10, 1]
        values['calibration.json']['foreign_reader'] = values['foreign_done.json']
    for name, value in values.items():
        (directory / name).write_text(json.dumps(value))
    with pytest.raises(ValueError):
        validate_controls(directory, 'new', declaration)
