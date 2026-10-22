"""CPU-only oracle, scheduler, classification and compiled fake-backend tests for G3."""
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks/harness'))
import g3_probe as probe  # noqa: E402

UUID = 'GPU-00000000-0000-0000-0000-000000000000'
SIZE = 65536


def arm(mode):
    native = {'backend': 'NATIVE_CUDA_CUFILE', 'mode': mode, 'gpu_uuid': UUID, 'bytes': SIZE,
              'device_memory': True, 'compat_disabled': True, 'stats_available': True,
              'route_read_bytes': SIZE, 'read_operations': 1, 'total_reads': 1,
              'p2p_reads': 1, 'nvfs_reads': 0, 'posix_reads': 0, 'unaligned_reads': 0,
              'dr_reads': 0, 'read_errors': 0, 'elapsed_s': 0.01}
    watch = {'outcome': 'ok', 'checks': {name: {'outcome': 'ok'} for name in
                                       ['suspend', 'power', 'foreign_reads']},
             'gpu_clock': {'outcome': 'ok'}, 'gpu': {'id': UUID},
             'box_stamp_before': {'gpu_processes': []},
             'box_stamp_after': {'gpu_processes': []}}
    return {'mode': mode, 'returncode': 0, 'native': native, 'watch': watch, 'integrity': True}


def block(performance=False):
    rows = [arm(mode) for mode in ['control', 'direct', 'direct', 'control']]
    if performance:
        for row in rows:
            row['native'].update(read_operations=5, route_read_bytes=SIZE*5,
                                 timed_samples=3, elapsed_samples_s=[0.01]*3)
    return rows


def test_independent_fixture_and_corruption_detection(tmp_path):
    work = tmp_path / 'fixture'
    probe.prepare(work, SIZE)
    value = json.loads((work / 'fixture.json').read_text())
    assert value['sha256'] == probe.expected_fixture_sha(SIZE)
    data = work / 'fixture.bin'
    assert probe.integrity(data, value['sha256'], SIZE)
    corrupted = bytearray(data.read_bytes())
    corrupted[19] ^= 1
    data.write_bytes(corrupted)
    assert not probe.integrity(data, value['sha256'], SIZE)
    with pytest.raises(FileExistsError):
        probe.prepare(work, SIZE)


@pytest.mark.parametrize('size', [0, -1, 4096, 65535, True, 67174400])
def test_fixture_limits(size, tmp_path):
    with pytest.raises(ValueError):
        probe.prepare(tmp_path / 'bad', size)
    assert not (tmp_path / 'bad').exists()


def test_direct_and_supported_negative_classification():
    rows = block()
    assert probe.classify(rows, SIZE) == 'DIRECT_PATH_VERIFIED'
    for index in [1, 2]:
        rows[index].update(returncode=2, integrity=False, native={}, native_error={
            'event': 'CUFILE_ERROR', 'stage': 'handle_register',
            'explicit_not_supported': True, 'code': 5008})
    assert probe.classify(rows, SIZE) == 'UNSUPPORTED_ON_TESTED_STACK'
    rows[2]['native_error']['stage'] = 'driver_open'
    assert probe.classify(rows, SIZE) == 'NO_VERDICT'


@pytest.mark.parametrize('field,value', [
    ('backend', 'SYNTHETIC_CPU_MOCK'), ('device_memory', False), ('compat_disabled', False),
    ('stats_available', False), ('route_read_bytes', SIZE-1), ('total_reads', 0),
    ('p2p_reads', 0), ('posix_reads', 1), ('unaligned_reads', 1), ('dr_reads', 1),
    ('read_errors', 1), ('route_read_bytes', True), ('nvfs_reads', -1),
    ('total_reads', 1.0), ('read_operations', 0),
])
def test_bad_route_is_never_direct(field, value):
    rows = block()
    rows[1]['native'][field] = value
    assert probe.classify(rows, SIZE) == 'NO_VERDICT'


@pytest.mark.parametrize('mutation', ['missing', 'order', 'control', 'unknown', 'void', 'clock'])
def test_invalid_controls_and_incomplete_abba(mutation):
    rows = block()
    if mutation == 'missing':
        rows.pop()
    elif mutation == 'order':
        rows.reverse()
        rows[0]['mode'] = 'direct'
    elif mutation == 'control':
        rows[0]['integrity'] = False
    elif mutation == 'clock':
        rows[0]['watch']['gpu_clock']['outcome'] = 'unknown'
    else:
        rows[1]['watch']['checks']['foreign_reads']['outcome'] = mutation
        rows[1]['watch']['outcome'] = mutation
    assert probe.classify(rows, SIZE) == 'NO_VERDICT'


def test_missing_driver_or_permission_is_not_unsupported():
    rows = block()
    for index in [1, 2]:
        rows[index].update(returncode=2, integrity=False, native={}, native_error={
            'event': 'CUFILE_ERROR', 'stage': 'driver_open',
            'explicit_not_supported': False, 'code': 5025})
    assert probe.classify(rows, SIZE) == 'NO_VERDICT'


def test_peer_guards():
    watch = arm('control')['watch']
    probe.validate_watch_peers(watch)
    watch['box_stamp_before']['gpu_processes'] = [{'pid': 123}]
    with pytest.raises(ValueError):
        probe.validate_watch_peers(watch)


def authority():
    return {'authorized': True, 'scope': 'g3-primitive-only', 'boot_id': 'boot',
            'gpu_uuid': UUID, 'instance_id': 123, 'started_unix_s': 1000,
            'deadline_unix_s': 5000, 'rate_usd_h': 1,
            'budget_cap_usd': 2, 'spent_before_lease_usd': 0,
            'traffic_and_cleanup_reserve_usd': 0.2}


@pytest.mark.parametrize('field,value', [
    ('authorized', False), ('scope', 'training'), ('boot_id', 'old'),
    ('gpu_uuid', 'any'), ('instance_id', 0), ('deadline_unix_s', 2000),
    ('budget_cap_usd', 0.01), ('rate_usd_h', float('nan')),
    ('started_unix_s', 3000), ('traffic_and_cleanup_reserve_usd', -1),
    ('spent_before_lease_usd', 2),
])
def test_lease_guards(field, value):
    value_dict = authority()
    probe.validate_authority(value_dict, 2000, 'boot')
    value_dict[field] = value
    with pytest.raises(ValueError):
        probe.validate_authority(value_dict, 2000, 'boot')


def test_uncommitted_rule_is_refused():
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        probe.verify_rule('0'*40, probe.sha(ROOT / 'benchmarks/g3/PROTOCOL.md'))


def test_raw_completion_duplication_rejected():
    raw = json.dumps({'event': 'READ_COMPLETE'})
    with pytest.raises(ValueError):
        probe.raw_record(raw + '\n' + raw)


def test_repeat_stopping_and_void_replacement():
    rows = [row for _ in range(5) for row in block(performance=True)]
    result = probe.performance_summary(rows, SIZE)
    assert result['status'] == 'PERFORMANCE_COMPARISON_COMPLETE'
    assert result['primitive_latency_ratio'] == 1
    assert probe.performance_summary(rows + block(True), SIZE)['status'] == 'NO_VERDICT'
    rows = copy.deepcopy(rows)
    rows[0]['watch']['outcome'] = 'void'
    assert probe.performance_summary(rows, SIZE)['status'] == 'MORE_BLOCKS_REQUIRED'
    assert probe.performance_summary(rows + block(True), SIZE)['status'] == (
        'PERFORMANCE_COMPARISON_COMPLETE')


@pytest.mark.parametrize('mutation', ['nan', 'median', 'unknown', 'notfive', 'fallback'])
def test_performance_rejects_bad_evidence(mutation):
    rows = [row for _ in range(5) for row in block(True)]
    if mutation == 'nan':
        rows[0]['native']['elapsed_samples_s'][0] = float('nan')
    elif mutation == 'median':
        rows[0]['native']['elapsed_s'] = 123
    elif mutation == 'unknown':
        rows[0]['watch']['outcome'] = 'unknown'
    elif mutation == 'notfive':
        rows[0]['native']['read_operations'] = 1
    else:
        rows[1]['native']['posix_reads'] = 1
    assert probe.performance_summary(rows, SIZE)['status'] == 'NO_VERDICT'


@pytest.fixture(scope='session')
def mock_binary(tmp_path_factory):
    if os.name != 'posix':
        pytest.skip('Native G3 and its CPU substitute use POSIX file APIs')
    compiler = shutil.which('clang++') or shutil.which('c++')
    if not compiler:
        pytest.skip('No local C++ compiler for explicitly synthetic tests')
    output = tmp_path_factory.mktemp('g3-cpu-build') / 'mock_probe'
    command = [compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror',
               '-DG3_TEST_BACKEND=1', '-DG3_WITH_STATS=1',
               '-I', str(ROOT / 'tests/fixtures/g3'),
               str(ROOT / 'benchmarks/g3/direct_probe.cpp'), '-o', str(output)]
    subprocess.run(command, check=True, capture_output=True, timeout=30)
    return output


@pytest.mark.parametrize('mode,iterations', [('control', '1'), ('direct', '1'),
                                            ('control', '5'), ('direct', '5')])
def test_compiled_mock_control_and_direct_bytes(tmp_path, mock_binary, mode, iterations):
    probe.prepare(tmp_path / 'fixture', SIZE)
    source = tmp_path / 'fixture/fixture.bin'
    destination = tmp_path / 'readback.bin'
    result = subprocess.run([str(mock_binary), mode, str(source), str(SIZE),
                             str(destination), UUID, 'on', iterations], capture_output=True,
                            text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    row = probe.raw_record(result.stdout)
    assert row['backend'] == 'SYNTHETIC_CPU_MOCK'
    assert destination.read_bytes() == source.read_bytes()
    assert row['read_operations'] == int(iterations)
    assert row['timed_samples'] == (1 if iterations == '1' else 3)
    assert not probe.direct_route(row, SIZE)  # Mock can never count as GDS evidence.


@pytest.mark.parametrize('variable,stage', [('G3_MOCK_COMPAT', 'compatibility_mode'),
                                           ('G3_MOCK_CORRUPT', 'per_iteration_integrity'),
                                           ('G3_MOCK_UNSUPPORTED', 'handle_register')])
def test_compiled_mock_negative_controls(tmp_path, mock_binary, variable, stage):
    probe.prepare(tmp_path / 'fixture', SIZE)
    env = dict(os.environ, **{variable: '1'})
    result = subprocess.run([str(mock_binary), 'direct', str(tmp_path/'fixture/fixture.bin'),
                             str(SIZE), str(tmp_path/'readback.bin'), UUID, 'on', '1'],
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2 and stage in result.stdout + result.stderr
    assert not (tmp_path / 'readback.bin').exists()


def test_cpu_mock_audit_cannot_be_accepted(tmp_path):
    # Audit has to check coverage before reading any claimed hardware verdict.
    (tmp_path/'files_sha256.json').write_text('{}')
    (tmp_path/'result.json').write_text('{"status":"DIRECT_PATH_VERIFIED"}')
    with pytest.raises(ValueError, match='coverage'):
        probe.audit(tmp_path)


def topology():
    commands = {name: {'status': 'COMPLETED', 'returncode': 0, 'truncated': False}
                for name in ['gpu', 'mount', 'block_devices']}
    commands['mount']['stdout'] = json.dumps({'filesystems': [
        {'fstype': 'ext4', 'maj:min': '259:1'}]})
    commands['block_devices']['stdout'] = json.dumps({'blockdevices': [
        {'kname': 'nvme0n1', 'type': 'disk', 'tran': 'nvme', 'rota': False,
         'maj:min': '259:0', 'children': [
             {'kname': 'nvme0n1p1', 'type': 'part', 'pkname': 'nvme0n1',
              'maj:min': '259:1'}]}]})
    return {'commands': commands}


def test_visible_nvme_partition_resolution():
    assert probe.local_nvme_device(topology()) == 'nvme0n1'


@pytest.mark.parametrize('mutation', ['overlay', 'missing', 'virtio', 'rotating', 'lvm'])
def test_storage_not_attributable_to_local_nvme_is_blocked(mutation):
    value = topology()
    if mutation == 'overlay':
        value['commands']['mount']['stdout'] = json.dumps({'filesystems': [
            {'fstype': 'overlay', 'maj:min': '259:1'}]})
    elif mutation == 'missing':
        value['commands']['mount']['returncode'] = 1
    else:
        devices = json.loads(value['commands']['block_devices']['stdout'])
        disk = devices['blockdevices'][0]
        if mutation == 'virtio':
            disk['tran'] = 'virtio'
        elif mutation == 'rotating':
            disk['rota'] = True
        else:
            disk['type'] = 'lvm'
        value['commands']['block_devices']['stdout'] = json.dumps(devices)
    with pytest.raises(ValueError):
        probe.local_nvme_device(value)


def test_calibration_requires_own_and_independent_foreign_controls():
    import g3_controls
    rows = {}
    for name, outcome in [('clean', 'ok'), ('own', 'ok'), ('foreign', 'void')]:
        rows[name] = {'outcome': outcome, 'checks': {
            'foreign_reads': {'outcome': outcome}, 'power': {'outcome': 'ok'},
            'suspend': {'outcome': 'ok'}}, 'gpu_clock': {'outcome': 'ok'}}
    assert g3_controls.controls_ok(rows)
    rows['foreign']['checks']['foreign_reads']['outcome'] = 'ok'
    assert not g3_controls.controls_ok(rows)
    assert g3_controls.independent([10, 3, 1], 11)
    assert not g3_controls.independent([10, 11, 1], 11)
    mapping = [{'before': 10, 'after': 74, 'requested_bytes': 64}]*3
    assert g3_controls.mapping_ok(mapping)
    assert not g3_controls.mapping_ok(mapping[:2])
    assert not g3_controls.mapping_ok([{'before': 10, 'after': 73, 'requested_bytes': 64}]*3)


def test_live_library_changes_and_preload_are_refused(tmp_path, monkeypatch):
    path = tmp_path/'libexample.so'
    path.write_bytes(b'initial runtime')
    values = {str(path): probe.sha(path)}
    monkeypatch.delenv('LD_PRELOAD', raising=False)
    probe.verify_linked_libraries(values)
    path.write_bytes(b'changed runtime')
    with pytest.raises(ValueError):
        probe.verify_linked_libraries(values)
    monkeypatch.setenv('LD_PRELOAD', '/unapproved.so')
    with pytest.raises(ValueError):
        probe.verify_linked_libraries(values)


def test_native_dependency_discovery_requires_both_runtime_libraries(tmp_path):
    runtime = tmp_path/'libcudart.so.13'
    cufile = tmp_path/'libcufile.so.0'
    runtime.write_bytes(b'unit test runtime')
    cufile.write_bytes(b'unit test cuFile')
    record = {'status': 'COMPLETED', 'returncode': 0, 'truncated': False,
              'stdout': f'libcudart.so.13 => {runtime} (0x0)\nlibcufile.so.0 => {cufile} (0x1)'}
    hashes = probe.linked_library_hashes(record)
    assert hashes[str(runtime)] == probe.sha(runtime)
    record['stdout'] = f'libcudart.so.13 => {runtime} (0x0)'
    with pytest.raises(ValueError):
        probe.linked_library_hashes(record)
