"""Prepare/build/run/audit the bounded G3 primitive; CUDA work requires frozen authority."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

from g3_capability import collect_inventory, run_command
from rich.console import Console

REPO = Path(__file__).parents[2]
LIMIT_BYTES = 64 * 1024**2
SOURCES = ['benchmarks/g3/direct_probe.cpp', 'benchmarks/g3/PROTOCOL.md',
           'benchmarks/harness/g3_probe.py', 'benchmarks/harness/g3_capability.py',
           'benchmarks/harness/g3_controls.py', 'benchmarks/harness/arm_validity.py',
           'benchmarks/harness/g3_archive.py']


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def manifest(root: Path) -> None:
    values = {str(path.relative_to(root)): sha(path) for path in root.rglob('*')
              if path.is_file() and path.name != 'files_sha256.json'}
    write_json(root / 'files_sha256.json', values)


def fixture_bytes(block: int) -> bytes:
    # Distinct 4KiB blocks, rather than a trivially repeating byte pattern.
    return hashlib.shake_256(b'soup-g3-v1\0' + block.to_bytes(8, 'little')).digest(4096)



def expected_fixture_sha(size: int) -> str:
    if type(size) is not int or not 65536 <= size <= LIMIT_BYTES or size % 65536:
        raise ValueError('Invalid bounded fixture size')
    digest = hashlib.sha256()
    for block in range(size // 4096):
        digest.update(fixture_bytes(block))
    return digest.hexdigest()


def prepare(root: Path, size: int = LIMIT_BYTES) -> None:
    if type(size) is not int or not 65536 <= size <= LIMIT_BYTES or size % 65536:
        raise ValueError('Fixture must be 64KiB-aligned, between64KiB and64MiB')
    root.mkdir(exist_ok=False)
    path = root / 'fixture.bin'
    with path.open('xb') as stream:
        for block in range(size // 4096):
            stream.write(fixture_bytes(block))
        stream.flush()
        os.fsync(stream.fileno())
    digest = hashlib.sha256()
    for block in range(size // 4096):
        digest.update(fixture_bytes(block))
    if sha(path) != digest.hexdigest():
        raise ValueError('Fixture independent readback mismatch')
    write_json(root / 'fixture.json', {'bytes': size, 'sha256': digest.hexdigest(),
                                     'generator': 'shake256-g3-v1-per4KiB'})
    manifest(root)



def linked_library_hashes(record: dict) -> dict:
    if (record.get('status') != 'COMPLETED' or record.get('returncode') != 0
            or record.get('truncated') is not False):
        raise ValueError('Native shared-library dependency discovery failed')
    paths = re.findall(r'(?:=>\s+|^\s*)(/[^\s]+)', record['stdout'], re.MULTILINE)
    values = {os.path.realpath(path): sha(Path(path)) for path in paths}
    if not any('libcufile.so' in path for path in values) or not any(
            'libcudart.so' in path for path in values):
        raise ValueError('Expected CUDA/cuFile dynamic libraries not attributable')
    return values


def verify_linked_libraries(value: dict) -> None:
    if not value or os.environ.get('LD_PRELOAD'):
        raise ValueError('Native libraries missing or unapproved preload present')
    for name, digest in value.items():
        if not Path(name).is_absolute() or sha(Path(name)) != digest:
            raise ValueError('Native runtime library changed')


def build(root: Path, cuda: Path, stats: bool) -> None:
    if sys.platform != 'linux':
        raise ValueError('Native build requires Linux; CPU mock is test-only')
    root.mkdir(exist_ok=False)
    compiler = shutil.which('c++')
    if compiler is None:
        raise ValueError('C++ compiler unavailable; environment blocked')
    binary = root / 'g3_native'
    command = [compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
               '-I' + str(cuda / 'include'), str(REPO / SOURCES[0]),
               '-L' + str(cuda / 'lib64'), '-Wl,-rpath,' + str(cuda / 'lib64'),
               '-lcudart', '-lcufile', '-o', str(binary)]
    if stats:
        command.insert(1, '-DG3_WITH_STATS=1')
    result = run_command(command, timeout=120)
    write_json(root / 'compile.json', result)
    if result.get('returncode') != 0 or result.get('status') != 'COMPLETED':
        raise ValueError('Native build failed; keep compiler evidence, do not claim unsupported')
    libraries = run_command(['ldd', str(binary)])
    hashes = linked_library_hashes(libraries)
    write_json(root / 'build.json', {'backend': 'NATIVE_CUDA_CUFILE', 'stats': stats,
                                    'source_sha256': sha(REPO / SOURCES[0]),
                                    'binary_sha256': sha(binary), 'command': command,
                                    'compiler': run_command([compiler, '--version']),
                                    'linked_libraries': libraries, 'libraries_sha256': hashes,
                                    'headers_sha256': {name: sha(cuda/'include'/name)
                                        for name in ['cufile.h', 'cufile_stats.h',
                                                     'cuda_runtime_api.h']
                                        if (cuda/'include'/name).is_file()}})
    manifest(root)



def local_nvme_device(inventory: dict) -> str:
    """Resolve the target mount's major:minor to one visible, non-rotational NVMe namespace."""
    commands = inventory['commands']
    for name in ['gpu', 'mount', 'block_devices']:
        row = commands.get(name, {})
        if (row.get('status') != 'COMPLETED' or row.get('returncode') != 0
                or row.get('truncated') is not False):
            raise ValueError('Required live topology evidence missing: ' + name)
    mounts = json.loads(commands['mount']['stdout'])['filesystems']
    if len(mounts) != 1 or mounts[0]['fstype'] not in ['ext4', 'xfs']:
        raise ValueError('This rule requires a single local ext4/xfs target mount')
    by_id, by_name = {}, {}
    def visit(nodes, parent=None):
        for node in nodes:
            entry = dict(node)
            entry['parent_name'] = node.get('pkname') or parent
            by_id[node['maj:min']] = entry
            by_name[node['kname']] = entry
            visit(node.get('children', []), node['kname'])
    visit(json.loads(commands['block_devices']['stdout'])['blockdevices'])
    target = by_id.get(mounts[0]['maj:min'])
    seen = set()
    while target and target['type'] == 'part':
        if target['kname'] in seen:
            raise ValueError('Cycle in block topology')
        seen.add(target['kname'])
        target = by_name.get(target.get('parent_name'))
    if (not target or target['type'] != 'disk' or target.get('tran') != 'nvme'
            or target.get('rota') not in [False, 0]
            or not re.fullmatch(r'nvme[0-9]+n[0-9]+', target['kname'])):
        raise ValueError('Target is not attributable to one visible NVMe namespace')
    return target['kname']


def validate_authority(value: dict, now: float, boot: str) -> None:
    if (value.get('authorized') is not True or value.get('scope') != 'g3-primitive-only'
            or value.get('boot_id') != boot or type(value.get('instance_id')) is not int
            or value['instance_id'] <= 0
            or not re.fullmatch(r'GPU-[a-fA-F0-9-]{36}', value.get('gpu_uuid', ''))):
        raise ValueError('Explicit G3 authority and exact current host required')
    fields = ['budget_cap_usd', 'started_unix_s', 'deadline_unix_s',
              'rate_usd_h', 'traffic_and_cleanup_reserve_usd', 'spent_before_lease_usd']
    if any(type(value.get(name)) not in [int, float] or not math.isfinite(value[name])
           for name in fields):
        raise ValueError('Finite lease budget required')
    if (not value['started_unix_s'] <= now < value['deadline_unix_s']
            or value['deadline_unix_s'] - now < 940
            or value['deadline_unix_s'] - value['started_unix_s'] > 10800
            or value['budget_cap_usd'] <= 0 or value['rate_usd_h'] <= 0
            or value['traffic_and_cleanup_reserve_usd'] < 0 or value['spent_before_lease_usd'] < 0):
        raise ValueError('Insufficient bounded lease or invalid budget')
    projected = (value['rate_usd_h'] * (value['deadline_unix_s'] - value['started_unix_s'])
                 / 3600 + value['traffic_and_cleanup_reserve_usd']
                 + value['spent_before_lease_usd'])
    if projected > value['budget_cap_usd']:
        raise ValueError('Cumulative lease estimate exceeds authorized cap')


def verify_rule(commit: str, digest: str) -> None:
    if not re.fullmatch(r'[a-f0-9]{40}', commit) or not re.fullmatch(r'[a-f0-9]{64}', digest):
        raise ValueError('Exact committed rule identity required')
    if sha(REPO / 'benchmarks/g3/PROTOCOL.md') != digest:
        raise ValueError('Rule SHA differs')
    for name in SOURCES:
        raw = subprocess.run(['git', 'show', commit + ':' + name], cwd=REPO,
                             capture_output=True, check=True, timeout=30).stdout
        if raw != (REPO / name).read_bytes():
            raise ValueError('Source differs from committed rule: ' + name)


def check_controls(value: dict, boot: str) -> bool:
    from g3_controls import HARNESS_SHA, controls_ok, independent, mapping_ok
    reader = value.get('foreign_reader', {})
    return (value.get('status') == 'PASS_DISK_SENSOR_CONTROLS'
            and value.get('boot_id') == boot and value.get('harness_sha256') == HARNESS_SHA
            and controls_ok(value.get('controls', {})) and mapping_ok(value.get('mapping', []))
            and independent(reader.get('ancestors', []), reader.get('watcher_pid', 0)))


def raw_record(raw: str) -> dict | None:
    rows = [json.loads(line) for line in raw.splitlines() if line.strip().startswith('{')]
    completed = [row for row in rows if row.get('event') == 'READ_COMPLETE']
    if len(completed) > 1:
        raise ValueError('Duplicate native completion')
    return completed[0] if completed else None


def integrity(path: Path, expected_sha: str, size: int) -> bool:
    return (path.is_file() and not path.is_symlink() and path.stat().st_size == size
            and sha(path) == expected_sha)


def direct_route(row: dict, size: int) -> bool:
    fields = ['route_read_bytes', 'total_reads', 'p2p_reads', 'nvfs_reads', 'posix_reads',
              'unaligned_reads', 'dr_reads', 'read_errors']
    if any(type(row.get(name)) is not int or row[name] < 0 for name in fields):
        return False
    return (row.get('backend') == 'NATIVE_CUDA_CUFILE'
            and row.get('device_memory') is True and row.get('compat_disabled') is True
            and row.get('stats_available') is True
            and type(row.get('read_operations')) is int and row['read_operations'] in [1, 5]
            and row['route_read_bytes'] == size * row['read_operations']
            and row['total_reads'] > 0
            and row['p2p_reads'] + row['nvfs_reads'] == row['total_reads']
            and all(row[name] == 0 for name in ['posix_reads', 'unaligned_reads',
                                               'dr_reads', 'read_errors']))


def watched_ok(row: dict) -> bool:
    watch = row.get('watch', {})
    return (watch.get('outcome') == 'ok'
            and watch.get('gpu_clock', {}).get('outcome') == 'ok'
            and all(watch.get('checks', {}).get(name, {}).get('outcome') == 'ok'
                    for name in ['suspend', 'power', 'foreign_reads']))


def classify(rows: list[dict], size: int) -> str:
    if len(rows) != 4 or [row.get('mode') for row in rows] != ['control', 'direct',
                                                               'direct', 'control']:
        return 'NO_VERDICT'
    if not all(watched_ok(row) for row in rows):
        return 'NO_VERDICT'
    controls = [row for row in rows if row['mode'] == 'control']
    if not all(row.get('integrity') is True and row.get('returncode') == 0
               and row.get('native', {}).get('backend') == 'NATIVE_CUDA_CUFILE'
               and row.get('native', {}).get('device_memory') is True for row in controls):
        return 'NO_VERDICT'
    direct = [row for row in rows if row['mode'] == 'direct']
    if all(row.get('integrity') is True and row.get('returncode') == 0
           and direct_route(row.get('native', {}), size) for row in direct):
        return 'DIRECT_PATH_VERIFIED'
    # Only explicit native API rejection on the same stage twice; not errno or missing tools.
    errors = [row.get('native_error', {}) for row in direct]
    if (all(row.get('returncode') == 2 for row in direct) and errors[0] == errors[1]
            and errors[0].get('event') == 'CUFILE_ERROR'
            and errors[0].get('explicit_not_supported') is True
            and errors[0].get('stage') in ['driver_open', 'handle_register', 'buffer_register']):
        return 'UNSUPPORTED_ON_TESTED_STACK'
    return 'NO_VERDICT'



def performance_summary(rows: list[dict], size: int) -> dict:
    from arm_validity import Run, blocks_to_add, valid_blocks
    if not rows or len(rows) % 4:
        return {'status': 'NO_VERDICT', 'reason': 'Incomplete block'}
    blocks = []
    valid = []
    for start in range(0, len(rows), 4):
        block = rows[start:start+4]
        if [row.get('mode') for row in block] != ['control', 'direct', 'direct', 'control']:
            return {'status': 'NO_VERDICT', 'reason': 'Order differs'}
        sample = []
        for row in block:
            native = row.get('native', {})
            values = native.get('elapsed_samples_s', [])
            if (row.get('returncode') != 0 or row.get('integrity') is not True
                    or native.get('backend') != 'NATIVE_CUDA_CUFILE'
                    or native.get('read_operations') != 5 or native.get('timed_samples') != 3
                    or len(values) != 3
                    or type(native.get('elapsed_s')) not in [int, float]
                    or not math.isfinite(native['elapsed_s']) or native['elapsed_s'] <= 0
                    or any(type(v) not in [int, float] or not math.isfinite(v) or v <= 0
                           for v in values)
                    or abs(statistics.median(values) - native.get('elapsed_s', 0)) > 1e-5
                    or (row['mode'] == 'direct' and not direct_route(native, size))):
                return {'status': 'NO_VERDICT', 'reason': 'Invalid primitive evidence'}
            outcome = row.get('watch', {}).get('outcome')
            clock = row.get('watch', {}).get('gpu_clock', {}).get('outcome')
            if (outcome not in ['ok', 'void'] or clock not in ['ok', 'void']
                    or any(row.get('watch', {}).get('checks', {}).get(name, {}).get('outcome')
                           not in ['ok', 'void'] for name in
                           ['suspend', 'power', 'foreign_reads'])):
                return {'status': 'NO_VERDICT', 'reason': 'Required check unknown'}
            effective = 'ok' if watched_ok(row) else 'void'
            sample.append(Run(native['elapsed_s'], effective))
        blocks.append(sample)
        if all(run.outcome == 'ok' for run in sample):
            valid.append(block)
    additional = blocks_to_add(blocks)
    if additional:
        return {'status': 'NO_VERDICT' if len(blocks) >= 10 else 'MORE_BLOCKS_REQUIRED',
                'blocks_to_add': additional, 'total_blocks': len(blocks),
                'valid_blocks': len(valid_blocks(blocks))}
    # Reject optional stopping or extra blocks after the frozen rule already stopped.
    if any(blocks_to_add(blocks[:end]) == 0 for end in range(1, len(blocks))):
        return {'status': 'NO_VERDICT', 'reason': 'Runs after predetermined stopping point'}
    left = [row['native']['elapsed_s'] for block in valid for row in block
            if row['mode'] == 'control']
    right = [row['native']['elapsed_s'] for block in valid for row in block
             if row['mode'] == 'direct']
    return {'status': 'PERFORMANCE_COMPARISON_COMPLETE', 'valid_blocks': len(valid),
            'total_blocks': len(blocks), 'control_mean_run_median_s': statistics.fmean(left),
            'direct_mean_run_median_s': statistics.fmean(right),
            'primitive_latency_ratio': statistics.fmean(left) / statistics.fmean(right),
            'instrumented_cufile_stats': True, 'training_speedup_claim': False,
            'total_pinned_host_memory': 'UNKNOWN; explicit allocation and kernel counters only'}


def validate_watch_peers(watch: dict) -> None:
    for name in ['box_stamp_before', 'box_stamp_after']:
        if watch.get(name, {}).get('gpu_processes') != []:
            raise ValueError('GPU peers unknown or present; do not interfere with their jobs')


def run_capability(args: argparse.Namespace) -> None:
    from arm_validity import ArmWatch, disk_device
    if sys.platform != 'linux':
        raise ValueError('Controlled native execution requires Linux')
    boot_path = Path('/proc/sys/kernel/random/boot_id')
    boot = boot_path.read_text().strip()
    authority = json.loads(args.authority.read_text())
    validate_authority(authority, time.time(), boot)
    verify_rule(args.commit, args.rule_sha)
    inventory = collect_inventory(args.work)
    namespace = local_nvme_device(inventory)
    if authority['gpu_uuid'] not in inventory['commands']['gpu']['stdout']:
        raise ValueError('Selected GPU absent from live inventory')
    controls = json.loads(args.controls.read_text())
    if not check_controls(controls, boot) or controls['device'] != disk_device(str(args.work)):
        raise ValueError('Same-boot, same-disk independently calibrated F1 controls required')
    fixture = json.loads((args.work / 'fixture.json').read_text())
    if (fixture['sha256'] != expected_fixture_sha(fixture['bytes'])
            or not integrity(args.work / 'fixture.bin', fixture['sha256'], fixture['bytes'])):
        raise ValueError('Fixture changed')
    build_info = json.loads((args.build / 'build.json').read_text())
    binary = args.build / 'g3_native'
    if (build_info.get('backend') != 'NATIVE_CUDA_CUFILE'
            or build_info.get('source_sha256') != sha(REPO / SOURCES[0])
            or build_info.get('binary_sha256') != sha(binary)
            or b'SYNTHETIC_CPU_MOCK' in binary.read_bytes()
            or any('G3_TEST_BACKEND' in word for word in build_info['command'])):
        raise ValueError('Native binary/build identity differs or CPU mock detected')
    verify_linked_libraries(build_info['libraries_sha256'])
    peers = run_command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                         '--format=csv,noheader'])
    if peers.get('returncode') != 0 or peers.get('stdout', '').strip():
        raise ValueError('GPU peers present or unknown; window must be booked')
    phase = args.phase
    if phase == 'performance':
        prior = audit(args.work / 'capability_001')
        prior_identity = json.loads((args.work / 'capability_001/identity.json').read_text())
        if (prior['status'] != 'DIRECT_PATH_VERIFIED'
                or prior_identity['boot_id'] != boot
                or prior_identity['gpu_uuid'] != authority['gpu_uuid']
                or prior_identity['binary_sha256'] != sha(binary)):
            raise ValueError('Same-host verified capability required before timing study')
    output = args.work / (phase + '_001')
    output.mkdir(exist_ok=False)
    write_json(output / 'inventory.json', inventory)
    write_json(output / 'authority.json', authority)
    write_json(output / 'controls.json', controls)
    write_json(output / 'build.json', build_info)
    write_json(output / 'fixture.json', fixture)
    for name in SOURCES:
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / name).read_bytes())
    write_json(output / 'identity.json', {'boot_id': boot, 'gpu_uuid': authority['gpu_uuid'],
                                        'commit': args.commit, 'rule_sha256': args.rule_sha,
                                        'phase': phase, 'nvme_namespace': namespace,
                                        'binary_sha256': sha(binary), 'instance_id':
                                        authority['instance_id'], 'performance_claim': False,
                                        'source_sha256': {name: sha(REPO/name)
                                                          for name in SOURCES}})
    strict = {'logging': {'dir': str(output.absolute()), 'level': 'ERROR'},
              'properties': {'allow_compat_mode': False, 'force_compat_mode': False,
                              'rdma_dynamic_routing': False}}
    write_json(output / 'cufile.strict.json', strict)
    shutil.copy2(binary, output / 'native_binary')
    rows = []
    original_env = {name: os.environ.get(name) for name in [
        'CUDA_VISIBLE_DEVICES', 'CUFILE_ENV_PATH_JSON', 'CUFILE_FORCE_COMPAT_MODE']}
    os.environ['CUDA_VISIBLE_DEVICES'] = authority['gpu_uuid']
    os.environ['CUFILE_ENV_PATH_JSON'] = str((output / 'cufile.strict.json').absolute())
    os.environ.pop('CUFILE_FORCE_COMPAT_MODE', None)
    def timeout_handler(signum, frame):
        raise TimeoutError('G3 phase absolute900s bound')
    previous_alarm = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(900)
    try:
        for index in range(4 if phase == 'capability' else 40):
            mode = ['control', 'direct', 'direct', 'control'][index % 4]
            if (sha(output/'native_binary') != build_info['binary_sha256']
                    or boot_path.read_text().strip() != boot
                    or time.time() + 140 > authority['deadline_unix_s']):
                raise ValueError('Boot transition or bounded window exhausted')
            verify_linked_libraries(build_info['libraries_sha256'])
            readback = output / f'{index}_{mode}.bin'
            argv = [str((output/'native_binary').absolute()), mode,
                    str((args.work / 'fixture.bin').absolute()),
                    str(fixture['bytes']), str(readback.absolute()), authority['gpu_uuid'],
                    'on' if build_info['stats'] else 'off', '1' if phase == 'capability' else '5']
            watch = ArmWatch(arm=mode, round=index // 2 + 1, run=index + 1,
                             required=['suspend', 'power', 'foreign_reads'],
                             model_path=str(args.work))
            log_dir = output/'logs'/str(index)
            log_dir.mkdir(parents=True, exist_ok=False)
            watch.start()
            try:
                raw = run_command(argv, timeout=120, cwd=log_dir)
            finally:
                record = watch.stop()
            write_json(output / f'{index}_{mode}_raw.json', raw)
            write_json(output / f'{index}_{mode}_watch.json', record)
            row = {'mode': mode, 'returncode': raw.get('returncode'), 'watch': record,
                   'integrity': integrity(readback, fixture['sha256'], fixture['bytes'])}
            if raw.get('status') == 'COMPLETED' and not raw.get('truncated'):
                row['native'] = raw_record(raw['stdout']) or {}
                errors = [json.loads(line) for line in raw['stdout'].splitlines()
                          if line.startswith('{')
                          and json.loads(line).get('event') == 'CUFILE_ERROR']
                if len(errors) == 1:
                    row['native_error'] = errors[0]
            rows.append(row)
            write_json(output / f'{index}_{mode}_checked.json', row)
            validate_watch_peers(record)
            if raw.get('status') != 'COMPLETED':
                raise ValueError('Native process did not complete; no further arms')
            if mode == 'control' and (row['returncode'] != 0 or not row['integrity']):
                raise ValueError('Positive control failed; preserve and stop')
            if phase == 'performance' and index % 4 == 3:
                summary = performance_summary(rows, fixture['bytes'])
                if summary['status'] != 'MORE_BLOCKS_REQUIRED':
                    break
        summary = ({'status': classify(rows, fixture['bytes'])} if phase == 'capability'
                   else performance_summary(rows, fixture['bytes']))
        write_json(output / 'result.json', {**summary, 'rows': rows,
                                           'performance_study': phase == 'performance',
                                           'performance_claim': summary['status'] ==
                                           'PERFORMANCE_COMPARISON_COMPLETE',
                                           'scope': 'one declared mount and selected GPU only'})
    except BaseException as error:
        write_json(output / 'failure.json', {'error': repr(error), 'automatic_retry': False})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)
        for name, value in original_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        manifest(output)


def audit(output: Path) -> dict:
    pins = json.loads((output / 'files_sha256.json').read_text())
    actual_names = {str(path.relative_to(output)) for path in output.rglob('*')
                    if path.is_file() and path.name != 'files_sha256.json'}
    if set(pins) != actual_names:
        raise ValueError('Manifest coverage differs')
    for name, digest in pins.items():
        target = output / name
        if (Path(name).is_absolute() or '..' in Path(name).parts or target.is_symlink()
                or os.path.commonpath([os.path.realpath(output), os.path.realpath(target)])
                != os.path.realpath(output)
                or sha(target) != digest):
            raise ValueError('Preserved file differs: ' + name)
    fixture = json.loads((output / 'fixture.json').read_text())
    identity = json.loads((output / 'identity.json').read_text())
    verify_rule(identity['commit'], identity['rule_sha256'])
    if local_nvme_device(json.loads((output/'inventory.json').read_text())) != (
            identity['nvme_namespace']):
        raise ValueError('Preserved target topology differs')
    if fixture['sha256'] != expected_fixture_sha(fixture['bytes']):
        raise ValueError('Fixture oracle differs')
    build_info = json.loads((output / 'build.json').read_text())
    if (sha(output / 'native_binary') != identity['binary_sha256']
            or identity['binary_sha256'] != build_info['binary_sha256']
            or b'SYNTHETIC_CPU_MOCK' in (output / 'native_binary').read_bytes()):
        raise ValueError('Native binary provenance differs')
    for name in SOURCES:
        if (sha(REPO/name) != identity['source_sha256'][name]
                or not (output/'source'/name).is_file()
                or sha(output/'source'/name) != identity['source_sha256'][name]):
            raise ValueError('Source missing')
    if (sha(output/'source'/SOURCES[0]) != build_info['source_sha256']
            or sha(output/'source/benchmarks/g3/PROTOCOL.md') != identity['rule_sha256']):
        raise ValueError('Frozen rule/native source differs')
    if not check_controls(json.loads((output/'controls.json').read_text()), identity['boot_id']):
        raise ValueError('Calibration differs')
    rows = []
    runtime_pairs = set()
    cufile_versions = set()
    indices = sorted(int(path.name.split('_')[0]) for path in output.glob('*_raw.json'))
    if indices != list(range(len(indices))):
        raise ValueError('Run indices differ')
    # Recompute from raw process output and actual readback bytes, never trust checked/result JSON.
    for index in indices:
        mode = ['control', 'direct', 'direct', 'control'][index % 4]
        prefix = f'{index}_{mode}'
        raw = json.loads((output / (prefix + '_raw.json')).read_text())
        watch = json.loads((output / (prefix + '_watch.json')).read_text())
        validate_watch_peers(watch)
        if watch.get('gpu', {}).get('id') != identity['gpu_uuid']:
            raise ValueError('Watch GPU identity differs')
        row = {'mode': mode, 'watch': watch, 'returncode': raw.get('returncode'),
               'integrity': integrity(output / (prefix + '.bin'), fixture['sha256'],
                                      fixture['bytes'])}
        if raw.get('status') == 'COMPLETED' and not raw.get('truncated'):
            row['native'] = raw_record(raw['stdout']) or {}
            errors = [json.loads(line) for line in raw['stdout'].splitlines()
                      if line.startswith('{')
                          and json.loads(line).get('event') == 'CUFILE_ERROR']
            if len(errors) == 1:
                row['native_error'] = errors[0]
            native = row['native']
            if native:
                operations = 1 if identity['phase'] == 'capability' else 5
                if (type(native.get('read_operations')) is not int
                        or native['read_operations'] != operations
                        or type(native.get('cpu_s')) not in [int, float]
                        or not math.isfinite(native['cpu_s']) or native['cpu_s'] < 0
                        or native.get('explicit_pinned_host_bytes') != (
                            fixture['bytes'] if mode == 'control' else 0)
                        or any(type(native.get(key)) is not int or native[key] <= 0
                               for key in ['cuda_runtime', 'cuda_driver'])):
                    raise ValueError('Native runtime/operation/accounting evidence differs')
                runtime_pairs.add((native['cuda_runtime'], native['cuda_driver']))
                if mode == 'direct':
                    cufile_versions.add(native.get('cufile_version'))
            if native and (native.get('gpu_uuid') != identity['gpu_uuid']
                           or native.get('mode') != mode
                           or native.get('bytes') != fixture['bytes']):
                raise ValueError('Native identity/count differs')
        rows.append(row)
    if len(runtime_pairs) != 1 or len(cufile_versions) > 1:
        raise ValueError('Runtime changed between arms')
    result = (classify(rows, fixture['bytes']) if identity['phase'] == 'capability'
              else performance_summary(rows, fixture['bytes'])['status'])
    original = json.loads((output / 'result.json').read_text())
    if rows != original.get('rows'):
        raise ValueError('Recorded row derivation differs from raw evidence')
    if identity['phase'] == 'performance':
        expected = performance_summary(rows, fixture['bytes'])
        if any(original.get(key) != value for key, value in expected.items()):
            raise ValueError('Performance derivation differs')
    if result != original['status']:
        raise ValueError('Independent audit verdict differs')
    return {'status': result, 'local_raw_readback_audit': 'PASS',
            'performance_claim': result == 'PERFORMANCE_COMPARISON_COMPLETE'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    prepare_args = modes.add_parser('prepare')
    prepare_args.add_argument('--work', type=Path, required=True)
    prepare_args.add_argument('--bytes', type=int, default=LIMIT_BYTES)
    build_args = modes.add_parser('build')
    build_args.add_argument('--out', type=Path, required=True)
    build_args.add_argument('--cuda-root', type=Path, default=Path('/usr/local/cuda'))
    build_args.add_argument('--stats', choices=['on', 'off'], default='on')
    run_args = modes.add_parser('capability')
    for name in ['work', 'build', 'authority', 'controls']:
        run_args.add_argument('--' + name, type=Path, required=True)
    run_args.add_argument('--phase', choices=['capability', 'performance'], default='capability')
    run_args.add_argument('--commit', required=True)
    run_args.add_argument('--rule-sha', required=True)
    audit_args = modes.add_parser('audit')
    audit_args.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.work, args.bytes)
    elif args.mode == 'build':
        build(args.out, args.cuda_root, args.stats == 'on')
    elif args.mode == 'capability':
        run_capability(args)
    else:
        Console().print(json.dumps(audit(args.out), indent=2))


if __name__ == '__main__':
    main()
