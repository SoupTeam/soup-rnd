"""Read-only host admission for one explicitly declared NVML infrastructure recovery."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

from boundary_confirmation import read_contract
from policy_pilot_support import digest, verify_runtime
from run_support import contained

REPO = Path(__file__).parents[2]


def read_declaration(path: Path, expected_sha: str) -> tuple[dict, dict]:
    if digest(path) != expected_sha:
        raise ValueError('Recovery declaration SHA differs')
    declaration = json.loads(path.read_text())
    if (declaration['scope'] != 'one-nvml-infrastructure-recovery'
            or declaration['new_series_attempts'] != 1
            or declaration['cumulative_budget_usd'] != 5):
        raise ValueError('Only one separately declared recovery within cumulative5USD')
    for name, sha in declaration['source_sha256'].items():
        if digest(contained(REPO, name)) != sha:
            raise ValueError('Recovery source differs: ' + name)
    base = declaration['base_contract']
    spec, recipe, _ = read_contract(argparse.Namespace(
        spec=contained(REPO, base['path']), spec_sha=base['sha256']))
    if declaration['model_order'] != spec['model_order']:
        raise ValueError('Recovery cannot alter model order')
    return declaration, recipe


def verify_prior(archive: Path, declaration: dict) -> None:
    prior = declaration['prior_failure']
    if (archive.stat().st_size != prior['archive_bytes']
            or digest(archive) != prior['archive_sha256']):
        raise ValueError('Preserved failed attempt archive differs')
    with tarfile.open(archive, 'r:gz') as handle:
        names = handle.getnames()
        if len(set(names)) != len(names):
            raise ValueError('Duplicate archived paths')
        if any(re.search(r'/(?:diagnostic|confirmation)_(?:original|boundary)\.jsonl$', n)
               for n in names):
            raise ValueError('Recovery exception requires zero generation files')
        for name, sha in prior['required_members_sha256'].items():
            member = handle.getmember(name)
            if not member.isfile() or member.size > 1024**2:
                raise ValueError('Invalid failed-attempt evidence member')
            with handle.extractfile(member) as source:
                raw = source.read()
            if hashlib.sha256(raw).hexdigest() != sha:
                raise ValueError('Failed-attempt evidence differs')
        with handle.extractfile('runs/initial/failure.json') as source:
            failure = json.load(source)
        with handle.extractfile('runs/initial.launcher.json') as source:
            launcher = json.load(source)
    if ('Driver/library version mismatch' not in failure['error']
            or failure['retry_allowed'] is not False
            or launcher['exitcode'] != 1 or launcher['hard_timeout'] is not False
            or launcher['spec_sha256'] != declaration['base_contract']['sha256']):
        raise ValueError('Exception covers only the pinned pre-generation NVML failure')


def validate_host(snapshot: dict, declaration: dict) -> None:
    if hashlib.sha256(snapshot['boot_id'].encode()).hexdigest() == (
            declaration['prior_failure']['boot_id_sha256']):
        raise ValueError('A new boot after the driver update is required')
    if hashlib.sha256(snapshot['gpu_uuid'].encode()).hexdigest() != (
            declaration['target_gpu_uuid_sha256']):
        raise ValueError('Different GPU is outside this recovery')
    expected = declaration['expected_driver_version']
    if any(snapshot[name] != expected for name in ['loaded_driver', 'module_on_disk',
                                                  'nvml_reported_driver']):
        raise ValueError('Loaded driver, installed module and NVML must agree')
    if snapshot['package_writers'] or snapshot['dpkg_audit'].strip():
        raise ValueError('Package update still running or package state incomplete')


def validate_controls(directory: Path, boot_id: str, declaration: dict) -> dict:
    if (digest(directory.parent / 'calibrate_guest_disk_v2.py') !=
            declaration['calibration_source_sha256']
            or digest(directory.parent / 'arm_validity.py') != declaration['harness_sha256']):
        raise ValueError('Control source differs')
    controls = json.loads((directory / 'calibration.json').read_text())
    ready = json.loads((directory / 'foreign_ready.json').read_text())
    done = json.loads((directory / 'foreign_done.json').read_text())
    if (controls['status'] != 'PASS_GUEST_CONTROLS_ONLY'
            or controls['harness_sha256'] != declaration['harness_sha256']
            or ready['boot_id'] != boot_id or ready['watcher_pid'] != done['watcher_pid']
            or done != controls['foreign_reader']
            or done['watcher_pid'] in done['ancestors']
            or done['started_unix_s'] < ready['unix_s']):
        raise ValueError('Fresh independent controls on this boot required')
    for name, expected in [('clean', 'ok'), ('own', 'ok'), ('foreign', 'void')]:
        row = controls['controls'][name]
        if (row['outcome'] != expected or row['checks']['foreign_reads']['outcome'] != expected
                or row['gpu_clock']['outcome'] != 'ok'
                or any(row['checks'][key]['outcome'] != 'ok' for key in ['suspend', 'power'])):
            raise ValueError('F1 or clocks control failed')
    return {name: digest(directory / name) for name in [
        'calibration.json', 'foreign_ready.json', 'foreign_done.json']}


def host_snapshot(uuid: str) -> dict:
    def command(argv: list[str]) -> str:
        return subprocess.run(argv, capture_output=True, text=True, check=True,
                              timeout=20).stdout.strip()

    writers = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            words = (process / 'cmdline').read_bytes().split(b'\0')[:2]
        except FileNotFoundError:
            continue
        if any(os.fsdecode(word).rsplit('/', 1)[-1] in {
                'apt', 'apt-get', 'dpkg', 'unattended-upgrade', 'unattended-upgrades'}
               for word in words):
            writers.append(int(process.name))
    nvml = command(['nvidia-smi', '--id=' + uuid, '--query-gpu=uuid,driver_version',
                    '--format=csv,noheader']).split(',')
    loaded = re.search(r'Kernel Module.*?\b(\d{3,4}\.\d+\.\d+)\b',
                       Path('/proc/driver/nvidia/version').read_text())
    if loaded is None or len(nvml) != 2:
        raise ValueError('Cannot establish driver identity')
    return {'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'gpu_uuid': nvml[0].strip(), 'nvml_reported_driver': nvml[1].strip(),
            'loaded_driver': loaded.group(1),
            'module_on_disk': command(['modinfo', '-F', 'version', 'nvidia']),
            'dpkg_audit': command(['dpkg', '--audit']), 'package_writers': writers}


def main() -> None:
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'host'])
    parser.add_argument('--declaration', type=Path, required=True)
    parser.add_argument('--declaration-sha', required=True)
    for name in ['prior-archive', 'controls', 'authority', 'out']:
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--authority-sha')
    args = parser.parse_args()
    declaration, recipe = read_declaration(args.declaration, args.declaration_sha)
    if args.mode == 'check':
        if args.prior_archive:
            verify_prior(args.prior_archive, declaration)
        Console().print('RECOVERY_SOURCE_AND_PRIOR_EVIDENCE_CHECK_PASS; no GPU operation')
        return
    if any(getattr(args, name) is None for name in [
            'prior_archive', 'controls', 'authority', 'authority_sha', 'out']):
        parser.error('Recovery authority, old archive, fresh controls and unique out required')
    config = json.loads(args.authority.read_text())
    if (sys.platform != 'linux' or digest(args.authority) != args.authority_sha
            or config.get('recovery_execution_authorized') is not True
            or config.get('recovery_contract_sha256') != args.declaration_sha
            or config.get('instance_id') != declaration['target_instance_id']
            or config.get('total_cumulative_cap_usd') != 5
            or os.environ.get('SOUP_G6_INSTANCE_ID') != str(config['instance_id'])
            or os.environ.get('CUDA_VISIBLE_DEVICES') != config['gpu_uuid']):
        raise ValueError('New current-host recovery execution authority required')
    old, new = [os.path.realpath(config[key]) for key in ['original_runs_root', 'runs_root']]
    if (not all(Path(config[key]).is_absolute() for key in ['original_runs_root', 'runs_root'])
            or os.path.commonpath([old, new]) in [old, new]):
        raise ValueError('Separate non-overlapping absolute roots required; preserve old claims')
    destination = contained(Path(new), str(args.out))
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The receipt itself is an exclusive claim; retain failed preflights without retry.
    with destination.open('x') as target:
        result = {'status': 'BLOCKED_PREFLIGHT', 'recovery_sha256': args.declaration_sha,
                  'authority_sha256': args.authority_sha, 'generation': False,
                  'physical_host_isolation_certified': False, 'quality_verdict': 'NO_VERDICT'}
        try:
            verify_prior(args.prior_archive, declaration)
            snapshot = host_snapshot(config['gpu_uuid'])
            validate_host(snapshot, declaration)
            verify_runtime(recipe)
            result['controls_sha256'] = validate_controls(
                args.controls, snapshot['boot_id'], declaration)
            result.update(status='PASS_RECOVERY_PREFLIGHT_ONLY', host=snapshot)
        except Exception as error:
            result['error'] = repr(error)
            raise
        finally:
            json.dump(result, target, indent=2)
            target.write('\n')
    Console().print('PASS_RECOVERY_PREFLIGHT_ONLY; model generation not launched')


if __name__ == '__main__':
    main()
