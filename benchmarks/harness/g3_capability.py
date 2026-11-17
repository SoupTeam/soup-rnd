"""G3 phase 1: bounded, read-only inventory; never certifies a direct DMA path.

No torch/CUDA imports, fixture reads, package installation or driver changes.
The chosen data path must already exist. Outputs are exclusive and SHA-pinned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

from rich.console import Console

OUTPUT_LIMIT = 131072
COMMAND_TIMEOUT = 15
SCHEMA = 'g3-inventory-v1'


def run_command(
    argv: list[str], timeout: float = COMMAND_TIMEOUT, cwd: Path | None = None,
) -> dict:
    """Bound a diagnostic process group and retain raw output, including errors."""
    started = time.monotonic()
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            child = subprocess.Popen(
                argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                start_new_session=os.name == "posix", cwd=cwd,
            )
        except OSError as error:
            return {'argv': argv, 'status': 'UNAVAILABLE', 'error': str(error),
                    'elapsed_s': time.monotonic() - started}
        status = 'COMPLETED'
        try:
            child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            status = 'TIMEOUT'
            try:
                os.killpg(child.pid, signal.SIGKILL) if os.name == "posix" else child.kill()
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                status = 'TIMEOUT_UNREAPED'
        except BaseException:
            try:
                os.killpg(child.pid, signal.SIGKILL) if os.name == "posix" else child.kill()
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            raise
        streams = {}
        truncated = False
        for name, stream in [('stdout', stdout), ('stderr', stderr)]:
            stream.seek(0)
            data = stream.read(OUTPUT_LIMIT + 1)
            truncated |= len(data) > OUTPUT_LIMIT
            streams[name] = data[:OUTPUT_LIMIT].decode('utf-8', errors='replace')
        return {'argv': argv, 'status': status, 'returncode': child.returncode,
                'truncated': truncated, 'pid': child.pid, 'elapsed_s': time.monotonic() - started,
                **streams}


def read_small(path: Path) -> dict:
    """Retain bounded configuration evidence without loading native libraries."""
    try:
        with path.open('rb') as stream:
            raw = stream.read(OUTPUT_LIMIT + 1)
    except OSError as error:
        return {'path': str(path), 'status': 'UNAVAILABLE', 'error': str(error)}
    return {'path': str(path), 'status': 'READ', 'truncated': len(raw) > OUTPUT_LIMIT,
            'text': raw[:OUTPUT_LIMIT].decode('utf-8', errors='replace')}


def classify_inventory(system: str, commands: dict) -> dict:
    """Inventory is never sufficient for SUPPORTED or UNSUPPORTED verdicts."""
    if system != 'Linux':
        status, reasons = 'OUT_OF_SCOPE_PLATFORM', ['This collector targets Linux GDS only']
    else:
        reasons = []
        for name in ['gpu', 'mount', 'block_devices', 'gdscheck']:
            row = commands.get(name, {})
            if (row.get('status') != 'COMPLETED' or row.get('returncode') != 0
                    or row.get('truncated') is not False or not row.get('stdout', '').strip()):
                reasons.append(name + ': missing, failed, timed out, empty or truncated')
        status = 'ENVIRONMENT_BLOCKED' if reasons else 'CONTROLLED_PROBE_REQUIRED'
    return {'status': status, 'reasons': reasons, 'support_verdict': 'NO_VERDICT',
            'direct_path_verified': False, 'performance_measured': False,
            'note': 'Missing software, nvidia-fs absence and API success are not DMA verdicts.'}


def collect_inventory(
    data_path: Path,
    *,
    system: str | None = None,
    runner: Callable = run_command,
    which: Callable = shutil.which,
) -> dict:
    path = Path(os.path.realpath(data_path))
    if not path.exists() or not path.is_dir():
        raise ValueError('Data path must be an existing directory on the intended filesystem')
    system = system or platform.system()
    commands = {}
    files = {}
    if system == 'Linux':
        specs = {
            'gpu': ['nvidia-smi', '--query-gpu=uuid,name,driver_version,pci.bus_id',
                    '--format=csv,noheader'],
            'mount': ['findmnt', '--json', '--target', str(path),
                      '--output', 'SOURCE,TARGET,FSTYPE,OPTIONS,MAJ:MIN'],
            'block_devices': ['lsblk', '--json', '--output',
                              'NAME,KNAME,TYPE,TRAN,SIZE,ROTA,MAJ:MIN,PKNAME,MOUNTPOINTS'],
            'pci_topology': ['lspci', '-Dnn'],
            'gpu_topology': ['nvidia-smi', 'topo', '-m'],
        }
        gdscheck = which('gdscheck') or which('gdscheck.py')
        if not gdscheck:
            for candidate in [Path('/usr/local/cuda/gds/tools/gdscheck'),
                              Path('/usr/local/cuda/gds/tools/gdscheck.py')]:
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    gdscheck = str(candidate)
                    break
        if gdscheck:
            specs['gdscheck'] = [gdscheck, '-p']
        for name, argv in specs.items():
            commands[name] = runner(argv)
        config = os.environ.get('CUFILE_ENV_PATH_JSON') or '/etc/cufile.json'
        files = {name: read_small(Path(filename)) for name, filename in {
            'boot_id': '/proc/sys/kernel/random/boot_id',
            'modules': '/proc/modules', 'cufile_config': config,
            'nvidia_fs_stats': '/proc/driver/nvidia-fs/stats',
        }.items()}
    return {'schema': SCHEMA, 'started_as': 'READ_ONLY_INVENTORY', 'unix_s': time.time(),
            'system': system, 'kernel': platform.release(), 'machine': platform.machine(),
            'data_path': str(path), 'commands': commands, 'files': files,
            'environment': {key: os.environ.get(key) for key in [
                'CUFILE_FORCE_COMPAT_MODE', 'CUFILE_USE_PCIP2PDMA', 'CUFILE_ENV_PATH_JSON']},
            'classification': classify_inventory(system, commands),
            'limitations': ['Mount/block/GPU topology still requires review for the target path.',
                            'No model or fixture was loaded; no throughput was measured.',
                            'No native cuFile probe or positive control has run.']}


def write_bundle(output: Path, report: dict) -> None:
    """Exclusive bundle, with the manifest computed from file readbacks."""
    output.mkdir(parents=True, exist_ok=False)
    payloads = {
        'inventory.json': (json.dumps(report, indent=2) + '\n').encode(),
        'collector.py': Path(__file__).read_bytes(),
    }
    hashes = {}
    for name, raw in payloads.items():
        with (output / name).open('xb') as stream:
            stream.write(raw)
        stored = (output / name).read_bytes()
        if stored != raw:
            raise OSError('Bundle readback differs: ' + name)
        hashes[name] = hashlib.sha256(stored).hexdigest()
    with (output / 'files_sha256.json').open('x') as stream:
        json.dump(hashes, stream, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output exists; inventory evidence cannot be overwritten')
    report = collect_inventory(args.data_path)
    write_bundle(args.out, report)
    Console().print(report['classification']['status'] + '; support=NO_VERDICT')


if __name__ == '__main__':
    main()
