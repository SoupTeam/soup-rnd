"""Offline rejection gate for a new G3 NVMe/nvidia-fs candidate; never executes CUDA."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from g3_probe import local_nvme_device
from rich.console import Console

# Conservative selection shortlist, not an exhaustive list of supported hardware.
GPU_SHORTLIST = {'A40', 'L4', 'L40S', 'A100'}


def completed_text(row: dict) -> str:
    if (row.get('status') != 'COMPLETED' or row.get('returncode') != 0
            or row.get('truncated') is not False):
        return ''
    return row.get('stdout', '')


def file_text(row: dict) -> str:
    if row.get('status') != 'READ' or row.get('truncated') is not False:
        return ''
    return row.get('text', '')


def assess(inventory: dict) -> dict:
    """Reject incomplete or unsuitable inventory; passing still requires live review."""
    reasons = []
    commands, files = inventory.get('commands', {}), inventory.get('files', {})
    if inventory.get('schema') != 'g3-inventory-v1' or inventory.get('system') != 'Linux':
        reasons.append('A complete Linux g3-inventory-v1 record is required')
    gpu_rows = list(csv.reader(completed_text(commands.get('gpu', {})).splitlines()))
    gpu_uuid = None
    if len(gpu_rows) != 1 or len(gpu_rows[0]) != 4:
        reasons.append('Exactly one attributable GPU is required')
    else:
        gpu_uuid, name, _, _ = [value.strip() for value in gpu_rows[0]]
        tokens = set(re.findall(r'[A-Za-z0-9]+', name))
        if not tokens.intersection(GPU_SHORTLIST) or re.search(r'GeForce|RTX', name, re.I):
            reasons.append('GPU is outside the conservative professional GPU shortlist')
        if not gpu_uuid.startswith('GPU-'):
            reasons.append('GPU UUID missing')
    device = None
    try:
        device = local_nvme_device(inventory)
        mount = json.loads(commands['mount']['stdout'])['filesystems'][0]
        if mount['fstype'] == 'ext4' and 'data=ordered' not in mount.get('options', '').split(','):
            reasons.append('EXT4 must explicitly report data=ordered')
    except (KeyError, ValueError, TypeError, IndexError):
        reasons.append('Target is not attributable to a supported local NVMe mount')
    modules = file_text(files.get('modules', {}))
    if not re.search(r'^nvidia_fs\s', modules, re.M):
        reasons.append('Loaded nvidia_fs transport is not evidenced')
    if not file_text(files.get('nvidia_fs_stats', {})).strip():
        reasons.append('Readable nvidia-fs counters are required')
    boot = file_text(files.get('boot_id', {})).strip()
    if not re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', boot):
        reasons.append('Boot identity missing')
    diagnostic = completed_text(commands.get('gdscheck', {}))
    nvme = re.findall(r'^\s*NVMe\s*:\s*(.+)$', diagnostic, re.M)
    if len(nvme) != 1 or not re.fullmatch(r'(?:Supported|nvfs(?:, compat)?)', nvme[0].strip()):
        reasons.append('gdscheck does not identify the selected NVMe nvidia-fs route')
    if not re.search(r'GPU index 0\b[^\n]*\bsupports GDS\b', diagnostic):
        reasons.append('gdscheck GPU 0 support diagnostic missing')
    environment = inventory.get('environment', {})
    if environment.get('CUFILE_FORCE_COMPAT_MODE') not in (None, '', '0', 'false'):
        reasons.append('Compatibility override is present')
    if environment.get('CUFILE_USE_PCIP2PDMA') not in (None, '', '0', 'false'):
        reasons.append('P2PDMA is outside this nvidia-fs-only selection gate')
    return {
        'schema': 'g3-readiness-v1',
        'status': 'REJECT_CANDIDATE' if reasons else 'CANDIDATE_REQUIRES_LIVE_REVIEW',
        'reasons': reasons, 'support_verdict': 'NO_VERDICT',
        'direct_path_verified': False, 'performance_measured': False,
        'execution_authorized': False, 'gpu_uuid': gpu_uuid, 'boot_id': boot,
        'device': device,
        'scope': 'Offline inventory screen; does not authenticate provenance or freshness. '
                 'Recollect on the selected host and review before controlled execution.',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    raw = args.inventory.read_bytes()
    if len(raw) > 2 * 1024**2:
        parser.error('Inventory exceeds 2 MiB')
    result = assess(json.loads(raw))
    result['inventory_sha256'] = hashlib.sha256(raw).hexdigest()
    result['gate_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    Console().print(result['status'] + '; support=NO_VERDICT; no execution authorized')


if __name__ == '__main__':
    main()
