"""Private execution authority and immutable sequential claims for a fresh pilot."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from importlib.metadata import version
from pathlib import Path

from portable_generation_probe import validate_launch
from run_support import contained, runtime

REPO = Path(__file__).parents[2]
ARMS = ['baseline', 'coherence_filter', 'length_curriculum', 'reverse_length_curriculum']


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path, expected: str, artifact_root: Path) -> dict:
    if digest(path) != expected:
        raise ValueError('Science contract SHA differs')
    spec = json.loads(path.read_text())
    if (spec.get('scope') != 'fresh-matched-policy-pilot-only'
            or spec['arm_order'] != ARMS or spec['limits']['pilot_train_targets'] != 65536
            or spec['recipe']['targets_per_update'] != 8192
            or spec['data']['budget_per_arm'] != 5_000_000):
        raise ValueError('Not the fixed four-policy pilot contract')
    for name, sha in spec['source_sha256'].items():
        if digest(contained(REPO, name)) != sha:
            raise ValueError('Frozen source differs: ' + name)
    for name, sha in spec['artifact_sha256'].items():
        if digest(contained(artifact_root, name)) != sha:
            raise ValueError('Frozen artifact differs: ' + name)
    wheels = spec['runtime_wheels']
    if digest(contained(REPO, wheels['path'])) != wheels['sha256']:
        raise ValueError('Pinned wheel lock differs')
    for item in (spec['suite'], spec['input_plan']):
        if digest(contained(REPO, item['path'])) != item['sha256']:
            raise ValueError('Frozen public suite or input plan differs')
    spec['launch_contract_sha256'] = expected
    return spec


def load_authority(path: Path, expected: str, science_sha: str) -> dict:
    if digest(path) != expected:
        raise ValueError('Private execution configuration SHA differs')
    launch = json.loads(path.read_text())
    validate_launch(launch)
    if not isinstance(launch.get('runs_root'), str) or not Path(launch['runs_root']).is_absolute():
        raise ValueError('Explicit absolute private runs root required')
    if launch.get('contract_sha256') != science_sha:
        raise ValueError('Private authority belongs to another science contract')
    return launch


def bound(spec: dict, mode: str) -> int:
    ceiling = {'cuda-gate': 600, 'pilot': 4800}
    if mode not in ceiling:
        raise ValueError('Only a numerical gate and pilot are authorized here')
    value = spec['limits'][mode + '_total_s']
    if type(value) is not int or not 0 < value <= ceiling[mode]:
        raise ValueError('Invalid bounded stage duration')
    if spec.get('supervisor_grace_s') != 40:
        raise ValueError('Supervisor grace differs')
    return value


def admit(receipt: dict, spec: dict, launch: dict, launch_sha: str,
          mode: str, arm: str | None, now: float) -> None:
    validate_launch(launch)
    stage_bound = bound(spec, mode)
    if (receipt.get('account') != launch['account'] or receipt.get('mode') != mode
            or receipt.get('instance_id') != launch['instance_id']
            or receipt.get('gpu_uuid') != launch['gpu_uuid']
            or receipt.get('contract_sha256') != spec['launch_contract_sha256']
            or launch.get('contract_sha256') != spec['launch_contract_sha256']
            or receipt.get('launch_config_sha256') != launch_sha
            or receipt.get('authorized_for_this_stage') is not True):
        raise ValueError('Explicit account, stage and private configuration binding required')
    if mode == 'pilot':
        if arm not in spec['arm_order'] or receipt.get('arm') != arm:
            raise ValueError('Stage admission names another arm')
        remaining = len(spec['arm_order']) - spec['arm_order'].index(arm)
        seconds = remaining * (stage_bound + 40)
    else:
        seconds = stage_bound + 40 + len(spec['arm_order']) * (bound(spec, 'pilot') + 40)
    for key in ('observed_unix_s', 'balance_usd', 'total_account_rate_usd_h'):
        value = receipt.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Invalid actual Billing number')
    age = now - receipt['observed_unix_s']
    rate = receipt['total_account_rate_usd_h']
    if not 0 <= age <= 300 or rate <= 0:
        raise ValueError('Fresh actual Billing and positive account rate required')
    required = 1.25 * rate * (seconds + 300 + age) / 3600 + .5
    if receipt['balance_usd'] - launch['account_floor_usd'] <= required:
        raise ValueError('Insufficient funds for remaining pilot series plus preservation reserve')


def verify_runtime(spec: dict) -> None:
    if sys.version.split()[0] != spec['python'] or runtime() != spec['runtime_target']:
        raise ValueError('Exact interpreter/core versions required')
    if any(version(name) != expected
           for name, expected in spec['runtime_wheels']['packages'].items()):
        raise ValueError('A pinned package version differs')


def preceding(claims: Path, spec: dict, arm: str, launch_sha: str) -> str:
    if arm not in spec['arm_order']:
        raise ValueError('Unknown arm')
    if (claims / ('pilot_' + arm + '.json')).exists():
        raise ValueError('An existing claim forbids retry or redeployment')
    expected = spec['expected_initial_state_sha256']
    for previous in spec['arm_order'][:spec['arm_order'].index(arm)]:
        claim = json.loads((claims / ('pilot_' + previous + '.json')).read_text())
        if claim['launch_config_sha256'] != launch_sha:
            raise ValueError('Private configuration changed between arms')
        out = Path(claim['out'])
        if (out / 'failure.json').exists():
            raise ValueError('Preceding arm failed; no subsequent arm may start')
        external = json.loads(Path(str(out) + '.launcher.json').read_text())
        if (external.get('exitcode') != 0 or external.get('hard_timeout') is not False
                or external.get('contract_sha256') != spec['launch_contract_sha256']
                or external.get('bound_s') != bound(spec, 'pilot')):
            raise ValueError('Preceding supervisor failed or did not complete')
        files = json.loads((out / 'files_sha256.json').read_text())
        receipt_name = previous + '/receipt.json'
        if receipt_name not in files:
            raise ValueError('Missing preserved arm receipt')
        for name, sha in files.items():
            if digest(contained(out, name)) != sha:
                raise ValueError('Preceding evidence was changed')
        receipt = json.loads((out / receipt_name).read_text())
        if (receipt['status'] != 'ARM_COMPLETED_REVIEW_PENDING'
                or receipt['F1_timing_accepted'] is not True
                or receipt['initial_state_sha256'] != expected):
            raise ValueError('Preceding arm lacks matched identity or accepted F1 timing')
    return expected
