"""Prospective fresh two-arm attempt after a preserved user pause; no optimizer resume."""
from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from policy_continuation import load_specs as load_parent
from policy_continuation import worker
from policy_pilot_support import digest
from reload_followup import verify_files, verify_phases
from run import write
from run_support import contained

REPO = Path(__file__).parents[2]
ARMS = ['length_curriculum', 'reverse_length_curriculum']


def verify_series(series: Path, spec: dict, parent: dict, inventory: dict) -> None:
    proof = spec['series']
    verify_files(series, proof['files_sha256'])
    completed = series / 'pilot_coherence_filter_001'
    paused = series / 'pilot_length_curriculum_001'
    for root, arm in [(completed, 'coherence_filter'), (paused, 'length_curriculum')]:
        required = {str(root.relative_to(series) / name) for name in [
            'prospective_run.json', 'inventory.json', arm + '/actual_trace.jsonl']}
        required.add(root.name + '.launcher.json')
        required.update(str(path.relative_to(series)) for path in root.rglob('*')
                        if path.is_file())
        if not required.issubset(proof['files_sha256']):
            raise ValueError('Series manifest omits required evidence')
        meta = json.loads((root / 'prospective_run.json').read_text())
        if (meta['contract_sha'] != spec['parent']['sha256'] or meta['arm'] != arm
                or meta['source_baseline_end_to_end_timing_validated'] is not False
                or json.loads((root / 'inventory.json').read_text()) != inventory):
            raise ValueError('Preserved series provenance differs')
    outer = json.loads((series / (completed.name + '.launcher.json')).read_text())
    if (outer['exitcode'] != 0 or outer['hard_timeout'] is not False
            or outer['contract_sha256'] != spec['parent']['sha256'] or outer['bound_s'] != 4800):
        raise ValueError('Coherence supervisor incomplete')
    receipt = json.loads((completed / 'coherence_filter/receipt.json').read_text())
    if (receipt['status'] != 'ARM_COMPLETED_REVIEW_PENDING'
            or receipt['training']['actual_useful_tokens'] != 65536
            or receipt['training']['applied_optimizer_updates'] != 8
            or receipt['training']['AMP_exercised'] is not True
            or receipt['reload_exact'] is not True or receipt['F1_timing_accepted'] is not True
            or receipt['initial_state_sha256'] != parent['expected_initial_state_sha256']):
        raise ValueError('Coherence is not an accepted completed source')
    verify_phases(completed / 'coherence_filter')
    reload_phase = completed / 'coherence_filter/reload_diagnostics_phase.json'
    check = json.loads(reload_phase.read_text())['F1']
    if check['outcome'] != 'ok' or check['gpu_clock']['outcome'] != 'ok':
        raise ValueError('Coherence reload phase rejected')
    outer = json.loads((series / (paused.name + '.launcher.json')).read_text())
    if (outer['exitcode'] != -15 or outer['hard_timeout'] is not False
            or outer['contract_sha256'] != spec['parent']['sha256'] or outer['bound_s'] != 4800):
        raise ValueError('Expected preserved user pause')
    directory = paused / 'length_curriculum'
    forbidden = [paused / 'receipt.json', directory / 'receipt.json',
                 directory / 'checkpoint.json', directory / 'adapter.safetensors',
                 directory / 'peft_adapter']
    if any(path.exists() for path in forbidden):
        raise ValueError('Paused attempt must have no completed receipt or saved adapter')
    trace = directory / 'actual_trace.jsonl'
    steps = [json.loads(line) for line in trace.read_text().splitlines()]
    updates = [row for row in steps if row['event'] == 'optimizer_step']
    if (len(updates) != 8 or [row['update'] for row in updates] != list(range(1, 9))
            or any(row['actual_useful_tokens'] != 8192 for row in updates)
            or updates[-1]['consumed_useful_tokens'] != 65536):
        raise ValueError('Paused training exposure differs')
    for name, count in [('initial_eval', 256), ('initial_diagnostics', 24), ('final_eval', 15)]:
        if len((directory / (name + '.jsonl')).read_text().splitlines()) != count:
            raise ValueError('Paused evaluation boundary differs')
    for name in ['initial_eval', 'initial_diagnostics', 'train']:
        check = json.loads((directory / (name + '_phase.json')).read_text())['F1']
        if check['outcome'] != 'ok' or check['gpu_clock']['outcome'] != 'ok':
            raise ValueError('Completed paused phase rejected')


def load_specs(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    if digest(args.spec) != args.spec_sha:
        raise ValueError('Recovery contract SHA differs')
    spec = json.loads(args.spec.read_text())
    if (spec.get('scope') != 'two-pilots-new-attempt-after-user-pause'
            or spec.get('arm_order') != ARMS or spec.get('optimizer_resume') is not False
            or spec.get('replace_old_receipts') is not False
            or spec.get('full_training_authorized') is not False):
        raise ValueError('Only one prospective fresh two-arm attempt is allowed')
    verify_files(REPO, spec['source_sha256'])
    old_args = copy.copy(args)
    old_args.spec = contained(REPO, spec['parent']['path'])
    old_args.spec_sha = spec['parent']['sha256']
    old_spec, parent, inventory = load_parent(old_args)
    verify_series(args.series, spec, parent, inventory)
    parent = copy.deepcopy(parent)
    parent['arm_order'] = ARMS
    parent['launch_contract_sha256'] = args.spec_sha
    # The unchanged worker records the already pinned baseline reload follow-up.
    spec['verification'] = old_spec['verification']
    return spec, parent, inventory


def main() -> None:
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'pilot'])
    for name in ['spec', 'reference', 'verification', 'series', 'artifact-root',
                 'data', 'cuda-gate', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--spec-sha', required=True)
    for name in ['launch-config', 'admission', 'model-dir']:
        parser.add_argument('--' + name, type=Path)
    for name in ['launch-config-sha', 'protocol-commit']:
        parser.add_argument('--' + name)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    spec, parent, inventory = load_specs(args)
    if args.mode == 'check':
        Console().print('RECOVERY_REFERENCE_AUDIT_PASS; no GPU or training')
        return
    if any(getattr(args, name) is None for name in ['launch_config', 'admission', 'model_dir',
                                                   'launch_config_sha', 'protocol_commit', 'arm']):
        parser.error('Private stage authority, model, pushed commit and arm required')
    for root in [args.series, args.reference, args.verification]:
        real_root, real_out = os.path.realpath(root), os.path.realpath(args.out)
        if os.path.commonpath([real_root, real_out]) in [real_root, real_out]:
            parser.error('New output must be separate from preserved evidence')
    if args.worker:
        worker(args, spec, parent, inventory)
        return
    if args.out.exists():
        parser.error('Existing output forbids retry')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       G6_CONTINUATION_EXTERNAL_BOUND_S='4800')
    timed_out = False
    with Path(str(args.out) + '.launcher.log').open('x') as log:
        child = subprocess.Popen([sys.executable, __file__, *sys.argv[1:], '--worker'],
                                 env=environment, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            status = child.wait(timeout=4830)
        except BaseException as error:
            timed_out = isinstance(error, subprocess.TimeoutExpired)
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
            if not timed_out:
                raise
            status = child.returncode
    write(Path(str(args.out) + '.launcher.json'), {
        'exitcode': status, 'hard_timeout': timed_out, 'bound_s': 4800,
        'supervisor_grace_s': 40, 'contract_sha256': args.spec_sha})
    if status or timed_out:
        raise SystemExit(status or 1)


if __name__ == '__main__':
    main()
