"""Prospective three-arm continuation linked to a separately verified paused baseline."""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from consumer import source_rows
from policy_pilot_support import (
    admit,
    bound,
    digest,
    load_authority,
    preceding,
    verify_runtime,
)
from prepare import take_budget
from reload_followup import compare_outputs, evidence, verify_files
from run import execute_arm, write
from run_support import Guard, contained, gpu_inventory, verify_model

REPO = Path(__file__).parents[2]
ARMS = ['coherence_filter', 'length_curriculum', 'reverse_length_curriculum']


def linked_verification(reference: Path, verification: Path, spec: dict) -> dict:
    proof = spec['verification']
    required = {'verification_001/receipt.json', 'verification_001.launcher.json',
                'verification_001/resumed_reload_diagnostics_phase.json',
                'verification_001/reload_diagnostics.jsonl'}
    if not required.issubset(proof['files_sha256']):
        raise ValueError('Verification manifest omits required proof')
    verify_files(verification, proof['files_sha256'])
    root = verification / 'verification_001'
    receipt = json.loads((root / 'receipt.json').read_text())
    outer = json.loads((verification / 'verification_001.launcher.json').read_text())
    if (receipt.get('status') != 'VERIFIED_RELOAD_DIAGNOSTICS_ONLY'
            or receipt.get('cases') != 24 or receipt.get('new_training') is not False
            or receipt.get('optimizer_resume') is not False
            or receipt.get('original_attempt_status') != 'USER_PAUSED_INCOMPLETE'
            or receipt.get('original_end_to_end_timing_validated') is not False
            or receipt.get('remaining_pilots_unlocked') is not False
            or receipt.get('contract_sha256') != proof['sha256']
            or receipt.get('launch_config_sha256') != proof['launch_config_sha256']):
        raise ValueError('Successful verification-only linked proof required')
    if (outer.get('exitcode') != 0 or outer.get('hard_timeout') is not False
            or outer.get('contract_sha256') != proof['sha256'] or outer.get('bound_s') != 2400):
        raise ValueError('Verification supervisor did not complete')
    check = json.loads((root / 'resumed_reload_diagnostics_phase.json').read_text())['F1']
    if check.get('outcome') != 'ok' or check.get('gpu_clock', {}).get('outcome') != 'ok':
        raise ValueError('Verification F1/clocks rejected')
    old = reference / 'pilot_baseline_001/baseline'
    ids = [json.loads(line)['id']
           for line in (old / 'final_diagnostics.jsonl').read_text().splitlines()]
    if len(ids) != 24:
        raise ValueError('All original diagnostic cases required')
    compare_outputs(old / 'final_diagnostics.jsonl', old / 'reload_diagnostics.jsonl',
                    root / 'reload_diagnostics.jsonl', ids)
    return receipt['inventory']


def load_specs(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    if digest(args.spec) != args.spec_sha:
        raise ValueError('Continuation contract SHA differs')
    spec = json.loads(args.spec.read_text())
    if (spec.get('scope') != 'remaining-three-pilots-after-user-pause'
            or spec.get('arm_order') != ARMS or spec.get('baseline_original_completed') is not False
            or spec.get('baseline_end_to_end_timing_validated') is not False):
        raise ValueError('Only explicit three-arm continuation scope allowed')
    verify_files(REPO, spec['source_sha256'])
    parent_args = copy.copy(args)
    parent_args.spec = contained(REPO, spec['verification']['path'])
    parent_args.spec_sha = spec['verification']['sha256']
    _, original = evidence(parent_args)
    inventory = linked_verification(args.reference, args.verification, spec)
    gate = json.loads(args.cuda_gate.read_text())
    if (digest(args.cuda_gate) != spec['training_gate_sha256']
            or gate.get('status') != 'PASS_TINY_CUDA_NUMERICAL_GATE'
            or gate.get('contract_sha256') != original['launch_contract_sha256']
            or gate.get('runtime') != original['runtime_target']
            or gate.get('inventory') != inventory):
        raise ValueError('Preserved numerical gate differs from verified setup')
    derived = copy.deepcopy(original)
    derived['arm_order'] = ARMS
    derived['launch_contract_sha256'] = args.spec_sha
    return spec, derived, inventory


def worker(args: argparse.Namespace, spec: dict, parent: dict, expected_inventory: dict) -> None:
    launch = load_authority(args.launch_config, args.launch_config_sha, args.spec_sha)
    admission = json.loads(args.admission.read_text())
    admit(admission, parent, launch, args.launch_config_sha, 'pilot', args.arm, time.time())
    if (os.environ.get('G6_CONTINUATION_EXTERNAL_BOUND_S') != '4800'
            or os.environ.get('SOUP_G6_INSTANCE_ID') != str(launch['instance_id'])
            or os.environ.get('CUDA_VISIBLE_DEVICES') != launch['gpu_uuid']
            or not re.fullmatch('[0-9a-f]{40}', args.protocol_commit)):
        raise ValueError('Approved bounded supervisor, host/GPU and verified commit required')
    verify_runtime(parent)
    inventory = gpu_inventory(launch['gpu_uuid'])
    if inventory != expected_inventory:
        raise ValueError('Inventory changed from source and linked verification')
    runs = Path(launch['runs_root'])
    destination = contained(runs, str(args.out))
    if destination.exists():
        raise ValueError('Existing output forbids retry')
    claims = contained(runs, '.claims/' + args.spec_sha)
    claims.mkdir(parents=True, exist_ok=True)
    expected = preceding(claims, parent, args.arm, args.launch_config_sha)
    write(claims / ('pilot_' + args.arm + '.json'), {
        'out': str(destination), 'launch_config_sha256': args.launch_config_sha,
        'protocol_commit': args.protocol_commit})
    destination.mkdir(parents=True, exist_ok=False)
    write(destination / 'prospective_run.json', {
        'mode': 'pilot', 'arm': args.arm, 'contract_sha': args.spec_sha,
        'launch_config_sha256': args.launch_config_sha, 'protocol_commit': args.protocol_commit,
        'source_baseline_status': 'USER_PAUSED_INCOMPLETE',
        'source_baseline_end_to_end_timing_validated': False,
        'linked_verification_contract_sha256': spec['verification']['sha256'],
        'fresh_original_model_and_adapter': True})
    try:
        import torch

        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        guard = Guard(destination, bound(parent, 'pilot'), gpu=True, uuid=launch['gpu_uuid'])
        guard()
        write(destination / 'model_verification.json',
              verify_model(args.model_dir, parent['model_files'], guard))
        write(destination / 'inventory.json', inventory)
        write(destination / 'admission.json', admission)
        rows = take_budget(list(source_rows(args.data / (args.arm + '.jsonl'))), 65536)
        evaluation = list(source_rows(args.data / 'eval.jsonl'))
        suite = json.loads(contained(REPO, parent['suite']['path']).read_text())
        directory = destination / args.arm
        directory.mkdir(exist_ok=False)
        result = execute_arm(parent, rows, evaluation, directory, guard,
                             model_dir=args.model_dir, synthetic=False,
                             budget=65536, lr_budget=5_000_000, arm=args.arm,
                             suite=suite, expected_initial=expected)
        if not result['F1_timing_accepted']:
            raise ValueError('F1 timing rejected; preserve and block continuation')
        write(destination / 'receipt.json', {'status': 'COMPLETED_REVIEW_PENDING',
              'arms': {args.arm: result}, 'quality_verdict': 'NO_VERDICT'})
        write(destination / 'files_sha256.json', {
            str(path.relative_to(destination)): digest(path)
            for path in sorted(destination.rglob('*')) if path.is_file()})
    except Exception as error:
        write(destination / 'failure.json', {'error': repr(error), 'no_retry': True})
        raise


def review(args: argparse.Namespace, spec: dict, parent: dict, inventory: dict) -> dict:
    rows = {}
    baseline = args.reference / 'pilot_baseline_001/baseline'
    initial_ce = None
    launch_hashes = set()
    for root in args.runs:
        if (root / 'failure.json').exists():
            raise ValueError('Failed continuation arm')
        manifest = json.loads((root / 'files_sha256.json').read_text())
        verify_files(root, manifest)
        meta = json.loads((root / 'prospective_run.json').read_text())
        arm = meta['arm']
        if not {'receipt.json', 'prospective_run.json', 'inventory.json',
                arm + '/receipt.json'}.issubset(manifest):
            raise ValueError('Continuation manifest omits required proof')
        launch_hashes.add(meta['launch_config_sha256'])
        if arm not in ARMS or arm in rows or meta['contract_sha'] != args.spec_sha:
            raise ValueError('Unknown, duplicate or mismatched continuation arm')
        if meta['source_baseline_end_to_end_timing_validated'] is not False:
            raise ValueError('Interrupted original cannot gain an end-to-end timing')
        outer = json.loads(Path(str(root) + '.launcher.json').read_text())
        if (outer['exitcode'] != 0 or outer['hard_timeout'] is not False
                or outer['contract_sha256'] != args.spec_sha or outer['bound_s'] != 4800):
            raise ValueError('Continuation supervisor incomplete')
        if json.loads((root / 'inventory.json').read_text()) != inventory:
            raise ValueError('Continuation inventory differs')
        value = json.loads((root / arm / 'receipt.json').read_text())
        if (value['status'] != 'ARM_COMPLETED_REVIEW_PENDING'
                or value['training']['actual_useful_tokens'] != 65536
                or value['training']['AMP_exercised'] is not True
                or value['reload_exact'] is not True or value['F1_timing_accepted'] is not True
                or value['initial_state_sha256'] != parent['expected_initial_state_sha256']):
            raise ValueError('Actual exposure, AMP, reload, timing or initialization differs')
        ce = value['initial_eval']['mean_ce']
        if initial_ce is not None and ce != initial_ce:
            raise ValueError('Initial eval differs across continuation arms')
        initial_ce = ce
        rows[arm] = {'final_ce': value['final_eval']['mean_ce'],
                     'training': value['training'], 'diagnostics': value['diagnostics'],
                     'initial_state_sha256': value['initial_state_sha256']}
    if set(rows) != set(ARMS) or len(launch_hashes) != 1:
        raise ValueError('All three continuation arms required')
    initial_path = baseline / 'initial_eval.jsonl'
    initial = [json.loads(line) for line in initial_path.read_text().splitlines()]
    initial_mean = sum(row['sum_nll'] for row in initial) / sum(
        row['tokens']['useful_loss_tokens'] for row in initial)
    if initial_ce != initial_mean:
        raise ValueError('Initial eval differs from preserved baseline')
    return {'status': 'CONTINUED_PILOT_RESOURCE_REVIEW_PENDING', 'continuation_arms': rows,
            'baseline_original_status': 'USER_PAUSED_INCOMPLETE',
            'baseline_train_phase': json.loads((baseline / 'train_phase.json').read_text()),
            'baseline_end_to_end_timing_validated': False,
            'quality_verdict': 'NO_VERDICT', 'quality_ranking_authorized': False,
            'pilot_prefixes_differ_in_composition': True, 'full_training_authorized': False,
            'no_reconstruction_of_interrupted_original_receipt': True,
            'single_seed_exploratory': True, 'independent_safety_review_pending': True}


def main() -> None:
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'pilot', 'review'])
    for name in ['spec', 'reference', 'verification', 'artifact-root', 'data', 'cuda-gate', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--spec-sha', required=True)
    for name in ['launch-config', 'admission', 'model-dir']:
        parser.add_argument('--' + name, type=Path)
    for name in ['launch-config-sha', 'protocol-commit']:
        parser.add_argument('--' + name)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--runs', type=Path, nargs=3)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    spec, parent, inventory = load_specs(args)
    if args.mode == 'check':
        Console().print('CONTINUATION_REFERENCE_AUDIT_PASS; no GPU or training')
        return
    if args.mode == 'review':
        if not args.runs:
            parser.error('Three completed continuation run roots required')
        result = review(args, spec, parent, inventory)
        args.out.mkdir(parents=True, exist_ok=False)
        write(args.out / 'review.json', result)
        return
    if any(getattr(args, name) is None for name in ['launch_config', 'admission', 'model_dir',
                                                   'launch_config_sha', 'protocol_commit', 'arm']):
        parser.error('Explicit private stage authority, model, commit and arm required')
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
