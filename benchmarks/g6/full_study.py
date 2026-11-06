"""Prospective four-policy 5M-target study using unchanged training and evaluation."""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

from consumer import source_rows
from policy_pilot_support import digest, load_authority, verify_runtime
from policy_recovery import load_specs as load_parent
from prepare import take_budget
from reload_followup import verify_files
from run import execute_arm, write
from run_support import Guard, contained, gpu_inventory, verify_model
from summarize import summarize

REPO = Path(__file__).parents[2]
ARMS = ['baseline', 'coherence_filter', 'length_curriculum', 'reverse_length_curriculum']
PHASES = ['initial_eval', 'initial_diagnostics', 'train', 'final_eval', 'final_diagnostics',
          'raw_reload_eval', 'peft_reload_eval', 'reload_diagnostics']
BOUND = 14400


def completed(root: Path, arm: str, science: str, inventory: dict,
              targets: int, updates: int, bound: int) -> dict:
    if (root / 'failure.json').exists():
        raise ValueError('Failed arm cannot be accepted')
    manifest = json.loads((root / 'files_sha256.json').read_text())
    required = {'receipt.json', 'prospective_run.json', 'inventory.json', arm + '/receipt.json'}
    required.update(arm + '/' + name + '_phase.json' for name in PHASES)
    required.update(str(path.relative_to(root)) for path in root.rglob('*')
                    if path.is_file() and path.name != 'files_sha256.json')
    if not required.issubset(manifest):
        raise ValueError('Completion manifest omits required evidence')
    verify_files(root, manifest)
    meta = json.loads((root / 'prospective_run.json').read_text())
    outer = json.loads(Path(str(root) + '.launcher.json').read_text())
    if (meta['arm'] != arm or meta['contract_sha'] != science
            or meta.get('mode') != ('pilot' if targets == 65536 else 'full')
            or outer['contract_sha256'] != science or outer['bound_s'] != bound
            or outer['exitcode'] != 0 or outer['hard_timeout'] is not False
            or json.loads((root / 'inventory.json').read_text()) != inventory):
        raise ValueError('Completed arm provenance or external termination differs')
    receipt = json.loads((root / arm / 'receipt.json').read_text())
    training = receipt['training']
    if (receipt['status'] != 'ARM_COMPLETED_REVIEW_PENDING'
            or training['actual_useful_tokens'] != targets
            or training['applied_optimizer_updates'] != updates
            or training['AMP_exercised'] is not True or receipt['reload_exact'] is not True
            or receipt['frozen_base_unchanged'] is not True
            or receipt['F1_timing_accepted'] is not True
            or receipt['final_eval']['mean_ce'] != receipt['raw_reload_eval']['mean_ce']
            or receipt['final_eval']['mean_ce'] != receipt['peft_reload_eval']['mean_ce']):
        raise ValueError('Exposure, AMP, frozen base, reload or eval differs')
    directory = root / arm
    for name in ['initial_eval', 'final_eval', 'raw_reload_eval', 'peft_reload_eval']:
        evaluation = directory / (name + '.jsonl')
        rows = [json.loads(line) for line in evaluation.read_text().splitlines()]
        count = sum(row['tokens']['useful_loss_tokens'] for row in rows)
        if (len(rows) != 256 or count != 322332
                or receipt[name]['rows'] != 256 or receipt[name]['useful_targets'] != count
                or sum(row['sum_nll'] for row in rows) / count != receipt[name]['mean_ce']):
            raise ValueError('Held-out evaluation coverage or weighted CE differs')
    trace_path = directory / 'actual_trace.jsonl'
    trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
    steps = [row for row in trace if row['event'] == 'optimizer_step']
    expected_last = targets - (updates - 1) * 8192
    if (len(steps) != updates or [row['update'] for row in steps] != list(range(1, updates + 1))
            or any(row['actual_useful_tokens'] != 8192 for row in steps[:-1])
            or steps[-1]['actual_useful_tokens'] != expected_last
            or [row['consumed_useful_tokens'] for row in steps]
            != [min(index * 8192, targets) for index in range(1, updates + 1)]):
        raise ValueError('Actual update trace differs from fixed exposure')
    for name in PHASES:
        check = json.loads((directory / (name + '_phase.json')).read_text())['F1']
        if check['outcome'] != 'ok' or check.get('gpu_clock', {}).get('outcome') != 'ok':
            raise ValueError('Rejected ownership, clocks or phase validity')
    for name in ['initial_diagnostics', 'final_diagnostics', 'reload_diagnostics']:
        if len((directory / (name + '.jsonl')).read_text().splitlines()) != 24:
            raise ValueError('Incomplete diagnostic suite')
    if (directory / 'final_diagnostics.jsonl').read_bytes() != (
            directory / 'reload_diagnostics.jsonl').read_bytes():
        raise ValueError('Reload diagnostic bytes differ')
    return receipt


def load_specs(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    if digest(args.spec) != args.spec_sha:
        raise ValueError('Full-study science SHA differs')
    spec = json.loads(args.spec.read_text())
    if (spec.get('scope') != 'four-policy-full-5M-targets'
            or spec.get('arm_order') != ARMS or spec.get('targets_per_arm') != 5_000_000
            or spec.get('updates_per_arm') != 611 or spec.get('last_update_targets') != 2880
            or spec.get('total_s_per_arm') != BOUND
            or spec.get('baseline_pilot_original_end_to_end_timing_validated') is not False
            or spec.get('financial_reserve_policy') != 'explicit_user_waiver_only'
            or spec.get('full_training_authorized') is not True):
        raise ValueError('Only the fixed bounded four-policy full study is allowed')
    verify_files(REPO, spec['source_sha256'])
    old_args = copy.copy(args)
    old_args.spec = contained(REPO, spec['parent']['path'])
    old_args.spec_sha = spec['parent']['sha256']
    _, parent, inventory = load_parent(old_args)
    verify_files(args.pilots, spec['pilots']['files_sha256'])
    for arm in ARMS[1:]:
        root = args.pilots / ('pilot_' + arm + '_001')
        required = {str(path.relative_to(args.pilots))
                    for path in root.rglob('*') if path.is_file()}
        required.add(root.name + '.launcher.json')
        if not required.issubset(spec['pilots']['files_sha256']):
            raise ValueError('Pinned pilot map omits evidence')
        proof = completed(root, arm,
                          spec['pilots']['contracts'][arm], inventory, 65536, 8, 4800)
        if proof['initial_state_sha256'] != parent['expected_initial_state_sha256']:
            raise ValueError('Pilot initialization differs')
    parent = copy.deepcopy(parent)
    parent['arm_order'] = ARMS
    parent['launch_contract_sha256'] = args.spec_sha
    return spec, parent, inventory


def admit(admission: dict, launch: dict, science_sha: str, launch_sha: str, arm: str) -> None:
    if (launch.get('credit_reserve_waived_by_user') is not True
            or admission.get('credit_reserve_waived_by_user') is not True):
        raise ValueError('Explicit user financial-reserve waiver required for this protocol')
    for key in ['account', 'instance_id', 'gpu_uuid']:
        if admission.get(key) != launch[key]:
            raise ValueError('Private account or instance differs')
    if (arm not in ARMS or admission.get('arm') != arm or admission.get('mode') != 'full'
            or admission.get('contract_sha256') != science_sha
            or admission.get('launch_config_sha256') != launch_sha
            or admission.get('authorized_for_this_stage') is not True):
        raise ValueError('Explicit science-bound full-arm authority required')
    # User waived financial reserves only. No correctness, identity or time check is waived.


def preceding(runs: Path, arm: str, args: argparse.Namespace, parent: dict,
              inventory: dict) -> Path:
    claims = contained(runs, '.claims/' + args.spec_sha)
    claims.mkdir(parents=True, exist_ok=True)
    own = claims / ('full_' + arm + '.json')
    if own.exists() or args.out.exists():
        raise ValueError('Existing claim or output forbids retry')
    for previous in ARMS[:ARMS.index(arm)]:
        claim = json.loads((claims / ('full_' + previous + '.json')).read_text())
        if claim['launch_config_sha256'] != args.launch_config_sha:
            raise ValueError('Private configuration changed between full arms')
        proof = completed(Path(claim['out']), previous, args.spec_sha, inventory,
                          5_000_000, 611, BOUND)
        if proof['initial_state_sha256'] != parent['expected_initial_state_sha256']:
            raise ValueError('Preceding full initialization differs')
    return own


def worker(args: argparse.Namespace, spec: dict, parent: dict, inventory: dict) -> None:
    launch = load_authority(args.launch_config, args.launch_config_sha, args.spec_sha)
    admission = json.loads(args.admission.read_text())
    admit(admission, launch, args.spec_sha, args.launch_config_sha, args.arm)
    if (os.environ.get('G6_FULL_EXTERNAL_BOUND_S') != str(BOUND)
            or os.environ.get('SOUP_G6_INSTANCE_ID') != str(launch['instance_id'])
            or os.environ.get('CUDA_VISIBLE_DEVICES') != launch['gpu_uuid']
            or not re.fullmatch('[0-9a-f]{40}', args.protocol_commit)):
        raise ValueError('Approved supervisor, host/GPU and verified pushed commit required')
    verify_runtime(parent)
    if gpu_inventory(launch['gpu_uuid']) != inventory:
        raise ValueError('GPU inventory changed')
    runs = Path(launch['runs_root'])
    args.out = contained(runs, str(args.out))
    claim = preceding(runs, args.arm, args, parent, inventory)
    write(claim, {'out': str(args.out), 'launch_config_sha256': args.launch_config_sha,
                  'protocol_commit': args.protocol_commit})
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / 'prospective_run.json', {
        'mode': 'full', 'arm': args.arm, 'contract_sha': args.spec_sha,
        'launch_config_sha256': args.launch_config_sha, 'protocol_commit': args.protocol_commit,
        'baseline_pilot_original_end_to_end_timing_validated': False,
        'fresh_original_model_adapter_optimizer': True, 'optimizer_resume': False})
    try:
        import torch

        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        guard = Guard(args.out, BOUND, gpu=True, uuid=launch['gpu_uuid'])
        write(args.out / 'model_verification.json', verify_model(
            args.model_dir, parent['model_files'], guard))
        write(args.out / 'inventory.json', inventory)
        write(args.out / 'admission.json', admission)
        rows = take_budget(list(source_rows(args.data / (args.arm + '.jsonl'))), 5_000_000)
        evaluation = list(source_rows(args.data / 'eval.jsonl'))
        suite = json.loads(contained(REPO, parent['suite']['path']).read_text())
        directory = args.out / args.arm
        directory.mkdir(exist_ok=False)
        result = execute_arm(parent, rows, evaluation, directory, guard,
                             model_dir=args.model_dir, synthetic=False,
                             budget=5_000_000, lr_budget=5_000_000, arm=args.arm,
                             suite=suite, expected_initial=parent['expected_initial_state_sha256'])
        if not result['F1_timing_accepted']:
            raise ValueError('Invalid full-arm timing; preserve and block progression')
        write(args.out / 'receipt.json', {'status': 'COMPLETED_REVIEW_PENDING',
              'arms': {args.arm: result}, 'quality_verdict': 'NO_VERDICT'})
        write(args.out / 'files_sha256.json', {str(path.relative_to(args.out)): digest(path)
              for path in sorted(args.out.rglob('*')) if path.is_file()})
    except Exception as error:
        write(args.out / 'failure.json', {'error': repr(error), 'no_retry': True})
        raise


def main() -> None:
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'full', 'review'])
    for name in ['spec', 'reference', 'verification', 'series', 'pilots', 'artifact-root',
                 'data', 'cuda-gate', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--spec-sha', required=True)
    for name in ['launch-config', 'admission', 'model-dir']:
        parser.add_argument('--' + name, type=Path)
    for name in ['launch-config-sha', 'protocol-commit']:
        parser.add_argument('--' + name)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--runs', type=Path, nargs=4)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    spec, parent, inventory = load_specs(args)
    if args.mode == 'check':
        Console().print('FULL_REFERENCE_AUDIT_PASS; no GPU or training')
        return
    if args.mode == 'review':
        if not args.runs:
            parser.error('Four completed full run roots required')
        configurations = set()
        exposure = {}
        for root in args.runs:
            meta = json.loads((root / 'prospective_run.json').read_text())
            configurations.add(meta['launch_config_sha256'])
            proof = completed(root, meta['arm'], args.spec_sha, inventory,
                              5_000_000, 611, BOUND)
            if proof['initial_state_sha256'] != parent['expected_initial_state_sha256']:
                raise ValueError('Full initialization differs from fixed source')
            from consumer import windows
            from engine import audit_runtime

            data = take_budget(list(source_rows(args.data / (meta['arm'] + '.jsonl'))), 5_000_000)
            audit_runtime(root / meta['arm'] / 'actual_trace.jsonl',
                          windows(data, 5_000_000, 8192), 5_000_000,
                          parent['recipe']['lr'], 5_000_000)
            training = proof['training']
            exposure[meta['arm']] = {
                'useful_loss_tokens': training['actual_useful_tokens'],
                'processed_input_slots': training['processed_input_slots'],
                'nonpadding_input_tokens': training['processed_input_slots']
                - training['padded_input_slots'],
                'padded_input_slots': training['padded_input_slots']}
        if len(configurations) != 1:
            raise ValueError('Full private configuration changed')
        args.out.mkdir(parents=True, exist_ok=False)
        result = summarize(parent, args.runs, 'full', args.spec_sha, args.out)
        for arm, counts in exposure.items():
            result['rows'][arm]['tokens'] = counts
        result.update(baseline_pilot_original_end_to_end_timing_validated=False,
                      quality_ranking_is_single_seed_exploratory=True,
                      independent_safety_review_pending=True,
                      training_phase_time_excludes_preparation_eval_and_reload=True)
        write(args.out / 'comparison.json', result)
        return
    if any(getattr(args, name) is None for name in ['launch_config', 'admission', 'model_dir',
                                                   'launch_config_sha', 'protocol_commit', 'arm']):
        parser.error('Full private stage authority, model, pushed commit and arm required')
    for root in [args.reference, args.verification, args.series, args.pilots]:
        old, new = os.path.realpath(root), os.path.realpath(args.out)
        if os.path.commonpath([old, new]) in [old, new]:
            parser.error('New full output must be separate from preserved evidence')
    if args.worker:
        worker(args, spec, parent, inventory)
        return
    if args.out.exists():
        parser.error('Existing output forbids retry')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       G6_FULL_EXTERNAL_BOUND_S=str(BOUND))
    timed_out = False
    with Path(str(args.out) + '.launcher.log').open('x') as log:
        child = subprocess.Popen([sys.executable, __file__, *sys.argv[1:], '--worker'],
                                 env=environment, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            status = child.wait(timeout=BOUND + 30)
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
        'exitcode': status, 'hard_timeout': timed_out, 'bound_s': BOUND,
        'supervisor_grace_s': 40, 'contract_sha256': args.spec_sha})
    if status or timed_out:
        raise SystemExit(status or 1)


if __name__ == '__main__':
    main()
