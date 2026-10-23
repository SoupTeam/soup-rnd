"""Prospective read-only reload verification; never resumes training or old receipts."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from policy_pilot import plan_data
from policy_pilot_support import digest, load_authority, load_contract, verify_runtime
from portable_generation_probe import validate_launch
from run import base_model, phase, tokenizer_for, write
from run_support import Guard, contained, gpu_inventory, verify_model

REPO = Path(__file__).parents[2]
BOUND = 2400
PHASES = ['initial_eval', 'initial_diagnostics', 'train', 'final_eval',
          'final_diagnostics', 'raw_reload_eval', 'peft_reload_eval']


def verify_files(root: Path, manifest: dict) -> None:
    for name, sha in manifest.items():
        if digest(contained(root, name)) != sha:
            raise ValueError('Preserved evidence changed: ' + name)


def verify_phases(root: Path) -> None:
    for name in PHASES:
        value = json.loads((root / (name + '_phase.json')).read_text())['F1']
        if value.get('outcome') != 'ok' or value.get('gpu_clock', {}).get('outcome') != 'ok':
            raise ValueError('A completed source phase lacks accepted F1/clocks')


def compare_outputs(expected: Path, partial: Path, actual: Path, ids: list[str]) -> int:
    rows = [json.loads(line) for line in actual.read_text().splitlines()]
    if [row['id'] for row in rows] != ids or len(set(ids)) != len(ids):
        raise ValueError('Missing, duplicate or reordered diagnostic cases')
    if actual.read_bytes() != expected.read_bytes():
        raise ValueError('Reload tokens, text or scores differ from preserved final diagnostics')
    if not actual.read_bytes().startswith(partial.read_bytes()):
        raise ValueError('Preserved partial output is not an exact prefix')
    return len(rows)


def admit(receipt: dict, launch: dict, science_sha: str, launch_sha: str, now: float) -> None:
    validate_launch(launch)
    for key in ['account', 'instance_id', 'gpu_uuid']:
        if receipt.get(key) != launch[key]:
            raise ValueError('Private host/account mismatch')
    if (receipt.get('mode') != 'reload-verification'
            or receipt.get('contract_sha256') != science_sha
            or launch.get('contract_sha256') != science_sha
            or receipt.get('launch_config_sha256') != launch_sha
            or receipt.get('authorized_for_this_stage') is not True):
        raise ValueError('Explicit verification-only science and private authority required')
    for key in ['observed_unix_s', 'balance_usd', 'total_account_rate_usd_h']:
        value = receipt.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Finite actual Billing required')
    age = now - receipt['observed_unix_s']
    rate = receipt['total_account_rate_usd_h']
    if not 0 <= age <= 300 or rate <= 0:
        raise ValueError('Fresh Billing and positive rate required')
    reserve = 1.25 * rate * (BOUND + 40 + 300 + age) / 3600 + .5
    if receipt['balance_usd'] - launch['account_floor_usd'] <= reserve:
        raise ValueError('Verification and preservation exceed balance floor')


def evidence(args: argparse.Namespace) -> tuple[dict, dict]:
    if digest(args.spec) != args.spec_sha:
        raise ValueError('Follow-up contract SHA differs')
    spec = json.loads(args.spec.read_text())
    if spec.get('scope') != 'reload-diagnostics-only' or spec.get('total_s') != BOUND:
        raise ValueError('Wrong scope or time bound')
    verify_files(REPO, spec['source_sha256'])
    parent = spec['parent']
    original = load_contract(contained(REPO, parent['path']), parent['sha256'], args.artifact_root)
    manifest_path = contained(REPO, spec['reference_manifest']['path'])
    if digest(manifest_path) != spec['reference_manifest']['sha256']:
        raise ValueError('Reference manifest differs')
    verify_files(args.reference, json.loads(manifest_path.read_text()))
    root = contained(args.reference, 'pilot_baseline_001')
    meta = json.loads((root / 'prospective_run.json').read_text())
    external = json.loads((args.reference / 'pilot_baseline_001.launcher.json').read_text())
    if (meta['contract_sha'] != parent['sha256'] or meta['mode'] != 'pilot'
            or meta['arm'] != 'baseline' or external['exitcode'] != -15
            or external['hard_timeout'] is not False or (root / 'receipt.json').exists()):
        raise ValueError('Expected preserved user-interrupted original only')
    verify_phases(root / 'baseline')
    checkpoint = json.loads((root / 'baseline/checkpoint.json').read_text())
    if (checkpoint['final_state'] != spec['trained_state_sha256']
            or checkpoint['frozen_base'] != spec['frozen_base_sha256']
            or checkpoint['optimizer_state_saved'] is not False):
        raise ValueError('Checkpoint provenance differs')
    if digest(root / 'baseline/adapter.safetensors') != spec['adapter_sha256']:
        raise ValueError('Trained adapter differs')
    if plan_data(original, args.data) != json.loads(
            contained(REPO, original['input_plan']['path']).read_text()):
        raise ValueError('Frozen prepared inputs differ')
    from consumer import source_rows, windows
    from engine import audit_runtime
    from prepare import take_budget

    rows = take_budget(list(source_rows(args.data / 'baseline.jsonl')), 65536)
    audit_runtime(root / 'baseline/actual_trace.jsonl', windows(rows, 65536, 8192),
                  65536, original['recipe']['lr'], 5_000_000)
    return spec, original


def worker(args: argparse.Namespace, spec: dict, original: dict) -> None:
    launch = load_authority(args.launch_config, args.launch_config_sha, args.spec_sha)
    admission = json.loads(args.admission.read_text())
    admit(admission, launch, args.spec_sha, args.launch_config_sha, time.time())
    if (os.environ.get('G6_RELOAD_EXTERNAL_BOUND_S') != str(BOUND)
            or os.environ.get('SOUP_G6_INSTANCE_ID') != str(launch['instance_id'])
            or os.environ.get('CUDA_VISIBLE_DEVICES') != launch['gpu_uuid']
            or not re.fullmatch('[0-9a-f]{40}', args.protocol_commit)):
        raise ValueError('Approved bounded supervisor, host/GPU and verified commit required')
    verify_runtime(original)
    inventory = gpu_inventory(launch['gpu_uuid'])
    source = contained(args.reference, 'pilot_baseline_001')
    if inventory != json.loads((source / 'inventory.json').read_text()):
        raise ValueError('GPU/runtime inventory differs from trained source')
    destination = contained(Path(launch['runs_root']), str(args.out))
    if destination.exists():
        raise ValueError('Existing follow-up output forbids retry')
    claims = contained(Path(launch['runs_root']), '.claims/' + args.spec_sha)
    claims.mkdir(parents=True, exist_ok=True)
    write(claims / 'reload-verification.json', {'out': str(destination)})
    destination.mkdir(parents=True, exist_ok=False)
    try:
        import torch
        from check_consumer import state_hash
        from eval_suite import generate_suite
        from peft import PeftModel

        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        guard = Guard(destination, BOUND, gpu=True, uuid=launch['gpu_uuid'])
        guard()
        write(destination / 'model_verification.json',
              verify_model(args.model_dir, original['model_files'], guard))
        base = base_model(original, args.model_dir, False)
        model = PeftModel.from_pretrained(base, source / 'baseline/peft_adapter',
                                         is_trainable=True, local_files_only=True,
                                         autocast_adapter_dtype=True)
        if (state_hash(model) != spec['trained_state_sha256']
                or state_hash(model, frozen_only=True) != spec['frozen_base_sha256']):
            raise ValueError('Loaded trained or frozen state differs')
        tokenizer = tokenizer_for(original, args.model_dir)
        suite = json.loads(contained(REPO, original['suite']['path']).read_text())
        output = destination / 'reload_diagnostics.jsonl'
        phase(destination, 'resumed_reload_diagnostics', guard, 1200,
              lambda: generate_suite(model, tokenizer, suite, output, 'cuda:0', guard),
              gpu=True, model_dir=args.model_dir, arm='baseline')
        count = compare_outputs(source / 'baseline/final_diagnostics.jsonl',
                                source / 'baseline/reload_diagnostics.jsonl', output,
                                [case['id'] for case in suite['cases']])
        guard.stage(600)
        if state_hash(model) != spec['trained_state_sha256']:
            raise ValueError('Verification modified weights')
        check = json.loads((destination / 'resumed_reload_diagnostics_phase.json').read_text())
        if (check['F1']['outcome'] != 'ok'
                or check['F1']['gpu_clock']['outcome'] != 'ok'):
            raise ValueError('Follow-up F1/clocks not accepted')
        write(destination / 'receipt.json', {
            'status': 'VERIFIED_RELOAD_DIAGNOSTICS_ONLY', 'cases': count,
            'contract_sha256': args.spec_sha, 'protocol_commit': args.protocol_commit,
            'launch_config_sha256': args.launch_config_sha, 'inventory': inventory,
            'new_training': False, 'optimizer_resume': False,
            'original_attempt_status': 'USER_PAUSED_INCOMPLETE',
            'original_end_to_end_timing_validated': False,
            'remaining_pilots_unlocked': False, 'quality_verdict': 'NO_VERDICT',
        })
        write(destination / 'files_sha256.json', {
            str(path.relative_to(destination)): digest(path)
            for path in sorted(destination.rglob('*')) if path.is_file()})
    except Exception as error:
        write(destination / 'failure.json', {'error': repr(error), 'no_retry': True})
        raise


def main() -> None:
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'run'])
    for name in ['spec', 'reference', 'artifact-root', 'data', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--spec-sha', required=True)
    for name in ['launch-config', 'admission', 'model-dir']:
        parser.add_argument('--' + name, type=Path)
    for name in ['launch-config-sha', 'protocol-commit']:
        parser.add_argument('--' + name)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    spec, original = evidence(args)
    if args.mode == 'check':
        Console().print('OFFLINE_REFERENCE_AUDIT_PASS; no CUDA or training')
        return
    if any(getattr(args, name) is None for name in ['launch_config', 'admission', 'model_dir',
                                                   'launch_config_sha', 'protocol_commit']):
        parser.error('Private authority, fresh admission, model and commit required')
    if args.worker:
        worker(args, spec, original)
        return
    if args.out.exists():
        parser.error('Existing output forbids retry')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       G6_RELOAD_EXTERNAL_BOUND_S=str(BOUND))
    timed_out = False
    with Path(str(args.out) + '.launcher.log').open('x') as log:
        child = subprocess.Popen([sys.executable, __file__, *sys.argv[1:], '--worker'],
                                 env=environment, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            status = child.wait(timeout=BOUND + 30)
        except (subprocess.TimeoutExpired, BaseException) as error:
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
    write(Path(str(args.out) + '.launcher.json'), {'exitcode': status, 'hard_timeout': timed_out,
          'contract_sha256': args.spec_sha, 'bound_s': BOUND, 'supervisor_grace_s': 40})
    if status or timed_out:
        raise SystemExit(status or 1)


if __name__ == '__main__':
    main()
