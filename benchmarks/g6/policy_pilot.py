"""Offline planning and one explicitly admitted arm of a fresh matched G6 pilot."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

from consumer import audit_plan, fingerprint, inspect_source, source_rows, windows
from policy_pilot_support import (
    admit,
    bound,
    digest,
    load_authority,
    load_contract,
    preceding,
    verify_runtime,
)
from prepare import take_budget
from run import execute_arm, write
from run_support import Guard, contained, gpu_inventory, verify_model


def plan_data(spec: dict, data: Path) -> dict:
    if digest(data / 'manifest.json') != spec['data']['manifest_sha256']:
        raise ValueError('Prepared manifest differs')
    if digest(data / 'eval.jsonl') != spec['data']['eval_sha256']:
        raise ValueError('Eval bytes differ')
    evaluation = list(source_rows(data / 'eval.jsonl'))
    if (len(evaluation) != spec['data']['eval_rows']
            or sum(len(row['input_ids']) - 1 for row in evaluation)
            != spec['data']['eval_useful_targets']):
        raise ValueError('Held-out exposure differs')
    eval_ids = {row['id'] for row in evaluation}
    multisets = {}
    proof = {'status': 'OFFLINE_INPUT_PLAN_ONLY', 'arms': {}, 'CUDA_training_proven': False}
    for arm in spec['arm_order']:
        source = inspect_source(data, arm, spec['data']['manifest_sha256'])
        selected = list(source_rows(Path(source['path'])))
        if eval_ids.intersection(row['id'] for row in selected):
            raise ValueError('Train/eval source identity overlap')
        multisets[arm] = Counter((row['id'], fingerprint(row['input_ids'])) for row in selected)
        full_budget = spec['data']['budget_per_arm']
        pilot_budget = spec['limits']['pilot_train_targets']
        pilot = take_budget(selected, pilot_budget)
        plans = {}
        for name, rows, budget in [('full', selected, full_budget), ('pilot', pilot, pilot_budget)]:
            size = spec['recipe']['targets_per_update']
            plan = audit_plan(windows(rows, budget, size), budget, size)
            plan['attention_length_squared_proxy'] = sum(
                ((len(piece['input_ids']) + 7) // 8 * 8) ** 2
                for window in windows(rows, budget, size) for piece in window['pieces']
            )
            plans[name] = plan
        proof['arms'][arm] = {'source_sha256': source['sha256'], **plans}
    if any(multisets[name] != multisets['baseline']
           for name in ['length_curriculum', 'reverse_length_curriculum']):
        raise ValueError('Full order policies changed the baseline token multiset')
    proof['full_order_policies_same_multiset'] = True
    proof['pilot_prefixes_do_not_establish_quality_ranking'] = True
    return proof


def pilot(args: argparse.Namespace, spec: dict, proof: dict) -> None:
    launch = load_authority(args.launch_config, args.launch_config_sha, args.spec_sha)
    if not re.fullmatch(r'[0-9a-f]{40}', args.protocol_commit):
        raise ValueError('Verified user-created commit required')
    if (os.environ.get('SOUP_G6_INSTANCE_ID') != str(launch['instance_id'])
            or os.environ.get('CUDA_VISIBLE_DEVICES') != launch['gpu_uuid']
            or os.environ.get('G6_POLICY_EXTERNAL_BOUND_S') != str(bound(spec, 'pilot'))):
        raise ValueError('Approved host/GPU and bounded supervisor required')
    admission = json.loads(args.admission.read_text())
    admit(admission, spec, launch, args.launch_config_sha, 'pilot', args.arm, time.time())
    verify_runtime(spec)
    inventory = gpu_inventory(launch['gpu_uuid'])
    gate = json.loads(args.cuda_gate.read_text())
    if (gate.get('status') != 'PASS_TINY_CUDA_NUMERICAL_GATE'
            or gate.get('contract_sha256') != args.spec_sha
            or gate.get('launch_config_sha256') != args.launch_config_sha
            or gate.get('runtime') != spec['runtime_target'] or gate.get('inventory') != inventory):
        raise ValueError('Numerical training gate belongs to another setup or is missing')
    gate_external = json.loads(
        Path(str(args.cuda_gate.parent) + '.launcher.json').read_text())
    if ((args.cuda_gate.parent / 'failure.json').exists()
            or gate_external.get('exitcode') != 0
            or gate_external.get('hard_timeout') is not False
            or gate_external.get('contract_sha256') != args.spec_sha
            or gate_external.get('bound_s') != bound(spec, 'cuda-gate')):
        raise ValueError('Numerical gate supervisor failed or did not complete')
    runs = Path(launch['runs_root'])
    destination = contained(runs, str(args.out))
    claims = contained(runs, '.claims/' + args.spec_sha)
    claims.mkdir(parents=True, exist_ok=True)
    expected = preceding(claims, spec, args.arm, args.launch_config_sha)
    if destination.exists():
        raise ValueError('Existing output forbids restart')
    write(claims / ('pilot_' + args.arm + '.json'), {
        'out': str(destination), 'launch_config_sha256': args.launch_config_sha,
        'protocol_commit': args.protocol_commit,
    })
    destination.mkdir(parents=True, exist_ok=False)
    write(destination / 'prospective_run.json', {
        'mode': 'pilot', 'arm': args.arm, 'contract_sha': args.spec_sha,
        'launch_config_sha256': args.launch_config_sha, 'protocol_commit': args.protocol_commit,
        'new_series_not_historical_redeployment': True,
    })
    try:
        os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', HF_HUB_OFFLINE='1',
                          TRANSFORMERS_OFFLINE='1')
        import torch

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        guard = Guard(destination, bound(spec, 'pilot'), gpu=True, uuid=launch['gpu_uuid'])
        guard()
        write(destination / 'model_verification.json',
              verify_model(args.model_dir, spec['model_files'], guard))
        write(destination / 'inventory.json', inventory)
        write(destination / 'admission.json', admission)
        write(destination / 'input_plan.json', proof)
        selected = list(source_rows(args.data / (args.arm + '.jsonl')))
        directory = destination / args.arm
        directory.mkdir(exist_ok=False)
        suite = json.loads(contained(Path(__file__).parents[2], spec['suite']['path']).read_text())
        result = execute_arm(
            spec, take_budget(selected, spec['limits']['pilot_train_targets']),
            list(source_rows(args.data / 'eval.jsonl')), directory, guard,
            model_dir=args.model_dir, synthetic=False,
            budget=spec['limits']['pilot_train_targets'], lr_budget=5_000_000,
            arm=args.arm, suite=suite, expected_initial=expected,
        )
        if not result['F1_timing_accepted']:
            raise ValueError('A required F1/clock check failed; preserve and stop series')
        write(destination / 'receipt.json', {'status': 'COMPLETED_REVIEW_PENDING',
              'arms': {args.arm: result}, 'quality_verdict': 'NO_VERDICT'})
        write(destination / 'files_sha256.json', {
            str(path.relative_to(destination)): digest(path)
            for path in sorted(destination.rglob('*')) if path.is_file()
        })
    except Exception as error:
        write(destination / 'failure.json', {'error': repr(error),
              'partial_outputs_retained': True, 'no_retry': True, 'quality_verdict': 'NO_VERDICT'})
        raise


def main() -> None:
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'pilot'))
    for name in ('spec', 'artifact-root', 'data', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--spec-sha', required=True)
    for name in ('launch-config', 'admission', 'model-dir', 'cuda-gate'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--launch-config-sha')
    parser.add_argument('--protocol-commit')
    parser.add_argument('--arm')
    args = parser.parse_args()
    spec = load_contract(args.spec, args.spec_sha, args.artifact_root)
    proof = plan_data(spec, args.data)
    proof_path = contained(Path(__file__).parents[2], spec['input_plan']['path'])
    expected = json.loads(proof_path.read_text())
    if proof != expected:
        raise ValueError('Input plan differs from committed offline audit')
    if args.mode == 'plan':
        args.out.mkdir(parents=True, exist_ok=False)
        write(args.out / 'input_plan.json', proof)
        Console().print('OFFLINE_INPUT_PLAN_PASS; no GPU or training')
    else:
        if any(getattr(args, name) is None for name in (
                'launch_config', 'launch_config_sha', 'protocol_commit', 'admission',
                'model_dir', 'cuda_gate', 'arm')):
            parser.error('Pilot requires private launch, admission, commit and CUDA gate arguments')
        pilot(args, spec, proof)


if __name__ == '__main__':
    main()
