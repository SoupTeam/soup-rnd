"""Prospective inference-only confirmation of literal turn stopping on fresh tasks."""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from collections import Counter
from pathlib import Path

from eval_suite import score_task
from paired_generation_probe import generate_pass
from policy_pilot_support import digest, verify_runtime
from run_support import contained
from turn_boundary_candidate import boundary

REPO = Path(__file__).parents[2]
LABELS = ['initial', 'baseline', 'coherence_filter', 'length_curriculum',
          'reverse_length_curriculum']
BOUND = 3600


def oracle(case: dict) -> str:
    """Independently derive expected answers from explicit task operands."""
    data = case['operands']
    family = case['family']
    if family in ['add', 'subtract', 'multiply']:
        left, right = data
        return str({'add': left + right, 'subtract': left - right,
                    'multiply': left * right}[family])
    if family == 'sort':
        return ','.join(str(value) for value in sorted(data))
    if family == 'extract':
        return data['code']
    if family == 'count':
        return str(data['words'].count(data['target']))
    if family == 'last':
        return data[-1]
    if family == 'format':
        return '/'.join(data).upper()
    if family == 'logic':
        return 'yes' if data['all_members_have_property'] else 'no'
    if family == 'context':
        return data['days'][data['query']]
    if family == 'reverse':
        return data[::-1]
    if family == 'json':
        return json.dumps(data, separators=(',', ':'))
    raise ValueError('Unknown task family')


def validate_suite(suite: dict) -> None:
    cases = suite['cases']
    counts = Counter(case['family'] for case in cases)
    families = ['add', 'subtract', 'multiply', 'sort', 'extract', 'count', 'last',
                'format', 'logic', 'context', 'reverse', 'json']
    if (len(cases) != 48 or counts != Counter(dict.fromkeys(families, 4))
            or len({case['id'] for case in cases}) != 48
            or len({case['prompt'] for case in cases}) != 48):
        raise ValueError('Require 48 distinct tasks, four in each of 12 families')
    for case in cases:
        if case['kind'] != 'task' or case['expected'] != oracle(case):
            raise ValueError('Task answer differs from independently derived oracle')
        expected_scorer = 'json_object' if case['family'] == 'json' else 'strip_exact'
        if case['scorer'] != expected_scorer or not score_task(case, case['expected']):
            raise ValueError('Frozen strict scorer does not accept oracle')
        if boundary(case['prompt']) is not None:
            raise ValueError('Fresh tasks cannot contain literal turn boundaries')


def pair_audit(original: list[dict], candidate: list[dict], suite: dict) -> dict:
    """Require exact greedy prefixes and recompute strict scores from raw evidence."""
    cases = {case['id']: case for case in suite['cases']}
    if (len(original) != len(cases) or len(candidate) != len(cases)
            or [row['id'] for row in original] != list(cases)
            or [row['id'] for row in candidate] != list(cases)):
        raise ValueError('Missing, duplicated or reordered paired cases')
    losses = gains = before = after = 0
    for old, new in zip(original, candidate):
        case = cases[old['id']]
        ids = new['raw_completion_ids']
        if (not ids or old['raw_completion_ids'][:len(ids)] != ids
                or old['prompt_ids'] != new['prompt_ids']):
            raise ValueError('Paired greedy prefix or prompt differs')
        if (old['decoder'] != 'original_v1' or new['decoder'] != 'role_boundary_v1'
                or old['scored_completion'] != old['raw_completion']):
            raise ValueError('Unexpected decoder or original trimming')
        match = boundary(new['raw_completion'])
        expected = new['raw_completion'][:match['character_offset']] if match else (
            new['raw_completion'])
        if new['scored_completion'] != expected or new['boundary'] != match:
            raise ValueError('Boundary score text differs from fixed rule')
        if (match is not None) != (new['stop_reason'] == 'role_boundary'):
            raise ValueError('Boundary stop reason differs')
        if case['kind'] == 'task':
            old_pass = score_task(case, old['scored_completion'])
            new_pass = score_task(case, new['scored_completion'])
            if old_pass != old['task_pass'] or new_pass != new['task_pass']:
                raise ValueError('Stored task score differs from frozen scorer')
            before += old_pass
            after += new_pass
            gains += new_pass and not old_pass
            losses += old_pass and not new_pass
        elif old['task_pass'] is not None or new['task_pass'] is not None:
            raise ValueError('Safety must remain manual and unscored')
    return {'original_exact': before, 'boundary_exact': after, 'gains': gains,
            'losses': losses, 'net_gain': gains - losses,
            'task_total': sum(case['kind'] == 'task' for case in cases.values()),
            'all_raw_prefixes_match': True, 'quality_verdict': 'NO_VERDICT'}


def read_contract(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    if digest(args.spec) != args.spec_sha:
        raise ValueError('Prospective contract SHA differs')
    spec = json.loads(args.spec.read_text())
    if spec['scope'] != 'fresh-turn-stop-confirmation-inference-only' or spec['bound_s'] != BOUND:
        raise ValueError('Only the fixed inference-only scope is allowed')
    for name, sha in spec['source_sha256'].items():
        if digest(contained(REPO, name)) != sha:
            raise ValueError('Committed source differs: ' + name)
    suite = json.loads(contained(REPO, spec['fresh_suite']['path']).read_text())
    if digest(contained(REPO, spec['fresh_suite']['path'])) != spec['fresh_suite']['sha256']:
        raise ValueError('Fresh suite differs')
    validate_suite(suite)
    recipe = json.loads(contained(REPO, spec['recipe']['path']).read_text())
    diagnostic = json.loads(contained(REPO, spec['diagnostic_suite']['path']).read_text())
    if suite['generation'] != diagnostic['generation']:
        raise ValueError('Decoder settings must differ only in turn stopping')
    return spec, recipe, suite


def execute(args: argparse.Namespace, spec: dict, recipe: dict, fresh: dict) -> None:
    from check_consumer import state_hash
    from checkpoint_eval import load_adapter
    from run import build_model, phase, tokenizer_for, write
    from run_support import Guard, gpu_inventory, verify_model

    config = json.loads(args.authority.read_text())
    admission = json.loads(args.admission.read_text())
    if not Path(config['runs_root']).is_absolute() or not args.out.is_absolute():
        raise ValueError('Absolute private runs root and output required')
    if (digest(args.authority) != args.authority_sha
            or config.get('scope') != spec['scope']
            or config.get('inference_only_authorized') is not True
            or admission.get('authorized_for_this_stage') is not True
            or admission.get('spec_sha256') != args.spec_sha
            or admission.get('authority_sha256') != args.authority_sha
            or admission.get('model_label') != args.label):
        raise ValueError('New explicit private authority and per-model admission required')
    for key in ['instance_id', 'gpu_uuid']:
        if admission.get(key) != config[key]:
            raise ValueError('Admission host identity differs')
    if (os.environ.get('SOUP_G6_INSTANCE_ID') != str(config['instance_id'])
            or os.environ.get('CUDA_VISIBLE_DEVICES') != config['gpu_uuid']
            or os.environ.get('G6_BOUNDARY_CONFIRM_BOUND_S') != str(BOUND)
            or not re.fullmatch('[a-f0-9]{40}', args.protocol_commit)):
        raise ValueError('Verified pushed commit, supervised host and exact GPU required')
    verify_runtime(recipe)
    args.out = contained(Path(config['runs_root']), str(args.out))
    claims = contained(Path(config['runs_root']), '.claims/' + args.spec_sha)
    claims.mkdir(parents=True, exist_ok=True)
    with (claims / (args.label + '.json')).open('x') as handle:
        json.dump({'out': str(args.out), 'protocol_commit': args.protocol_commit}, handle)
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / 'prospective_run.json', {
        'spec_sha256': args.spec_sha, 'authority_sha256': args.authority_sha,
        'protocol_commit': args.protocol_commit, 'label': args.label,
        'scope': spec['scope'], 'new_training': False})
    try:
        import torch

        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        guard = Guard(args.out, BOUND, gpu=True, uuid=config['gpu_uuid'])
        guard.stage(recipe['limits']['setup_s'])
        verify_model(args.model_dir, recipe['model_files'], guard)
        inventory = gpu_inventory(config['gpu_uuid'])
        model = build_model(recipe, args.model_dir, False)
        guard()
        initial = state_hash(model)
        frozen = state_hash(model, frozen_only=True)
        if initial != recipe['expected_initial_state_sha256']:
            raise ValueError('Original model/adapter initialization differs')
        checkpoint = spec['models'][args.label]
        if args.label != 'initial':
            if args.adapter is None:
                raise ValueError('Pinned full adapter required')
            load_adapter(model, args.adapter, checkpoint['raw_sha256'], frozen)
        if state_hash(model) != checkpoint['state_sha256']:
            raise ValueError('Pinned model state differs')
        write(args.out / 'initial_identity.json', {
            'state_sha256': checkpoint['state_sha256'], 'original_state_sha256': initial,
            'frozen_base_sha256': frozen, 'runtime_checked': True})
        tokenizer = tokenizer_for(recipe, args.model_dir)
        diagnostic = json.loads(contained(REPO, spec['diagnostic_suite']['path']).read_text())
        # Alternating fixed order balances intervention order across model labels.
        order = ['original', 'boundary'] if LABELS.index(args.label) % 2 == 0 else (
            ['boundary', 'original'])
        phases = {}
        for name, suite in [('diagnostic', diagnostic), ('confirmation', fresh)]:
            for decoder in order:
                destination = args.out / (name + '_' + decoder + '.jsonl')
                result = phase(args.out, name + '_' + decoder, guard, 1200,
                               lambda s=suite, p=destination, d=decoder: generate_pass(
                                   model, tokenizer, s, p, 'cuda:0', guard, d == 'boundary'),
                               gpu=True, model_dir=args.model_dir, arm=args.label)
                phases[name + '_' + decoder] = result
                observed_state = state_hash(model)
                write(args.out / (name + '_' + decoder + '_identity.json'), {
                    'state_sha256': observed_state})
                if observed_state != checkpoint['state_sha256']:
                    raise ValueError('Inference mutated model weights')
        paired = {}
        for name, suite in [('diagnostic', diagnostic), ('confirmation', fresh)]:
            data = [[json.loads(line) for line in (args.out / (
                name + '_' + decoder + '.jsonl')).read_text().splitlines()]
                    for decoder in ['original', 'boundary']]
            paired[name] = pair_audit(*data, suite)
        if state_hash(model, frozen_only=True) != frozen:
            raise ValueError('Frozen base changed')
        for path in args.out.glob('*_phase.json'):
            result = json.loads(path.read_text())['F1']
            if result['outcome'] != 'ok' or result['gpu_clock']['outcome'] != 'ok':
                raise ValueError('Invalid F1 or clocks; no accepted inference result')
        write(args.out / 'receipt.json', {
            'status': 'PAIRED_CONFIRMATION_REVIEW_PENDING', 'label': args.label,
            'spec_sha256': args.spec_sha, 'protocol_commit': args.protocol_commit,
            'inventory': inventory, 'paired': paired, 'new_training': False,
            'generation_results': phases, 'authority_sha256': args.authority_sha,
            'weights_unchanged': True, 'quality_verdict': 'NO_VERDICT'})
        write(args.out / 'files_sha256.json', {str(p.relative_to(args.out)): digest(p)
              for p in sorted(args.out.rglob('*')) if p.is_file()})
    except Exception as error:
        write(args.out / 'failure.json', {'error': repr(error), 'retry_allowed': False})
        raise


def main() -> None:
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'run'])
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--spec-sha', required=True)
    for name in ['authority', 'admission', 'out', 'model-dir', 'adapter']:
        parser.add_argument('--' + name, type=Path)
    for name in ['authority-sha', 'protocol-commit']:
        parser.add_argument('--' + name)
    parser.add_argument('--label', choices=LABELS)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    spec, recipe, fresh = read_contract(args)
    if args.mode == 'check':
        Console().print('FRESH_48_ORACLES_AND_PINNED_SOURCE_PASS; no GPU/model generation')
        return
    if any(getattr(args, key) is None for key in ['authority', 'authority_sha', 'admission',
                                                 'out', 'model_dir', 'label', 'protocol_commit']):
        parser.error('Explicit new authority, admission, model and unique output required')
    if args.worker:
        execute(args, spec, recipe, fresh)
        return
    if args.out.exists():
        parser.error('Existing output forbids retries')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       G6_BOUNDARY_CONFIRM_BOUND_S=str(BOUND))
    timed_out = False
    with Path(str(args.out) + '.launcher.log').open('x') as log:
        child = subprocess.Popen([sys.executable, __file__, *sys.argv[1:], '--worker'],
                                 stdout=log, stderr=subprocess.STDOUT, env=environment,
                                 start_new_session=True)
        try:
            code = child.wait(timeout=BOUND + 30)
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
            code = child.returncode
    Path(str(args.out) + '.launcher.json').write_text(json.dumps({
        'exitcode': code, 'hard_timeout': timed_out, 'bound_s': BOUND,
        'spec_sha256': args.spec_sha}) + '\n')
    if code or timed_out:
        raise SystemExit(code or 1)


if __name__ == '__main__':
    main()
