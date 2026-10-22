"""Prospective format-prompt inference comparison; frozen legacy results remain intact."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from boundary_confirmation import oracle
from eval_suite import score_task
from paired_generation_probe import generate_pass as base_generate_pass
from policy_pilot_support import digest, verify_runtime
from run_support import contained
from turn_boundary_candidate import boundary, inspect_prefix

REPO = Path(__file__).parents[2]
LABELS = ['initial', 'baseline']
BOUND = 3600


def validate_suite(suite: dict, old: dict) -> None:
    cases = suite['cases']
    families = {c['family'] for c in old['cases']}
    if (len(cases) != 96 or Counter(c['family'] for c in cases) !=
            Counter(dict.fromkeys(families, 8)) or len({c['id'] for c in cases}) != 96
            or len({c['prompt'] for c in cases}) != 96):
        raise ValueError('Require96 distinct balanced tasks')
    old_prompts = {c['prompt'].casefold().strip() for c in old['cases']}
    for case in cases:
        if (case['kind'] != 'task' or case['expected'] != oracle(case)
                or not score_task(case, case['expected'])):
            raise ValueError('Invalid oracle')
        if case['scorer'] != ('json_object' if case['family'] == 'json' else 'strip_exact'):
            raise ValueError('Scorer changed')
        if case['prompt'].casefold().strip() in old_prompts or boundary(case['prompt']):
            raise ValueError('Old prompt overlap or role marker')


def rendered(prompt: str, candidate: bool, prefix: str) -> str:
    return f"user: {prefix if candidate else ''}{prompt}\nassistant:"


def generate_pass(model: object, tokenizer: object, suite: dict, out: Path,
                  device: str, guard: object, candidate: bool, prefix: str) -> dict:
    changed = copy.deepcopy(suite)
    if candidate:
        for case in changed['cases']:
            case['prompt'] = prefix + case['prompt']
    # Both conditions use identical role_boundary_v1; only prompt text differs.
    return base_generate_pass(model, tokenizer, changed, out, device, guard, True)


def pair_audit(comparator: list[dict], candidate: list[dict], suite: dict,
               tokenizer: object, prefix: str) -> dict:
    cases = suite['cases']
    if any([r['id'] for r in rows] != [c['id'] for c in cases]
           for rows in [comparator, candidate]):
        raise ValueError('Missing or reordered cases')
    before = after = gains = losses = prompt_before = prompt_after = 0
    for case, left, right in zip(cases, comparator, candidate):
        scores = []
        for row, changed in [(left, False), (right, True)]:
            expected_ids = [tokenizer.bos_token_id, *tokenizer.encode(
                rendered(case['prompt'], changed, prefix), add_special_tokens=False)]
            if row['prompt_ids'] != expected_ids or row['decoder'] != 'role_boundary_v1':
                raise ValueError('Prompt or decoder differs from assigned condition')
            ids = row['raw_completion_ids']
            if not ids or len(ids) > 128:
                raise ValueError('Invalid completion token count')
            inspected = inspect_prefix(ids, tokenizer)
            match = inspected['boundary']
            reason = 'role_boundary' if match else ('eos' if ids[-1] ==
                     tokenizer.eos_token_id else 'max_new_tokens' if len(ids) == 128 else None)
            if (row['raw_completion'] != inspected['raw_completion']
                    or row['scored_completion'] != inspected['candidate_scored_text']
                    or row['boundary'] != match or reason is None or row['stop_reason'] != reason):
                raise ValueError('Raw token/text/boundary evidence differs')
            score = score_task(case, row['scored_completion']) if case['kind'] == 'task' else None
            if (row['task_pass'] != score or (case['kind'] == 'safety' and
                    row.get('safety_review') != 'PENDING')):
                raise ValueError('Stored score or manual safety status differs')
            scores.append(score)
        prompt_before += len(left['prompt_ids'])
        prompt_after += len(right['prompt_ids'])
        if case['kind'] == 'task':
            before += scores[0]
            after += scores[1]
            gains += scores[1] and not scores[0]
            losses += scores[0] and not scores[1]
    return {'comparator_exact': before, 'candidate_exact': after, 'gains': gains,
            'losses': losses, 'net_gain': gains - losses, 'prompt_tokens_comparator': prompt_before,
            'prompt_tokens_candidate': prompt_after, 'same_prefix_claim': False,
            'quality_verdict': 'NO_VERDICT'}


def read_contract(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    if digest(args.spec) != args.spec_sha:
        raise ValueError('Contract SHA differs')
    spec = json.loads(args.spec.read_text())
    if (spec['scope'] != 'fixed-format-demonstrations-inference-only'
            or spec['bound_s'] != BOUND or spec['model_order'] != LABELS
            or spec['new_training_allowed'] is not False or not spec['candidate_prefix']):
        raise ValueError('Fixed inference-only scope differs')
    for name, sha in spec['source_sha256'].items():
        if digest(contained(REPO, name)) != sha:
            raise ValueError('Pinned source differs: ' + name)
    values = []
    for key in ['recipe', 'fresh_suite', 'diagnostic_suite', 'development_suite']:
        entry = spec[key]
        path = contained(REPO, entry['path'])
        if digest(path) != entry['sha256']:
            raise ValueError('Pinned artifact differs: ' + key)
        values.append(json.loads(path.read_text()))
    recipe, fresh, diagnostic, old = values
    if fresh['generation'] != diagnostic['generation']:
        raise ValueError('Decoder settings differ')
    validate_suite(fresh, old)
    return spec, recipe, fresh


def operational_admission(config: dict, admission: dict, now: float, boot_id: str) -> None:
    """Reject stale host/finance evidence and any stage beyond the new total cap."""
    if (config.get('total_cap_usd') != 5 or config.get('account') != 'soup-rnd'
            or admission.get('budget_verified') is not True
            or admission.get('boot_id') != boot_id or config.get('boot_id') != boot_id):
        raise ValueError('New cumulative5USD authority and current boot required')
    names = ['observed_unix_s', 'operator_deadline_unix_s', 'spent_usd',
             'rate_usd_h', 'remaining_noncompute_reserve_usd']
    if any(type(admission.get(k)) not in [int, float] or not math.isfinite(admission[k])
           for k in names):
        raise ValueError('Finite actual financial observation required')
    if (not 0 <= now - admission['observed_unix_s'] <= 300
            or not 0 <= admission['spent_usd'] < 5 or admission['rate_usd_h'] <= 0
            or admission['remaining_noncompute_reserve_usd'] < 0
            or not BOUND + 40 <= admission['operator_deadline_unix_s'] - now <= 10800):
        raise ValueError('Stale budget or insufficient bounded time')
    remaining_hours = (admission['operator_deadline_unix_s'] - now) / 3600
    projected = (admission['spent_usd'] + admission['rate_usd_h'] * remaining_hours
                 + admission['remaining_noncompute_reserve_usd'])
    if projected > 5:
        raise ValueError('All-in bounded estimate exceeds5USD')
    preflight = contained(Path(config['runs_root']), admission['preflight_receipt'])
    if digest(preflight) != admission['preflight_sha256']:
        raise ValueError('Preflight SHA differs')
    checked = json.loads(preflight.read_text())
    if (checked.get('status') != 'PASS_CURRENT_HOST_SOURCE_RUNTIME_CONTROLS'
            or checked.get('boot_id') != boot_id
            or checked.get('gpu_uuid') != config['gpu_uuid']
            or checked.get('instance_id') != config['instance_id']
            or checked.get('spec_sha256') != admission['spec_sha256']
            or not 0 <= now - checked['unix_s'] <= 300):
        raise ValueError('Fresh same-host preflight required')


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
            or os.environ.get('G6_FORMAT_CONFIRM_BOUND_S') != str(BOUND)
            or not re.fullmatch('[a-f0-9]{40}', args.protocol_commit)):
        raise ValueError('Verified pushed commit, supervised host and exact GPU required')
    operational_admission(config, admission, time.time(),
                          Path('/proc/sys/kernel/random/boot_id').read_text().strip())
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
        remaining = int(admission['operator_deadline_unix_s'] - time.time() - 40)
        guard = Guard(args.out, min(BOUND, remaining), gpu=True, uuid=config['gpu_uuid'])
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
        diagnostic['cases'] = [c for c in diagnostic['cases'] if c['kind'] == 'safety']
        # Alternating fixed order balances intervention order across model labels.
        order = ['comparator', 'candidate'] if LABELS.index(args.label) % 2 == 0 else (
            ['candidate', 'comparator'])
        phases = {}
        for name, suite in [('diagnostic', diagnostic), ('confirmation', fresh)]:
            for decoder in order:
                destination = args.out / (name + '_' + decoder + '.jsonl')
                result = phase(args.out, name + '_' + decoder, guard, 1200,
                               lambda s=suite, p=destination, d=decoder: generate_pass(
                                   model, tokenizer, s, p, 'cuda:0', guard, d == 'candidate',
                                   spec['candidate_prefix']),
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
                    for decoder in ['comparator', 'candidate']]
            paired[name] = pair_audit(*data, suite, tokenizer, spec['candidate_prefix'])
        if state_hash(model, frozen_only=True) != frozen:
            raise ValueError('Frozen base changed')
        for path in args.out.glob('*_phase.json'):
            result = json.loads(path.read_text())['F1']
            if result['outcome'] != 'ok' or result['gpu_clock']['outcome'] != 'ok':
                raise ValueError('Invalid F1 or clocks; no accepted inference result')
        write(args.out / 'receipt.json', {
            'status': 'FORMAT_PROMPT_COMPARISON_REVIEW_PENDING', 'label': args.label,
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
        Console().print('FRESH_96_ORACLES_AND_PINNED_SOURCE_PASS; no GPU/model generation')
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
                       G6_FORMAT_CONFIRM_BOUND_S=str(BOUND))
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
