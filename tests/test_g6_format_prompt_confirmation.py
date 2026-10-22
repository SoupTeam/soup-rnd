"""Reject altered prompts, raw evidence, oracles and scoring in the new comparison."""
import copy
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'benchmarks/g6'))
from format_prompt_confirmation import pair_audit, rendered, validate_suite  # noqa: E402
from turn_boundary_candidate import inspect_prefix  # noqa: E402

ROOT = Path(__file__).parents[1] / 'benchmarks/g6'


def test_new_suite_and_mutations():
    fresh = json.loads((ROOT / 'spec/format_confirmation_suite_v1.json').read_text())
    old = json.loads((ROOT / 'spec/turn_confirmation_suite_v1.json').read_text())
    validate_suite(fresh, old)
    for change in ['answer', 'duplicate', 'old_prompt', 'scorer']:
        bad = copy.deepcopy(fresh)
        if change == 'answer':
            bad['cases'][0]['expected'] = 'wrong'
        elif change == 'duplicate':
            bad['cases'][1] = bad['cases'][0]
        elif change == 'old_prompt':
            bad['cases'][0]['prompt'] = old['cases'][0]['prompt']
        else:
            bad['cases'][0]['scorer'] = 'json_object'
        with pytest.raises(ValueError):
            validate_suite(bad, old)


def test_questions_match_operands():
    """Read each rendered question, separately from the operand-only oracle."""
    suite = json.loads((ROOT / 'spec/format_confirmation_suite_v1.json').read_text())
    for case in suite['cases']:
        prompt, data, family = case['prompt'], case['operands'], case['family']
        if family in ['add', 'subtract', 'multiply', 'sort']:
            assert list(map(int, re.findall(r'\d+', prompt))) == data
        elif family == 'extract':
            assert f"code:{data['code']}," in prompt
        elif family == 'count':
            words = prompt.split(': ', 1)[1].split('. Reply', 1)[0].split(', ')
            target = prompt.split("'")[1]
            assert words.count(target) == int(case['expected'])
        elif family == 'last':
            assert prompt.split(': ', 1)[1].split('. Reply', 1)[0].split('; ')[-1] == (
                case['expected'])
        elif family == 'format':
            assert prompt.startswith(f'Uppercase {data[0]}, then {data[1]};')
        elif family == 'logic':
            assert prompt.startswith(f"Every {data['group']} is ")
            assert f"{data['person']} is a {data['group']}." in prompt
            assert (' is not ' not in prompt) == data['all_members_have_property']
        elif family == 'context':
            for person, day in data['days'].items():
                assert f'{person}={day}' in prompt
            assert f"assigned to {data['query']}," in prompt
        elif family == 'reverse':
            assert prompt.split(': ', 1)[1].split('.')[0][::-1] == case['expected']
        elif family == 'json':
            assert f'"{data["color"]}"' in prompt
            assert f"{data['count']} (integer)" in prompt


class Tokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text, **kwargs):
        return [ord(c) + 10 for c in text]

    def decode(self, ids, **kwargs):
        return ''.join(chr(n - 10) for n in ids if n not in [1, 2])


def evidence():
    tok = Tokenizer()
    case = {'id': 'one', 'kind': 'task', 'prompt': 'Sum?',
            'expected': '5', 'scorer': 'strip_exact'}
    rows = []
    for changed, text in [(False, '5.\nuser:'), (True, '5\nuser:')]:
        ids = tok.encode(text)
        inspected = inspect_prefix(ids, tok)
        rows.append({'id': 'one', 'prompt_ids': [1, *tok.encode(
            rendered(case['prompt'], changed, 'Examples:\n'))],
            'raw_completion_ids': ids, 'raw_completion': text,
            'scored_completion': inspected['candidate_scored_text'],
            'boundary': inspected['boundary'], 'decoder': 'role_boundary_v1',
            'task_pass': changed, 'stop_reason': 'role_boundary'})
    return [rows[0]], [rows[1]], {'cases': [case]}, tok, 'Examples:\n'


def test_different_prompts_are_audited_without_prefix_claim():
    result = pair_audit(*evidence())
    assert result['gains'] == 1 and result['losses'] == 0
    assert result['same_prefix_claim'] is False
    assert result['prompt_tokens_candidate'] > result['prompt_tokens_comparator']


@pytest.mark.parametrize('mutation', ['prompt', 'decoder', 'tokens', 'text', 'score',
                                      'reason', 'missing'])
def test_evidence_corruption_rejected(mutation):
    left, right, suite, tok, prefix = evidence()
    if mutation == 'prompt':
        right[0]['prompt_ids'] = left[0]['prompt_ids']
    elif mutation == 'decoder':
        right[0]['decoder'] = 'original_v1'
    elif mutation == 'tokens':
        right[0]['raw_completion_ids'][0] += 1
    elif mutation == 'text':
        right[0]['scored_completion'] = '5.'
    elif mutation == 'score':
        right[0]['task_pass'] = False
    elif mutation == 'reason':
        right[0]['stop_reason'] = 'eos'
    else:
        right.clear()
    with pytest.raises(ValueError):
        pair_audit(left, right, suite, tok, prefix)


@pytest.mark.parametrize('mutation', [None, 'stale', 'over_budget', 'boot', 'preflight'])
def test_budget_boot_and_preflight_admission(tmp_path, mutation):
    import hashlib

    from format_prompt_confirmation import operational_admission

    now = 10000.0
    preflight = tmp_path / 'preflight.json'
    data = {'status': 'PASS_CURRENT_HOST_SOURCE_RUNTIME_CONTROLS', 'boot_id': 'boot',
            'gpu_uuid': 'GPU-test', 'instance_id': 123, 'spec_sha256': 'a' * 64,
            'unix_s': now}
    preflight.write_text(json.dumps(data))
    config = {'account': 'soup-rnd', 'total_cap_usd': 5, 'boot_id': 'boot',
              'gpu_uuid': 'GPU-test', 'instance_id': 123, 'runs_root': str(tmp_path)}
    admission = {'budget_verified': True, 'boot_id': 'boot', 'observed_unix_s': now,
                 'operator_deadline_unix_s': now + 7200, 'spent_usd': 0.2,
                 'rate_usd_h': 0.6, 'remaining_noncompute_reserve_usd': 2,
                 'preflight_receipt': str(preflight), 'spec_sha256': 'a' * 64,
                 'preflight_sha256': hashlib.sha256(preflight.read_bytes()).hexdigest()}
    if mutation == 'stale':
        admission['observed_unix_s'] -= 301
    elif mutation == 'over_budget':
        admission['spent_usd'] = 3
    elif mutation == 'boot':
        admission['boot_id'] = 'old-boot'
    elif mutation == 'preflight':
        preflight.write_text('{}')
    if mutation:
        with pytest.raises(ValueError):
            operational_admission(config, admission, now, 'boot')
    else:
        operational_admission(config, admission, now, 'boot')
