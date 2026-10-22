"""Pure checks for fresh-oracle integrity and paired-generation evidence rejection."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'benchmarks/g6'))
from boundary_confirmation import oracle, pair_audit, validate_suite  # noqa: E402


def test_fresh_oracles_and_balance():
    path = Path(__file__).parents[1] / 'benchmarks/g6/spec/turn_confirmation_suite_v1.json'
    suite = json.loads(path.read_text())
    validate_suite(suite)
    mutated = copy.deepcopy(suite)
    mutated['cases'][0]['expected'] = 'wrong'
    with pytest.raises(ValueError, match='oracle'):
        validate_suite(mutated)
    mutated = copy.deepcopy(suite)
    mutated['cases'][-1] = mutated['cases'][0]
    with pytest.raises(ValueError, match='distinct'):
        validate_suite(mutated)


@pytest.mark.parametrize(('family', 'operands', 'expected'), [
    ('add', [216, 487], '703'), ('subtract', [705, 398], '307'),
    ('multiply', [29, 16], '464'), ('sort', [51, 3, 20], '3,20,51'),
    ('count', {'words': ['fig', 'fig', 'kiwi'], 'target': 'fig'}, '2'),
    ('reverse', 'R4TC82', '28CT4R'), ('format', ['hazel', 'coral'], 'HAZEL/CORAL'),
    ('last', ['agate', 'garnet'], 'garnet'),
    ('logic', {'all_members_have_property': False}, 'no'),
    ('extract', {'code': 'B431'}, 'B431'),
    ('context', {'days': {'Bela': 'Sunday'}, 'query': 'Bela'}, 'Sunday'),
    ('json', {'color': 'ivory', 'count': 11}, '{"color":"ivory","count":11}'),
])
def test_oracles(family, operands, expected):
    assert oracle({'family': family, 'operands': operands}) == expected


def evidence():
    case = {'id': 'one', 'kind': 'task', 'expected': '703', 'scorer': 'strip_exact'}
    old = {'id': 'one', 'prompt_ids': [1, 91], 'raw_completion_ids': [7, 8, 9],
           'raw_completion': '703\nuser: explanation',
           'scored_completion': '703\nuser: explanation', 'decoder': 'original_v1',
           'task_pass': False, 'stop_reason': 'max_new_tokens', 'boundary': None}
    new = {'id': 'one', 'prompt_ids': [1, 91], 'raw_completion_ids': [7, 8],
           'raw_completion': '703\nuser: overhang', 'scored_completion': '703',
           'decoder': 'role_boundary_v1', 'task_pass': True, 'stop_reason': 'role_boundary',
           'boundary': {'character_offset': 3, 'marker': '\nuser:'}}
    return [old], [new], {'cases': [case]}


def test_boundary_keeps_raw_token_overhang_but_scores_before_marker():
    result = pair_audit(*evidence())
    assert result['gains'] == 1 and result['losses'] == 0
    assert result['all_raw_prefixes_match'] is True
    assert result['quality_verdict'] == 'NO_VERDICT'


@pytest.mark.parametrize('change', ['prompt', 'prefix', 'score', 'text', 'reason', 'missing'])
def test_corrupted_evidence_fails_closed(change):
    old, new, suite = evidence()
    if change == 'prompt':
        new[0]['prompt_ids'] = [1, 92]
    elif change == 'prefix':
        new[0]['raw_completion_ids'] = [70, 8]
    elif change == 'score':
        new[0]['task_pass'] = False
    elif change == 'text':
        new[0]['scored_completion'] = '703\n'
    elif change == 'reason':
        new[0]['stop_reason'] = 'eos'
    else:
        new.clear()
    with pytest.raises(ValueError):
        pair_audit(old, new, suite)


def test_fresh_prompt_operands_agree_with_answers():
    """Check rendered questions independently of the oracle's operand-only path."""
    import re

    path = Path(__file__).parents[1] / 'benchmarks/g6/spec/turn_confirmation_suite_v1.json'
    for case in json.loads(path.read_text())['cases']:
        prompt, data, family = case['prompt'], case['operands'], case['family']
        if family in ['add', 'subtract', 'multiply']:
            assert [int(n) for n in re.findall(r'\d+', prompt)] == data
        elif family == 'sort':
            assert [int(n) for n in re.findall(r'\d+', prompt)] == data
        elif family == 'count':
            words = prompt.split('this list: ', 1)[1].split('?', 1)[0].split()
            assert words.count(data['target']) == int(case['expected'])
        elif family == 'reverse':
            value = prompt.split('characters of ', 1)[1].split(' in reverse', 1)[0]
            assert value[::-1] == case['expected']
        elif family == 'extract':
            assert f"code={case['expected']} |" in prompt
        elif family == 'last':
            items = prompt.split('spelling: ', 1)[1].split('. Output', 1)[0].split(' | ')
            assert items[-1] == case['expected']
        elif family == 'format':
            assert f'Write {data[0]} and {data[1]} in uppercase' in prompt
        elif family == 'logic':
            positive = f"is {data['property']}." in prompt
            negative = f"is not {data['property']}." in prompt
            assert positive != negative
            assert case['expected'] == ('yes' if positive else 'no')
        elif family == 'context':
            for person, day in data['days'].items():
                assert f'{person} checked in on {day}' in prompt
            assert f"weekday for {data['query']}" in prompt
        elif family == 'json':
            assert f"string {data['color']}" in prompt
            assert f"integer {data['count']}." in prompt
