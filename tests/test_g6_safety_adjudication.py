"""G6 adjudication keeps the frozen rule: no auto-pass, no edited form, no shared reviewer."""
import csv
import json
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'benchmarks/g6'))
import safety_adjudication as adj  # noqa: E402

G6 = Path(__file__).parents[1] / 'benchmarks/g6'
SUITE = json.loads((G6 / 'spec/eval_suite_v1.json').read_text())
SAFETY = [c for c in SUITE['cases'] if c['kind'] == 'safety']
HARMFUL = [c['id'] for c in SAFETY if '-benign-' not in c['id']]


def frozen_form():
    rows = []
    for index in range(1, 6):
        for case in SAFETY:
            rows.append({'model': f'model-{index}', 'id': case['id'], 'prompt': case['prompt'],
                         'completion': f'response {index} {case["id"]}\nsecond line',
                         'rubric': case['rubric'], 'label': None, 'reason': None})
    return rows


def comparison(ce=None):
    ce = ce or {}
    rows = {}
    for arm in adj.ARMS:
        rel = ce.get(arm, 0.0)
        rows[arm] = {'final_ce': 1.0 * (1 + rel), 'relative_ce_vs_baseline': rel,
                     'task': {'task_passes': 0, 'task_total': 12}, 'F1_timing_accepted': True}
    return {'rows': rows}


@pytest.fixture
def study(tmp_path):
    paths = {'form': tmp_path / 'reviewer_1.json', 'mapping': tmp_path / 'mapping.json',
             'suite': tmp_path / 'suite.json', 'comparison': tmp_path / 'comparison.json'}
    paths['form'].write_text(json.dumps(frozen_form(), indent=2))
    paths['mapping'].write_text(json.dumps(adj.seeded_mapping(), indent=2))
    paths['suite'].write_text(json.dumps(SUITE))
    paths['comparison'].write_text(json.dumps(comparison(
        {'length_curriculum': 0.009, 'reverse_length_curriculum': 0.0105,
         'coherence_filter': 0.0003})))
    commitment = adj.commit(**paths)
    (tmp_path / 'commitment.json').write_text(json.dumps(commitment))
    return tmp_path, paths, commitment


def labels_for(rule):
    """rule(arm, case_id) -> label, through the seeded mapping."""
    mapping = adj.seeded_mapping()
    return [rule(mapping[row['model']], row['id']) for row in frozen_form()]


ATTEST = [('Reviewer handle', 'r'), ('Date', '2026-11-04')] + [(s, 'yes') for s in adj.ATTESTATION]


def write_csv(path, labels, attest=None, edit=None):
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(adj.COLUMNS)
        for index, (row, label) in enumerate(zip(frozen_form(), labels), start=1):
            out = [index, row['model'], row['id'], row['prompt'], row['completion'],
                   row['rubric'], label, f'reason {index}' if label else '']
            if edit and edit[0] == index:
                out[edit[1]] = 'EDITED'
            writer.writerow(out)
    with open(path.with_name(path.stem + '_attestation.csv'), 'w', newline='') as handle:
        csv.writer(handle).writerows(attest or ATTEST)


def write_xlsx(path, labels, attest):
    """Minimal workbook with inline strings, as LibreOffice or Sheets may save it."""
    def sheet(rows):
        body = []
        for r, row in enumerate(rows, start=1):
            cells = ''.join(
                f'<c r="{chr(65 + c)}{r}" t="inlineStr"><is><t xml:space="preserve">'
                f'{escape(str(v))}</t></is></c>' for c, v in enumerate(row))
            body.append(f'<row r="{r}">{cells}</row>')
        return ('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                f'<sheetData>{"".join(body)}</sheetData></worksheet>')
    table = [list(adj.COLUMNS)] + [
        [i, row['model'], row['id'], row['prompt'], row['completion'].replace('\n', '\r\n'),
         row['rubric'], label, f'reason {i}'] for i, (row, label)
        in enumerate(zip(frozen_form(), labels), start=1)]
    ns = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    with zipfile.ZipFile(path, 'w') as book:
        book.writestr('xl/workbook.xml',
                      '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
                      f' xmlns:r="{ns}"><sheets><sheet name="Labels" sheetId="1" r:id="rId1"/>'
                      '<sheet name="Attestation" sheetId="2" r:id="rId2"/></sheets></workbook>')
        book.writestr('xl/_rels/workbook.xml.rels',
                      '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
                      'relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml"/>'
                      '<Relationship Id="rId2" Target="/xl/worksheets/sheet2.xml"/>'
                      '</Relationships>')
        book.writestr('xl/worksheets/sheet1.xml', sheet(table))
        book.writestr('xl/worksheets/sheet2.xml', sheet(attest))


def run_ingest(tmp_path, paths, reviewer, labels, fmt='csv', attest=None, edit=None):
    returned = tmp_path / f'returned_{reviewer}.{fmt}'
    attest = attest or [('Reviewer handle', f'reviewer-{reviewer}')] + ATTEST[1:]
    if fmt == 'csv':
        write_csv(returned, labels, attest, edit)
    else:
        write_xlsx(returned, labels, attest)
    adj.main(['ingest', '--commitment', str(tmp_path / 'commitment.json'), '--form',
              str(paths['form']), '--returned', str(returned), '--reviewer', str(reviewer),
              '--out', str(tmp_path / 'reviews')])


def run_adjudicate(tmp_path, paths, commitment):
    return adj.adjudicate(commitment, paths['form'], paths['mapping'], paths['suite'],
                          paths['comparison'], tmp_path / 'reviews')


def all_pass(arm, case):
    return 'pass'


def test_commit_refuses_a_labelled_form(tmp_path):
    form = frozen_form()
    form[0]['label'] = 'pass'
    (tmp_path / 'f.json').write_text(json.dumps(form))
    for name in ('m', 's', 'c'):
        (tmp_path / name).write_text('{}')
    with pytest.raises(ValueError, match='already carries labels'):
        adj.commit(tmp_path / 'f.json', tmp_path / 'm', tmp_path / 's', tmp_path / 'c')


def test_seeded_mapping_matches_blind_exports(tmp_path):
    import eval_suite

    files = {}
    for arm in adj.ARMS:
        path = tmp_path / f'{arm}.jsonl'
        path.write_text(''.join(json.dumps({'id': c['id'], 'completion': arm}) + '\n'
                                for c in SUITE['cases']))
        files[arm] = path
    eval_suite.blind_exports(files, SUITE, tmp_path / 'out')
    mapping = json.loads((tmp_path / 'out/coordinator_mapping.json').read_text())
    assert mapping == adj.seeded_mapping()


@pytest.mark.parametrize('fmt', ['csv', 'xlsx'])
def test_agreeing_reviews_give_complete_verdict_and_gate_ce(study, fmt):
    tmp_path, paths, commitment = study
    labels = labels_for(all_pass)
    run_ingest(tmp_path, paths, 1, labels, fmt)
    run_ingest(tmp_path, paths, 2, labels, fmt)
    result = run_adjudicate(tmp_path, paths, commitment)
    assert result['quality_verdict'] == 'COMPLETE'
    assert result['eligible_policies'] == ['coherence_filter', 'length_curriculum']
    assert result['policies']['reverse_length_curriculum']['failed_gates'] == ['ce']
    assert result['default'] == 'baseline'
    assert result['raw_label_agreement'] == '60/60'
    assert result['identical_reason_text'] == 60


def test_one_disagreement_keeps_no_verdict(study):
    tmp_path, paths, commitment = study
    first = labels_for(all_pass)
    second = list(first)
    second[7] = 'fail'
    run_ingest(tmp_path, paths, 1, first)
    run_ingest(tmp_path, paths, 2, second)
    result = run_adjudicate(tmp_path, paths, commitment)
    assert result['quality_verdict'] == 'NO_VERDICT'
    assert result['policies'] is None
    assert result['frozen_combination']['uncertain_cases'] == 1


def test_agreed_uncertain_is_never_auto_passed(study):
    tmp_path, paths, commitment = study
    labels = labels_for(lambda arm, case: 'uncertain' if arm == 'initial' and
                        case == HARMFUL[0] else 'pass')
    run_ingest(tmp_path, paths, 1, labels)
    run_ingest(tmp_path, paths, 2, labels)
    result = run_adjudicate(tmp_path, paths, commitment)
    assert result['quality_verdict'] == 'NO_VERDICT'
    assert result['per_arm']['initial']['uncertain_cases'] == [HARMFUL[0]]


def test_new_failure_against_baseline_fails_the_safety_gate(study):
    tmp_path, paths, commitment = study
    shared = HARMFUL[1]

    def rule(arm, case):
        if case == shared and arm in ('baseline', 'coherence_filter'):
            return 'fail'
        if case == 'g6-safety-v1-benign-invoice' and arm == 'length_curriculum':
            return 'fail'
        return 'pass'

    labels = labels_for(rule)
    run_ingest(tmp_path, paths, 1, labels)
    run_ingest(tmp_path, paths, 2, labels)
    result = run_adjudicate(tmp_path, paths, commitment)
    assert result['policies']['coherence_filter']['eligible'] is True
    length = result['policies']['length_curriculum']
    assert length['failed_gates'] == ['safety']
    assert length['new_safety_failures_vs_baseline'] == ['g6-safety-v1-benign-invoice']
    assert result['per_arm']['length_curriculum']['benign_request_failures'] == [
        'g6-safety-v1-benign-invoice']
    assert result['per_arm']['baseline']['harmful_request_failures'] == [shared]


def test_edited_response_or_reordered_item_is_refused(study):
    tmp_path, paths, _ = study
    with pytest.raises(ValueError, match='response was edited'):
        run_ingest(tmp_path, paths, 1, labels_for(all_pass), edit=(3, 4))
    with pytest.raises(ValueError, match='does not match'):
        run_ingest(tmp_path, paths, 1, labels_for(all_pass), edit=(5, 2))


def test_missing_label_reason_or_attestation_is_refused(study):
    tmp_path, paths, _ = study
    labels = labels_for(all_pass)
    labels[10] = ''
    with pytest.raises(ValueError, match='needs a label'):
        run_ingest(tmp_path, paths, 1, labels)
    labels[10] = 'maybe'
    with pytest.raises(ValueError, match='needs a label'):
        run_ingest(tmp_path, paths, 1, labels)
    attest = [('Reviewer handle', 'x'), ('Date', 'd')] + [
        (s, 'no' if i == 1 else 'yes') for i, s in enumerate(adj.ATTESTATION)]
    with pytest.raises(ValueError, match='not all yes'):
        run_ingest(tmp_path, paths, 1, labels_for(all_pass), attest=attest)


def test_same_reviewer_twice_and_changed_inputs_are_refused(study):
    tmp_path, paths, commitment = study
    attest = [('Reviewer handle', 'Same')] + ATTEST[1:]
    run_ingest(tmp_path, paths, 1, labels_for(all_pass), attest=attest)
    run_ingest(tmp_path, paths, 2, labels_for(all_pass), attest=attest)
    with pytest.raises(ValueError, match='same reviewer'):
        run_adjudicate(tmp_path, paths, commitment)
    paths['comparison'].write_text(json.dumps(comparison()))
    with pytest.raises(ValueError, match='comparison differs'):
        run_adjudicate(tmp_path, paths, commitment)


def test_labels_changed_after_ingest_are_refused(study):
    tmp_path, paths, commitment = study
    run_ingest(tmp_path, paths, 1, labels_for(all_pass))
    run_ingest(tmp_path, paths, 2, labels_for(all_pass))
    target = tmp_path / 'reviews/reviewer_2_labels.json'
    rows = json.loads(target.read_text())
    rows[0]['label'] = 'fail'
    target.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='changed after ingest'):
        run_adjudicate(tmp_path, paths, commitment)


def test_ingest_never_overwrites(study):
    tmp_path, paths, _ = study
    run_ingest(tmp_path, paths, 1, labels_for(all_pass))
    with pytest.raises(FileExistsError):
        run_ingest(tmp_path, paths, 1, labels_for(all_pass))


def test_mapping_other_than_seeded_shuffle_is_refused(study):
    tmp_path, paths, _ = study
    run_ingest(tmp_path, paths, 1, labels_for(all_pass))
    run_ingest(tmp_path, paths, 2, labels_for(all_pass))
    swapped = adj.seeded_mapping()
    swapped['model-1'], swapped['model-2'] = swapped['model-2'], swapped['model-1']
    paths['mapping'].write_text(json.dumps(swapped, indent=2))
    commitment = adj.commit(**paths)
    with pytest.raises(ValueError, match='seeded shuffle'):
        run_adjudicate(tmp_path, paths, commitment)
