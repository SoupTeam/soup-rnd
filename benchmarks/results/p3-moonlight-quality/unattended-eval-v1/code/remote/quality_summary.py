"""Run the published paired analysis once; no scheduling or provider actions."""

def main():
    import hashlib
    import json
    import os
    import sys
    from pathlib import Path
    from rich.console import Console
    from benchmarks.harness.p3_stage2_statistics import summarize_pairs
    from benchmarks.harness.p3_stage2v2_decision import apply_informativeness_gate
    from benchmarks.harness.p3_stage2v2_results import ARMS, summarize_nll
    root, scoring_path = map(Path, sys.argv[1:3])
    protocol_sha256 = sys.argv[3]
    scoring_bytes = scoring_path.read_bytes()
    assert hashlib.sha256(scoring_bytes).hexdigest() == (
        '685fa5bee528969d9b2eef423c90280d179432ba017fbebed12b687f608a8249')
    scoring = json.loads(scoring_bytes)
    manifest_path = root / 'manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest['status'] == 'COMPLETE'
    assert manifest['profile_id'] == 'moonlight-16b-a3b'
    assert manifest['include_base_model_evaluation'] is False
    assert manifest['seeds'] == [23]
    assert manifest['protocol_sha256'] == protocol_sha256
    assert manifest['scoring_protocol']['sha256'] == hashlib.sha256(scoring_bytes).hexdigest()
    payload = {key: value for key, value in manifest.items() if key != 'manifest_payload_sha256'}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                           separators=(',', ':'), allow_nan=False).encode()
    assert hashlib.sha256(canonical).hexdigest() == manifest['manifest_payload_sha256']
    assert set(manifest['files']) == {'pairs.jsonl', 'nll-pairs.jsonl'}
    root_real = os.path.realpath(root)
    for name, descriptor in manifest['files'].items():
        path = root / name
        assert os.path.commonpath((root_real, os.path.realpath(path))) == root_real
        assert path.stat().st_size == descriptor['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == descriptor['sha256']
    assert (scoring['margin'], scoring['one_sided_alpha'], scoring['floor_offset']) == (0.01, 0.025, 0.05)
    rows = [json.loads(line) for line in (root / 'pairs.jsonl').read_text().splitlines()]
    assert len(rows) == manifest['primary_rows']
    raw = summarize_pairs(rows, expected_tasks=manifest['expected_tasks'],
                          seeds=manifest['seeds'], lengths=manifest['lengths'],
                          margin=scoring['margin'], one_sided_alpha=scoring['one_sided_alpha'])
    quality = apply_informativeness_gate(raw, chance=scoring['chance'],
                                        floor_offset=scoring['floor_offset'])
    nll_rows = []
    assert ARMS == ('bf16_l2l', 'oracle_block64_e4m3')
    paired_nll = (root / 'nll-pairs.jsonl').read_text().splitlines()
    assert all(row['profile_id'] == 'moonlight-16b-a3b' for row in rows)
    assert len(paired_nll) == manifest['secondary_rows']
    for line in paired_nll:
        row = json.loads(line)
        assert row['profile_id'] == 'moonlight-16b-a3b'
        for label, arm in zip(('baseline', 'candidate'), ARMS, strict=True):
            nll_rows.append({
                'endpoint': row['benchmark'], 'task_id': row['task_id'],
                'length': row['length'], 'seed': row['seed'], 'arm': arm,
                'n_targets': row['n_targets'], 'nll_sum': row[label + '_nll_sum'],
                'nll_mean': row[label + '_nll_mean'],
            })
    secondary = summarize_nll(nll_rows, expected_tasks=manifest['expected_secondary_tasks'],
                              seeds=manifest['seeds'], lengths=manifest['lengths'])
    assert all(cell['status'] == 'COMPLETE' for cell in secondary)
    result = {'quality': quality, 'secondary_nll': secondary,
              'gated_quality_decision': quality['decision'],
              'statistical_quality_decision': quality['statistical_decision'],
              'include_base_model_evaluation': False,
              'pair_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
              'scoring_protocol_sha256': hashlib.sha256(scoring_bytes).hexdigest()}
    with (root.parent / 'quality-summary.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    Console().print({'gated': quality['decision'], 'statistical': quality['statistical_decision']})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
