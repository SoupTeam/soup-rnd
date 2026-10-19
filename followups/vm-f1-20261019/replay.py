import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tarfile
from pathlib import Path

import torch

ROOT = Path(__file__).parent
parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, required=True)
REPO = parser.parse_args().repo
control = ROOT
export = json.loads((control/'export.json').read_text())
archive_path = control/'raw.tgz'
assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == export['sha256']
out = control/'extracted'
out.mkdir(exist_ok=True)
with tarfile.open(archive_path) as archive:
    for member in archive.getmembers():
        target = os.path.realpath(out/member.name)
        assert os.path.commonpath([str(out),target]) == str(out)
        assert member.isfile() or member.isdir()
    archive.extractall(out)
source_sha = (control/'source-sha.txt').read_text().strip()
source_files = 0
with tarfile.open(out/'source.tar') as archive:
    for member in archive.getmembers():
        if member.isfile():
            data = archive.extractfile(member).read()
            expected = subprocess.check_output(['git','show',source_sha+':'+member.name],cwd=REPO)
            assert data.replace(b'\r\n',b'\n') == expected.replace(b'\r\n',b'\n')
            source_files += 1
audit = {'source_sha':source_sha,'source_files_verified':source_files,
         'archive_sha256':export['sha256'],'scope':'current BF16 descriptors, single GPU cohort'}
results = out/'results'
alias = json.loads((results/'native-gates/large-alias.json').read_text())
pt = results/'native-gates/large-alias.pt'
assert hashlib.sha256(pt.read_bytes()).hexdigest() == alias['witness_sha256']
values = torch.load(pt,map_location='cpu',weights_only=True)
audit['large_alias'] = {
    'forward_exact':torch.equal(values['forward'],values['native_forward']),
    'dx_exact':torch.equal(values['dx'],values['native_dx']),
    'all_finite':all(bool(torch.isfinite(v).all()) for v in values.values()),
    'negative_control_detected':bool(torch.isfinite(values['broken_dx']).all())
        and not torch.equal(values['broken_dx'],values['native_dx']),
    'recompute_calls':alias['recompute_calls']}
if not alias['passed']:
    audit['formal_status']='NOT_RUN: correctness failed'
else:
    assert all(audit['large_alias'][key] for key in
        ('forward_exact','dx_exact','all_finite','negative_control_detected'))
    assert alias['recompute_calls']==2
    prepared = results/'prepare'
    def compare(expected,actual):
        assert set(expected)==set(actual) and len(expected)==34
        assert all(torch.equal(expected[k],actual[k]) and bool(torch.isfinite(actual[k]).all())
                   for k in expected)
        gradients=[v for k,v in actual.items() if k.startswith('gradient:')]
        assert len(gradients)==16 and all(bool(torch.count_nonzero(v)) for v in gradients)
        return {'finite_exact_quantities':34,'nonzero_gradients':16}
    native=torch.load(prepared/'native.pt',map_location='cpu',weights_only=True)
    candidate=torch.load(prepared/'candidate.pt',map_location='cpu',weights_only=True)
    assert len(native)==len(candidate)==3
    audit['streaming_updates']=[compare(a,b) for a,b in zip(native,candidate)]
    resident=torch.load(prepared/'resident-gate/witness.pt',map_location='cpu',weights_only=True)
    assert len(resident)==3
    audit['resident_updates']=[compare(r['resident'],r['candidate']) for r in resident]
    dispatch=json.loads((results/'dispatch/result.json').read_text())
    assert dispatch['passed'] and dispatch['dispatch_counts']['candidate']>0
    audit['dispatch_counts']=dispatch['dispatch_counts']
    sys.path.insert(0,str(REPO/'benchmarks/harness'))
    from d1_rules import summarize
    from d1_reference import validate_reference
    summary_path=results/'f1/summary.json'
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else None
    if summary is None:
        stop=json.loads((results/'controller-stop.json').read_text())
        assert stop['formal_status']=='NO_VERDICT'
        audit['controller_stop']=stop
    rows=[]
    for path in sorted((results/'f1').glob('block-*-run-*/watched.json')):
        row=json.loads(path.read_text())
        timings=row['step_wall_ms']
        assert len(timings)==200 and all(math.isfinite(x) and x>0 for x in timings)
        assert statistics.median(timings)==row['median_step_ms']
        assert row['actual_timed_steps']==200 and row['actual_warmup_steps']==20
        assert row['source_sha']==source_sha
        if row['arm']=='B':assert row['dispatch_counts']['candidate']>0
        rows.append(row)
    replay=summarize(rows)
    if summary is not None:assert replay['comparisons']==summary['comparisons']
    else:assert replay['status']=='NO_VERDICT'
    audit['formal_status']=replay['status']
    audit['comparisons']=replay['comparisons']
    audit['arms_verified']=len(rows)
    audit['timed_steps_verified']=sum(len(r['step_wall_ms']) for r in rows)
    clocks=[c for r in rows for w in (r,r['dense_gemm_reference']) for c in w['clock_sm_mhz']]
    audit['sm_clock_range_mhz']=[min(clocks),max(clocks)]
    audit['dense_reference_validations']=[validate_reference(r,r['dense_gemm_reference']) for r in rows]
    audit['telemetry_power_limit_mw']=sorted({v for r in rows for v in r['power_limit_mw']})
    audit['gpu_uuid']=rows[0]['gpu_uuid']
    audit['sizing_extra']=summary.get('sizing_extra') if summary else None
(control/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit))
