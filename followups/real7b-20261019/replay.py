import hashlib,json,math,os,statistics,subprocess,sys,tarfile
from pathlib import Path
import torch
root=Path(__file__).parent
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,required=True)
repo=parser.parse_args().repo
export=json.loads((root/'export.json').read_text())
archive=root/'raw.tgz'
if not archive.exists():
 info=json.loads((root/'raw-parts.json').read_text())
 with archive.open('xb') as output:
  for part in info['parts']:
   data=(root/part['name']).read_bytes()
   assert len(data)==part['bytes'] and hashlib.sha256(data).hexdigest()==part['sha256']
   output.write(data)
assert hashlib.sha256(archive.read_bytes()).hexdigest()==export['sha256']
out=root/'extracted';out.mkdir(exist_ok=True)
with tarfile.open(archive) as handle:
 for member in handle.getmembers():
  dest=os.path.realpath(out/member.name)
  assert os.path.commonpath([str(out),dest])==str(out)
  assert member.isfile() or member.isdir()
 handle.extractall(out)
sha=(root/'source-sha.txt').read_text().strip();count=0
with tarfile.open(out/'source.tar') as handle:
 for member in handle.getmembers():
  if not member.isfile():continue
  data=handle.extractfile(member).read()
  expected=subprocess.check_output(['git','show',sha+':'+member.name],cwd=repo)
  assert data.replace(b'\r\n',b'\n')==expected.replace(b'\r\n',b'\n'),member.name
  count+=1
sys.path.insert(0,str(repo/'benchmarks/harness'))
from d1_rules import compare_tensors,peer_check
from arm_validity import arm_outcome
from d1_reference import validate_reference
results=out/'results'
report={'source_sha':sha,'source_files_verified':count,'archive_sha256':export['sha256']}
repair=json.loads((root/'io-preparation-repair.json').read_text())
old_sha=repair['original_source_sha']
changed=subprocess.check_output(['git','diff','--name-only',old_sha,sha],cwd=repo,text=True).splitlines()
assert changed==['benchmarks/harness/d1_real7b.py']
prior=results/'attempt-01-preparation/source.tar'
plan=json.loads((root/'booking-plan.json').read_text())
assert hashlib.sha256(prior.read_bytes()).hexdigest()==plan['source_tar_sha256']
report['io_repair']={'original_source_sha':old_sha,'corrected_source_sha':sha,'prior_source_tar_sha256':plan['source_tar_sha256'],'only_changed_file':changed[0],'native_gate_numeric_source_unchanged':True}

if not (results/'native-gates/large-alias.json').exists():
 report['status']='STOPPED_BEFORE_CUDA_GATE'
 (root/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report));raise SystemExit(0)
alias=json.loads((results/'native-gates/large-alias.json').read_text())
pt=results/'native-gates/large-alias.pt'
assert hashlib.sha256(pt.read_bytes()).hexdigest()==alias['witness_sha256']
v=torch.load(pt,map_location='cpu',weights_only=True)
report['large_alias']={'forward_exact':torch.equal(v['forward'],v['native_forward']),'dx_exact':torch.equal(v['dx'],v['native_dx']),'all_finite':all(bool(torch.isfinite(value).all()) for value in v.values()),'negative_control_detected':bool(torch.isfinite(v['broken_dx']).all()) and not torch.equal(v['broken_dx'],v['native_dx']),'recompute_calls':alias['recompute_calls']}
assert all(report['large_alias'][key] for key in ['forward_exact','dx_exact','all_finite','negative_control_detected']) and report['large_alias']['recompute_calls']==2

prepared=json.loads((results/'model-prepared.json').read_text())
upstream=json.loads((root/'upstream-weight-identities.json').read_text())
expected={name:value['oid'] for name,value in upstream['weights'].items() if name.startswith('model-')}
assert prepared['model_file_hashes']==expected and prepared['revision']==upstream['revision']
assert prepared['layers']==32 and prepared['tier']=='ram' and prepared['buffers']==2 and prepared['pin'] is False
assert hashlib.sha256((results/'inputs.pt').read_bytes()).hexdigest()==prepared['inputs_sha256']
report['pretrained_checkpoint_verified']={'model':upstream['model'],'revision':upstream['revision'],'all_three_weight_shas_match_upstream_lfs':True,'layers':32,'inputs_sha256':prepared['inputs_sha256']}
run=results/'real7b'
if not (run/'correctness.json').exists():
 report['status']='STOPPED_BEFORE_COMPARISON'
else:
 actual=json.loads((run/'correctness.json').read_text())
 bundles=[]
 for mode in ['native','candidate']:
  info=json.loads((run/mode/'result.json').read_text())
  witness=run/mode/'witness.pt'
  assert hashlib.sha256(witness.read_bytes()).hexdigest()==info['witness_sha256']
  assert info['source_sha']==sha
  assert all(x>0 for x in info['nonzero_gradients'])
  assert all(math.isfinite(x) for x in info['losses'])
  if mode=='candidate':assert info['dispatch_counts']['candidate']>0
  bundles.append(torch.load(witness,map_location='cpu',weights_only=True))
 assert all(len(bundle)==3 for bundle in bundles)
 comparisons=[compare_tensors(b,a) for a,b in zip(*bundles,strict=True)]
 assert comparisons==actual['updates']
 report['correctness_passed']=all(row['passed'] for row in comparisons)
 report['quantities_verified']=sum(len(row['quantities']) for row in comparisons)
 report['native_losses']=json.loads((run/'native/result.json').read_text())['losses']
 report['nonzero_gradients']=json.loads((run/'native/result.json').read_text())['nonzero_gradients']
 report['differing_quantities']=[[q for q in row['quantities'] if not q['exact']] for row in comparisons]
 if not report['correctness_passed']:
  assert not list(run.glob('arm-*'))
  report['status']='PARITY_FAIL_NO_TIMING'
 else:
  assert (run/'summary.json').exists()
  summary=json.loads((run/'summary.json').read_text())
  rows=[]
  for path in sorted(run.glob('arm-*/watched.json')):
   row=json.loads(path.read_text());rows.append(row)
   assert row['source_sha']==sha
   assert len(row['step_wall_ms'])==row['actual_timed_steps']==20
   assert row['actual_warmup_steps']==5
   assert all(math.isfinite(x) and x>0 for x in row['step_wall_ms'])
   assert statistics.median(row['step_wall_ms'])==row['median_step_ms']
   assert row['reference_validation']==validate_reference(row,row['dense_gemm_reference'])
   life=row['peer_lifecycle']
   assert row['peers_clean']==peer_check(life['before'],life['during'],life['after'],row['gpu_uuid'])
   validity=row['validity']
   assert validity['outcome']==arm_outcome(validity['checks'],validity['required'])
  assert len(rows)==4 and [row['arm'] for row in rows]==list('ABBA')
  for field in ['median_ms','peak_allocated_bytes','peak_reserved_bytes']:
   key={'median_ms':'median_step_ms'}.get(field,field)
   aggregate=statistics.median if field=='median_ms' else max
   expected={arm:aggregate(row[key] for row in rows if row['arm']==arm) for arm in 'AB'}
   assert summary[field]==expected
  windows=[window for row in rows for window in (row,row['dense_gemm_reference'])]
  clocks=[value for window in windows for value in window['clock_sm_mhz']]
  limits=[value for window in windows for value in window['power_limit_mw']]
  assert summary['clock_range_mhz']==[min(clocks),max(clocks)]
  clean=(min(clocks)>0 and max(clocks)<=min(clocks)*1.05 and len(set(limits))==1
         and all(row['validity']['outcome']=='ok' and row['peers_clean'] and row['reference_validation']['passed'] for row in rows))
  assert summary['status']==('CLEAN_DESCRIPTIVE_SCREEN' if clean else 'NO_VERDICT_TIMING')
  assert summary['formal_f1'] is False and summary['quality_evaluation'] is False
  report.update(status=summary['status'],arms_verified=4,timed_steps_verified=80,median_ms=summary['median_ms'],peak_allocated_bytes=summary['peak_allocated_bytes'],peak_reserved_bytes=summary['peak_reserved_bytes'],clock_range_mhz=summary['clock_range_mhz'])
(root/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
