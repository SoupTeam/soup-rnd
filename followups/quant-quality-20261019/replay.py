"""Independent CPU replay of saved original-BF16/NF4 observations."""
import argparse,hashlib,io,json,math,subprocess,tarfile
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,required=True)
args=parser.parse_args();root=Path(__file__).parent
archive=root/('cuda-results.tgz' if (root/'cuda-results.tgz').exists() else 'raw.tgz')
sha=lambda value: hashlib.sha256(value).hexdigest()
export=json.loads((root/'export.json').read_text()); data=archive.read_bytes()
assert sha(data)==export['sha256'] and len(data)==export['bytes']
with tarfile.open(fileobj=io.BytesIO(data)) as handle:
 def read(name): return handle.extractfile(name).read()
 def js(name): return json.loads(read(name))
 assert read('results/exit-code.txt').strip()==b'0'
 assert not any('network-budget-stop' in m.name for m in handle.getmembers())
 source=read('source.tar');source_sha=(root/'source-sha.txt').read_text().strip()
 expected=subprocess.check_output(['git','archive','--format=tar',source_sha,'benchmarks/harness/d1_quant_quality.py','benchmarks/d1-quant-quality-rule-20261019.md'],cwd=args.repo)
 assert source==expected
 with tarfile.open(fileobj=io.BytesIO(source)) as src: source_count=sum(m.isfile() for m in src.getmembers())
 prepared=js('results/quality/prepared.json'); comparison=js('results/quality/comparison.json')
 assert prepared['dataset_sha256']=='5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'
 assert prepared['revision']=='caa1feb0e54d415e2df31207e5f4e273e33509b1'
 assert prepared['dataset_revision']=='b08601e04326c79dfdd32d625aee71d232d685c3'
 assert prepared['blocks']==32 and prepared['length']==512 and prepared['scored_tokens']==16352
 weights={
 'model-00001-of-00003.safetensors':'2d7de8eb9bd554cccc999649fd5a3a28a44d09042879d6c1cedd13f59c86f7a8',
 'model-00002-of-00003.safetensors':'79a5f871a0fbb7919f263d78caff0d4d3b7a1d444552db3011ada70f48fd5ee4',
 'model-00003-of-00003.safetensors':'ba7ae59358f726f5f5bc5264fd8a3aec907f42e6009d7b22454007d18723616a'}
 assert prepared['weight_hashes']==weights
 starts=[i*(prepared['token_stream_length']-512)//31 for i in range(32)]
 assert prepared['window_starts']==starts and all(b-a>=512 for a,b in zip(starts,starts[1:]))
 inputs=read('results/quality/inputs.npy');assert sha(inputs)==prepared['inputs_sha256']
 assert np.load(io.BytesIO(inputs),allow_pickle=False).shape==(32,512)
 arrays=[];metas=[]
 for arm in ('bf16','nf4'):
  meta=js('results/quality/'+arm+'.json');metas.append(meta)
  assert not meta['training'] and not meta['candidate_kernel'] and not meta['ram_streaming']
  assert meta['inputs_sha256']==prepared['inputs_sha256']
  payload=read('results/quality/'+arm+'-tokens.npz');assert sha(payload)==meta['raw_sha256']
  with np.load(io.BytesIO(payload),allow_pickle=False) as npz: array={key:npz[key].copy() for key in npz.files}
  assert array['nll'].shape==(32,511) and np.isfinite(array['nll']).all()
  assert array['predictions'].shape==(32,511)
  mean=float(array['nll'].astype(np.float64).mean())
  assert mean==meta['nll'] and math.isclose(math.exp(mean),meta['perplexity'],rel_tol=1e-14)
  arrays.append(array)
 assert metas[0]['packages']==metas[1]['packages'] and metas[0]['gpu']==metas[1]['gpu']
 assert len(metas[0]['quantized_modules'])==0 and len(metas[1]['quantized_modules'])==224
 delta=metas[1]['nll']-metas[0]['nll']
 assert comparison['delta_nll']==delta
 assert math.isclose(comparison['relative_perplexity_change_percent'],100*math.expm1(delta),rel_tol=1e-14)
 agreement=100*float((arrays[0]['predictions']==arrays[1]['predictions']).mean())
 assert comparison['top1_agreement_percent']==agreement
 for idx,arm in enumerate(('bf16','nf4')):
  assert comparison[arm+'_nll']==metas[idx]['nll']
  assert comparison[arm+'_perplexity']==metas[idx]['perplexity']
 report={'status':'CPU_REPLAY_PASS','source_commit':source_sha,'source_files_verified':source_count,
 'raw_sha256':sha(data),'comparison':comparison,'gpu':read('results/gpu.txt').decode(),
 'packages':metas[0]['packages'],'network':js('results/network-guard.json')}
(root/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
