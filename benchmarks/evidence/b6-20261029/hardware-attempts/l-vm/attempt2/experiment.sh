#!/bin/bash
set -euo pipefail
export PYTHONPATH=/root/b6/repo-revised:/root/b6/repo-revised/src
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCHDYNAMO_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
cd /root/b6/repo-revised
echo download > /root/b6/phase2.txt
/root/b6/venv/bin/python benchmarks/harness/b6_source_plan.py \
 --revision e815299b0bcbac849fa540c768ef21845365c9eb \
 --output /root/b6/results/attempt2/source-plan | tee /root/b6/results/attempt2/source-plan.log
/root/b6/venv/bin/python - <<'DOWNLOAD' | tee /root/b6/results/attempt2/download.log
from huggingface_hub import snapshot_download
from pathlib import Path
from rich.console import Console
import shutil,json,time
root=Path('/root/b6/results/attempt2')
plan=json.loads((root/'source-plan/plan.json').read_text())
required=plan['source_bytes']+plan['cache_budget_bytes']+4*2**30
free=shutil.disk_usage('/root/b6').free
receipt={'free_before_source_bytes':free,'source_bytes':plan['source_bytes'],'cache_budget_bytes':plan['cache_budget_bytes'],'additional_reserve_bytes':4*2**30,'required_bytes':required}
(root/'disk-plan.json').write_text(json.dumps(receipt,indent=2));Console().print(receipt)
assert free>=required,(free,required)
start=time.time()
(root/'download-started.json').write_text(json.dumps({'unix':start}))
snapshot_download('deepseek-ai/DeepSeek-V3',revision='e815299b0bcbac849fa540c768ef21845365c9eb',local_dir='/root/b6/source',allow_patterns=['*.safetensors','*.json','tokenizer.model','*.tiktoken','*.txt'],max_workers=8,token=False)
files=list(Path('/root/b6/source').glob('*.safetensors'))
assert len(files)==163 and sum(p.stat().st_size for p in files)==688586727753
receipt={'start_unix':start,'end_unix':time.time(),'seconds':time.time()-start,'source_bytes':sum(p.stat().st_size for p in files),'files':len(files)}
(root/'download.json').write_text(json.dumps(receipt,indent=2));Console().print(receipt)
DOWNLOAD
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/root/b6/venv/bin/python benchmarks/harness/b6_real_moe.py \
 --source /root/b6/source --cache /root/b6/cache --output /root/b6/results/attempt2/real \
 --data examples/b6-pilot-data.json --revision e815299b0bcbac849fa540c768ef21845365c9eb \
 --steps 2 --max-length 64 --shard-threads 6 \
 --evaluation-protocol benchmarks/protocols/b6-acceptance-20261029/protocol.json \
 --evaluation-data benchmarks/protocols/b6-acceptance-20261029/evaluation-data.json \
 --ram-phase-file /root/b6/phase2.txt \
 --source-plan /root/b6/results/attempt2/source-plan/plan.json | tee /root/b6/results/attempt2/real.log
cp /root/b6/cache/index.json /root/b6/results/attempt2/cache-index.json
