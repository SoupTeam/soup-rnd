#!/bin/bash
set -euo pipefail
cd /root/b6/repo
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
echo download > /root/b6/phase.txt
touch /root/b6/download-started
/root/b6/venv/bin/python - <<'DOWNLOAD' | tee /root/b6/results/download.log
from huggingface_hub import snapshot_download
from pathlib import Path
import shutil,json,time
required=688586727753+519579082752+4*2**30
free=shutil.disk_usage('/root/b6').free
assert free>=required,(free,required)
start=time.time()
snapshot_download('deepseek-ai/DeepSeek-V3',revision='e815299b0bcbac849fa540c768ef21845365c9eb',local_dir='/root/b6/source',allow_patterns=['*.safetensors','*.json','tokenizer.model','*.tiktoken','*.txt'],max_workers=8,token=False)
files=list(Path('/root/b6/source').glob('*.safetensors'))
assert len(files)==163 and sum(p.stat().st_size for p in files)==688586727753
print(json.dumps({'seconds':time.time()-start,'source_bytes':sum(p.stat().st_size for p in files),'files':len(files)}))
DOWNLOAD
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/root/b6/venv/bin/python benchmarks/harness/b6_real_moe.py \
 --source /root/b6/source --cache /root/b6/cache --output /root/b6/results/real \
 --data examples/b6-pilot-data.json --revision e815299b0bcbac849fa540c768ef21845365c9eb \
 --steps 2 --max-length 64 --shard-threads 8 \
 --evaluation-protocol benchmarks/protocols/b6-acceptance-20261029/protocol.json \
 --evaluation-data benchmarks/protocols/b6-acceptance-20261029/evaluation-data.json \
 --ram-phase-file /root/b6/phase.txt | tee /root/b6/results/real.log
cp /root/b6/cache/index.json /root/b6/results/cache-index.json
