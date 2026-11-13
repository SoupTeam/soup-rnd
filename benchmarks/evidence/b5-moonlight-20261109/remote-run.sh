#!/bin/bash
# B5 ticket 03: environment record, pinned download, the run, post record. Run after remote-setup.sh.
# Usage: remote-run.sh ATTEMPT   (output goes to /root/b5/results/ATTEMPT)
set -uo pipefail
ATTEMPT=${1:?attempt name}
R=/root/b5/results/$ATTEMPT
mkdir -p "$R"
cd /root/b5/repo
export PYTHONPATH=/root/b5/repo:/root/b5/repo/src
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
PY=/root/b5/venv/bin/python
{
  date -u +%FT%TZ; uname -a; uptime; lscpu; free -b; df -B1 /root /dev/shm
  nvidia-smi -q; nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm --format=csv
  cat /proc/cpuinfo | grep -m1 'model name'; cat /sys/fs/cgroup/memory.max 2>/dev/null
} > "$R/environment.txt" 2>&1
cp /root/b5/pip-freeze.txt "$R/pip-freeze.txt"
df -B1 /root > "$R/df-0-start.txt"
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,clocks.sm,power.draw,temperature.gpu \
  --format=csv -l 2 > "$R/nvidia-smi-samples.csv" 2>&1 &
SMI=$!
echo download > "$R/phase.txt"
START=$(date +%s)
$PY - <<'PYEOF' 2>&1 | tee "$R/download.log"
import json, time
from pathlib import Path
from huggingface_hub import snapshot_download
t = time.time()
snapshot_download("moonshotai/Moonlight-16B-A3B", revision="476b36a473d4467f94469414bef6cee75c9c8172",
                  local_dir="/root/b5/source", max_workers=8, token=False,
                  allow_patterns=["*.safetensors", "*.json", "*.model", "*.py", "*.txt"])
files = list(Path("/root/b5/source").glob("*.safetensors"))
rec = {"seconds": time.time() - t, "files": len(files), "source_bytes": sum(p.stat().st_size for p in files)}
Path("/root/b5/results/download.json").write_text(json.dumps(rec))
print(rec)
PYEOF
cp /root/b5/results/download.json "$R/download.json" 2>/dev/null
df -B1 /root > "$R/df-1-after-download.txt"
echo run > "$R/phase.txt"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY benchmarks/harness/b5_real_moe.py \
  --config benchmarks/protocols/b5-moonlight-20261109/soup.yaml \
  --source /root/b5/source --cache /root/b5/cache --output "$R/run" \
  --data benchmarks/protocols/b5-moonlight-20261109/data.json \
  --model moonshotai/Moonlight-16B-A3B --revision 476b36a473d4467f94469414bef6cee75c9c8172 \
  --kind real --steps 4 --shard-threads 6 2>&1 | tee "$R/run.log"
CODE=${PIPESTATUS[0]}
echo "$CODE" > "$R/exit-code.txt"
df -B1 /root > "$R/df-2-after-run.txt"
du -sb /root/b5/source /root/b5/cache > "$R/du.txt" 2>&1
kill $SMI 2>/dev/null
echo done > "$R/phase.txt"
exit "$CODE"
