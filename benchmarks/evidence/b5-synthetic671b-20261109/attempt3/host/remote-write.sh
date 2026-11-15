#!/bin/bash
# B5 ticket 06: environment record and the write of the synthetic DeepSeek-V3-shaped store.
# Run after remote-setup.sh. Output goes to /root/b5/results/ATTEMPT (default: store).
set -uo pipefail
R=/root/b5/results/${1:-store}
mkdir -p "$R"
cd /root/b5/repo
export PYTHONPATH=/root/b5/repo:/root/b5/repo/src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TORCHDYNAMO_DISABLE=1
PY=/root/b5/venv/bin/python
{
  date -u +%FT%TZ; uname -a; uptime; lscpu; free -b; df -B1 /root
  nvidia-smi -q; nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm --format=csv
  cat /sys/fs/cgroup/memory.max 2>/dev/null; cat /sys/fs/cgroup/memory.swap.max 2>/dev/null
  ls -l /proc/sys/vm/drop_caches; mount | grep -E " / "
} > "$R/environment.txt" 2>&1
cp /root/b5/pip-freeze.txt "$R/pip-freeze.txt"
df -B1 /root > "$R/df-0-before-write.txt"
echo write > "$R/phase.txt"
$PY benchmarks/harness/b5_synth_store.py write --shape deepseek-v3 --dtype bfloat16 \
  --seed 20261109 --out /root/b5/synthetic-v3 --record "$R/store-record.json" 2>&1 | tee "$R/write.log"
CODE=${PIPESTATUS[0]}
echo "$CODE" > "$R/write-exit-code.txt"
df -B1 /root > "$R/df-1-after-write.txt"
du -sb /root/b5/synthetic-v3 > "$R/du-store.txt" 2>&1
echo done > "$R/phase.txt"
exit "$CODE"
