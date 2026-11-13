#!/bin/bash
# B5 ticket 06: finish the store write after the instance was stopped half way (94 of 122
# files were complete). Same shape, seed and dtype as remote-write.sh; complete files are kept.
set -uo pipefail
R=/root/b5/results/store
cd /root/b5/repo
export PYTHONPATH=/root/b5/repo:/root/b5/repo/src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TORCHDYNAMO_DISABLE=1
PY=/root/b5/venv/bin/python
df -B1 /root > "$R/df-2-before-resume.txt"
ls -l /root/b5/synthetic-v3 > "$R/ls-before-resume.txt"
echo resume > "$R/phase.txt"
$PY benchmarks/harness/b5_synth_store.py write --shape deepseek-v3 --dtype bfloat16 \
  --seed 20261109 --out /root/b5/synthetic-v3 --record "$R/store-record.json" --resume \
  2>&1 | tee "$R/resume.log"
CODE=${PIPESTATUS[0]}
echo "$CODE" > "$R/resume-exit-code.txt"
df -B1 /root > "$R/df-3-after-write.txt"
du -sb /root/b5/synthetic-v3 > "$R/du-store.txt" 2>&1
echo done > "$R/phase.txt"
exit "$CODE"
