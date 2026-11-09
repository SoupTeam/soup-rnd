#!/bin/bash
# B5 ticket 06: the three-step run on the store. Usage: remote-run.sh ATTEMPT [--cold|--warm]
# --cold (default) drops the page cache right before step 0; --warm runs again without a drop.
set -uo pipefail
ATTEMPT=${1:?attempt name}; MODE=${2:---cold}
R=/root/b5/results/$ATTEMPT
mkdir -p "$R"
cd /root/b5/repo
export PYTHONPATH=/root/b5/repo:/root/b5/repo/src
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1
PY=/root/b5/venv/bin/python
{ date -u +%FT%TZ; uptime; free -b; df -B1 /root; nvidia-smi --query-gpu=name,memory.total --format=csv; } > "$R/environment.txt" 2>&1
df -B1 /root > "$R/df-0-start.txt"
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,clocks.sm,power.draw,temperature.gpu \
  --format=csv -l 2 > "$R/nvidia-smi-samples.csv" 2>&1 &
SMI=$!
echo run > "$R/phase.txt"
EXTRA=(); [ "$MODE" = "--cold" ] && EXTRA=(--cold)
$PY benchmarks/harness/b5_real_moe.py \
  --config benchmarks/protocols/b5-synthetic671b-20261109/soup.yaml \
  --store /root/b5/synthetic-v3 --output "$R/run" --steps 3 --device cuda "${EXTRA[@]}" 2>&1 | tee "$R/run.log"
CODE=${PIPESTATUS[0]}
echo "$CODE" > "$R/exit-code.txt"
df -B1 /root > "$R/df-1-after-run.txt"
kill $SMI 2>/dev/null
WARM=(); [ "$MODE" = "--warm" ] && WARM=(--warm)
$PY benchmarks/harness/b5_cold_acceptance.py --journal "$R/run/journal.jsonl" \
  --store-record /root/b5/results/store/store-record.json "${WARM[@]}" > "$R/acceptance-verdict.json" 2>&1
echo $? > "$R/acceptance-exit-code.txt"
echo done > "$R/phase.txt"
exit "$CODE"
