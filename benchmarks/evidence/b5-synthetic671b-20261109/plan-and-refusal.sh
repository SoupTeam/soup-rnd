#!/bin/sh
# B5 report 3: size plan of the DeepSeek-V3-shaped store and the refusal at 200 GB free.
# Run from the repo root: PYTHONPATH=$PWD/src sh plan-and-refusal.sh PYTHON OUT_DIR EVIDENCE_DIR
PY=$1; OUT=$2; E=$3
$PY benchmarks/harness/b5_synth_store.py plan --shape deepseek-v3 --dtype bfloat16 \
    --seed 20261109 --out "$OUT" > "$E/plan.out" 2>&1
echo "plan exit code $?" > "$E/plan-exit-code.txt"
$PY benchmarks/harness/b5_synth_store.py write --shape deepseek-v3 --dtype bfloat16 \
    --seed 20261109 --out "$OUT" --record "$E/refusal-200gb.json" \
    --assume-free-bytes 200000000000 > "$E/refusal-200gb.out" 2>&1
echo "write exit code $?" > "$E/refusal-200gb-exit-code.txt"
if [ -e "$OUT" ]; then echo "destination exists after the refusal" > "$E/refusal-200gb-destination.txt"
else echo "destination $OUT does not exist after the refusal" > "$E/refusal-200gb-destination.txt"; fi
