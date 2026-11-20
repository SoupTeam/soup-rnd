#!/bin/bash
# F1 for the segmented MBR step (ticket 07): A0 against B0s, then B0 against B0s.
# Driver: benchmarks/harness/p72_f1_blocks.py (--arms a0-b0s, --arms b0-b0s); rule:
# benchmarks/mbr-invariance-f1-rule-20261119.md. Per-run records go to f1_records.jsonl as they end.
R=/workspace/p3-baseline
cd $R
export PYTHONPATH=$R/repo/src:$R/repo/benchmarks/harness
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
for ARMS in a0-b0s b0-b0s; do
  echo "$(date +%s) start f1 $ARMS" >> $R/series.log
  echo "f1_running_$ARMS" > $R/phase.txt
  $R/venv/bin/python $R/repo/benchmarks/harness/p72_f1_blocks.py --root $R \
    --output $R/results/f1-$ARMS --arms $ARMS --host-cache-mib 12288 --timed-steps 3 \
    --max-blocks 10 --label "RTX 3060 (vast.ai), Moonlight, $ARMS" > $R/results/f1-$ARMS.log 2>&1
  echo "$(date +%s) f1 $ARMS exit $?" >> $R/series.log
done
echo f1_b0s_done > $R/phase.txt
