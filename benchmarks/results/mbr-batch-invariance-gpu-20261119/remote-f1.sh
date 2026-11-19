#!/bin/bash
# MBR batch-invariance F1 (ticket 07): B0 (native GEMM) against B1 (invariant GEMM), both complete +
# MBR group 8 + prefetch. Driver: benchmarks/harness/p72_f1_blocks.py --arms b0-b1 (rule: benchmarks/mbr-invariance-f1-rule-20261119.md).
R=/workspace/p3-baseline
cd $R
export PYTHONPATH=$R/repo/src:$R/repo/benchmarks/harness
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
echo "$(date +%s) start f1" >> $R/series.log
echo f1_running > $R/phase.txt
$R/venv/bin/python $R/repo/benchmarks/harness/p72_f1_blocks.py --root $R --output $R/results/f1-b0-b1 \
  --arms b0-b1 --host-cache-mib 12288 --timed-steps 3 --max-blocks 10 \
  --label "RTX 3060 (vast.ai), Moonlight MBR, B0 vs B1" > $R/results/f1-b0-b1.log 2>&1
rc=$?
echo "$(date +%s) f1 exit $rc" >> $R/series.log
echo "f1_done rc=$rc" > $R/phase.txt
