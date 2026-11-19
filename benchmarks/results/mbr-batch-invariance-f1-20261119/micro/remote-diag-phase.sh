#!/bin/bash
# Diagnosis 1: instrumented phase breakdown of one B0 and one B1 step (no torch.profiler).
R=/workspace/p3-baseline
cd $R
export PYTHONPATH=$R/repo/src:$R/repo/benchmarks/harness
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
COMPLETE="--expert-transfer packed --expert-host-cache-mib 12288"
COMMON="--phase-profile-only --phase-profile-blocks 1 --timed-steps 1"
mkdir -p $R/diag2
B=$R/repo/benchmarks/harness/p3_route_preserving_mbr_soup_bridge.py
$R/venv/bin/python $B --root $R --output $R/diag2/B0 \
  --profile-kinds mbr --mbr-group-size 8 $COMPLETE --expert-prefetch $COMMON > $R/diag2/B0.log 2>&1
echo "B0 rc=$?" >> $R/diag2/status.txt
$R/venv/bin/python $B --root $R --output $R/diag2/B1 \
  --profile-kinds mbr --mbr-group-size 8 $COMPLETE --expert-prefetch --expert-gemm invariant $COMMON > $R/diag2/B1.log 2>&1
echo "B1 rc=$?" >> $R/diag2/status.txt
echo diag2_done > $R/diag2/done
