#!/bin/bash
# Diagnosis 4: segment-count tuples of one real segmented step (B0s configuration).
R=/workspace/p3-baseline
cd $R
export PYTHONPATH=$R/repo/src:$R/repo/benchmarks/harness
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
rm -f $R/plans-stats.done $R/plans_stats.json
$R/venv/bin/python $R/plans_stats.py $R/repo/benchmarks/harness/p3_route_preserving_mbr_soup_bridge.py \
  $R/plans_stats.json --root $R --output $R/diag4 --profile-kinds mbr --mbr-group-size 8 \
  --expert-transfer packed --expert-host-cache-mib 12288 --expert-prefetch --expert-gemm segmented \
  --phase-profile-only --phase-profile-blocks 0 --timed-steps 1 > $R/plans-stats.log 2>&1
echo "rc=$?" > $R/plans-stats.done
