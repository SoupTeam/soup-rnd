#!/bin/bash
# Diagnosis 3: instrumented phase breakdown of one B0s step (segmented native expert GEMM).
R=/workspace/p3-baseline
cd $R
export PYTHONPATH=$R/repo/src:$R/repo/benchmarks/harness
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
COMPLETE="--expert-transfer packed --expert-host-cache-mib 12288"
COMMON="--phase-profile-only --phase-profile-blocks 1 --timed-steps 1"
mkdir -p $R/diag3
B=$R/repo/benchmarks/harness/p3_route_preserving_mbr_soup_bridge.py
$R/venv/bin/python $B --root $R --output $R/diag3/B0s \
  --profile-kinds mbr --mbr-group-size 8 $COMPLETE --expert-prefetch --expert-gemm segmented \
  $COMMON > $R/diag3/B0s.log 2>&1
echo "B0s rc=$?" >> $R/diag3/status.txt
echo diag3_done > $R/diag3/done
