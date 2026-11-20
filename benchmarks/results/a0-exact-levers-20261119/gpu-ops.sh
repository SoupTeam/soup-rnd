#!/bin/bash
# GPU profile of one step with the idle-gap histogram, for B0 and the exact MBR.  usage: gpu-gaps.sh LABEL
R=/workspace/p3-baseline
LABEL=${1:-host}
cd $R
export PYTHONPATH=$R/repo/src:$R/repo/benchmarks/harness PATH=$R/venv/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
B=repo/benchmarks/harness/p3_route_preserving_mbr_soup_bridge.py
MBR="--profile-kinds mbr --mbr-group-size 8 --expert-transfer packed --expert-host-cache-mib 12288 --expert-prefetch"
OUT=$R/results/ops-$LABEL; rm -rf $OUT; mkdir -p $OUT
for run in b0 ltfo; do
  X=""; [ $run = ltfo ] && X="--expert-gemm lt-fused --mbr-ordered-backward"
  SOUP_GPU_TRACE_WINDOWS=1 SOUP_GPU_PROFILE_JSON=$OUT/$run python $B --root $R --output $OUT/o-$run --phase-profile-only --phase-profile-blocks 0 --timed-steps 2 $MBR $X > $OUT/$run.log 2>&1
  rm -rf $OUT/o-$run
done
echo done > $OUT/DONE
