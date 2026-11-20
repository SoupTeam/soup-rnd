#!/bin/bash
# Causal profiling on one host: one arm per point, delay on for odd steps only, all in one process.
# usage: causal.sh LABEL "arm=spec arm=spec ..."   (STEPS env, default 14)
R=/workspace/p3-baseline
LABEL=$1; ARMS=$2
cd $R
export PYTHONPATH=$R/repo/src:$R/repo/benchmarks/harness PATH=$R/venv/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
B=repo/benchmarks/harness/p3_route_preserving_mbr_soup_bridge.py
MBR="--profile-kinds mbr --mbr-group-size 8 --expert-transfer packed --expert-host-cache-mib 12288 --expert-prefetch --expert-gemm lt-fused --mbr-ordered-backward --expert-fast-gather"
OUT=$R/results/causal-$LABEL; mkdir -p $OUT
for arm in $ARMS; do
  name=${arm%%=*}; spec=${arm#*=}
  python $B --root $R --output $OUT/o-$name --phase-profile-only --phase-profile-blocks 0 --timed-steps ${STEPS:-14} $MBR --causal-delay "$spec" --alternate-causal > $OUT/$name.log 2>&1
  cp $OUT/o-$name/result_phase_profile_only.json $OUT/$name.json; rm -rf $OUT/o-$name
done
echo done > $OUT/DONE
