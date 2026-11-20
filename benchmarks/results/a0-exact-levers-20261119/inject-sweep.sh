#!/bin/bash
# Causal-profiling sweep: how much does wall time move when GPU time or CPU time is added before every
# fused expert call?  usage: inject-sweep.sh LABEL "ARM ARM ..."   ARM = name:gpu_us:cpu_us
R=/workspace/p3-baseline
LABEL=$1; ARMS=${2:-"base:0:0"}
cd $R
export PYTHONPATH=$R/repo/src:$R/repo/benchmarks/harness PATH=$R/venv/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
B=repo/benchmarks/harness/p3_route_preserving_mbr_soup_bridge.py
MBR="--profile-kinds mbr --mbr-group-size 8 --expert-transfer packed --expert-host-cache-mib 12288 --expert-prefetch --expert-gemm lt-fused --mbr-ordered-backward $EXTRA"
OUT=$R/results/inject-$LABEL; mkdir -p $OUT
for arm in $ARMS; do
  IFS=: read name g c <<< "$arm"
  echo "arm $name gpu=$g cpu=$c"
  SOUP_INJECT_GPU_US=$g SOUP_INJECT_CPU_US=$c SOUP_INJECT_COUNT_FILE=$OUT/$name.count \
    python $B --root $R --output $OUT/$name --phase-profile-only --phase-profile-blocks 0 --timed-steps ${STEPS:-3} $MBR > $OUT/$name.log 2>&1
  cp $OUT/$name/result_phase_profile_only.json $OUT/$name.json 2>/dev/null
  rm -rf $OUT/$name
done
echo done > $OUT/DONE
