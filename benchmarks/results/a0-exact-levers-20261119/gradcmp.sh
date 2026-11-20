#!/bin/bash
# First-step gradients of arms A and C on the training data (one step each).   usage: gradcmp.sh "C A"
R=/workspace/p3-baseline
cd $R
export PYTHONPATH=$R/repo/src:$R/repo/benchmarks/harness PATH=$R/venv/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
B=repo/benchmarks/harness/p3_route_preserving_mbr_soup_bridge.py
CACHE="--expert-transfer packed --expert-host-cache-mib 12288"
declare -A FLAGS
FLAGS[A]="--profile-kinds l2l $CACHE"
FLAGS[B]="--profile-kinds mbr --mbr-group-size 8 $CACHE --expert-prefetch"
FLAGS[C]="--profile-kinds mbr --mbr-group-size 8 $CACHE --expert-prefetch --expert-gemm lt-fused --mbr-ordered-backward --expert-fast-gather --expert-async-lease"
FLAGS[C2]=${FLAGS[C]}
FLAGS[A2]=${FLAGS[A]}
FLAGS[D]="--profile-kinds mbr --mbr-group-size 8 $CACHE --expert-prefetch --expert-gemm lt-fused --mbr-ordered-backward"
mkdir -p $R/results/gradcmp
for arm in $1; do
  rm -rf $R/results/gradcmp-$arm
  python $B --root $R --output $R/results/gradcmp-$arm --phase-profile-only --phase-profile-blocks 0 \
    --train-data $R/qual/train_data.pt --deterministic --train-steps 1 --train-dump-grads $R/results/gradcmp/$arm.pt ${FLAGS[$arm]} > $R/results/gradcmp-$arm.log 2>&1
  echo "arm $arm exit $?" >> $R/results/gradcmp.log
done
echo done > $R/results/gradcmp-DONE
