#!/bin/bash
# 40-step training of each arm on the same data, adapter digest per step and held-out NLL.   usage: train-run.sh "C A B"
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
for arm in $1; do
  rm -rf $R/results/train-$arm
  python $B --root $R --output $R/results/train-$arm --phase-profile-only --phase-profile-blocks 0 \
    --deterministic --train-data $R/qual/train_data.pt --train-eval ${FLAGS[$arm]} > $R/results/train-$arm.log 2>&1
  echo "arm $arm exit $?" >> $R/results/train.log
done
echo done > $R/results/train-DONE
