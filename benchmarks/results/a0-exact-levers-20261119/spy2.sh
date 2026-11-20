#!/bin/bash
# py-spy (active samples, per thread, 200 Hz) of the exact MBR with fast gather and async lease.   usage: spy2.sh LABEL
R=/workspace/p3-baseline
LABEL=$1
cd $R
export PYTHONPATH=$R/repo/src:$R/repo/benchmarks/harness PATH=$R/venv/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
B=repo/benchmarks/harness/p3_route_preserving_mbr_soup_bridge.py
MBR="--profile-kinds mbr --mbr-group-size 8 --expert-transfer packed --expert-host-cache-mib 12288 --expert-prefetch --expert-gemm lt-fused --mbr-ordered-backward --expert-fast-gather --expert-async-lease"
OUT=$R/results/spy2-$LABEL; rm -rf $OUT; mkdir -p $OUT
py-spy record -r 200 -f raw --threads -o $OUT/spy.raw -- python $B --root $R --output $OUT/o --phase-profile-only --phase-profile-blocks 0 --timed-steps 4 $MBR > $OUT/spy.log 2>&1
cp $OUT/o/result_phase_profile_only.json $OUT/result.json; rm -rf $OUT/o
echo done > $OUT/DONE
