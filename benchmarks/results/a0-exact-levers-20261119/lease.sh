#!/bin/bash
# async_lease: GPU test, then in-process alternating A/B (off/on) of --expert-async-lease with --expert-fast-gather.
R=/workspace/p3-baseline
LABEL=$1
cd $R/repo
export PYTHONPATH=$R/repo/src:$R/repo/benchmarks/harness PATH=$R/venv/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
OUT=$R/results/lease-$LABEL; rm -rf $OUT; mkdir -p $OUT
python -m pytest tests/test_expert_async_lease_cuda.py -q -c /dev/null --rootdir=. -p no:cacheprovider -m gpu > $OUT/pytest.log 2>&1
cd $R
B=repo/benchmarks/harness/p3_route_preserving_mbr_soup_bridge.py
MBR="--profile-kinds mbr --mbr-group-size 8 --expert-transfer packed --expert-host-cache-mib 12288 --expert-prefetch --expert-gemm lt-fused --mbr-ordered-backward --expert-fast-gather"
python $B --root $R --output $OUT/o-alt --phase-profile-only --phase-profile-blocks 0 --timed-steps 16 $MBR --alternate-async-lease > $OUT/alt.log 2>&1
cp $OUT/o-alt/result_phase_profile_only.json $OUT/alt.json; rm -rf $OUT/o-alt
echo done > $OUT/DONE
