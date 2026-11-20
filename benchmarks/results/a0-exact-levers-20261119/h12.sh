#!/bin/bash
# H1/H2 checks on one host: (a) in-process A/B of --expert-fast-gather, (b) per-thread CPU profile of the
# exact MBR, (c) GIL occupancy by py-spy --gil.   usage: h12.sh LABEL
R=/workspace/p3-baseline
LABEL=${1:-host}
cd $R
export PYTHONPATH=$R/repo/src:$R/repo/benchmarks/harness PATH=$R/venv/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
B=repo/benchmarks/harness/p3_route_preserving_mbr_soup_bridge.py
MBR="--profile-kinds mbr --mbr-group-size 8 --expert-transfer packed --expert-host-cache-mib 12288 --expert-prefetch --expert-gemm lt-fused --mbr-ordered-backward"
OUT=$R/results/h12-$LABEL; rm -rf $OUT; mkdir -p $OUT
python $B --root $R --output $OUT/o-alt --phase-profile-only --phase-profile-blocks 0 --timed-steps 16 $MBR --alternate-fast-gather > $OUT/alt.log 2>&1
cp $OUT/o-alt/result_phase_profile_only.json $OUT/alt.json; rm -rf $OUT/o-alt
SOUP_GPU_TRACE_WINDOWS=1 SOUP_GPU_PROFILE_JSON=$OUT/threads python $B --root $R --output $OUT/o-thr --phase-profile-only --phase-profile-blocks 0 --timed-steps 2 $MBR > $OUT/threads.log 2>&1
cp $OUT/o-thr/result_phase_profile_only.json $OUT/threads_result.json; rm -rf $OUT/o-thr
which py-spy >/dev/null 2>&1 || pip install -q py-spy >/dev/null 2>&1
py-spy record --gil -r 200 -f raw --threads -o $OUT/gil.raw -- python $B --root $R --output $OUT/o-gil --phase-profile-only --phase-profile-blocks 0 --timed-steps 3 $MBR > $OUT/gil.log 2>&1
cp $OUT/o-gil/result_phase_profile_only.json $OUT/gil_result.json; rm -rf $OUT/o-gil
echo done > $OUT/DONE
