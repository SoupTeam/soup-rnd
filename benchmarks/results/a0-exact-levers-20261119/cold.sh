#!/bin/bash
# Step-by-step curves of fresh processes: exact MBR without / with --expert-fast-gather, ordered A B B A.
R=/workspace/p3-baseline
cd $R
export PYTHONPATH=$R/repo/src:$R/repo/benchmarks/harness PATH=$R/venv/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
B=repo/benchmarks/harness/p3_route_preserving_mbr_soup_bridge.py
MBR="--profile-kinds mbr --mbr-group-size 8 --expert-transfer packed --expert-host-cache-mib 12288 --expert-prefetch --expert-gemm lt-fused --mbr-ordered-backward"
OUT=$R/results/cold; rm -rf $OUT; mkdir -p $OUT
for arm in off1:"" on1:"--expert-fast-gather" on2:"--expert-fast-gather" off2:""; do
  name=${arm%%:*}; extra=${arm#*:}
  python $B --root $R --output $OUT/o-$name --phase-profile-only --phase-profile-blocks 0 --timed-steps 10 $MBR $extra > $OUT/$name.log 2>&1
  cp $OUT/o-$name/result_phase_profile_only.json $OUT/$name.json
  cp $OUT/o-$name/progress.json $OUT/$name.progress.json
  rm -rf $OUT/o-$name
done
echo done > $OUT/DONE
