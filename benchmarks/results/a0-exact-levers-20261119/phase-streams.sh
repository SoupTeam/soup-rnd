#!/bin/bash
# In-process effect of the streams: synchronised phase profile of the exact MBR step with 1 and 3
# streams, alternated (1 3 3 1 1 3 ...) so drift averages out. Each run is one process, one
# profiled step plus the unprofiled timed step the bridge records. Compare with phase_streams.py.
R=/workspace/p3-baseline
cd $R
export PYTHONPATH=$R/repo/src:$R/repo/benchmarks/harness PATH=$R/venv/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
OUT=$R/results/phstreams
rm -rf $OUT && mkdir -p $OUT
EXACT="--profile-kinds mbr --mbr-group-size 8 --expert-transfer packed --expert-host-cache-mib 12288 --expert-prefetch --expert-gemm lt-fused --mbr-ordered-backward"
i=0
for n in ${ORDER:-1 3 3 1 1 3 3 1}; do
  i=$((i+1))
  $R/venv/bin/python repo/benchmarks/harness/p3_route_preserving_mbr_soup_bridge.py --root $R \
    --output $OUT/run${i}_s$n --phase-profile-only --phase-profile-blocks 1 --timed-steps 2 \
    $EXACT --expert-streams $n > $OUT/run${i}_s$n.log 2>&1
done
echo phstreams_done
