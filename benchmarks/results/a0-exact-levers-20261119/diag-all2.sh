#!/bin/bash
# The same CPU + GPU diagnosis on every host (run it identically on both, compare the folders).
#   1. host_probe.json: CPU cost per operation, copy bandwidths, disk read, GPU settings
#   2. for B0 and the exact MBR (lt-fused + ordered backward), each in its own process:
#        wall      : 3 unprofiled timed steps (result_phase_profile_only.json)
#        gpu       : torch.profiler CUDA trace of one step, reduced to gpu.N.json
#        dmon      : nvidia-smi dmon once per second during the unprofiled steps
#        pyspy     : py-spy all-thread profile (raw stacks)
#        phases    : synchronised phase profile
# Output: $R/results/diag/<run>/..., packed into /root/diag-<label>.tgz
R=/workspace/p3-baseline
LABEL=${1:-host}
cd $R
export PYTHONPATH=$R/repo/src:$R/repo/benchmarks/harness PATH=$R/venv/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
OUT=$R/results/diag2
rm -rf $OUT && mkdir -p $OUT
PY=$R/venv/bin/python
B=repo/benchmarks/harness/p3_route_preserving_mbr_soup_bridge.py
MBR="--profile-kinds mbr --mbr-group-size 8 --expert-transfer packed --expert-host-cache-mib 12288 --expert-prefetch"
which py-spy >/dev/null 2>&1 || $PY -m pip install -q py-spy >/dev/null 2>&1
DISK_FILE=$(ls -S $R/cache/*/* 2>/dev/null | head -1)
$PY repo/benchmarks/harness/host_probe.py $OUT/host_probe.json "$DISK_FILE" > $OUT/host_probe.log 2>&1
for run in b0 ltfo; do
  X=""; [ $run = ltfo ] && X="--expert-gemm lt-fused --mbr-ordered-backward"
  D=$OUT/$run; mkdir -p $D
  echo "diag $run wall"
  nvidia-smi dmon -s pucm -d 1 > $D/dmon.txt 2>&1 & DM=$!
  $PY $B --root $R --output $D/wall --phase-profile-only --phase-profile-blocks 0 --timed-steps 3 $MBR $X > $D/wall.log 2>&1
  kill $DM 2>/dev/null
  echo "diag $run gpu"
  SOUP_GPU_PROFILE_JSON=$D/gpu $PY $B --root $R --output $D/gpuprof --phase-profile-only --phase-profile-blocks 0 --timed-steps 1 $MBR $X > $D/gpu.log 2>&1
  echo "diag $run pyspy"
  py-spy record -r 100 -f raw --threads -o $D/pyspy.raw -- $PY $B --root $R --output $D/spy --phase-profile-only --phase-profile-blocks 0 --timed-steps 3 $MBR $X > $D/pyspy.log 2>&1
  echo "diag $run phases"
  $PY $B --root $R --output $D/phases --phase-profile-only --phase-profile-blocks 1 --timed-steps 1 $MBR $X > $D/phases.log 2>&1
  rm -rf $D/spy $D/gpuprof
done
cd $R/results && tar czf /root/diag2-$LABEL.tgz diag2
echo diag_done
