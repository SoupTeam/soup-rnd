#!/bin/bash
# Diagnosis 2: segmented native expert GEMM against A0 (series style of ticket 06). One bridge call per run, one output dir per run.
# Usage: remote-series.sh RUN [RUN...]   RUN = name from the table below. Leaves the host on failure.
R=/workspace/p3-baseline
cd $R
mkdir -p $R/results/seg
export PYTHONPATH=$R/repo/src:$R/repo/benchmarks/harness
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
COMPLETE="--expert-transfer packed --expert-host-cache-mib 12288"
COMMON="--phase-profile-only --phase-profile-blocks 0 --timed-steps 2"
args_for() {
  case "$1" in
    O0*)  echo "--profile-kinds l2l --expert-transfer sync" ;;
    Ap*)  echo "--profile-kinds l2l $COMPLETE --expert-gemm padded" ;;
    Bp*)  echo "--profile-kinds mbr --mbr-group-size 8 $COMPLETE --expert-prefetch --expert-gemm padded" ;;
    A0*)  echo "--profile-kinds l2l $COMPLETE" ;;
    A1*)  echo "--profile-kinds l2l $COMPLETE --expert-gemm invariant" ;;
    B0s*) echo "--profile-kinds mbr --mbr-group-size 8 $COMPLETE --expert-prefetch --expert-gemm segmented" ;;
    B0*)  echo "--profile-kinds mbr --mbr-group-size 8 $COMPLETE --expert-prefetch" ;;
    B1g1*) echo "--profile-kinds mbr --mbr-group-size 1 $COMPLETE --expert-prefetch --expert-gemm invariant" ;;
    B1g2*) echo "--profile-kinds mbr --mbr-group-size 2 $COMPLETE --expert-prefetch --expert-gemm invariant" ;;
    B1g4*) echo "--profile-kinds mbr --mbr-group-size 4 $COMPLETE --expert-prefetch --expert-gemm invariant" ;;
    B1s*) echo "--profile-kinds mbr --mbr-group-size 8 --expert-transfer sync --expert-gemm invariant" ;;
    B1*)  echo "--profile-kinds mbr --mbr-group-size 8 $COMPLETE --expert-prefetch --expert-gemm invariant" ;;
    *) return 1 ;;
  esac
}
for run in "$@"; do
  echo "run_$run" > $R/phase.txt
  A=$(args_for "$run") || { echo "unknown run $run" >> $R/series.log; echo SERIES_FAILED > $R/phase.txt; exit 1; }
  echo "$(date +%s) start $run: $A" >> $R/series.log
  $R/venv/bin/python repo/benchmarks/harness/p3_route_preserving_mbr_soup_bridge.py --root $R \
    --output $R/results/seg/$run --dump-grads $R/results/seg/$run.grads.pt $COMMON $A > $R/results/seg-$run.log 2>&1 \
    || { echo "$(date +%s) FAILED $run" >> $R/series.log; echo SERIES_FAILED > $R/phase.txt; exit 1; }
  echo "$(date +%s) done $run" >> $R/series.log
done
echo series_done > $R/phase.txt
