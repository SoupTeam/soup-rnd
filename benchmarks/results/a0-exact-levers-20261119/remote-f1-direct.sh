#!/bin/bash
# A0-exact levers on Moonlight (RTX 3060): environment, cache, digest series, then F1.
# Stages (each writes $R/phase.txt; failures leave the host up):
#   setup  : venv (torch 2.14.1+cu130 and the versions of the earlier Moonlight runs), repo install
#   cache  : p72_prepare_moonlight_cache.py
#   series : one bridge call per arm with --dump-grads; grad digests must match A0 for the exact arms
#   f1     : p72_f1_blocks.py --arms b0-b0sd, then --arms b0-ltd
# Usage: [RUNS="..."] [F1_ARMS="..."] remote-f1-direct.sh [setup] [cache] [micro] [series] [f1]
#   micro  : benchmarks/results/micro_lt_ext.py (copied there by the upload) on one expert
R=/workspace/p3-baseline
mkdir -p $R/results/series
cd $R
export PYTHONPATH=$R/repo/src:$R/repo/benchmarks/harness
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 USE_TF=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
PY=$R/venv/bin/python
export PATH=$R/venv/bin:$PATH  # ninja for the lt-merged extension build
log() { echo "$(date +%s) $*" >> $R/series.log; }
fail() { log "FAILED $*"; echo "FAILED_$1" > $R/phase.txt; exit 1; }

stage_setup() {
  echo setup > $R/phase.txt
  mkdir -p $R/repo && tar xzf /root/repo.tgz -C $R/repo || fail setup
  python3 -m venv --help >/dev/null 2>&1 && python3 -c "import ensurepip" 2>/dev/null || \
    timeout 300 bash -c "apt-get update -qq && apt-get install -y -qq python3.12-venv python3.12-dev" || fail setup
  python3 -m venv $R/venv || fail setup
  $PY -m pip install -q --upgrade pip uv || fail setup
  $R/venv/bin/uv pip install --python $PY -q torch==2.14.1 \
    --index-url https://download.pytorch.org/whl/cu130 || fail setup
  $R/venv/bin/uv pip install --python $PY -q transformers==5.19.0 peft==0.21.2 \
    accelerate==1.15.0 bitsandbytes==0.50.2 safetensors==0.8.0 tiktoken==0.14.0 \
    blobfile==3.1.0 huggingface-hub==2.4.0 numpy==2.1.3 psutil==7.2.2 pyyaml==6.0.3 \
    rich==13.9.4 pydantic==2.14.1 ninja "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-cccl==13.0.*" \
    || fail setup  # headers for the lt-merged extension on images without a toolkit
  $R/venv/bin/uv pip install --python $PY -q --no-deps -e $R/repo || fail setup
  $PY -c "import torch, transformers; print(torch.__version__, transformers.__version__)" \
    >> $R/series.log 2>&1 || fail setup
  log setup done
}

stage_cache() {
  echo cache > $R/phase.txt
  $PY repo/benchmarks/harness/p72_prepare_moonlight_cache.py --root $R \
    > $R/results/cache.log 2>&1 || fail cache
  log cache done
}

COMPLETE="--expert-transfer packed --expert-host-cache-mib 12288"
COMMON="--phase-profile-only --phase-profile-blocks 0 --timed-steps 2"
MBR="--profile-kinds mbr --mbr-group-size 8 $COMPLETE --expert-prefetch"
args_for() {
  case "$1" in
    A0)    echo "--profile-kinds l2l $COMPLETE" ;;
    B0)    echo "$MBR" ;;
    B0s)   echo "$MBR --expert-gemm segmented" ;;
    B0sd)  echo "$MBR --expert-gemm segmented --expert-direct-backward" ;;
    Blt)   echo "$MBR --expert-gemm lt-merged" ;;
    Bltd)  echo "$MBR --expert-gemm lt-merged --expert-direct-backward" ;;
    *) return 1 ;;
  esac
}

stage_series() {
  for run in ${RUNS:-A0 B0s B0sd Bltd Blt B0}; do
    echo "series_$run" > $R/phase.txt
    A=$(args_for $run)
    log "start $run: $A"
    $PY repo/benchmarks/harness/p3_route_preserving_mbr_soup_bridge.py --root $R \
      --output $R/results/series/$run --dump-grads $R/results/series/$run.grads.pt \
      $COMMON $A > $R/results/series-$run.log 2>&1 || fail series_$run
    log "done $run"
  done
}

stage_micro() {
  echo micro > $R/phase.txt
  (cd $R/repo/benchmarks/harness && $PY $R/repo/benchmarks/results/micro_lt_ext.py \
    > $R/results/micro_lt_ext.out 2>&1) || fail micro
  log micro done
}

stage_f1() {
  for ARMS in ${F1_ARMS:-b0-b0sd b0-ltd}; do
    echo "f1_$ARMS" > $R/phase.txt
    log "start f1 $ARMS"
    $PY repo/benchmarks/harness/p72_f1_blocks.py --root $R --output $R/results/f1-$ARMS \
      --arms $ARMS --host-cache-mib 12288 --timed-steps 3 --max-blocks 10 \
      --label "RTX 3060 (vast.ai), Moonlight, $ARMS" > $R/results/f1-$ARMS.log 2>&1
    log "f1 $ARMS exit $?"
  done
}

for stage in "$@"; do "stage_$stage"; done
echo "done_$*" > $R/phase.txt
