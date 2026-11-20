#!/usr/bin/env bash
set -euo pipefail

# Turn-key RTX 4060 Ti 8 GB (native BF16) + 32 GiB RAM (swap=0) + physical NVMe runner
# for the DeepSeek P3 Unified GO stack:
#   Track 1: M1 (decoupled sub-window=1) + M2+ (zero-materialization streaming) +
#            H2 (ABA-safe window NF4 dequant) + M3 (LargeLayerBufferPool release)
#   Track 2: C++17 PackedStreamAsync (io_workers=4, ring_slots=8, cross-layer prefetch) +
#            2-slot async DMA + CUDA NF4 decode + PR #73 batch_invariant Triton GEMM
#   Track 3: Chunked CE head (PR #63 bridge, chunk=256)
#
# Usage (Kanana 30B-A3B, default):
#   P3_WORK=/mnt/nvme/kanana bash benchmarks/harness/run_p3_unified_go_4060ti.sh
#
# Usage (DeepSeek-V3 671B custom local checkpoint):
#   P3_CASE=custom P3_MODEL_DIR=/mnt/nvme/deepseek-v3/weights P3_WORK=/mnt/nvme/deepseek-v3 \
#   P3_HOST_CACHE_GIB=18.0 bash benchmarks/harness/run_p3_unified_go_4060ti.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
P3_CASE="${P3_CASE:-kanana}"
P3_WORK="${P3_WORK:-/mnt/nvme/p3-kanana-work}"
P3_MODEL_DIR="${P3_MODEL_DIR:-}"
P3_HOST_CACHE_GIB="${P3_HOST_CACHE_GIB:-18.0}"
P3_IO_WORKERS="${P3_IO_WORKERS:-4}"
P3_RING_SLOTS="${P3_RING_SLOTS:-8}"
P3_OUT_DIR="${P3_OUT_DIR:-$REPO_ROOT/benchmarks/results/unified-go-4060ti-$(date +%Y%m%d-%H%M%S)}"
P3_EXT_DIR="${P3_EXT_DIR:-$P3_OUT_DIR/ext}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export PYTHONPATH="$P3_EXT_DIR:$REPO_ROOT/src:$REPO_ROOT"

mkdir -p "$P3_EXT_DIR" "$P3_OUT_DIR/obj"

echo "=== [1/4] Building C++17 + CUDA NF4 _native_expert_stream_async ==="
python "$REPO_ROOT/src/soup_cli/utils/_native_expert_async/build_ext.py" \
  --with-cuda-nf4 build_ext \
  --build-lib "$P3_EXT_DIR" \
  --build-temp "$P3_OUT_DIR/obj"

echo "=== [2/4] Verifying Unified GO unit & parity test suite ==="
python -m pytest -o addopts="" -v \
  "$REPO_ROOT/tests/test_p3_decoupled_subwindow.py" \
  "$REPO_ROOT/tests/test_l2l_chunked_ce.py" \
  "$REPO_ROOT/tests/test_p3_unified_go.py"

EXTRA_MODEL_ARGS=()
if [[ -n "$P3_MODEL_DIR" ]]; then
  EXTRA_MODEL_ARGS+=(--model-dir "$P3_MODEL_DIR")
fi

echo "=== [3/4] Paired A-B-B-A (k=64, S=64): std_mbr vs unified_go ==="
python -u "$REPO_ROOT/benchmarks/harness/p3_unified_go_bench.py" \
  --case "$P3_CASE" \
  "${EXTRA_MODEL_ARGS[@]}" \
  --work "$P3_WORK" \
  --device cuda:0 --dtype bfloat16 --storage nf4 \
  --k 64 --seq-len 64 --subwindow 1 \
  --expert-gemm batch_invariant \
  --expert-loader cpp_async_2slot \
  --io-workers "$P3_IO_WORKERS" --ring-slots "$P3_RING_SLOTS" \
  --cross-layer-prefetch --cuda-nf4-decode \
  --head-loss chunked --ce-chunk 256 \
  --host-cache-gib "$P3_HOST_CACHE_GIB" \
  --arms std_mbr unified_go unified_go std_mbr \
  --output "$P3_OUT_DIR/abba-k64-s64.json"

echo "=== [4/4] Long-context high-k stress (k=64, S=512): unified_go (32,768 tok/step) ==="
python -u "$REPO_ROOT/benchmarks/harness/p3_unified_go_bench.py" \
  --case "$P3_CASE" \
  "${EXTRA_MODEL_ARGS[@]}" \
  --work "$P3_WORK" --prepared \
  --device cuda:0 --dtype bfloat16 --storage nf4 \
  --k 64 --seq-len 512 --subwindow 1 \
  --expert-gemm batch_invariant \
  --expert-loader cpp_async_2slot \
  --io-workers "$P3_IO_WORKERS" --ring-slots "$P3_RING_SLOTS" \
  --cross-layer-prefetch --cuda-nf4-decode \
  --head-loss chunked --ce-chunk 256 \
  --host-cache-gib "$P3_HOST_CACHE_GIB" \
  --arms unified_go unified_go \
  --output "$P3_OUT_DIR/stress-k64-s512.json"

echo "Done! Results saved under $P3_OUT_DIR"
