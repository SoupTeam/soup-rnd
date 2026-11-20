// Elementwise kernels of the fused expert (``--expert-gemm lt-fused``). Each one replaces a chain
// of ATen ops and must give the bits of that chain, op for op, rounding for rounding:
//
//   silu(x)          -> bf16( x / (1 + expf(-x)) )                    (torch silu, float opmath)
//   mul(bf16, bf16)  -> bf16( float(a) * float(b) )
//   silu_backward    -> bf16( dy * s * (1 + x * (1 - s)) ), s = 1 / (1 + expf(-x))
//
// The formulas are copied from ATen's ActivationSiluKernel.cu and BinaryMulKernel.cu; compile
// flags are nvcc's defaults like torch's (no fast math), so expf, division and fused
// multiply-adds are contracted the same way. tests/test_a0_exact_lt_kernels_gpu.py compares every
// kernel with the ATen chain bit for bit.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ __nv_bfloat16 to_bf16(float v) { return __float2bfloat16_rn(v); }

__global__ void swiglu_fwd_kernel(const __nv_bfloat16* h, __nv_bfloat16* a, int64_t rows,
                                  int64_t inter) {
  const int64_t n = rows * inter;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t r = i / inter;
    const int64_t c = i - r * inter;
    const float gate = __bfloat162float(h[r * 2 * inter + c]);
    const float up = __bfloat162float(h[r * 2 * inter + inter + c]);
    const float s = __bfloat162float(to_bf16(silu_f(gate)));  // silu output is bf16
    a[i] = to_bf16(s * up);
  }
}

// grad_h = cat([silu_backward(bf16(grad_a * up), gate), bf16(grad_a * s)], -1) with
// s = bf16(silu(gate)); h is [rows, 2 * inter] (gate | up), grad_a [rows, inter].
__global__ void swiglu_bwd_kernel(const __nv_bfloat16* h, const __nv_bfloat16* grad_a,
                                  __nv_bfloat16* grad_h, int64_t rows, int64_t inter) {
  const int64_t n = rows * inter;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t r = i / inter;
    const int64_t c = i - r * inter;
    const float gate = __bfloat162float(h[r * 2 * inter + c]);
    const float up = __bfloat162float(h[r * 2 * inter + inter + c]);
    const float ga = __bfloat162float(grad_a[i]);
    const float s_bf = __bfloat162float(to_bf16(silu_f(gate)));
    const float dy = __bfloat162float(to_bf16(ga * up));  // grad_a * up, rounded to bf16
    const float sig = 1.0f / (1.0f + expf(-gate));
    grad_h[r * 2 * inter + c] = to_bf16(dy * sig * (1.0f + gate * (1.0f - sig)));
    grad_h[r * 2 * inter + inter + c] = to_bf16(ga * s_bf);
  }
}

// out = float(y) * w   (bf16 y [rows, cols], fp32 route weight per row)
__global__ void scale_fwd_kernel(const __nv_bfloat16* y, const float* w, int64_t w_stride,
                                 float* out, int64_t rows, int64_t cols) {
  const int64_t n = rows * cols;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t r = i / cols;
    out[i] = __bfloat162float(y[i]) * w[r * w_stride];
  }
}

template <typename G>
__device__ __forceinline__ float load_g(const G* g, int64_t i);
template <>
__device__ __forceinline__ float load_g<float>(const float* g, int64_t i) { return g[i]; }
template <>
__device__ __forceinline__ float load_g<__nv_bfloat16>(const __nv_bfloat16* g, int64_t i) {
  return __bfloat162float(g[i]);
}

// grad_y = bf16(float(g) * w) and prod = float(g) * float(y), one pass over g.
template <typename G>
__global__ void grad_scale_kernel(const G* g, const __nv_bfloat16* y, const float* w,
                                  int64_t w_stride, __nv_bfloat16* grad_y, float* prod,
                                  int64_t rows, int64_t cols) {
  const int64_t n = rows * cols;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t r = i / cols;
    const float gv = load_g<G>(g, i);
    if (grad_y != nullptr) grad_y[i] = to_bf16(gv * w[r * w_stride]);
    if (prod != nullptr) prod[i] = gv * __bfloat162float(y[i]);
  }
}

constexpr int kMaxSegs = 32;

// Row blocks of one layout copied into another in one launch: segment s takes ``rows[s]`` rows
// from row ``src_row[s]`` of the source to row ``dst_row[s]`` of the destination. One thread
// block per row, 16-byte vector moves; row_bytes must be a multiple of 16.
struct SegTable {
  int64_t src_row[kMaxSegs];
  int64_t dst_row[kMaxSegs];
  int64_t rows[kMaxSegs];
  int n;
};

__global__ void copy_segments_kernel(const char* src, char* dst, SegTable t, int64_t row_bytes) {
  int64_t r = blockIdx.x;
  int s = 0;
  while (s < t.n - 1 && r >= t.rows[s]) {
    r -= t.rows[s];
    ++s;
  }
  const int4* sp = reinterpret_cast<const int4*>(src + (t.src_row[s] + r) * row_bytes);
  int4* dp = reinterpret_cast<int4*>(dst + (t.dst_row[s] + r) * row_bytes);
  for (int64_t c = threadIdx.x; c < row_bytes / 16; c += blockDim.x) dp[c] = sp[c];
}

int blocks_for(int64_t n) {
  const int64_t b = (n + 255) / 256;
  return static_cast<int>(b > 4096 ? 4096 : (b < 1 ? 1 : b));
}

// ---- Row sums with ATen's reduction tree --------------------------------------------------------
// ``prod[rows, hid].sum(1, keepdim=True)`` per segment, one launch for all segments. ATen's
// reduce_kernel (Reduce.cuh) picks its tree from the segment's row count R: with hid vectorised by
// 4 (dim0 = hid / 4), dim0_pow2 = min(last_pow2(dim0), 512), block_width = min(dim0_pow2, 32),
// block_height = min(last_pow2(R), 512 / block_width), block_width = min(dim0_pow2,
// 512 / block_height). Every output row is one block row of block_width lanes: lane t adds
// the float4 at t, t + bx, ... into four accumulators, the accumulators are combined in order,
// then lanes combine through shared memory (offsets bx/2 down to 32) and warp shuffles (offsets
// 16, 8, ..., 1 in the CUDA build of PyTorch 2.14). This mirrors that code for float input with 16-byte aligned rows.

constexpr int kMaxSumSegs = 32;

struct SumTable {
  int n;
  int rows[kMaxSumSegs];
  int height[kMaxSumSegs];
  int width[kMaxSumSegs];
  int64_t row0[kMaxSumSegs];
};

__global__ void segment_row_sum_kernel(const float* __restrict__ prod, float* __restrict__ out,
                                       SumTable t, int64_t hid) {
  int b = blockIdx.x;
  int s = 0;
  for (; s < t.n; ++s) {
    const int blocks = (t.rows[s] + t.height[s] - 1) / t.height[s];
    if (b < blocks) break;
    b -= blocks;
  }
  const int bx = t.width[s];
  const int height = t.height[s];
  const bool active = threadIdx.x < bx * height;
  const int tx = threadIdx.x % bx;
  const int ty = threadIdx.x / bx;
  const int row = b * height + ty;
  __shared__ float shared[512];
  float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
  if (active && row < t.rows[s]) {
    const float4* in = reinterpret_cast<const float4*>(prod + (t.row0[s] + row) * hid);
    const int vecs = static_cast<int>(hid / 4);
    for (int idx = tx; idx < vecs; idx += bx) {
      const float4 v = in[idx];
      v0 = v0 + v.x;
      v1 = v1 + v.y;
      v2 = v2 + v.z;
      v3 = v3 + v.w;
    }
  }
  float value = v0;
  value = value + v1;
  value = value + v2;
  value = value + v3;
  int dim_x = bx;
  const int base = tx + ty * bx;
  if (dim_x > 32) {
    if (active) shared[base] = value;
    for (int offset = dim_x / 2; offset >= 32; offset >>= 1) {
      __syncthreads();
      if (active && tx < offset && tx + offset < bx) {
        value = value + shared[base + offset];
        shared[base] = value;
      }
    }
    dim_x = 32;
  }
  __syncthreads();
  for (int offset = dim_x >> 1; offset > 0; offset >>= 1) {
    const float other = __shfl_down_sync(0xffffffffu, value, offset);
    value = value + other;
  }
  if (active && tx == 0 && row < t.rows[s]) out[t.row0[s] + row] = value;
}

int last_pow2(int v) {
  int p = 1;
  while (p * 2 <= v) p *= 2;
  return p;
}

}  // namespace

void launch_swiglu_fwd(const void* h, void* a, int64_t rows, int64_t inter, cudaStream_t stream) {
  swiglu_fwd_kernel<<<blocks_for(rows * inter), 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(h), static_cast<__nv_bfloat16*>(a), rows, inter);
}

void launch_swiglu_bwd(const void* h, const void* grad_a, void* grad_h, int64_t rows,
                       int64_t inter, cudaStream_t stream) {
  swiglu_bwd_kernel<<<blocks_for(rows * inter), 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(h), static_cast<const __nv_bfloat16*>(grad_a),
      static_cast<__nv_bfloat16*>(grad_h), rows, inter);
}

void launch_scale_fwd(const void* y, const float* w, int64_t w_stride, float* out, int64_t rows,
                      int64_t cols, cudaStream_t stream) {
  scale_fwd_kernel<<<blocks_for(rows * cols), 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(y), w, w_stride, out, rows, cols);
}

void launch_grad_scale(const void* g, bool g_is_bf16, const void* y, const float* w,
                       int64_t w_stride, void* grad_y, float* prod, int64_t rows, int64_t cols,
                       cudaStream_t stream) {
  const int blocks = blocks_for(rows * cols);
  if (g_is_bf16) {
    grad_scale_kernel<__nv_bfloat16><<<blocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(g), static_cast<const __nv_bfloat16*>(y), w, w_stride,
        static_cast<__nv_bfloat16*>(grad_y), prod, rows, cols);
  } else {
    grad_scale_kernel<float><<<blocks, 256, 0, stream>>>(
        static_cast<const float*>(g), static_cast<const __nv_bfloat16*>(y), w, w_stride,
        static_cast<__nv_bfloat16*>(grad_y), prod, rows, cols);
  }
}

void launch_copy_segments(const void* src, void* dst, const int64_t* src_row,
                          const int64_t* dst_row, const int64_t* rows, int n, int64_t row_bytes,
                          cudaStream_t stream) {
  SegTable table;
  table.n = n;
  int64_t total = 0;
  for (int i = 0; i < n; ++i) {
    table.src_row[i] = src_row[i];
    table.dst_row[i] = dst_row[i];
    table.rows[i] = rows[i];
    total += rows[i];
  }
  if (total == 0) return;
  copy_segments_kernel<<<static_cast<unsigned>(total), 256, 0, stream>>>(
      static_cast<const char*>(src), static_cast<char*>(dst), table, row_bytes);
}

bool launch_segment_row_sum(const float* prod, float* out, const int64_t* counts, int n,
                            int64_t hid, cudaStream_t stream) {
  // ATen vectorises the reduction only for hid > 128 (multiple of 4); keep to what is mirrored
  if (n < 1 || n > kMaxSumSegs || hid <= 128 || hid % 4 != 0) return false;
  if (reinterpret_cast<std::uintptr_t>(prod) % 16 != 0) return false;
  SumTable table;
  table.n = n;
  const int dim0 = static_cast<int>(hid / 4);
  const int dim0_pow2 = dim0 < 512 ? last_pow2(dim0) : 512;
  int64_t row0 = 0;
  int blocks = 0;
  for (int i = 0; i < n; ++i) {
    const int rows = static_cast<int>(counts[i]);
    const int dim1_pow2 = rows < 512 ? last_pow2(rows) : 512;
    const int block_width0 = dim0_pow2 < 32 ? dim0_pow2 : 32;
    const int height = dim1_pow2 < 512 / block_width0 ? dim1_pow2 : 512 / block_width0;
    const int width = dim0_pow2 < 512 / height ? dim0_pow2 : 512 / height;
    table.rows[i] = rows;
    table.height[i] = height;
    table.width[i] = width;
    table.row0[i] = row0;
    row0 += rows;
    blocks += (rows + height - 1) / height;
  }
  if (blocks == 0) return true;
  segment_row_sum_kernel<<<static_cast<unsigned>(blocks), 512, 0, stream>>>(prod, out, table, hid);
  return true;
}
