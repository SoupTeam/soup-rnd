// A0-exact merged expert GEMM: cuBLASLt with a caller-chosen algo, on torch's Lt handle and stream.
//
// out[M, m] = x[M, k] @ (transposed ? w.T : w), bf16 in and out, fp32 compute, alpha 1, beta 0.
// In cuBLAS's column-major view this is D[m, M] = op(A) * B with A = w as stored and B = x, the
// same arguments torch passes to cublasGemmEx for F.linear (transposed, "tn") and for the input
// gradient g @ W ("nn"). An algo is the raw cublasLtMatmulAlgo_t (8 x uint64) in an int64 tensor.
// Built by benchmarks/harness/a0_exact_lt.py:load_extension (torch JIT, needs CUDA headers).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#include <cstdint>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <tuple>
#include <utility>
#include <vector>

#ifdef A0_HAVE_KERNELS
// Raw launchers of a0_exact_lt_kernels.cu.
void launch_swiglu_fwd(const void* h, void* a, int64_t rows, int64_t inter, cudaStream_t stream);
void launch_swiglu_bwd(const void* h, const void* grad_a, void* grad_h, int64_t rows,
                       int64_t inter, cudaStream_t stream);
void launch_scale_fwd(const void* y, const float* w, int64_t w_stride, float* out, int64_t rows,
                      int64_t cols, cudaStream_t stream);
void launch_grad_scale(const void* g, bool g_is_bf16, const void* y, const float* w,
                       int64_t w_stride, void* grad_y, float* prod, int64_t rows, int64_t cols,
                       cudaStream_t stream);
bool launch_segment_row_sum(const float* prod, float* out, const int64_t* counts, int n,
                            int64_t hid, cudaStream_t stream);
void launch_copy_segments(const void* src, void* dst, const int64_t* src_row,
                          const int64_t* dst_row, const int64_t* rows, int n, int64_t row_bytes,
                          cudaStream_t stream);

#endif

namespace {

// torch's default cuBLAS workspace for sm_86 (aten/src/ATen/cuda/CublasHandlePool.cpp).
constexpr size_t kWorkspaceBytes = 4096ull * 1024 * 2 + 16ull * 1024 * 8;
static_assert(sizeof(cublasLtMatmulAlgo_t) == 8 * sizeof(int64_t), "unexpected algo size");

#define LT_CHECK(expr)                                                                   \
  do {                                                                                   \
    cublasStatus_t status_ = (expr);                                                     \
    TORCH_CHECK(status_ == CUBLAS_STATUS_SUCCESS, #expr " failed with cuBLAS status ", \
                static_cast<int>(status_));                                              \
  } while (0)

struct Problem {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr;
  cublasLtMatrixLayout_t b = nullptr;
  cublasLtMatrixLayout_t c = nullptr;
  int64_t m = 0;

  Problem(const at::Tensor& x, const at::Tensor& w, bool transposed) {
    check_inputs(x, w, transposed);
    init(x.size(0), x.size(1), transposed ? w.size(0) : w.size(1), transposed);
  }

  Problem(int64_t rows, int64_t k, int64_t out_cols, bool transposed) {
    init(rows, k, out_cols, transposed);
  }

  static void check_inputs(const at::Tensor& x, const at::Tensor& w, bool transposed) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "x and w must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16,
                "x and w must be bf16");
    TORCH_CHECK(x.dim() == 2 && w.dim() == 2, "x and w must be 2-D");
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous(), "x and w must be contiguous");
    TORCH_CHECK((transposed ? w.size(1) : w.size(0)) == x.size(1), "inner dimensions differ");
  }

  void init(int64_t rows, int64_t k, int64_t out_cols, bool transposed) {
    m = out_cols;
    const cublasOperation_t ta = transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t tb = CUBLAS_OP_N;
    LT_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    if (transposed) {  // w [m, k] row-major is column-major [k, m], leading dimension k
      LT_CHECK(cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, k, m, k));
    } else {  // w [k, m] row-major is column-major [m, k], leading dimension m
      LT_CHECK(cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, m, k, m));
    }
    LT_CHECK(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, k, rows, k));
    LT_CHECK(cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, m, rows, m));
  }

  Problem(const Problem&) = delete;
  Problem& operator=(const Problem&) = delete;

  ~Problem() {
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (op) cublasLtMatmulDescDestroy(op);
  }
};

at::Tensor heuristics(const at::Tensor& x, const at::Tensor& w, bool transposed, int64_t limit) {
  TORCH_CHECK(limit > 0, "limit must be positive");
  c10::cuda::CUDAGuard guard(x.device());
  Problem p(x, w, transposed);
  cublasLtMatmulPreference_t pref = nullptr;
  LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  size_t workspace = kWorkspaceBytes;
  cublasStatus_t status = cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace));
  std::vector<cublasLtMatmulHeuristicResult_t> results(static_cast<size_t>(limit));
  int found = 0;
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulAlgoGetHeuristic(at::cuda::getCurrentCUDABlasLtHandle(), p.op, p.a,
                                            p.b, p.c, p.c, pref, static_cast<int>(limit),
                                            results.data(), &found);
  }
  cublasLtMatmulPreferenceDestroy(pref);
  if (status != CUBLAS_STATUS_SUCCESS) found = 0;
  std::vector<const cublasLtMatmulHeuristicResult_t*> ok;
  for (int i = 0; i < found; ++i) {
    if (results[i].state == CUBLAS_STATUS_SUCCESS) ok.push_back(&results[i]);
  }
  at::Tensor out = at::empty({static_cast<int64_t>(ok.size()), 8}, at::kLong);
  for (size_t i = 0; i < ok.size(); ++i) {
    std::memcpy(out[static_cast<int64_t>(i)].data_ptr(), &ok[i]->algo, sizeof(cublasLtMatmulAlgo_t));
  }
  return out;
}

// Every (algo id, tile, stages, swizzle, custom option) cuBLASLt offers for this problem without
// split-K that passes cublasLtMatmulAlgoCheck, as 8 x int64 each: the candidate pool of the search
// for one algo that reproduces the A0 bits for many row counts.
at::Tensor enumerate_algos(const at::Tensor& x, const at::Tensor& w, bool transposed,
                           int64_t max_custom) {
  c10::cuda::CUDAGuard guard(x.device());
  Problem p(x, w, transposed);
  cublasLtHandle_t handle = at::cuda::getCurrentCUDABlasLtHandle();
  constexpr cublasComputeType_t kCompute = CUBLAS_COMPUTE_32F;
  int ids[512];
  int n_ids = 0;
  LT_CHECK(cublasLtMatmulAlgoGetIds(handle, kCompute, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF,
                                    CUDA_R_16BF, CUDA_R_16BF, 512, ids, &n_ids));
  auto cap_list = [&](const cublasLtMatmulAlgo_t& base, cublasLtMatmulAlgoCapAttributes_t attr) {
    std::vector<uint32_t> v(1024);
    size_t written = 0;
    if (cublasLtMatmulAlgoCapGetAttribute(&base, attr, v.data(), v.size() * sizeof(uint32_t),
                                          &written) != CUBLAS_STATUS_SUCCESS) {
      v.assign(1, 0);  // undefined: keep the algo's default
      return v;
    }
    v.resize(std::max<size_t>(written / sizeof(uint32_t), 1));
    return v;
  };
  auto cap_value = [&](const cublasLtMatmulAlgo_t& base, cublasLtMatmulAlgoCapAttributes_t attr) {
    uint32_t value = 0;
    size_t written = 0;
    if (cublasLtMatmulAlgoCapGetAttribute(&base, attr, &value, sizeof(value), &written) !=
        CUBLAS_STATUS_SUCCESS) {
      return uint32_t{0};
    }
    return value;
  };
  std::vector<cublasLtMatmulAlgo_t> found;
  for (int ii = 0; ii < n_ids; ++ii) {
    cublasLtMatmulAlgo_t base;
    if (cublasLtMatmulAlgoInit(handle, kCompute, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF,
                               CUDA_R_16BF, CUDA_R_16BF, ids[ii], &base) != CUBLAS_STATUS_SUCCESS) {
      continue;
    }
    const auto tiles = cap_list(base, CUBLASLT_ALGO_CAP_TILE_IDS);
    const auto stages = cap_list(base, CUBLASLT_ALGO_CAP_STAGES_IDS);
    const uint32_t swizzles = cap_value(base, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT);
    const uint32_t customs =
        std::min<uint32_t>(cap_value(base, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX),
                           static_cast<uint32_t>(max_custom));
    for (uint32_t tile : tiles) {
      for (uint32_t stage : stages) {
        for (uint32_t swizzle = 0; swizzle <= std::min<uint32_t>(swizzles, 1); ++swizzle) {
          for (uint32_t custom = 0; custom <= customs; ++custom) {
            cublasLtMatmulAlgo_t algo = base;
            const uint32_t split = 1;
            const uint32_t scheme = CUBLASLT_REDUCTION_SCHEME_NONE;
            auto set = [&](cublasLtMatmulAlgoConfigAttributes_t attr, const uint32_t& value) {
              return cublasLtMatmulAlgoConfigSetAttribute(&algo, attr, &value, sizeof(value)) ==
                     CUBLAS_STATUS_SUCCESS;
            };
            if (tile != 0 && !set(CUBLASLT_ALGO_CONFIG_TILE_ID, tile)) continue;
            if (stage != 0 && !set(CUBLASLT_ALGO_CONFIG_STAGES_ID, stage)) continue;
            if (!set(CUBLASLT_ALGO_CONFIG_SPLITK_NUM, split)) continue;
            if (!set(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, scheme)) continue;
            if (!set(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, swizzle)) continue;
            if (!set(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, custom)) continue;
            cublasLtMatmulHeuristicResult_t result;
            std::memset(&result, 0, sizeof(result));
            if (cublasLtMatmulAlgoCheck(handle, p.op, p.a, p.b, p.c, p.c, &algo, &result) ==
                    CUBLAS_STATUS_SUCCESS &&
                result.state == CUBLAS_STATUS_SUCCESS && result.workspaceSize <= kWorkspaceBytes) {
              found.push_back(algo);
            }
          }
        }
      }
    }
  }
  at::Tensor out = at::empty({static_cast<int64_t>(found.size()), 8}, at::kLong);
  for (size_t i = 0; i < found.size(); ++i) {
    std::memcpy(out[static_cast<int64_t>(i)].data_ptr(), &found[i], sizeof(cublasLtMatmulAlgo_t));
  }
  return out;
}

at::Tensor matmul(const at::Tensor& x, const at::Tensor& w, bool transposed,
                  const at::Tensor& algo_tensor) {
  TORCH_CHECK(algo_tensor.device().is_cpu() && algo_tensor.scalar_type() == at::kLong &&
                  algo_tensor.numel() == 8,
              "algo must be a CPU int64 tensor of 8 values");
  c10::cuda::CUDAGuard guard(x.device());
  Problem p(x, w, transposed);
  cublasLtMatmulAlgo_t algo;
  std::memcpy(&algo, algo_tensor.contiguous().data_ptr(), sizeof(algo));
  at::Tensor out = at::empty({x.size(0), p.m}, x.options());
  at::Tensor workspace =
      at::empty({static_cast<int64_t>(kWorkspaceBytes)}, x.options().dtype(at::kByte));
  const float alpha = 1.0f;
  const float beta = 0.0f;
  LT_CHECK(cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(), p.op, &alpha, w.data_ptr(),
                          p.a, x.data_ptr(), p.b, &beta, out.data_ptr(), p.c, out.data_ptr(), p.c,
                          &algo, workspace.data_ptr(), kWorkspaceBytes,
                          at::cuda::getCurrentCUDAStream()));
  return out;
}

// Descriptors per (rows, k, out_cols, transposed), built once per process. The map is shared by
// every thread that calls the extension, so lookups and inserts take a mutex (the returned
// reference stays valid: map nodes do not move).
Problem& cached_problem(int64_t rows, int64_t k, int64_t out_cols, bool transposed) {
  static std::mutex mutex;
  static std::map<std::tuple<int64_t, int64_t, int64_t, bool>, std::unique_ptr<Problem>> cache;
  std::lock_guard<std::mutex> lock(mutex);
  auto key = std::make_tuple(rows, k, out_cols, transposed);
  auto it = cache.find(key);
  if (it == cache.end()) {
    it = cache.emplace(key, std::make_unique<Problem>(rows, k, out_cols, transposed)).first;
  }
  return *it->second;
}

at::Tensor& shared_workspace(const at::Tensor& like) {
  // One cuBLAS workspace per (device, stream): kernels on different streams must not share one
  // (cuBLAS reproducibility notes), and the stream is whichever is current at the call.
  static thread_local std::map<std::pair<int, cudaStream_t>, at::Tensor> pool;
  at::Tensor& workspace =
      pool[std::make_pair(static_cast<int>(like.get_device()),
                          static_cast<cudaStream_t>(at::cuda::getCurrentCUDAStream()))];
  if (!workspace.defined()) {
    workspace = at::empty({static_cast<int64_t>(kWorkspaceBytes)}, like.options().dtype(at::kByte));
  }
  return workspace;
}

bool aligned(const at::Tensor& t) {
  return reinterpret_cast<std::uintptr_t>(t.data_ptr()) % 256 == 0;
}

// The whole planned expert call: for every group, either native mm per segment (empty algo, the
// A0 call itself) or one forced-algo Lt call on the group's rows, written back in segment order.
// Returns the output and the indices of groups Lt refused (their rows are left unset).
std::tuple<at::Tensor, std::vector<int64_t>> merged(
    const at::Tensor& x, const at::Tensor& w, bool transposed, const std::vector<int64_t>& counts,
    const std::vector<std::vector<int64_t>>& groups,
    const std::vector<std::vector<int64_t>>& algos) {
  Problem::check_inputs(x, w, transposed);
  TORCH_CHECK(groups.size() == algos.size(), "one algo entry per group");
  c10::cuda::CUDAGuard guard(x.device());
  const int64_t k = x.size(1);
  const int64_t out_cols = transposed ? w.size(0) : w.size(1);
  std::vector<int64_t> offset(counts.size() + 1, 0);
  for (size_t i = 0; i < counts.size(); ++i) offset[i + 1] = offset[i] + counts[i];
  TORCH_CHECK(offset.back() == x.size(0), "counts must cover the rows of x");
  const at::Tensor wt = transposed ? w.t() : w;
  at::Tensor& workspace = shared_workspace(x);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  std::vector<int64_t> failed;
  // Each segment's output rows, then one cat in segment order: a gather (one cat) and the Lt
  // call per group, instead of one copy per segment each way.
  std::vector<at::Tensor> pieces(counts.size());
  at::Tensor only_output;  // set when a single group covers every row in order
  for (size_t g = 0; g < groups.size(); ++g) {
    const auto& segs = groups[g];
    if (algos[g].empty()) {  // the A0 call: a fresh mm per segment, as torch.mm makes it
      for (int64_t i : segs) pieces[i] = at::mm(x.narrow(0, offset[i], counts[i]), wt);
      continue;
    }
    TORCH_CHECK(algos[g].size() == 8, "an algo is 8 int64 values");
    cublasLtMatmulAlgo_t algo;
    std::memcpy(&algo, algos[g].data(), sizeof(algo));
    int64_t rows = 0;
    bool adjacent = true;
    std::vector<at::Tensor> parts;
    parts.reserve(segs.size());
    for (size_t j = 0; j < segs.size(); ++j) {
      rows += counts[segs[j]];
      if (j > 0 && segs[j] != segs[j - 1] + 1) adjacent = false;
      parts.push_back(x.narrow(0, offset[segs[j]], counts[segs[j]]));
    }
    if (rows == 0) continue;
    at::Tensor in = x.narrow(0, offset[segs[0]], rows);
    if (!(adjacent && aligned(in))) in = at::cat(parts);  // one gather into a fresh buffer
    at::Tensor dst = at::empty({rows, out_cols}, x.options());
    Problem& p = cached_problem(rows, k, out_cols, transposed);
    const cublasStatus_t status = cublasLtMatmul(
        at::cuda::getCurrentCUDABlasLtHandle(), p.op, &alpha, w.data_ptr(), p.a, in.data_ptr(),
        p.b, &beta, dst.data_ptr(), p.c, dst.data_ptr(), p.c, &algo, workspace.data_ptr(),
        kWorkspaceBytes, at::cuda::getCurrentCUDAStream());
    if (status != CUBLAS_STATUS_SUCCESS) {
      failed.push_back(static_cast<int64_t>(g));
      for (int64_t i : segs) pieces[i] = at::empty({counts[i], out_cols}, x.options());
      continue;
    }
    if (groups.size() == 1 && adjacent && segs.size() == counts.size()) only_output = dst;
    int64_t pos = 0;
    for (int64_t i : segs) {
      pieces[i] = dst.narrow(0, pos, counts[i]);
      pos += counts[i];
    }
  }
  for (size_t i = 0; i < pieces.size(); ++i) {  // segments in no group (a partial rerun)
    if (!pieces[i].defined()) pieces[i] = at::empty({counts[i], out_cols}, x.options());
  }
  at::Tensor out = only_output.defined() ? only_output : at::cat(pieces);
  return std::make_tuple(out, failed);
}

// ---- Fused expert (``--expert-gemm lt-fused``): the algo table lives here, per GEMM form
// (out_cols, k, transposed) a vector indexed by row count; segments are grouped by algo identity,
// Lt writes straight into the output rows (every row stride of the expert shapes is a multiple of
// 256 bytes), non-adjacent groups are gathered and scattered with raw device copies.

struct AlgoEntry {
  cublasLtMatmulAlgo_t algo;
  int64_t id = -1;
};
using FormKey = std::tuple<int64_t, int64_t, bool>;
std::map<FormKey, std::vector<AlgoEntry>>& algo_table() {
  static std::map<FormKey, std::vector<AlgoEntry>> table;
  return table;
}
// Guards the algo table: ``register_algo`` writes it (and may reallocate a form's vector), every
// segment matmul reads it and holds ``AlgoEntry`` pointers into it for the length of the call.
std::shared_mutex g_algo_mutex;

// (out_cols, k, transposed, algo id) -> smallest merged row count cuBLASLt refused
using RefusedKey = std::tuple<int64_t, int64_t, bool, int64_t>;
std::mutex g_refused_mutex;
std::map<RefusedKey, int64_t> g_refused;

std::optional<int64_t> smallest_refused(const RefusedKey& key) {
  std::lock_guard<std::mutex> lock(g_refused_mutex);
  auto it = g_refused.find(key);
  if (it == g_refused.end()) return std::nullopt;
  return it->second;
}

void note_refused(const RefusedKey& key, int64_t rows) {
  std::lock_guard<std::mutex> lock(g_refused_mutex);
  auto it = g_refused.find(key);
  if (it == g_refused.end()) {
    g_refused.emplace(key, rows);
  } else {
    it->second = std::min(rows, it->second);
  }
}

void register_algo(int64_t out_cols, int64_t k, bool transposed, int64_t rows,
                   const std::vector<int64_t>& algo) {
  TORCH_CHECK(algo.size() == 8, "an algo is 8 int64 values");
  TORCH_CHECK(rows > 0, "rows must be positive");
  std::unique_lock<std::shared_mutex> lock(g_algo_mutex);
  static std::map<std::vector<int64_t>, int64_t> ids;
  auto it = ids.find(algo);
  const int64_t id = it == ids.end() ? (ids[algo] = static_cast<int64_t>(ids.size())) : it->second;
  auto& form = algo_table()[std::make_tuple(out_cols, k, transposed)];
  if (static_cast<int64_t>(form.size()) <= rows) form.resize(rows + 1);
  std::memcpy(&form[rows].algo, algo.data(), sizeof(cublasLtMatmulAlgo_t));
  form[rows].id = id;
}

// One cuBLASLt handle per thread and device. ``at::cuda::getCurrentCUDABlasLtHandle`` hands out
// the shared cuBLAS handle and sets its stream and workspace on every call; a forced-algo Lt
// call takes the stream and the workspace as arguments, so none of that is needed here. The algo
// fixes the kernel, so the handle does not change the bits.
struct LtHandle {
  cublasLtHandle_t handle = nullptr;
  ~LtHandle() {
    if (handle) cublasLtDestroy(handle);
  }
};

cublasLtHandle_t own_lt_handle() {
  constexpr int kMaxDevices = 16;
  static thread_local LtHandle handles[kMaxDevices];
  const int device = static_cast<int>(at::cuda::current_device());
  TORCH_CHECK(device >= 0 && device < kMaxDevices, "device index out of range");
  if (!handles[device].handle) LT_CHECK(cublasLtCreate(&handles[device].handle));
  return handles[device].handle;
}

// The cuBLAS workspace of (device, stream), looked up with a one-entry cache in front of the map.
// The workspace tensor of a (device, stream) is created once by ``shared_workspace`` and never
// reallocated, so the cached pointer stays valid; the cache is per thread like the map.
void* workspace_ptr(const at::Tensor& like, cudaStream_t stream) {
  static thread_local cudaStream_t last_stream = nullptr;
  static thread_local int last_device = -1;
  static thread_local void* last_ptr = nullptr;
  const int device = static_cast<int>(like.get_device());
  if (last_ptr && last_stream == stream && last_device == device) return last_ptr;
  last_ptr = shared_workspace(like).data_ptr();
  last_stream = stream;
  last_device = device;
  return last_ptr;
}

bool lt_raw(const void* in, int64_t rows, int64_t k, const at::Tensor& w, bool transposed,
            const cublasLtMatmulAlgo_t& algo, void* out, const at::Tensor& like,
            cudaStream_t stream) {
  const int64_t out_cols = transposed ? w.size(0) : w.size(1);
  Problem& p = cached_problem(rows, k, out_cols, transposed);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  return cublasLtMatmul(own_lt_handle(), p.op, &alpha, w.data_ptr(), p.a, in, p.b, &beta, out,
                        p.c, out, p.c, &algo, workspace_ptr(like, stream), kWorkspaceBytes,
                        stream) == CUBLAS_STATUS_SUCCESS;
}

void device_copy(void* dst, const void* src, size_t bytes) {
  C10_CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice,
                                 at::cuda::getCurrentCUDAStream()));
}

bool aligned_ptr(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 256 == 0; }

// Scratch buffers of the fused expert. The staging slots are one arena per (device, stream):
// one preallocated block serves the forward and the backward and every visit, instead of a
// tensor per slot and thread. A tensor per slot let a few small buffers pin 160/320 MiB
// segments that the caching allocator had made for larger, already freed allocations. Slots
// have fixed offsets in the arena, so the pointers of one call never overlap.
//
// The arena is sized by ``prealloc_scratch`` (the launch options of the benchmark); a slot that
// does not fit (a larger visit than the arena was sized for, or a gather slot left out) falls
// back to the per-thread per-slot pool below, which is also what runs when no arena was sized.
enum Slot { kH, kA, kY, kGradA, kGradY, kGradH, kProd, kGatherIn, kGatherOut, kSlots };

constexpr int kScratchSlots = static_cast<int>(kSlots);
constexpr int64_t kScratchAlign = 256;

struct ScratchPlan {
  int64_t bytes[kScratchSlots] = {};
};

struct ScratchArena {
  at::Tensor buf;
  int64_t off[kScratchSlots] = {};
  int64_t len[kScratchSlots] = {};
  int64_t bytes = 0;
};

std::mutex g_arena_mutex;
std::map<std::pair<int, cudaStream_t>, ScratchArena> g_arenas;
ScratchPlan g_arena_plan;
bool g_arena_planned = false;

int64_t align_up(int64_t value, int64_t align) {
  return (value + align - 1) / align * align;
}

// Sizes of one visit. The gather slots take the largest row in/out bytes of any ``seg_matmul_raw``
// call of the visit, so a forward and a backward share one layout even when only one uses them.
ScratchPlan scratch_plan(int64_t rows, int64_t hid, int64_t inter, bool backward, bool gather) {
  ScratchPlan plan;
  plan.bytes[kH] = rows * 2 * inter * 2;
  plan.bytes[kA] = rows * inter * 2;
  plan.bytes[kY] = rows * hid * 2;
  if (gather) {
    const int64_t row_bytes = std::max(2 * hid, 4 * inter);
    plan.bytes[kGatherIn] = rows * row_bytes;
    plan.bytes[kGatherOut] = rows * row_bytes;
  }
  if (backward) {
    plan.bytes[kGradY] = rows * hid * 2;
    plan.bytes[kProd] = rows * hid * 4;
    plan.bytes[kGradA] = rows * inter * 2;
    plan.bytes[kGradH] = rows * 2 * inter * 2;
  }
  return plan;
}

void install_arena(ScratchArena& arena, const ScratchPlan& plan, int device) {
  int64_t offset = 0;
  for (int slot = 0; slot < kScratchSlots; ++slot) {
    arena.off[slot] = offset;
    arena.len[slot] = plan.bytes[slot];
    offset += align_up(plan.bytes[slot], kScratchAlign);
  }
  arena.bytes = offset;
  arena.buf = at::empty({offset},
                        at::TensorOptions().dtype(at::kByte).device(at::kCUDA, device));
}

// Size the arena for the largest visit the run will make. ``now`` allocates at the call (used at
// extension load, before the weights); otherwise the first expert call materializes it.
void prealloc_scratch(int64_t rows, int64_t hid, int64_t inter, bool now, bool gather) {
  TORCH_CHECK(rows > 0 && hid > 0 && inter > 0, "scratch arena needs positive rows, hid, inter");
  const ScratchPlan plan = scratch_plan(rows, hid, inter, true, gather);
  std::lock_guard<std::mutex> lock(g_arena_mutex);
  g_arena_plan = plan;
  g_arena_planned = true;
  if (!now) return;
  const int device = static_cast<int>(at::cuda::current_device());
  const auto stream = static_cast<cudaStream_t>(at::cuda::getCurrentCUDAStream());
  install_arena(g_arenas[std::make_pair(device, stream)], plan, device);
}

std::map<std::string, int64_t> scratch_stats() {
  std::lock_guard<std::mutex> lock(g_arena_mutex);
  int64_t bytes = 0;
  for (const auto& entry : g_arenas) bytes += entry.second.bytes;
  return {
      {"arenas", static_cast<int64_t>(g_arenas.size())},
      {"bytes", bytes},
      {"planned", g_arena_planned ? 1 : 0},
  };
}

void* legacy_scratch(int slot, int64_t bytes, const at::Tensor& like) {
  static thread_local std::map<std::tuple<int, int, cudaStream_t>, at::Tensor> pool;
  const auto key = std::make_tuple(static_cast<int>(like.get_device()), slot,
                                   static_cast<cudaStream_t>(at::cuda::getCurrentCUDAStream()));
  at::Tensor& t = pool[key];
  if (!t.defined() || t.numel() < bytes) {
    t = at::empty({std::max<int64_t>(bytes, 256)}, like.options().dtype(at::kByte));
  }
  return t.data_ptr();
}

void* scratch(int slot, int64_t bytes, const at::Tensor& like) {
  if (bytes <= 0 || slot < 0 || slot >= kScratchSlots) return legacy_scratch(slot, bytes, like);
  const int device = static_cast<int>(like.get_device());
  const auto stream = static_cast<cudaStream_t>(at::cuda::getCurrentCUDAStream());
  {
    std::lock_guard<std::mutex> lock(g_arena_mutex);
    auto it = g_arenas.find(std::make_pair(device, stream));
    if (it == g_arenas.end() && g_arena_planned) {
      it = g_arenas.emplace(std::make_pair(device, stream), ScratchArena()).first;
      install_arena(it->second, g_arena_plan, device);
    }
    if (it != g_arenas.end() && it->second.len[slot] >= bytes) {
      return static_cast<char*>(it->second.buf.data_ptr()) + it->second.off[slot];
    }
  }
  return legacy_scratch(slot, bytes, like);
}

// Number of CUDA streams the Lt calls of one segment matmul are spread over (1: the current
// stream only). Each group of segments writes its own rows of the output, so groups are
// independent; the join waits on every side stream before the next op reads the output.
int g_streams = 1;
void set_streams(int64_t n) {
  TORCH_CHECK(n >= 1 && n <= 8, "streams must be between 1 and 8");
  g_streams = static_cast<int>(n);
}

c10::cuda::CUDAStream side_stream(int index, c10::DeviceIndex device) {
  static thread_local std::map<std::pair<int, int>, c10::cuda::CUDAStream> streams;
  const auto key = std::make_pair(static_cast<int>(device), index);
  auto it = streams.find(key);
  if (it == streams.end()) {
    it = streams.emplace(key, c10::cuda::getStreamFromPool(false, device)).first;
  }
  return it->second;
}

constexpr int kMaxCopySegs = 32;

// Row blocks of ``src`` laid out as ``dst``: one kernel launch (a raw device copy per segment
// without the kernels build).
void copy_segments(const void* src, void* dst, const std::vector<int64_t>& src_row,
                   const std::vector<int64_t>& dst_row, const std::vector<int64_t>& rows,
                   int64_t row_bytes) {
  const int n = static_cast<int>(rows.size());
#ifdef A0_HAVE_KERNELS
  if (n <= kMaxCopySegs && row_bytes % 16 == 0) {
    launch_copy_segments(src, dst, src_row.data(), dst_row.data(), rows.data(), n, row_bytes,
                         at::cuda::getCurrentCUDAStream());
    return;
  }
#endif
  for (int i = 0; i < n; ++i) {
    device_copy(static_cast<char*>(dst) + dst_row[i] * row_bytes,
                static_cast<const char*>(src) + src_row[i] * row_bytes, rows[i] * row_bytes);
  }
}

// Per-call planning buffers of seg_matmul_raw, kept per thread so a call allocates nothing for them
// (the vectors keep their capacity from call to call; the function is not re-entrant).
struct Plan {
  std::vector<const AlgoEntry*> entry;  // A0 algo of each segment
  std::vector<int64_t> group_id;        // algo id of each group, in order of first appearance
  std::vector<int> group_of;            // group index of each segment
  std::vector<int> gstart, cursor;      // range of each group in ``order`` (size groups + 1)
  std::vector<int64_t> order;           // segments grouped by algo, index order inside a group
  std::vector<int64_t> pos;             // first row of each segment in the working layout
  std::vector<int64_t> src_row, dst_row, seg_rows;
};

Plan& plan() {
  static thread_local Plan p;
  return p;
}

// out = in @ (w.T if transposed else w) per segment with the A0 algo of each segment's row count,
// one Lt call per group of segments sharing an algo; ``in`` and ``out`` are raw rows.
void seg_matmul_raw(const void* in, const at::Tensor& w, bool transposed,
                    const std::vector<int64_t>& counts, const std::vector<int64_t>& offset,
                    void* out, const at::Tensor& like) {
  const int64_t k = transposed ? w.size(1) : w.size(0);
  const int64_t out_cols = transposed ? w.size(0) : w.size(1);
  const size_t n = counts.size();
  std::shared_lock<std::shared_mutex> algo_lock(g_algo_mutex);  // entries point into the table
  auto form_it = algo_table().find(std::make_tuple(out_cols, k, transposed));
  TORCH_CHECK(form_it != algo_table().end(), "no A0 algo registered for this GEMM form");
  const auto& form = form_it->second;
  Plan& P = plan();
  P.entry.resize(n);
  P.group_of.resize(n);
  P.group_id.clear();
  for (size_t i = 0; i < n; ++i) {
    TORCH_CHECK(counts[i] < static_cast<int64_t>(form.size()) && form[counts[i]].id >= 0,
                "no A0 algo registered for ", counts[i], " rows");
    P.entry[i] = &form[counts[i]];
    size_t g = 0;
    while (g < P.group_id.size() && P.group_id[g] != P.entry[i]->id) ++g;
    if (g == P.group_id.size()) P.group_id.push_back(P.entry[i]->id);
    P.group_of[i] = static_cast<int>(g);
  }
  const size_t groups = P.group_id.size();
  P.gstart.assign(groups + 1, 0);
  for (size_t i = 0; i < n; ++i) ++P.gstart[P.group_of[i] + 1];
  for (size_t g = 0; g < groups; ++g) P.gstart[g + 1] += P.gstart[g];
  P.cursor.assign(P.gstart.begin(), P.gstart.end() - 1);
  P.order.resize(n);
  for (size_t i = 0; i < n; ++i) P.order[P.cursor[P.group_of[i]]++] = static_cast<int64_t>(i);
  const size_t esize = 2;  // bf16
  const int64_t in_bytes = k * esize;
  const int64_t out_bytes = out_cols * esize;
  // Groups of one algo are laid out back to back: when the segments are not already in that
  // order, one kernel launch permutes the rows in and one puts the results back.
  bool identity = true;
  for (size_t j = 0; j < n; ++j) identity = identity && P.order[j] == static_cast<int64_t>(j);
  P.pos.resize(n);  // first row of each segment in the working layout
  const char* in_p = static_cast<const char*>(in);
  char* out_p = static_cast<char*>(out);
  int64_t total_rows = 0;
  for (size_t j = 0; j < n; ++j) total_rows += counts[j];
  P.src_row.clear();
  P.dst_row.clear();
  P.seg_rows.clear();
  if (identity) {
    for (size_t j = 0; j < n; ++j) P.pos[j] = offset[j];
  } else {
    int64_t at = 0;
    for (int64_t i : P.order) {
      P.pos[i] = at;
      P.src_row.push_back(offset[i]);
      P.dst_row.push_back(at);
      P.seg_rows.push_back(counts[i]);
      at += counts[i];
    }
    char* gin = static_cast<char*>(scratch(kGatherIn, total_rows * in_bytes, like));
    copy_segments(in, gin, P.src_row, P.dst_row, P.seg_rows, in_bytes);
    in_p = gin;
    out_p = static_cast<char*>(scratch(kGatherOut, total_rows * out_bytes, like));
  }
  auto in_at = [&](int64_t i) { return in_p + P.pos[i] * in_bytes; };
  auto out_at = [&](int64_t i) { return out_p + P.pos[i] * out_bytes; };
  const bool spread = g_streams > 1 && groups > 1;
  const c10::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();
  std::vector<at::cuda::CUDAEvent> joined;
  if (spread) {
    at::cuda::CUDAEvent ready;
    ready.record(main_stream);
    const int sides = std::min<int>(g_streams - 1, static_cast<int>(groups) - 1);
    joined.resize(sides);
    for (int s = 0; s < sides; ++s) ready.block(side_stream(s, like.get_device()));
  }
  std::vector<bool> used(joined.size(), false);
  for (size_t g = 0; g < groups; ++g) {
    const int64_t* segs = P.order.data() + P.gstart[g];
    const size_t count = static_cast<size_t>(P.gstart[g + 1] - P.gstart[g]);
    // group 0 stays on the current stream, the others rotate over the side streams
    c10::cuda::CUDAStream stream = main_stream;
    int side = -1;
    if (spread && g > 0) {
      side = static_cast<int>((g - 1) % joined.size());
      stream = side_stream(side, like.get_device());
      used[side] = true;
    }
    // the guard (a stream exchange and a device check) only matters when the group leaves the
    // current stream
    std::optional<c10::cuda::CUDAStreamGuard> guard;
    if (side >= 0) guard.emplace(stream);
    const cudaStream_t raw_stream = stream;
    int64_t rows = 0;
    for (size_t j = 0; j < count; ++j) rows += counts[segs[j]];
    const auto refusal = std::make_tuple(out_cols, k, transposed, P.entry[segs[0]]->id);
    const std::optional<int64_t> refused_below = smallest_refused(refusal);
    if (count > 1 && (!refused_below || rows < *refused_below)) {
      TORCH_CHECK(aligned_ptr(in_at(segs[0])) && aligned_ptr(out_at(segs[0])),
                  "unaligned segment rows (hidden sizes must be multiples of 128)");
      if (lt_raw(in_at(segs[0]), rows, k, w, transposed, P.entry[segs[0]]->algo, out_at(segs[0]),
                 like, raw_stream)) {
        continue;
      }
      note_refused(refusal, rows);
    }
    for (size_t j = 0; j < count; ++j) {  // one call per segment with its own algo
      const int64_t i = segs[j];
      TORCH_CHECK(aligned_ptr(in_at(i)) && aligned_ptr(out_at(i)),
                  "unaligned segment rows (hidden sizes must be multiples of 128)");
      TORCH_CHECK(lt_raw(in_at(i), counts[i], k, w, transposed, P.entry[i]->algo, out_at(i), like, raw_stream),
                  "cuBLASLt refused the A0 algo of ", counts[i], " rows");
    }
  }
  if (spread) {  // the current stream waits for every side stream that worked
    for (size_t s = 0; s < joined.size(); ++s) {
      if (!used[s]) continue;
      joined[s].record(side_stream(static_cast<int>(s), like.get_device()));
      joined[s].block(main_stream);
    }
  }
  if (!identity) copy_segments(out_p, out, P.dst_row, P.src_row, P.seg_rows, out_bytes);
}

// ATen-tensor form (fallback path and the old ops chain).
at::Tensor seg_matmul(const at::Tensor& x, const at::Tensor& w, bool transposed,
                      const std::vector<int64_t>& counts, const std::vector<int64_t>& offset) {
  TORCH_CHECK(x.is_contiguous(), "segment input must be contiguous");
  const int64_t out_cols = transposed ? w.size(0) : w.size(1);
  at::Tensor out = at::empty({x.size(0), out_cols}, x.options());
  seg_matmul_raw(x.data_ptr(), w, transposed, counts, offset, out.data_ptr(), x);
  return out;
}

std::vector<int64_t> offsets_of(const std::vector<int64_t>& counts, int64_t rows) {
  std::vector<int64_t> offset(counts.size() + 1, 0);
  for (size_t i = 0; i < counts.size(); ++i) offset[i + 1] = offset[i] + counts[i];
  TORCH_CHECK(offset.back() == rows, "counts must cover the rows");
  return offset;
}

void check_expert(const at::Tensor& hidden, const at::Tensor& gate_up, const at::Tensor& down) {
  TORCH_CHECK(hidden.is_cuda() && hidden.scalar_type() == at::kBFloat16 && hidden.dim() == 2,
              "hidden must be 2-D bf16 CUDA");
  TORCH_CHECK(gate_up.is_contiguous() && down.is_contiguous(), "expert weights must be contiguous");
  TORCH_CHECK(gate_up.scalar_type() == at::kBFloat16 && down.scalar_type() == at::kBFloat16,
              "expert weights must be bf16");
}

#ifdef A0_HAVE_KERNELS
bool weights_ok(const at::Tensor& weights, int64_t rows) {
  return weights.scalar_type() == at::kFloat && weights.dim() == 2 && weights.size(0) == rows &&
         weights.size(1) == 1 && weights.stride(0) >= 1;
}

at::Tensor expert_forward_kernels(const at::Tensor& x, const at::Tensor& weights,
                                  const at::Tensor& gate_up, const at::Tensor& down,
                                  const std::vector<int64_t>& counts,
                                  const std::vector<int64_t>& offset) {
  const int64_t rows = x.size(0);
  const int64_t hid = x.size(1);
  const int64_t inter = down.size(1);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  void* h = scratch(kH, rows * 2 * inter * 2, x);
  seg_matmul_raw(x.data_ptr(), gate_up, true, counts, offset, h, x);
  void* a = scratch(kA, rows * inter * 2, x);
  launch_swiglu_fwd(h, a, rows, inter, stream);
  void* y = scratch(kY, rows * hid * 2, x);
  seg_matmul_raw(a, down, true, counts, offset, y, x);
  at::Tensor out = at::empty({rows, hid}, weights.options());
  launch_scale_fwd(y, weights.data_ptr<float>(), weights.stride(0), out.data_ptr<float>(), rows,
                   hid, stream);
  return out;
}

std::tuple<at::Tensor, at::Tensor> expert_backward_kernels(
    const at::Tensor& x, const at::Tensor& weights, const at::Tensor& gate_up,
    const at::Tensor& down, const at::Tensor& grad_out, const std::vector<int64_t>& counts,
    const std::vector<int64_t>& offset, bool need_hidden, bool need_weights) {
  const int64_t rows = x.size(0);
  const int64_t hid = x.size(1);
  const int64_t inter = down.size(1);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  void* h = scratch(kH, rows * 2 * inter * 2, x);
  seg_matmul_raw(x.data_ptr(), gate_up, true, counts, offset, h, x);
  void* y = nullptr;
  if (need_weights) {
    void* a = scratch(kA, rows * inter * 2, x);
    launch_swiglu_fwd(h, a, rows, inter, stream);
    y = scratch(kY, rows * hid * 2, x);
    seg_matmul_raw(a, down, true, counts, offset, y, x);
  }
  const bool g_bf16 = grad_out.scalar_type() == at::kBFloat16;
  void* grad_y = need_hidden ? scratch(kGradY, rows * hid * 2, x) : nullptr;
  float* prod = need_weights ? static_cast<float*>(scratch(kProd, rows * hid * 4, x)) : nullptr;
  launch_grad_scale(grad_out.data_ptr(), g_bf16, y, weights.data_ptr<float>(), weights.stride(0),
                    grad_y, prod, rows, hid, stream);
  at::Tensor grad_weights;
  if (need_weights) {
    grad_weights = at::empty({rows, 1}, weights.options());
    // one launch with ATen's own reduction tree per segment; ATen itself when that is not mirrored
    if (!launch_segment_row_sum(prod, grad_weights.data_ptr<float>(), counts.data(),
                                static_cast<int>(counts.size()), hid, stream)) {
      const auto f32 = weights.options();
      for (size_t i = 0; i < counts.size(); ++i) {
        at::Tensor dst =
            at::from_blob(grad_weights.data_ptr<float>() + offset[i], {counts[i], 1}, f32);
        at::Tensor src = at::from_blob(prod + offset[i] * hid, {counts[i], hid}, f32);
        at::sum_out(dst, src, {1}, true);
      }
    }
  }
  at::Tensor grad_hidden;
  if (need_hidden) {
    void* grad_a = scratch(kGradA, rows * inter * 2, x);
    seg_matmul_raw(grad_y, down, false, counts, offset, grad_a, x);
    void* grad_h = scratch(kGradH, rows * 2 * inter * 2, x);
    launch_swiglu_bwd(h, grad_a, grad_h, rows, inter, stream);
    grad_hidden = at::empty({rows, hid}, x.options());
    seg_matmul_raw(grad_h, gate_up, false, counts, offset, grad_hidden.data_ptr(), x);
  }
  return std::make_tuple(grad_hidden, grad_weights);
}
#endif  // A0_HAVE_KERNELS

// expert_mlp(hidden) * weights, as WeightedFrozenExpert.forward computes it.
at::Tensor expert_forward(const at::Tensor& hidden, const at::Tensor& weights,
                          const at::Tensor& gate_up, const at::Tensor& down,
                          const std::vector<int64_t>& counts) {
  check_expert(hidden, gate_up, down);
  c10::cuda::CUDAGuard guard(hidden.device());
  at::AutoDispatchBelowAutograd no_autograd;  // skip the VariableType layer: same kernels, less CPU
  const at::Tensor x = hidden.contiguous();
  const auto offset = offsets_of(counts, x.size(0));
#ifdef A0_HAVE_KERNELS
  if (weights_ok(weights, x.size(0))) {
    return expert_forward_kernels(x, weights, gate_up, down, counts, offset);
  }
#endif
  const at::Tensor h = seg_matmul(x, gate_up, true, counts, offset);
  const auto halves = h.chunk(2, -1);
  const at::Tensor a = at::mul(at::silu(halves[0]), halves[1]);
  const at::Tensor y = seg_matmul(a, down, true, counts, offset);
  return at::mul(y, weights);
}

// The direct backward of expert_vjp (benchmarks/harness/direct_expert_backward.py), same ops;
// the route-weight product is taken once (elementwise), its reduction per segment.
std::tuple<at::Tensor, at::Tensor> expert_backward(
    const at::Tensor& hidden, const at::Tensor& weights, const at::Tensor& gate_up,
    const at::Tensor& down, const at::Tensor& grad_out, const std::vector<int64_t>& counts,
    bool need_hidden, bool need_weights) {
  check_expert(hidden, gate_up, down);
  c10::cuda::CUDAGuard guard(hidden.device());
  at::AutoDispatchBelowAutograd no_autograd;  // skip the VariableType layer: same kernels, less CPU
  const at::Tensor x = hidden.contiguous();
  const auto offset = offsets_of(counts, x.size(0));
#ifdef A0_HAVE_KERNELS
  const bool g_ok = grad_out.is_contiguous() && (grad_out.scalar_type() == at::kBFloat16 ||
                                                 grad_out.scalar_type() == at::kFloat);
  if (weights_ok(weights, x.size(0)) && g_ok) {
    return expert_backward_kernels(x, weights, gate_up, down, grad_out, counts, offset,
                                   need_hidden, need_weights);
  }
#endif
  const at::Tensor h = seg_matmul(x, gate_up, true, counts, offset);
  const auto halves = h.chunk(2, -1);
  const at::Tensor& gate = halves[0];
  const at::Tensor& up = halves[1];
  const at::Tensor s = at::silu(gate);
  const at::Tensor a = at::mul(s, up);
  at::Tensor y;
  if (need_weights) y = seg_matmul(a, down, true, counts, offset);
  const auto out_dtype = need_weights ? y.scalar_type() : a.scalar_type();
  const at::Tensor g = grad_out.to(at::promote_types(out_dtype, weights.scalar_type()));
  at::Tensor grad_weights;
  if (need_weights) {
    const at::Tensor prod = at::mul(g, y).to(weights.scalar_type());
    grad_weights = at::empty({x.size(0), 1}, weights.options());
    for (size_t i = 0; i < counts.size(); ++i) {
      at::Tensor dst = grad_weights.narrow(0, offset[i], counts[i]);
      at::sum_out(dst, prod.narrow(0, offset[i], counts[i]), {1}, true);
    }
  }
  at::Tensor grad_hidden;
  if (need_hidden) {
    const at::Tensor grad_y = at::mul(g, weights).to(out_dtype).contiguous();
    const at::Tensor grad_a = seg_matmul(grad_y, down, false, counts, offset);
    const at::Tensor grad_gate = at::silu_backward(at::mul(grad_a, up), gate);
    const at::Tensor grad_h = at::cat({grad_gate, at::mul(grad_a, s)}, -1);
    grad_hidden = seg_matmul(grad_h, gate_up, false, counts, offset).to(hidden.scalar_type());
  }
  return std::make_tuple(grad_hidden, grad_weights);
}

#ifdef A0_HAVE_KERNELS
void t_copy_segments(const at::Tensor& src, at::Tensor& dst, const std::vector<int64_t>& src_row,
                     const std::vector<int64_t>& dst_row, const std::vector<int64_t>& rows) {
  TORCH_CHECK(src.is_contiguous() && dst.is_contiguous() && src.size(1) == dst.size(1),
              "contiguous tensors with the same row width");
  copy_segments(src.data_ptr(), dst.data_ptr(), src_row, dst_row, rows,
                src.size(1) * src.element_size());
}

// Test entry points: each kernel on its own, to compare with the ATen chain it replaces.
at::Tensor t_swiglu_fwd(const at::Tensor& h) {
  const int64_t inter = h.size(1) / 2;
  at::Tensor a = at::empty({h.size(0), inter}, h.options());
  launch_swiglu_fwd(h.contiguous().data_ptr(), a.data_ptr(), h.size(0), inter,
                    at::cuda::getCurrentCUDAStream());
  return a;
}
at::Tensor t_swiglu_bwd(const at::Tensor& h, const at::Tensor& grad_a) {
  const int64_t inter = h.size(1) / 2;
  at::Tensor out = at::empty_like(h);
  launch_swiglu_bwd(h.contiguous().data_ptr(), grad_a.contiguous().data_ptr(), out.data_ptr(),
                    h.size(0), inter, at::cuda::getCurrentCUDAStream());
  return out;
}
at::Tensor t_scale_fwd(const at::Tensor& y, const at::Tensor& w) {
  at::Tensor out = at::empty({y.size(0), y.size(1)}, w.options());
  launch_scale_fwd(y.contiguous().data_ptr(), w.data_ptr<float>(), w.stride(0),
                   out.data_ptr<float>(), y.size(0), y.size(1), at::cuda::getCurrentCUDAStream());
  return out;
}
at::Tensor t_segment_row_sum(const at::Tensor& prod, const std::vector<int64_t>& counts) {
  TORCH_CHECK(prod.is_cuda() && prod.scalar_type() == at::kFloat && prod.is_contiguous() &&
                  prod.dim() == 2,
              "prod must be a contiguous 2-D float CUDA tensor");
  at::Tensor out = at::empty({prod.size(0), 1}, prod.options());
  TORCH_CHECK(launch_segment_row_sum(prod.data_ptr<float>(), out.data_ptr<float>(), counts.data(),
                                     static_cast<int>(counts.size()), prod.size(1),
                                     at::cuda::getCurrentCUDAStream()),
              "row sum shape not mirrored");
  return out;
}
std::tuple<at::Tensor, at::Tensor> t_grad_scale(const at::Tensor& g, const at::Tensor& y,
                                                const at::Tensor& w, bool want_grad_y,
                                                bool want_prod) {
  at::Tensor grad_y, prod;
  if (want_grad_y) grad_y = at::empty({g.size(0), g.size(1)}, y.options());
  if (want_prod) prod = at::empty({g.size(0), g.size(1)}, w.options());
  launch_grad_scale(g.contiguous().data_ptr(), g.scalar_type() == at::kBFloat16,
                    y.contiguous().data_ptr(), w.data_ptr<float>(), w.stride(0),
                    want_grad_y ? grad_y.data_ptr() : nullptr,
                    want_prod ? prod.data_ptr<float>() : nullptr, g.size(0), g.size(1),
                    at::cuda::getCurrentCUDAStream());
  return std::make_tuple(grad_y, prod);
}
#endif

}  // namespace

// Busy-wait for ``us`` microseconds with the GIL released (causal profiling: CPU time of the calling
// thread only, see causal_probe.py).
void spin_us(double us) {
  pybind11::gil_scoped_release release;
  const auto end = std::chrono::steady_clock::now() +
                   std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                       std::chrono::duration<double, std::micro>(us));
  while (std::chrono::steady_clock::now() < end) {
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spin_us", &spin_us, "busy-wait with the GIL released");
  m.def("heuristics", &heuristics, "Lt heuristic algos for x @ (w.T if transposed else w)");
  m.def("enumerate_algos", &enumerate_algos, "all split-K-free algo configs that pass AlgoCheck");
  m.def("matmul", &matmul, "x @ (w.T if transposed else w) with a forced Lt algo");
  m.def("set_streams", &set_streams, "spread the Lt calls of a segment matmul over n streams");
  m.def("merged", &merged, "planned expert call: native or forced-algo Lt per segment group");
  m.def("register_algo", &register_algo, "store the A0 algo of (out_cols, k, transposed, rows)");
  m.def("expert_forward", &expert_forward, "expert_mlp(hidden) * weights, merged A0-exact GEMMs");
  m.def("expert_backward", &expert_backward, "direct expert backward, merged A0-exact GEMMs");
  m.def("prealloc_scratch", &prealloc_scratch,
        "size the shared scratch arena for the largest expert visit (now: allocate at once)");
  m.def("scratch_stats", &scratch_stats, "arena count and bytes of the shared scratch");
#ifdef A0_HAVE_KERNELS
  m.attr("has_kernels") = true;
  m.def("copy_segments", &t_copy_segments);
  m.def("swiglu_fwd", &t_swiglu_fwd);
  m.def("swiglu_bwd", &t_swiglu_bwd);
  m.def("scale_fwd", &t_scale_fwd);
  m.def("grad_scale", &t_grad_scale);
  m.def("segment_row_sum", &t_segment_row_sum);
#else
  m.attr("has_kernels") = false;
#endif
}
