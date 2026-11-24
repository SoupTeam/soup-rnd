# Op probe: expert GEMM row counts (Moonlight-16B-A3B, NVIDIA GeForce RTX 3050 Laptop GPU)

Window: op probe, no model; arm: operation; mode: bits (strict deterministic); torch 2.14.1+cu130 (CUDA 13.0).

## Kernel buckets

The first row count of each cuBLAS kernel bucket, a kernel with another stage count being a different reduction order:

### gate_up (2816, 2048)

| first M in bucket | kernel |
|---|---|
| 1 | void gemv2T_kernel_val<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, 128, 16, 4, 4, false, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16>, float>, float, float) |
| 2 | ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn |
| 17 | ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn |
| 32 | ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn |
| 40 | ampere_s16816gemm_bf16_256x128_ldg8_stages_32x3_tn,void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, true, false, false, false>(cublasLt::cublasSplitKParams<float>, float const*, __nv_bfloat16 const*, __nv_bfloat16*, __nv_bfloat16*, float const*, float const*, __nv_bfloat16 const*, float const*, __nv_bfloat16*, void*, long, float*, int*, float*, float*, float const*, float const*, float const*, float const*, float const*) |

The first rows of a small call against the same rows inside a larger call:

| M small | M large | equal | n_diff / numel | max abs diff |
|---|---|---|---|---|
| 15 | 31 | True | 0 / 42240 | 0 |
| 1 | 2 | False | 2 / 2816 | 0.000244141 |
| 8 | 16 | True | 0 / 22528 | 0 |

Per-row calls against the joined call (the row-wise exact variant):

| M small | M large | equal | n_diff / numel | max abs diff |
|---|---|---|---|---|
| 15 | 31 | False | 17 / 42240 | 0.00195312 |
| 1 | 2 | False | 2 / 2816 | 0.000244141 |
| 8 | 16 | False | 10 / 22528 | 0.00195312 |

Per-micro-batch calls against the joined call (the segmented exact variant):

| split | M large | equal | n_diff / numel | max abs diff |
|---|---|---|---|---|
| [15, 16] | 31 | True | 0 / 87296 | 0 |

Zero padding to a shared row count, one nonzero row against two:

| pad | equal | n_diff / numel | max abs diff |
|---|---|---|---|
| 32 | True | 0 / 2816 | 0 |
| 512 | True | 0 / 2816 | 0 |

`allow_bf16_reduced_precision_reduction = False`:

| pair | equal | n_diff / numel | max abs diff |
|---|---|---|---|
| 15 vs 31 | True | 0 / 42240 | 0 |
| 1 vs 2 | False | 2 / 2816 | 0.000244141 |

Kernels with the flag off: `{'1': ['void gemv2T_kernel_val<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, 128, 16, 4, 4, false, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16>, float>, float, float)'], '2': ['ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn'], '15': ['ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn'], '16': ['ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn'], '31': ['ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn'], '32': ['ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn']}`

### down (2048, 1408)

| first M in bucket | kernel |
|---|---|
| 1 | void gemv2T_kernel_val<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, 128, 16, 4, 4, false, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16>, float>, float, float) |
| 2 | ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn |
| 17 | void cublasLt::splitKreduce_kernel<32, 16, int, __nv_bfloat16, __nv_bfloat16, float, __nv_bfloat16, false, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, true, false, false, false>(cublasLt::cublasSplitKParams<float>, __nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*, __nv_bfloat16*, float const*, float const*, __nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*, void*, long, float*, int*, float*, float*, float const*, float const*, float const*, float const*, float const*),void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_64x6_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_64x6_tn_align8::Params) |
| 32 | ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn |
| 40 | void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8::Params) |

The first rows of a small call against the same rows inside a larger call:

| M small | M large | equal | n_diff / numel | max abs diff |
|---|---|---|---|---|
| 15 | 31 | False | 11928 / 30720 | 0.0078125 |
| 1 | 2 | False | 2 / 2048 | 6.10352e-05 |
| 8 | 16 | True | 0 / 16384 | 0 |

Per-row calls against the joined call (the row-wise exact variant):

| M small | M large | equal | n_diff / numel | max abs diff |
|---|---|---|---|---|
| 15 | 31 | False | 11930 / 30720 | 0.0078125 |
| 1 | 2 | False | 2 / 2048 | 6.10352e-05 |
| 8 | 16 | False | 7 / 16384 | 0.00390625 |

Per-micro-batch calls against the joined call (the segmented exact variant):

| split | M large | equal | n_diff / numel | max abs diff |
|---|---|---|---|---|
| [15, 16] | 31 | False | 24691 / 63488 | 0.0078125 |

Zero padding to a shared row count, one nonzero row against two:

| pad | equal | n_diff / numel | max abs diff |
|---|---|---|---|
| 32 | True | 0 / 2048 | 0 |
| 512 | True | 0 / 2048 | 0 |

`allow_bf16_reduced_precision_reduction = False`:

| pair | equal | n_diff / numel | max abs diff |
|---|---|---|---|
| 15 vs 31 | False | 15 / 30720 | 0.00195312 |
| 1 vs 2 | False | 2 / 2048 | 6.10352e-05 |

Kernels with the flag off: `{'1': ['void gemv2T_kernel_val<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, 128, 16, 4, 4, false, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16>, float>, float, float)'], '2': ['ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn'], '15': ['ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn'], '16': ['ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn'], '31': ['void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, true, false, false, false>(cublasLt::cublasSplitKParams<float>, float const*, __nv_bfloat16 const*, __nv_bfloat16*, __nv_bfloat16*, float const*, float const*, __nv_bfloat16 const*, float const*, __nv_bfloat16*, void*, long, float*, int*, float*, float*, float const*, float const*, float const*, float const*, float const*)', 'void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8::Params)'], '32': ['ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn']}`
