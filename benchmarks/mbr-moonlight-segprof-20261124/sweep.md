# Per-M kernel cost of the routed-expert GEMMs (Moonlight-16B-A3B)

Moonlight-16B-A3B on NVIDIA GeForce RTX 3050 Laptop GPU sm_86 4 GiB, torch 2.14.1+cu130: 20 calls per M after 5 warm-up calls, CUDA-only profile, no model.

## gate_up: Moonlight-16B-A3B weight (2816, 2048) on NVIDIA GeForce RTX 3050 Laptop GPU sm_86 4 GiB, torch 2.14.1+cu130

| M | kernel | calls/reps | µs/call | µs/row |
|---|---|---|---|---|
| 1 | total | 1.0 | 65.427 | 65.427 |
| 1 | void gemv2T_kernel_val<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, 128, 16, 4, 4, false, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16>, float>, float, float) | 1.0 | 65.427 | 65.427 |
| 2 | total | 1.0 | 77.136 | 38.568 |
| 2 | ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn | 1.0 | 77.136 | 38.568 |
| 4 | total | 1.0 | 77.328 | 19.332 |
| 4 | ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn | 1.0 | 77.328 | 19.332 |
| 6 | total | 1.0 | 77.566 | 12.928 |
| 6 | ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn | 1.0 | 77.566 | 12.928 |
| 8 | total | 1.0 | 78.038 | 9.755 |
| 8 | ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn | 1.0 | 78.038 | 9.755 |
| 12 | total | 1.0 | 78.446 | 6.537 |
| 12 | ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn | 1.0 | 78.446 | 6.537 |
| 16 | total | 1.0 | 78.765 | 4.923 |
| 16 | ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn | 1.0 | 78.765 | 4.923 |
| 17 | total | 1.0 | 77.927 | 4.584 |
| 17 | ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn | 1.0 | 77.927 | 4.584 |
| 24 | total | 1.0 | 78.504 | 3.271 |
| 24 | ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn | 1.0 | 78.504 | 3.271 |
| 31 | total | 1.0 | 78.896 | 2.545 |
| 31 | ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn | 1.0 | 78.896 | 2.545 |
| 32 | total | 1.0 | 93.039 | 2.907 |
| 32 | ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn | 1.0 | 93.039 | 2.907 |
| 40 | total | 1.0 | 175.569 | 4.389 |
| 40 | ampere_s16816gemm_bf16_256x128_ldg8_stages_32x3_tn | 1.0 | 164.873 | 4.122 |
| 40 | void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, true, false, false, false>(cublasLt::cublasSplitKParams<float>, float const*, __nv_bfloat16 const*, __nv_bfloat16*, __nv_bfloat16*, float const*, float const*, __nv_bfloat16 const*, float const*, __nv_bfloat16*, void*, long, float*, int*, float*, float*, float const*, float const*, float const*, float const*, float const*) | 1.0 | 10.696 | 0.267 |
| 48 | total | 1.0 | 176.732 | 3.682 |
| 48 | ampere_s16816gemm_bf16_256x128_ldg8_stages_32x3_tn | 1.0 | 165.201 | 3.442 |
| 48 | void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, true, false, false, false>(cublasLt::cublasSplitKParams<float>, float const*, __nv_bfloat16 const*, __nv_bfloat16*, __nv_bfloat16*, float const*, float const*, __nv_bfloat16 const*, float const*, __nv_bfloat16*, void*, long, float*, int*, float*, float*, float const*, float const*, float const*, float const*, float const*) | 1.0 | 11.532 | 0.240 |
| 64 | total | 1.0 | 162.966 | 2.546 |
| 64 | ampere_s16816gemm_bf16_256x128_ldg8_stages_32x3_tn | 1.0 | 148.498 | 2.320 |
| 64 | void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, true, false, false, false>(cublasLt::cublasSplitKParams<float>, float const*, __nv_bfloat16 const*, __nv_bfloat16*, __nv_bfloat16*, float const*, float const*, __nv_bfloat16 const*, float const*, __nv_bfloat16*, void*, long, float*, int*, float*, float*, float const*, float const*, float const*, float const*, float const*) | 1.0 | 14.468 | 0.226 |
| 96 | total | 1.0 | 112.602 | 1.173 |
| 96 | ampere_bf16_s16816gemm_bf16_64x128_ldg8_f2f_stages_32x6_tn | 1.0 | 112.602 | 1.173 |
| 128 | total | 1.0 | 112.937 | 0.882 |
| 128 | ampere_bf16_s16816gemm_bf16_64x128_ldg8_f2f_stages_32x6_tn | 1.0 | 112.937 | 0.882 |
| 192 | total | 1.0 | 222.506 | 1.159 |
| 192 | ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_32x3_tn | 1.0 | 222.506 | 1.159 |
| 256 | total | 1.0 | 208.805 | 0.816 |
| 256 | ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_tn | 1.0 | 208.805 | 0.816 |
| 384 | total | 1.0 | 345.093 | 0.899 |
| 384 | ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_tn | 1.0 | 345.093 | 0.899 |
| 512 | total | 1.0 | 403.403 | 0.788 |
| 512 | ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_tn | 1.0 | 403.403 | 0.788 |

## down: Moonlight-16B-A3B weight (2048, 1408) on NVIDIA GeForce RTX 3050 Laptop GPU sm_86 4 GiB, torch 2.14.1+cu130

| M | kernel | calls/reps | µs/call | µs/row |
|---|---|---|---|---|
| 1 | total | 1.0 | 34.107 | 34.107 |
| 1 | void gemv2T_kernel_val<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, 128, 16, 4, 4, false, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16>, float>, float, float) | 1.0 | 34.107 | 34.107 |
| 2 | total | 1.0 | 40.116 | 20.058 |
| 2 | ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn | 1.0 | 40.116 | 20.058 |
| 4 | total | 1.0 | 40.414 | 10.104 |
| 4 | ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn | 1.0 | 40.414 | 10.104 |
| 6 | total | 1.0 | 40.647 | 6.774 |
| 6 | ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn | 1.0 | 40.647 | 6.774 |
| 8 | total | 1.0 | 40.682 | 5.085 |
| 8 | ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn | 1.0 | 40.682 | 5.085 |
| 12 | total | 1.0 | 40.955 | 3.413 |
| 12 | ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn | 1.0 | 40.955 | 3.413 |
| 16 | total | 1.0 | 41.205 | 2.575 |
| 16 | ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn | 1.0 | 41.205 | 2.575 |
| 17 | total | 1.0 | 50.649 | 2.979 |
| 17 | void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_64x6_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_64x6_tn_align8::Params) | 1.0 | 46.202 | 2.718 |
| 17 | void cublasLt::splitKreduce_kernel<32, 16, int, __nv_bfloat16, __nv_bfloat16, float, __nv_bfloat16, false, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, true, false, false, false>(cublasLt::cublasSplitKParams<float>, __nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*, __nv_bfloat16*, float const*, float const*, __nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*, void*, long, float*, int*, float*, float*, float const*, float const*, float const*, float const*, float const*) | 1.0 | 4.447 | 0.262 |
| 24 | total | 1.0 | 51.500 | 2.146 |
| 24 | void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_64x6_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_64x6_tn_align8::Params) | 1.0 | 46.775 | 1.949 |
| 24 | void cublasLt::splitKreduce_kernel<32, 16, int, __nv_bfloat16, __nv_bfloat16, float, __nv_bfloat16, false, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, true, false, false, false>(cublasLt::cublasSplitKParams<float>, __nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*, __nv_bfloat16*, float const*, float const*, __nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*, void*, long, float*, int*, float*, float*, float const*, float const*, float const*, float const*, float const*) | 1.0 | 4.725 | 0.197 |
| 31 | total | 1.0 | 52.431 | 1.691 |
| 31 | void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_64x6_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_64x6_tn_align8::Params) | 1.0 | 47.469 | 1.531 |
| 31 | void cublasLt::splitKreduce_kernel<32, 16, int, __nv_bfloat16, __nv_bfloat16, float, __nv_bfloat16, false, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, true, false, false, false>(cublasLt::cublasSplitKParams<float>, __nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*, __nv_bfloat16*, float const*, float const*, __nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*, void*, long, float*, int*, float*, float*, float const*, float const*, float const*, float const*, float const*) | 1.0 | 4.962 | 0.160 |
| 32 | total | 1.0 | 37.867 | 1.183 |
| 32 | ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn | 1.0 | 37.867 | 1.183 |
| 40 | total | 1.0 | 38.258 | 0.956 |
| 40 | void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8::Params) | 1.0 | 38.258 | 0.956 |
| 48 | total | 1.0 | 38.554 | 0.803 |
| 48 | void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8::Params) | 1.0 | 38.554 | 0.803 |
| 64 | total | 1.0 | 39.263 | 0.613 |
| 64 | void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x3_tn_align8::Params) | 1.0 | 39.263 | 0.613 |
| 96 | total | 1.0 | 54.310 | 0.566 |
| 96 | ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_32x6_tn | 1.0 | 54.310 | 0.566 |
| 128 | total | 1.0 | 54.425 | 0.425 |
| 128 | ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_32x6_tn | 1.0 | 54.425 | 0.425 |
| 192 | total | 1.0 | 108.608 | 0.566 |
| 192 | ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_tn | 1.0 | 108.608 | 0.566 |
| 256 | total | 1.0 | 115.958 | 0.453 |
| 256 | ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_tn | 1.0 | 115.958 | 0.453 |
| 384 | total | 1.0 | 198.100 | 0.516 |
| 384 | void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8::Params) | 1.0 | 198.100 | 0.516 |
| 512 | total | 1.0 | 198.935 | 0.389 |
| 512 | void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8::Params) | 1.0 | 198.935 | 0.389 |

