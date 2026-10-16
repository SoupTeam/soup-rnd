"""Bench-only WMMA hypothesis, compiled with the installed CUDA NVRTC library."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

_CACHE = {}

SOURCE = r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
using namespace nvcuda;
using scalar = SCALAR;
extern "C" __global__ void fused(
    const scalar* x, const unsigned char* packed, const void* absmax,
    const float* code, const float* absmax2, const float* code2, const float* offset,
    scalar* output, int M, int N, int K, int blocksize, int nested, int block2, int transpose) {
  __shared__ __align__(32) scalar A[TILE_SIZE*REDUCE_TILE];
  __shared__ __align__(32) scalar B[TILE_SIZE*REDUCE_TILE];
  __shared__ __align__(32) float C[TILE_SIZE*TILE_SIZE];
  int outsize = transpose ? K : N;
  int reduce = transpose ? N : K;
  int outbase = blockIdx.x * TILE_SIZE, mbase = blockIdx.y * TILE_SIZE;
  int warp = threadIdx.x / 32;
  int warp_out = warp % (TILE_SIZE/16), warp_m = warp / (TILE_SIZE/16);
  wmma::fragment<wmma::matrix_a,16,16,16,scalar,wmma::row_major> a;
  wmma::fragment<wmma::matrix_b,16,16,16,scalar,wmma::col_major> b;
  wmma::fragment<wmma::accumulator,16,16,16,float> c;
  wmma::fill_fragment(c,0.0f);
  for (int start=0; start<reduce; start+=REDUCE_TILE) {
    for (int q=threadIdx.x; q<TILE_SIZE*REDUCE_TILE; q+=blockDim.x) {
      // Decode contiguous original W rows, then transpose in shared memory.
      int i=transpose ? q%TILE_SIZE : q/REDUCE_TILE;
      int j=transpose ? q/TILE_SIZE : q%REDUCE_TILE;
      int out=outbase+i, inner=start+j;
      float weight=0.0f;
      if (out<outsize && inner<reduce) {
        long long pos=transpose ? (long long)inner*K+out : (long long)out*K+inner;
        int byte=packed[pos/2];
        int nibble=(pos%2==0) ? byte>>4 : byte&15;
        long long scale=pos/BLOCKSIZE;
        float magnitude;
        if (nested) {
          magnitude=code2[((const unsigned char*)absmax)[scale]]*absmax2[scale/BLOCK2];
          magnitude=magnitude+offset[0];
        } else magnitude=((const float*)absmax)[scale];
        weight=code[nibble]*magnitude;
      }
      A[i*REDUCE_TILE+j]=CONVERT(weight);
      int row=mbase+q/REDUCE_TILE, red=start+q%REDUCE_TILE;
      B[q]=(row<M && red<reduce) ? x[(long long)row*reduce+red] : CONVERT(0.0f);
    }
    __syncthreads();
    for (int inner=0; inner<REDUCE_TILE; inner+=16) {
      wmma::load_matrix_sync(a,A+warp_out*16*REDUCE_TILE+inner,REDUCE_TILE);
      wmma::load_matrix_sync(b,B+warp_m*16*REDUCE_TILE+inner,REDUCE_TILE);
      wmma::mma_sync(c,a,b,c);
    }
    __syncthreads();
  }
  wmma::store_matrix_sync(C+warp_m*16*TILE_SIZE+warp_out*16,c,TILE_SIZE,wmma::mem_col_major);
  __syncthreads();
  for (int q=threadIdx.x; q<TILE_SIZE*TILE_SIZE; q+=blockDim.x) {
    int row=mbase+q/TILE_SIZE, col=outbase+q%TILE_SIZE;
    if (row<M && col<outsize) output[(long long)row*outsize+col]=CONVERT(C[q]);
  }
}
"""


def check(library: ctypes.CDLL, function: str, *args: Any) -> None:
    code = getattr(library, function)(*args)
    if code:
        raise RuntimeError(f"{function} failed: {code}")


def compile_kernel(
    dtype: Any,
    blocksize: int = 64,
    block2: int = 256,
    *,
    tile: int = 16,
    reduction_tile: int = 16,
) -> tuple[Any, ...]:
    import torch

    if blocksize <= 0 or block2 < 0:
        raise ValueError("invalid NF4 scale block sizes")
    if tile not in (16, 32, 64) or reduction_tile not in (16, 32, 64, 128):
        raise ValueError("invalid research WMMA tile")
    key = (torch.cuda.current_device(), dtype, int(blocksize), int(block2), tile, reduction_tile)
    if key in _CACHE:
        return _CACHE[key]
    packages = Path(torch.__file__).parents[1]
    nvrtc_path = next(packages.glob("nvidia/cuda_nvrtc/lib/libnvrtc.so*"))
    nvrtc = ctypes.CDLL(str(nvrtc_path))
    driver = ctypes.CDLL("libcuda.so.1")
    program = ctypes.c_void_p()
    scalar, conversion = (
        ("__half", "__float2half_rn")
        if dtype == torch.float16
        else ("__nv_bfloat16", "__float2bfloat16_rn")
    )
    source = (
        SOURCE.replace("SCALAR", scalar)
        .replace("CONVERT", conversion)
        .replace("BLOCKSIZE", str(int(blocksize)))
        .replace("BLOCK2", str(max(1, int(block2))))
        .replace("TILE_SIZE", str(tile))
        .replace("REDUCE_TILE", str(reduction_tile))
        .encode()
    )
    check(
        nvrtc,
        "nvrtcCreateProgram",
        ctypes.byref(program),
        ctypes.c_char_p(source),
        ctypes.c_char_p(b"fused.cu"),
        0,
        None,
        None,
    )
    major, minor = torch.cuda.get_device_capability()
    headers = [
        packages / "nvidia/cuda_runtime/include",
        packages / "triton/backends/nvidia/include",
        packages / "nvidia/cuda_cccl/include",
    ]
    options = [
        b"--std=c++17",
        b"--fmad=false",
        f"--gpu-architecture=compute_{major}{minor}".encode(),
    ]
    options += [("--include-path=" + str(p)).encode() for p in headers if p.exists()]
    arguments = (ctypes.c_char_p * len(options))(*options)
    status = nvrtc.nvrtcCompileProgram(program, len(options), arguments)
    size = ctypes.c_size_t()
    check(nvrtc, "nvrtcGetProgramLogSize", program, ctypes.byref(size))
    log = ctypes.create_string_buffer(size.value)
    check(nvrtc, "nvrtcGetProgramLog", program, log)
    if status:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(program))
        raise RuntimeError("NVRTC compile failed: " + log.value.decode())
    check(nvrtc, "nvrtcGetPTXSize", program, ctypes.byref(size))
    ptx = ctypes.create_string_buffer(size.value)
    check(nvrtc, "nvrtcGetPTX", program, ptx)
    check(nvrtc, "nvrtcDestroyProgram", ctypes.byref(program))
    module, function = ctypes.c_void_p(), ctypes.c_void_p()
    check(driver, "cuModuleLoadDataEx", ctypes.byref(module), ptx, 0, None, None)
    check(driver, "cuModuleGetFunction", ctypes.byref(function), module, ctypes.c_char_p(b"fused"))
    _CACHE[key] = (driver, module, function, ptx.raw, log.value.decode())
    return _CACHE[key]


def matmul(
    value: Any,
    saved: tuple,
    metadata: tuple,
    transpose: bool,
    *,
    tile: int = 16,
    reduction_tile: int = 16,
) -> Any:
    import torch

    if value.device.type != "cuda" or value.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("WMMA experiment requires CUDA FP16/BF16")
    n, k, blocksize, nested, block2, weight_dtype = metadata
    if weight_dtype != value.dtype:
        raise ValueError("WMMA experiment requires identical input and weight dtypes")
    value = value.contiguous()
    reduce, outsize = (n, k) if transpose else (k, n)
    if value.shape[-1] != reduce:
        raise ValueError("WMMA dimension mismatch")
    m = value.numel() // reduce
    result = torch.empty((*value.shape[:-1], outsize), device=value.device, dtype=value.dtype)
    driver, _module, function, _ptx, _log = compile_kernel(
        value.dtype, blocksize, block2, tile=tile, reduction_tile=reduction_tile
    )
    pointers = [ctypes.c_void_p(t.data_ptr()) for t in (value, *saved, result)]
    scalars = [ctypes.c_int(i) for i in (m, n, k, blocksize, int(nested), block2, int(transpose))]
    parameters = (ctypes.c_void_p * 15)(
        *(ctypes.cast(ctypes.byref(arg), ctypes.c_void_p) for arg in pointers + scalars)
    )
    check(
        driver,
        "cuLaunchKernel",
        function,
        (outsize + tile - 1) // tile,
        (m + tile - 1) // tile,
        1,
        32 * (tile // 16) ** 2,
        1,
        1,
        0,
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        parameters,
        None,
    )
    return result
