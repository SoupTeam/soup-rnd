"""Bench-only WMMA hypothesis, compiled with the installed CUDA NVRTC library."""

from __future__ import annotations

import ctypes
from pathlib import Path

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
  __shared__ __align__(32) scalar A[256];
  __shared__ __align__(32) scalar B[256];
  __shared__ __align__(32) float C[256];
  int outsize = transpose ? K : N;
  int reduce = transpose ? N : K;
  int outbase = blockIdx.x * 16, mbase = blockIdx.y * 16;
  wmma::fragment<wmma::matrix_a,16,16,16,scalar,wmma::row_major> a;
  wmma::fragment<wmma::matrix_b,16,16,16,scalar,wmma::col_major> b;
  wmma::fragment<wmma::accumulator,16,16,16,float> c;
  wmma::fill_fragment(c,0.0f);
  for (int start=0; start<reduce; start+=16) {
    for (int q=threadIdx.x; q<256; q+=32) {
      int i=q/16, j=q%16, out=outbase+i, inner=start+j;
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
      A[q]=CONVERT(weight);
      B[q]=(mbase+i<M && inner<reduce) ? x[(long long)(mbase+i)*reduce+inner] : CONVERT(0.0f);
    }
    __syncthreads();
    wmma::load_matrix_sync(a,A,16);
    wmma::load_matrix_sync(b,B,16);
    wmma::mma_sync(c,a,b,c);
    __syncthreads();
  }
  wmma::store_matrix_sync(C,c,16,wmma::mem_col_major);
  __syncthreads();
  for (int q=threadIdx.x; q<256; q+=32) {
    int row=mbase+q/16, col=outbase+q%16;
    if (row<M && col<outsize) output[(long long)row*outsize+col]=CONVERT(C[q]);
  }
}
"""


def check(library, function, *args):
    code = getattr(library, function)(*args)
    if code:
        raise RuntimeError(f"{function} failed: {code}")


def compile_kernel(dtype, blocksize=64, block2=256):
    import torch

    if blocksize <= 0 or block2 < 0:
        raise ValueError("invalid NF4 scale block sizes")
    key = (torch.cuda.current_device(), dtype, int(blocksize), int(block2))
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


def matmul(value, saved, metadata, transpose):
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
    driver, _module, function, _ptx, _log = compile_kernel(value.dtype, blocksize, block2)
    pointers = [ctypes.c_void_p(t.data_ptr()) for t in (value, *saved, result)]
    scalars = [ctypes.c_int(i) for i in (m, n, k, blocksize, int(nested), block2, int(transpose))]
    parameters = (ctypes.c_void_p * 15)(
        *(ctypes.cast(ctypes.byref(arg), ctypes.c_void_p) for arg in pointers + scalars)
    )
    check(
        driver,
        "cuLaunchKernel",
        function,
        (outsize + 15) // 16,
        (m + 15) // 16,
        1,
        32,
        1,
        1,
        0,
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        parameters,
        None,
    )
    return result
