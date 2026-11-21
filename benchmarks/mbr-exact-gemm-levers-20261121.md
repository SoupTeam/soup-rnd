# Exact expert GEMM and host levers on Kanana, RTX 4060 Ti (ticket 02, D1 gates)

Date 2026-11-21. Kanana-2-30B-A3B NF4+BF16 on a Vast RTX 4060 Ti 16 GiB with the allocator limited to 8 GiB, torch 2.14.1+cu130, MBPP-train text, one block C-E-F-G-G-F-E-C (two runs per arm and window, 3 steps, median after warm-up) in normal mode; bits in deterministic mode (`use_deterministic_algorithms(True)`). Raw rows are in `.scratch/mbr-unified/results/vast-4060ti-levers-20261121` (outside git).

Arms, each on the one before: **C** = route-preserving MBR + ordered backward (the baseline of ticket 01), **E** = C + `lt-fused`, **F** = E + `ExpertBufferPool.async_lease`, **G** = F + `StreamedExperts.fast_row_gather`.

## Bits against A0 (gradient digest, deterministic mode, S=64)

| k | A0 | C | E | F | G |
|---|---|---|---|---|---|
| 1 | `309caf99` | equal | equal | equal | equal |
| 2 | `8b73cd25` | **differs** (`42e6bf8d`) | equal | **differs** (`b6d38a10`) | equal |
| 4 | `a3d3709a` | **differs** (`bea4d019`) | equal | equal | equal |
| 8 | `94f79a58` | **differs** (`573cfcd1`) | equal | equal | equal |

Repeat of k=2 (3 runs each, same setting): F, E and A0 all gave `8b73cd25` every time. F differed in the first run only (1 of 4 runs).

## Speed and peak VRAM (normal mode)

| k | S | arm | step s | alloc GiB | reserved GiB | NVML GiB | status |
|---|---|---|---|---|---|---|---|
| 8 | 64 | C | 33.0 | 1.10 | 1.16 | 1.35 | ok |
| 8 | 64 | E | 32.8 | 1.14 | 1.21 | 1.40 | ok |
| 8 | 64 | F | 31.5 | 1.14 | 1.21 | 1.40 | ok |
| 8 | 64 | G | 31.2 | 1.14 | 1.21 | 1.40 | ok |
| 8 | 512 | C | 35.3 | 2.03 | 2.12 | 2.31 | ok |
| 8 | 512 | E | 36.1 | 3.14 | 3.96 | 4.15 | ok |
| 8 | 512 | F | 32.8 | 3.14 | 3.96 | 4.15 | ok |
| 8 | 512 | G | 32.8 | 3.14 | 3.96 | 4.15 | ok |
| 64 | 64 | C | 54.1 | 3.72 | 4.03 | 4.22 | ok |
| 64 | 64 | E | 63.9 | 3.83 | 4.21 | 4.40 | ok |
| 64 | 64 | F | 60.9 | 3.83 | 4.21 | 4.40 | ok |
| 64 | 64 | G | 60.7 | 3.83 | 4.21 | 4.39 | ok |
| 64 | 512 | C | - | 7.98 | 8.00 | 3.68 | OOM |
| 64 | 512 | E | - | 7.97 | 7.99 | 8.18 | OOM |
| 64 | 512 | F | - | 7.97 | 7.99 | 5.31 | OOM |
| 64 | 512 | G | - | 7.97 | 7.99 | 5.33 | OOM |

## Gates (`lever_gate`, speed needs 3% in 3 of 4 windows; t-test not computed with one block)

| lever | bits | VRAM | speed gain by window (8,64 / 8,512 / 64,64) | outcome by the ticket rule |
|---|---|---|---|---|
| E vs C | pass | fail | +0.6% / -2.3% / -18.1% | bits kept, no gain, more VRAM: dropped unless kept for the bits |
| F vs E | fail at k=[2] | pass | +4.2% / +9.2% / +4.6% | bits flaky at k=2: off by default, own row |
| G vs F | pass | pass | +0.7% / +0.0% / +0.3% | bits kept, no gain: dropped |

Window k=64, S=512 is OOM at 8 GiB for every arm, so no lever removes an OOM.

## Notes

- The first attempt of the GPU tests segfaulted on this host: a CUDA 12.6 toolkit beside a cu130 torch linked the wrong cuBLASLt. The toolchain check now reports a cuBLAS major that differs from torch's, and the build supports the nvidia pip wheels (cccl headers, `libcublasLt.so.13`, nvvm and crt pinned to the nvcc version).
- `test_a0_exact_lt_kernels_gpu.py`: 35 passed on sm_89, including the PyTorch 2.14 series check.
- One block instead of five (user decision for ticket 01): the t>2.78 criterion is not computed, so the speed verdicts are provisional.
- E uses more memory (S=512: 3.14 against 2.03 GiB allocated) and is slower at k=64 (-18%); the cause is not investigated.
