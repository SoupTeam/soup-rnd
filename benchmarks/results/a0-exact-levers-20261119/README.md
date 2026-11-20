# A0-exact levers: results on RTX 3060 (2026-11-19 and 2026-11-20)

**Final report:** [`benchmarks/mbr-exact-speed-20261120.md`](../../mbr-exact-speed-20261120.md). The exact
MBR (bit-equal to L2L A0) is 75% faster per step than A0 and 5.6% (fast-CPU host) and 9.1% (slow-CPU
host) faster than stock route-preserving MBR (B0); adapters after 40 deterministic training steps are
bit-equal to A0 and across both hosts.

What is tracked in git here (the working folder held about 500 MB; the rest stays outside git):

| Path | What it is |
|---|---|
| `abc/utah-final/abc_report.json`, `abc/norway-final-report.json` | A-B-C-C-B-A reports: per-run times, fetches, logical bytes, validity, paired differences |
| `f1d/<host>/b0-ltfo*/f1_report.json` | two-arm F1 reports (A-B-B-A blocks) before `async_lease` |
| `train/<host>-det/{A,B,C}.json` | 40-step deterministic training per arm: loss, adapter digest and expert fetches per step, held-out NLL per example (`deterministic_note` says the field was annotated after the run) |
| `out/` | text outputs of the micro tests and profiles quoted in the report |
| `*.sh` | launch scripts of every run in the report (`abc-run.sh`, `train-run.sh`, `gradcmp.sh`, `lease.sh`, `inject-sweep.sh`, `causal.sh`, `diag-all2.sh`, `gpu-windows.sh`, `h12.sh`, ...) |

Not tracked: py-spy and Chrome traces, gradient and adapter tensors (`*.pt`), the scratch Python scripts
that produced the micro outputs and tables (`micro_*.py`, `*_analyze.py`, `abc_peek.py`, ...). The
harness modules they relied on are in `benchmarks/harness/`.

# Micro results of 2026-11-19 (A0-exact levers, step 1)

Goal: cut B0s' +43% step cost vs B0 while staying bit-equal to A0 (per-segment `cublasGemmEx`).
Plan and sources: `.scratch/mbr-triton/research-a0-exact.md` (not tracked).

Host: vast.ai instance 61445434, RTX 3060 (driver 615.71), Utah, the same host type as the A0/B0s
series. torch 2.14.1+cu130, Triton 3.8.0, nvmath-python 1.0.1 (cuda-bindings upgraded to 13.5).
One Moonlight expert, synthetic weights, four GEMM forms (fwd gate_up, fwd down, grad_x down,
grad_x gate_up). Microbenchmarks only, no training step.

| Script | Output |
|---|---|
| `micro_kernel_map.py` | `out/micro_kernel_map.out` |
| `micro_lt_forced.py` | `out/micro_lt_forced.out` |
| `micro_lt_v2.py` | `out/micro_lt_v2.out` |

## Findings

1. **Rows do not depend on neighbours inside one kernel.** Whenever segments a, b and a + b get the
   same cuBLAS kernel set, the a + b call gives the same bits per row: 1359 of 1359 pairs over the
   four forms, including sliced-K and split-K kernels.
2. **Kernel bands.** M = 2..16 is one kernel in fwd gate_up, grad_x down and grad_x gate_up; fwd down
   has 2..4 and 5..16. M = 1 is a GEMV (forward) or a different split-K kernel (backward). Above 16
   the heuristic alternates between 2-3 kernels.
3. **cuBLASLt reproduces A0.** For every form and every M = 1..64, one of the Lt heuristic algos
   (nvmath, operands `x`, `W.T`) gives exactly the A0 bits (256 of 256). torch's own
   `preferred_blas_library("cublaslt")` does not (heuristic picks another algo for M 2..16 and others).
4. **Merged call with a forced A0 algo keeps the bits.** Buckets keyed by the full serialized algo,
   40 groups drawn like a real step: all slices equal A0 in all four forms. Algos for M = 1 do not
   accept larger n, so 1-row segments stay separate; batched 1-row calls do not match A0 reliably.
5. **Speed is limited by the prototype.** Calls per group drop from 6.7 to 4.1-4.75; GPU time per
   group: gate_up forms 0.29 -> 0.21 ms and 0.34 -> 0.25 ms, down forms equal or slightly slower
   (B0 is 0.05-0.11 ms). nvmath adds Python overhead per call, and the synthetic sizes have more
   rows (mean 119) than the real step (mean 51).
6. **Multi-stream** keeps the bits (all equal) and gains 10-15%.
7. **Triton** fixed-tile kernel equals cuBLAS for 48 of 64 M in fwd gate_up (not checked which M;
   likely the non-sliced 128x64 kernels) and is M-invariant 64/64.

## Extension run (`micro_lt_ext.py`, `out/micro_lt_ext.out`)

Instance 61456255, RTX 3060 (driver 615.71), Norway, image `vastai/base-image:cuda-13.2-mini-py312`,
torch 2.14.1+cu130, extension built against the `nvidia` wheels (headers from `nvidia-cuda-nvcc`
and `nvidia-cuda-cccl` 13.0). One expert, forward + backward, 60 groups (6.4 segments, 7 rows
per segment on average).

- `torch.mm(g, W)` equals the autograd input gradient of `F.linear` for M = 1..64 (both weights).
- An A0-exact Lt algo exists for every M = 1..64 in all four forms; 4-5 distinct algos per form.
- lt-merged equals segmented bit for bit (out and grad_hidden) on all groups; 2 merged sizes were
  refused by Lt and ran per segment.
- Direct backward (H3, same GEMMs and `aten.silu_backward`, no autograd graph) equals autograd.

| Variant | GPU kernels, ms/group | Wall, ms/group |
|---|---|---|
| segmented (B0s), autograd | 0.97 | 3.29 |
| lt-merged, autograd | 0.51 | 3.85 |
| segmented, direct backward | 1.00 | 1.48 |
| lt-merged, direct backward | 0.51 | 1.87 |
| B0 (one call, not A0-exact), direct | 0.20 | 0.36 |

The micro is CPU-bound (event time equals wall time). Merging halves the GPU kernel time, but the
Python planning per call costs more than it saves; the direct backward removes most of the
autograd overhead.

## Open

- A low-overhead path (C++ extension on torch's Lt handle and workspace) to measure the real gain.
- 1-row segments (the most frequent size) still cost one call each.
- Real segment-size distribution per form instead of the synthetic one.

## Moonlight step (RTX 3060, Utah host), 2026-11-19

Folders: `moonlight-series/` (Python merge), `moonlight-cpp/` (C++ merge v1 + phase profiles),
`moonlight-cpp2/` (C++ merge with one gather/one assemble cat). Script: `remote-f1-direct.sh`.

Bits: every exact arm gave the A0 gradient digest `d0a663ace110` (216 LoRA tensors equal);
B0 differs (`23a4369d4773`), so the check can go red.

| B against B0 (F1 pilot, 5 blocks, 0 void, suspend/power ok) | Bits = A0 | Mean diff | t |
|---|---|---|---|
| B0s (earlier F1) | yes | +43% | sig. |
| B0s + direct backward (`b0-b0sd`) | yes | +37.8% | 38.4 |
| lt-merged (Python) + direct backward (`b0-ltd`) | yes | +27.3% | 31.8 |
| lt-merged (C++ v1) + direct backward | yes | +31.6% | 31.1 |
| lt-merged (C++, cat gather/assemble) + direct backward | yes | +28.7% | 34.4 |

`gpu_clock` was not reviewed by hand (the F1 rule asks for it). Runs of different rows are on
different rentals, so differences of a few points between rows are within session variance.

Phase profile (synchronised, seconds per step): B0 vs Bltd (C++ v1) differ by +1.04 in the expert
backward GEMM and +0.72 in the expert forward GEMM; everything else is equal. Counters for one
process: 9 432 expert GEMM calls, 6.3 segments and 2.5 merged groups per call on average, about
32 us Python lookup and 135 us in the C++ call per call (v1). Remaining gap is CPU work per
expert call; the 10% threshold is not met by any A0-exact arm.
