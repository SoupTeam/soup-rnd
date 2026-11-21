# lt-fused diagnosis: where we stopped (2026-11-21)

Branch `mbr-unified`, ticket 02 (exact expert GEMM into `src`). The D1 gates are in
[mbr-exact-gemm-levers-20261121.md](mbr-exact-gemm-levers-20261121.md): `lt-fused` (arm E) is
bit-equal to A0 at k = 1, 2, 4, 8, but not faster than the ordered arm C, and on Kanana
(RTX 4060 Ti, 8 GiB allocator limit) it holds more memory (S = 512: 3.14 against 2.03 GiB
allocated) and is 18% slower at k = 64, S = 64.

## Suspects

1. **Memory.** `FusedExpert` cached a rows x K input per row count. Fixed in commit 73d72016
   (test `test_fused_expert_does_not_keep_a_sample_tensor_per_row_count`); not yet measured on a GPU.
2. **k = 64 slowdown.** Hypothesis: segments of one row (GEMV) have no A0-exact cuBLASLt algo, so
   `FusedExpert.ensure` returns False and `direct_expert_backward` runs the Python path
   (`_python_forward`, `expert_vjp`) for the whole visit. Measurement rows of the lt-fused arms now
   carry `fused_visits` (`fused_forward`, `python_forward`, ...) and `algo_table` stats.

## What to run on a small CUDA card (sm_80+, PyTorch 2.14, CUDA build tools)

```
export CUDA_HOME=$(python -c "import nvidia,os;print(os.path.join(list(nvidia.__path__)[0],'cu13'))")
PYTHONPATH=src:. python benchmarks/harness/lever_tiny_smoke.py --out out 2>&1 | tee smoke.log
```

It runs arms A0, C, E, F, G on a tiny synthetic MoE. Look at the bit verdicts and at
`fused=` in the E, F, G lines: `python_forward` above zero confirms suspect 2.

Pip packages for the build (they must match torch's CUDA major, here 13): `nvidia-cuda-nvcc`,
`nvidia-cuda-cccl`, `nvidia-nvvm`, `nvidia-cuda-crt` all `==13.0.*`, and `ninja`. A system CUDA 12
toolkit beside a cu130 torch links the wrong cuBLASLt and segfaults; `toolchain_error()` now says so.

## Result on RTX 3050 Laptop (sm_86, 3.7 GiB, tiny MoE, 2026-11-21)

Environment: torch 2.14.1+cu130, `toolchain_error` None. The GPU kernel tests
(`test_a0_exact_lt_kernels_gpu.py`) pass, 35 of 35.

- Bits: arms C, E, F, G are `equal` to A0 at k = 1, 2, 4, 8.
- Suspect 2 is not seen here. `python_forward` and `python_backward` are 0 for E, F, G at every k
  up to 64 and S up to 512. The tiny model does not reproduce the slowdown: at k = 64, S = 512 the
  step is 1.51 s for C and 1.54 s for E. This run neither confirms nor rules out the hypothesis
  for Kanana.
- Suspect 1 shows in allocation. E allocates more than C in every window: 121 against 104 MiB
  (S = 64, k = 8), 335 against 317 MiB (S = 512, k = 8), 364 against 331 MiB (S = 512, k = 64).
  The gap is 17 to 33 MiB on the tiny model. Re-measure on Kanana.
- Step times here (0.04 to 1.5 s) come from a tiny model and are not speed verdicts.

## Open

- `async_lease` (arm F): +4 to +9% speed, but 1 of 4 k = 2 runs differed from A0 (3 repeats equal).
- Is `lt-fused` usable below sm_80 (T4 is sm_75)? The algo table is built on the card at run time,
  so it may adapt; the sm_80 guard in `expert_gemm.require_lt_fused` is conservative, not proven.
- One block instead of five: the speed verdicts are provisional.
