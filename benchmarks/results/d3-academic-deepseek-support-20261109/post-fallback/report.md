# Post-fallback DeepSeek trainer smoke

Date: 2026-11-09
Checkpoint: `ByteDance-Seed/academic-ds-9B`, revision `2a205727f616995ad4aba57806f8e2d0d2f28743`
GPU: NVIDIA RTX 3090 24 GB
Software: Python 3.10.12, PyTorch 2.11.0+cu130, CUDA runtime 13.0
Harness source SHA-256: `9c1a029037d317434b94d5f4273eb2cc8e7a6b6d9f4202c9dd76161301824a4a`

## Purpose

Verify the runtime fallback selected after the B14 row check: for the bounded
academic-ds-9B recipe, `backend=grouped` must use grouped expert restore with
native eager per-expert GEMMs, without constructing GroupedGemm or requiring its
cuBLAS workspace. Compare it with `backend=sequential` on the exact same source,
checkpoint, synthetic batch, seed and device.

## Results

| Codec | Backend | Loss | Updated trainable tensors | Grouped matrices | Updated adapter SHA-256 |
|---|---|---:|---:|---:|---|
| NF4 | grouped restore fallback | 0.9548282623291016 | 160/160 | 0 | `8300086541357601ecf2911948db9f8954626c50244cbec6f96d172037dfee13` |
| NF4 | sequential control | 0.9548282623291016 | 160/160 | 0 | `8300086541357601ecf2911948db9f8954626c50244cbec6f96d172037dfee13` |
| Nested NF4 | grouped restore fallback | 0.9574334025382996 | 160/160 | 0 | `f53ba93be41cbb7fab592c57e7294bff584b345ae819d0212f6ae705c840821f` |
| Nested NF4 | sequential control | 0.9574334025382996 | 160/160 | 0 | `f53ba93be41cbb7fab592c57e7294bff584b345ae819d0212f6ae705c840821f` |

All four runs used 512 synthetic tokens, seed 102 and one SGD step. Each began
from adapter hash `66cf16f48ca78034af6d3ed34695eebb6c1a3bcc99b7c975a49868f55509cfa0`.
The grouped runs report `expert_compute_backend=grouped-restore-eager` and
`expert_gemm=null`; sequential controls report `native-eager`. Losses and saved
adapter bytes match exactly within each codec. The runner metadata for every arm
is retained beside this report.

## Limits

This is a one-step integration/equivalence smoke for one checkpoint, synthetic
input, one GPU and the test-only eager-attention path. It does not measure speed,
physical disk I/O, convergence, production data, the L2L workflow, or general
DeepSeek compatibility. The older four-run result in the parent support report
used the pre-fallback grouped-GEMM implementation and remains preserved as
historical evidence; its grouped-vs-sequential match does not validate GEMM at
the disputed B14 row counts.
