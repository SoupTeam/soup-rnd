# B14 grouped-GEMM row check against the current D3 worktree

Date: 2026-11-09
Base commit: `1b4547d40e26fcc6d68785fac53b11bda6a9ab6d`; the measured source includes the uncommitted D3 worktree changes identified by the source hashes in `protocol.json`.
GPU: NVIDIA RTX 3090 24 GB, compute capability 8.6
Software: Python 3.10.12, PyTorch 2.11.0+cu130, CUDA runtime 13.0, Transformers 5.19.0.

## Scope

This reruns only the two disputed B14 routed-row buckets, with exactly 32 or 64 rows per expert. It uses the B14 diagnostic harness, eight synthetic experts with academic-ds-9B projection geometry (`hidden=2048`, `intermediate=1408`, BF16, top-k 1), seed 5 and 20 interleaved timing samples. Arms are sequential D3, grouped restore, and grouped restore plus grouped GEMM. The outputs and gradients are compared with a resident-layer reference.

Grouped GEMM was run under cuBLASLt with both BF16 reduced-precision-reduction flags set to false, including split-K. Both shapes, `(2048, 2816)` and `(1408, 2048)`, were admitted. The harness and relevant source files transferred to the GPU were SHA-256 checked against the local worktree; see `protocol.json`.

## Results

| Rows per expert | Arm | Exact output / hidden grad / route-weight grad | Max abs diff (output, hidden grad, route-weight grad) | CUDA kernels | Median forward+backward (ms) |
|---:|---|---|---|---:|---:|
| 32 | Sequential | yes / yes / yes | 0 / 0 / 0 | 385 | 51.348 |
| 32 | Grouped restore | yes / yes / yes | 0 / 0 / 0 | 385 | 51.092 |
| 32 | Grouped GEMM | **no / yes / no** | 0.0078125 / 0 / 0.0625 | 373 | 52.052 |
| 64 | Sequential | yes / yes / yes | 0 / 0 / 0 | 385 | 52.029 |
| 64 | Grouped restore | yes / yes / yes | 0 / 0 / 0 | 385 | 51.229 |
| 64 | Grouped GEMM | **no / yes / no** | 0.0078125 / 0 / 0.25 | 373 | 52.856 |

At both row counts, grouped GEMM uses 4 GEMM kernels instead of 16, reducing the total layer count by 12 (385 to 373). The 262 backward kernels remain unchanged. Grouped GEMM did not improve the measured end-to-end median here: it was about 1.4% slower than sequential at 32 rows and 1.6% slower at 64 rows. This is a two-bucket, one-GPU diagnostic with synthetic weights and routes, not a speed claim.

## Conclusion

PyTorch 2.11 works for this cuBLASLt grouped-GEMM path with the stated BF16 flags. The non-bitwise result is reproduced qualitatively on the current D3 worktree at both 32 and 64 rows: output and route-weight gradients differ, while hidden gradients match. The exact maximum differences are not identical to the teammate's report (ours are 0.0078125 for output at both sizes; route-weight gradient 0.0625 and 0.25), so this does not establish matching input/seed or reproduce their precise error values. No tolerance was introduced and the grouped-GEMM arm is not accepted as numerically equivalent for these cases.

The first GPU invocation used a stale scratch copy of some D3 modules; it was excluded. This report is from the rerun after transferring the current local source package and verifying its hashes. No model checkpoint, real routing, full-model training, packed-only path, or additional row ranges were tested.

## Runtime decision

The current D3 trainer now keeps academic-ds-9B on grouped restore with native per-expert GEMMs. These two shapes are removed from the generic grouped-GEMM validated-shape allowlist, and D3 preflight no longer requires a cuBLAS handle/workspace for this DeepSeek grouped-restore path. This preserves the bitwise-exact restore path while the grouped-GEMM numerical contract remains unresolved. A post-fallback full-model one-step smoke was subsequently run for both NF4 codecs and matched fresh sequential controls; see `../d3-academic-deepseek-support-20261109/post-fallback/report.md`. This remains a correctness/integration smoke, not a performance result.
