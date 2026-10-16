# D1 amendment G: diagnose a separate A100 environment before the full grid

Dated 2026-10-15, committed before any A100 CUDA fixture or timing. RTX 3090
cohort F completed its ablation and QA comparisons, each with five valid ABBA
blocks. The QA clock range was 1860-1950 MHz (ratio 1.048387), and its 20
complete training runs remain attributable to source ca1c5b00. Its first
FP16 M=1, N=K=4096 dense/native projection comparison spanned 1755-1980 MHz
(ratio 1.128205) and was rejected. All 20 projection attempts stay rejected;
the controller stopped before the remaining grid. No sample filtering or
clock normalization repairs them.

Every E/F raw file was downloaded and hash-verified, including void arms,
preludes and failures: 758 files, 1,722,061,867 logical bytes. Lossless archive
SHA-256 22b1c446ea9ef9fddcc117525c39bb53540d058018af5aa71480ccdd9c47e990.
Rental 56073519 was deleted and verified absent. Observed cumulative D1
charges are $1.01, subject to provider settlement; the user cap remains $10.

A VM alternative was not allocated. The team service key cannot create
personal SSH keys, and VM creation requires one before allocation. The
browser tool failed during initialization, before any UI interaction. No
account SSH setting changed. Failed API calls and candidate offers remain
in the local control ledger. Do not treat VM access or clock locking as verified.

## Fixed diagnostic cohort

Try one separately bounded A100 SXM4 rental at <=$0.60/hour including 100 GB
disk. Keep the pinned torch 2.11.0 CUDA 12.6 / bitsandbytes 0.50.2 /
transformers 5.17.0 / peft 0.20.0 stack and retain the full resolved environment.
Pin the physical GPU UUID explicitly, recording original visibility variables
without exporting any credential. Do not pool its timings with RTX 3090.

First run the large-alias/streaming fixtures in FP16 and BF16 and the existing
#842/#968 GPU tests. Then run ONE fresh-process ABBA block of each:

- BF16 M=1, N=K=4096, dense/native;
- BF16 M=2048, N=K=4096, native/forced.

For these diagnostic projection arms, use >=20 steps AND 5 seconds warm-up,
then 200 synchronized timed steps, retaining split forward/backward and combined
events, both allocator peaks, dense GEMM ceiling and all raw telemetry.
Use the unchanged original <=5% actual timed-work clock guard and unchanged
F1 sensors, UUID, process, numerical and environment checks. A one-block
screen has no performance verdict. Its purpose is to decide whether the
candidate hardware can satisfy the validity requirements before investing
in the full grid, not to choose a winning implementation or drop bad timings.

Record application/default/actual clock settings and attempt the supported
clock-lock command only on this dedicated rental's visible GPU. A denial is
retained; it cannot be reinterpreted as a locked clock. If factory/provider
clocks alone pass the diagnostic, any complete study still requires a fresh
committed rule, source cohort, all 96 cases and repeated QA/ablation on that GPU.
If they fail, stop instead of relaxing the guard. A successful lock must be
reset before release. Independent review and the cumulative budget stay required.
