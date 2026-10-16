# D1: bounded arithmetic diagnosis, 2026-10-16

The previous GPU witness decoded every weight exactly but differed from native
bitsandbytes in four forward and ten dX elements. Test accumulation order before
any training or timing. Production dispatch remains unchanged.

`harness/d1_arithmetic_screen.py` declares twelve tile/reduction configurations
before execution. Calibration uses seed 842, FP16, nonnested, M128/N64/K128;
all failed configurations and candidate outputs are retained. Forward and dX
may select different configurations. Selection requires zero differing elements.

If both directions have an exact calibration candidate, validate on seeds 843
and 1977, FP16 and BF16, nested and nonnested scales, shapes (128,64,128),
(133,67,96), (256,256,512). Stop independent validation at its first failure.
Calibration alone never establishes correctness. No tolerance relaxation and
no replacement of the native reference. Native GPU kernel names are collected
with Torch profiler when available; this is diagnosis, not performance evidence.

A separately pinned follow-up tests twelve configurations: chunks of 4/8/16
products, chained/independent accumulation, and both operand orientations.
Masking unused reduction lanes keeps the minimum Triton MMA tile size at 16.
The transposed output formulation mirrors cuBLAS's column-major orientation.
FP16 and BF16 are evaluated independently, so failure of one cannot hide evidence
for the other. Each dtype needs 16 independent cases (two seeds, two nesting
modes, four shapes including M512/N4096/K4096), stopping at its first failure.
Only a fully validated dtype could proceed to checkpoint and full-step gates.

After those configurations failed, the physical profiler identified native
`cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_*_align8` kernels.
The next concrete hypothesis uses CUDA WMMA 16x16x16, with the same FP16/BF16
weight rounding, compiled by the installed NVRTC; no new package download.
Both dtypes get seed 842 calibration followed by the same 16 held-out cases,
stopping each dtype at its first numerical failure. Identity decode is checked
on calibration/failure; native bitsandbytes stays the reference. No downstream
training, acceptance or speed claim is implied by small-shape agreement.

Run one physical CUDA instance, single GPU, no model download, with an independent
local deletion guard and per-instance remote watchdog. Bound this iteration to
$0.20, keeping total short screens below the previously approved $0.25 and
cumulative D1 below $10. Export and hash the raw evidence, then delete the rental.
The user's subsequent instruction authorizes correcting arithmetic and continuing
short checks; it does not authorize a new large benchmark campaign.

Even a passing 24-case validation does not close D1: checkpoint/negative control,
large recycled buffers, streamed decoder/LoRA gradients, then full-step memory
and timing remain required. A failed arithmetic gate stops those downstream runs.
