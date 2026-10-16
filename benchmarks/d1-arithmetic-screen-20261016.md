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

Run one physical CUDA instance, single GPU, no model download, with an independent
local deletion guard and per-instance remote watchdog. Bound this iteration to
$0.20, keeping total short screens below the previously approved $0.25 and
cumulative D1 below $10. Export and hash the raw evidence, then delete the rental.
The user's subsequent instruction authorizes correcting arithmetic and continuing
short checks; it does not authorize a new large benchmark campaign.

Even a passing 24-case validation does not close D1: checkpoint/negative control,
large recycled buffers, streamed decoder/LoRA gradients, then full-step memory
and timing remain required. A failed arithmetic gate stops those downstream runs.
