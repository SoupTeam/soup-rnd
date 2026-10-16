# D2 stack-v3: complete steps under observed default GPU Boost

Declared before any accepted timing on 2026-10-14. Both strict stack-v2 hosts
denied clock/power controls. RTX 3090 failed thermal/clock checks; its retry
and the A100 pilot failed the <=5% clock check. Those runs remain VOID. No
timing result from them is promoted to a valid comparison.

The user delegated the choice between further hardware searches and an explicit
default-Boost experiment. This experiment asks how long the configured complete
training stacks take on the available A100 under ordinary GPU Boost. It does
not establish fixed-frequency kernel efficiency, equal training quality, or
an isolated LoRA speedup. No clock or power override succeeded or is applied.

All configuration, numerical admission, native FP32 operator proofs, per-backend
repeatability, worker isolation and measurements follow
`d2-stack-comparison-v2-protocol.md`. Same frozen checkpoint and fixtures, four
backends in eight 50-step gates, FP16/BF16, Soup/Liger and Soup/Unsloth, three
ABBA cycles per pair: 48 fresh timing workers. Each warms up for 100 updates,
then restores adapters and creates a fresh optimizer/scaler before 50 measured
complete updates. No input, seed, optimizer or numerical tolerance changes.

The sole hardware admission change is explicit `--default-boost` (valid only
with `--stack-comparison`): arm and combined-cycle clocks must have >=3 finite,
positive observations, but their variation is observed rather than capped at
5%. Preserve the strict <=5% diagnostic without relabelling it PASS. Retain all
frequency, power, temperature and throttle samples; report clock ranges by
backend and cycle. Thermal/hardware throttle, missing observations, wrong UUID,
another process or occupied before/after still invalidate the run. Default
strict invocations and the old independent auditor retain the <=5% criterion.

Report all three candidate/Soup wall-time ratios, their median and range,
absolute full-step wall/CUDA-event times, target tokens/sec and peak allocated
and reserved memory. A ratio >1 means Soup is faster. No selective cycles or
confidence interval inferred from three observations. Disclose cross-stack
numerical FAILs and clock variation alongside speed; no general speedup claim.

Independent CPU replay must validate all retained tensors and lifecycle records
and recompute ratios using the declared clock policy. Missing/invalid arms make
the matrix incomplete. Source and this declaration are archived before launch.

The original cumulative 3-hour/$1.50 authorization and absolute destruction
deadline 2026-10-14 08:18:06 UTC remain unchanged. No third rental or extension.
