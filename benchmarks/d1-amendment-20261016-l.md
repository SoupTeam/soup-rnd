# D1 amendment L: separately sampled dense reference and bounded follow-up

Dated 2026-10-16, before any L CUDA work. No L GPU has been rented or measured.
K and its original raw archive, packet, source identities and verdicts stay
unchanged. All 17 terminal K NO_VERDICT cases failed the unchanged 5% clock
guard. K's dense reference CUDA events had no separate sampled telemetry
window, so they do not establish a frequency-matched dense ceiling.

## Reference measurement

Keep the primary projection settings: >=20 warm-up calls and >=5 seconds,
200 timed checkpointed forward/backward steps, fixed dtype/shape/seed, fresh
process, ABBA blocks, F1 sizing and the existing validity rules. Record primary
allocated and reserved peaks before allocating oracle tensors or dense weights.

After correctness witnesses, measure dense `F.linear` forward in the same
process and on the same GPU. Warm up >=20 calls AND >=1 second; time >=200
calls AND >=1 second. Synchronize each CUDA-event pair. The reference statistic
is the median per-call CUDA event, replacing the old 20-call batched mean for
this new source only. It is not an end-to-end forward/backward multiplier.

Save distinct primary/reference monotonic time boundaries, actual counts and
durations, individual events, and full NVML samples. The reference must have
>=3 available samples inside its actual timed window; the primary must have
>=1. Validate raw sample UUIDs against the CUDA UUID, finite positive SM clocks
and power limits, memory-controller utilization in [0,100], and the aggregates
against raw samples. All primary/reference clocks across each ABBA comparison
must satisfy max/min <=1.05. The power limit must be unchanged across both
windows and all arms. Missing/mismatched data invalidates the block. Offline
reporting recomputes this check rather than trusting a saved `passed` field.

## Cohort and spending boundary

The confirmed cost snapshot is $8.061 against a cumulative $10 authorization:
$1.939 remains before any late adjustment. No extra budget is assumed. A new
physical GPU/source is a new cohort with fresh CUDA correctness and GPU
regressions. Never append its timings to K, combine ratios across devices, or
present a selected follow-up as the complete 98-case K grid.

The runner's optional repeated `--case NAME` declares a separate selected
follow-up. It freezes its own plan/source before arms; changing that plan on
resume is rejected. The full-grid index continues to reject reduced plans.

Before paying for a measurement series, establish a fixed-clock candidate and
a conservative cost/export deadline. If clock control is unavailable, do not
repeat the grid hoping to get lucky. A targeted series may address K's missing
operating regions but cannot supply K's missing same-session reference windows.
A full new 98-case cohort requires a separately feasible resource plan within
the cumulative cap, or explicit new budget / a team's GPU. CPU tests of this
amendment are preparation, not CUDA measurement or full D1 acceptance.
