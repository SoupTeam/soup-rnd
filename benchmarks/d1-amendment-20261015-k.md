# D1 amendment K: projection grid first, separately conditioned RAM-tier cases

Dated 2026-10-15, committed before K arms on the same physical A100 as J:
`GPU-b0245e84-f202-902b-9331-f732ab021e57`, rental 56108892.
J diagnostics passed, but the first full-study ablation baseline arm had
actual timed-work SM clocks 1185–1335 MHz, violating the unchanged <=5% guard.
The J controller was explicitly stopped before wasting the remaining planned
ablation repeats. Every completed and interrupted J attempt is retained.
Neither J timings nor its diagnostic blocks are pooled into K.

Both GPU locked clocks and application clocks were denied by the provider.
The application-clock permission query itself was unsupported. No lock or
successful fixed-clock policy is claimed; retain every attempted command.

The complete new K plan still contains all 98 cases, under a new source and
fresh output directory. Re-run both CUDA correctness fixtures and GPU tests.
Run all 96 projection comparisons FIRST using unchanged >=20 steps AND
5 seconds warm-up, 200 timed steps, five valid ABBA blocks and F1 top-ups.
This places the already screened frozen-projection workload before the separate
bursty RAM-tier model comparisons. Per-comparison actual-clock and all other
validity requirements remain unchanged; no failed timing is filtered or fixed.

For the target-card ablation, increase fixed warm-up from 20 to 200 steps,
then keep 100 timed steps at sequence 512. It has no optimizer and the cached
zero arm remains timing-only garbage math. Longer conditioning is a new
experiment, not a repair of J. It may still fail validity and must say so.

For QA, restore F's independent 200-step synthetic thermal prelude in a
separate fresh process before each primary arm. No weights, allocator or RNG
state transfer into the primary training process. The primary run remains
exactly 20 warm-up + 256 timed optimizer updates, with unchanged model, data,
seed and held-out generation. Record prelude time separately; it cannot be
reported as additional QA training or removed from the raw experiment cost.

The immutable resource/budget deadline from J is unchanged: no later than
10.5 hours from its allocation, with 1.5 hours reserved for export and the
same cumulative $10 cap. No extra rental is authorized by this amendment.
Keep J evidence, all failed clock commands and K evidence in the same lossless
export, with separate output/source identities. Full index acceptance still
requires all 98 valid cases plus numerical audits and independent review.
